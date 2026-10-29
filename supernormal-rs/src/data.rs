use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};

#[derive(Clone, Serialize, Deserialize)]
pub struct Patch {
    pub origin: [f32; 3],
    pub dirs: [[f32; 3]; 9],
    pub plane: [f32; 3],
    pub inverse: [[[f32; 3]; 3]; 9],
    pub normals: [[f32; 3]; 9],
    pub mask: [f32; 9],
}

pub struct View {
    pub width: usize,
    pub height: usize,
    pub normals: Vec<[half::f16; 3]>,
    pub mask: Vec<bool>,
    pub origin: Vector3<f32>,
    pub rotation: Matrix3<f32>,
    pub inv_k: Matrix3<f32>,
    /// Lazily computed world-space ray directions `[height * width]` and analytical
    /// inverse basis rows `[height * width]` for legacy CPU patch gathering (`--cpu-patch-upload`
    /// or `check-render`), avoiding 1.76 GB of host RAM allocation during GPU training.
    pub cached_rays: std::sync::OnceLock<(Vec<[f32; 3]>, Vec<[[f32; 3]; 3]>)>,
    /// Precomputed sum of 9 pixel masks in the 3x3 neighborhood centered at `(x, y)`
    /// (`[height * width]`, stored as `u8` in `0..=9` for a 4x smaller L3 cache footprint).
    pub patch_mask_sums: Vec<u8>,
}

/// Device-resident view tables and cropped lossless `f16` normal/mask banks uploaded
/// once at startup so the GPU can compute ray-sphere intersections and gather 3x3
/// patch normals/masks directly from compact `[N, 4]` `(v, x, y, jitter)` seeds.
#[derive(Clone)]
pub struct DeviceDataset {
    /// `[num_views, 30]`:
    /// `[origin(3), plane(3), u_col(3), v_col(3), w_col(3), inv_k(9), min_x, min_y, bbox_w, bbox_h, bank, bank_offset]`
    pub view_meta: Tensor<2>,
    /// Up to 3 banks of cropped bounding-box pixels `[bank_pixels, 2]`, each `< 128 MiB`
    /// and `< 2^24` rows so indices and 24-bit integer normal encodings are exact in `f32`.
    pub pixel_banks: [Tensor<2>; 3],
}

pub struct Dataset {
    pub views: Vec<View>,
    pub scale: f32,
    pub offset: [f32; 3],
    pub device_cache: std::sync::OnceLock<DeviceDataset>,
}

fn read_matrix(npz: &mut NpzReader<File>, name: &str) -> Result<Array2<f64>> {
    if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(name) {
        return Ok(a);
    }
    let a: Array2<f32> = npz.by_name(name)?;
    Ok(a.mapv(|x| x as f64))
}

impl Dataset {
    pub fn load(path: &Path) -> Result<Self> {
        let mut npz = NpzReader::new(File::open(path.join("cameras_sphere.npz"))?)?;
        let mut files: Vec<_> = std::fs::read_dir(path.join("normal_world_space_sdmunips"))?
            .filter_map(|x| x.ok().map(|x| x.path()))
            .filter(|x| x.extension().is_some_and(|x| x == "exr"))
            .collect();
        files.sort();
        ensure!(!files.is_empty(), "no normal maps");

        let mut views = vec![];
        let mut scale = 1.;
        let mut offset = [0.; 3];
        for file in files {
            let id = file
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<usize>()?;
            let world = read_matrix(&mut npz, &format!("world_mat_{id}"))?;
            let sm = read_matrix(&mut npz, &format!("scale_mat_{id}"))?;
            if views.is_empty() {
                scale = sm[[0, 0]] as f32;
                offset = std::array::from_fn(|i| sm[[i, 3]] as f32);
            }
            let projection = world.dot(&sm);
            let mut a = Matrix3::from_fn(|r, c| projection[[r, c]] as f32);
            let mut b = Vector3::from_fn(|r, _| projection[[r, 3]] as f32);
            if a.determinant() < 0. {
                a = -a;
                b = -b;
            }
            let origin = -a.try_inverse().context("singular camera")? * b;
            let row3 = a.row(2).transpose();
            let k33 = row3.norm();
            let r3 = row3 / k33;
            let row2 = a.row(1).transpose();
            let k23 = row2.dot(&r3);
            let temp = row2 - r3 * k23;
            let k22 = temp.norm();
            let r2 = temp / k22;
            let row1 = a.row(0).transpose();
            let k13 = row1.dot(&r3);
            let k12 = row1.dot(&r2);
            let temp = row1 - r3 * k13 - r2 * k12;
            let k11 = temp.norm();
            let r1 = temp / k11;
            let rotation = Matrix3::from_columns(&[r1, r2, r3]);
            let k = Matrix3::new(k11, k12, k13, 0., k22, k23, 0., 0., k33) / k33;
            let inv_k = k.try_inverse().context("singular intrinsics")?;

            let normal = image::open(&file)?.to_rgb32f();
            let (w, h) = normal.dimensions();
            let mask = image::open(path.join("mask").join(format!(
                "{}.png",
                file.file_stem().unwrap().to_str().unwrap()
            )))?
            .to_rgb8();
            ensure!(mask.dimensions() == (w, h), "normal/mask size mismatch");
            let masks: Vec<bool> = mask.pixels().map(|p| p[0] != 0).collect();
            let normals = normal
                .pixels()
                .zip(&masks)
                .map(|(p, m)| {
                    if *m {
                        p.0.map(half::f16::from_f32)
                    } else {
                        [half::f16::ZERO; 3]
                    }
                })
                .collect();

            let width = w as usize;
            let height = h as usize;
            let patch_mask_sums: Vec<u8> = (0..width * height)
                .into_par_iter()
                .map(|idx| {
                    let xx = idx % width;
                    let yy = idx / width;
                    if xx == 0 || xx + 1 >= width || yy == 0 || yy + 1 >= height {
                        return 0u8;
                    }
                    let mut sum = 0u8;
                    for py in 0..3 {
                        let row_off = (yy + py - 1) * width;
                        for px in 0..3 {
                            if masks[row_off + (xx + px - 1)] {
                                sum += 1;
                            }
                        }
                    }
                    sum
                })
                .collect();

            views.push(View {
                width,
                height,
                normals,
                mask: masks,
                origin,
                rotation,
                inv_k,
                cached_rays: std::sync::OnceLock::new(),
                patch_mask_sums,
            });
        }
        Ok(Self {
            views,
            scale,
            offset,
            device_cache: std::sync::OnceLock::new(),
        })
    }

    pub fn device_dataset(&self, device: &Device) -> DeviceDataset {
        self.device_cache
            .get_or_init(|| {
                use crate::field::tensor;
                const MAX_BANK_PIXELS: usize = 15_500_000;
                let mut meta = Vec::with_capacity(self.views.len() * 30);
                let mut banks: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                let mut cur_bank = 0usize;

                for v in &self.views {
                    let mut min_x = v.width;
                    let mut max_x = 0usize;
                    let mut min_y = v.height;
                    let mut max_y = 0usize;
                    for yy in 0..v.height {
                        let row_off = yy * v.width;
                        for xx in 0..v.width {
                            if v.mask[row_off + xx] {
                                if xx < min_x {
                                    min_x = xx;
                                }
                                if xx > max_x {
                                    max_x = xx;
                                }
                                if yy < min_y {
                                    min_y = yy;
                                }
                                if yy > max_y {
                                    max_y = yy;
                                }
                            }
                        }
                    }
                    let (bx, by, bw, bh) = if min_x <= max_x && min_y <= max_y {
                        (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
                    } else {
                        (0, 0, 0, 0)
                    };
                    let bbox_pixels = bw * bh;
                    if cur_bank < 2
                        && (banks[cur_bank].len() / 2) + bbox_pixels > MAX_BANK_PIXELS
                    {
                        cur_bank += 1;
                    }
                    let bank_offset = banks[cur_bank].len() / 2;
                    assert!(
                        bank_offset + bbox_pixels < (1 << 24),
                        "device dataset pixel bank exceeds exact f32 integer range"
                    );
                    for yy in by..by + bh {
                        let row_off = yy * v.width;
                        for xx in bx..bx + bw {
                            let idx = row_off + xx;
                            if v.mask[idx] {
                                let n = v.normals[idx];
                                let nx_bits = u32::from(n[0].to_bits());
                                let ny_bits = u32::from(n[1].to_bits());
                                let nz_bits = u32::from(n[2].to_bits());
                                let f0 = (nx_bits | ((ny_bits & 0xFF) << 16)) as f32;
                                let f1 = (nz_bits | ((ny_bits >> 8) << 16)) as f32;
                                banks[cur_bank].push(f0);
                                banks[cur_bank].push(f1);
                            } else {
                                banks[cur_bank].push(-1.0_f32);
                                banks[cur_bank].push(0.0_f32);
                            }
                        }
                    }

                    let origin: [f32; 3] = v.origin.into();
                    let u_col = v.rotation.column(0).into_owned();
                    let v_col = v.rotation.column(1).into_owned();
                    let plane = v.rotation.column(2).into_owned();
                    let w_col = u_col.cross(&v_col);
                    meta.extend(origin);
                    meta.extend([plane[0], plane[1], plane[2]]);
                    meta.extend([u_col[0], u_col[1], u_col[2]]);
                    meta.extend([v_col[0], v_col[1], v_col[2]]);
                    meta.extend([w_col[0], w_col[1], w_col[2]]);
                    for r in 0..3 {
                        for c in 0..3 {
                            meta.push(v.inv_k[(r, c)]);
                        }
                    }
                    meta.extend([
                        bx as f32,
                        by as f32,
                        bw as f32,
                        bh as f32,
                        cur_bank as f32,
                        bank_offset as f32,
                    ]);
                }
                let pixel_banks = banks.map(|mut b| {
                    if b.is_empty() {
                        b.extend([-1.0_f32, 0.0_f32]);
                    }
                    let rows = b.len() / 2;
                    tensor(b, [rows, 2], device)
                });
                DeviceDataset {
                    view_meta: tensor(meta, [self.views.len(), 30], device),
                    pixel_banks,
                }
            })
            .clone()
    }

    /// Sample `count` random patch coordinates `(v, x, y)` and return both the
    /// coordinate list and the exact `normal_den` without materializing `Vec<Patch>`.
    pub fn random_patch_coords(
        &self,
        count: usize,
        rng: &mut impl Rng,
    ) -> (Vec<(usize, usize, usize)>, f32) {
        let mut coords = Vec::with_capacity(count);
        let mut mask_sum = 0u32;
        for _ in 0..count {
            let v = rng.random_range(0..self.views.len());
            let view = &self.views[v];
            let x = rng.random_range(1..view.width - 2);
            let y = rng.random_range(1..view.height - 2);
            mask_sum += u32::from(view.patch_mask_sums[y * view.width + x]);
            coords.push((v, x, y));
        }
        (coords, mask_sum as f32 + 1e-5)
    }

    pub fn random_patches(&self, count: usize, rng: &mut impl Rng) -> Vec<Patch> {
        // Draw random coordinates sequentially to preserve exact deterministic RNG state,
        // then gather precomputed patch ray bases and 3x3 inversions in parallel.
        let coords: Vec<(usize, usize, usize)> = (0..count)
            .map(|_| {
                let v = rng.random_range(0..self.views.len());
                let view = &self.views[v];
                let x = rng.random_range(1..view.width - 2);
                let y = rng.random_range(1..view.height - 2);
                (v, x, y)
            })
            .collect();

        let views = &self.views;
        coords
            .into_par_iter()
            .map(|(v, x, y)| views[v].patch(x, y))
            .collect()
    }
}

impl View {
    pub fn patch(&self, x: usize, y: usize) -> Patch {
        let (cached_dirs, cached_inverses) = self.cached_rays.get_or_init(|| {
            let u_col = self.rotation.column(0).into_owned();
            let v_col = self.rotation.column(1).into_owned();
            let w_col = u_col.cross(&v_col);
            (0..self.width * self.height)
                .into_par_iter()
                .map(|idx| {
                    let xx = idx % self.width;
                    let yy = idx / self.width;
                    let dir = self.rotation
                        * (self.inv_k * Vector3::new(xx as f32, yy as f32, 1.)).normalize();
                    let det = dir.dot(&w_col);
                    let inv = if det.abs() > 1e-12 {
                        let inv_det = 1.0 / det;
                        let c0 = w_col * inv_det;
                        let c1 = v_col.cross(&dir) * inv_det;
                        let c2 = dir.cross(&u_col) * inv_det;
                        [
                            [c0[0], c1[0], c2[0]],
                            [c0[1], c1[1], c2[1]],
                            [c0[2], c1[2], c2[2]],
                        ]
                    } else {
                        let basis = Matrix3::from_rows(&[
                            dir.transpose(),
                            u_col.transpose(),
                            v_col.transpose(),
                        ]);
                        let m = basis.try_inverse().unwrap();
                        std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
                    };
                    let dir_arr: [f32; 3] = dir.into();
                    (dir_arr, inv)
                })
                .unzip()
        });
        let origin = self.origin.into();
        let plane = self.rotation.column(2).into_owned().into();
        let mut patch = Patch {
            origin,
            dirs: [[0.; 3]; 9],
            plane,
            inverse: [[[0.; 3]; 3]; 9],
            normals: [[0.; 3]; 9],
            mask: [0.; 9],
        };
        for py in 0..3 {
            for px in 0..3 {
                let i = py * 3 + px;
                let xx = x + px - 1;
                let yy = y + py - 1;
                let idx = yy * self.width + xx;
                patch.dirs[i] = cached_dirs[idx];
                patch.inverse[i] = cached_inverses[idx];
                patch.normals[i] = self.normals[idx].map(half::f16::to_f32);
                patch.mask[i] = if self.mask[idx] { 1. } else { 0. };
            }
        }
        patch
    }
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn point(o: [f32; 3], d: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| o[i] + d[i] * t)
}

pub fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot(
        std::array::from_fn(|i| a[i] - b[i]),
        std::array::from_fn(|i| a[i] - b[i]),
    )
    .sqrt()
}
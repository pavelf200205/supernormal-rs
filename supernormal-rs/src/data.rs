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
    /// Precomputed world-space ray directions `[height * width]` and analytical
    /// inverse basis rows `[height * width]` so per-step patch sampling becomes
    /// a zero-compute table lookup.
    pub cached_dirs: Vec<[f32; 3]>,
    pub cached_inverses: Vec<[[f32; 3]; 3]>,
    /// Precomputed sum of 9 pixel masks in the 3x3 neighborhood centered at `(x, y)`
    /// (`[height * width]`), enabling O(1) `normal_den` accumulation without
    /// constructing 2,048 `Patch` structs on the host.
    pub patch_mask_sums: Vec<f32>,
}

/// Device-resident view tables uploaded once at startup so the GPU can gather
/// `march_rays` `[N, 12]`, `geom_rays` `[N, 39]`, and `packed_patches` `[N, 123]`
/// from compact `[N, 5]` patch coordinates without per-step 1.4 MB host uploads.
#[derive(Clone)]
pub struct DeviceDataset {
    /// `[num_views, 24]`: `[origin(3), plane(3), u_col(3), v_col(3), w_col(3), inv_k(9)]`
    pub view_meta: Tensor<2>,
}

pub struct Dataset {
    pub views: Vec<View>,
    pub scale: f32,
    pub offset: [f32; 3],
    pub uniform_view_dims: bool,
    pub view_pixel_cdf: Vec<u64>,
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
            let masks: Vec<bool> = mask.pixels().map(|p| p[0] > 127).collect();
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
            let u_col = rotation.column(0).into_owned();
            let v_col = rotation.column(1).into_owned();
            let w_col = u_col.cross(&v_col);
            let (cached_dirs, cached_inverses): (Vec<[f32; 3]>, Vec<[[f32; 3]; 3]>) = (0..width
                * height)
                .into_par_iter()
                .map(|idx| {
                    let xx = idx % width;
                    let yy = idx / width;
                    let dir =
                        rotation * (inv_k * Vector3::new(xx as f32, yy as f32, 1.)).normalize();
                    // Analytical inverse of [dir^T; u^T; v^T] using orthonormal camera axes (u, v, w = u x v):
                    // det = dir . w; col0 = w / det, col1 = (v x dir) / det, col2 = (dir x u) / det.
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
                .unzip::<[f32; 3], [[f32; 3]; 3], Vec<[f32; 3]>, Vec<[[f32; 3]; 3]>>();

            let patch_mask_sums: Vec<f32> = (0..width * height)
                .into_par_iter()
                .map(|idx| {
                    let xx = idx % width;
                    let yy = idx / width;
                    if xx == 0 || xx + 1 >= width || yy == 0 || yy + 1 >= height {
                        return 0.0_f32;
                    }
                    let mut sum = 0.0_f32;
                    for py in 0..3 {
                        for px in 0..3 {
                            if masks[(yy + py - 1) * width + (xx + px - 1)] {
                                sum += 1.0_f32;
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
                cached_dirs,
                cached_inverses,
                patch_mask_sums,
            });
        }
        let uniform_view_dims = views
            .windows(2)
            .all(|w| w[0].width == w[1].width && w[0].height == w[1].height);
        let mut cum = 0u64;
        let view_pixel_cdf: Vec<u64> = views
            .iter()
            .map(|v| {
                let valid_px =
                    (v.width.saturating_sub(3) * v.height.saturating_sub(3)).max(1) as u64;
                cum += valid_px;
                cum
            })
            .collect();

        Ok(Self {
            views,
            scale,
            offset,
            uniform_view_dims,
            view_pixel_cdf,
            device_cache: std::sync::OnceLock::new(),
        })
    }

    #[inline]
    fn sample_view_index(&self, rng: &mut impl Rng) -> usize {
        if self.uniform_view_dims {
            rng.random_range(0..self.views.len())
        } else {
            let total_px = *self.view_pixel_cdf.last().expect("non-empty views");
            let pick = rng.random_range(0..total_px);
            self.view_pixel_cdf.partition_point(|&c| c <= pick)
        }
    }

    pub fn device_dataset(&self, device: &Device) -> DeviceDataset {
        self.device_cache
            .get_or_init(|| {
                use crate::field::tensor;
                let mut meta = Vec::with_capacity(self.views.len() * 24);
                for v in &self.views {
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
                }
                DeviceDataset {
                    view_meta: tensor(meta, [self.views.len(), 24], device),
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
        let mut mask_sum = 0.0_f32;
        for _ in 0..count {
            let v = self.sample_view_index(rng);
            let view = &self.views[v];
            let x = rng.random_range(1..view.width - 2);
            let y = rng.random_range(1..view.height - 2);
            mask_sum += view.patch_mask_sums[y * view.width + x];
            coords.push((v, x, y));
        }
        (coords, mask_sum + 1e-5)
    }

    /// Sample `count` random patches + ray jitter and upload a single `[count, 40]`
    /// descriptor tensor (`[v, x, y, 9 * (nx, ny, nz, mask), jitter]`) to `device`,
    /// driving both GPU ray-sphere marching and 3x3 patch table construction.
    pub fn random_patch_coords_tensor(
        &self,
        count: usize,
        rng: &mut impl Rng,
        device: &Device,
    ) -> (Tensor<2>, f32) {
        let mut draws = Vec::with_capacity(count);
        let mut mask_sum = 0.0_f32;
        for _ in 0..count {
            let v = self.sample_view_index(rng);
            let view = &self.views[v];
            let x = rng.random_range(1..view.width - 2);
            let y = rng.random_range(1..view.height - 2);
            let jitter = rng.random::<f32>();
            mask_sum += view.patch_mask_sums[y * view.width + x];
            draws.push((v, x, y, jitter));
        }
        let mut flat = vec![0.0_f32; count * 40];
        let views = &self.views;
        flat.par_chunks_exact_mut(40)
            .zip(draws.par_iter())
            .for_each(|(dst, &(v, x, y, jitter))| {
                let view = &views[v];
                dst[0] = v as f32;
                dst[1] = x as f32;
                dst[2] = y as f32;
                let mut k = 3usize;
                for py in 0..3 {
                    let row_off = (y + py - 1) * view.width;
                    for px in 0..3 {
                        let idx = row_off + (x + px - 1);
                        let n = view.normals[idx];
                        dst[k] = n[0].to_f32();
                        dst[k + 1] = n[1].to_f32();
                        dst[k + 2] = n[2].to_f32();
                        dst[k + 3] = if view.mask[idx] { 1.0_f32 } else { 0.0_f32 };
                        k += 4;
                    }
                }
                dst[39] = jitter;
            });
        (crate::field::tensor(flat, [count, 40], device), mask_sum + 1e-5)
    }

    pub fn random_patches(&self, count: usize, rng: &mut impl Rng) -> Vec<Patch> {
        // Draw random coordinates sequentially to preserve exact deterministic RNG state,
        // then gather precomputed patch ray bases and 3x3 inversions in parallel.
        let coords: Vec<(usize, usize, usize)> = (0..count)
            .map(|_| {
                let v = self.sample_view_index(rng);
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

    /// Compute a conservative, resolution-adaptive, close-up-safe 3D Euclidean
    /// Signed Distance Field `hull_sdf` (`res^3`, layout `(cx * res + cy) * res + cz`,
    /// `> 0` outside object, `< 0` inside visual hull) from the dataset masks.
    pub fn compute_visual_hull_sdf(&self, res: usize) -> Option<(Vec<f32>, usize)> {
        const INF_SQ: f64 = 1e12;

        fn edt_1d_sq(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
            let n = f.len();
            if n == 0 {
                return;
            }
            let mut k = 0usize;
            v[0] = 0;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            for q in 1..n {
                let q_f = q as f64;
                let fq = f[q] + q_f * q_f;
                let mut s;
                loop {
                    let vk = v[k];
                    let vk_f = vk as f64;
                    s = (fq - (f[vk] + vk_f * vk_f)) / (2.0 * (q_f - vk_f));
                    if s > z[k] || k == 0 {
                        break;
                    }
                    k -= 1;
                }
                if s <= z[k] {
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                } else {
                    k += 1;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = f64::INFINITY;
                }
            }
            k = 0;
            for (q, out) in d.iter_mut().enumerate() {
                let q_f = q as f64;
                while z[k + 1] < q_f {
                    k += 1;
                }
                let dx = q_f - v[k] as f64;
                *out = dx * dx + f[v[k]];
            }
        }

        struct ViewHullData {
            a: [[f32; 3]; 3],
            b: [f32; 3],
            width: usize,
            height: usize,
            focal_max: f32,
            calib_margin_px: f32,
            outside_dist_px: Vec<f32>,
        }

        let view_hulls: Vec<ViewHullData> = self
            .views
            .par_iter()
            .filter_map(|view| {
                if !view.mask.iter().any(|&m| m) {
                    return None;
                }
                let w = view.width;
                let h = view.height;
                let k = view.inv_k.try_inverse()?;
                let rt = view.rotation.transpose();
                let a_mat = k * rt;
                let b_vec = -(a_mat * view.origin);
                let focal_max = k[(0, 0)].abs().max(k[(1, 1)].abs());
                // Resolution-proportional angular safety dilation (~0.8% of image size,
                // e.g. ~4 px at 512x512, ~8 px at 960/1024, ~16 px at 2048x2048).
                let calib_margin_px = (0.008 * (w.max(h) as f32)).max(4.0);

                // Exact 2D Euclidean distance in pixels to nearest `mask == true` pixel.
                let mut grid: Vec<f64> = view
                    .mask
                    .iter()
                    .map(|&m| if m { 0.0 } else { INF_SQ })
                    .collect();
                // Pass 1: columns (length h)
                let mut col_in = vec![0.0_f64; h];
                let mut col_out = vec![0.0_f64; h];
                let mut v_buf = vec![0usize; h];
                let mut z_buf = vec![0.0_f64; h + 1];
                for x in 0..w {
                    for y in 0..h {
                        col_in[y] = grid[y * w + x];
                    }
                    edt_1d_sq(&col_in, &mut col_out, &mut v_buf, &mut z_buf);
                    for y in 0..h {
                        grid[y * w + x] = col_out[y];
                    }
                }
                // Pass 2: rows (length w)
                let mut row_in = vec![0.0_f64; w];
                let mut row_out = vec![0.0_f64; w];
                let mut rv_buf = vec![0usize; w];
                let mut rz_buf = vec![0.0_f64; w + 1];
                let mut outside_dist_px = vec![0.0_f32; w * h];
                for y in 0..h {
                    let off = y * w;
                    row_in.copy_from_slice(&grid[off..off + w]);
                    edt_1d_sq(&row_in, &mut row_out, &mut rv_buf, &mut rz_buf);
                    for x in 0..w {
                        outside_dist_px[off + x] = row_out[x].max(0.0).sqrt() as f32;
                    }
                }

                Some(ViewHullData {
                    a: std::array::from_fn(|r| std::array::from_fn(|c| a_mat[(r, c)])),
                    b: [b_vec[0], b_vec[1], b_vec[2]],
                    width: w,
                    height: h,
                    focal_max,
                    calib_margin_px,
                    outside_dist_px,
                })
            })
            .collect();

        if view_hulls.is_empty() {
            return None;
        }

        let r = res;
        let rr = r * r;
        let total = r * rr;
        let voxel_size = 2.0_f32 / r as f32;
        let r_vox = 0.5_f32 * 3.0_f32.sqrt() * voxel_size;

        let inside: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|cell| {
                let cx = cell / rr;
                let cy = (cell / r) % r;
                let cz = cell % r;
                let px = ((cx as f32 + 0.5) / r as f32) * 2.0 - 1.0;
                let py = ((cy as f32 + 0.5) / r as f32) * 2.0 - 1.0;
                let pz = ((cz as f32 + 0.5) / r as f32) * 2.0 - 1.0;
                if (px * px + py * py + pz * pz).sqrt() - r_vox > 1.0 {
                    return false;
                }
                for vh in &view_hulls {
                    let qz = vh.a[2][0] * px + vh.a[2][1] * py + vh.a[2][2] * pz + vh.b[2];
                    if qz <= 1e-3 {
                        continue;
                    }
                    let u = (vh.a[0][0] * px + vh.a[0][1] * py + vh.a[0][2] * pz + vh.b[0]) / qz;
                    let vv = (vh.a[1][0] * px + vh.a[1][1] * py + vh.a[1][2] * pz + vh.b[1]) / qz;
                    let vox_px = r_vox * vh.focal_max / qz;
                    let w_f = vh.width as f32;
                    let h_f = vh.height as f32;
                    // Close-up frustum safety: if the projected voxel is not strictly
                    // inside `[vox_px, W - 1 - vox_px] x [vox_px, H - 1 - vox_px]`,
                    // it may extend outside a cropped view — treat as unobserved.
                    if u < vox_px
                        || u > w_f - 1.0 - vox_px
                        || vv < vox_px
                        || vv > h_f - 1.0 - vox_px
                    {
                        continue;
                    }
                    let xi = (u.round() as usize).min(vh.width - 1);
                    let yi = (vv.round() as usize).min(vh.height - 1);
                    let dist_px = vh.outside_dist_px[yi * vh.width + xi];
                    if dist_px > vh.calib_margin_px + vox_px {
                        return false;
                    }
                }
                true
            })
            .collect();

        let inside_count = inside.iter().filter(|&&b| b).count();
        if inside_count * 2000 < total || inside_count == total {
            return None;
        }

        let edt_3d = |target_true: bool| -> Vec<f64> {
            let mut vol: Vec<f64> = inside
                .iter()
                .map(|&b| if b == target_true { 0.0 } else { INF_SQ })
                .collect();
            // Z pass (stride 1)
            vol.par_chunks_exact_mut(r).for_each(|line| {
                let mut f = vec![0.0_f64; r];
                let mut v = vec![0usize; r];
                let mut z = vec![0.0_f64; r + 1];
                f.copy_from_slice(line);
                edt_1d_sq(&f, line, &mut v, &mut z);
            });
            // Y pass (stride r) within each X slab (size rr)
            vol.par_chunks_exact_mut(rr).for_each(|slab| {
                let mut f = vec![0.0_f64; r];
                let mut d = vec![0.0_f64; r];
                let mut v = vec![0usize; r];
                let mut z = vec![0.0_f64; r + 1];
                for cz in 0..r {
                    for cy in 0..r {
                        f[cy] = slab[cy * r + cz];
                    }
                    edt_1d_sq(&f, &mut d, &mut v, &mut z);
                    for cy in 0..r {
                        slab[cy * r + cz] = d[cy];
                    }
                }
            });
            // X pass (stride rr): transpose by (cy, cz) fibers in parallel
            let prev = vol.clone();
            let fibers: Vec<(usize, Vec<f64>)> = (0..rr)
                .into_par_iter()
                .map(|yz| {
                    let mut f = vec![0.0_f64; r];
                    let mut d = vec![0.0_f64; r];
                    let mut v = vec![0usize; r];
                    let mut z = vec![0.0_f64; r + 1];
                    for cx in 0..r {
                        f[cx] = prev[cx * rr + yz];
                    }
                    edt_1d_sq(&f, &mut d, &mut v, &mut z);
                    (yz, d)
                })
                .collect();
            for (yz, d) in fibers {
                for cx in 0..r {
                    vol[cx * rr + yz] = d[cx];
                }
            }
            vol
        };

        let d_to_inside = edt_3d(true);
        let d_to_outside = edt_3d(false);
        let hull_sdf: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|i| {
                let signed_vox = if inside[i] {
                    -(d_to_outside[i].max(0.0).sqrt() as f32 - 0.5)
                } else {
                    d_to_inside[i].max(0.0).sqrt() as f32 - 0.5
                };
                signed_vox * voxel_size
            })
            .collect();

        Some((hull_sdf, inside_count))
    }
}

impl View {
    pub fn patch(&self, x: usize, y: usize) -> Patch {
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
                patch.dirs[i] = self.cached_dirs[idx];
                patch.inverse[i] = self.cached_inverses[idx];
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
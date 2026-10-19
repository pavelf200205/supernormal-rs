use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::Rng;
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
}
pub struct Dataset {
    pub views: Vec<View>,
    pub scale: f32,
    pub offset: [f32; 3],
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
            views.push(View {
                width: w as usize,
                height: h as usize,
                normals,
                mask: masks,
                origin,
                rotation,
                inv_k: k.try_inverse().context("singular intrinsics")?,
            });
        }
        Ok(Self {
            views,
            scale,
            offset,
        })
    }
    pub fn random_patches(&self, count: usize, rng: &mut impl Rng) -> Vec<Patch> {
        (0..count)
            .map(|_| {
                let view = &self.views[rng.random_range(0..self.views.len())];
                view.patch(
                    rng.random_range(1..view.width - 2),
                    rng.random_range(1..view.height - 2),
                )
            })
            .collect()
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
                let dir = self.rotation
                    * (self.inv_k * Vector3::new(xx as f32, yy as f32, 1.)).normalize();
                let basis = Matrix3::from_rows(&[
                    dir.transpose(),
                    self.rotation.column(0).transpose(),
                    self.rotation.column(1).transpose(),
                ]);
                let inv = basis.try_inverse().unwrap();
                patch.dirs[i] = dir.into();
                patch.inverse[i] = std::array::from_fn(|r| std::array::from_fn(|c| inv[(r, c)]));
                patch.normals[i] = self.normals[yy * self.width + xx].map(half::f16::to_f32);
                patch.mask[i] = if self.mask[yy * self.width + xx] {
                    1.
                } else {
                    0.
                };
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

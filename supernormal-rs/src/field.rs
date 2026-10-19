use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Serialize, Deserialize, Clone)]
pub struct ParameterMeta {
    pub name: String,
    pub shape: [usize; 2],
    pub file: String,
}
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub step: usize,
    pub active_levels: usize,
    pub parameters: Vec<ParameterMeta>,
}
pub struct Parameter {
    pub value: Tensor<2>,
    pub m: Tensor<2>,
    pub v: Tensor<2>,
    pub updates: usize,
}
pub struct Field {
    pub params: BTreeMap<String, Parameter>,
    pub active: usize,
    pub step: usize,
    pub device: Device,
    pub peak_reserved_bytes: std::cell::Cell<u64>,
}

pub fn tensor<const D: usize>(data: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(data, shape), device)
}
pub fn ints(data: Vec<i32>, device: &Device) -> Tensor<1, Int> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), device)
}
pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() % 4 == 0, "invalid float file");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
pub fn write_f32(path: &Path, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

#[derive(Default)]
pub struct Gradients {
    pub params: BTreeMap<String, Tensor<2>>,
    pub host_tables: Vec<Vec<f32>>,
}

impl Gradients {
    pub fn insert(&mut self, name: String, tensor: Tensor<2>) {
        self.params.insert(name, tensor);
    }
    pub fn into_tensors(mut self, device: &Device) -> BTreeMap<String, Tensor<2>> {
        for (level, data) in self.host_tables.into_iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            let name = format!("encoding.embeddings.{level}");
            let entries = data.len() / 2;
            let g = tensor(data, [entries, 2], device);
            self.params
                .entry(name)
                .and_modify(|v| *v = (v.clone() + g.clone()).detach())
                .or_insert(g);
        }
        self.params
    }
}

pub struct FeatureChunk {
    pub feat: Tensor<2>,
    pub pts_tensor: Tensor<2>,
    pub point_count: usize,
}

impl Field {
    pub fn new(device: Device, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut params = BTreeMap::new();
        let mut add = |name: String, values: Vec<f32>, shape: [usize; 2]| {
            params.insert(
                name,
                Parameter {
                    value: tensor(values, shape, &device).require_grad(),
                    m: Tensor::zeros(shape, &device),
                    v: Tensor::zeros(shape, &device),
                    updates: 0,
                },
            );
        };
        for l in 0..14 {
            let res = (32f64 * 1.3195079107728942f64.powi(l)).floor() as usize;
            let entries = ((res + 1).pow(3)).min(524288);
            add(
                format!("encoding.embeddings.{l}"),
                (0..entries * 2)
                    .map(|_| rng.random_range(-1e-4f32..1e-4))
                    .collect(),
                [entries, 2],
            );
        }
        let mut normal = || {
            let u = rng.random::<f32>().max(1e-12);
            let v = rng.random::<f32>();
            (-2. * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        };
        for (l, input, output) in [(0, 31, 64), (1, 64, 1)] {
            let values: Vec<f32> = (0..input * output)
                .map(|i| {
                    if l == 0 {
                        if i % input < 3 {
                            normal() * (2f32 / 64.).sqrt()
                        } else {
                            0.
                        }
                    } else {
                        normal() * 1e-4 + std::f32::consts::PI.sqrt() / 8.
                    }
                })
                .collect();
            let magnitude = values
                .chunks(input)
                .map(|row| row.iter().map(|x| x * x).sum::<f32>().sqrt())
                .collect();
            add(format!("lin{l}.weight_v"), values, [output, input]);
            add(format!("lin{l}.weight_g"), magnitude, [output, 1]);
            add(
                format!("lin{l}.bias"),
                vec![if l == 0 { 0. } else { -0.8 }; output],
                [1, output],
            );
        }
        add("variance".into(), vec![0.5], [1, 1]);
        let _ = device.sync();
        device.memory_cleanup();
        let peak = device
            .memory_pool_usage()
            .map(|u| u.bytes_reserved)
            .unwrap_or(0);
        Self {
            params,
            active: 0,
            step: 0,
            device,
            peak_reserved_bytes: std::cell::Cell::new(peak),
        }
    }
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let m: Manifest = serde_json::from_slice(&std::fs::read(path.join("model.json"))?)?;
        ensure!(m.format_version == 1, "unsupported model version");
        let mut params = BTreeMap::new();
        for p in m.parameters {
            let data = read_f32(&path.join(&p.file))?;
            ensure!(
                data.len() == p.shape.iter().product::<usize>(),
                "invalid shape for {}",
                p.name
            );
            let value = tensor(data, p.shape, &device).require_grad();
            let state = path.join(format!("{}.adam.json", p.file));
            let (m, v, updates) = if state.exists() {
                let updates = serde_json::from_slice(&std::fs::read(state)?)?;
                let m = tensor(
                    read_f32(&path.join(format!("{}.m", p.file)))?,
                    p.shape,
                    &device,
                );
                let v = tensor(
                    read_f32(&path.join(format!("{}.v", p.file)))?,
                    p.shape,
                    &device,
                );
                (m, v, updates)
            } else {
                (
                    Tensor::zeros(p.shape, &device),
                    Tensor::zeros(p.shape, &device),
                    0,
                )
            };
            params.insert(
                p.name,
                Parameter {
                    value,
                    m,
                    v,
                    updates,
                },
            );
        }
        let _ = device.sync();
        device.memory_cleanup();
        let peak = device
            .memory_pool_usage()
            .map(|u| u.bytes_reserved)
            .unwrap_or(0);
        Ok(Self {
            params,
            active: m.active_levels,
            step: m.step,
            device,
            peak_reserved_bytes: std::cell::Cell::new(peak),
        })
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        let mut manifest = Manifest {
            format_version: 1,
            step: self.step,
            active_levels: self.active,
            parameters: vec![],
        };
        for (name, p) in &self.params {
            let file = format!("{}.bin", name.replace('.', "_"));
            write_f32(
                &path.join(&file),
                &p.value.clone().try_into_vec_as::<f32>()?,
            )?;
            write_f32(
                &path.join(format!("{file}.m")),
                &p.m.clone().try_into_vec_as::<f32>()?,
            )?;
            write_f32(
                &path.join(format!("{file}.v")),
                &p.v.clone().try_into_vec_as::<f32>()?,
            )?;
            std::fs::write(
                path.join(format!("{file}.adam.json")),
                serde_json::to_vec(&p.updates)?,
            )?;
            manifest.parameters.push(ParameterMeta {
                name: name.clone(),
                shape: p.value.dims(),
                file,
            });
        }
        std::fs::write(
            path.join("model.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(())
    }
    pub fn p(&self, name: &str) -> Tensor<2> {
        self.params
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .value
            .clone()
    }
    pub fn forward(&self, points: &[[f32; 3]]) -> Tensor<2> {
        self.forward_impl(points, true)
    }
    pub fn evaluate(&self, points: &[[f32; 3]], chunk: usize) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(points.len());
        for pts in points.chunks(chunk) {
            values.extend(self.forward_impl(pts, false).try_into_vec_as::<f32>()?);
            self.trim_memory();
        }
        Ok(values)
    }

    pub fn compute_features_impl(
        &self,
        points: &[[f32; 3]],
        training: bool,
    ) -> (Tensor<2>, Vec<Vec<i32>>, Vec<Vec<f32>>) {
        use rayon::prelude::*;
        let n = points.len();
        assert!(n > 0);
        let xyz = tensor(
            points.iter().flatten().copied().collect(),
            [n, 3],
            &self.device,
        );
        let active_levels = self.active.min(14);
        let (all_indices, all_weights): (Vec<Vec<i32>>, Vec<Vec<f32>>) = (0..active_levels)
            .into_par_iter()
            .map(|level| {
                let res = (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as u32;
                let dense = (res as usize + 1).pow(3) <= 524288;
                let mut indices = Vec::with_capacity(n * 8);
                let mut weights = Vec::with_capacity(n * 8);
                for point in points {
                    let pos = point.map(|p| ((p + 1.) * 0.5).clamp(0., 1. - 1e-7) * res as f32);
                    let base = pos.map(|x| x.floor() as u32);
                    let w = [
                        pos[0] - base[0] as f32,
                        pos[1] - base[1] as f32,
                        pos[2] - base[2] as f32,
                    ];
                    for corner in 0..8 {
                        let mut v = [0u32; 3];
                        let mut weight = 1.;
                        for d in 0..3 {
                            let b = (corner >> d) & 1;
                            v[d] = base[d] + b;
                            weight *= if b == 0 { 1. - w[d] } else { w[d] };
                        }
                        let idx = if dense {
                            v[0] + v[1] * (res + 1) + v[2] * (res + 1) * (res + 1)
                        } else {
                            (v[0] ^ v[1].wrapping_mul(2654435761) ^ v[2].wrapping_mul(805459861))
                                & 524287
                        };
                        indices.push(idx as i32);
                        weights.push(weight);
                    }
                }
                (indices, weights)
            })
            .unzip();

        let mut features = vec![xyz];
        for level in 0..active_levels {
            let table = if training {
                self.p(&format!("encoding.embeddings.{level}"))
            } else {
                self.p(&format!("encoding.embeddings.{level}")).detach()
            };
            let indices = &all_indices[level];
            let weights = &all_weights[level];
            let selected = table
                .select(0, ints(indices.clone(), &self.device))
                .reshape([n, 8, 2]);
            features.push(
                (selected * tensor(weights.clone(), [n, 8, 1], &self.device))
                    .sum_dim(1)
                    .reshape([n, 2]),
            );
        }
        if self.active < 14 {
            features.push(Tensor::zeros([n, (14 - self.active) * 2], &self.device));
        }
        (Tensor::cat(features, 1), all_indices, all_weights)
    }

    pub fn mlp_forward(&self, x: Tensor<2>, training: bool) -> Tensor<2> {
        let param = |name: &str| {
            let p = self.p(name);
            if training { p } else { p.detach() }
        };
        let mut h = x;
        for l in 0..2 {
            let v = param(&format!("lin{l}.weight_v"));
            let g = param(&format!("lin{l}.weight_g"));
            let w = v.clone() * g / (v.clone() * v).sum_dim(1).sqrt();
            h = h.matmul(w.transpose()) + param(&format!("lin{l}.bias"));
            if l == 0 {
                let scaled = h.clone() * 100.;
                let soft = (scaled.clone().clamp_max(20.).exp() + 1.).log() / 100.;
                h = soft.mask_where(scaled.greater_elem(20.), h);
            }
        }
        h
    }

    pub fn compute_features_gpu(&self, xyz: Tensor<2>) -> Tensor<2> {
        let n = xyz.shape().dims::<2>()[0];
        let mut features = Tensor::zeros([n, 31], &self.device);
        let active_levels = self.active.min(14) as u32;
        for level in 0..active_levels {
            let res = (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as u32;
            let dense = (res as usize + 1).pow(3) <= 524288;
            let table = self.p(&format!("encoding.embeddings.{level}")).detach();
            features = crate::kernels::forward_level(
                xyz.clone(),
                table,
                features,
                res,
                level,
                dense,
                level == 0,
            );
        }
        features
    }

    fn forward_impl(&self, points: &[[f32; 3]], training: bool) -> Tensor<2> {
        let (feat, _, _) = self.compute_features_impl(points, training);
        self.mlp_forward(feat, training)
    }

    pub fn evaluate_and_cache(
        &self,
        points: &[[f32; 3]],
        chunk: usize,
    ) -> Result<(Vec<f32>, Vec<FeatureChunk>)> {
        let mut values = Vec::with_capacity(points.len());
        let mut chunks = Vec::with_capacity(points.len().div_ceil(chunk));
        for pts in points.chunks(chunk) {
            let n = pts.len();
            let pts_tensor = tensor(
                pts.iter().flatten().copied().collect(),
                [n, 3],
                &self.device,
            );
            let feat = self.compute_features_gpu(pts_tensor.clone());
            let y = self.mlp_forward(feat.clone(), false);
            values.extend(y.try_into_vec_as::<f32>()?);
            chunks.push(FeatureChunk {
                feat,
                pts_tensor,
                point_count: n,
            });
        }
        Ok((values, chunks))
    }

    pub fn inv_s(&self) -> Tensor<2> {
        (self.p("variance") * 10.).exp().clamp(1e-6, 1e6)
    }

    /// Flush lazy work before measuring, then release unused cached pages. This
    /// is a cache target, not a hard cap on live tensors or driver allocations.
    pub fn trim_memory(&self) {
        let _ = self.device.sync();
        if let Some(usage) = self.device.memory_pool_usage() {
            if usage.bytes_reserved > self.peak_reserved_bytes.get() {
                self.peak_reserved_bytes.set(usage.bytes_reserved);
            }
            if usage.bytes_reserved > 256 * 1024 * 1024 {
                self.device.memory_cleanup();
            }
        }
    }

    pub fn accumulate_gradients_cached(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: &[f32],
        accumulated: &mut Gradients,
    ) {
        let active_levels = self.active.min(14);
        for level in 0..active_levels {
            let table_name = format!("encoding.embeddings.{level}");
            if !accumulated.params.contains_key(&table_name) {
                let entries = self.params.get(&table_name).unwrap().value.dims()[0];
                accumulated
                    .params
                    .insert(table_name, Tensor::zeros([entries, 2], &self.device));
            }
        }

        let mut offset = 0;
        for chunk in cached {
            let n = chunk.point_count;
            let ds = &adjoints[offset..offset + n];
            offset += n;

            let feat = chunk.feat.detach().autodiff().require_grad();
            let output = self.mlp_forward(feat.clone(), true);
            let loss = (output * tensor(ds.to_vec(), [n, 1], &self.device)).sum();
            let grads = loss.backward();

            // 1. MLP parameters
            for name in [
                "lin0.weight_v",
                "lin0.weight_g",
                "lin0.bias",
                "lin1.weight_v",
                "lin1.weight_g",
                "lin1.bias",
            ] {
                let p = self.params.get(name).unwrap();
                if let Some(g) = p.value.grad(&grads) {
                    accumulated
                        .params
                        .entry(name.to_string())
                        .and_modify(|v| *v = (v.clone() + g.clone()).detach())
                        .or_insert(g);
                }
            }

            // 2. Hash table embeddings on GPU
            if let Some(d_feat_tensor) = feat.grad(&grads) {
                for level in 0..active_levels {
                    let res = (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as u32;
                    let dense = (res as usize + 1).pow(3) <= 524288;
                    let table_name = format!("encoding.embeddings.{level}");
                    let table_grad = accumulated.params.remove(&table_name).unwrap();
                    let updated = crate::kernels::backward_level(
                        chunk.pts_tensor.clone(),
                        d_feat_tensor.clone(),
                        table_grad,
                        res,
                        level as u32,
                        dense,
                    );
                    accumulated.params.insert(table_name, updated);
                }
            }
            drop(grads);
        }
    }

    pub fn accumulate_gradients(
        &self,
        points: &[[f32; 3]],
        adjoints: &[f32],
        chunk: usize,
        accumulated: &mut Gradients,
    ) {
        assert_eq!(points.len(), adjoints.len());
        let (_, cached) = self.evaluate_and_cache(points, chunk).unwrap();
        self.accumulate_gradients_cached(cached, adjoints, accumulated);
    }
    pub fn apply_gradients(&mut self, gradients: BTreeMap<String, Tensor<2>>, lr: f64) {
        for (name, grad) in gradients {
            let p = self.params.get_mut(&name).unwrap();
            p.updates += 1;
            let grad = grad.detach();
            p.m = (p.m.clone() * 0.9 + grad.clone() * 0.1).detach();
            p.v = (p.v.clone() * 0.999 + grad.clone() * grad * 0.001).detach();
            let m = p.m.clone() / (1. - 0.9f64.powi(p.updates as i32));
            let v = p.v.clone() / (1. - 0.999f64.powi(p.updates as i32));
            p.value = (p.value.clone().detach() - m / (v.sqrt() + 1e-8) * lr)
                .detach()
                .require_grad();
        }
    }
}

#[derive(Deserialize)]
struct Reference {
    points: Vec<[f32; 3]>,
    sdf: Vec<f32>,
    loss: f32,
    grad_norms: BTreeMap<String, f32>,
}
pub fn verify(path: &Path, device: Device) -> Result<()> {
    let field = Field::load(path, device)?;
    let r: Reference = serde_json::from_slice(&std::fs::read(path.join("reference.json"))?)?;
    let y = field.forward(&r.points);
    let values = y.clone().try_into_vec_as::<f32>()?;
    let max_err = values
        .iter()
        .zip(&r.sdf)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let loss = (y.clone() * y).mean();
    let loss_val: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    println!(
        "field forward max_abs={max_err:0.8e}; loss={loss_val:?}, reference={}",
        r.loss
    );
    ensure!(max_err < 2e-4, "field output mismatch");
    for (name, expected_norm) in r.grad_norms {
        let p = field.params.get(&name).context("missing parameter")?;
        let g = p
            .value
            .grad(&grads)
            .context(format!("missing gradient {name}"))?
            .try_into_vec_as::<f32>()?;
        let expected = read_f32(&path.join(format!("{}.grad.bin", name.replace('.', "_"))))?;
        let err = g
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let norm = g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let max_ref = expected.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!("{name}: grad_max_abs={err:0.3e} norm={norm:0.6e} reference={expected_norm:0.6e}");
        ensure!(err < 2e-4 + max_ref * 2e-3, "gradient mismatch {name}");
    }
    Ok(())
}

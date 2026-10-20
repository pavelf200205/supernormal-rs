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
    pub packed_embedding: Option<Parameter>,
    pub packed_metadata: Option<Tensor<2>>,
    pub level_offsets: Vec<usize>,
    pub level_u32_updates: Vec<usize>,
    pub variance_host: f32,
    inference_weights: std::cell::RefCell<Option<(Tensor<2>, Tensor<2>)>>,
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

    pub fn into_tensors(mut self, field: &Field) -> BTreeMap<String, Tensor<2>> {
        if let Some(packed) = self.params.remove("encoding.packed") {
            let mut offset = 0;
            for level in 0..field.active.min(14) {
                let name = format!("encoding.embeddings.{level}");
                let entries = field.params[&name].value.dims()[0];
                self.params
                    .insert(name, packed.clone().slice([offset..offset + entries, 0..2]));
                offset += entries;
            }
        }
        for (level, data) in self.host_tables.into_iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            let name = format!("encoding.embeddings.{level}");
            let entries = data.len() / 2;
            let g = tensor(data, [entries, 2], &field.device);
            self.params
                .entry(name)
                .and_modify(|v| *v = (v.clone() + g.clone()).without_autodiff())
                .or_insert(g);
        }
        self.params
    }
}

pub struct FeatureChunk {
    pub feat: Tensor<2>,
    pub preactivation: Tensor<2>,
    pub pts_tensor: Tensor<2>,
    pub point_count: usize,
    pub padded_count: usize,
}

impl Field {
    fn build_packed_embedding(
        params: &BTreeMap<String, Parameter>,
        device: &Device,
    ) -> (Parameter, Tensor<2>, Vec<usize>, Vec<usize>) {
        let mut values = Vec::with_capacity(14);
        let mut ms = Vec::with_capacity(14);
        let mut vs = Vec::with_capacity(14);
        let mut metadata = Vec::with_capacity(28);
        let mut offsets = Vec::with_capacity(15);
        let mut updates = Vec::with_capacity(14);
        let mut offset = 0usize;
        offsets.push(0);

        for level in 0..14 {
            let res = (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as usize;
            let name = format!("encoding.embeddings.{level}");
            let p = &params[&name];
            let rows = p.value.dims()[0];
            metadata.extend([res as f32, offset as f32]);
            offset += rows;
            offsets.push(offset);
            updates.push(p.updates);
            values.push(p.value.clone().without_autodiff());
            ms.push(p.m.clone().without_autodiff());
            vs.push(p.v.clone().without_autodiff());
        }

        let packed_param = Parameter {
            value: Tensor::cat(values, 0),
            m: Tensor::cat(ms, 0),
            v: Tensor::cat(vs, 0),
            updates: updates[0],
        };
        let meta_tensor = tensor(metadata, [14, 2], device);
        (packed_param, meta_tensor, offsets, updates)
    }

    pub fn sync_packed_to_params(&mut self) {
        if let Some(packed) = &self.packed_embedding {
            for level in 0..14 {
                let start = self.level_offsets[level];
                let end = self.level_offsets[level + 1];
                let name = format!("encoding.embeddings.{level}");
                if let Some(p) = self.params.get_mut(&name) {
                    p.value = packed
                        .value
                        .clone()
                        .without_autodiff()
                        .slice([start..end, 0..2])
                        .autodiff()
                        .require_grad();
                    p.m = packed
                        .m
                        .clone()
                        .without_autodiff()
                        .slice([start..end, 0..2]);
                    p.v = packed
                        .v
                        .clone()
                        .without_autodiff()
                        .slice([start..end, 0..2]);
                    p.updates = self.level_u32_updates[level];
                }
            }
        }
    }

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

        let (packed_embedding, packed_metadata, level_offsets, level_u32_updates) =
            Self::build_packed_embedding(&params, &device);

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
            packed_embedding: Some(packed_embedding),
            packed_metadata: Some(packed_metadata),
            level_offsets,
            level_u32_updates,
            variance_host: 0.5,
            inference_weights: std::cell::RefCell::new(None),
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

        let variance_host: f32 = params["variance"]
            .value
            .clone()
            .without_autodiff()
            .into_scalar();
        let (packed_embedding, packed_metadata, level_offsets, level_u32_updates) =
            Self::build_packed_embedding(&params, &device);

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
            packed_embedding: Some(packed_embedding),
            packed_metadata: Some(packed_metadata),
            level_offsets,
            level_u32_updates,
            variance_host,
            inference_weights: std::cell::RefCell::new(None),
        })
    }

    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.sync_packed_to_params();
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

    #[inline]
    fn static_bucket_size(n: usize, chunk: usize) -> usize {
        if n == chunk || chunk < 1024 {
            return n;
        }
        n.next_power_of_two().max(1024).min(chunk)
    }

    /// One mapping/readback for the entire query, rather than a host wait for
    /// every chunk. Only the small SDF outputs are concatenated, not features.
    fn read_outputs(outputs: Vec<(usize, Tensor<2>)>) -> Result<Vec<f32>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        let outputs = outputs
            .into_iter()
            .map(|(n, y)| {
                if y.dims()[0] == n {
                    y
                } else {
                    y.slice([0..n, 0..1])
                }
            })
            .collect();
        Ok(Tensor::cat(outputs, 0).try_into_vec_as::<f32>()?)
    }

    fn enqueue_evaluation(&self, points: &[[f32; 3]], chunk: usize) -> Vec<(usize, Tensor<2>)> {
        let mut pending = Vec::with_capacity(points.len().div_ceil(chunk));
        for pts in points.chunks(chunk) {
            let n = pts.len();
            let padded_n = Self::static_bucket_size(n, chunk);
            let mut flat = Vec::with_capacity(padded_n * 3);
            flat.extend(pts.iter().flatten().copied());
            flat.resize(padded_n * 3, 0.0);

            let xyz = tensor(flat, [padded_n, 3], &self.device);
            let features = self.compute_features_gpu(xyz);
            let y = self.mlp_forward(features, false);
            pending.push((n, y));
        }

        pending
    }

    pub fn evaluate(&self, points: &[[f32; 3]], chunk: usize) -> Result<Vec<f32>> {
        Self::read_outputs(self.enqueue_evaluation(points, chunk))
    }

    /// Read variance in the same transfer as visibility SDFs. This is the first
    /// CPU consumer of the previous Adam update, after ray/point preparation.
    pub fn evaluate_visibility(&mut self, points: &[[f32; 3]], chunk: usize) -> Result<Vec<f32>> {
        let mut pending = self.enqueue_evaluation(points, chunk);
        pending.push((1, self.p("variance").without_autodiff()));
        let mut values = Self::read_outputs(pending)?;
        self.variance_host = values.pop().expect("visibility query includes variance");
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
                self.p(&format!("encoding.embeddings.{level}"))
                    .without_autodiff()
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
        let h = self.mlp_hidden(x, training);
        self.mlp_output(h, training)
    }

    fn effective_weight(&self, layer: usize, training: bool) -> Tensor<2> {
        if training {
            let v = self.p(&format!("lin{layer}.weight_v"));
            let g = self.p(&format!("lin{layer}.weight_g"));
            return v.clone() * g / (v.clone() * v).sum_dim(1).sqrt();
        }
        if self.inference_weights.borrow().is_none() {
            let build = |layer| {
                let v = self.p(&format!("lin{layer}.weight_v")).without_autodiff();
                let g = self.p(&format!("lin{layer}.weight_g")).without_autodiff();
                (v.clone() * g / (v.clone() * v).sum_dim(1).sqrt()).without_autodiff()
            };
            *self.inference_weights.borrow_mut() = Some((build(0), build(1)));
        }
        let cache = self.inference_weights.borrow();
        let weights = cache.as_ref().unwrap();
        if layer == 0 {
            weights.0.clone()
        } else {
            weights.1.clone()
        }
    }

    fn mlp_hidden(&self, x: Tensor<2>, training: bool) -> Tensor<2> {
        let param = |name: &str| {
            let p = self.p(name);
            if training { p } else { p.without_autodiff() }
        };
        let w = self.effective_weight(0, training);
        let h = x.matmul(w.transpose()) + param("lin0.bias");
        Self::softplus(h)
    }

    fn softplus(h: Tensor<2>) -> Tensor<2> {
        let scaled = h.clone() * 100.;
        let soft = (scaled.clone().clamp_max(20.).exp() + 1.).log() / 100.;
        soft.mask_where(scaled.greater_elem(20.), h)
    }

    fn mlp_output(&self, h: Tensor<2>, training: bool) -> Tensor<2> {
        let bias = self.p("lin1.bias");
        let bias = if training {
            bias
        } else {
            bias.without_autodiff()
        };
        let w = self.effective_weight(1, training);
        h.matmul(w.transpose()) + bias
    }

    fn output_weight_gradients(
        &self,
        hidden: Tensor<2>,
        adjoints: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        self.weight_gradients(1, adjoints.transpose().matmul(hidden))
    }

    fn weight_gradients(&self, layer: usize, dw: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let v = self.p(&format!("lin{layer}.weight_v")).without_autodiff();
        let g = self.p(&format!("lin{layer}.weight_g")).without_autodiff();
        let norm_squared = (v.clone() * v.clone()).sum_dim(1);
        let norm = norm_squared.clone().sqrt();
        let dot = (dw.clone() * v.clone()).sum_dim(1);
        let dg = dot.clone() / norm.clone();
        let dv = (dw - v * (dot / norm_squared)) * (g / norm);
        (dv.without_autodiff(), dg.without_autodiff())
    }

    fn packed_tables(&self) -> (Tensor<2>, Tensor<2>) {
        let tables = self.packed_embedding.as_ref().unwrap().value.clone();
        let metadata = self.packed_metadata.as_ref().unwrap().clone();
        (tables, metadata)
    }

    pub fn compute_features_gpu(&self, xyz: Tensor<2>) -> Tensor<2> {
        let n = xyz.shape().dims::<2>()[0];
        let active = self.active.min(14);
        if active == 0 {
            return Tensor::cat(vec![xyz, Tensor::zeros([n, 28], &self.device)], 1);
        }
        let features = Tensor::zeros([n, 31], &self.device);
        let (tables, metadata) = self.packed_tables();
        crate::kernels::forward_packed(xyz, tables, metadata, features, active)
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
        let num_chunks = points.len().div_ceil(chunk);
        let mut chunks = Vec::with_capacity(num_chunks);
        let mut pending_outputs = Vec::with_capacity(num_chunks);

        let w0_t = self.effective_weight(0, false).transpose();
        let b0 = self.p("lin0.bias").without_autodiff();

        for pts in points.chunks(chunk) {
            let n = pts.len();
            let padded_n = Self::static_bucket_size(n, chunk);
            let mut flat = Vec::with_capacity(padded_n * 3);
            flat.extend(pts.iter().flatten().copied());
            flat.resize(padded_n * 3, 0.0);

            let pts_tensor = tensor(flat, [padded_n, 3], &self.device);
            let feat = self.compute_features_gpu(pts_tensor.clone());
            let preactivation = feat.clone().matmul(w0_t.clone()) + b0.clone();
            let hidden = Self::softplus(preactivation.clone());
            let y = self.mlp_output(hidden, false);

            pending_outputs.push((n, y));
            chunks.push(FeatureChunk {
                feat,
                preactivation,
                pts_tensor,
                point_count: n,
                padded_count: padded_n,
            });
        }

        Ok((Self::read_outputs(pending_outputs)?, chunks))
    }

    #[inline]
    pub fn inv_s_host(&self) -> f32 {
        (self.variance_host * 10.).exp().clamp(1e-6, 1e6)
    }

    pub fn trim_memory(&self) {
        let _ = self.device.sync();
        if let Some(usage) = self.device.memory_pool_usage() {
            self.peak_reserved_bytes
                .set(self.peak_reserved_bytes.get().max(usage.bytes_reserved));
            if usage.bytes_reserved > 1024 * 1024 * 1024 {
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
        let (_, metadata) = self.packed_tables();
        if !accumulated.params.contains_key("encoding.packed") {
            let entries = self.level_offsets[active_levels];
            accumulated.params.insert(
                "encoding.packed".into(),
                Tensor::zeros([entries, 2], &self.device),
            );
        }

        let w0 = self.effective_weight(0, false);
        let w1 = self.effective_weight(1, false);

        let mut offset = 0;
        let mut dw0_total: Option<Tensor<2>> = None;
        let mut dw1_total: Option<Tensor<2>> = None;
        let mut db0_total: Option<Tensor<2>> = None;
        let mut db1_total: Option<Tensor<2>> = None;

        for chunk in cached {
            let n = chunk.point_count;
            let padded_n = chunk.padded_count;
            let ds = &adjoints[offset..offset + n];
            offset += n;

            // Skip chunk completely if all adjoints in this chunk are zero
            if ds.iter().all(|&x| x == 0.0) {
                continue;
            }

            let mut ds_padded = Vec::with_capacity(padded_n);
            ds_padded.extend_from_slice(ds);
            ds_padded.resize(padded_n, 0.0);

            let ds_tensor = tensor(ds_padded, [padded_n, 1], &self.device);
            let hidden = Self::softplus(chunk.preactivation.clone());
            let dw1_chunk = ds_tensor.clone().transpose().matmul(hidden);

            let scaled = chunk.preactivation * 100.;
            let sigmoid = ((-scaled.clone()).exp() + 1.).recip();
            let derivative = sigmoid.mask_fill(scaled.greater_elem(20.), 1.);
            let dh = ds_tensor.clone() * w1.clone() * derivative;
            let dw0_chunk = dh.clone().transpose().matmul(chunk.feat);
            let d_feat_tensor = dh.clone().matmul(w0.clone());

            let db0_chunk = dh.sum_dim(0);
            let db1_chunk = ds_tensor.sum_dim(0);

            dw0_total = Some(match dw0_total {
                Some(acc) => (acc + dw0_chunk).without_autodiff(),
                None => dw0_chunk,
            });
            dw1_total = Some(match dw1_total {
                Some(acc) => (acc + dw1_chunk).without_autodiff(),
                None => dw1_chunk,
            });
            db0_total = Some(match db0_total {
                Some(acc) => (acc + db0_chunk).without_autodiff(),
                None => db0_chunk,
            });
            db1_total = Some(match db1_total {
                Some(acc) => (acc + db1_chunk).without_autodiff(),
                None => db1_chunk,
            });

            if active_levels > 0 {
                let table_grad = accumulated.params.remove("encoding.packed").unwrap();
                let updated = crate::kernels::backward_packed(
                    chunk.pts_tensor,
                    d_feat_tensor,
                    metadata.clone(),
                    table_grad,
                    active_levels,
                );
                accumulated.params.insert("encoding.packed".into(), updated);
            }
        }

        if let (Some(dw0), Some(dw1), Some(db0), Some(db1)) =
            (dw0_total, dw1_total, db0_total, db1_total)
        {
            let (dv0, dg0) = self.weight_gradients(0, dw0);
            let (dv1, dg1) = self.weight_gradients(1, dw1);

            for (name, grad) in [
                ("lin0.weight_v", dv0),
                ("lin0.weight_g", dg0),
                ("lin0.bias", db0),
                ("lin1.weight_v", dv1),
                ("lin1.weight_g", dg1),
                ("lin1.bias", db1),
            ] {
                accumulated
                    .params
                    .entry(name.to_string())
                    .and_modify(|v| *v = (v.clone() + grad.clone()).without_autodiff())
                    .or_insert(grad);
            }
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

    pub fn apply_gradients_packed(&mut self, mut gradients: Gradients, lr: f64) {
        self.inference_weights.get_mut().take();

        if let Some(packed_grad) = gradients.params.remove("encoding.packed") {
            let active = self.active.min(14);
            if active > 0 {
                // Upload O(levels) scalars, rather than two O(table entries)
                // correction tensors and full-table slice/copy expressions.
                let mut corrections = Vec::with_capacity(active * 3);
                for level in 0..active {
                    self.level_u32_updates[level] += 1;
                    let updates = self.level_u32_updates[level] as i32;
                    corrections.extend([
                        self.level_offsets[level + 1] as f32,
                        (lr / (1.0 - 0.9f64.powi(updates))) as f32,
                        (1.0 / (1.0 - 0.999f64.powi(updates)).sqrt()) as f32,
                    ]);
                }
                let packed = self.packed_embedding.take().unwrap();
                let [value, m, v] = crate::kernels::adam_packed(
                    packed.value,
                    packed.m,
                    packed.v,
                    packed_grad.without_autodiff(),
                    tensor(corrections, [active, 3], &self.device),
                );
                self.packed_embedding = Some(Parameter {
                    value,
                    m,
                    v,
                    updates: self.level_u32_updates[0],
                });
            }
        }
        for (name, grad) in gradients.params {
            let p = self.params.get_mut(&name).unwrap();
            p.updates += 1;
            let grad = grad.without_autodiff();
            p.m = (p.m.clone() * 0.9 + grad.clone() * 0.1).without_autodiff();
            p.v = (p.v.clone() * 0.999 + grad.clone() * grad * 0.001).without_autodiff();
            let m = p.m.clone() / (1. - 0.9f64.powi(p.updates as i32));
            let v = p.v.clone() / (1. - 0.999f64.powi(p.updates as i32));
            let updated = (p.value.clone().without_autodiff() - m / (v.sqrt() + 1e-8) * lr)
                .without_autodiff();
            // Visibility refreshes the variance mirror in its SDF readback.
            // Do not insert a scalar readback/wait into the optimizer here.
            p.value = updated.autodiff().require_grad();
        }
    }

    pub fn apply_gradients(&mut self, gradients: BTreeMap<String, Tensor<2>>, lr: f64) {
        self.inference_weights.get_mut().take();
        for (name, grad) in gradients {
            let p = self.params.get_mut(&name).unwrap();
            p.updates += 1;
            let grad = grad.without_autodiff();
            p.m = (p.m.clone() * 0.9 + grad.clone() * 0.1).without_autodiff();
            p.v = (p.v.clone() * 0.999 + grad.clone() * grad * 0.001).without_autodiff();
            let m = p.m.clone() / (1. - 0.9f64.powi(p.updates as i32));
            let v = p.v.clone() / (1. - 0.999f64.powi(p.updates as i32));
            let updated = (p.value.clone().without_autodiff() - m / (v.sqrt() + 1e-8) * lr)
                .without_autodiff();
            if name == "variance" {
                self.variance_host = updated.clone().into_scalar();
            }
            p.value = updated.autodiff().require_grad();
        }
        let (packed_embedding, packed_metadata, level_offsets, level_u32_updates) =
            Self::build_packed_embedding(&self.params, &self.device);
        self.packed_embedding = Some(packed_embedding);
        self.packed_metadata = Some(packed_metadata);
        self.level_offsets = level_offsets;
        self.level_u32_updates = level_u32_updates;
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
    let (gpu_values, gpu_cache) = field.evaluate_and_cache(&r.points, 8192)?;
    ensure!(
        gpu_cache
            .iter()
            .all(|c| !c.preactivation.is_require_grad() && !c.feat.is_require_grad()),
        "cached inference unexpectedly tracks gradients"
    );
    drop(gpu_cache);
    let gpu_error = gpu_values
        .iter()
        .zip(&r.sdf)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("packed GPU field forward max_abs={gpu_error:0.8e}");
    ensure!(gpu_error < 2e-4, "packed GPU field output mismatch");
    let max_err = values
        .iter()
        .zip(&r.sdf)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let loss = (y.clone() * y).mean();
    let loss_val: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    let (features, _, _) = field.compute_features_impl(&r.points, false);
    let hidden = field.mlp_hidden(features, false);
    let adjoint_values: Vec<f32> = values
        .iter()
        .map(|&sdf| 2. * sdf / values.len() as f32)
        .collect();
    let adjoints = tensor(adjoint_values.clone(), [values.len(), 1], &field.device);
    let (output_dv, output_dg) = field.output_weight_gradients(hidden, adjoints);
    let mut packed_gradients = Gradients::default();
    field.accumulate_gradients(&r.points, &adjoint_values, 31, &mut packed_gradients);
    let packed_gradients = packed_gradients.into_tensors(&field);
    println!(
        "field forward max_abs={max_err:0.8e}; loss={loss_val:?}, reference={}",
        r.loss
    );
    ensure!(max_err < 2e-4, "field output mismatch");
    for (name, expected_norm) in r.grad_norms {
        let p = field.params.get(&name).context("missing parameter")?;
        let mut g = p
            .value
            .grad(&grads)
            .context(format!("missing gradient {name}"))?
            .try_into_vec_as::<f32>()?;
        if name == "lin1.weight_g" || name == "lin1.weight_v" {
            let analytic = if name == "lin1.weight_g" {
                output_dg.clone()
            } else {
                output_dv.clone()
            };
            println!(
                "{name}: Burn autodiff norm={}",
                g.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
            g = analytic.try_into_vec_as::<f32>()?;
        }
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
        let packed = packed_gradients
            .get(&name)
            .context(format!("missing packed gradient {name}"))?
            .clone()
            .try_into_vec_as::<f32>()?;
        let packed_err = packed
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{name}: packed_grad_max_abs={packed_err:0.3e}");
        ensure!(
            packed_err < 2e-4 + max_ref * 2e-3,
            "packed gradient mismatch {name}"
        );
    }
    Ok(())
}

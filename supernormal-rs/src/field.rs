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

fn default_scale_cutoff() -> f32 {
    0.45
}
fn default_base_resolution() -> usize {
    32
}
fn default_max_resolution() -> usize {
    1175
}
fn default_log2_hashmap_size() -> u32 {
    19
}

pub fn hash_level_resolution(base_res: usize, max_res: usize, level: usize) -> usize {
    if base_res == 32 && max_res == 1175 {
        (32f64 * 1.3195079107728942f64.powi(level as i32)).floor() as usize
    } else {
        let growth = (max_res as f64 / base_res as f64).powf(1.0 / 13.0);
        (base_res as f64 * growth.powi(level as i32)).floor() as usize
    }
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub step: usize,
    pub active_levels: usize,
    pub parameters: Vec<ParameterMeta>,
    #[serde(default)]
    pub scale_encoding: bool,
    #[serde(default = "default_scale_cutoff")]
    pub scale_cutoff: f32,
    #[serde(default = "default_base_resolution")]
    pub base_resolution: usize,
    #[serde(default = "default_max_resolution")]
    pub max_resolution: usize,
    #[serde(default = "default_log2_hashmap_size")]
    pub log2_hashmap_size: u32,
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
    pub is_cpu: bool,
    pub peak_reserved_bytes: std::cell::Cell<u64>,
    pub packed_embedding: Option<Parameter>,
    pub packed_metadata: Option<Tensor<2>>,
    pub level_offsets: Vec<usize>,
    pub level_u32_updates: Vec<usize>,
    pub variance_host: f32,
    pub fused_field: bool,
    pub cpu_visibility: bool,
    pub tiled_gradients: bool,
    pub local_hash: bool,
    pub compact_gradients: bool,
    pub wide_gradient_tiles: bool,
    pub corner_parallel: bool,
    pub aggregate_hash: bool,
    pub level_adam: bool,
    pub skip_untouched_adam: bool,
    pub direct_adjoint_views: bool,
    pub active_parameter_columns: bool,
    pub selective_backward_weights: bool,
    pub point_tiles: usize,
    pub subgroup_reduce: bool,
    pub aggregation_levels: usize,
    pub cross_corners: bool,
    pub recompute_parameter_gradients: bool,
    pub reuse_endpoints: bool,
    pub chunked_eval: bool,
    pub feature_only_cache: bool,
    pub fused_feature_replay: bool,
    pub fused_hash_replay: bool,
    pub selective_forward_weights: bool,
    pub device_occupancy_update: bool,
    pub register_gradient_tiles: bool,
    pub wide_parameter_rows: bool,
    pub hybrid_activation_cache: bool,
    pub fused_mlp_packing: bool,
    pub fused_backward_params: bool,
    pub front_prune: bool,
    pub scale_encoding: bool,
    pub scale_cutoff: f32,
    pub base_resolution: usize,
    pub max_resolution: usize,
    pub log2_hashmap_size: u32,
    packed_mlp: std::cell::RefCell<Option<Tensor<2>>>,
    packed_mlp_state: Option<(Tensor<2>, usize)>,
    inference_weights: std::cell::RefCell<Option<(Tensor<2>, Tensor<2>)>>,
    persistent_table_grad: std::cell::RefCell<Option<Tensor<2>>>,
}

pub fn tensor<const D: usize>(data: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    let total: usize = shape.iter().product();
    Tensor::<1>::from_data(TensorData::new(data, [total]), device).reshape(shape)
}

/// Returns `true` if `device` is backed by Burn's CubeCL GPU backend (`Fusion<CubeBackend>`),
/// and `false` if it is a pure-CPU backend such as `Device::flex()`.
pub fn is_gpu_device(device: &Device) -> bool {
    Tensor::<1>::empty([1], device)
        .inner()
        .try_into_primitive::<burn_fusion::Fusion<burn_cubecl::CubeBackend>>()
        .is_ok()
}

/// Allocate an uninitialized 2D tensor with guaranteed contiguous row-major
/// strides on both WGPU and CUDA (avoiding CUDA's 512-byte pitched 2D layout).
#[inline]
pub fn empty_2d(rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::<1>::empty([rows * cols], device).reshape([rows, cols])
}

/// Allocate a zeroed 2D tensor with guaranteed contiguous row-major strides.
#[inline]
pub fn zeros_2d(rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::<1>::zeros([rows * cols], device).reshape([rows, cols])
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
    /// Raw `(dw0, db0, dw1)` for single-workgroup fused WeightNorm + Adam + pack.
    pub raw_mlp: Option<(Tensor<2>, Tensor<2>, Tensor<2>)>,
}

impl Gradients {
    pub fn insert(&mut self, name: String, tensor: Tensor<2>) {
        self.params.insert(name, tensor);
    }

    pub fn expand_raw_mlp(&mut self, field: &Field) {
        if let Some((dw0, db0, dw1)) = self.raw_mlp.take() {
            let (dv0, dg0) = field.weight_gradients(0, dw0);
            let (dv1, dg1) = field.weight_gradients(1, dw1);
            for (name, grad) in [
                ("lin0.weight_v", dv0),
                ("lin0.weight_g", dg0),
                ("lin0.bias", db0),
                ("lin1.weight_v", dv1),
                ("lin1.weight_g", dg1),
            ] {
                self.params
                    .entry(name.to_string())
                    .and_modify(|v| *v = (v.clone() + grad.clone()).without_autodiff())
                    .or_insert(grad);
            }
        }
    }

    pub fn into_tensors(mut self, field: &Field) -> BTreeMap<String, Tensor<2>> {
        self.expand_raw_mlp(field);
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
    pub feat: Option<Tensor<2>>,
    pub preactivation: Option<Tensor<2>>,
    pub pts_tensor: Tensor<2>,
    pub point_count: usize,
    pub padded_count: usize,
}

/// Submitted forward work whose small SDF output has not yet been mapped.
pub struct PendingEvaluation {
    outputs: Vec<(usize, Tensor<2>)>,
    cache: Vec<FeatureChunk>,
}

impl PendingEvaluation {
    pub fn into_device(self) -> (Tensor<2>, Vec<FeatureChunk>) {
        let mut outputs: Vec<Tensor<2>> = self
            .outputs
            .into_iter()
            .map(|(n, y)| {
                if y.dims()[0] == n {
                    y
                } else {
                    y.slice([0..n, 0..1])
                }
            })
            .collect();
        let sdf = if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 0)
        };
        (sdf, self.cache)
    }

    pub fn finish(self) -> Result<(Vec<f32>, Vec<FeatureChunk>)> {
        Ok((Field::read_outputs(self.outputs)?, self.cache))
    }
}

/// Baseline activation budget: three full feature + preactivation chunks.
/// Large batches can retain features alone, reserving one preactivation workspace
/// within that budget. Uncached chunks retain the existing streamed fallback.
const MAX_CACHED_CHUNKS: usize = 6;

impl Field {
    #[inline]
    pub fn hashmap_size(&self) -> usize {
        1usize << self.log2_hashmap_size
    }

    fn build_packed_embedding(
        params: &BTreeMap<String, Parameter>,
        base_resolution: usize,
        max_resolution: usize,
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
            let res = hash_level_resolution(base_resolution, max_resolution, level);
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

    const MLP_PARAM_SPEC: [(&'static str, usize, [usize; 2]); 7] = [
        ("lin0.weight_v", 1984, [64, 31]),
        ("lin0.weight_g", 64, [64, 1]),
        ("lin0.bias", 64, [1, 64]),
        ("lin1.weight_v", 64, [1, 64]),
        ("lin1.weight_g", 1, [1, 1]),
        ("lin1.bias", 1, [1, 1]),
        ("variance", 1, [1, 1]),
    ];

    fn build_packed_mlp_state(
        params: &BTreeMap<String, Parameter>,
    ) -> Option<(Tensor<2>, usize)> {
        let updates = params.get("lin0.weight_v")?.updates;
        if Self::MLP_PARAM_SPEC
            .iter()
            .any(|(name, _, _)| params.get(*name).is_none_or(|p| p.updates != updates))
        {
            return None;
        }
        let mut vals = Vec::with_capacity(7);
        let mut ms = Vec::with_capacity(7);
        let mut vs = Vec::with_capacity(7);
        for &(name, len, _) in &Self::MLP_PARAM_SPEC {
            let p = &params[name];
            vals.push(p.value.clone().without_autodiff().reshape([1, len]));
            ms.push(p.m.clone().without_autodiff().reshape([1, len]));
            vs.push(p.v.clone().without_autodiff().reshape([1, len]));
        }
        let state = Tensor::cat(
            vec![
                Tensor::cat(vals, 1),
                Tensor::cat(ms, 1),
                Tensor::cat(vs, 1),
            ],
            0,
        );
        Some((state, updates))
    }

    fn sync_packed_mlp_to_params(&mut self) {
        if let Some((state, updates)) = &self.packed_mlp_state {
            let mut col = 0usize;
            for &(name, len, shape) in &Self::MLP_PARAM_SPEC {
                if let Some(p) = self.params.get_mut(name) {
                    p.value = state
                        .clone()
                        .slice([0..1, col..col + len])
                        .reshape(shape)
                        .autodiff()
                        .require_grad();
                    p.m = state.clone().slice([1..2, col..col + len]).reshape(shape);
                    p.v = state.clone().slice([2..3, col..col + len]).reshape(shape);
                    p.updates = *updates;
                }
                col += len;
            }
        }
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
        self.sync_packed_mlp_to_params();
    }

    #[allow(dead_code)]
    pub fn new(device: Device, seed: u64) -> Self {
        Self::new_with_encoding(
            device,
            seed,
            default_base_resolution(),
            default_max_resolution(),
            default_log2_hashmap_size(),
        )
    }

    pub fn new_with_encoding(
        device: Device,
        seed: u64,
        base_resolution: usize,
        max_resolution: usize,
        log2_hashmap_size: u32,
    ) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut params = BTreeMap::new();
        let mut add = |name: String, values: Vec<f32>, shape: [usize; 2]| {
            params.insert(
                name,
                Parameter {
                    value: tensor(values, shape, &device).require_grad(),
                    m: zeros_2d(shape[0], shape[1], &device),
                    v: zeros_2d(shape[0], shape[1], &device),
                    updates: 0,
                },
            );
        };
        let hashmap_size = 1usize << log2_hashmap_size;
        for l in 0..14usize {
            let res = hash_level_resolution(base_resolution, max_resolution, l);
            let entries = (res + 1).saturating_pow(3).min(hashmap_size);
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
            Self::build_packed_embedding(&params, base_resolution, max_resolution, &device);
        let packed_mlp_state = Self::build_packed_mlp_state(&params);

        let is_cpu = !is_gpu_device(&device);
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
            is_cpu,
            peak_reserved_bytes: std::cell::Cell::new(peak),
            packed_embedding: Some(packed_embedding),
            packed_metadata: Some(packed_metadata),
            level_offsets,
            level_u32_updates,
            variance_host: 0.5,
            fused_field: !is_cpu,
            cpu_visibility: is_cpu,
            tiled_gradients: !is_cpu,
            local_hash: true,
            compact_gradients: true,
            wide_gradient_tiles: true,
            corner_parallel: true,
            aggregate_hash: true,
            level_adam: true,
            skip_untouched_adam: true,
            direct_adjoint_views: true,
            active_parameter_columns: true,
            selective_backward_weights: true,
            point_tiles: 4,
            subgroup_reduce: true,
            aggregation_levels: 14,
            cross_corners: false,
            recompute_parameter_gradients: false,
            reuse_endpoints: true,
            chunked_eval: false,
            feature_only_cache: true,
            fused_feature_replay: true,
            fused_hash_replay: true,
            selective_forward_weights: true,
            device_occupancy_update: !is_cpu,
            register_gradient_tiles: true,
            wide_parameter_rows: true,
            hybrid_activation_cache: false,
            fused_mlp_packing: !is_cpu,
            fused_backward_params: !is_cpu,
            front_prune: true,
            scale_encoding: false,
            scale_cutoff: default_scale_cutoff(),
            base_resolution,
            max_resolution,
            log2_hashmap_size,
            packed_mlp: std::cell::RefCell::new(None),
            packed_mlp_state,
            inference_weights: std::cell::RefCell::new(None),
            persistent_table_grad: std::cell::RefCell::new(None),
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
                    zeros_2d(p.shape[0], p.shape[1], &device),
                    zeros_2d(p.shape[0], p.shape[1], &device),
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
            Self::build_packed_embedding(&params, m.base_resolution, m.max_resolution, &device);
        let packed_mlp_state = Self::build_packed_mlp_state(&params);

        let is_cpu = !is_gpu_device(&device);
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
            is_cpu,
            peak_reserved_bytes: std::cell::Cell::new(peak),
            packed_embedding: Some(packed_embedding),
            packed_metadata: Some(packed_metadata),
            level_offsets,
            level_u32_updates,
            variance_host,
            fused_field: !is_cpu,
            cpu_visibility: is_cpu,
            tiled_gradients: !is_cpu,
            local_hash: true,
            compact_gradients: true,
            wide_gradient_tiles: true,
            corner_parallel: true,
            aggregate_hash: true,
            level_adam: true,
            skip_untouched_adam: true,
            direct_adjoint_views: true,
            active_parameter_columns: true,
            selective_backward_weights: true,
            point_tiles: 4,
            subgroup_reduce: true,
            aggregation_levels: 14,
            cross_corners: false,
            recompute_parameter_gradients: false,
            reuse_endpoints: true,
            chunked_eval: false,
            feature_only_cache: true,
            fused_feature_replay: true,
            fused_hash_replay: true,
            selective_forward_weights: true,
            device_occupancy_update: !is_cpu,
            register_gradient_tiles: true,
            wide_parameter_rows: true,
            hybrid_activation_cache: false,
            fused_mlp_packing: !is_cpu,
            fused_backward_params: !is_cpu,
            front_prune: true,
            scale_encoding: m.scale_encoding,
            scale_cutoff: m.scale_cutoff,
            base_resolution: m.base_resolution,
            max_resolution: m.max_resolution,
            log2_hashmap_size: m.log2_hashmap_size,
            packed_mlp: std::cell::RefCell::new(None),
            packed_mlp_state,
            inference_weights: std::cell::RefCell::new(None),
            persistent_table_grad: std::cell::RefCell::new(None),
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
            scale_encoding: self.scale_encoding,
            scale_cutoff: self.scale_cutoff,
            base_resolution: self.base_resolution,
            max_resolution: self.max_resolution,
            log2_hashmap_size: self.log2_hashmap_size,
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
    pub fn static_bucket_size(n: usize, chunk: usize) -> usize {
        if n == chunk || chunk < 1024 {
            return n.max(1);
        }
        n.next_power_of_two().max(1024).min(chunk)
    }

    /// Compute the total allocated row capacity for a device-resident point buffer
    /// `[count, 3]` so its tail chunk of size `count % chunk` already has exact
    /// `static_bucket_size(tail, chunk)` capacity, eliminating `Tensor::zeros` +
    /// `Tensor::cat` padding copies during chunked evaluation.
    #[inline]
    pub fn padded_point_capacity(count: usize, chunk: usize) -> usize {
        if count == 0 {
            return 1024.min(chunk.max(1));
        }
        let full_chunks = count / chunk;
        let rem = count % chunk;
        if rem == 0 {
            full_chunks * chunk
        } else {
            full_chunks * chunk + Self::static_bucket_size(rem, chunk)
        }
    }

    /// One mapping/readback for the entire query, rather than a host wait for
    /// every chunk. Only the small SDF outputs are concatenated, not features.
    fn read_outputs(outputs: Vec<(usize, Tensor<2>)>) -> Result<Vec<f32>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        let outputs: Vec<Tensor<2>> = outputs
            .into_iter()
            .map(|(n, y)| {
                if y.dims()[0] == n {
                    y
                } else {
                    y.slice([0..n, 0..1])
                }
            })
            .collect();
        let cat = if outputs.len() == 1 {
            outputs.into_iter().next().unwrap()
        } else {
            Tensor::cat(outputs, 0)
        };
        Ok(cat.try_into_vec_as::<f32>()?)
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
            let y = if self.fused_field {
                let [y, _, _] = self.fused_forward_valid(xyz, n, false);
                y
            } else {
                let features = self.compute_features_gpu(xyz);
                self.mlp_forward(features, false)
            };
            pending.push((n, y));
        }

        pending
    }

    /// Evaluate points that are already device-resident `[padded_cap, 3]` with
    /// `valid_count` active points without downloading or re-uploading coordinates.
    pub fn evaluate_device_tensor_valid(
        &self,
        points: Tensor<2>,
        valid_count: usize,
        chunk: usize,
    ) -> Tensor<2> {
        let total_rows = points.dims()[0];
        if valid_count <= chunk && total_rows == Self::static_bucket_size(valid_count, chunk) {
            return if self.fused_field {
                let [y, _, _] = self.fused_forward_valid(points, valid_count, false);
                y
            } else {
                let features = self.compute_features_gpu(points);
                let y = self.mlp_forward(features, false);
                if y.dims()[0] == valid_count {
                    y
                } else {
                    y.slice([0..valid_count, 0..1])
                }
            };
        }
        let pt_cols = points.dims()[1];
        let mut outputs: Vec<Tensor<2>> = (0..valid_count)
            .step_by(chunk)
            .map(|start| {
                let n = (valid_count - start).min(chunk);
                let padded_n = Self::static_bucket_size(n, chunk);
                let xyz = if start == 0 && total_rows == padded_n {
                    points.clone()
                } else if start + padded_n <= total_rows {
                    points.clone().slice([start..start + padded_n, 0..pt_cols])
                } else {
                    let pts = points.clone().slice([start..start + n, 0..pt_cols]);
                    if padded_n == n {
                        pts
                    } else {
                        Tensor::cat(
                            vec![pts, empty_2d(padded_n - n, pt_cols, &self.device)],
                            0,
                        )
                    }
                };
                if self.fused_field {
                    let [y, _, _] = self.fused_forward_valid(xyz, n, false);
                    y
                } else {
                    let features = self.compute_features_gpu(xyz);
                    let y = self.mlp_forward(features, false);
                    if y.dims()[0] == n {
                        y
                    } else {
                        y.slice([0..n, 0..1])
                    }
                }
            })
            .collect();
        if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 0)
        }
    }

    pub fn evaluate_device_tensor(&self, points: Tensor<2>, chunk: usize) -> Tensor<2> {
        let count = points.dims()[0];
        self.evaluate_device_tensor_valid(points, count, chunk)
    }

    /// Evaluate center-ray endpoints `[padded_cap, 3|4]` using device-resident `count_prefix`
    /// (`count_mode = 2`: `min(count_prefix[rows * 2 - 1], max_ep_cap)`), avoiding host count readback.
    pub fn evaluate_device_tensor_prefix(
        &self,
        points: Tensor<2>,
        max_ep_cap: usize,
        count_prefix: Tensor<2>,
        chunk: usize,
    ) -> Tensor<2> {
        let dims = points.dims();
        let total_rows = dims[0];
        let pt_cols = dims[1];
        let weights = self.packed_mlp_weights();
        let (tables, metadata) = self.packed_tables();
        let active = self.active.min(14);
        let hashmap_size = self.hashmap_size();
        let mut outputs: Vec<Tensor<2>> = (0..max_ep_cap.max(1))
            .step_by(chunk)
            .map(|start| {
                let n = (max_ep_cap.max(1) - start).min(chunk);
                let padded_n = Self::static_bucket_size(n, chunk);
                let xyz = if start == 0 && total_rows == padded_n {
                    points.clone()
                } else if start + padded_n <= total_rows {
                    points.clone().slice([start..start + padded_n, 0..pt_cols])
                } else {
                    let pts = points.clone().slice([start..start + n, 0..pt_cols]);
                    if padded_n == n {
                        pts
                    } else {
                        Tensor::cat(
                            vec![pts, empty_2d(padded_n - n, pt_cols, &self.device)],
                            0,
                        )
                    }
                };
                let [y, _, _] = crate::kernels::fused_field_device_count(
                    [
                        xyz,
                        tables.clone(),
                        metadata.clone(),
                        weights.clone(),
                        count_prefix.clone(),
                    ],
                    n,
                    start,
                    max_ep_cap,
                    2,
                    active,
                    false,
                    false,
                    self.local_hash,
                    self.corner_parallel,
                    self.point_tiles,
                    self.subgroup_reduce,
                    self.selective_forward_weights,
                    hashmap_size,
                    self.scale_cutoff,
                );
                y
            })
            .collect();
        if outputs.len() == 1 {
            outputs.pop().unwrap()
        } else {
            Tensor::cat(outputs, 0)
        }
    }

    pub fn evaluate(&self, points: &[[f32; 3]], chunk: usize) -> Result<Vec<f32>> {
        Self::read_outputs(self.enqueue_evaluation(points, chunk))
    }

    /// Read variance in the same transfer as visibility SDFs. This is the first
    /// CPU consumer of the previous Adam update, after ray/point preparation.
    pub fn visibility_counts(
        &mut self,
        points: &[[f32; 3]],
        chunk: usize,
        pairs: Vec<f32>,
        ranges: Vec<f32>,
    ) -> Result<Vec<usize>> {
        let outputs = self
            .enqueue_evaluation(points, chunk)
            .into_iter()
            .map(|(n, y)| {
                if y.dims()[0] == n {
                    y
                } else {
                    y.slice([0..n, 0..1])
                }
            })
            .collect();
        let rays = ranges.len() / 2;
        let samples = pairs.len() / 2;
        let result = crate::kernels::visibility(
            [
                Tensor::cat(outputs, 0),
                tensor(pairs, [samples, 2], &self.device),
                tensor(ranges, [rays, 2], &self.device),
                self.p("variance").without_autodiff(),
                empty_2d(rays + 1, 1, &self.device),
            ],
            self.reuse_endpoints,
        );
        let mut values = result.try_into_vec_as::<f32>()?;
        self.variance_host = values.pop().expect("visibility variance");
        Ok(values.into_iter().map(|v| v as usize).collect())
    }

    /// Evaluate visibility on device-resident center points, pairs, and ranges,
    /// returning `[vis_out [rays + 1, 2], vis_m_begin [rays, 1]]` without host transfers.
    pub fn visibility_device(
        &self,
        unique_pts: Tensor<2>,
        valid_ep_count: usize,
        pairs: Tensor<2>,
        ranges: Tensor<2>,
        chunk: usize,
    ) -> [Tensor<2>; 2] {
        let rays = ranges.dims()[0];
        let sdf = self.evaluate_device_tensor_valid(unique_pts, valid_ep_count, chunk);
        crate::kernels::visibility_ext(
            [
                sdf,
                pairs,
                ranges,
                self.p("variance").without_autodiff(),
                empty_2d(rays + 1, 2, &self.device),
                empty_2d(rays.max(1), 1, &self.device),
            ],
            self.reuse_endpoints,
            self.front_prune,
        )
    }

    /// Fully asynchronous visibility pass reading active endpoint count from `march_prefix` on GPU.
    pub fn visibility_device_async(
        &self,
        unique_pts: Tensor<2>,
        max_ep_cap: usize,
        march_prefix: Tensor<2>,
        pairs: Tensor<2>,
        ranges: Tensor<2>,
        chunk: usize,
    ) -> [Tensor<2>; 2] {
        let rays = ranges.dims()[0];
        let sdf = if self.fused_field {
            self.evaluate_device_tensor_prefix(unique_pts, max_ep_cap, march_prefix, chunk)
        } else {
            self.evaluate_device_tensor_valid(unique_pts, max_ep_cap, chunk)
        };
        crate::kernels::visibility_ext(
            [
                sdf,
                pairs,
                ranges,
                self.p("variance").without_autodiff(),
                empty_2d(rays + 1, 2, &self.device),
                empty_2d(rays.max(1), 1, &self.device),
            ],
            self.reuse_endpoints,
            self.front_prune,
        )
    }

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
        let base_res = self.base_resolution;
        let max_res = self.max_resolution;
        let hashmap_size = self.hashmap_size();
        let hash_mask = (hashmap_size - 1) as u32;
        let (all_indices, all_weights): (Vec<Vec<i32>>, Vec<Vec<f32>>) = (0..active_levels)
            .into_par_iter()
            .map(|level| {
                let res = hash_level_resolution(base_res, max_res, level) as u32;
                let dense = (res as usize + 1).saturating_pow(3) <= hashmap_size;
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
                                & hash_mask
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
        let dims = xyz.dims();
        let n = dims[0];
        let cols = dims[1];
        if self.is_cpu {
            let flat = xyz.try_into_vec_as::<f32>().expect("CPU xyz read");
            let pts: Vec<[f32; 3]> = flat
                .chunks_exact(cols)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            return self.compute_features_impl(&pts, false).0;
        }
        let active = self.active.min(14);
        if active == 0 {
            return Tensor::cat(vec![xyz, zeros_2d(n, 28, &self.device)], 1);
        }
        let features = if active < 14 {
            zeros_2d(n, 31, &self.device)
        } else {
            empty_2d(n, 31, &self.device)
        };
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
        self.enqueue_cached(points, chunk).finish()
    }

    fn fused_forward_valid(&self, xyz: Tensor<2>, valid_n: usize, cache: bool) -> [Tensor<2>; 3] {
        self.fused_forward_cache_valid(xyz, valid_n, cache, cache)
    }

    fn packed_mlp_weights(&self) -> Tensor<2> {
        if self.packed_mlp.borrow().is_none() {
            let weights = if self.fused_mlp_packing {
                crate::kernels::pack_mlp_weights([
                    self.effective_weight(0, false),
                    self.p("lin0.bias").without_autodiff(),
                    self.effective_weight(1, false),
                    self.p("lin1.bias").without_autodiff(),
                ])
            } else {
                Tensor::cat(
                    vec![
                        self.effective_weight(0, false)
                            .transpose()
                            .reshape([1, 1984]),
                        self.p("lin0.bias").without_autodiff().reshape([1, 64]),
                        self.effective_weight(1, false).reshape([1, 64]),
                        self.p("lin1.bias").without_autodiff().reshape([1, 1]),
                    ],
                    1,
                )
            };
            *self.packed_mlp.borrow_mut() = Some(weights);
        }
        self.packed_mlp.borrow().as_ref().unwrap().clone()
    }

    pub fn update_occupancy_device(
        &self,
        points: Tensor<2>,
        prefix: Tensor<2>,
        ids: Tensor<2>,
        values: Tensor<2>,
    ) -> Tensor<2> {
        let weights = self.packed_mlp_weights();
        let (tables, metadata) = self.packed_tables();
        crate::kernels::fused_occupancy_field(
            [points, tables, metadata, weights, prefix, ids, values],
            self.active.min(14),
            self.local_hash,
            self.corner_parallel,
            self.point_tiles,
            self.subgroup_reduce,
            self.selective_forward_weights,
            self.hashmap_size(),
        )
    }

    fn fused_forward_cache_valid(
        &self,
        xyz: Tensor<2>,
        valid_n: usize,
        cache_features: bool,
        cache_preactivation: bool,
    ) -> [Tensor<2>; 3] {
        let weights = self.packed_mlp_weights();
        let (tables, metadata) = self.packed_tables();
        crate::kernels::fused_field(
            [xyz, tables, metadata, weights],
            valid_n,
            self.active.min(14),
            cache_features,
            cache_preactivation,
            self.local_hash,
            self.corner_parallel,
            self.point_tiles,
            self.subgroup_reduce,
            self.selective_forward_weights,
            self.hashmap_size(),
            self.scale_cutoff,
        )
    }

    pub fn enqueue_cached(&self, points: &[[f32; 3]], chunk: usize) -> PendingEvaluation {
        let inputs = points.chunks(chunk).map(|pts| {
            let n = pts.len();
            let padded = Self::static_bucket_size(n, chunk);
            let mut flat: Vec<_> = pts.iter().flatten().copied().collect();
            flat.resize(padded * 3, 0.);
            (n, tensor(flat, [padded, 3], &self.device))
        });
        self.enqueue_cached_inputs(inputs)
    }

    /// Consume generated GPU coordinates `[padded_cap, 3|4]` with `valid_count`
    /// points in static power-of-two `chunk` buckets without downloading/re-uploading
    /// or issuing `Tensor::zeros` + `Tensor::cat` padding copies.
    pub fn enqueue_cached_device_valid(
        &self,
        points: Tensor<2>,
        valid_count: usize,
        chunk: usize,
    ) -> PendingEvaluation {
        let dims = points.dims();
        let total_rows = dims[0];
        let pt_cols = dims[1];
        let inputs = (0..valid_count).step_by(chunk).map(|start| {
            let n = (valid_count - start).min(chunk);
            let padded = Self::static_bucket_size(n, chunk);
            let pts = if start == 0 && total_rows == padded {
                points.clone()
            } else if start + padded <= total_rows {
                points.clone().slice([start..start + padded, 0..pt_cols])
            } else {
                let sub = points.clone().slice([start..start + n, 0..pt_cols]);
                if padded == n {
                    sub
                } else {
                    Tensor::cat(
                        vec![sub, empty_2d(padded - n, pt_cols, &self.device)],
                        0,
                    )
                }
            };
            (n, pts)
        });
        self.enqueue_cached_inputs(inputs)
    }

    /// Enqueue cached 3x3 stencil forward evaluation driven by device-resident `vis_prefix`
    /// (`count_mode = 3`: `min(vis_prefix[rows * 2 - 1], max_ep_cap) * 9`) using feature-only
    /// retention (`cache_preactivation = false`) so backward replays `h` from shared memory.
    pub fn enqueue_cached_device_prefix(
        &self,
        points: Tensor<2>,
        max_ep_cap: usize,
        vis_prefix: Tensor<2>,
        chunk: usize,
    ) -> PendingEvaluation {
        let max_points = (max_ep_cap * 9).max(1);
        let dims = points.dims();
        let total_rows = dims[0];
        let pt_cols = dims[1];
        let weights = self.packed_mlp_weights();
        let (tables, metadata) = self.packed_tables();
        let active = self.active.min(14);
        let hashmap_size = self.hashmap_size();
        let mut chunks = Vec::new();
        let mut pending_outputs = Vec::new();
        for start in (0..max_points).step_by(chunk) {
            let n = (max_points - start).min(chunk);
            let padded_n = Self::static_bucket_size(n, chunk);
            let pts_tensor = if start == 0 && total_rows == padded_n {
                points.clone()
            } else if start + padded_n <= total_rows {
                points.clone().slice([start..start + padded_n, 0..pt_cols])
            } else {
                let sub = points.clone().slice([start..start + n, 0..pt_cols]);
                if padded_n == n {
                    sub
                } else {
                    Tensor::cat(
                        vec![sub, empty_2d(padded_n - n, pt_cols, &self.device)],
                        0,
                    )
                }
            };
            let [y, feat, _] = crate::kernels::fused_field_device_count(
                [
                    pts_tensor.clone(),
                    tables.clone(),
                    metadata.clone(),
                    weights.clone(),
                    vis_prefix.clone(),
                ],
                n,
                start,
                max_ep_cap,
                3,
                active,
                true,
                false,
                self.local_hash,
                self.corner_parallel,
                self.point_tiles,
                self.subgroup_reduce,
                self.selective_forward_weights,
                hashmap_size,
                self.scale_cutoff,
            );
            pending_outputs.push((n, y));
            chunks.push(FeatureChunk {
                feat: Some(feat),
                preactivation: None,
                pts_tensor,
                point_count: n,
                padded_count: padded_n,
            });
        }
        PendingEvaluation {
            outputs: pending_outputs,
            cache: chunks,
        }
    }

    fn enqueue_cached_inputs(
        &self,
        inputs: impl Iterator<Item = (usize, Tensor<2>)>,
    ) -> PendingEvaluation {
        // Look ahead at shapes only; no activation evaluation or GPU readback.
        let inputs: Vec<_> = inputs.collect();
        let bytes = |rows: usize, columns: usize| rows.saturating_mul(columns).saturating_mul(4);
        let budget = inputs
            .iter()
            .take(MAX_CACHED_CHUNKS)
            .fold(0usize, |total, (_, pts)| {
                total.saturating_add(bytes(pts.dims()[0], 95))
            });
        let full_bytes = inputs.iter().fold(0usize, |total, (_, pts)| {
            total.saturating_add(bytes(pts.dims()[0], 95))
        });
        let feature_only = self.fused_field && self.feature_only_cache && full_bytes > budget;
        // Materialized replay needs a streamed preactivation workspace. Fused
        // replay keeps h in registers, making that space available to features
        // without increasing the previous retained-cache budget.
        let pre_workspace = if self.fused_feature_replay && !self.recompute_parameter_gradients {
            0
        } else {
            inputs
                .iter()
                .map(|(_, pts)| bytes(pts.dims()[0], 64))
                .max()
                .unwrap_or(0)
        };
        let feature_budget = budget.saturating_sub(pre_workspace);
        // The established/default cache path does not need planning vectors.
        // Only the opt-in hybrid path computes a two-pass storage plan.
        let hybrid_plan = if feature_only && self.hybrid_activation_cache {
            let mut used = 0usize;
            let mut plan: Vec<(bool, bool)> = inputs
                .iter()
                .map(|(_, points)| {
                    let cost = bytes(points.dims()[0], 31);
                    let keep = cost <= feature_budget.saturating_sub(used);
                    if keep {
                        used += cost;
                    }
                    (keep, false)
                })
                .collect();
            let mut spare = feature_budget.saturating_sub(used);
            for ((_, points), (keep, cache_pre)) in inputs.iter().zip(&mut plan) {
                let cost = bytes(points.dims()[0], 64);
                if *keep && cost <= spare {
                    *cache_pre = true;
                    spare -= cost;
                }
            }
            Some(plan)
        } else {
            None
        };
        let mut retained_bytes = 0usize;
        if feature_only {
            crate::diagnostics::count("field.feature_only_batches", 1);
        }
        let mut chunks = Vec::new();
        let mut pending_outputs = Vec::new();
        let w0_t = if !self.fused_field {
            Some(self.effective_weight(0, false).transpose())
        } else {
            None
        };
        let b0 = if !self.fused_field {
            Some(self.p("lin0.bias").without_autodiff())
        } else {
            None
        };
        for (chunk_idx, (n, pts_tensor)) in inputs.into_iter().enumerate() {
            let padded_n = pts_tensor.dims()[0];
            crate::diagnostics::count("field.cached_forward_points", n);
            crate::diagnostics::count("field.cached_capacity_points", padded_n);
            if self.fused_field {
                let (keep_cache, cache_pre) = if let Some(plan) = &hybrid_plan {
                    plan[chunk_idx]
                } else if feature_only {
                    (
                        bytes(padded_n, 31) <= feature_budget.saturating_sub(retained_bytes),
                        false,
                    )
                } else {
                    let keep = chunk_idx < MAX_CACHED_CHUNKS;
                    (keep, keep)
                };
                if keep_cache {
                    retained_bytes += bytes(padded_n, if cache_pre { 95 } else { 31 });
                }
                let [y, feat, preactivation] =
                    self.fused_forward_cache_valid(pts_tensor.clone(), n, keep_cache, cache_pre);
                if keep_cache {
                    crate::diagnostics::count(
                        if !cache_pre {
                            "field.feature_only_cached_points"
                        } else {
                            "field.full_cached_points"
                        },
                        n,
                    );
                }
                pending_outputs.push((n, y));
                chunks.push(FeatureChunk {
                    feat: if keep_cache { Some(feat) } else { None },
                    preactivation: if cache_pre { Some(preactivation) } else { None },
                    pts_tensor,
                    point_count: n,
                    padded_count: padded_n,
                });
            } else {
                let feat = self.compute_features_gpu(pts_tensor.clone());
                let preactivation =
                    feat.clone().matmul(w0_t.clone().unwrap()) + b0.clone().unwrap();
                let hidden = Self::softplus(preactivation.clone());
                let y = self.mlp_output(hidden, false);
                pending_outputs.push((n, y));
                chunks.push(FeatureChunk {
                    feat: Some(feat),
                    preactivation: Some(preactivation),
                    pts_tensor,
                    point_count: n,
                    padded_count: padded_n,
                });
            }
        }

        if feature_only {
            crate::diagnostics::count("field.retained_activation_bytes", retained_bytes);
        }
        PendingEvaluation {
            outputs: pending_outputs,
            cache: chunks,
        }
    }

    #[inline]
    pub fn inv_s_host(&self) -> f32 {
        (self.variance_host * 10.).exp().clamp(1e-6, 1e6)
    }

    pub fn trim_memory(&self) {
        crate::diagnostics::count("memory.pool_checks", 1);
        if let Some(usage) = self.device.memory_pool_usage() {
            self.peak_reserved_bytes
                .set(self.peak_reserved_bytes.get().max(usage.bytes_reserved));
            if usage.bytes_reserved > 768 * 1024 * 1024 {
                crate::diagnostics::count("memory.cleanup_calls", 1);
                let _ = self.device.sync();
                self.device.memory_cleanup();
                return;
            }
        }
        // A maintenance check must still submit pending lazy work.
        let _ = self.device.flush();
    }

    pub fn accumulate_gradients_cached(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: &[f32],
        accumulated: &mut Gradients,
    ) {
        if cached.is_empty() {
            return;
        }
        let adjoints = tensor(adjoints.to_vec(), [adjoints.len(), 1], &self.device);
        self.accumulate_gradients_device(cached, adjoints, accumulated);
    }

    pub fn accumulate_gradients_device(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: Tensor<2>,
        accumulated: &mut Gradients,
    ) {
        self.accumulate_gradients_device_impl(cached, adjoints, accumulated, true);
    }

    pub fn accumulate_training_gradients_device(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: Tensor<2>,
        accumulated: &mut Gradients,
    ) {
        self.accumulate_gradients_device_impl(cached, adjoints, accumulated, false);
    }

    /// Fused backward + parameter gradient accumulation driven on-device by `vis_prefix`
    /// (`device_counts = true`, passing `vis_prefix` in the unused `pre` slot since
    /// `replay_features = true` rebuilds `h` in registers from `shared_feat`).
    pub fn accumulate_training_gradients_device_prefix(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: Tensor<2>,
        vis_prefix: Tensor<2>,
        max_ep_cap: usize,
        accumulated: &mut Gradients,
    ) {
        if cached.is_empty() {
            return;
        }
        let active_levels = self.active.min(14);
        let (_, metadata) = self.packed_tables();
        if !accumulated.params.contains_key("encoding.packed") {
            let total_entries = *self.level_offsets.last().unwrap();
            let grad_buf = self
                .persistent_table_grad
                .borrow_mut()
                .take()
                .unwrap_or_else(|| zeros_2d(total_entries, 2, &self.device));
            accumulated
                .params
                .insert("encoding.packed".into(), grad_buf);
        }
        let weights = self
            .packed_mlp
            .borrow()
            .as_ref()
            .expect("cached forward packs weights")
            .clone();
        let hashmap_size = self.hashmap_size();
        let mut offset = 0usize;
        let mut fused_partials: Option<(Tensor<2>, Tensor<2>)> = None;

        for chunk in cached {
            let n = chunk.point_count;
            let feat = chunk.feat.expect("async cached forward retains features");
            let table_grad = accumulated.params.remove("encoding.packed").unwrap();
            let (dw0p, dw1p, accumulate) = match fused_partials.take() {
                Some((p0, p1)) => (p0, p1, true),
                None => (
                    empty_2d(448, 2048, &self.device),
                    empty_2d(448, 64, &self.device),
                    false,
                ),
            };
            let [updated, next_dw0p, next_dw1p] = crate::kernels::field_backward_fused_ext(
                [
                    vis_prefix.clone(),
                    feat,
                    adjoints.clone(),
                    weights.clone(),
                    metadata.clone(),
                    table_grad,
                    dw0p,
                    dw1p,
                ],
                n,
                max_ep_cap,
                true,
                active_levels,
                self.local_hash,
                self.aggregate_hash,
                self.aggregation_levels,
                true,
                offset,
                self.selective_backward_weights,
                accumulate,
                self.active_parameter_columns,
                hashmap_size,
                self.scale_cutoff,
            );
            accumulated.params.insert("encoding.packed".into(), updated);
            fused_partials = Some((next_dw0p, next_dw1p));
            offset += n;
        }

        if let Some((dw0p, dw1p)) = fused_partials {
            let [dw0_f, db0_f, dw1_f] = crate::kernels::reduce_mlp_partials(dw0p, dw1p);
            accumulated.raw_mlp = Some(match accumulated.raw_mlp.take() {
                Some((a0, ab0, a1)) => (
                    (a0 + dw0_f).without_autodiff(),
                    (ab0 + db0_f).without_autodiff(),
                    (a1 + dw1_f).without_autodiff(),
                ),
                None => (dw0_f, db0_f, dw1_f),
            });
        }
    }

    fn accumulate_gradients_device_impl(
        &self,
        cached: Vec<FeatureChunk>,
        adjoints: Tensor<2>,
        accumulated: &mut Gradients,
        need_bias: bool,
    ) {
        let active_levels = self.active.min(14);
        let (tables, metadata) = self.packed_tables();
        if !self.is_cpu && !accumulated.params.contains_key("encoding.packed") {
            let total_entries = *self.level_offsets.last().unwrap();
            let grad_buf = self
                .persistent_table_grad
                .borrow_mut()
                .take()
                .unwrap_or_else(|| zeros_2d(total_entries, 2, &self.device));
            accumulated
                .params
                .insert("encoding.packed".into(), grad_buf);
        }

        let w0 = if !self.fused_field {
            Some(self.effective_weight(0, false))
        } else {
            None
        };
        let w1 = if !self.fused_field {
            Some(self.effective_weight(1, false))
        } else {
            None
        };

        let mut offset = 0;
        let mut dw0_total: Option<Tensor<2>> = None;
        let mut dw1_total: Option<Tensor<2>> = None;
        let mut db0_total: Option<Tensor<2>> = None;
        let mut db1_total: Option<Tensor<2>> = None;
        let mut fused_partials: Option<(Tensor<2>, Tensor<2>)> = None;
        let single_chunk = cached.len() == 1;

        for chunk in cached {
            let n = chunk.point_count;
            crate::diagnostics::count("field.backward_points", n);
            let padded_n = chunk.padded_count;
            // The default fused training kernel can address its chunk inside
            // the shared global adjoint buffer. Reference/recompute/bias paths
            // still receive a local slice with their existing semantics.
            let direct_adjoints = self.direct_adjoint_views
                && self.fused_field
                && !need_bias
                && !self.recompute_parameter_gradients;
            let adjoint_offset = if direct_adjoints { offset } else { 0 };
            let mut ds_tensor = if direct_adjoints || (single_chunk && adjoints.dims()[0] == n) {
                adjoints.clone()
            } else {
                adjoints.clone().slice([offset..offset + n, 0..1])
            };
            offset += n;
            if crate::diagnostics::enabled() {
                // Address diagnostics inspect chunk-local adjoints. This debug
                // slice does not replace the direct buffer used by backward.
                let sampled_ds = if direct_adjoints {
                    ds_tensor
                        .clone()
                        .slice([adjoint_offset..adjoint_offset + n, 0..1])
                } else {
                    ds_tensor.clone()
                };
                crate::diagnostics::sample_addresses(
                    &chunk.pts_tensor,
                    &sampled_ds,
                    n,
                    active_levels,
                );
            }

            let mut replay_features = false;
            let mut replay_hash = false;
            let (mut feat, preactivation) = match (chunk.feat, chunk.preactivation) {
                (Some(f), Some(p)) => (f, p),
                (Some(f), None)
                    if self.fused_field
                        && self.fused_feature_replay
                        && !self.recompute_parameter_gradients =>
                {
                    replay_features = true;
                    crate::diagnostics::count("field.fused_feature_replay_points", n);
                    // Placeholder only: the shader rebuilds h in registers from f.
                    // Keep this separate from the feature output binding.
                    (f, empty_2d(1, 64, &self.device))
                }
                (Some(f), None) => {
                    crate::diagnostics::count("field.mlp_only_recomputed_points", n);
                    let weights = self
                        .packed_mlp
                        .borrow()
                        .as_ref()
                        .expect("forward packs MLP weights")
                        .clone();
                    let pre = crate::kernels::cached_mlp_preactivation(
                        f.clone(),
                        weights,
                        n,
                        active_levels,
                    );
                    (f, pre)
                }
                (None, _)
                    if self.fused_field
                        && self.fused_hash_replay
                        && self.fused_feature_replay
                        && !self.recompute_parameter_gradients
                        && !self.scale_encoding
                        && self.log2_hashmap_size == 19 =>
                {
                    replay_features = true;
                    replay_hash = true;
                    crate::diagnostics::count("field.fused_hash_replay_points", n);
                    // Backward writes f and consumes h directly in registers.
                    // No forward SDF or [N,64] preactivation allocation is needed.
                    (
                        empty_2d(padded_n, 31, &self.device),
                        empty_2d(1, 64, &self.device),
                    )
                }
                _ => {
                    crate::diagnostics::count("field.recomputed_points", n);
                    let [_, f, p] = self.fused_forward_valid(chunk.pts_tensor.clone(), n, true);
                    (f, p)
                }
            };

            let use_fused_bwd = self.fused_backward_params
                && self.fused_field
                && self.tiled_gradients
                && self.corner_parallel
                && !self.recompute_parameter_gradients
                && self.wide_gradient_tiles
                && self.register_gradient_tiles
                && !self.cross_corners
                && !replay_hash;
            if use_fused_bwd {
                let weights = self
                    .packed_mlp
                    .borrow()
                    .as_ref()
                    .expect("cached forward packs weights")
                    .clone();
                let table_grad = accumulated.params.remove("encoding.packed").unwrap();
                let (dw0p, dw1p, accumulate) = match fused_partials.take() {
                    Some((p0, p1)) => (p0, p1, true),
                    None => (
                        empty_2d(448, 2048, &self.device),
                        empty_2d(448, 64, &self.device),
                        false,
                    ),
                };
                let [updated, next_dw0p, next_dw1p] = crate::kernels::field_backward_fused(
                    [
                        preactivation,
                        feat,
                        ds_tensor.clone(),
                        weights,
                        metadata.clone(),
                        table_grad,
                        dw0p,
                        dw1p,
                    ],
                    n,
                    active_levels,
                    self.local_hash,
                    self.aggregate_hash,
                    self.aggregation_levels,
                    replay_features,
                    adjoint_offset,
                    self.selective_backward_weights,
                    accumulate,
                    self.active_parameter_columns,
                    self.hashmap_size(),
                    self.scale_cutoff,
                );
                accumulated.params.insert("encoding.packed".into(), updated);
                fused_partials = Some((next_dw0p, next_dw1p));
                if need_bias {
                    let db1_chunk = ds_tensor.sum_dim(0);
                    db1_total = Some(match db1_total {
                        Some(acc) => (acc + db1_chunk).without_autodiff(),
                        None => db1_chunk,
                    });
                }
                continue;
            }

            let recompute =
                self.fused_field && self.tiled_gradients && self.recompute_parameter_gradients;
            let parameter_inputs = if recompute {
                Some((
                    preactivation.clone(),
                    ds_tensor.clone(),
                    self.packed_mlp.borrow().as_ref().unwrap().clone(),
                ))
            } else {
                None
            };
            let (dh, d_feat_tensor, dw1_chunk, db1_chunk) = if self.fused_field {
                let weights = self
                    .packed_mlp
                    .borrow()
                    .as_ref()
                    .expect("cached forward packs weights")
                    .clone();
                let table_grad = accumulated.params.remove("encoding.packed").unwrap();
                let [updated, dh, dw1, backward_feat] = crate::kernels::field_backward(
                    [
                        preactivation,
                        ds_tensor.clone(),
                        weights,
                        chunk.pts_tensor.clone(),
                        metadata.clone(),
                        feat,
                        table_grad,
                        tables.clone(),
                    ],
                    n,
                    active_levels,
                    self.local_hash,
                    self.compact_gradients,
                    self.corner_parallel,
                    self.aggregate_hash,
                    self.aggregation_levels,
                    self.cross_corners,
                    !recompute,
                    replay_features,
                    adjoint_offset,
                    self.selective_backward_weights,
                    replay_hash,
                );
                feat = backward_feat;
                accumulated.params.insert("encoding.packed".into(), updated);
                (
                    dh,
                    None,
                    dw1.sum_dim(0),
                    if need_bias {
                        Some(ds_tensor.sum_dim(0))
                    } else {
                        None
                    },
                )
            } else {
                let db1_chunk = if need_bias {
                    Some(ds_tensor.clone().sum_dim(0))
                } else {
                    None
                };
                if padded_n != n {
                    ds_tensor = Tensor::cat(
                        vec![ds_tensor, Tensor::zeros([padded_n - n, 1], &self.device)],
                        0,
                    );
                }
                let hidden = Self::softplus(preactivation.clone());
                let dw1 = ds_tensor.clone().transpose().matmul(hidden);
                let scaled = preactivation * 100.;
                let derivative = ((-scaled.clone()).exp() + 1.)
                    .recip()
                    .mask_fill(scaled.greater_elem(20.), 1.);
                let dh = ds_tensor * w1.clone().unwrap() * derivative;
                let df = dh.clone().matmul(w0.clone().unwrap());
                (dh, Some(df), dw1, db1_chunk)
            };
            let (dw0_chunk, db0_chunk) = if self.tiled_gradients {
                if let Some((pre, ds, weights)) = parameter_inputs {
                    crate::diagnostics::count("field.recomputed_parameter_points", n);
                    crate::sampling_gpu::parameter_gradients_recompute(
                        pre,
                        feat,
                        ds,
                        weights,
                        n,
                        self.wide_gradient_tiles,
                        self.register_gradient_tiles,
                    )
                } else {
                    crate::sampling_gpu::parameter_gradients(
                        dh,
                        feat,
                        n,
                        self.wide_gradient_tiles,
                        self.register_gradient_tiles,
                        if self.fused_field && self.active_parameter_columns {
                            3 + 2 * active_levels
                        } else {
                            31
                        },
                        self.wide_parameter_rows,
                    )
                }
            } else {
                let dh_valid = if padded_n == n {
                    dh
                } else {
                    dh.slice([0..n, 0..64])
                };
                let feat_valid = if padded_n == n {
                    feat
                } else {
                    feat.slice([0..n, 0..31])
                };
                (
                    dh_valid.clone().transpose().matmul(feat_valid),
                    dh_valid.sum_dim(0),
                )
            };

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
            if let Some(db1_chunk) = db1_chunk {
                db1_total = Some(match db1_total {
                    Some(acc) => (acc + db1_chunk).without_autodiff(),
                    None => db1_chunk,
                });
            }

            if let Some(d_feat_tensor) = d_feat_tensor.filter(|_| active_levels > 0) {
                if self.is_cpu {
                    let pt_cols = chunk.pts_tensor.dims()[1];
                    let pts_flat = chunk
                        .pts_tensor
                        .try_into_vec_as::<f32>()
                        .expect("CPU pts read");
                    let df_flat = d_feat_tensor
                        .try_into_vec_as::<f32>()
                        .expect("CPU d_feat read");
                    let hashmap_size = self.hashmap_size();
                    let hash_mask = (hashmap_size - 1) as u32;
                    if accumulated.host_tables.len() < active_levels {
                        accumulated.host_tables.resize_with(active_levels, Vec::new);
                    }
                    for level in 0..active_levels {
                        let res =
                            hash_level_resolution(self.base_resolution, self.max_resolution, level)
                                as u32;
                        let entries = (res as usize + 1).saturating_pow(3).min(hashmap_size);
                        let dense = (res as usize + 1).saturating_pow(3) <= hashmap_size;
                        let table = &mut accumulated.host_tables[level];
                        if table.is_empty() {
                            table.resize(entries * 2, 0.0_f32);
                        }
                        let col = 3 + level * 2;
                        for p_idx in 0..n {
                            let g0 = df_flat[p_idx * 31 + col];
                            let g1 = df_flat[p_idx * 31 + col + 1];
                            if g0 == 0.0 && g1 == 0.0 {
                                continue;
                            }
                            let p = [
                                pts_flat[p_idx * pt_cols],
                                pts_flat[p_idx * pt_cols + 1],
                                pts_flat[p_idx * pt_cols + 2],
                            ];
                            let pos =
                                p.map(|x| ((x + 1.) * 0.5).clamp(0., 1. - 1e-7) * res as f32);
                            let base = pos.map(|x| x.floor() as u32);
                            let w = [
                                pos[0] - base[0] as f32,
                                pos[1] - base[1] as f32,
                                pos[2] - base[2] as f32,
                            ];
                            for corner in 0..8u32 {
                                let b0 = corner & 1;
                                let b1 = (corner >> 1) & 1;
                                let b2 = (corner >> 2) & 1;
                                let v0 = base[0] + b0;
                                let v1 = base[1] + b1;
                                let v2 = base[2] + b2;
                                let wt = (if b0 == 0 { 1. - w[0] } else { w[0] })
                                    * (if b1 == 0 { 1. - w[1] } else { w[1] })
                                    * (if b2 == 0 { 1. - w[2] } else { w[2] });
                                let idx = if dense {
                                    (v0 + v1 * (res + 1) + v2 * (res + 1) * (res + 1)) as usize
                                } else {
                                    ((v0 ^ v1.wrapping_mul(2654435761) ^ v2.wrapping_mul(805459861))
                                        & hash_mask) as usize
                                };
                                table[idx * 2] += wt * g0;
                                table[idx * 2 + 1] += wt * g1;
                            }
                        }
                    }
                } else {
                    let table_grad = accumulated.params.remove("encoding.packed").unwrap();
                    let updated = crate::kernels::backward_packed(
                        chunk.pts_tensor,
                        d_feat_tensor,
                        metadata.clone(),
                        table_grad,
                        active_levels,
                        n,
                    );
                    accumulated.params.insert("encoding.packed".into(), updated);
                }
            }
        }

        if let Some((dw0p, dw1p)) = fused_partials {
            let [dw0_f, db0_f, dw1_f] = crate::kernels::reduce_mlp_partials(dw0p, dw1p);
            dw0_total = Some(match dw0_total {
                Some(acc) => (acc + dw0_f).without_autodiff(),
                None => dw0_f,
            });
            db0_total = Some(match db0_total {
                Some(acc) => (acc + db0_f).without_autodiff(),
                None => db0_f,
            });
            dw1_total = Some(match dw1_total {
                Some(acc) => (acc + dw1_f).without_autodiff(),
                None => dw1_f,
            });
        }

        if let (Some(dw0), Some(dw1), Some(db0)) = (dw0_total, dw1_total, db0_total) {
            if !need_bias
                && self.fused_field
                && self.fused_mlp_packing
                && self.packed_mlp_state.is_some()
                && db1_total.is_none()
            {
                accumulated.raw_mlp = Some(match accumulated.raw_mlp.take() {
                    Some((a0, ab0, a1)) => (
                        (a0 + dw0).without_autodiff(),
                        (ab0 + db0).without_autodiff(),
                        (a1 + dw1).without_autodiff(),
                    ),
                    None => (dw0, db0, dw1),
                });
            } else {
                let (dv0, dg0) = self.weight_gradients(0, dw0);
                let (dv1, dg1) = self.weight_gradients(1, dw1);

                for (name, grad) in [
                    ("lin0.weight_v", dv0),
                    ("lin0.weight_g", dg0),
                    ("lin0.bias", db0),
                    ("lin1.weight_v", dv1),
                    ("lin1.weight_g", dg1),
                ] {
                    accumulated
                        .params
                        .entry(name.to_string())
                        .and_modify(|v| *v = (v.clone() + grad.clone()).without_autodiff())
                        .or_insert(grad);
                }
            }
        }
        if let Some(db1) = db1_total {
            accumulated
                .params
                .entry("lin1.bias".into())
                .and_modify(|g| *g = (g.clone() + db1.clone()).without_autodiff())
                .or_insert(db1);
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

    pub fn apply_gradients_packed(&mut self, gradients: Gradients, lr: f64) {
        self.apply_gradients_packed_with_stats(gradients, None, lr);
    }

    pub fn apply_gradients_packed_with_stats(
        &mut self,
        mut gradients: Gradients,
        device_stats: Option<&Tensor<2>>,
        lr: f64,
    ) {
        if self.is_cpu {
            if let Some(stats) = device_stats {
                gradients
                    .params
                    .entry("variance".into())
                    .or_insert_with(|| stats.clone().slice([0..1, 3..4]));
                gradients
                    .params
                    .entry("lin1.bias".into())
                    .or_insert_with(|| stats.clone().slice([0..1, 4..5]));
            }
            let tensors = gradients.into_tensors(self);
            self.apply_gradients(tensors, lr);
            return;
        }
        self.inference_weights.get_mut().take();
        self.packed_mlp.get_mut().take();

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
                let active_rows = self.level_offsets[active];
                let packed = self.packed_embedding.take().unwrap();
                let [value, m, v, zeroed_grad] = crate::kernels::adam_packed(
                    packed.value,
                    packed.m,
                    packed.v,
                    packed_grad.without_autodiff(),
                    tensor(corrections, [active, 3], &self.device),
                    active_rows,
                    (0..active)
                        .map(|l| self.level_offsets[l + 1] - self.level_offsets[l])
                        .max()
                        .unwrap(),
                    self.level_adam,
                    self.skip_untouched_adam,
                );
                self.packed_embedding = Some(Parameter {
                    value,
                    m,
                    v,
                    updates: self.level_u32_updates[0],
                });
                *self.persistent_table_grad.borrow_mut() = Some(zeroed_grad);
            } else {
                *self.persistent_table_grad.borrow_mut() = Some(packed_grad.without_autodiff());
            }
        }

        if self.fused_field
            && self.fused_mlp_packing
            && gradients.params.is_empty()
            && device_stats.is_some()
            && gradients.raw_mlp.is_some()
            && self.packed_mlp_state.is_some()
        {
            let stats = device_stats.unwrap();
            let (dw0, db0, dw1) = gradients.raw_mlp.take().unwrap();
            let (state, mut updates) = self.packed_mlp_state.take().unwrap();
            updates += 1;
            let inv_bc1 = 1.0 / (1.0 - 0.9f64.powi(updates as i32));
            let inv_bc2 = 1.0 / (1.0 - 0.999f64.powi(updates as i32));
            let [new_state, packed_mlp, new_var] = crate::kernels::fused_mlp_step(
                state,
                dw0,
                db0,
                dw1,
                stats.clone(),
                inv_bc1,
                inv_bc2,
                lr,
            );
            self.packed_mlp_state = Some((new_state, updates));
            *self.packed_mlp.borrow_mut() = Some(packed_mlp);
            self.params.get_mut("variance").unwrap().value = new_var.autodiff().require_grad();
            return;
        }

        self.sync_packed_mlp_to_params();
        gradients.expand_raw_mlp(self);
        if let Some(stats) = device_stats {
            gradients
                .params
                .entry("variance".into())
                .or_insert_with(|| stats.clone().slice([0..1, 3..4]));
            gradients
                .params
                .entry("lin1.bias".into())
                .or_insert_with(|| stats.clone().slice([0..1, 4..5]));
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
        self.packed_mlp_state = Self::build_packed_mlp_state(&self.params);
    }

    pub fn apply_gradients(&mut self, gradients: BTreeMap<String, Tensor<2>>, lr: f64) {
        self.sync_packed_mlp_to_params();
        self.inference_weights.get_mut().take();
        self.packed_mlp.get_mut().take();
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
            Self::build_packed_embedding(
                &self.params,
                self.base_resolution,
                self.max_resolution,
                &self.device,
            );
        self.packed_embedding = Some(packed_embedding);
        self.packed_metadata = Some(packed_metadata);
        self.level_offsets = level_offsets;
        self.level_u32_updates = level_u32_updates;
        self.packed_mlp_state = Self::build_packed_mlp_state(&self.params);
    }

    pub fn reset_optimizer_state(&mut self) {
        self.sync_packed_to_params();
        self.inference_weights.get_mut().take();
        self.packed_mlp.get_mut().take();
        self.persistent_table_grad.get_mut().take();
        for p in self.params.values_mut() {
            let dims = p.value.dims();
            p.m = zeros_2d(dims[0], dims[1], &self.device);
            p.v = zeros_2d(dims[0], dims[1], &self.device);
            p.updates = 0;
        }
        let (packed_embedding, packed_metadata, level_offsets, level_u32_updates) =
            Self::build_packed_embedding(
                &self.params,
                self.base_resolution,
                self.max_resolution,
                &self.device,
            );
        self.packed_embedding = Some(packed_embedding);
        self.packed_metadata = Some(packed_metadata);
        self.level_offsets = level_offsets;
        self.level_u32_updates = level_u32_updates;
        self.packed_mlp_state = Self::build_packed_mlp_state(&self.params);
    }

    /// Rapidly fit the coarse SDF field (`active = 2`) to a conservative 3D visual-hull
    /// signed distance grid `hull_sdf` (`res^3`, layout `(cx * res + cy) * res + cz`,
    /// `> 0` outside, `< 0` inside), then reset Adam moments before Step 1.
    pub fn warmstart_from_hull_sdf(&mut self, hull_sdf: &[f32], res: usize, seed: u64) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x5648_554C_4C30_u64);
        let rr = res * res;
        let band_cells: Vec<usize> = hull_sdf
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d.abs() < 0.08).then_some(i))
            .collect();

        let sample_trilinear = |p: [f32; 3]| -> f32 {
            let max_c = (res - 1) as f32;
            let gx = ((p[0] + 1.0) * 0.5 * res as f32 - 0.5).clamp(0.0, max_c);
            let gy = ((p[1] + 1.0) * 0.5 * res as f32 - 0.5).clamp(0.0, max_c);
            let gz = ((p[2] + 1.0) * 0.5 * res as f32 - 0.5).clamp(0.0, max_c);
            let ix = (gx as usize).min(res - 2);
            let iy = (gy as usize).min(res - 2);
            let iz = (gz as usize).min(res - 2);
            let fx = (gx - ix as f32).clamp(0.0, 1.0);
            let fy = (gy - iy as f32).clamp(0.0, 1.0);
            let fz = (gz - iz as f32).clamp(0.0, 1.0);
            let base = ix * rr + iy * res + iz;
            let c000 = hull_sdf[base];
            let c001 = hull_sdf[base + 1];
            let c010 = hull_sdf[base + res];
            let c011 = hull_sdf[base + res + 1];
            let c100 = hull_sdf[base + rr];
            let c101 = hull_sdf[base + rr + 1];
            let c110 = hull_sdf[base + rr + res];
            let c111 = hull_sdf[base + rr + res + 1];
            let c00 = c000 * (1.0 - fz) + c001 * fz;
            let c01 = c010 * (1.0 - fz) + c011 * fz;
            let c10 = c100 * (1.0 - fz) + c101 * fz;
            let c11 = c110 * (1.0 - fz) + c111 * fz;
            let c0 = c00 * (1.0 - fy) + c01 * fy;
            let c1 = c10 * (1.0 - fy) + c11 * fy;
            c0 * (1.0 - fx) + c1 * fx
        };

        let prev_active = self.active;
        self.active = 2;
        let n_pts = if self.is_cpu { 4096usize } else { 16384usize };
        let pt_cols = if self.scale_encoding { 4 } else { 3 };
        let inv_r = 1.0_f32 / res as f32;

        for _ in 0..80 {
            let mut pts_flat = Vec::with_capacity(n_pts * pt_cols);
            let mut targets = Vec::with_capacity(n_pts);
            for k in 0..n_pts {
                let p = if k < n_pts / 2 && !band_cells.is_empty() {
                    let cell = band_cells[rng.random_range(0..band_cells.len())];
                    let cx = cell / rr;
                    let cy = (cell / res) % res;
                    let cz = cell % res;
                    [
                        ((cx as f32 + rng.random::<f32>()) * inv_r) * 2.0 - 1.0,
                        ((cy as f32 + rng.random::<f32>()) * inv_r) * 2.0 - 1.0,
                        ((cz as f32 + rng.random::<f32>()) * inv_r) * 2.0 - 1.0,
                    ]
                } else {
                    [
                        rng.random_range(-1.0_f32..1.0),
                        rng.random_range(-1.0_f32..1.0),
                        rng.random_range(-1.0_f32..1.0),
                    ]
                };
                pts_flat.extend(p);
                if self.scale_encoding {
                    pts_flat.push(0.002_f32);
                }
                targets.push(sample_trilinear(p));
            }
            let pts_t = tensor(pts_flat, [n_pts, pt_cols], &self.device);
            let target_t = tensor(targets, [n_pts, 1], &self.device);
            let pending = self.enqueue_cached_device_valid(pts_t, n_pts, n_pts);
            let (pred_t, cached) = pending.into_device();
            let adj = ((pred_t - target_t).clamp(-0.5, 0.5) * (2.0 / n_pts as f32))
                .without_autodiff();
            let db1 = adj.clone().sum_dim(0);
            let stats = Tensor::cat(vec![zeros_2d(1, 4, &self.device), db1], 1);
            let mut grads = Gradients::default();
            self.accumulate_training_gradients_device(cached, adj, &mut grads);
            self.apply_gradients_packed_with_stats(grads, Some(&stats), 5e-3);
        }

        self.reset_optimizer_state();
        self.active = prev_active;
        let _ = self.device.sync();
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
        gpu_cache.iter().all(|c| {
            c.preactivation
                .as_ref()
                .is_none_or(|t| !t.is_require_grad())
                && c.feat.as_ref().is_none_or(|t| !t.is_require_grad())
        }),
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

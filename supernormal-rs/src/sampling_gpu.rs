//! GPU sampling and occupancy primitives. Outputs use FP32 indices, bounded by
//! the host to the exact-integer range. Sample order is stable within each ray.
use burn_backend::ExecutionError;
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId,
        TensorIr,
    },
};
use cubecl::prelude::*;
type GpuBackend = Fusion<CubeBackend>;

/// Retrieve a float tensor handle guaranteed to have strictly contiguous row-major
/// strides. Zero-cost no-op on WGPU; ensures CUDA pitched 2D tensors match flat
/// row-major indexing in custom CubeCL kernels.
#[inline]
fn get_contig(
    handles: &mut HandleContainer<CubeFusionHandle>,
    ir: &TensorIr,
) -> burn_cubecl::tensor::CubeTensor {
    let t = handles.get_float_tensor::<CubeBackend>(ir);
    if t.is_contiguous() {
        t
    } else {
        burn_cubecl::kernel::into_contiguous(t)
    }
}

/// Round dynamic allocation row counts to a coarse bucket (4 steps per power-of-two
/// octave) so Burn's wgpu memory pool reuses buffers in O(1) across steps instead
/// of allocating a fresh GPU buffer every time sample/point counts fluctuate.
#[inline]
pub fn bucket_capacity(n: usize) -> usize {
    let max_f32_int = (1usize << 24) - 1;
    if n <= 4096 {
        return 4096;
    }
    let p = n.next_power_of_two();
    let half = p >> 1;
    let step = (half >> 2).max(1024);
    (n.div_ceil(step) * step).min(max_f32_int)
}

#[cube(launch)]
fn march_kernel(
    rays: &Tensor<f32>,
    flags: &Tensor<f32>,
    samples: &mut Tensor<f32>,
    counts: &mut Tensor<f32>,
) {
    let ray = ABSOLUTE_POS;
    if ray >= rays.shape(0) {
        terminate!();
    }
    let b = ray * rays.shape(1);
    let ox = rays[b];
    let oy = rays[b + 1];
    let oz = rays[b + 2];
    let dx = rays[b + 3];
    let dy = rays[b + 4];
    let dz = rays[b + 5];
    let near = rays[b + 6];
    let far = rays[b + 7];
    let step = rays[b + 8];
    let offset = usize::cast_from(rays[b + 9]);
    let capacity = usize::cast_from(rays[b + 10]);
    let r = usize::cast_from(rays[b + 11]);
    let half_r = f32::cast_from(r) * 0.5_f32;
    let r_max = f32::cast_from(r - 1);
    let safe_dx = if f32::abs(dx) < 1e-10_f32 {
        1e-10_f32
    } else {
        dx
    };
    let safe_dy = if f32::abs(dy) < 1e-10_f32 {
        1e-10_f32
    } else {
        dy
    };
    let safe_dz = if f32::abs(dz) < 1e-10_f32 {
        1e-10_f32
    } else {
        dz
    };
    let inv_dx = 1.0_f32 / (safe_dx * half_r);
    let inv_dy = 1.0_f32 / (safe_dy * half_r);
    let inv_dz = 1.0_f32 / (safe_dz * half_r);
    let inv_step = 1.0_f32 / step;
    let mut count = 0usize;
    let mut ep_count = 0usize;
    let mut prev_end_bits = 0u32;
    let mut s = 0usize;
    while s < capacity {
        let t = near + f32::cast_from(s) * step;
        let end = t + step;
        let mid = (t + end) * 0.5_f32;
        if mid >= far {
            break;
        }
        let px = ox + dx * mid;
        let py = oy + dy * mid;
        let pz = oz + dz * mid;
        let inside = px >= -1.0_f32
            && px <= 1.0_f32
            && py >= -1.0_f32
            && py <= 1.0_f32
            && pz >= -1.0_f32
            && pz <= 1.0_f32;
        let gx = (px + 1.0_f32) * half_r;
        let gy = (py + 1.0_f32) * half_r;
        let gz = (pz + 1.0_f32) * half_r;
        let cx = usize::cast_from(f32::clamp(gx, 0.0_f32, r_max));
        let cy = usize::cast_from(f32::clamp(gy, 0.0_f32, r_max));
        let cz = usize::cast_from(f32::clamp(gz, 0.0_f32, r_max));
        let cell = (cx * r + cy) * r + cz;
        if inside && flags[cell] > 0.0_f32 {
            samples[(offset + count) * 2] = t;
            samples[(offset + count) * 2 + 1] = end;
            let t_bits = u32::reinterpret(t);
            if count > 0 && t_bits == prev_end_bits {
                ep_count += 1;
            } else {
                ep_count += 2;
            }
            prev_end_bits = u32::reinterpret(end);
            count += 1;
            s += 1;
        } else if inside {
            let pl_x = if dx >= 0.0_f32 {
                f32::floor(gx) + 1.0_f32
            } else {
                f32::ceil(gx) - 1.0_f32
            };
            let pl_y = if dy >= 0.0_f32 {
                f32::floor(gy) + 1.0_f32
            } else {
                f32::ceil(gy) - 1.0_f32
            };
            let pl_z = if dz >= 0.0_f32 {
                f32::floor(gz) + 1.0_f32
            } else {
                f32::ceil(gz) - 1.0_f32
            };
            let dt_voxel = f32::min(
                (pl_x - gx) * inv_dx,
                f32::min((pl_y - gy) * inv_dy, (pl_z - gz) * inv_dz),
            );
            let skip = usize::cast_from(f32::clamp(
                f32::ceil(dt_voxel * inv_step),
                1.0_f32,
                f32::cast_from(capacity - s),
            ));
            s += usize::min(skip, capacity - s);
        } else {
            s += 1;
        }
    }
    if counts.shape(1) == 2 {
        counts[ray * 2] = f32::cast_from(count);
        counts[ray * 2 + 1] = f32::cast_from(ep_count);
    } else {
        counts[ray] = f32::cast_from(count);
    }
}

#[cube(launch)]
fn compact_kernel(
    rays: &Tensor<f32>,
    samples: &Tensor<f32>,
    prefix: &Tensor<f32>,
    out: &mut Tensor<f32>,
) {
    let ray = ABSOLUTE_POS;
    if ray >= rays.shape(0) {
        terminate!();
    }
    let stride = prefix.shape(1);
    let begin = if ray == 0 {
        0usize
    } else {
        usize::cast_from(prefix[(ray - 1) * stride])
    };
    let end = usize::cast_from(prefix[ray * stride]);
    let offset = usize::cast_from(rays[ray * rays.shape(1) + 9]);
    for i in 0..end - begin {
        out[(begin + i) * 3] = f32::cast_from(ray);
        out[(begin + i) * 3 + 1] = samples[(offset + i) * 2];
        out[(begin + i) * 3 + 2] = samples[(offset + i) * 2 + 1];
    }
}

/// Fused ray-level compaction + deduplicated center-ray endpoint materialization:
/// writes `marched` `[total_marched, 3]`, `unique_pts` `[unique_ep_count, 3|4]`,
/// `pairs` `[total_marched, 2]`, and `ranges` `[rays, 2]` in a single pass.
#[cube(launch)]
fn compact_and_center_endpoints_kernel(
    rays: &Tensor<f32>,
    samples: &Tensor<f32>,
    prefix: &Tensor<f32>,
    marched: &mut Tensor<f32>,
    unique_pts: &mut Tensor<f32>,
    pairs: &mut Tensor<f32>,
    ranges: &mut Tensor<f32>,
    #[comptime] scale_encoding: bool,
) {
    let ray = ABSOLUTE_POS;
    if ray >= rays.shape(0) {
        terminate!();
    }
    let s_begin = if ray == 0 {
        0usize
    } else {
        usize::cast_from(prefix[(ray - 1) * 2])
    };
    let s_end = usize::cast_from(prefix[ray * 2]);
    let mut ep_cursor = if ray == 0 {
        0usize
    } else {
        usize::cast_from(prefix[(ray - 1) * 2 + 1])
    };
    ranges[ray * 2] = f32::cast_from(s_begin);
    ranges[ray * 2 + 1] = f32::cast_from(s_end);
    let r_stride = rays.shape(1);
    let rb = ray * r_stride;
    let offset = usize::cast_from(rays[rb + 9]);
    let ox = rays[rb];
    let oy = rays[rb + 1];
    let oz = rays[rb + 2];
    let dx = rays[rb + 3];
    let dy = rays[rb + 4];
    let dz = rays[rb + 5];
    let mut scale_per_t = 0.0_f32;
    if scale_encoding {
        scale_per_t = rays[rb + 12];
    }
    let p_stride = if scale_encoding { 4usize } else { 3usize };
    let ray_f = f32::cast_from(ray);
    let mut prev_end_bits = 0u32;
    for i in 0..s_end - s_begin {
        let t_start = samples[(offset + i) * 2];
        let t_end = samples[(offset + i) * 2 + 1];
        let m_idx = s_begin + i;
        marched[m_idx * 3] = ray_f;
        marched[m_idx * 3 + 1] = t_start;
        marched[m_idx * 3 + 2] = t_end;

        let t_bits = u32::reinterpret(t_start);
        let reuse = i > 0 && t_bits == prev_end_bits;
        prev_end_bits = u32::reinterpret(t_end);
        let mut start_id = ep_cursor;
        let mut end_id = ep_cursor + 1;
        if reuse {
            start_id = ep_cursor - 1;
            end_id = ep_cursor;
            ep_cursor += 1;
        } else {
            unique_pts[start_id * p_stride] = ox + dx * t_start;
            unique_pts[start_id * p_stride + 1] = oy + dy * t_start;
            unique_pts[start_id * p_stride + 2] = oz + dz * t_start;
            if scale_encoding {
                unique_pts[start_id * 4 + 3] = t_start * scale_per_t;
            }
            ep_cursor += 2;
        }
        unique_pts[end_id * p_stride] = ox + dx * t_end;
        unique_pts[end_id * p_stride + 1] = oy + dy * t_end;
        unique_pts[end_id * p_stride + 2] = oz + dz * t_end;
        if scale_encoding {
            unique_pts[end_id * 4 + 3] = t_end * scale_per_t;
        }
        pairs[m_idx * 2] = f32::cast_from(start_id);
        pairs[m_idx * 2 + 1] = f32::cast_from(end_id);
    }
}

/// Build `endpoints` `[ep_count, 2]`, `descriptors` `[vis_count, 5]`,
/// compact `mapping` `[vis_count, 1]` (storing `start_ep * 9`, since `end_ep`
/// is always `start_ep + 1` so the 18 sample points are always contiguous at
/// `start_ep * 9 + 0..18`), and compact endpoint-level `reverse` `[ep_count, 2]`
/// directly on the GPU without the 9-pixel inner loop.
#[cube(launch)]
fn build_visible_geometry_tables_kernel(
    marched: &Tensor<f32>,
    vis_m_begin: &Tensor<f32>,
    vis_prefix: &Tensor<f32>,
    endpoints: &mut Tensor<f32>,
    descriptors: &mut Tensor<f32>,
    mapping: &mut Tensor<f32>,
    reverse: &mut Tensor<f32>,
    patch_start: u32,
    patch_count: u32,
    s_base: u32,
    ep_base: u32,
) {
    let local_ray = ABSOLUTE_POS;
    let ray = usize::cast_from(patch_start) + local_ray;
    if local_ray >= usize::cast_from(patch_count) || ray >= vis_prefix.shape(0) {
        terminate!();
    }
    let m_begin = usize::cast_from(vis_m_begin[ray]);
    let v_begin_global = if ray == 0 {
        0usize
    } else {
        usize::cast_from(vis_prefix[(ray - 1) * 2])
    };
    let v_end_global = usize::cast_from(vis_prefix[ray * 2]);
    if v_end_global <= v_begin_global {
        terminate!();
    }
    let v_begin = v_begin_global - usize::cast_from(s_base);
    let v_len = v_end_global - v_begin_global;
    let mut ep_cursor = (if ray == 0 {
        0usize
    } else {
        usize::cast_from(vis_prefix[(ray - 1) * 2 + 1])
    }) - usize::cast_from(ep_base);

    let patch_f = f32::cast_from(local_ray);
    for k in 0..v_len {
        let m_idx = m_begin + k;
        let t_start = marched[m_idx * 3 + 1];
        let t_end = marched[m_idx * 3 + 2];
        let mut reuse_prev = false;
        if k > 0 {
            let s_bits = u32::reinterpret(t_start);
            let e_prev = u32::reinterpret(marched[(m_idx - 1) * 3 + 2]);
            if s_bits == e_prev {
                reuse_prev = true;
            }
        }
        let mut reuse_next = false;
        if k + 1 < v_len {
            let e_bits = u32::reinterpret(t_end);
            let s_next = u32::reinterpret(marched[(m_idx + 1) * 3 + 1]);
            if e_bits == s_next {
                reuse_next = true;
            }
        }
        let mut start_ep = ep_cursor;
        let mut end_ep = ep_cursor + 1;
        if reuse_prev {
            start_ep = ep_cursor - 1;
            end_ep = ep_cursor;
            ep_cursor += 1;
        } else {
            endpoints[start_ep * 2] = patch_f;
            endpoints[start_ep * 2 + 1] = t_start;
            ep_cursor += 2;
        }
        endpoints[end_ep * 2] = patch_f;
        endpoints[end_ep * 2 + 1] = t_end;

        let vis_idx = v_begin + k;
        descriptors[vis_idx * 5] = patch_f;
        descriptors[vis_idx * 5 + 1] = t_start;
        descriptors[vis_idx * 5 + 2] = t_end;
        descriptors[vis_idx * 5 + 3] = f32::cast_from(start_ep);
        descriptors[vis_idx * 5 + 4] = f32::cast_from(end_ep);

        // Store the base point index `start_ep * 9` once per sample (18x smaller
        // than storing 18 indices per sample) and store endpoint-level reverse
        // codes `vis_idx * 2 + is_end` once per endpoint (9x smaller).
        mapping[vis_idx] = f32::cast_from(start_ep * 9);
        if !reuse_prev {
            reverse[start_ep * 2] = f32::cast_from(vis_idx * 2);
            reverse[start_ep * 2 + 1] = -1.0_f32;
        }
        reverse[end_ep * 2] = f32::cast_from(vis_idx * 2 + 1);
        reverse[end_ep * 2 + 1] = -1.0_f32;
        if reuse_next {
            reverse[end_ep * 2 + 1] = f32::cast_from((vis_idx + 1) * 2);
        }
    }
}

#[cube]
fn pcg_hash_u32(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

#[cube]
fn hash_to_unit_f32(h: u32) -> f32 {
    f32::cast_from(h >> 8u32) * (1.0_f32 / 16777216.0_f32)
}

/// Generate jittered 3D cell coordinates `[valid_n, 3]` directly on the GPU for
/// a contiguous slice of occupancy cells `[start_cell .. start_cell + valid_n]`.
#[cube(launch)]
fn occupancy_grid_points_kernel(
    points: &mut Tensor<f32>,
    valid_n: u32,
    r: u32,
    start_cell: u32,
    seed: u32,
) {
    let k = ABSOLUTE_POS;
    if k >= usize::cast_from(valid_n) {
        terminate!();
    }
    let cell = start_cell + u32::cast_from(k);
    let rr = r * r;
    let cx = cell / rr;
    let cy = (cell / r) % r;
    let cz = cell % r;
    let inv_r = 1.0_f32 / f32::cast_from(r);
    let base = cell * 3u32;
    let u0 = hash_to_unit_f32(pcg_hash_u32(base ^ seed));
    let u1 = hash_to_unit_f32(pcg_hash_u32((base + 1u32) ^ (seed + 0x9E3779B9u32)));
    let u2 = hash_to_unit_f32(pcg_hash_u32((base + 2u32) ^ (seed + 0x3C6EF372u32)));
    points[k * 3] = ((f32::cast_from(cx) + u0) * inv_r) * 2.0_f32 - 1.0_f32;
    points[k * 3 + 1] = ((f32::cast_from(cy) + u1) * inv_r) * 2.0_f32 - 1.0_f32;
    points[k * 3 + 2] = ((f32::cast_from(cz) + u2) * inv_r) * 2.0_f32 - 1.0_f32;
}

/// Apply occupancy EMA update in-place for a contiguous cell slice `[start_cell .. +valid_n]`.
#[cube(launch)]
fn occupancy_apply_slice_kernel(
    sdf: &Tensor<f32>,
    values: &mut Tensor<f32>,
    valid_n: u32,
    start_cell: u32,
) {
    let k = ABSOLUTE_POS;
    if k >= usize::cast_from(valid_n) {
        terminate!();
    }
    let cell = usize::cast_from(start_cell) + k;
    let density = 1.0_f32 / (1.0_f32 + f32::exp(80.0_f32 * sdf[k]));
    values[cell] = f32::max(values[cell] * 0.95_f32, density);
}

/// Count selected occupancy cells per block (`flags[c] > 0.0` or 1 stratified random
/// cell per 4-cell group), selecting 100% of occupied cells + 25% (`n / 4`) of
/// unoccupied cells without duplicates or CPU readbacks of `flags`.
#[cube(launch)]
fn occupancy_count_kernel(
    flags: &Tensor<f32>,
    counts: &mut Tensor<f32>,
    total_cells: u32,
    cells_per_thread: u32,
    seed: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid >= counts.shape(0) {
        terminate!();
    }
    let start = tid * usize::cast_from(cells_per_thread);
    let end = usize::min(
        start + usize::cast_from(cells_per_thread),
        usize::cast_from(total_cells),
    );
    let mut count = 0usize;
    for c in start..end {
        let cu = u32::cast_from(c);
        let pick = pcg_hash_u32((cu >> 2u32) ^ seed) & 3u32;
        if flags[c] > 0.0_f32 || (cu & 3u32) == pick {
            count += 1usize;
        }
    }
    counts[tid] = f32::cast_from(count);
}

/// Compact selected cell IDs `[total_selected, 1]` and generate jittered 3D query
/// points `[total_selected, 3]` directly on the GPU.
#[cube(launch)]
fn occupancy_compact_points_kernel(
    flags: &Tensor<f32>,
    prefix: &Tensor<f32>,
    selected_ids: &mut Tensor<f32>,
    points: &mut Tensor<f32>,
    total_cells: u32,
    cells_per_thread: u32,
    r: u32,
    seed: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid >= prefix.shape(0) {
        terminate!();
    }
    let start = tid * usize::cast_from(cells_per_thread);
    let end = usize::min(
        start + usize::cast_from(cells_per_thread),
        usize::cast_from(total_cells),
    );
    let mut cursor = if tid == 0 {
        0usize
    } else {
        usize::cast_from(prefix[tid - 1])
    };
    let rr = r * r;
    let inv_r = 1.0_f32 / f32::cast_from(r);
    for c in start..end {
        let cu = u32::cast_from(c);
        let pick = pcg_hash_u32((cu >> 2u32) ^ seed) & 3u32;
        if flags[c] > 0.0_f32 || (cu & 3u32) == pick {
            selected_ids[cursor] = f32::cast_from(c);
            let cx = cu / rr;
            let cy = (cu / r) % r;
            let cz = cu % r;
            let base = cu * 3u32;
            let u0 = hash_to_unit_f32(pcg_hash_u32(base ^ seed));
            let u1 = hash_to_unit_f32(pcg_hash_u32((base + 1u32) ^ (seed + 0x9E3779B9u32)));
            let u2 = hash_to_unit_f32(pcg_hash_u32((base + 2u32) ^ (seed + 0x3C6EF372u32)));
            points[cursor * 3] = ((f32::cast_from(cx) + u0) * inv_r) * 2.0_f32 - 1.0_f32;
            points[cursor * 3 + 1] = ((f32::cast_from(cy) + u1) * inv_r) * 2.0_f32 - 1.0_f32;
            points[cursor * 3 + 2] = ((f32::cast_from(cz) + u2) * inv_r) * 2.0_f32 - 1.0_f32;
            cursor += 1usize;
        }
    }
}

/// Apply occupancy EMA update in-place for compacted unique `selected_ids`.
#[cube(launch)]
fn occupancy_apply_selected_kernel(
    selected_ids: &Tensor<f32>,
    sdf: &Tensor<f32>,
    values: &mut Tensor<f32>,
    valid_n: u32,
) {
    let k = ABSOLUTE_POS;
    if k >= usize::cast_from(valid_n) {
        terminate!();
    }
    let cell = usize::cast_from(selected_ids[k]);
    let density = 1.0_f32 / (1.0_f32 + f32::exp(80.0_f32 * sdf[k]));
    values[cell] = f32::max(values[cell] * 0.95_f32, density);
}

#[derive(Debug)]
struct MarchOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for MarchOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([r, f, s, c], [o0, o1]) = self.desc.as_fixed();
        let [r, f, s, c] = [r, f, s, c].map(|x| get_contig(handles, x));
        let client = r.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.MarchOp");
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new_1d(32);
        march_kernel::launch(
            &client,
            CubeCount::Static(n.div_ceil(32) as u32, 1, 1),
            dim,
            r.into_tensor_arg(),
            f.into_tensor_arg(),
            s.clone().into_tensor_arg(),
            c.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&o0.id, s);
        handles.register_float_tensor::<CubeBackend>(&o1.id, c);
        Ok(())
    }
}
pub fn march(inputs: [burn::tensor::Tensor<2>; 4]) -> [burn::tensor::Tensor<2>; 2] {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [2usize, 3usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_march",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
    );
    let result: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            MarchOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
struct CompactOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for CompactOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([r, s, p, o], [o0]) = self.desc.as_fixed();
        let [r, s, p, o] = [r, s, p, o].map(|x| get_contig(handles, x));
        let client = r.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.CompactOp");
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        compact_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            r.into_tensor_arg(),
            s.into_tensor_arg(),
            p.into_tensor_arg(),
            o.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&o0.id, o);
        Ok(())
    }
}
pub fn compact(inputs: [burn::tensor::Tensor<2>; 4]) -> [burn::tensor::Tensor<2>; 1] {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [3usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_compact",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
    );
    let result: [_; 1] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            CompactOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
struct CompactAndCenterEndpointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for CompactAndCenterEndpointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([r, s, p, m, u, pr, rg], [om, ou, opr, org]) = self.desc.as_fixed();
        let [r, s, p, m, u, pr, rg] =
            [r, s, p, m, u, pr, rg].map(|x| get_contig(handles, x));
        let scale_encoding = u.meta.shape().dims::<2>()[1] == 4;
        let client = r.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.CompactAndCenterEndpointsOp");
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new_1d(32);
        compact_and_center_endpoints_kernel::launch(
            &client,
            CubeCount::Static(n.div_ceil(32) as u32, 1, 1),
            dim,
            r.into_tensor_arg(),
            s.into_tensor_arg(),
            p.into_tensor_arg(),
            m.clone().into_tensor_arg(),
            u.clone().into_tensor_arg(),
            pr.clone().into_tensor_arg(),
            rg.clone().into_tensor_arg(),
            scale_encoding,
        );
        handles.register_float_tensor::<CubeBackend>(&om.id, m);
        handles.register_float_tensor::<CubeBackend>(&ou.id, u);
        handles.register_float_tensor::<CubeBackend>(&opr.id, pr);
        handles.register_float_tensor::<CubeBackend>(&org.id, rg);
        Ok(())
    }
}
pub fn compact_and_center_endpoints(
    rays: burn::tensor::Tensor<2>,
    samples: burn::tensor::Tensor<2>,
    prefix: burn::tensor::Tensor<2>,
    total_marched: usize,
    padded_ep_cap: usize,
    scale_encoding: bool,
) -> [burn::tensor::Tensor<2>; 4] {
    let n_rays = rays.dims()[0];
    let device = rays.device();
    let m_cap = bucket_capacity(total_marched);
    let pt_cols = if scale_encoding { 4 } else { 3 };
    let m_buf = crate::field::empty_2d(m_cap, 3, &device);
    let u_buf = crate::field::empty_2d(padded_ep_cap, pt_cols, &device);
    let pr_buf = crate::field::empty_2d(m_cap, 2, &device);
    let rg_buf = crate::field::empty_2d(n_rays, 2, &device);
    let inputs = [rays, samples, prefix, m_buf, u_buf, pr_buf, rg_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [3usize, 4usize, 5usize, 6usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_compact_and_center_endpoints",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
    );
    let result: [_; 4] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            CompactAndCenterEndpointsOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
struct BuildVisibleGeometryTablesOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for BuildVisibleGeometryTablesOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([m, mp, vp, e, d, map, r], [oe, od, omap, or]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(patch_start),
            ScalarIr::UInt(patch_count),
            ScalarIr::UInt(s_base),
            ScalarIr::UInt(ep_base),
            ScalarIr::UInt(_ep_count),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid visible geometry table scalars")
        };
        let [m, mp, vp, e, d, map, r] =
            [m, mp, vp, e, d, map, r].map(|x| get_contig(handles, x));
        let client = m.client.clone();
        let _timing = crate::diagnostics::start(&client, "geometry_tables");
        let n_rays = (*patch_count as usize).max(1);
        let dim = CubeDim::new_1d(32);
        build_visible_geometry_tables_kernel::launch(
            &client,
            CubeCount::Static(n_rays.div_ceil(32) as u32, 1, 1),
            dim,
            m.into_tensor_arg(),
            mp.into_tensor_arg(),
            vp.into_tensor_arg(),
            e.clone().into_tensor_arg(),
            d.clone().into_tensor_arg(),
            map.clone().into_tensor_arg(),
            r.clone().into_tensor_arg(),
            *patch_start as u32,
            *patch_count as u32,
            *s_base as u32,
            *ep_base as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&oe.id, e);
        handles.register_float_tensor::<CubeBackend>(&od.id, d);
        handles.register_float_tensor::<CubeBackend>(&omap.id, map);
        handles.register_float_tensor::<CubeBackend>(&or.id, r);
        Ok(())
    }
}
pub fn build_visible_geometry_tables(
    marched: burn::tensor::Tensor<2>,
    march_prefix: burn::tensor::Tensor<2>,
    vis_prefix: burn::tensor::Tensor<2>,
    patch_start: usize,
    patch_count: usize,
    s_base: usize,
    ep_base: usize,
    vis_count: usize,
    ep_count: usize,
) -> [burn::tensor::Tensor<2>; 4] {
    let device = marched.device();
    let vis_cap = bucket_capacity(vis_count);
    let ep_cap = bucket_capacity(ep_count);
    let e_buf = crate::field::empty_2d(ep_cap, 2, &device);
    let d_buf = crate::field::empty_2d(vis_cap, 5, &device);
    let map_buf = crate::field::empty_2d(vis_cap, 1, &device);
    let r_buf = crate::field::empty_2d(ep_cap, 2, &device);
    let inputs = [
        marched,
        march_prefix,
        vis_prefix,
        e_buf,
        d_buf,
        map_buf,
        r_buf,
    ]
    .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [3usize, 4usize, 5usize, 6usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_build_visible_geometry_tables",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(patch_start as u64),
            ScalarIr::UInt(patch_count as u64),
            ScalarIr::UInt(s_base as u64),
            ScalarIr::UInt(ep_base as u64),
            ScalarIr::UInt(ep_count as u64),
        ],
    );
    let result: [_; 4] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            BuildVisibleGeometryTablesOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
struct OccupancyGridPointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyGridPointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p], [op]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(valid_n),
            ScalarIr::UInt(r),
            ScalarIr::UInt(start_cell),
            ScalarIr::UInt(seed),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid occupancy_grid_points scalars")
        };
        let p = get_contig(handles, p);
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.OccupancyGridPointsOp");
        let n = (*valid_n as usize).max(1);
        let dim = CubeDim::new(&client, n);
        occupancy_grid_points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            p.clone().into_tensor_arg(),
            *valid_n as u32,
            *r as u32,
            *start_cell as u32,
            *seed as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        Ok(())
    }
}
pub fn occupancy_grid_points(
    padded_n: usize,
    valid_n: usize,
    r: usize,
    start_cell: usize,
    seed: u32,
    device: &burn::tensor::Device,
) -> burn::tensor::Tensor<2> {
    let p_buf = burn::tensor::Tensor::<2>::empty([padded_n, 3], device);
    let [p] = [p_buf].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = p.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), p.shape.clone(), p.dtype);
    let desc = CustomOpIr::with_scalars(
        "supernormal_occupancy_grid_points",
        &[p.into_ir()],
        &[out],
        vec![
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(r as u64),
            ScalarIr::UInt(start_cell as u64),
            ScalarIr::UInt(seed as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyGridPointsOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

#[derive(Debug)]
struct OccupancyApplySliceOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyApplySliceOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([s, v], [ov]) = self.desc.as_fixed();
        let [ScalarIr::UInt(valid_n), ScalarIr::UInt(start_cell)] = self.desc.scalars.as_slice()
        else {
            panic!("invalid occupancy_apply_slice scalars")
        };
        let [s, v] = [s, v].map(|x| get_contig(handles, x));
        let v = if v.can_mut() { v } else { v.copy() };
        let client = v.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.OccupancyApplySliceOp");
        let n = (*valid_n as usize).max(1);
        let dim = CubeDim::new(&client, n);
        occupancy_apply_slice_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            s.into_tensor_arg(),
            v.clone().into_tensor_arg(),
            *valid_n as u32,
            *start_cell as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&ov.id, v);
        Ok(())
    }
}
pub fn occupancy_apply_slice(
    sdf: burn::tensor::Tensor<2>,
    values: burn::tensor::Tensor<2>,
    valid_n: usize,
    start_cell: usize,
) -> burn::tensor::Tensor<2> {
    let inputs = [sdf, values].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[1].shape.clone(),
        inputs[1].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_occupancy_apply_slice",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(start_cell as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyApplySliceOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

#[derive(Debug)]
struct OccupancyCountOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyCountOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([f, c], [oc]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(total_cells),
            ScalarIr::UInt(cells_per_thread),
            ScalarIr::UInt(seed),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid occupancy_count scalars")
        };
        let [f, c] = [f, c].map(|x| get_contig(handles, x));
        let client = f.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.OccupancyCountOp");
        let n = c.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        occupancy_count_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            f.into_tensor_arg(),
            c.clone().into_tensor_arg(),
            *total_cells as u32,
            *cells_per_thread as u32,
            *seed as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&oc.id, c);
        Ok(())
    }
}
pub fn occupancy_count(
    flags: burn::tensor::Tensor<2>,
    num_blocks: usize,
    total_cells: usize,
    cells_per_thread: usize,
    seed: u32,
) -> burn::tensor::Tensor<2> {
    let device = flags.device();
    let c_buf = burn::tensor::Tensor::<2>::empty([num_blocks, 1], &device);
    let inputs = [flags, c_buf].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[1].shape.clone(),
        inputs[1].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_occupancy_count",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![
            ScalarIr::UInt(total_cells as u64),
            ScalarIr::UInt(cells_per_thread as u64),
            ScalarIr::UInt(seed as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyCountOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

#[derive(Debug)]
struct OccupancyCompactPointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyCompactPointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([f, pr, id, pt], [oid, opt]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(total_cells),
            ScalarIr::UInt(cells_per_thread),
            ScalarIr::UInt(r),
            ScalarIr::UInt(seed),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid occupancy_compact_points scalars")
        };
        let [f, pr, id, pt] = [f, pr, id, pt].map(|x| get_contig(handles, x));
        let client = f.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.OccupancyCompactPointsOp");
        let n = pr.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        occupancy_compact_points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            f.into_tensor_arg(),
            pr.into_tensor_arg(),
            id.clone().into_tensor_arg(),
            pt.clone().into_tensor_arg(),
            *total_cells as u32,
            *cells_per_thread as u32,
            *r as u32,
            *seed as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&oid.id, id);
        handles.register_float_tensor::<CubeBackend>(&opt.id, pt);
        Ok(())
    }
}
pub fn occupancy_compact_points(
    flags: burn::tensor::Tensor<2>,
    prefix: burn::tensor::Tensor<2>,
    total_selected: usize,
    padded_pts_cap: usize,
    total_cells: usize,
    cells_per_thread: usize,
    r: usize,
    seed: u32,
) -> [burn::tensor::Tensor<2>; 2] {
    let device = flags.device();
    let id_cap = bucket_capacity(total_selected);
    let id_buf = burn::tensor::Tensor::<2>::empty([id_cap, 1], &device);
    let pt_buf = burn::tensor::Tensor::<2>::empty([padded_pts_cap, 3], &device);
    let inputs = [flags, prefix, id_buf, pt_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [2usize, 3usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_occupancy_compact_points",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(total_cells as u64),
            ScalarIr::UInt(cells_per_thread as u64),
            ScalarIr::UInt(r as u64),
            ScalarIr::UInt(seed as u64),
        ],
    );
    let result: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyCompactPointsOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
struct OccupancyApplySelectedOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyApplySelectedOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([id, s, v], [ov]) = self.desc.as_fixed();
        let [ScalarIr::UInt(valid_n)] = self.desc.scalars.as_slice() else {
            panic!("invalid occupancy_apply_selected scalars")
        };
        let [id, s, v] = [id, s, v].map(|x| get_contig(handles, x));
        let v = if v.can_mut() { v } else { v.copy() };
        let client = v.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.OccupancyApplySelectedOp");
        let n = (*valid_n as usize).max(1);
        let dim = CubeDim::new(&client, n);
        occupancy_apply_selected_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            id.into_tensor_arg(),
            s.into_tensor_arg(),
            v.clone().into_tensor_arg(),
            *valid_n as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&ov.id, v);
        Ok(())
    }
}
pub fn occupancy_apply_selected(
    selected_ids: burn::tensor::Tensor<2>,
    sdf: burn::tensor::Tensor<2>,
    values: burn::tensor::Tensor<2>,
    valid_n: usize,
) -> burn::tensor::Tensor<2> {
    let inputs =
        [selected_ids, sdf, values].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[2].shape.clone(),
        inputs[2].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_occupancy_apply_selected",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![ScalarIr::UInt(valid_n as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyApplySelectedOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

// Default: an 8x32 tile reuses each staged hidden derivative across all
// features and bias, instead of loading it independently for four 8x8 tiles.
// Keep the original 8x8 layout available for independent A/B comparisons.
#[cube(launch)]
fn parameter_kernel(
    dh: &Tensor<f32>,
    features: &Tensor<f32>,
    partials: &mut Tensor<f32>,
    pre: &Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    valid_rows: u32,
    #[comptime] tile_cols: usize,
    #[comptime] recompute: bool,
) {
    let unit = usize::cast_from(UNIT_POS);
    let tile = usize::cast_from(CUBE_POS_X);
    let part = usize::cast_from(CUBE_POS_Y);
    let units = 8 * tile_cols;
    let row = unit / tile_cols;
    let col = unit % tile_cols;
    let tiles_per_row = 32 / tile_cols;
    let row_base = (tile / tiles_per_row) * 8;
    let col_base = (tile % tiles_per_row) * tile_cols;
    let valid_n = usize::cast_from(valid_rows);
    let mut hs = Shared::new_slice(256usize);
    let mut xs = Shared::new_slice(32 * tile_cols);
    let mut sum = 0.0_f32;
    for block in 0..8usize {
        let sample_base = part * 256 + block * 32;
        for load in 0..4usize {
            let index = unit + load * units;
            if index < 256 {
                let sample = sample_base + index / 8;
                let channel = index % 8;
                let mut h = 0.0_f32;
                if sample < valid_n {
                    if recompute {
                        let d = ds[sample * ds.stride(0)];
                        if d != 0.0_f32 {
                            let scaled = pre
                                [sample * pre.stride(0) + (row_base + channel) * pre.stride(1)]
                                * 100.0_f32;
                            let exp_pos = f32::exp(f32::min(scaled, 20.0_f32));
                            let derivative = if scaled > 20.0_f32 {
                                1.0_f32
                            } else {
                                exp_pos / (1.0_f32 + exp_pos)
                            };
                            h = d * weights[2048 + row_base + channel] * derivative;
                        }
                    } else {
                        h = dh[sample * dh.stride(0) + (row_base + channel) * dh.stride(1)];
                    }
                }
                hs[index] = h;
            }
            let sample = sample_base + index / tile_cols;
            let feature = col_base + index % tile_cols;
            let mut x = 0.0_f32;
            if sample < valid_n {
                if feature == 31 {
                    x = 1.0_f32;
                } else {
                    x = features[sample * features.stride(0) + feature * features.stride(1)];
                }
            }
            xs[index] = x;
        }
        sync_cube();
        for j in 0..32usize {
            sum += hs[j * 8 + row] * xs[j * tile_cols + col];
        }
        // No next tile overwrites shared storage after the final block;
        // each thread writes only its own output accumulators below.
        if block < 7usize {
            sync_cube();
        }
    }
    partials[part * 2048 + (row_base + row) * 32 + col_base + col] = sum;
}

// 16x32 outputs, four accumulators per thread. Compared with the 8x32
// kernel this halves feature-tile loads and workgroup/barrier count per part.
// Derivatives are already computed by field backward: this path adds no exp().
#[cube(launch)]
fn parameter_register_kernel(
    dh: &Tensor<f32>,
    features: &Tensor<f32>,
    partials: &mut Tensor<f32>,
    valid_rows: u32,
    active_columns: u32,
) {
    let unit = usize::cast_from(UNIT_POS);
    let row_base = usize::cast_from(CUBE_POS_X) * 16;
    let part = usize::cast_from(CUBE_POS_Y);
    let row = unit / 16;
    let col = unit % 16;
    let valid_n = usize::cast_from(valid_rows);
    let active_cols = usize::cast_from(active_columns);
    // Pack active input features followed by bias into the left of the tile.
    // This avoids executing a second half-tile just to process column-31 bias.
    let first_active = col <= active_cols;
    let second_active = col + 16 <= active_cols;
    let mut hs = Shared::new_slice(512usize);
    let mut xs = Shared::new_slice(1024usize);
    let mut sum00 = 0.0_f32;
    let mut sum01 = 0.0_f32;
    let mut sum10 = 0.0_f32;
    let mut sum11 = 0.0_f32;
    for block in 0..8usize {
        let sample_base = part * 256 + block * 32;
        for load in 0..4usize {
            let index = unit + load * 128;
            let sample = sample_base + index / 16;
            let channel = row_base + index % 16;
            let mut h = 0.0_f32;
            if sample < valid_n {
                h = dh[sample * dh.stride(0) + channel * dh.stride(1)];
            }
            hs[index] = h;
        }
        for load in 0..8usize {
            let index = unit + load * 128;
            let sample = sample_base + index / 32;
            let feature = index % 32;
            let mut x = 0.0_f32;
            if sample < valid_n {
                if feature == active_cols {
                    x = 1.0_f32;
                } else if feature < active_cols {
                    x = features[sample * features.stride(0) + feature * features.stride(1)];
                }
            }
            xs[index] = x;
        }
        sync_cube();
        if first_active && second_active {
            for j in 0..32usize {
                let h0 = hs[j * 16 + row];
                let h1 = hs[j * 16 + row + 8];
                let x0 = xs[j * 32 + col];
                let x1 = xs[j * 32 + col + 16];
                sum00 += h0 * x0;
                sum01 += h0 * x1;
                sum10 += h1 * x0;
                sum11 += h1 * x1;
            }
        } else if first_active {
            for j in 0..32usize {
                let x = xs[j * 32 + col];
                sum00 += hs[j * 16 + row] * x;
                sum10 += hs[j * 16 + row + 8] * x;
            }
        } else if second_active {
            for j in 0..32usize {
                let x = xs[j * 32 + col + 16];
                sum01 += hs[j * 16 + row] * x;
                sum11 += hs[j * 16 + row + 8] * x;
            }
        }
        // No next tile overwrites shared storage after the final block;
        // each thread writes only its own output accumulators below.
        if block < 7usize {
            sync_cube();
        }
    }
    let row0 = part * 2048 + (row_base + row) * 32;
    let row1 = row0 + 256;
    // Every inactive dense output has a unique zero writer. Active/bias
    // locations are excluded, so remapped bias stores cannot race with zeroes.
    if col >= active_cols && col != 31 {
        partials[row0 + col] = 0.0_f32;
        partials[row1 + col] = 0.0_f32;
    }
    if col + 16 >= active_cols && col + 16 != 31 {
        partials[row0 + col + 16] = 0.0_f32;
        partials[row1 + col + 16] = 0.0_f32;
    }
    if first_active {
        let mut output_col = col;
        if col == active_cols {
            output_col = 31usize;
        }
        partials[row0 + output_col] = sum00;
        partials[row1 + output_col] = sum10;
    }
    if second_active {
        let mut output_col = col + 16;
        if col + 16 == active_cols {
            output_col = 31usize;
        }
        partials[row0 + output_col] = sum01;
        partials[row1 + output_col] = sum11;
    }
}

// A 32-neuron tile halves repeated feature staging; eight accumulators per thread.
#[cube(launch)]
fn parameter_register_wide_rows_kernel(
    dh: &Tensor<f32>,
    features: &Tensor<f32>,
    partials: &mut Tensor<f32>,
    valid_rows: u32,
    active_columns: u32,
) {
    let unit = usize::cast_from(UNIT_POS);
    let row_base = usize::cast_from(CUBE_POS_X) * 32;
    let part = usize::cast_from(CUBE_POS_Y);
    let row = unit / 16;
    let col = unit % 16;
    let valid_n = usize::cast_from(valid_rows);
    let active_cols = usize::cast_from(active_columns);
    // Pack active input features followed by bias into the left of the tile.
    // This avoids executing a second half-tile just to process column-31 bias.
    let first_active = col <= active_cols;
    let second_active = col + 16 <= active_cols;
    let mut hs = Shared::new_slice(1024usize);
    let mut xs = Shared::new_slice(1024usize);
    let mut sum00 = 0.0_f32;
    let mut sum01 = 0.0_f32;
    let mut sum10 = 0.0_f32;
    let mut sum11 = 0.0_f32;
    let mut sum20 = 0.0_f32;
    let mut sum21 = 0.0_f32;
    let mut sum30 = 0.0_f32;
    let mut sum31 = 0.0_f32;
    for block in 0..8usize {
        let sample_base = part * 256 + block * 32;
        for load in 0..8usize {
            let index = unit + load * 128;
            let sample = sample_base + index / 32;
            let channel = row_base + index % 32;
            let mut h = 0.0_f32;
            if sample < valid_n {
                h = dh[sample * dh.stride(0) + channel * dh.stride(1)];
            }
            hs[index] = h;
        }
        for load in 0..8usize {
            let index = unit + load * 128;
            let sample = sample_base + index / 32;
            let feature = index % 32;
            let mut x = 0.0_f32;
            if sample < valid_n {
                if feature == active_cols {
                    x = 1.0_f32;
                } else if feature < active_cols {
                    x = features[sample * features.stride(0) + feature * features.stride(1)];
                }
            }
            xs[index] = x;
        }
        sync_cube();
        if first_active && second_active {
            for j in 0..32usize {
                let x0 = xs[j * 32 + col];
                let x1 = xs[j * 32 + col + 16];
                let h0 = hs[j * 32 + row];
                sum00 += h0 * x0;
                sum01 += h0 * x1;
                let h1 = hs[j * 32 + row + 8];
                sum10 += h1 * x0;
                sum11 += h1 * x1;
                let h2 = hs[j * 32 + row + 16];
                sum20 += h2 * x0;
                sum21 += h2 * x1;
                let h3 = hs[j * 32 + row + 24];
                sum30 += h3 * x0;
                sum31 += h3 * x1;
            }
        } else if first_active {
            for j in 0..32usize {
                let x0 = xs[j * 32 + col];
                let h0 = hs[j * 32 + row];
                sum00 += h0 * x0;
                let h1 = hs[j * 32 + row + 8];
                sum10 += h1 * x0;
                let h2 = hs[j * 32 + row + 16];
                sum20 += h2 * x0;
                let h3 = hs[j * 32 + row + 24];
                sum30 += h3 * x0;
            }
        } else if second_active {
            for j in 0..32usize {
                let x1 = xs[j * 32 + col + 16];
                let h0 = hs[j * 32 + row];
                sum01 += h0 * x1;
                let h1 = hs[j * 32 + row + 8];
                sum11 += h1 * x1;
                let h2 = hs[j * 32 + row + 16];
                sum21 += h2 * x1;
                let h3 = hs[j * 32 + row + 24];
                sum31 += h3 * x1;
            }
        }
        // No next tile overwrites shared storage after the final block;
        // each thread writes only its own output accumulators below.
        if block < 7usize {
            sync_cube();
        }
    }
    let row0 = part * 2048 + (row_base + row + 0) * 32;
    let row1 = part * 2048 + (row_base + row + 8) * 32;
    let row2 = part * 2048 + (row_base + row + 16) * 32;
    let row3 = part * 2048 + (row_base + row + 24) * 32;
    // Disjoint writers cover the full dense partial, including inactive columns.
    if col >= active_cols && col != 31 {
        partials[row0 + col] = 0.0_f32;
        partials[row1 + col] = 0.0_f32;
        partials[row2 + col] = 0.0_f32;
        partials[row3 + col] = 0.0_f32;
    }
    if col + 16 >= active_cols && col + 16 != 31 {
        partials[row0 + col + 16] = 0.0_f32;
        partials[row1 + col + 16] = 0.0_f32;
        partials[row2 + col + 16] = 0.0_f32;
        partials[row3 + col + 16] = 0.0_f32;
    }
    if first_active {
        let mut output_col = col;
        if col == active_cols {
            output_col = 31usize;
        }
        partials[row0 + output_col] = sum00;
        partials[row1 + output_col] = sum10;
        partials[row2 + output_col] = sum20;
        partials[row3 + output_col] = sum30;
    }
    if second_active {
        let mut output_col = col + 16;
        if col + 16 == active_cols {
            output_col = 31usize;
        }
        partials[row0 + output_col] = sum01;
        partials[row1 + output_col] = sum11;
        partials[row2 + output_col] = sum21;
        partials[row3 + output_col] = sum31;
    }
}

#[derive(Debug)]
struct ParameterOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for ParameterOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([h, f, p, pre, ds, w], [out]) = self.desc.as_fixed();
        let [h, f, p, pre, ds, w] =
            [h, f, p, pre, ds, w].map(|x| get_contig(handles, x));
        let [
            ScalarIr::UInt(wide),
            ScalarIr::UInt(valid_rows),
            ScalarIr::UInt(recompute),
            ScalarIr::UInt(register_tiles),
            ScalarIr::UInt(active_columns),
            ScalarIr::UInt(wide_rows),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid parameter-gradient arguments")
        };
        let tile_cols = if *wide != 0 { 32usize } else { 8usize };
        let client = h.client.clone();
        let _timing = crate::diagnostics::start(&client, "parameter_gradients");
        let parts = (*valid_rows as usize).div_ceil(256).max(1);
        if *register_tiles != 0 && *wide != 0 && *recompute == 0 && *wide_rows != 0 {
            parameter_register_wide_rows_kernel::launch(
                &client,
                CubeCount::Static(2, parts as u32, 1),
                CubeDim::new_1d(128),
                h.into_tensor_arg(),
                f.into_tensor_arg(),
                p.clone().into_tensor_arg(),
                *valid_rows as u32,
                *active_columns as u32,
            );
        } else if *register_tiles != 0 && *wide != 0 && *recompute == 0 {
            parameter_register_kernel::launch(
                &client,
                CubeCount::Static(4, parts as u32, 1),
                CubeDim::new_1d(128),
                h.into_tensor_arg(),
                f.into_tensor_arg(),
                p.clone().into_tensor_arg(),
                *valid_rows as u32,
                *active_columns as u32,
            );
        } else {
            parameter_kernel::launch(
                &client,
                CubeCount::Static((2048 / (8 * tile_cols)) as u32, parts as u32, 1),
                CubeDim::new_1d((8 * tile_cols) as u32),
                h.into_tensor_arg(),
                f.into_tensor_arg(),
                p.clone().into_tensor_arg(),
                pre.into_tensor_arg(),
                ds.into_tensor_arg(),
                w.into_tensor_arg(),
                *valid_rows as u32,
                tile_cols,
                *recompute != 0,
            );
        }
        handles.register_float_tensor::<CubeBackend>(&out.id, p);
        Ok(())
    }
}
pub fn parameter_gradients(
    dh: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    valid_rows: usize,
    wide: bool,
    register_tiles: bool,
    active_columns: usize,
    wide_rows: bool,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    parameter_gradients_impl(
        dh.clone(),
        features,
        dh.clone(),
        dh.clone(),
        dh,
        valid_rows,
        wide,
        register_tiles,
        active_columns,
        wide_rows,
        false,
    )
}

pub fn parameter_gradients_recompute(
    pre: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    ds: burn::tensor::Tensor<2>,
    weights: burn::tensor::Tensor<2>,
    valid_rows: usize,
    wide: bool,
    register_tiles: bool,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    parameter_gradients_impl(
        pre.clone(),
        features,
        pre,
        ds,
        weights,
        valid_rows,
        wide,
        register_tiles,
        31,
        false,
        true,
    )
}

fn parameter_gradients_impl(
    dh: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    pre: burn::tensor::Tensor<2>,
    ds: burn::tensor::Tensor<2>,
    weights: burn::tensor::Tensor<2>,
    valid_rows: usize,
    wide: bool,
    register_tiles: bool,
    active_columns: usize,
    wide_rows: bool,
    recompute: bool,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    use burn::tensor::Tensor;
    assert!(active_columns <= 31);
    let parts = valid_rows.div_ceil(256).max(1);
    assert!(parts <= 65535, "parameter-gradient query too large");
    // Use uninitialized output allocation instead of zero-filling before full overwrite.
    let buffer = crate::field::empty_2d(parts, 2048, &dh.device());
    let inputs = [dh, features, buffer, pre, ds, weights]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[2].shape.clone(),
        inputs[2].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_parameter_gradients",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![
            ScalarIr::UInt(wide as u64),
            ScalarIr::UInt(valid_rows as u64),
            ScalarIr::UInt(recompute as u64),
            ScalarIr::UInt(register_tiles as u64),
            ScalarIr::UInt(active_columns as u64),
            ScalarIr::UInt(wide_rows as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            ParameterOp { desc },
        )
        .outputs();
    let summed = Tensor::<2>::from_primitive::<GpuBackend>(out)
        .sum_dim(0)
        .reshape([64, 32]);
    (
        summed.clone().slice([0..64, 0..31]),
        summed.slice([0..64, 31..32]).transpose(),
    )
}

// Separate launches establish a storage dependency: geometry reads the exact
// same rounded world coordinates that the field evaluates, including origin.
#[cube(launch)]
fn points_kernel(
    rays: &Tensor<f32>,
    endpoints: &Tensor<f32>,
    points: &mut Tensor<f32>,
    valid_points: u32,
    #[comptime] scale_encoding: bool,
) {
    let id = ABSOLUTE_POS;
    if id >= usize::cast_from(valid_points) {
        terminate!();
    }
    let endpoint = id / 9;
    let pixel = id % 9;
    let patch = usize::cast_from(endpoints[endpoint * 2]);
    let base = patch * 40;
    let t_center = endpoints[endpoint * 2 + 1];
    let t = t_center * rays[base + 30 + pixel];
    let p_stride = if scale_encoding { 4usize } else { 3usize };
    for d in 0..3usize {
        points[id * p_stride + d] = rays[base + d] + rays[base + 3 + pixel * 3 + d] * t;
    }
    if scale_encoding {
        points[id * 4 + 3] = t_center * rays[base + 39];
    }
}

#[cube]
fn point_distance(points: &Tensor<f32>, a: usize, b: usize) -> f32 {
    let stride = points.stride(0);
    let x = points[a * stride] - points[b * stride];
    let y = points[a * stride + 1] - points[b * stride + 1];
    let z = points[a * stride + 2] - points[b * stride + 2];
    (x * x + y * y + z * z).sqrt()
}

#[cube(launch)]
fn geometry_kernel(
    rays: &Tensor<f32>,
    samples: &Tensor<f32>,
    points: &Tensor<f32>,
    geometry: &mut Tensor<f32>,
    valid_sample_pixels: u32,
) {
    let id = ABSOLUTE_POS;
    if id >= usize::cast_from(valid_sample_pixels) {
        terminate!();
    }
    let sample = id / 9;
    let pixel = id % 9;
    let patch = usize::cast_from(samples[sample * 5]);
    let base = usize::cast_from(samples[sample * 5 + 3]) * 9;
    let row = pixel / 3;
    let col = pixel % 3;
    let mut l = pixel;
    let mut r = pixel;
    let mut u = pixel;
    let mut d = pixel;
    if col > 0 {
        l = pixel - 1;
    }
    if col < 2 {
        r = pixel + 1;
    }
    if row > 0 {
        u = pixel - 3;
    }
    if row < 2 {
        d = pixel + 3;
    }
    let dx = if col == 1 {
        point_distance(points, base + l, base + pixel)
            + point_distance(points, base + pixel, base + r)
    } else {
        point_distance(points, base + l, base + r)
    };
    let dy = if row == 1 {
        point_distance(points, base + u, base + pixel)
            + point_distance(points, base + pixel, base + d)
    } else {
        point_distance(points, base + u, base + d)
    };
    let dt = (samples[sample * 5 + 2] - samples[sample * 5 + 1]) * rays[patch * 40 + 30 + pixel];
    geometry[id * 3] = 1.0_f32 / dt;
    geometry[id * 3 + 1] = 1.0_f32 / dx;
    geometry[id * 3 + 2] = 1.0_f32 / dy;
}

#[derive(Debug)]
struct GeometryOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for GeometryOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([r, e, s, p, g], [po, go]) = self.desc.as_fixed();
        let [ScalarIr::UInt(valid_points), ScalarIr::UInt(valid_samples)] =
            self.desc.scalars.as_slice()
        else {
            panic!("invalid geometry scalars")
        };
        let [r, e, s, p, g] = [r, e, s, p, g].map(|x| get_contig(handles, x));
        let scale_encoding = p.meta.shape().dims::<2>()[1] == 4;
        let client = r.client.clone();
        let _timing = crate::diagnostics::start(&client, "geometry");
        let n_pts = (*valid_points as usize).max(1);
        let dim_pts = CubeDim::new(&client, n_pts);
        points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n_pts, dim_pts),
            dim_pts,
            r.clone().into_tensor_arg(),
            e.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            *valid_points as u32,
            scale_encoding,
        );
        let n_sp = ((*valid_samples as usize) * 9).max(1);
        let dim_sp = CubeDim::new(&client, n_sp);
        geometry_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n_sp, dim_sp),
            dim_sp,
            r.into_tensor_arg(),
            s.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            g.clone().into_tensor_arg(),
            n_sp as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&po.id, p);
        handles.register_float_tensor::<CubeBackend>(&go.id, g);
        Ok(())
    }
}

pub fn geometry(
    inputs: [burn::tensor::Tensor<2>; 5],
    valid_points: usize,
    valid_samples: usize,
) -> [burn::tensor::Tensor<2>; 2] {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [3usize, 4usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_geometry",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(valid_points as u64),
            ScalarIr::UInt(valid_samples as u64),
        ],
    );
    let outputs: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            GeometryOp { desc },
        )
        .outputs();
    outputs.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

/// Build `geom_rays` `[n_patches, 39]` and `packed_patches` `[n_patches, 123]`
/// in parallel across `n_patches * 9` threads on the GPU from compact `[n_patches, 39]`
/// patch descriptors (`[v, x, y, 9 * (nx, ny, nz, mask)]`) and the `[num_views, 24]`
/// camera matrix table (`view_meta`), avoiding multi-gigabyte per-pixel GPU buffers.
#[cube(launch)]
fn gather_patch_tables_kernel(
    coords: &Tensor<f32>,
    view_meta: &Tensor<f32>,
    geom_rays: &mut Tensor<f32>,
    packed_patches: &mut Tensor<f32>,
    patch_start_u: u32,
    s_base: f32,
    inv: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
) {
    let id = ABSOLUTE_POS;
    let n_patches = geom_rays.shape(0);
    if id >= n_patches * 9 {
        terminate!();
    }
    let p = id / 9;
    let i = id % 9;
    let patch_start = usize::cast_from(patch_start_u);

    let c_idx = (patch_start + p) * 39;
    let v = usize::cast_from(coords[c_idx]);
    let x = coords[c_idx + 1];
    let y = coords[c_idx + 2];

    let vm = v * 24;
    let ox = view_meta[vm];
    let oy = view_meta[vm + 1];
    let oz = view_meta[vm + 2];
    let plx = view_meta[vm + 3];
    let ply = view_meta[vm + 4];
    let plz = view_meta[vm + 5];
    let ux = view_meta[vm + 6];
    let uy = view_meta[vm + 7];
    let uz = view_meta[vm + 8];
    let vx = view_meta[vm + 9];
    let vy = view_meta[vm + 10];
    let vz = view_meta[vm + 11];
    let wx = view_meta[vm + 12];
    let wy = view_meta[vm + 13];
    let wz = view_meta[vm + 14];
    let k00 = view_meta[vm + 15];
    let k01 = view_meta[vm + 16];
    let k02 = view_meta[vm + 17];
    let k10 = view_meta[vm + 18];
    let k11 = view_meta[vm + 19];
    let k12 = view_meta[vm + 20];
    let k20 = view_meta[vm + 21];
    let k21 = view_meta[vm + 22];
    let k22 = view_meta[vm + 23];

    let gr = p * 40;
    let pb = p * 123;
    if i == 0 {
        geom_rays[gr] = ox;
        geom_rays[gr + 1] = oy;
        geom_rays[gr + 2] = oz;
        let g_ray = patch_start + p;
        packed_patches[pb] = -(f32::cast_from(g_ray)) - 1.0_f32;
        packed_patches[pb + 1] = s_base;
        packed_patches[pb + 2] = inv;
        packed_patches[pb + 3] = normal_den;
        packed_patches[pb + 4] = pixel_den;
        packed_patches[pb + 5] = sample_den;
    }

    // Center ray (px = 1, py = 1 -> xx = x, yy = y) dot plane:
    let c_cx = k00 * x + k01 * y + k02;
    let c_cy = k10 * x + k11 * y + k12;
    let c_cz = k20 * x + k21 * y + k22;
    let c_inv_len = 1.0_f32 / f32::sqrt(c_cx * c_cx + c_cy * c_cy + c_cz * c_cz);
    let d4x = (ux * c_cx + vx * c_cy + plx * c_cz) * c_inv_len;
    let d4y = (uy * c_cx + vy * c_cy + ply * c_cz) * c_inv_len;
    let d4z = (uz * c_cx + vz * c_cy + plz * c_cz) * c_inv_len;
    let center_dot = d4x * plx + d4y * ply + d4z * plz;
    if i == 4 {
        let ang_pix = f32::sqrt(f32::abs(k00 * k11)) * f32::abs(center_dot);
        let t_base_c = c_idx + 3 + 16;
        let gnx = coords[t_base_c];
        let gny = coords[t_base_c + 1];
        let gnz = coords[t_base_c + 2];
        let gmask = coords[t_base_c + 3];
        let cos_view = if gmask > 0.5_f32 {
            f32::clamp(-(d4x * gnx + d4y * gny + d4z * gnz), 0.15_f32, 1.0_f32)
        } else {
            1.0_f32
        };
        geom_rays[gr + 39] = ang_pix / f32::sqrt(cos_view);
    }

    let px = f32::cast_from(i % 3) - 1.0_f32;
    let py = f32::cast_from(i / 3) - 1.0_f32;
    let xx = x + px;
    let yy = y + py;
    let cx = k00 * xx + k01 * yy + k02;
    let cy = k10 * xx + k11 * yy + k12;
    let cz = k20 * xx + k21 * yy + k22;
    let inv_len = 1.0_f32 / f32::sqrt(cx * cx + cy * cy + cz * cz);
    let dx = (ux * cx + vx * cy + plx * cz) * inv_len;
    let dy = (uy * cx + vy * cy + ply * cz) * inv_len;
    let dz = (uz * cx + vz * cy + plz * cz) * inv_len;

    geom_rays[gr + 3 + i * 3] = dx;
    geom_rays[gr + 3 + i * 3 + 1] = dy;
    geom_rays[gr + 3 + i * 3 + 2] = dz;
    geom_rays[gr + 30 + i] = center_dot / (dx * plx + dy * ply + dz * plz);

    // Analytical 3x3 basis inverse [w_col / det, (v_col x dir) / det, (dir x u_col) / det]:
    let det = dx * wx + dy * wy + dz * wz;
    let inv_det = 1.0_f32 / det;
    let c0x = wx * inv_det;
    let c0y = wy * inv_det;
    let c0z = wz * inv_det;
    let c1x = (vy * dz - vz * dy) * inv_det;
    let c1y = (vz * dx - vx * dz) * inv_det;
    let c1z = (vx * dy - vy * dx) * inv_det;
    let c2x = (dy * uz - dz * uy) * inv_det;
    let c2y = (dz * ux - dx * uz) * inv_det;
    let c2z = (dx * uy - dy * ux) * inv_det;

    let vb = pb + 6 + i * 13;
    packed_patches[vb] = c0x;
    packed_patches[vb + 1] = c1x;
    packed_patches[vb + 2] = c2x;
    packed_patches[vb + 3] = c0y;
    packed_patches[vb + 4] = c1y;
    packed_patches[vb + 5] = c2y;
    packed_patches[vb + 6] = c0z;
    packed_patches[vb + 7] = c1z;
    packed_patches[vb + 8] = c2z;

    let t_base = c_idx + 3 + i * 4;
    packed_patches[vb + 9] = coords[t_base];
    packed_patches[vb + 10] = coords[t_base + 1];
    packed_patches[vb + 11] = coords[t_base + 2];
    packed_patches[vb + 12] = coords[t_base + 3];
}

#[derive(Debug)]
struct GatherPatchTablesOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for GatherPatchTablesOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([c, vm, gr, pp], [ogr, opp]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(patch_start),
            ScalarIr::Float(s_base),
            ScalarIr::Float(inv),
            ScalarIr::Float(normal_den),
            ScalarIr::Float(pixel_den),
            ScalarIr::Float(sample_den),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid gather_patch_tables scalars")
        };
        let [c, vm, gr, pp] = [c, vm, gr, pp].map(|x| get_contig(handles, x));
        let client = c.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.GatherPatchTablesOp");
        let n = gr.meta.shape().dims::<2>()[0] * 9;
        let dim = CubeDim::new(&client, n);
        gather_patch_tables_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            c.into_tensor_arg(),
            vm.into_tensor_arg(),
            gr.clone().into_tensor_arg(),
            pp.clone().into_tensor_arg(),
            *patch_start as u32,
            *s_base as f32,
            *inv as f32,
            *normal_den as f32,
            *pixel_den as f32,
            *sample_den as f32,
        );
        handles.register_float_tensor::<CubeBackend>(&ogr.id, gr);
        handles.register_float_tensor::<CubeBackend>(&opp.id, pp);
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
pub fn gather_patch_tables(
    coords: burn::tensor::Tensor<2>,
    view_meta: burn::tensor::Tensor<2>,
    patch_start: usize,
    s_base: usize,
    inv: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
    patch_count: usize,
) -> [burn::tensor::Tensor<2>; 2] {
    let device = coords.device();
    let gr_buf = crate::field::empty_2d(patch_count, 40, &device);
    let pp_buf = crate::field::empty_2d(patch_count, 123, &device);
    let inputs = [coords, view_meta, gr_buf, pp_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [2usize, 3usize]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_gather_patch_tables",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(patch_start as u64),
            ScalarIr::Float(s_base as f64),
            ScalarIr::Float(inv as f64),
            ScalarIr::Float(normal_den as f64),
            ScalarIr::Float(pixel_den as f64),
            ScalarIr::Float(sample_den as f64),
        ],
    );
    let result: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            GatherPatchTablesOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

/// Generate `[slab_count * res * res, 3]` 3D coordinate slabs starting at slice `z_start`
/// directly on the GPU for Marching Cubes extraction, eliminating host-to-device uploads.
#[cube(launch)]
fn mesh_slab_points_kernel(params: &Tensor<f32>, points: &mut Tensor<f32>) {
    let i = ABSOLUTE_POS;
    if i >= points.shape(0) {
        terminate!();
    }
    let res = usize::cast_from(params[0]);
    let inv_res_m1 = params[1];
    let z_start = params[2];
    let plane = res * res;
    let slab = i / plane;
    let rem = i % plane;
    let x = rem % res;
    let y = rem / res;
    let z = z_start + f32::cast_from(slab);
    points[i * 3] = -1.0_f32 + 2.0_f32 * f32::cast_from(x) * inv_res_m1;
    points[i * 3 + 1] = -1.0_f32 + 2.0_f32 * f32::cast_from(y) * inv_res_m1;
    points[i * 3 + 2] = -1.0_f32 + 2.0_f32 * z * inv_res_m1;
}

#[derive(Debug)]
struct MeshSlabPointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for MeshSlabPointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([pr, p], [op]) = self.desc.as_fixed();
        let [pr, p] = [pr, p].map(|x| get_contig(handles, x));
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.MeshSlabPointsOp");
        let n = p.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        mesh_slab_points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            pr.into_tensor_arg(),
            p.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        Ok(())
    }
}
pub fn mesh_slab_points(
    res: usize,
    z: usize,
    device: &burn::tensor::Device,
) -> burn::tensor::Tensor<2> {
    mesh_slab_points_batch(res, z, 1, device)
}

pub fn mesh_slab_points_batch(
    res: usize,
    z_start: usize,
    slab_count: usize,
    device: &burn::tensor::Device,
) -> burn::tensor::Tensor<2> {
    use crate::field::tensor;
    let n = slab_count * res * res;
    let inv = 1.0_f32 / (res - 1) as f32;
    let params = tensor(vec![res as f32, inv, z_start as f32], [3, 1], device);
    let out_buf = burn::tensor::Tensor::<2>::empty([n, 3], device);
    let inputs = [params, out_buf].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[1].shape.clone(),
        inputs[1].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_mesh_slab_points",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            MeshSlabPointsOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

/// Single-workgroup inclusive prefix scan of `[n, 2]` `(count, ep_count)` pairs
/// fused with packing the scalar readback packet (`mode 0`: `[t0, t1]`,
/// `mode 1`: `[t0, t1, extra[0..3]]`, `mode 2`: `[t0, t1, counts[n*2]]`),
/// eliminating Burn's multi-dispatch `cumsum(0)`, `slice`, `reshape`, and `cat`.
#[cube(launch)]
fn scan_2d_pairs_and_pack_kernel(
    counts: &Tensor<f32>,
    extra: &Tensor<f32>,
    prefix: &mut Tensor<f32>,
    meta: &mut Tensor<f32>,
    n_rows: u32,
    #[comptime] mode: u32,
) {
    let unit = usize::cast_from(UNIT_POS);
    let n = usize::cast_from(n_rows);
    let chunk = (n + 255usize) / 256usize;
    let start = unit * chunk;
    let end = usize::min(start + chunk, n);
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    for i in start..end {
        s0 += counts[i * 2];
        s1 += counts[i * 2 + 1];
    }
    let mut sh0 = Shared::new_slice(256usize);
    let mut sh1 = Shared::new_slice(256usize);
    sh0[unit] = s0;
    sh1[unit] = s1;
    sync_cube();
    let mut offset = 1usize;
    for _ in 0..8usize {
        let v0 = if unit >= offset {
            sh0[unit - offset]
        } else {
            0.0_f32
        };
        let v1 = if unit >= offset {
            sh1[unit - offset]
        } else {
            0.0_f32
        };
        sync_cube();
        if unit >= offset {
            sh0[unit] += v0;
            sh1[unit] += v1;
        }
        sync_cube();
        offset = offset * 2usize;
    }
    let mut base0 = if unit == 0usize {
        0.0_f32
    } else {
        sh0[unit - 1usize]
    };
    let mut base1 = if unit == 0usize {
        0.0_f32
    } else {
        sh1[unit - 1usize]
    };
    for i in start..end {
        base0 += counts[i * 2];
        base1 += counts[i * 2 + 1];
        prefix[i * 2] = base0;
        prefix[i * 2 + 1] = base1;
    }
    if unit == 0usize {
        meta[0] = sh0[255];
        meta[1] = sh1[255];
        if mode == 1u32 {
            meta[2] = extra[0];
            meta[3] = extra[1];
            meta[4] = extra[2];
        } else if mode == 2u32 {
            meta[2] = counts[n * 2];
        }
    }
}

#[derive(Debug)]
struct Scan2dPairsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for Scan2dPairsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([c, e, p, m], [op, om]) = self.desc.as_fixed();
        let [ScalarIr::UInt(n_rows), ScalarIr::UInt(mode)] = self.desc.scalars.as_slice() else {
            panic!("invalid scan_2d_pairs scalars")
        };
        let [c, e, p, m] = [c, e, p, m].map(|x| get_contig(handles, x));
        let client = c.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.Scan2dPairsOp");
        scan_2d_pairs_and_pack_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(256),
            c.into_tensor_arg(),
            e.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            m.clone().into_tensor_arg(),
            *n_rows as u32,
            *mode as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        handles.register_float_tensor::<CubeBackend>(&om.id, m);
        Ok(())
    }
}
pub fn scan_2d_pairs_and_pack(
    counts: burn::tensor::Tensor<2>,
    extra: Option<burn::tensor::Tensor<2>>,
    n_rows: usize,
    mode: u32,
) -> [burn::tensor::Tensor<2>; 2] {
    let device = counts.device();
    let meta_len = match mode {
        1 => 5,
        2 => 3,
        _ => 2,
    };
    let extra_t = extra.unwrap_or_else(|| counts.clone());
    let prefix_buf = burn::tensor::Tensor::<2>::empty([n_rows, 2], &device);
    let meta_buf = burn::tensor::Tensor::<2>::empty([meta_len, 1], &device);
    let inputs = [counts, extra_t, prefix_buf, meta_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [2usize, 3]
        .into_iter()
        .map(|i| {
            TensorIr::uninit(
                client.create_empty_handle(),
                inputs[i].shape.clone(),
                inputs[i].dtype,
            )
        })
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_scan_2d_pairs",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![ScalarIr::UInt(n_rows as u64), ScalarIr::UInt(mode as u64)],
    );
    let result: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            Scan2dPairsOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[cube(launch)]
fn scan_1d_kernel(counts: &Tensor<f32>, prefix: &mut Tensor<f32>, n_rows: u32) {
    let unit = usize::cast_from(UNIT_POS);
    let n = usize::cast_from(n_rows);
    let chunk = (n + 255usize) / 256usize;
    let start = unit * chunk;
    let end = usize::min(start + chunk, n);
    let mut s = 0.0_f32;
    for i in start..end {
        s += counts[i];
    }
    let mut sh = Shared::new_slice(256usize);
    sh[unit] = s;
    sync_cube();
    let mut offset = 1usize;
    for _ in 0..8usize {
        let v = if unit >= offset {
            sh[unit - offset]
        } else {
            0.0_f32
        };
        sync_cube();
        if unit >= offset {
            sh[unit] += v;
        }
        sync_cube();
        offset = offset * 2usize;
    }
    let mut base = if unit == 0usize {
        0.0_f32
    } else {
        sh[unit - 1usize]
    };
    for i in start..end {
        base += counts[i];
        prefix[i] = base;
    }
}

#[derive(Debug)]
struct Scan1dOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for Scan1dOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([c, p], [op]) = self.desc.as_fixed();
        let [ScalarIr::UInt(n_rows)] = self.desc.scalars.as_slice() else {
            panic!("invalid scan_1d scalars")
        };
        let [c, p] = [c, p].map(|x| get_contig(handles, x));
        let client = c.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.Scan1dOp");
        scan_1d_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(256),
            c.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            *n_rows as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        Ok(())
    }
}
pub fn scan_1d(counts: burn::tensor::Tensor<2>) -> burn::tensor::Tensor<2> {
    let n_rows = counts.dims()[0];
    let device = counts.device();
    let prefix_buf = burn::tensor::Tensor::<2>::empty([n_rows, 1], &device);
    let inputs = [counts, prefix_buf].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[1].shape.clone(),
        inputs[1].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_scan_1d",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![ScalarIr::UInt(n_rows as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            Scan1dOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

/// Generate coarse probe grid coordinates `[(nb + 1)^3, 3]` directly on the GPU
/// for conservative block-skipping Marching Cubes. Each coarse axis index `c` maps
/// to exact fine grid index `min(c * block_size, res - 1)`.
#[cube(launch)]
fn mesh_coarse_points_kernel(
    points: &mut Tensor<f32>,
    res: u32,
    block_size: u32,
    c_res: u32,
    inv_res_m1: f32,
) {
    let i = ABSOLUTE_POS;
    if i >= points.shape(0) {
        terminate!();
    }
    let cr = usize::cast_from(c_res);
    let plane = cr * cr;
    let cz = i / plane;
    let rem = i % plane;
    let cy = rem / cr;
    let cx = rem % cr;
    let max_idx = usize::cast_from(res - 1u32);
    let bs = usize::cast_from(block_size);
    let gx = usize::min(cx * bs, max_idx);
    let gy = usize::min(cy * bs, max_idx);
    let gz = usize::min(cz * bs, max_idx);
    points[i * 3] = -1.0_f32 + 2.0_f32 * f32::cast_from(gx) * inv_res_m1;
    points[i * 3 + 1] = -1.0_f32 + 2.0_f32 * f32::cast_from(gy) * inv_res_m1;
    points[i * 3 + 2] = -1.0_f32 + 2.0_f32 * f32::cast_from(gz) * inv_res_m1;
}

#[derive(Debug)]
struct MeshCoarsePointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for MeshCoarsePointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p], [op]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(res),
            ScalarIr::UInt(block_size),
            ScalarIr::UInt(c_res),
            ScalarIr::Float(inv_res_m1),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid mesh_coarse_points scalars")
        };
        let p = get_contig(handles, p);
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.MeshCoarsePointsOp");
        let n = p.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        mesh_coarse_points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            p.clone().into_tensor_arg(),
            *res as u32,
            *block_size as u32,
            *c_res as u32,
            *inv_res_m1 as f32,
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        Ok(())
    }
}
pub fn mesh_coarse_points(
    res: usize,
    block_size: usize,
    device: &burn::tensor::Device,
) -> burn::tensor::Tensor<2> {
    let nb = (res - 1).div_ceil(block_size);
    let c_res = nb + 1;
    let total = c_res * c_res * c_res;
    let inv_res_m1 = 1.0_f32 / (res - 1) as f32;
    let p_buf = burn::tensor::Tensor::<2>::empty([total, 3], device);
    let p = p_buf.inner().try_into_primitive::<GpuBackend>().unwrap();
    let client = p.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), p.shape.clone(), p.dtype);
    let desc = CustomOpIr::with_scalars(
        "supernormal_mesh_coarse_points",
        &[p.into_ir()],
        &[out],
        vec![
            ScalarIr::UInt(res as u64),
            ScalarIr::UInt(block_size as u64),
            ScalarIr::UInt(c_res as u64),
            ScalarIr::Float(inv_res_m1 as f64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            MeshCoarsePointsOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

/// Generate `[z_count * K, 3]` 3D fine-grid points for a sparse z-band directly on
/// the GPU from a compact `[K, 2]` list of active `(x, y)` grid coordinates.
#[cube(launch)]
fn mesh_sparse_band_points_kernel(
    xy_coords: &Tensor<f32>,
    points: &mut Tensor<f32>,
    z_start: u32,
    inv_res_m1: f32,
) {
    let i = ABSOLUTE_POS;
    if i >= points.shape(0) {
        terminate!();
    }
    let k = xy_coords.shape(0);
    let lz = i / k;
    let idx = i % k;
    let x = xy_coords[idx * 2];
    let y = xy_coords[idx * 2 + 1];
    let z = f32::cast_from(usize::cast_from(z_start) + lz);
    points[i * 3] = -1.0_f32 + 2.0_f32 * x * inv_res_m1;
    points[i * 3 + 1] = -1.0_f32 + 2.0_f32 * y * inv_res_m1;
    points[i * 3 + 2] = -1.0_f32 + 2.0_f32 * z * inv_res_m1;
}

#[derive(Debug)]
struct MeshSparseBandPointsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for MeshSparseBandPointsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([xy, p], [op]) = self.desc.as_fixed();
        let [ScalarIr::UInt(z_start), ScalarIr::Float(inv_res_m1)] = self.desc.scalars.as_slice()
        else {
            panic!("invalid mesh_sparse_band_points scalars")
        };
        let [xy, p] = [xy, p].map(|x| get_contig(handles, x));
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "sampling.MeshSparseBandPointsOp");
        let n = p.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        mesh_sparse_band_points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            xy.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            *z_start as u32,
            *inv_res_m1 as f32,
        );
        handles.register_float_tensor::<CubeBackend>(&op.id, p);
        Ok(())
    }
}
pub fn mesh_sparse_band_points(
    xy_coords: burn::tensor::Tensor<2>,
    res: usize,
    z_start: usize,
    z_count: usize,
) -> burn::tensor::Tensor<2> {
    let k = xy_coords.dims()[0];
    let total = k * z_count;
    let inv_res_m1 = 1.0_f32 / (res - 1) as f32;
    let device = xy_coords.device();
    let p_buf = crate::field::empty_2d(total, 3, &device);
    let inputs = [xy_coords, p_buf].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[1].shape.clone(),
        inputs[1].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_mesh_sparse_band_points",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![
            ScalarIr::UInt(z_start as u64),
            ScalarIr::Float(inv_res_m1 as f64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            MeshSparseBandPointsOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

/// SNSR scale-aware mesh extraction helper (SMEM, arXiv:2603.20337 Sec. 3.4):
/// trilinearly sample the precomputed visibility/mask/occlusion-aware 3D `scale_grid`
/// `[c_res^3, 1]` in O(1) per vertex and emit `[N, 4]` `[x, y, z, s_local]`.
#[cube(launch)]
fn attach_mesh_scales_kernel(
    points_xyz: &Tensor<f32>,
    scale_grid: &Tensor<f32>,
    points_xyzw: &mut Tensor<f32>,
    c_res: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = points_xyz.shape(0);
    if idx >= n {
        terminate!();
    }
    let px = points_xyz[idx * 3];
    let py = points_xyz[idx * 3 + 1];
    let pz = points_xyz[idx * 3 + 2];
    let max_c = f32::cast_from(c_res - 1u32);
    let gx = f32::clamp((px + 1.0_f32) * 0.5_f32 * max_c, 0.0_f32, max_c);
    let gy = f32::clamp((py + 1.0_f32) * 0.5_f32 * max_c, 0.0_f32, max_c);
    let gz = f32::clamp((pz + 1.0_f32) * 0.5_f32 * max_c, 0.0_f32, max_c);
    let max_idx = usize::cast_from(c_res) - 2usize;
    let ix = usize::min(usize::cast_from(f32::floor(gx)), max_idx);
    let iy = usize::min(usize::cast_from(f32::floor(gy)), max_idx);
    let iz = usize::min(usize::cast_from(f32::floor(gz)), max_idx);
    let fx = f32::clamp(gx - f32::cast_from(ix), 0.0_f32, 1.0_f32);
    let fy = f32::clamp(gy - f32::cast_from(iy), 0.0_f32, 1.0_f32);
    let fz = f32::clamp(gz - f32::cast_from(iz), 0.0_f32, 1.0_f32);
    let stride_y = usize::cast_from(c_res);
    let stride_z = stride_y * stride_y;
    let base = iz * stride_z + iy * stride_y + ix;
    let c000 = scale_grid[base];
    let c100 = scale_grid[base + 1usize];
    let c010 = scale_grid[base + stride_y];
    let c110 = scale_grid[base + stride_y + 1usize];
    let c001 = scale_grid[base + stride_z];
    let c101 = scale_grid[base + stride_z + 1usize];
    let c011 = scale_grid[base + stride_z + stride_y];
    let c111 = scale_grid[base + stride_z + stride_y + 1usize];
    let c00 = c000 * (1.0_f32 - fx) + c100 * fx;
    let c10 = c010 * (1.0_f32 - fx) + c110 * fx;
    let c01 = c001 * (1.0_f32 - fx) + c101 * fx;
    let c11 = c011 * (1.0_f32 - fx) + c111 * fx;
    let c0 = c00 * (1.0_f32 - fy) + c10 * fy;
    let c1 = c01 * (1.0_f32 - fy) + c11 * fy;
    let best_s = c0 * (1.0_f32 - fz) + c1 * fz;
    points_xyzw[idx * 4] = px;
    points_xyzw[idx * 4 + 1] = py;
    points_xyzw[idx * 4 + 2] = pz;
    points_xyzw[idx * 4 + 3] = best_s;
}

#[derive(Debug)]
struct AttachMeshScalesOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for AttachMeshScalesOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([px, sg, pw], [opw]) = self.desc.as_fixed();
        let [ScalarIr::UInt(c_res)] = self.desc.scalars.as_slice() else {
            panic!("invalid attach_mesh_scales scalars")
        };
        let [px, sg, pw] = [px, sg, pw].map(|x| get_contig(handles, x));
        let client = px.client.clone();
        let n = px.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        attach_mesh_scales_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            px.into_tensor_arg(),
            sg.into_tensor_arg(),
            pw.clone().into_tensor_arg(),
            *c_res as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&opw.id, pw);
        Ok(())
    }
}

pub fn attach_mesh_scales(
    points_xyz: burn::tensor::Tensor<2>,
    scale_grid: burn::tensor::Tensor<2>,
    c_res: usize,
) -> burn::tensor::Tensor<2> {
    let n = points_xyz.dims()[0];
    let device = points_xyz.device();
    let pw_buf = crate::field::empty_2d(n, 4, &device);
    let inputs = [points_xyz, scale_grid, pw_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[2].shape.clone(),
        inputs[2].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_attach_mesh_scales",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![ScalarIr::UInt(c_res as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            AttachMeshScalesOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

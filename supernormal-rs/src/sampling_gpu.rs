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
    let b = ray * 12;
    let near = rays[b + 6];
    let far = rays[b + 7];
    let step = rays[b + 8];
    let offset = usize::cast_from(rays[b + 9]);
    let capacity = usize::cast_from(rays[b + 10]);
    let r = usize::cast_from(rays[b + 11]);
    let half_r = f32::cast_from(r) * 0.5_f32;
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
        let mut cell = 0usize;
        let mut inside = true;
        let mut dt_voxel = 3.402823466e38_f32;
        for axis in 0..3usize {
            let d = rays[b + 3 + axis];
            let pt = rays[b + axis] + d * mid;
            if pt < -1.0_f32 || pt > 1.0_f32 {
                inside = false;
            }
            let grid = (pt + 1.0_f32) * half_r;
            let c = usize::cast_from(f32::clamp(grid, 0.0_f32, f32::cast_from(r - 1)));
            cell = cell * r + c;
            let plane = if d >= 0.0_f32 {
                f32::floor(grid) + 1.0_f32
            } else {
                f32::ceil(grid) - 1.0_f32
            };
            let safe_d = if f32::abs(d) < 1e-10_f32 {
                1e-10_f32
            } else {
                d
            };
            let dt = (plane - grid) * (1.0_f32 / safe_d) / half_r;
            dt_voxel = f32::min(dt_voxel, dt);
        }
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
        } else {
            let skip = if inside {
                usize::cast_from(f32::clamp(
                    f32::ceil(dt_voxel / step),
                    1.0_f32,
                    f32::cast_from(capacity - s),
                ))
            } else {
                1usize
            };
            // Clamp before addition to prevent overflow for nearly parallel rays.
            s += usize::min(skip, capacity - s);
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
    let offset = usize::cast_from(rays[ray * 12 + 9]);
    for i in 0..end - begin {
        out[(begin + i) * 3] = f32::cast_from(ray);
        out[(begin + i) * 3 + 1] = samples[(offset + i) * 2];
        out[(begin + i) * 3 + 2] = samples[(offset + i) * 2 + 1];
    }
}

/// Fused ray-level compaction + deduplicated center-ray endpoint materialization:
/// writes `marched` `[total_marched, 3]`, `unique_pts` `[unique_ep_count, 3]`,
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
    let offset = usize::cast_from(rays[ray * 12 + 9]);
    let rb = ray * 12;
    let mut prev_end_bits = 0u32;
    for i in 0..s_end - s_begin {
        let t_start = samples[(offset + i) * 2];
        let t_end = samples[(offset + i) * 2 + 1];
        let m_idx = s_begin + i;
        marched[m_idx * 3] = f32::cast_from(ray);
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
            for d in 0..3usize {
                unique_pts[start_id * 3 + d] = rays[rb + d] + rays[rb + 3 + d] * t_start;
            }
            ep_cursor += 2;
        }
        for d in 0..3usize {
            unique_pts[end_id * 3 + d] = rays[rb + d] + rays[rb + 3 + d] * t_end;
        }
        pairs[m_idx * 2] = f32::cast_from(start_id);
        pairs[m_idx * 2 + 1] = f32::cast_from(end_id);
    }
}

/// Initialize reverse occurrence slots `[n_entries, 2]` to `-1.0`.
#[cube(launch)]
fn init_reverse_kernel(reverse: &mut Tensor<f32>, valid_entries: u32) {
    let i = ABSOLUTE_POS;
    if i >= usize::cast_from(valid_entries) {
        terminate!();
    }
    reverse[i] = -1.0_f32;
}

/// Build `endpoints` `[ep_count, 2]`, `descriptors` `[vis_count, 5]`,
/// compact `mapping` `[vis_count, 1]` (storing `start_ep * 9`, since `end_ep`
/// is always `start_ep + 1` so the 18 sample points are always contiguous at
/// `start_ep * 9 + 0..18`), and compact endpoint-level `reverse` `[ep_count, 2]`
/// directly on the GPU without the 9-pixel inner loop.
#[cube(launch)]
fn build_visible_geometry_tables_kernel(
    marched: &Tensor<f32>,
    march_prefix: &Tensor<f32>,
    vis_prefix: &Tensor<f32>,
    endpoints: &mut Tensor<f32>,
    descriptors: &mut Tensor<f32>,
    mapping: &mut Tensor<f32>,
    reverse: &mut Tensor<f32>,
    patch_start: u32,
    s_base: u32,
    ep_base: u32,
) {
    let local_ray = ABSOLUTE_POS;
    let ray = usize::cast_from(patch_start) + local_ray;
    if ray >= vis_prefix.shape(0) || local_ray >= vis_prefix.shape(0) - usize::cast_from(patch_start)
    {
        terminate!();
    }
    let m_begin = if ray == 0 {
        0usize
    } else {
        usize::cast_from(march_prefix[(ray - 1) * 2])
    };
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
        }
        reverse[end_ep * 2] = f32::cast_from(vis_idx * 2 + 1);
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
    let u0 = hash_to_unit_f32(pcg_hash_u32(cell ^ seed));
    let u1 = hash_to_unit_f32(pcg_hash_u32(cell ^ (seed + 0x9E3779B9u32)));
    let u2 = hash_to_unit_f32(pcg_hash_u32(cell ^ (seed + 0x3C6EF372u32)));
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
            let u0 = hash_to_unit_f32(pcg_hash_u32(cu ^ seed));
            let u1 = hash_to_unit_f32(pcg_hash_u32(cu ^ (seed + 0x9E3779B9u32)));
            let u2 = hash_to_unit_f32(pcg_hash_u32(cu ^ (seed + 0x3C6EF372u32)));
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
        let [r, f, s, c] = [r, f, s, c].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = r.client.clone();
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        march_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
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
        let [r, s, p, o] = [r, s, p, o].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = r.client.clone();
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
            [r, s, p, m, u, pr, rg].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = r.client.clone();
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        compact_and_center_endpoints_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            r.into_tensor_arg(),
            s.into_tensor_arg(),
            p.into_tensor_arg(),
            m.clone().into_tensor_arg(),
            u.clone().into_tensor_arg(),
            pr.clone().into_tensor_arg(),
            rg.clone().into_tensor_arg(),
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
) -> [burn::tensor::Tensor<2>; 4] {
    let n_rays = rays.dims()[0];
    let device = rays.device();
    let m_cap = bucket_capacity(total_marched);
    let m_buf = burn::tensor::Tensor::<2>::empty([m_cap, 3], &device);
    let u_buf = burn::tensor::Tensor::<2>::empty([padded_ep_cap, 3], &device);
    let pr_buf = burn::tensor::Tensor::<2>::empty([m_cap, 2], &device);
    let rg_buf = burn::tensor::Tensor::<2>::empty([n_rays, 2], &device);
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
            ScalarIr::UInt(ep_count),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid visible geometry table scalars")
        };
        let [m, mp, vp, e, d, map, r] =
            [m, mp, vp, e, d, map, r].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = m.client.clone();
        let n_rev = ((*ep_count as usize) * 2).max(1);
        let dim_rev = CubeDim::new(&client, n_rev);
        init_reverse_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n_rev, dim_rev),
            dim_rev,
            r.clone().into_tensor_arg(),
            n_rev as u32,
        );
        let n_rays = (*patch_count as usize).max(1);
        let dim = CubeDim::new(&client, n_rays);
        build_visible_geometry_tables_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n_rays, dim),
            dim,
            m.into_tensor_arg(),
            mp.into_tensor_arg(),
            vp.into_tensor_arg(),
            e.clone().into_tensor_arg(),
            d.clone().into_tensor_arg(),
            map.clone().into_tensor_arg(),
            r.clone().into_tensor_arg(),
            *patch_start as u32,
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
    let e_buf = burn::tensor::Tensor::<2>::empty([ep_cap, 2], &device);
    let d_buf = burn::tensor::Tensor::<2>::empty([vis_cap, 5], &device);
    let map_buf = burn::tensor::Tensor::<2>::empty([vis_cap, 1], &device);
    let r_buf = burn::tensor::Tensor::<2>::empty([ep_cap, 2], &device);
    let inputs = [marched, march_prefix, vis_prefix, e_buf, d_buf, map_buf, r_buf]
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
        let p = handles.get_float_tensor::<CubeBackend>(p);
        let client = p.client.clone();
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
        let [s, v] = [s, v].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let v = if v.can_mut() { v } else { v.copy() };
        let client = v.client.clone();
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
        let [f, c] = [f, c].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = f.client.clone();
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
        let [f, pr, id, pt] = [f, pr, id, pt].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = f.client.clone();
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
        let [id, s, v] = [id, s, v].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let v = if v.can_mut() { v } else { v.copy() };
        let client = v.client.clone();
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
    valid_rows: u32,
    #[comptime] tile_cols: usize,
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
                    h = dh[sample * dh.stride(0) + (row_base + channel) * dh.stride(1)];
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
        sync_cube();
    }
    partials[part * 2048 + (row_base + row) * 32 + col_base + col] = sum;
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
        let ([h, f, p], [out]) = self.desc.as_fixed();
        let [h, f, p] = [h, f, p].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let [ScalarIr::UInt(wide), ScalarIr::UInt(valid_rows)] = self.desc.scalars.as_slice()
        else {
            panic!("invalid parameter-gradient arguments")
        };
        let tile_cols = if *wide != 0 { 32usize } else { 8usize };
        let client = h.client.clone();
        let parts = (*valid_rows as usize).div_ceil(256).max(1);
        parameter_kernel::launch(
            &client,
            CubeCount::Static((2048 / (8 * tile_cols)) as u32, parts as u32, 1),
            CubeDim::new_1d((8 * tile_cols) as u32),
            h.into_tensor_arg(),
            f.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            *valid_rows as u32,
            tile_cols,
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, p);
        Ok(())
    }
}
pub fn parameter_gradients(
    dh: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    valid_rows: usize,
    wide: bool,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    use burn::tensor::Tensor;
    let parts = valid_rows.div_ceil(256).max(1);
    assert!(parts <= 65535, "parameter-gradient query too large");
    // Use uninitialized output allocation instead of zero-filling before full overwrite.
    let buffer = Tensor::empty([parts, 2048], &dh.device());
    let inputs =
        [dh, features, buffer].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
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
) {
    let id = ABSOLUTE_POS;
    if id >= usize::cast_from(valid_points) {
        terminate!();
    }
    let endpoint = id / 9;
    let pixel = id % 9;
    let patch = usize::cast_from(endpoints[endpoint * 2]);
    let base = patch * 39;
    let t = endpoints[endpoint * 2 + 1] * rays[base + 30 + pixel];
    for d in 0..3usize {
        points[id * 3 + d] = rays[base + d] + rays[base + 3 + pixel * 3 + d] * t;
    }
}

#[cube]
fn point_distance(points: &Tensor<f32>, a: usize, b: usize) -> f32 {
    let x = points[a * 3] - points[b * 3];
    let y = points[a * 3 + 1] - points[b * 3 + 1];
    let z = points[a * 3 + 2] - points[b * 3 + 2];
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
    let dt = (samples[sample * 5 + 2] - samples[sample * 5 + 1]) * rays[patch * 39 + 30 + pixel];
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
        let [r, e, s, p, g] = [r, e, s, p, g].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = r.client.clone();
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
    params: &Tensor<f32>,
    geom_rays: &mut Tensor<f32>,
    packed_patches: &mut Tensor<f32>,
) {
    let id = ABSOLUTE_POS;
    let n_patches = geom_rays.shape(0);
    if id >= n_patches * 9 {
        terminate!();
    }
    let p = id / 9;
    let i = id % 9;
    let patch_start = usize::cast_from(params[0]);
    let s_base = params[1];
    let inv = params[2];
    let normal_den = params[3];
    let pixel_den = params[4];
    let sample_den = params[5];

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

    let gr = p * 39;
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
        let ([c, vm, pr, gr, pp], [ogr, opp]) = self.desc.as_fixed();
        let [c, vm, pr, gr, pp] =
            [c, vm, pr, gr, pp].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = c.client.clone();
        let n = gr.meta.shape().dims::<2>()[0] * 9;
        let dim = CubeDim::new(&client, n);
        gather_patch_tables_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            c.into_tensor_arg(),
            vm.into_tensor_arg(),
            pr.into_tensor_arg(),
            gr.clone().into_tensor_arg(),
            pp.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&ogr.id, gr);
        handles.register_float_tensor::<CubeBackend>(&opp.id, pp);
        Ok(())
    }
}
pub fn gather_patch_tables(
    coords: burn::tensor::Tensor<2>,
    view_meta: burn::tensor::Tensor<2>,
    params: burn::tensor::Tensor<2>,
    patch_count: usize,
) -> [burn::tensor::Tensor<2>; 2] {
    let device = coords.device();
    let gr_buf = burn::tensor::Tensor::<2>::empty([patch_count, 39], &device);
    let pp_buf = burn::tensor::Tensor::<2>::empty([patch_count, 123], &device);
    let inputs = [coords, view_meta, params, gr_buf, pp_buf]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
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
        "supernormal_gather_patch_tables",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
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

/// Generate a `[res * res, 3]` 3D coordinate slab at slice `z` directly on the GPU
/// for Marching Cubes extraction, eliminating 1,024 host-to-device 12 MB point uploads.
#[cube(launch)]
fn mesh_slab_points_kernel(params: &Tensor<f32>, points: &mut Tensor<f32>) {
    let i = ABSOLUTE_POS;
    if i >= points.shape(0) {
        terminate!();
    }
    let res = usize::cast_from(params[0]);
    let inv_res_m1 = params[1];
    let z_coord = params[2];
    let x = i % res;
    let y = i / res;
    points[i * 3] = -1.0_f32 + 2.0_f32 * f32::cast_from(x) * inv_res_m1;
    points[i * 3 + 1] = -1.0_f32 + 2.0_f32 * f32::cast_from(y) * inv_res_m1;
    points[i * 3 + 2] = z_coord;
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
        let [pr, p] = [pr, p].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = p.client.clone();
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
    use crate::field::tensor;
    let n = res * res;
    let inv = 1.0_f32 / (res - 1) as f32;
    let z_coord = -1.0_f32 + 2.0_f32 * (z as f32) * inv;
    let params = tensor(vec![res as f32, inv, z_coord], [3, 1], device);
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

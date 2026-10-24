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

/// Initialize reverse occurrence slots `[n_points, 2]` to `-1.0`.
#[cube(launch)]
fn init_reverse_kernel(reverse: &mut Tensor<f32>) {
    let i = ABSOLUTE_POS;
    if i >= reverse.shape(0) * 2 {
        terminate!();
    }
    reverse[i] = -1.0_f32;
}

/// Build `endpoints` `[ep_count, 2]`, `descriptors` `[vis_count, 5]`,
/// `mapping` `[vis_count, 18]`, and `reverse` `[ep_count * 9, 2]` in parallel
/// per ray directly from `marched` + `march_prefix` + `vis_prefix` without
/// atomics or intermediate sample compaction.
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
    // Check bounds via descriptor count indirectly: only launch `end - start` threads.
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

        for j in 0..9usize {
            let start_pt = start_ep * 9 + j;
            let end_pt = end_ep * 9 + j;
            mapping[vis_idx * 18 + j] = f32::cast_from(start_pt);
            mapping[vis_idx * 18 + 9 + j] = f32::cast_from(end_pt);
            if !reuse_prev {
                reverse[start_pt * 2] = f32::cast_from(vis_idx * 18 + j);
            }
            reverse[end_pt * 2] = f32::cast_from(vis_idx * 18 + 9 + j);
            if reuse_next {
                reverse[end_pt * 2 + 1] = f32::cast_from((vis_idx + 1) * 18 + j);
            }
        }
    }
}

#[cube(launch)]
fn occupancy_kernel(
    sdf: &Tensor<f32>,
    heads: &Tensor<f32>,
    next: &Tensor<f32>,
    old: &Tensor<f32>,
    out: &mut Tensor<f32>,
) {
    let cell = ABSOLUTE_POS;
    if cell >= old.shape(0) {
        terminate!();
    }
    let mut value = old[cell];
    let mut link = heads[cell];
    // Each cell owns an ordered list: duplicate random cell IDs apply decay
    // repeatedly in the same order as the reference, without racing scatters.
    while link >= 0.0_f32 {
        let index = usize::cast_from(link);
        let density = 1.0_f32 / (1.0_f32 + f32::exp(80.0_f32 * sdf[index]));
        value = f32::max(value * 0.95_f32, density);
        link = next[index];
    }
    out[cell] = value;
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
    unique_ep_count: usize,
) -> [burn::tensor::Tensor<2>; 4] {
    let n_rays = rays.dims()[0];
    let device = rays.device();
    let m_buf = burn::tensor::Tensor::<2>::empty([total_marched, 3], &device);
    let u_buf = burn::tensor::Tensor::<2>::empty([unique_ep_count, 3], &device);
    let pr_buf = burn::tensor::Tensor::<2>::empty([total_marched, 2], &device);
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
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid visible geometry table scalars")
        };
        let [m, mp, vp, e, d, map, r] =
            [m, mp, vp, e, d, map, r].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = m.client.clone();
        let n_rev = r.meta.shape().dims::<2>()[0] * 2;
        let dim_rev = CubeDim::new(&client, n_rev);
        init_reverse_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n_rev, dim_rev),
            dim_rev,
            r.clone().into_tensor_arg(),
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
    let e_buf = burn::tensor::Tensor::<2>::empty([ep_count, 2], &device);
    let d_buf = burn::tensor::Tensor::<2>::empty([vis_count, 5], &device);
    let map_buf = burn::tensor::Tensor::<2>::empty([vis_count, 18], &device);
    let r_buf = burn::tensor::Tensor::<2>::empty([ep_count * 9, 2], &device);
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
struct OccupancyOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for OccupancyOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([s, h, n, v, o], [o0]) = self.desc.as_fixed();
        let [s, h, n, v, o] = [s, h, n, v, o].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = s.client.clone();
        let cell_count = v.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, cell_count);
        occupancy_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, cell_count, dim),
            dim,
            s.into_tensor_arg(),
            h.into_tensor_arg(),
            n.into_tensor_arg(),
            v.into_tensor_arg(),
            o.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&o0.id, o);
        Ok(())
    }
}
pub fn occupancy(inputs: [burn::tensor::Tensor<2>; 5]) -> [burn::tensor::Tensor<2>; 1] {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [4usize]
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
        "supernormal_occupancy",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
    );
    let result: [_; 1] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            OccupancyOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

// Default: an 8x32 tile reuses each staged hidden derivative across all
// features and bias, instead of loading it independently for four 8x8 tiles.
// Keep the original 8x8 layout available for independent A/B comparisons.
#[cube(launch)]
fn parameter_kernel(
    dh: &Tensor<f32>,
    features: &Tensor<f32>,
    partials: &mut Tensor<f32>,
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
                if sample < dh.shape(0) {
                    h = dh[sample * dh.stride(0) + (row_base + channel) * dh.stride(1)];
                }
                hs[index] = h;
            }
            let sample = sample_base + index / tile_cols;
            let feature = col_base + index % tile_cols;
            let mut x = 0.0_f32;
            if sample < dh.shape(0) {
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
        let [ScalarIr::UInt(wide)] = self.desc.scalars.as_slice() else {
            panic!("invalid parameter-gradient arguments")
        };
        let tile_cols = if *wide != 0 { 32usize } else { 8usize };
        let client = h.client.clone();
        let parts = h.meta.shape().dims::<2>()[0].div_ceil(256);
        parameter_kernel::launch(
            &client,
            CubeCount::Static((2048 / (8 * tile_cols)) as u32, parts as u32, 1),
            CubeDim::new_1d((8 * tile_cols) as u32),
            h.into_tensor_arg(),
            f.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            tile_cols,
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, p);
        Ok(())
    }
}
pub fn parameter_gradients(
    dh: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    wide: bool,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    use burn::tensor::Tensor;
    let parts = dh.dims()[0].div_ceil(256);
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
        vec![ScalarIr::UInt(wide as u64)],
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
fn points_kernel(rays: &Tensor<f32>, endpoints: &Tensor<f32>, points: &mut Tensor<f32>) {
    let id = ABSOLUTE_POS;
    if id >= points.shape(0) {
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
) {
    let id = ABSOLUTE_POS;
    if id >= samples.shape(0) * 9 {
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
        let [r, e, s, p, g] = [r, e, s, p, g].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = r.client.clone();
        let n = p.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        points_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            r.clone().into_tensor_arg(),
            e.into_tensor_arg(),
            p.clone().into_tensor_arg(),
        );
        let n = s.meta.shape().dims::<2>()[0] * 9;
        let dim = CubeDim::new(&client, n);
        geometry_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            r.into_tensor_arg(),
            s.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            g.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&po.id, p);
        handles.register_float_tensor::<CubeBackend>(&go.id, g);
        Ok(())
    }
}

pub fn geometry(inputs: [burn::tensor::Tensor<2>; 5]) -> [burn::tensor::Tensor<2>; 2] {
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
        vec![],
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

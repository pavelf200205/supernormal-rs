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
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr,
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
    counts[ray] = f32::cast_from(count);
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
    let begin = if ray == 0 {
        0usize
    } else {
        usize::cast_from(prefix[ray - 1])
    };
    let end = usize::cast_from(prefix[ray]);
    let offset = usize::cast_from(rays[ray * 12 + 9]);
    for i in 0..end - begin {
        out[(begin + i) * 3] = f32::cast_from(ray);
        out[(begin + i) * 3 + 1] = samples[(offset + i) * 2];
        out[(begin + i) * 3 + 2] = samples[(offset + i) * 2 + 1];
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

// 8x8 parameter tile, 32 samples staged at a time. The last feature column
// is the bias, so one pass computes both dW and db. Partials avoid atomics.
#[cube(launch)]
fn parameter_kernel(dh: &Tensor<f32>, features: &Tensor<f32>, partials: &mut Tensor<f32>) {
    let unit = usize::cast_from(UNIT_POS);
    let tile = usize::cast_from(CUBE_POS_X);
    let part = usize::cast_from(CUBE_POS_Y);
    let row = unit / 8;
    let col = unit % 8;
    let row_base = (tile / 4) * 8;
    let col_base = (tile % 4) * 8;
    let mut hs = Shared::new_slice(256usize);
    let mut xs = Shared::new_slice(256usize);
    let mut sum = 0.0_f32;
    for block in 0..8usize {
        let sample_base = part * 256 + block * 32;
        for load in 0..4usize {
            let index = unit + load * 64;
            let sample = sample_base + index / 8;
            let channel = index % 8;
            let mut h = 0.0_f32;
            let mut x = 0.0_f32;
            if sample < dh.shape(0) {
                h = dh[sample * dh.stride(0) + (row_base + channel) * dh.stride(1)];
                let feature = col_base + channel;
                if feature == 31 {
                    x = 1.0_f32;
                } else {
                    x = features[sample * features.stride(0) + feature * features.stride(1)];
                }
            }
            hs[index] = h;
            xs[index] = x;
        }
        sync_cube();
        for j in 0..32usize {
            sum += hs[j * 8 + row] * xs[j * 8 + col];
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
        let client = h.client.clone();
        let parts = h.meta.shape().dims::<2>()[0].div_ceil(256);
        parameter_kernel::launch(
            &client,
            CubeCount::Static(32, parts as u32, 1),
            CubeDim::new_1d(64),
            h.into_tensor_arg(),
            f.into_tensor_arg(),
            p.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, p);
        Ok(())
    }
}
pub fn parameter_gradients(
    dh: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    use burn::tensor::Tensor;
    let parts = dh.dims()[0].div_ceil(256);
    assert!(parts <= 65535, "parameter-gradient query too large");
    let buffer = Tensor::zeros([parts, 2048], &dh.device());
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
        vec![],
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
    let mut dx = 0.0_f32;
    if col == 1 {
        dx = point_distance(points, base + l, base + pixel)
            + point_distance(points, base + pixel, base + r);
    } else {
        dx = point_distance(points, base + l, base + r);
    }
    let mut dy = 0.0_f32;
    if row == 1 {
        dy = point_distance(points, base + u, base + pixel)
            + point_distance(points, base + pixel, base + d);
    } else {
        dy = point_distance(points, base + u, base + d);
    }
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

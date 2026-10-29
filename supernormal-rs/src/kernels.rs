use cubecl::prelude::*;

#[cube(launch)]
pub fn hash_grid_packed_forward_kernel(
    points: &Tensor<f32>,
    tables: &Tensor<f32>,
    metadata: &Tensor<f32>,
    features: &mut Tensor<f32>,
    active: u32,
) {
    let point_idx = ABSOLUTE_POS;
    if point_idx >= points.shape(0) {
        terminate!();
    }
    let p0 = points[point_idx * 3];
    let p1 = points[point_idx * 3 + 1];
    let p2 = points[point_idx * 3 + 2];
    let row = point_idx * 31;
    features[row] = p0;
    features[row + 1] = p1;
    features[row + 2] = p2;

    let norm0 = f32::clamp((p0 + 1.) * 0.5, 0., 1. - 1e-7);
    let norm1 = f32::clamp((p1 + 1.) * 0.5, 0., 1. - 1e-7);
    let norm2 = f32::clamp((p2 + 1.) * 0.5, 0., 1. - 1e-7);
    for level in 0..active {
        let res = u32::cast_from(metadata[usize::cast_from(level) * 2]);
        let offset = u32::cast_from(metadata[usize::cast_from(level) * 2 + 1]);
        let c0 = norm0 * f32::cast_from(res);
        let c1 = norm1 * f32::cast_from(res);
        let c2 = norm2 * f32::cast_from(res);
        let b0 = f32::floor(c0);
        let b1 = f32::floor(c1);
        let b2 = f32::floor(c2);
        let w0 = c0 - b0;
        let w1 = c1 - b1;
        let w2 = c2 - b2;
        let base0 = u32::cast_from(b0);
        let base1 = u32::cast_from(b1);
        let base2 = u32::cast_from(b2);
        let r1 = res + 1;
        let dense = r1 * r1 * r1 <= 524288;
        let mut feat0 = 0.0_f32;
        let mut feat1 = 0.0_f32;
        for corner in 0..8u32 {
            let bit0 = corner & 1;
            let bit1 = (corner >> 1) & 1;
            let bit2 = (corner >> 2) & 1;
            let v0 = base0 + bit0;
            let v1 = base1 + bit1;
            let v2 = base2 + bit2;
            let kw0 = if bit0 == 0 { 1. - w0 } else { w0 };
            let kw1 = if bit1 == 0 { 1. - w1 } else { w1 };
            let kw2 = if bit2 == 0 { 1. - w2 } else { w2 };
            let weight = kw0 * kw1 * kw2;
            let idx = if dense {
                v0 + v1 * r1 + v2 * r1 * r1
            } else {
                (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
            };
            let entry = usize::cast_from(offset + idx) * 2;
            feat0 += weight * tables[entry];
            feat1 += weight * tables[entry + 1];
        }
        let col = row + 3 + usize::cast_from(level) * 2;
        features[col] = feat0;
        features[col + 1] = feat1;
    }
}

#[cube(launch)]
pub fn hash_grid_level_forward_kernel<F: Float>(
    points: &Tensor<F>,       // [N, 3]
    table: &Tensor<F>,        // [entries, 2]
    features: &mut Tensor<F>, // [N, 31]
    res: u32,
    level: u32,
    #[comptime] dense: bool,
    #[comptime] copy_xyz: bool,
) {
    let point_idx = ABSOLUTE_POS;
    let n = points.shape(0);
    if point_idx >= n {
        terminate!();
    }

    let p0 = points[point_idx * 3];
    let p1 = points[point_idx * 3 + 1];
    let p2 = points[point_idx * 3 + 2];

    if copy_xyz {
        features[point_idx * 31] = p0;
        features[point_idx * 31 + 1] = p1;
        features[point_idx * 31 + 2] = p2;
    }

    let res_f = F::cast_from(res);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let half = F::new(0.5_f32);
    let norm_max = one - F::new(1e-7_f32);

    let norm0 = F::clamp((p0 + one) * half, zero, norm_max);
    let norm1 = F::clamp((p1 + one) * half, zero, norm_max);
    let norm2 = F::clamp((p2 + one) * half, zero, norm_max);

    let c0 = norm0 * res_f;
    let c1 = norm1 * res_f;
    let c2 = norm2 * res_f;

    let b0 = F::floor(c0);
    let b1 = F::floor(c1);
    let b2 = F::floor(c2);

    let w0 = c0 - b0;
    let w1 = c1 - b1;
    let w2 = c2 - b2;

    let base0 = u32::cast_from(b0);
    let base1 = u32::cast_from(b1);
    let base2 = u32::cast_from(b2);

    let mut feat0 = zero;
    let mut feat1 = zero;

    let res_plus_1 = res + 1;
    let res_plus_1_sq = res_plus_1 * res_plus_1;

    for corner in 0..8u32 {
        let bit0 = corner & 1;
        let bit1 = (corner >> 1) & 1;
        let bit2 = (corner >> 2) & 1;

        let v0 = base0 + bit0;
        let v1 = base1 + bit1;
        let v2 = base2 + bit2;

        let kw0 = if bit0 == 0 { one - w0 } else { w0 };
        let kw1 = if bit1 == 0 { one - w1 } else { w1 };
        let kw2 = if bit2 == 0 { one - w2 } else { w2 };
        let weight = kw0 * kw1 * kw2;

        let idx_u32 = if dense {
            v0 + v1 * res_plus_1 + v2 * res_plus_1_sq
        } else {
            (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
        };

        let idx = usize::cast_from(idx_u32);
        let t0 = table[idx * 2];
        let t1 = table[idx * 2 + 1];

        feat0 = feat0 + weight * t0;
        feat1 = feat1 + weight * t1;
    }

    let out_col = 3 + usize::cast_from(level) * 2;
    features[point_idx * 31 + out_col] = feat0;
    features[point_idx * 31 + out_col + 1] = feat1;
}

#[cube]
pub fn atomic_add_f32(table: &Tensor<Atomic<u32>>, idx: usize, val: f32) {
    let mut cur_u32 = table[idx].load();
    let mut done = false;
    while !done {
        let cur_f32 = f32::reinterpret(cur_u32);
        let new_f32 = cur_f32 + val;
        let new_u32 = u32::reinterpret(new_f32);
        let old_u32 = table[idx].compare_exchange_weak(cur_u32, new_u32);
        if old_u32 == cur_u32 {
            done = true;
        } else {
            cur_u32 = old_u32;
        }
    }
}

#[cube(launch)]
pub fn hash_grid_level_backward_kernel(
    points: &Tensor<f32>,
    d_features: &Tensor<f32>,
    table_grad: &Tensor<Atomic<u32>>,
    res: u32,
    level: u32,
    #[comptime] dense: bool,
) {
    let point_idx = ABSOLUTE_POS;
    let n = points.shape(0);
    if point_idx >= n {
        terminate!();
    }

    let p0 = points[point_idx * 3];
    let p1 = points[point_idx * 3 + 1];
    let p2 = points[point_idx * 3 + 2];

    let res_f = f32::cast_from(res);
    let zero = 0.0_f32;
    let one = 1.0_f32;
    let half = 0.5_f32;
    let norm_max = one - 1e-7_f32;

    let norm0 = f32::clamp((p0 + one) * half, zero, norm_max);
    let norm1 = f32::clamp((p1 + one) * half, zero, norm_max);
    let norm2 = f32::clamp((p2 + one) * half, zero, norm_max);

    let c0 = norm0 * res_f;
    let c1 = norm1 * res_f;
    let c2 = norm2 * res_f;

    let b0 = f32::floor(c0);
    let b1 = f32::floor(c1);
    let b2 = f32::floor(c2);

    let w0 = c0 - b0;
    let w1 = c1 - b1;
    let w2 = c2 - b2;

    let base0 = u32::cast_from(b0);
    let base1 = u32::cast_from(b1);
    let base2 = u32::cast_from(b2);

    let res_plus_1 = res + 1;
    let res_plus_1_sq = res_plus_1 * res_plus_1;

    let in_col = 3 + usize::cast_from(level) * 2;
    let g0 = d_features[point_idx * 31 + in_col];
    let g1 = d_features[point_idx * 31 + in_col + 1];

    for corner in 0..8u32 {
        let bit0 = corner & 1;
        let bit1 = (corner >> 1) & 1;
        let bit2 = (corner >> 2) & 1;

        let v0 = base0 + bit0;
        let v1 = base1 + bit1;
        let v2 = base2 + bit2;

        let kw0 = if bit0 == 0 { one - w0 } else { w0 };
        let kw1 = if bit1 == 0 { one - w1 } else { w1 };
        let kw2 = if bit2 == 0 { one - w2 } else { w2 };
        let weight = kw0 * kw1 * kw2;

        let idx_u32 = if dense {
            v0 + v1 * res_plus_1 + v2 * res_plus_1_sq
        } else {
            (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
        };

        let idx = usize::cast_from(idx_u32);

        let g0_w = weight * g0;
        let g1_w = weight * g1;
        if g0_w != zero {
            atomic_add_f32(table_grad, idx * 2, g0_w);
        }
        if g1_w != zero {
            atomic_add_f32(table_grad, idx * 2 + 1, g1_w);
        }
    }
}

#[cube(launch)]
pub fn hash_grid_packed_backward_kernel(
    points: &Tensor<f32>,
    d_features: &Tensor<f32>,
    metadata: &Tensor<f32>,
    table_grad: &Tensor<Atomic<u32>>,
    active: u32,
    valid_n: u32,
) {
    let point_idx = ABSOLUTE_POS;
    if point_idx >= usize::cast_from(valid_n) {
        terminate!();
    }

    let p0 = points[point_idx * 3];
    let p1 = points[point_idx * 3 + 1];
    let p2 = points[point_idx * 3 + 2];
    let norm0 = f32::clamp((p0 + 1.) * 0.5, 0., 1. - 1e-7);
    let norm1 = f32::clamp((p1 + 1.) * 0.5, 0., 1. - 1e-7);
    let norm2 = f32::clamp((p2 + 1.) * 0.5, 0., 1. - 1e-7);

    for level in 0..active {
        let col = point_idx * 31 + 3 + usize::cast_from(level) * 2;
        let g0 = d_features[col];
        let g1 = d_features[col + 1];

        if g0 != 0.0 || g1 != 0.0 {
            let res = u32::cast_from(metadata[usize::cast_from(level) * 2]);
            let offset = u32::cast_from(metadata[usize::cast_from(level) * 2 + 1]);
            let c0 = norm0 * f32::cast_from(res);
            let c1 = norm1 * f32::cast_from(res);
            let c2 = norm2 * f32::cast_from(res);
            let b0 = f32::floor(c0);
            let b1 = f32::floor(c1);
            let b2 = f32::floor(c2);
            let w0 = c0 - b0;
            let w1 = c1 - b1;
            let w2 = c2 - b2;
            let base0 = u32::cast_from(b0);
            let base1 = u32::cast_from(b1);
            let base2 = u32::cast_from(b2);
            let r1 = res + 1;
            let dense = r1 * r1 * r1 <= 524288;

            for corner in 0..8u32 {
                let bit0 = corner & 1;
                let bit1 = (corner >> 1) & 1;
                let bit2 = (corner >> 2) & 1;
                let v0 = base0 + bit0;
                let v1 = base1 + bit1;
                let v2 = base2 + bit2;
                let kw0 = if bit0 == 0 { 1. - w0 } else { w0 };
                let kw1 = if bit1 == 0 { 1. - w1 } else { w1 };
                let kw2 = if bit2 == 0 { 1. - w2 } else { w2 };
                let weight = kw0 * kw1 * kw2;
                let idx = if dense {
                    v0 + v1 * r1 + v2 * r1 * r1
                } else {
                    (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                };
                let entry = usize::cast_from(offset + idx) * 2;
                let wg0 = weight * g0;
                let wg1 = weight * g1;
                if wg0 != 0. {
                    atomic_add_f32(table_grad, entry, wg0);
                }
                if wg1 != 0. {
                    atomic_add_f32(table_grad, entry + 1, wg1);
                }
            }
        }
    }
}

use burn_backend::ExecutionError;
use burn_cubecl::CubeBackend;
use burn_cubecl::fusion::{CubeFusionHandle, FusionCubeRuntime};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId,
        TensorIr,
    },
};

type GpuBackend = Fusion<CubeBackend>;

/// One pass over active embedding entries. Inactive levels and their Adam
/// moments are left untouched; bias corrections are independent per level.
#[cube(launch)]
fn packed_adam_kernel(
    value: &mut Tensor<f32>,
    first: &mut Tensor<f32>,
    second: &mut Tensor<f32>,
    gradient: &mut Tensor<f32>,
    corrections: &Tensor<f32>,
    active_elems: u32,
    #[comptime] level_dispatch: bool,
) {
    let mut index = ABSOLUTE_POS;
    let mut level = 0usize;
    if level_dispatch {
        level = usize::cast_from(CUBE_POS_Y);
        let mut start = 0usize;
        if level > 0 {
            start = usize::cast_from(corrections[(level - 1) * 3]) * 2;
        }
        let end = usize::cast_from(corrections[level * 3]) * 2;
        index = start + usize::cast_from(CUBE_POS_X) * 256 + usize::cast_from(UNIT_POS);
        if index >= end {
            terminate!();
        }
    } else {
        if index >= usize::cast_from(active_elems) {
            terminate!();
        }
        let row = index / 2;
        for candidate in 1..corrections.shape(0) {
            if row >= usize::cast_from(corrections[(candidate - 1) * 3]) {
                level = candidate;
            }
        }
    }
    let step_scale = corrections[level * 3 + 1];
    let inv_bc2_sqrt = corrections[level * 3 + 2];
    let g = gradient[index];
    gradient[index] = 0.0_f32;
    let m = first[index] * 0.9 + g * 0.1;
    let v = second[index] * 0.999 + g * g * 0.001;
    value[index] = value[index] - m * step_scale / (f32::sqrt(v) * inv_bc2_sqrt + 1e-8);
    first[index] = m;
    second[index] = v;
}

#[derive(Debug)]
struct PackedAdamOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for PackedAdamOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([v_ir, m_ir, s_ir, g_ir, c_ir], [ov_ir, om_ir, os_ir, og_ir]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(active_rows),
            ScalarIr::UInt(max_level_rows),
            ScalarIr::UInt(level_dispatch),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("Invalid scalar arguments for packed_adam")
        };
        let value = handles.get_float_tensor::<CubeBackend>(v_ir);
        let first = handles.get_float_tensor::<CubeBackend>(m_ir);
        let second = handles.get_float_tensor::<CubeBackend>(s_ir);
        let gradient = handles.get_float_tensor::<CubeBackend>(g_ir);
        let corrections = handles.get_float_tensor::<CubeBackend>(c_ir);
        // Preserve outstanding views (for example checkpoint parameter views).
        // Normal training transfers sole ownership and updates in place.
        let value = if value.can_mut() {
            value
        } else {
            crate::diagnostics::count(
                "adam.value_copy_bytes",
                value.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            value.copy()
        };
        let first = if first.can_mut() {
            first
        } else {
            crate::diagnostics::count(
                "adam.first_copy_bytes",
                first.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            first.copy()
        };
        let second = if second.can_mut() {
            second
        } else {
            crate::diagnostics::count(
                "adam.second_copy_bytes",
                second.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            second.copy()
        };
        let gradient = if gradient.can_mut() {
            gradient
        } else {
            crate::diagnostics::count(
                "adam.gradient_copy_bytes",
                gradient.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            gradient.copy()
        };
        let client = value.client.clone();
        let _timing = crate::diagnostics::start(&client, "adam");
        let n = (*active_rows as usize) * 2;
        let cube_dim = if *level_dispatch != 0 {
            CubeDim::new_1d(256)
        } else {
            CubeDim::new(&client, n)
        };
        let cube_count = if *level_dispatch != 0 {
            CubeCount::Static(
                ((*max_level_rows as usize) * 2).div_ceil(256) as u32,
                corrections.meta.shape().dims::<2>()[0] as u32,
                1,
            )
        } else {
            cubecl::calculate_cube_count_elemwise(&client, n, cube_dim)
        };
        packed_adam_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            value.clone().into_tensor_arg(),
            first.clone().into_tensor_arg(),
            second.clone().into_tensor_arg(),
            gradient.clone().into_tensor_arg(),
            corrections.into_tensor_arg(),
            n as u32,
            *level_dispatch != 0,
        );
        handles.register_float_tensor::<CubeBackend>(&ov_ir.id, value);
        handles.register_float_tensor::<CubeBackend>(&om_ir.id, first);
        handles.register_float_tensor::<CubeBackend>(&os_ir.id, second);
        handles.register_float_tensor::<CubeBackend>(&og_ir.id, gradient);
        Ok(())
    }
}

pub fn adam_packed(
    value: burn::tensor::Tensor<2>,
    first: burn::tensor::Tensor<2>,
    second: burn::tensor::Tensor<2>,
    gradient: burn::tensor::Tensor<2>,
    corrections: burn::tensor::Tensor<2>,
    active_rows: usize,
    max_level_rows: usize,
    level_dispatch: bool,
) -> [burn::tensor::Tensor<2>; 4] {
    let [v, m, s, g, c] = [value, first, second, gradient, corrections]
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = v.client.clone();
    let outputs = [&v, &m, &s, &g]
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype));
    let desc = CustomOpIr::with_scalars(
        "supernormal_packed_adam",
        &[
            v.into_ir(),
            m.into_ir(),
            s.into_ir(),
            g.into_ir(),
            c.into_ir(),
        ],
        &outputs,
        vec![
            ScalarIr::UInt(active_rows as u64),
            ScalarIr::UInt(max_level_rows as u64),
            ScalarIr::UInt(level_dispatch as u64),
        ],
    );
    let result: [_; 4] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            PackedAdamOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

#[derive(Debug)]
pub struct HashGridPackedForwardOp {
    pub desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for HashGridPackedForwardOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p_ir, t_ir, m_ir, f_ir], [out_ir]) = self.desc.as_fixed();
        let [ScalarIr::UInt(active)] = self.desc.scalars.as_slice() else {
            panic!("Invalid scalar arguments for hash_grid_packed_forward")
        };
        let points = handles.get_float_tensor::<CubeBackend>(p_ir);
        let tables = handles.get_float_tensor::<CubeBackend>(t_ir);
        let metadata = handles.get_float_tensor::<CubeBackend>(m_ir);
        let features = handles.get_float_tensor::<CubeBackend>(f_ir);
        let client = points.client.clone();
        let n = points.meta.shape().dims::<2>()[0];
        let cube_dim = CubeDim::new(&client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(&client, n, cube_dim);
        hash_grid_packed_forward_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            points.into_tensor_arg(),
            tables.into_tensor_arg(),
            metadata.into_tensor_arg(),
            features.clone().into_tensor_arg(),
            *active as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&out_ir.id, features);
        Ok(())
    }
}

pub fn forward_packed(
    points: burn::tensor::Tensor<2>,
    tables: burn::tensor::Tensor<2>,
    metadata: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    active: usize,
) -> burn::tensor::Tensor<2> {
    let p_prim = points.inner().try_into_primitive::<GpuBackend>().unwrap();
    let t_prim = tables.inner().try_into_primitive::<GpuBackend>().unwrap();
    let m_prim = metadata.inner().try_into_primitive::<GpuBackend>().unwrap();
    let f_prim = features.inner().try_into_primitive::<GpuBackend>().unwrap();
    let client = p_prim.client.clone();
    let shape = f_prim.shape.clone();
    let dtype = f_prim.dtype;
    let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
    let desc = CustomOpIr::with_scalars(
        "hash_grid_packed_forward",
        &[
            p_prim.into_ir(),
            t_prim.into_ir(),
            m_prim.into_ir(),
            f_prim.into_ir(),
        ],
        &[out_ir],
        vec![ScalarIr::UInt(active as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            HashGridPackedForwardOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out).autodiff()
}

#[derive(Debug)]
pub struct HashGridLevelForwardOp {
    pub desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for HashGridLevelForwardOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p_ir, t_ir, f_ir], [out_ir]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(res),
            ScalarIr::UInt(level),
            ScalarIr::Bool(dense),
            ScalarIr::Bool(copy_xyz),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("Invalid scalar arguments for hash_grid_level_forward")
        };

        let points = handles.get_float_tensor::<CubeBackend>(p_ir);
        let table = handles.get_float_tensor::<CubeBackend>(t_ir);
        let features = handles.get_float_tensor::<CubeBackend>(f_ir);

        let client = points.client.clone();
        let n = points.meta.shape().dims::<2>()[0];
        let cube_dim = CubeDim::new(&client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(&client, n, cube_dim);

        hash_grid_level_forward_kernel::launch::<f32>(
            &client,
            cube_count,
            cube_dim,
            points.into_tensor_arg(),
            table.into_tensor_arg(),
            features.clone().into_tensor_arg(),
            *res as u32,
            *level as u32,
            *dense,
            *copy_xyz,
        );

        handles.register_float_tensor::<CubeBackend>(&out_ir.id, features);
        Ok(())
    }
}

pub fn forward_level(
    points: burn::tensor::Tensor<2>,
    table: burn::tensor::Tensor<2>,
    features: burn::tensor::Tensor<2>,
    res: u32,
    level: u32,
    dense: bool,
    copy_xyz: bool,
) -> burn::tensor::Tensor<2> {
    let p_prim = points.inner().try_into_primitive::<GpuBackend>().unwrap();
    let t_prim = table.inner().try_into_primitive::<GpuBackend>().unwrap();
    let f_prim = features.inner().try_into_primitive::<GpuBackend>().unwrap();

    let client = p_prim.client.clone();
    let shape = f_prim.shape.clone();
    let dtype = f_prim.dtype;

    let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
    let desc = CustomOpIr::with_scalars(
        "hash_grid_level_forward",
        &[p_prim.into_ir(), t_prim.into_ir(), f_prim.into_ir()],
        &[out_ir],
        vec![
            ScalarIr::UInt(res as u64),
            ScalarIr::UInt(level as u64),
            ScalarIr::Bool(dense),
            ScalarIr::Bool(copy_xyz),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            HashGridLevelForwardOp { desc },
        )
        .outputs();

    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out).autodiff()
}

#[derive(Debug)]
pub struct HashGridLevelBackwardOp {
    pub desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for HashGridLevelBackwardOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p_ir, df_ir, tg_ir], [out_ir]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(res),
            ScalarIr::UInt(level),
            ScalarIr::Bool(dense),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("Invalid scalar arguments for hash_grid_level_backward")
        };

        let points = handles.get_float_tensor::<CubeBackend>(p_ir);
        let d_features = handles.get_float_tensor::<CubeBackend>(df_ir);
        let table_grad = handles.get_float_tensor::<CubeBackend>(tg_ir);

        let client = points.client.clone();
        let n = points.meta.shape().dims::<2>()[0];
        let cube_dim = CubeDim::new(&client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(&client, n, cube_dim);

        hash_grid_level_backward_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            points.into_tensor_arg(),
            d_features.into_tensor_arg(),
            table_grad.clone().into_tensor_arg(),
            *res as u32,
            *level as u32,
            *dense,
        );

        handles.register_float_tensor::<CubeBackend>(&out_ir.id, table_grad);
        Ok(())
    }
}

pub fn backward_level(
    points: burn::tensor::Tensor<2>,
    d_features: burn::tensor::Tensor<2>,
    table_grad: burn::tensor::Tensor<2>,
    res: u32,
    level: u32,
    dense: bool,
) -> burn::tensor::Tensor<2> {
    let p_prim = points.inner().try_into_primitive::<GpuBackend>().unwrap();
    let df_prim = d_features
        .inner()
        .try_into_primitive::<GpuBackend>()
        .unwrap();
    let tg_prim = table_grad
        .inner()
        .try_into_primitive::<GpuBackend>()
        .unwrap();

    let client = p_prim.client.clone();
    let shape = tg_prim.shape.clone();
    let dtype = tg_prim.dtype;

    let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
    let desc = CustomOpIr::with_scalars(
        "hash_grid_level_backward",
        &[p_prim.into_ir(), df_prim.into_ir(), tg_prim.into_ir()],
        &[out_ir],
        vec![
            ScalarIr::UInt(res as u64),
            ScalarIr::UInt(level as u64),
            ScalarIr::Bool(dense),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            HashGridLevelBackwardOp { desc },
        )
        .outputs();

    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

#[derive(Debug)]
pub struct HashGridPackedBackwardOp {
    pub desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for HashGridPackedBackwardOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p_ir, df_ir, m_ir, tg_ir], [out_ir]) = self.desc.as_fixed();
        let [ScalarIr::UInt(active), ScalarIr::UInt(valid_n)] = self.desc.scalars.as_slice() else {
            panic!("Invalid scalar arguments for hash_grid_packed_backward")
        };
        let points = handles.get_float_tensor::<CubeBackend>(p_ir);
        let d_features = handles.get_float_tensor::<CubeBackend>(df_ir);
        let metadata = handles.get_float_tensor::<CubeBackend>(m_ir);
        let table_grad = handles.get_float_tensor::<CubeBackend>(tg_ir);
        let client = points.client.clone();
        let n = (*valid_n as usize).max(1);
        let cube_dim = CubeDim::new(&client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(&client, n, cube_dim);
        hash_grid_packed_backward_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            points.into_tensor_arg(),
            d_features.into_tensor_arg(),
            metadata.into_tensor_arg(),
            table_grad.clone().into_tensor_arg(),
            *active as u32,
            *valid_n as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&out_ir.id, table_grad);
        Ok(())
    }
}

pub fn backward_packed(
    points: burn::tensor::Tensor<2>,
    d_features: burn::tensor::Tensor<2>,
    metadata: burn::tensor::Tensor<2>,
    table_grad: burn::tensor::Tensor<2>,
    active: usize,
    valid_n: usize,
) -> burn::tensor::Tensor<2> {
    let p_prim = points.inner().try_into_primitive::<GpuBackend>().unwrap();
    let df_prim = d_features
        .inner()
        .try_into_primitive::<GpuBackend>()
        .unwrap();
    let m_prim = metadata.inner().try_into_primitive::<GpuBackend>().unwrap();
    let tg_prim = table_grad
        .inner()
        .try_into_primitive::<GpuBackend>()
        .unwrap();
    let client = p_prim.client.clone();
    let shape = tg_prim.shape.clone();
    let dtype = tg_prim.dtype;
    let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
    let desc = CustomOpIr::with_scalars(
        "hash_grid_packed_backward",
        &[
            p_prim.into_ir(),
            df_prim.into_ir(),
            m_prim.into_ir(),
            tg_prim.into_ir(),
        ],
        &[out_ir],
        vec![
            ScalarIr::UInt(active as u64),
            ScalarIr::UInt(valid_n as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            HashGridPackedBackwardOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

pub fn test_hash_grid_backward_parity(device: &burn::tensor::Device) -> anyhow::Result<()> {
    use burn::tensor::TensorData;
    let pts = vec![
        -0.754f32, 0.456, -0.789, -0.999, 0.999, 0.000, 0.250, -0.500, 0.750, 0.888, -0.222, 0.666,
    ];
    let n = 4;
    let points = burn::tensor::Tensor::<2>::from_data(TensorData::new(pts.clone(), [n, 3]), device);

    // Create synthetic d_features [n, 31]
    let mut df_data = vec![0f32; n * 31];
    for i in 0..n {
        for c in 0..31 {
            df_data[i * 31 + c] = ((i * 31 + c) as f32 + 1.0) * 0.05;
        }
    }
    let d_features =
        burn::tensor::Tensor::<2>::from_data(TensorData::new(df_data.clone(), [n, 31]), device);

    // Test Level 0: res = 32, dense = true
    let res = 32u32;
    let entries = 35937;
    let table_grad = burn::tensor::Tensor::<2>::zeros([entries, 2], device);
    let out_grad = backward_level(points.clone(), d_features.clone(), table_grad, res, 0, true);
    let gpu_grad: Vec<f32> = out_grad.into_data().try_to_vec::<f32>()?;

    // CPU reference
    let mut ref_grad = vec![0f32; entries * 2];
    for i in 0..n {
        let p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
        let g0 = df_data[i * 31 + 3];
        let g1 = df_data[i * 31 + 4];

        let mut base = [0u32; 3];
        let mut w = [0f32; 3];
        for d in 0..3 {
            let coord = ((p[d] + 1.) * 0.5).clamp(0., 1. - 1e-7) * res as f32;
            base[d] = coord.floor() as u32;
            w[d] = coord - base[d] as f32;
        }

        for corner in 0..8 {
            let mut v = [0u32; 3];
            let mut weight = 1.;
            for d in 0..3 {
                let b = (corner >> d) & 1;
                v[d] = base[d] + b;
                weight *= if b == 0 { 1. - w[d] } else { w[d] };
            }
            let idx = (v[0] + v[1] * (res + 1) + v[2] * (res + 1) * (res + 1)) as usize;
            ref_grad[idx * 2] += weight * g0;
            ref_grad[idx * 2 + 1] += weight * g1;
        }
    }

    let mut max_diff0 = 0f32;
    for k in 0..entries * 2 {
        let diff = (gpu_grad[k] - ref_grad[k]).abs();
        if diff > max_diff0 {
            max_diff0 = diff;
        }
    }
    println!("Level 0 backward GPU vs CPU max diff: {max_diff0:.2e}");
    anyhow::ensure!(max_diff0 < 1e-5, "Mismatch on level 0 backward!");

    // Test Level 5: res = 128, dense = false (hash encoding)
    let res5 = 128u32;
    let entries5 = 524288;
    let table_grad5 = burn::tensor::Tensor::<2>::zeros([entries5, 2], device);
    let out_grad5 = backward_level(points, d_features, table_grad5, res5, 5, false);
    let gpu_grad5: Vec<f32> = out_grad5.into_data().try_to_vec::<f32>()?;

    let mut ref_grad5 = vec![0f32; entries5 * 2];
    for i in 0..n {
        let p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
        let g0 = df_data[i * 31 + 3 + 5 * 2];
        let g1 = df_data[i * 31 + 3 + 5 * 2 + 1];

        let mut base = [0u32; 3];
        let mut w = [0f32; 3];
        for d in 0..3 {
            let coord = ((p[d] + 1.) * 0.5).clamp(0., 1. - 1e-7) * res5 as f32;
            base[d] = coord.floor() as u32;
            w[d] = coord - base[d] as f32;
        }

        for corner in 0..8 {
            let mut v = [0u32; 3];
            let mut weight = 1.;
            for d in 0..3 {
                let b = (corner >> d) & 1;
                v[d] = base[d] + b;
                weight *= if b == 0 { 1. - w[d] } else { w[d] };
            }
            let idx = ((v[0] ^ v[1].wrapping_mul(2654435761) ^ v[2].wrapping_mul(805459861))
                & 524287) as usize;
            ref_grad5[idx * 2] += weight * g0;
            ref_grad5[idx * 2 + 1] += weight * g1;
        }
    }

    let mut max_diff5 = 0f32;
    for k in 0..entries5 * 2 {
        let diff = (gpu_grad5[k] - ref_grad5[k]).abs();
        if diff > max_diff5 {
            max_diff5 = diff;
        }
    }
    println!("Level 5 backward GPU vs CPU max diff: {max_diff5:.2e}");
    anyhow::ensure!(max_diff5 < 1e-5, "Mismatch on level 5 hashed backward!");

    println!("Hash grid GPU backward kernel passed parity test for both dense and hashed levels!");
    Ok(())
}

pub fn test_hash_grid_forward_parity(device: &burn::tensor::Device) -> anyhow::Result<()> {
    use burn::tensor::TensorData;
    let pts = vec![
        -0.754f32, 0.456, -0.789, -0.999, 0.999, 0.000, 0.250, -0.500, 0.750, 0.888, -0.222, 0.666,
    ];
    let points = burn::tensor::Tensor::<2>::from_data(TensorData::new(pts.clone(), [4, 3]), device);
    let mut features = burn::tensor::Tensor::<2>::zeros([4, 31], device);

    // Test Level 0: res = 32, dense = true
    let res = 32u32;
    let entries = 35937;
    let mut table_data = Vec::with_capacity(entries * 2);
    for i in 0..entries * 2 {
        table_data.push(((i % 100) as f32) * 0.01);
    }
    let table = burn::tensor::Tensor::<2>::from_data(
        TensorData::new(table_data.clone(), [entries, 2]),
        device,
    );

    features = forward_level(points, table, features, res, 0, true, true);
    let out_data: Vec<f32> = features.clone().into_data().try_to_vec::<f32>()?;

    // Compute reference on CPU
    for i in 0..4 {
        let p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
        // check xyz
        assert!((out_data[i * 31] - p[0]).abs() < 1e-6);
        assert!((out_data[i * 31 + 1] - p[1]).abs() < 1e-6);
        assert!((out_data[i * 31 + 2] - p[2]).abs() < 1e-6);

        // compute ref
        let mut base = [0u32; 3];
        let mut w = [0f32; 3];
        for d in 0..3 {
            let coord = ((p[d] + 1.) * 0.5).clamp(0., 1. - 1e-7) * res as f32;
            base[d] = coord.floor() as u32;
            w[d] = coord - base[d] as f32;
        }
        let mut ref_f0 = 0f32;
        let mut ref_f1 = 0f32;
        for corner in 0..8 {
            let mut v = [0u32; 3];
            let mut weight = 1.;
            for d in 0..3 {
                let b = (corner >> d) & 1;
                v[d] = base[d] + b;
                weight *= if b == 0 { 1. - w[d] } else { w[d] };
            }
            let idx = (v[0] + v[1] * (res + 1) + v[2] * (res + 1) * (res + 1)) as usize;
            ref_f0 += weight * table_data[idx * 2];
            ref_f1 += weight * table_data[idx * 2 + 1];
        }
        let gpu_f0 = out_data[i * 31 + 3];
        let gpu_f1 = out_data[i * 31 + 4];
        let diff0 = (gpu_f0 - ref_f0).abs();
        let diff1 = (gpu_f1 - ref_f1).abs();
        println!(
            "Level 0 Point {i}: GPU=({gpu_f0:.6}, {gpu_f1:.6}) CPU=({ref_f0:.6}, {ref_f1:.6}) diff=({diff0:.2e}, {diff1:.2e})"
        );
        anyhow::ensure!(diff0 < 1e-5 && diff1 < 1e-5, "Mismatch on level 0 forward!");
    }

    // Test Level 5: res = 128, dense = false (hash encoding)
    let res5 = 128u32;
    let entries5 = 524288;
    let mut table_data5 = Vec::with_capacity(entries5 * 2);
    for i in 0..entries5 * 2 {
        table_data5.push(((i % 127) as f32) * 0.005);
    }
    let table5 = burn::tensor::Tensor::<2>::from_data(
        TensorData::new(table_data5.clone(), [entries5, 2]),
        device,
    );
    let points5 =
        burn::tensor::Tensor::<2>::from_data(TensorData::new(pts.clone(), [4, 3]), device);
    features = forward_level(points5, table5, features, res5, 5, false, false);
    let out_data5: Vec<f32> = features.into_data().try_to_vec::<f32>()?;

    for i in 0..4 {
        let p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
        let mut base = [0u32; 3];
        let mut w = [0f32; 3];
        for d in 0..3 {
            let coord = ((p[d] + 1.) * 0.5).clamp(0., 1. - 1e-7) * res5 as f32;
            base[d] = coord.floor() as u32;
            w[d] = coord - base[d] as f32;
        }
        let mut ref_f0 = 0f32;
        let mut ref_f1 = 0f32;
        for corner in 0..8 {
            let mut v = [0u32; 3];
            let mut weight = 1.;
            for d in 0..3 {
                let b = (corner >> d) & 1;
                v[d] = base[d] + b;
                weight *= if b == 0 { 1. - w[d] } else { w[d] };
            }
            let idx = ((v[0] ^ v[1].wrapping_mul(2654435761) ^ v[2].wrapping_mul(805459861))
                & 524287) as usize;
            ref_f0 += weight * table_data5[idx * 2];
            ref_f1 += weight * table_data5[idx * 2 + 1];
        }
        let gpu_f0 = out_data5[i * 31 + 3 + 5 * 2];
        let gpu_f1 = out_data5[i * 31 + 3 + 5 * 2 + 1];
        let diff0 = (gpu_f0 - ref_f0).abs();
        let diff1 = (gpu_f1 - ref_f1).abs();
        println!(
            "Level 5 Point {i}: GPU=({gpu_f0:.6}, {gpu_f1:.6}) CPU=({ref_f0:.6}, {ref_f1:.6}) diff=({diff0:.2e}, {diff1:.2e})"
        );
        anyhow::ensure!(
            diff0 < 1e-5 && diff1 < 1e-5,
            "Mismatch on level 5 hashed forward!"
        );
    }
    println!("Hash grid GPU forward kernel passed parity test for both dense and hashed levels!");
    Ok(())
}

// Four points per workgroup. Hash interpolation groups neighboring points by
// level; when `corner_parallel` is active (default), all 224 threads across
// (4 points x 14 levels x 4 corner-pairs) issue trilinear corner lookups in
// parallel instead of serializing 8 corners on 56 threads while 200 sit idle.
#[cube]
fn field_feature_index(slot: usize, feature: usize, #[comptime] local_hash: bool) -> usize {
    if local_hash {
        feature * 4 + slot
    } else {
        slot * 32 + feature
    }
}

#[cube(launch)]
fn fused_field_kernel(
    points: &Tensor<f32>,
    tables: &Tensor<f32>,
    metadata: &Tensor<f32>,
    weights: &Tensor<f32>,
    output: &mut Tensor<f32>,
    features: &mut Tensor<f32>,
    preactivation: &mut Tensor<f32>,
    active: u32,
    valid_n: u32,
    #[comptime] cache: bool,
    #[comptime] local_hash: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] point_tiles: usize,
    #[comptime] subgroup_width: usize,
) {
    // CubeCL workgroup coordinates are u32; tensor/shared indices are usize.
    let mut unit = usize::cast_from(UNIT_POS);
    if subgroup_width > 0 {
        // Define logical neuron ownership using actual subgroup IDs; do not
        // assume the backend maps local invocation IDs to subgroups linearly.
        unit = usize::cast_from(PLANE_POS) * subgroup_width + usize::cast_from(UNIT_POS_PLANE);
    }
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            shared_weights[index] = weights[index];
        }
    }
    sync_cube();
    // Reuse the staged weights for several bounded four-point tiles.
    for point_tile in 0..point_tiles {
        let lane = unit % 64;
        let slot = unit / 64;
        let group =
            usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * point_tiles + point_tile;
        let point_idx = group * 4 + slot;
        let n_valid = usize::cast_from(valid_n);
        let valid = point_idx < n_valid;
        let mut shared_features = Shared::new_slice(128usize);
        let mut partials = Shared::new_slice(if corner_parallel { 512usize } else { 256usize });
        if lane < 32 {
            shared_features[field_feature_index(slot, lane, local_hash)] = 0.0_f32;
        }
        sync_cube();
        if valid {
            let p0 = points[point_idx * 3];
            let p1 = points[point_idx * 3 + 1];
            let p2 = points[point_idx * 3 + 2];
            if lane == 0 {
                shared_features[field_feature_index(slot, 0, local_hash)] = p0;
                shared_features[field_feature_index(slot, 1, local_hash)] = p1;
                shared_features[field_feature_index(slot, 2, local_hash)] = p2;
            }
        }
        if corner_parallel {
            // Map 256 threads to (hash_slot in 0..4, corner_pair in 0..4, level in 0..16).
            // Up to 4 * 4 * 14 = 224 threads concurrently fetch 2 corners each (448 lookups
            // in flight per workgroup), then reduce the 4 corner-pairs in shared memory.
            let hash_slot = unit & 3usize;
            let corner_pair = (unit >> 2) & 3usize;
            let level = unit >> 4;
            let hash_point = group * 4 + hash_slot;
            let mut feat0 = 0.0_f32;
            let mut feat1 = 0.0_f32;
            if hash_point < n_valid && level < usize::cast_from(active) {
                let p0 = points[hash_point * 3];
                let p1 = points[hash_point * 3 + 1];
                let p2 = points[hash_point * 3 + 2];
                let norm0 = f32::clamp((p0 + 1.) * 0.5, 0., 1. - 1e-7);
                let norm1 = f32::clamp((p1 + 1.) * 0.5, 0., 1. - 1e-7);
                let norm2 = f32::clamp((p2 + 1.) * 0.5, 0., 1. - 1e-7);
                let res = u32::cast_from(metadata[level * 2]);
                let offset = u32::cast_from(metadata[level * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;
                for c_sub in 0..2u32 {
                    let corner = u32::cast_from(corner_pair) * 2 + c_sub;
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1. - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1. - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1. - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    feat0 += weight * tables[entry];
                    feat1 += weight * tables[entry + 1];
                }
            }
            partials[unit * 2] = feat0;
            partials[unit * 2 + 1] = feat1;
            sync_cube();
            let red_slot = unit & 3usize;
            let red_level = unit >> 2;
            if red_level < usize::cast_from(active) {
                let base_p = (red_level * 16 + red_slot) * 2;
                let sum0 = (partials[base_p] + partials[base_p + 8])
                    + (partials[base_p + 16] + partials[base_p + 24]);
                let sum1 = (partials[base_p + 1] + partials[base_p + 9])
                    + (partials[base_p + 17] + partials[base_p + 25]);
                shared_features[field_feature_index(red_slot, 3 + red_level * 2, local_hash)] =
                    sum0;
                shared_features[field_feature_index(red_slot, 4 + red_level * 2, local_hash)] =
                    sum1;
            }
        } else {
            // Legacy 56-thread hash evaluation: one thread loops over all 8 corners.
            let hash_slot = if local_hash { unit % 4 } else { slot };
            let level = if local_hash { unit / 4 } else { lane };
            let hash_point = group * 4 + hash_slot;
            if hash_point < n_valid && level < usize::cast_from(active) {
                let p0 = points[hash_point * 3];
                let p1 = points[hash_point * 3 + 1];
                let p2 = points[hash_point * 3 + 2];
                let norm0 = f32::clamp((p0 + 1.) * 0.5, 0., 1. - 1e-7);
                let norm1 = f32::clamp((p1 + 1.) * 0.5, 0., 1. - 1e-7);
                let norm2 = f32::clamp((p2 + 1.) * 0.5, 0., 1. - 1e-7);
                let res = u32::cast_from(metadata[usize::cast_from(level) * 2]);
                let offset = u32::cast_from(metadata[usize::cast_from(level) * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;
                let mut feat0 = 0.0_f32;
                let mut feat1 = 0.0_f32;
                for corner in 0..8u32 {
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1. - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1. - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1. - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    feat0 += weight * tables[entry];
                    feat1 += weight * tables[entry + 1];
                }
                shared_features[field_feature_index(hash_slot, 3 + level * 2, local_hash)] = feat0;
                shared_features[field_feature_index(hash_slot, 4 + level * 2, local_hash)] = feat1;
            }
        }
        sync_cube();
        let mut contribution = 0.0_f32;
        if valid {
            let mut h = 0.0_f32;
            // Packed input-major weights give adjacent lanes adjacent reads.
            for k in 0..(3 + usize::cast_from(active) * 2) {
                h += shared_features[field_feature_index(slot, k, local_hash)]
                    * shared_weights[k * 64 + lane];
            }
            h += shared_weights[1984 + lane];
            if cache {
                preactivation[point_idx * 64 + lane] = h;
                if lane < 31 {
                    features[point_idx * 31 + lane] =
                        shared_features[field_feature_index(slot, lane, local_hash)];
                }
            }
            let scaled = h * 100.;
            let soft = f32::ln(1. + f32::exp(f32::min(scaled, 20.))) / 100.;
            let activated = if scaled > 20. { h } else { soft };
            contribution = activated * shared_weights[2048 + lane];
        }
        if subgroup_width > 0 {
            let sum = plane_sum(contribution);
            if subgroup_width == 64 {
                if valid && lane == 0 {
                    output[point_idx] = sum + shared_weights[2112];
                }
            } else {
                partials[unit] = sum;
                sync_cube();
                if valid && lane == 0 {
                    output[point_idx] =
                        partials[slot * 64] + partials[slot * 64 + 32] + shared_weights[2112];
                }
            }
        } else {
            partials[unit] = contribution;
            sync_cube();
            // Fixed tree reduction, no float atomics and no subgroup-size assumptions.
            for stage in 0..6usize {
                let width = 32usize >> stage;
                if lane < width {
                    partials[unit] += partials[unit + width];
                }
                sync_cube();
            }
            if valid && lane == 0 {
                output[point_idx] = partials[slot * 64] + shared_weights[2112];
            }
        }
        // The next hash tile reuses partials after every lane has consumed them.
        if point_tiles > 1 {
            sync_cube();
        }
    }
}

#[derive(Debug)]
struct FusedFieldOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for FusedFieldOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p, t, m, w, y, f, h], [oy, of, oh]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(active),
            ScalarIr::UInt(valid_n),
            ScalarIr::UInt(cache),
            ScalarIr::UInt(local_hash),
            ScalarIr::UInt(corner_parallel),
            ScalarIr::UInt(point_tiles),
            ScalarIr::UInt(subgroup_reduce),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid fused field arguments")
        };
        let [points, tables, metadata, weights, output, features, hidden] =
            [p, t, m, w, y, f, h].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let client = points.client.clone();
        let _timing = crate::diagnostics::start(&client, "field_forward");
        let props = client.properties();
        let hw = &props.hardware;
        let subgroup_width = if *subgroup_reduce != 0
            && props
                .features
                .plane
                .contains(cubecl::ir::features::Plane::Ops)
            && hw.plane_size_min == hw.plane_size_max
            && (hw.plane_size_min == 32 || hw.plane_size_min == 64)
        {
            hw.plane_size_min as usize
        } else {
            0usize
        };
        let point_tiles = (*point_tiles as usize).clamp(1, 4);
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| eprintln!("Field forward: {} points/group; subgroup reduction width {} (0 = portable shared tree)", point_tiles * 4, subgroup_width));
        let groups = (*valid_n as usize).div_ceil(4 * point_tiles).max(1);
        let gx = groups.min(65535).max(1);
        fused_field_kernel::launch(
            &client,
            CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
            CubeDim::new_1d(256),
            points.into_tensor_arg(),
            tables.into_tensor_arg(),
            metadata.into_tensor_arg(),
            weights.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            features.clone().into_tensor_arg(),
            hidden.clone().into_tensor_arg(),
            *active as u32,
            *valid_n as u32,
            *cache != 0,
            *local_hash != 0,
            *corner_parallel != 0,
            point_tiles,
            subgroup_width,
        );
        handles.register_float_tensor::<CubeBackend>(&oy.id, output);
        handles.register_float_tensor::<CubeBackend>(&of.id, features);
        handles.register_float_tensor::<CubeBackend>(&oh.id, hidden);
        Ok(())
    }
}

pub fn fused_field(
    inputs: [burn::tensor::Tensor<2>; 4],
    valid_n: usize,
    active: usize,
    cache: bool,
    local_hash: bool,
    corner_parallel: bool,
    point_tiles: usize,
    subgroup_reduce: bool,
) -> [burn::tensor::Tensor<2>; 3] {
    let n = inputs[0].dims()[0];
    let device = inputs[0].device();
    // `fused_field_kernel` zeroes all 32 `shared_features` slots and writes all
    // 31 columns of `features` for every valid point, so `Tensor::empty` is safe
    // regardless of `active`.
    let buffers = [
        burn::tensor::Tensor::<2>::empty([valid_n.max(1), 1], &device),
        burn::tensor::Tensor::<2>::empty([if cache { n } else { 1 }, 31], &device),
        burn::tensor::Tensor::<2>::empty([if cache { n } else { 1 }, 64], &device),
    ];
    let inputs: Vec<_> = inputs
        .into_iter()
        .chain(buffers)
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap())
        .collect();
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = inputs[4..]
        .iter()
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_fused_field",
        &inputs.into_iter().map(|t| t.into_ir()).collect::<Vec<_>>(),
        &outputs,
        vec![
            ScalarIr::UInt(active as u64),
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(cache as u64),
            ScalarIr::UInt(local_hash as u64),
            ScalarIr::UInt(corner_parallel as u64),
            ScalarIr::UInt(point_tiles as u64),
            ScalarIr::UInt(subgroup_reduce as u64),
        ],
    );
    let result: [_; 3] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            FusedFieldOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

// Barrier-free per-pixel ray compositor:
// 1. Background rays (`begin == end`, ~70-80% of random patches) immediately
//    evaluate their mask loss and exit in a few cycles without waiting at
//    multi-patch `sync_cube()` barriers.
// 2. When `compact_ids` is true (device-resident pipeline), `ids` has shape
//    `[vis_cap, 1]` storing `base_pt = start_ep * 9`. Because `end_ep` is
//    always `start_ep + 1`, all 18 points of sample `s` are contiguous at
//    `sdf[base_pt .. base_pt + 18]`, reducing index loads by 6x and turning all
//    4 stencil neighbor lookups (`l, r, u, d`) into contiguous L1 cache hits.
// 3. Forward caches `(c0, c1, s0, s1)` in `scratch[sb + 5..9]` so the reverse
//    pass executes zero `exp()` calls and uses branchless predication.
// Component-major within a nine-pixel sample: adjacent pixel lanes read/write
// adjacent words. Keep the former 10-float pixel records as an A/B variant.
#[cube]
fn composite_scratch_index(
    sample: usize,
    pixel: usize,
    component: usize,
    #[comptime] coalesced: bool,
) -> usize {
    if coalesced {
        sample * 90 + component * 9 + pixel
    } else {
        (sample * 9 + pixel) * 10 + component
    }
}

#[cube(launch)]
fn composite_kernel(
    sdf: &Tensor<f32>,
    geometry: &Tensor<f32>,
    ids: &Tensor<f32>,
    patches: &Tensor<f32>,
    vis_prefix: &Tensor<f32>,
    scratch: &mut Tensor<f32>,
    stats: &mut Tensor<f32>,
    #[comptime] compact_ids: bool,
    #[comptime] coalesced_scratch: bool,
    #[comptime] reuse_endpoints: bool,
) {
    let ray = ABSOLUTE_POS;
    if ray >= patches.shape(0) * 9 {
        terminate!();
    }
    let p = ray / 9;
    let pixel = ray % 9;
    let pb = p * 123;
    let begin = if patches[pb] < 0.0_f32 {
        let g_ray = usize::cast_from(-patches[pb] - 1.0_f32);
        let s_base = usize::cast_from(patches[pb + 1]);
        let g_begin = if g_ray == 0 {
            0usize
        } else {
            usize::cast_from(vis_prefix[(g_ray - 1) * 2])
        };
        g_begin - s_base
    } else {
        usize::cast_from(patches[pb])
    };
    let end = if patches[pb] < 0.0_f32 {
        let g_ray = usize::cast_from(-patches[pb] - 1.0_f32);
        let s_base = usize::cast_from(patches[pb + 1]);
        usize::cast_from(vis_prefix[g_ray * 2]) - s_base
    } else {
        usize::cast_from(patches[pb + 1])
    };
    let inv = patches[pb + 2];
    let normal_den = patches[pb + 3];
    let pixel_den = patches[pb + 4];
    let sample_den = patches[pb + 5];

    let mut normals = Array::<f32>::new(3usize);
    let mut dn = Array::<f32>::new(3usize);
    let mut masks = 0.0_f32;
    let mut trans = 1.0_f32;
    let mut dm = 0.0_f32;
    let mut future = 0.0_f32;
    for i in 0..3usize {
        normals[i] = 0.0_f32;
        dn[i] = 0.0_f32;
    }

    let col = pixel % 3;
    let row = pixel / 3;
    let l = if col == 0 { pixel } else { pixel - 1 };
    let r = if col == 2 { pixel } else { pixel + 1 };
    let u = if row == 0 { pixel } else { pixel - 3 };
    let d = if row == 2 { pixel } else { pixel + 3 };

    let mut prev_end_id = 0usize;
    let mut prev_c1 = 0.0_f32;
    for s in begin..end {
        let mut base_pt = 0usize;
        if compact_ids {
            base_pt = usize::cast_from(ids[s]);
        }
        let ib = s * 18;
        let i0 = if compact_ids {
            base_pt + pixel
        } else {
            usize::cast_from(ids[ib + pixel])
        };
        let i1 = if compact_ids {
            base_pt + 9 + pixel
        } else {
            usize::cast_from(ids[ib + 9 + pixel])
        };
        let il = if compact_ids {
            base_pt + l
        } else {
            usize::cast_from(ids[ib + l])
        };
        let ir = if compact_ids {
            base_pt + r
        } else {
            usize::cast_from(ids[ib + r])
        };
        let iu = if compact_ids {
            base_pt + u
        } else {
            usize::cast_from(ids[ib + u])
        };
        let id = if compact_ids {
            base_pt + d
        } else {
            usize::cast_from(ids[ib + d])
        };
        let s0 = sdf[i0];
        let s1 = sdf[i1];
        let sl = sdf[il];
        let sr = sdf[ir];
        let su = sdf[iu];
        let sd = sdf[id];
        let gb = s * 27 + pixel * 3;
        let dt = (s1 - s0) * geometry[gb];
        let dx = (sr - sl) * geometry[gb + 1];
        let dy = (sd - su) * geometry[gb + 2];
        let reuse = reuse_endpoints && s > begin && i0 == prev_end_id;
        let mut c0 = prev_c1;
        if !reuse {
            c0 = 1.0_f32 / (1.0_f32 + f32::exp(-s0 * inv));
        }
        let c1 = 1.0_f32 / (1.0_f32 + f32::exp(-s1 * inv));
        prev_c1 = c1;
        prev_end_id = i1;
        let a = f32::clamp((c0 - c1 + 1e-5_f32) / (c0 + 1e-5_f32), 0.0_f32, 1.0_f32);
        let w = trans * a;
        scratch[composite_scratch_index(s, pixel, 3, coalesced_scratch)] = trans;
        scratch[composite_scratch_index(s, pixel, 4, coalesced_scratch)] = a;
        scratch[composite_scratch_index(s, pixel, 5, coalesced_scratch)] = c0;
        scratch[composite_scratch_index(s, pixel, 6, coalesced_scratch)] = c1;
        scratch[composite_scratch_index(s, pixel, 7, coalesced_scratch)] = s0;
        scratch[composite_scratch_index(s, pixel, 8, coalesced_scratch)] = s1;
        trans *= f32::max(1.0_f32 - a, 1e-10_f32);
        masks += w;
        for j in 0..3usize {
            let vb = pb + 6 + pixel * 13 + j * 3;
            let g = patches[vb] * dt + patches[vb + 1] * dx + patches[vb + 2] * dy;
            scratch[composite_scratch_index(s, pixel, j, coalesced_scratch)] = g;
            normals[j] += w * g;
        }
    }

    let mut nl = 0.0_f32;
    let mut ml = 0.0_f32;
    let mut el = 0.0_f32;
    let mut dinv = 0.0_f32;
    let mut bias = 0.0_f32;
    let mask = patches[pb + 6 + pixel * 13 + 12];
    for j in 0..3usize {
        let err = (normals[j] - patches[pb + 6 + pixel * 13 + 9 + j]) * mask;
        nl += err * err / normal_den;
        dn[j] = 2.0_f32 * err * mask / normal_den;
    }
    let m = f32::clamp(masks, 1e-5_f32, 1.0_f32 - 1e-5_f32);
    ml += -(mask * f32::ln(m) + (1.0_f32 - mask) * f32::ln(1.0_f32 - m)) / pixel_den;
    if masks >= 1e-5_f32 && masks <= 1.0_f32 - 1e-5_f32 {
        dm = (m - mask) / (m * (1.0_f32 - m) * pixel_den);
    }

    for rev in 0..(end - begin) {
        let s = end - 1 - rev;
        let g0 = scratch[composite_scratch_index(s, pixel, 0, coalesced_scratch)];
        let g1 = scratch[composite_scratch_index(s, pixel, 1, coalesced_scratch)];
        let g2 = scratch[composite_scratch_index(s, pixel, 2, coalesced_scratch)];
        let t = scratch[composite_scratch_index(s, pixel, 3, coalesced_scratch)];
        let a = scratch[composite_scratch_index(s, pixel, 4, coalesced_scratch)];
        let c0 = scratch[composite_scratch_index(s, pixel, 5, coalesced_scratch)];
        let c1 = scratch[composite_scratch_index(s, pixel, 6, coalesced_scratch)];
        let s0 = scratch[composite_scratch_index(s, pixel, 7, coalesced_scratch)];
        let s1 = scratch[composite_scratch_index(s, pixel, 8, coalesced_scratch)];
        let norm = f32::sqrt(g0 * g0 + g1 * g1 + g2 * g2);
        el += (norm - 1.0_f32) * (norm - 1.0_f32) / sample_den;
        let eg = 2.0_f32 * (norm - 1.0_f32) / sample_den / f32::max(norm, 1e-12_f32);
        let dg0 = t * a * dn[0] + eg * g0;
        let dg1 = t * a * dn[1] + eg * g1;
        let dg2 = t * a * dn[2] + eg * g2;
        let vb = pb + 6 + pixel * 13;
        let gb = s * 27 + pixel * 3;
        let dp0 =
            (patches[vb] * dg0 + patches[vb + 3] * dg1 + patches[vb + 6] * dg2) * geometry[gb];
        let dp1 = (patches[vb + 1] * dg0 + patches[vb + 4] * dg1 + patches[vb + 7] * dg2)
            * geometry[gb + 1];
        let dp2 = (patches[vb + 2] * dg0 + patches[vb + 5] * dg1 + patches[vb + 8] * dg2)
            * geometry[gb + 2];
        scratch[composite_scratch_index(s, pixel, 5, coalesced_scratch)] = dp0;
        scratch[composite_scratch_index(s, pixel, 6, coalesced_scratch)] = dp1;
        scratch[composite_scratch_index(s, pixel, 7, coalesced_scratch)] = dp2;
        let gw = dn[0] * g0 + dn[1] * g1 + dn[2] * g2 + dm;
        let one_minus_a = 1.0_f32 - a;
        let q = f32::max(one_minus_a, 1e-10_f32);
        let tail = if one_minus_a >= 1e-10_f32 {
            future
        } else {
            0.0_f32
        };
        let da = t * (gw - tail);
        future = gw * a + q * future;
        let denom = c0 + 1e-5_f32;
        let raw = (c0 - c1 + 1e-5_f32) / denom;
        // Branchless predication (s05_branchless_predication): evaluate z0/z1 via
        // register select and issue a single unconditional store per scratch slot.
        let active_raw = raw >= 0.0_f32 && raw <= 1.0_f32;
        let z0_raw = da * c1 / (denom * denom) * c0 * (1.0_f32 - c0);
        let z1_raw = -da / denom * c1 * (1.0_f32 - c1);
        let z0 = if active_raw { z0_raw } else { 0.0_f32 };
        let z1 = if active_raw { z1_raw } else { 0.0_f32 };
        scratch[composite_scratch_index(s, pixel, 8, coalesced_scratch)] = z0 * inv;
        scratch[composite_scratch_index(s, pixel, 9, coalesced_scratch)] = z1 * inv;
        bias += z0 * inv + z1 * inv;
        dinv += z0 * s0 + z1 * s1;
    }
    stats[ray * 5] = nl;
    stats[ray * 5 + 1] = ml;
    stats[ray * 5 + 2] = el;
    stats[ray * 5 + 3] = dinv;
    stats[ray * 5 + 4] = bias;
}

#[derive(Debug)]
struct CompositeOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for CompositeOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([s, g, i, p, vp, w, d, l, r], [od, ol]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(valid_points),
            ScalarIr::UInt(compact_tables),
            ScalarIr::UInt(coalesced_scratch),
            ScalarIr::UInt(reuse_endpoints),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid composite scalars")
        };
        let [s, g, i, p, vp, w, d, l, r] =
            [s, g, i, p, vp, w, d, l, r].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = s.client.clone();
        let rays = (p.meta.shape().dims::<2>()[0] * 9).max(1);
        let dim_rays = CubeDim::new(&client, rays);
        let compact = *compact_tables != 0;
        let timing = crate::diagnostics::start(&client, "composite");
        composite_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, rays, dim_rays),
            dim_rays,
            s.into_tensor_arg(),
            g.into_tensor_arg(),
            i.into_tensor_arg(),
            p.into_tensor_arg(),
            vp.into_tensor_arg(),
            w.clone().into_tensor_arg(),
            l.clone().into_tensor_arg(),
            compact,
            *coalesced_scratch != 0,
            *reuse_endpoints != 0,
        );
        drop(timing);
        let _timing = crate::diagnostics::start(&client, "composite_gather");
        let count = (*valid_points as usize).max(1);
        let dim = CubeDim::new(&client, count);
        composite_gather_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, count, dim),
            dim,
            w.into_tensor_arg(),
            r.into_tensor_arg(),
            d.clone().into_tensor_arg(),
            *valid_points as u32,
            compact,
            *coalesced_scratch != 0,
        );
        handles.register_float_tensor::<CubeBackend>(&od.id, d);
        handles.register_float_tensor::<CubeBackend>(&ol.id, l);
        Ok(())
    }
}
pub fn composite_gpu(
    inputs: [burn::tensor::Tensor<2>; 9],
    valid_points: usize,
    compact_tables: bool,
    coalesced_scratch: bool,
    reuse_endpoints: bool,
) -> [burn::tensor::Tensor<2>; 2] {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = inputs[6..8]
        .iter()
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_composite",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(valid_points as u64),
            ScalarIr::UInt(compact_tables as u64),
            ScalarIr::UInt(coalesced_scratch as u64),
            ScalarIr::UInt(reuse_endpoints as u64),
        ],
    );
    let result: [_; 2] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            CompositeOp { desc },
        )
        .outputs();
    result.map(burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>)
}

// Keep total shared memory at 2,821 floats (11,284 bytes < 12 KB) so 4 blocks
// of 256 threads on some Ampere configurations; actual residency also depends
// on register allocation and device limits.
// Both the 64-neuron `dh * W0` feature-gradient dot product (16 iterations per
// thread across 224 threads) and the 8-corner trilinear hash scatter (2 corners
// per thread across 224 threads) execute across 87.5% of the workgroup.
#[cube(launch)]
fn field_backward_kernel(
    pre: &Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    dh: &mut Tensor<f32>,
    dw1: &mut Tensor<f32>,
    points: &Tensor<f32>,
    metadata: &Tensor<f32>,
    table_grad: &Tensor<Atomic<u32>>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    #[comptime] local_hash: bool,
    #[comptime] compact_gradients: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] cross_corners: bool,
    #[comptime] write_dh: bool,
) {
    let unit = usize::cast_from(UNIT_POS);
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let target = if local_hash && index < 1984 {
                (index % 64) * 31 + index / 64
            } else {
                index
            };
            shared_weights[target] = weights[index];
        }
    }
    sync_cube();
    let lane = unit % 64;
    let slot = unit / 64;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let point = group * 4 + slot;
    let n_valid = usize::cast_from(valid_n);
    let mut shared = Shared::new_slice(256usize);
    // Reused first for `output_grads` `[256]` and then for 224-thread partial
    // feature gradients `[448]`, keeping shared memory under 12 KB per block.
    let mut partials = Shared::new_slice(if corner_parallel {
        448usize
    } else if compact_gradients {
        256usize
    } else {
        1usize
    });
    let mut slot_ds = Shared::new_slice(4usize);
    let mut output_grad = 0.0_f32;
    let mut v = 0.0_f32;
    let mut ds_val = 0.0_f32;
    if point < n_valid {
        ds_val = ds[point];
        if ds_val != 0.0_f32 {
            let h = pre[point * 64 + lane];
            let scaled = h * 100.0_f32;
            let exp_pos = f32::exp(f32::min(scaled, 20.0_f32));
            let derivative = if scaled > 20.0_f32 {
                1.0_f32
            } else {
                exp_pos / (1.0_f32 + exp_pos)
            };
            v = ds_val * shared_weights[2048 + lane] * derivative;
            let soft = f32::ln(1.0_f32 + exp_pos) * 0.01_f32;
            let hidden = if scaled > 20.0_f32 { h } else { soft };
            output_grad = ds_val * hidden;
        }
        if write_dh {
            dh[point * 64 + lane] = v;
        }
        if !compact_gradients {
            dw1[point * 64 + lane] = output_grad;
        }
    }
    if lane == 0 {
        slot_ds[slot] = ds_val;
    }
    let shared_index = if local_hash { lane * 4 + slot } else { unit };
    shared[shared_index] = v;
    if compact_gradients {
        partials[unit] = output_grad;
    }
    sync_cube();
    if compact_gradients && slot == 0 {
        if group < dw1.shape(0) {
            dw1[group * 64 + lane] = (partials[lane] + partials[64 + lane])
                + (partials[128 + lane] + partials[192 + lane]);
        }
    }
    if slot_ds[0] == 0.0_f32
        && slot_ds[1] == 0.0_f32
        && slot_ds[2] == 0.0_f32
        && slot_ds[3] == 0.0_f32
    {
        terminate!();
    }
    if corner_parallel {
        if compact_gradients {
            sync_cube();
        }
        // Map 224 threads across (hash_slot in 0..4, sub_part in 0..4, level in 0..14).
        // Phase 1: Each thread accumulates 16 of the 64 hidden neurons (4x faster than
        // 56 threads looping 64 times) and writes `[p_g0, p_g1]` to `partials`.
        let hash_slot = unit & 3usize;
        let sub_part = (unit >> 2) & 3usize;
        let level = unit >> 4;
        let hash_point = group * 4 + hash_slot;
        if level < 14usize {
            let mut p_g0 = 0.0_f32;
            let mut p_g1 = 0.0_f32;
            if hash_point < n_valid
                && level < usize::cast_from(active)
                && slot_ds[hash_slot] != 0.0_f32
            {
                let col = 3 + level * 2;
                let j_base = sub_part * 16;
                for j_off in 0..16usize {
                    let j = j_base + j_off;
                    let hi = if local_hash {
                        j * 4 + hash_slot
                    } else {
                        hash_slot * 64 + j
                    };
                    let w_idx0 = if local_hash {
                        j * 31 + col
                    } else {
                        col * 64 + j
                    };
                    let w_idx1 = if local_hash {
                        j * 31 + col + 1
                    } else {
                        (col + 1) * 64 + j
                    };
                    let h = shared[hi];
                    p_g0 += h * shared_weights[w_idx0];
                    p_g1 += h * shared_weights[w_idx1];
                }
            }
            partials[unit * 2] = p_g0;
            partials[unit * 2 + 1] = p_g1;
        }
        sync_cube();
        // Phase 2: All 224 threads sum the 4 partial dot products for `(level, hash_slot)`
        // and scatter their 2 trilinear corners concurrently (448 lookups/atomics per block).
        let mut scatter_keys = Array::<f32>::new(2usize);
        let mut scatter_g0 = Array::<f32>::new(2usize);
        let mut scatter_g1 = Array::<f32>::new(2usize);
        for i in 0..2usize {
            scatter_keys[i] = -1.0_f32;
            scatter_g0[i] = 0.0_f32;
            scatter_g1[i] = 0.0_f32;
        }
        if hash_point < n_valid && level < usize::cast_from(active) && slot_ds[hash_slot] != 0.0_f32
        {
            let base_p = (level * 16 + hash_slot) * 2;
            let g0 = (partials[base_p] + partials[base_p + 8])
                + (partials[base_p + 16] + partials[base_p + 24]);
            let g1 = (partials[base_p + 1] + partials[base_p + 9])
                + (partials[base_p + 17] + partials[base_p + 25]);
            if g0 != 0.0_f32 || g1 != 0.0_f32 {
                let norm0 = f32::clamp(
                    (points[hash_point * 3] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (points[hash_point * 3 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (points[hash_point * 3 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res = u32::cast_from(metadata[level * 2]);
                let offset = u32::cast_from(metadata[level * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;
                for c_sub in 0..2u32 {
                    let corner = u32::cast_from(sub_part) * 2 + c_sub;
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1.0_f32 - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1.0_f32 - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1.0_f32 - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    let wg0 = weight * g0;
                    let wg1 = weight * g1;
                    if aggregate_hash && level < usize::cast_from(aggregation_levels) {
                        let c = usize::cast_from(c_sub);
                        scatter_keys[c] = f32::cast_from(idx);
                        scatter_g0[c] = wg0;
                        scatter_g1[c] = wg1;
                    } else {
                        if wg0 != 0.0_f32 {
                            atomic_add_f32(table_grad, entry, wg0);
                        }
                        if wg1 != 0.0_f32 {
                            atomic_add_f32(table_grad, entry + 1, wg1);
                        }
                    }
                }
            }
        }
        if aggregate_hash {
            // Repurpose the weight tile after every lane finishes reading it.
            // Keys are level-local integer addresses, exactly representable in
            // f32. This adds no shared-memory allocation.
            sync_cube();
            if unit < 224usize {
                for c in 0..2usize {
                    let staging = c * 224 + unit;
                    shared_weights[staging] = scatter_keys[c];
                    shared_weights[448 + staging] = scatter_g0[c];
                    shared_weights[896 + staging] = scatter_g1[c];
                }
            }
            sync_cube();
            if unit < 224usize
                && level < usize::cast_from(active)
                && level < usize::cast_from(aggregation_levels)
            {
                let peers = unit - hash_slot;
                let offset = usize::cast_from(metadata[level * 2 + 1]);
                for c in 0..2usize {
                    let key = scatter_keys[c];
                    let mut leader = key >= 0.0_f32;
                    let peer_count = if cross_corners { 32usize } else { 4usize };
                    let owner = if cross_corners {
                        c * 16 + sub_part * 4 + hash_slot
                    } else {
                        hash_slot
                    };
                    for other in 0..peer_count {
                        let staging = if cross_corners {
                            (other / 16) * 224 + level * 16 + other % 16
                        } else {
                            c * 224 + peers + other
                        };
                        if other < owner && shared_weights[staging] == key {
                            leader = false;
                        }
                    }
                    if leader {
                        let mut sum0 = 0.0_f32;
                        let mut sum1 = 0.0_f32;
                        for other in 0..peer_count {
                            let staging = if cross_corners {
                                (other / 16) * 224 + level * 16 + other % 16
                            } else {
                                c * 224 + peers + other
                            };
                            if shared_weights[staging] == key {
                                sum0 += shared_weights[448 + staging];
                                sum1 += shared_weights[896 + staging];
                            }
                        }
                        let entry = (offset + usize::cast_from(key)) * 2;
                        if sum0 != 0.0_f32 {
                            atomic_add_f32(table_grad, entry, sum0);
                        }
                        if sum1 != 0.0_f32 {
                            atomic_add_f32(table_grad, entry + 1, sum1);
                        }
                    }
                }
            }
        }
    } else {
        let hash_slot = if local_hash { unit % 4 } else { slot };
        let level = if local_hash { unit / 4 } else { lane };
        let hash_point = group * 4 + hash_slot;
        if hash_point < n_valid && level < usize::cast_from(active) && slot_ds[hash_slot] != 0.0_f32
        {
            let col = 3 + level * 2;
            let mut g0 = 0.0_f32;
            let mut g1 = 0.0_f32;
            for j in 0..64usize {
                let hi = if local_hash {
                    j * 4 + hash_slot
                } else {
                    hash_slot * 64 + j
                };
                let w0 = if local_hash {
                    j * 31 + col
                } else {
                    col * 64 + j
                };
                let w1 = if local_hash {
                    j * 31 + col + 1
                } else {
                    (col + 1) * 64 + j
                };
                let h = shared[hi];
                g0 += h * shared_weights[w0];
                g1 += h * shared_weights[w1];
            }
            if g0 != 0.0_f32 || g1 != 0.0_f32 {
                let norm0 = f32::clamp(
                    (points[hash_point * 3] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (points[hash_point * 3 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (points[hash_point * 3 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res = u32::cast_from(metadata[usize::cast_from(level) * 2]);
                let offset = u32::cast_from(metadata[usize::cast_from(level) * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;

                for corner in 0..8u32 {
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1.0_f32 - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1.0_f32 - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1.0_f32 - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    let wg0 = weight * g0;
                    let wg1 = weight * g1;
                    if wg0 != 0.0_f32 {
                        atomic_add_f32(table_grad, entry, wg0);
                    }
                    if wg1 != 0.0_f32 {
                        atomic_add_f32(table_grad, entry + 1, wg1);
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct FieldBackwardOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for FieldBackwardOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p, d, w, x, m, t, h, g], [ot, oh, og]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(active),
            ScalarIr::UInt(valid_n),
            ScalarIr::UInt(local_hash),
            ScalarIr::UInt(compact_gradients),
            ScalarIr::UInt(corner_parallel),
            ScalarIr::UInt(aggregate_hash),
            ScalarIr::UInt(aggregation_levels),
            ScalarIr::UInt(cross_corners),
            ScalarIr::UInt(write_dh),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid backward scalars")
        };
        let [p, d, w, x, m, t, h, g] =
            [p, d, w, x, m, t, h, g].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let t = if t.can_mut() {
            t
        } else {
            crate::diagnostics::count(
                "hash_backward.gradient_copy_bytes",
                t.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            t.copy()
        };
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "field_backward");
        let groups = (*valid_n as usize).div_ceil(4).max(1);
        let gx = groups.min(65535).max(1);
        let native_add = client
            .properties()
            .atomic_type_usage(Type::atomic(f32::elem_type_native()))
            .contains(cubecl::ir::features::AtomicUsage::Add);
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| {
            eprintln!(
                "Hash gradients: {}; hash layout: {}; corner parallelism: {}; local aggregation: {}",
                if native_add {
                    "native float atomics"
                } else {
                    "compare-and-swap atomics"
                },
                if *local_hash != 0 {
                    "neighboring points grouped by level"
                } else {
                    "legacy point-major"
                },
                if *corner_parallel != 0 {
                    "256-thread corner-parallel"
                } else {
                    "legacy 56-thread sequential corners"
                },
                if *aggregate_hash != 0 && *corner_parallel != 0 && *aggregation_levels > 0 {
                    "enabled"
                } else {
                    "disabled"
                },
            );
        });
        if native_add {
            field_backward_native_kernel::launch(
                &client,
                CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
                CubeDim::new_1d(256),
                p.into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                h.clone().into_tensor_arg(),
                g.clone().into_tensor_arg(),
                x.into_tensor_arg(),
                m.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *local_hash != 0,
                *compact_gradients != 0,
                *corner_parallel != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *cross_corners != 0,
                *write_dh != 0,
            );
        } else {
            field_backward_kernel::launch(
                &client,
                CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
                CubeDim::new_1d(256),
                p.into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                h.clone().into_tensor_arg(),
                g.clone().into_tensor_arg(),
                x.into_tensor_arg(),
                m.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *local_hash != 0,
                *compact_gradients != 0,
                *corner_parallel != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *cross_corners != 0,
                *write_dh != 0,
            );
        }
        handles.register_float_tensor::<CubeBackend>(&ot.id, t);
        handles.register_float_tensor::<CubeBackend>(&oh.id, h);
        handles.register_float_tensor::<CubeBackend>(&og.id, g);
        Ok(())
    }
}
pub fn field_backward(
    inputs: [burn::tensor::Tensor<2>; 6],
    valid_n: usize,
    active: usize,
    local_hash: bool,
    compact_gradients: bool,
    corner_parallel: bool,
    aggregate_hash: bool,
    aggregation_levels: usize,
    cross_corners: bool,
    write_dh: bool,
) -> [burn::tensor::Tensor<2>; 3] {
    use burn::tensor::Tensor as BT;
    let n = inputs[0].dims()[0];
    let device = inputs[0].device();
    let dw1_rows = if compact_gradients {
        valid_n.div_ceil(4).max(1)
    } else {
        valid_n.max(1)
    };
    let inputs: Vec<_> = inputs
        .into_iter()
        .chain([
            BT::empty([if write_dh { n } else { 1 }, 64], &device),
            BT::empty([dw1_rows, 64], &device),
        ])
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap())
        .collect();
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = inputs[5..]
        .iter()
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_field_backward",
        &inputs.into_iter().map(|t| t.into_ir()).collect::<Vec<_>>(),
        &outputs,
        vec![
            ScalarIr::UInt(active as u64),
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(local_hash as u64),
            ScalarIr::UInt(compact_gradients as u64),
            ScalarIr::UInt(corner_parallel as u64),
            ScalarIr::UInt(aggregate_hash as u64),
            ScalarIr::UInt(aggregation_levels as u64),
            ScalarIr::UInt(cross_corners as u64),
            ScalarIr::UInt(write_dh as u64),
        ],
    );
    let result: [_; 3] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            FieldBackwardOp { desc },
        )
        .outputs();
    result.map(BT::<2>::from_primitive::<GpuBackend>)
}

#[cube(launch)]
fn composite_gather_kernel(
    scratch: &Tensor<f32>,
    reverse: &Tensor<f32>,
    ds: &mut Tensor<f32>,
    valid_points: u32,
    #[comptime] compact_reverse: bool,
    #[comptime] coalesced_scratch: bool,
) {
    let point = ABSOLUTE_POS;
    if point >= usize::cast_from(valid_points) {
        terminate!();
    }
    let mut total = 0.0_f32;
    if compact_reverse {
        let ep = point / 9;
        let pixel = point % 9;
        let p_col = pixel % 3;
        let p_row = pixel / 3;
        for occurrence in 0..2usize {
            let code = reverse[ep * 2 + occurrence];
            if code >= 0.0_f32 {
                let c = usize::cast_from(code);
                let sample = c / 2;
                let is_end = (c & 1usize) != 0usize;
                if is_end {
                    total += scratch[composite_scratch_index(sample, pixel, 5, coalesced_scratch)]
                        + scratch[composite_scratch_index(sample, pixel, 9, coalesced_scratch)];
                } else {
                    total += -scratch[composite_scratch_index(sample, pixel, 5, coalesced_scratch)]
                        + scratch[composite_scratch_index(sample, pixel, 8, coalesced_scratch)];
                    if p_col == 0 {
                        total -=
                            scratch[composite_scratch_index(sample, pixel, 6, coalesced_scratch)];
                    }
                    if p_col > 0 {
                        total += scratch
                            [composite_scratch_index(sample, pixel - 1, 6, coalesced_scratch)];
                    }
                    if p_col < 2 {
                        total -= scratch
                            [composite_scratch_index(sample, pixel + 1, 6, coalesced_scratch)];
                    }
                    if p_col == 2 {
                        total +=
                            scratch[composite_scratch_index(sample, pixel, 6, coalesced_scratch)];
                    }
                    if p_row == 0 {
                        total -=
                            scratch[composite_scratch_index(sample, pixel, 7, coalesced_scratch)];
                    }
                    if p_row > 0 {
                        total += scratch
                            [composite_scratch_index(sample, pixel - 3, 7, coalesced_scratch)];
                    }
                    if p_row < 2 {
                        total -= scratch
                            [composite_scratch_index(sample, pixel + 3, 7, coalesced_scratch)];
                    }
                    if p_row == 2 {
                        total +=
                            scratch[composite_scratch_index(sample, pixel, 7, coalesced_scratch)];
                    }
                }
            }
        }
    } else {
        for occurrence in 0..2usize {
            let code = reverse[point * 2 + occurrence];
            if code >= 0.0_f32 {
                let code = usize::cast_from(code);
                let sample = code / 18;
                let corner = code % 18;
                let pixel = corner % 9;
                if corner >= 9 {
                    total += scratch[composite_scratch_index(sample, pixel, 5, coalesced_scratch)]
                        + scratch[composite_scratch_index(sample, pixel, 9, coalesced_scratch)];
                } else {
                    total += -scratch[composite_scratch_index(sample, pixel, 5, coalesced_scratch)]
                        + scratch[composite_scratch_index(sample, pixel, 8, coalesced_scratch)];
                    let p_col = pixel % 3;
                    let p_row = pixel / 3;
                    if p_col == 0 {
                        total -=
                            scratch[composite_scratch_index(sample, pixel, 6, coalesced_scratch)];
                    }
                    if p_col > 0 {
                        total += scratch
                            [composite_scratch_index(sample, pixel - 1, 6, coalesced_scratch)];
                    }
                    if p_col < 2 {
                        total -= scratch
                            [composite_scratch_index(sample, pixel + 1, 6, coalesced_scratch)];
                    }
                    if p_col == 2 {
                        total +=
                            scratch[composite_scratch_index(sample, pixel, 6, coalesced_scratch)];
                    }
                    if p_row == 0 {
                        total -=
                            scratch[composite_scratch_index(sample, pixel, 7, coalesced_scratch)];
                    }
                    if p_row > 0 {
                        total += scratch
                            [composite_scratch_index(sample, pixel - 3, 7, coalesced_scratch)];
                    }
                    if p_row < 2 {
                        total -= scratch
                            [composite_scratch_index(sample, pixel + 3, 7, coalesced_scratch)];
                    }
                    if p_row == 2 {
                        total +=
                            scratch[composite_scratch_index(sample, pixel, 7, coalesced_scratch)];
                    }
                }
            }
        }
    }
    ds[point] = total;
}

#[cube(launch)]
fn visibility_kernel(
    sdf: &Tensor<f32>,
    pairs: &Tensor<f32>,
    ranges: &Tensor<f32>,
    variance: &Tensor<f32>,
    out: &mut Tensor<f32>,
    #[comptime] reuse_endpoints: bool,
) {
    let ray = ABSOLUTE_POS;
    if ray >= ranges.shape(0) {
        terminate!();
    }
    let inv = f32::clamp(f32::exp(variance[0] * 10.0_f32), 1e-6_f32, 1e6_f32);
    let begin = usize::cast_from(ranges[ray * 2]);
    let end = usize::cast_from(ranges[ray * 2 + 1]);
    let mut trans = 1.0_f32;
    let mut count = 0usize;
    let mut ep_count = 0usize;
    let mut prev_end_id = 0usize;
    let mut prev_c1 = 0.0_f32;
    for sample in begin..end {
        if trans < 1e-8_f32 {
            break;
        }
        let i0 = usize::cast_from(pairs[sample * 2]);
        let i1 = usize::cast_from(pairs[sample * 2 + 1]);
        let reuse = reuse_endpoints && count > 0 && i0 == prev_end_id;
        if count > 0 && i0 == prev_end_id {
            ep_count += 1;
        } else {
            ep_count += 2;
        }
        prev_end_id = i1;
        count += 1;
        let a = sdf[i0];
        let b = sdf[i1];
        // sigmoid(a*inv), written with the same sign convention as CPU alpha.
        let mut c = prev_c1;
        if !reuse {
            c = 1.0_f32 / (1.0_f32 + f32::exp(-a * inv));
        }
        let d = 1.0_f32 / (1.0_f32 + f32::exp(-b * inv));
        prev_c1 = d;
        let alpha = f32::clamp((c - d + 1e-5_f32) / (c + 1e-5_f32), 0.0_f32, 1.0_f32);
        trans *= f32::max(1.0_f32 - alpha, 1e-10_f32);
    }
    if out.shape(1) == 2 {
        out[ray * 2] = f32::cast_from(count);
        out[ray * 2 + 1] = f32::cast_from(ep_count);
        if ray == 0 {
            out[ranges.shape(0) * 2] = variance[0];
            out[ranges.shape(0) * 2 + 1] = variance[0];
        }
    } else {
        out[ray] = f32::cast_from(count);
        if ray == 0 {
            out[ranges.shape(0)] = variance[0];
        }
    }
}
#[derive(Debug)]
struct VisibilityOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for VisibilityOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([s, p, r, v, o], [out]) = self.desc.as_fixed();
        let [s, p, r, v, o] = [s, p, r, v, o].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let [ScalarIr::UInt(reuse_endpoints)] = self.desc.scalars.as_slice() else {
            panic!("invalid visibility scalars")
        };
        let client = s.client.clone();
        let _timing = crate::diagnostics::start(&client, "visibility");
        let n = r.meta.shape().dims::<2>()[0];
        let dim = CubeDim::new(&client, n);
        visibility_kernel::launch(
            &client,
            cubecl::calculate_cube_count_elemwise(&client, n, dim),
            dim,
            s.into_tensor_arg(),
            p.into_tensor_arg(),
            r.into_tensor_arg(),
            v.into_tensor_arg(),
            o.clone().into_tensor_arg(),
            *reuse_endpoints != 0,
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, o);
        Ok(())
    }
}
pub fn visibility(
    inputs: [burn::tensor::Tensor<2>; 5],
    reuse_endpoints: bool,
) -> burn::tensor::Tensor<2> {
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[4].shape.clone(),
        inputs[4].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_visibility",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![ScalarIr::UInt(reuse_endpoints as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            VisibilityOp { desc },
        )
        .outputs();
    burn::tensor::Tensor::<2>::from_primitive::<GpuBackend>(out)
}

// Native float add variant; selected only when the runtime advertises support.
#[cube(launch)]
fn field_backward_native_kernel(
    pre: &Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    dh: &mut Tensor<f32>,
    dw1: &mut Tensor<f32>,
    points: &Tensor<f32>,
    metadata: &Tensor<f32>,
    table_grad: &Tensor<Atomic<f32>>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    #[comptime] local_hash: bool,
    #[comptime] compact_gradients: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] cross_corners: bool,
    #[comptime] write_dh: bool,
) {
    let unit = usize::cast_from(UNIT_POS);
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let target = if local_hash && index < 1984 {
                (index % 64) * 31 + index / 64
            } else {
                index
            };
            shared_weights[target] = weights[index];
        }
    }
    sync_cube();
    let lane = unit % 64;
    let slot = unit / 64;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let point = group * 4 + slot;
    let n_valid = usize::cast_from(valid_n);
    let mut shared = Shared::new_slice(256usize);
    // Reused first for `output_grads` `[256]` and then for 224-thread partial
    // feature gradients `[448]`, keeping shared memory under 12 KB per block.
    let mut partials = Shared::new_slice(if corner_parallel {
        448usize
    } else if compact_gradients {
        256usize
    } else {
        1usize
    });
    let mut slot_ds = Shared::new_slice(4usize);
    let mut output_grad = 0.0_f32;
    let mut v = 0.0_f32;
    let mut ds_val = 0.0_f32;
    if point < n_valid {
        ds_val = ds[point];
        if ds_val != 0.0_f32 {
            let h = pre[point * 64 + lane];
            let scaled = h * 100.0_f32;
            let exp_pos = f32::exp(f32::min(scaled, 20.0_f32));
            let derivative = if scaled > 20.0_f32 {
                1.0_f32
            } else {
                exp_pos / (1.0_f32 + exp_pos)
            };
            v = ds_val * shared_weights[2048 + lane] * derivative;
            let soft = f32::ln(1.0_f32 + exp_pos) * 0.01_f32;
            let hidden = if scaled > 20.0_f32 { h } else { soft };
            output_grad = ds_val * hidden;
        }
        if write_dh {
            dh[point * 64 + lane] = v;
        }
        if !compact_gradients {
            dw1[point * 64 + lane] = output_grad;
        }
    }
    if lane == 0 {
        slot_ds[slot] = ds_val;
    }
    let shared_index = if local_hash { lane * 4 + slot } else { unit };
    shared[shared_index] = v;
    if compact_gradients {
        partials[unit] = output_grad;
    }
    sync_cube();
    if compact_gradients && slot == 0 {
        if group < dw1.shape(0) {
            dw1[group * 64 + lane] = (partials[lane] + partials[64 + lane])
                + (partials[128 + lane] + partials[192 + lane]);
        }
    }
    if slot_ds[0] == 0.0_f32
        && slot_ds[1] == 0.0_f32
        && slot_ds[2] == 0.0_f32
        && slot_ds[3] == 0.0_f32
    {
        terminate!();
    }
    if corner_parallel {
        if compact_gradients {
            sync_cube();
        }
        // Map 224 threads across (hash_slot in 0..4, sub_part in 0..4, level in 0..14).
        // Phase 1: Each thread accumulates 16 of the 64 hidden neurons (4x faster than
        // 56 threads looping 64 times) and writes `[p_g0, p_g1]` to `partials`.
        let hash_slot = unit & 3usize;
        let sub_part = (unit >> 2) & 3usize;
        let level = unit >> 4;
        let hash_point = group * 4 + hash_slot;
        if level < 14usize {
            let mut p_g0 = 0.0_f32;
            let mut p_g1 = 0.0_f32;
            if hash_point < n_valid
                && level < usize::cast_from(active)
                && slot_ds[hash_slot] != 0.0_f32
            {
                let col = 3 + level * 2;
                let j_base = sub_part * 16;
                for j_off in 0..16usize {
                    let j = j_base + j_off;
                    let hi = if local_hash {
                        j * 4 + hash_slot
                    } else {
                        hash_slot * 64 + j
                    };
                    let w_idx0 = if local_hash {
                        j * 31 + col
                    } else {
                        col * 64 + j
                    };
                    let w_idx1 = if local_hash {
                        j * 31 + col + 1
                    } else {
                        (col + 1) * 64 + j
                    };
                    let h = shared[hi];
                    p_g0 += h * shared_weights[w_idx0];
                    p_g1 += h * shared_weights[w_idx1];
                }
            }
            partials[unit * 2] = p_g0;
            partials[unit * 2 + 1] = p_g1;
        }
        sync_cube();
        // Phase 2: All 224 threads sum the 4 partial dot products for `(level, hash_slot)`
        // and scatter their 2 trilinear corners concurrently (448 lookups/atomics per block).
        let mut scatter_keys = Array::<f32>::new(2usize);
        let mut scatter_g0 = Array::<f32>::new(2usize);
        let mut scatter_g1 = Array::<f32>::new(2usize);
        for i in 0..2usize {
            scatter_keys[i] = -1.0_f32;
            scatter_g0[i] = 0.0_f32;
            scatter_g1[i] = 0.0_f32;
        }
        if hash_point < n_valid && level < usize::cast_from(active) && slot_ds[hash_slot] != 0.0_f32
        {
            let base_p = (level * 16 + hash_slot) * 2;
            let g0 = (partials[base_p] + partials[base_p + 8])
                + (partials[base_p + 16] + partials[base_p + 24]);
            let g1 = (partials[base_p + 1] + partials[base_p + 9])
                + (partials[base_p + 17] + partials[base_p + 25]);
            if g0 != 0.0_f32 || g1 != 0.0_f32 {
                let norm0 = f32::clamp(
                    (points[hash_point * 3] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (points[hash_point * 3 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (points[hash_point * 3 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res = u32::cast_from(metadata[level * 2]);
                let offset = u32::cast_from(metadata[level * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;
                for c_sub in 0..2u32 {
                    let corner = u32::cast_from(sub_part) * 2 + c_sub;
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1.0_f32 - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1.0_f32 - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1.0_f32 - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    let wg0 = weight * g0;
                    let wg1 = weight * g1;
                    if aggregate_hash && level < usize::cast_from(aggregation_levels) {
                        let c = usize::cast_from(c_sub);
                        scatter_keys[c] = f32::cast_from(idx);
                        scatter_g0[c] = wg0;
                        scatter_g1[c] = wg1;
                    } else {
                        if wg0 != 0.0_f32 {
                            table_grad[entry].fetch_add(wg0);
                        }
                        if wg1 != 0.0_f32 {
                            table_grad[entry + 1].fetch_add(wg1);
                        }
                    }
                }
            }
        }
        if aggregate_hash {
            // Repurpose the weight tile after every lane finishes reading it.
            // Keys are level-local integer addresses, exactly representable in
            // f32. This adds no shared-memory allocation.
            sync_cube();
            if unit < 224usize {
                for c in 0..2usize {
                    let staging = c * 224 + unit;
                    shared_weights[staging] = scatter_keys[c];
                    shared_weights[448 + staging] = scatter_g0[c];
                    shared_weights[896 + staging] = scatter_g1[c];
                }
            }
            sync_cube();
            if unit < 224usize
                && level < usize::cast_from(active)
                && level < usize::cast_from(aggregation_levels)
            {
                let peers = unit - hash_slot;
                let offset = usize::cast_from(metadata[level * 2 + 1]);
                for c in 0..2usize {
                    let key = scatter_keys[c];
                    let mut leader = key >= 0.0_f32;
                    let peer_count = if cross_corners { 32usize } else { 4usize };
                    let owner = if cross_corners {
                        c * 16 + sub_part * 4 + hash_slot
                    } else {
                        hash_slot
                    };
                    for other in 0..peer_count {
                        let staging = if cross_corners {
                            (other / 16) * 224 + level * 16 + other % 16
                        } else {
                            c * 224 + peers + other
                        };
                        if other < owner && shared_weights[staging] == key {
                            leader = false;
                        }
                    }
                    if leader {
                        let mut sum0 = 0.0_f32;
                        let mut sum1 = 0.0_f32;
                        for other in 0..peer_count {
                            let staging = if cross_corners {
                                (other / 16) * 224 + level * 16 + other % 16
                            } else {
                                c * 224 + peers + other
                            };
                            if shared_weights[staging] == key {
                                sum0 += shared_weights[448 + staging];
                                sum1 += shared_weights[896 + staging];
                            }
                        }
                        let entry = (offset + usize::cast_from(key)) * 2;
                        if sum0 != 0.0_f32 {
                            table_grad[entry].fetch_add(sum0);
                        }
                        if sum1 != 0.0_f32 {
                            table_grad[entry + 1].fetch_add(sum1);
                        }
                    }
                }
            }
        }
    } else {
        let hash_slot = if local_hash { unit % 4 } else { slot };
        let level = if local_hash { unit / 4 } else { lane };
        let hash_point = group * 4 + hash_slot;
        if hash_point < n_valid && level < usize::cast_from(active) && slot_ds[hash_slot] != 0.0_f32
        {
            let col = 3 + level * 2;
            let mut g0 = 0.0_f32;
            let mut g1 = 0.0_f32;
            for j in 0..64usize {
                let hi = if local_hash {
                    j * 4 + hash_slot
                } else {
                    hash_slot * 64 + j
                };
                let w0 = if local_hash {
                    j * 31 + col
                } else {
                    col * 64 + j
                };
                let w1 = if local_hash {
                    j * 31 + col + 1
                } else {
                    (col + 1) * 64 + j
                };
                let h = shared[hi];
                g0 += h * shared_weights[w0];
                g1 += h * shared_weights[w1];
            }
            if g0 != 0.0_f32 || g1 != 0.0_f32 {
                let norm0 = f32::clamp(
                    (points[hash_point * 3] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (points[hash_point * 3 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (points[hash_point * 3 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res = u32::cast_from(metadata[usize::cast_from(level) * 2]);
                let offset = u32::cast_from(metadata[usize::cast_from(level) * 2 + 1]);
                let c0 = norm0 * f32::cast_from(res);
                let c1 = norm1 * f32::cast_from(res);
                let c2 = norm2 * f32::cast_from(res);
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                let w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                let base0 = u32::cast_from(b0);
                let base1 = u32::cast_from(b1);
                let base2 = u32::cast_from(b2);
                let r1 = res + 1;
                let dense = r1 * r1 * r1 <= 524288;

                for corner in 0..8u32 {
                    let bit0 = corner & 1;
                    let bit1 = (corner >> 1) & 1;
                    let bit2 = (corner >> 2) & 1;
                    let v0 = base0 + bit0;
                    let v1 = base1 + bit1;
                    let v2 = base2 + bit2;
                    let kw0 = if bit0 == 0 { 1.0_f32 - w0 } else { w0 };
                    let kw1 = if bit1 == 0 { 1.0_f32 - w1 } else { w1 };
                    let kw2 = if bit2 == 0 { 1.0_f32 - w2 } else { w2 };
                    let weight = kw0 * kw1 * kw2;
                    let idx = if dense {
                        v0 + v1 * r1 + v2 * r1 * r1
                    } else {
                        (v0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
                    };
                    let entry = usize::cast_from(offset + idx) * 2;
                    let wg0 = weight * g0;
                    let wg1 = weight * g1;
                    if wg0 != 0.0_f32 {
                        table_grad[entry].fetch_add(wg0);
                    }
                    if wg1 != 0.0_f32 {
                        table_grad[entry + 1].fetch_add(wg1);
                    }
                }
            }
        }
    }
}

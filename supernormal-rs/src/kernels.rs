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
    #[comptime] skip_untouched: bool,
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
    let g = gradient[index];
    let old_m = first[index];
    let old_v = second[index];
    // A zero gradient alone is not enough: historical moments must still
    // decay and move the parameter. Only an entirely zero Adam state is idle.
    if skip_untouched && g == 0.0_f32 && old_m == 0.0_f32 && old_v == 0.0_f32 {
        terminate!();
    }
    if !skip_untouched || g != 0.0_f32 {
        gradient[index] = 0.0_f32;
    }
    let step_scale = corrections[level * 3 + 1];
    let inv_bc2_sqrt = corrections[level * 3 + 2];
    let m = old_m * 0.9 + g * 0.1;
    let v = old_v * 0.999 + g * g * 0.001;
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
            ScalarIr::UInt(skip_untouched),
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
            *skip_untouched != 0,
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
    skip_untouched: bool,
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
            ScalarIr::UInt(skip_untouched as u64),
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

/// SNSR (Scale-encoded Neural Surface Representation) Butterworth low-pass
/// anti-aliasing weight for hash-grid level `res` given physical pixel footprint `scale`:
/// `u = 0.5 * cutoff * scale * res`. Flat passband (~1.0) when `scale <= cell_size`,
/// smooth roll-off to 0 for sub-pixel / grazing / far-view levels.
#[cube]
fn scale_level_weight(scale: f32, res: u32, cutoff: f32) -> f32 {
    let u = 0.5_f32 * cutoff * scale * f32::cast_from(res);
    let u2 = u * u;
    1.0_f32 / (1.0_f32 + u2 * u2)
}

// Identical interpolation and reduction order for forward and backward replay.
// All callers use a complete 256-thread group and uniform barriers.
#[cube]
fn field_hash_tile(
    points: &Tensor<f32>,
    tables: &Tensor<f32>,
    metadata: &Tensor<f32>,
    shared_features: &mut Shared<[f32]>,
    partials: &mut Shared<[f32]>,
    unit: usize,
    group: usize,
    n_valid: usize,
    active: u32,
    scale_cutoff: f32,
    #[comptime] local_hash: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] subgroup_width: usize,
    #[comptime] scale_encoding: bool,
) {
    let lane = unit % 64;
    let slot = unit / 64;
    let point_idx = group * 4 + slot;
    let valid = point_idx < n_valid;
    let active_cols = 3usize + usize::cast_from(active) * 2usize;
    let p_stride = if scale_encoding { 4usize } else { 3usize };
    // Disjoint writes to xyz [0..3] and inactive tail [active_cols..32] require
    // no pre-hash barrier: hash threads only write [3..active_cols].
    if lane < 3usize {
        shared_features[field_feature_index(slot, lane, local_hash)] = if valid {
            points[point_idx * p_stride + lane]
        } else {
            0.0_f32
        };
    } else if lane < 32usize && lane >= active_cols {
        shared_features[field_feature_index(slot, lane, local_hash)] = 0.0_f32;
    }
    if corner_parallel {
        // Map 256 threads to (hash_slot in 0..4, corner_pair in 0..4, level in 0..16).
        // Up to 4 * 4 * 14 = 224 threads concurrently fetch 2 corners each (448 lookups
        // in flight per workgroup), then reduce the 4 corner-pairs via warp shuffles
        // (zero shared traffic, zero barrier) or shared partials fallback.
        let hash_slot = unit & 3usize;
        let corner_pair = (unit >> 2) & 3usize;
        let level = unit >> 4;
        let hash_point = group * 4 + hash_slot;
        let mut feat0 = 0.0_f32;
        let mut feat1 = 0.0_f32;
        if hash_point < n_valid && level < usize::cast_from(active) {
            let p0 = points[hash_point * p_stride];
            let p1 = points[hash_point * p_stride + 1];
            let p2 = points[hash_point * p_stride + 2];
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
            let bit1 = u32::cast_from(corner_pair) & 1u32;
            let bit2 = (u32::cast_from(corner_pair) >> 1u32) & 1u32;
            let v1 = base1 + bit1;
            let v2 = base2 + bit2;
            let kw1 = if bit1 == 0u32 { 1.0_f32 - w1 } else { w1 };
            let kw2 = if bit2 == 0u32 { 1.0_f32 - w2 } else { w2 };
            let mut kw12 = kw1 * kw2;
            if scale_encoding {
                kw12 *= scale_level_weight(points[hash_point * 4 + 3], res, scale_cutoff);
            }
            let weight0 = (1.0_f32 - w0) * kw12;
            let weight1 = w0 * kw12;
            let idx0 = if dense {
                base0 + v1 * r1 + v2 * r1 * r1
            } else {
                (base0 ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
            };
            let idx1 = if dense {
                idx0 + 1u32
            } else {
                ((base0 + 1u32) ^ (v1 * 2654435761u32) ^ (v2 * 805459861u32)) & 524287u32
            };
            let entry0 = usize::cast_from(offset + idx0) * 2;
            let entry1 = usize::cast_from(offset + idx1) * 2;
            feat0 = weight0 * tables[entry0] + weight1 * tables[entry1];
            feat1 = weight0 * tables[entry0 + 1] + weight1 * tables[entry1 + 1];
        }
        if subgroup_width >= 32usize {
            // Uniform across all lanes in every plane: unit = level*16 + corner_pair*4 + hash_slot.
            // XOR 4 pairs (cp 0<->1, cp 2<->3); XOR 8 combines (0+1)<->(2+3), preserving exact order.
            let s0 = feat0 + plane_shuffle_xor(feat0, 4u32);
            let sum0 = s0 + plane_shuffle_xor(s0, 8u32);
            let s1 = feat1 + plane_shuffle_xor(feat1, 4u32);
            let sum1 = s1 + plane_shuffle_xor(s1, 8u32);
            if corner_pair == 0usize && level < usize::cast_from(active) {
                shared_features[field_feature_index(hash_slot, 3 + level * 2, local_hash)] = sum0;
                shared_features[field_feature_index(hash_slot, 4 + level * 2, local_hash)] = sum1;
            }
        } else {
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
        }
    } else {
        // Legacy 56-thread hash evaluation: one thread loops over all 8 corners.
        let hash_slot = if local_hash { unit % 4 } else { slot };
        let level = if local_hash { unit / 4 } else { lane };
        let hash_point = group * 4 + hash_slot;
        if hash_point < n_valid && level < usize::cast_from(active) {
            let p0 = points[hash_point * p_stride];
            let p1 = points[hash_point * p_stride + 1];
            let p2 = points[hash_point * p_stride + 2];
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
            if scale_encoding {
                let sw = scale_level_weight(points[hash_point * 4 + 3], res, scale_cutoff);
                feat0 *= sw;
                feat1 *= sw;
            }
            shared_features[field_feature_index(hash_slot, 3 + level * 2, local_hash)] = feat0;
            shared_features[field_feature_index(hash_slot, 4 + level * 2, local_hash)] = feat1;
        }
    }
    sync_cube();
}

// Occupancy cell IDs are unique; each selected cell has exactly one writer.
#[cube]
fn field_store_result(
    output: &mut Tensor<f32>,
    selected_ids: &Tensor<f32>,
    point: usize,
    sdf: f32,
    #[comptime] occupancy: bool,
) {
    if occupancy {
        let cell = usize::cast_from(selected_ids[point]);
        let density = 1.0_f32 / (1.0_f32 + f32::exp(80.0_f32 * sdf));
        output[cell] = f32::max(output[cell] * 0.95_f32, density);
    } else {
        output[point] = sdf;
    }
}

#[cube(launch)]
fn fused_field_kernel(
    points: &Tensor<f32>,
    tables: &Tensor<f32>,
    metadata: &Tensor<f32>,
    weights: &Tensor<f32>,
    count_prefix: &Tensor<f32>,
    selected_ids: &Tensor<f32>,
    output: &mut Tensor<f32>,
    features: &mut Tensor<f32>,
    preactivation: &mut Tensor<f32>,
    active: u32,
    valid_n: u32,
    scale_cutoff: f32,
    #[comptime] cache: bool,
    #[comptime] cache_preactivation: bool,
    #[comptime] local_hash: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] point_tiles: usize,
    #[comptime] subgroup_width: usize,
    #[comptime] selective_weights: bool,
    #[comptime] occupancy: bool,
    #[comptime] scale_encoding: bool,
) {
    // CubeCL workgroup coordinates are u32; tensor/shared indices are usize.
    let mut unit = usize::cast_from(UNIT_POS);
    if subgroup_width > 0 {
        // Define logical neuron ownership using actual subgroup IDs; do not
        // assume the backend maps local invocation IDs to subgroups linearly.
        unit = usize::cast_from(PLANE_POS) * subgroup_width + usize::cast_from(UNIT_POS_PLANE);
    }
    let group_base = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * point_tiles;
    let mut n_valid = usize::cast_from(valid_n);
    if occupancy {
        n_valid = usize::cast_from(count_prefix[count_prefix.shape(0) - 1]);
    }
    // Uniform workgroup exit before any barrier, weights or point reads.
    // Occupancy uses a bounded grid; the actual workload remains device-resident.
    if group_base * 4 >= n_valid {
        terminate!();
    }
    let mut shared_weights = Shared::new_slice(2113usize);
    let mut shared_features = Shared::new_slice(128usize);
    let mut partials = Shared::new_slice(if corner_parallel && subgroup_width >= 32usize {
        if subgroup_width == 64usize {
            1usize
        } else {
            8usize
        }
    } else if corner_parallel {
        512usize
    } else {
        256usize
    });
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            if !selective_weights || index >= 1984 || index / 64 < 3 + usize::cast_from(active) * 2
            {
                shared_weights[index] = weights[index];
            }
        }
    }
    sync_cube();
    let lane = unit % 64;
    let slot = unit / 64;
    // Occupancy uses a bounded persistent grid: stage weights once, then consume
    // as many compacted tiles as the device count requires. No capacity-sized
    // field dispatch and no CPU count readback are necessary.
    let group_stride =
        usize::cast_from(CUBE_COUNT_X) * usize::cast_from(CUBE_COUNT_Y) * point_tiles;
    let mut wave_base = group_base;
    let active_cols = 3usize + usize::cast_from(active) * 2usize;
    let fast_tile_sync = corner_parallel && subgroup_width == 32usize;
    while wave_base * 4 < n_valid {
        // Reuse the staged weights for several bounded four-point tiles.
        for point_tile in 0..point_tiles {
            let group = wave_base + point_tile;
            let point_idx = group * 4 + slot;
            let valid = point_idx < n_valid;
            field_hash_tile(
                points,
                tables,
                metadata,
                &mut shared_features,
                &mut partials,
                unit,
                group,
                n_valid,
                active,
                scale_cutoff,
                local_hash,
                corner_parallel,
                subgroup_width,
                scale_encoding,
            );
            let mut contribution = 0.0_f32;
            if valid {
                let mut h = 0.0_f32;
                // Packed input-major weights give adjacent lanes adjacent reads.
                for k in 0..active_cols {
                    h += shared_features[field_feature_index(slot, k, local_hash)]
                        * shared_weights[k * 64 + lane];
                }
                h += shared_weights[1984 + lane];
                if cache_preactivation {
                    preactivation[point_idx * 64 + lane] = h;
                }
                if cache {
                    let f_stride = if scale_encoding { 32usize } else { 31usize };
                    if lane < 31 {
                        features[point_idx * f_stride + lane] =
                            shared_features[field_feature_index(slot, lane, local_hash)];
                    } else if scale_encoding && lane == 31 {
                        features[point_idx * 32 + 31] = points[point_idx * 4 + 3];
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
                        field_store_result(
                            output,
                            selected_ids,
                            point_idx,
                            sum + shared_weights[2112],
                            occupancy,
                        );
                    }
                } else if fast_tile_sync {
                    if lane == 0usize || lane == 32usize {
                        partials[slot * 2 + (lane >> 5)] = sum;
                    }
                    // This single barrier publishes partials AND ensures all lanes
                    // finished reading shared_features before the next tile writes it.
                    sync_cube();
                    if valid && lane == 0usize {
                        field_store_result(
                            output,
                            selected_ids,
                            point_idx,
                            partials[slot * 2] + partials[slot * 2 + 1] + shared_weights[2112],
                            occupancy,
                        );
                    }
                } else {
                    partials[unit] = sum;
                    sync_cube();
                    if valid && lane == 0 {
                        field_store_result(
                            output,
                            selected_ids,
                            point_idx,
                            partials[slot * 64] + partials[slot * 64 + 32] + shared_weights[2112],
                            occupancy,
                        );
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
                    field_store_result(
                        output,
                        selected_ids,
                        point_idx,
                        partials[slot * 64] + shared_weights[2112],
                        occupancy,
                    );
                }
            }
            // The next hash tile reuses partials/shared_features unless fast_tile_sync
            // already synchronized after MLP reads and does not touch partials in hash.
            if point_tiles > 1 && !fast_tile_sync {
                sync_cube();
            }
        }
        if occupancy {
            if !fast_tile_sync {
                sync_cube();
            }
            wave_base += group_stride;
        } else {
            break;
        }
    }
}

// Pack already-normalized Burn weights. This kernel performs only copies;
// weight normalization and its floating-point reductions stay on the reference
// path. Padding the shared transpose tile avoids a 32-float row pitch.
#[cube(launch)]
fn pack_mlp_weights_kernel(
    w0: &Tensor<f32>,
    b0: &Tensor<f32>,
    w1: &Tensor<f32>,
    b1: &Tensor<f32>,
    packed: &mut Tensor<f32>,
) {
    let unit = usize::cast_from(UNIT_POS);
    let row_base = usize::cast_from(CUBE_POS_X) * 32;
    let mut tile = Shared::new_slice(1056usize);
    for load in 0..4usize {
        let index = unit + load * 256;
        let row = index / 32;
        let col = index % 32;
        if col < 31 {
            tile[row * 33 + col] = w0[(row_base + row) * w0.stride(0) + col * w0.stride(1)];
        }
    }
    sync_cube();
    for store in 0..4usize {
        let index = unit + store * 256;
        let row = index % 32;
        let col = index / 32;
        if col < 31 {
            packed[col * 64 + row_base + row] = tile[row * 33 + col];
        }
    }
    if unit < 32 {
        packed[1984 + row_base + unit] = b0[(row_base + unit) * b0.stride(1)];
    }
    if row_base == 0 {
        if unit < 64 {
            packed[2048 + unit] = w1[unit * w1.stride(1)];
        }
        if unit == 0 {
            packed[2112] = b1[0];
        }
    }
}

#[derive(Debug)]
struct PackMlpWeightsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for PackMlpWeightsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([w0, b0, w1, b1, packed], [out]) = self.desc.as_fixed();
        let [w0, b0, w1, b1, packed] =
            [w0, b0, w1, b1, packed].map(|t| handles.get_float_tensor::<CubeBackend>(t));
        let client = packed.client.clone();
        let _timing = crate::diagnostics::start(&client, "pack_mlp_weights");
        pack_mlp_weights_kernel::launch(
            &client,
            CubeCount::Static(2, 1, 1),
            CubeDim::new_1d(256),
            w0.into_tensor_arg(),
            b0.into_tensor_arg(),
            w1.into_tensor_arg(),
            b1.into_tensor_arg(),
            packed.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, packed);
        Ok(())
    }
}

pub fn pack_mlp_weights(inputs: [burn::tensor::Tensor<2>; 4]) -> burn::tensor::Tensor<2> {
    use burn::tensor::Tensor as BT;
    assert_eq!(inputs[0].dims(), [64, 31]);
    assert_eq!(inputs[1].dims(), [1, 64]);
    assert_eq!(inputs[2].dims(), [1, 64]);
    assert_eq!(inputs[3].dims(), [1, 1]);
    let device = inputs[0].device();
    let inputs: Vec<_> = inputs
        .into_iter()
        .chain([BT::empty([1, 2113], &device)])
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap())
        .collect();
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[4].shape.clone(),
        inputs[4].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_pack_mlp_weights",
        &inputs.into_iter().map(|t| t.into_ir()).collect::<Vec<_>>(),
        &[out],
        vec![],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            PackMlpWeightsOp { desc },
        )
        .outputs();
    BT::<2>::from_primitive::<GpuBackend>(out)
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
        let ([p, t, m, w, c, i, y, f, h], [oy, of, oh]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(active),
            ScalarIr::UInt(valid_n),
            ScalarIr::UInt(cache),
            ScalarIr::UInt(cache_preactivation),
            ScalarIr::UInt(local_hash),
            ScalarIr::UInt(corner_parallel),
            ScalarIr::UInt(point_tiles),
            ScalarIr::UInt(subgroup_reduce),
            ScalarIr::UInt(selective_weights),
            ScalarIr::UInt(occupancy),
            ScalarIr::UInt(scale_encoding),
            ScalarIr::Float(scale_cutoff),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid fused field arguments")
        };
        let [
            points,
            tables,
            metadata,
            weights,
            prefix,
            ids,
            output,
            features,
            hidden,
        ] = [p, t, m, w, c, i, y, f, h].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let output = if *occupancy != 0 && !output.can_mut() {
            output.copy()
        } else {
            output
        };
        let client = points.client.clone();
        let _timing = crate::diagnostics::start(
            &client,
            if *occupancy != 0 {
                "occupancy.fused_field_apply"
            } else {
                "field_forward"
            },
        );
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
        let mut groups = (*valid_n as usize).div_ceil(4 * point_tiles).max(1);
        if *occupancy != 0 {
            groups = groups.min(4096);
        }
        let gx = groups.min(65535).max(1);
        fused_field_kernel::launch(
            &client,
            CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
            CubeDim::new_1d(256),
            points.into_tensor_arg(),
            tables.into_tensor_arg(),
            metadata.into_tensor_arg(),
            weights.into_tensor_arg(),
            prefix.into_tensor_arg(),
            ids.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            features.clone().into_tensor_arg(),
            hidden.clone().into_tensor_arg(),
            *active as u32,
            *valid_n as u32,
            *scale_cutoff as f32,
            *cache != 0,
            *cache_preactivation != 0,
            *local_hash != 0,
            *corner_parallel != 0,
            point_tiles,
            subgroup_width,
            *selective_weights != 0,
            *occupancy != 0,
            *scale_encoding != 0,
        );
        handles.register_float_tensor::<CubeBackend>(&oy.id, output);
        handles.register_float_tensor::<CubeBackend>(&of.id, features);
        handles.register_float_tensor::<CubeBackend>(&oh.id, hidden);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fused_field(
    inputs: [burn::tensor::Tensor<2>; 4],
    valid_n: usize,
    active: usize,
    cache: bool,
    cache_preactivation: bool,
    local_hash: bool,
    corner_parallel: bool,
    point_tiles: usize,
    subgroup_reduce: bool,
    selective_weights: bool,
    scale_cutoff: f32,
) -> [burn::tensor::Tensor<2>; 3] {
    let dims = inputs[0].dims();
    let n = dims[0];
    let scale_encoding = dims[1] == 4;
    let feat_cols = if scale_encoding { 32 } else { 31 };
    let device = inputs[0].device();
    // `fused_field_kernel` zeroes all 32 `shared_features` slots and writes all
    // columns of `features` for every valid point, so `Tensor::empty` is safe
    // regardless of `active`.
    let buffers = [
        burn::tensor::Tensor::<2>::empty([valid_n.max(1), 1], &device),
        burn::tensor::Tensor::<2>::empty([if cache { n } else { 1 }, feat_cols], &device),
        burn::tensor::Tensor::<2>::empty([if cache_preactivation { n } else { 1 }, 64], &device),
    ];
    let [points, tables, metadata, weights] = inputs;
    let dummy = points.clone();
    fused_field_op(
        [points, tables, metadata, weights, dummy.clone(), dummy],
        buffers,
        valid_n,
        active,
        cache,
        cache_preactivation,
        local_hash,
        corner_parallel,
        point_tiles,
        subgroup_reduce,
        selective_weights,
        false,
        scale_encoding,
        scale_cutoff,
    )
}

/// Capacity-bounded occupancy evaluation + EMA, without an SDF buffer or count readback.
/// Inputs: compact points, hash table, metadata, MLP, count prefix, unique cell IDs, values.
pub fn fused_occupancy_field(
    inputs: [burn::tensor::Tensor<2>; 7],
    active: usize,
    local_hash: bool,
    corner_parallel: bool,
    point_tiles: usize,
    subgroup_reduce: bool,
    selective_weights: bool,
) -> burn::tensor::Tensor<2> {
    use burn::tensor::Tensor as BT;
    let [points, tables, metadata, weights, prefix, ids, values] = inputs;
    let cap = points.dims()[0];
    assert!(cap == values.dims()[0] && ids.dims()[0] >= cap);
    assert!(prefix.dims()[0] > 0 && prefix.dims()[1] == 1);
    let device = points.device();
    let [values, _, _] = fused_field_op(
        [points, tables, metadata, weights, prefix, ids],
        [
            values,
            BT::empty([1, 31], &device),
            BT::empty([1, 64], &device),
        ],
        cap,
        active,
        false,
        false,
        local_hash,
        corner_parallel,
        point_tiles,
        subgroup_reduce,
        selective_weights,
        true,
        false,
        0.0,
    );
    values
}

#[allow(clippy::too_many_arguments)]
fn fused_field_op(
    inputs: [burn::tensor::Tensor<2>; 6],
    buffers: [burn::tensor::Tensor<2>; 3],
    valid_n: usize,
    active: usize,
    cache: bool,
    cache_preactivation: bool,
    local_hash: bool,
    corner_parallel: bool,
    point_tiles: usize,
    subgroup_reduce: bool,
    selective_weights: bool,
    occupancy: bool,
    scale_encoding: bool,
    scale_cutoff: f32,
) -> [burn::tensor::Tensor<2>; 3] {
    let inputs: Vec<_> = inputs
        .into_iter()
        .chain(buffers)
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap())
        .collect();
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = inputs[6..]
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
            ScalarIr::UInt(cache_preactivation as u64),
            ScalarIr::UInt(local_hash as u64),
            ScalarIr::UInt(corner_parallel as u64),
            ScalarIr::UInt(point_tiles as u64),
            ScalarIr::UInt(subgroup_reduce as u64),
            ScalarIr::UInt(selective_weights as u64),
            ScalarIr::UInt(occupancy as u64),
            ScalarIr::UInt(scale_encoding as u64),
            ScalarIr::Float(scale_cutoff as f64),
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
// 3. Forward retains sigmoid endpoints, avoiding exp() in reverse. Compact
//    records reload immutable s0/s1 instead of storing redundant SDF values.
//    Seven forward components become four derivatives after local reads finish.
// Component-major within a nine-pixel sample: adjacent pixel lanes read/write
// adjacent words. The legacy ten-float cache and pixel layout are independent
// comparison switches.
#[cube]
fn composite_scratch_index(
    sample: usize,
    pixel: usize,
    component: usize,
    #[comptime] coalesced: bool,
    #[comptime] compact_cache: bool,
) -> usize {
    if coalesced {
        if compact_cache {
            sample * 63 + component * 9 + pixel
        } else {
            sample * 90 + component * 9 + pixel
        }
    } else {
        if compact_cache {
            (sample * 9 + pixel) * 7 + component
        } else {
            (sample * 9 + pixel) * 10 + component
        }
    }
}

// Forward inputs are dead before reverse writes these four derivatives.
// Legacy records keep their original dp0, dp1, dp2, z0, z1 slots.
#[cube]
fn composite_adjoint_index(
    sample: usize,
    pixel: usize,
    component: usize,
    #[comptime] coalesced: bool,
    #[comptime] compact_cache: bool,
) -> usize {
    if compact_cache {
        composite_scratch_index(sample, pixel, component, coalesced, compact_cache)
    } else {
        composite_scratch_index(sample, pixel, component + 5, coalesced, compact_cache)
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
    #[comptime] compact_cache: bool,
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

    let vb = pb + 6 + pixel * 13;
    let mask = patches[vb + 12];
    if begin >= end {
        let mut nl = 0.0_f32;
        if mask != 0.0_f32 {
            let e0 = -patches[vb + 9] * mask;
            let e1 = -patches[vb + 10] * mask;
            let e2 = -patches[vb + 11] * mask;
            nl = (e0 * e0 + e1 * e1 + e2 * e2) / normal_den;
        }
        let m = 1e-5_f32;
        let ml = -(mask * f32::ln(m) + (1.0_f32 - mask) * f32::ln(1.0_f32 - m)) / pixel_den;
        stats[ray * 5] = nl;
        stats[ray * 5 + 1] = ml;
        stats[ray * 5 + 2] = 0.0_f32;
        stats[ray * 5 + 3] = 0.0_f32;
        stats[ray * 5 + 4] = 0.0_f32;
        terminate!();
    }

    // Hoist loop-invariant 3x3 inverse basis matrix into registers once per ray
    // instead of reloading 9 floats from global memory on every sample in both loops.
    let m00 = patches[vb];
    let m01 = patches[vb + 1];
    let m02 = patches[vb + 2];
    let m10 = patches[vb + 3];
    let m11 = patches[vb + 4];
    let m12 = patches[vb + 5];
    let m20 = patches[vb + 6];
    let m21 = patches[vb + 7];
    let m22 = patches[vb + 8];

    let mut normals0 = 0.0_f32;
    let mut normals1 = 0.0_f32;
    let mut normals2 = 0.0_f32;
    let mut masks = 0.0_f32;
    let mut trans = 1.0_f32;
    let mut dm = 0.0_f32;
    let mut future = 0.0_f32;

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
        scratch[composite_scratch_index(s, pixel, 3, coalesced_scratch, compact_cache)] = trans;
        scratch[composite_scratch_index(s, pixel, 4, coalesced_scratch, compact_cache)] = a;
        scratch[composite_scratch_index(s, pixel, 5, coalesced_scratch, compact_cache)] = c0;
        scratch[composite_scratch_index(s, pixel, 6, coalesced_scratch, compact_cache)] = c1;
        if !compact_cache {
            scratch[composite_scratch_index(s, pixel, 7, coalesced_scratch, compact_cache)] = s0;
            scratch[composite_scratch_index(s, pixel, 8, coalesced_scratch, compact_cache)] = s1;
        }
        trans *= f32::max(1.0_f32 - a, 1e-10_f32);
        masks += w;
        let g0 = m00 * dt + m01 * dx + m02 * dy;
        let g1 = m10 * dt + m11 * dx + m12 * dy;
        let g2 = m20 * dt + m21 * dx + m22 * dy;
        scratch[composite_scratch_index(s, pixel, 0, coalesced_scratch, compact_cache)] = g0;
        scratch[composite_scratch_index(s, pixel, 1, coalesced_scratch, compact_cache)] = g1;
        scratch[composite_scratch_index(s, pixel, 2, coalesced_scratch, compact_cache)] = g2;
        normals0 += w * g0;
        normals1 += w * g1;
        normals2 += w * g2;
    }

    let mut el = 0.0_f32;
    let mut dinv = 0.0_f32;
    let mut bias = 0.0_f32;
    let err0 = (normals0 - patches[vb + 9]) * mask;
    let err1 = (normals1 - patches[vb + 10]) * mask;
    let err2 = (normals2 - patches[vb + 11]) * mask;
    let nl = (err0 * err0 + err1 * err1 + err2 * err2) / normal_den;
    let dn0 = 2.0_f32 * err0 * mask / normal_den;
    let dn1 = 2.0_f32 * err1 * mask / normal_den;
    let dn2 = 2.0_f32 * err2 * mask / normal_den;
    let m = f32::clamp(masks, 1e-5_f32, 1.0_f32 - 1e-5_f32);
    let ml = -(mask * f32::ln(m) + (1.0_f32 - mask) * f32::ln(1.0_f32 - m)) / pixel_den;
    if masks >= 1e-5_f32 && masks <= 1.0_f32 - 1e-5_f32 {
        dm = (m - mask) / (m * (1.0_f32 - m) * pixel_den);
    }

    for rev in 0..(end - begin) {
        let s = end - 1 - rev;
        let g0 = scratch[composite_scratch_index(s, pixel, 0, coalesced_scratch, compact_cache)];
        let g1 = scratch[composite_scratch_index(s, pixel, 1, coalesced_scratch, compact_cache)];
        let g2 = scratch[composite_scratch_index(s, pixel, 2, coalesced_scratch, compact_cache)];
        let t = scratch[composite_scratch_index(s, pixel, 3, coalesced_scratch, compact_cache)];
        let a = scratch[composite_scratch_index(s, pixel, 4, coalesced_scratch, compact_cache)];
        let c0 = scratch[composite_scratch_index(s, pixel, 5, coalesced_scratch, compact_cache)];
        let c1 = scratch[composite_scratch_index(s, pixel, 6, coalesced_scratch, compact_cache)];
        let (s0, s1) = if compact_cache {
            // SDF values are immutable throughout forward/reverse/gather.
            // Reloading them avoids caching two extra floats per pixel/sample.
            if compact_ids {
                let base = usize::cast_from(ids[s]);
                (sdf[base + pixel], sdf[base + 9 + pixel])
            } else {
                (
                    sdf[usize::cast_from(ids[s * 18 + pixel])],
                    sdf[usize::cast_from(ids[s * 18 + 9 + pixel])],
                )
            }
        } else {
            (
                scratch[composite_scratch_index(s, pixel, 7, coalesced_scratch, compact_cache)],
                scratch[composite_scratch_index(s, pixel, 8, coalesced_scratch, compact_cache)],
            )
        };
        let norm = f32::sqrt(g0 * g0 + g1 * g1 + g2 * g2);
        el += (norm - 1.0_f32) * (norm - 1.0_f32) / sample_den;
        let eg = 2.0_f32 * (norm - 1.0_f32) / sample_den / f32::max(norm, 1e-12_f32);
        let dg0 = t * a * dn0 + eg * g0;
        let dg1 = t * a * dn1 + eg * g1;
        let dg2 = t * a * dn2 + eg * g2;
        let gb = s * 27 + pixel * 3;
        let dp0 = (m00 * dg0 + m10 * dg1 + m20 * dg2) * geometry[gb];
        let dp1 = (m01 * dg0 + m11 * dg1 + m21 * dg2) * geometry[gb + 1];
        let dp2 = (m02 * dg0 + m12 * dg1 + m22 * dg2) * geometry[gb + 2];
        if !compact_cache {
            scratch[composite_adjoint_index(s, pixel, 0, coalesced_scratch, compact_cache)] = dp0;
        }
        scratch[composite_adjoint_index(s, pixel, 1, coalesced_scratch, compact_cache)] = dp1;
        scratch[composite_adjoint_index(s, pixel, 2, coalesced_scratch, compact_cache)] = dp2;
        let gw = dn0 * g0 + dn1 * g1 + dn2 * g2 + dm;
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
        if compact_cache {
            // Precombine exactly the endpoint expressions gather would evaluate.
            scratch[composite_adjoint_index(s, pixel, 0, coalesced_scratch, compact_cache)] =
                -dp0 + z0 * inv;
            scratch[composite_adjoint_index(s, pixel, 3, coalesced_scratch, compact_cache)] =
                dp0 + z1 * inv;
        } else {
            scratch[composite_adjoint_index(s, pixel, 3, coalesced_scratch, compact_cache)] =
                z0 * inv;
            scratch[composite_adjoint_index(s, pixel, 4, coalesced_scratch, compact_cache)] =
                z1 * inv;
        }
        bias += z0 * inv + z1 * inv;
        dinv += z0 * s0 + z1 * s1;
    }
    stats[ray * 5] = nl;
    stats[ray * 5 + 1] = ml;
    stats[ray * 5 + 2] = el;
    stats[ray * 5 + 3] = dinv;
    stats[ray * 5 + 4] = bias;
}

#[cube(launch)]
fn reduce_composite_stats_kernel(
    raw_stats: &Tensor<f32>,
    out_stats: &mut Tensor<f32>,
    rays: u32,
    vscale: f32,
) {
    let col = usize::cast_from(CUBE_POS_X);
    let unit = usize::cast_from(UNIT_POS);
    let n = usize::cast_from(rays);
    let mut sum = 0.0_f32;
    let mut i = unit;
    while i < n {
        sum += raw_stats[i * 5 + col];
        i += 256usize;
    }
    let mut sh = Shared::new_slice(256usize);
    sh[unit] = sum;
    sync_cube();
    let mut stride = 128usize;
    for _ in 0..8usize {
        if unit < stride {
            sh[unit] += sh[unit + stride];
        }
        sync_cube();
        stride = stride >> 1usize;
    }
    if unit == 0usize {
        let mut total = sh[0];
        if col == 3usize {
            total = total * vscale;
        }
        out_stats[col] = total;
    }
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
        let ([s, g, i, p, vp, w, d, l, r, rl], [od, orl]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(valid_points),
            ScalarIr::UInt(compact_tables),
            ScalarIr::UInt(coalesced_scratch),
            ScalarIr::UInt(compact_cache),
            ScalarIr::UInt(reuse_endpoints),
            ScalarIr::Float(vscale),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid composite scalars")
        };
        let [s, g, i, p, vp, w, d, l, r, rl] =
            [s, g, i, p, vp, w, d, l, r, rl].map(|x| handles.get_float_tensor::<CubeBackend>(x));
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
            *compact_cache != 0,
            *reuse_endpoints != 0,
        );
        drop(timing);
        let timing_g = crate::diagnostics::start(&client, "composite_gather");
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
            *compact_cache != 0,
        );
        drop(timing_g);
        let _timing_r = crate::diagnostics::start(&client, "composite_stats_reduce");
        reduce_composite_stats_kernel::launch(
            &client,
            CubeCount::Static(5, 1, 1),
            CubeDim::new_1d(256),
            l.into_tensor_arg(),
            rl.clone().into_tensor_arg(),
            rays as u32,
            *vscale as f32,
        );
        handles.register_float_tensor::<CubeBackend>(&od.id, d);
        handles.register_float_tensor::<CubeBackend>(&orl.id, rl);
        Ok(())
    }
}
pub fn composite_gpu(
    inputs: [burn::tensor::Tensor<2>; 9],
    valid_points: usize,
    compact_tables: bool,
    coalesced_scratch: bool,
    compact_cache: bool,
    reuse_endpoints: bool,
    vscale: f32,
) -> [burn::tensor::Tensor<2>; 2] {
    assert_eq!(inputs[5].dims()[1], if compact_cache { 63 } else { 90 });
    let device = inputs[0].device();
    let inputs: Vec<_> = inputs
        .into_iter()
        .chain([burn::tensor::Tensor::<2>::empty([1, 5], &device)])
        .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap())
        .collect();
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [6usize, 9]
        .into_iter()
        .map(|i| &inputs[i])
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_composite",
        &inputs.into_iter().map(|t| t.into_ir()).collect::<Vec<_>>(),
        &outputs,
        vec![
            ScalarIr::UInt(valid_points as u64),
            ScalarIr::UInt(compact_tables as u64),
            ScalarIr::UInt(coalesced_scratch as u64),
            ScalarIr::UInt(compact_cache as u64),
            ScalarIr::UInt(reuse_endpoints as u64),
            ScalarIr::Float(vscale as f64),
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

// Shared memory is 2,821 floats plus the optional 128-float replay tile
// (11,796 bytes); hash replay adds 64 scratch floats (12,052 bytes < 12 KiB).
// Actual residency also depends on register allocation and device limits.
// Both the 64-neuron `dh * W0` feature-gradient dot product (16 iterations per
// thread across 224 threads) and the 8-corner trilinear hash scatter (2 corners
// per thread across 224 threads) execute across 87.5% of the workgroup.
#[cube(launch)]
fn field_backward_kernel(
    pre: &Tensor<f32>,
    features: &mut Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    dh: &mut Tensor<f32>,
    dw1: &mut Tensor<f32>,
    points: &Tensor<f32>,
    metadata: &Tensor<f32>,
    tables: &Tensor<f32>,
    table_grad: &Tensor<Atomic<u32>>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    adjoint_offset: u32,
    #[comptime] local_hash: bool,
    #[comptime] compact_gradients: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] cross_corners: bool,
    #[comptime] write_dh: bool,
    #[comptime] replay_features: bool,
    #[comptime] selective_weights: bool,
    #[comptime] replay_hash: bool,
) {
    let unit = usize::cast_from(UNIT_POS);
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let mut needed = false;
            if index < 1984 {
                let column = index / 64;
                needed =
                    column < 3 + usize::cast_from(active) * 2 && (replay_features || column >= 3);
            } else if index < 2048 {
                needed = replay_features;
            } else if index < 2112 {
                needed = true;
            }
            if !selective_weights || needed {
                let target = if local_hash && index < 1984 {
                    (index % 64) * 31 + index / 64
                } else {
                    index
                };
                shared_weights[target] = weights[index];
            }
        }
    }
    let lane = unit % 64;
    let slot = unit / 64;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let point = group * 4 + slot;
    let n_valid = usize::cast_from(valid_n);
    // Hash interpolation scratch is reused by output/feature gradients below.
    // Cache hits retain their original allocation; misses add only 64 floats.
    let mut partials = Shared::new_slice(if replay_hash && corner_parallel {
        512usize
    } else if corner_parallel {
        448usize
    } else if compact_gradients || replay_hash {
        256usize
    } else {
        1usize
    });
    // Only replay variants need this 512-byte tile. Stage each feature once for
    // all 64 neurons; no full preactivation buffer is written or read.
    let mut replay = Shared::new_slice(if replay_features { 128usize } else { 1usize });
    if replay_hash {
        field_hash_tile(
            points,
            tables,
            metadata,
            &mut replay,
            &mut partials,
            unit,
            group,
            n_valid,
            active,
            0.0_f32,
            local_hash,
            corner_parallel,
            0usize,
            false,
        );
        // Parameter gradients consume these features after this dispatch.
        // All 31 columns are initialized, including inactive hash levels.
        if point < n_valid && lane < 31 {
            features[point * 31 + lane] = replay[field_feature_index(slot, lane, local_hash)];
        }
    } else if replay_features {
        if unit < 128 {
            let p = group * 4 + unit / 32;
            let col = unit % 32;
            let mut f = 0.0_f32;
            if p < n_valid && col < 3 + usize::cast_from(active) * 2 {
                f = features[p * features.stride(0) + col * features.stride(1)];
            }
            replay[unit] = f;
        }
    }
    // One uniform barrier publishes both packed weights and optional features.
    sync_cube();
    let mut shared = Shared::new_slice(256usize);
    // Reused first for `output_grads` `[256]` and then for 224-thread partial
    // feature gradients `[448]`, keeping shared memory under 12 KB per block.
    let mut slot_ds = Shared::new_slice(4usize);
    let mut output_grad = 0.0_f32;
    let mut v = 0.0_f32;
    let mut ds_val = 0.0_f32;
    if point < n_valid {
        ds_val = ds[(point + usize::cast_from(adjoint_offset)) * ds.stride(0)];
        if ds_val != 0.0_f32 {
            let mut h = 0.0_f32;
            if replay_features {
                for col in 0..(3 + usize::cast_from(active) * 2) {
                    let wi = if local_hash {
                        lane * 31 + col
                    } else {
                        col * 64 + lane
                    };
                    let fi = if replay_hash {
                        field_feature_index(slot, col, local_hash)
                    } else {
                        slot * 32 + col
                    };
                    h += replay[fi] * shared_weights[wi];
                }
                h += shared_weights[1984 + lane];
            } else {
                h = pre[point * 64 + lane];
            }
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
    if replay_features {
        sync_cube();
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
        let ([p, d, w, x, m, f, t, e, h, g], [ot, oh, og, of]) = self.desc.as_fixed();
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
            ScalarIr::UInt(replay_features),
            ScalarIr::UInt(adjoint_offset),
            ScalarIr::UInt(selective_weights),
            ScalarIr::UInt(replay_hash),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid backward scalars")
        };
        let [p, d, w, x, m, f, t, e, h, g] =
            [p, d, w, x, m, f, t, e, h, g].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let t = if t.can_mut() {
            t
        } else {
            crate::diagnostics::count(
                "hash_backward.gradient_copy_bytes",
                t.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            t.copy()
        };
        let f = if *replay_hash != 0 && !f.can_mut() {
            f.copy()
        } else {
            f
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
                f.clone().into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                h.clone().into_tensor_arg(),
                g.clone().into_tensor_arg(),
                x.into_tensor_arg(),
                m.into_tensor_arg(),
                e.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *adjoint_offset as u32,
                *local_hash != 0,
                *compact_gradients != 0,
                *corner_parallel != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *cross_corners != 0,
                *write_dh != 0,
                *replay_features != 0,
                *selective_weights != 0,
                *replay_hash != 0,
            );
        } else {
            field_backward_kernel::launch(
                &client,
                CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
                CubeDim::new_1d(256),
                p.into_tensor_arg(),
                f.clone().into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                h.clone().into_tensor_arg(),
                g.clone().into_tensor_arg(),
                x.into_tensor_arg(),
                m.into_tensor_arg(),
                e.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *adjoint_offset as u32,
                *local_hash != 0,
                *compact_gradients != 0,
                *corner_parallel != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *cross_corners != 0,
                *write_dh != 0,
                *replay_features != 0,
                *selective_weights != 0,
                *replay_hash != 0,
            );
        }
        handles.register_float_tensor::<CubeBackend>(&ot.id, t);
        handles.register_float_tensor::<CubeBackend>(&oh.id, h);
        handles.register_float_tensor::<CubeBackend>(&og.id, g);
        handles.register_float_tensor::<CubeBackend>(&of.id, f);
        Ok(())
    }
}
pub fn field_backward(
    inputs: [burn::tensor::Tensor<2>; 8],
    valid_n: usize,
    active: usize,
    local_hash: bool,
    compact_gradients: bool,
    corner_parallel: bool,
    aggregate_hash: bool,
    aggregation_levels: usize,
    cross_corners: bool,
    write_dh: bool,
    replay_features: bool,
    adjoint_offset: usize,
    selective_weights: bool,
    replay_hash: bool,
) -> [burn::tensor::Tensor<2>; 4] {
    use burn::tensor::Tensor as BT;
    assert!(
        adjoint_offset <= inputs[1].dims()[0] && valid_n <= inputs[1].dims()[0] - adjoint_offset
    );
    assert!(adjoint_offset <= u32::MAX as usize);
    assert!(!replay_hash || (replay_features && write_dh));
    let n = inputs[3].dims()[0];
    assert!(valid_n <= n && valid_n <= inputs[5].dims()[0]);
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
    let outputs: Vec<_> = [6usize, 8, 9, 5]
        .into_iter()
        .map(|i| &inputs[i])
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
            ScalarIr::UInt(replay_features as u64),
            ScalarIr::UInt(adjoint_offset as u64),
            ScalarIr::UInt(selective_weights as u64),
            ScalarIr::UInt(replay_hash as u64),
        ],
    );
    let result: [_; 4] = client
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
    #[comptime] compact_cache: bool,
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
                    if compact_cache {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            3,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    } else {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )] + scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            4,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                } else {
                    if compact_cache {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    } else {
                        total += -scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )] + scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            3,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col == 0 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col > 0 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel - 1,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col < 2 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel + 1,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col == 2 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row == 0 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row > 0 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel - 3,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row < 2 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel + 3,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row == 2 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
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
                    if compact_cache {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            3,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    } else {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )] + scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            4,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                } else {
                    if compact_cache {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    } else {
                        total += -scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            0,
                            coalesced_scratch,
                            compact_cache,
                        )] + scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            3,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    let p_col = pixel % 3;
                    let p_row = pixel / 3;
                    if p_col == 0 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col > 0 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel - 1,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col < 2 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel + 1,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_col == 2 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            1,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row == 0 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row > 0 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel - 3,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row < 2 {
                        total -= scratch[composite_adjoint_index(
                            sample,
                            pixel + 3,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
                    }
                    if p_row == 2 {
                        total += scratch[composite_adjoint_index(
                            sample,
                            pixel,
                            2,
                            coalesced_scratch,
                            compact_cache,
                        )];
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
        let dim = CubeDim::new_1d(32);
        visibility_kernel::launch(
            &client,
            CubeCount::Static(n.div_ceil(32) as u32, 1, 1),
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
    features: &mut Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    dh: &mut Tensor<f32>,
    dw1: &mut Tensor<f32>,
    points: &Tensor<f32>,
    metadata: &Tensor<f32>,
    tables: &Tensor<f32>,
    table_grad: &Tensor<Atomic<f32>>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    adjoint_offset: u32,
    #[comptime] local_hash: bool,
    #[comptime] compact_gradients: bool,
    #[comptime] corner_parallel: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] cross_corners: bool,
    #[comptime] write_dh: bool,
    #[comptime] replay_features: bool,
    #[comptime] selective_weights: bool,
    #[comptime] replay_hash: bool,
) {
    let unit = usize::cast_from(UNIT_POS);
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let mut needed = false;
            if index < 1984 {
                let column = index / 64;
                needed =
                    column < 3 + usize::cast_from(active) * 2 && (replay_features || column >= 3);
            } else if index < 2048 {
                needed = replay_features;
            } else if index < 2112 {
                needed = true;
            }
            if !selective_weights || needed {
                let target = if local_hash && index < 1984 {
                    (index % 64) * 31 + index / 64
                } else {
                    index
                };
                shared_weights[target] = weights[index];
            }
        }
    }
    let lane = unit % 64;
    let slot = unit / 64;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let point = group * 4 + slot;
    let n_valid = usize::cast_from(valid_n);
    // Hash interpolation scratch is reused by output/feature gradients below.
    // Cache hits retain their original allocation; misses add only 64 floats.
    let mut partials = Shared::new_slice(if replay_hash && corner_parallel {
        512usize
    } else if corner_parallel {
        448usize
    } else if compact_gradients || replay_hash {
        256usize
    } else {
        1usize
    });
    // Only replay variants need this 512-byte tile. Stage each feature once for
    // all 64 neurons; no full preactivation buffer is written or read.
    let mut replay = Shared::new_slice(if replay_features { 128usize } else { 1usize });
    if replay_hash {
        field_hash_tile(
            points,
            tables,
            metadata,
            &mut replay,
            &mut partials,
            unit,
            group,
            n_valid,
            active,
            0.0_f32,
            local_hash,
            corner_parallel,
            0usize,
            false,
        );
        // Parameter gradients consume these features after this dispatch.
        // All 31 columns are initialized, including inactive hash levels.
        if point < n_valid && lane < 31 {
            features[point * 31 + lane] = replay[field_feature_index(slot, lane, local_hash)];
        }
    } else if replay_features {
        if unit < 128 {
            let p = group * 4 + unit / 32;
            let col = unit % 32;
            let mut f = 0.0_f32;
            if p < n_valid && col < 3 + usize::cast_from(active) * 2 {
                f = features[p * features.stride(0) + col * features.stride(1)];
            }
            replay[unit] = f;
        }
    }
    // One uniform barrier publishes both packed weights and optional features.
    sync_cube();
    let mut shared = Shared::new_slice(256usize);
    // Reused first for `output_grads` `[256]` and then for 224-thread partial
    // feature gradients `[448]`, keeping shared memory under 12 KB per block.
    let mut slot_ds = Shared::new_slice(4usize);
    let mut output_grad = 0.0_f32;
    let mut v = 0.0_f32;
    let mut ds_val = 0.0_f32;
    if point < n_valid {
        ds_val = ds[(point + usize::cast_from(adjoint_offset)) * ds.stride(0)];
        if ds_val != 0.0_f32 {
            let mut h = 0.0_f32;
            if replay_features {
                for col in 0..(3 + usize::cast_from(active) * 2) {
                    let wi = if local_hash {
                        lane * 31 + col
                    } else {
                        col * 64 + lane
                    };
                    let fi = if replay_hash {
                        field_feature_index(slot, col, local_hash)
                    } else {
                        slot * 32 + col
                    };
                    h += replay[fi] * shared_weights[wi];
                }
                h += shared_weights[1984 + lane];
            } else {
                h = pre[point * 64 + lane];
            }
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
    if replay_features {
        sync_cube();
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

// Rebuild only the first MLP layer from retained features. The accumulation
// order and packed weights match fused_field_kernel; no hash lookup, softplus,
// final SDF output, or new per-point cache is needed in this pass.
#[cube(launch)]
fn cached_mlp_preactivation_kernel(
    features: &Tensor<f32>,
    weights: &Tensor<f32>,
    pre: &mut Tensor<f32>,
    valid_n: u32,
    active: u32,
) {
    let unit = usize::cast_from(UNIT_POS);
    let lane = unit % 64;
    let slot = unit / 64;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let point = group * 4 + slot;
    let valid = point < usize::cast_from(valid_n);
    let mut shared_weights = Shared::new_slice(2048usize);
    let mut shared_features = Shared::new_slice(128usize);
    for tile in 0..8usize {
        let i = unit + tile * 256;
        shared_weights[i] = weights[i];
    }
    if lane < 31 && valid {
        shared_features[slot * 32 + lane] =
            features[point * features.stride(0) + lane * features.stride(1)];
    }
    // All invocations, including padded/tail lanes, reach the barrier.
    sync_cube();
    if valid {
        let mut h = 0.0_f32;
        for k in 0..(3 + usize::cast_from(active) * 2) {
            h += shared_features[slot * 32 + k] * shared_weights[k * 64 + lane];
        }
        pre[point * 64 + lane] = h + shared_weights[1984 + lane];
    }
}

#[derive(Debug)]
struct CachedMlpOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for CachedMlpOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([f, w, p], [out]) = self.desc.as_fixed();
        let [ScalarIr::UInt(valid_n), ScalarIr::UInt(active)] = self.desc.scalars.as_slice() else {
            panic!("invalid cached MLP scalars")
        };
        let [f, w, p] = [f, w, p].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let client = f.client.clone();
        let _timing = crate::diagnostics::start(&client, "cached_mlp_preactivation");
        let groups = (*valid_n as usize).div_ceil(4).max(1);
        let gx = groups.min(65535);
        cached_mlp_preactivation_kernel::launch(
            &client,
            CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
            CubeDim::new_1d(256),
            f.into_tensor_arg(),
            w.into_tensor_arg(),
            p.clone().into_tensor_arg(),
            *valid_n as u32,
            *active as u32,
        );
        handles.register_float_tensor::<CubeBackend>(&out.id, p);
        Ok(())
    }
}
pub fn cached_mlp_preactivation(
    features: burn::tensor::Tensor<2>,
    weights: burn::tensor::Tensor<2>,
    valid_n: usize,
    active: usize,
) -> burn::tensor::Tensor<2> {
    use burn::tensor::Tensor as BT;
    assert!(valid_n > 0 && valid_n <= features.dims()[0] && active <= 14);
    let output = BT::empty([features.dims()[0], 64], &features.device());
    let inputs =
        [features, weights, output].map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        inputs[2].shape.clone(),
        inputs[2].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "supernormal_cached_mlp_preactivation",
        &inputs.map(|t| t.into_ir()),
        &[out],
        vec![
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(active as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            CachedMlpOp { desc },
        )
        .outputs();
    BT::<2>::from_primitive::<GpuBackend>(out)
}

// Persistent fused backward + parameter-gradient kernel (native float atomics):
// Eliminates the global `dh` [N, 64] buffer, eliminates the separate
// `parameter_gradients` dispatch and double `features` reads, stages MLP
// weights once per workgroup instead of once per 4 points, replaces 4 of 6
// workgroup barriers with warp-shuffle reductions & warp-shuffle same-address
// hash aggregation, and accumulates `dw0`/`db0`/`dw1` partials across chunks.
#[cube(launch)]
fn field_backward_fused_native_kernel(
    pre: &Tensor<f32>,
    features: &Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    metadata: &Tensor<f32>,
    table_grad: &Tensor<Atomic<f32>>,
    dw0_partials: &mut Tensor<f32>,
    dw1_partials: &mut Tensor<f32>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    adjoint_offset: u32,
    scale_cutoff: f32,
    #[comptime] local_hash: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] replay_features: bool,
    #[comptime] selective_weights: bool,
    #[comptime] accumulate: bool,
    #[comptime] subgroup_width: usize,
    #[comptime] active_columns_only: bool,
    #[comptime] scale_encoding: bool,
) {
    let mut unit = usize::cast_from(UNIT_POS);
    if subgroup_width > 0 {
        unit = usize::cast_from(PLANE_POS) * subgroup_width + usize::cast_from(UNIT_POS_PLANE);
    }
    let lane = unit % 64;
    let slot = unit / 64;
    let p_row = unit / 16;
    let p_col = unit % 16;
    let hash_slot = unit & 3usize;
    let sub_part = (unit >> 2) & 3usize;
    let level = unit >> 4;
    let active_cols = 3usize + usize::cast_from(active) * 2usize;
    let feat_cols = if active_columns_only {
        active_cols
    } else {
        usize::cast_from(31u32)
    };
    let first_active = p_col < feat_cols;
    let second_active = p_col + 16usize < feat_cols || p_col + 16usize == 31usize;

    let n_valid = usize::cast_from(valid_n);
    let total_groups = (n_valid + 3usize) / 4usize;
    let part = usize::cast_from(CUBE_POS_X);
    let stride = usize::cast_from(CUBE_COUNT_X);

    // Stage MLP weights ONCE per workgroup (never overwritten by aggregation).
    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let mut needed = false;
            if index < 1984 {
                let column = index / 64;
                needed = column < active_cols && (replay_features || column >= 3);
            } else if index < 2048 {
                needed = replay_features;
            } else if index < 2112 {
                needed = true;
            }
            if !selective_weights || needed {
                let target = if local_hash && index < 1984 {
                    (index % 64) * 31 + index / 64
                } else {
                    index
                };
                shared_weights[target] = weights[index];
            }
        }
    }
    let mut shared_dh = Shared::new_slice(272usize);
    let mut shared_feat = Shared::new_slice(128usize);
    let mut slot_ds = Shared::new_slice(4usize);
    let mut slot_scale = Shared::new_slice(4usize);
    let mut fallback_partials = Shared::new_slice(if subgroup_width == 0 { 448usize } else { 1usize });
    sync_cube();

    let mut acc00 = 0.0_f32;
    let mut acc01 = 0.0_f32;
    let mut acc10 = 0.0_f32;
    let mut acc11 = 0.0_f32;
    let mut acc20 = 0.0_f32;
    let mut acc21 = 0.0_f32;
    let mut acc30 = 0.0_f32;
    let mut acc31 = 0.0_f32;
    let mut dw1_reg = 0.0_f32;

    // Hoist loop-invariant hash-level metadata (2 registers) out of the group loop.
    let mut level_res = 0u32;
    let mut level_offset = 0usize;
    if level < usize::cast_from(active) {
        level_res = u32::cast_from(metadata[level * 2]);
        level_offset = usize::cast_from(metadata[level * 2 + 1]);
    }
    let lane_base = (u32::cast_from(UNIT_POS_PLANE) >> 2u32) << 2u32;
    let use_agg = aggregate_hash
        && subgroup_width >= 32usize
        && level < usize::cast_from(aggregation_levels);

    let tiles_per_part = (total_groups + stride - 1usize) / stride;
    let start_group = part * tiles_per_part;
    let end_group = usize::min(start_group + tiles_per_part, total_groups);
    let mut group = start_group;
    while group < end_group {
        if unit < 4usize {
            let p = group * 4 + unit;
            slot_ds[unit] = if p < n_valid {
                ds[(p + usize::cast_from(adjoint_offset)) * ds.stride(0)]
            } else {
                0.0_f32
            };
            if scale_encoding {
                slot_scale[unit] = if p < n_valid {
                    features[p * features.stride(0) + 31usize * features.stride(1)]
                } else {
                    0.0_f32
                };
            }
        }
        if unit < 128usize {
            let s = unit / 32;
            let col = unit % 32;
            let p = group * 4 + s;
            let mut f = 0.0_f32;
            if p < n_valid {
                if col == 31usize {
                    f = 1.0_f32;
                } else if col < feat_cols {
                    f = features[p * features.stride(0) + col * features.stride(1)];
                }
            }
            shared_feat[unit] = f;
        }
        sync_cube();

        let ds0 = slot_ds[0];
        let ds1 = slot_ds[1];
        let ds2 = slot_ds[2];
        let ds3 = slot_ds[3];
        if ds0 != 0.0_f32 || ds1 != 0.0_f32 || ds2 != 0.0_f32 || ds3 != 0.0_f32 {
            let point = group * 4 + slot;
            let ds_val = slot_ds[slot];
            let mut v = 0.0_f32;
            if point < n_valid && ds_val != 0.0_f32 {
                let mut h = 0.0_f32;
                if replay_features {
                    for col in 0..active_cols {
                        let wi = if local_hash {
                            lane * 31 + col
                        } else {
                            col * 64 + lane
                        };
                        h += shared_feat[slot * 32 + col] * shared_weights[wi];
                    }
                    h += shared_weights[1984 + lane];
                } else {
                    h = pre[point * 64 + lane];
                }
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
                dw1_reg += ds_val * hidden;
            }
            shared_dh[slot * 68 + lane] = v;
            sync_cube();

            // Fused in-register dw0 + db0 outer product (zero bank conflicts).
            for s in 0..4usize {
                if slot_ds[s] != 0.0_f32 {
                    let h0 = shared_dh[s * 68 + p_row];
                    let h1 = shared_dh[s * 68 + p_row + 16];
                    let h2 = shared_dh[s * 68 + p_row + 32];
                    let h3 = shared_dh[s * 68 + p_row + 48];
                    if first_active {
                        let x0 = shared_feat[s * 32 + p_col];
                        acc00 += h0 * x0;
                        acc10 += h1 * x0;
                        acc20 += h2 * x0;
                        acc30 += h3 * x0;
                    }
                    if second_active {
                        let x1 = shared_feat[s * 32 + p_col + 16];
                        acc01 += h0 * x1;
                        acc11 += h1 * x1;
                        acc21 += h2 * x1;
                        acc31 += h3 * x1;
                    }
                }
            }

            // 224-thread hash backward with bank-conflict-free interleaved j,
            // warp-shuffle reduction, and low-register 4-peer warp-shuffle aggregation.
            let hash_point = group * 4 + hash_slot;
            let active_lane = hash_point < n_valid
                && level < usize::cast_from(active)
                && slot_ds[hash_slot] != 0.0_f32;
            let mut p_g0 = 0.0_f32;
            let mut p_g1 = 0.0_f32;
            if active_lane {
                let col = 3 + level * 2;
                for j_off in 0..16usize {
                    let j = j_off * 4 + sub_part;
                    let h = shared_dh[hash_slot * 68 + j];
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
                    p_g0 += h * shared_weights[w0];
                    p_g1 += h * shared_weights[w1];
                }
            }
            let (g0, g1) = if subgroup_width >= 32usize {
                let s0 = p_g0 + plane_shuffle_xor(p_g0, 4u32);
                let fg0 = s0 + plane_shuffle_xor(s0, 8u32);
                let s1 = p_g1 + plane_shuffle_xor(p_g1, 4u32);
                let fg1 = s1 + plane_shuffle_xor(s1, 8u32);
                (fg0, fg1)
            } else {
                if level < 14usize {
                    fallback_partials[unit * 2] = p_g0;
                    fallback_partials[unit * 2 + 1] = p_g1;
                }
                sync_cube();
                if active_lane {
                    let base_p = (level * 16 + hash_slot) * 2;
                    let fg0 = (fallback_partials[base_p] + fallback_partials[base_p + 8])
                        + (fallback_partials[base_p + 16] + fallback_partials[base_p + 24]);
                    let fg1 = (fallback_partials[base_p + 1] + fallback_partials[base_p + 9])
                        + (fallback_partials[base_p + 17] + fallback_partials[base_p + 25]);
                    (fg0, fg1)
                } else {
                    (0.0_f32, 0.0_f32)
                }
            };

            let do_hash = active_lane && (g0 != 0.0_f32 || g1 != 0.0_f32);
            let mut base0 = 0u32;
            let mut w0 = 0.0_f32;
            let mut kw12 = 0.0_f32;
            let mut dense = false;
            let mut dense_yz = 0u32;
            let mut hash_yz = 0u32;
            if do_hash {
                let norm0 = f32::clamp(
                    (shared_feat[hash_slot * 32] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (shared_feat[hash_slot * 32 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (shared_feat[hash_slot * 32 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res_f = f32::cast_from(level_res);
                let c0 = norm0 * res_f;
                let c1 = norm1 * res_f;
                let c2 = norm2 * res_f;
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                base0 = u32::cast_from(b0);
                let r1 = level_res + 1u32;
                dense = r1 * r1 * r1 <= 524288u32;
                let bit1 = u32::cast_from(sub_part) & 1u32;
                let bit2 = (u32::cast_from(sub_part) >> 1u32) & 1u32;
                let v1 = u32::cast_from(b1) + bit1;
                let v2 = u32::cast_from(b2) + bit2;
                let kw1 = if bit1 == 0u32 { 1.0_f32 - w1 } else { w1 };
                let kw2 = if bit2 == 0u32 { 1.0_f32 - w2 } else { w2 };
                kw12 = kw1 * kw2;
                if scale_encoding {
                    kw12 *= scale_level_weight(slot_scale[hash_slot], level_res, scale_cutoff);
                }
                dense_yz = v1 * r1 + v2 * r1 * r1;
                hash_yz = (v1 * 2654435761u32) ^ (v2 * 805459861u32);
            }

            // Process corner 0 and corner 1 sequentially to minimize live registers.
            for c_sub in 0..2u32 {
                let mut key = 0xFFFF_FFFFu32;
                let mut wg0 = 0.0_f32;
                let mut wg1 = 0.0_f32;
                if do_hash {
                    let v0 = base0 + c_sub;
                    let weight = (if c_sub == 0u32 { 1.0_f32 - w0 } else { w0 }) * kw12;
                    let idx = if dense {
                        v0 + dense_yz
                    } else {
                        (v0 ^ hash_yz) & 524287u32
                    };
                    let t0 = weight * g0;
                    let t1 = weight * g1;
                    if use_agg {
                        key = idx;
                        wg0 = t0;
                        wg1 = t1;
                    } else {
                        let e = (level_offset + usize::cast_from(idx)) * 2;
                        if t0 != 0.0_f32 {
                            table_grad[e].fetch_add(t0);
                        }
                        if t1 != 0.0_f32 {
                            table_grad[e + 1].fetch_add(t1);
                        }
                    }
                }
                if aggregate_hash && subgroup_width >= 32usize {
                    let mut is_leader = key != 0xFFFF_FFFFu32;
                    let mut sum0 = wg0;
                    let mut sum1 = wg1;
                    for peer in 0..4u32 {
                        let pk = plane_shuffle(key, lane_base + peer);
                        let pg0 = plane_shuffle(wg0, lane_base + peer);
                        let pg1 = plane_shuffle(wg1, lane_base + peer);
                        let p_idx = usize::cast_from(peer);
                        if p_idx < hash_slot && pk == key {
                            is_leader = false;
                        }
                        if p_idx > hash_slot && pk == key {
                            sum0 += pg0;
                            sum1 += pg1;
                        }
                    }
                    if is_leader {
                        let entry = (level_offset + usize::cast_from(key)) * 2;
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
        group += 1usize;
        if group < end_group {
            sync_cube();
        }
    }

    // Write workgroup dw0 + db0 partials once per workgroup.
    let row0 = part * 2048 + p_row * 32;
    let row1 = part * 2048 + (p_row + 16) * 32;
    let row2 = part * 2048 + (p_row + 32) * 32;
    let row3 = part * 2048 + (p_row + 48) * 32;
    if accumulate {
        dw0_partials[row0 + p_col] += acc00;
        dw0_partials[row1 + p_col] += acc10;
        dw0_partials[row2 + p_col] += acc20;
        dw0_partials[row3 + p_col] += acc30;
        dw0_partials[row0 + p_col + 16] += acc01;
        dw0_partials[row1 + p_col + 16] += acc11;
        dw0_partials[row2 + p_col + 16] += acc21;
        dw0_partials[row3 + p_col + 16] += acc31;
    } else {
        dw0_partials[row0 + p_col] = acc00;
        dw0_partials[row1 + p_col] = acc10;
        dw0_partials[row2 + p_col] = acc20;
        dw0_partials[row3 + p_col] = acc30;
        dw0_partials[row0 + p_col + 16] = acc01;
        dw0_partials[row1 + p_col + 16] = acc11;
        dw0_partials[row2 + p_col + 16] = acc21;
        dw0_partials[row3 + p_col + 16] = acc31;
    }

    // Reduce dw1 across the 4 slots in shared_dh (free after loop) and write once.
    sync_cube();
    shared_dh[unit] = dw1_reg;
    sync_cube();
    if slot == 0usize {
        let sum = (shared_dh[lane] + shared_dh[64 + lane])
            + (shared_dh[128 + lane] + shared_dh[192 + lane]);
        if accumulate {
            dw1_partials[part * 64 + lane] += sum;
        } else {
            dw1_partials[part * 64 + lane] = sum;
        }
    }
}

// CAS-atomic companion to `field_backward_fused_native_kernel` for runtimes
// without native float atomic add.
#[cube(launch)]
fn field_backward_fused_cas_kernel(
    pre: &Tensor<f32>,
    features: &Tensor<f32>,
    ds: &Tensor<f32>,
    weights: &Tensor<f32>,
    metadata: &Tensor<f32>,
    table_grad: &Tensor<Atomic<u32>>,
    dw0_partials: &mut Tensor<f32>,
    dw1_partials: &mut Tensor<f32>,
    active: u32,
    valid_n: u32,
    aggregation_levels: u32,
    adjoint_offset: u32,
    scale_cutoff: f32,
    #[comptime] local_hash: bool,
    #[comptime] aggregate_hash: bool,
    #[comptime] replay_features: bool,
    #[comptime] selective_weights: bool,
    #[comptime] accumulate: bool,
    #[comptime] subgroup_width: usize,
    #[comptime] active_columns_only: bool,
    #[comptime] scale_encoding: bool,
) {
    let mut unit = usize::cast_from(UNIT_POS);
    if subgroup_width > 0 {
        unit = usize::cast_from(PLANE_POS) * subgroup_width + usize::cast_from(UNIT_POS_PLANE);
    }
    let lane = unit % 64;
    let slot = unit / 64;
    let p_row = unit / 16;
    let p_col = unit % 16;
    let hash_slot = unit & 3usize;
    let sub_part = (unit >> 2) & 3usize;
    let level = unit >> 4;
    let active_cols = 3usize + usize::cast_from(active) * 2usize;
    let feat_cols = if active_columns_only {
        active_cols
    } else {
        usize::cast_from(31u32)
    };
    let first_active = p_col < feat_cols;
    let second_active = p_col + 16usize < feat_cols || p_col + 16usize == 31usize;

    let n_valid = usize::cast_from(valid_n);
    let total_groups = (n_valid + 3usize) / 4usize;
    let part = usize::cast_from(CUBE_POS_X);
    let stride = usize::cast_from(CUBE_COUNT_X);

    let mut shared_weights = Shared::new_slice(2113usize);
    for tile in 0..9usize {
        let index = unit + tile * 256;
        if index < 2113 {
            let mut needed = false;
            if index < 1984 {
                let column = index / 64;
                needed = column < active_cols && (replay_features || column >= 3);
            } else if index < 2048 {
                needed = replay_features;
            } else if index < 2112 {
                needed = true;
            }
            if !selective_weights || needed {
                let target = if local_hash && index < 1984 {
                    (index % 64) * 31 + index / 64
                } else {
                    index
                };
                shared_weights[target] = weights[index];
            }
        }
    }
    let mut shared_dh = Shared::new_slice(272usize);
    let mut shared_feat = Shared::new_slice(128usize);
    let mut slot_ds = Shared::new_slice(4usize);
    let mut slot_scale = Shared::new_slice(4usize);
    let mut fallback_partials = Shared::new_slice(if subgroup_width == 0 { 448usize } else { 1usize });
    sync_cube();

    let mut acc00 = 0.0_f32;
    let mut acc01 = 0.0_f32;
    let mut acc10 = 0.0_f32;
    let mut acc11 = 0.0_f32;
    let mut acc20 = 0.0_f32;
    let mut acc21 = 0.0_f32;
    let mut acc30 = 0.0_f32;
    let mut acc31 = 0.0_f32;
    let mut dw1_reg = 0.0_f32;

    // Hoist loop-invariant hash-level metadata (2 registers) out of the group loop.
    let mut level_res = 0u32;
    let mut level_offset = 0usize;
    if level < usize::cast_from(active) {
        level_res = u32::cast_from(metadata[level * 2]);
        level_offset = usize::cast_from(metadata[level * 2 + 1]);
    }
    let lane_base = (u32::cast_from(UNIT_POS_PLANE) >> 2u32) << 2u32;
    let use_agg = aggregate_hash
        && subgroup_width >= 32usize
        && level < usize::cast_from(aggregation_levels);

    let tiles_per_part = (total_groups + stride - 1usize) / stride;
    let start_group = part * tiles_per_part;
    let end_group = usize::min(start_group + tiles_per_part, total_groups);
    let mut group = start_group;
    while group < end_group {
        if unit < 4usize {
            let p = group * 4 + unit;
            slot_ds[unit] = if p < n_valid {
                ds[(p + usize::cast_from(adjoint_offset)) * ds.stride(0)]
            } else {
                0.0_f32
            };
            if scale_encoding {
                slot_scale[unit] = if p < n_valid {
                    features[p * features.stride(0) + 31usize * features.stride(1)]
                } else {
                    0.0_f32
                };
            }
        }
        if unit < 128usize {
            let s = unit / 32;
            let col = unit % 32;
            let p = group * 4 + s;
            let mut f = 0.0_f32;
            if p < n_valid {
                if col == 31usize {
                    f = 1.0_f32;
                } else if col < feat_cols {
                    f = features[p * features.stride(0) + col * features.stride(1)];
                }
            }
            shared_feat[unit] = f;
        }
        sync_cube();

        let ds0 = slot_ds[0];
        let ds1 = slot_ds[1];
        let ds2 = slot_ds[2];
        let ds3 = slot_ds[3];
        if ds0 != 0.0_f32 || ds1 != 0.0_f32 || ds2 != 0.0_f32 || ds3 != 0.0_f32 {
            let point = group * 4 + slot;
            let ds_val = slot_ds[slot];
            let mut v = 0.0_f32;
            if point < n_valid && ds_val != 0.0_f32 {
                let mut h = 0.0_f32;
                if replay_features {
                    for col in 0..active_cols {
                        let wi = if local_hash {
                            lane * 31 + col
                        } else {
                            col * 64 + lane
                        };
                        h += shared_feat[slot * 32 + col] * shared_weights[wi];
                    }
                    h += shared_weights[1984 + lane];
                } else {
                    h = pre[point * 64 + lane];
                }
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
                dw1_reg += ds_val * hidden;
            }
            shared_dh[slot * 68 + lane] = v;
            sync_cube();

            for s in 0..4usize {
                if slot_ds[s] != 0.0_f32 {
                    let h0 = shared_dh[s * 68 + p_row];
                    let h1 = shared_dh[s * 68 + p_row + 16];
                    let h2 = shared_dh[s * 68 + p_row + 32];
                    let h3 = shared_dh[s * 68 + p_row + 48];
                    if first_active {
                        let x0 = shared_feat[s * 32 + p_col];
                        acc00 += h0 * x0;
                        acc10 += h1 * x0;
                        acc20 += h2 * x0;
                        acc30 += h3 * x0;
                    }
                    if second_active {
                        let x1 = shared_feat[s * 32 + p_col + 16];
                        acc01 += h0 * x1;
                        acc11 += h1 * x1;
                        acc21 += h2 * x1;
                        acc31 += h3 * x1;
                    }
                }
            }

            let hash_point = group * 4 + hash_slot;
            let active_lane = hash_point < n_valid
                && level < usize::cast_from(active)
                && slot_ds[hash_slot] != 0.0_f32;
            let mut p_g0 = 0.0_f32;
            let mut p_g1 = 0.0_f32;
            if active_lane {
                let col = 3 + level * 2;
                for j_off in 0..16usize {
                    let j = j_off * 4 + sub_part;
                    let h = shared_dh[hash_slot * 68 + j];
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
                    p_g0 += h * shared_weights[w0];
                    p_g1 += h * shared_weights[w1];
                }
            }
            let (g0, g1) = if subgroup_width >= 32usize {
                let s0 = p_g0 + plane_shuffle_xor(p_g0, 4u32);
                let fg0 = s0 + plane_shuffle_xor(s0, 8u32);
                let s1 = p_g1 + plane_shuffle_xor(p_g1, 4u32);
                let fg1 = s1 + plane_shuffle_xor(s1, 8u32);
                (fg0, fg1)
            } else {
                if level < 14usize {
                    fallback_partials[unit * 2] = p_g0;
                    fallback_partials[unit * 2 + 1] = p_g1;
                }
                sync_cube();
                if active_lane {
                    let base_p = (level * 16 + hash_slot) * 2;
                    let fg0 = (fallback_partials[base_p] + fallback_partials[base_p + 8])
                        + (fallback_partials[base_p + 16] + fallback_partials[base_p + 24]);
                    let fg1 = (fallback_partials[base_p + 1] + fallback_partials[base_p + 9])
                        + (fallback_partials[base_p + 17] + fallback_partials[base_p + 25]);
                    (fg0, fg1)
                } else {
                    (0.0_f32, 0.0_f32)
                }
            };

            let do_hash = active_lane && (g0 != 0.0_f32 || g1 != 0.0_f32);
            let mut base0 = 0u32;
            let mut w0 = 0.0_f32;
            let mut kw12 = 0.0_f32;
            let mut dense = false;
            let mut dense_yz = 0u32;
            let mut hash_yz = 0u32;
            if do_hash {
                let norm0 = f32::clamp(
                    (shared_feat[hash_slot * 32] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm1 = f32::clamp(
                    (shared_feat[hash_slot * 32 + 1] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let norm2 = f32::clamp(
                    (shared_feat[hash_slot * 32 + 2] + 1.0_f32) * 0.5_f32,
                    0.0_f32,
                    1.0_f32 - 1e-7_f32,
                );
                let res_f = f32::cast_from(level_res);
                let c0 = norm0 * res_f;
                let c1 = norm1 * res_f;
                let c2 = norm2 * res_f;
                let b0 = f32::floor(c0);
                let b1 = f32::floor(c1);
                let b2 = f32::floor(c2);
                w0 = c0 - b0;
                let w1 = c1 - b1;
                let w2 = c2 - b2;
                base0 = u32::cast_from(b0);
                let r1 = level_res + 1u32;
                dense = r1 * r1 * r1 <= 524288u32;
                let bit1 = u32::cast_from(sub_part) & 1u32;
                let bit2 = (u32::cast_from(sub_part) >> 1u32) & 1u32;
                let v1 = u32::cast_from(b1) + bit1;
                let v2 = u32::cast_from(b2) + bit2;
                let kw1 = if bit1 == 0u32 { 1.0_f32 - w1 } else { w1 };
                let kw2 = if bit2 == 0u32 { 1.0_f32 - w2 } else { w2 };
                kw12 = kw1 * kw2;
                if scale_encoding {
                    kw12 *= scale_level_weight(slot_scale[hash_slot], level_res, scale_cutoff);
                }
                dense_yz = v1 * r1 + v2 * r1 * r1;
                hash_yz = (v1 * 2654435761u32) ^ (v2 * 805459861u32);
            }

            for c_sub in 0..2u32 {
                let mut key = 0xFFFF_FFFFu32;
                let mut wg0 = 0.0_f32;
                let mut wg1 = 0.0_f32;
                if do_hash {
                    let v0 = base0 + c_sub;
                    let weight = (if c_sub == 0u32 { 1.0_f32 - w0 } else { w0 }) * kw12;
                    let idx = if dense {
                        v0 + dense_yz
                    } else {
                        (v0 ^ hash_yz) & 524287u32
                    };
                    let t0 = weight * g0;
                    let t1 = weight * g1;
                    if use_agg {
                        key = idx;
                        wg0 = t0;
                        wg1 = t1;
                    } else {
                        let e = (level_offset + usize::cast_from(idx)) * 2;
                        if t0 != 0.0_f32 {
                            atomic_add_f32(table_grad, e, t0);
                        }
                        if t1 != 0.0_f32 {
                            atomic_add_f32(table_grad, e + 1, t1);
                        }
                    }
                }
                if aggregate_hash && subgroup_width >= 32usize {
                    let mut is_leader = key != 0xFFFF_FFFFu32;
                    let mut sum0 = wg0;
                    let mut sum1 = wg1;
                    for peer in 0..4u32 {
                        let pk = plane_shuffle(key, lane_base + peer);
                        let pg0 = plane_shuffle(wg0, lane_base + peer);
                        let pg1 = plane_shuffle(wg1, lane_base + peer);
                        let p_idx = usize::cast_from(peer);
                        if p_idx < hash_slot && pk == key {
                            is_leader = false;
                        }
                        if p_idx > hash_slot && pk == key {
                            sum0 += pg0;
                            sum1 += pg1;
                        }
                    }
                    if is_leader {
                        let entry = (level_offset + usize::cast_from(key)) * 2;
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
        group += 1usize;
        if group < end_group {
            sync_cube();
        }
    }

    let row0 = part * 2048 + p_row * 32;
    let row1 = part * 2048 + (p_row + 16) * 32;
    let row2 = part * 2048 + (p_row + 32) * 32;
    let row3 = part * 2048 + (p_row + 48) * 32;
    if accumulate {
        dw0_partials[row0 + p_col] += acc00;
        dw0_partials[row1 + p_col] += acc10;
        dw0_partials[row2 + p_col] += acc20;
        dw0_partials[row3 + p_col] += acc30;
        dw0_partials[row0 + p_col + 16] += acc01;
        dw0_partials[row1 + p_col + 16] += acc11;
        dw0_partials[row2 + p_col + 16] += acc21;
        dw0_partials[row3 + p_col + 16] += acc31;
    } else {
        dw0_partials[row0 + p_col] = acc00;
        dw0_partials[row1 + p_col] = acc10;
        dw0_partials[row2 + p_col] = acc20;
        dw0_partials[row3 + p_col] = acc30;
        dw0_partials[row0 + p_col + 16] = acc01;
        dw0_partials[row1 + p_col + 16] = acc11;
        dw0_partials[row2 + p_col + 16] = acc21;
        dw0_partials[row3 + p_col + 16] = acc31;
    }

    sync_cube();
    shared_dh[unit] = dw1_reg;
    sync_cube();
    if slot == 0usize {
        let sum = (shared_dh[lane] + shared_dh[64 + lane])
            + (shared_dh[128 + lane] + shared_dh[192 + lane]);
        if accumulate {
            dw1_partials[part * 64 + lane] += sum;
        } else {
            dw1_partials[part * 64 + lane] = sum;
        }
    }
}

#[derive(Debug)]
struct FieldBackwardFusedOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for FieldBackwardFusedOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p, f, d, w, m, t, dw0p, dw1p], [ot, odw0, odw1]) = self.desc.as_fixed();
        let [
            ScalarIr::UInt(active),
            ScalarIr::UInt(valid_n),
            ScalarIr::UInt(local_hash),
            ScalarIr::UInt(aggregate_hash),
            ScalarIr::UInt(aggregation_levels),
            ScalarIr::UInt(replay_features),
            ScalarIr::UInt(adjoint_offset),
            ScalarIr::UInt(selective_weights),
            ScalarIr::UInt(accumulate),
            ScalarIr::UInt(active_columns_only),
            ScalarIr::UInt(scale_encoding),
            ScalarIr::Float(scale_cutoff),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid fused backward scalars")
        };
        let [p, f, d, w, m, t, dw0p, dw1p] = [p, f, d, w, m, t, dw0p, dw1p]
            .map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let t = if t.can_mut() {
            t
        } else {
            crate::diagnostics::count(
                "hash_backward.gradient_copy_bytes",
                t.meta.shape().dims::<2>().iter().product::<usize>() * 4,
            );
            t.copy()
        };
        let dw0p = if *accumulate != 0 && !dw0p.can_mut() {
            dw0p.copy()
        } else {
            dw0p
        };
        let dw1p = if *accumulate != 0 && !dw1p.can_mut() {
            dw1p.copy()
        } else {
            dw1p
        };
        let client = p.client.clone();
        let _timing = crate::diagnostics::start(&client, "field_backward_fused");
        let props = client.properties();
        let hw = &props.hardware;
        let subgroup_width = if props
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
        let native_add = props
            .atomic_type_usage(Type::atomic(f32::elem_type_native()))
            .contains(cubecl::ir::features::AtomicUsage::Add);
        let parts = dw0p.meta.shape().dims::<2>()[0];
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| {
            eprintln!(
                "Fused backward+params: {} atomics; subgroup width {}; {} persistent groups",
                if native_add { "native f32" } else { "CAS" },
                subgroup_width,
                parts,
            );
        });
        if native_add {
            field_backward_fused_native_kernel::launch(
                &client,
                CubeCount::Static(parts as u32, 1, 1),
                CubeDim::new_1d(256),
                p.into_tensor_arg(),
                f.into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                m.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                dw0p.clone().into_tensor_arg(),
                dw1p.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *adjoint_offset as u32,
                *scale_cutoff as f32,
                *local_hash != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *replay_features != 0,
                *selective_weights != 0,
                *accumulate != 0,
                subgroup_width,
                *active_columns_only != 0,
                *scale_encoding != 0,
            );
        } else {
            field_backward_fused_cas_kernel::launch(
                &client,
                CubeCount::Static(parts as u32, 1, 1),
                CubeDim::new_1d(256),
                p.into_tensor_arg(),
                f.into_tensor_arg(),
                d.into_tensor_arg(),
                w.into_tensor_arg(),
                m.into_tensor_arg(),
                t.clone().into_tensor_arg(),
                dw0p.clone().into_tensor_arg(),
                dw1p.clone().into_tensor_arg(),
                *active as u32,
                *valid_n as u32,
                *aggregation_levels as u32,
                *adjoint_offset as u32,
                *scale_cutoff as f32,
                *local_hash != 0,
                *aggregate_hash != 0 && *aggregation_levels > 0,
                *replay_features != 0,
                *selective_weights != 0,
                *accumulate != 0,
                subgroup_width,
                *active_columns_only != 0,
                *scale_encoding != 0,
            );
        }
        handles.register_float_tensor::<CubeBackend>(&ot.id, t);
        handles.register_float_tensor::<CubeBackend>(&odw0.id, dw0p);
        handles.register_float_tensor::<CubeBackend>(&odw1.id, dw1p);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn field_backward_fused(
    inputs: [burn::tensor::Tensor<2>; 8],
    valid_n: usize,
    active: usize,
    local_hash: bool,
    aggregate_hash: bool,
    aggregation_levels: usize,
    replay_features: bool,
    adjoint_offset: usize,
    selective_weights: bool,
    accumulate: bool,
    active_columns_only: bool,
    scale_cutoff: f32,
) -> [burn::tensor::Tensor<2>; 3] {
    use burn::tensor::Tensor as BT;
    assert!(
        adjoint_offset <= inputs[2].dims()[0] && valid_n <= inputs[2].dims()[0] - adjoint_offset
    );
    assert!(valid_n <= inputs[1].dims()[0]);
    let scale_encoding = inputs[1].dims()[1] == 32;
    let inputs = inputs.map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [5usize, 6, 7]
        .into_iter()
        .map(|i| &inputs[i])
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_field_backward_fused",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::UInt(active as u64),
            ScalarIr::UInt(valid_n as u64),
            ScalarIr::UInt(local_hash as u64),
            ScalarIr::UInt(aggregate_hash as u64),
            ScalarIr::UInt(aggregation_levels as u64),
            ScalarIr::UInt(replay_features as u64),
            ScalarIr::UInt(adjoint_offset as u64),
            ScalarIr::UInt(selective_weights as u64),
            ScalarIr::UInt(accumulate as u64),
            ScalarIr::UInt(active_columns_only as u64),
            ScalarIr::UInt(scale_encoding as u64),
            ScalarIr::Float(scale_cutoff as f64),
        ],
    );
    let result: [_; 3] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            FieldBackwardFusedOp { desc },
        )
        .outputs();
    result.map(BT::<2>::from_primitive::<GpuBackend>)
}

#[cube(launch)]
fn reduce_mlp_partials_kernel(
    dw0_partials: &Tensor<f32>,
    dw1_partials: &Tensor<f32>,
    dw0_out: &mut Tensor<f32>,
    db0_out: &mut Tensor<f32>,
    dw1_out: &mut Tensor<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= 2112usize {
        terminate!();
    }
    let parts = dw0_partials.shape(0);
    let mut sum = 0.0_f32;
    if idx < 2048usize {
        for p in 0..parts {
            sum += dw0_partials[p * 2048 + idx];
        }
        let r = idx / 32;
        let c = idx % 32;
        if c < 31usize {
            dw0_out[r * 31 + c] = sum;
        } else {
            db0_out[r] = sum;
        }
    } else {
        let lane = idx - 2048usize;
        for p in 0..parts {
            sum += dw1_partials[p * 64 + lane];
        }
        dw1_out[lane] = sum;
    }
}

#[derive(Debug)]
struct ReduceMlpPartialsOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for ReduceMlpPartialsOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([p0, p1, dw0, db0, dw1], [odw0, odb0, odw1]) = self.desc.as_fixed();
        let [p0, p1, dw0, db0, dw1] =
            [p0, p1, dw0, db0, dw1].map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let client = p0.client.clone();
        let _timing = crate::diagnostics::start(&client, "reduce_mlp_partials");
        reduce_mlp_partials_kernel::launch(
            &client,
            CubeCount::Static(17, 1, 1),
            CubeDim::new_1d(128),
            p0.into_tensor_arg(),
            p1.into_tensor_arg(),
            dw0.clone().into_tensor_arg(),
            db0.clone().into_tensor_arg(),
            dw1.clone().into_tensor_arg(),
        );
        handles.register_float_tensor::<CubeBackend>(&odw0.id, dw0);
        handles.register_float_tensor::<CubeBackend>(&odb0.id, db0);
        handles.register_float_tensor::<CubeBackend>(&odw1.id, dw1);
        Ok(())
    }
}

pub fn reduce_mlp_partials(
    dw0_partials: burn::tensor::Tensor<2>,
    dw1_partials: burn::tensor::Tensor<2>,
) -> [burn::tensor::Tensor<2>; 3] {
    use burn::tensor::Tensor as BT;
    let device = dw0_partials.device();
    let inputs = [
        dw0_partials,
        dw1_partials,
        BT::empty([64, 31], &device),
        BT::empty([1, 64], &device),
        BT::empty([1, 64], &device),
    ]
    .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [2usize, 3, 4]
        .into_iter()
        .map(|i| &inputs[i])
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_reduce_mlp_partials",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![],
    );
    let result: [_; 3] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            ReduceMlpPartialsOp { desc },
        )
        .outputs();
    result.map(BT::<2>::from_primitive::<GpuBackend>)
}

#[cube]
fn adam_scalar_step(
    state: &mut Tensor<f32>,
    idx: usize,
    grad: f32,
    inv_bc1: f32,
    inv_bc2: f32,
    lr: f32,
) -> f32 {
    let old_val = state[idx];
    let old_m = state[2179usize + idx];
    let old_v = state[4358usize + idx];
    let m = old_m * 0.9_f32 + grad * 0.1_f32;
    let v = old_v * 0.999_f32 + grad * grad * 0.001_f32;
    let m_hat = m * inv_bc1;
    let v_hat = v * inv_bc2;
    let new_val = old_val - m_hat / (f32::sqrt(v_hat) + 1e-8_f32) * lr;
    state[idx] = new_val;
    state[2179usize + idx] = m;
    state[4358usize + idx] = v;
    new_val
}

// Single-workgroup fused MLP WeightNorm backward + 7-parameter Adam update +
// WeightNorm forward + transposed `packed_mlp` packing.
// Replaces ~118 Burn ops and ~20 tiny GPU launches per step with one < 5 us launch.
// `mlp_state` `[3, 2179]` rows: 0 = value, 1 = m, 2 = v.
// Offsets within each 2179-float row:
//   0..1984:    lin0.weight_v [64, 31]
//   1984..2048: lin0.weight_g [64, 1]
//   2048..2112: lin0.bias     [1, 64]
//   2112..2176: lin1.weight_v [1, 64]
//   2176:       lin1.weight_g [1, 1]
//   2177:       lin1.bias     [1, 1]
//   2178:       variance      [1, 1]
#[cube(launch)]
fn fused_mlp_adam_pack_kernel(
    mlp_state: &mut Tensor<f32>,
    dw0: &Tensor<f32>,
    db0: &Tensor<f32>,
    dw1: &Tensor<f32>,
    stats: &Tensor<f32>,
    packed_mlp: &mut Tensor<f32>,
    variance_out: &mut Tensor<f32>,
    inv_bc1: f32,
    inv_bc2: f32,
    lr: f32,
) {
    let unit = usize::cast_from(UNIT_POS);
    let mut sh_norm = Shared::new_slice(64usize);
    let mut sh_dot = Shared::new_slice(64usize);
    let mut sh_meta = Shared::new_slice(4usize);

    // Threads 0..64: Layer 0 (64 rows of 31 cols) — independent per row!
    if unit < 64usize {
        let r = unit;
        let row_off = r * 31;
        let mut norm_sq = 0.0_f32;
        let mut dot = 0.0_f32;
        for c in 0..31usize {
            let v = mlp_state[row_off + c];
            let g = dw0[row_off + c];
            norm_sq += v * v;
            dot += g * v;
        }
        let norm = f32::sqrt(norm_sq);
        let old_g = mlp_state[1984usize + r];
        let dg = dot / norm;
        let dot_over_nsq = dot / norm_sq;
        let scale_v = old_g / norm;

        let new_g = adam_scalar_step(mlp_state, 1984usize + r, dg, inv_bc1, inv_bc2, lr);
        let new_b0 = adam_scalar_step(mlp_state, 2048usize + r, db0[r], inv_bc1, inv_bc2, lr);
        packed_mlp[1984usize + r] = new_b0;

        let mut new_norm_sq = 0.0_f32;
        for c in 0..31usize {
            let v = mlp_state[row_off + c];
            let dv = (dw0[row_off + c] - v * dot_over_nsq) * scale_v;
            let nv = adam_scalar_step(mlp_state, row_off + c, dv, inv_bc1, inv_bc2, lr);
            new_norm_sq += nv * nv;
        }
        let new_scale = new_g / f32::sqrt(new_norm_sq);
        for c in 0..31usize {
            packed_mlp[c * 64 + r] = mlp_state[row_off + c] * new_scale;
        }
    }

    // Threads 64..128: Layer 1 (1 row of 64 cols) + lin1.bias + variance.
    if unit >= 64usize {
        let k = unit - 64usize;
        let v1 = mlp_state[2112usize + k];
        sh_norm[k] = v1 * v1;
        sh_dot[k] = dw1[k] * v1;
    }
    sync_cube();

    if unit == 64usize {
        let mut norm_sq = 0.0_f32;
        let mut dot = 0.0_f32;
        for k in 0..64usize {
            norm_sq += sh_norm[k];
            dot += sh_dot[k];
        }
        let norm = f32::sqrt(norm_sq);
        let old_g1 = mlp_state[2176usize];
        let dg1 = dot / norm;
        let new_g1 = adam_scalar_step(mlp_state, 2176usize, dg1, inv_bc1, inv_bc2, lr);
        let new_b1 = adam_scalar_step(mlp_state, 2177usize, stats[4], inv_bc1, inv_bc2, lr);
        let new_var = adam_scalar_step(mlp_state, 2178usize, stats[3], inv_bc1, inv_bc2, lr);
        packed_mlp[2112usize] = new_b1;
        variance_out[0] = new_var;
        sh_meta[0] = dot / norm_sq;
        sh_meta[1] = old_g1 / norm;
        sh_meta[2] = new_g1;
    }
    sync_cube();

    if unit >= 64usize {
        let k = unit - 64usize;
        let v1 = mlp_state[2112usize + k];
        let dv1 = (dw1[k] - v1 * sh_meta[0]) * sh_meta[1];
        let nv1 = adam_scalar_step(mlp_state, 2112usize + k, dv1, inv_bc1, inv_bc2, lr);
        sh_norm[k] = nv1 * nv1;
    }
    sync_cube();

    if unit == 64usize {
        let mut new_norm_sq = 0.0_f32;
        for k in 0..64usize {
            new_norm_sq += sh_norm[k];
        }
        sh_meta[3] = sh_meta[2] / f32::sqrt(new_norm_sq);
    }
    sync_cube();

    if unit >= 64usize {
        let k = unit - 64usize;
        packed_mlp[2048usize + k] = mlp_state[2112usize + k] * sh_meta[3];
    }
}

#[derive(Debug)]
struct FusedMlpStepOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for FusedMlpStepOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([st, dw0, db0, dw1, stats, pmlp, vout], [ost, opmlp, ovout]) = self.desc.as_fixed();
        let [
            ScalarIr::Float(inv_bc1),
            ScalarIr::Float(inv_bc2),
            ScalarIr::Float(lr),
        ] = self.desc.scalars.as_slice()
        else {
            panic!("invalid fused mlp step scalars")
        };
        let [st, dw0, db0, dw1, stats, pmlp, vout] = [st, dw0, db0, dw1, stats, pmlp, vout]
            .map(|x| handles.get_float_tensor::<CubeBackend>(x));
        let st = if st.can_mut() { st } else { st.copy() };
        let client = st.client.clone();
        let _timing = crate::diagnostics::start(&client, "fused_mlp_adam_pack");
        fused_mlp_adam_pack_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(128),
            st.clone().into_tensor_arg(),
            dw0.into_tensor_arg(),
            db0.into_tensor_arg(),
            dw1.into_tensor_arg(),
            stats.into_tensor_arg(),
            pmlp.clone().into_tensor_arg(),
            vout.clone().into_tensor_arg(),
            *inv_bc1 as f32,
            *inv_bc2 as f32,
            *lr as f32,
        );
        handles.register_float_tensor::<CubeBackend>(&ost.id, st);
        handles.register_float_tensor::<CubeBackend>(&opmlp.id, pmlp);
        handles.register_float_tensor::<CubeBackend>(&ovout.id, vout);
        Ok(())
    }
}

pub fn fused_mlp_step(
    mlp_state: burn::tensor::Tensor<2>,
    dw0: burn::tensor::Tensor<2>,
    db0: burn::tensor::Tensor<2>,
    dw1: burn::tensor::Tensor<2>,
    stats: burn::tensor::Tensor<2>,
    inv_bc1: f64,
    inv_bc2: f64,
    lr: f64,
) -> [burn::tensor::Tensor<2>; 3] {
    use burn::tensor::Tensor as BT;
    let device = mlp_state.device();
    let inputs = [
        mlp_state,
        dw0,
        db0,
        dw1,
        stats,
        BT::empty([1, 2113], &device),
        BT::empty([1, 1], &device),
    ]
    .map(|t| t.inner().try_into_primitive::<GpuBackend>().unwrap());
    let client = inputs[0].client.clone();
    let outputs: Vec<_> = [0usize, 5, 6]
        .into_iter()
        .map(|i| &inputs[i])
        .map(|t| TensorIr::uninit(client.create_empty_handle(), t.shape.clone(), t.dtype))
        .collect();
    let desc = CustomOpIr::with_scalars(
        "supernormal_fused_mlp_step",
        &inputs.map(|t| t.into_ir()),
        &outputs,
        vec![
            ScalarIr::Float(inv_bc1),
            ScalarIr::Float(inv_bc2),
            ScalarIr::Float(lr),
        ],
    );
    let result: [_; 3] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            FusedMlpStepOp { desc },
        )
        .outputs();
    result.map(BT::<2>::from_primitive::<GpuBackend>)
}

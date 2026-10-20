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
) {
    let point_idx = ABSOLUTE_POS;
    if point_idx >= points.shape(0) {
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
        let [ScalarIr::UInt(active)] = self.desc.scalars.as_slice() else {
            panic!("Invalid scalar arguments for hash_grid_packed_backward")
        };
        let points = handles.get_float_tensor::<CubeBackend>(p_ir);
        let d_features = handles.get_float_tensor::<CubeBackend>(df_ir);
        let metadata = handles.get_float_tensor::<CubeBackend>(m_ir);
        let table_grad = handles.get_float_tensor::<CubeBackend>(tg_ir);
        let client = points.client.clone();
        let n = points.meta.shape().dims::<2>()[0];
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
        vec![ScalarIr::UInt(active as u64)],
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
use crate::{field::Field, mc_table::TRIANGLES};
use anyhow::{Result, ensure};
use std::{
    io::{BufWriter, Write},
    path::Path,
};
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn polygonize_cell(
    x: usize,
    y: usize,
    z: usize,
    res: usize,
    inv_res_m1: f32,
    scale: f32,
    offset: [f32; 3],
    lower: &[f32],
    upper: &[f32],
    xy_lower: &mut [u32],
    xy_upper: &mut [u32],
    z_edges: &mut [u32],
    dirty_xy_lower: Option<&mut Vec<usize>>,
    dirty_xy_upper: Option<&mut Vec<usize>>,
    dirty_z_edges: Option<&mut Vec<usize>>,
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
) {
    let row0 = y * res;
    let row1 = (y + 1) * res;
    let v0 = lower[row0 + x];
    let v1 = lower[row0 + x + 1];
    let v2 = lower[row1 + x + 1];
    let v3 = lower[row1 + x];
    let v4 = upper[row0 + x];
    let v5 = upper[row0 + x + 1];
    let v6 = upper[row1 + x + 1];
    let v7 = upper[row1 + x];
    let case = (if v0 < 0.0 { 1usize } else { 0 })
        | (if v1 < 0.0 { 2 } else { 0 })
        | (if v2 < 0.0 { 4 } else { 0 })
        | (if v3 < 0.0 { 8 } else { 0 })
        | (if v4 < 0.0 { 16 } else { 0 })
        | (if v5 < 0.0 { 32 } else { 0 })
        | (if v6 < 0.0 { 64 } else { 0 })
        | (if v7 < 0.0 { 128 } else { 0 });
    if case == 0 || case == 255 {
        return;
    }
    let values = [v0, v1, v2, v3, v4, v5, v6, v7];
    let coords = CORNERS.map(|c| [x + c[0], y + c[1], z + c[2]]);
    let mut dirty_xy_lower = dirty_xy_lower;
    let mut dirty_xy_upper = dirty_xy_upper;
    let mut dirty_z_edges = dirty_z_edges;
    for tri in TRIANGLES[case].chunks_exact(3) {
        if tri[0] < 0 {
            break;
        }
        let mut face = [0u32; 3];
        for j in 0..3 {
            let e_id = tri[j] as usize;
            let [a, b] = EDGES[e_id];
            let (entry_val, table_kind, slot_idx) = if e_id < 4 {
                let axis = if e_id == 0 || e_id == 2 { 0 } else { 1 };
                let ex = x + if e_id == 1 { 1 } else { 0 };
                let ey = y + if e_id == 2 { 1 } else { 0 };
                let idx = (ey * res + ex) * 2 + axis;
                (xy_lower[idx], 0u8, idx)
            } else if e_id < 8 {
                let axis = if e_id == 4 || e_id == 6 { 0 } else { 1 };
                let ex = x + if e_id == 5 { 1 } else { 0 };
                let ey = y + if e_id == 6 { 1 } else { 0 };
                let idx = (ey * res + ex) * 2 + axis;
                (xy_upper[idx], 1u8, idx)
            } else {
                let ex = x + if e_id == 9 || e_id == 10 { 1 } else { 0 };
                let ey = y + if e_id == 10 || e_id == 11 { 1 } else { 0 };
                let idx = ey * res + ex;
                (z_edges[idx], 2u8, idx)
            };
            let id = if entry_val == u32::MAX {
                let ca = coords[a];
                let cb = coords[b];
                let t = if (values[b] - values[a]).abs() < 1e-20 {
                    0.5
                } else {
                    -values[a] / (values[b] - values[a])
                };
                let v = std::array::from_fn(|d| {
                    ((-1.
                        + 2. * (ca[d] as f32 + (cb[d] as f32 - ca[d] as f32) * t) * inv_res_m1)
                        * scale)
                        + offset[d]
                });
                let new_id = vertices.len() as u32;
                vertices.push(v);
                match table_kind {
                    0 => {
                        xy_lower[slot_idx] = new_id;
                        if let Some(d) = dirty_xy_lower.as_deref_mut() {
                            d.push(slot_idx);
                        }
                    }
                    1 => {
                        xy_upper[slot_idx] = new_id;
                        if let Some(d) = dirty_xy_upper.as_deref_mut() {
                            d.push(slot_idx);
                        }
                    }
                    _ => {
                        z_edges[slot_idx] = new_id;
                        if let Some(d) = dirty_z_edges.as_deref_mut() {
                            d.push(slot_idx);
                        }
                    }
                }
                new_id
            } else {
                entry_val
            };
            face[j] = id;
        }
        faces.push(face);
    }
}

fn extract_sparse_cells(
    field: &Field,
    res: usize,
    query: usize,
    scale: f32,
    offset: [f32; 3],
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
) -> Result<()> {
    const BLOCK_SIZE: usize = 8;
    let nb = (res - 1).div_ceil(BLOCK_SIZE);
    let c_res = nb + 1;
    let inv_res_m1 = 1.0_f32 / (res - 1) as f32;
    let cell_size = 2.0_f32 * inv_res_m1;
    let block_diag = (BLOCK_SIZE as f32) * cell_size * 3.0_f32.sqrt();
    // Conservative Lipschitz safety band (3x block diagonal + 1-ring dilation
    // around sign-changing blocks) preserves exact surface topology while
    // skipping ~95% of empty 3D blocks.
    let band_margin = 3.0_f32 * block_diag;

    let coarse_pts = crate::sampling_gpu::mesh_coarse_points(res, BLOCK_SIZE, &field.device);
    let coarse_sdf = (-field.evaluate_device_tensor(coarse_pts, query)).try_into_vec_as::<f32>()?;
    let c_plane = c_res * c_res;

    let mut active = vec![false; nb * nb * nb];
    let mut sign_change = vec![false; nb * nb * nb];
    for bz in 0..nb {
        let cz0 = bz * c_plane;
        let cz1 = (bz + 1) * c_plane;
        for by in 0..nb {
            let cy0 = by * c_res;
            let cy1 = (by + 1) * c_res;
            for bx in 0..nb {
                let c0 = coarse_sdf[cz0 + cy0 + bx];
                let c1 = coarse_sdf[cz0 + cy0 + bx + 1];
                let c2 = coarse_sdf[cz0 + cy1 + bx];
                let c3 = coarse_sdf[cz0 + cy1 + bx + 1];
                let c4 = coarse_sdf[cz1 + cy0 + bx];
                let c5 = coarse_sdf[cz1 + cy0 + bx + 1];
                let c6 = coarse_sdf[cz1 + cy1 + bx];
                let c7 = coarse_sdf[cz1 + cy1 + bx + 1];
                let min_v = c0
                    .min(c1)
                    .min(c2)
                    .min(c3)
                    .min(c4)
                    .min(c5)
                    .min(c6)
                    .min(c7);
                let max_v = c0
                    .max(c1)
                    .max(c2)
                    .max(c3)
                    .max(c4)
                    .max(c5)
                    .max(c6)
                    .max(c7);
                let has_sign = min_v < 0.0 && max_v >= 0.0;
                let min_abs = c0
                    .abs()
                    .min(c1.abs())
                    .min(c2.abs())
                    .min(c3.abs())
                    .min(c4.abs())
                    .min(c5.abs())
                    .min(c6.abs())
                    .min(c7.abs());
                let idx = (bz * nb + by) * nb + bx;
                if has_sign {
                    sign_change[idx] = true;
                    active[idx] = true;
                } else if min_abs < band_margin {
                    active[idx] = true;
                }
            }
        }
    }
    for bz in 0..nb {
        for by in 0..nb {
            for bx in 0..nb {
                if !sign_change[(bz * nb + by) * nb + bx] {
                    continue;
                }
                let z_lo = bz.saturating_sub(1);
                let z_hi = (bz + 1).min(nb - 1);
                let y_lo = by.saturating_sub(1);
                let y_hi = (by + 1).min(nb - 1);
                let x_lo = bx.saturating_sub(1);
                let x_hi = (bx + 1).min(nb - 1);
                for nz in z_lo..=z_hi {
                    for ny in y_lo..=y_hi {
                        for nx in x_lo..=x_hi {
                            active[(nz * nb + ny) * nb + nx] = true;
                        }
                    }
                }
            }
        }
    }

    struct BandPlan {
        z0: usize,
        z1: usize,
        blocks: Vec<(usize, usize)>,
        xy_indices: Vec<usize>,
        xy_flat: Vec<f32>,
    }

    let plane_size = res * res;
    let mut seen_stamp = vec![0u32; plane_size];
    let mut plans: Vec<BandPlan> = Vec::new();
    let mut total_active_blocks = 0usize;
    for bz in 0..nb {
        let mut blocks = Vec::new();
        for by in 0..nb {
            let row = (bz * nb + by) * nb;
            for bx in 0..nb {
                if active[row + bx] {
                    blocks.push((bx, by));
                }
            }
        }
        if blocks.is_empty() {
            continue;
        }
        total_active_blocks += blocks.len();
        let stamp = (bz as u32) + 1;
        let mut xy_indices = Vec::with_capacity(blocks.len() * 72);
        let mut xy_flat = Vec::with_capacity(blocks.len() * 144);
        for &(bx, by) in &blocks {
            let x0 = bx * BLOCK_SIZE;
            let x1 = (x0 + BLOCK_SIZE).min(res - 1);
            let y0 = by * BLOCK_SIZE;
            let y1 = (y0 + BLOCK_SIZE).min(res - 1);
            for y in y0..=y1 {
                let row_off = y * res;
                for x in x0..=x1 {
                    let p_idx = row_off + x;
                    if seen_stamp[p_idx] != stamp {
                        seen_stamp[p_idx] = stamp;
                        xy_indices.push(p_idx);
                        xy_flat.push(x as f32);
                        xy_flat.push(y as f32);
                    }
                }
            }
        }
        let z0 = bz * BLOCK_SIZE;
        let z1 = (z0 + BLOCK_SIZE).min(res - 1);
        plans.push(BandPlan {
            z0,
            z1,
            blocks,
            xy_indices,
            xy_flat,
        });
    }
    println!(
        "Sparse mesh probe ({c_res}^3): {total_active_blocks}/{} active blocks ({:.2}%), {} bands",
        nb * nb * nb,
        100.0 * total_active_blocks as f64 / (nb * nb * nb) as f64,
        plans.len(),
    );
    if plans.is_empty() {
        return Ok(());
    }

    let enqueue_band = |plan: &BandPlan| -> burn::tensor::Tensor<2> {
        let k = plan.xy_indices.len();
        let z_count = plan.z1 - plan.z0 + 1;
        let xy_tensor = crate::field::tensor(plan.xy_flat.clone(), [k, 2], &field.device);
        let pts = crate::sampling_gpu::mesh_sparse_band_points(xy_tensor, res, plan.z0, z_count);
        let neg_sdf = -field.evaluate_device_tensor(pts, query);
        let _ = field.device.flush();
        neg_sdf
    };

    let mut pending = Some(enqueue_band(&plans[0]));
    let mut lower = vec![0.0_f32; plane_size];
    let mut upper = vec![0.0_f32; plane_size];
    let mut xy_lower = vec![u32::MAX; 2 * plane_size];
    let mut xy_upper = vec![u32::MAX; 2 * plane_size];
    let mut z_edges = vec![u32::MAX; plane_size];
    let mut dirty_xy_lower: Vec<usize> = Vec::with_capacity(65536);
    let mut dirty_xy_upper: Vec<usize> = Vec::with_capacity(65536);
    let mut dirty_z_edges: Vec<usize> = Vec::with_capacity(65536);
    let mut prev_z: Option<usize> = None;

    for i in 0..plans.len() {
        let cur_tensor = pending.take().expect("queued sparse band");
        if i + 1 < plans.len() {
            pending = Some(enqueue_band(&plans[i + 1]));
        }
        let band_sdf = cur_tensor.try_into_vec_as::<f32>()?;
        let plan = &plans[i];
        let k = plan.xy_indices.len();
        let z_steps = plan.z1 - plan.z0;

        if let Some(pz) = prev_z {
            if plan.z0 != pz + 1 {
                for &idx in &dirty_xy_lower {
                    xy_lower[idx] = u32::MAX;
                }
                dirty_xy_lower.clear();
            }
        }

        for (j, &p_idx) in plan.xy_indices.iter().enumerate() {
            lower[p_idx] = band_sdf[j];
        }
        for lz in 0..z_steps {
            let z = plan.z0 + lz;
            let plane_off = (lz + 1) * k;
            for (j, &p_idx) in plan.xy_indices.iter().enumerate() {
                upper[p_idx] = band_sdf[plane_off + j];
            }
            for &(bx, by) in &plan.blocks {
                let x0 = bx * BLOCK_SIZE;
                let x1 = (x0 + BLOCK_SIZE).min(res - 1);
                let y0 = by * BLOCK_SIZE;
                let y1 = (y0 + BLOCK_SIZE).min(res - 1);
                for y in y0..y1 {
                    for x in x0..x1 {
                        polygonize_cell(
                            x,
                            y,
                            z,
                            res,
                            inv_res_m1,
                            scale,
                            offset,
                            &lower,
                            &upper,
                            &mut xy_lower,
                            &mut xy_upper,
                            &mut z_edges,
                            Some(&mut dirty_xy_lower),
                            Some(&mut dirty_xy_upper),
                            Some(&mut dirty_z_edges),
                            vertices,
                            faces,
                        );
                    }
                }
            }
            std::mem::swap(&mut lower, &mut upper);
            for &idx in &dirty_xy_lower {
                xy_lower[idx] = u32::MAX;
            }
            dirty_xy_lower.clear();
            for &idx in &dirty_z_edges {
                z_edges[idx] = u32::MAX;
            }
            dirty_z_edges.clear();
            std::mem::swap(&mut xy_lower, &mut xy_upper);
            std::mem::swap(&mut dirty_xy_lower, &mut dirty_xy_upper);
            prev_z = Some(z);
        }
    }
    Ok(())
}

pub fn extract(
    field: &Field,
    path: &Path,
    res: usize,
    query: usize,
    scale: f32,
    offset: [f32; 3],
) -> Result<()> {
    ensure!(res >= 2 && query > 0, "invalid mesh resolution/query size");
    let mut vertices: Vec<[f32; 3]> = vec![];
    let mut faces: Vec<[u32; 3]> = vec![];

    if res >= 128 {
        extract_sparse_cells(field, res, query, scale, offset, &mut vertices, &mut faces)?;
    } else {
        let plane_size = res * res;
        let mut xy_lower = vec![u32::MAX; 2 * plane_size];
        let mut xy_upper = vec![u32::MAX; 2 * plane_size];
        let mut z_edges = vec![u32::MAX; plane_size];
        let inv_res_m1 = 1.0_f32 / (res - 1) as f32;

        const BATCH_SLABS: usize = 4;
        let enqueue_batch = |z_start: usize| -> Option<(usize, burn::tensor::Tensor<2>)> {
            if z_start >= res {
                return None;
            }
            let count = (res - z_start).min(BATCH_SLABS);
            let pts =
                crate::sampling_gpu::mesh_slab_points_batch(res, z_start, count, &field.device);
            let sdf = field.evaluate_device_tensor(pts, query);
            let neg_sdf = -sdf;
            let _ = field.device.flush();
            Some((count, neg_sdf))
        };

        let first_slab = crate::sampling_gpu::mesh_slab_points(res, 0, &field.device);
        let first_sdf = -field.evaluate_device_tensor(first_slab, query);
        let _ = field.device.flush();
        let mut next_batch = enqueue_batch(1);
        let mut lower = first_sdf.try_into_vec_as::<f32>()?;
        let mut batch_buf: Vec<f32> = Vec::new();
        let mut batch_offset = 0usize;
        let mut batch_count = 0usize;
        let mut next_z_to_enqueue = 1 + BATCH_SLABS;

        for z in 0..res - 1 {
            if batch_offset >= batch_count {
                let (count, tensor) = next_batch.take().expect("queued slab batch");
                if next_z_to_enqueue < res {
                    next_batch = enqueue_batch(next_z_to_enqueue);
                    next_z_to_enqueue += BATCH_SLABS;
                }
                batch_buf = tensor.try_into_vec_as::<f32>()?;
                batch_offset = 0;
                batch_count = count;
            }
            let upper = &batch_buf[batch_offset * plane_size..(batch_offset + 1) * plane_size];
            batch_offset += 1;

            for y in 0..res - 1 {
                for x in 0..res - 1 {
                    polygonize_cell(
                        x,
                        y,
                        z,
                        res,
                        inv_res_m1,
                        scale,
                        offset,
                        &lower,
                        upper,
                        &mut xy_lower,
                        &mut xy_upper,
                        &mut z_edges,
                        None,
                        None,
                        None,
                        &mut vertices,
                        &mut faces,
                    );
                }
            }
            lower.copy_from_slice(upper);
            std::mem::swap(&mut xy_lower, &mut xy_upper);
            xy_upper.fill(u32::MAX);
            z_edges.fill(u32::MAX);
            if z % 32 == 0 {
                println!(
                    "mesh slab {}/{} vertices={} faces={}",
                    z + 1,
                    res - 1,
                    vertices.len(),
                    faces.len()
                );
            }
        }
    }
    // Match reference's largest connected component, measured by triangle count.
    if !faces.is_empty() {
        let mut parent: Vec<usize> = (0..vertices.len()).collect();
        fn root(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for f in &faces {
            let a = root(&mut parent, f[0] as usize);
            for &v in &f[1..] {
                let b = root(&mut parent, v as usize);
                parent[b] = a;
            }
        }
        let mut counts = vec![0usize; vertices.len()];
        let mut largest = 0usize;
        let mut max_count = 0usize;
        for f in &faces {
            let r = root(&mut parent, f[0] as usize);
            counts[r] += 1;
            if counts[r] > max_count {
                max_count = counts[r];
                largest = r;
            }
        }
        faces.retain(|f| root(&mut parent, f[0] as usize) == largest);
        let mut remap = vec![u32::MAX; vertices.len()];
        let mut compact = vec![];
        for f in &mut faces {
            for v in f {
                let i = *v as usize;
                if remap[i] == u32::MAX {
                    remap[i] = compact.len() as u32;
                    compact.push(vertices[i]);
                }
                *v = remap[i];
            }
        }
        vertices = compact;
    }
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
        vertices.len(),
        faces.len()
    )?;
    out.write_all(bytemuck::cast_slice(&vertices))?;
    let mut face_buf = Vec::with_capacity(faces.len() * 13);
    for f in &faces {
        face_buf.push(3u8);
        face_buf.extend_from_slice(bytemuck::cast_slice(f));
    }
    out.write_all(&face_buf)?;
    out.flush()?;
    println!(
        "Wrote {} vertices, {} triangles to {}",
        vertices.len(),
        faces.len(),
        path.display()
    );
    Ok(())
}

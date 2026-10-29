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
        let pts = crate::sampling_gpu::mesh_slab_points_batch(res, z_start, count, &field.device);
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
            let row0 = y * res;
            let row1 = (y + 1) * res;
            for x in 0..res - 1 {
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
                    continue;
                }
                let values = [v0, v1, v2, v3, v4, v5, v6, v7];
                let coords = CORNERS.map(|c| [x + c[0], y + c[1], z + c[2]]);
                for tri in TRIANGLES[case].chunks_exact(3) {
                    if tri[0] < 0 {
                        break;
                    }
                    let mut face = [0u32; 3];
                    for j in 0..3 {
                        let e_id = tri[j] as usize;
                        let [a, b] = EDGES[e_id];
                        let entry = if e_id < 4 {
                            let axis = if e_id == 0 || e_id == 2 { 0 } else { 1 };
                            let ex = x + if e_id == 1 { 1 } else { 0 };
                            let ey = y + if e_id == 2 { 1 } else { 0 };
                            &mut xy_lower[(ey * res + ex) * 2 + axis]
                        } else if e_id < 8 {
                            let axis = if e_id == 4 || e_id == 6 { 0 } else { 1 };
                            let ex = x + if e_id == 5 { 1 } else { 0 };
                            let ey = y + if e_id == 6 { 1 } else { 0 };
                            &mut xy_upper[(ey * res + ex) * 2 + axis]
                        } else {
                            let ex = x + if e_id == 9 || e_id == 10 { 1 } else { 0 };
                            let ey = y + if e_id == 10 || e_id == 11 { 1 } else { 0 };
                            &mut z_edges[ey * res + ex]
                        };
                        if *entry == u32::MAX {
                            let ca = coords[a];
                            let cb = coords[b];
                            let t = if (values[b] - values[a]).abs() < 1e-20 {
                                0.5
                            } else {
                                -values[a] / (values[b] - values[a])
                            };
                            let v = std::array::from_fn(|d| {
                                ((-1.
                                    + 2. * (ca[d] as f32 + (cb[d] as f32 - ca[d] as f32) * t)
                                        * inv_res_m1)
                                    * scale)
                                    + offset[d]
                            });
                            let id = vertices.len() as u32;
                            vertices.push(v);
                            *entry = id;
                        }
                        face[j] = *entry;
                    }
                    faces.push(face);
                }
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

use crate::{field::Field, mc_table::TRIANGLES};
use anyhow::{Result, ensure};
use std::{
    collections::HashMap,
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
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    let layer = |z: usize| -> Result<Vec<f32>> {
        let pts: Vec<_> = (0..res * res)
            .map(|i| [i % res, i / res, z].map(|v| -1. + 2. * v as f32 / (res - 1) as f32))
            .collect();
        Ok(field
            .evaluate(&pts, query)?
            .into_iter()
            .map(|v| -v)
            .collect())
    };
    let mut lower = layer(0)?;
    for z in 0..res - 1 {
        let upper = layer(z + 1)?;
        for y in 0..res - 1 {
            for x in 0..res - 1 {
                let coords = CORNERS.map(|c| [x + c[0], y + c[1], z + c[2]]);
                let values = CORNERS.map(|c| {
                    if c[2] == 0 {
                        lower[(y + c[1]) * res + x + c[0]]
                    } else {
                        upper[(y + c[1]) * res + x + c[0]]
                    }
                });
                let case = (0..8).fold(0usize, |bits, i| {
                    bits | if values[i] < 0. { 1 << i } else { 0 }
                });
                for tri in TRIANGLES[case].chunks_exact(3) {
                    if tri[0] < 0 {
                        break;
                    }
                    let mut face = [0; 3];
                    for j in 0..3 {
                        let [a, b] = EDGES[tri[j] as usize];
                        let ca = coords[a];
                        let cb = coords[b];
                        let ia = (ca[2] * res + ca[1]) * res + ca[0];
                        let ib = (cb[2] * res + cb[1]) * res + cb[0];
                        let key = (ia.min(ib), ia.max(ib));
                        face[j] = *edges.entry(key).or_insert_with(|| {
                            let t = if (values[b] - values[a]).abs() < 1e-20 {
                                0.5
                            } else {
                                -values[a] / (values[b] - values[a])
                            };
                            let v = std::array::from_fn(|d| {
                                ((-1.
                                    + 2. * (ca[d] as f32 + (cb[d] as f32 - ca[d] as f32) * t)
                                        / (res - 1) as f32)
                                    * scale)
                                    + offset[d]
                            });
                            let id = vertices.len() as u32;
                            vertices.push(v);
                            id
                        });
                    }
                    faces.push(face);
                }
            }
        }
        lower = upper;
        // Keep only edges that may be reused by the next slab.
        edges.retain(|&(a, b), _| a / (res * res) >= z + 1 && b / (res * res) >= z + 1);
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
        let mut counts = HashMap::new();
        for f in &faces {
            *counts
                .entry(root(&mut parent, f[0] as usize))
                .or_insert(0usize) += 1;
        }
        let largest = *counts.iter().max_by_key(|(_, n)| **n).unwrap().0;
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
    for v in &vertices {
        for x in v {
            out.write_all(&x.to_le_bytes())?;
        }
    }
    for f in &faces {
        out.write_all(&[3])?;
        for i in f {
            out.write_all(&i.to_le_bytes())?;
        }
    }
    out.flush()?;
    println!(
        "Wrote {} vertices, {} triangles to {}",
        vertices.len(),
        faces.len(),
        path.display()
    );
    Ok(())
}

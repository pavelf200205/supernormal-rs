//! Packed CPU reference compositor with analytical derivatives. Field execution is on Burn.
//! This also serves as the oracle for portable GPU compositor kernels.

use crate::{
    data::{Patch, distance, dot, point},
    field::Field,
};
use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Occupancy {
    pub resolution: usize,
    pub values: Vec<f32>,
    pub flags: Vec<bool>,
    #[serde(skip)]
    pub device_values: Option<burn::tensor::Tensor<2>>,
    #[serde(skip)]
    pub device_flags: Option<burn::tensor::Tensor<2>>,
    #[serde(skip)]
    pub cpu_update: bool,
}

impl Occupancy {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            values: vec![0.; resolution.pow(3)],
            flags: vec![false; resolution.pow(3)],
            device_values: None,
            device_flags: None,
            cpu_update: false,
        }
    }

    pub fn sync_values(&mut self) -> Result<()> {
        if let Some(values) = self.device_values.take() {
            self.values = values.try_into_vec_as::<f32>()?;
        }
        Ok(())
    }

    pub fn occupied(&self, p: [f32; 3]) -> bool {
        if p.iter().any(|x| *x < -1. || *x > 1.) {
            return false;
        }
        let r = self.resolution;
        let c = p.map(|x| (((x + 1.) * 0.5 * r as f32) as usize).min(r - 1));
        self.flags[(c[0] * r + c[1]) * r + c[2]]
    }

    pub fn update(
        &mut self,
        field: &Field,
        step: usize,
        query: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let n = self.values.len();
        let ids: Vec<usize> = if step < 256 {
            (0..n).collect()
        } else {
            let mut ids: Vec<_> = (0..n / 4).map(|_| rng.random_range(0..n)).collect();
            let occupied: Vec<_> = self
                .flags
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| x.then_some(i))
                .collect();
            if occupied.len() > n / 4 {
                ids.extend((0..n / 4).map(|_| occupied[rng.random_range(0..occupied.len())]));
            } else {
                ids.extend(occupied);
            }
            ids
        };
        let r = self.resolution;

        let points: Vec<[f32; 3]> = ids
            .iter()
            .map(|&i| {
                let c = [i / (r * r), (i / r) % r, i % r];
                c.map(|v| ((v as f32 + rng.random::<f32>()) / r as f32) * 2. - 1.)
            })
            .collect();

        if !self.cpu_update {
            use crate::field::tensor;
            use burn::tensor::Tensor;
            anyhow::ensure!(
                ids.len() < (1 << 24),
                "occupancy update exceeds exact index range"
            );
            let sdf = field.evaluate_device(&points, query);
            let old = self
                .device_values
                .take()
                .unwrap_or_else(|| tensor(self.values.clone(), [n, 1], &field.device));
            let mut heads = vec![-1.0_f32; n];
            let mut next = vec![-1.0_f32; ids.len()];
            for (index, &id) in ids.iter().enumerate().rev() {
                next[index] = heads[id];
                heads[id] = index as f32;
            }
            let [values] = crate::sampling_gpu::occupancy([
                sdf,
                tensor(heads, [n, 1], &field.device),
                tensor(next, [ids.len(), 1], &field.device),
                old,
                Tensor::zeros([n, 1], &field.device),
            ]);
            let threshold = values.clone().mean().clamp_max(0.1).reshape([1, 1]);
            let flags = values.clone().greater(threshold).float();
            // Keep the occupied-ID sampling policy and RNG identical. Only the
            // flags are mirrored; values are downloaded when saving a checkpoint.
            self.flags = flags
                .clone()
                .try_into_vec_as::<f32>()?
                .into_iter()
                .map(|v| v != 0.)
                .collect();
            self.device_values = Some(values);
            self.device_flags = Some(flags);
            return Ok(());
        }
        self.sync_values()?;
        self.device_flags = None;
        let sdf = field.evaluate(&points, query)?;
        for (&i, &s) in ids.iter().zip(&sdf) {
            self.values[i] = (self.values[i] * 0.95).max(sigmoid(-80. * s));
        }

        let threshold =
            (self.values.iter().map(|&x| x as f64).sum::<f64>() / n as f64).min(0.1) as f32;
        for (flag, &value) in self.flags.iter_mut().zip(&self.values) {
            *flag = value > threshold;
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub patch: usize,
    pub start: f32,
    pub end: f32,
}

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub fn alpha(a: f32, b: f32, inv: f32) -> f32 {
    let c = sigmoid(a * inv);
    ((c - sigmoid(b * inv) + 1e-5) / (c + 1e-5)).clamp(0., 1.)
}

pub fn march(
    patches: &[Patch],
    occ: &Occupancy,
    step_size: f32,
    rng: &mut impl Rng,
) -> Vec<Sample> {
    let jitter: Vec<_> = patches
        .iter()
        .map(|patch| {
            let d = patch.dirs[4];
            let o = patch.origin;
            let a = dot(d, d);
            let b = 2. * dot(o, d);
            let c = dot(o, o) - 1.;
            let disc = b * b - 4. * a * c;
            (disc > 0.).then(|| rng.random::<f32>())
        })
        .collect();

    let r = occ.resolution as f32;
    let half_r = r * 0.5;

    let by_patch: Vec<Vec<Sample>> = patches
        .par_iter()
        .enumerate()
        .map(|(p, patch)| {
            let mut samples = Vec::new();
            let d = patch.dirs[4];
            let o = patch.origin;
            let a = dot(d, d);
            let b = 2. * dot(o, d);
            let c = dot(o, o) - 1.;
            let disc = b * b - 4. * a * c;
            if disc <= 0. {
                return samples;
            }
            let near = (-b - disc.sqrt()) / (2. * a) + jitter[p].unwrap() * step_size;
            let far = (-b + disc.sqrt()) / (2. * a);
            let steps = ((far - near) / step_size).ceil().max(0.) as usize;
            let inv_d = d.map(|v| 1.0 / if v.abs() < 1e-10 { 1e-10 } else { v });

            let mut s = 0usize;
            while s < steps {
                let t = near + s as f32 * step_size;
                let end = t + step_size;
                let mid = (t + end) * 0.5;
                if mid >= far {
                    break;
                }
                let pt = point(o, d, mid);
                if occ.occupied(pt) {
                    samples.push(Sample {
                        patch: p,
                        start: t,
                        end,
                    });
                    s += 1;
                } else if pt.iter().any(|&x| x < -1. || x > 1.) {
                    s += 1;
                } else {
                    let mut dt_voxel = f32::MAX;
                    for axis in 0..3 {
                        let grid_pos = (pt[axis] + 1.0) * half_r;
                        let next_plane = if d[axis] >= 0.0 {
                            grid_pos.floor() + 1.0
                        } else {
                            grid_pos.ceil() - 1.0
                        };
                        let dt_axis = (next_plane - grid_pos) * inv_d[axis] / half_r;
                        if dt_axis < dt_voxel {
                            dt_voxel = dt_axis;
                        }
                    }
                    let skip = ((dt_voxel / step_size).ceil() as usize).max(1);
                    s += skip;
                }
            }
            samples
        })
        .collect();
    by_patch.into_iter().flatten().collect()
}

pub fn visible(
    field: &mut Field,
    patches: &[Patch],
    samples: Vec<Sample>,
    query: usize,
) -> Result<Vec<Sample>> {
    if samples.is_empty() {
        return Ok(samples);
    }

    let n = samples.len();
    let mut unique_pts = Vec::with_capacity(n + patches.len());
    let mut start_idx = Vec::with_capacity(n);
    let mut end_idx = Vec::with_capacity(n);

    for (i, s) in samples.iter().enumerate() {
        let p = &patches[s.patch];
        let reuse = i > 0
            && samples[i - 1].patch == s.patch
            && s.start.to_bits() == samples[i - 1].end.to_bits();
        if reuse {
            start_idx.push(end_idx[i - 1]);
        } else {
            let idx = unique_pts.len();
            unique_pts.push(point(p.origin, p.dirs[4], s.start));
            start_idx.push(idx);
        }
        let idx = unique_pts.len();
        unique_pts.push(point(p.origin, p.dirs[4], s.end));
        end_idx.push(idx);
    }

    if field.cpu_visibility {
        let sdf = field.evaluate_visibility(&unique_pts, query)?;
        let inv = field.inv_s_host();
        let mut trans = vec![1f32; patches.len()];
        let mut out = Vec::with_capacity(n);
        for (i, s) in samples.into_iter().enumerate() {
            let a = alpha(sdf[start_idx[i]], sdf[end_idx[i]], inv);
            let t = &mut trans[s.patch];
            if *t >= 1e-8 {
                out.push(s);
            }
            *t *= (1. - a).max(1e-10);
        }
        return Ok(out);
    }
    assert!(
        unique_pts.len() < (1 << 24) && n < (1 << 24),
        "visibility indices exceed f32 range"
    );
    let pairs = start_idx
        .iter()
        .zip(&end_idx)
        .flat_map(|(&a, &b)| [a as f32, b as f32])
        .collect();
    let mut ranges = Vec::with_capacity(patches.len() * 2);
    let mut cursor = 0;
    for patch in 0..patches.len() {
        let begin = cursor;
        while cursor < n && samples[cursor].patch == patch {
            cursor += 1;
        }
        ranges.extend([begin as f32, cursor as f32]);
    }
    let mut remaining = field.visibility_counts(&unique_pts, query, pairs, ranges)?;
    let out = samples
        .into_iter()
        .filter(|s| {
            let count = &mut remaining[s.patch];
            if *count == 0 {
                false
            } else {
                *count -= 1;
                true
            }
        })
        .collect();
    Ok(out)
}

pub fn sample_and_dedup_points(
    patches: &[Patch],
    samples: &[Sample],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<usize>) {
    let n = samples.len();
    let mut points = Vec::with_capacity(n * 18);
    let mut unique = Vec::with_capacity(n * 10);
    let mut indices = Vec::with_capacity(n * 18);
    // A patch's ray-plane projection is invariant across all of its samples.
    // Keep the original arithmetic order, but do the nine divisions once.
    let ratios: Vec<[f32; 9]> = patches
        .iter()
        .map(|p| {
            let center = dot(p.dirs[4], p.plane);
            std::array::from_fn(|i| center / dot(p.dirs[i], p.plane))
        })
        .collect();

    for (s, sample) in samples.iter().enumerate() {
        let p = &patches[sample.patch];
        let ratios = &ratios[sample.patch];

        let reuse_start = s > 0
            && samples[s - 1].patch == sample.patch
            && sample.start.to_bits() == samples[s - 1].end.to_bits();

        if reuse_start {
            let prev_end_base = (s - 1) * 18 + 9;
            for i in 0..9 {
                points.push(points[prev_end_base + i]);
                indices.push(indices[prev_end_base + i]);
            }
        } else {
            for i in 0..9 {
                let pt = point(p.origin, p.dirs[i], sample.start * ratios[i]);
                let idx = unique.len();
                unique.push(pt);
                points.push(pt);
                indices.push(idx);
            }
        }

        for i in 0..9 {
            let pt = point(p.origin, p.dirs[i], sample.end * ratios[i]);
            let idx = unique.len();
            unique.push(pt);
            points.push(pt);
            indices.push(idx);
        }
    }
    (points, unique, indices)
}

pub fn unique_points(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<usize>) {
    let mut lookup = ahash::AHashMap::with_capacity(points.len());
    let mut unique = Vec::with_capacity(points.len());
    let indices = points
        .iter()
        .map(|&point| {
            *lookup.entry(point.map(f32::to_bits)).or_insert_with(|| {
                let i = unique.len();
                unique.push(point);
                i
            })
        })
        .collect();
    (unique, indices)
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct Loss {
    pub normal: f64,
    pub mask: f64,
    pub eikonal: f64,
}

impl Loss {
    pub fn total(&self) -> f64 {
        self.normal + self.mask + self.eikonal
    }
}

pub struct Composite {
    pub loss: Loss,
    pub sdf_grad: Vec<f32>,
    pub variance_grad: f32,
    pub sdf_bias_grad: f32,
    pub normals: Vec<[f32; 3]>,
    pub masks: Vec<f32>,
}

pub fn composite(
    patches: &[Patch],
    samples: &[Sample],
    points: &[[f32; 3]],
    sdf: &[f32],
    variance: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
) -> Composite {
    let count = samples.len();
    let inv = (variance * 10.).exp().clamp(1e-6, 1e6);

    let is_monotonic = samples.windows(2).all(|w| w[0].patch <= w[1].patch);

    if !is_monotonic {
        return composite_fallback(
            patches, samples, points, sdf, variance, normal_den, pixel_den, sample_den,
        );
    }

    let mut patch_ranges = vec![(0usize, 0usize); patches.len()];
    let mut s_idx = 0;
    while s_idx < count {
        let p = samples[s_idx].patch;
        let start = s_idx;
        while s_idx < count && samples[s_idx].patch == p {
            s_idx += 1;
        }
        patch_ranges[p] = (start, s_idx);
    }

    let mut ds = vec![0f32; sdf.len()];
    let mut normals = vec![[0f32; 3]; patches.len() * 9];
    let mut masks = vec![0f32; patches.len() * 9];

    let chunk_patches = 64usize;
    let num_patch_chunks = patches.len().div_ceil(chunk_patches);

    let mut normals_rest = normals.as_mut_slice();
    let mut masks_rest = masks.as_mut_slice();
    let mut ds_rest = ds.as_mut_slice();

    struct PatchChunkTask<'a> {
        patches: &'a [Patch],
        ranges: &'a [(usize, usize)],
        normals_out: &'a mut [[f32; 3]],
        masks_out: &'a mut [f32],
        ds_slice: &'a mut [f32],
        s_offset: usize,
    }

    let mut tasks = Vec::with_capacity(num_patch_chunks);
    let mut prev_s_end = 0usize;

    for c in 0..num_patch_chunks {
        let p_start = c * chunk_patches;
        let p_end = (p_start + chunk_patches).min(patches.len());
        let p_len = p_end - p_start;

        let mut s_start = prev_s_end;
        let mut s_end = prev_s_end;
        for &(a, b) in &patch_ranges[p_start..p_end] {
            if b > a {
                if s_start == prev_s_end && s_end == prev_s_end {
                    s_start = a;
                }
                s_end = b;
            }
        }
        let ds_len = (s_end - prev_s_end) * 18;
        let s_offset = prev_s_end;
        prev_s_end = s_end;

        let (n_head, n_tail) = normals_rest.split_at_mut(p_len * 9);
        normals_rest = n_tail;
        let (m_head, m_tail) = masks_rest.split_at_mut(p_len * 9);
        masks_rest = m_tail;
        let (ds_head, ds_tail) = ds_rest.split_at_mut(ds_len);
        ds_rest = ds_tail;

        tasks.push(PatchChunkTask {
            patches: &patches[p_start..p_end],
            ranges: &patch_ranges[p_start..p_end],
            normals_out: n_head,
            masks_out: m_head,
            ds_slice: ds_head,
            s_offset,
        });
    }

    let partials: Vec<(f64, f64, f64, f64, f64)> = tasks
        .into_par_iter()
        .map(|task| {
            let mut local_normal_loss = 0f64;
            let mut local_mask_loss = 0f64;
            let mut local_eikonal_loss = 0f64;
            let mut local_dinv = 0f64;
            let mut local_bias_grad = 0f64;

            let mut gradients = Vec::new();
            let mut steps_cache = Vec::new();
            let mut alpha_values = Vec::new();
            let mut weights = Vec::new();
            let mut ts = Vec::new();
            let mut gw = Vec::new();

            for (local_p, patch) in task.patches.iter().enumerate() {
                let (s_start, s_end) = task.ranges[local_p];
                let s_count = s_end - s_start;

                let norm_out = &mut task.normals_out[local_p * 9..(local_p + 1) * 9];
                let mask_out = &mut task.masks_out[local_p * 9..(local_p + 1) * 9];

                let dir4_plane = dot(patch.dirs[4], patch.plane);
                let dt_ratios: [f32; 9] =
                    std::array::from_fn(|i| dir4_plane / dot(patch.dirs[i], patch.plane));

                let mut trans = [1f32; 9];

                gradients.clear();
                gradients.resize(s_count * 9, [0f32; 3]);
                steps_cache.clear();
                steps_cache.resize(s_count * 9, [0f32; 3]);
                alpha_values.clear();
                alpha_values.resize(s_count * 9, 0f32);
                weights.clear();
                weights.resize(s_count * 9, 0f32);
                ts.clear();
                ts.resize(s_count * 9, 0f32);
                gw.clear();
                gw.resize(s_count * 9, 0f32);

                for ls in 0..s_count {
                    let s = s_start + ls;
                    let sample = &samples[s];
                    let base = s * 18;
                    let sample_dt = sample.end - sample.start;

                    for i in 0..9 {
                        let row = i / 3;
                        let col = i % 3;
                        let (left, right) = match col {
                            0 => (i, i + 1),
                            1 => (i - 1, i + 1),
                            _ => (i - 1, i),
                        };
                        let (up, down) = match row {
                            0 => (i, i + 3),
                            1 => (i - 3, i + 3),
                            _ => (i - 3, i),
                        };
                        let dx = if col == 1 {
                            distance(points[base + i - 1], points[base + i])
                                + distance(points[base + i], points[base + i + 1])
                        } else {
                            distance(points[base + left], points[base + right])
                        };
                        let dy = if row == 1 {
                            distance(points[base + i - 3], points[base + i])
                                + distance(points[base + i], points[base + i + 3])
                        } else {
                            distance(points[base + up], points[base + down])
                        };
                        let dt = sample_dt * dt_ratios[i];
                        let inv_dt = 1. / dt;
                        let inv_dx = 1. / dx;
                        let inv_dy = 1. / dy;

                        let proj = [
                            (sdf[base + 9 + i] - sdf[base + i]) * inv_dt,
                            (sdf[base + right] - sdf[base + left]) * inv_dx,
                            (sdf[base + down] - sdf[base + up]) * inv_dy,
                        ];
                        let g = patch.inverse[i].map(|r| dot(r, proj));
                        let idx = ls * 9 + i;
                        gradients[idx] = g;
                        steps_cache[idx] = [inv_dt, inv_dx, inv_dy];

                        let a = alpha(sdf[base + i], sdf[base + 9 + i], inv);
                        alpha_values[idx] = a;
                        ts[idx] = trans[i];
                        let w = trans[i] * a;
                        weights[idx] = w;
                        trans[i] *= (1. - a).max(1e-10);
                        mask_out[i] += w;
                        for d in 0..3 {
                            norm_out[i][d] += w * g[d];
                        }
                    }
                }

                let mut dn = [[0f32; 3]; 9];
                let mut dm = [0f32; 9];
                for i in 0..9 {
                    let mask = patch.mask[i];
                    for d in 0..3 {
                        let err = (norm_out[i][d] - patch.normals[i][d]) * mask;
                        local_normal_loss += (err * err / normal_den) as f64;
                        dn[i][d] = 2. * err * mask / normal_den;
                    }
                    let m = mask_out[i].clamp(1e-5, 1. - 1e-5);
                    local_mask_loss +=
                        (-(mask * m.ln() + (1. - mask) * (1. - m).ln()) / pixel_den) as f64;
                    if mask_out[i] >= 1e-5 && mask_out[i] <= 1. - 1e-5 {
                        dm[i] = (m - mask) / (m * (1. - m) * pixel_den);
                    }
                }

                for ls in 0..s_count {
                    let s = s_start + ls;
                    let ds_base = (s - task.s_offset) * 18;
                    for i in 0..9 {
                        let idx = ls * 9 + i;
                        let g = gradients[idx];
                        let norm = dot(g, g).sqrt();
                        local_eikonal_loss += ((norm - 1.).powi(2) / sample_den) as f64;
                        let dg: [f32; 3] = std::array::from_fn(|d| {
                            weights[idx] * dn[i][d]
                                + 2. * (norm - 1.) / sample_den * g[d] / norm.max(1e-12)
                        });
                        let v = patch.inverse[i];
                        let dp: [f32; 3] = std::array::from_fn(|d| {
                            v[0][d] * dg[0] + v[1][d] * dg[1] + v[2][d] * dg[2]
                        });
                        gw[idx] = dot(dn[i], g) + dm[i];

                        let row = i / 3;
                        let col = i % 3;
                        let (l, r) = match col {
                            0 => (i, i + 1),
                            1 => (i - 1, i + 1),
                            _ => (i - 1, i),
                        };
                        let (u, d) = match row {
                            0 => (i, i + 3),
                            1 => (i - 3, i + 3),
                            _ => (i - 3, i),
                        };
                        let [inv_dt, inv_dx, inv_dy] = steps_cache[idx];
                        task.ds_slice[ds_base + i] -= dp[0] * inv_dt;
                        task.ds_slice[ds_base + 9 + i] += dp[0] * inv_dt;
                        task.ds_slice[ds_base + l] -= dp[1] * inv_dx;
                        task.ds_slice[ds_base + r] += dp[1] * inv_dx;
                        task.ds_slice[ds_base + u] -= dp[2] * inv_dy;
                        task.ds_slice[ds_base + d] += dp[2] * inv_dy;
                    }
                }

                let mut future = [0f32; 9];
                for ls in (0..s_count).rev() {
                    let s = s_start + ls;
                    let base = s * 18;
                    let ds_base = (s - task.s_offset) * 18;
                    for i in 0..9 {
                        let idx = ls * 9 + i;
                        let a = alpha_values[idx];
                        let q = (1. - a).max(1e-10);
                        let da = ts[idx] * (gw[idx] - if 1. - a >= 1e-10 { future[i] } else { 0. });
                        future[i] = gw[idx] * a + q * future[i];

                        let s0 = sdf[base + i];
                        let s1 = sdf[base + 9 + i];
                        let c0 = sigmoid(s0 * inv);
                        let c1 = sigmoid(s1 * inv);
                        let denom = c0 + 1e-5;
                        let raw = (c0 - c1 + 1e-5) / denom;
                        if (0. ..=1.).contains(&raw) {
                            let dc0 = da * c1 / (denom * denom);
                            let dc1 = -da / denom;
                            let z0 = dc0 * c0 * (1. - c0);
                            let z1 = dc1 * c1 * (1. - c1);
                            task.ds_slice[ds_base + i] += z0 * inv;
                            task.ds_slice[ds_base + 9 + i] += z1 * inv;
                            local_bias_grad += f64::from(z0 * inv) + f64::from(z1 * inv);
                            local_dinv += f64::from(z0 * s0 + z1 * s1);
                        }
                    }
                }
            }

            (
                local_normal_loss,
                local_mask_loss,
                local_eikonal_loss,
                local_dinv,
                local_bias_grad,
            )
        })
        .collect();

    let mut loss = Loss::default();
    let mut dinv = 0f64;
    let mut sdf_bias_grad = 0f64;
    for (nl, ml, el, di, bg) in partials {
        loss.normal += nl;
        loss.mask += ml;
        loss.eikonal += el;
        dinv += di;
        sdf_bias_grad += bg;
    }

    let variance_grad = if (1e-6..=1e6).contains(&(variance * 10.).exp()) {
        (dinv as f32) * 10. * inv
    } else {
        0.
    };

    Composite {
        loss,
        sdf_grad: ds,
        variance_grad,
        sdf_bias_grad: sdf_bias_grad as f32,
        normals,
        masks,
    }
}

fn composite_fallback(
    patches: &[Patch],
    samples: &[Sample],
    points: &[[f32; 3]],
    sdf: &[f32],
    variance: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
) -> Composite {
    let count = samples.len();
    let inv = (variance * 10.).exp().clamp(1e-6, 1e6);
    let mut gradients = vec![[0f32; 3]; count * 9];
    let mut alpha_values = vec![0.; count * 9];
    let mut weights = vec![0.; count * 9];
    let mut ts = vec![0.; count * 9];
    let mut trans = vec![1.; patches.len() * 9];
    let mut normals = vec![[0.; 3]; patches.len() * 9];
    let mut masks = vec![0.; patches.len() * 9];
    for (s, sample) in samples.iter().enumerate() {
        let patch = &patches[sample.patch];
        let base = s * 18;
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let (left, right) = match col {
                0 => (i, i + 1),
                1 => (i - 1, i + 1),
                _ => (i - 1, i),
            };
            let (up, down) = match row {
                0 => (i, i + 3),
                1 => (i - 3, i + 3),
                _ => (i - 3, i),
            };
            let dx = if col == 1 {
                distance(points[base + i - 1], points[base + i])
                    + distance(points[base + i], points[base + i + 1])
            } else {
                distance(points[base + left], points[base + right])
            };
            let dy = if row == 1 {
                distance(points[base + i - 3], points[base + i])
                    + distance(points[base + i], points[base + i + 3])
            } else {
                distance(points[base + up], points[base + down])
            };
            let dt = (sample.end - sample.start) * dot(patch.dirs[4], patch.plane)
                / dot(patch.dirs[i], patch.plane);
            let proj = [
                (sdf[base + 9 + i] - sdf[base + i]) / dt,
                (sdf[base + right] - sdf[base + left]) / dx,
                (sdf[base + down] - sdf[base + up]) / dy,
            ];
            let g = patch.inverse[i].map(|r| dot(r, proj));
            let index = s * 9 + i;
            gradients[index] = g;
            let a = alpha(sdf[base + i], sdf[base + 9 + i], inv);
            alpha_values[index] = a;
            let pixel = sample.patch * 9 + i;
            ts[index] = trans[pixel];
            let w = trans[pixel] * a;
            weights[index] = w;
            trans[pixel] *= (1. - a).max(1e-10);
            masks[pixel] += w;
            for d in 0..3 {
                normals[pixel][d] += w * g[d];
            }
        }
    }

    let mut loss = Loss::default();
    let mut dn = vec![[0.; 3]; normals.len()];
    let mut dm = vec![0.; masks.len()];
    for (p, patch) in patches.iter().enumerate() {
        for i in 0..9 {
            let k = p * 9 + i;
            let mask = patch.mask[i];
            for d in 0..3 {
                let err = (normals[k][d] - patch.normals[i][d]) * mask;
                loss.normal += (err * err / normal_den) as f64;
                dn[k][d] = 2. * err * mask / normal_den;
            }
            let m = masks[k].clamp(1e-5, 1. - 1e-5);
            loss.mask += (-(mask * m.ln() + (1. - mask) * (1. - m).ln()) / pixel_den) as f64;
            if masks[k] >= 1e-5 && masks[k] <= 1. - 1e-5 {
                dm[k] = (m - mask) / (m * (1. - m) * pixel_den);
            }
        }
    }

    let mut ds = vec![0f32; sdf.len()];
    let mut gw = vec![0.; count * 9];
    for (s, sample) in samples.iter().enumerate() {
        let patch = &patches[sample.patch];
        let base = s * 18;
        for i in 0..9 {
            let k = s * 9 + i;
            let pixel = sample.patch * 9 + i;
            let g = gradients[k];
            let norm = dot(g, g).sqrt();
            loss.eikonal += ((norm - 1.).powi(2) / sample_den) as f64;
            let dg: [f32; 3] = std::array::from_fn(|d| {
                weights[k] * dn[pixel][d] + 2. * (norm - 1.) / sample_den * g[d] / norm.max(1e-12)
            });
            let v = patch.inverse[i];
            let dp: [f32; 3] =
                std::array::from_fn(|d| v[0][d] * dg[0] + v[1][d] * dg[1] + v[2][d] * dg[2]);
            gw[k] = dot(dn[pixel], g) + dm[pixel];
            let row = i / 3;
            let col = i % 3;
            let (l, r) = match col {
                0 => (i, i + 1),
                1 => (i - 1, i + 1),
                _ => (i - 1, i),
            };
            let (u, d) = match row {
                0 => (i, i + 3),
                1 => (i - 3, i + 3),
                _ => (i - 3, i),
            };
            let dx = if col == 1 {
                distance(points[base + i - 1], points[base + i])
                    + distance(points[base + i], points[base + i + 1])
            } else {
                distance(points[base + l], points[base + r])
            };
            let dy = if row == 1 {
                distance(points[base + i - 3], points[base + i])
                    + distance(points[base + i], points[base + i + 3])
            } else {
                distance(points[base + u], points[base + d])
            };
            let dt = (sample.end - sample.start) * dot(patch.dirs[4], patch.plane)
                / dot(patch.dirs[i], patch.plane);
            ds[base + i] -= dp[0] / dt;
            ds[base + 9 + i] += dp[0] / dt;
            ds[base + l] -= dp[1] / dx;
            ds[base + r] += dp[1] / dx;
            ds[base + u] -= dp[2] / dy;
            ds[base + d] += dp[2] / dy;
        }
    }

    let mut future = vec![0.; patches.len() * 9];
    let mut dinv = 0.;
    let mut sdf_bias_grad = 0f64;
    for s in (0..count).rev() {
        let sample = &samples[s];
        let base = s * 18;
        for i in 0..9 {
            let k = s * 9 + i;
            let pixel = sample.patch * 9 + i;
            let a = alpha_values[k];
            let q = (1. - a).max(1e-10);
            let da = ts[k] * (gw[k] - if 1. - a >= 1e-10 { future[pixel] } else { 0. });
            future[pixel] = gw[k] * a + q * future[pixel];
            let s0 = sdf[base + i];
            let s1 = sdf[base + 9 + i];
            let c0 = sigmoid(s0 * inv);
            let c1 = sigmoid(s1 * inv);
            let denom = c0 + 1e-5;
            let raw = (c0 - c1 + 1e-5) / denom;
            if (0. ..=1.).contains(&raw) {
                let dc0 = da * c1 / (denom * denom);
                let dc1 = -da / denom;
                let z0 = dc0 * c0 * (1. - c0);
                let z1 = dc1 * c1 * (1. - c1);
                ds[base + i] += z0 * inv;
                ds[base + 9 + i] += z1 * inv;
                sdf_bias_grad += f64::from(z0 * inv) + f64::from(z1 * inv);
                dinv += z0 * s0 + z1 * s1;
            }
        }
    }
    let variance_grad = if (1e-6..=1e6).contains(&(variance * 10.).exp()) {
        dinv * 10. * inv
    } else {
        0.
    };
    Composite {
        loss,
        sdf_grad: ds,
        variance_grad,
        sdf_bias_grad: sdf_bias_grad as f32,
        normals,
        masks,
    }
}

#[derive(Deserialize)]
struct RenderFixture {
    patches: Vec<Patch>,
    samples: Vec<Sample>,
    points: Vec<[f32; 3]>,
    sdf: Vec<f32>,
    variance: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
    loss: Loss,
    sdf_grad: Vec<f32>,
    variance_grad: f32,
    normals: Vec<[f32; 3]>,
    masks: Vec<f32>,
}

pub fn verify_step(
    model: &std::path::Path,
    path: &std::path::Path,
    device: burn::tensor::Device,
) -> Result<()> {
    use crate::field::{read_f32, tensor};
    use std::collections::BTreeMap;
    let mut field = Field::load(model, device)?;
    let f: RenderFixture = serde_json::from_slice(&std::fs::read(path.join("render.json"))?)?;
    let (unique, indices) = unique_points(&f.points);
    let unique_sdf = field.evaluate(&unique, 8192)?;
    let sdf: Vec<_> = indices.iter().map(|&i| unique_sdf[i]).collect();
    let c = composite(
        &f.patches,
        &f.samples,
        &f.points,
        &sdf,
        field.variance_host,
        f.normal_den,
        f.pixel_den,
        f.sample_den,
    );
    let loss_error = (c.loss.total() - f.loss.total()).abs();
    println!(
        "Combined loss: native={} torch={} abs={loss_error:e}",
        c.loss.total(),
        f.loss.total()
    );
    anyhow::ensure!(
        loss_error < 0.002 + 0.002 * f.loss.total().abs(),
        "combined loss mismatch"
    );
    let mut gradients = crate::field::Gradients::default();
    let mut adjoints = vec![0.; unique.len()];
    for (&i, &g) in indices.iter().zip(&c.sdf_grad) {
        adjoints[i] += g;
    }
    field.accumulate_gradients(&unique, &adjoints, 31, &mut gradients);
    gradients.insert(
        "lin1.bias".into(),
        tensor(vec![c.sdf_bias_grad], [1, 1], &field.device),
    );
    gradients.insert(
        "variance".into(),
        tensor(vec![c.variance_grad], [1, 1], &field.device),
    );
    let gradients = gradients.into_tensors(&field);
    let mut oracle_gradients = BTreeMap::new();
    for (name, g) in &gradients {
        let expected = read_f32(&path.join(format!("{}.step.grad.bin", name.replace('.', "_"))))?;
        let actual = g.clone().try_into_vec_as::<f32>()?;
        anyhow::ensure!(
            actual.len() == expected.len(),
            "gradient shape mismatch {name}"
        );
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let peak = expected.iter().map(|v| v.abs()).fold(0f32, f32::max);
        println!("{name}: gradient max_abs={error:e} reference_peak={peak:e}");
        anyhow::ensure!(
            error <= 0.005 + peak * 0.01,
            "combined gradient mismatch {name}"
        );
        oracle_gradients.insert(name.clone(), tensor(expected, g.dims(), &field.device));
    }

    let mut oracle_adam = Field::load(model, field.device.clone())?;
    oracle_adam.apply_gradients(oracle_gradients, 5e-4);
    field.apply_gradients(gradients, 5e-4);
    for (name, p) in &field.params {
        let expected = read_f32(&path.join(format!("{}.step.value.bin", name.replace('.', "_"))))?;
        let actual = p.value.clone().try_into_vec_as::<f32>()?;
        let oracle = oracle_adam.p(name).try_into_vec_as::<f32>()?;
        let optimizer_error = oracle
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        anyhow::ensure!(
            optimizer_error <= 2e-6,
            "Adam arithmetic mismatch {name}: {optimizer_error:e}"
        );
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        println!(
            "{name}: same-gradient Adam max_abs={optimizer_error:e}; combined update max_abs={error:e} rms={rms:e}"
        );
        anyhow::ensure!(
            error <= 2e-3 && rms <= 2e-5,
            "combined update mismatch {name}"
        );
    }
    println!("Complete reconstruction optimizer step passed.");
    Ok(())
}

pub fn verify(path: &std::path::Path, data: Option<&std::path::Path>) -> Result<()> {
    let f: RenderFixture = serde_json::from_slice(&std::fs::read(path.join("render.json"))?)?;
    let c = composite(
        &f.patches,
        &f.samples,
        &f.points,
        &f.sdf,
        f.variance,
        f.normal_den,
        f.pixel_den,
        f.sample_den,
    );
    let check = |name: &str, a: &[f32], b: &[f32], abs: f32, rel: f32| -> Result<()> {
        anyhow::ensure!(a.len() == b.len(), "{name} shape mismatch");
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{name}: max_abs={max:.7e}");
        anyhow::ensure!(
            a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() <= abs + rel * b.abs()),
            "{name} mismatch"
        );
        Ok(())
    };
    check(
        "normal loss",
        &[c.loss.normal as f32],
        &[f.loss.normal as f32],
        1e-5,
        1e-4,
    )?;
    check(
        "mask loss",
        &[c.loss.mask as f32],
        &[f.loss.mask as f32],
        1e-5,
        1e-4,
    )?;
    check(
        "eikonal loss",
        &[c.loss.eikonal as f32],
        &[f.loss.eikonal as f32],
        1e-5,
        1e-4,
    )?;
    check("SDF adjoint", &c.sdf_grad, &f.sdf_grad, 2e-4, 2e-3)?;
    check(
        "variance adjoint",
        &[c.variance_grad],
        &[f.variance_grad],
        2e-4,
        2e-3,
    )?;
    check(
        "normals",
        &c.normals.iter().flatten().copied().collect::<Vec<_>>(),
        &f.normals.iter().flatten().copied().collect::<Vec<_>>(),
        1e-5,
        1e-4,
    )?;
    check("masks", &c.masks, &f.masks, 1e-5, 1e-4)?;

    if let Some(path_data) = data {
        let ds = crate::data::Dataset::load(path_data)?;
        let v = &ds.views[0];
        let p = v.patch(v.width / 2, v.height / 2);
        #[derive(Deserialize)]
        struct Camera {
            width: usize,
            height: usize,
            origin: [f32; 3],
            dirs: Vec<[f32; 3]>,
            normals: Vec<[f32; 3]>,
        }
        let camera: Camera = serde_json::from_slice(&std::fs::read(path.join("camera.json"))?)?;
        anyhow::ensure!(
            (v.width, v.height) == (camera.width, camera.height),
            "image dimension mismatch"
        );
        check("camera origin", &p.origin, &camera.origin, 1e-5, 1e-5)?;
        check(
            "camera rays",
            &p.dirs.iter().flatten().copied().collect::<Vec<_>>(),
            &camera.dirs.iter().flatten().copied().collect::<Vec<_>>(),
            1e-5,
            1e-5,
        )?;
        check(
            "EXR normals",
            &p.normals.iter().flatten().copied().collect::<Vec<_>>(),
            &camera.normals.iter().flatten().copied().collect::<Vec<_>>(),
            1e-6,
            1e-6,
        )?;
    }

    let eps = 1e-4;
    let shifted_loss = |shift: f32| {
        let shifted: Vec<_> = f.sdf.iter().map(|v| v + shift).collect();
        composite(
            &f.patches,
            &f.samples,
            &f.points,
            &shifted,
            f.variance,
            f.normal_den,
            f.pixel_den,
            f.sample_den,
        )
        .loss
        .total()
    };
    let numeric_bias = ((shifted_loss(eps) - shifted_loss(-eps)) / f64::from(2. * eps)) as f32;
    anyhow::ensure!(
        (numeric_bias - c.sdf_bias_grad).abs() < 0.03 + 0.02 * numeric_bias.abs(),
        "SDF bias finite difference mismatch"
    );
    for index in (0..f.sdf.len()).step_by(13) {
        let mut plus = f.sdf.clone();
        let mut minus = f.sdf.clone();
        plus[index] += eps;
        minus[index] -= eps;
        let a = composite(
            &f.patches,
            &f.samples,
            &f.points,
            &plus,
            f.variance,
            f.normal_den,
            f.pixel_den,
            f.sample_den,
        )
        .loss
        .total();
        let b = composite(
            &f.patches,
            &f.samples,
            &f.points,
            &minus,
            f.variance,
            f.normal_den,
            f.pixel_den,
            f.sample_den,
        )
        .loss
        .total();
        let numeric = ((a - b) / (2. * eps as f64)) as f32;
        anyhow::ensure!(
            (numeric - c.sdf_grad[index]).abs() < 0.03 + 0.02 * numeric.abs(),
            "finite difference mismatch at {index}"
        );
    }
    println!("Packed renderer, gradients and finite differences passed.");
    Ok(())
}

/// Geometry stays on the CPU for now; SDFs and their adjoints stay on device.
/// `indices` must come from sample_and_dedup_points (no sharing across patches).
pub fn composite_device(
    patches: &[Patch],
    samples: &[Sample],
    points: &[[f32; 3]],
    indices: &[usize],
    sdf: burn::tensor::Tensor<2>,
    variance: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
) -> (burn::tensor::Tensor<2>, burn::tensor::Tensor<2>) {
    use crate::field::tensor;
    use burn::tensor::Tensor;
    let device = sdf.device();
    let n = sdf.dims()[0];
    assert!(
        n < (1 << 24),
        "compositor index exceeds exact f32 integer range"
    );
    let mut geometry = Vec::with_capacity(samples.len() * 27);
    for (s, sample) in samples.iter().enumerate() {
        let p = &patches[sample.patch];
        let base = s * 18;
        for i in 0..9 {
            let col = i % 3;
            let row = i / 3;
            let l = if col == 0 { i } else { i - 1 };
            let r = if col == 2 { i } else { i + 1 };
            let u = if row == 0 { i } else { i - 3 };
            let d = if row == 2 { i } else { i + 3 };
            let dx = if col == 1 {
                distance(points[base + l], points[base + i])
                    + distance(points[base + i], points[base + r])
            } else {
                distance(points[base + l], points[base + r])
            };
            let dy = if row == 1 {
                distance(points[base + u], points[base + i])
                    + distance(points[base + i], points[base + d])
            } else {
                distance(points[base + u], points[base + d])
            };
            let dt =
                (sample.end - sample.start) * (dot(p.dirs[4], p.plane) / dot(p.dirs[i], p.plane));
            geometry.extend([1. / dt, 1. / dx, 1. / dy]);
        }
    }
    let mut packed = Vec::with_capacity(patches.len() * 123);
    let mut cursor = 0;
    let inv = (variance * 10.).exp().clamp(1e-6, 1e6);
    for (pi, p) in patches.iter().enumerate() {
        let begin = cursor;
        while cursor < samples.len() && samples[cursor].patch == pi {
            cursor += 1;
        }
        packed.extend([
            begin as f32,
            cursor as f32,
            inv,
            normal_den,
            pixel_den,
            sample_den,
        ]);
        for i in 0..9 {
            packed.extend(p.inverse[i].iter().flatten().copied());
            packed.extend(p.normals[i]);
            packed.push(p.mask[i]);
        }
    }
    assert_eq!(cursor, samples.len());
    let count = samples.len().max(1);
    geometry.resize(count * 27, 0.);
    let mut mapping: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
    mapping.resize(count * 18, 0.);
    assert!(
        indices.len() < (1 << 24),
        "compositor occurrence index exceeds f32 range"
    );
    let mut reverse = vec![-1.0_f32; n * 2];
    for (occurrence, &id) in indices.iter().enumerate() {
        let slot = if reverse[id * 2] < 0. {
            id * 2
        } else {
            id * 2 + 1
        };
        assert!(reverse[slot] < 0., "unexpected cross-patch point sharing");
        reverse[slot] = occurrence as f32;
    }
    let [adjoints, stats] = crate::kernels::composite_gpu([
        sdf,
        tensor(geometry, [count, 27], &device),
        tensor(mapping, [count, 18], &device),
        tensor(packed, [patches.len(), 123], &device),
        Tensor::zeros([count, 90], &device),
        Tensor::zeros([n, 1], &device),
        Tensor::zeros([patches.len() * 9, 5], &device),
        tensor(reverse, [n, 2], &device),
    ]);
    let stats = stats.sum_dim(0);
    let vscale = if (1e-6..=1e6).contains(&(variance * 10.).exp()) {
        10. * inv
    } else {
        0.
    };
    let stats = stats * tensor(vec![1., 1., 1., vscale, 1.], [1, 5], &device);
    (adjoints, stats)
}

/// March into bounded per-ray storage, then compact in stable order on the GPU.
/// A scalar total determines the final allocation; existing geometry preparation
/// still consumes the compact samples on the host.
pub fn march_device(
    patches: &[Patch],
    occ: &mut Occupancy,
    step_size: f32,
    rng: &mut impl Rng,
    device: &burn::tensor::Device,
) -> Result<Vec<Sample>> {
    use crate::field::tensor;
    use burn::tensor::Tensor;
    let mut rays = Vec::with_capacity(patches.len() * 12);
    let mut capacity = 0usize;
    for patch in patches {
        let d = patch.dirs[4];
        let o = patch.origin;
        let a = dot(d, d);
        let b = 2. * dot(o, d);
        let c = dot(o, o) - 1.;
        let disc = b * b - 4. * a * c;
        let (near, far, count) = if disc > 0. {
            let near = (-b - disc.sqrt()) / (2. * a) + rng.random::<f32>() * step_size;
            let far = (-b + disc.sqrt()) / (2. * a);
            (
                near,
                far,
                ((far - near) / step_size).ceil().max(0.) as usize,
            )
        } else {
            (0., 0., 0)
        };
        rays.extend(o);
        rays.extend(d);
        rays.extend([
            near,
            far,
            step_size,
            capacity as f32,
            count as f32,
            occ.resolution as f32,
        ]);
        capacity = capacity
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("march capacity overflow"))?;
    }
    anyhow::ensure!(
        capacity < (1 << 24),
        "march batch exceeds exact index range; reduce batch size"
    );
    if capacity == 0 {
        return Ok(Vec::new());
    }
    let flags = occ
        .device_flags
        .get_or_insert_with(|| {
            tensor(
                occ.flags.iter().map(|&v| if v { 1. } else { 0. }).collect(),
                [occ.flags.len(), 1],
                device,
            )
        })
        .clone();
    let rays = tensor(rays, [patches.len(), 12], device);
    let [samples, counts] = crate::sampling_gpu::march([
        rays.clone(),
        flags,
        Tensor::zeros([capacity, 2], device),
        Tensor::zeros([patches.len(), 1], device),
    ]);
    let prefix = counts.cumsum(0);
    let total = prefix
        .clone()
        .slice([patches.len() - 1..patches.len(), 0..1])
        .try_into_vec_as::<f32>()?[0] as usize;
    anyhow::ensure!(total <= capacity, "invalid compacted sample count");
    if total == 0 {
        return Ok(Vec::new());
    }
    let [packed] =
        crate::sampling_gpu::compact([rays, samples, prefix, Tensor::zeros([total, 3], device)]);
    let packed = packed.try_into_vec_as::<f32>()?;
    Ok(packed
        .chunks_exact(3)
        .map(|s| Sample {
            patch: s[0] as usize,
            start: s[1],
            end: s[2],
        })
        .collect())
}

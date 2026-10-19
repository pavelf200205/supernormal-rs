//! Packed CPU reference compositor with analytical derivatives. Field execution is on Burn.
//! This also serves as the oracle for portable GPU compositor kernels.
use crate::{
    data::{Patch, distance, dot, point},
    field::Field,
};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Occupancy {
    pub resolution: usize,
    pub values: Vec<f32>,
    pub flags: Vec<bool>,
}
impl Occupancy {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            values: vec![0.; resolution.pow(3)],
            flags: vec![false; resolution.pow(3)],
        }
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
        // Advanced indexed assignment in Torch reads the previous occupancy once,
        // even when an index occurs more than once in the sampled list.
        let previous = self.values.clone();
        for chunk in ids.chunks(query) {
            let points: Vec<[f32; 3]> = chunk
                .iter()
                .map(|&i| {
                    let c = [i / (r * r), (i / r) % r, i % r];
                    c.map(|v| ((v as f32 + rng.random::<f32>()) / r as f32) * 2. - 1.)
                })
                .collect();
            let sdf = field.evaluate(&points, query)?;
            for (&i, &s) in chunk.iter().zip(&sdf) {
                self.values[i] = (previous[i] * 0.95).max(sigmoid(-80. * s));
            }
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
    let mut samples = vec![];
    for (p, patch) in patches.iter().enumerate() {
        let d = patch.dirs[4];
        let o = patch.origin;
        let a = dot(d, d);
        let b = 2. * dot(o, d);
        let c = dot(o, o) - 1.;
        let disc = b * b - 4. * a * c;
        if disc <= 0. {
            continue;
        }
        let near = (-b - disc.sqrt()) / (2. * a) + rng.random::<f32>() * step_size;
        let far = (-b + disc.sqrt()) / (2. * a);
        let steps = ((far - near) / step_size).ceil().max(0.) as usize;
        for s in 0..steps {
            let t = near + s as f32 * step_size;
            let end = t + step_size;
            let mid = (t + end) * 0.5;
            if mid < far && occ.occupied(point(o, d, mid)) {
                samples.push(Sample {
                    patch: p,
                    start: t,
                    end,
                });
            }
        }
    }
    samples
}
pub fn visible(
    field: &Field,
    patches: &[Patch],
    samples: Vec<Sample>,
    query: usize,
) -> Result<Vec<Sample>> {
    if samples.is_empty() {
        return Ok(samples);
    }
    let pts: Vec<_> = samples
        .iter()
        .flat_map(|s| {
            let p = &patches[s.patch];
            [
                point(p.origin, p.dirs[4], s.start),
                point(p.origin, p.dirs[4], s.end),
            ]
        })
        .collect();
    let sdf = field.evaluate(&pts, query)?;
    let inv = field.inv_s().into_scalar();
    let mut trans = vec![1f32; patches.len()];
    let mut out = vec![];
    for (i, s) in samples.into_iter().enumerate() {
        let a = alpha(sdf[2 * i], sdf[2 * i + 1], inv);
        let t = &mut trans[s.patch];
        if *t >= 1e-8 {
            out.push(s);
        }
        *t *= (1. - a).max(1e-10);
    }
    Ok(out)
}
pub fn sample_points(patches: &[Patch], samples: &[Sample]) -> Vec<[f32; 3]> {
    let mut points = Vec::with_capacity(samples.len() * 18);
    for s in samples {
        let p = &patches[s.patch];
        let center = dot(p.dirs[4], p.plane);
        for t in [s.start, s.end] {
            for d in p.dirs {
                points.push(point(p.origin, d, t * center / dot(d, p.plane)));
            }
        }
    }
    points
}
/// Share exactly equal endpoints; never quantize coordinates or change gradients.
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
                // DFD terms are exactly invariant under a constant SDF shift.
                // Sum only opacity derivatives to avoid cancellation noise being
                // amplified by Adam into a spurious output-bias update.
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
/// Tests the complete field -> renderer -> adjoint -> field -> Adam chain.
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
        field.p("variance").into_scalar(),
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
    // Force several chunks to test accumulation, rather than just a single backward.
    field.accumulate_gradients(&unique, &adjoints, 31, &mut gradients);
    gradients.insert(
        "lin1.bias".into(),
        tensor(vec![c.sdf_bias_grad], [1, 1], &field.device),
    );
    gradients.insert(
        "variance".into(),
        tensor(vec![c.variance_grad], [1, 1], &field.device),
    );
    let gradients = gradients.into_tensors(&field.device);
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
    // Separate optimizer arithmetic from first-step sensitivity to tiny gradient
    // differences in finite differences / reductions across different GPU backends.
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
    // Numerical derivatives supplement the independent Torch oracle.
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

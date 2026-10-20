use crate::{
    data::Dataset,
    field::{Field, tensor},
    render::{self, Loss, Occupancy},
};
use anyhow::{Result, ensure};
use burn::tensor::Device;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{io::Write, path::PathBuf};

#[derive(clap::Args, Clone, Serialize, Deserialize)]
pub struct TrainArgs {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, default_value_t = 30000)]
    pub steps: usize,
    #[arg(long, default_value_t = 30000)]
    pub end_iter: usize,
    #[arg(long, default_value_t = 2048)]
    pub batch: usize,
    #[arg(long, default_value_t = 2048)]
    pub microbatch: usize,
    #[arg(long, default_value_t = 65536)]
    pub query: usize,
    #[arg(long, default_value_t = 128)]
    pub occupancy_resolution: usize,
    #[arg(long, default_value_t = 5000)]
    pub save_freq: usize,
    #[arg(long, default_value_t = 0)]
    pub mesh_freq: usize,
    #[arg(long, default_value_t = 256)]
    pub mesh_resolution: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    #[serde(default)]
    pub profile_stages: bool,
}

#[derive(Serialize, Deserialize)]
struct State {
    rng: ChaCha8Rng,
    occupancy: Occupancy,
    args: TrainArgs,
}

pub fn train(args: TrainArgs, device: Device) -> Result<()> {
    ensure!(
        args.batch > 0
            && args.microbatch > 0
            && args.query > 0
            && args.end_iter > 500
            && args.occupancy_resolution >= 2,
        "invalid sizes"
    );
    ensure!(
        !args.out.join("metrics.csv").exists(),
        "output already contains a run; select a fresh output directory"
    );
    std::fs::create_dir_all(&args.out)?;
    let dataset = Dataset::load(&args.data)?;
    println!(
        "Loaded {} views, first {}x{}",
        dataset.views.len(),
        dataset.views[0].width,
        dataset.views[0].height
    );
    let mut field = if let Some(path) = &args.model {
        Field::load(path, device)?
    } else {
        Field::new(device, args.seed)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut occ = Occupancy::new(args.occupancy_resolution);
    let mut restored_state = false;
    if let Some(path) = &args.model {
        if path.join("training.json").exists() {
            let state: State = serde_json::from_slice(&std::fs::read(path.join("training.json"))?)?;
            rng = state.rng;
            occ = state.occupancy;
            restored_state = true;
            ensure!(
                state.args.batch == args.batch && state.args.end_iter == args.end_iter,
                "resume batch/schedule differs"
            );
        } else {
            eprintln!(
                "Model-only import: optimizer/occupancy/RNG may be fresh; this is not an exact resume."
            );
        }
    }
    let final_step = (field.step + args.steps).min(args.end_iter);
    let mut metrics = std::io::BufWriter::new(std::fs::File::create(args.out.join("metrics.csv"))?);
    writeln!(
        metrics,
        "step,normal,mask,eikonal,total,samples,seconds,active_levels"
    )?;
    std::fs::write(
        args.out.join("config.json"),
        serde_json::to_vec_pretty(&args)?,
    )?;

    if args.mesh_freq > 0 && field.step == 0 {
        crate::mesh::extract(
            &field,
            &args.out.join("mesh_000000.ply"),
            args.mesh_resolution,
            args.query,
            dataset.scale,
            dataset.offset,
        )?;
    }

    let mut first = true;
    while field.step < final_step {
        let started = std::time::Instant::now();
        let step = field.step;
        if (first && !restored_state) || step % 8 == 0 {
            occ.update(
                &field,
                if first && !occ.flags.iter().any(|x| *x) {
                    0
                } else {
                    step
                },
                args.query,
                &mut rng,
            )?;
        }
        first = false;
        let occupancy_seconds = started.elapsed().as_secs_f64();
        field.active = (step / 2000 + 1).min(14);
        let patches = dataset.random_patches(args.batch, &mut rng);
        let step_size = 10f32.powf(-2. - step as f32 / args.end_iter as f32);
        let march_started = std::time::Instant::now();
        let marched = render::march(&patches, &occ, step_size, &mut rng);
        let march_seconds = march_started.elapsed().as_secs_f64();
        let visible_started = std::time::Instant::now();
        let samples = render::visible(&field, &patches, marched, args.query)?;
        let visibility_seconds = visible_started.elapsed().as_secs_f64();
        if samples.is_empty() {
            anyhow::bail!(
                "empty occupancy batch at step {step}; checkpoint/config requires investigation"
            );
        }
        let normal_den = patches.iter().flat_map(|p| p.mask).sum::<f32>() + 1e-5;
        let sampling_seconds = started.elapsed().as_secs_f64() - occupancy_seconds;
        let pixel_den = (patches.len() * 9) as f32;
        let sample_den = (samples.len() * 9) as f32;

        let mut gradients = crate::field::Gradients::default();
        let mut total = Loss::default();
        let variance: f32 = field.variance_host;
        let mut variance_grad = 0f32;
        let mut sdf_bias_grad = 0f64;
        let mut t_eval = 0.0;
        let mut t_points = 0.0;
        let mut t_comp = 0.0;
        let mut t_bwd = 0.0;

        for start in (0..patches.len()).step_by(args.microbatch) {
            let end = (start + args.microbatch).min(patches.len());
            let local_storage: Vec<render::Sample>;
            let local: &[render::Sample] = if start == 0 && end == patches.len() {
                &samples
            } else {
                local_storage = samples
                    .iter()
                    .filter(|s| s.patch >= start && s.patch < end)
                    .map(|s| render::Sample {
                        patch: s.patch - start,
                        start: s.start,
                        end: s.end,
                    })
                    .collect();
                &local_storage
            };
            let pp = &patches[start..end];

            let t0 = std::time::Instant::now();
            let (points, unique, indices) = render::sample_and_dedup_points(pp, local);
            t_points += t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            let (unique_sdf, cache) = field.evaluate_and_cache(&unique, args.query)?;
            let sdf: Vec<f32> = indices
                .iter()
                .map(|&i| unsafe { *unique_sdf.get_unchecked(i) })
                .collect();
            t_eval += t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let result = render::composite(
                pp, local, &points, &sdf, variance, normal_den, pixel_den, sample_den,
            );
            t_comp += t1.elapsed().as_secs_f64();

            total.normal += result.loss.normal;
            total.mask += result.loss.mask;
            total.eikonal += result.loss.eikonal;
            variance_grad += result.variance_grad;
            sdf_bias_grad += f64::from(result.sdf_bias_grad);

            let mut adjoints = vec![0f32; unique.len()];
            for (&i, &g) in indices.iter().zip(&result.sdf_grad) {
                unsafe {
                    *adjoints.get_unchecked_mut(i) += g;
                }
            }
            let t2 = std::time::Instant::now();
            field.accumulate_gradients_cached(cache, &adjoints, &mut gradients);
            if args.profile_stages {
                field
                    .device
                    .sync()
                    .map_err(|e| anyhow::anyhow!("profiling sync failed: {e:?}"))?;
            }
            t_bwd += t2.elapsed().as_secs_f64();
        }

        gradients.insert(
            "variance".into(),
            tensor(vec![variance_grad], [1, 1], &field.device),
        );
        gradients.insert(
            "lin1.bias".into(),
            tensor(vec![sdf_bias_grad as f32], [1, 1], &field.device),
        );
        let factor = if step < 500 {
            step as f64 / 500.
        } else {
            let progress = (step - 500) as f64 / (args.end_iter - 500) as f64;
            ((progress * std::f64::consts::PI).cos() + 1.) * 0.5 * 0.95 + 0.05
        };
        let t_adam_start = std::time::Instant::now();
        field.apply_gradients_packed(gradients, 5e-4 * factor);
        field.step += 1;

        if field.step % 200 == 0 {
            field.trim_memory();
        } else {
            field
                .device
                .sync()
                .map_err(|e| anyhow::anyhow!("GPU synchronization failed: {e:?}"))?;
        }
        let t_adam = t_adam_start.elapsed().as_secs_f64();

        let secs = started.elapsed().as_secs_f64();
        println!(
            "step={} loss={:.7} normal={:.7} mask={:.7} eikonal={:.7} samples={} time={secs:.2}s",
            field.step,
            total.total(),
            total.normal,
            total.mask,
            total.eikonal,
            samples.len()
        );
        println!(
            "  occupancy={occupancy_seconds:.2}s sampling={sampling_seconds:.2}s (march={march_seconds:.2}s visible={visibility_seconds:.2}s) points={t_points:.2}s eval={t_eval:.2}s composite={t_comp:.2}s bwd={t_bwd:.2}s adam={t_adam:.2}s",
        );
        if field.step == final_step || args.profile_stages {
            println!(
                "Burn memory pool (not total driver/process usage): {:?}",
                field.device.memory_pool_usage()
            );
            println!(
                "Peak observed pool reservation before cleanup: {} bytes",
                field.peak_reserved_bytes.get()
            );
        }
        writeln!(
            metrics,
            "{},{},{},{},{},{},{},{}",
            field.step,
            total.normal,
            total.mask,
            total.eikonal,
            total.total(),
            samples.len(),
            secs,
            field.active
        )?;
        metrics.flush()?;
        if args.save_freq > 0 && field.step % args.save_freq == 0 || field.step == final_step {
            let dest = args.out.join(format!("checkpoint_{:06}", field.step));
            let temp = args
                .out
                .join(format!("checkpoint_{:06}.partial", field.step));
            field.save(&temp)?;
            let state = State {
                rng: rng.clone(),
                occupancy: Occupancy {
                    resolution: occ.resolution,
                    values: occ.values.clone(),
                    flags: occ.flags.clone(),
                },
                args: args.clone(),
            };
            std::fs::write(temp.join("training.json"), serde_json::to_vec(&state)?)?;
            std::fs::rename(temp, dest)?;
        }
        if args.mesh_freq > 0 && field.step % args.mesh_freq == 0 {
            crate::mesh::extract(
                &field,
                &args.out.join(format!("mesh_{:06}.ply", field.step)),
                args.mesh_resolution,
                args.query,
                dataset.scale,
                dataset.offset,
            )?;
        }
    }
    Ok(())
}
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
    /// Patches per overlapped forward/compositor group; 0 disables lookahead.
    #[arg(long, default_value_t = 0)]
    #[serde(default = "default_pipeline_patches")]
    pub pipeline_patches: usize,
    /// Use the separate hash encoder and Burn matmuls for comparison.
    #[arg(long)]
    #[serde(default)]
    pub unfused_field: bool,
    /// Retain the CPU compositor for numerical/performance comparison.
    #[arg(long)]
    #[serde(default)]
    pub cpu_composite: bool,
    /// Use CPU visibility filtering for comparison.
    #[arg(long)]
    #[serde(default)]
    pub cpu_visibility: bool,
    /// Use the reference CPU ray marcher.
    #[arg(long)]
    #[serde(default)]
    pub cpu_march: bool,
    /// Update occupancy values on the CPU.
    #[arg(long)]
    #[serde(default)]
    pub cpu_occupancy: bool,
    /// Use Burn matmul and reductions for first-layer parameter gradients.
    #[arg(long)]
    #[serde(default)]
    pub untiled_gradients: bool,
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
    /// Plain log/CSV flush interval; interactive progress refreshes at most 4 Hz.
    #[arg(long, default_value_t = 10)]
    #[serde(default = "default_log_every")]
    pub log_every: usize,
}

fn default_log_every() -> usize {
    10
}

fn default_pipeline_patches() -> usize {
    0
}

struct PreparedBatch {
    start: usize,
    end: usize,
    samples: Vec<render::Sample>,
    points: Vec<[f32; 3]>,
    indices: Vec<usize>,
    unique_count: usize,
    geometry: Option<burn::tensor::Tensor<2>>,
    pending: crate::field::PendingEvaluation,
}

fn prepare_batch(
    field: &Field,
    patches: &[crate::data::Patch],
    samples: &[render::Sample],
    start: usize,
    size: usize,
    query: usize,
    cpu_composite: bool,
) -> PreparedBatch {
    let end = (start + size).min(patches.len());
    let first = samples.partition_point(|s| s.patch < start);
    let last = samples.partition_point(|s| s.patch < end);
    let samples: Vec<_> = samples[first..last]
        .iter()
        .map(|s| render::Sample {
            patch: s.patch - start,
            start: s.start,
            end: s.end,
        })
        .collect();
    let (points, indices, unique_count, geometry, pending) = if cpu_composite || samples.is_empty()
    {
        let (points, unique, indices) =
            render::sample_and_dedup_points(&patches[start..end], &samples);
        let pending = field.enqueue_cached(&unique, query);
        (points, indices, unique.len(), None, pending)
    } else {
        let (unique, indices, geometry) =
            render::prepare_geometry_device(&patches[start..end], &samples, &field.device);
        let count = unique.dims()[0];
        let pending = field.enqueue_cached_device(unique, query);
        (Vec::new(), indices, count, Some(geometry), pending)
    };
    PreparedBatch {
        start,
        end,
        samples,
        points,
        indices,
        unique_count,
        geometry,
        pending,
    }
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
            && args.log_every > 0
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
    field.fused_field = !args.unfused_field;
    field.cpu_visibility = args.cpu_visibility;
    field.tiled_gradients = !args.untiled_gradients;
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
    occ.cpu_update = args.cpu_occupancy;
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

    let mut progress = crate::progress::TrainingProgress::new(final_step);
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
        let marched = if args.cpu_march {
            render::march(&patches, &occ, step_size, &mut rng)
        } else {
            render::march_device(&patches, &mut occ, step_size, &mut rng, &field.device)?
        };
        let march_seconds = march_started.elapsed().as_secs_f64();
        let visible_started = std::time::Instant::now();
        // Previous backward/Adam kernels run while we prepare patches and rays.
        // Visibility reads its SDFs and the updated variance in one transfer.
        let samples = render::visible(&mut field, &patches, marched, args.query)?;
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
        let mut device_stats: Option<burn::tensor::Tensor<2>> = None;
        let mut variance_grad = 0f32;
        let mut sdf_bias_grad = 0f64;
        let mut t_eval = 0.0;
        let mut t_points = 0.0;
        let mut t_comp = 0.0;
        let mut t_bwd = 0.0;

        let group_size = if args.pipeline_patches == 0 {
            args.microbatch
        } else {
            args.microbatch.min(args.pipeline_patches)
        };
        let t0 = std::time::Instant::now();
        let mut prepared = Some(prepare_batch(
            &field,
            &patches,
            &samples,
            0,
            group_size,
            args.query,
            args.cpu_composite,
        ));
        t_points += t0.elapsed().as_secs_f64();
        while let Some(batch) = prepared.take() {
            let PreparedBatch {
                start,
                end,
                samples: local,
                points,
                indices,
                unique_count,
                geometry,
                pending,
            } = batch;
            let pp = &patches[start..end];
            let t0 = std::time::Instant::now();
            if !args.cpu_composite && unique_count > 0 {
                let (sdf, cache) = pending.into_device();
                t_eval += t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let (adjoints, stats) = render::composite_device(
                    pp,
                    &local,
                    geometry.expect("GPU geometry"),
                    &indices,
                    sdf,
                    variance,
                    normal_den,
                    pixel_den,
                    sample_den,
                );
                device_stats = Some(match device_stats {
                    Some(old) => old + stats,
                    None => stats,
                });
                t_comp += t1.elapsed().as_secs_f64();
                let t2 = std::time::Instant::now();
                field.accumulate_gradients_device(cache, adjoints, &mut gradients);
                if args.profile_stages {
                    field
                        .device
                        .sync()
                        .map_err(|e| anyhow::anyhow!("profiling sync failed: {e:?}"))?;
                }
                t_bwd += t2.elapsed().as_secs_f64();
                if end < patches.len() {
                    prepared = Some(prepare_batch(
                        &field,
                        &patches,
                        &samples,
                        end,
                        group_size,
                        args.query,
                        args.cpu_composite,
                    ));
                }
                continue;
            }
            let (unique_sdf, cache) = pending.finish()?;
            // Submit the next forward before CPU compositing. Only two groups'
            // activations are live, and all groups use the same model weights.
            if end < patches.len() && args.pipeline_patches > 0 {
                prepared = Some(prepare_batch(
                    &field,
                    &patches,
                    &samples,
                    end,
                    group_size,
                    args.query,
                    args.cpu_composite,
                ));
                field
                    .device
                    .flush()
                    .map_err(|e| anyhow::anyhow!("forward submission failed: {e:?}"))?;
            }
            let sdf: Vec<f32> = indices
                .iter()
                .map(|&i| unsafe { *unique_sdf.get_unchecked(i) })
                .collect();
            t_eval += t0.elapsed().as_secs_f64();

            let t1 = std::time::Instant::now();
            let result = render::composite(
                pp, &local, &points, &sdf, variance, normal_den, pixel_den, sample_den,
            );
            t_comp += t1.elapsed().as_secs_f64();

            total.normal += result.loss.normal;
            total.mask += result.loss.mask;
            total.eikonal += result.loss.eikonal;
            variance_grad += result.variance_grad;
            sdf_bias_grad += f64::from(result.sdf_bias_grad);

            let mut adjoints = vec![0f32; unique_count];
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
            if end < patches.len() && prepared.is_none() {
                prepared = Some(prepare_batch(
                    &field,
                    &patches,
                    &samples,
                    end,
                    group_size,
                    args.query,
                    args.cpu_composite,
                ));
            }
        }

        // Keep the two optimizer scalars device-resident. Reading loss before
        // Adam used to force an idle gap between backward and the update.
        if let Some(stats) = &device_stats {
            gradients.insert(
                "variance".into(),
                stats.clone().slice([0..1, 3..4]) + variance_grad,
            );
            gradients.insert(
                "lin1.bias".into(),
                stats.clone().slice([0..1, 4..5]) + sdf_bias_grad as f32,
            );
        } else {
            gradients.insert(
                "variance".into(),
                tensor(vec![variance_grad], [1, 1], &field.device),
            );
            gradients.insert(
                "lin1.bias".into(),
                tensor(vec![sdf_bias_grad as f32], [1, 1], &field.device),
            );
        }
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
        } else if args.profile_stages || field.step == final_step {
            field
                .device
                .sync()
                .map_err(|e| anyhow::anyhow!("GPU synchronization failed: {e:?}"))?;
        } else {
            // Submit lazy work without waiting for the GPU to become idle.
            // The next visibility/field readback is the required dependency.
            field
                .device
                .flush()
                .map_err(|e| anyhow::anyhow!("GPU submission failed: {e:?}"))?;
        }
        let t_adam = t_adam_start.elapsed().as_secs_f64();

        // Read metrics only after submitting the optimizer, so the CPU cannot
        // delay the update while waiting for compositor statistics.
        if let Some(stats) = device_stats {
            let values = stats.slice([0..1, 0..3]).try_into_vec_as::<f32>()?;
            total.normal += values[0] as f64;
            total.mask += values[1] as f64;
            total.eikonal += values[2] as f64;
        }
        let secs = started.elapsed().as_secs_f64();
        let report =
            args.profile_stages || field.step % args.log_every == 0 || field.step == final_step;
        if args.profile_stages {
            progress.clear();
            println!(
                "step={} occupancy={occupancy_seconds:.3}s sampling={sampling_seconds:.3}s (march={march_seconds:.3}s visible={visibility_seconds:.3}s) points={t_points:.3}s eval={t_eval:.3}s composite={t_comp:.3}s bwd={t_bwd:.3}s adam={t_adam:.3}s",
                field.step,
            );
        }
        if args.profile_stages {
            println!(
                "Burn memory pool (not total driver/process usage): {:?}",
                field.device.memory_pool_usage()
            );
            println!(
                "Peak observed pool reservation before cleanup: {} bytes",
                field.peak_reserved_bytes.get()
            );
        }
        progress.update(field.step, secs, total.total(), report);
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
        if report {
            metrics.flush()?;
        }
        if args.save_freq > 0 && field.step % args.save_freq == 0 || field.step == final_step {
            progress.clear();
            metrics.flush()?;
            let dest = args.out.join(format!("checkpoint_{:06}", field.step));
            eprintln!("Saving checkpoint {}", dest.display());
            let temp = args
                .out
                .join(format!("checkpoint_{:06}.partial", field.step));
            field.save(&temp)?;
            occ.sync_values()?;
            let state = State {
                rng: rng.clone(),
                occupancy: Occupancy {
                    resolution: occ.resolution,
                    values: occ.values.clone(),
                    flags: occ.flags.clone(),
                    device_values: None,
                    device_flags: None,
                    cpu_update: false,
                },
                args: args.clone(),
            };
            std::fs::write(temp.join("training.json"), serde_json::to_vec(&state)?)?;
            std::fs::rename(temp, dest)?;
        }
        if args.mesh_freq > 0 && field.step % args.mesh_freq == 0 {
            progress.clear();
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

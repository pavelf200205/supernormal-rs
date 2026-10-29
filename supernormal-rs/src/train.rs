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
    /// Restore point-major hash lookup/scatter ownership for comparison.
    #[arg(long)]
    #[serde(default)]
    pub legacy_hash_layout: bool,
    /// Restore per-point output-weight gradients instead of workgroup partials.
    #[arg(long)]
    #[serde(default)]
    pub legacy_output_gradients: bool,
    /// Restore the original 8x8 first-layer gradient tiles.
    #[arg(long)]
    #[serde(default)]
    pub legacy_gradient_tiles: bool,
    /// Restore 56-thread sequential-corner hash forward/backward instead of 256-thread corner parallelism.
    #[arg(long)]
    #[serde(default)]
    pub legacy_corner_loop: bool,
    /// Disable local same-address hash-gradient aggregation.
    #[arg(long)]
    #[serde(default)]
    pub no_hash_aggregation: bool,
    /// Restore pixel-major compositor scratch storage.
    #[arg(long)]
    #[serde(default)]
    pub legacy_composite_scratch: bool,
    /// Restore ten-float compositor records and separate endpoint derivative loads.
    #[arg(long)]
    #[serde(default)]
    pub legacy_composite_cache: bool,
    #[arg(long)]
    #[serde(default)]
    pub legacy_adam_dispatch: bool,
    /// Update/write every active hash entry, including entirely zero Adam state.
    #[arg(long)]
    #[serde(default)]
    pub dense_hash_adam: bool,
    /// Restore per-query adjoint slice/copy operations in fused training backward.
    #[arg(long)]
    #[serde(default)]
    pub legacy_adjoint_slicing: bool,
    /// Compute first-layer gradients for all 31 features before levels activate.
    #[arg(long)]
    #[serde(default)]
    pub dense_parameter_gradients: bool,
    /// Stage all packed MLP weights in backward, including unused columns/bias.
    #[arg(long)]
    #[serde(default)]
    pub dense_backward_weight_loads: bool,
    #[arg(long)]
    #[serde(default)]
    pub no_endpoint_reuse: bool,
    /// Retain the former three-full-chunk activation cache for comparison.
    #[arg(long)]
    #[serde(default)]
    pub legacy_activation_cache: bool,
    /// Materialize feature-only preactivations in a separate dispatch for comparison.
    #[arg(long)]
    #[serde(default)]
    pub legacy_feature_replay: bool,
    /// Restore a separate full forward replay for chunks outside the feature cache.
    #[arg(long)]
    #[serde(default)]
    pub legacy_hash_replay: bool,
    /// Stage every first-layer weight in forward before hash levels activate.
    #[arg(long)]
    #[serde(default)]
    pub dense_forward_weight_loads: bool,
    /// Restore occupancy count readback, chunked field queries and separate EMA application.
    #[arg(long)]
    #[serde(default)]
    pub legacy_occupancy_readback: bool,
    /// Restore one-output-per-thread parameter-gradient tiles.
    #[arg(long)]
    #[serde(default)]
    pub legacy_parameter_kernel: bool,
    /// Restore 16-neuron parameter tiles instead of sharing features across 32 neurons.
    #[arg(long)]
    #[serde(default)]
    pub legacy_parameter_rows: bool,
    /// Experimental: fill spare feature-cache capacity with full preactivations.
    #[arg(long, conflicts_with = "no_hybrid_cache")]
    #[serde(default)]
    pub hybrid_activation_cache: bool,
    /// Explicitly disable experimental hybrid caching (already off by default).
    #[arg(long)]
    #[serde(default)]
    pub no_hybrid_cache: bool,
    /// Use Burn transpose and concatenation instead of direct packing of normalized weights.
    #[arg(long)]
    #[serde(default)]
    pub legacy_mlp_packing: bool,
    /// Forward workgroup weight-reuse tile: 4, 8, or 16 points.
    #[arg(long, default_value_t = 4)]
    #[serde(default = "default_field_points")]
    pub field_points_per_group: usize,
    #[arg(long)]
    #[serde(default)]
    pub legacy_field_reduction: bool,
    #[arg(long, default_value_t = 14)]
    #[serde(default = "default_aggregation_levels")]
    pub hash_aggregation_levels: usize,
    #[arg(long)]
    #[serde(default)]
    pub hash_aggregation_cross_corners: bool,
    /// Avoid the global dh buffer by recomputing derivatives in parameter tiles.
    #[arg(long)]
    #[serde(default)]
    pub recompute_parameter_gradients: bool,
    /// Restore host sample readbacks between march, visibility, and geometry preparation.
    #[arg(long)]
    #[serde(default)]
    pub host_sample_compaction: bool,
    /// Split device-resident forward/backward queries into `--query` chunks instead of single-dispatch evaluation.
    #[arg(long)]
    #[serde(default)]
    pub chunked_eval: bool,
    /// Gather 3x3 patch ray/normal/inverse structs on the CPU instead of device-resident view tables.
    #[arg(long)]
    #[serde(default)]
    pub cpu_patch_upload: bool,
    /// Read loss statistics synchronously at the end of every step instead of piggybacking on the next step's march readback.
    #[arg(long)]
    #[serde(default)]
    pub sync_loss: bool,
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
    /// Optional per-custom-op timing windows and logical work/copy counters.
    #[arg(long)]
    #[serde(default)]
    pub diagnostics: bool,
    /// Maintenance interval; zero disables periodic trimming (warmup/mesh cleanup remain).
    #[arg(long, default_value_t = 200)]
    #[serde(default = "default_trim_interval")]
    pub trim_interval: usize,
    /// Plain log/CSV flush interval; interactive progress refreshes at most 4 Hz.
    #[arg(long, default_value_t = 10)]
    #[serde(default = "default_log_every")]
    pub log_every: usize,
}

fn default_field_points() -> usize {
    4
}
fn default_aggregation_levels() -> usize {
    14
}

fn default_trim_interval() -> usize {
    200
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
    device_tables: Option<render::DevicePreparedGeometry>,
    pending: Option<crate::field::PendingEvaluation>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch(
    field: &Field,
    dataset: &Dataset,
    patches: &[crate::data::Patch],
    total_patches: usize,
    samples: &[render::Sample],
    device_vis: Option<&render::DeviceVisibleBatch>,
    start: usize,
    size: usize,
    query: usize,
    variance: f32,
    normal_den: f32,
    pixel_den: f32,
    sample_den: f32,
    cpu_composite: bool,
) -> PreparedBatch {
    let end = (start + size).min(total_patches);
    if let Some(vis_batch) = device_vis {
        if let Some(prep) = render::prepare_geometry_from_device_batch(
            dataset,
            patches,
            total_patches,
            vis_batch,
            start,
            end,
            query,
            variance,
            normal_den,
            pixel_den,
            sample_den,
            &field.device,
        ) {
            let count = prep.valid_points;
            let pending =
                field.enqueue_cached_device_valid(prep.unique_points.clone(), count, query);
            return PreparedBatch {
                start,
                end,
                samples: Vec::new(),
                points: Vec::new(),
                indices: Vec::new(),
                unique_count: count,
                geometry: None,
                device_tables: Some(prep),
                pending: Some(pending),
            };
        } else {
            return PreparedBatch {
                start,
                end,
                samples: Vec::new(),
                points: Vec::new(),
                indices: Vec::new(),
                unique_count: 0,
                geometry: None,
                device_tables: None,
                pending: None,
            };
        }
    }
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
        (points, indices, unique.len(), None, Some(pending))
    } else {
        let (unique, count, indices, geometry) =
            render::prepare_geometry_device(&patches[start..end], &samples, query, &field.device);
        let pending = field.enqueue_cached_device_valid(unique, count, query);
        (Vec::new(), indices, count, Some(geometry), Some(pending))
    };
    PreparedBatch {
        start,
        end,
        samples,
        points,
        indices,
        unique_count,
        geometry,
        device_tables: None,
        pending,
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    rng: ChaCha8Rng,
    occupancy: Occupancy,
    args: TrainArgs,
}

struct DeferredStepMetric {
    step: usize,
    samples: usize,
    secs: f64,
    active: usize,
    report: bool,
    stats: burn::tensor::Tensor<2>,
}

pub fn train(args: TrainArgs, device: Device) -> Result<()> {
    let command_clock = std::time::Instant::now();
    let mut checkpoint_seconds = 0.0f64;
    let mut mesh_seconds = 0.0f64;
    crate::diagnostics::configure(args.diagnostics);
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
    field.local_hash = !args.legacy_hash_layout;
    field.compact_gradients = !args.legacy_output_gradients;
    field.wide_gradient_tiles = !args.legacy_gradient_tiles;
    field.corner_parallel = !args.legacy_corner_loop;
    field.aggregate_hash = !args.no_hash_aggregation;
    ensure!(
        [4, 8, 16].contains(&args.field_points_per_group),
        "field-points-per-group must be 4, 8, or 16"
    );
    ensure!(
        args.hash_aggregation_levels <= 14,
        "hash-aggregation-levels must be <= 14"
    );
    field.point_tiles = args.field_points_per_group / 4;
    field.subgroup_reduce = !args.legacy_field_reduction;
    field.aggregation_levels = args.hash_aggregation_levels;
    field.cross_corners = args.hash_aggregation_cross_corners;
    field.recompute_parameter_gradients = args.recompute_parameter_gradients;
    field.level_adam = !args.legacy_adam_dispatch;
    field.skip_untouched_adam = !args.dense_hash_adam;
    field.direct_adjoint_views = !args.legacy_adjoint_slicing;
    field.active_parameter_columns = !args.dense_parameter_gradients;
    field.selective_backward_weights = !args.dense_backward_weight_loads;
    field.reuse_endpoints = !args.no_endpoint_reuse;
    field.feature_only_cache = !args.legacy_activation_cache;
    field.fused_feature_replay = !args.legacy_feature_replay;
    field.fused_hash_replay = !args.legacy_hash_replay;
    field.selective_forward_weights = !args.dense_forward_weight_loads;
    field.device_occupancy_update = !args.legacy_occupancy_readback;
    field.register_gradient_tiles = !args.legacy_parameter_kernel;
    field.wide_parameter_rows = !args.legacy_parameter_rows;
    field.hybrid_activation_cache = args.hybrid_activation_cache && !args.no_hybrid_cache;
    field.fused_mlp_packing = !args.legacy_mlp_packing;
    field.chunked_eval = args.chunked_eval;
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

    let use_device_pipeline = !args.cpu_march
        && !args.cpu_visibility
        && !args.cpu_composite
        && !args.host_sample_compaction;
    let use_gpu_patch_gather = use_device_pipeline && !args.cpu_patch_upload;
    if use_gpu_patch_gather {
        let _ = dataset.device_dataset(&field.device);
    }

    let mut progress = crate::progress::TrainingProgress::new(final_step);
    let mut first = true;
    let mut deferred_metric: Option<DeferredStepMetric> = None;

    let flush_deferred = |dm: DeferredStepMetric,
                          loss_override: Option<Loss>,
                          progress: &mut crate::progress::TrainingProgress,
                          metrics: &mut std::io::BufWriter<std::fs::File>|
     -> Result<()> {
        let loss = if let Some(l) = loss_override {
            l
        } else {
            let values = dm.stats.slice([0..1, 0..3]).try_into_vec_as::<f32>()?;
            Loss {
                normal: values[0] as f64,
                mask: values[1] as f64,
                eikonal: values[2] as f64,
            }
        };
        progress.update(dm.step, dm.secs, loss.total(), dm.report);
        writeln!(
            metrics,
            "{},{},{},{},{},{},{},{}",
            dm.step,
            loss.normal,
            loss.mask,
            loss.eikonal,
            loss.total(),
            dm.samples,
            dm.secs,
            dm.active
        )?;
        if dm.report {
            metrics.flush()?;
        }
        Ok(())
    };

    while field.step < final_step {
        let started = std::time::Instant::now();
        let step = field.step;
        if (first && !restored_state) || step % 8 == 0 {
            occ.update(
                &field,
                if first && !restored_state { 0 } else { step },
                args.query,
                &mut rng,
            )?;
            if args.cpu_march && !occ.cpu_update {
                occ.sync_values()?;
            }
        }
        first = false;
        let occupancy_seconds = started.elapsed().as_secs_f64();
        field.active = (step / 2000 + 1).min(14);
        let (patches, patch_coords, normal_den) = if use_gpu_patch_gather {
            let (coords, nd) = dataset.random_patch_coords(args.batch, &mut rng);
            (Vec::new(), Some(coords), nd)
        } else {
            let p = dataset.random_patches(args.batch, &mut rng);
            let nd = p.iter().flat_map(|patch| patch.mask).sum::<f32>() + 1e-5;
            (p, None, nd)
        };
        let total_patches = args.batch;
        let step_size = 10f32.powf(-2. - step as f32 / args.end_iter as f32);

        let group_size = if args.pipeline_patches == 0 {
            args.microbatch
        } else {
            args.microbatch.min(args.pipeline_patches)
        };

        let (samples, device_vis, visible_count, march_seconds, visibility_seconds) =
            if use_device_pipeline {
                let prev_stats = deferred_metric.as_ref().map(|dm| dm.stats.clone());
                let (vis_batch, prev_loss, m_sec, v_sec) = render::march_and_filter_device(
                    &mut field,
                    &dataset,
                    patch_coords.as_deref(),
                    &patches,
                    &mut occ,
                    step_size,
                    args.query,
                    group_size,
                    prev_stats,
                    &mut rng,
                )?;
                if let Some(dm) = deferred_metric.take() {
                    flush_deferred(dm, prev_loss, &mut progress, &mut metrics)?;
                }
                let count = vis_batch.total_visible;
                (Vec::new(), Some(vis_batch), count, m_sec, v_sec)
            } else {
                if let Some(dm) = deferred_metric.take() {
                    flush_deferred(dm, None, &mut progress, &mut metrics)?;
                }
                let march_started = std::time::Instant::now();
                let marched = if args.cpu_march {
                    render::march(&patches, &occ, step_size, &mut rng)
                } else {
                    render::march_device(&patches, &mut occ, step_size, &mut rng, &field.device)?
                };
                let m_sec = march_started.elapsed().as_secs_f64();
                let visible_started = std::time::Instant::now();
                // Previous backward/Adam kernels run while we prepare patches and rays.
                // Visibility reads its SDFs and the updated variance in one transfer.
                let samples = render::visible(&mut field, &patches, marched, args.query)?;
                let v_sec = visible_started.elapsed().as_secs_f64();
                let count = samples.len();
                (samples, None, count, m_sec, v_sec)
            };

        if visible_count == 0 {
            anyhow::bail!(
                "empty occupancy batch at step {step}; checkpoint/config requires investigation"
            );
        }
        let sampling_seconds = started.elapsed().as_secs_f64() - occupancy_seconds;
        let pixel_den = (total_patches * 9) as f32;
        let sample_den = (visible_count * 9) as f32;

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

        let t0 = std::time::Instant::now();
        let mut prepared = Some(prepare_batch(
            &field,
            &dataset,
            &patches,
            total_patches,
            &samples,
            device_vis.as_ref(),
            0,
            group_size,
            args.query,
            variance,
            normal_den,
            pixel_den,
            sample_den,
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
                device_tables,
                pending,
            } = batch;
            let patch_count = end - start;
            let pp = if patches.is_empty() {
                &[][..]
            } else {
                &patches[start..end]
            };
            let t0 = std::time::Instant::now();
            if !args.cpu_composite && unique_count == 0 {
                if end < total_patches {
                    let tp = std::time::Instant::now();
                    prepared = Some(prepare_batch(
                        &field,
                        &dataset,
                        &patches,
                        total_patches,
                        &samples,
                        device_vis.as_ref(),
                        end,
                        group_size,
                        args.query,
                        variance,
                        normal_den,
                        pixel_den,
                        sample_den,
                        args.cpu_composite,
                    ));
                    t_points += tp.elapsed().as_secs_f64();
                }
                continue;
            }
            if !args.cpu_composite && unique_count > 0 {
                let (sdf, cache) = pending.expect("GPU pending evaluation").into_device();
                t_eval += t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let (adjoints, stats) = if let Some(tables) = device_tables {
                    render::composite_device_tables(
                        pp,
                        patch_count,
                        tables.patch_start,
                        tables.s_base,
                        tables.visible_count,
                        tables.vis_prefix,
                        tables.packed_patches,
                        tables.geometry,
                        tables.mapping,
                        tables.reverse,
                        sdf,
                        variance,
                        normal_den,
                        pixel_den,
                        sample_den,
                        !args.legacy_composite_scratch,
                        !args.legacy_composite_cache,
                        !args.no_endpoint_reuse,
                    )
                } else {
                    render::composite_device(
                        pp,
                        &local,
                        geometry.expect("GPU geometry"),
                        &indices,
                        sdf,
                        variance,
                        normal_den,
                        pixel_den,
                        sample_den,
                        !args.legacy_composite_scratch,
                        !args.legacy_composite_cache,
                        !args.no_endpoint_reuse,
                    )
                };
                device_stats = Some(match device_stats {
                    Some(old) => old + stats,
                    None => stats,
                });
                t_comp += t1.elapsed().as_secs_f64();
                let t2 = std::time::Instant::now();
                field.accumulate_training_gradients_device(cache, adjoints, &mut gradients);
                if args.profile_stages {
                    field
                        .device
                        .sync()
                        .map_err(|e| anyhow::anyhow!("profiling sync failed: {e:?}"))?;
                }
                t_bwd += t2.elapsed().as_secs_f64();
                if end < total_patches {
                    let tp = std::time::Instant::now();
                    prepared = Some(prepare_batch(
                        &field,
                        &dataset,
                        &patches,
                        total_patches,
                        &samples,
                        device_vis.as_ref(),
                        end,
                        group_size,
                        args.query,
                        variance,
                        normal_den,
                        pixel_den,
                        sample_den,
                        args.cpu_composite,
                    ));
                    t_points += tp.elapsed().as_secs_f64();
                }
                continue;
            }
            let (unique_sdf, cache) = pending.expect("CPU pending evaluation").finish()?;
            // Submit the next forward before CPU compositing. Only two groups'
            // activations are live, and all groups use the same model weights.
            if end < total_patches && args.pipeline_patches > 0 {
                prepared = Some(prepare_batch(
                    &field,
                    &dataset,
                    &patches,
                    total_patches,
                    &samples,
                    device_vis.as_ref(),
                    end,
                    group_size,
                    args.query,
                    variance,
                    normal_den,
                    pixel_den,
                    sample_den,
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
            if end < total_patches && prepared.is_none() {
                prepared = Some(prepare_batch(
                    &field,
                    &dataset,
                    &patches,
                    total_patches,
                    &samples,
                    device_vis.as_ref(),
                    end,
                    group_size,
                    args.query,
                    variance,
                    normal_den,
                    pixel_den,
                    sample_den,
                    args.cpu_composite,
                ));
            }
        }

        // Keep the two optimizer scalars device-resident. Reading loss before
        // Adam used to force an idle gap between backward and the update.
        if let Some(stats) = &device_stats {
            if gradients.mlp_raw.is_some() && variance_grad == 0.0 && sdf_bias_grad == 0.0 {
                gradients.device_stats = Some(stats.clone());
            } else {
                gradients.insert(
                    "variance".into(),
                    stats.clone().slice([0..1, 3..4]) + variance_grad,
                );
                gradients.insert(
                    "lin1.bias".into(),
                    stats.clone().slice([0..1, 4..5]) + sdf_bias_grad as f32,
                );
            }
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

        if field.step == 1000 {
            // Release early warmup high-water buffers once the initial 1-900 step
            // sample spike subsides so VRAM stays at its steady low baseline.
            let _ = field.device.sync();
            field.device.memory_cleanup();
        } else if args.trim_interval > 0 && field.step % args.trim_interval == 0 {
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

        let secs = started.elapsed().as_secs_f64();
        let report =
            args.profile_stages || field.step % args.log_every == 0 || field.step == final_step;
        let will_checkpoint =
            args.save_freq > 0 && field.step % args.save_freq == 0 || field.step == final_step;
        let will_mesh = args.mesh_freq > 0 && field.step % args.mesh_freq == 0;

        if args.profile_stages {
            progress.clear();
            println!(
                "step={} occupancy={occupancy_seconds:.3}s sampling={sampling_seconds:.3}s (march={march_seconds:.3}s visible={visibility_seconds:.3}s) points={t_points:.3}s eval={t_eval:.3}s composite={t_comp:.3}s bwd={t_bwd:.3}s adam={t_adam:.3}s",
                field.step,
            );
            println!(
                "Burn memory pool (not total driver/process usage): {:?}",
                field.device.memory_pool_usage()
            );
            println!(
                "Peak observed pool reservation before cleanup: {} bytes",
                field.peak_reserved_bytes.get()
            );
        }

        // Defer reading `device_stats` until the next iteration's march readback
        // so the GPU executes forward + composite + backward + Adam asynchronously
        // while the CPU immediately prepares the next iteration's patches.
        if let Some(stats) = device_stats {
            if use_device_pipeline
                && !args.sync_loss
                && !args.profile_stages
                && !will_checkpoint
                && !will_mesh
            {
                deferred_metric = Some(DeferredStepMetric {
                    step: field.step,
                    samples: visible_count,
                    secs,
                    active: field.active,
                    report,
                    stats,
                });
            } else {
                let values = stats.slice([0..1, 0..3]).try_into_vec_as::<f32>()?;
                total.normal += values[0] as f64;
                total.mask += values[1] as f64;
                total.eikonal += values[2] as f64;
                progress.update(field.step, secs, total.total(), report);
                writeln!(
                    metrics,
                    "{},{},{},{},{},{},{},{}",
                    field.step,
                    total.normal,
                    total.mask,
                    total.eikonal,
                    total.total(),
                    visible_count,
                    secs,
                    field.active
                )?;
                if report {
                    metrics.flush()?;
                }
            }
        } else {
            progress.update(field.step, secs, total.total(), report);
            writeln!(
                metrics,
                "{},{},{},{},{},{},{},{}",
                field.step,
                total.normal,
                total.mask,
                total.eikonal,
                total.total(),
                visible_count,
                secs,
                field.active
            )?;
            if report {
                metrics.flush()?;
            }
        }

        if will_checkpoint {
            let checkpoint_clock = std::time::Instant::now();
            if let Some(dm) = deferred_metric.take() {
                flush_deferred(dm, None, &mut progress, &mut metrics)?;
            }
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
                    est_marched: 0,
                    est_endpoints: 0,
                },
                args: args.clone(),
            };
            std::fs::write(temp.join("training.json"), serde_json::to_vec(&state)?)?;
            std::fs::rename(temp, dest)?;
            checkpoint_seconds += checkpoint_clock.elapsed().as_secs_f64();
        }
        if will_mesh {
            let mesh_clock = std::time::Instant::now();
            if let Some(dm) = deferred_metric.take() {
                flush_deferred(dm, None, &mut progress, &mut metrics)?;
            }
            progress.clear();
            crate::mesh::extract(
                &field,
                &args.out.join(format!("mesh_{:06}.ply", field.step)),
                args.mesh_resolution,
                args.query,
                dataset.scale,
                dataset.offset,
            )?;
            let _ = field.device.sync();
            field.device.memory_cleanup();
            mesh_seconds += mesh_clock.elapsed().as_secs_f64();
        }
    }
    if let Some(dm) = deferred_metric.take() {
        flush_deferred(dm, None, &mut progress, &mut metrics)?;
    }
    field
        .device
        .sync()
        .map_err(|e| anyhow::anyhow!("final synchronization failed: {e:?}"))?;
    let elapsed = command_clock.elapsed().as_secs_f64();
    std::fs::write(
        args.out.join("wall_time.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "command_seconds": elapsed, "includes_loading_checkpointing_and_mesh_extraction": true,
            "final_step": field.step, "diagnostics_enabled": args.diagnostics,
            "checkpoint_seconds": checkpoint_seconds, "mesh_seconds": mesh_seconds,
            "seconds_excluding_checkpoint_and_mesh": (elapsed - checkpoint_seconds - mesh_seconds).max(0.0),
            "excludes_diagnostic_report_serialization": true,
        }))?,
    )?;
    crate::diagnostics::report(&args.out.join("diagnostics.json"))?;
    Ok(())
}

# SuperNormal-RS: optimization history and engineering findings

Snapshot: 2026-10-29, through commit `6d28744` and completed experiment
`after_george_kopanas_upgrades12`. This is a consolidated handoff document.
It combines repository evidence with observations reported during development.

## 1. Evidence and scope

- **Saved measurements:** experiment `metrics.csv`, `config.json` and
  `wall_time.json` files. These support completed-run comparisons below.
- **Historical validation:** numerical checks recorded in `VALIDATION.md`.
  They validate the versions and fixtures tested at the time, not every later
  kernel or every possible configuration.
- **User observations:** earlier reconstruction times, GPU power, total-system
  VRAM, mesh appearance and Radeon 840M results reported in the conversation.
  These are valuable observations but not controlled hardware captures.
- **Source inspection:** establishes implemented mechanisms and defaults, not
  actual cache hit rates, register allocation, occupancy or achieved bandwidth.

The recent optimization commits were developed without running builds, tests,
training or GPU captures, at the user's request. Subsequent full runs were
performed by the user. This documentation update launches none of those tasks.
Existing Markdown files are historical snapshots; current source and saved
run evidence take precedence over outdated performance claims.

## 2. Goal and starting point

The goal was a native reconstruction application using Rust, Burn and WGPU,
without Python/CUDA installation as a runtime requirement. Scope is the actual
SuperNormal reconstruction pipeline. Dataset preparation, particularly SAM,
was explicitly excluded.

The source repository was `pavelf200205/SuperNormal_memory_optimized`, with
`main`, `sub2gb` and `pure_pytorch_version` representing different stages of
memory reduction and portability work. The original CUDA implementation used
tiny-cuda-nn and nerfacc. The pure-PyTorch branch provided the more portable
algorithmic reference; the CUDA version remained a performance/quality reference.
Exact dependency revisions belong to the original environment setup script;
this document does not invent or substitute commit IDs for those dependencies.

User-reported starting observations:

| Implementation | Observation |
|---|---|
| Original SuperNormal before memory reductions | Nearly filled a 12 GB GPU and could run out of memory |
| Memory-optimized main branch | About 3.9 GB total-system VRAM during training |
| sub2gb branch | About 1.9 GB total-system VRAM; batch 1024, slower and somewhat lower quality |
| Pure PyTorch | About 2.6 GB total-system VRAM; roughly 3x CUDA training time |

Those memory figures include Windows/driver/other system usage and are not
allocator live-byte measurements. They are not directly comparable to Burn's
reserved-memory counters or guaranteed minimum GPU capacities.

The pure-PyTorch 30k horse mesh looked very close to the CUDA result. Its 5k
and 10k meshes had less detail. That does not establish that reduced memory
caused slower detail formation; sampling, optimizer trajectories, batch sizes,
precision and numerical differences are alternative explanations.

## 3. What tiny-cuda-nn and nerfacc taught us

Their advantage is not simply the CUDA language. Important principles include:

- Fuse small operations to reduce dispatch overhead and intermediate tensors.
- Reuse small MLP weights and activations close to the arithmetic that uses them.
- Expose many independent hash lookups and avoid serial work in mostly idle groups.
- Keep ray sampling, compaction and volume rendering on the device.
- Avoid unnecessary sample evaluations, host transfers and synchronization.
- Use compact representations, suitable memory layouts and efficient gradient scatter.

Several of these principles have now been implemented in portable CubeCL/WGPU
kernels. This does not reproduce tiny-cuda-nn's entire execution model or its
CUDA-specific matrix primitives, and the application still has several distinct
GPU passes. “Fused backward” refers to a field backward kernel, not one kernel
covering sampling, every loss, every parameter gradient and Adam.

## 4. Development history

### 4.1 Correctness-first native reconstruction

`9ddfe4b` introduced the Rust/Burn rewrite with analytical backward and mesh
extraction. Early development and Gemini work fixed gradient mismatches until
the native pipeline produced recognizable and then correct-looking meshes.

Historical checkpoints and fixtures were essential: forward SDF agreement alone
was insufficient. Output-layer weight-normalization gradients, output-bias
cancellation, renderer adjoints and optimizer updates required separate checks.
The early native pipeline was much slower than the CUDA baseline.

### 4.2 Move hash work and repeated field evaluation onto the GPU

| Commit | Main change |
|---|---|
| `bbf7894` | Feature chunk caching, point deduplication and parallel host scatter |
| `5890b3a` | CubeCL/WGPU hash interpolation and CAS atomic gradient scatter |
| `71aa60e` | Packed hash evaluation and corrected output-weight gradients |
| `c52a731` | Historical approximately 1 it/s milestone |
| `82a90aa` | Historical 15 it/s milestone |
| `c5fc22d` | Further intermediate speed work |
| `b3230f1` | Updating console progress display with it/s |
| `74c760f` | Fused hash-grid plus MLP forward, enabled by default |

Earlier code could spend substantial time evaluating the same field during
visibility, training and backward. Packing levels and caching normalized weights
removed repeated work and avoided many small dispatches. Inference graph
lifetime also mattered: inference must detach from autodiff, not retain a large
training graph inadvertently.

### 4.3 GPU rendering, backward, sampling and geometry

| Commit | Main change |
|---|---|
| `4229418` | GPU compositing/loss derivatives and fused field backward |
| `45bab15` | GPU visibility, smaller readbacks, nine-pixel compositing parallelism, shared MLP weights, native float atomic add with CAS fallback |
| `c61213e` | GPU ray marching with voxel skipping, stable compaction, occupancy updates and tiled first-layer gradients |
| `ded7dbc` | GPU point generation and finite-difference geometry; device variance/bias handling removes another optimizer readback |
| `72306cf` | Remove mesh extraction at iteration zero |
| `24a14d6` | Improve cache locality |
| `c3bcb39` | Further memory-traffic improvements |
| `ac1b3a4` | Parallelize hash corners and keep visibility/geometry tables on the GPU |

User-reported full-run milestones progressed from approximately 85 minutes,
through 34 minutes, to just under 27 minutes with 1024-resolution exports every
10k steps. Other earlier comparisons used different export frequencies and
resolutions; these are milestones, not a controlled speedup series.

The no-iteration-zero request was implemented explicitly. Expensive exports
should follow the requested schedule rather than run unconditionally at startup.

### 4.4 Cache locality and memory traffic

Neighboring points grouped by hash level improve access organization. Corner
parallelism spreads interpolation/scatter across up to 224 of 256 threads,
instead of leaving most threads idle while a small subset loops over corners.
At fewer active levels, fewer threads perform useful hash work.

Other changes included precomputed ray bases, device patch gathering, stable
endpoint reuse, compact mapping/reverse tables, uninitialized output allocation
when every consumed entry is written, and bucketed capacity reuse to reduce
allocator churn. Exact equal endpoints can share field evaluation; approximate
point merging would alter the calculation and is a different proposal.

The user reported a prior locality comparison improving average instantaneous
rates from 25.94 to 28.54 it/s and late rates from 10.33 to 13.92 it/s. These are
historical reported statistics, not recomputed full-wall-time gains. Averaging
instantaneous rates differs from dividing steps by elapsed time.

### 4.5 George Kopanas adaptations and subsequent commits

| Commit | Change / finding |
|---|---|
| `03e589d` | George Kopanas upgrades part 1 snapshot |
| `0653795` | Part 2: local same-address hash-gradient aggregation and component-major compositor scratch |
| `0f7a659` | Scheduling improvements and microbatch bounds fix |
| `f2f3a57` | Budgeted feature-only activation cache |
| `5ebd691` | Fused feature replay in backward and register-tiled parameter gradients |
| `9806d3b` | Hybrid activation caching plus custom raw-weight normalization/packing experiment; followed by run-8 failure |
| `567cfd2` | Restore Burn normalization; hybrid caching off; copy-only normalized-weight packing; remove final redundant tile barrier |
| `d7c4bd9` | Compact compositor scratch and precombined endpoint derivatives |
| `6a3ac1d` | Zero-state Adam fast path, active parameter columns, direct adjoints and selective backward weights |
| `4523d4f` | Fused hash replay on cache misses; GPU-driven occupancy and selective forward weights |
| `6d28744` | 32-neuron parameter-gradient tiles, halving repeated feature staging |

## 5. Completed-run measurements

The table below is recomputed from the saved files. A blank wall time means no
completed wall-time summary was found. Run numbers identify experiment folders,
not guaranteed binary hashes; not every folder records an independently verified
executable revision.

| Run | Logged steps | Command wall seconds | Logged training seconds | Mesh seconds | Steps 20k–30k it/s | Final loss |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 30,000 | 773.405 | 712.396 | 55.299 | 25.948 | 0.067116 |
| 6 | 30,000 | 775.865 | 714.943 | 54.738 | 25.955 | 0.067040 |
| 7 | 30,000 | 695.321 | 635.595 | 53.444 | 29.636 | 0.067039 |
| 8 | 30,000 | 799.038 | 741.611 | 51.754 | 25.625 | 9.906942 |
| 9 | 30,000 | 686.937 | 628.080 | 53.167 | 29.783 | 0.067101 |
| 10 | 30,000 | 665.873 | 605.598 | 54.254 | 30.441 | 0.067016 |
| 11 | 30,000 | 677.199 | 616.942 | 54.389 | 29.294 | 0.067191 |
| 12 | 30,000 | 654.495 | 591.837 | 57.185 | 30.695 | 0.067023 |

For runs 10–12, the saved settings are horse_960, fresh training, seed 42,
batch/microbatch 2048, query 131072, 30k steps, occupancy resolution 128,
checkpoint and mesh frequencies 10k, mesh resolution 1024, diagnostics off.
Configurations differ by output paths and newly introduced switches. Configuration
files do not establish clock, thermal, driver or concurrent-process equality.

### Interpretation

- **Run 5 → 6:** Feature-only caching helped early throughput but did not establish
  an overall win; command time increased by approximately 0.32%.
- **Run 6 → 7:** Fused feature replay plus register tiling produced a substantial
  combined gain: approximately 10.4% lower wall time and 14.2% higher late rate.
  Individual contributions were not isolated.
- **Run 8:** A correctness failure, not a valid equivalent-workload benchmark.
  Final total loss reached about 9.91; late samples collapsed from about 70,793
  to 2,124 per iteration. Divergence began at step 2. Hybrid caching and custom
  normalization/packing changed together, so the cause was not isolated.
- **Run 9:** Healthy scalar losses returned after restoring reference
  normalization and disabling hybrid caching. This recovery does not prove
  which run-8 change was responsible.
- **Run 10:** Another measurable overall gain, with healthy scalar losses.
- **Run 11:** GPU-driven occupancy/replay changes improved early throughput but
  increased recorded training time by 1.87% and full wall time by about 1.70%.
  No net performance win was demonstrated for the combined update.
- **Run 12:** Wider parameter tiles recovered the regression and beat run 10.
  Versus run 11: training time fell 4.07%, command wall time fell 3.35%, and
  22.70 seconds were saved despite mesh extraction taking 2.80 seconds longer.
  Versus run 10, command wall time fell approximately 1.71%.

### Detailed recent throughput

Rates are window step count divided by summed recorded duration.

| Window | Run 10 | Run 11 | Run 12 |
|---|---:|---:|---:|
| Steps 1–2,000 | 39.36 | 39.97 | 41.63 |
| Steps 2,001–10,000 | 113.76 | 118.17 | 120.12 |
| Steps 10,001–20,000 | 64.12 | 63.36 | 66.04 |
| Steps 20,001–30,000 | 30.44 | 29.29 | 30.70 |
| Steps 24,001–30,000 | 26.91 | 25.80 | 27.02 |

Late mean samples/iteration were 70,505, 71,482 and 71,122 respectively.
Late sample throughput increased from approximately 2.094 million/s in run 11
to 2.183 million/s in run 12 (+4.26%). Sample count decreased only 0.50%.
That supports a computational improvement, but does not isolate one kernel from
all differences in execution. The last 10k steps still consume about 55% of run
12's recorded training time.

## 6. Current pipeline and optimization mechanisms

### Field and gradients

The network combines XYZ with 14 two-channel hash levels (31 inputs total),
a 64-neuron softplus hidden layer and scalar SDF output. Levels activate
progressively. Packed weights, shared staging and native/CAS gradient scatter
avoid many separate Burn tensor operations.

The backward kernel evaluates activation derivatives, feature adjoints and
hash scatter. First-layer parameter gradients still use a separate tiled pass
and partial reduction. Weight-normalization derivatives and some optimizer work
remain separate. There is no single fully fused end-to-end training kernel.

The current default parameter tile covers 32 neurons per workgroup. Two groups
cover all 64 neurons per 256-point partition, rather than four groups of 16.
Features are staged half as many times, derivative loads cover contiguous 32-
channel segments, and each thread uses eight accumulators. Global buffer sizes
are unchanged. Shared storage rises from 6 to 8 KiB/group; register pressure and
residency remain hardware-dependent tradeoffs.

### Activation cache and replay

The cache is budgeted from the former three full chunks of 31 features plus
64 preactivations. Small workloads can retain full activations. Larger workloads
retain more feature-only chunks within the same budget. Fused feature replay
reconstructs hidden preactivations inside backward instead of writing another
large activation buffer. At query 131072, the established budget can retain nine
full feature chunks in feature-only mode.

For uncached chunks, fused hash replay reconstructs features and hidden values
inside backward. It returns features as an explicit dependency for parameter
gradients, avoiding a separate full-forward replay and discarded SDF. A full
131072-point miss avoids a 32 MiB preactivation buffer plus associated traffic.
This helps only actual cache misses. It is not a universal per-iteration saving.
Native and CAS variants share the forward interpolation helper.

### Compositing and scratch

Each pixel ray has a barrier-free serial compositor; the nine rays of a patch
execute in parallel. Component-major scratch improves neighboring-lane access.
Compact records use 63 rather than 90 floats/sample, a 30% capacity reduction.
Consumed slots are reused for derivatives; endpoint terms are precombined before
gather. This requires careful lifetime tracking and consistent indexing in both
producer and consumer kernels.

### Optimizer and inactive work

Hash Adam updates active levels with independent bias-correction counters and
clears gradients in the same pass. Its persistent gradient buffer is reused.
The zero-state fast path skips an entry only when gradient, first moment and
second moment are all exactly zero. A zero gradient with nonzero moments still
requires decay and parameter motion; ordinary sparse skipping would change Adam.

Active-column parameter gradients and selective weight staging omit work for
unactivated hash levels. The forward staging reduction is largest early: 449
rather than 2113 floats/group with one active level, and no reduction at all 14.
Direct global adjoint offsets avoid per-query slicing on eligible paths.

### Occupancy and remaining readbacks

GPU occupancy uses the same stratified selection/jitter and EMA policy as its
preceding GPU implementation. After the 256-step warmup, its selected count can
stay on-device. At most 4096 persistent field workgroups reuse weights while
consuming compacted points and apply EMA directly, without an SDF buffer.
XYZ plus ID capacity is bounded to 32 MiB at occupancy resolution 128.
Larger grids, explicit chunked evaluation and reference paths retain the old
readback route. This can increase scratch memory compared with exact-sized
selection; total VRAM is not guaranteed unchanged.

Marching and visibility still require host-visible counts in the ordinary
pipeline. GPU-resident tables do not imply that all CPU/GPU synchronization has
been eliminated. Loss statistics piggyback on a later required readback.

## 7. Defaults and comparison controls

All switches below are training arguments unless explicitly described otherwise.
Read `src/train.rs` for authoritative defaults and interactions.

| Default behavior | Comparison / alternative |
|---|---|
| Fused field forward/backward | `--unfused-field` |
| GPU compositor / visibility / march / occupancy | `--cpu-composite`, `--cpu-visibility`, `--cpu-march`, `--cpu-occupancy` |
| GPU patch gather and compact sample tables | `--cpu-patch-upload`, `--host-sample-compaction` |
| Endpoint reuse | `--no-endpoint-reuse` |
| Level-oriented hash layout / corner parallelism | `--legacy-hash-layout`, `--legacy-corner-loop` |
| Compact output gradient partials | `--legacy-output-gradients` |
| Tiled register parameter gradients | `--untiled-gradients`, `--legacy-gradient-tiles`, `--legacy-parameter-kernel` |
| 32-neuron parameter tiles | `--legacy-parameter-rows` restores 16 |
| Local hash aggregation, up to 14 levels | `--no-hash-aggregation`; adjust `--hash-aggregation-levels` |
| Component-major compact compositor scratch | `--legacy-composite-scratch`, `--legacy-composite-cache` |
| Per-level Adam dispatch / exact zero-state skip | `--legacy-adam-dispatch`, `--dense-hash-adam` |
| Active parameter columns / selective backward weights | `--dense-parameter-gradients`, `--dense-backward-weight-loads` |
| Direct adjoint offsets | `--legacy-adjoint-slicing` |
| Budgeted feature cache / fused feature replay | `--legacy-activation-cache`, `--legacy-feature-replay` |
| Fused uncached hash replay | `--legacy-hash-replay` |
| Selective forward weights | `--dense-forward-weight-loads` |
| GPU-driven occupancy count/evaluation/EMA | `--legacy-occupancy-readback` |
| Copy-only normalized MLP packing | `--legacy-mlp-packing` |
| Deferred loss readback | `--sync-loss` forces synchronization |

Not enabled by default: `--hybrid-activation-cache`,
`--hash-aggregation-cross-corners`, `--recompute-parameter-gradients`,
`--chunked-eval`, `--profile-stages`, `--diagnostics` and pipeline lookahead.
Forward points/group defaults to 4; 8 and 16 are alternatives. The user reported
8-point groups plus cross-corner aggregation and gradient recomputation slower
in an earlier combination; their individual contributions were not isolated.
Pipeline lookahead previously slowed training and defaults to zero.

Current CLI defaults are batch 2048, microbatch 2048 and query 65536. Recent
comparison runs explicitly use query 131072. Do not remove these arguments and
assume the workload is unchanged. Batch changes the optimization problem;
microbatch/query change partitioning and memory requirements. Increasing query
previously provided a small gain, not a guaranteed monotonic improvement.

## 8. Correctness findings and failure modes

Historical fixture checks included imported Python checkpoints at 5k, 10k and
30k, field gradients, render adjoints, independent Adam updates and mesh extraction
from the same checkpoint. Examples recorded in `VALIDATION.md` include SDF error
5.96e-7, small field-gradient discrepancies and resolution-96 mesh nearest-vertex
agreement. These are fixture-specific historical results.

Important lessons:

1. Correct forward values do not guarantee correct weight-normalization gradients.
2. Near-cancelling bias gradients can be amplified by Adam's small denominator.
3. Changed reduction grouping can change training trajectories without being a
   bug; large loss/sample collapse is not ordinary harmless rounding.
4. `Tensor::empty` is safe only if every consumed element is initialized. Tail
   rows, inactive columns and dummy outputs require explicit ownership rules.
5. Native float atomics and CAS must implement equivalent mathematical updates;
   neither guarantees deterministic sum ordering.
6. Workgroup barriers must be uniform, shared slots must not be overwritten too
   early, and compiler dominance errors are distinct from Rust compile errors.
7. Copy-on-write handling protects outstanding checkpoint/tensor aliases during
   in-place updates. Fusion IR must declare regenerated buffers as dependencies.
8. Do not drop samples merely because alpha is zero: eikonal and neighboring
   finite-difference losses can still depend on them.
9. The normal-map validation path in the Python branch was reported broken;
   reference runs should set `val_normal_freq` above `end_iter` (e.g. 30001).
10. Run-8 damaged checkpoints are not repaired by changing the executable.
    Resume a known-good checkpoint or start fresh.

Build issues encountered included invalid CubeCL scalar/tensor argument use,
expanded-value type mismatches, borrow-checker conflicts, unavailable client
methods and an invalid `self` reference in an associated load function. Source
formatting is not evidence of successful Rust compilation or shader validation.

## 9. Mesh quality and extraction

Mesh extraction time is a separate component of end-to-end reconstruction.
Resolution, export frequency, coordinate transforms, threshold and connected-
component filtering affect comparisons. Exporting at 1024 is not comparable to
an earlier 512 or 96 export in either cost or apparent detail.

A quality regression was reported during optimization, but the conversation did
not establish mesh extraction as its cause. To distinguish extraction from
training differences, extract the same known-good checkpoint with old and new
extractors using identical settings. Then compare different trained checkpoints
with one fixed extractor. Check topology, components, surface distance and
normals in addition to screenshots and scalar loss. Latest healthy losses alone
do not settle geometric parity.

## 10. Integrated GPUs, bandwidth and power

The user reported Radeon 840M training initially about 2x slower than RTX 3060,
then falling to roughly 1.3 it/s near completion. The quoted total was 1 hour
and 46 seconds. This is a historical observation with different software/settings
from the newest 10–11 minute RTX runs, not a current controlled adapter comparison.

Narrower/shared memory bandwidth is a plausible contributor, but the charts do
not prove it is the sole cause. More active hash levels, smaller marching steps,
more surviving samples, scatter contention, cache capacity, residency, shared
memory/register pressure, thermal limits and shared CPU/GPU power can all matter.

Low GPU power draw is not a direct measurement of spare usable throughput.
Latency-bound memory access, atomics, barriers, CPU waits and specialized
arithmetic can all produce low power without an easy arithmetic speedup.
Saturating TDP is not the optimization objective; shorter validated reconstruction
with acceptable memory and quality is.

The application targets WGPU-capable graphics APIs, with Vulkan, DX12 and Metal
selection exposed. API support alone does not guarantee that every device meets
all resource limits or driver requirements. The i3-1315U Intel iGPU is an
intended target; this host did not provide that adapter for validation. Brush's
ability to run on it supports the portability approach, not proof that every
SuperNormal kernel will run correctly or efficiently there. A Windows binary
is not a universal OS/architecture binary; OS graphics drivers remain required.

## 11. Profiling is possible; optimization is not blind trial and error

Existing CSVs measure elapsed time, losses, sample count and active levels.
`--profile-stages` and `--diagnostics` provide more detailed timing/work evidence,
but synchronization changes the execution being measured. Normal asynchronous
step timings can charge preceding GPU work to the next host stage.

Hardware-specific GPU profilers and graphics captures can, where supported,
provide dispatch timelines, generated code, memory traffic, cache statistics,
occupancy and stall evidence. Counter availability depends on adapter, driver,
API and tool. No such capture was performed for the recent commits. Do not label
source-level load counts as measured DRAM bytes or assert a cache bottleneck
without supporting measurements.

Late slowdown is substantially workload-related: sample throughput was relatively
stable in several windows even as it/s fell. More samples and levels create more
work. That does not rule out cache or contention effects. Samples/s is also only
a partial normalization: occupancy, visibility, levels and fixed costs differ.

## 12. Benchmark and validation protocol

- Keep dataset, seed, batch, schedule, API, adapter, power settings, query sizes,
  mesh frequency/resolution and background workload fixed.
- Save command/config, binary revision, metrics and successful wall-time summary.
  A power-interrupted run without its final summary is not a completed benchmark.
- Use steps divided by summed seconds. Arithmetic means of instantaneous it/s
  exaggerate throughput when iteration durations vary.
- Report full command wall time, logged training time, mesh time and checkpoints
  separately. Never silently compare export-inclusive and export-free durations.
- Repeat comparisons; one pair of runs provides evidence, not statistical certainty.
- Use exact native checkpoint resumes for short early/middle/late comparisons.
  Python model-only imports can rebuild occupancy/RNG and are not exact resumes.
- `--steps` is additional updates; `--end-iter` retains the global schedule.
  Preserve batch/schedule when exact-resume checks require them.
- Compare gradients/updates on fixed fixtures before spending a full 30k run on
  a major numerical change. Inspect losses, sample counts and meshes together.

Example from the crate directory (run by the user):

```powershell
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --out experiments\comparison --steps 30000 --end-iter 30000 --batch 2048 --microbatch 2048 --query 131072 --save-freq 10000 --mesh-freq 10000 --mesh-resolution 1024
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model experiments\known_good\checkpoint_020000 --out experiments\resume_comparison --steps 1000 --end-iter 30000 --batch 2048 --microbatch 2048 --query 131072
```

Global adapter/API options precede the subcommand, for example
`.\target\release\supernormal.exe --api dx12 --integrated train ...`.
The standalone mesh subcommand can inspect one fixed checkpoint independently.

## 13. Remaining opportunities, in priority order

1. **Profile late training:** it consumes about 55% of training time in the final
   third. Separate parameter gradients, hash scatter, field evaluation, Adam,
   occupancy and required synchronization before another large rewrite.
2. **Isolate the newest changes:** compare the same late native checkpoint with
   and without `--legacy-parameter-rows`; independently assess occupancy and hash
   replay. The combined run-11 update has no demonstrated net win.
3. **Reduce remaining count synchronization:** bounded capacities/device counts
   could remove marching/visibility host waits, but require overflow handling,
   tail initialization and a deliberate memory budget.
4. **Further parameter-gradient/backward integration:** reduce intermediate
   derivative traffic without exploding partial buffers or register pressure.
   A recomputation variant exists, but was not consistently faster.
5. **Hash scatter contention/locality:** quantify collision rates by level and
   compare aggregation scopes; more aggregation can cost more than it saves.
6. **Investigate Adam/normalization/small dispatch overhead:** preserve exact
   moment decay and normalization chain rules. Run 8 makes numerical changes
   particularly deserving of fixed-fixture validation.
7. **Mesh extraction:** now roughly 55–57 seconds of recent total wall time;
   optimize it independently if end-to-end time is the goal.
8. **Integrated-adapter validation:** test actual DX12/Vulkan device limits,
   memory, driver behavior and early/late performance rather than infer all
   behavior from RTX results.

Mixed precision, approximate math, aggressive pruning, batch/schedule changes
and cache-policy changes are future experiments with quality implications, not
already-validated free speedups. No claim is made that the implementation has
reached a hardware limit or exhausted all useful optimizations.

## 14. George Kopanas article: what was and was not integrated

Reference: <https://grgkopanas.github.io/3dgs_cuda_blog/> and its six stage diffs.
`KOPANAS_AUDIT.md` records the detailed inspection and later follow-ups.

Adapted principles include local aggregation of equal-address gradients,
shared weight staging, compact scratch/partials, memory-lifetime reuse,
zero-adjoint work avoidance, reduced redundant stores and better lane ownership.
The current aggregation is a portable local reduction, not a literal CUDA
half-warp folding implementation.

Gaussian ellipse binning, spherical-harmonic streaming, splat contributor masks,
Gaussian opacity/conic formulas and Gaussian-specific thresholds are not direct
SuperNormal replacements. CUDA PTX approximations and fixed warp assumptions
are not portable defaults. Neither “all article tricks are integrated” nor
“nothing more can be optimized” is supported by the audit.

## 15. Supporting documents and source map

- [VALIDATION.md](VALIDATION.md): early numerical fixtures and continuation tests.
- [KOPANAS_AUDIT.md](KOPANAS_AUDIT.md): article comparison and implementation follow-ups.
- [RUN6_ANALYSIS.md](RUN6_ANALYSIS.md), [RUN7_ANALYSIS.md](RUN7_ANALYSIS.md),
  [RUN8_ANALYSIS.md](RUN8_ANALYSIS.md): measured history and failed experiment.
- [TRAINING_TRAFFIC_UPGRADES.md](TRAINING_TRAFFIC_UPGRADES.md): Adam, active columns,
  direct adjoints and backward staging.
- [COMPOSITOR_MEMORY_OPTIMIZATIONS.md](COMPOSITOR_MEMORY_OPTIMIZATIONS.md): scratch accounting.
- [FUSED_REPLAY_OCCUPANCY.md](FUSED_REPLAY_OCCUPANCY.md): bounded occupancy and replay.
- [RUN11_PARAMETER_TILING.md](RUN11_PARAMETER_TILING.md): run-11 comparison and wider tiles.
- `src/field.rs`: field/cache orchestration, normalized weights, gradients and optimizer.
- `src/kernels.rs`: field, compositor, visibility and hash/Adam custom kernels.
- `src/sampling_gpu.rs`: sampling, occupancy, geometry and parameter-gradient kernels.
- `src/render.rs`: device pipeline and CPU renderer reference.
- `src/train.rs`: scheduling, defaults, checkpointing, metrics and export timings.
- `src/mesh.rs`: extraction; `src/diagnostics.rs`: optional instrumentation.

Historical `OPTIMIZATIONS.md` and older README paragraphs contain claims that
were later revised. In particular, shader-source mechanisms do not prove
hardware bottlenecks were resolved, every documented prototype survived, or
all adapters were validated. Use this history with current source and saved
measurements rather than treating any early snapshot as a current guarantee.

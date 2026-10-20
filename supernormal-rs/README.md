# SuperNormal reconstruction in Rust + Burn

Native reconstruction for the `pure_pytorch_version` branch's `own_objects.conf` configuration. Python, CUDA, tiny-cuda-nn and nerfacc are not runtime dependencies. Development comparison scripts in `tools/` use the existing Python environment.

Implemented: EXR/PNG/NPZ ingestion, camera decomposition and patch rays, progressive hash encoding, weight-normalized SDF MLP, occupancy updates, sphere marching, visibility filtering, directional finite differences, packed normal/mask integration, all three training losses, analytical renderer and MLP backward, WGPU hash-grid backward, Adam, native checkpoint/resume, and streaming marching cubes with largest-component filtering.

The neural field, hash-grid interpolation and gradient scatter, and optimizer use WGPU. Ray sampling and the analytical compositor run on CPU, with point uploads and SDF readbacks between stages. Field inference for visibility and occupancy also uses the packed WGPU hash kernel. The earlier native training run was cancelled; short continuations from the 5k, 10k, and 30k Python checkpoints are measured in `VALIDATION.md`. Sustained speed and integrated-GPU performance remain to be validated.

## Build and diagnose

```powershell
cargo build --release --locked
.\target\release\supernormal.exe doctor
.\target\release\supernormal.exe --api dx12 doctor
.\target\release\supernormal.exe --api vulkan doctor
.\target\release\supernormal.exe --integrated doctor
```

Windows MSVC builds link the C runtime statically via `.cargo/config.toml`. The executable still uses the operating system and installed graphics driver. Each OS/architecture requires its own build. Intel UHD in the i3-1315U is an intended WGPU target; successful execution on that specific laptop remains to be verified. `--integrated` selects the first integrated adapter. `--cpu` uses Burn Flex for debugging.

## Train

Run from this directory. The original dataset layout is consumed directly; there is no preprocessing step.

```powershell
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --out runs\horse --steps 100 --save-freq 100
```

The defaults preserve the reference's 2,048 patches, 3x3 patch size, 30,000-update schedule, learning-rate warmup, occupancy resolution 128 and progressive bandwidth. `--steps` limits additional updates without compressing that schedule. `--microbatch` and `--query` bound working memory while accumulating the complete batch gradient before one Adam update. Adjacent samples share exactly equal endpoints. Normal maps stay in host FP16 storage; GPU arithmetic uses FP32.

Normal training submits backward/Adam work without an iteration-end GPU wait, overlapping it with the next iteration's CPU preparation. Visibility reads the updated variance together with its SDF values; no stale optimizer values are used. Query chunks share one SDF readback, and packed Adam updates active embedding entries in a single kernel with per-level bias correction. The experimental point/level hash layout was reverted following a runtime shader compiler dominance error; the previously working point-wise kernels are used again.

Interactive terminals show one updating progress line with percentage, step count, smoothed training **it/s**, elapsed time, ETA, and loss, refreshed at most four times per second. Checkpoint/mesh messages appear above it. Redirected output uses plain progress lines every 10 steps (`--log-every` controls this interval and CSV flushing); CSV still records every update. Speed/ETA use a rolling training-time window and exclude checkpoint/mesh export time. Use `--profile-stages` for detailed synchronized diagnostics, which disables iteration overlap. Normal per-step timings can include GPU work submitted by the preceding step; compare throughput over multiple iterations. The shader workaround and progress display have not been compiled or tested.

Resume into a fresh output directory:

```powershell
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\horse\checkpoint_000100 --out runs\horse_continued --steps 100
```

Native checkpoints contain weights, Adam moments and counters, occupancy and RNG state. Model-only imports lack some of that state and print a warning. The native schedule continues globally on resume; the Python runner's local-loop schedule reset is intentionally not reproduced. Changing batch size or schedule on exact resume is rejected.

Mesh export uses normalized coordinates by default. Add `--data` to apply the dataset's world scale and offset.

```powershell
.\target\release\supernormal.exe mesh --model runs\horse\checkpoint_000100 --out runs\horse\mesh.ply --resolution 256
```

`--save-freq` controls checkpoints; `--mesh-freq` and `--mesh-resolution` enable periodic meshes. Normal-image validation is currently omitted, including the broken Python validation path.

## Fused field forward

The default training path now uses a cooperative CubeCL hash-grid + MLP forward
kernel for occupancy, visibility SDF queries, and cached training evaluation.
Four points share a 256-thread workgroup: hash levels evaluate in parallel,
features stay in shared memory, and 64 lanes evaluate the hidden neurons before
a tree reduction produces the SDF. Inactive hash features are skipped in the
first layer. Packed input-major MLP weights are cached until the optimizer updates
them; the kernel uses seven storage buffers and 1.5 KiB of workgroup scratch.
Training writes the same features and preactivations required by the existing
analytical backward; inference does not write those large caches. The GPU compositor and fused backward described below are now the training defaults. This is FP32, without CUDA or subgroup-size
requirements, and is not a tensor-core implementation of tiny-cuda-nn.

Use `--unfused-field` to compare with the previous encoder + Burn matmul path.
No build or runtime validation has been performed for this kernel. Floating-point
summation order differs, and speedup is not established. Use a fresh output
folder for each comparison; no checkpoint format or training schedule changed.

## Reproduce comparisons

Training now overlaps the next group's field forward pass with the current
group's CPU loss/compositor. Lookahead is disabled by default (`--pipeline-patches 0`) following a measured
regression. When enabled, the group size is capped by `--microbatch`. This is lookahead within one optimizer batch,
not stale-weight or asynchronous optimization: all groups use the same weights,
global loss denominators, and one Adam update. Only one future group is prepared
at a time. Smaller groups reduce activation size but add dispatches/readbacks;
256, 512, and 1024 are useful comparison points. `--pipeline-patches 0` restores
sequential scheduling with the requested microbatch size.

This source change has not been compiled or benchmarked. Grouping changes
floating-point reduction order. Compare throughput and losses from the same
checkpoint in fresh output directories. Stage timings overlap and cannot be
interpreted as separate GPU kernel timings; evaluation timing also includes
lookahead preparation/submission. Low board power alone is not proof of CPU
starvation: memory-bound kernels can also consume relatively little power.

The local `experiments/` directory contains exported fixtures. These are development artifacts, not required for normal use.

```powershell
.\target\release\supernormal.exe verify experiments\field_30k
.\target\release\supernormal.exe verify-render experiments\render --data ..\data\own_objects_normals\horse_960
.\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k
```

`export_reference.py` exports the pure-PyTorch field. `export_render_reference.py` supplies independent differentiable compositor fixtures. `validate_native.py` additionally calls the repository's actual renderer with fixed samples and compares native meshes with Python marching cubes. Fixing samples isolates numerical correctness from different RNG implementations.

Current scope is the supplied DFD/L2, 14-level hash-grid configuration. Arbitrary HOCON settings, FD/AD training, preprocessing and original tiny-cuda-nn checkpoint layouts are not supported. An imported Python checkpoint proving inference parity does not by itself prove native training convergence or a speedup. See `VALIDATION.md` for measured results and outstanding validation.

The original reconstruction implementation is MIT licensed; see `LICENSE`. The marching-cubes table comes from scikit-image; see `THIRD_PARTY_NOTICES.md`.


## GPU compositor and fused backward (unvalidated)

Training now keeps field SDFs and unique-point adjoints on the GPU. One invocation owns a pixel ray and performs ordered volume compositing,
normal/mask/eikonal losses and reverse transmittance differentiation. A second
GPU gather combines spatial stencil derivatives and reused endpoint adjoints.
The nine pixels now run independently, without float atomics in this stage. It relies
on `sample_and_dedup_points`, which never shares indices between patches. Geometry
preparation, marching and visibility filtering still involve the CPU. Five loss
and scalar-gradient totals are read back per optimizer batch; large SDF/adjoint
arrays no longer cross the CPU boundary in the default training path.

The fused backward computes softplus derivatives, output-weight gradient
contributions, feature gradients and hash-table atomic accumulation in one
workgroup kernel. It removes the intermediate feature-gradient buffer and
separate hash-backward launch. First-layer weight-gradient matmul, parameter
reductions, weight-normalization derivatives and Adam remain separate operations.
This is not a fully fused optimizer or a complete nerfacc replacement.

Use `--cpu-composite` for the previous CPU loss implementation and
`--unfused-field` for separate field forward/backward operations. Combine both
flags to compare with the earlier implementation. No builds, numerical checks,
or speed measurements were run for these changes. GPU loss accumulation uses
FP32 rather than the CPU's mixed FP32/FP64 accumulation, and derivative summation
order differs. Check short resumed runs and loss/mesh parity before a long run.


## Visibility and parallel compositor changes (source only)

Visibility transmittance scanning and early termination now run on the GPU.
Only one retained-prefix count per patch plus variance is read back. The CPU
still compacts the existing ordered sample list and prepares field points.
`--cpu-visibility` selects the previous implementation for comparison.

The compositor uses one invocation per pixel, with scalar ray state and only
three-element normal arrays, followed by an atomics-free stencil gather.
Its scratch storage increases from 45 to 90 floats per sample. Reverse endpoint
mapping requires each unique point to have at most two occurrences, as guaranteed
by `sample_and_dedup_points`; this is explicitly checked on the host.

Fused MLP forward and backward cooperatively load packed weights into shared
memory once per workgroup, reusing them across four points. The fused hash
backward selects native FP32 atomic addition when CubeCL advertises that exact
capability, and otherwise uses the portable CAS implementation. Native atomic
accumulation may change rounding and denormal behavior.

These changes have not been built or tested. GPU marching, GPU occupancy-grid
maintenance, fully device-side compaction, and tiled matrix/parameter-gradient
execution are still outstanding; this is not completion of all performance work.


## GPU sampling, occupancy and parameter reductions (unvalidated)

The default training path now marches rays on the GPU using the existing voxel
boundary skipping rule. Host sphere intersections and jitter preserve the RNG
consumption order. Each ray writes into bounded storage, followed by a device
prefix sum and stable device compaction. One total-count scalar is read for
allocation, then the compact samples are read for the existing CPU geometry
preparation. Thus compaction is GPU-side, but the complete sampling-to-field
pipeline is not yet device-resident. Initial temporary sample storage can be
larger than the CPU marcher's packed output.

Occupancy EMA updates and thresholding run on the GPU. Ordered per-cell lists
preserve repeated decay when random cell IDs repeat. Occupied-cell selection and
jitter remain on the host; flags are mirrored for that selection policy. Values
stay on the device until checkpoint serialization, which explicitly refreshes
the host copy. The mean threshold now uses an FP32 device reduction rather than
a CPU FP64 sum; cells very near the threshold can differ.

First-layer weight and bias gradients use 8x8 parameter tiles with 32 samples
staged in shared memory, accumulating 256-sample partials without atomics. Burn
reduces the partials. Output-layer reductions, weight-normalization derivatives
and Adam remain separate operations.

Comparison switches (independent): `--cpu-march`, `--cpu-occupancy`, and
`--untiled-gradients`. Combine all three to disable this set of changes. Existing
`--cpu-visibility`, `--cpu-composite`, and `--unfused-field` switches remain.
Checkpoint format and training schedules are preserved. No compilation, runtime
parity checks or timing measurements were performed for this set. Early-training
speedup is a hypothesis, not a measured result; compare the same checkpoints and
watch memory use before committing to a full reconstruction.

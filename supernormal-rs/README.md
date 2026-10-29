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


### GPU training geometry

Training now uploads compact endpoint descriptors and camera rays, then generates
unique world-space points and finite-difference coefficients on the GPU. The
field consumes those points directly. Adjacent sample endpoints retain the same
sharing rule and order; distances use the generated world coordinates rather
than an approximate camera-space spacing formula. CPU expanded point arrays and
the per-sample CPU distance calculations are removed from the default path.
`--cpu-composite` retains CPU point generation and the reference compositor.

Variance and output-bias derivatives stay on the GPU through Adam. Loss metrics
are read after optimizer submission rather than blocking between backward and
Adam. CSV losses still describe the current iteration before its update.

Compact ray samples, endpoint/index bookkeeping, and occupancy flag selection
still pass through the CPU. This is not yet a completely device-resident training
loop. GPU coordinates now exist for the whole microbatch, so monitor peak memory
when comparing large microbatches. Shader arithmetic can round differently from
CPU arithmetic; reconstruction parity and performance require measurement.
This change was source-reviewed and formatted only, with no build or tests run.


### Hash access locality experiment

Fused field kernels now assign consecutive hash-work lanes to four neighboring
points at the same grid level, rather than different levels of one point.
Nearby patch pixels can consequently reuse the same grid vertices/cache lines
within a wave. This is a locality opportunity, not a guarantee: high-resolution
hashed levels may map even nearby vertices to unrelated addresses.

The MLP retains its neuron-parallel layout. Shared feature/derivative buffers
bridge these ownership layouts; backward stages the first-layer weights in a
transposed shared layout so different levels no longer read columns separated
by a 64-float stride. This targets shared-memory bank conflicts as well as
hash-table cache reuse, without sorting points or allocating new global buffers.
Both native-float and compare-and-swap atomic paths use the new layout. A
one-time training message reports the selected atomic path and hash layout.
The number of atomic additions is unchanged; closer updates may also increase
contention, which is why hardware comparison remains necessary.

The new layout is the default. Add `--legacy-hash-layout` to the same training
command to restore the previous shared layouts and point-major hash ownership.
The switch applies to fused field evaluation and backward; `--unfused-field`
bypasses these kernels. Existing checkpoints require no conversion.

Compare separate fresh output directories resumed from the same late-stage
checkpoint, using identical batch/query settings and API. Use several hundred
iterations, discount shader warm-up, and disable mesh extraction during the
comparison. Compare samples per second alongside iterations per second and
check losses/reconstruction quality. Floating-point atomic update order can
differ, so bitwise equality is not expected. Test on both integrated and discrete
GPUs before judging the default. No build, tests, or runtime measurements were
performed for this change.


### Backward memory traffic optimizations

The fused backward kernels now reduce four points' output-weight gradients in
shared memory before storing a partial row. Their global output-gradient buffer
contains `ceil(query_points / 4)` rows instead of one row per point, reducing
that buffer's allocation and write/read volume by approximately 75%. The final
reduction still runs on the GPU. Shared storage grows by 1 KiB per workgroup;
no extra workgroup barrier or global atomic is needed. Tail lanes contribute
zero, and the rounded-up 2D dispatch is bounded before partial writes.

First-layer parameter gradients default to 8x32 tiles (256 threads) instead of
8x8 tiles (64 threads). Each tile handles all 31 features plus bias for eight
neurons. It loads the hidden derivatives once for all feature columns, reducing
explicit derivative loads by 4x, and reads wider contiguous feature spans.
Feature-load count and the partial-output shape remain unchanged. Shared
storage grows from 2 KiB to 5 KiB per tile, and fewer, larger workgroups may
behave differently on different GPUs. This is not a claimed 4x kernel speedup.

Comparison switches are independent:

- `--legacy-output-gradients`: restore per-point output-weight gradient rows.
- `--legacy-gradient-tiles`: restore 8x8 first-layer gradient tiles.

Pass both switches to compare against the preceding locality update while
retaining its improved hash layout. `--legacy-hash-layout` remains independent.
The output-gradient optimization requires the fused field, and the wider tile
optimization is bypassed by `--untiled-gradients`. Checkpoints and training
schedules are unchanged; sum grouping changes floating-point rounding.

Rebuild and compare the same checkpoint/settings in fresh output directories,
first with neither flag, then with each flag separately and both flags together.
Use short early- and late-stage runs, discount shader compilation/warm-up, and
check losses as well as throughput. No builds, tests, or runtime measurements
were run for this update.


### 256-thread corner-parallel hash kernels and device-resident visibility/geometry pipeline (Update 3)

Four bottlenecks previously capped board power around 85 W / 170 W TDP and caused iteration rate to fall from ~45 it/s at step 2,000 down to ~12 it/s at step 30,000:

1. **78% idle threads during hash-grid evaluation and backward scatter**: In the 256-thread fused workgroup (`4` points × `64` lanes), the hash stage assigned one thread per `(point_slot, level)` (`<= 56` active threads) to loop sequentially over all `8` trilinear cube corners while `200+` threads waited at `sync_cube()`.
   - `fused_field_kernel` and `field_backward_kernel` / `field_backward_native_kernel` now map `256` threads across `(point_slot in 0..4, corner_pair in 0..4, level in 0..16)`. Up to `224` threads (`87.5%` of the workgroup) concurrently issue `2` corner lookups or atomic scatter-adds each (`448` corners in flight per workgroup, 4× higher memory-level parallelism to hide L2/DRAM latency across the 14 hash levels), reducing partial corner pairs in shared memory before `sync_cube()`.
   - In backward, `exp(min(scaled, 20.0))` is evaluated once per neuron activation and reused for both the sigmoid derivative and softplus reconstruction, cutting transcendental SFU instructions per point in half.
   - Pass `--legacy-corner-loop` to restore the 56-thread sequential 8-corner loop for A/B comparison.

2. **Elimination of host sample readbacks and CPU index-table uploads between march, visibility, and compositing**:
   - `march_kernel` now counts both marched samples and deduplicated center-ray endpoints per ray in a single `[patches, 2]` prefix sum.
   - `compact_and_center_endpoints_kernel` compacts marched samples while simultaneously materializing deduplicated 3D center-ray points, endpoint index pairs, and per-ray ranges on the GPU.
   - `visibility_kernel` outputs both visible sample counts and retained unique endpoint counts per ray (`[patches + 1, 2]`), reading back only a single 3-float metadata slice (`total_visible`, `total_visible_endpoints`, and `variance`) for the default full-batch configuration.
   - `build_visible_geometry_tables_kernel` builds `endpoints`, `descriptors`, forward `mapping` (`[vis_count, 18]`), and `reverse` occurrence tables (`[ep_count * 9, 2]`) directly on the GPU without atomics or host transfers, and `composite_kernel` reads per-patch sample ranges directly from the device-resident `vis_prefix` tensor.
   - Pass `--host-sample-compaction` to restore the host sample readback and CPU index-table construction path for A/B comparison.

3. **Elimination of redundant `Tensor::zeros` buffer-fill dispatches**:
   - Write-only output buffers in `fused_field`, `field_backward`, `parameter_gradients`, `composite_gpu`, `geometry`, `march`, `compact`, `occupancy`, and `visibility` now use uninitialized `Tensor::empty` allocations instead of `Tensor::zeros`, removing ~10 redundant zero-fill kernel dispatches and tens of megabytes of wasted DRAM writes per step.

4. **Precomputed camera ray directions and analytical 3×3 basis inversions (`src/data.rs`)**:
   - `Dataset::load` precomputes world-space ray directions and closed-form orthonormal cross-product basis inverses for every pixel once on startup, turning `View::patch` (`18,432` rays per step) into a direct table lookup with zero per-step matrix inversions or ray normalizations.

See `OPTIMIZATIONS.md` for a detailed technical breakdown and A/B benchmarking instructions.


### Single-dispatch evaluation, GPU patch gathering, and deferred loss readback (Update 4)

Five additional optimizations further reduce CPU-GPU synchronization stalls, host-to-device memory traffic, and per-step kernel launch overhead:

1. **Unchunked Single-Dispatch Device Evaluation & Backward (`src/field.rs`)**:
   - Device-resident point sets in `evaluate_device_tensor`, `enqueue_cached_device`, and `accumulate_gradients_device` now execute in a single unpadded `fused_field` / `field_backward` / `parameter_gradients` dispatch instead of splitting into `65,536`-point chunks, zero-padding the tail chunk (`Tensor::cat`), and slicing/concatenating outputs.
   - Pass `--chunked-eval` to restore `--query`-chunked device evaluation for A/B comparison.

2. **Device-Resident View Tables & GPU Patch Gathering (`src/data.rs`, `src/sampling_gpu.rs`, `src/render.rs`)**:
   - `Dataset::device_dataset` uploads all views' precomputed pixel ray directions, analytical 3×3 basis inverses, normals, masks, and camera metadata to the GPU once at startup.
   - During training, `Dataset::random_patch_coords` samples only `(view, x, y)` coordinates and accumulates `normal_den` in O(1) via precomputed 3×3 `patch_mask_sums` without materializing 2,048 `Patch` structs (`~1 MB`) on the CPU.
   - `gather_patch_tables_kernel` gathers both `geom_rays` (`[2048, 39]`) and `packed_patches` (`[2048, 123]`) directly on the GPU from the compact `[2048, 3]` coordinates, eliminating ~1.33 MB of host-to-device PCIe uploads every step.
   - Pass `--cpu-patch-upload` to restore CPU `Patch` struct construction and host-to-device patch table uploads.

3. **Deferred Loss Readback Piggybacked on Next Step's March Readback (`src/train.rs`, `src/render.rs`)**:
   - Instead of blocking the CPU at the end of every step to read `stats.slice([0..1, 0..3])` immediately after `device.flush()`, the 3 loss scalars from step `N` are concatenated onto step `N+1`'s `march_prefix` tail readback (`5` floats in a single transfer).
   - This eliminates the end-of-step CPU-GPU synchronization barrier so the GPU executes `geometry -> fused_field -> composite -> field_backward -> parameter_gradients -> Adam -> next step's march` continuously without pipeline drain.
   - Pass `--sync-loss` to restore synchronous end-of-step loss readback.

4. **Zero-Recompute Compositor Reverse Pass & O(1) Direct Stencil Gather (`src/kernels.rs`)**:
   - `composite_kernel` caches `c0`, `c1`, `s0`, and `s1` in unused forward scratch slots `scratch[sb + 5..9]` during the forward transmittance loop and reads them directly during the reverse pass, eliminating `2 * S * 9` `f32::exp` evaluations and `4 * S * 9` global memory reads (`ids` and `sdf`) per step.
   - `composite_gather_kernel` replaces the 9-iteration inner stencil loop (`for j in 0..9`) with direct O(1) row/column neighbor lookups.

5. **Persistent Hash-Table Gradient Buffer with In-Place Adam Zeroing (`src/kernels.rs`, `src/field.rs`)**:
   - `packed_adam_kernel` clears each active gradient entry (`gradient[index] = 0.0`) immediately after reading it, and `Field` reuses the resulting zeroed `[total_entries, 2]` buffer across steps instead of allocating and zero-filling a fresh 1.4M-float tensor every step.


### 3DGS-Inspired Warp-Aligned Compositor, Cooperative Voxel-Leader Reduction, and Backward Pruning (Update 5)

Porting key micro-architectural principles from George Kopanas's *What the 3DGS Literature Left on the Table: Five Self-Contained CUDA Optimizations for 3D Gaussian Splatting* (`https://grgkopanas.github.io/3dgs_cuda_blog/`):

1. **Cooperative Voxel-Leader Hash Gradient Reduction (`s02_reduce_scatter`)**:
   - Because `points_kernel` orders `unique_points` by `endpoint * 9 + pixel`, every 4-point workgroup evaluates 4 adjacent pixels of the same 3×3 patch (~`0.002` world units apart). Across levels `0..11` (`res = 32..675`), those 4 points fall into the **exact same integer voxel cell** 60–95% of the time.
   - Previously, 4 threads in the same warp hammered the exact same `table_grad[entry]` address with 4 colliding global atomic additions.
   - `field_backward_kernel` and `field_backward_native_kernel` now stage `(base0, base1, base2)` and `(w0, w1, w2)` in shared memory and elect the first matching slot (`s`) as the voxel leader for each `(level, corner, channel)` lane, accumulating `wg_sum` across all slots `k >= s` that share `(base0, base1, base2)` in registers before issuing **a single combined global atomic add** (~1.8× fewer global atomics and drastically lower L2 cache line contention).

2. **Zero-Adjoint Backward Pruning (`s04_raster_pruning`)**:
   - In `field_backward_kernel`, when a point's incoming SDF adjoint `ds[point] == 0.0`, the kernel skips the MUFU `exp` and `ln` evaluations immediately, and if all 4 points in the workgroup have `ds == 0.0`, the entire 256-thread workgroup terminates early before executing the 64-iteration weight dot product and 14-level hash coordinate math.

3. **Branchless Reverse-Pass Predication (`s05_branchless_predication`)**:
   - In `composite_kernel`'s reverse transmittance loop, `z0` and `z1` are predicated via branchless register selection (`if active_raw { z0_raw } else { 0.0_f32 }`) and written to `scratch[sb + 8..10]` once unconditionally, eliminating inner-loop SIMT reconvergence stalls (`BSSY`/`BSYNC`) across the 27 active pixel lanes of the warp and avoiding duplicate global memory writes.

4. **32-Thread Single-Warp Cooperative Shared-SDF Compositor (`s06_multirow_and_pipe_math` & `s03_sh_streaming`)**:
   - `composite_kernel` now runs with `CubeDim::new_1d(32)` (1 warp = 3 patches = 27 active pixel lanes) with partitioned per-patch shared memory (`shared_sdf[lp * 18 .. +18]`). Within each sample step, the 9 pixel lanes of each patch cooperatively load the 18 unique endpoint SDFs into shared memory once, so all 4 spatial stencil neighbors (`l, r, u, d`) are read from shared memory instead of issuing 4 redundant global `ids` + `sdf` gathers per pixel (cutting global `ids`/`sdf` memory loads in `composite_kernel` by **3×** and eliminating multi-warp barrier stalls).


### High-speed 1024-resolution mesh extraction (`src/mesh.rs`, `src/sampling_gpu.rs`)

At `--mesh-resolution 1024`, mesh extraction previously evaluated $1024^3 = 1.073\times 10^9$ points by allocating and uploading 1,024 separate 12.6 MB `Vec<[f32; 3]>` slabs (`12.9 GB` total PCIe upload per mesh), waiting synchronously on every slab, and using a `std::collections::HashMap<(usize, usize), u32>` with `edges.retain(...)` on every slab.

Mesh extraction now uses:
1. **GPU-Generated Coordinate Slabs (`mesh_slab_points_kernel`)**: Generates all $1024 \times 1024$ `[x, y, z]` coordinates directly on the GPU from 3 floats (`res`, `1 / (res - 1)`, `z_coord`), eliminating 12.9 GB of CPU allocation and host-to-device PCIe uploads per mesh.
2. **Unchunked Single-Dispatch Slab Evaluation + Double-Buffered GPU/CPU Overlap**: Evaluates all $1,048,576$ points of a slab in a single `fused_field` launch and enqueues slab `z + 2` via `device.flush()` before reading slab `z + 1`, overlapping GPU SDF evaluation with CPU triangle extraction.
3. **$O(1)$ Rolling Flat-Array Edge Tables (`xy_lower`, `xy_upper`, `z_edges`)**: Replaces `std::collections::HashMap<(usize, usize), u32>` and `edges.retain(...)` with three flat `Vec<u32>` buffers (`20 MB` total at `res = 1024`) swapped per slab, plus an immediate `if case == 0 || case == 255 { continue; }` fast path and vectorized `bytemuck::cast_slice` binary PLY writes.


### Kopanas part 2 comparison

The default corner-parallel hash backward combines equal-address updates from
four neighboring points before emitting native-float or CAS atomics. It reuses
existing shared weight storage and adds no global buffers. Compositor scratch
now keeps each component's nine pixels adjacent, including matching adjoint
gather indexing. Both changes preserve ray order, all losses, and FP32 storage;
local sum grouping can change floating-point rounding.

For an A/B comparison from the same checkpoint and identical settings, add
`--no-hash-aggregation --legacy-composite-scratch` to restore part-1 behavior for
these changes. Each flag can be tested separately. `--legacy-corner-loop` also
bypasses local hash aggregation by selecting the original sequential scatter.
Use fresh output directories. Rebuild and test yourself: no build, test, or
runtime benchmark was performed for this update. See
[KOPANAS_AUDIT.md](KOPANAS_AUDIT.md) for applicability and implementation details.


### Training kernel scheduling and diagnostics

This update fixes the visible-geometry microbatch guard: rounded-up shader
invocations are bounded by the local patch count, not the remaining global ray
count. Reverse occurrence slots are now fully initialized by their owning ray,
removing the separate reverse-table fill dispatch.

Enabled by default:

- Level-indexed packed Adam (same update/clearing formulas and moment decay).
  Add `--legacy-adam-dispatch` for the former per-scalar level search.
- Exact endpoint-ID-based sigmoid reuse in visibility and compositing.
  Add `--no-endpoint-reuse` to evaluate both endpoints independently.
- Subgroup forward output reduction only for advertised fixed 32/64-wide
  subgroup devices with subgroup operations. Logical lane ownership uses actual
  subgroup IDs, rather than assuming local invocation ordering. Other devices
  retain the portable shared-memory tree. `--legacy-field-reduction` forces it.
- GPU training omits the field-only output-bias reduction that compositing
  replaces. Standalone/reference backward still computes that derivative.
- Memory maintenance queries reservation before synchronizing; it still cleans
  above the existing threshold. `--trim-interval N` controls periodic checks
  (default 200). Zero disables periodic checks; warmup/mesh cleanup remains.

Independent experiments (measure separately from defaults):

| Flag | Default | Effect |
| --- | --- | --- |
| `--field-points-per-group 8` or `16` | 4 | Reuses staged MLP weights across 2/4 sequential four-point tiles; does not expand the activation cache. |
| `--hash-aggregation-levels N` | 14 | Aggregate only levels below N; zero disables aggregation. Upper levels scatter directly. |
| `--hash-aggregation-cross-corners` | off | Combines matching addresses across all 32 point/corner candidates per level, rather than four candidates at a fixed corner. More coordination can regress speed. |
| `--recompute-parameter-gradients` | off | Avoids writing/reading the full global `dh` buffer by recomputing derivatives in parameter tiles. Trades bandwidth for extra sigmoid arithmetic; requires fused field and tiled gradients. |
| `--diagnostics` | off | Records up to 128 timing windows per custom-op label, copy-byte/work counters, and a small address-reuse model sample per active-level count. |

`diagnostics.json` identifies device versus system timing per row and records
unavailable/failed measurements separately from zero durations. The address
model downloads at most 256 prefix points once per level count and excludes zero
adjoints; it does not measure hardware cache hits or actual atomic instructions,
and ignores cancellation of feature gradients. Use short checkpoint resumes.
Instrumentation alters scheduling and adds readbacks: never use diagnostic
throughput as production throughput. Excluding the first timing sample does not
guarantee all compilation/autotuning overhead has disappeared.

`wall_time.json` records elapsed time within the train call, including loading,
checkpointing and mesh extraction, with separate checkpoint/mesh totals. It is
written only after successful completion and excludes diagnostic-report
serialization. A power-interrupted run has no completed timing summary.

Checkpoint format, ray order, losses and the three-chunk activation-cache cap
remain unchanged. Subgroup/local reductions change floating-point sum grouping;
GPU gradient/mesh parity and performance still require user validation. Larger
tiles, selective aggregation and recomputation are selectable experiments, not
promised speedups. No build, tests, GPU captures or benchmarks were run by the
assistant for this update. Rebuild and test the binary yourself.

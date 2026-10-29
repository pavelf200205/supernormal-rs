# SuperNormal-RS Performance Optimizations (Update 3)

## Overview

In Update 2, SuperNormal-RS already surpassed the original PyTorch (`tiny-cuda-nn` + `nerfacc`) implementation in iteration speed, but exhibited two clear hardware symptoms:

1. **Low GPU Power Utilization (\~85 W out of 170 W TDP)** on an NVIDIA RTX 3060.

2. **Progressive Throughput Decay**: Iteration speed started around \~45 it/s at step 2,000 (`active_levels = 1`) and decayed almost linearly to \~12 it/s by step 30,000 (`active_levels = 14`).

This update resolves the root causes of both bottlenecks across `src/kernels.rs`, `src/sampling_gpu.rs`, `src/render.rs`, `src/field.rs`, `src/train.rs`, and `src/data.rs`.

## Summary of Bottlenecks and Fixes

| Optimization Area | Previous Bottleneck | Update 3 Implementation | CLI A/B Flag to Revert | 
| ----- | ----- | ----- | ----- | 
| **1. 256-Thread Corner-Parallel Hash Forward & Backward** | Only `4 * active` (`<= 56`) of `256` threads in each workgroup executed hash-grid lookups/scatters, looping sequentially over `8` corners while `200+` threads sat idle behind `sync_cube()`. | Maps `256` threads across `(point_slot in 0..4, corner_pair in 0..4, level in 0..16)`. Up to `224` threads (`87.5%` of workgroup) issue `2` corners each in parallel (`4×` higher memory-level parallelism), reducing partial pairs in shared memory. | `--legacy-corner-loop` | 
| **2. Device-Resident March, Visibility & Geometry Tables** | Every step forced 3 synchronous GPU-to-CPU stalls and multi-megabyte host-to-device uploads (`marched` download, CPU `unique_pts` upload, `visibility` download, CPU `mapping` `[count, 18]` and `reverse` `[n, 2]` construction & upload). | Fuses endpoint counting into `march_kernel` and `visibility_kernel`, generates center-ray endpoints on device (`compact_and_center_endpoints_kernel`), builds `endpoints`, `descriptors`, `mapping`, and `reverse` on the GPU without atomics (`build_visible_geometry_tables_kernel`), and reads per-patch ranges from `vis_prefix` in `composite_kernel`. | `--host-sample-compaction` | 
| **3. Uninitialized Output Allocations & SFU Deduplication** | `Tensor::zeros` dispatched \~10 buffer-clearing kernels per step for buffers immediately overwritten by custom ops; `field_backward_kernel` computed `exp()` twice per neuron. | Replaces write-only `Tensor::zeros` buffers with `Tensor::empty` across all custom ops and reuses a single `exp(min(scaled, 20.0))` evaluation for both sigmoid derivative and softplus activation in backward. | Always active | 
| **4. Precomputed Pixel Ray Bases & Analytical Inverses** | `View::patch` normalized `18,432` rays and ran `18,432` general 3×3 matrix inversions (`basis.try_inverse()`) on the CPU every step. | Precomputes world-space ray directions and closed-form orthonormal cross-product inverses for all pixels during `Dataset::load`, turning `View::patch` into a direct memory lookup. | Always active | 

## Detailed Technical Breakdown

### 1. 256-Thread Corner-Parallel Hash Kernels (`src/kernels.rs`)

In both `fused_field_kernel` and `field_backward_kernel` / `field_backward_native_kernel`, each workgroup has `256` threads (`CubeDim::new_1d(256)`) processing `4` points (`slot in 0..4`, `lane in 0..64`).

* **Why power was capped at 85 W and speed dropped with `active_levels`**:
  During the MLP phase, all `256` threads are active (`4` points × `64` hidden neurons). However, during the hash-grid phase, work was mapped by `(slot, level)`, which meant at most `4 * 14 = 56` threads (`21.8%` of the workgroup) were active while `200` threads waited at `sync_cube()`. Worse, as `active_levels` grew from `1` to `14` (every 2,000 steps), levels `5..13` each span `524,288 × 2` floats (`4 MB` per level, `36 MB` total), exceeding the GPU's L2 cache. Because each of the 56 active threads executed a sequential `for corner in 0..8` loop, DRAM latency could not be hidden by warp-level concurrency.

* **How Update 3 fixes it**:

  * **Forward (`fused_field_kernel`)**: Threads are mapped as:

    * `hash_slot = unit & 3` (`4` points)

    * `corner_pair = (unit >> 2) & 3` (`4` pairs of trilinear cube corners)

    * `level = unit >> 4` (`16` level slots, covering all `0..14` levels)
      Now `4 × 4 × active` threads (up to `224` threads at `active = 14`) issue memory loads simultaneously—putting `448` trilinear corner lookups in flight per workgroup instead of `112`. Each thread accumulates its 2 corners and writes `[feat0, feat1]` to `partials[unit * 2 .. unit * 2 + 2]`. After `sync_cube()`, threads `0..56` sum the 4 corner-pairs in shared memory and write `shared_features`.

  * **Backward (`field_backward_kernel` & `field_backward_native_kernel`)**:

    1. Threads `0..56` compute feature adjoints `(g0, g1)` from shared `dh` and `shared_weights` and stage them in `grad_staging` (`112` floats in shared memory).

    2. After `sync_cube()`, all `224` `(slot, corner_pair, level)` threads read `(g0, g1)` from `grad_staging` and scatter-add their `2` trilinear corners concurrently into `table_grad`.

  * **Transcendental SFU Deduplication**:
    Previously, `field_backward_kernel` evaluated `f32::exp(-scaled)` for the sigmoid derivative and then evaluated `f32::exp(f32::min(scaled, 20.0))` a second time on the next line for `softplus`. Using the identity $\sigma(x) = \frac{e^x}{1 + e^x}$, both expressions now share a single `exp_pos = f32::exp(f32::min(scaled, 20.0_f32))` evaluation.

### 2. Device-Resident March, Visibility, and Geometry Pipeline (`src/sampling_gpu.rs`, `src/render.rs`, `src/train.rs`)

Previously, even with GPU ray marching, GPU visibility, GPU geometry, and GPU compositing enabled, the CPU still acted as a middleman three times per step:

1. `march_device` read back `total`, ran `compact`, and downloaded all `[total, 3]` marched samples (`try_into_vec_as::<f32>()`) to the CPU.

2. `visible` looped over the marched samples on the CPU to deduplicate center-ray endpoints, uploaded `unique_pts`, `pairs`, and `ranges` back to the GPU, ran `visibility_kernel`, and blocked on downloading `[patches + 1, 1]` counts to filter `Vec<Sample>` on the CPU.

3. `prepare_geometry_device` and `composite_device` looped over the surviving `Vec<Sample>` on the CPU to build `endpoints`, `descriptors`, `mapping` (`[count, 18]`), and `reverse` (`[n, 2]`), uploading several megabytes of index tables back to the GPU every step.

In Update 3 (`march_and_filter_device` + `prepare_geometry_from_device_batch` + `composite_device_tables`):

* **Fused March & Center-Endpoint Counting (`march_kernel`)**: Each ray thread counts both its marched intervals (`count`) and its deduplicated center-ray endpoints (`ep_count`, incremented by `1` when `t_start == prev_end` and `2` otherwise), writing `[patches, 2]`. A single `counts.cumsum(0)` and a 2-float tail slice readback (`[total_marched, unique_ep_count]`) provides exact allocation sizes for both buffers at once.

* **Fused Compaction & Center-Endpoint Generation (`compact_and_center_endpoints_kernel`)**: One GPU launch compacts `marched` (`[total_marched, 3]`) and simultaneously writes `unique_pts` (`[unique_ep_count, 3]`), `pairs` (`[total_marched, 2]`), and `ranges` (`[patches, 2]`) on the device.

* **Dual-Count Visibility (`visibility_kernel`)**: Each ray thread tracks both visible sample count and retained unique endpoint count (`[patches + 1, 2]`), followed by `vis_counts.cumsum(0)`. For the default `microbatch == batch`, only a 3-float slice (`[total_visible, total_visible_endpoints, variance]`) is read back to the host.

* **Zero-Atomic GPU Geometry & Reverse-Index Builder (`build_visible_geometry_tables_kernel`)**:
  Because each endpoint is either an unshared `start` of sample $k$ (1 occurrence) or the `end` of sample $k$ optionally reused as the `start` of sample $k+1$ within the same ray (at most 2 occurrences), each ray thread deterministically populates `endpoints` (`[ep_count, 2]`), `descriptors` (`[vis_count, 5]`), `mapping` (`[vis_count, 18]`), and `reverse` (`[ep_count * 9, 2]`) directly on the GPU without atomics or host transfers.

* **Direct Device Prefix Lookup in Compositor (`composite_kernel`)**:
  `composite_kernel` reads each patch's `[begin, end)` visible sample range directly from `vis_prefix` on the GPU when `patches[pb] < 0.0`.

### 3. Eliminating Redundant `Tensor::zeros` Dispatches (`src/kernels.rs`, `src/sampling_gpu.rs`, `src/field.rs`, `src/render.rs`)

Burn's `Tensor::zeros` allocates a buffer and enqueues a GPU fill kernel. Across `fused_field`, `field_backward`, `parameter_gradients`, `composite_gpu`, `geometry`, `march`, `compact`, `occupancy`, and `visibility`, \~10 output buffers per step were being zero-filled by Burn immediately before custom CubeCL kernels overwrote every element. Switching these write-only output buffers to `Tensor::empty` eliminates those redundant dispatches and reduces memory bandwidth overhead.

### 4. Precomputed Analytical Camera Ray Bases (`src/data.rs`)

For every pixel $(x, y)$ in a view, the ray direction $\mathbf{d}$ and the 3×3 basis matrix

$$
B = \begin{bmatrix} \mathbf{d}^\top \\ \mathbf{u}^\top \\ \mathbf{v}^\top \end{bmatrix}
$$

depend only on the static camera intrinsics and extrinsics. Furthermore, because the camera's $\mathbf{u}$ and $\mathbf{v}$ axes (columns 0 and 1 of `rotation`) are orthonormal with $\mathbf{w} = \mathbf{u} \times \mathbf{v}$, the inverse $B^{-1}$ has the exact closed form:

$$
B^{-1} = \frac{1}{\mathbf{d} \cdot \mathbf{w}} \begin{bmatrix} \mathbf{w} & \mathbf{v} \times \mathbf{d} & \mathbf{d} \times \mathbf{u} \end{bmatrix}
$$

`Dataset::load` now precomputes `cached_dirs` and `cached_inverses` in parallel across all views once on startup, reducing `View::patch` during training to a fast array copy.

## Benchmarking & A/B Comparison Guide

To benchmark the full Update 3 pipeline against Update 2 from an existing checkpoint:

```
# 1. Build release binary
cargo build --release --locked

# 2. Run Update 3 (all optimizations enabled by default)
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_update3 --steps 500 --save-freq 0

# 3. Isolate 256-thread corner parallelism vs legacy 56-thread corner loop
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_legacy_corners --steps 500 --save-freq 0 --legacy-corner-loop

# 4. Isolate device-resident sample/geometry pipeline vs host sample compaction
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_host_compaction --steps 500 --save-freq 0 --host-sample-compaction

# 5. Restore both to match Update 2 kernel/pipeline behavior
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_update2_baseline --steps 500 --save-freq 0 --legacy-corner-loop --host-sample-compaction




```

## Update 4 Optimizations: Eliminating Per-Step Pipeline Drains, PCIe Uploads, and Chunk Overhead

While Update 3 parallelized the 256-thread hash kernels and kept sample compaction on the GPU, five additional bottlenecks remained across the step lifecycle:

| Optimization Area | Previous Bottleneck (Update 3) | Update 4 Implementation | CLI A/B Flag to Revert | 
| ----- | ----- | ----- | ----- | 
| **1. Unchunked Single-Dispatch Device Evaluation & Backward** | `evaluate_device_tensor`, `enqueue_cached_device`, and `accumulate_gradients_device` split GPU-resident point buffers into `65,536`-point chunks, zero-padded the tail chunk (`Tensor::cat`), and sliced/concatenated outputs (`Tensor::cat`). | Evaluates all device-resident points (`<= 16.7M`) in a single unpadded `fused_field`, `field_backward`, and `parameter_gradients` dispatch, eliminating 3–5× kernel launch and concatenation overhead per step. | `--chunked-eval` | 
| **2. Device-Resident View Tables & GPU Patch Gathering** | Every step gathered 2,048 `Patch` structs (`~1 MB`) on the CPU and uploaded `geom_rays` (`[2048, 39]` = 319 KB) and `packed_patches` (`[2048, 123]` = 1.01 MB) over PCIe. | Uploads all view camera metadata and pixel `[dir(3), inv(9), normal(3), mask(1)]` tables to the GPU once at startup (`Dataset::device_dataset`), computes `normal_den` in O(1) via precomputed 3×3 `patch_mask_sums`, and gathers `geom_rays` + `packed_patches` on the GPU from compact `[2048, 3]` `(view, x, y)` coordinates (`gather_patch_tables_kernel`). | `--cpu-patch-upload` | 
| **3. Deferred Loss Readback (Continuous Async Step Overlap)** | At the end of every step, `stats.slice([0..1, 0..3]).try_into_vec_as::<f32>()` forced a synchronous GPU pipeline drain immediately after `device.flush()`, preventing the CPU from preparing step `N+1` while the GPU executed step `N` backward + Adam. | Piggybacks step `N`'s 3 loss scalars onto step `N+1`'s `march_prefix` tail readback (`5` floats in a single transfer), allowing the GPU to execute `geometry -> fused_field -> composite -> backward -> Adam -> next step march` continuously without an end-of-step stall. | `--sync-loss` | 
| **4. Zero-Recompute Compositor Reverse Pass & O(1) Stencil Gather** | `composite_kernel` recomputed `f32::exp(-s0 * inv)` and `f32::exp(-s1 * inv)` and re-read global `ids` and `sdf` in the reverse loop; `composite_gather_kernel` looped over all `j in 0..9` pixels to check 4 stencil neighbors. | Caches `c0`, `c1`, `s0`, `s1` in unused forward scratch slots `scratch[sb + 5..9]` (saving `2 * S * 9` `exp()` calls and `4 * S * 9` global memory reads per step) and replaces the 9-iteration gather loop with direct O(1) neighbor indexing. | Always active | 
| **5. Persistent Hash-Gradient Buffer with In-Place Adam Zeroing** | `accumulate_gradients_device` allocated and zero-filled a new `[entries, 2]` tensor (`up to 1.4M floats` = 5.6 MB) every step. | `packed_adam_kernel` clears `gradient[index] = 0.0` in-place immediately after reading `g`, and `Field` reuses the persistent zeroed buffer across steps. | Always active | 

### Update 4 Benchmarking Commands

```
# 1. Build release binary
cargo build --release --locked

# 2. Run Update 4 (all optimizations enabled by default)
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_update4 --steps 500 --save-freq 0

# 3. Compare against Update 3 behavior (revert chunking, CPU patch uploads, and synchronous loss readback)
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\checkpoint_025000 --out runs\bench_update3_equiv --steps 500 --save-freq 0 --chunked-eval --cpu-patch-upload --sync-loss




```

## Update 5 Optimizations: Porting 3DGS Micro-Architectural Techniques (`grgkopanas.github.io/3dgs_cuda_blog`)

Inspired by George Kopanas's *What the 3DGS Literature Left on the Table: Five Self-Contained CUDA Optimizations for 3D Gaussian Splatting* (`s01_literature` through `s06_multirow_and_pipe_math`), Update 5 ports the exact hardware-level mechanisms from those progressive CUDA diffs into SuperNormal-RS's CubeCL kernels (`src/kernels.rs`):

| 3DGS Blog Diff Stage | SuperNormal-RS Bottleneck Solved | Update 5 Implementation | 
| ----- | ----- | ----- | 
| **`s02_reduce_scatter`** | 4 adjacent patch points in a workgroup (`~0.002` world units apart) fall into the **exact same voxel cell** across levels `0..11`, causing 4 threads in the same warp to issue 4 colliding global `atomic_add` / `fetch_add` calls to the same `table_grad[entry]` address. | **Cooperative Voxel-Leader Hash Gradient Reduction**: Stages `(base0, base1, base2)` and `(w0, w1, w2)` for the 4 slots in shared memory (`base_staging`, `w_staging`). For each `(level, corner, channel)` lane, the first slot with a given voxel `(base0, base1, base2)` sums `wg_sum` across all matching slots `k >= s` in registers and emits **1 combined global atomic add** (\~1.8× fewer global atomics, eliminating intra-workgroup L2 atomic collisions). | 
| **`s03_sh_streaming`** | Bulk `cudaMemsetAsync` / `Tensor::zeros` zero-fill dispatches and redundant global reads across neighboring lanes. | **In-Place Adam Gradient Zeroing & Cooperative Staging**: `packed_adam_kernel` zeroes active hash-table gradient entries in-place as they are consumed (`gradient[index] = 0.0`), and `composite_kernel` cooperatively stages 18 unique endpoint SDFs per patch in shared memory. | 
| **`s04_raster_pruning`** | Points with `ds[point] == 0.0` (e.g. unshared endpoints or zero-gradient samples) still executed `f32::exp`, `f32::ln`, the 64-step hidden-weight loop, and 14-level hash coordinate math; occluded samples beyond transmittance saturation still ran through backward. | **Zero-Adjoint Backward Pruning & Visibility Compaction**: `visibility_kernel` truncates ray sample ranges at transmittance saturation (`trans < 1e-8`) before field evaluation, while `field_backward_kernel` skips `exp`/`ln` and the 64-neuron dot product when `ds[point] == 0.0` and terminates the entire 256-thread workgroup (`terminate!()`) when all 4 slots have `ds == 0.0`. | 
| **`s05_branchless_predication`** | In `composite_kernel`'s reverse transmittance loop, a conditional `if raw >= 0.0 && raw <= 1.0` block with global stores (`scratch[sb + 8..10]`) caused SIMT `BSSY`/`BSYNC` reconvergence stalls across the 27 active pixel lanes of the warp and issued two global writes per slot when `raw` was in bounds. | **Branchless Reverse-Pass Predication**: Computes `z0` and `z1` via branchless register selection (`if active_raw { z0_raw } else { 0.0 }`) and issues a single unconditional global write to `scratch[sb + 8]` and `scratch[sb + 9]`, eliminating inner-loop warp reconvergence barriers and duplicate global stores. | 
| **`s06_multirow_and_pipe_math`** | `composite_kernel` previously launched 256-thread blocks (\~28 patches per block) where every pixel independently gathered `s0`, `s1`, `sdf[l]`, `sdf[r]`, `sdf[u]`, and `sdf[d]` from global memory (`54` global `ids` + `sdf` reads per sample instead of `18`) and synchronized all 8 warps at block-wide barriers. | **32-Thread Single-Warp Cooperative Shared-SDF Compositor**: Launches `CubeDim::new_1d(32)` (1 warp = 3 patches = 27 active pixel lanes) with partitioned per-patch shared memory (`shared_sdf[lp * 18 .. +18]`), cutting global `ids`/`sdf` reads by **3×**, eliminating multi-warp barrier stalls, and caching `(c0, c1, s0, s1)` in forward scratch to eliminate `2 * S * 9` `exp` calls in reverse. | 

### High-speed 1024-resolution mesh extraction (`src/mesh.rs`, `src/sampling_gpu.rs`)

At `--mesh-resolution 1024`, mesh extraction previously evaluated $1024^3 = 1.073\times 10^9$ points by allocating and uploading 1,024 separate 12.6 MB `Vec<[f32; 3]>` slabs (`12.9 GB` total PCIe upload per mesh), waiting synchronously on every slab, and using a `std::collections::HashMap<(usize, usize), u32>` with `edges.retain(...)` on every slab.

Mesh extraction now uses:

1. **GPU-Generated Coordinate Slabs (`mesh_slab_points_kernel`)**: Generates all $1024 \times 1024$ `[x, y, z]` coordinates directly on the GPU from 3 floats (`res`, `1 / (res - 1)`, `z_coord`), eliminating 12.9 GB of CPU allocation and host-to-device PCIe uploads per mesh.

2. **Unchunked Single-Dispatch Slab Evaluation + Double-Buffered GPU/CPU Overlap**: Evaluates all $1,048,576$ points of a slab in a single `fused_field` launch and enqueues slab `z + 2` via `device.flush()` before reading slab `z + 1`, overlapping GPU SDF evaluation with CPU triangle extraction.

3. $O(1)$ **Rolling Flat-Array Edge Tables (`xy_lower`, `xy_upper`, `z_edges`)**: Replaces `std::collections::HashMap<(usize, usize), u32>` and `edges.retain(...)` with three flat `Vec<u32>` buffers (`20 MB` total at `res = 1024`) swapped per slab, plus an immediate `if case == 0 || case == 255 { continue; }` fast path and vectorized `bytemuck::cast_slice` binary PLY writes.

## Update 6 Optimizations: Telemetry-Driven Fixes for `Occupancy::update`, Compositor Barriers, and the >4 GB Early VRAM Spike

Analysis of the 30,000-step `metrics.csv` (`1,038.91s` total step time) and Windows Task Manager GPU telemetry uncovered the exact root causes of both the remaining iteration time and the early `> 4 GB` VRAM spike on steps 1–3,000:

### Telemetry Breakdown from `metrics.csv`

| Component in `metrics.csv` | Measured Time | Share of Run | Root Cause | 
| ----- | ----- | ----- | ----- | 
| **1. `Occupancy::update` Overhead (`step % 8 == 0`, 3,750 steps)** | **296.4 s (4.94 min)** | **28.5%** | Regular steps averaged `23.4 ms`, while every 8th step averaged `100.8 ms` (`+77.4 ms` overhead per update). Every 8 steps, the CPU generated `0.6M–2.1M` jittered 3D points on a single thread (`1.8M–6.3M` RNG calls), built an `8.4 MB` linked list (`heads`/`next`) in RAM, uploaded `~22 MB` over PCIe, and downloaded `8.4 MB` (`flags`) back to the host while the GPU sat idle. | 
| **2. Per-Sample Level-Independent Cost (`0.311 ms / 1k samples`)** | **412.6 s (6.88 min)** | **39.7%** | (a) Update 5's 32-thread 1-warp `composite_kernel` grouped 3 patches with `2 * max_len` `sync_cube()` barriers, capping Ampere SM occupancy at 33% (`16 blocks/SM`) and forcing the 70–80% background rays (`0` samples) to loop through `50–150` barrier iterations whenever grouped with a foreground ray. (b) `build_visible_geometry_tables_kernel` ran a 9-pixel inner loop writing `36` floats per sample (`mapping` `[vis_count, 18]` and `reverse` `[ep_count * 9, 2]`). | 
| **3. Early Sample Spike (`steps 1–900`, up to `215,025` samples = `1.95M` points) & Unbucketed Pool Thrashing** | **66.6 s + >4 GB VRAM** | **6.4%** | Disabling chunking (`chunked_eval = false`) in Update 4 caused `Field` to allocate single giant `[1.95M, 64]` (`preactivation` = `500 MB`), `[1.95M, 31]` (`feat` = `242 MB`), and `[1.95M, 64]` (`dh` = `500 MB`) tensors (`~1.25 GB` per step) with a different exact row count on every step. Because `wgpu`'s memory pool cannot reuse a smaller buffer when the next step's point count increases, multiple multi-hundred-megabyte buffers accumulated in the pool (`> 4 GB` VRAM) until `step % 200` cleanup. | 
| **4. `field_backward_kernel` Shared Memory Cliff (`> 12 KB`) & 56-Thread Dot Product** | **156.0 s (2.60 min)** | **15.0%** | Adding `base_staging` (`672 B`) + `w_staging` (`672 B`) pushed shared memory to `12,308 B` (`4 * 12,308 = 49,232 > 48 KB`), dropping RTX 3060 occupancy from 4 to 3 blocks/SM (`-25%`), while only 56 of 256 threads computed the 64-neuron `dh * W0` dot product. | 

### Update 6 Architectural Fixes

1. **100% GPU-Resident `Occupancy::update` (`src/sampling_gpu.rs`, `src/render.rs`)**:

   * Eliminates all CPU point generation, CPU linked lists (`heads`/`next`), PCIe point uploads, and PCIe `flags` downloads during training (`flags` and `values` are only synced to the host when saving a checkpoint).

   * During warmup (`step < 256`), `occupancy_grid_points_kernel` and `occupancy_apply_slice_kernel` generate jittered 3D cell coordinates via PCG32 hashing and update `device_values` in-place on the GPU in reusable `chunk`-sized slices.

   * After warmup (`step >= 256`), `occupancy_count_kernel` and `occupancy_compact_points_kernel` select 100% of occupied cells plus 1 stratified random cell per 4-cell block (`n / 4` unoccupied cells) directly on the GPU without duplicates or linked lists, cutting `Occupancy::update` from `~79 ms` to `~2.5 ms` (**saving \~4.7 minutes over 30,000 steps**).

2. **Flat \~1.6–1.8 GB VRAM from Step 1 via Bucketed Capacities & `MAX_CACHED_CHUNKS = 3` (`src/field.rs`, `src/sampling_gpu.rs`, `src/render.rs`)**:

   * All transient per-step buffers (`samples`, `marched`, `pairs`, `endpoints`, `descriptors`, `mapping`, `reverse`, `geometry`, `scratch`) use `bucket_capacity(n)` (at most 4 bucket steps per power-of-two octave), and `unique_points` is allocated at `Field::padded_point_capacity(valid_points, query)` so every chunk slice already has exact static power-of-two shape `[padded_n, 3]` with **zero `Tensor::zeros` + `Tensor::cat` padding copies**.

   * `Field` caps simultaneously resident cached `(feat, preactivation)` chunks in VRAM at `MAX_CACHED_CHUNKS = 3` (`<= 149 MB` at `--query 131072`). During the step 1–900 warmup spike (`up to 1.95M points`), any chunk beyond the first 3 streams through a single pooled `[131072, 31]` / `[131072, 64]` buffer pair in `accumulate_gradients_device`, preventing the `> 4 GB` early VRAM spike and releasing warmup high-water buffers at `step == 1000`.

3. **Barrier-Free Compositor & Compact Contiguous Endpoint Indexing (`src/kernels.rs`, `src/sampling_gpu.rs`)**:

   * Because `end_ep` is always `start_ep + 1` in `build_visible_geometry_tables_kernel`, all 18 points of sample `s` are contiguous at `sdf[start_ep * 9 .. start_ep * 9 + 18]`.

   * `mapping` is shrunk **18×** (`[vis_cap, 1]` storing `start_ep * 9`), `reverse` is shrunk **9×** (`[ep_cap, 2]` storing endpoint-level occurrence codes), and the 9-iteration inner pixel loop in `build_visible_geometry_tables_kernel` is completely eliminated.

   * `composite_kernel` launches 1 thread per pixel ray at full SM warp occupancy with **zero `sync_cube()` barriers**, allowing the 70–80% background rays (`begin == end`) to exit immediately while foreground rays read contiguous SDFs from L1 cache.

4. **Restored 4 Blocks/SM Occupancy (`< 12 KB` Shared Memory) & 224-Thread Parallel MLP Dot Product in `field_backward_kernel` (`src/kernels.rs`)**:

   * Reuses a single `448`-float `partials` shared buffer for both `output_grads` and `grad_staging`, reducing shared memory from `12,308 B` to `11,284 B` (`< 12 KB`, restoring 4 blocks/SM = 1,024 threads/SM on Ampere).

   * Splits the 64-neuron `dh * W0` dot product across all **224 threads** (`16` iterations per thread instead of `64` on 56 threads), followed by 224-thread parallel trilinear corner scatter.


## Update 7 Optimizations: Eliminating Per-Step CPU Bottlenecks & Mid-Step GPU Pipeline Drains (Targeting >85% GPU Load on Older CPUs)

After reaching `< 11 minutes` total reconstruction time, Task Manager telemetry showed **50–60% GPU 3D load** alongside noticeable **per-step CPU utilization**. Inspecting the step lifecycle revealed four remaining CPU-to-GPU bottlenecks that starved the RTX 3060 on an older CPU:

| Bottleneck | Why It Capped GPU Load at 50–60% & Loaded the CPU | Update 7 Implementation |
| ----- | ----- | ----- |
| **1. Per-Step Single-Threaded CPU Ray-Sphere & 18,432-Pixel `f16` Normal/Mask Gathering (`src/render.rs`, `src/data.rs`, `src/sampling_gpu.rs`)** | Every step ran a single-threaded CPU loop over all 2,048 patches that: (a) read `view.cached_dirs` across `442 MB` of host RAM (2,048 L3 cache misses/step), (b) solved 2,048 ray-sphere intersections on the CPU, (c) looped over `2,048 × 9 = 18,432` pixels across 40 views in scattered DRAM, (d) executed **`55,296` `half::f16::to_f32` conversions per step (`1.66 billion` over 30,000 steps)**, and (e) uploaded **`418 KB` (`104,448` floats: `[2048, 12]` + `[2048, 39]`)** over PCIe every step (`12.5 GB` total). Startup also allocated `1.76 GB` of host RAM for `cached_dirs` + `cached_inverses` and `147 MB` for `f32` `patch_mask_sums`. | **Cropped Lossless `f16` GPU Pixel Banks & `march_from_coords_kernel`**: Uploads per-view tight mask-bounding-box normals & masks once at startup into `DeviceDataset::pixel_banks` using a NaN/FTZ-immune 24-bit integer float encoding (`f0 = -1.0` when unmasked; exact `nx, ny, nz` `f16` bitfields in `f0, f1` when masked, unpacked on the GPU via `unpack_f16_to_f32` in `gather_patch_tables_kernel`). The CPU now uploads only **`[2048, 4]` (`[v, x, y, jitter]` = `32 KB`, 13× smaller)** and `march_from_coords_kernel` computes center ray directions, ray-sphere intersections, and occupancy ray marching directly on the GPU. `cached_rays` (`1.76 GB`) is now lazy (`OnceLock`) and `patch_mask_sums` is shrunk **4×** (`Vec<u8>`, `36.8 MB`), fitting in CPU L3 cache. |
| **2. Two Synchronous GPU-to-CPU Pipeline Drains Per Step (`src/render.rs`, `src/sampling_gpu.rs`, `src/kernels.rs`)** | Every step drained the WDDM GPU queue to idle **twice**: first after `march_kernel` (`march_prefix` tail readback for `total_marched` and `unique_ep_count`), and a second time ~1.5 ms later after `visibility_kernel` (`vis_prefix` tail readback for `total_visible`, `total_visible_endpoints`, and `variance`). Each WDDM staging-buffer map + OS thread wake-up on an older CPU left a `1–2 ms` GPU idle gap (`0%` 3D load). | **Single-Readback Speculative March + Visibility Pipeline (`visibility_device_speculative`)**: Tracks adaptive capacities (`1.5×` previous step's `total_marched` and `unique_ep_count`, floor `32,768`). `compact_and_center_endpoints_kernel` and `fused_visibility_field` read `total_marched` and `unique_ep_count` **directly from `march_prefix` on the GPU**, executing `march_from_coords -> scan_2d_pairs -> compact_and_center_endpoints -> fused_visibility_field -> visibility -> scan_vis_and_pack_meta` in **one continuous GPU submission** with **only 1 CPU-GPU readback per step** (cutting 30,000 full GPU pipeline stalls out of training). |
| **3. 11 Multi-Dispatch `cumsum` / `slice` / `reshape` / `cat` Chains Around Readbacks (`src/sampling_gpu.rs`, `src/render.rs`)** | Before each readback, Burn launched generic axis-0 `.cumsum(0)` kernels followed by `.slice()`, `.reshape()`, and `Tensor::cat()` copy kernels (11 small kernel dispatches per step on the critical sync path). | **Fused Single-Workgroup Shared-Memory Scans (`scan_2d_pairs_kernel`, `scan_1d_counts_kernel`, `scan_vis_and_pack_meta_kernel`)**: Scans all 2,048 `[count, ep_count]` pairs in shared memory inside a single 256-thread workgroup (`< 2 µs`) and packs `[total_marched, unique_ep_count, total_visible, total_visible_endpoints, variance, loss_normal, loss_mask, loss_eikonal]` into `[1, 8]` in one launch. |
| **4. ~124 Per-Step CPU Burn Tensor Graph Ops & ~16 Small Kernel Launches in MLP Weight-Norm + Adam + Weight Packing (`src/kernels.rs`, `src/field.rs`, `src/sampling_gpu.rs`)** | Every step constructed ~124 Burn tensor ops on the CPU (`3.7M` ops over 30,000 steps): 4 ops to reduce/slice `parameter_gradients`, 2 ops + a CPU tensor upload to reduce/scale `composite_gpu` stats, 1 op for `dw1.sum_dim(0)`, 22 ops for `weight_gradients(0)` & `weight_gradients(1)`, 4 ops to slice `variance`/`lin1.bias` grads, 84 ops for the 7-parameter Adam loop, and 12 ops + `pack_mlp_weights_kernel` for `effective_weight(0/1)`. | **Fused Reductions & 1-Workgroup `fused_mlp_adam_and_pack_kernel`**: `ParameterOp`, `CompositeOp`, and `FieldBackwardOp` now fuse `reduce_parameter_partials_kernel`, `reduce_composite_stats_kernel` (with scalar `vscale`), and `reduce_dw1_kernel`. Then a **single 1-workgroup (256-thread) kernel (`fused_mlp_adam_and_pack_kernel`)** computes Layer 0 & Layer 1 WeightNorm backward gradients, updates all `2,179` floats of the 7 MLP/variance parameters via Adam in-place, normalizes the updated weights, and writes the transposed `packed_mlp` `[1, 2113]` + updated `variance` `[1, 1]` in one `< 5 µs` launch. |
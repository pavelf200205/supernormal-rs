# SuperNormal-RS Performance Optimizations (Update 3)

## Overview

In Update 2, SuperNormal-RS already surpassed the original PyTorch (`tiny-cuda-nn` + `nerfacc`) implementation in iteration speed, but exhibited two clear hardware symptoms:
1. **Low GPU Power Utilization (~85 W out of 170 W TDP)** on an NVIDIA RTX 3060.
2. **Progressive Throughput Decay**: Iteration speed started around ~45 it/s at step 2,000 (`active_levels = 1`) and decayed almost linearly to ~12 it/s by step 30,000 (`active_levels = 14`).

This update resolves the root causes of both bottlenecks across `src/kernels.rs`, `src/sampling_gpu.rs`, `src/render.rs`, `src/field.rs`, `src/train.rs`, and `src/data.rs`.

## Summary of Bottlenecks and Fixes

| Optimization Area | Previous Bottleneck | Update 3 Implementation | CLI A/B Flag to Revert |
| :--- | :--- | :--- | :--- |
| **1. 256-Thread Corner-Parallel Hash Forward & Backward** | Only `4 * active` (`<= 56`) of `256` threads in each workgroup executed hash-grid lookups/scatters, looping sequentially over `8` corners while `200+` threads sat idle behind `sync_cube()`. | Maps `256` threads across `(point_slot in 0..4, corner_pair in 0..4, level in 0..16)`. Up to `224` threads (`87.5%` of workgroup) issue `2` corners each in parallel (`4×` higher memory-level parallelism), reducing partial pairs in shared memory. | `--legacy-corner-loop` |
| **2. Device-Resident March, Visibility & Geometry Tables** | Every step forced 3 synchronous GPU-to-CPU stalls and multi-megabyte host-to-device uploads (`marched` download, CPU `unique_pts` upload, `visibility` download, CPU `mapping` `[count, 18]` and `reverse` `[n, 2]` construction & upload). | Fuses endpoint counting into `march_kernel` and `visibility_kernel`, generates center-ray endpoints on device (`compact_and_center_endpoints_kernel`), builds `endpoints`, `descriptors`, `mapping`, and `reverse` on the GPU without atomics (`build_visible_geometry_tables_kernel`), and reads per-patch ranges from `vis_prefix` in `composite_kernel`. | `--host-sample-compaction` |
| **3. Uninitialized Output Allocations & SFU Deduplication** | `Tensor::zeros` dispatched ~10 buffer-clearing kernels per step for buffers immediately overwritten by custom ops; `field_backward_kernel` computed `exp()` twice per neuron. | Replaces write-only `Tensor::zeros` buffers with `Tensor::empty` across all custom ops and reuses a single `exp(min(scaled, 20.0))` evaluation for both sigmoid derivative and softplus activation in backward. | Always active |
| **4. Precomputed Pixel Ray Bases & Analytical Inverses** | `View::patch` normalized `18,432` rays and ran `18,432` general 3×3 matrix inversions (`basis.try_inverse()`) on the CPU every step. | Precomputes world-space ray directions and closed-form orthonormal cross-product inverses for all pixels during `Dataset::load`, turning `View::patch` into a direct memory lookup. | Always active |

## Detailed Technical Breakdown

### 1. 256-Thread Corner-Parallel Hash Kernels (`src/kernels.rs`)

In both `fused_field_kernel` and `field_backward_kernel` / `field_backward_native_kernel`, each workgroup has `256` threads (`CubeDim::new_1d(256)`) processing `4` points (`slot in 0..4`, `lane in 0..64`).
- **Why power was capped at 85 W and speed dropped with `active_levels`**:
  During the MLP phase, all `256` threads are active (`4` points × `64` hidden neurons). However, during the hash-grid phase, work was mapped by `(slot, level)`, which meant at most `4 * 14 = 56` threads (`21.8%` of the workgroup) were active while `200` threads waited at `sync_cube()`. Worse, as `active_levels` grew from `1` to `14` (every 2,000 steps), levels `5..13` each span `524,288 × 2` floats (`4 MB` per level, `36 MB` total), exceeding the GPU's L2 cache. Because each of the 56 active threads executed a sequential `for corner in 0..8` loop, DRAM latency could not be hidden by warp-level concurrency.
- **How Update 3 fixes it**:
  - **Forward (`fused_field_kernel`)**: Threads are mapped as:
    - `hash_slot = unit & 3` (`4` points)
    - `corner_pair = (unit >> 2) & 3` (`4` pairs of trilinear cube corners)
    - `level = unit >> 4` (`16` level slots, covering all `0..14` levels)
    Now `4 × 4 × active` threads (up to `224` threads at `active = 14`) issue memory loads simultaneously—putting `448` trilinear corner lookups in flight per workgroup instead of `112`. Each thread accumulates its 2 corners and writes `[feat0, feat1]` to `partials[unit * 2 .. unit * 2 + 2]`. After `sync_cube()`, threads `0..56` sum the 4 corner-pairs in shared memory and write `shared_features`.
  - **Backward (`field_backward_kernel` & `field_backward_native_kernel`)**:
    1. Threads `0..56` compute feature adjoints `(g0, g1)` from shared `dh` and `shared_weights` and stage them in `grad_staging` (`112` floats in shared memory).
    2. After `sync_cube()`, all `224` `(slot, corner_pair, level)` threads read `(g0, g1)` from `grad_staging` and scatter-add their `2` trilinear corners concurrently into `table_grad`.
  - **Transcendental SFU Deduplication**:
    Previously, `field_backward_kernel` evaluated `f32::exp(-scaled)` for the sigmoid derivative and then evaluated `f32::exp(f32::min(scaled, 20.0))` a second time on the next line for `softplus`. Using the identity $\sigma(x) = \frac{e^x}{1 + e^x}$, both expressions now share a single `exp_pos = f32::exp(f32::min(scaled, 20.0_f32))` evaluation.

### 2. Device-Resident March, Visibility, and Geometry Pipeline (`src/sampling_gpu.rs`, `src/render.rs`, `src/train.rs`)

Previously, even with GPU ray marching, GPU visibility, GPU geometry, and GPU compositing enabled, the CPU still acted as a middleman three times per step:
1. `march_device` read back `total`, ran `compact`, and downloaded all `[total, 3]` marched samples (`try_into_vec_as::<f32>()`) to the CPU.
2. `visible` looped over the marched samples on the CPU to deduplicate center-ray endpoints, uploaded `unique_pts`, `pairs`, and `ranges` back to the GPU, ran `visibility_kernel`, and blocked on downloading `[patches + 1, 1]` counts to filter `Vec<Sample>` on the CPU.
3. `prepare_geometry_device` and `composite_device` looped over the surviving `Vec<Sample>` on the CPU to build `endpoints`, `descriptors`, `mapping` (`[count, 18]`), and `reverse` (`[n, 2]`), uploading several megabytes of index tables back to the GPU every step.

In Update 3 (`march_and_filter_device` + `prepare_geometry_from_device_batch` + `composite_device_tables`):
- **Fused March & Center-Endpoint Counting (`march_kernel`)**: Each ray thread counts both its marched intervals (`count`) and its deduplicated center-ray endpoints (`ep_count`, incremented by `1` when `t_start == prev_end` and `2` otherwise), writing `[patches, 2]`. A single `counts.cumsum(0)` and a 2-float tail slice readback (`[total_marched, unique_ep_count]`) provides exact allocation sizes for both buffers at once.
- **Fused Compaction & Center-Endpoint Generation (`compact_and_center_endpoints_kernel`)**: One GPU launch compacts `marched` (`[total_marched, 3]`) and simultaneously writes `unique_pts` (`[unique_ep_count, 3]`), `pairs` (`[total_marched, 2]`), and `ranges` (`[patches, 2]`) on the device.
- **Dual-Count Visibility (`visibility_kernel`)**: Each ray thread tracks both visible sample count and retained unique endpoint count (`[patches + 1, 2]`), followed by `vis_counts.cumsum(0)`. For the default `microbatch == batch`, only a 3-float slice (`[total_visible, total_visible_endpoints, variance]`) is read back to the host.
- **Zero-Atomic GPU Geometry & Reverse-Index Builder (`build_visible_geometry_tables_kernel`)**:
  Because each endpoint is either an unshared `start` of sample $k$ (1 occurrence) or the `end` of sample $k$ optionally reused as the `start` of sample $k+1$ within the same ray (at most 2 occurrences), each ray thread deterministically populates `endpoints` (`[ep_count, 2]`), `descriptors` (`[vis_count, 5]`), `mapping` (`[vis_count, 18]`), and `reverse` (`[ep_count * 9, 2]`) directly on the GPU without atomics or host transfers.
- **Direct Device Prefix Lookup in Compositor (`composite_kernel`)**:
  `composite_kernel` reads each patch's `[begin, end)` visible sample range directly from `vis_prefix` on the GPU when `patches[pb] < 0.0`.

### 3. Eliminating Redundant `Tensor::zeros` Dispatches (`src/kernels.rs`, `src/sampling_gpu.rs`, `src/field.rs`, `src/render.rs`)

Burn's `Tensor::zeros` allocates a buffer and enqueues a GPU fill kernel. Across `fused_field`, `field_backward`, `parameter_gradients`, `composite_gpu`, `geometry`, `march`, `compact`, `occupancy`, and `visibility`, ~10 output buffers per step were being zero-filled by Burn immediately before custom CubeCL kernels overwrote every element. Switching these write-only output buffers to `Tensor::empty` eliminates those redundant dispatches and reduces memory bandwidth overhead.

### 4. Precomputed Analytical Camera Ray Bases (`src/data.rs`)

For every pixel $(x, y)$ in a view, the ray direction $\mathbf{d}$ and the 3×3 basis matrix
$$B = \begin{bmatrix} \mathbf{d}^\top \\ \mathbf{u}^\top \\ \mathbf{v}^\top \end{bmatrix}$$
depend only on the static camera intrinsics and extrinsics. Furthermore, because the camera's $\mathbf{u}$ and $\mathbf{v}$ axes (columns 0 and 1 of `rotation`) are orthonormal with $\mathbf{w} = \mathbf{u} \times \mathbf{v}$, the inverse $B^{-1}$ has the exact closed form:
$$B^{-1} = \frac{1}{\mathbf{d} \cdot \mathbf{w}} \begin{bmatrix} \mathbf{w} & \mathbf{v} \times \mathbf{d} & \mathbf{d} \times \mathbf{u} \end{bmatrix}$$
`Dataset::load` now precomputes `cached_dirs` and `cached_inverses` in parallel across all views once on startup, reducing `View::patch` during training to a fast array copy.

## Benchmarking & A/B Comparison Guide

To benchmark the full Update 3 pipeline against Update 2 from an existing checkpoint:

```powershell
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

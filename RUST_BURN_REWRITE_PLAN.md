# SuperNormal reconstruction: Rust + Burn/WGPU rewrite plan

Prepared 2026-10-19 from `pavelf200205/SuperNormal_memory_optimized`, branch `pure_pytorch_version`, commit `c4a99143ffc4374e9eff636bd23e3b6ee143b571`.

Reference checkout: `E:\SuperNormal_Rust_Burn_Framework_Rewrite\SuperNormal_memory_optimized`.

## Recommendation and scope

Build a native Rust application with Burn for the learnable SDF, automatic differentiation, and Adam; use CubeCL backend extensions for performance-critical operations on WGPU. Keep a portable CPU implementation for correctness and machines without usable GPU compute. Start with directional finite difference (DFD) training, establish parity against the tiny-cuda-nn/NerfAcc baseline, then optimize memory and kernels. Use the vendored CUDA kernels as executable semantic references, not as the implementation language for the port.

The finished reconstruction pipeline must load prepared datasets, train, resume, render normals/depth, extract and clean meshes, and evaluate geometry and normal errors without Python or CUDA. Temporary Python fixture exporters are development tools, not runtime dependencies. All data preparation, RAW conversion, camera-calibration conversion, normal estimation and SAM work is deferred by user request. Prepared EXR normals, PNG masks and NPZ cameras are the input boundary.

## Revised baseline and branch comparison

User clarification: `pure_pytorch_version` is an experimental intermediate portability step with plausible outputs, not a diligently validated reference. Treat `main` as the primary available CUDA reference and `sub2gb` as the low-memory CUDA variant. Both should be measured; neither a commit message nor visual plausibility proves parity with the original research implementation. Earlier pre-optimization commit `7979c5f` is available if differences need historical investigation.

Fetched and inspected all three branches without changing the active checkout:

| Branch / commit | Findings | Role |
|---|---|---|
| `main` / `85f48645b75dd131b50a7822a11e06ec625ac5ac` | tiny-cuda-nn encoding and patched NerfAcc; CPU f16 normal storage, boolean masks, on-demand patch geometry, SDF checkpointing and spatially chunked meshing. Commit describes fitting 4 GB; not measured here. | Primary CUDA numerical/quality reference. |
| `sub2gb` / `1787c7be30af24add8286ffa5f09ca632495994d` | Only runner and renderer differ from main: hard-coded 1024-patch batch, 65536-point SDF/occupancy chunks, checkpointed chunked SDF, explicit cleanup, CPU-first checkpoint loading and VRAM logging. Fields, datasets and configs are unchanged. Commit describes under 2 GB; not measured here. | Memory strategy reference; compare at equal logical batch size before attributing quality differences to chunking. |
| `pure_pytorch_version` / `c4a99143ffc4374e9eff636bd23e3b6ee143b571` | Replaces encoding and renderer; runner returns to main's contents, removing sub2gb's forced batch, profiler and CPU-first checkpoint loading. Still hard-codes CUDA in runner/loader. | Readable candidate implementation, checked against CUDA at each stage. |

Two concrete semantic differences require early fixtures:

* CUDA fields pass raw scene coordinates to `tcnn.Encoding`; the PyTorch grid explicitly maps `[-1,1]` to `[0,1]` and clamps. Determine tiny-cuda-nn's exact installed grid scale, offsets, rounding, hash function, table alignment and out-of-range behavior before defining the Rust encoder. Do not assume weights can be copied between these encoders without an explicit layout/coordinate mapping.
* Vendored patch-weight CUDA forward multiplies transmittance by `(1-alpha)` without a floor; its backward clamps a denominator to `1e-10`. PyTorch instead floors `(1-alpha)` in the forward cumulative product and differentiates that expression. Saturated-alpha forward/backward tests must resolve this difference. The previous plan's implication that the forward floor matches CUDA was incorrect.

Other priorities: compare actual sampled intervals and visibility, not only rendered images; verify that occupancy skipping preserves the reference step locations; compare all active bandwidths and f16/f32 behavior. Record intentional improvements separately from parity fixes.

## Intel Core i3-1315U target

The i3-1315U has Intel UHD Graphics for 13th Gen processors, 64 execution units and DirectX 12.1 support according to [Intel's processor specifications](https://www.intel.com/content/www/us/en/products/sku/232136/intel-core-i31315u-processor-10m-cache-up-to-4-50-ghz/specifications.html). Intel lists Vulkan 1.3 for this graphics family in its [API support table](https://www.intel.com/content/www/us/en/support/articles/000005524/graphics.html). [Brush](https://github.com/ArthurBrussee/brush) explicitly uses Burn and WebGPU-compatible technology and supports Intel GPUs.

Engineering conclusion: this is a suitable intended GPU-accelerated target through WGPU DX12/Vulkan, using the same class of compute stack as Brush. Make the i3-1315U a named acceptance target, not merely an optional fallback machine. GPU adapter selection must be explicit and logged; CPU fallback must never masquerade as a successful iGPU test. Actual application compatibility and performance remain unverified until a forward/backward spike and reconstruction run execute on that laptop and driver. Brush running there is useful evidence for the graphics stack, not proof that every SuperNormal kernel will work or run at the same speed. Test available RAM and single/dual-channel memory configuration because CPU and iGPU share memory bandwidth.

## Available reference environment and dataset-driven verification

Environment supplied by user: `E:\Projects\MVPS_Studio\.venv`. Read-only check successfully loaded Python 3.10.19, PyTorch 2.8.0+cu129 and detected an NVIDIA GeForce RTX 3060 with CUDA available. Module discovery found tinycudann, nerfacc, pyhocon, pyexr, mcubes, open3d and cv2. This confirms discovery, not all extension imports or kernel execution. Initial sandboxed Torch loading failed with access denied; the approved unsandboxed check succeeded. Do not reinstall or modify this shared environment unnecessarily.

Before baseline runs, verify imported module paths and that installed NerfAcc exposes the repository's custom patch operations; an editable install may point at a different checkout. Use isolated branch snapshots/processes and explicit import paths. Smoke-test both tiny-cuda-nn and NerfAcc forward/backward and record installed/build versions; keep compilation caches and outputs in designated work areas where possible.

A real prepared dataset substantially improves fidelity. Required: normal EXRs, matching masks, `cameras_sphere.npz`, the exact config/command and intended view exclusions. Useful extras: known-good checkpoint, resulting mesh, training log, and ground-truth mesh/normals. Ground truth is helpful but not required for baseline agreement; without it, agreement does not establish absolute geometric correctness.

Verification ladder:

1. Run main and sub2gb as-is to characterize existing behavior, then under explicitly matched batch sizes/configurations. Capture runtime, peak allocated/reserved CUDA memory, losses, normal maps and meshes. Audit failed evaluation/resume paths independently from training.
2. Export actual dataset camera rays, patch IDs, occupancy state, sample positions, jitter, weights and parameter state. Matching random seeds alone is insufficient across libraries/backends.
3. Compare PyTorch against CUDA one component at a time: camera/rays, hash encoding, SDF, occupancy/marching, patch geometry, alpha/weights, DFD normals, losses, gradients and one Adam update. If parameter layouts differ, build a mapping or use controlled synthetic tables/analytic SDFs to isolate the renderer.
4. Port each validated component to Rust and rerun the same fixture tests on CPU and WGPU. Use file-based fixture boundaries initially; no cross-language differentiable bridge is required. Test backward as well as forward before replacing the next component.
5. Perform short identical-input training trajectories, then full reconstructions and held-out normal comparisons. Evaluate final mesh distances/F-score against ground truth if available, otherwise report agreement with baseline meshes as agreement only.
6. Repeat on the i3-1315U, preserving the logical batch via accumulation and measuring working set and runtime. Keep original and optimized kernels selectable until regressions are resolved.

This replaces the earlier assumption that matching the pure-PyTorch branch alone establishes a faithful rewrite.

“Works everywhere” means capability-tested platforms, not every graphics chip. Initial release targets Windows DX12/Vulkan, Linux Vulkan, and macOS Metal, including supported Intel and AMD integrated GPUs and Apple integrated graphics. CPU fallback is required. Browser WebGPU and Android/iOS are later qualification targets with the same algorithms, different I/O and memory limits. Old GPUs without adequate compute support cannot be made compatible by selecting WGPU. No CUDA-speed or fixed training-time promise is justified before measurements.

## Findings from the source

| Source | Actual behavior and port implications |
|---|---|
| `models/fields.py` | Pure-PyTorch multiresolution hash grid; separate trainable table per level; dense low levels and hashed high levels; trilinear interpolation; progressive active levels. SDF MLP uses geometric initialization, weight normalization, Softplus(beta=100), optional input concatenation/skips. Variance is `exp(10 * variance)` with downstream clipping. |
| `models/renderer.py` | Implements occupancy, ray marching, padded segmented transmittance, patch/pixel rendering, DFD/FD/AD normals, checkpointed SDF queries, and tiled marching cubes directly in PyTorch. No active NerfAcc import. |
| `models/dataset_loader.py` | EXR world normals, PNG masks, NPZ camera matrices; CPU f16 normals and boolean masks; sampled patches transferred to GPU. Device is still hard-coded to CUDA. Camera decomposition uses OpenCV. |
| `exp_runner.py` | CUDA-only device initialization/selection; Adam, warmup/cosine LR, geometric step-size schedule, occupancy updates, bandwidth progression, checkpoints, visualization and metrics. |
| `config/*.conf` | Both use 14 levels, 2 features/level, hash cap 2^19, base resolution 32, scale 1.3195079107728942, 64-wide single-hidden-layer SDF, 3x3 patches, 2048 patches/update, DFD. DiLiGenT runs 5000 steps; own-object config runs 30000. |
| `models/cd_and_fscore.py` | Bidirectional mean **unsquared** nearest-neighbor distance, averaged across directions; F-score threshold 0.5 in the dataset's world units (comment assumes millimeters). |
| `data_capture_and_preprocessing/*` | DNG conversion/grouping, interactive SAM masks, Metashape XML conversion, camera-to-world normals, and NeuS2 export. Includes external-tool dependencies and hard-coded capture assumptions. |
| `utilities/utils.py` | Cropping, angular errors, RGBA conversion, and video output. |
| `third_parties/nerfacc-0.3.5/*` | Legacy vendored CUDA package, examples, docs, and tests. Preserve attribution/reference material; replace application-used semantics rather than ship an unrelated general-purpose NerfAcc clone. |
| Setup/download/run shell scripts | Replace with documented native CLI commands, dependency-free release packages where feasible, and a dataset fetch command with checksums and explicit destinations. |

README/setup instructions describe older CUDA requirements and performance; they are not an accurate specification of this branch's current implementation.

### Existing inconsistencies to resolve explicitly

These are static source findings, not runtime-confirmed failures:

* Resume uses `exp_time_dir` although it is initialized only for a fresh run. Checkpoints omit occupancy, active bandwidth, RNG and sampling state; some schedules use local `iter_i`, which restarts on resume.
* `eval_mae` references `dataset.masks_np`, while the loader now stores `masks`. `image_at` references an unset `images_lis`; pixel validation references `validate_resolution_level` rather than the configured normal-resolution field.
* Occupancy resolution and sigmoid slope are hard-coded to 128 and 80 in the renderer despite corresponding config keys.
* Endpoint reuse tests distance equality but does not also check ray/patch identity. The Rust implementation should require both identity and adjacency; record this as an intentional correctness fix.
* Empty occupancy iterations do not increment `iter_step` although the outer loop advances. Specify attempted versus successful steps and migration behavior instead of copying an ambiguous schedule.
* Patch borders, zero-sized outputs, invalid sphere intersections, nearly parallel rays, empty meshes and zero precision-plus-recall need defined handling.

Maintain a behavior-change ledger: fixture-compatible behavior versus intentional fixes. Freeze imported weights and explicit sample locations before comparing algorithms so random initialization and sampling do not obscure discrepancies.

## Architecture

Use a Cargo workspace under a new `rust/` directory in the checkout when implementation begins:

| Crate | Responsibility |
|---|---|
| `sn-core` | Validated configuration, scene/camera types, coordinate conventions, RNG, memory budgets, errors and checkpoint schema. |
| `sn-data` | EXR/PNG/NPZ ingestion, CPU tile cache, camera decomposition, patch selection and ray generation. |
| `sn-field` | Burn hash-grid parameters, SDF MLP, weight normalization, variance and analytical spatial derivatives. |
| `sn-render` | Occupancy, sampling, patch-plane intersections, alpha/transmittance, normals, depth and losses. |
| `sn-kernels` | CPU reference operations plus CubeCL/WGPU implementations, explicit backward operations and fusion registration. |
| `sn-train` | Custom training loop, gradient accumulation, schedules, profiling, cancellation and resumable checkpoints. |
| `sn-mesh` | Streaming field evaluation, marching cubes, seam welding, component filtering, PLY/OBJ and CPU BVH/KD-tree evaluation. |
| `sn-cli` | `doctor`, `convert-config`, `inspect-data`, `fetch-data`, `train`, `render`, `mesh`, `evaluate`, `convert-checkpoint`. |

Candidate non-ML dependencies: serde/TOML, clap, image/exr, ndarray-npy, nalgebra, rayon and tracing. Validate each candidate's formats, licensing, platform support and versions during implementation; these are not yet a tested dependency set. Prefer small Rust implementations for marching cubes, BVH and KD-tree integration over carrying Open3D/VTK/Embree into the runtime. Keep optional video encoding separate from reconstruction dependencies.

### Burn version decision

Target Burn **0.22.0**, then lock the exact dependency graph and Rust toolchain after a compile-and-gradient spike. Current official docs use runtime-selected `Device` and tensors without backend generics; do not mix older 0.21 examples into application code. Enable WGPU, autodiff/optimizer support, CPU Flex and extension support as needed; take the compatible CubeCL version from Burn's dependency graph. Keep fusion optional until custom operations pass forward/backward tests in both modes. The 0.22 docs.rs build failure is a documentation-host issue to investigate, not proof that the crate cannot compile.

Official references checked on 2026-10-19:

* [Burn 0.22 API and backend features](https://burn.dev/docs/burn/).
* [Burn WGPU 0.22 package and configuration](https://docs.rs/crate/burn-wgpu/0.22.0).
* [Backend extensions and fusion contract](https://burn.dev/books/burn/advanced/backend-extension/index.html).
* [WGPU limits](https://docs.rs/wgpu/latest/wgpu/struct.Limits.html) and [graphics backend tiers](https://docs.rs/wgpu/latest/wgpu/struct.Backends.html).

The extension API requires concrete backend implementations and explicit custom derivatives; registration does not generate a backward pass. Content-dependent output lengths also need careful handling under fusion. Use fixed-capacity tensors with valid-count tensors or an explicit synchronization boundary for packed results. All version/API claims must be verified against the locked build during Phase 1.

## Algorithm and kernel design

### Hash encoding and network

First establish CUDA-compatible grid resolution rounding, table layout, coordinate transform, interpolation corner order and level activation from installed tiny-cuda-nn fixtures. The pure-PyTorch choices are candidate semantics, not the authority; retain its explicit layout only as a separately labeled compatibility mode if needed. Use wrapping 32-bit integer multiply/XOR/mask for hashing; test equivalence to the signed PyTorch constants. Keep levels separately allocated and skip inactive levels. Preserve zero-filled inactive features and the iteration-zero activation ordering.

Initially implement with differentiable Burn primitives and compare imported parameter fixtures. Implement weight normalization as trainable direction/magnitude parameters with the correct reduction axis; folding it into inference weights is not equivalent for training. Match PyTorch Softplus's stable threshold behavior, initialization distributions, concatenation order, skip scaling and variance clipping.

The optimized hash backward must sum gradients from repeated indices and collisions. Do not require native floating-point atomics. Establish a bounded key/value emission plus sort/segmented-reduction implementation using portable integer operations, or prove the selected Burn scatter implementation works on every baseline adapter. Benchmark optional integer-CAS float accumulation separately; avoid making its contention or numerical ordering the only correctness path. Account for reduction scratch in the memory budget.

### Sampling and occupancy

Implement the branch's uniform step lattice with occupancy midpoint tests first. Preserve detached sample selection, stratification, occupancy warmup (first 256 attempted steps), sampled-cell updates thereafter, max-decay update and `min(mean(occs), threshold)` binarization. Freeze sampled cell indices/jitter in parity tests, including duplicate indices.

Represent ray/patch samples with offsets/counts and packed intervals. Count, prefix-scan and fill in bounded buffers; overflow splits work and retries without changing the random samples. Avoid a global `[rays, maximum_steps, patch_pixels, ...]` allocation. Decode occupancy coordinates from linear indices instead of storing a persistent int64 XYZ grid. Start with u32 occupancy flags; consider bit packing later.

DDA empty-cell skipping is a later optimization: retain the original step lattice and interval acceptance so it does not silently change sample locations. Handle no-hit rays and finite ranges explicitly. Keep center-ray visibility filtering separate from nine-pixel patch compositing.

### Patch rendering and gradients

Preserve center-ray marching, the camera-plane distance ratio for all nine rays, and endpoint reuse only within contiguous intervals of the same patch. Match alpha:

`alpha = clamp((sigmoid(s0*inv_s) - sigmoid(s1*inv_s) + 1e-5) / (sigmoid(s0*inv_s) + 1e-5), 0, 1)`.

Compute exclusive segmented transmittance and weight `T*alpha` with the CUDA-reference semantics established by fixtures; specifically test the forward-product versus backward-denominator clamp difference described above. Start with one worker per patch pixel walking its contiguous segment; optimize with hierarchical scans only after profiling. Use a reverse recurrence for backward with clamp semantics, avoiding division by zero at opaque samples. Reductions stay within each segment rather than globally scattering patch outputs.

DFD reconstructs the gradient from ray-direction differences, centered patch-interior differences, one-sided patch-border differences and the inverse directional basis. Preserve the actual distances and the unnormalized weighted normal used by the training loss. Normalize only where the reference evaluation does.

FD remains a supported mode with epsilon 1e-3 and six shifted evaluations. Stream those evaluations rather than materializing all six position arrays simultaneously.

AD training differentiates through spatial gradients and therefore needs mixed second derivatives. Do not assume ordinary Burn backward provides PyTorch `create_graph=True`. Phase 1 probes this. If unavailable, implement an analytical spatial Jacobian through trilinear encoding, normalized-weight linear layers and Softplus using differentiable Burn expressions; parameter backward then computes the mixed derivatives. Add the divergence/Laplacian API with explicit second spatial derivatives if retained as public functionality. Test cell-boundary conventions against PyTorch; never silently substitute FD for AD. The initial DFD milestone is not full feature completion.

### Losses, training and checkpoints

Preserve normal L1/L2 sums divided by foreground pixel count plus epsilon; eikonal mean across all sampled patch points; clipped BCE averaged across all patch pixels; Adam defaults; warmup/cosine LR and logarithmic step-size decay. Do not replace Adam with AdamW.

Use whole-patch microbatches. For exact logical-batch accumulation, normal loss uses the global mask denominator, BCE the global pixel count, and eikonal the global valid sample count. Determine sample counts with detached sampling before backward, retaining small metadata or regenerating with fixed seeds. Average-of-microbatch-means is incorrect when sample counts differ. Perform one optimizer update per logical batch and freeze occupancy/weights until all its microbatches finish.

Checkpoint model, Adam moments and counters, active levels, attempted/successful steps, occupancy values/flags, RNG state, schedules, configuration, coordinate transforms and format version. Write atomically and test interrupted/resumed equivalence. Provide model-only legacy `.pth` conversion with explicit weight-normalization and tensor-layout mapping; historical full-state resume cannot be exact because the source did not save all required state.

## Memory and portability budget

The pure-PyTorch layout under the default configuration has 11,878,328 hash parameters (recalculate for the validated tiny-cuda-nn-compatible layout): about **45.31 MiB f32**; parameters + gradients + two Adam moments total **181.25 MiB**, excluding temporary copies and activations. This is calculated from the source's per-level sizes, not a measured peak.

At 2048 patches, 9 pixels and 256 intervals, one 64-channel f32 activation alone is **1.125 GiB**. A unit-sphere crossing at a 0.001 step can approach 2000 intervals before occupancy filtering. The reference's 262144-point forward chunks do not by themselves bound all retained backward state or concatenated outputs.

Required controls:

* Explicit adapter/API selection and a `doctor` report of feature/limit support, shader compilation and a small forward/backward result.
* f32 arithmetic and u32/i32 indexing as baseline. No requirement for f16, f64, int64, subgroups, tensor cores, native float atomics or discrete VRAM.
* Configurable working-set budget, conservative allocation estimates and bounded buffer sizes checked against device limits. WGPU limits are not a reliable report of free system/GPU memory.
* Start an integrated-GPU profile around 32–128 patches per microbatch and 8K–32K field points per query; these are tuning seeds, not validated defaults. Preserve the logical batch through accumulation.
* Capacity exhaustion reduces microbatch work and retries deterministically. Device loss reloads the last complete checkpoint; do not assume a failed allocation leaves the device usable.
* CPU cache for normals/masks with a bounded decoded-view/tile budget, transferring only requested patches. Unified memory still competes with the OS and must be bounded.
* Recompute field activations where necessary with explicit backward support; release each microbatch graph promptly. Profile fusion caches, scratch, upload/readback and optimizer temporaries too.
* Smaller hash tables or lower reconstruction resolution are explicitly labeled quality presets, not automatic invisible changes.

## Meshes and metrics

Evaluate the SDF in small query batches inside 32–64-voxel bricks with a one-voxel overlap. Emit marching-cubes faces incrementally, key boundary vertices by global grid edges, and weld seams before connected-component filtering. Preserve the reference query sign (`-SDF`), triangle orientation and scene-to-world transform. A 1024^3 scalar field alone is 4 GiB f32, so never allocate the complete volume. Budget final mesh storage too; use an on-disk intermediate for very large meshes.

Port largest-triangle-component selection, masked camera-ray first-hit sampling with a CPU BVH, normal-angle statistics and the exact unsquared Chamfer/F-score definitions. Preserve held-out views and world units. Provide explicit empty-data outputs instead of NaNs/division by zero. Reproduce EXR/PNG normal/depth/error outputs, CSV metrics, image cropping and optional video/report generation.

Data preparation and SAM are deferred; no preprocessing milestone is required for this reconstruction release.

## Delivery phases and gates

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0. Freeze behavior | Source inventory, source-bug ledger, config mapping, minimal synthetic scene and fixture exporter; obtain one prepared reconstruction dataset and establish main/sub2gb CUDA baselines. | Fixed weights, samples, intermediate tensors, gradients and one Adam update exported; minimally repaired Python harness changes documented. |
| 1. Feasibility | Locked Burn build; CPU and WGPU tiny training; hash gather/scatter, segmented recurrence backward, device limits, AD strategy and checkpoint spike. | Finite matching gradients on the i3-1315U integrated adapter; working CPU fallback; explicit decision for every unsupported primitive. |
| 2. Data and field | Native formats/cameras, both configs, hash grid, MLP, variance, weight norm and schedules. | Reprojection, dense/hash boundary/collision, SDF and parameter-gradient fixture tests pass. |
| 3. Complete DFD path | Occupancy, detached sampling, patch geometry, rendering, losses, Adam and exact accumulation. | One-step parity, synthetic sphere convergence, short real-scene run and correct empty-ray behavior. |
| 4. Integrated-GPU performance | Bounded packed kernels, backward/recompute, adapter profiles and instrumentation. | Completes sustained training within declared budget on real integrated hardware; no mandatory CUDA or optional GPU features. |
| 5. Reconstruction outputs | Checkpoint/resume, DFD/FD/AD modes, pixel/patch validation, mesh extraction/cleanup, metrics and legacy import. | Resume equivalence; seam tests; full benchmark runs and documented quality differences. |
| 6. Reconstruction packaging | Portable CLI, prepared-data examples, checkpoint conversion, metrics/reports and release packaging. | Reconstruction workflow matrix complete; no data preparation dependencies. |
| 7. Qualification | Windows/Linux/macOS packages and tests on Intel, AMD, Apple, NVIDIA plus CPU. Then browser/mobile qualification. | Measured support matrix, reproducible examples and no unqualified “all GPUs” claim. |

Dependencies: Phase 1 is the go/no-go gate; Phase 3 depends on field/data parity; kernel optimization follows reference correctness; full completion requires Phase 6, not just successful DFD training. Planning estimate for one experienced Rust/GPU engineer: roughly 10–16 engineering weeks for reconstruction and native desktop qualification; preprocessing/SAM are excluded. Browser/mobile are later scope. These are scope estimates, not commitments; revise after the first feasibility spike and access to target hardware/data.

## Verification and definition of done

Use the Python branches for baseline runs, component fixtures and regression comparisons during development; the shipped reconstruction application is Rust-only. Unit/gradient tests run in Rust, including collision-heavy hash tables, inactive levels, alpha near 0/1, uneven and empty segments, invalid intersections, patch borders, all-empty masks and chunk seams. Check finite differences away from interpolation/clamp discontinuities. Provisional f32 gates are `atol=1e-5, rtol=1e-4` for small forward fixtures and `atol=1e-4, rtol=1e-3` for gradients; revise only with recorded conditioning evidence, not blanket relaxed tolerances.

For long runs compare multiple seeds with matched samples/settings and report normal MAE, Chamfer, F-score, wall time, samples/second and peak CPU/GPU working sets. Establish quality acceptance bands from baseline run variance before optimizing. Bitwise equivalence across GPU reductions is not expected. Performance comparisons must identify adapter, driver, API, power mode and resolution.

Required hardware runs include the i3-1315U on Windows (DX12 and Vulkan where available), AMD integrated graphics, Linux integrated graphics, Apple Silicon Metal, NVIDIA WGPU and CPU-only execution. Build-only CI is not GPU validation. Add browser/mobile only after their own memory, device-loss and I/O tests pass.

The complete rewrite is done when all in-scope reconstruction workflows have Rust equivalents; core training/render/mesh/evaluation have no Python/CUDA runtime requirement; default DFD and alternative FD/AD semantics are accounted for; bounded-memory runs succeed on qualified integrated GPUs; checkpoints resume reproducibly; and quality/performance/support limitations are published with measurements.

## Work completed for this planning task

Cloned and verified pure_pytorch_version, fetched and compared main/sub2gb and their history, inspected reconstruction sources and CUDA patch kernels, calculated the PyTorch hash layout memory, checked official Burn/WGPU/Intel and Brush sources, and verified Torch/CUDA availability in the supplied environment. No dataset/model downloads, training, Rust prototype or hardware benchmark were performed. The Python checkout remains unchanged; this plan is stored beside it.

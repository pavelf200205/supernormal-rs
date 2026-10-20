# Validation log — 2026-10-19

Machine tested: NVIDIA RTX 3060, Windows, Burn 0.22.0 / WGPU. Intel i3-1315U hardware has not been tested. No native 30,000-step reconstruction has been launched.

## Numerical checks

- Imported pure-PyTorch 30k checkpoint: SDF maximum absolute error `5.96e-7` on 128 sampled positions. Field parameter gradients matched, with largest observed absolute discrepancy about `1.49e-7`.
- Independent packed compositor fixture: normal/mask output errors below `1.2e-7`; SDF and variance adjoints within approximately `1.2e-4`. Numerical finite differences passed.
- Dataset check: camera origins and rays within `2.39e-7`; selected EXR normal values matched exactly after FP16 quantization.
- Actual repository renderer, fixed sample positions: combined loss error `1.31e-6` before endpoint sharing. Chunked field backward and endpoint sharing also passed the continuation optimizer comparison.
- Adam arithmetic is tested independently using identical gradients and saved checkpoint moments. This distinguishes optimizer errors from finite-difference/reduction differences between backends. The complete update comparison records maximum and RMS parameter discrepancies; tiny gradients can amplify into isolated update differences (observed maximum about `1.22e-4` in a hash table, RMS about `1.38e-7`). These are not bit-identical trajectories.
- Fresh-Adam testing exposed a spurious output-bias gradient of about `3e-6`, caused by reduction cancellation, which Adam amplified into a roughly `5e-4` first update. The renderer now computes the bias derivative explicitly from opacity: DFD derivatives are exactly invariant under a constant SDF shift. Final retest results are recorded below when available.
- DirectX 12 and Vulkan device/autodiff smoke tests passed. The imported field and gradient test also passed through explicit DX12 selection.

## Mesh

At resolution 96, native and Python extraction of the same imported checkpoint both produced 4,872 vertices and 9,740 triangles after largest-component filtering. Symmetric nearest-vertex distance: maximum `1.8883e-5`, mean `1.6837e-7`, in normalized object coordinates. This verifies checkpoint inference and extraction; it is not evidence of a completed native training run.

## Training and memory experiments

- Two fresh initialization smoke steps completed, including checkpoint writing.
- Two full-batch continuation steps (2,048 patches; schedule extended to 30,002 solely to permit this test) completed. The imported checkpoint contains Adam history, but its occupancy and RNG are rebuilt, so this is not exact continuation of the original Python run.
- With microbatch 64 and query 32,768, the version before endpoint reuse took 64.16 and 47.09 seconds; endpoint reuse took 37.35 and 31.20 seconds. Corresponding losses and sample counts agreed to displayed precision. These are short diagnostic measurements, with other compilation activity during the session, not controlled benchmark results.
- Endpoint reuse alone was insufficient for low memory: the allocator reported 142.6 MB live and 5.65 GB reserved after the run. Query 131,072 / microbatch 256 hit WGPU OOM. That configuration is not recommended.
- A 512 MiB unused-cache cleanup threshold has been added at query/backward boundaries. This is a cache target, not a hard VRAM cap. Its measured results and default-size tests must be used before making low-memory claims.

## Distribution and remaining work

PE import inspection found only Windows system DLLs; no Python, CUDA, LibTorch or Visual C++ runtime DLL import. Graphics drivers and operating-system libraries remain required. Other platforms require their own builds and tests.

Remaining: sustained native convergence testing; performance work on hash-grid backward and CPU/GPU transfers; integrated-GPU validation; and broader configuration support. Current reconstruction covers the supplied DFD/L2 configuration. Normal-image validation, preprocessing and original tiny-cuda-nn checkpoint layouts are outside the implemented scope.

## Progress on 2026-10-20

The 500-step scratch run produced a recognizable horse mesh, then two optimization commits landed: `bbf7894` cached features and parallelized host scatter; `5890b3a` moved hash interpolation and gradient scatter to CubeCL/WGPU. The fused kernels passed the development field and gradient comparisons. The original description above records the earlier implementation, so its performance and memory figures are historical.

A full 30,000-step run was launched with 2,048 patches, microbatch 64, query 65,536, and checkpoint/mesh extraction every 200 steps at mesh resolution 128. At inspection, its log had advanced past step 950 and produced meshes at 200, 400, 600, and 800. Steps 801–900 averaged about 1.64 seconds and 200,216 ray samples per iteration, with an active hash-grid level count of **one**. The code adds a level every 2,000 iterations; this timing is not a measured cost for all 14 levels. This run was subsequently cancelled.

## Checkpoint continuation benchmarks — 2026-10-20

The Gemini training run mentioned above was cancelled. Separate short runs imported the 5k, 10k, and 30k pure-PyTorch checkpoints on the RTX 3060. All used 2,048 patches, microbatch 64, query 65,536, and no periodic mesh extraction. The first step was excluded because it rebuilds occupancy and compiles shaders; imported parameter and available Adam state were restored, but occupancy and RNG were fresh. These are therefore performance samples, not exact trajectory continuations. The 30k schedule was extended eight steps solely for timing.

| Checkpoint | Active levels | Timed steps | Median step | Median ray samples | Range |
|---|---:|---:|---:|---:|---:|
| 5k | 3 | 3 | 0.449 s | 14,947 | 0.338–0.486 s |
| 10k | 6 | 9 | 0.764 s | 19,901 | 0.447–3.725 s |
| 30k | 14 | 7 | 2.181 s | 86,937 | 1.100–5.753 s |

The 10k repeat and 30k runs have intermittent multi-second stalls in evaluation, sampling, or backward. The timing paths are `experiments/benchmark_resume_{5k,10k_repeat,30k}/metrics.csv`; the 10k first pass is separate. The 30k Burn allocator reported a peak reservation of 6.11 GB before cleanup, not total process VRAM. The old 1.3–1.6 s/step observation was from an early one-level run and cannot represent late training.

Additional 5k and 10k field checks exposed wrong Burn autodiff gradients for the output layer's weight-normalization parameters despite forward values matching PyTorch. Explicit chain-rule derivatives for both `lin1.weight_g` and `lin1.weight_v` now match the exported PyTorch gradients at 5k, 10k, and 30k, with maximum absolute errors below `6e-8` for the output weight vector.

An experimental packed hash-grid implementation combines all active levels into one WGPU forward dispatch and one hash-gradient dispatch per chunk. At 30k, seven timed continuation steps had median `1.227 s` versus `2.181 s` in the earlier per-level build, but mean `2.948 s` versus `3.009 s` because multi-second stalls remained. Fast individual steps improved from `1.100 s` to `0.971 s`. This is a small noisy sample, not a reliable 2× throughput gain. The packed implementation still keeps ray traversal, visibility, compositing, and MLP backpropagation outside these hash kernels; it is not a fully fused training step.

The final build explicitly compared packed GPU forward and chunked packed gradients with the 5k, 10k, and 30k PyTorch field references; all passed. In a repeated eight-step 30k sample, normal steps took `1.00–1.27 s`, while intermittent stalls reached `4.87–5.57 s`. The new timer split showed ray traversal itself took only `0.01–0.02 s`; visibility SDF evaluation took `0.18–0.22 s` on normal steps, and training field evaluation took `0.62–0.99 s`. Normal backward was `0.05–0.09 s`. This evidence makes field evaluation, GPU readbacks, and stall elimination higher priorities than porting the ray loop or fusing backward at any cost. The repeat log is `experiments/benchmark_resume_30k_packed_timed.log`.

The 5k packed forward/backward reference check also passed through explicit DX12 selection on the RTX 3060, although the first DX12 verification took several minutes. No integrated adapter is present on this host (`--integrated doctor` returns `AdapterUnavailable`), so Intel UHD correctness, memory use, and speed remain unverified.

Visibility and occupancy inference were then switched from CPU-built hash indices to the already-validated packed GPU hash kernel, and normalized inference weights were cached for each optimizer step. In a final eight-step 30k continuation, visibility fell from about `0.18–0.22 s` to `0.04 s` on normal steps; the fastest complete step fell from `1.00 s` to `0.77 s`. Seven timed steps had median `1.076 s` and mean `2.867 s`; the mean remains distorted by multi-second stalls. Peak Burn allocator reservation fell from `6.03 GB` to `3.75 GB`. This is measured on the RTX 3060 and does not establish performance on Intel integrated graphics. See `experiments/benchmark_resume_30k_gpu_infer.log`.

A longer 25-step 30k continuation found that the final 13 steps settled at median `0.943 s`, mean `0.947 s`, and range `0.763–1.113 s`. Earlier steps still had occasional 3–4 s spikes attributed in the log to whichever GPU-synchronized stage was active; their cause is not yet isolated. This stable late sample is about 1 iteration/s, still far from the original CUDA baseline near 25 iterations/s reported by the user. See `experiments/benchmark_resume_30k_gpu_infer_long.log`.

For a like-for-like final-build comparison, each pure-PyTorch checkpoint was run for 25 additional steps, then the last 13 timed steps were summarized. The first-step occupancy rebuild/shader initialization is excluded. These are still model-only imports, not exact RNG/occupancy resumes.

| Checkpoint | Active hash levels | Median ray samples | Median step | Mean step | Range |
|---|---:|---:|---:|---:|---:|
| 5k | 3 | 15,376 | 0.290 s | 0.584 s | 0.257–2.053 s |
| 10k | 6 | 19,966 | 0.304 s | 0.418 s | 0.280–1.543 s |
| 30k | 14 | 86,725 | 0.943 s | 0.947 s | 0.763–1.113 s |

Logs: `experiments/benchmark_resume_5k_final_long.log`, `experiments/benchmark_resume_10k_final_long.log`, and `experiments/benchmark_resume_30k_gpu_infer_long.log`.

## Cached MLP backward and inference graph lifetime

Commit `71aa60e` preserves the preceding validated implementation. The next change caches hidden preactivations and activations, then computes softplus, both weight-normalization derivatives, biases, and feature adjoints explicitly. It reuses normalized weights and avoids repeating the forward MLP during backward. Matrix products still use Burn; this is not a fully fused MLP kernel.

Synchronized profiling exposed an attribution error in earlier timings: backward work queued on the GPU was charged to the next forward readback. One diagnostic update measured roughly 0.12 s forward and 0.62 s backward. CPU point generation/deduplication is now timed separately. Those diagnostic runs failed after two updates and are not valid throughput benchmarks.

The first cached-backward candidate passed the 5k, 10k, and 30k field/gradient fixtures but leaked live GPU allocations during training. Burn 0.22's `detach()` preserves `requires_grad`, unlike the PyTorch behavior assumed by the earlier code. Removing automatic backward left unintended inference/optimizer graphs unconsumed; live pool allocation exceeded 7 GB. Microbatch cleanup alone did not fix it. Inference, analytical gradient calculations, and Adam now use `without_autodiff()`, and parameter leaves explicitly regain autodiff after each update. The field verification also asserts that cached activations do not require gradients.

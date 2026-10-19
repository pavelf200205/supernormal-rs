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

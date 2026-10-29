# Kopanas optimization audit — 2026-10-29

Scope: source inspection of the current working tree, including existing
uncommitted updates. All eight supplied article/diff pages were downloaded and
reviewed. No builds, tests, GPU captures, or benchmarks were run. Presence of
source mechanisms does not establish numerical parity or machine-code behavior.

Conclusion: the code implements several related principles, but it does not
implement every technique, and some Update 5 documentation describes experiments
that are absent from the current kernels.

| Article stage | Current SuperNormal status |
| --- | --- |
| [s00 baseline](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s00_baseline.html) | A reference 3DGS wrapper with timing instrumentation, not a SuperNormal optimization to port. Our stage timers are separate and do not demonstrate equivalent GPU instrumentation. |
| [s00 to s01](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s00_to_s01.html) | Partial conceptual overlap: occupancy/visibility pruning and compact device tables. Gaussian ellipse binning, tile masks, SH loads, and two-stage depth/tile sorting have no direct counterpart here. Warp aggregation of colliding hash gradients is absent. |
| [s01 to s02](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s01_to_s02.html) | The article's halving reduce-scatter and two-splat half-warp folding are absent. Compact output-weight gradient sums and shared-memory feature-gradient reductions are different reductions, not proof of integration of this technique. |
| [s02 to s03](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s02_to_s03.html) | Analogous memory-traffic improvements exist: write-only output allocation, reusable gradient storage, gradient clearing fused with Adam, and limited activation caching. Four-window SH streaming and ballot-based zeroing of untouched SH rows are not implemented; there are no SH tensors in this reconstruction pipeline. |
| [s03 to s04](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s03_to_s04.html) | Partial adaptation: transmittance truncation before full field evaluation and zero-adjoint backward skipping. No exact per-pixel contributor masks, packed Gaussian contributor IDs, or splat-touched flags. |
| [s04 to s05](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s04_to_s05.html) | Partial adaptation: compositor reverse-pass derivatives use register selections and unconditional scratch stores. No explicit approximate reciprocal/PTX path or rounding-pinned Gaussian quadratic evaluation. Source-level selections alone do not prove branchless generated machine code. |
| [s05 to s06](https://grgkopanas.github.io/3dgs_cuda_blog/diffs/s05_to_s06.html) | The current compositor already has no workgroup barriers, but does not implement two vertical pixels per thread, independent warp staging rings, folded log-opacity Gaussian evaluation, vertical derivative moments, or sign-bit transmittance termination. Most of those formulas are Gaussian-specific. |

## Findings in the current source

1. **Documented voxel-leader aggregation is missing.**
   `src/kernels.rs::field_backward_kernel` and
   `field_backward_native_kernel` compute `weight * g0/g1` per point/corner and
   issue `atomic_add_f32` or `fetch_add` directly. Both the default corner-parallel
   path and the legacy corner-loop path lack same-address aggregation. There are
   no `base_staging`, `w_staging`, leader selection, or `wg_sum` mechanisms claimed
   in the Update 5 notes. The compact `dw1` reduction targets MLP output weights;
   it does not reduce hash-table atomic traffic.

2. **The documented cooperative compositor was superseded.**
   `src/kernels.rs::composite_kernel` assigns one invocation per pixel ray,
   gathers SDF neighbors directly, and uses no shared SDF tile or `sync_cube`.
   `CompositeOp` chooses `CubeDim::new`, rather than forcing 32 threads.
   Compact endpoint indexing and forward caching of sigmoid/SDF values remain.
   Removing workgroup barriers serves a related purpose, but the claimed 3x
   cooperative SDF load reduction is not present in this version.

3. **Gradient clearing is genuinely fused with Adam.**
   `packed_adam_kernel` reads each active gradient and resets it to zero as part
   of the parameter update. `Field::apply_gradients_packed` retains that cleared
   buffer for the next iteration. The initial allocation is zeroed. Sparse
   updates must not simply skip untouched entries: Adam moments can remain
   nonzero and still require decay and parameter updates.

4. **Pruning must preserve SuperNormal's losses.**
   `visibility_kernel` truncates the center-ray prefix at transmittance below
   `1e-8`. Backward skips expensive work for zero point adjoints and exits a
   four-point group if all four are zero. This is not exact nine-pixel contributor
   pruning. A sample with zero alpha can still contribute to the eikonal loss or
   neighboring finite-difference derivatives. Dropping it merely because it did
   not blend would change the objective.

5. **Hardware claims need measurement.**
   Shared-memory staging, corner parallelism, wider parameter tiles and compact
   gradient partials are visible in source. Exact resident blocks, register
   pressure, cache hit rates, and elimination of specific reconvergence
   instructions require profiling/generated-code inspection. A100 launch bounds,
   fixed 32-lane shuffles, PTX intrinsics, and the article's performance ratios
   cannot be assumed to carry over to WGPU on RTX or integrated AMD GPUs.

## Recommended next work

The most directly relevant missing adaptation is combining hash-gradient updates
that target the same table address within a local group. Prototype it behind an
independent comparison flag in both native-float and CAS paths, preserve all
contributions, and compare early/late checkpoints. The current four neighboring
points provide a candidate group, but equal addresses are not guaranteed at fine
levels and extra staging/barriers can outweigh atomic savings.

Do not copy the article's Gaussian thresholds or formulas into SDF compositing.
Profile first-layer reduction, hash scattering, compositor traffic, and Adam
separately before choosing another mapping or precision change. Keep numerical
and reconstruction validation separate from source-presence checks.

Reference overview: https://grgkopanas.github.io/3dgs_cuda_blog/

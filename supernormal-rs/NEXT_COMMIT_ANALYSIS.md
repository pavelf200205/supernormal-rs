# Next-commit performance analysis

Source baseline: `0653795` (George Kopanas upgrades part 2).
Read-only runtime inspection on 2026-10-29: no builds, training, tests, or GPU
captures. Only this analysis document was added. Source observations below are
confirmed; speedups, compiler behavior and hardware bottlenecks are hypotheses.
The latest power-interrupted run is not a completed reconstruction benchmark.

## First: correctness and trustworthy measurements

**Microbatch dispatch guard needs correction.** In
`sampling_gpu.rs::build_visible_geometry_tables_kernel`, the local ray is bounded
by the remaining global ray count, not the requested local `patch_count`.
`BuildVisibleGeometryTablesOp` knows that count but passes only `patch_start`,
`s_base`, and `ep_base` to the shader. If a non-final microbatch dispatch rounds
up, extra lanes can process the next microbatch's rays and write beyond the
logical local tables. Rounded capacity can hide an invalid write; it does not
make it safe. Pass the local count to the kernel and bound `local_ray` explicitly.
Aligned default batches can avoid triggering this, so it is not evidence that
the reported successful reconstructions were corrupted.

**Current stage times are not isolated kernel times.** `--profile-stages` places
host timers around deferred submissions, with synchronization after backward
and Adam and readbacks in sampling. A stage can inherit work queued earlier.
Use GPU timestamps or profiler ranges, plus full wall-clock timing, to attribute
field forward, hash backward, parameter reductions, compositing, Adam,
occupancy and readbacks separately. Do not infer a memory-bound kernel from
power alone or from these CPU timer values.

Collect valid center-ray endpoints, valid nine-pixel field points, recomputed
activation points, active levels, conditional-copy bytes, allocation/cleanup
counts, and logical samples. Samples/second alone misses endpoint reuse and
recomputation. Keep detailed diagnostics optional, sampled, and out of normal
training; counters in every atomic hot path would perturb the measurement.

## Ranked implementation opportunities

| Priority | Source observation | Candidate change | Main tradeoff |
| --- | --- | --- | --- |
| 1 | Packed Adam loops over every active level boundary for every parameter scalar (`kernels.rs::packed_adam_kernel`). | Dispatch by level using a second grid dimension, derive each level's start/end directly, and retain one fused value/moment/clear pass. | Unequal level sizes cause some masked lanes; no extra persistent buffers needed. |
| 1 | Every fused field workgroup stages 2,113 MLP floats for just four points. | Evaluate more nearby points per workgroup or reuse weights across several four-point tiles. | More live state/shared memory can reduce concurrency. Keep a bounded-memory fallback. |
| 1 | Six workgroup-wide reduction barriers are used for the scalar output of each four-point forward group. | Capability-gated subgroup reductions, with shared-memory reduction only between subgroups when necessary. | Must support variable subgroup widths and retain the portable fallback; never assume 32 lanes or uniform masked participation. |
| 1 | Part-2 aggregation always pays two additional barriers and leader comparisons; it matches four points only at the same corner position. | Measure matches by level; compare selective aggregation and broader same-address matching across corner positions. | Fine-level matches may be rare. Broader reduction can consume more instructions/registers than it saves. |
| 1 | Visibility recomputes both endpoint sigmoids for each retained interval. | Reuse the previous end sigmoid when the current start has exactly the same endpoint ID. | Divergence may erase some savings. Preserve the existing clamp, threshold and evaluation values. |
| 2 | The compositor likewise evaluates sigmoid at both interval endpoints. | Reuse cached end values for a matching next start; hoist genuinely invariant denominator/basis loads. | Must match actual endpoint IDs, not assume floating-point contiguity. Additional live registers may hurt. |
| 2 | Full `dh[N,64]`, features `[N,31]`, and output-weight partials cross kernel boundaries. | Fuse bounded-tile parameter-gradient generation with backward, or generate all small MLP parameter partials in one coordinated pass. | A naive fully fused outer product can explode register use, partial storage or atomics. Preserve the small live-cache cap. |
| 2 | Nine pixel threads per patch share a ray range, but neighboring patches have unequal lengths. | Group patches by length, or compare alternative pixel/ray mappings while retaining sample order within each ray. | Extra compaction/order restoration and changed sum order; sorting overhead must be included. |
| 2 | Sampler has two mandatory scalar count readbacks in a normal device iteration; occupancy adds another every eight steps after warmup. | Reuse bounded capacity buffers with device valid counts and indirect/count-guarded dispatch where supported. | Requires explicit overflow handling and a memory ceiling. Worst-case over-allocation would lose the VRAM achievement. |
| 2 | March, compaction, visibility, and visible-geometry table construction traverse ray-length loops with one invocation per ray. | Parallelize pointwise preparation across compact samples after counts are known; retain only sequential transmittance work per ray. | Prefix/dependency work can dominate short rays; stable endpoint sharing must be preserved. |
| 2 | `trim_memory` synchronizes before even checking its reservation threshold; training calls it every 200 steps. | Sample pool usage without an unconditional wait where the API allows; clean only after sustained excess or at established boundaries. | Retain warmup/mesh cleanup and a memory budget. Removing cleanup indiscriminately risks higher VRAM. |
| 2 | Backward and Adam contain `can_mut` fallbacks that copy large buffers. | Add copy counters; fix ownership/lifetimes if normal training actually triggers copies. | Outstanding views must remain valid. Source presence alone is not proof copies are happening. |
| 3 | Small MLP Adam, weight normalization and packing remain separate Burn expressions. | Consolidate small-parameter optimizer/normalization/packing operations after profiling dispatch overhead. | Burn fusion may already eliminate much of this; manual fusion needs optimizer parity checks. |
| 3 | `lin1.bias` is reduced through field backward and later overwritten by compositor statistics in training. | Allow training to skip the redundant bias reduction, while retaining it for field-only/reference consumers. | Backend laziness may already drop unused work; first confirm it executes. |
| 3 | Geometry writes 27 coefficients/sample; descriptors store five floats/sample. | Fuse compatible preparation stages or compute exact coefficients near their consumer; compact unused descriptor fields. | Recomputing sqrt/divisions can be slower. Analytical spacing changes rounding and needs gradient validation. |
| 3 | Initial per-level embeddings/moments coexist with packed storage; checkpoint sync creates per-level slices. | Serialize directly from packed state, and avoid keeping unnecessary duplicate/alias state during training. | Preserve checkpoint format and reference tools; whether slices alias or copy depends on backend implementation. |
| 3 | 1024 mesh extraction evaluates a full dense volume, reads slabs back, and builds triangles serially on CPU. | Profile separately; compare parallel CPU cell classification or bounded sparse-block extraction. | Sparse skipping requires valid conservative bounds: an SDF network is not guaranteed to be 1-Lipschitz. Preserve topology and bounded memory. |

## Why the weight-staging and Adam observations matter

Staging 2,113 FP32 weights means 8,452 logical bytes per four-point workgroup,
about 2,113 bytes per point per forward launch. At one million points that is
about 2.1 GB of logical load requests, plus shared stores, although the tiny
weight tensor is cache-friendly and this is NOT a DRAM-traffic estimate. The
opportunity is reduced staging/barrier/instruction cost, not a claim that all
these requests miss cache. Larger point tiles and subgroup MLP mappings should
be independent experiments, not one massive rewrite.

Packed Adam touches value, both moments and gradient for every active scalar.
Four read/write streams imply approximately 32 logical bytes/scalar, before
compiler/cache effects and correction metadata. At all 14 levels the tables
contain roughly 12 million feature scalars: approximately 380 MB of useful
logical streaming traffic per update. Eliminating the per-scalar level search
is feasible without sparse-Adam semantics. Updating only currently touched
entries is not equivalent: old moments can still cause parameter updates on
zero-gradient entries.

## Hidden scope limits of part 2

Aggregation merges up to four matching keys within each level/corner group.
Adjacent cells may reach the same vertex through different corners; those are
not merged. Cross-workgroup collisions remain. It is therefore not a universal
atomic-traffic elimination mechanism. The staged keys compare table addresses,
so combining accidental hash collisions is mathematically valid as well.

Component-major scratch is contiguous for a sample's nine pixels, not all 32 or
64 lanes. A hardware subgroup can contain several unrelated patch ranges and
split execution as they finish at different times. A better mapping must improve
that divergence without bringing back multi-patch barriers. Evaluate this
separately from changing the scratch representation.

## Other experiments to defer until profiling justifies them

- Pack the two adjacent hash features into explicitly vectorized loads/stores if
  generated code does not already do so; verify alignment and backend support.
- Avoid repeated coordinate normalization/base/fraction work across four
  corner-pair threads by staging compact point/level data; compare added barriers
  and shared traffic against the arithmetic saved.
- Specialize early versus late level-count mappings so early training does not
  use a fixed 256-thread hash layout with very few participating levels.
- Improve launch ordering or reorder evaluation points spatially only with a
  reversible permutation of SDFs and adjoints. Preserve ray order in compositing.
- Fuse occupancy apply/threshold preparation or reduce its allocation readback,
  but keep the current selection schedule and RNG behavior when claiming parity.
- Make the three-chunk activation cache configurable under an explicit memory
  budget. Extra caching avoids recomputation but is not VRAM-neutral; it is not
  the preferred default for integrated GPUs.
- CPU patch coordinates, normals/masks and sphere intersections still require
  host preparation and small uploads. Move them only if they become material;
  uploading the entire dataset could increase VRAM substantially.
- Mixed precision is a separate numerical project. The dataset already stores
  normals as f16 on the host; that does not establish safety of f16 hash state,
  adjoints, softplus or optimizer moments.

## Proposed next commit boundary

1. Correct the microbatch guard and add optional accurate timing/copy/aggregation
   diagnostics. Distinguish no-GPU-timestamp support from a zero-duration result.
2. Add independent variants for level-indexed Adam, exact endpoint sigmoid reuse,
   and selective aggregation. Keep part-2 switches and unchanged defaults until
   the user compares them, or clearly identify any newly enabled experiment.
3. Prototype larger point tiles/subgroup reductions as a separately switchable
   kernel variant. Preserve current kernels for devices lacking capabilities.
4. Leave broad sorting, sparse geometry skipping, mixed precision and aggressive
   indirect allocation for later unless captures show they dominate.

The user remains responsible for builds and tests. For comparisons, use matched
checkpoints around 2k/10k/20k/28k, identical seed/batch/query/API, fresh output
directories and warmed shaders. Measure repeated wall-clock windows and valid
field points/second, not averages of reciprocal CPU submission times. First
exclude mesh/checkpoint intervals to isolate kernels, then restore the actual
mesh cadence and measure full reconstruction wall time. Validate gradients and
optimizer updates, then loss trajectories/meshes, before accepting a speedup.
Neither the article's A100 ratios nor low GPU power predicts an additive gain.


## Implementation follow-up

The commit following this analysis implements the proposed next-commit boundary:
local microbatch bounds; level-indexed Adam with legacy fallback; exact endpoint
sigmoid reuse; fixed-subgroup forward reduction with portable fallback; optional
8/16-point forward weight reuse; configurable aggregation level cutoff and
cross-corner matching in both atomic variants; optional parameter-gradient
recomputation to eliminate the full dh buffer; removal of the unused GPU-training
bias reduction and reverse-table fill; conditional memory-maintenance waits;
copy/work counters; backend timing windows; address-only aggregation estimates;
and completed-run wall-clock/mesh/checkpoint timings. All have source review only.

The broader roadmap experiments explicitly deferred above remain deferred:
indirect count allocation, general spatial sorting, sparse mesh skipping,
mixed precision, full outer-product/optimizer fusion, and checkpoint ownership
redesign. They are not proven fixes and would require the newly available
measurements before selecting implementations. The existing low-memory state,
all loss terms, and optimizer moment semantics are retained.

# Run 6 findings and next implementation

Compared `experiments/after_george_kopanas_upgrades5` and
`experiments/after_george_kopanas_upgrades6`. Both contain 30,000 metrics rows
and successful wall-time summaries. Saved configurations match apart from the
new activation-cache option and output directory. Neither enabled diagnostics.

| Measurement | Run 5 | Run 6 |
| --- | ---: | ---: |
| Elapsed train call, including loading and exports | 773.405 s | 775.865 s |
| Mesh extraction | 55.299 s | 54.738 s |
| Checkpoint saves | 1.240 s | 1.262 s |
| Sum of logged iteration durations | 712.396 s | 714.943 s |
| Steps 1–1,000 effective it/s | 17.919 | 19.176 |
| Steps 1,001–6,000 effective it/s | 101.079 | 108.454 |
| Steps 6,001–10,000 effective it/s | 89.799 | 88.558 |
| Steps 10,001–20,000 effective it/s | 56.436 | 53.693 |
| Steps 20,001–30,000 effective it/s | 25.948 | 25.955 |
| Steps 24,001–30,000 effective it/s | 22.461 | 22.663 |

Rates above are row count divided by summed seconds, not averages of reciprocal
durations. Export time is accounted for separately by `wall_time.json`.

The full run is 0.32% longer; this does not establish a material regression from
one pair of runs. Early training improves, but the feature-only cache has not
demonstrated an overall speedup. It saves hash replay while introducing a
separate first-layer reconstruction and preactivation write/read.

Late slowdown largely tracks increased work: run 6 processes 16,436 visible
samples/iteration in steps 1,001–6,000 versus 70,683 in steps 20,001–30,000.
Their logged costs are approximately 561 and 545 ns/sample respectively. This
does not measure cache misses, atomic contention, or GPU bandwidth, but argues
against attributing the entire slowdown to progressively worse cache behavior.
Sample counts exclude some other work such as occupancy updates and visibility.

Final logged losses are close (run 5 versus run 6): normal 0.044517 versus
0.044537; mask 0.019376 versus 0.019308; weighted eikonal 0.003222 versus
0.003195. This is not a geometric quality or gradient-equivalence test. Sample
counts differ slightly, so these are not identical-workload measurements.

## Implemented next

1. **Fuse feature replay into backward.** Both native-float and CAS hash
   backward variants stage four cached feature vectors in 512 bytes of shared
   memory, rebuild each neuron preactivation in registers, and immediately use
   it for hidden/output derivatives. Packed weight loading and feature staging
   share one uniform barrier. No extra full-size preactivation tensor or
   standalone replay dispatch is needed on this path. Zero-adjoint points skip
   first-layer arithmetic. Feature accumulation order and FP32 arithmetic are
   retained at source level; shader compilation may affect numerical results.
   The existing cache budget can now hold nine full feature chunks rather than
   seven at query 131072, without reserving a 32 MiB preactivation workspace.

2. **Register-tiled first-layer parameter gradients.** A 128-thread group
   computes a 16×32 tile with four scalar accumulators per thread. Compared with
   the previous 256-thread 8×32 layout, each 256-sample part needs four groups
   instead of eight, and loads each feature tile four times instead of eight.
   Each output retains the same 256-sample accumulation order and partial-buffer
   layout/reduction. No extra sigmoid computations, global gradient buffer, or
   floating-point atomics are introduced. Shared storage per group is 6 KiB
   rather than 5 KiB; actual occupancy/performance requires measurement.

Both paths are enabled by default. `--legacy-feature-replay` restores the prior
materialized replay and workspace reservation; `--legacy-parameter-kernel`
restores the prior parameter kernel. `--legacy-gradient-tiles` still selects the
original 8×8 kernel. `--recompute-parameter-gradients` retains materialized replay
and its existing parameter kernel because it requires preactivations directly.

Sampling, occupancy, losses, mesh extraction, checkpoints, optimizer equations
and batch/query defaults are unchanged. Sampler count readbacks remain a future
target; eliminating them needs bounded device allocations and correct overflow
handling, not speculative oversizing that undermines the memory budget.

## Validation status

Read-only CSV aggregation, source review, rustfmt, and `git diff --check` only.
No build, tests, training, benchmark, or GPU capture was run. Rebuild and test
the same command first. To attribute changes, compare matched short checkpoint
resumes with either fallback individually and both fallbacks together, using
fresh output directories. Initial shader compilation can distort short runs.
Diagnostics can distinguish field-backward and parameter-gradient time, but
perturb throughput. No measured speedup or VRAM guarantee is claimed yet.

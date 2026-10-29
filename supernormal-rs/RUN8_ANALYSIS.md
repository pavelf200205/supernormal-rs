# Run 8 regression and mitigation

Compared the saved configurations, 30,000-row metrics files and completed
wall-time summaries in `experiments/after_george_kopanas_upgrades7` and
`experiments/after_george_kopanas_upgrades8`. Training settings match; the new
run-8 hybrid-cache and raw-weight packing paths were both enabled. Neither run
enabled kernel diagnostics. These files do not verify hardware/power settings.

| Measurement | Run 7 | Run 8 |
| --- | ---: | ---: |
| Total elapsed train call | 695.321 s | 799.038 s |
| Mesh extraction | 53.444 s | 51.754 s |
| Checkpoints | 1.238 s | 0.985 s |
| Steps 1,001–6,000 effective it/s | 118.368 | 89.712 |
| Steps 10,001–20,000 effective it/s | 59.396 | 52.235 |
| Steps 20,001–30,000 effective it/s | 29.636 | 25.625 |
| Late visible samples/iteration (20,001–30,000) | 70,793 | 2,124 |
| Final normal loss | 0.044565 | 0.148206 |
| Final mask loss | 0.019229 | 9.745985 |
| Final weighted eikonal loss | 0.003244 | 0.012752 |
| Final total loss | 0.067039 | 9.906942 |

Effective rates are row count divided by summed iteration durations. The
14.9% longer elapsed time is a regression, but the loss/sample data show a more
important correctness failure. These timings do not represent equivalent
reconstructions. They cannot establish a cache, bandwidth, or occupancy
performance bottleneck in a converging run.

The first logged losses agree within a few millionths and both runs start with
46,609 visible samples. Step 2 already diverges: run 7 has loss 7.8455 and 46,246
samples, while run 8 has loss 9.0182 and only 6,814 samples. The failure begins
around the first backward/update cycle, before late-stage complexity matters.

Both changes were enabled together and no intermediate parameter/gradient
captures exist. The logs do not isolate which change caused the failure.
It would be unjustified to label this ordinary reduction rounding or claim a
proven single root cause. The previous commit enabled unvalidated changes too
broadly; the following update removes the new normalization implementation and
restores the established cache policy by default.

## Changes

1. **Reference normalization, copy-only packing.** Remove the 65-workgroup
   hand-written weight-normalization kernel. Compute effective weights with
   the existing Burn expressions and cache, as in run 7. A copy-only packing
   kernel transposes/stores those normalized weights and biases directly into
   the packed field layout. It introduces no floating-point arithmetic on
   weights. Two 256-thread workgroups use padded 32×33 shared transpose tiles
   and one barrier each, replacing separate transpose/concatenation outputs.
   Both input loads and transposed stores follow adjacent columns/rows.
   Shape assertions check all four input matrices. `--legacy-mlp-packing`
   restores the entire run-7 Burn packing path if needed. No raw-weight
   normalization experiment remains available in the packing kernel.

2. **Hybrid caching is experimental and off by default.** The established
   run-7 feature-only policy is restored. `--hybrid-activation-cache` explicitly
   opts into filling unused cache capacity with preactivations. The older
   `--no-hybrid-cache` option remains accepted; the two switches conflict.
   Avoid the two-pass planner and its vectors entirely on the default path.
   The cache budget and fused backward/parameter kernels from run 7 remain.

3. **Remove redundant final parameter-tile barriers.** The scalar and register
   parameter-gradient kernels do not reuse shared memory after the eighth
   sample block. Omit that final barrier, retaining all barriers before reads
   and before the next block overwrites shared storage. This reduces barriers
   from 16 to 15 per group without changing output arithmetic or buffers.

Optimizer formulas, checkpoints, sampling and loss equations are unchanged.
New packing consumes normalized values without a new reduction grouping; its
GPU execution has not yet been validated. Driver VRAM and speed still require
measurement. No speedup or confirmed correctness recovery is claimed yet.

## Validation and next run

Read-only CSV analysis, source review, rustfmt and whitespace checks only.
No build, tests, benchmarks, GPU captures or reconstruction were launched.
Rebuild and use the same parameters in a fresh output directory, with hybrid
caching left off. Start fresh or resume a known-good run-7 checkpoint: updating
the binary does not repair parameters already trained into the run-8 failure.

First confirm healthy losses with a short continuation. If the new copy-only
packing still diverges, repeat with `--legacy-mlp-packing` to restore both
run-7 cache and packing behavior; keep other settings fixed. Do not spend another
full run treating high loss and collapsed samples as a performance comparison.

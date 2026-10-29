# Run 11 analysis and wider parameter-gradient tiles

Sources: `experiments/after_george_kopanas_upgrades10/metrics.csv` and
`experiments/after_george_kopanas_upgrades11/metrics.csv`, both 30,000 steps.
Rates below are steps divided by summed recorded seconds, not arithmetic means
of instantaneous rates. These are training timings, not complete export-inclusive
wall times; mesh extraction occurs at 10k, 20k and 30k.

| Metric | Run 10 | Run 11 |
|---|---:|---:|
| Recorded training seconds | 605.598 | 616.942 |
| Steps 1–2k, it/s | 39.36 | 39.97 |
| Steps 2,001–10k, it/s | 113.76 | 118.17 |
| Steps 10,001–20k, it/s | 64.12 | 63.36 |
| Steps 20,001–30k, it/s | 30.44 | 29.29 |
| Final total loss | 0.06701594 | 0.06719131 |

Run 11's recorded training time increased 1.87%. It improved early throughput,
but later throughput regressed. Late mean samples also rose from 70,505 to
71,482 (1.39%). Different sampled workloads prevent attributing the entire
regression to kernel performance. Similar scalar losses do not establish mesh
parity. The last update therefore has no demonstrated overall performance win.

## Next implementation

The first-layer parameter-gradient kernel previously launched four workgroups
per 256-point partition, each computing 16 neurons. Each group separately staged
the same input features. The new default computes 32 neurons per group:

- Two workgroups instead of four, halving repeated feature loads, feature shared
  writes and group-level synchronization executions for each partition.
- Contiguous 32-channel derivative loads instead of 16-channel segments.
- Eight register accumulators per thread reuse each feature across four neurons.
- The same 256-point partials, accumulation order per output, active-column
  handling, bias placement and subsequent reduction remain in use.

Global buffer sizes and persistent VRAM allocations are unchanged. Shared memory
per group rises from 6 KiB to 8 KiB, and register usage increases; reduced hardware
residency could offset the savings. This targets all ordinary register-tiled
parameter-gradient queries, including cached points, rather than only cache misses
or occupancy refreshes. Arithmetic count is unchanged; half the workgroups does
not imply twice the training speed.

`--legacy-parameter-rows` selects the original 16-neuron kernel independently.
Legacy parameter kernels and gradient-recompute mode retain their existing paths.
Use identical settings and a fresh output directory to compare throughput and
quality. No build, tests or GPU execution were performed. Source review, rustfmt
and `git diff --check` only.

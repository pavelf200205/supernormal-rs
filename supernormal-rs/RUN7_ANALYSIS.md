# Run 7 comparison and follow-up

Sources: `experiments/after_george_kopanas_upgrades6` and
`experiments/after_george_kopanas_upgrades7`, using each run's `metrics.csv`,
`config.json`, and `wall_time.json`. Both completed 30,000 iterations with
batch/microbatch 2048, query 131072, and 1024-resolution mesh exports every
10,000 iterations. Saved configurations differ only in output paths and the
new legacy flags, which are false in run 7. Hardware/power settings cannot be
verified from these files.

| Measurement | Run 6 | Run 7 |
| --- | ---: | ---: |
| Elapsed train call including loading and exports | 775.865 s | 695.321 s |
| Mesh extraction | 54.738 s | 53.444 s |
| Checkpoint saves | 1.262 s | 1.238 s |
| Sum of logged iteration durations | 714.943 s | 635.595 s |
| Steps 1–1,000 effective it/s | 19.176 | 21.048 |
| Steps 1,001–6,000 effective it/s | 108.454 | 118.368 |
| Steps 6,001–10,000 effective it/s | 88.558 | 99.867 |
| Steps 10,001–20,000 effective it/s | 53.693 | 59.396 |
| Steps 20,001–30,000 effective it/s | 25.955 | 29.636 |
| Steps 24,001–30,000 effective it/s | 22.663 | 25.840 |

Rates are row counts divided by summed iteration durations, not arithmetic
averages of instantaneous rates. Mesh/checkpoint costs are measured separately
in the successful wall-time summary. Full elapsed time decreased by 10.4%;
logged iteration time decreased by 11.1%. Late-window throughput increased by
14.2%. These measurements support keeping the previous fused replay and
register-tiled parameter-gradient defaults, but do not isolate their individual
contributions.

Late sample counts are almost unchanged: 70,683 versus 70,793 per iteration in
steps 20,001–30,000. Their logged cost falls from about 545 to 477 ns/sample.
This is consistent with reduced overhead/work per sample rather than simply
processing fewer samples. It is not a measurement of hardware bandwidth or
cache misses. Increasing sample counts still explain much of the remaining
slowdown as training proceeds.

Final normal/mask/weighted-eikonal losses are respectively
0.044537/0.019308/0.003195 in run 6 and
0.044565/0.019229/0.003244 in run 7. Final total loss differs by approximately
0.0000013. Close scalar losses do not establish mesh or gradient equivalence.

## Next changes

### Hybrid activation cache

The previous policy switched every retained chunk to feature-only storage as
soon as full activations exceeded the budget. For example, four full query
chunks retain only 62 MiB of features inside a 142.5 MiB budget; all four replay
the first layer even though there is space to retain two preactivation chunks.

The new planner first retains exactly the same feature chunks as before, then
fills spare bytes with preactivations for those retained chunks. Four full
chunks now retain all four feature chunks and two preactivation chunks, using
126 MiB. Five full chunks use 141.5 MiB with two preactivation chunks; six or
seven full chunks can retain one preactivation chunk. Eight or nine full chunks
have insufficient spare space for a full preactivation chunk, though a small
tail may fit. Planning uses padded shapes and handles tails individually.

The existing three-full-chunk byte ceiling remains unchanged. Materialized
legacy replay still reserves its temporary workspace first. Retaining features
has priority: adding preactivations never evicts a feature chunk. The planner
changes storage/replay decisions only, not sample order or loss equations.
`--no-hybrid-cache` restores run 7's feature-only policy. The existing
`--legacy-activation-cache` restores the older first-three-full-chunks policy.

The existing diagnostic full/feature-only cached-point counters now reflect the
actual per-chunk decision. `field.retained_activation_bytes` sums the planned
retained bytes over feature-cache batches; it is not peak VRAM or a live byte
gauge. Total driver memory still depends on allocation pooling and lifetimes.

### Fused MLP weight preparation

Fused forward previously normalized the two weight matrices using Burn
expressions, transposed the first matrix, and concatenated weights and biases
into the 2113-float shader layout after each optimizer update. A dedicated
kernel now performs row normalization and writes that layout directly. It
launches 65 workgroups of 64 threads with a 256-byte shared reduction tile.
Every packed entry is written exactly once. There are no intermediate effective
weight matrices, transpose outputs, concatenation copies, or host readbacks in
this packing path.

Packed weights retain their existing lifetime: cached within an iteration and
invalidated by the optimizer. Optimizer state and checkpoint formats are
unchanged. The kernel uses the same `v * g / sqrt(sum(v*v))` formula, without a
new epsilon or precision change. Its reduction grouping can differ from Burn,
so bitwise equivalence is not promised. The fused field also serves mesh
inference: extraction topology and settings are unchanged, but small SDF
rounding differences are possible. `--legacy-mlp-packing` restores Burn packing
independently of the cache switch. Diagnostics add `pack_mlp_weights` timing.

Both changes are enabled by default. They target remaining replay and dispatch
costs; no new speedup is claimed before measurement. Sample-count readbacks,
sorting/divergence, and broader optimizer fusion remain separate future work.

## Validation and next comparison

Only read-only CSV analysis, source inspection, formatting, and whitespace
checks were performed. No build, tests, GPU capture, benchmark, or training run
was launched. Rebuild and use the same training settings with a fresh output
directory. To isolate the new changes, compare matched checkpoint resumes with
`--no-hybrid-cache` and `--legacy-mlp-packing` separately, then together. Discard
initial compilation effects; check losses and mesh quality as well as speed.

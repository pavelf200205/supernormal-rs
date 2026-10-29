# Training traffic and inactive-work upgrades

Run 9 completed in 686.937 seconds versus run 7's 695.321 seconds. Its final
loss was 0.067101 versus 0.067039, and late throughput was 29.78 versus 29.64 it/s.
These saved metrics support a small overall gain and healthy scalar losses;
they are not a geometric parity test or attribution of individual changes.

This update targets four other costs, enabled by default:

## Hash Adam: exact zero-state fast path

Read gradient, first moment and second moment before fetching the parameter.
When all three are exactly zero, keep the parameter/state unchanged and exit.
A zero gradient with historical momentum still executes Adam normally. This is
not sparse Adam that ignores moment decay. Processed entries use the existing
formulas, correction counts and epsilon. If the gradient is already zero, avoid
its redundant clear store even when nonzero moments require an update.

For an entirely idle FP32 entry, logical parameter/state traffic falls from
four reads plus four writes (32 bytes) to three reads (12 bytes), excluding
correction metadata and cache effects. The number of such entries is workload
dependent; memory transactions and overall savings are not measured. Additional
branches may cost time on dense workloads. No counter readback or new buffers
are introduced. --dense-hash-adam restores unconditional updates/clear writes.

## Active first-layer parameter gradients

The default register-tiled kernel computes only active hash features plus XYZ
and bias. Bias is packed immediately after active features inside the shared
tile, so processing bias does not force execution of the other half-tile.
Outputs are mapped back into the same 64x32 partial layout; inactive output
columns are explicitly zeroed. Inactive zero writers never touch the remapped
bias slot. Every valid partial output has exactly one writer, including tails.
The existing partial reduction, weight-normalization derivative and optimizer
remain unchanged. This assumes finite values in healthy training.

With one active level, six columns (five features plus bias) require products
instead of 32. This is an arithmetic count, not an 81% kernel speedup: warp
execution, staging, barriers, and output writes remain. At all 14 levels the
same full dot-product loop is used. Active column count is a runtime scalar,
so progression does not require 14 new shader variants. Reference/unfused,
legacy tile and parameter-recomputation paths retain dense behavior.
--dense-parameter-gradients passes all 31 feature columns for comparison.

## Direct chunk adjoints

Fused GPU training backward reads the shared global adjoint tensor with a
bounds-checked chunk offset rather than constructing a Tensor::slice for every
query. Both native-float and CAS backward shaders honor the offset/stride.
Reference, standalone bias-reduction and parameter-recompute paths keep local
slices. Address diagnostics create their own debug slice while the real kernel
still receives the direct buffer. No production readback is added.
--legacy-adjoint-slicing restores the previous per-query slicing path.

## Selective backward weight staging

Load only the weights actually read by the active backward computation. With
cached preactivations, this means active hash columns and output weights; XYZ
columns and first-layer bias are unnecessary. Feature replay also loads active
XYZ columns and first-layer bias. Output bias is never needed here. Both hash
layouts, native/CAS atomics, and corner mappings use the same selection rule.
The shared array retains its existing size and aggregation reuse/barriers;
unloaded slots are never read before aggregation overwrites its own staging.

At one active level without replay, staging falls from 2113 to 192 floats per
four-point group. At 14 levels without replay it is 1856 floats. Replay loads
more, and the reduction at full activation is only one unused output-bias float.
These are logical loads/shared writes, not DRAM byte counts or a guaranteed
speedup. --dense-backward-weight-loads restores full staging.

## Scope and validation

No sampling, occupancy, loss weights, precision, normalization, cache budget,
mesh extraction or checkpoint format changes. No new large allocations.
The four fallback switches are independent and restore the previous behavior
for these paths when supplied together. Existing comparison switches remain.

Source inspection, rustfmt and git diff --check only; no build, tests, training,
GPU captures or benchmarks were run. Rebuild and test unchanged batch/query
settings in a fresh directory, first checking a short resume from a known-good
checkpoint. Measure early and late phases separately: inactive-column/staging
savings favor earlier phases, while idle hash states and eliminated query copies
may also help later. Neither performance nor numerical parity is confirmed.

# Fused backward replay and device-driven occupancy

Three changes are enabled by default. Compilation, shader execution, numerical
parity and speedup have **not** been tested. Only source review, rustfmt and
`git diff --check` were performed; rebuild and test locally.

## Occupancy: remove the count readback and reuse weights

After the 256-step full-grid warmup, occupancy selection still uses the same
seed, selected cells, prefix sum and jittered coordinates. Its count stays on
the GPU. Compaction reserves capacity for every grid cell, which cannot overflow
because each cell is emitted at most once.

A field dispatch of at most 4,096 workgroups reads that count and processes
successive tiles with a grid-stride loop. Each group stages MLP weights once and
reuses them across its tiles. Only valid compacted points are evaluated. The
kernel directly applies the existing sigmoid/EMA expression to unique cell IDs,
eliminating the intermediate SDF buffer, query slicing/concatenation and separate
EMA dispatch. Mean/threshold calculation remains on the existing Burn path.

This replaces the blocking count download every eight training steps. It does
not remove the separate marching/visibility count readbacks. The grid-stride
loop has uniform barriers, disjoint point ownership, and a uniform exit before
any uninitialized tail data can be read. Outstanding value-buffer aliases are
protected by the existing copy-on-write pattern.

Temporary XYZ and ID capacity is bounded to 32 MiB for a 128³ grid. This can
increase occupancy scratch usage relative to exact-sized selection buffers;
overall VRAM is not guaranteed unchanged. Larger grids, `--chunked-eval`, CPU
occupancy and unfused fields retain the old path. Training query/cache limits
are unchanged. Persistent dispatch trades scheduling overhead and repeated
weight loads for longer-lived workgroups; hardware-dependent regressions remain
possible. Use `--legacy-occupancy-readback` for independent comparison.

## Backward: fuse hash replay when the activation cache runs out

Previously, uncached chunks reran full forward, writing features, 64 hidden
preactivations and an unused SDF result. The new path reconstructs hash features
inside backward, computes hidden preactivations in registers, and immediately
consumes them for derivatives. The features are returned as an explicit Fusion
output dependency for the following parameter-gradient kernel.

Forward and backward share the same interpolation helper, including corner
summation order. Native-float and CAS atomics use the same replay path. Inactive
features are zeroed, valid output rows have unique writers, and existing
parameter reductions, weight normalization and Adam remain unchanged.

For a 131,072-point cache miss this removes a 32 MiB preactivation allocation
and its writes/reads, the discarded SDF work, and one kernel dispatch. It does
not benefit chunks already retained by the cache. Shared scratch is reused;
the replay variant uses 12,052 bytes per workgroup, excluding registers and
compiler overhead. `--legacy-hash-replay` restores full forward replay.
`--legacy-feature-replay` and parameter-gradient recomputation also retain their
existing separate replay paths. The activation-cache budget is unchanged.

## Forward: stage only active first-layer weights

All fused forward queries load active W0 columns plus both biases and W1.
With one active hash level, this is 449 floats instead of 2,113 per group.
With all 14 levels active it is unchanged. Arithmetic and active-level schedules
are unchanged. `--dense-forward-weight-loads` restores full staging.

## Comparing the update

Use the same dataset, seed, batch/query settings and export schedule in a fresh
output directory. Check losses and meshes as well as elapsed time. An early
resume and a late resume exercise different costs. All three new fallback flags
together disable this update's optimized paths. Existing diagnostics additionally
report `field.fused_hash_replay_points`, `occupancy.device_count_updates` and
the `occupancy.fused_field_apply` timing span; enabling synchronized diagnostics
changes throughput and should be kept separate from performance comparisons.

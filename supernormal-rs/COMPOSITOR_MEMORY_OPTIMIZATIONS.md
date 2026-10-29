# Compact compositor cache

Default GPU compositing now uses seven floats per pixel/sample rather than ten.
Forward retains gradient XYZ, transmittance, alpha and sigmoid endpoints.
Reverse reloads the two original SDF values from the immutable SDF tensor,
loads its cached values before overwriting them, and reuses slots 0-3 for:

- start-endpoint derivative: -dp0 + z0 * inv
- horizontal stencil derivative: dp1
- vertical stencil derivative: dp2
- end-endpoint derivative: dp0 + z1 * inv

Each invocation owns its pixel records. Other pixels consume final stencil
values only in the subsequent gather dispatch. Gather reads precombined endpoint
derivatives rather than loading two separate values for each occurrence.
Compact and expanded reverse tables preserve their previous stencil signs/order.
No samples, eikonal terms, thresholds, variance/bias derivatives, exponentials,
atomics, workgroup barriers or readbacks are added/dropped. GPU compiler
contraction can still change rounding when expressions move between dispatches.

## Traffic

Scratch capacity drops from 90 to 63 floats per nine-pixel sample: 30% less.
For 100,000 allocated rows this is 36.0 MB versus 25.2 MB (10.3 MiB saved).
Forward/reverse scratch stores fall from 14 to 11 per pixel, saving 108 logical
store bytes per sample. Endpoint gathering also needs fewer loads. These are
source-level counts, not DRAM counters or an overall speedup. Reloading SDF and
index values is the tradeoff; actual cache hits and driver VRAM require measurement.

Enabled by default. --legacy-composite-cache restores ten-float records and
separate endpoint expressions. --legacy-composite-scratch independently switches
component-major/pixel-major layout; all four combinations use matching indexing
and allocations. The custom-op wrapper asserts the scratch width.

## Kopanas follow-up

Source: https://grgkopanas.github.io/3dgs_cuda_blog/

The saved article/diffs and repository audit were reviewed. Reopening the
article through the web tool failed. Reducing live intermediates/global traffic
is the principle adapted here. Grouped hash corners, shared MLP weights, compact
gradient partials, local same-address aggregation, buffer reuse and zero-adjoint
exits already exist, with some experiments optional.

Not every possible optimization is integrated. CUDA half-warp folding and PTX
approximations are not portable drop-ins. Gaussian SH streaming, tile binning,
contributor thresholds and opacity/conic formulas have different dependencies.
Zero-alpha samples can still contribute to eikonal and stencil derivatives.
Sampler count readbacks and more field/optimizer fusion remain future work;
the article's rasterizer speed ratios do not predict speedups in this pipeline.
See KOPANAS_AUDIT.md for the earlier detailed mapping.

## Validation

Source review, rustfmt and whitespace checks only: no builds, tests, GPU runs or
benchmarks. No successful new run has appeared after the run-8 failure and its
mitigation. Rebuild and compare short resumes from a known-good checkpoint with
and without --legacy-composite-cache, using fresh output directories. Check
losses and geometry before full-run timing. Performance/parity remain unverified.

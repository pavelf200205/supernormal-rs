# SuperNormal kernel performance investigation

2026-10-19. Source inspection only; no competing GPU workload was launched while the user's 30k-step run is in progress. Bottleneck rankings and speedups require profiling afterward.

## Exact reference sources

`create_env.sh` pins tiny-cuda-nn to `2ec562e853e6f482b5d09168705205f46358fb39`. Cloned to `tiny-cuda-nn-reference/` beside the application and checked out detached at that commit. The script does **not** pin a separate NerfAcc Git commit: it installs the local, modified `third_parties/nerfacc-0.3.5/nerfacc-0.3.5/` tree. The application repository's pinned commit therefore identifies the relevant NerfAcc code. Do not substitute upstream 0.3.5 without its patch-based operations. The setup pin does not independently prove which tiny-cuda-nn binary is installed in the supplied environment.

## What is accelerated here

SuperNormal uses `tcnn.Encoding`, followed by ordinary PyTorch Linear/weight-normalization/Softplus layers. It does not use `tcnn.Network` or `NetworkWithInputEncoding`. Claims about tiny-cuda-nn's fully fused neural networks or Tensor Core MLPs cannot explain the encoding replacement's speed difference in this repository.

### Hash encoding

Pinned `include/tiny-cuda-nn/encodings/grid.h` implements grid lookup, coordinate arithmetic, corner weights and eight-corner interpolation within a specialized GPU kernel. The launch spans points and levels; each block processes one level, with cache locality explicitly motivating the layout. Compile-time dimensions/features allow unrolled loops and vector loads. Output layout favors coalesced writes, with a transpose where needed. Backward directly accumulates corner contributions into table gradients with custom atomics and supports explicit spatial derivatives.

The PyTorch replacement's active-level loop builds many separate tensors for coordinates, corner indices and interpolation weights, performs eight `index_select` operations per level, multiplies and adds their outputs, then concatenates levels. Fourteen active levels imply 112 gather operations per SDF invocation, before counting arithmetic and backward. This is an operation count, not a measured kernel count. Eager execution incurs dispatch, tensor allocation, memory traffic and autograd bookkeeping between these operations. GPU arithmetic still executes on CUDA: the difference is the granularity and arrangement of the work, not that Python computes each sample on the CPU.

Its per-level parameter allocation and skipping inactive levels are useful improvements worth retaining. The original wrapper computes the encoding and masks inactive features afterward. Both training paths also use checkpointing, so recomputation amplifies field-evaluation cost; it is not unique to the PyTorch branch.

The pinned wrapper selects native encoding precision at construction. Its forward casts parameters to native precision even if the surrounding module is converted to float32. Source supports efficient half2 atomic gradient accumulation on suitable NVIDIA architectures. Thus `.to(torch.float32)` alone does not establish a fully-f32 native encoding. Verify installed native precision before benchmarking. Float32 master parameters/Adam state may still exist; do not infer that total training memory is halved.

### Ray marching

Vendored `cuda/csrc/ray_marching.cu` assigns a thread to a ray, advances through its interval, tests occupancy and writes only accepted samples. For AABB contraction it computes the next voxel boundary and advances the regular stepping lattice through empty space. This implementation still uses a small stepping loop to reach the boundary; it does not magically remove all stepping work. It does avoid repeated occupancy tests and dense candidate-position buffers for that span.

It counts samples, prefix-sums counts, allocates packed output and reruns the march to fill it. There is a host `.item<int>()` synchronization for the total count, so the CUDA implementation is neither synchronization-free nor single-pass.

The PyTorch implementation constructs candidate `ts`, `te`, midpoints and XYZ arrays for each ray chunk, tests every candidate's occupancy, then compacts with `nonzero`/indexing. Chunking bounds some temporary storage but still performs work on rejected positions. Repeated `.item()`, GPU-value conditionals and dynamic compaction introduce additional host/device coordination. Sparse occupancy and fine step sizes are particularly likely to expose this overhead.

The exact old NerfAcc path computes center-ray alpha and filters visibility **after** occupancy marching. Do not attribute avoidance of all occluded SDF queries to an imagined fully fused early-termination marcher.

### Patch transmittance and backward

Vendored `render_weight.cu` maps threads to patch/pixel pairs. Each walks a contiguous sample segment, keeps transmittance in a scalar, and writes weights. The custom backward makes linear passes over the same packed segment. It needs no rectangular padding to the longest ray.

PyTorch `_segmented_transmittance_and_weights` computes counts, reads the maximum length, allocates a padded `[active rays, max samples, patch pixels, 1]` tensor, scatters alpha into it, runs cumprod and gathers the packed result back. Work scales with the padded volume rather than just valid samples. For illustration, an average length of 100 and maximum 2000 produces 20x as many padded slots, not a claim of 20x total runtime.

Final weighted accumulation in the vendored `vol_rendering.py` already uses PyTorch `scatter_add_`, as does the replacement. Occupancy updates also retain substantial PyTorch work. The CUDA baseline is a hybrid, not an entirely fused pipeline.

## What carries over to Rust/WGPU

| Optimization | Portability strategy |
|---|---|
| Fused coordinate/hash/interpolation kernel | CubeCL or WGSL kernel with specialized feature counts and level-local workgroups. |
| Packed count/scan/fill marching | Portable compute kernels, bounded capacity and explicit overflow handling. |
| Empty-voxel skipping | Port the algorithm with fixture-verified interval/rounding behavior. |
| Packed patch-weight forward/backward | Portable per-segment recurrence, optionally hierarchical scans after profiling. |
| Reduced allocations and recomputation control | Buffer reuse, whole-patch microbatches, explicit backward/recompute. |
| Custom table-gradient accumulation | Harder: benchmark portable integer-CAS float accumulation or bounded sort/reduce; native float atomics only as optional capability-specific paths. |
| Native half2 atomics, CUDA launch/layout tuning | No universal direct equivalent; tune per backend and retain a portable baseline. |
| Half precision | Optional capability/quality-tested optimization, not a minimum requirement. |

Burn supports custom backend operations with handwritten backward implementations. Registering one does not automatically derive derivatives or fuse it into neighboring operations: [Burn extension documentation](https://burn.dev/books/burn/advanced/backend-extension/index.html).

Replacing Python with Rust while expressing the same large sequence of tensor operations is insufficient. Burn fusion may reduce intermediate work, but it should not be assumed to discover the packed marcher or specialized hash backward on its own. The intended design is Burn plus specialized portable kernels, preserving algorithmic optimizations while replacing CUDA implementations.

Expect room to improve substantially over this eager PyTorch version. Do not promise equality with tuned CUDA on the same NVIDIA GPU, nor NVIDIA-class throughput from a laptop iGPU. Hardware bandwidth, power limits, atomics and drivers still matter.

## Fidelity discoveries from the pinned source

The encoding difference is deeper than an unconfirmed coordinate normalization issue:

* tiny-cuda-nn uses `scale = base * exp2(level * log2(scale_factor)) - 1`, vertex resolution `ceil(scale)+1`, and interpolation coordinates `scale*x + 0.5`.
* Dense table sizes use that vertex resolution, align to a multiple of eight entries, then cap for hashing.
* The PyTorch port instead clamps normalized scene coordinates, multiplies by a floored cell resolution, uses `(res+1)^3` dense entries, and has no half-cell offset.
* The coherent-prime hash factors do match the intended three-dimensional constants, but that alone does not make the full encoders equivalent.

These are source-level differences, not proof that either reconstruction is unusable. A similar final mesh can coexist with different internal representations and optimization trajectories. Renderer-only fixtures should use analytic SDFs or controlled outputs; encoding parity needs the exact baseline convention and table mapping.

## Profiling after the user's run

1. Freeze weights, active levels, points, occupancy, patch IDs and random jitter. Match logical batch size; sub2gb forces 1024 while other branches default to 2048.
2. Benchmark encoding forward/table backward, marcher, patch weights forward/backward, and complete training steps separately with warmup and GPU-aware timing.
3. Measure early/mid/late training, because active levels, step sizes and occupancy evolve. Include occupancy-update and ordinary steps separately.
4. Record valid versus candidate/padded sample counts, peak allocated/reserved memory, GPU time, host gaps and native precision. Separate mesh/evaluation pauses from training throughput.
5. Validate numerical behavior before accepting each faster implementation. Keep fixed-input kernel comparisons distinct from end-to-end quality comparisons.

No measured speedup or percentage attribution is established by this investigation.

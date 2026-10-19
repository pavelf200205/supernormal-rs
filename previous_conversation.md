# Plan Rust Burn rewrite

> please clone the pavelf200205/SuperNormal_memory_optimized repository, check out the pure_pytorch_version
>   branch, then come up with a plan to rewrite it in its entirety into Rust + Burn framework (WGPU target)
>   so that it works everywhere, not just on CUDA, but also on embedded laptop graphics cards.

<details><summary>6 previous messages</summary>

> I’ll clone the repository, check out `pure_pytorch_version`, and inspect the training and rendering pipeline. I’ll then develop a migration plan around Burn’s WGPU backend, including the parts that need custom GPU kernels and the constraints of integrated graphics.
>
> <details><summary>Explored 3 files, ran 8 commands, searched the web</summary>
>
> - Ran `Get-Location; rg --files -g AGENTS.md -g Cargo.toml -g README.md -g '*requirements*' -g pyproject.toml -g .git`
>
> - Ran `git clone --branch pure_pytorch_version --single-branch https://github.com/pavelf200205/SuperNormal_memory_optimized.git`
>
> - Ran `git clone --branch pure_pytorch_version --single-branch https://github.com/pavelf200205/SuperNormal_memory_optimized.git`
>
> - Ran `Get-ChildItem -Force; Get-ChildItem -Path E:\AGENTS.md,E:\SuperNormal_Rust_Burn_Framework_Rewrite\AGENTS.md -ErrorAction SilentlyContinue | Get-Content`
>
> - Ran `git status --short --branch; git rev-parse HEAD; rg --files -g AGENTS.md -g '!*.ipynb';`
>
> Searched the web for `site.burn.dev WGPU custom kernel autodiff CubeCL backend ...`
>
> - Ran `Get-ChildItem -Force; rg --files --hidden --no-ignore -g '!.git/**'`
>
> - Ran `Get-Content README.md,create_env.sh,config\diligent.conf,config\own_objects.conf`
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./models/fields.py`
> - Read `./models/renderer.py`
> - Read `./models/dataset_loader.py`
>
> </details>
>
> - Ran `Get-Content models\renderer.py -TotalCount 280; Get-Content models\fields.py | Select-Object -Skip 217`
>
> </details>
>
> The branch is checked out. Its hash encoding and renderer have already been converted to PyTorch, but the dataset loader and experiment runner still hard-code CUDA. The main Rust challenges are preserving patch-based rendering and its gradients while bounding memory use; changing the tensor backend alone won’t solve those.
>
> <details><summary>Ran 4 commands, searched the web</summary>
>
> - Ran `Get-Content models\renderer.py | Select-Object -Skip 280 -First 430`
>
> - Ran `rg -n --no-ignore '^(class |def |    def )|cuda|loss|checkpoint|Adam|bandwidth|sampling_step|occ_eval|gradient_method' exp_runner.py models\cd_and_fscore.py utilities\utils.py data_capture_and_preprocessing; Get-Content exp_runner.py -TotalCount 330`
>
> - Ran `Get-Content models\renderer.py | Select-Object -Skip 710 -First 220; Get-Content data_capture_and_preprocessing\README.md; Get-Content models\cd_and_fscore.py; Get-Content exp_runner.py | Select-Object -Skip 482 -First 151; Get-Content data_capture_and_preprocessing\iPhone_mvps_data_preprocessing.py -TotalCount 45; Get-Content data_capture_and_preprocessing\gather_and_convert_normal_map.py -TotalCount 70`
>
> Searched the web for `https://burn.dev/books/burn/advanced/backend-extension/index.html`
>
> - Ran `Get-Content exp_runner.py | Select-Object -Skip 507 -First 127; Get-Content utilities\utils.py; Get-Content data_capture_and_preprocessing\metashape2neus.py -TotalCount 100; Get-Content data_capture_and_preprocessing\metashape2neus2_json_and_images.py -TotalCount 70; git status --porcelain`
>
> </details>
>
> I’m structuring the plan around a faithful DFD training implementation first, followed by portable kernel optimizations. DFD is the branch’s default and avoids the higher-order autodiff needed by its `ad` mode.
>
> The default hash tables occupy about 45 MiB; parameters, gradients, and Adam moments together need roughly 181 MiB before activations. Sample buffers are the larger memory risk, so the plan will use bounded patch batches, packed samples, and streaming mesh extraction. I also found resume and evaluation inconsistencies that need explicit fixes during the port.
>
> <details><summary>Created a file, edited a file, explored a file, ran 2 commands</summary>
>
> - Wrote `./RUST_BURN_REWRITE_PLAN.md` (+173 -0)
>
> - Ran `rg -n --no-ignore 'num_view|num_image_per_view|subtract|ambient' data_capture_and_preprocessing\iPhone_mvps_data_preprocessing.py; git status --short --branch; git rev-parse HEAD`
>
> - Ran `Get-Item RUST_BURN_REWRITE_PLAN.md; Select-String -Path RUST_BURN_REWRITE_PLAN.md -Pattern 'DD A|15-view'`
>
> <details><summary>Explored a file</summary>
>
> - Read `./data_capture_and_preprocessing/iPhone_mvps_data_preprocessing.py`
>
> </details>
>
> - Wrote `./RUST_BURN_REWRITE_PLAN.md` (+2 -2)
>
> Tool call
>
> Tool: open_in_codex
> Status: completed
>
> </details>
</details>

Cloned the repository into `SuperNormal_memory_optimized` and checked out **`pure_pytorch_version`** at `c4a99143`. The checkout is clean.

Created the detailed [Rust + Burn/WGPU rewrite plan](./RUST_BURN_REWRITE_PLAN.md), covering architecture, every application workflow, migration phases, memory budgets and validation gates.

The recommended approach is:

1. Establish numerical parity with the existing DFD training pipeline.
2. Implement Burn models with portable CubeCL/WGPU kernels.
3. Bound memory through patch microbatches, packed samples and streaming mesh extraction.
4. Complete checkpointing, alternative gradient modes, evaluation and preprocessing.
5. Qualify Intel/AMD integrated graphics, Apple Metal and other desktop GPUs, with CPU fallback.

The main risks are hash-gradient accumulation, differentiable segmented rendering and higher-order derivatives for AD mode. WGPU broadens hardware support, but compatibility and practical training speed still need testing on real laptops.

This completes the inspection and planning; implementation and benchmarks have not started.

> Do not bother with the data preparation scripts just yet (especially SAM). i am only interested in the actual supernormal reconstruction pipeline at the moment. also, please confirm that the app will be able to utilize the iGPU of the i3-1315U just like the Brush gaussian splatting app does.
> Additionally, please explore other branches of the repository (main and sub2gb).
> Also, answer the question: if i provide you with an actual dataset needed for the reconstruction, will you be able to perform the rewrite more faithfully by porting parts step by step while verifying that they work similarly to the original version? The fully configured UV virtual environment is located at E:\Projects\MVPS_Studio\.venv (should work with all branches).
> Here's one more thing that i want to say: the pure_pytorch_version is an intermediate step in my journey of making SuperNormal more accessible with the previous steps related to reducing the memory usage on Nvidia GPUs (the the other 2 branches). The pure_pytorch_version haven't been tested and compared diligently against the baseline implementation that uses tiny-cuda-nn and nerfacc, but it seems to work and produces a plausible result.

<details><summary>6 previous messages</summary>

> I’ll narrow the plan to reconstruction, compare `main` and `sub2gb` with `pure_pytorch_version`, and check the i3-1315U’s graphics support against Burn/WGPU and Brush. I’ll also inspect the supplied environment so the validation plan can use the CUDA implementation as the baseline, rather than assuming the unverified PyTorch port is equivalent.
>
> <details><summary>Ran 6 commands, searched the web</summary>
>
> - Ran `git fetch origin main sub2gb; git branch -a; git log -1 --oneline origin/pure_pytorch_version`
>
> Searched the web for `site.intel.com i3-1315U processor graphics DirectX Vulkan ...`
>
> - Ran `git fetch origin main:refs/remotes/origin/main sub2gb:refs/remotes/origin/sub2gb`
>
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B -c "import sys, importlib.util; import torch; print('python',sys.version); print('torch',torch.__version__,'cuda_build',torch.version.cuda,'cuda_available',torch.cuda.is_available()); print('gpus',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print('modules', {m: bool(importlib.util.find_spec(m)) for m in ['tinycudann','nerfacc','pyhocon','pyexr','mcubes','open3d','cv2']})"`
>
> - Ran `git log --all --oneline --decorate -15; git diff --stat origin/main origin/sub2gb; git diff --stat origin/sub2gb origin/pure_pytorch_version; git diff origin/main origin/sub2gb -- models/fields.py models/dataset_loader.py config exp_runner.py`
>
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B -c "import sys, importlib.util; import torch; print('python',sys.version); print('torch',torch.__version__,'cuda_build',torch.version.cuda,'cuda_available',torch.cuda.is_available()); print('gpus',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print('modules', {m: bool(importlib.util.find_spec(m)) for m in ['tinycudann','nerfacc','pyhocon','pyexr','mcubes','open3d','cv2']})"`
>
> - Ran `git diff origin/main origin/sub2gb -- models/renderer.py; git show origin/main:models/fields.py; git show --stat a329415; git show --stat 85f4864; git diff 7979c5f origin/main -- models/renderer.py models/dataset_loader.py exp_runner.py`
>
> </details>
>
> The branch history matches your description: `main` retains tiny-cuda-nn and NerfAcc; `sub2gb` adds smaller SDF chunks and forces a 1024-patch batch; `pure_pytorch_version` then replaces the encoding and rendering implementations.
>
> One important difference is already visible: the PyTorch encoder normalizes and clamps positions to `[0,1]`, while the CUDA branch passes scene coordinates directly to tiny-cuda-nn. That needs a targeted comparison before choosing the Rust encoding behavior.
>
> Your environment loads PyTorch successfully outside the sandbox and detects an RTX 3060, so we have local CUDA hardware for baseline comparisons.
>
> <details><summary>Edited a file, ran 5 commands, searched the web</summary>
>
> - Ran `git rev-parse origin/main origin/sub2gb; git diff origin/sub2gb origin/pure_pytorch_version -- exp_runner.py; git show origin/main:third_parties/nerfacc-0.3.5/nerfacc-0.3.5/nerfacc/cuda/csrc/render_weight.cu | Select-String -Pattern 'patch|1e-10|backward' -Context 2,3; Get-Content 'E:\Projects\MVPS_Studio\.venv\Lib\site-packages\tinycudann\modules.py' -TotalCount 55`
>
> Searched the web for `site.intel.com "i3-1315U" "64" "1.25" graphics specifications`
>
> - Ran `Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\cuda\csrc\render_weight.cu | Select-Object -Skip 80 -First 42; Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\cuda\csrc\render_weight.cu | Select-Object -Skip 300 -First 45; git diff --quiet origin/main origin/sub2gb -- models/fields.py config models/dataset_loader.py; git diff --quiet origin/main origin/pure_pytorch_version -- exp_runner.py`
>
> - Ran `Get-Content 'E:\Projects\MVPS_Studio\.venv\Lib\site-packages\tinycudann\modules.py' -TotalCount 55`
>
> - Wrote `./RUST_BURN_REWRITE_PLAN.md` (+47 -3)
>
> - Ran `Get-Content RUST_BURN_REWRITE_PLAN.md | Select-Object -Skip 165 -First 115`
>
> - Ran ``$planPath = 'E:\SuperNormal_Rust_Burn_Framework_Rewrite\RUST_BURN_REWRITE_PLAN.md'
> $planText = [IO.File]::ReadAllText($planPath)
> $planText = $planText.Replace('| `sn-preprocess` | Capture conversion, XML/camera/normal conversion, mask tooling, optional Burn SAM inference. |', '')
> $planText = $planText.Replace('`evaluate`, `preprocess`, `convert-checkpoint`', '`evaluate`, `convert-checkpoint`')
> $planText = $planText.Replace('Treat video encoding and RAW decoding as separately packaged capabilities if portable native libraries are necessary.', 'Keep optional video encoding separate from reconstruction dependencies.')
> $planText = $planText.Replace('Preserve the branch''s exact grid resolution rounding, `(resolution+1)^3` dense sizing, x-fast dense layout, coordinate clamp to `[0, 1-1e-7]`, interpolation corner order and level activation.', 'First establish CUDA-compatible grid resolution rounding, table layout, coordinate transform, interpolation corner order and level activation from installed tiny-cuda-nn fixtures. The pure-PyTorch choices are candidate semantics, not the authority; retain its explicit layout only as a separately labeled compatibility mode if needed.')
> $planText = $planText.Replace('Compute exclusive segmented transmittance from `max(1-alpha, 1e-10)` and weight `T*alpha`.', 'Compute exclusive segmented transmittance and weight `T*alpha` with the CUDA-reference semantics established by fixtures; specifically test the forward-product versus backward-denominator clamp difference described above.')
> $planText = $planText.Replace('Default configuration has 11,878,328 hash parameters:', 'The pure-PyTorch layout under the default configuration has 11,878,328 hash parameters (recalculate for the validated tiny-cuda-nn-compatible layout):')
> $planText = $planText.Replace('## Meshes, metrics and preprocessing completeness', '## Meshes and metrics')
> $planText = [regex]::Replace($planText, '(?s)Preprocessing completion requires:.*?## Delivery phases and gates', 'Data preparation and SAM are deferred; no preprocessing milestone is required for this reconstruction release.' + [Environment]::NewLine + [Environment]::NewLine + '## Delivery phases and gates')
> $planText = $planText.Replace('obtain one benchmark dataset and representative RAW/model inputs.', 'obtain one prepared reconstruction dataset and establish main/sub2gb CUDA baselines.')
> $planText = $planText.Replace('Finite matching gradients on at least one Intel/AMD integrated adapter;', 'Finite matching gradients on the i3-1315U integrated adapter;')
> $planText = $planText.Replace('| 6. Remaining utilities | RAW and camera/normal conversion, SAM UI/model work, reports/video, portable CLI and packaging. | Repository-owned user workflows covered by a feature matrix; external input providers explicitly identified. |', '| 6. Reconstruction packaging | Portable CLI, prepared-data examples, checkpoint conversion, metrics/reports and release packaging. | Reconstruction workflow matrix complete; no data preparation dependencies. |')
> $planText = $planText.Replace('roughly 10–16 engineering weeks for core reconstruction and native desktop qualification, plus 4–8 or more for full preprocessing/SAM and additional platforms.', 'roughly 10–16 engineering weeks for reconstruction and native desktop qualification; preprocessing/SAM are excluded. Browser/mobile are later scope.')
> $planText = $planText.Replace('Use Python only to produce development reference fixtures.', 'Use the Python branches for baseline runs, component fixtures and regression comparisons during development; the shipped reconstruction application is Rust-only.')
> $planText = $planText.Replace('Windows Intel integrated graphics (DX12 and Vulkan where available)', 'the i3-1315U on Windows (DX12 and Vulkan where available)')
> $planText = $planText.Replace('all repository-owned workflows have Rust equivalents;', 'all in-scope reconstruction workflows have Rust equivalents;')
> $planText = $planText.Replace('Cloned and verified the requested branch/commit, inspected the application and preprocessing sources plus legacy dependency layout, calculated default hash memory, and checked current official Burn/WGPU documentation.', 'Cloned and verified pure_pytorch_version, fetched and compared main/sub2gb and their history, inspected reconstruction sources and CUDA patch kernels, calculated the PyTorch hash layout memory, checked official Burn/WGPU/Intel and Brush sources, and verified Torch/CUDA availability in the supplied environment.')
> [IO.File]::WriteAllText($planPath, $planText)
> Select-String -Path $planPath -Pattern 'SAM|preprocess|tiny-cuda|i3-1315U'``
>
> </details>
>
> The revised plan now uses the CUDA branches as the reference and treats the PyTorch version as a candidate to validate. It also removes preparation scripts and SAM from the current scope.
>
> The i3-1315U has the graphics API support needed for the planned WGPU path, so GPU-accelerated reconstruction on it is a reasonable target. I’ve made it a specific acceptance test: we’ll verify forward/backward kernels and a real reconstruction on that iGPU before calling it supported.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran ``$planPath = 'E:\SuperNormal_Rust_Burn_Framework_Rewrite\RUST_BURN_REWRITE_PLAN.md'
> $planText = [IO.File]::ReadAllText($planPath)
> $planText = $planText.Replace('CPU BVH/KD-tree evaluation. |' + [char]10 + [char]10 + '| `sn-cli`', 'CPU BVH/KD-tree evaluation. |' + [char]10 + '| `sn-cli`')
> [IO.File]::WriteAllText($planPath, $planText)
> Get-Content $planPath | Select-Object -Skip 94 -First 17``
> - Ran `git status --short --branch; git branch -r`
>
> </details>
</details>

I’ve updated the [rewrite plan](./RUST_BURN_REWRITE_PLAN.md) to cover **only reconstruction**. Data preparation and SAM are deferred.

**Yes—the i3-1315U’s iGPU is a suitable target for GPU acceleration through Burn/WGPU, using the same underlying approach as Brush.** It has Intel UHD graphics with DirectX 12.1 support, and Intel lists Vulkan 1.3 for that graphics family. Brush explicitly supports Intel GPUs through Burn and WebGPU-compatible technology. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/232136/intel-core-i31315u-processor-10m-cache-up-to-4-50-ghz/specifications.html), [API support](https://www.intel.com/content/www/us/en/support/articles/000005524/graphics.html), [Brush](https://github.com/ArthurBrussee/brush).

That establishes hardware suitability; confirming our application works requires testing its kernels and training on that laptop. I’ve made the i3-1315U a specific acceptance target. Training speed will depend on memory bandwidth, available RAM and workload—it won’t necessarily resemble Brush’s speed.

I fetched and compared both other branches:

| Branch | What I found |
|---|---|
| `main` | Retains tiny-cuda-nn and patched NerfAcc, with CPU-resident normal maps, checkpointing and chunked mesh extraction. |
| `sub2gb` | Adds 65,536-point SDF/occupancy chunks, forces 1,024 patches per batch, adds memory profiling and loads checkpoints through CPU. |
| `pure_pytorch_version` | Replaces encoding/rendering, but reverts the runner to `main`, losing several `sub2gb` changes. |

Your clarification changes the validation strategy: **the CUDA implementation should be our reference; the PyTorch version is a useful intermediate, not established ground truth.** I found differences worth checking immediately:

- PyTorch normalizes/clamps encoding coordinates; the CUDA branch passes scene coordinates directly to tiny-cuda-nn.
- PyTorch and the vendored CUDA patch-weight kernel handle transmittance clamping differently.

Neither proves poor reconstruction, but plausible-looking meshes alone won’t establish equivalence.

**Yes, providing an actual dataset will make a step-by-step rewrite substantially more faithful.** We can:

1. Record CUDA baseline reconstructions and intermediate results.
2. Compare camera rays, encoding, SDF, sampling and rendering individually.
3. Verify gradients and one optimizer update—not just forward outputs.
4. Replace each validated component with Rust.
5. Compare short training trajectories, complete meshes, quality metrics and memory use.

I checked your environment: PyTorch **2.8.0+cu129** loads and detects an **RTX 3060**. The required modules are discoverable; their custom kernels still need smoke tests.

Provide the prepared dataset’s local path and the config/command you normally use. A known-good checkpoint or mesh would also help; ground truth is useful but not required for comparison against the original.

> while i am running a full 30k iteration reconstruction on the pure_pytorch_version branch to verify that it works properly and produces a similar result to the baseline CUDA-only implementation, please explore the tiny-cuda-nn and nerfacc (look at the env setup script in the repository for the exact commits). What makes it so much faster than the pure pytorch version?
> By the way, i do understand that by rewriting the whole pipeline without these modules (tcnn and nerfacc) we are sacrificing their optimizations. Or do we?

<details><summary>4 previous messages</summary>

> I’ll inspect the pinned tiny-cuda-nn version and the vendored NerfAcc kernels, then compare their execution paths with this branch’s PyTorch replacements. That should distinguish optimizations we can preserve in Rust/WGPU from advantages specific to CUDA hardware.
>
> <details><summary>Ran 5 commands, searched the web</summary>
>
> - Ran `Get-Content create_env.sh; Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\cuda\csrc\ray_marching.cu -TotalCount 240; Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\grid.py | Select-Object -Skip 180 -First 150`
>
> - Ran `git clone https://github.com/NVlabs/tiny-cuda-nn.git tiny-cuda-nn-reference`
>
> - Ran `git checkout --detach 2ec562e853e6f482b5d09168705205f46358fb39; rg -n 'kernel_grid|atomicAdd|grid_scale|grid_resolution|pos_fract|N_FEATURES_PER_LEVEL|__half|interpolation_type|CoherentPrime' include/tiny-cuda-nn/encodings/grid.h; rg -n 'Encoding|preferred_precision|precision|loss_scale' bindings/torch/tinycudann/modules.py bindings/torch/tinycudann/bindings.cpp`
>
> - Ran `Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\ray_marching.py | Select-Object -Skip 150 -First 150; rg -n 'def accumulate|scatter_add|class _.*Alpha|patch_based' third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\vol_rendering.py; rg -n 'item\(|tolist\(|nonzero|cumprod|alphas_pad|index_select|checkpoint|range\(active' models/renderer.py models/fields.py`
>
> - Ran `Get-Content include\tiny-cuda-nn\encodings\grid.h | Select-Object -Skip 48 -First 125; Get-Content include\tiny-cuda-nn\encodings\grid.h | Select-Object -Skip 650 -First 85; Get-Content include\tiny-cuda-nn\encodings\grid.h | Select-Object -Skip 750 -First 60; rg -n 'grid_scale|grid_resolution|pos_fract|coherent_prime_hash|atomic_add_gmem' include\tiny-cuda-nn\common_device.h; Get-Content bindings\torch\tinycudann\modules.py | Select-Object -Skip 160 -First 36`
>
> Searched the web for `https://github.com/NVlabs/tiny-cuda-nn/tree/2ec562e853e6f482b5d09168705205f46358fb39`
>
> </details>
>
> The main speed advantage is specialized GPU kernels, not simply C++ versus Python. Also, this repository uses tiny-cuda-nn only for the hash encoding—the MLP remains PyTorch, so its “fully fused MLP” optimization isn’t responsible here.
>
> NerfAcc avoids building dense candidate-sample tensors and computes patch weights directly over packed samples. Those approaches can be carried into WGPU. One portability challenge is tiny-cuda-nn’s optimized gradient atomics, which need a different implementation on GPUs without equivalent support.
>
> <details><summary>Created a file, ran 2 commands</summary>
>
> - Ran `Get-Content include\tiny-cuda-nn\common_device.h | Select-Object -Skip 645 -First 78; Get-Content include\tiny-cuda-nn\common_device.h | Select-Object -Skip 823 -First 49; Get-Content include\tiny-cuda-nn\encodings\grid.h | Select-Object -Skip 244 -First 75; rg -n 'preferred_precision|network_precision_t' src\cpp_api.cu include\tiny-cuda-nn\common.h; git status --short --branch`
> - Ran `Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\cuda\csrc\ray_marching.cu | Select-Object -Skip 248 -First 65; Get-Content third_parties\nerfacc-0.3.5\nerfacc-0.3.5\nerfacc\vol_rendering.py | Select-Object -Skip 980 -First 34; git log -1 --format='%h %s' -- third_parties/nerfacc-0.3.5; git status --short`
> - Wrote `./KERNEL_PERFORMANCE_NOTES.md` (+79 -0)
>
> </details>
</details>

**We don’t have to sacrifice most of their optimizations.** We lose their ready-made CUDA implementations, but can reproduce many of the important techniques in portable GPU kernels. That is a different proposition from translating the current PyTorch operations directly into Burn operations.

I inspected tiny-cuda-nn at the exact setup-script commit, `2ec562e853e6f482b5d09168705205f46358fb39`. NerfAcc has no separate commit pinned there: the script installs the repository’s **locally modified 0.3.5 source**, including its patch-rendering kernels.

The main differences are:

| Area | CUDA baseline | Pure-PyTorch branch |
|---|---|---|
| Hash encoding | Specialized kernel combines hashing, corner lookup and interpolation | Many separate tensor operations and eight gathers per active level |
| Encoding backward | Custom kernel accumulates contributions directly into table gradients | Autograd traverses the individual gather/interpolation operations |
| Ray marching | Walks rays on the GPU, skips empty voxels and produces packed samples | Builds candidate-position tensors, tests occupancy, then filters them |
| Patch weights | Walks packed sample segments with custom forward/backward kernels | Pads segments to the longest ray, computes cumulative products, then gathers results |
| Execution overhead | Fewer, larger operations with specialized memory layouts | More intermediate tensors, dispatches and CPU/GPU synchronization |

These are source-based explanations; I haven’t measured their relative contribution or launched competing GPU benchmarks during your reconstruction.

**An important distinction: SuperNormal does not use tiny-cuda-nn’s fully fused MLP.** It uses `tcnn.Encoding`; the subsequent linear layers and Softplus remain PyTorch. So the famous fused-network/Tensor Core optimization is not the explanation for this particular difference.

For the encoder alone, 14 active levels mean **112 separate gather operations per SDF evaluation** in the PyTorch implementation, plus coordinate arithmetic, interpolation and backward. tiny-cuda-nn combines that work into specialized kernels, keeps intermediate values local, and arranges execution for cache locality. Its native encoding can also use half precision and optimized paired gradient atomics; calling `.to(torch.float32)` on its wrapper doesn’t necessarily make the internal kernel float32.

The padded rendering calculation is another substantial candidate. If rays average 100 samples but the longest has 2,000, a rectangular representation has **20 times as many slots** as the packed representation. That isn’t a prediction of 20× overall slowdown, but it illustrates the avoidable work.

There are limits to these explanations: the old NerfAcc implementation still synchronizes to allocate packed outputs, and final weighted accumulation already uses PyTorch `scatter_add_`. It isn’t an entirely fused pipeline.

**What we can preserve in Rust/WGPU:**

- Fused hash lookup and interpolation.
- Packed samples and empty-space skipping.
- Specialized patch-weight forward and backward.
- Buffer reuse, bounded memory and activation recomputation.
- Skipping inactive hash levels—something your PyTorch version already does usefully.

The harder parts are architecture-specific atomics, precision choices and kernel tuning. We’ll need portable alternatives and optional faster paths where supported. Burn supports custom operations and handwritten backward implementations, but doesn’t automatically invent these algorithms for us. [Burn’s extension documentation](https://burn.dev/books/burn/advanced/backend-extension/index.html).

So **“Rust + Burn” alone is not the optimization; “Burn + carefully designed portable GPU kernels” is.** I expect substantial opportunity to outperform the current eager PyTorch implementation, but matching tuned CUDA on the same NVIDIA GPU needs measurement. An iGPU’s lower bandwidth and power budget remain separate constraints.

I also confirmed additional encoding differences: tiny-cuda-nn uses a half-cell offset and different resolution/table-sizing conventions. Your 30k run is valuable evidence about reconstruction quality, but even a very similar mesh won’t establish that the intermediate encoding is equivalent.

The detailed findings and proposed profiling procedure are saved in [Kernel performance notes](./KERNEL_PERFORMANCE_NOTES.md).

> I have some excellent news! The full 30k iteration training run with the pure pytorch version of supernormal has just finished, and the mesh looks virtually identical to what i would get with the baseline implementation. The training process took only 3 times more time than the original implementation, which i find to be still acceptable (but please try to make it faster in the rust version by keeping the optimizations discussed above). Also, the VRAM consumption throughout the whole training process remained very low: the total system VRAM at just 2.6 GB tops, whereas with the main branch (the one with sub 4gb optimizations) the system VRAM is typically at 3.9 GB during training, the sub2gb branch further decreases the system VRAM to just 1.9 GB out of which the majority is due to idle driver and windows consumption (so i guess on an actual 2 GB GPU it would have run just fine). Although the sub2gb branch is slower and the quality is also a bit worse due to 1024 batch size.
> By the way, the original SuperNormal implementation consumed nearly all of my 12 GBs of VRAM and often crashed due to OOM. So the existing reimplementations alone (sub 4gb, sub 2 gb and pure pytorch) are already a very big deal. The problem is that it still requires manual VENV installation which is especially challenging on non-nvidia GPUs and impossible on many embedded mobile graphics chips (such as the one inside the i3-1315U chipset). I want a fully static binary that works absolutely everywhere (everywhere = everywhere where there is a DX12/Vulkan 1.3 capable GPU). Just like brush enables gaussian splatting reconstruction on wide spectrum of GPUs.
> Now, i have placed my dataset under the root project directory E:\SuperNormal_Rust_Burn_Framework_Rewrite. The command to launch the reconstruction is python exp_runner.py --conf .\config\own_objects.conf --obj_name horse_960. The data folder must be located within the same folder as the exp_runner.py (you can copy the dataset or create a symbolic link). Also, please check the own_objects.conf. As you might have noticed, the normal map validation in the pure pytorch version is broken, so make sure that val_normal_freq is set higher than end_iter (30001, for example). the save_freq dictates how often the checkpoint is saved, the val_mesh_freq dictates how often the mesh is extracted.
> Now, please proceed with the reimplementation in Rust: the full toolchain is already installed on the machine. Please perform extensive experiments to ensure that the rust reimplementation matches the original (but please don't launch the full 30k reconstruction unless it is strictly necessary). You are free to modify the code of the existing branches (just don't push anything to the remote please) or write new test python scripts for extra debug information.
> Additional info: you can find the results of the pure_pytorch_version test run in the pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06 folder, there are checkpoints and meshes exported every 5k iterations. The results of original supernormal execution on the same dataset are available original_supernormal_exp_horse_960 (note that there are results only for 5k, 10k and 30k iterations).
>
> Wait, actually, upon further inspection i noticed that the level of detail in meshes exported at 5k and 10k with pure pytorch version is worse than that of the original version. But the full 30k iteration meshlooks remarkably close to the original supernormal result. Not a big deal, i guess. Or may be even better for us if this peculiarity is directly related to reduced VRAM footprint.

<details><summary>7 previous messages</summary>

> I’ll start with the dataset, configs and saved checkpoints, then build and test the Rust pipeline component by component. I’ll use short training runs and checkpoint comparisons rather than a full 30k run.
>
> The slower emergence of detail is worth investigating: the changed hash-grid coordinates and level schedule could explain it, independently of memory savings. I’ll treat the 30k visual match as encouraging evidence, while checking the intermediate behavior explicitly. The packaging target will be a self-contained executable with no Python or CUDA installation; GPU driver and device-capability checks will still be necessary.
>
> <details><summary>Created 4 files, edited 2 files, ran 7 commands</summary>
>
> - Ran `Get-ChildItem -Force; rg --files --hidden --no-ignore -g AGENTS.md -g '!tiny-cuda-nn-reference/**' -g '!SuperNormal_memory_optimized/.git/**'; rustc --version; cargo --version`
> - Ran `Get-Content config\own_objects.conf; Get-Content exp_runner.py | Select-Object -Skip 633; Get-ChildItem ..\data; Get-ChildItem ..\pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06; Get-ChildItem "$env:USERPROFILE\.cargo\registry\src" -ErrorAction SilentlyContinue`
> - Ran `Get-ChildItem data\own_objects_normals\horse_960; Get-ChildItem pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06\checkpoints; Get-ChildItem original_supernormal_exp_horse_960 -Recurse -File | Select-Object FullName,Length -First 25`
> - Ran `Get-ChildItem "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f" -Name | Where-Object { $_ -match '^(burn|cubecl|wgpu|safetensors|ndarray-npy|marching|exr|hocon)' }; Get-Content "$env:USERPROFILE\.cargo\config.toml" -ErrorAction SilentlyContinue`
> - Wrote `./SuperNormal_memory_optimized/config/own_objects.conf` (+2 -2)
> - Wrote `./supernormal-rs/Cargo.toml` (+24 -0)
> - Wrote `./supernormal-rs/src/main.rs` (+11 -0)
> - Ran `cargo check`
> - Ran `Get-Content "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-core-0.22.0\src\optim\adam.rs" -TotalCount 110; rg -n 'pub fn (select|slice|scatter|mask_where|clamp|require_grad|grad|from_data|into_data|sum_dim)' "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src" | Select-Object -First 20`
> - Ran `Get-ChildItem "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src\tensor\api" -Name; rg -n 'cumprod|pub fn powf|pub fn powi|pub fn detach|pub fn floor' "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src\tensor\api"; New-Item -ItemType Junction -Path data -Target E:\SuperNormal_Rust_Burn_Framework_Rewrite\data`
> - Wrote `./supernormal-rs/src/field.rs` (+136 -0)
> - Wrote `./supernormal-rs/src/main.rs` (+11 -1)
> - Wrote `./supernormal-rs/tools/export_reference.py` (+53 -0)
>
> </details>
>
> The project now has a Burn/WGPU build and a development exporter for checkpoint-based numerical tests. I’ve also linked the dataset into the Python checkout and set `val_normal_freq` to `30001`.
>
> The first test will compare Rust SDF outputs and parameter gradients against your saved PyTorch checkpoints. That isolates the encoding and network from ray sampling, so any mismatch is easier to locate.
>
> <details><summary>Created 6 files, edited 4 files, ran 6 commands</summary>
>
> - Ran `$srcPath='E:\SuperNormal_Rust_Burn_Framework_Rewrite\supernormal-rs\src\field.rs'
> $s=[IO.File]::ReadAllText($srcPath)
> $s=[regex]::Replace($s,'(?<![\w.])\.(\d)','0.$1')
> [IO.File]::WriteAllText($srcPath,$s)`
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\export_reference.py --checkpoint ..\pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06\checkpoints\ckpt_030000.pth --out experiments\field_30k`
> - Ran `cargo check`
> - Wrote `./supernormal-rs/src/data.rs` (+65 -0)
> - Wrote `./supernormal-rs/src/render.rs` (+103 -0)
> - Wrote `./supernormal-rs/Cargo.toml` (+1 -0)
> - Wrote `./supernormal-rs/src/field.rs` (+12 -3)
> - Wrote `./supernormal-rs/src/main.rs` (+2 -0)
> - Ran `$p='src\field.rs'; $s=[IO.File]::ReadAllText((Join-Path (Get-Location) $p)); $s=$s.Replace("replace('.','_')",'replace(''.'',"_")').Replace('p.shape.iter().product()','p.shape.iter().product::<usize>()'); [IO.File]::WriteAllText((Join-Path (Get-Location) $p),$s)
> cargo check --offline`
> - Wrote `./supernormal-rs/.cargo/config.toml` (+2 -0)
> - Wrote `./supernormal-rs/src/field.rs` (+1 -1)
> - Wrote `./supernormal-rs/src/render.rs` (+1 -1)
> - Ran `rg -n 'name = "(marching|isosurface)' "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f" -g Cargo.toml --max-count 1`
> - Wrote `./supernormal-rs/src/field.rs` (+20 -0)
> - Wrote `./supernormal-rs/src/train.rs` (+82 -0)
> - Wrote `./supernormal-rs/tools/export_render_reference.py` (+58 -0)
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\export_render_reference.py --out experiments\render`
> - Wrote `./supernormal-rs/src/main.rs` (+17 -2)
> - Wrote `./supernormal-rs/src/mesh.rs` (+48 -0)
>
> </details>
>
> I’m building a packed renderer with explicit derivatives alongside the Burn field implementation. This gives us a reference that avoids padding every ray to the longest ray and lets field activations be recomputed in bounded chunks during backward.
>
> The initial packed compositor runs in Rust on the CPU; the neural field runs through WGPU. I’ll measure that split before claiming a speed improvement, then move costly operations into GPU kernels where the tests support it.
>
> <details><summary>Edited 2 files, ran 4 commands</summary>
>
> - Wrote `./supernormal-rs/src/render.rs` (+38 -0)
> - Ran `rg -n 'pub fn|pub async fn' "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src\device.rs" | Select-Object -First 50; Get-Content "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-wgpu-0.22.0\Cargo.toml" | Select-Object -First 125; Get-Content src\mc_table.rs -TotalCount 3`
> - Ran `Get-Content "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src\device\wgpu.rs" -TotalCount 150; Get-Content "$env:USERPROFILE\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-tensor-0.22.0\src\device.rs" | Select-Object -Skip 875 -First 98`
> - Wrote `./supernormal-rs/tools/export_render_reference.py` (+16 -2)
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\export_render_reference.py --out experiments\step_30k --checkpoint ..\pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06\checkpoints\ckpt_030000.pth`
> - Ran `cargo build --release --offline`
>
> </details>
>
> System error
>
> You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/settings/usage to purchase more credits or try again at 7:57 PM.
</details>

> proceed with the rust reimplementation. but i have to ask you to be a little more conservative with the token usage. we are burning through tokens really fast

<details><summary>21 previous messages</summary>

> I’ll continue with shorter updates and targeted checks. Next I’ll finish the build, run the numerical comparisons, and test a few training steps before adding more optimizations.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Test-Path target\release\supernormal.exe; Get-ChildItem experiments\step_30k -Name -ErrorAction SilentlyContinue | Select-Object -First 3; cargo fmt; cargo build --release --offline`
> - Ran `.\target\release\supernormal.exe doctor; .\target\release\supernormal.exe verify experiments\field_30k; .\target\release\supernormal.exe verify-render experiments\render --data ..\data\own_objects_normals\horse_960`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> The first comparisons passed: Rust matches the checkpoint’s field values within \(6×10^{-7}\), and the renderer’s gradients, camera rays, and loaded normals agree with Python. I’m now checking complete optimizer steps and short training runs.
>
> <details><summary>Edited 3 files, ran 10 commands</summary>
>
> - Ran `Get-Content src/train.rs; Get-Content src/main.rs; Get-Content src/render.rs | Select-Object -Last 170; Get-Content src/field.rs | Select-Object -Last 130`
> - Ran `Get-Content tools/export_render_reference.py | Select-Object -Last 70; Get-Content src/render.rs | Select-Object -First 180; Get-Content src/field.rs | Select-Object -First 175; rg --files -g AGENTS.md -g README.md -g '*LICENSE*' -g '*NOTICES*'`
> - Ran `rg -n 'bindwidth|update_learning|occ_eval|normal_loss|eikonal|near_far|alpha_thre|stratified' exp_runner.py models/renderer.py models/fields.py models/dataset_loader.py; rg -n 'march|t_min|mid|occ|update|warmup' models/torch*`
> - Ran `Get-Content models/renderer.py | Select-Object -Skip 125 -First 63; Get-Content models/renderer.py | Select-Object -Skip 408 -First 57; Get-Content models/renderer.py | Select-Object -Skip 628 -First 19; Get-Content exp_runner.py | Select-Object -Skip 140 -First 25; Get-Content config/own_objects.conf`
> - Wrote `./supernormal-rs/src/main.rs` (+5 -0)
> - Wrote `./supernormal-rs/src/render.rs` (+4 -1)
> - Wrote `./supernormal-rs/src/train.rs` (+4 -2)
> - Ran `Get-ChildItem -Path E:\SuperNormal_Rust_Burn_Framework_Rewrite -Filter AGENTS.md -Force; Get-Content E:\Projects\MVPS_Studio\.venv\Lib\site-packages\scikit_image-0.25.2.dist-info\licenses\LICENSE.txt -ErrorAction SilentlyContinue | Select-Object -First 50`
> - Wrote `./supernormal-rs/src/render.rs` (+40 -0)
> - Ran `rg --files E:/Projects/MVPS_Studio/.venv/Lib/site-packages -g '*LICENSE*' -g '*COPYING*' | Select-String 'skimage|scikit_image'`
> - Ran `.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --out experiments\smoke_initial --steps 2 --batch 8 --microbatch 8 --occupancy-resolution 32 --save-freq 1`
> - Ran `Get-ChildItem E:\Projects\MVPS_Studio\.venv\Lib\site-packages -Directory -Filter '*image*'; Get-Content src/data.rs | Select-Object -First 140; Get-Content Cargo.toml`
> - Ran `Get-Content models/renderer.py | Select-Object -Skip 650 -First 145; Get-Content models/renderer.py | Select-Object -Skip 930 -First 170; Get-Content models/dataset_loader.py | Select-Object -Skip 125 -First 105`
> - Ran `Get-Content E:\Projects\MVPS_Studio\.venv\Lib\site-packages\scikit_image-0.21.0.dist-info\LICENSE.txt | Select-Object -First 95`
>
> </details>
>
> The native program has completed its first two training steps and saved checkpoints. I also fixed resume behavior so restarting a run won’t introduce an extra occupancy update. Next are the full optimizer comparison and mesh extraction checks.
>
> <details><summary>Ran 8 commands</summary>
>
> - Ran `Get-Process cargo,rustc,link -ErrorAction SilentlyContinue | Select-Object Name,Id,CPU,StartTime; Get-Content E:\Projects\MVPS_Studio\.venv\Lib\site-packages\scikit_image-0.21.0.dist-info\LICENSE.txt | Select-Object -Skip 101 -First 36; Get-Content ..\SuperNormal_memory_optimized\LICENSE* -ErrorAction SilentlyContinue | Select-Object -First 30`
> - Ran `cargo fmt; cargo build --release --offline`
> - Ran `Get-ChildItem target\release\supernormal.exe | Select-Object Length,LastWriteTime; Get-Content src/mesh.rs | Select-Object -First 180; Get-Content .cargo/config.toml; Get-ChildItem experiments -Directory | Select-Object Name`
> - Ran `rg -n 'pub fn (identity|enumerate|wgpu)|pub enum WgpuDevice|pub struct DeviceIdentity|pub enum DeviceType|fn memory_pool_usage' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-*0.22.0/src C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-* -g '*.rs'; Get-Content tools/export_reference.py | Select-Object -First 75`
> - Ran `.\target\release\supernormal.exe mesh --model experiments\field_30k --out experiments\horse_30k_96.ply --resolution 96 --query 16384`
> - Ran `Get-Content models/renderer.py | Select-Object -Skip 588 -First 44; Get-Content models/renderer.py | Select-Object -Skip 790 -First 100; Get-Content exp_runner.py | Select-Object -Skip 279 -First 38; rg -n 'pub fn (identity|enumerate|wgpu)|pub enum WgpuDevice|pub enum DeviceType|fn memory_pool_usage' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f -g '*.rs' -g '!windows*' -m 2 | Select-Object -First 25`
> - Ran `Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/wgpu.rs | Select-Object -First 100; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.11.0/src/device/wgpu.rs | Select-Object -First 105; rg -n 'identity|enumerate|pub use.*Device' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/device.rs`
> - Ran `Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.11.0/src/device/wgpu.rs | Select-Object -First 68; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/device.rs | Select-Object -Skip 1158 -First 35; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/wgpu.rs | Select-Object -Skip 100 -First 22; Get-Content ..\SuperNormal_memory_optimized\models/renderer.py | Select-Object -Skip 892 -First 24`
>
> </details>
>
> horse_30k_96.ply does indeed look like a horse!!!
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./supernormal-rs/src/main.rs` (+18 -3)
> - Wrote `./supernormal-rs/src/train.rs` (+5 -1)
>
> </details>
>
> That confirms the checkpoint import and native mesh extraction are producing recognizable geometry. This mesh uses your trained Python checkpoint; the next milestone is verifying that Rust’s own training updates preserve the same behavior.
>
> <details><summary>Created a file, edited 2 files, ran 4 commands</summary>
>
> - Ran `Copy-Item -LiteralPath ..\SuperNormal_memory_optimized\LICENSE -Destination LICENSE; Copy-Item -LiteralPath E:\Projects\MVPS_Studio\.venv\Lib\site-packages\scikit_image-0.21.0.dist-info\LICENSE.txt -Destination THIRD_PARTY_NOTICES.md`
> - Ran `cargo fmt; cargo check --offline`
> - Wrote `./supernormal-rs/tools/validate_native.py` (+91 -0)
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\validate_native.py --model experiments\field_30k --fixture experiments\step_30k --out experiments\actual_step_30k --mesh experiments\horse_30k_96.ply`
> - Wrote `./supernormal-rs/src/render.rs` (+11 -0)
> - Wrote `./supernormal-rs/src/train.rs` (+7 -3)
> - Ran `Get-Content src/field.rs | Select-Object -Skip 193 -First 30; Get-Content src/render.rs | Select-Object -Skip 195 -First 120`
>
> </details>
>
> I’m adding endpoint reuse so neighboring samples can share field evaluations without changing the gradients. The next checks compare a full Rust training update against the repository’s renderer and measure mesh agreement numerically.
>
> <details><summary>Edited a file, ran 4 commands</summary>
>
> - Ran `rg -n 'name = "supernormal"' Cargo.lock; Get-Content experiments/smoke_initial/metrics.csv; Get-ChildItem experiments/actual_step_30k -Filter report.json -ErrorAction SilentlyContinue; rg -n 'pub fn sync' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/device.rs`
> - Ran `cargo rustc --release --offline --bin supernormal -- -C lto=off -C codegen-units=16`
> - Ran `Get-Content src/data.rs | Select-Object -Skip 140 -First 65; Get-Content experiments/actual_step_30k/report.json; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-tensor-0.22.0/src/device.rs | Select-Object -Skip 768 -First 18`
> - Wrote `./supernormal-rs/src/data.rs` (+4 -4)
> - Ran `.\target\release\supernormal.exe --help`
>
> </details>
>
> The mesh comparison passed: Rust and Python produced the same 4,872 vertices and 9,740 triangles at resolution 96, with a maximum vertex discrepancy below \(0.000019\) in normalized coordinates. I’m finishing the training-update checks and adding explicit integrated-GPU selection.
>
> <details><summary>Created 2 files, edited 3 files, ran 5 commands</summary>
>
> - Ran `Get-Process | Where-Object { $_.ProcessName -match 'cargo|rustc|link|supernormal' } | Select-Object ProcessName,Id,CPU,StartTime; Get-Item target/release/supernormal.exe | Select-Object Length,LastWriteTimeUtc; Get-ChildItem target/release/deps -Filter '*supernormal*' | Select-Object Name,LastWriteTimeUtc,Length`
> - Wrote `./supernormal-rs/.gitignore` (+5 -0)
> - Wrote `./supernormal-rs/README.md` (+61 -0)
> - Wrote `./supernormal-rs/src/field.rs` (+12 -0)
> - Ran `Get-Content src/render.rs | Select-Object -Skip 448 -First 85; Get-Content src/train.rs | Select-Object -Skip 155 -First 40`
> - Ran `Get-Content src/render.rs | Select-Object -Skip 392 -First 62`
> - Wrote `./supernormal-rs/src/render.rs` (+9 -14)
> - Wrote `./supernormal-rs/src/train.rs` (+1 -16)
> - Ran `.\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k`
> - Ran `rg -n 'optimizer|params_to_train' ..\SuperNormal_memory_optimized\exp_runner.py | Select-Object -First 9; Get-Content tools/export_reference.py | Select-Object -Skip 30 -First 19`
>
> </details>
>
> The full renderer loss agrees within about \(1.3×10^{-6}\). The fresh-Adam comparison exposed a sensitivity near zero gradients: tiny numerical differences can produce noticeable first updates. I’m testing continuation with the checkpoint’s saved optimizer history as well, and retaining the fresh-Adam discrepancy in the validation notes.
>
> <details><summary>Created a file, edited 4 files, ran 10 commands</summary>
>
> - Wrote `./supernormal-rs/src/train.rs` (+1 -1)
> - Wrote `./supernormal-rs/tools/export_reference.py` (+10 -1)
> - Wrote `./supernormal-rs/tools/validate_native.py` (+9 -1)
> - Ran `cargo fmt; cargo build --release --offline > experiments\build.log 2>&1`
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\export_reference.py --checkpoint ..\pure_pytorch_exp\own_objects\horse_960\exp_2026_10_19_15_27_06\checkpoints\ckpt_030000.pth --out experiments\field_30k`
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\validate_native.py --model experiments\field_30k --fixture experiments\step_30k --out experiments\actual_step_30k`
> - Ran `Get-Content experiments/build.log -Tail 8; .\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k > experiments\optimizer_comparison.log 2>&1`
> - Wrote `./supernormal-rs/tools/inspect_binary.py` (+26 -0)
> - Ran `Get-Content experiments/optimizer_comparison.log -Tail 24; & 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\inspect_binary.py target\release\supernormal.exe`
> - Ran `.\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k > experiments\optimizer_comparison.log 2>&1; Get-Content experiments\optimizer_comparison.log -Tail 23`
> - Ran `cargo check --offline --quiet`
> - Ran `Copy-Item -LiteralPath target\release\supernormal.exe -Destination experiments\supernormal_before_reuse.exe; .\experiments\supernormal_before_reuse.exe train --data ..\data\own_objects_normals\horse_960 --model experiments\field_30k --out experiments\late_full_batch_before_reuse --steps 2 --end-iter 30002 --microbatch 64 --query 32768 > experiments\late_before.log 2>&1`
> - Wrote `./supernormal-rs/src/render.rs` (+12 -2)
> - Ran `cargo fmt; cargo check --offline --quiet; Get-Content experiments/late_before.log -Tail 5; Get-Content experiments/build.log -Tail 5`
> - Ran `cargo build --release --offline > experiments\build_final.log 2>&1`
>
> </details>
>
> The executable’s imports contain only Windows system libraries—no Python, CUDA, or Visual C++ runtime DLL dependency. A two-step test at the full 2,048-patch batch size is running now, including occupancy rebuilding from your 30k checkpoint.
>
> <details><summary>Edited a file, ran 5 commands</summary>
>
> - Ran `Get-Content experiments\late_before.log -Tail 8; Get-Content experiments\build.log -Tail 4; Get-Content experiments\build_final.log -Tail 4`
> - Ran `& 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\export_reference.py --out experiments\field_init --active 1`
> - Ran `Get-Content experiments\late_before.log -Tail 5; Get-Content experiments\build.log -Tail 3`
> - Wrote `./supernormal-rs/src/train.rs` (+3 -0)
> - Ran `cargo fmt; & 'E:\Projects\MVPS_Studio\.venv\Scripts\python.exe' -B tools\validate_native.py --model experiments\field_init --fixture experiments\render --out experiments\actual_step_init`
> - Ran `.\experiments\supernormal_before_reuse.exe train --data ..\data\own_objects_normals\horse_960 --model experiments\field_30k --out experiments\late_larger_chunks_before_reuse --steps 2 --end-iter 30002 --microbatch 256 --query 131072 > experiments\late_larger_before.log 2>&1`
>
> </details>
>
> Both full-batch continuation steps completed with finite losses. The conservative chunk settings are still slow—about 47 seconds for the second step—so I’m testing larger chunks and endpoint reuse. Speed parity is still an open task.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content experiments\build.log -Tail 4; Get-Content experiments\build_final.log -Tail 4; Get-Content experiments\late_larger_before.log -Tail 4`
>
> </details>
</details>

> please continue

<details><summary>7 previous messages</summary>

> I’ll continue with the memory checks first: the larger-chunk experiment hit a WGPU out-of-memory error. I’ll keep the conservative settings while testing endpoint reuse and checkpoint resume.
>
> <details><summary>Edited 2 files, ran 8 commands</summary>
>
> - Ran `Get-Content experiments/build_final.log -Tail 6; Get-Content experiments/late_larger_before.log -Tail 12; Get-Content src/main.rs | Select-Object -First 65; Get-Content src/train.rs | Select-Object -Last 25`
> - Ran `.\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k > experiments\optimizer_final.log 2>&1; Get-Content experiments\optimizer_final.log -Tail 8`
> - Ran `.\target\release\supernormal.exe --api dx12 doctor; .\target\release\supernormal.exe --api vulkan doctor; .\target\release\supernormal.exe verify-step experiments\field_init experiments\actual_step_init > experiments\optimizer_init.log 2>&1; Get-Content experiments\optimizer_init.log -Tail 5`
> - Ran `.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model experiments\field_30k --out experiments\late_full_batch_reuse --steps 2 --end-iter 30002 --microbatch 64 --query 32768 > experiments\late_reuse.log 2>&1`
> - Ran `Get-Content src/render.rs | Select-Object -Skip 285 -First 109; Get-Content experiments\optimizer_init.log -First 14; Get-Content experiments\late_reuse.log -Tail 6`
> - Wrote `./supernormal-rs/src/render.rs` (+8 -0)
> - Wrote `./supernormal-rs/src/train.rs` (+3 -0)
> - Ran `rg -n 'select_assign|select_backward' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-autodiff-0.22.0/src/ops/tensor.rs -m 8; Get-Content experiments/late_reuse.log -Tail 6; cargo fmt; cargo check --offline --quiet`
> - Ran `Copy-Item -LiteralPath target/release/supernormal.exe -Destination experiments/supernormal_reuse.exe; cargo build --release --offline > experiments/build_bias_fix.log 2>&1`
> - Ran `Get-Content experiments/late_reuse.log -Tail 7; .\experiments\supernormal_reuse.exe --api dx12 verify experiments\field_30k > experiments\dx12_field.log 2>&1; Get-Content experiments\dx12_field.log -Tail 3`
>
> </details>
>
> Endpoint reuse reduced the second full-batch step from 47 to 31 seconds with essentially unchanged losses. The memory report also showed an important issue: only 143 MB was live, but Burn’s allocator retained about 5.7 GB. I’ll address that before calling the implementation suitable for low-memory GPUs.
>
> <details><summary>Created a file, edited 3 files, ran 8 commands</summary>
>
> - Ran `rg -n 'pub (struct|enum) MemoryConfiguration|enum MemoryPoolType|pub struct.*Memory|pub fn memory_cleanup' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-* -g '*.rs'`
> - Ran `rg -n 'pub (struct|enum) MemoryConfiguration|pub enum MemoryPoolType|pub struct MemoryPool' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f -g '*.rs' -m 2 | Select-Object -First 16; Get-Content src/field.rs | Select-Object -Skip 292 -First 43`
> - Ran `Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.11.0/src/memory_management/policy.rs | Select-Object -First 210; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-backend-0.22.0/src/backend/memory_pools.rs | Select-Object -First 70`
> - Ran `rg -n 'fn (sync|memory_cleanup|memory_pool_usage)' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-fusion-0.22.0/src/backend.rs; rg -n 'fn memory_cleanup' C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.11.0/src -g '*.rs'; Get-Content src/field.rs | Select-Object -Skip 289 -First 45; Get-Content src/main.rs | Select-Object -Skip 62 -First 20`
> - Ran `Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-fusion-0.22.0/src/backend.rs | Select-Object -Skip 150 -First 29; Get-Content C:/Users/Pavel/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.11.0/src/compute/server.rs | Select-Object -Skip 705 -First 25; Get-Content experiments/dx12_field.log -Tail 3`
> - Wrote `./supernormal-rs/src/field.rs` (+16 -0)
> - Wrote `./supernormal-rs/src/train.rs` (+2 -0)
> - Ran `cargo fmt; cargo check --offline --quiet; cargo build --release --offline > experiments\build_memory.log 2>&1`
> - Ran `rg -n 'peak_reserved|trim_memory|drop\(grads\)' src/field.rs src/train.rs; Get-Content experiments/build_bias_fix.log -Tail 3; Get-Content experiments/build_memory.log -Tail 3`
> - Wrote `./supernormal-rs/src/render.rs` (+7 -0)
> - Wrote `./supernormal-rs/tools/compare_checkpoints.py` (+21 -0)
> - Ran `cargo fmt; cargo check --offline --quiet`
>
> </details>
>
> I found the fresh-Adam discrepancy: floating-point cancellation was creating a small false gradient for the SDF’s output bias. Directional differences are invariant to a constant bias shift, so I’ve made that derivative explicit. DirectX 12 and Vulkan also passed the device checks; the DX12 field and gradient comparison passed.
>
> <details><summary>Created a file, ran 2 commands</summary>
>
> - Ran `Get-Content experiments/build_bias_fix.log -Tail 3; Get-Content experiments/build_memory.log -Tail 3; Get-Content experiments/optimizer_final.log | Select-String 'combined update max_abs' | Select-Object -First 4`
> - Wrote `./supernormal-rs/VALIDATION.md` (+31 -0)
> - Ran `Get-Content experiments/build_bias_fix.log -Tail 3; Get-Content experiments/build_memory.log -Tail 3`
>
> </details>
>
> System error
>
> You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/settings/usage to purchase more credits or try again at Oct 20th, 2026 1:00 AM.
</details>

# SuperNormal reconstruction in Rust + Burn

Native reconstruction for the `pure_pytorch_version` branch's `own_objects.conf` configuration. Python, CUDA, tiny-cuda-nn and nerfacc are not runtime dependencies. Development comparison scripts in `tools/` use the existing Python environment.

Implemented: EXR/PNG/NPZ ingestion, camera decomposition and patch rays, progressive hash encoding, weight-normalized SDF MLP, occupancy updates, sphere marching, visibility filtering, directional finite differences, packed normal/mask integration, all three training losses, analytical renderer and MLP backward, WGPU hash-grid backward, Adam, native checkpoint/resume, and streaming marching cubes with largest-component filtering.

The neural field, hash-grid interpolation and gradient scatter, and optimizer use WGPU. Ray sampling and the analytical compositor run on CPU, with point uploads and SDF readbacks between stages. Field inference for visibility and occupancy also uses the packed WGPU hash kernel. The earlier native training run was cancelled; short continuations from the 5k, 10k, and 30k Python checkpoints are measured in `VALIDATION.md`. Sustained speed and integrated-GPU performance remain to be validated.

## Build and diagnose

```powershell
cargo build --release --locked
.\target\release\supernormal.exe doctor
.\target\release\supernormal.exe --api dx12 doctor
.\target\release\supernormal.exe --api vulkan doctor
.\target\release\supernormal.exe --integrated doctor
```

Windows MSVC builds link the C runtime statically via `.cargo/config.toml`. The executable still uses the operating system and installed graphics driver. Each OS/architecture requires its own build. Intel UHD in the i3-1315U is an intended WGPU target; successful execution on that specific laptop remains to be verified. `--integrated` selects the first integrated adapter. `--cpu` uses Burn Flex for debugging.

## Train

Run from this directory. The original dataset layout is consumed directly; there is no preprocessing step.

```powershell
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --out runs\horse --steps 100 --save-freq 100
```

The defaults preserve the reference's 2,048 patches, 3x3 patch size, 30,000-update schedule, learning-rate warmup, occupancy resolution 128 and progressive bandwidth. `--steps` limits additional updates without compressing that schedule. `--microbatch` and `--query` bound working memory while accumulating the complete batch gradient before one Adam update. Adjacent samples share exactly equal endpoints. Normal maps stay in host FP16 storage; GPU arithmetic uses FP32.

Resume into a fresh output directory:

```powershell
.\target\release\supernormal.exe train --data ..\data\own_objects_normals\horse_960 --model runs\horse\checkpoint_000100 --out runs\horse_continued --steps 100
```

Native checkpoints contain weights, Adam moments and counters, occupancy and RNG state. Model-only imports lack some of that state and print a warning. The native schedule continues globally on resume; the Python runner's local-loop schedule reset is intentionally not reproduced. Changing batch size or schedule on exact resume is rejected.

Mesh export uses normalized coordinates by default. Add `--data` to apply the dataset's world scale and offset.

```powershell
.\target\release\supernormal.exe mesh --model runs\horse\checkpoint_000100 --out runs\horse\mesh.ply --resolution 256
```

`--save-freq` controls checkpoints; `--mesh-freq` and `--mesh-resolution` enable periodic meshes. Normal-image validation is currently omitted, including the broken Python validation path.

## Reproduce comparisons

The local `experiments/` directory contains exported fixtures. These are development artifacts, not required for normal use.

```powershell
.\target\release\supernormal.exe verify experiments\field_30k
.\target\release\supernormal.exe verify-render experiments\render --data ..\data\own_objects_normals\horse_960
.\target\release\supernormal.exe verify-step experiments\field_30k experiments\actual_step_30k
```

`export_reference.py` exports the pure-PyTorch field. `export_render_reference.py` supplies independent differentiable compositor fixtures. `validate_native.py` additionally calls the repository's actual renderer with fixed samples and compares native meshes with Python marching cubes. Fixing samples isolates numerical correctness from different RNG implementations.

Current scope is the supplied DFD/L2, 14-level hash-grid configuration. Arbitrary HOCON settings, FD/AD training, preprocessing and original tiny-cuda-nn checkpoint layouts are not supported. An imported Python checkpoint proving inference parity does not by itself prove native training convergence or a speedup. See `VALIDATION.md` for measured results and outstanding validation.

The original reconstruction implementation is MIT licensed; see `LICENSE`. The marching-cubes table comes from scikit-image; see `THIRD_PARTY_NOTICES.md`.

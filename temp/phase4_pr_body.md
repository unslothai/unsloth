## Summary

Phase 4 of porting the richer diffusion stack onto the image-generation backend, building on #6675 (Phase 2: memory / speed / precision) and #6670 (Phase 1: device policy). This adds the **native stable-diffusion.cpp engine**, the CPU and Apple-Silicon tier of a two-engine strategy that mirrors the chat backend's llama.cpp shell-out.

Diffusers stays the default and only path on CUDA / ROCm / XPU. This engine exists for the hardware diffusers serves poorly (CPU, Apple MPS, and very low VRAM budgets), and it consumes the same split GGUF assets Studio already curates for the diffusers path, so a model loaded for one engine needs no re-download for the other.

Everything is additive: new modules plus a standalone installer and smoke script. Nothing in the existing diffusers path changes, so this is a no-op until the native engine is selected. Stacked on #6675; I will rebase onto main and retarget the base once the lower phases land.

## What it does

- Pure sd-cli command builder (`sd_cpp_args.py`). Maps each family to its text-encoder flag (Z-Image's Qwen3 to `--llm`, Qwen-Image's Qwen2-VL to `--qwen2vl`, FLUX.1's CLIP-L + T5 to `--clip_l` / `--t5xxl`) and the diffusers memory policy (`none` / `group` / `model` / `sequential`) to sd.cpp's own offload flags (`--offload-to-cpu`, `--clip-on-cpu`, `--vae-on-cpu`, `--vae-tiling`, `--diffusion-fa`). One user-facing memory knob drives both engines identically.
- The engine (`sd_cpp_engine.py`). `SdCppEngine` over a located `sd-cli`: a `find_sd_cpp_binary()` with the same precedence as the llama finder (env override, then the Studio install root, then an in-tree build, then PATH), an `is_available` / `version` probe, and a one-shot subprocess `generate` that streams progress and returns the written PNG. `runtime_env()` prepends the binary's own directory to the platform library path (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH`) so a prebuilt's bundled `libstable-diffusion.so` resolves. `select_diffusion_engine()` is the pure routing decision: GPU backends stay on diffusers, CPU / MPS take the native engine when its binary is present, and a `prefer_native` override can force it anywhere.
- Prebuilt installer (`install_sd_cpp_prebuilt.py`). Resolves and downloads the per-host stable-diffusion.cpp release zip (macOS-arm64 / Metal, Linux x86_64 CPU, plus Vulkan / ROCm / Windows variants) into the Studio install root, where the finder picks it up. `resolve_release_asset()` is a pure, unit-tested host-to-asset matrix, so Apple Silicon and CPU users get a working binary with nothing to compile.
- Smoke harness (`scripts/sd_cpp_smoke.py`). Drives the real engine over a built or installed `sd-cli` and a set of split GGUF assets, the native analogue of `scripts/diffusion_bench.py`.

## Verification (single B200 box)

Built `sd-cli` from source (CUDA) and installed the prebuilt (CPU), then generated Z-Image-Turbo Q4_K end to end through `SdCppEngine`, all producing coherent images:

| binary | memory mode | offload flags | generation |
| --- | --- | --- | --- |
| source (CUDA) | `balanced` | `--offload-to-cpu --diffusion-fa` | 5.0 s |
| source (CUDA) | `low_vram` | `--offload-to-cpu --clip-on-cpu --vae-on-cpu --vae-tiling --diffusion-fa` | 13.4 s |
| prebuilt (CPU) | `low_vram` | (same) | 50.4 s on CPU |

The CPU prebuilt run exercises the dynamically-linked path and confirms `runtime_env()` lets the bundled shared library load. The installer was verified live: `resolve_release_asset` selects the correct asset for Linux (CPU / Vulkan / ROCm), macOS-arm64, and Windows (avx2 / cuda12 / vulkan) against a real release, and a real download + extract produced a runnable `sd-cli`.

## Tests

CPU only, with subprocess and the filesystem stubbed, run from `studio/backend`:

```
python -m pytest tests/test_sd_cpp_args.py tests/test_sd_cpp_engine.py tests/test_sd_cpp_install.py
```

49 new tests covering the command builder, the offload and family mappings, the binary finder precedence, the version probe, `generate` (success, nonzero exit, missing output, missing binary), `runtime_env`, the engine routing matrix, and the installer's host-to-asset resolver. The full diffusion suite is 166 passing.

## Out of scope (later)

Wiring the route / backend to dispatch to the native engine (asset resolution for the split VAE / text-encoder files, status reporting which engine is active) and the image-to-image / editing / LoRA / upscale / video feature surface. This PR lands the engine, installer, and routing decision as a self-contained, tested unit.

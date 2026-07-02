## Summary

Wires the native stable-diffusion.cpp engine into the **live** diffusion route so that when no usable
CUDA / ROCm / XPU GPU is present (CPU, and Apple MPS when explicitly enabled), generation runs on the
native `sd-cli` engine instead of diffusers, with diffusers as the guaranteed fallback. Stacked on the
Phase 15 pre-quant pass (#6717).

The native engine has shipped since Phase 4 (#6679) but was never reachable from the route, which always
drove the diffusers `DiffusionBackend`. So a CPU / Mac user got the slow, RAM-heavy diffusers path even
though the faster engine was already in the tree. Measured on this box (192-core CPU, same Q4_K_M
transformer GGUF), the native engine is the right CPU engine:

| model (CPU, 512px) | diffusers | sd.cpp | sd.cpp speed | sd.cpp RAM |
| --- | ---: | ---: | ---: | ---: |
| FLUX.1-schnell | 98.5s | 60.3s | **1.6x** | **1.6x less** |
| Z-Image-Turbo | 99.1s | 71.9s | **1.4x** | **1.5x less** |
| Qwen-Image | 243.8s | 88.4s | **2.8x** | **2.2x less** |

(FLUX.1 CPU vs ComfyUI: sd.cpp 60.3s / 18.9 GB beats ComfyUI 102.4s / 71.2 GB too.) The decision
function `select_diffusion_engine` already existed and was unit-tested; this PR is the routing and
lifecycle integration around it, not a rewrite. The diffusers path stays the default and only path on
CUDA / ROCm / XPU, and the universal fallback.

## What it does

- **`diffusion_engine_router.py` (new).** Centralised engine selection at load time, remembered for the
  rest of the load so generate / unload / status / progress all act on the same engine. Built on
  `select_diffusion_engine(backend, ...)`; the device backend comes from
  `resolve_diffusion_device_target().backend`. Records why a fallback to diffusers happened and exposes it
  in status. Env knobs (one canonical interpretation each):
  - `UNSLOTH_DIFFUSION_ENGINE=auto|diffusers|sd_cpp` force an engine
  - `UNSLOTH_DIFFUSION_SD_CPP=auto|0|1` enable / disable the native route
  - `UNSLOTH_DIFFUSION_SD_CPP_MPS=0|1` allow native on Apple MPS (default off)
  - `UNSLOTH_DIFFUSION_SD_CPP_INSTALL=auto|0|1` allow lazy binary install

- **`sd_cpp_backend.py` (new) `SdCppDiffusionBackend`.** Mirrors the public surface the route uses on the
  diffusers backend (`begin_load` / `load_progress` / `generate` / `generate_progress` / `unload` /
  `status`), backed by `SdCppEngine`. It lazily installs the `sd-cli` binary on first use, fetches the
  per-family single-file assets, runs the load on a daemon thread with a download-progress phase, parses
  sd-cli step lines for per-step progress, and supports cancellation. Import-light: no torch / diffusers,
  so selecting it on a CPU box does not pull the GPU stack.

- **`diffusion_families.py`.** Each family gains its native single-file asset mapping (`sd_cpp_vae`,
  `sd_cpp_text_encoders`, `sd_cpp_vae_format`) using the same hashable-tuple pattern as `prequant_repos`.
  The transformer GGUF is reused from the diffusers download path; only the single-file VAE + text
  encoders are new fetches (the diffusers base repo ships those sharded, which sd-cli cannot read). Z-Image
  and FLUX.2-klein use Qwen3-4B, FLUX.1 uses CLIP-L + T5, Qwen-Image uses Qwen2.5-VL; FLUX.2 uses the
  `flux2` VAE latent format.

- **`sd_cpp_engine.py`.** Adds cancellation to `generate` / `upscale`: an optional `cancel_event` polled
  while the child runs, plus a process-group kill (`SdCppCancelled`) so a superseding load / unload /
  arbiter eviction can hard-stop the subprocess and its children. User cancellation does not trigger a
  diffusers fallback.

- **`routes/inference.py` + `gpu_arbiter.py`.** The load handler resolves the device, selects + activates
  the engine before evicting chat (so a fallback never strands a half-native load), and the other handlers
  and the arbiter evictor act on the active engine via the router. The status response gains `engine` and
  `fallback_reason`.

## Selection and fallback

Selection is deterministic and happens before the slow load. Diffusers is chosen (with a recorded reason)
whenever the native route is disabled, the device has a usable GPU, MPS is not enabled, the family has no
native asset mapping, or the `sd-cli` binary is unavailable. Scope is text-to-image (the route's only
mode); image-to-image / edit / LoRA are not exposed there.

## Testing

- `test_sd_cpp_backend.py` (new): asset resolution per family, guidance mapping, status shape, generate
  returns images with per-image seeds, `--vae-format` for FLUX.2, cancellation surfaces as cancelled (not
  a crash), progress parsing, load validation, lazy-install gating, unload semantics.
- `test_diffusion_engine_router.py` (new): the full selection matrix (cpu -> sd_cpp, gpu -> diffusers,
  opt-out -> diffusers, MPS default vs opt-in, unsupported family, missing binary, forced sd_cpp), and the
  status annotation.
- `test_diffusion_routes.py`: a route-level test asserting a CPU host with an available binary reports
  `engine=sd_cpp`; the existing fixture now drives the router transparently.
- `test_sd_cpp_engine.py`: the two no-binary tests are now hermetic (forced no-binary) so they pass
  regardless of a locally installed `sd-cli`.

All 313 diffusion / sd.cpp tests pass. Verified end-to-end on CPU (CUDA hidden): the router selected the
native engine, fetched the registry assets, and `sd-cli` produced an image, with `status.engine == sd_cpp`.

## Notes

- Everything is additive and opt-out-able: on a CUDA box this is a no-op (diffusers as before).
- The single-file asset repos are pinned to public, verified sources (Comfy-Org / black-forest-labs /
  comfyanonymous / unsloth GGUF). sd-cli flag construction stays isolated in `sd_cpp_args.py` (already
  test-covered), since upstream notes its CLI flags can change.

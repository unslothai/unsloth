# Plan: Unsloth Studio diffusion workflows + Images UI redesign

Stacked as NEW PRs on top of the existing 14-PR diffusion stack above unslothai/unsloth#6658
(treated as one logical base). Goal: cover ~80% of common real-world diffusion workflows,
across macOS/Linux/Windows/CPU, optimizing performance, accuracy, and memory.

## Current state (verified by recon)

- Backend is **text-to-image only** end-to-end. No image/mask/control plumbing in
  `DiffusionGenerateRequest`, `/images/generate`, or `DiffusionBackend.generate()`.
- **diffusers 0.38.0 already imports every pipeline we need**: `*Img2ImgPipeline`,
  `*InpaintPipeline`, `FluxFillPipeline`, `*ControlNetPipeline`/`ControlNetModel`,
  `FluxKontextPipeline`, `QwenImageEditPipeline`/`QwenImageEditPlusPipeline`,
  `StableDiffusion(Latent)UpscalePipeline`. No diffusers upgrade required.
- The native **sd.cpp engine already has dormant fields** (`init_img`, `strength`, `mask`,
  `ref_images`) and a complete `upscale()` path — never wired to the request/route.
- **All advanced LOAD options are already wired** end-to-end (speed_mode, transformer_quant
  fp8/int8/nvfp4/mxfp8, attention_backend, memory_mode, cpu_offload, transformer_cache,
  vae_tiling status). The Advanced panel is mostly a FRONTEND surfacing job.
- Frontend has `Tabs` (`components/ui/tabs.tsx`) and `Accordion` ready. No dropzone and no
  mask/brush canvas — both greenfield. `SliderField` is the customizer number input.
- `_EDIT_KEYWORDS = ("edit","kontext","inpaint","layered")` rejects edit/inpaint/Kontext/
  Layered repos at family detection.
- sd.cpp binary is downloaded prebuilt from **upstream leejet/stable-diffusion.cpp** (no
  unsloth mirror, no checksum/manifest/version pin, not wired into setup.sh). llama.cpp uses
  an `unslothai/llama.cpp` mirror with manifest+sha256+version pin+source fallback.
- Chat "Images" pill = provider-side (OpenAI/Gemini) hosted tool, separate from local diffusion.

## Workflow popularity ranking (what to build for 80% coverage)

1. txt2img (keep, polished) — done
2. img2img / variations — **P0**
3. inpainting (mask edit) — **P0**
4. upscaling / hires fix — **P0**
5. ControlNet (Canny/Depth/Pose/Lineart/Tile) — **P0/P1**
6. outpainting (canvas extend) — **P1**
7. instruction image editing (Qwen-Image-Edit, FLUX Kontext) — **P1**
8. style transfer / reference — **P1** (via img2img/edit/control)
9. batch generation/edit/upscale — **P1**
10. LoRA/style packs — **P2**

## Editing-model decisions (researched)

- **Qwen-Image-Edit / Edit-2511**: popular, best-in-class clean targeted edits + multilingual
  text. **Support** (instruction edit, mask-optional).
- **FLUX.1 Kontext**: popular, character-consistent in-context editing. **Support** (note:
  Kontext-dev is non-commercial/gated — surface license, don't block local custom models).
- **Qwen-Image-Layered**: newer, niche (Photoshop RGBA layer decomposition). Needs a dedicated
  pipeline (`additional_t_cond`) — **defer** (keep rejected for now; optional later behind a
  layered-specific view). This already crashed the standard path (the earlier bug).

## UI design — workflow tabs (inside ImagesPage, route/nav unchanged)

`Tabs` across the top of the controls area. Combine related workflows:
- **Create** — txt2img (current behavior preserved)
- **Transform** — img2img + style transfer (upload + strength/denoise + presets)
- **Edit** — inpaint (mask brush/upload/invert/feather, masked-vs-whole) + instruction edit
  (Qwen-Image-Edit / FLUX Kontext, mask-optional)
- **Extend** — outpainting (directional handles, aspect presets, overlap/feather)
- **Control** — ControlNet (one control slot first: Canny/Depth/Pose/Lineart/Tile + preview)
- **Enhance** — upscaling (ESRGAN/RealESRGAN + latent/tiled) 
- **Advanced Options** — Accordion surfacing existing load knobs (speed/compile/attention/
  quant fp8/int8/nvfp4/memory/offload/vae tiling/cache) with Auto defaults + resolved values.

Capability gating: a workflow/control is shown enabled only when the selected engine+family+
device+quant supports it; otherwise disabled with a plain-language "why".

## Backend architecture

- Extend `DiffusionGenerateRequest`: optional `workflow` (txt2img|img2img|inpaint|outpaint|
  control|edit|upscale), `init_image` (b64), `mask_image` (b64), `control_image` (b64),
  `strength`, `controlnet_conditioning_scale`, `control_start/end`, `upscale_factor`,
  `ref_images`. Add an image-decode (b64→PIL) helper (none exists).
- `DiffusionFamily`: add optional pipeline-class slots (`img2img_pipeline_class`,
  `inpaint_pipeline_class`, `edit_pipeline_class`, `controlnet_pipeline_class` + control repos).
  Build the right pipeline around the already-loaded `transformer=` (reuse `_assemble_pipe`
  shape); swap/cache pipeline class per workflow without reloading the transformer where
  possible.
- `generate()` kwarg builder must branch: img2img/edit pipelines take `image=`/`strength=` and
  reject `width/height`; inpaint adds `mask_image=`; control adds `control_image=`. Gate each
  kwarg via `inspect.signature`.
- Capability resolver: maps engine+family+device+quant → supported workflows + reasons; echoed
  in run metadata so the UI shows what actually ran.
- Memory planner must account for input/latent size, control models, VAE decode, upscale.

## PR breakdown (stacked, small, capability-gated)

- **PR-1 UI fixes + workflow shell**: fix number-input spinner overlap (DONE in tree), tab
  scaffold (Create/Transform/Edit/Extend/Control/Enhance/Advanced), Advanced Options accordion
  surfacing existing load knobs, capability banner, loading/empty/error states.
- **PR-2 Backend workflow contract + capability registry**: extend request/response, decode
  helper, per-family pipeline slots, resolver. No new behavior yet beyond txt2img.
- **PR-3 img2img (Transform)**: backend + Transform tab + dropzone (adapt from
  shared-composer `addFiles`/`PendingImageThumb`). Smoke test low vs high denoise.
- **PR-4 inpaint + instruction edit (Edit)**: mask canvas (greenfield), inpaint pipeline,
  Qwen-Image-Edit/FLUX Kontext edit; relax `_EDIT_KEYWORDS` → route to edit family.
- **PR-5 outpaint (Extend)**: expanded-canvas inpaint, directional handles, feather/overlap.
- **PR-6 ControlNet (Control)**: one control slot + preprocessor preview + strength/start/end.
- **PR-7 upscaling (Enhance)**: wire dormant sd.cpp `upscale()` + diffusers upscale + `/images/upscale`.
- **PR-8 Advanced panel polish + FP8/INT8 verification matrix**.
- **PR-9 sd.cpp prebuilt packaging**: mirror to `unslothai/stable-diffusion.cpp`, manifest+
  sha256+version pin+`--published-repo`+source fallback, wire into setup.sh (ref
  install_llama_prebuilt.py).
- **PR-10 cross-platform staging validation** (danielhanchen staging repos, small GGUFs).
- **PR-11 Playwright tests + screenshots/GIFs per tab** (studio_test_kit / unsloth_studio_workflow).
- **PR-12 batch + multi-control + reproducibility polish** (later).

## Done so far

- Fixed the customizer number-input spinner overlap (`SliderField` in images-page.tsx): native
  spinners covered the value on the narrow field; now fully suppressed (webkit inner+outer +
  Firefox `appearance:textfield`) and field widened to `w-14`. Frontend rebuilt clean.

## Verification

- Playwright (studio_test_kit) per tab: screenshots + GIFs, capability gating, upload/mask,
  progress/cancel/error, gallery.
- B200 functional: load + generate one image per workflow per representative family.
- FP8 + INT8 verified (build matrix: SDXL/FLUX/Qwen-Image/Qwen-Image-Edit/GGUF; measure
  black-image/NaN rate, peak VRAM, time-to-first-image, prompt adherence, source preservation).
- Cross-platform staging (Linux CUDA/CPU, Windows CUDA/CPU, macOS MPS) with small GGUFs.

## Delivery

New branch(es) off the current tip; new stacked PRs. Commit/push only when asked.

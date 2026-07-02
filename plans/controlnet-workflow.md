# Plan: ControlNet for the Studio Images workflow (stacked on #6769)

## Context & research

ControlNet is the **#2 most-used "beyond text-to-image" diffusion workflow** after LoRA (now shipped
in #6771). It conditions generation on a spatial control map so the output follows a structure. Web
research (this session) on ComfyUI / Forge / A1111 usage:

- The dominant control types are **Depth, Canny (edges), and OpenPose (human pose)**.
- The biggest 2025 shift is toward **Union / all-in-one ControlNets** that bundle many control modes in
  one model: **InstantX / Shakker-Labs `FLUX.1-dev-ControlNet-Union-Pro`** for FLUX, and **`xinsir/
  controlnet-union-sdxl-1.0`** for SDXL. SDXL has no official ControlNet (community: xinsir, TheMistoAI,
  BRIA). SD1.5 has the original lllyasviel set.
- Sources: comfyui-wiki.com ControlNet collections (flux-1 / sdxl), stable-diffusion-art.com ControlNet
  ComfyUI, education.civitai.com ControlNet guide, stablediffusiontutorials.com Qwen-Image ControlNets.

**Both Studio backends can do ControlNet** (verified against the live tree):
- diffusers has `FluxControlNetPipeline` / `FluxControlNetModel`, `StableDiffusionXLControlNetPipeline` /
  `ControlNetModel` / `ControlNetUnionModel`, `QwenImageControlNetPipeline`, `FluxControlNetInpaintPipeline`
  (diffusers 0.38 in the studio venv). Also `FluxControlPipeline` (the Flux.1-Canny/Depth "Control"
  in-model variants).
- native sd.cpp (`stable-diffusion.cpp`, the #6769 sd-cli base) has `--control-net <path>`,
  `--control-image <path>`, `--control-strength <f>`, `--control-net-cpu`, and a built-in
  `preprocess_canny` (examples/cli/main.cpp:704, examples/common/common.cpp:422+).

Studio has **no ControlNet wiring today**: `diffusion_families.py` has no controlnet fields and
`sd_cpp_args.py` has no `--control-*`. This adds it, mirroring the LoRA architecture (#6771) and the
existing img2img/inpaint/reference workflow patterns.

## Scope (this PR = diffusers ControlNet, Union-first)

Ship the highest-value slice first; keep it shippable and consistent with the shipped LoRA design.

- **In scope:** diffusers ControlNet for the families with the strongest ecosystems and pipeline support:
  **FLUX** (FluxControlNetPipeline + Union Pro), **SDXL** (StableDiffusionXLControlNetPipeline + xinsir
  Union), **Qwen-Image** (QwenImageControlNetPipeline). Single ControlNet per generation. A `control_type`
  hint (canny / depth / pose / tile / passthrough). **Canny preprocessing built in** (cheap, cv2/PIL) plus
  **passthrough** for user-supplied control maps (depth/pose maps made elsewhere, matching ComfyUI where
  preprocessing is separate). Strength + guidance start/end. Discovery endpoint + family-gated picker.
- **Out of scope (follow-ups):** native sd.cpp ControlNet (`--control-net`, needs GGUF ControlNet assets +
  family support probe); heavy preprocessors (Depth-Anything, OpenPose detector) as optional server-side
  auto-preprocess; multi-ControlNet stacking; ControlNet + inpaint combo.

## Key facts (verified)

- diffusers ControlNet pipelines are built with `Pipeline.from_pipe(base_pipe, controlnet=cn_model)` (or
  `from_pretrained(base, controlnet=...)`), so the resident base modules are reused with **no reload** --
  same `from_pipe` machinery the img2img/inpaint/edit workflows already use (`diffusion.py` ~:1104-1130,
  `_workflow_pipe`). The ControlNet model (`ControlNetModel` / `FluxControlNetModel` / `ControlNetUnionModel`)
  is a small extra module loaded once and cached on `_LoadState`.
- ControlNet models are **family-specific** (SD1.5 CN != FLUX CN != SDXL CN). So discovery must be
  **family-gated**, exactly like the LoRA picker's `supports_lora`/family filter.
- Generate-time contract mirrors reference/inpaint: a control image (b64) + params, threaded through
  `routes/inference.py` into both backends (diffusers serves it; native rejects clearly until the
  follow-up wires `--control-*`).
- Reuse: `diffusion_lora.py` discovery/resolve/family-gate patterns; the reference-image upload component +
  the LoRA picker UI shape; `_workflow_pipe` from_pipe; `hf_hub_download_with_xet_fallback`.

## Approach

### Families (`core/inference/diffusion_families.py`)
- Add per-family ControlNet declaration: `controlnet_pipeline_class` (e.g. "FluxControlNetPipeline",
  "StableDiffusionXLControlNetPipeline", "QwenImageControlNetPipeline"), `controlnet_model_class`
  ("FluxControlNetModel" / "ControlNetModel" / a union class), and a small curated list of recommended
  ControlNet repos tagged by control type. Expose a `controlnet: bool` capability (like `reference`).

### Discovery (`core/inference/diffusion_controlnet.py`, new -- mirrors diffusion_lora.py)
- `list_controlnets(family)` = curated family-tagged repos + a local scan; `resolve_controlnet(id, family,
  hf_token)` downloads via the xet-fallback helper; `preprocess_control(image, control_type)` (canny via
  cv2/PIL; passthrough otherwise); `supports_controlnet(engine, family, model_kind, quant)` gate
  (diffusers bf16 / bnb-4bit yes; GGUF-via-diffusers + torchao fp8/int8 dense = no, same rule as LoRA;
  native = follow-up).

### Backend -- diffusers (`core/inference/diffusion.py`)
- A ControlNet manager parallel to `_apply_loras`: load the requested `ControlNetModel` once (cache on
  `_LoadState`, reset on unload/model change), build the CN pipeline via `from_pipe(base, controlnet=...)`
  in `_workflow_pipe`, and pass `control_image` + `controlnet_conditioning_scale` +
  `control_guidance_start/end` at generate time. Never fuse; CN model stays bf16.

### Backend -- native (`core/inference/sd_cpp_backend.py`, sd_cpp_args.py) -- FOLLOW-UP
- Add `--control-net` / `--control-image` / `--control-strength` to the arg builder and a GGUF-ControlNet
  resolve; gate to families sd.cpp supports. Deferred out of this PR.

### Routes + request models (`models/inference.py`, `routes/inference.py`)
- Add optional `controlnet: ControlNetSpec` to `DiffusionGenerateRequest` (`{id, image, control_type,
  strength (0..2, default 1), guidance_start (0..1), guidance_end (0..1)}`); thread into `backend.generate`;
  surface `supports_controlnet` in status; persist the chosen CN + type in gallery recipe metadata.
- New `GET /api/models/diffusion-controlnets?family=` (mirror the LoRA discovery route).

### Frontend (`features/images/images-page.tsx`, `api.ts`)
- A "ControlNet" control in the left rail (reuse the reference-image uploader + a control-type Select +
  ControlNet-model Select gated by `supports_controlnet`/family + a strength SliderField). Show a small
  preview of the preprocessed control map. Thread `controlnet` into `generateDiffusionImage`; omit when no
  control image.

## Verification
- **Unit:** request validation (optional/empty unchanged; bad strength rejected; unsupported family/quant
  rejected); discovery (family filter, resolve, canny preprocess shape); diffusers manager (loads CN once,
  from_pipe built, scale threaded, reset on model change) with a fake pipe; routes (no-CN path unchanged).
- **Live smoke (critical):** on GPU 4, drive the real diffusers backend with a real family + Union CN and a
  canny control image; same prompt/seed at strength 0 vs 0.8; assert (a) output DIFFERS from no-control and
  (b) the strong-control output structurally follows the control map (edge-overlap / SSIM vs the control).
- **Playwright (`unsloth_studio_workflow`):** upload a control image, pick type + model + strength,
  generate; capture screenshots/GIF against the live secure studio.
- Full `pytest studio/backend/tests/` green; frontend `vite build` clean; ruff clean.

## Delivery
- New branch `diffusion-controlnet` off `diffusion-image-workflows` (#6769 head) in an isolated worktree,
  sibling to #6771 (LoRA) and #6772 (fp8 fix). PR on `unslothai/unsloth`, base = diffusion-image-workflows,
  part of the single logical stack rooted on #6763 (continuation of #6658). Commit as Daniel Han; no AI/bot
  mentions, no emojis, no em dashes.
- Follow-ups: native sd.cpp ControlNet; server-side Depth/OpenPose auto-preprocessors; multi-ControlNet;
  ControlNet+inpaint.

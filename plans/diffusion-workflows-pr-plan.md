# Stacked-PR plan: Studio diffusion workflows (Images redesign)

Branch tip: `diffusion-eager-and-compile-cache` (latest commit "Phase 16 review fixes").
Remote: `oobabooga/unsloth`. New PRs stack on top of the existing diffusion stack
(ultimately on top of unslothai/unsloth#6658), treated as one logical change.

Nothing here is committed yet (commit/push only on explicit request).

## CRITICAL: the working tree holds TWO uncommitted streams, and three core files INTERMINGLE them

A full `git status` / marker audit (branch `diffusion-eager-and-compile-cache`, tip "Phase 16
review fixes") shows the uncommitted tree is NOT a clean single feature. There are two streams:

A) **The eager/compile-cache phase** (the branch's own in-progress work, NOT this session's —
   zero of this feature's markers). Purely-its files, safe to NOT touch in the workflow PRs:
   - new modules: `diffusion_arch_patches.py`, `diffusion_compile_cache.py`,
     `diffusion_eager_patches.py`, `diffusion_gguf_compile.py`, `diffusion_patch_backend.py`
   - new tests: `test_diffusion_arch_patches.py`, `test_diffusion_compile_cache.py`,
     `test_diffusion_eager_patches.py`, `test_diffusion_gguf_compile.py`
   - modified: `diffusion_speed.py`, `test_diffusion_speed.py`, `conftest.py`,
     `scripts/diffusion_bench.py`, and ~25 untracked `scripts/*bench*/*probe*/*orchestrator*`.

B) **The Images workflows feature** (this session): the workflow engine + frontend + installer.

**The two streams INTERMINGLE inside three shared files and are NOT separable by file:**
  - `studio/backend/core/inference/diffusion.py` — this feature's workflow hunks are interleaved
    with the eager/compile wiring (imports at L67-75; `install_arch_patches`/`compile_cache.begin`/
    `.restore`/`.save` and the `eager_patched`/`compile_cache_ctx` state throughout
    `load_pipeline`/`generate`/`unload`). A single `diffusion.py` cannot go into one PR without the
    other stream's hunks.
  - `studio/backend/models/inference.py` — this feature's `init_image`/`mask_image`/
    `reference_images`/`upscale`/`model_kind` fields sit next to the pre-existing `speed_mode`/
    `transformer_prequant_path` fields in the same request models.
  - `studio/backend/tests/test_diffusion_backend.py` — this feature's workflow tests sit next to
    the pre-existing `test_failed_load_rolls_back_eager_patches` (imports `diffusion_eager_patches`).

**Implication / options (USER DECIDES — it is their branch + their eager/compile work):**
  - CLEANLY separable now (purely this feature, can be committed/PR'd on their own anytime):
    frontend `images-page.tsx` + `api.ts` + `pickers.tsx`, and the sd.cpp installer
    `install_sd_cpp_prebuilt.py` + `test_sd_cpp_install.py`. (These are PR 2 and PR 3 below.)
  - The backend engine (PR 1) CANNOT be cleanly split from the eager/compile phase via files.
    Realistic paths: (a) finalize + commit the eager/compile phase first, then this feature's
    backend lands as a clean diff on top; or (b) commit both streams together as the branch's
    next chunk (consistent with treating the stack as one logical change); or (c) a manual
    `git add -p` hunk split of the three shared files (tedious, risks a non-compiling
    intermediate). NOT auto-doable safely without the owner's intent for the eager/compile work.

## Proposed stack (3 PRs, bottom to top)

### PR 1 - Backend: diffusion workflow engine (safetensors + image-conditioned + editing)
Files:
- `studio/backend/core/inference/diffusion.py` (the feature hunks: three load "kinds"
  gguf/single_file/pipeline; `_workflow_pipe` via `from_pipe(torch_dtype=None)`;
  `_align_vae_dtype`; `generate()` routing for reference/img2img/inpaint/upscale/edit;
  image-conditioned width/height from the input image (but reference + txt2img use the slider
  size); `upscale` (hires fix) branch on the img2img pipe; `reference` (FLUX.2) branch that
  passes the image(s) to the loaded pipe directly (no from_pipe, no strength) incl. multi-
  reference (`reference_images` combined into a list, capped at 3 extra); branch ORDER
  inpaint/upscale before reference so a mask/upscale request on a reference family still routes
  right; `_family_workflows` (adds "upscale" wherever img2img is supported, "reference" for
  reference families, "outpaint" only for size-preserving inpaint families); `kind` on state +
  `model_kind` in status; `load_progress` double-count fix). NOTE: this file ALSO carries
  pre-existing speed hunks if any landed here - review per-hunk and exclude non-feature hunks.
- `studio/backend/core/inference/diffusion_families.py` (img2img/inpaint pipeline slots;
  `edit` flag + `reference` flag + `inpaint_preserves_size` flag; `qwen-image-edit` +
  `flux.1-kontext` families; flux.2-klein gains reference + inpaint (no outpaint: FLUX.2
  normalizes to ~1MP); `detect_family` longest-match + leftover-reject; `layered` reject).
- `studio/backend/core/inference/diffusion_engine_router.py` (model_kind -> diffusers for
  non-gguf kinds).
- `studio/backend/core/inference/diffusion_memory.py` (`estimate_safetensors_dense_mib`).
- `studio/backend/core/inference/sd_cpp_backend.py` (model_kind passthrough; reject
  img2img/inpaint on the native engine).
- `studio/backend/models/inference.py` (load request: optional gguf_filename, model_kind,
  init/mask/strength, advanced knobs; status: workflows, model_kind).
- `studio/backend/routes/inference.py` (model_kind forwarding; ValueError -> 400;
  exc_info logging).
- Tests: `test_diffusion_backend.py`, `test_diffusion_routes.py`.

Title: `Studio diffusion: safetensors + image-conditioned + instruction-editing workflows`
Summary: Adds non-GGUF safetensors loading (full bnb-4bit pipelines + single-file fp8,
gated to unsloth/*), the image-conditioned workflows (img2img, inpaint, outpaint via the
inpaint path) built with `Pipeline.from_pipe` for zero-extra-VRAM component reuse, and
instruction editing as its own family kind (Qwen-Image-Edit-2511 + FLUX.1-Kontext-dev).
Fixes two bugs: `from_pipe` defaulting to a float32 recast that crashed torchao-quantized
transformers, and image-conditioned calls forcing the slider size onto the input image.

### PR 2 - Frontend: redesigned Images page (workflow tabs + Advanced Options)
Files:
- `studio/frontend/src/features/images/images-page.tsx` (workflow tabs Create/Transform/
  Inpaint/Extend/Upscale/Reference/Edit; capability gating + auto-switch; `MaskCanvas`;
  `buildOutpaint`; Upscale tab with Scale + Detail-strength sliders; Reference tab (FLUX.2,
  reference image + add/remove extra references, no strength); Advanced Options accordion gated
  to GGUF for transformer-quant; spinner-overlap fix).
- `studio/frontend/src/features/images/api.ts` (request/status types incl. model_kind, upscale,
  reference_images).
- `studio/frontend/src/components/assistant-ui/model-selector/pickers.tsx` (curated
  safetensors + edit GGUF rows; `SUPPORTED_EDIT_KEYWORDS` un-hide; layered hide).

Title: `Studio Images: workflow tabs (create/transform/inpaint/extend/upscale/reference/edit) + Advanced Options`
Summary: Redesigns the Images page around capability-gated workflow tabs with a brush mask
editor, client-side outpaint, a hires-fix upscale tab, a FLUX.2 reference tab, an instruction-edit
tab, and an Advanced Options panel (speed/quant/attention/memory/step-cache/offload), plus the
number-input spinner fix.

### PR 3 - sd.cpp prebuilt installer hardening
Files: `studio/install_sd_cpp_prebuilt.py`, `studio/backend/tests/test_sd_cpp_install.py`.
Title: `Studio sd.cpp: pin release + verify sha256 + mirror-ready source`
Summary: Pins the stable-diffusion.cpp release (was tracking `latest`), verifies each
download's sha256 against GitHub's published asset digest before extract/execute, adds a
download timeout, and makes the source repo configurable (`UNSLOTH_SD_CPP_REPO`) so a
future unslothai mirror needs no code change. Cleanly separable from the rest.

## Pre-PR review (done)
An independent 3-angle review (backend correctness, frontend/UX, security/robustness) ran over
the full session diff. No High findings; the load-gating to `unsloth/*` and the multi-reference
count caps were verified intact end-to-end. Fixes applied before the PRs:
- Frontend [Med]: multi-reference slots no longer renumber mid-edit (dropped the eager
  `filter(Boolean)` in the per-slot onChange; empties dropped only at send).
- Backend [Med]: upscale now caps the ABSOLUTE output (longest side <= 2048), not just the
  factor, so a large upload * 4x can't OOM.
- Backend/security [Med]: `_decode_b64_image` rejects images > 4096px/side (uniform guard for
  init/mask/reference vs decompression-bomb / OOM inputs); base64 image fields capped at 32 MiB.
- Security [Low]: the native sd.cpp engine guard also rejects `reference_images`.
All covered by new tests (82 backend pass) and a post-fix e2e (all five workflows still pass).

## Post-deploy user feedback fixes (done)
From live use of the deployed studio:
- Backend [Med]: image-conditioned workflows passed the raw upload size to pipelines that
  require multiples of 16 (Z-Image/Qwen/FLUX), so an odd upload (e.g. 186px) failed with
  "Height must be divisible by 16". Added `_snap_to_multiple` and auto-resize init (and the
  matched mask) to the nearest /16 for img2img/inpaint/extend/edit. Verified live: a 186x250
  Transform and Inpaint now return 200 at 192x256. Tests added.
- Frontend [Med]: the Advanced options (FP8/INT8 quant, speed, attention, memory) were a
  collapsed, muted accordion at the bottom of the left rail that users missed (HF screenshots
  discussion #25). Moved them into a RIGHT-DOCKED panel mirroring Chat's settings panel.
  Per follow-up (discussion #26): CLOSED by default, toggled by a SINGLE fixed top-bar button
  using Chat's `LayoutAlignRightIcon` that stays in the exact same position in both states
  (verified x/y identical open vs closed) and highlights when open. Controls extracted to a
  render-local `advancedControls`; unused Accordion import removed.

## Constraints for execution (when authorized)
- Write as the user; no AI/bot mentions, no emojis, no em dashes; well-formatted bodies.
- `gh auth status` first. Push to `oobabooga/unsloth`, stack on the current branch.
- Keep PR 3 independent; PR 2 depends on PR 1 (frontend needs the backend contract).
- Re-run `pytest studio/backend/tests/test_diffusion_*.py test_sd_cpp_install.py` +
  frontend `tsc`/`build` before each PR.

## Out of scope / follow-ups
Scope decisions are backed by HF download data in `plans/diffusion-popularity-findings.md`.
The seven shipped workflows are create, transform, inpaint, extend, upscale, reference, edit.
- Publish the unslothai/stable-diffusion.cpp mirror + macOS/Windows staging (#152).
- ControlNet / style-transfer: the goal's "most popular" set is covered by the seven shipped
  workflows. ControlNet on the modern diffusers/FLUX/Qwen stack is niche by downloads
  (~3K/30d), so deferring it is data-backed, not just an effort call.
- FLUX.2-klein inpaint and multi-reference are DONE (shipped). Outpaint is intentionally not
  offered for FLUX.2 (it scales >1MP inputs to ~1MP). No further FLUX.2 follow-ups outstanding.

# Diffusion workflow popularity findings (HF download data)

Read-only HF metadata pull (`scripts/investigate_popularity.py`), to ground the Studio
Images scope against what people actually download. Downloads are HF's 30-day count and
all-time count; pulled 2026-06-30.

## Qwen-Image-Edit vs Qwen-Image-Layered (the explicit "determine popularity" question)

| Model | dl / 30d | dl all-time | likes | pipeline |
|---|---:|---:|---:|---|
| Qwen/Qwen-Image-Edit-2509 | 511,996 | 2,942,966 | 1,185 | image-to-image |
| Qwen/Qwen-Image-Edit-2511 | 162,185 | 1,088,015 | 1,087 | image-to-image |
| Qwen/Qwen-Image-Edit (base) | 70,728 | 1,161,044 | 2,440 | image-to-image |
| **Qwen-Image-Edit (all variants)** | **~745,000** | **~5,192,000** | - | - |
| Qwen/Qwen-Image-Layered | 51,303 | 234,785 | 1,112 | image-text-to-image |
| unsloth/Qwen-Image-Edit-2511-GGUF | 218,313 | - | - | image-to-image |

**Conclusion:** Qwen-Image-Edit is ~10-14x more downloaded than Layered (combined 745K/30d
vs 51K, 5.2M vs 235K all-time). Shipping Edit (2511 + the unsloth GGUF, which alone pulls
218K/30d) and rejecting/deferring Layered is the correct, data-backed call. Layered also
needs a dedicated pipeline (`additional_t_cond=True`) the standard QwenImagePipeline can't
drive, so it would be both niche AND extra engineering. Reject stands.

## ControlNet is niche on the modern (diffusers/FLUX/Qwen) stack

| Model | dl / 30d | dl all-time | likes |
|---|---:|---:|---:|
| InstantX/FLUX.1-dev-Controlnet-Canny | 2,891 | 136,727 | 194 |
| lllyasviel/ControlNet (SD1.5-era) | 0 | 14 | 3,820 |
| stabilityai/stable-diffusion-x4-upscaler | 10,040 | 2,976,405 | 725 |

**Conclusion:** ControlNet's large user base lives in the older SD1.5 / A1111 ecosystem, not
the diffusers/FLUX/Qwen stack Studio targets (the modern FLUX ControlNet is ~3K/30d). It is
NOT part of the "most popular ~80%" for current-gen models, so deferring it is justified by
the data, not just by effort. The dedicated x4 upscaler is also low 30-day (10K) though high
all-time; our generic hires-fix upscale (img2img re-detail) covers the use case for any
loaded family without an extra model.

## The shipped six cover the popular workflows

Top text-to-image (HF list, 30-day): SD1.5 (1.78M), SDXL (1.32M), FLUX.1-dev (1.09M),
dreamshaper-7 (1.03M), **Tongyi-MAI/Z-Image-Turbo (886K)**, sd-turbo (684K), SD3.5-medium
(606K), sdxl-turbo (598K), Qwen-Image-Lightning (483K). All are plain txt2img -> our Create
tab; the GGUF/bnb families + Z-Image cover the modern ones.

Top image-to-image (HF list, 30-day): Qwen-Image-Edit-2509 (512K) -> Edit tab; SDXL-refiner
(162K) -> Upscale/Transform; Kontext (150K) -> Edit tab.

So Create / Transform / Inpaint / Extend / Upscale / Edit map onto the head of both
distributions.

## SHIPPED: FLUX.2-klein image (reference) conditioning

> Status: IMPLEMENTED + verified live (2026-06-30). `flux.2-klein` now has `reference=True`;
> the backend exposes a "reference" workflow that passes the image to the loaded
> Flux2KleinPipeline directly (no from_pipe, no strength, output at the requested size); the
> frontend has a "Reference" tab. Verified with `scripts/verify_reference_http.py` on
> `unsloth/FLUX.2-klein-4B-GGUF` (Q4_K_M): a reference-conditioned 1024x1024 result is
> non-blank, correctly sized, and DIFFERS from the identical-seed plain txt2img.
>
> FLUX.2-klein ALSO gained inpaint (`Flux2KleinInpaintPipeline` via from_pipe; verified with
> `scripts/verify_klein_inpaint.py`). It does NOT get outpaint/extend: FLUX.2 scales any >1MP
> input down to ~1MP, so a padded outpaint canvas shrinks back. "outpaint" is now a distinct
> capability advertised only for size-preserving inpaint families (`inpaint_preserves_size`).
> Multi-reference is shipped too (the pipeline accepts a list; the Reference tab has add/remove
> slots, backend caps at 3 extra; verified with `scripts/verify_multiref_http.py`: two
> references differ from one at the same seed). The analysis that motivated the work follows.

The data surfaced this gap (now closed):

| Model | dl / 30d | pipeline |
|---|---:|---|
| black-forest-labs/FLUX.2-klein-4B | 470,482 | image-to-image (#2 overall) |
| black-forest-labs/FLUX.2-dev | 271,037 | image-to-image |
| **unsloth/FLUX.2-klein-4B-GGUF** | **243,307** | image-to-image |
| black-forest-labs/FLUX.2-klein-9B | 178,964 | image-to-image |

`flux.2-klein` is ALREADY a registered family in `diffusion_families.py` (txt2img only,
base `FLUX.2-klein-4B`, open repo). But `Flux2KleinPipeline.__call__` natively accepts an
`image` argument (verified in diffusers 0.38.0; params: image, prompt, height, width,
num_inference_steps, guidance_scale -- NOTE: no `strength`). FLUX.2 is a unified
text-to-image + reference/edit model: the SAME loaded pipe does both, depending on whether
`image` is passed. Today Studio exposes only txt2img for it, so the popular image-editing
mode of the #2 image-to-image model is unreachable.

### Why it's a separate PR, not a tail-of-session add
FLUX.2 reference conditioning is a DIFFERENT semantic from the shipped workflows:
- No `strength` (it is reference-conditioning, not a denoise blend like img2img).
- Output size comes from width/height (txt2img-style), not from the input image size, so the
  image-conditioned width/height rule we added for img2img/inpaint/upscale does NOT apply.
- FLUX.2 supports MULTIPLE reference images; single-image is the common case but the UX
  should not preclude multi-ref.
This needs: read the Flux2KleinPipeline source for exact `image` semantics (list vs single,
how it is resized/tiled, recommended guidance), decide the UX (a "Reference" workflow that is
available alongside Create for `reference=True` families, distinct from the strength-based
Transform tab), then verify on the open FLUX.2-klein-4B base (and the unsloth GGUF) with a
reference image before/after.

### Sketch (for the follow-up PR)
- `diffusion_families.py`: add `reference: bool = False`; set `reference=True` on flux.2-klein.
- `_family_workflows`: when `fam.reference`, expose `"reference"` (in addition to txt2img).
- `generate()`: a `reference` branch that passes `image` to `state.pipe` directly (no
  from_pipe, no strength), with width/height = the requested size (NOT the input size).
- Frontend: a "Reference" tab (image dropzone + prompt), gated to `reference` families;
  Create stays pure txt2img for the same model.
- Verify: load unsloth/FLUX.2-klein-4B-GGUF, pass a reference image, confirm the output is
  conditioned on it and differs from a no-image run at the same seed.

## Net
The seven shipped workflows (create, transform, inpaint, extend, upscale, reference, edit)
cover the popular ~80% across both the txt2img and image-to-image distributions, including the
#1 image-to-image model (Qwen-Image-Edit) and the #2 (FLUX.2-klein, now via the reference tab).
ControlNet / SD1.5-era ControlNet remain deferred with data backing (niche on the modern stack).

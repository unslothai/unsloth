# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for diffusion model identification.

No torch/diffusers imports here: everything in this module is a pure function of
its string/path arguments so it can be unit-tested without the heavy runtime.

A diffusion checkpoint published as a single-file GGUF only carries the
transformer weights; the matching VAE / text encoders / scheduler come from a
companion ``diffusers`` base repo. ``DiffusionFamily`` maps a checkpoint to the
diffusers classes and base repo needed to assemble the full pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional


# Runtime->route contract: the RuntimeError messages a backend raises for
# client-recoverable generate states. The /images/generate route matches these
# EXACTLY to return 409 (vs a sanitized 500 for real failures), so both engines
# must raise them verbatim -- keep them named here, not as scattered literals.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Pipeline kwarg carrying the guidance value. Most use "guidance_scale";
    # Qwen-Image's distilled guidance is off, so its real CFG is "true_cfg_scale".
    cfg_kwarg: str = "guidance_scale"
    # The pipe attribute holding the denoiser module. DiT families expose it as
    # ``pipe.transformer`` (the default); U-Net families (SDXL) as ``pipe.unet``.
    # Read wherever the backend touches the denoiser generically (VAE dtype
    # alignment, optimisation guards), so a U-Net family works without assuming a
    # ``transformer`` attribute exists.
    denoiser_attr: str = "transformer"
    # True when a single-file ``.safetensors`` checkpoint is the WHOLE pipeline
    # (U-Net + VAE + text encoders), not a transformer-only file. SDXL ships this
    # way, so the loader calls ``pipeline_class.from_single_file`` on it directly
    # rather than ``transformer_class.from_single_file`` + a companion base repo.
    # DiT families leave this False (their single file is transformer-only).
    single_file_is_pipeline: bool = False
    # Optional diffusers pipeline classes for image-conditioned workflows. The backend
    # builds these around the ALREADY-loaded transformer/VAE/text-encoder via
    # ``Pipeline.from_pipe`` (no extra weights, no reload), so a family only needs the
    # class name here to gain the workflow. None = the family does not support it (the
    # UI gates the workflow off). The base text-to-image pipeline is ``pipeline_class``.
    img2img_pipeline_class: Optional[str] = None
    inpaint_pipeline_class: Optional[str] = None
    # ControlNet: the diffusers ControlNet pipeline + model classes for this family. The backend
    # loads the (small) ControlNet model via from_pretrained and builds the pipeline via
    # ``Pipeline.from_pipe(base, controlnet=model)`` around the resident modules (no reload),
    # then passes the control image + conditioning scale at generate time. None on both = the
    # family has no diffusers ControlNet support and the UI gates the workflow off.
    controlnet_pipeline_class: Optional[str] = None
    controlnet_model_class: Optional[str] = None
    # True when the inpaint pipeline keeps the input canvas size, so it can also drive
    # outpaint (extend), where the padded canvas is LARGER than the original. False for
    # FLUX.2 (its pipelines scale any >1MP input down to ~1MP, which shrinks an outpaint
    # canvas back and defeats the extend). Such families get Inpaint but not Extend.
    inpaint_preserves_size: bool = True
    # True for instruction-editing families (Qwen-Image-Edit / FLUX Kontext): the model's
    # OWN pipeline (``pipeline_class``) is the edit pipeline -- it takes an input image plus
    # a text instruction and has no plain text-to-image mode. So these expose only the
    # "edit" workflow, require an input image at generate time, and the loaded pipe is used
    # directly (no from_pipe). ``base_repo`` here is the matching diffusers repo that
    # supplies the VAE / text-encoder / processor / scheduler for the GGUF transformer.
    edit: bool = False
    # True for families whose OWN text-to-image pipeline ALSO accepts reference image(s)
    # (FLUX.2: Flux2KleinPipeline takes an optional ``image`` arg). Unlike ``edit`` these
    # families still do plain text-to-image (no image), and unlike img2img the conditioning
    # is reference-based, not a denoise blend: there is no ``strength`` and the output size
    # comes from the requested width/height, not the reference's size. The loaded pipe is
    # used directly (no from_pipe). Exposes a "reference" workflow alongside "txt2img".
    reference: bool = False
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True for families whose activations overflow float16's finite range
    # (~6.5e4) and produce inf -> NaN latents -> a black image. The backend
    # promotes a resolved float16 to float32 for these at load time.
    fp16_incompatible: bool = False
    # Set False only for a family whose denoiser block does not compile cleanly with
    # regional torch.compile. Now consulted on the GGUF path too (compile runs on the
    # GGUF transformer); all current families compile, so this stays True.
    supports_torch_compile: bool = True
    # Optional pre-quantized transformer checkpoints, as (scheme, repo_id) pairs (a
    # hashable mapping). When the fast transformer_quant path resolves a scheme with a
    # hosted checkpoint, the loader fetches the already-quantized weights instead of
    # materialising the dense bf16 transformer on the GPU (much lower load VRAM + a
    # smaller download). Empty until checkpoints are hosted -> behaviour is unchanged.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Native (stable-diffusion.cpp) single-file assets, used only when the no-GPU
    # sd.cpp engine is selected (CPU / Apple). The transformer GGUF is shared with
    # the diffusers path; sd-cli additionally needs a single-file VAE and text
    # encoder(s), because the diffusers base repo ships those sharded and sd-cli
    # cannot read that layout. Each asset is a hashable (repo_id, filename) the
    # backend fetches with hf_hub_download. ``sd_cpp_text_encoders`` carries a
    # trailing SdCppModelFiles field name (clip_l / t5xxl / llm / qwen2vl / clip_g)
    # so the backend maps each file onto the right sd-cli flag. Empty -> the family
    # has no native mapping and the sd.cpp route falls back to diffusers.
    sd_cpp_vae: Optional[tuple[str, str]] = None
    # VAE latent-format override for sd-cli (--vae-format): "flux2" for the FLUX.2
    # autoencoder, None (auto) otherwise.
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Family-specific sd-cli sampler settings, applied on the native path so the
    # output matches the model's supported invocation (e.g. Qwen-Image needs
    # --sampling-method euler --flow-shift 3 per stable-diffusion.cpp's docs). None
    # leaves sd-cli's defaults (correct for the distilled flux/z-image families).
    sd_cpp_sampling_method: Optional[str] = None
    sd_cpp_flow_shift: Optional[float] = None


# Keyed by architecture, not per model variant: a checkpoint's specific base repo
# is read from its HF base_model tag at load time, so one entry covers Turbo/full,
# schnell/dev, etc. base_repo here is only a fallback. Only archs whose diffusers
# transformer supports from_single_file load here (ERNIE-Image does not, yet; LTX
# video models are out of scope). FLUX.2-klein-9B shares the klein family (its base
# repo is resolved per-variant), and FLUX.2-dev has its own family below.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        aliases = ("flux1", "flux-1"),
        img2img_pipeline_class = "FluxImg2ImgPipeline",
        inpaint_pipeline_class = "FluxInpaintPipeline",
        controlnet_pipeline_class = "FluxControlNetPipeline",
        controlnet_model_class = "FluxControlNetModel",
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = (
            ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip_l"),
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", "t5xxl"),
        ),
    ),
    # FLUX.2-klein is a distinct pipeline (Flux2KleinPipeline) with a Qwen3 text
    # encoder, not the Mistral-based Flux2Pipeline; it must precede a generic flux
    # match. The Mistral-based Flux2Pipeline is the separate flux.2-dev family below.
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = ("flux2-klein",),
        # Flux2KleinPipeline natively accepts reference image(s) via its `image` arg, so it
        # exposes a "reference" workflow on top of plain text-to-image. It has a dedicated
        # inpaint pipeline too (no img2img one), so it also gets inpaint + extend (outpaint).
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        # FLUX.2 scales >1MP inputs down to ~1MP, so outpaint (a larger canvas) can't grow.
        inpaint_preserves_size = False,
        # FLUX.2 uses a distinct 32-channel autoencoder; sd-cli needs the latent
        # format override. The single-file VAE ships in Comfy-Org/flux2-dev (the
        # klein-4B repo only has a sharded diffusers VAE). Shares Qwen3-4B with z-image.
        sd_cpp_vae = ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    # FLUX.2-dev is the full (non-distilled) FLUX.2. It uses the Mistral-based
    # Flux2Pipeline, distinct from klein's Qwen3-based Flux2KleinPipeline, so it needs
    # its own entry. Its base diffusers repo is gated (gated=auto) but reachable with an
    # HF token. text-to-image only: diffusers 0.38 ships no Flux2 img2img / inpaint
    # pipeline for dev. VAE + Mistral text encoder come from the open Comfy-Org/flux2-dev
    # mirror for the sd-cli path (shares the FLUX.2 32-channel AE with klein).
    DiffusionFamily(
        name = "flux.2-dev",
        pipeline_class = "Flux2Pipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        aliases = ("flux2-dev", "flux2dev"),
        sd_cpp_vae = ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            (
                "Comfy-Org/flux2-dev",
                "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
                "llm",
            ),
        ),
    ),
    DiffusionFamily(
        # Instruction editing with FLUX. FluxKontextPipeline takes an input image + an edit
        # instruction; the GGUF transformer is the standard FluxTransformer2DModel, with the
        # T5/CLIP text encoders + VAE from the base diffusers repo. cfg defaults to
        # guidance_scale (FLUX). Most-specific aliases first so detect_family prefers this
        # over the plain "flux.1" family and un-rejects the "kontext" keyword for it.
        name = "flux.1-kontext",
        pipeline_class = "FluxKontextPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        aliases = ("flux.1-kontext-dev", "flux1-kontext", "flux-kontext", "kontext"),
        edit = True,
    ),
    DiffusionFamily(
        # Instruction editing (image-in + text-instruction-out). The 2511 checkpoint ships
        # as QwenImageEditPlusPipeline (multi-image-capable); the GGUF transformer is the
        # standard QwenImageTransformer2DModel, with the VAE / Qwen2.5-VL text-encoder /
        # image processor / scheduler coming from the base diffusers repo. Most-specific
        # aliases first so detect_family prefers this over the plain "qwen-image" family.
        name = "qwen-image-edit",
        pipeline_class = "QwenImageEditPlusPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        cfg_kwarg = "true_cfg_scale",
        aliases = (
            "qwen-image-edit-2511",
            "qwen-image-edit-2509",
            "qwen-image-edit",
            "qwen_image_edit",
            "qwenimageedit",
        ),
        edit = True,
    ),
    DiffusionFamily(
        name = "qwen-image",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image", "qwenimage"),
        img2img_pipeline_class = "QwenImageImg2ImgPipeline",
        inpaint_pipeline_class = "QwenImageInpaintPipeline",
        controlnet_pipeline_class = "QwenImageControlNetPipeline",
        controlnet_model_class = "QwenImageControlNetModel",
        sd_cpp_vae = ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        # The Qwen2.5-VL text encoder as a Q4_K_M GGUF keeps the CPU RAM win (the
        # bf16 safetensors encoder is ~15 GB). sd-cli's --qwen2vl is an alias of --llm.
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
        ),
        # Qwen-Image's supported sd.cpp invocation (docs/qwen_image.md).
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 3.0,
    ),
    DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        aliases = ("zimage", "z_image"),
        img2img_pipeline_class = "ZImageImg2ImgPipeline",
        inpaint_pipeline_class = "ZImageInpaintPipeline",
        # Z-Image's MLP down-projections peak near 9e5, which overflows float16.
        fp16_incompatible = True,
        sd_cpp_vae = ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    # SDXL is the one U-Net family here: the denoiser is ``pipe.unet``
    # (UNet2DConditionModel), not a DiT ``pipe.transformer``, and a single-file
    # ``.safetensors`` is the WHOLE pipeline rather than a transformer-only file.
    # So it declares ``denoiser_attr = "unet"`` + ``single_file_is_pipeline = True``
    # and loads via the pipeline class (from_pretrained for a repo, from_single_file
    # for a single .safetensors). The base repo supplies both CLIP text encoders,
    # the VAE and the scheduler on the pipeline path. img2img / inpaint / ControlNet
    # are the standard SDXL pipelines, built around the resident modules via
    # from_pipe like every other family. There is no GGUF/single-file transformer
    # path for SDXL (the whole checkpoint is one file), and no native sd.cpp mapping
    # yet, so the no-GPU route falls back to diffusers.
    DiffusionFamily(
        name = "sdxl",
        pipeline_class = "StableDiffusionXLPipeline",
        transformer_class = "UNet2DConditionModel",
        base_repo = "stabilityai/stable-diffusion-xl-base-1.0",
        aliases = ("stable-diffusion-xl", "sd-xl", "sd_xl", "sdxl-turbo", "sdxl-base"),
        denoiser_attr = "unet",
        single_file_is_pipeline = True,
        img2img_pipeline_class = "StableDiffusionXLImg2ImgPipeline",
        inpaint_pipeline_class = "StableDiffusionXLInpaintPipeline",
        controlnet_pipeline_class = "StableDiffusionXLControlNetPipeline",
        controlnet_model_class = "ControlNetModel",
    ),
)

# Editing / inpaint checkpoints share an arch keyword but need a different
# pipeline and an input image, which this text-to-image backend doesn't drive.
# "layered" rejects Qwen-Image-Layered: its transformer sets additional_t_cond=True
# and expects an extra addition_t_cond input that the standard QwenImagePipeline
# never supplies, so it loads but crashes at the first denoise step. Rejecting it
# here fails the load fast with a clear message and hides it from the picker.
_EDIT_KEYWORDS = ("edit", "kontext", "inpaint", "layered")


def _token_in_needle(token: str, needle: str) -> bool:
    """True when ``token`` appears in ``needle`` as a whole path/name segment, i.e.
    delimited by a separator (``- _ . / \\``) or a string boundary, not merely as a
    raw substring. This keeps multi-part tokens matching where they should
    ('qwen-image-edit' in 'qwen-image-edit-2511') while preventing a short token from
    matching inside an unrelated word ('kontext' must not match 'kontextual', 'edit'
    must not match 'edition')."""
    return re.search(r"(?:^|[-_./\\])" + re.escape(token) + r"(?:$|[-_./\\])", needle) is not None


def _best_family_match(needle: str) -> Optional[DiffusionFamily]:
    """The family whose name/alias is the LONGEST whole-segment token of ``needle``.
    Longest = most specific, so an edit checkpoint ('...qwen-image-edit-2511...')
    matches the 'qwen-image-edit' family rather than the generic 'qwen-image' one.
    Segment matching (not raw substring) stops a short alias like 'kontext' from
    hijacking an unrelated path such as '.../kontextual/z-image-...gguf'."""
    best: Optional[tuple[DiffusionFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or len(token) > best[1]):
                best = (fam, len(token))
    return best[0] if best else None


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Resolve a ``DiffusionFamily`` from a repo id, or an explicit override.

    ``override`` matches a family ``name`` or alias exactly. Otherwise the most-specific
    family whose name/alias is a substring of the repo id wins. Supported editing families
    (Qwen-Image-Edit) match here; unsupported editing/inpaint/layered checkpoints that only
    share a base family's arch keyword are still rejected (None), because they need a
    different pipeline + input this backend's base text-to-image path doesn't drive.
    """
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        return None
    needle = repo_id.lower()
    match = _best_family_match(needle)
    if match is not None:
        # Don't let a generic base family (e.g. qwen-image) swallow a variant it can't run
        # (qwen-image-LAYERED, ...-Inpaint): if the id still carries a reject keyword the
        # matched family does not itself declare, reject so the load fails fast + clearly.
        # Scope the keyword check to the LAST path component (the model id or
        # filename), not arbitrary parent directories: a valid file selected as
        # repo_id `/models/edit` + filename `Z-Image-Turbo-Q4.gguf` must not be
        # rejected because a parent folder happens to be named `edit`. The
        # combined `repo_id/gguf_filename` fallback passes the filename last.
        basename = re.split(r"[/\\]+", needle)[-1]
        matched_tokens = (match.name, *match.aliases)
        if any(
            _token_in_needle(kw, basename) and not any(kw in tok for tok in matched_tokens)
            for kw in _EDIT_KEYWORDS
        ):
            return None
        return match
    return None


def supported_family_names() -> tuple[str, ...]:
    """Family names accepted as ``family_override`` and shown in the unknown-model
    error. Kept in registry order so the message lists what the backend can load."""
    return tuple(fam.name for fam in _FAMILIES)


def detect_family_for_pick(
    repo_id: str,
    gguf_filename: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[DiffusionFamily]:
    """``detect_family``, falling back to the combined path/filename for a direct
    local ``.gguf`` pick. The frontend splits such a pick into (parent dir, basename),
    so the family keyword can live only in the filename (e.g.
    ``/models/z-image-turbo-Q4_K_M.gguf``) while the parent directory carries none;
    scan the combined string too when the directory alone is undetectable. Only a
    fallback, so remote ``org/name`` picks and explicit overrides behave exactly as
    ``detect_family``. Shared by both engines so validation and load can't diverge."""
    fam = detect_family(repo_id, override)
    if fam is None and gguf_filename and not override:
        fam = detect_family(f"{repo_id}/{gguf_filename}", override)
    return fam


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


# Default (steps, guidance) per model for callers that can't pass them — namely
# the OpenAI /v1/images/generations endpoint, whose spec has no step/guidance
# knobs. Distilled "turbo/schnell" models want few steps and no CFG; the full
# "dev" models want more steps and real CFG. Matched by substring, most specific
# first — the same scheme and values as the UI's MODEL_DEFAULTS table
# (studio/frontend/src/features/images/images-page.tsx); keep the two in sync.
_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("z-image-turbo", 9, 0.0),
    ("flux.1-schnell", 4, 0.0),
    ("flux.1", 28, 3.5),
    ("flux.2-klein", 4, 0.0),
    ("qwen-image", 20, 4.0),
    ("z-image", 20, 4.0),
)
# Unrecognised model: distilled few-step / no-CFG shape, matching the UI fallback.
_GENERATION_DEFAULT_FALLBACK = (9, 0.0)


def default_generation_params(*identifiers: Optional[str]) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded model. The first identifier that
    names a known model wins (the repo id, then the resolved base repo), so a
    local-path load — whose repo id is just a filesystem path that may not name
    the model — still resolves via its base repo. Within an identifier, keys are
    matched as substrings, most specific first (the same scheme as the UI)."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, steps, guidance in _GENERATION_DEFAULTS:
            if key in needle:
                return steps, guidance
    return _GENERATION_DEFAULT_FALLBACK


def family_prequant_repo(fam: DiffusionFamily, scheme: str) -> Optional[str]:
    """The hosted pre-quantized transformer repo for ``scheme`` in this family, or None."""
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """True when the family has the single-file VAE + text-encoder mapping the
    native sd.cpp engine needs. A family without it can only run on diffusers, so
    the no-GPU route falls back rather than routing to sd-cli."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


def resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve ``gguf_filename`` to a file under ``repo_root``, rejecting escapes.

    ``gguf_filename`` is user-supplied, so an absolute path or a ``..`` segment
    could otherwise read outside the local repo directory.
    """
    if (
        Path(gguf_filename).is_absolute()
        or PurePosixPath(gguf_filename).is_absolute()
        or gguf_filename.startswith(("/", "\\"))
        or "\\" in gguf_filename
    ):
        raise ValueError("gguf_filename must be a relative path inside the repo.")
    rel = PurePosixPath(gguf_filename)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("gguf_filename must not contain '', '.', or '..' segments.")
    # Resolve symlinks before the containment check: the guards above stop
    # lexical escapes, but a symlink inside the repo could still point outside it.
    repo_real = repo_root.resolve()
    child = repo_root.joinpath(*rel.parts).resolve()
    if child != repo_real and repo_real not in child.parents:
        raise ValueError("gguf_filename must resolve to a file inside the repo.")
    if not child.is_file():
        raise FileNotFoundError(f"'{gguf_filename}' is not a file under {repo_root}.")
    return child

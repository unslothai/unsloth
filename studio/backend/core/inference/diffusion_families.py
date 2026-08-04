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


# Runtime->route contract: the /images/generate route matches these messages EXACTLY for a 409 (vs a 500), so both engines raise them verbatim.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Pipeline kwarg carrying guidance. Most use "guidance_scale"; Qwen-Image real CFG is "true_cfg_scale".
    cfg_kwarg: str = "guidance_scale"
    # The pipe attribute holding the denoiser: ``pipe.transformer`` for DiT families, ``pipe.unet`` for SDXL.
    denoiser_attr: str = "transformer"
    # True when a single-file ``.safetensors`` is the WHOLE pipeline (SDXL), so the loader calls ``from_single_file``.
    single_file_is_pipeline: bool = False
    # True for families needing MULTIPLE denoisers no single file carries (Ideogram 4), so only a full pipeline load is valid.
    pipeline_only: bool = False
    # Optional diffusers pipeline classes for image-conditioned workflows, built via ``Pipeline.from_pipe`` (no reload). None = unsupported.
    img2img_pipeline_class: Optional[str] = None
    inpaint_pipeline_class: Optional[str] = None
    # ControlNet pipeline + model classes: the model loads via from_pretrained, the pipeline via ``from_pipe``. None on both = no support.
    controlnet_pipeline_class: Optional[str] = None
    controlnet_model_class: Optional[str] = None
    # True when the inpaint pipeline keeps the canvas size, so it can also drive outpaint. False for FLUX.2 (it scales >1MP inputs to ~1MP).
    inpaint_preserves_size: bool = True
    # True for instruction-editing families (Qwen-Image-Edit / FLUX Kontext): the pipeline IS the edit pipeline (image +
    # instruction, no plain text-to-image). ``base_repo`` supplies the VAE / text-encoder / processor / scheduler.
    edit: bool = False
    # True for families whose text-to-image pipeline ALSO accepts reference image(s) (FLUX.2 ``image``): no ``strength``, size from width/height.
    reference: bool = False
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True for families whose activations overflow float16 (-> black image); the backend promotes a resolved float16 to float32.
    fp16_incompatible: bool = False
    # False only for a family whose denoiser block does not compile cleanly with regional torch.compile.
    supports_torch_compile: bool = True
    # Optional pre-quantized transformer checkpoints as (scheme, repo_id): fetched instead of the dense bf16 (lower load VRAM + download).
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Hosted checkpoints for NON-DEFAULT bases as (base_repo, scheme, repo_id), base_repo lowercased: one family entry covers
    # variants whose weights differ. Resolution prefers an exact variant match, then falls back to ``prequant_repos``.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id). Layerwise-fp8 only: the cast is deterministic, so the artifact is bit-identical while skipping the dense TE download.
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Native (sd.cpp) single-file assets, used only on the no-GPU sd.cpp engine. The transformer GGUF is shared with
    # diffusers; sd-cli also needs a (repo_id, filename) VAE + text encoder(s), each with a trailing SdCppModelFiles field name.
    sd_cpp_vae: Optional[tuple[str, str]] = None
    # VAE latent-format override for sd-cli (--vae-format): "flux2" for FLUX.2, None otherwise.
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Family-specific sd-cli sampler settings so native output matches the model's supported invocation. None leaves sd-cli defaults.
    sd_cpp_sampling_method: Optional[str] = None
    sd_cpp_flow_shift: Optional[float] = None
    # True when Studio can TRAIN a LoRA on this family; the training-start path refuses a non-trainable family up front.
    trainable: bool = False
    # Recommended base repos to train FROM, most-preferred first (e.g. a QLoRA prequant repo, then bf16). Surfaced by the Train UI.
    train_base_repos: tuple[str, ...] = field(default_factory = tuple)
    # When set, deploying a LoRA trained on this family loads THIS repo instead (Krea: train on Raw, preview on Turbo). Same precision both sides.
    deploy_base_repo: Optional[str] = None


# Keyed by architecture, not per variant: the base repo is read from the HF base_model tag at load time, so one entry covers Turbo/full, schnell/dev.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        # Hosted pre-quantized DiT checkpoints (gate-validated vs same-seed bf16). The loader verifies the baked base_model_id, so a non-default base falls back to dense-quantize.
        prequant_repos = (
            ("int8", "unsloth/FLUX.1-schnell-FP8"),
            ("fp8", "unsloth/FLUX.1-schnell-FP8"),
        ),
        # Checkpoints baked from the dev / Krea-dev weights (same arch, different weights); without these every int8/fp8 load pays the dense download + on-the-fly quantise.
        prequant_variant_repos = (
            ("black-forest-labs/flux.1-dev", "int8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-dev", "fp8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "int8", "unsloth/FLUX.1-Krea-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "fp8", "unsloth/FLUX.1-Krea-dev-FP8"),
        ),
        # Pre-cast T5-XXL (9.52 -> 5.90 GB; CLIP-L stays dense). One artifact serves schnell/dev/Krea-dev (T5 shards are byte-identical).
        te_prequant_repos = (("fp8", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8"),),
        aliases = ("flux1", "flux-1"),
        # LoRA training targets FLUX.1-dev via the DiT trainer (QLoRA nf4); the dev repo is gated.
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.1-dev",),
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
    # FLUX.2-klein is Flux2KleinPipeline (Qwen3 encoder), not the Mistral Flux2Pipeline, so it must precede a generic flux match.
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-klein-4B-FP8"),
            ("fp8", "unsloth/FLUX.2-klein-4B-FP8"),
        ),
        aliases = ("flux2-klein",),
        # LoRA training via the DiT trainer (QLoRA nf4 by default); klein-4B is not gated.
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.2-klein-4B",),
        # Flux2KleinPipeline takes reference image(s) via `image`, so it exposes a "reference" workflow atop text-to-image. Inpaint but no img2img.
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        # FLUX.2 scales >1MP inputs to ~1MP, so outpaint can't grow.
        inpaint_preserves_size = False,
        # FLUX.2's 32-channel AE needs the latent-format override; the single-file VAE ships in Comfy-Org/flux2-dev. Shares Qwen3-4B with z-image.
        sd_cpp_vae = ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    # FLUX.2-dev: full (non-distilled) FLUX.2 on the Mistral Flux2Pipeline, so its own entry. Gated base, text-to-image only; VAE + Mistral encoder come from Comfy-Org/flux2-dev for sd-cli.
    DiffusionFamily(
        name = "flux.2-dev",
        pipeline_class = "Flux2Pipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-dev-FP8"),
            ("fp8", "unsloth/FLUX.2-dev-FP8"),
        ),
        # Pre-cast Mistral-Small-24B conditioner (bf16 ~48 GB dense, ~24.7 GB pre-cast).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/FLUX.2-dev-FP8"),),
        aliases = ("flux2-dev", "flux2dev"),
        # LoRA training via the DiT trainer (QLoRA nf4); the gated base needs an HF token with the FLUX.2-dev license accepted.
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.2-dev",),
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
        # FLUX instruction editing: FluxKontextPipeline takes an image + instruction. Specific aliases first so detect_family prefers this over "flux.1".
        name = "flux.1-kontext",
        pipeline_class = "FluxKontextPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        aliases = ("flux.1-kontext-dev", "flux1-kontext", "flux-kontext", "kontext"),
        edit = True,
    ),
    DiffusionFamily(
        # Qwen instruction editing: the 2511 checkpoint ships as QwenImageEditPlusPipeline. Specific aliases first so detect_family prefers this over "qwen-image".
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
        # int8 only: fp8 is family-denied (_FAMILY_SCHEME_DENY) so a repo entry would be dead.
        prequant_repos = (("int8", "unsloth/Qwen-Image-FP8"),),
        # Pre-cast Qwen2.5-VL-7B (16.6 -> 8.8 GB). The DiT fp8 denial is a transformer-scheme rule; the layerwise TE cast is unaffected.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image", "qwenimage"),
        # LoRA training via the DiT trainer, defaulting to the prequant nf4 repo (QLoRA).
        trainable = True,
        train_base_repos = ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "Qwen/Qwen-Image"),
        img2img_pipeline_class = "QwenImageImg2ImgPipeline",
        inpaint_pipeline_class = "QwenImageInpaintPipeline",
        controlnet_pipeline_class = "QwenImageControlNetPipeline",
        controlnet_model_class = "QwenImageControlNetModel",
        sd_cpp_vae = ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        # Qwen2.5-VL as a Q4_K_M GGUF keeps the CPU RAM win (bf16 encoder is ~15 GB). sd-cli --qwen2vl aliases --llm.
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
        prequant_repos = (
            ("int8", "unsloth/Z-Image-Turbo-FP8"),
            ("fp8", "unsloth/Z-Image-Turbo-FP8"),
        ),
        # Pre-cast Qwen3-4B TE (8.04 -> 4.41 GB). NOT shared with flux.2-klein-4B: klein TE retrained layer 35 MLP (up/down_proj maxdiff 0.86 vs this checkpoint).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Z-Image-Turbo-FP8"),),
        aliases = ("zimage", "z_image"),
        # LoRA training via the DiT trainer (bf16); defaults to the prequant nf4 repo for QLoRA.
        trainable = True,
        train_base_repos = ("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", "Tongyi-MAI/Z-Image-Turbo"),
        img2img_pipeline_class = "ZImageImg2ImgPipeline",
        inpaint_pipeline_class = "ZImageInpaintPipeline",
        # Z-Image's MLP down-projections peak near 9e5, which overflows float16.
        fp16_incompatible = True,
        sd_cpp_vae = ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    # Krea 2 (diffusers >= 0.39): a ~12B single-stream DiT with a Qwen3-VL-4B encoder and the Qwen-Image VAE. Loaded per-component because the repo ships transformers-5.x configs.
    DiffusionFamily(
        name = "krea-2",
        pipeline_class = "Krea2Pipeline",
        transformer_class = "Krea2Transformer2DModel",
        base_repo = "krea/Krea-2-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Krea-2-Turbo-FP8"),
            ("fp8", "unsloth/Krea-2-Turbo-FP8"),
        ),
        # Pre-cast Qwen3-VL-4B TE (8.88 -> 4.83 GB); handed into load_krea2_pipeline directly (assembly never sees pipe_kwargs).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Krea-2-Turbo-FP8"),),
        aliases = ("krea2",),
        # LoRA training via the DiT trainer (no prequant repo yet, so nf4 quantizes on the fly). Krea guidance: train on the undistilled Raw, run adapters on Turbo.
        trainable = True,
        train_base_repos = ("krea/Krea-2-Raw", "krea/Krea-2-Turbo"),
        # Adapters trained on Raw run on Turbo; deploy previews them there (same bf16 precision).
        deploy_base_repo = "krea/Krea-2-Turbo",
        # Exported bf16-only; fp16 unvalidated upstream, so keep the fp16 fallback off like z-image.
        fp16_incompatible = True,
    ),
    # Lumina Image 2.0: a 2.6B single-stream DiT with a Gemma2-2B encoder and a standard AutoencoderKL, so the generic
    # from_pretrained path loads it. NOT aliased to bare "lumina": Lumina-Next checkpoints are a different arch.
    DiffusionFamily(
        name = "lumina-2",
        pipeline_class = "Lumina2Pipeline",
        transformer_class = "Lumina2Transformer2DModel",
        base_repo = "Alpha-VLLM/Lumina-Image-2.0",
        # Gate-validated hosted checkpoints (28/28 pairs each; LPIPS mean 0.146 int8 / 0.116 fp8).
        prequant_repos = (
            ("int8", "unsloth/Lumina-Image-2.0-FP8"),
            ("fp8", "unsloth/Lumina-Image-2.0-FP8"),
        ),
        # Pre-cast Gemma2-2B TE. The Hub stores it fp32 (10.46 GB), so the 3.20 GB artifact is a 3.3x download cut.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Lumina-Image-2.0-FP8"),),
        aliases = ("lumina-image-2.0", "lumina-image-2", "lumina2"),
        # Published and validated bf16-only upstream; keep the fp16 fallback off like z-image.
        fp16_incompatible = True,
    ),
    # HunyuanImage 2.1 (diffusers >= 0.39): a 17B dual-stream DiT with a Qwen2.5-VL encoder, a ByT5 glyph encoder and the
    # 32x HunyuanImage VAE. 2K-native; CFG runs inside the repo guider and the call knob is distilled_guidance_scale.
    DiffusionFamily(
        name = "hunyuanimage-2.1",
        # Hosted checkpoints, verified bit-identical to on-the-fly quantize (the guider pipeline is not run-to-run deterministic, so same-seed LPIPS mixes trajectory divergence with harness noise).
        prequant_repos = (
            ("int8", "unsloth/HunyuanImage-2.1-FP8"),
            ("fp8", "unsloth/HunyuanImage-2.1-FP8"),
        ),
        # The Qwen2.5-VL TE is byte-identical to Qwen-Image (verified sha256), so reuse that artifact: 16.58 -> 8.84 GB. ByT5 stays dense.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        pipeline_class = "HunyuanImagePipeline",
        transformer_class = "HunyuanImageTransformer2DModel",
        base_repo = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        cfg_kwarg = "distilled_guidance_scale",
        aliases = ("hunyuanimage-2.1-diffusers", "hunyuanimage2.1"),
        # Exported bf16-only; keep the fp16 fallback off like z-image / krea-2.
        fp16_incompatible = True,
    ),
    # HiDream-I1: a 17B MoE DiT with FOUR text encoders. The repos ship CLIP-L/CLIP-G/T5-XXL but NOT the Llama-3.1-8B
    # text_encoder_4 their model_index names, so the loader assembles it from the open unsloth mirror. One family covers Full/Dev/Fast.
    DiffusionFamily(
        name = "hidream-i1",
        # Hosted checkpoints: 28/28 per-case gate pairs per scheme (LPIPS 0.291 int8 / 0.278 fp8); int8 bit-identical to on-the-fly quantize.
        prequant_repos = (
            ("int8", "unsloth/HiDream-I1-Full-FP8"),
            ("fp8", "unsloth/HiDream-I1-Full-FP8"),
        ),
        # Pre-cast Llama-3.1-8B TE4 (16.1 -> 8.1 GB). The generic TE pass only covers text_encoder.._3, so TE4 engages via hidream_te4_kwargs.
        te_prequant_repos = (("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),),
        pipeline_class = "HiDreamImagePipeline",
        transformer_class = "HiDreamImageTransformer2DModel",
        base_repo = "HiDream-ai/HiDream-I1-Full",
        aliases = ("hidream", "hidream-i1-full", "hidream-i1-dev", "hidream-i1-fast"),
        # Exported bf16-only; keep the fp16 fallback off like the other modern DiTs.
        fp16_incompatible = True,
    ),
    # Ideogram 4 (diffusers >= 0.39): a 34-layer DiT PAIR (conditional + unconditional, ~9B each, so planning counts two)
    # with a Qwen3-VL encoder. No bf16 ships: -fp8 is the family base, -nf4 carries bnb-4bit. CFG takes guidance_scale OR a per-step schedule.
    DiffusionFamily(
        name = "ideogram-4",
        pipeline_class = "Ideogram4Pipeline",
        transformer_class = "Ideogram4Transformer2DModel",
        base_repo = "ideogram-ai/ideogram-4-fp8",
        aliases = ("ideogram4", "ideogram-v4", "ideogram"),
        # Two DiTs assembled per-component, so no transformer-only single-file / GGUF load.
        pipeline_only = True,
    ),
    # SDXL is the one U-Net family: the denoiser is ``pipe.unet`` and a single-file ``.safetensors`` is the WHOLE pipeline. img2img / inpaint / ControlNet are the standard SDXL pipelines. No GGUF path.
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
        # SDXL uses the U-Net LoRA trainer.
        trainable = True,
        train_base_repos = (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
        ),
    ),
)


def trainable_family_names() -> tuple[str, ...]:
    """Names of families Studio can train a LoRA on, in registry order."""
    return tuple(fam.name for fam in _FAMILIES if fam.trainable)


# The family whose CFG uses a guidance_scale/guidance_schedule pair. Named here so the two modules cannot drift.
IDEOGRAM4_FAMILY_NAME = "ideogram-4"

# The family whose generate call carries the card CFG-truncation ratio. Named here so the two modules cannot drift.
LUMINA2_FAMILY_NAME = "lumina-2"


# Models Studio deliberately does NOT support, reason surfaced verbatim in the load error, keyed by lowercase repo-id substring. The bar is a diffusers pipeline.
_EXCLUDED_MODELS: tuple[tuple[str, str], ...] = (
    (
        # "-3" scoped so a future HunyuanImage 2.x with a diffusers pipeline falls through normally.
        "hunyuanimage-3",
        "HunyuanImage-3.0 has no diffusers pipeline (it is an 80B autoregressive MoE "
        "that requires trust_remote_code), so Studio does not support it.",
    ),
)


def excluded_model_reason(repo_id: str) -> Optional[str]:
    """The stated reason ``repo_id`` is unsupported, or None when it is simply unknown."""
    needle = (repo_id or "").lower()
    for token, reason in _EXCLUDED_MODELS:
        if _token_in_needle(token, needle):
            return reason
    return None


# Editing / inpaint checkpoints share an arch keyword but need a different pipeline + input image. "layered" rejects Qwen-Image-Layered, whose transformer expects an extra input.
_EDIT_KEYWORDS = ("edit", "kontext", "inpaint", "layered")


def _token_in_needle(token: str, needle: str) -> bool:
    """True when ``token`` appears in ``needle`` as a whole segment (delimited by ``- _ . / \\`` or
    a boundary), not a raw substring, so 'qwen-image-edit' matches '...-2511' but 'kontext' doesn't
    match 'kontextual'."""
    return re.search(r"(?:^|[-_./\\])" + re.escape(token) + r"(?:$|[-_./\\])", needle) is not None


def _best_family_match(needle: str) -> Optional[DiffusionFamily]:
    """The family whose name/alias is the LONGEST whole-segment token of ``needle`` (longest = most
    specific, so '...qwen-image-edit-2511...' matches 'qwen-image-edit', not 'qwen-image')."""
    best: Optional[tuple[DiffusionFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or len(token) > best[1]):
                best = (fam, len(token))
    return best[0] if best else None


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Resolve a ``DiffusionFamily`` from a repo id, or an explicit override.

    ``override`` matches a family ``name``/alias exactly; otherwise the most-specific family whose
    name/alias is a substring of the repo id wins. Supported editing families match here;
    unsupported editing/inpaint/layered checkpoints sharing only an arch keyword are rejected (None).
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
        # Do not let a generic family (qwen-image) swallow a variant it cannot run (qwen-image-LAYERED). Scoped to the LAST path component so a parent folder named `edit` does not reject a valid file.
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
    """Family names accepted as ``family_override`` and shown in the unknown-model error (registry
    order)."""
    return tuple(fam.name for fam in _FAMILIES)


def detect_family_for_pick(
    repo_id: str,
    gguf_filename: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[DiffusionFamily]:
    """``detect_family``, falling back to the combined path/filename for a local ``.gguf`` pick
    where the family keyword lives only in the filename. Only a fallback, so remote picks and
    overrides behave exactly as ``detect_family``. Shared by both engines."""
    fam = detect_family(repo_id, override)
    if fam is None and gguf_filename and not override:
        fam = detect_family(f"{repo_id}/{gguf_filename}", override)
    return fam


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


# Default (steps, guidance) per model for callers that cannot pass them. Matched by substring, most specific first; same values as the UI MODEL_DEFAULTS table, keep in sync.
_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("z-image-turbo", 9, 0.0),
    # FLUX.1 Krea dev is a FLUX.1-dev finetune, NOT a Krea-2: 28 steps at guidance 4.5. Must precede the generic "krea" key.
    ("flux.1-krea", 28, 4.5),
    # Krea 2 Raw (undistilled): 52 steps / guidance 3.5. Must precede the generic "krea" key.
    ("krea-2-raw", 52, 3.5),
    # Krea 2 Turbo (distilled): 8 steps, no CFG. "krea" then covers Turbo and other krea ids but Raw.
    ("krea", 8, 0.0),
    ("flux.1-schnell", 4, 0.0),
    ("kontext", 28, 2.5),  # editing: before the generic flux.1
    ("flux.1", 28, 3.5),
    ("flux.2-klein", 4, 0.0),
    ("flux.2-dev", 28, 4.0),  # full (non-distilled)
    ("qwen-image", 20, 4.0),
    ("z-image", 20, 4.0),
    # Lumina Image 2.0 card: 50 steps, guidance 4 (plus cfg_trunc_ratio 0.25, which the loader passes itself).
    ("lumina", 50, 4.0),
    # HunyuanImage 2.1 card: 50 steps; guidance feeds distilled_guidance_scale, while real CFG runs inside the guiders.
    ("hunyuanimage", 50, 3.25),
    # HiDream-I1 upstream: Full 50 steps / guidance 5; the distilled Dev (28) and Fast (16) run guidance-free.
    ("hidream-i1-dev", 28, 0.0),
    ("hidream-i1-fast", 16, 0.0),
    ("hidream", 50, 5.0),
    # Ideogram 4 card: 48 steps, guidance 7 (its schedule tapers the last 3 steps; the loader keeps that taper at these defaults).
    ("ideogram", 48, 7.0),
    # SDXL: Turbo distilled; base wants ~30 steps + CFG ~7. "sdxl-turbo" precedes "sdxl".
    ("sdxl-turbo", 3, 0.0),
    ("stable-diffusion-xl", 30, 7.0),
    ("sdxl", 30, 7.0),
)
# Unrecognised model: distilled few-step / no-CFG shape, matching the UI fallback.
_GENERATION_DEFAULT_FALLBACK = (9, 0.0)


def default_generation_params(*identifiers: Optional[str]) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded model. The first identifier naming a known model
    wins (repo id, then resolved base repo), so a local-path load still resolves via its base repo.
    Keys matched as substrings, most specific first."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, steps, guidance in _GENERATION_DEFAULTS:
            if key in needle:
                return steps, guidance
    return _GENERATION_DEFAULT_FALLBACK


def family_prequant_repo(
    fam: DiffusionFamily,
    scheme: str,
    base_repo: Optional[str] = None,
) -> Optional[str]:
    """The hosted pre-quantized transformer repo for ``scheme`` in this family, or None.

    ``base_repo`` (when known) selects a variant-specific checkpoint first: a checkpoint is
    baked from ONE base's weights and the loader refuses it for any other base, so a variant
    without its own entry still returns the family default (harmless: the base_model_id
    validation then falls back to dense-quantise, exactly as before this table existed)."""
    base = (base_repo or "").strip().lower()
    if base:
        for entry_base, entry_scheme, repo_id in fam.prequant_variant_repos:
            if entry_base == base and entry_scheme == scheme:
                return repo_id
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def assert_pipeline_class_available(pipeline_class: str, family_name: str) -> None:
    """Raise ``ValueError`` before any download when the installed diffusers has no
    ``pipeline_class``.

    The newer families (Flux2Klein, Z-Image, Krea 2, LTX-2, HunyuanImage) only exist from diffusers
    0.39, and the packaging leaves an older diffusers installable on Python 3.9 -- diffusers dropped
    3.9 in 0.38 and this project still supports it, so the 0.39 floor has to be conditional or the
    whole extra becomes unresolvable. Without this check the getattr chain died with a bare
    AttributeError deep in the load, after the checkpoint had already been fetched, which is an
    expensive way to learn the environment is too old. Krea 2 already guarded itself this way; this
    is the same check for every family, run from validation.

    ``ValueError``, like every other unloadable-pick refusal ``validate_load_request`` raises, so
    the routes map it to 400 with the message intact. A ``RuntimeError`` instead reached
    ``/images/load``'s 409 (the code that otherwise means "a load is already in progress") and
    escaped ``/images/download-plan``, which catches only (ValueError, FileNotFoundError), as a bare
    500 with the message lost."""
    try:
        import diffusers
    except ImportError:
        # Not this check's business: it answers "is the installed diffusers new enough for this family", and with
        # nothing installed there is no version to judge. Refusing would also break the native sd.cpp engine, which
        # serves GGUF picks on a CPU or Apple host without diffusers. A pick that really needs it fails later, in
        # the loader. The one thing that must not happen is a raise: ModuleNotFoundError is not the ValueError the
        # routes map to 400, so it escapes /images/download-plan as a bare 500 with the message lost.
        return

    if hasattr(diffusers, pipeline_class):
        return
    raise ValueError(
        f"'{family_name}' needs diffusers >= 0.39.0 ({pipeline_class}); this environment has "
        f"diffusers {getattr(diffusers, '__version__', 'unknown')}. Upgrade with: "
        f"pip install -U diffusers (which needs Python >= 3.10; diffusers dropped 3.9 in 0.38)."
    )


def family_pipeline_available(fam: Optional[DiffusionFamily]) -> bool:
    """True when the installed diffusers actually has this family's pipeline class.

    The boolean twin of ``assert_pipeline_class_available``, for the listing routes: the newer
    families exist only from diffusers 0.39, and the packaging leaves an older diffusers
    installable on Python 3.9 (diffusers dropped 3.9 in 0.38, so the 0.39 floor has to be
    conditional or the extra becomes unresolvable). Advertising Z-Image or Krea 2 in the picker
    on such an environment offers a pick that can only fail, and no `pip install -U diffusers`
    can fix it without also upgrading Python. Fails OPEN (True) when diffusers cannot be
    imported at all, so a listing never hides a model over an unrelated import problem. The
    attribute lookup is inside the guard for the same reason: diffusers resolves its pipelines
    lazily, so the class name is only a hasattr for a name it does not know -- for one it does, the
    lookup imports that pipeline module and can raise something other than AttributeError."""
    if fam is None:
        return False
    try:
        import diffusers
        return hasattr(diffusers, fam.pipeline_class)
    except Exception:  # noqa: BLE001 -- no diffusers here: the load path reports it properly
        return True


def family_gguf_loadable(fam: DiffusionFamily) -> bool:
    """True when a GGUF transformer can be assembled for this family.

    The two exclusions mirror the ones ``DiffusionBackend.validate_load_request`` raises on (which
    keep their own specific messages): a family whose single file IS the whole pipeline has no
    transformer-only GGUF, and a multi-denoiser family has no single transformer to swap. Exposed
    so the model-listing routes can classify a GGUF the same way the loader would, instead of
    keeping a second hand-maintained list that drifts."""
    return not fam.single_file_is_pipeline and not fam.pipeline_only


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """True when the family has the single-file VAE + text-encoder mapping sd.cpp needs; without it
    the no-GPU route falls back to diffusers."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


# FLUX.2-klein 9B pairs with Qwen3-8B, the 4B with the family-default Qwen3-4B (a mismatched encoder fails deep in sd-cli), so pick per variant.
_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS = (
    (
        "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "split_files/text_encoders/qwen_3_8b.safetensors",
        "llm",
    ),
)


def sd_cpp_text_encoders_for(
    fam: DiffusionFamily,
    repo_id: Optional[str] = None,
    gguf_filename: Optional[str] = None,
) -> tuple[tuple[str, str, str], ...]:
    """The sd.cpp text encoders for a specific load.

    FLUX.2-klein picks by variant (9B needs Qwen3-8B, 4B the family default) keyed on the load
    identity (repo id + GGUF filename); every other family returns its static table."""
    if fam.name == "flux.2-klein":
        identity = f"{repo_id or ''}/{gguf_filename or ''}".lower()
        # Match the size token on its own: klein-BASE-9B is 9B too, and matching "klein-9b"
        # literally handed it the 4B text encoder.
        if _token_in_needle("9b", identity) or "klein9b" in identity:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
    return fam.sd_cpp_text_encoders


# FLUX.2 sizes the adaLN modulation projection as (6 * inner_dim, inner_dim), and inner_dim is the
# only thing that differs between the klein sizes and dev. sd.cpp keys its own FLUX.2 version
# detection on this same tensor, and GGUF metadata cannot help: our files and leejet's carry no kv
# pairs at all, and city96/orabazes write general.architecture = "flux" for FLUX.1 and FLUX.2 alike.
_FLUX2_PROBE_TENSOR = "double_stream_modulation_img.lin.weight"
_FLUX2_INNER_DIMS = {3072: "FLUX.2-klein-4B / klein-base-4B",
                     4096: "FLUX.2-klein-9B / klein-base-9B",
                     6144: "FLUX.2-dev"}
_FLUX2_BASE_INNER_DIM = {
    "black-forest-labs/flux.2-klein-4b": 3072,
    "black-forest-labs/flux.2-klein-base-4b": 3072,
    "black-forest-labs/flux.2-klein-9b": 4096,
    "black-forest-labs/flux.2-klein-base-9b": 4096,
    "black-forest-labs/flux.2-dev": 6144,
}


def gguf_flux2_inner_dim(path) -> Optional[int]:
    """``inner_dim`` of a FLUX.2 GGUF, read from its header, or None if it cannot be determined."""
    try:
        from gguf import GGUFReader

        for t in GGUFReader(str(path)).tensors:
            if t.name == _FLUX2_PROBE_TENSOR or t.name.endswith("." + _FLUX2_PROBE_TENSOR):
                # GGUF stores dims reversed relative to torch, so the input dim leads.
                return int(t.shape[0])
    except Exception:
        return None
    return None


def assert_flux2_gguf_matches_base(fam, base_repo: str, gguf_path) -> None:
    """Fail early, and legibly, when a FLUX.2 GGUF is paired with a different-size base config.

    Without this the mismatch surfaces from inside the GGUF quantizer as a bare shape error
    ("expected torch.Size([18432, 3072]), decodes to (24576, 4096)") that names neither the file
    nor the repo. Fail-open by construction: any unreadable file, non-FLUX.2 tensor set, or
    unmapped base leaves the load exactly as it was."""
    if gguf_path is None or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return
    want = _FLUX2_BASE_INNER_DIM.get((base_repo or "").strip().lower())
    got = gguf_flux2_inner_dim(gguf_path)
    if want is None or got is None or want == got:
        return
    raise ValueError(
        f"'{Path(str(gguf_path)).name}' is a {_FLUX2_INNER_DIMS.get(got, f'FLUX.2 variant with inner_dim {got}')} "
        f"checkpoint, but it is being loaded against '{base_repo}', which is "
        f"{_FLUX2_INNER_DIMS.get(want, f'inner_dim {want}')}. Pass base_repo for the matching "
        f"variant, or pick the GGUF that matches the selected model."
    )


def resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve ``gguf_filename`` (user-supplied) to a file under ``repo_root``, rejecting absolute
    paths and ``..`` escapes."""
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
    # Resolve symlinks before the containment check (the lexical guards miss a symlink escape).
    repo_real = repo_root.resolve()
    child = repo_root.joinpath(*rel.parts).resolve()
    if child != repo_real and repo_real not in child.parents:
        raise ValueError("gguf_filename must resolve to a file inside the repo.")
    if not child.is_file():
        raise FileNotFoundError(f"'{gguf_filename}' is not a file under {repo_root}.")
    return child

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


# Runtime->route contract: RuntimeError messages for client-recoverable generate states. The
# /images/generate route matches these EXACTLY for a 409 (vs a 500), so both engines raise them
# verbatim -- named here, not as scattered literals.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Pipeline kwarg carrying guidance. Most use "guidance_scale"; Qwen-Image's real CFG is
    # "true_cfg_scale" (its distilled guidance is off).
    cfg_kwarg: str = "guidance_scale"
    # The pipe attribute holding the denoiser: DiT families ``pipe.transformer`` (default), U-Net
    # families (SDXL) ``pipe.unet``. Read wherever the backend touches the denoiser generically.
    denoiser_attr: str = "transformer"
    # True when a single-file ``.safetensors`` is the WHOLE pipeline (SDXL), so the loader calls
    # ``pipeline_class.from_single_file`` directly. DiT families leave this False (transformer-only).
    single_file_is_pipeline: bool = False
    # True for families whose pipeline needs MULTIPLE denoisers no single file carries (Ideogram 4:
    # conditional + unconditional_transformer), so only a full ``pipeline`` load is valid;
    # validate_load_request rejects single-file / GGUF kinds up front.
    pipeline_only: bool = False
    # Optional diffusers pipeline classes for image-conditioned workflows, built around the resident
    # modules via ``Pipeline.from_pipe`` (no reload). None = unsupported (UI gates it off).
    img2img_pipeline_class: Optional[str] = None
    inpaint_pipeline_class: Optional[str] = None
    # ControlNet pipeline + model classes: the backend loads the model via from_pretrained and
    # builds the pipeline via ``from_pipe(base, controlnet=model)`` (no reload). None on both =
    # no support (UI gates it off).
    controlnet_pipeline_class: Optional[str] = None
    controlnet_model_class: Optional[str] = None
    # True when the inpaint pipeline keeps the canvas size, so it can also drive outpaint. False for
    # FLUX.2 (it scales >1MP inputs to ~1MP, shrinking an outpaint canvas) -> Inpaint but not Extend.
    inpaint_preserves_size: bool = True
    # True for instruction-editing families (Qwen-Image-Edit / FLUX Kontext): the OWN pipeline IS the
    # edit pipeline (image + instruction, no plain text-to-image), used directly (no from_pipe).
    # ``base_repo`` supplies the VAE / text-encoder / processor / scheduler for the GGUF transformer.
    edit: bool = False
    # True for families whose text-to-image pipeline ALSO accepts reference image(s) (FLUX.2's
    # ``image`` arg). Unlike ``edit`` they still do plain text-to-image; unlike img2img the
    # conditioning is reference-based (no ``strength``, output size from width/height). Used directly.
    reference: bool = False
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True for families whose activations overflow float16 (-> inf/NaN -> black image); the backend
    # promotes a resolved float16 to float32 for these.
    fp16_incompatible: bool = False
    # False only for a family whose denoiser block doesn't compile cleanly with regional
    # torch.compile. Consulted on the GGUF path too; all current families compile, so this stays True.
    supports_torch_compile: bool = True
    # Optional pre-quantized transformer checkpoints as (scheme, repo_id) pairs. When the fast quant
    # path resolves a scheme with a hosted checkpoint, the loader fetches the already-quantized
    # weights instead of the dense bf16 (lower load VRAM + smaller download). Empty -> unchanged.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Hosted checkpoints for NON-DEFAULT bases of the family, as (base_repo, scheme, repo_id)
    # triples with base_repo lowercased. One family entry covers several published variants
    # (flux.1: schnell/dev/Krea-dev) whose weights differ, so each variant needs its own baked
    # checkpoint; the loader's base_model_id validation correctly refuses the default entry for
    # them. Resolution prefers an exact variant match, then falls back to ``prequant_repos``.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id) triples
    # (component is the pipeline attribute, e.g. "text_encoder"). Serves the layerwise-fp8
    # storage scheme only: the cast is a deterministic transform, so the stored artifact is
    # bit-identical to dense-load-then-cast while skipping the multi-GB dense TE download
    # (see diffusion_te_prequant.py). Empty -> the TE loads dense and casts as before.
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Native (sd.cpp) single-file assets, used only on the no-GPU sd.cpp engine. The transformer GGUF
    # is shared with diffusers; sd-cli also needs a single-file VAE + text encoder(s) (the base repo
    # ships those sharded). Each is a (repo_id, filename); ``sd_cpp_text_encoders`` carries a trailing
    # SdCppModelFiles field name (clip_l / t5xxl / llm / qwen2vl / clip_g) for the sd-cli flag. Empty
    # -> no native mapping (sd.cpp route falls back to diffusers).
    sd_cpp_vae: Optional[tuple[str, str]] = None
    # VAE latent-format override for sd-cli (--vae-format): "flux2" for FLUX.2, None otherwise.
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Family-specific sd-cli sampler settings so the native output matches the model's supported
    # invocation (e.g. Qwen-Image needs euler + flow-shift 3). None leaves sd-cli defaults.
    sd_cpp_sampling_method: Optional[str] = None
    sd_cpp_flow_shift: Optional[float] = None
    # True when Studio can TRAIN a LoRA on this family (a trainer is registered). Opt-in per family
    # (each arch needs its own loop); the training-start path refuses a non-trainable family up front.
    trainable: bool = False
    # Recommended base repos to train FROM, most-preferred first (e.g. a QLoRA prequant repo, then
    # bf16). Surfaced by the Train UI.
    train_base_repos: tuple[str, ...] = field(default_factory = tuple)
    # When set, deploying a LoRA trained on this family loads THIS repo instead of the trained-on
    # checkpoint (Krea: train on Raw, preview on Turbo). Both sides must be the same precision so the
    # swap never enlarges the load. Unset elsewhere.
    deploy_base_repo: Optional[str] = None


# Keyed by architecture, not per variant: a checkpoint's specific base repo is read from its HF
# base_model tag at load time, so one entry covers Turbo/full, schnell/dev, etc. (base_repo here is
# a fallback). Only archs whose diffusers transformer supports from_single_file load here.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        # Hosted pre-quantized DiT checkpoints (gate-validated vs same-seed bf16). The loader
        # verifies the checkpoint's baked base_model_id against the repo actually being loaded,
        # so a non-default base (e.g. FLUX.1-dev under this family) safely falls back to the
        # dense-quantize path instead of loading schnell weights.
        prequant_repos = (
            ("int8", "unsloth/FLUX.1-schnell-FP8"),
            ("fp8", "unsloth/FLUX.1-schnell-FP8"),
        ),
        # Gate-validated checkpoints baked from the dev / Krea-dev weights (same arch, different
        # weights): without these entries the default schnell checkpoint is refused for those
        # bases and every int8/fp8 load pays the dense download + on-the-fly quantise.
        prequant_variant_repos = (
            ("black-forest-labs/flux.1-dev", "int8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-dev", "fp8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "int8", "unsloth/FLUX.1-Krea-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "fp8", "unsloth/FLUX.1-Krea-dev-FP8"),
        ),
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
    # FLUX.2-klein is Flux2KleinPipeline (Qwen3 encoder), not the Mistral Flux2Pipeline; must
    # precede a generic flux match. The Mistral Flux2Pipeline is the flux.2-dev family below.
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
        # Flux2KleinPipeline takes reference image(s) via `image`, so it exposes a "reference"
        # workflow atop text-to-image. It has an inpaint pipeline (no img2img) -> inpaint + extend.
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        # FLUX.2 scales >1MP inputs to ~1MP, so outpaint can't grow.
        inpaint_preserves_size = False,
        # FLUX.2's 32-channel AE needs the latent-format override; the single-file VAE ships in
        # Comfy-Org/flux2-dev (klein-4B has only a sharded diffusers VAE). Shares Qwen3-4B with z-image.
        sd_cpp_vae = ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    # FLUX.2-dev: full (non-distilled) FLUX.2 on the Mistral Flux2Pipeline (distinct from klein), so
    # its own entry. Base repo is gated. Text-to-image only (no Flux2 img2img/inpaint in diffusers
    # 0.38). VAE + Mistral encoder come from the open Comfy-Org/flux2-dev mirror for sd-cli.
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
        # LoRA training via the DiT trainer (QLoRA nf4 by default); the base repo is gated, so
        # training requires an HF token with the FLUX.2-dev license accepted.
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
        # FLUX instruction editing: FluxKontextPipeline takes an image + edit instruction; the GGUF
        # transformer is standard FluxTransformer2DModel. Specific aliases first so detect_family
        # prefers this over "flux.1" and un-rejects the "kontext" keyword.
        name = "flux.1-kontext",
        pipeline_class = "FluxKontextPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        aliases = ("flux.1-kontext-dev", "flux1-kontext", "flux-kontext", "kontext"),
        edit = True,
    ),
    DiffusionFamily(
        # Qwen instruction editing: the 2511 checkpoint ships as QwenImageEditPlusPipeline
        # (multi-image); the GGUF transformer is standard QwenImageTransformer2DModel. Specific
        # aliases first so detect_family prefers this over "qwen-image".
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
        # Pre-cast Qwen2.5-VL-7B (bf16 ~16.6 GB dense, ~8.8 GB pre-cast). The DiT fp8 denial
        # is a transformer-scheme rule; the layerwise TE cast is unaffected.
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
        # Qwen2.5-VL as a Q4_K_M GGUF keeps the CPU RAM win (bf16 encoder is ~15 GB). sd-cli's
        # --qwen2vl aliases --llm.
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
    # Krea 2 (diffusers >= 0.39): a ~12B single-stream DiT with a Qwen3-VL-4B encoder and the
    # Qwen-Image VAE. Loaded per-component (diffusion_krea2.py) because the repo ships
    # transformers-5.x configs. No GGUF/sd.cpp mapping yet.
    DiffusionFamily(
        name = "krea-2",
        pipeline_class = "Krea2Pipeline",
        transformer_class = "Krea2Transformer2DModel",
        base_repo = "krea/Krea-2-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Krea-2-Turbo-FP8"),
            ("fp8", "unsloth/Krea-2-Turbo-FP8"),
        ),
        aliases = ("krea2",),
        # LoRA training via the DiT trainer (no prequant repo yet, so nf4 quantizes on the fly).
        # Krea's guidance: train on the undistilled Raw, run adapters on Turbo, so Raw is the
        # default training base and Turbo the inference/base repo.
        trainable = True,
        train_base_repos = ("krea/Krea-2-Raw", "krea/Krea-2-Turbo"),
        # Adapters trained on Raw run on Turbo; deploy previews them there (same bf16 precision).
        deploy_base_repo = "krea/Krea-2-Turbo",
        # Exported bf16-only; fp16 unvalidated upstream, so keep the fp16 fallback off like z-image.
        fp16_incompatible = True,
    ),
    # Lumina Image 2.0: a 2.6B single-stream DiT with a Gemma2-2B encoder and a standard
    # 16-channel AutoencoderKL, all transformers-4.x-compatible, so the generic
    # from_pretrained pipeline path loads it. No GGUF/sd.cpp mapping exists upstream.
    # NOT aliased to bare "lumina": Lumina-Next checkpoints are a different arch
    # (LuminaText2ImgPipeline) and must stay unknown rather than crash mid-load.
    DiffusionFamily(
        name = "lumina-2",
        pipeline_class = "Lumina2Pipeline",
        transformer_class = "Lumina2Transformer2DModel",
        base_repo = "Alpha-VLLM/Lumina-Image-2.0",
        # Gate-validated hosted checkpoints (28/28 pairs each; LPIPS mean 0.146 int8 /
        # 0.116 fp8 vs same-seed bf16).
        prequant_repos = (
            ("int8", "unsloth/Lumina-Image-2.0-FP8"),
            ("fp8", "unsloth/Lumina-Image-2.0-FP8"),
        ),
        aliases = ("lumina-image-2.0", "lumina-image-2", "lumina2"),
        # Published and validated bf16-only upstream; keep the fp16 fallback off like z-image.
        fp16_incompatible = True,
    ),
    # HunyuanImage 2.1 (diffusers >= 0.39): a 17B dual-stream DiT with a Qwen2.5-VL text
    # encoder, a ByT5 glyph encoder, and the 32x-compression HunyuanImage VAE. The community
    # mirror also ships guider/ocr_guider components (AdaptiveProjectedMixGuidance), which
    # 0.39 loads natively, so the generic from_pretrained pipeline path covers the whole
    # stack. 2K-native (the card recipe renders 2048x2048); classifier-free guidance runs
    # inside the repo's guider at its baked scale, and the call's own guidance knob is
    # distilled_guidance_scale (there is no guidance_scale kwarg). Distinct from
    # HunyuanImage-3.0, which stays excluded above: 2.1 has a real diffusers pipeline.
    DiffusionFamily(
        name = "hunyuanimage-2.1",
        # Hosted checkpoints, verified bit-identical to on-the-fly quantize (the family's
        # guider pipeline is not run-to-run deterministic, so same-seed LPIPS vs bf16 blends
        # trajectory divergence with harness noise; per-case hard checks pass and the drift is
        # compositional, reviewed visually).
        prequant_repos = (
            ("int8", "unsloth/HunyuanImage-2.1-FP8"),
            ("fp8", "unsloth/HunyuanImage-2.1-FP8"),
        ),
        pipeline_class = "HunyuanImagePipeline",
        transformer_class = "HunyuanImageTransformer2DModel",
        base_repo = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        cfg_kwarg = "distilled_guidance_scale",
        aliases = ("hunyuanimage-2.1-diffusers", "hunyuanimage2.1"),
        # Exported bf16-only; keep the fp16 fallback off like z-image / krea-2.
        fp16_incompatible = True,
    ),
    # HiDream-I1: a 17B MoE DiT (16 double + 32 single layers, 4 routed experts) with FOUR text
    # encoders. The repos ship CLIP-L/CLIP-G/T5-XXL but NOT the Llama-3.1-8B text_encoder_4 their
    # model_index names: the loader assembles it from the open unsloth mirror
    # (diffusion_hidream.py). Full / Dev / Fast share the arch, so one family covers all three
    # (per-variant step/guidance defaults below). city96 publishes a GGUF but the GGUF path would
    # need the same TE4 assembly for tiny demand, so no GGUF artifact is wired yet.
    DiffusionFamily(
        name = "hidream-i1",
        # Hosted checkpoints: 28/28 per-case gate pairs per scheme (LPIPS suite means 0.291
        # int8 / 0.278 fp8, the 50-step trajectory band); int8 verified bit-identical to
        # on-the-fly quantize across all 1615 state dict tensors.
        prequant_repos = (
            ("int8", "unsloth/HiDream-I1-Full-FP8"),
            ("fp8", "unsloth/HiDream-I1-Full-FP8"),
        ),
        pipeline_class = "HiDreamImagePipeline",
        transformer_class = "HiDreamImageTransformer2DModel",
        base_repo = "HiDream-ai/HiDream-I1-Full",
        aliases = ("hidream", "hidream-i1-full", "hidream-i1-dev", "hidream-i1-fast"),
        # Exported bf16-only; keep the fp16 fallback off like the other modern DiTs.
        fp16_incompatible = True,
    ),
    # Ideogram 4 (diffusers >= 0.39): a 34-layer DiT PAIR (conditional + unconditional_transformer
    # for dual-branch CFG, both ~9B, so memory planning counts two DiTs) with a Qwen3-VL encoder.
    # No bf16 checkpoint: ideogram-4-fp8 (raw float8, upcast on load) is the highest-precision
    # artifact and the family base; the -nf4 repos carry bnb-4bit quantization_configs. All gated.
    # No GGUF/sd.cpp mapping. CFG quirk: the pipeline takes guidance_scale OR a per-step
    # guidance_schedule (see the loader's IDEOGRAM4 branch).
    DiffusionFamily(
        name = "ideogram-4",
        pipeline_class = "Ideogram4Pipeline",
        transformer_class = "Ideogram4Transformer2DModel",
        base_repo = "ideogram-ai/ideogram-4-fp8",
        aliases = ("ideogram4", "ideogram-v4", "ideogram"),
        # Two DiTs assembled per-component, so no transformer-only single-file / GGUF load.
        pipeline_only = True,
    ),
    # SDXL is the one U-Net family: the denoiser is ``pipe.unet`` and a single-file ``.safetensors``
    # is the WHOLE pipeline, so it sets ``denoiser_attr="unet"`` + ``single_file_is_pipeline=True``
    # and loads via the pipeline class. img2img / inpaint / ControlNet are the standard SDXL
    # pipelines via from_pipe. No GGUF/single-file transformer path and no sd.cpp mapping.
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


# The family whose CFG uses a guidance_scale/guidance_schedule pair (the loader special-cases the
# call). Named here so the two modules can't drift.
IDEOGRAM4_FAMILY_NAME = "ideogram-4"

# The family whose generate call carries the card's CFG-truncation ratio (the loader
# special-cases the call). Named here so the two modules can't drift.
LUMINA2_FAMILY_NAME = "lumina-2"


# Models Studio deliberately does NOT support, reason surfaced verbatim in the load error (vs the
# generic unknown-family message). Keyed by a lowercase repo-id substring. The bar is a diffusers
# pipeline: HunyuanImage-3.0 is an 80B MoE needing AutoModelForCausalLM + trust_remote_code (RCE
# out of the question).
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


# Editing / inpaint checkpoints share an arch keyword but need a different pipeline + input image.
# "layered" rejects Qwen-Image-Layered (its transformer expects an extra addition_t_cond input
# the standard pipeline never supplies, so it crashes at the first denoise). Fails the load fast.
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
        # Don't let a generic family (qwen-image) swallow a variant it can't run
        # (qwen-image-LAYERED): if the id carries a reject keyword the matched family doesn't
        # declare, reject. Scope the check to the LAST path component so a parent folder named
        # `edit` doesn't reject a valid file (the repo_id/filename fallback passes the filename last).
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


# Default (steps, guidance) per model for callers that can't pass them (the OpenAI
# /v1/images/generations endpoint has no step/guidance knobs). Matched by substring, most specific
# first -- same values as the UI's MODEL_DEFAULTS table (images-page.tsx); keep in sync.
_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("z-image-turbo", 9, 0.0),
    # FLUX.1 Krea dev is a FLUX.1-dev finetune (flux.1 family), NOT a Krea-2: its card runs
    # 28 steps at guidance 4.5. Must precede the generic "krea" key below, which would
    # otherwise hand it Krea-2-Turbo's 8-step no-CFG recipe.
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
    # Lumina Image 2.0 model-card: 50 steps, guidance 4 (plus cfg_trunc_ratio 0.25, which the
    # loader passes itself; see LUMINA2_FAMILY_NAME).
    ("lumina", 50, 4.0),
    # HunyuanImage 2.1 model-card: 50 steps; the guidance value feeds the call's
    # distilled_guidance_scale (default 3.25), while real CFG runs inside the repo guiders.
    ("hunyuanimage", 50, 3.25),
    # HiDream-I1 upstream inference.py: Full 50 steps / guidance 5; the distilled Dev (28) and
    # Fast (16) run guidance-free. Specific keys precede the generic "hidream" (Full + fallback).
    ("hidream-i1-dev", 28, 0.0),
    ("hidream-i1-fast", 16, 0.0),
    ("hidream", 50, 5.0),
    # Ideogram 4 model-card: 48 steps, guidance 7 (its schedule tapers the last 3 steps to 3.0;
    # the loader keeps that taper when the request matches these defaults exactly).
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
    fam: DiffusionFamily, scheme: str, base_repo: Optional[str] = None
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


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """True when the family has the single-file VAE + text-encoder mapping sd.cpp needs; without it
    the no-GPU route falls back to diffusers."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


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

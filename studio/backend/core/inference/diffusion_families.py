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

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Optional, Sequence
from utils.paths.path_utils import is_appledouble_metadata


# Runtime->route contract: the /images/generate route matches these messages EXACTLY for a 409 (vs a 500), so both engines raise them verbatim.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class LoadIdentity:
    """What a caller's derived request parameters depend on, as one comparable value.

    ``repo_id`` alone is not one: /images/load takes ``base_repo`` and ``family_override``
    independently of the path, so a local checkpoint reloads as a different model while the
    path stays put, and the images route derives steps/guidance from ``base_repo`` and its
    edit-only verdict from the family (#9448). Loads agreeing on all three derive identical
    parameters, which is exactly when accepting one for the other is correct.

    A type rather than a tuple, so pinning a bare repo id compares unequal and is refused
    instead of matching some other shape by accident.
    """

    repo_id: str
    base_repo: str
    family: str


def load_identity(repo_id, base_repo, family) -> LoadIdentity:
    """``LoadIdentity`` for one load. None and "" describe the same absent field."""
    return LoadIdentity(str(repo_id or ""), str(base_repo or ""), str(family or ""))


class DiffusionModelReplacedError(RuntimeError):
    """Both engines' ``generate`` refusing a ``LoadIdentity`` that is no longer loaded.

    Keeps a caller's per-model steps/guidance and workflow verdict, taken from an earlier
    ``status()`` read, off a model they never validated (#9448). Here rather than in
    ``diffusion`` so the native engine can raise it without importing the torch backend.
    """

    def __init__(self, expected: LoadIdentity, actual: LoadIdentity):
        super().__init__(
            f"The image model was replaced while this request waited "
            f"(expected {expected.repo_id!r}, loaded {actual.repo_id!r}); "
            "retry with fresh parameters."
        )
        self.expected = expected
        self.actual = actual


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
    # Bases (lowercased) with NO hosted checkpoint, which must not inherit ``prequant_repos``: the family
    # fallback names an artifact baked from different weights, and planning acts on it before the load's
    # base_model_id check can refuse it. Only for a base whose weights genuinely differ from the default.
    prequant_excluded_bases: tuple[str, ...] = field(default_factory = tuple)
    # Preferred checkpoint FILENAME for a scheme, as (scheme, filename), overriding the
    # ``<Model>-<SCHEME>.pt`` name ``prequant_repo_filename`` derives. The derived name stays on as
    # the fallback, so a repo hosting BOTH an old and a new artifact serves the new one to a build
    # that asks for it by name and the old one to every build that does not. That is what lets a
    # rotated (v2) checkpoint ship without regressing an already-installed Studio, which would
    # otherwise refuse the v2 tag and fall all the way back to the dense download.
    # A row may also be (scheme, task, filename), which names the artifact for ONE task and beats
    # the task-agnostic row; see ``family_prequant_filename``.
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
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
    # Variant-specific training-base to inference-base mappings. FLUX.2 Klein trains on an
    # undistilled base and runs the adapter on the matching 4-step checkpoint, so its 4B and 9B
    # variants cannot share the single family-wide deploy_base_repo above.
    deploy_base_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)

    def deploy_base_for(self, trained_base: str) -> str:
        """The inference checkpoint paired with ``trained_base``, or the input unchanged."""
        key = canonical_base(trained_base).lower()
        for training_repo, inference_repo in self.deploy_base_repos:
            if canonical_base(training_repo).lower() == key:
                return inference_repo
        return self.deploy_base_repo or trained_base


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
        # Byte-identical mirror of comfyanonymous/flux_text_encoders (CLIP-L + T5-XXL fp16).
        sd_cpp_text_encoders = (
            ("unsloth/flux-text-encoders", "clip_l.safetensors", "clip_l"),
            ("unsloth/flux-text-encoders", "t5xxl_fp16.safetensors", "t5xxl"),
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
        # Train the undistilled bases, then preview their adapters on the matching 4-step models.
        # Both vendor ids resolve through the ungated mirrors at fetch time.
        trainable = True,
        train_base_repos = (
            "black-forest-labs/FLUX.2-klein-base-4B",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        deploy_base_repos = (
            (
                "black-forest-labs/FLUX.2-klein-base-4B",
                "black-forest-labs/FLUX.2-klein-4B",
            ),
            (
                "black-forest-labs/FLUX.2-klein-base-9B",
                "black-forest-labs/FLUX.2-klein-9B",
            ),
        ),
        # Flux2KleinPipeline takes reference image(s) via `image`, so it exposes a "reference" workflow atop text-to-image. Inpaint but no img2img.
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        # FLUX.2 scales >1MP inputs to ~1MP, so outpaint can't grow.
        inpaint_preserves_size = False,
        # FLUX.2's 32-channel AE needs the latent-format override. The VAE is mirrored on its own because BFL licensed it Apache-2.0 while the rest of FLUX.2-dev is non-commercial. Shares Qwen3-4B with z-image.
        sd_cpp_vae = ("unsloth/FLUX.2-VAE", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            (
                "unsloth/Z-Image-Turbo-ComfyUI",
                "split_files/text_encoders/qwen_3_4b.safetensors",
                "llm",
            ),
        ),
    ),
    # FLUX.2-dev: full (non-distilled) FLUX.2 on the Mistral Flux2Pipeline, so its own entry. Gated base, text-to-image only; sd-cli takes the Mistral encoder from unsloth/FLUX.2-dev-ComfyUI and the VAE from unsloth/FLUX.2-VAE.
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
        sd_cpp_vae = ("unsloth/FLUX.2-VAE", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            (
                "unsloth/FLUX.2-dev-ComfyUI",
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
        # int8 only: no fp8 DiT checkpoint is published for this family yet. fp8 is no longer
        # denied for inference, so adding one here would now be live rather than dead.
        prequant_repos = (("int8", "unsloth/Qwen-Image-FP8"),),
        # Pre-cast Qwen2.5-VL-7B (16.6 -> 8.8 GB). Always was independent of the DiT scheme rules.
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
        # Byte-identical mirror of Comfy-Org/Qwen-Image_ComfyUI.
        sd_cpp_vae = ("unsloth/Qwen-Image-ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
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
        # Both hosted checkpoints are baked from the distilled Turbo transformer, so the undistilled base has none and must quantize its own dense weights.
        prequant_excluded_bases = ("tongyi-mai/z-image",),
        # Pre-cast Qwen3-4B TE (8.04 -> 4.41 GB), shared with the undistilled base (byte-identical encoder). NOT shared with flux.2-klein-4B: klein TE retrained layer 35 MLP (up/down_proj maxdiff 0.86 vs this checkpoint).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Z-Image-Turbo-FP8"),),
        aliases = ("zimage", "z_image"),
        # LoRA training via the DiT trainer (bf16); defaults to the prequant nf4 repo for QLoRA.
        trainable = True,
        # The undistilled base is what the upstream DreamBooth recipe trains on. No deploy pairing:
        # its adapters preview on the base itself at the 20-step / guidance 4 recipe.
        train_base_repos = (
            "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
            "Tongyi-MAI/Z-Image-Turbo",
            "Tongyi-MAI/Z-Image",
        ),
        img2img_pipeline_class = "ZImageImg2ImgPipeline",
        inpaint_pipeline_class = "ZImageInpaintPipeline",
        # Z-Image's MLP down-projections peak near 9e5, which overflows float16.
        fp16_incompatible = True,
        # Byte-identical mirror of Comfy-Org/z_image_turbo (AE + Qwen3-4B).
        sd_cpp_vae = ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Z-Image-Turbo-ComfyUI",
                "split_files/text_encoders/qwen_3_4b.safetensors",
                "llm",
            ),
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


def detect_family_by_pipeline_class(class_name: Optional[str]) -> Optional[DiffusionFamily]:
    """The family a saved pipeline's ``model_index.json`` ``_class_name`` names, or None.

    Evidence out of the checkpoint rather than out of its name, the counterpart of a GGUF's
    ``general.architecture``: an HF cache snapshot's leaf is a commit hash, so the listing had no
    name to match and hid a model the load path accepts (#8407).

    Only the BASE class matches. The loader instantiates ``fam.pipeline_class``
    (``diffusion.py:2946``), never the declared class, so tagging an inpaint or img2img checkpoint
    would list it and then load it through the wrong pipeline (its UNet input shape differs), which
    is the listing-versus-loader split this exists to close. A variant stays untagged, the answer
    it got before the index was read at all."""
    key = (class_name or "").strip()
    if not key:
        return None
    for fam in _FAMILIES:
        if fam.pipeline_class and fam.pipeline_class == key:
            return fam
    return None


def pipeline_class_from_index(path: Optional[str]) -> Optional[str]:
    """The ``_class_name`` the diffusers pipeline saved at ``path`` declares, or None.

    Size-capped and schema-free: neither a listing nor a load may be held up by whatever a scan
    folder contains. ``_class_name`` is a LIST for a remote-code community pipeline, which Studio
    cannot load, so only a plain string answers.

    ``utf-8-sig`` because PowerShell writes JSON with a BOM and a hand-authored index is ordinary
    beside a converted checkpoint; read as ``utf-8`` it raises and the model stays hidden, #8407
    again. ``RecursionError`` (a nesting bomb, not a ``ValueError``) is caught too: both callers
    wrap this in a blanket except that reads a raise as detection having succeeded."""
    root = Path(path or "")
    if not str(root):
        return None
    for name in ("model_index.json", "modular_model_index.json"):
        try:
            index = root / name
            if not index.is_file() or index.stat().st_size > 1_000_000:
                continue
            payload = json.loads(index.read_text(encoding = "utf-8-sig"))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            value = payload.get("_class_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def detect_family_by_pipeline_index(path: Optional[str]) -> Optional[DiffusionFamily]:
    """The family the pipeline saved at ``path`` declares in its ``model_index.json``, or None.

    The path-shaped counterpart of ``detect_family_by_pipeline_class``, used by BOTH the listing
    and the loader so the picker and ``validate_load_request`` answer off the same evidence (#8407).

    Carries over ``detect_family``'s variant guard: a directory NAMED for a checkpoint the matched
    family cannot run (``...-layered``) is still refused, so the index only adds models whose name
    said nothing, never overrides a name that said no."""
    fam = detect_family_by_pipeline_class(pipeline_class_from_index(path))
    if fam is None:
        return None
    basename = re.split(r"[/\\]+", str(path).lower())[-1]
    matched_tokens = (fam.name, *fam.aliases)
    if any(
        _token_in_needle(kw, basename) and not any(kw in tok for tok in matched_tokens)
        for kw in _EDIT_KEYWORDS
    ):
        return None
    return fam


def detect_family_for_pick(
    repo_id: str,
    gguf_filename: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[DiffusionFamily]:
    """``detect_family``, falling back to the combined path/filename for a local ``.gguf`` pick
    where the family keyword lives only in the filename, and then to the saved pipeline class of a
    local diffusers pipeline directory. Only fallbacks, so remote picks and overrides behave
    exactly as ``detect_family``. Shared by both engines.

    The index keeps the listing and the loader on one answer: the listing classifies a moved
    pipeline from its ``model_index.json`` (its directory name is a commit hash), and the pick sent
    back is that same opaque path with no family_override, so without this the model is shown as
    text-to-image and then refused as an unsupported family (#8407)."""
    fam = None
    if not override:
        # The checkpoint's own declaration outranks any guess made from its path: a family keyword
        # in ANY ancestor segment otherwise shadows the index (a QwenImagePipeline under
        # `.../flux.1/checkpoint` matched FLUX and never reached it), so the listing, which reads
        # the index, named one family and the loader another for one directory.
        # Remote picks are unaffected: with no local index this is None and the name-based paths
        # below run as before.
        fam = detect_family_by_pipeline_index(repo_id)
    if fam is None:
        fam = detect_family(repo_id, override)
    if fam is None and gguf_filename and not override:
        fam = detect_family(f"{repo_id}/{gguf_filename}", override)
    return fam


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


# Byte-identical unsloth mirrors of the vendor bases: a GGUF/FP8 pick ships only the denoiser, so
# companions come from the base, which is a 401 for gated vendors and a third-party fetch for the
# rest. Swapped at the fetch sites only, never in ``resolve_base_repo``, whose result keys the
# UPSTREAM-id tables below (see ``canonical_base``).
# A mirror stands in for the WHOLE base, bf16 pipeline loads included, so it must be a complete
# copy: a companions-only repo breaks every pick that needs the transformer.
_GATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev"),
    ("black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell"),
    ("black-forest-labs/FLUX.1-Kontext-dev", "unsloth/FLUX.1-Kontext-dev"),
    ("black-forest-labs/FLUX.1-Krea-dev", "unsloth/FLUX.1-Krea-dev"),
    ("black-forest-labs/FLUX.2-dev", "unsloth/FLUX.2-dev"),
    ("black-forest-labs/FLUX.2-klein-9B", "unsloth/FLUX.2-klein-9B"),
    ("black-forest-labs/FLUX.2-klein-base-9B", "unsloth/FLUX.2-klein-base-9B"),
    ("krea/Krea-2-Turbo", "unsloth/Krea-2-Turbo"),
    ("krea/Krea-2-Raw", "unsloth/Krea-2-Raw"),
    ("ideogram-ai/ideogram-4-fp8", "unsloth/ideogram-4-fp8"),
    ("ideogram-ai/ideogram-4-nf4", "unsloth/ideogram-4-nf4"),
    ("ideogram-ai/ideogram-4-nf4-diffusers", "unsloth/ideogram-4-nf4-diffusers"),
)

# Mirrored to drop the third-party fetch, NOT to route around a gate. Every licence here
# permits redistribution, and each mirror carries the upstream licence text plus the notice
# that licence prescribes. Qwen-Image-2512 is the one no other redirect could reach: its
# companions are named by the artifact repo's base_model card tag, not the family table.
#
# Kept apart from the gated pairs because the two answer different questions. Both redirect
# a fetch, but only a GATED upstream justifies overriding a user's existing cache: for these
# the upstream is reachable without credentials, so a complete local snapshot must keep being
# used rather than re-pulled from the mirror.
_UNGATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512"),
    ("Qwen/Qwen-Image", "unsloth/Qwen-Image"),
    ("Qwen/Qwen-Image-Edit-2511", "unsloth/Qwen-Image-Edit-2511"),
    ("black-forest-labs/FLUX.2-klein-4B", "unsloth/FLUX.2-klein-4B"),
    # Lookup is by exact id, so every VARIANT a pick can resolve to needs its own row: the base-4B
    # is what `unsloth/FLUX.2-klein-base-4B-GGUF` resolves to from its card tag, and Dev / Fast are
    # offered directly as bf16 pipeline picks. Without these three the table silently misses them.
    ("black-forest-labs/FLUX.2-klein-base-4B", "unsloth/FLUX.2-klein-base-4B"),
    ("Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo"),
    ("Alpha-VLLM/Lumina-Image-2.0", "unsloth/Lumina-Image-2.0"),
    ("HiDream-ai/HiDream-I1-Full", "unsloth/HiDream-I1-Full"),
    ("HiDream-ai/HiDream-I1-Dev", "unsloth/HiDream-I1-Dev"),
    ("HiDream-ai/HiDream-I1-Fast", "unsloth/HiDream-I1-Fast"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "unsloth/stable-diffusion-xl-base-1.0"),
    ("stabilityai/sdxl-turbo", "unsloth/sdxl-turbo"),
    # NOT mirrored: hunyuanvideo-community/HunyuanImage-2.1-Diffusers. The Tencent Hunyuan
    # Community License permits distribution "exclusively in the Territory", and the Territory
    # excludes the EU, the UK and South Korea. A public Hub repo distributes worldwide, so that
    # mirror cannot be made compliant and the family keeps fetching upstream.
)
_MIRROR_PAIRS: tuple[tuple[str, str], ...] = _GATED_MIRROR_PAIRS + _UNGATED_MIRROR_PAIRS
_GATED_MIRRORS: dict[str, str] = {u.lower(): m for u, m in _MIRROR_PAIRS}
_MIRROR_UPSTREAM: dict[str, str] = {m.lower(): u for u, m in _MIRROR_PAIRS}
_GATED_UPSTREAMS: frozenset[str] = frozenset(u.lower() for u, _m in _GATED_MIRROR_PAIRS)


def mirror_repo(repo_id: Optional[str]) -> Optional[str]:
    """The unsloth mirror of ``repo_id``, or None when it is not a mirrored vendor base."""
    return _GATED_MIRRORS.get((repo_id or "").strip().lower())


def upstream_is_gated(repo_id: Optional[str]) -> bool:
    """True when ``repo_id`` is a vendor base the Hub refuses without accepted terms.

    Distinct from "has a mirror": most of the mirror table is ungated and exists only to keep
    the fetch inside ``unsloth/*``. Only the gated half justifies overriding a user's cache.
    """
    return (repo_id or "").strip().lower() in _GATED_UPSTREAMS


def canonical_base(repo_id: Optional[str]) -> str:
    """A mirror id mapped back to the upstream it copies, else ``repo_id`` unchanged.

    Base-keyed tables hold UPSTREAM ids, so every lookup normalises here: a mirror reaching
    ``_FLUX2_BASE_INNER_DIM`` misses, and that shape guard fails OPEN, so it goes silent.
    """
    base = (repo_id or "").strip()
    return _MIRROR_UPSTREAM.get(base.lower(), base)


# What counts as a real weight file, vs the stray config an interrupted pull leaves behind.
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".ckpt", ".gguf"})


def _root_holds_upstream(root: Path, repo_id: str, wanted: Sequence[str]) -> bool:
    """``_upstream_is_cached`` for ONE cache root. Never raises."""
    try:
        repo = root / f"models--{repo_id.replace('/', '--')}"
        ref = repo / "refs" / "main"
        revs = (
            [repo / "snapshots" / ref.read_text(encoding = "utf-8").strip()]
            if ref.is_file()
            else sorted((repo / "snapshots").iterdir())
        )
        for rev in revs:
            if not rev.is_dir():
                continue
            if wanted:
                if all((rev / name).exists() for name in wanted):
                    return True
            elif any(
                p.suffix.lower() in _WEIGHT_SUFFIXES
                and p.is_file()
                and not is_appledouble_metadata(p)
                for p in rev.rglob("*")
            ):
                return True
        return False
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return False


def _upstream_is_cached(
    repo_id: str,
    files: Optional[Sequence[str]] = None,
    *,
    other_root: bool = False,
) -> bool:
    """Whether the upstream load is SATISFIABLE from the local cache.

    Not "has any blob": one config left by an interrupted or previously-tokened pull would pin every
    later load to the gated upstream and re-raise the 401 the mirror exists to avoid. So the revision
    must hold ``files``, else a real weight file. ``.incomplete`` downloads have no snapshot symlink,
    so they count as absent.

    Only the revision ``refs/main`` names counts, the one a gated fetch falls back to: on a 401 the
    HEAD fails and ``hf_hub_download`` resolves the ref to ONE commit, so a complete but superseded
    revision would read as cached and hand the loader a repo it cannot fetch from. With no ref (a
    commit-pinned download) any revision counts, as before.

    Reads the LIVE cache root; huggingface_hub's import-time constant goes stale after a
    cache-folder change. ``other_root`` adds that constant back, for the callers whose fetch passes
    ``reuse_other_cache_root``: those resolve each file through whichever root holds it, so bytes
    left in the pre-change root really do satisfy the load. OFF by default, because a
    ``from_pretrained`` is pinned to the live root and cannot see the other one -- counting those
    bytes there would send a gated base back to the 401 the mirror exists to avoid.
    """
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        roots = [Path(active_hf_hub_cache())]
        if other_root:
            from huggingface_hub import constants

            # Exactly what ``cache_dir = None`` resolves to, i.e. the root the per-file reuse
            # probe falls back to (file_download reads constants.HF_HUB_CACHE per call).
            fallback = Path(constants.HF_HUB_CACHE)
            if fallback != roots[0]:
                roots.append(fallback)
        wanted = tuple(files or ())
        return any(_root_holds_upstream(root, repo_id, wanted) for root in roots)
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return False


def cache_holds_files(repo_id: str, files: Sequence[str]) -> bool:
    """Whether ``repo_id``'s local cache holds EVERY name in ``files``.

    The same revision rule ``_upstream_is_cached`` applies, exposed for callers that need to know a
    component is complete rather than merely started: a partial pull leaves some shards resident,
    and "some" is not a cache hit for anything that then decides not to download the rest.

    The LIVE root only. It is tempting to count the import-time root as well, since
    ``_prefetch_files`` passes ``reuse_other_cache_root`` and would not re-fetch from it, but the
    prefetch is not the consumer that matters here: the dense fast path this verdict unlocks calls
    ``from_pretrained(cache_dir = hub_cache_dir())``, which is pinned to the live root and cannot
    see the other one. A hit there would widen the plan and then download the whole transformer
    again after eviction, which is the exact outcome the check exists to prevent.
    """
    return bool(files) and _upstream_is_cached(repo_id, tuple(files))


# Lowercased unsloth mirror -> the community repack the tables named before it. The mirrors are
# byte identical, so an existing install that already pulled the repack holds the very same bytes
# under the OLD repo key: the HF cache is keyed by repo id, so re-pointing the table alone would
# re-download up to tens of GB on upgrade and fail outright offline.
_SD_CPP_LEGACY_SOURCES: dict[str, str] = {
    "unsloth/flux-text-encoders": "comfyanonymous/flux_text_encoders",
    "unsloth/qwen-image-comfyui": "Comfy-Org/Qwen-Image_ComfyUI",
    "unsloth/flux.2-vae": "Comfy-Org/flux2-dev",
    "unsloth/flux.2-dev-comfyui": "Comfy-Org/flux2-dev",
    "unsloth/z-image-turbo-comfyui": "Comfy-Org/z_image_turbo",
    "unsloth/flux.2-klein-9b-comfyui": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
    "unsloth/wan2.2-ti2v-5b-gguf": "QuantStack/Wan2.2-TI2V-5B-GGUF",
}


def legacy_source_repo(repo_id: Optional[str]) -> Optional[str]:
    """The community repack ``repo_id`` mirrors, or None when it is not one of the mirrors."""
    return _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())


def prefer_cached_legacy_source(repo_id: str, files: Optional[Sequence[str]] = None) -> str:
    """``repo_id``, or the community repack it mirrors when THAT already satisfies ``files``.

    The mirror stays the preferred source for a fresh install; this only spares an existing one
    from re-fetching bytes it already has under the old repo key. Same ``_upstream_is_cached``
    probe the gated swap uses, so an interrupted or partial repack does not win.

    Both cache roots count: the sd.cpp fetch passes ``reuse_other_cache_root``, so a repack left
    behind by a cache-folder change is still reusable, and only the repo id can reach it -- once
    the id has become the mirror those bytes are unreachable and the load re-pulls several GB
    (offline, it fails outright).

    PURE: table lookup + local stat, no network, so the staging plan and the fetch agree.
    """
    legacy = _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())
    if not legacy:
        return repo_id
    return legacy if _upstream_is_cached(legacy, files, other_root = True) else repo_id


def _is_local_path(base: str) -> bool:
    """Whether ``base`` exists on disk, i.e. a local dir rather than a Hub id.

    A user can clone a base into a relative dir named exactly like the vendor id
    (``black-forest-labs/FLUX.1-dev``), which the loaders deliberately treat as local. OSError from
    an id with invalid path characters just means "not a local path".
    """
    try:
        return Path(base or "").expanduser().exists()
    except OSError:
        return False


def prefer_ungated_mirror(
    base: str,
    hf_token: Optional[str] = None,
    *,
    files: Optional[Sequence[str]] = None,
) -> str:
    """``base``, or its ungated unsloth mirror when that is the better repo to FETCH from.

    A GGUF/FP8 pick carries only the denoiser, so a gated base 401s on the companions; the mirrors
    are byte identical, so this drops the gate without changing a weight. Fetch only: the upstream
    id stays what the picker and status() show, what saved configs hold and what a trained LoRA's
    base_model tag records. The one visible swap is the download manager row, which must name the
    repo actually being pulled -- staging the gated id there is the 401 this exists to remove.

    Declines to today's behaviour under ``UNSLOTH_DIFFUSION_NO_MIRROR``, for a local path, or when
    the upstream already satisfies the load from cache and switching would re-pull tens of GiB.
    ``files`` sharpens that last test to the names about to be fetched; without it any weight counts.

    PURE: table lookup, env read, local stat, NO network. This runs inside pipeline assembly, where
    an earlier ``model_info`` probe of the mirror put a Hub round trip on the load path and broke
    four download-plan tests. A missing mirror surfaces as the ordinary download error.
    ``hf_token`` is unused, kept so callers need not care.
    """
    del hf_token  # noqa: F841 -- signature stability only
    mirror = mirror_repo(base)
    if not mirror or os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip():
        return base
    # A local path is never a Hub id: rewriting one sends loads the other sites resolve on disk
    # (``Path(base).exists()``) to the Hub, skipping the copy already downloaded.
    if _is_local_path(base):
        return base
    return base if _upstream_is_cached(base, files) else mirror


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
    # The undistilled base variants need their model-card 50-step CFG recipe. Keep this before
    # the generic distilled key, which covers both 4B and 9B 4-step checkpoints.
    ("flux.2-klein-base", 50, 4.0),
    ("flux.2-klein", 4, 1.0),
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
    without its own entry still returns the family default. That is harmless only while the
    default is close enough that planning around it costs nothing, since the base_model_id
    validation refuses the artifact well after the plan was made. A base whose weights really
    differ belongs in ``prequant_excluded_bases``, which returns None here instead."""
    # Both tables are keyed on lowercased upstream ids.
    base = canonical_base(base_repo).lower()
    if base:
        # getattr, because the video loader calls this with a VideoFamily, which has no such
        # field. A plain attribute read raises AttributeError, resolve_prequant_source swallows it
        # in its bare except and hands back None, and every video family silently loses its hosted
        # prequant checkpoint to the dense path whenever a base_repo is passed.
        if base in (getattr(fam, "prequant_excluded_bases", ()) or ()):
            return None
        for entry_base, entry_scheme, repo_id in fam.prequant_variant_repos:
            if entry_base == base and entry_scheme == scheme:
                return repo_id
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def family_prequant_filename(
    fam: DiffusionFamily,
    scheme: str,
    task: Optional[str] = None,
) -> Optional[str]:
    """The preferred checkpoint filename this family declares for ``scheme``, or None.

    ``None`` means "use the derived ``<Model>-<SCHEME>.pt`` name", which is every family but the
    ones shipping a second artifact under the same repo and scheme. Not variant-keyed: the
    filename says WHICH artifact, the repo says which base.

    Rows come in two shapes. ``(scheme, filename)`` is the historical one and is TASK-AGNOSTIC.
    ``(scheme, task, filename)`` names an artifact for one task only and wins over the agnostic
    row when ``task`` matches. That distinction exists because a family can hold several denoiser
    PARTITIONS in one repo (MiniMax-H3: keyframe vs reference), whose checkpoints have identical
    key sets and identical metadata and so cannot be told apart by any later check -- picking the
    wrong one generates from the wrong partition rather than failing.

    ``task = None`` therefore sees only the agnostic rows, and a scheme with no row for the task
    asked for falls back to the agnostic one, i.e. exactly today's behaviour. Malformed rows are
    skipped rather than raising: this runs on a refusal path where a table typo must not 500."""
    wanted = (task or "").strip().lower()
    agnostic: Optional[str] = None
    for entry in getattr(fam, "prequant_filenames", ()) or ():
        if not isinstance(entry, (tuple, list)):
            continue
        if len(entry) == 2:
            entry_scheme, filename = entry
            if entry_scheme == scheme and agnostic is None and filename:
                agnostic = filename
        elif len(entry) == 3:
            entry_scheme, entry_task, filename = entry
            if (
                wanted
                and entry_scheme == scheme
                and (entry_task or "").strip().lower() == wanted
                and filename
            ):
                return filename
    return agnostic


# The release where diffusers' own requires-python went ">= 3.10.0", which makes 0.36.0 the newest
# a supported Python 3.9 host can resolve.
_DIFFUSERS_DROPPED_PY39 = "0.37.0"

# First diffusers release exporting each pipeline class, read off ``src/diffusers/__init__.py`` at
# the upstream tags and cross-checked against each release's requires-python on PyPI. An unlisted
# class gets a version-free "a newer diffusers" instead of a number, since the ones left out are
# older than any release in play -- and ``family_pipeline_available`` reads one as available, so
# every class the listing probes belongs here. This exists so the remedy is true: telling a
# 3.9 host that Z-Image needs Python >= 3.10 sends it to upgrade the interpreter when
# ``pip install -U diffusers`` (0.36.0 there) would have been enough.
_PIPELINE_MIN_DIFFUSERS: dict[str, str] = {
    # MiniMax-H3 is judged by its transformer, first available in diffusers 0.40.0.
    "MiniMaxH3Transformer3DModel": "0.40.0",
    "Flux2Pipeline": "0.36.0",
    "ZImagePipeline": "0.36.0",
    "ZImageImg2ImgPipeline": "0.36.0",
    "HunyuanImagePipeline": "0.36.0",
    "HunyuanVideo15Pipeline": "0.36.0",
    "QwenImageControlNetPipeline": "0.36.0",
    "QwenImageEditPlusPipeline": "0.36.0",
    "Flux2KleinPipeline": "0.37.0",
    "ZImageInpaintPipeline": "0.37.0",
    "LTX2Pipeline": "0.37.0",
    "Flux2KleinInpaintPipeline": "0.38.0",
    "Ideogram4Pipeline": "0.39.0",
    "Krea2Pipeline": "0.39.0",
    # Older than the 0.35 baseline, but listed anyway: the packaging leaves an UNCONSTRAINED
    # diffusers installable below 3.10, so an already-present ancient one satisfies the pin, and
    # quoting the 0.39 floor at a family that has existed since 0.30 is the same wrong remedy.
    "QwenImagePipeline": "0.35.0",
    "QwenImageImg2ImgPipeline": "0.35.0",
    "QwenImageInpaintPipeline": "0.35.0",
    "FluxKontextPipeline": "0.35.0",
    "HiDreamImagePipeline": "0.34.0",
    # WanPipeline arrived with Wan2.1 in 0.33.0. The shipped Wan2.2 family wants weights only
    # 0.35 carries, but this table answers class presence, like the attribute probe before it.
    "WanPipeline": "0.33.0",
    "Lumina2Pipeline": "0.33.0",
    "FluxPipeline": "0.30.0",
    "FluxImg2ImgPipeline": "0.30.0",
    "FluxInpaintPipeline": "0.30.0",
}


def _version_tuple(v: str) -> tuple[int, ...]:
    """``"0.37.0" -> (0, 37, 0)`` for ordering. Numeric so 0.9 sorts below 0.10, which a string
    compare gets backwards; a non-numeric part stops the parse rather than raising."""
    out: list[int] = []
    for part in str(v).split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def pipeline_class_requirement(pipeline_class: str) -> tuple[Optional[str], bool]:
    """``(minimum diffusers version, whether that minimum also needs Python >= 3.10)``.

    ``None`` for a class with no entry. That is deliberately not the packaging floor: an unlisted
    class is one old enough that no release in play lacks it (StableDiffusionXLPipeline goes back
    past 0.29), so naming 0.39 would send a supported Python 3.9 host to upgrade its interpreter
    for a class every diffusers it can install already has. Without an entry the refusal says
    "a newer diffusers" and stops there, which is true whatever the class."""
    minimum = _PIPELINE_MIN_DIFFUSERS.get(pipeline_class)
    if minimum is None:
        return None, False
    return minimum, _version_tuple(minimum) >= _version_tuple(_DIFFUSERS_DROPPED_PY39)


def _too_old_message(pipeline_class: str, family_name: str, installed: str) -> str:
    """The refusal text: what is missing, what is installed, and a remedy this interpreter can
    actually carry out."""
    minimum, needs_py310 = pipeline_class_requirement(pipeline_class)
    if minimum is None:
        return (
            f"'{family_name}' needs a newer diffusers ({pipeline_class}); this environment has "
            f"diffusers {installed}. Upgrade with: pip install -U diffusers."
        )
    remedy = f"Upgrade with: pip install -U 'diffusers>={minimum}'."
    if needs_py310:
        remedy += (
            f" diffusers dropped Python 3.9 in {_DIFFUSERS_DROPPED_PY39}, so that release needs "
            f"Python >= 3.10 too."
        )
    return (
        f"'{family_name}' needs diffusers >= {minimum} ({pipeline_class}); this environment has "
        f"diffusers {installed}. {remedy}"
    )


def _dummy_required_backends(cls: object) -> tuple[str, ...]:
    """The backends diffusers says ``cls`` REQUIRES, when ``cls`` is one of its placeholders.

    Required, not missing: ``_backends`` is the class's full requirement list, so a placeholder
    standing in because transformers is absent still lists torch beside it. Naming them all as
    missing, and prescribing a reinstall, is how you tell someone with a working ROCm or CUDA
    build of torch to replace it.

    With a required backend absent (torch, transformers, ...), diffusers still EXPORTS every
    pipeline name, as a ``DummyObject``-metaclassed stand-in from ``diffusers.utils.dummy_*``
    whose ``from_pretrained`` raises ``ImportError`` on the first call. ``hasattr`` therefore
    answers True for a class that cannot be used, which is exactly the "importable" answer the
    strict gate must not accept. Empty tuple for a real class."""
    if not str(getattr(cls, "__module__", "")).startswith("diffusers.utils.dummy"):
        return ()
    backends = getattr(cls, "_backends", None) or ()
    return tuple(str(b) for b in backends)


def assert_pipeline_class_available(
    pipeline_class: str,
    family_name: str,
    *,
    strict: bool = False,
) -> None:
    """Raise ``ValueError`` before any download when the installed diffusers has no
    ``pipeline_class``.

    The newer families (Flux2Klein, Z-Image, Krea 2, LTX-2, HunyuanImage) only exist from a
    diffusers newer than the 0.35 baseline, and the packaging leaves an older one installable on
    Python 3.9 -- diffusers dropped 3.9 in 0.37 and this project still supports it, so the 0.39
    floor has to be conditional or the whole extra becomes unresolvable. Which release a family
    needs differs per class, so the refusal reads it from ``_PIPELINE_MIN_DIFFUSERS`` rather than
    quoting the floor at everyone. Without this check the getattr chain died with a bare
    AttributeError deep in the load, after the checkpoint had already been fetched, which is an
    expensive way to learn the environment is too old. Krea 2 already guarded itself this way; this
    is the same check for every family, run from validation.

    ``strict`` decides what an *unimportable* diffusers means. Inference (the default) stays
    silent: it only answers "is the installed diffusers new enough", and the native sd.cpp engine
    serves GGUF picks on a CPU or Apple host that has no diffusers at all. Training passes
    ``strict = True``, because its child is an ``mp.get_context("spawn")`` process in the SAME
    interpreter -- an import that fails here fails there too, only after the route has reserved the
    training slot and freed the resident GPU models.

    ``ValueError``, like every other unloadable-pick refusal ``validate_load_request`` raises, so
    the routes map it to 400 with the message intact. A ``RuntimeError`` instead reached
    ``/images/load``'s 409 (the code that otherwise means "a load is already in progress") and
    escaped ``/images/download-plan``, which catches only (ValueError, FileNotFoundError), as a bare
    500 with the message lost."""
    try:
        import diffusers
        present = hasattr(diffusers, pipeline_class)
        dummy_backends = _dummy_required_backends(getattr(diffusers, pipeline_class, None))
    except Exception as exc:  # noqa: BLE001 -- see below: this check must never raise anything but its own ValueError
        # Not this check's business under the default: it answers "is the installed diffusers new enough for this
        # family", and with nothing importable there is no version to judge. Refusing would also break the native
        # sd.cpp engine, which serves GGUF picks on a CPU or Apple host without diffusers. A pick that really needs
        # it fails later, in the loader. The one thing that must not happen is a raise of the wrong type:
        # ModuleNotFoundError is not the ValueError the routes map to 400, so it escapes /images/download-plan as a
        # bare 500 with the message lost.
        #
        # The attribute probe is inside the try for the same reason. diffusers' top level is a lazy module, so
        # ``hasattr`` is what actually imports the pipeline's submodule, and when that submodule's own dependencies
        # are unsatisfiable it raises RuntimeError ("Failed to import diffusers.pipelines...") -- which hasattr does
        # NOT swallow, since it only absorbs AttributeError. A partially usable diffusers install therefore escaped
        # this guard exactly the way a missing one used to.
        if strict:
            raise ValueError(
                f"'{family_name}' needs diffusers ({pipeline_class}), which this environment "
                f"cannot import: {exc}. Install or repair it with: pip install -U diffusers."
            ) from None
        return

    if present and dummy_backends:
        # A placeholder, not the pipeline. Under the default this is left alone like every other
        # unusable install; strict refuses, because the trainer child imports the same placeholder
        # and its from_pretrained raises only after the GPU residents are gone.
        if not strict:
            return
        raise ValueError(
            f"'{family_name}' needs diffusers ({pipeline_class}), but this diffusers exports it as "
            f"a placeholder, which it does when a backend it requires is unavailable. That class "
            f"requires: {', '.join(dummy_backends)}. Check which of those this environment is "
            f"missing and install it."
        )

    if present:
        return
    raise ValueError(
        _too_old_message(
            pipeline_class, family_name, str(getattr(diffusers, "__version__", "unknown"))
        )
    )


def _module_namespace_is_unreadable(module: Any) -> bool:
    """Return whether probing attributes could import code or read a partial module."""
    if hasattr(type(module), "__getattr__"):
        return True
    if callable(getattr(module, "__getattr__", None)):
        return True
    return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))


def _installed_diffusers_version() -> Optional[str]:
    """Read the installed diffusers version without importing it."""
    module = sys.modules.get("diffusers")
    if module is not None:
        try:
            installed = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001 -- a module that raises on __version__ just falls through
            installed = None
        if isinstance(installed, str) and installed.strip():
            return installed.strip()
    try:
        from importlib.metadata import version
        installed = version("diffusers")
    except Exception:  # noqa: BLE001 -- not installed / unreadable metadata: caller fails open
        return None
    return installed.strip() if isinstance(installed, str) and installed.strip() else None


def _installed_at_least(installed: str, minimum: str) -> bool:
    """Whether an INSTALLED version satisfies ``minimum``, judged on its release numbers.

    Not ``_version_tuple``, which is for the clean constants in the table above: a vendor build
    carries a PEP 440 local suffix (``0.40.0+dfsg``) that stops the numeric parse mid-version, and
    a git install carries ``.dev0``, which strict PEP 440 sorts below its own release. Both HAVE
    the class, so compare the release they were cut from. An unreadable version answers OPEN."""
    try:
        from packaging.version import Version
        return Version(Version(installed).base_version) >= Version(minimum)
    except Exception:  # noqa: BLE001 -- an unparseable version must not hide a model
        return True


def family_probe_class(fam: Any) -> str:
    """The class whose presence in the installed diffusers actually proves ``fam`` is loadable.

    Normally that is ``fam.pipeline_class``. ``ModularPipeline`` is the exception: it is the
    generic entry point for every Modular Diffusers workflow, not a family, and it has existed for
    several releases, so a diffusers that predates MiniMax-H3's own blocks still answers hasattr
    for it. Probe the family's own transformer class there instead, which is the thing the load
    actually needs. Shared by the listing probe and by both training gates so a family cannot be
    hidden from the picker and simultaneously accepted by /diffusion/start."""
    name = str(getattr(fam, "pipeline_class", "") or "")
    if name == "ModularPipeline":
        return str(getattr(fam, "transformer_class", None) or name)
    return name


def family_pipeline_available(fam: Optional[DiffusionFamily]) -> bool:
    """True when the installed diffusers actually has this family's pipeline class.

    The boolean twin of ``assert_pipeline_class_available``, for the listing routes: the newer
    families exist only from diffusers 0.39, and the packaging leaves an older diffusers
    installable on Python 3.9 (diffusers dropped 3.9 in 0.37, so the 0.39 floor has to be
    conditional or the extra becomes unresolvable). Advertising Z-Image or Krea 2 in the picker
    on such an environment offers a pick that can only fail, and no `pip install -U diffusers`
    can fix it without also upgrading Python. Fails OPEN (True) when diffusers cannot be
    imported at all, so a listing never hides a model over an unrelated import problem.

    Uses installed-version metadata because probing diffusers' lazy attributes imports pipeline
    dependencies. The load path remains the final availability check."""
    if fam is None:
        return False
    # A modular family is judged on its own transformer class, not on the generic
    # ``ModularPipeline`` entry point -- see ``family_probe_class``.
    name = family_probe_class(fam)
    # No class name to probe means the record is not one this helper can judge, which is the same
    # position as a missing diffusers: answer OPEN. ``family_probe_class`` reads the attribute with
    # a default, so a record without ``pipeline_class`` reaches here as "" rather than raising into
    # the guard below, and ``hasattr(diffusers, "")`` is False -- which would hide the model.
    if not name:
        return True
    if "diffusers" in sys.modules:
        module = sys.modules["diffusers"]
        # A None entry blocks imports, so preserve the existing fail-open behavior.
        if module is None:
            return True
        if not _module_namespace_is_unreadable(module):
            try:
                return hasattr(module, name)
            except Exception:  # noqa: BLE001 -- a probe failure must not hide a model
                return True
    minimum, _needs_py310 = pipeline_class_requirement(name)
    # Unlisted classes predate the version gates in this table.
    if minimum is None:
        return True
    installed = _installed_diffusers_version()
    if installed is None:
        return True
    return _installed_at_least(installed, minimum)


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
# Byte-identical mirror of Comfy-Org/vae-text-encorder-for-flux-klein-9b.
_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS = (
    (
        "unsloth/FLUX.2-klein-9B-ComfyUI",
        "split_files/text_encoders/qwen_3_8b.safetensors",
        "llm",
    ),
)


def sd_cpp_companion_only_repo_ids() -> frozenset[str]:
    """Lowercased ids of repos that exist ONLY to hand sd.cpp a single-file VAE / text encoder.

    They carry no DENOISER, so no diffusion pipeline loads from one. Third-party repacks never
    cleared the cached-model trust gate, but their ``unsloth/*`` mirrors do, so they need excluding
    by hand or each becomes an un-loadable On Device row. Derived from the tables, minus repos that
    are also a real base: FLUX.1-schnell ships the FLUX.1 VAE and IS loadable.

    Diffusion-only, and callers must keep it that way: the set includes
    ``unsloth/Qwen2.5-VL-7B-Instruct-GGUF``, which is a perfectly good CHAT model that sd.cpp also
    borrows as a text encoder. Hiding this set from a chat or GGUF listing would take a real model
    away from a user who downloaded it to chat with. Today's only consumer is the non-GGUF
    cached-model listing, which never sees a GGUF-only repo."""
    companions: set[str] = set()
    loadable: set[str] = set()
    for fam in _FAMILIES:
        if fam.sd_cpp_vae:
            companions.add(fam.sd_cpp_vae[0])
        companions.update(repo for repo, _f, _k in fam.sd_cpp_text_encoders)
        loadable.add(fam.base_repo)
        loadable.update(fam.train_base_repos)
        if fam.deploy_base_repo:
            loadable.add(fam.deploy_base_repo)
        loadable.update(repo for _scheme, repo in fam.prequant_repos)
        loadable.update(repo for _base, _scheme, repo in fam.prequant_variant_repos)
        loadable.update(repo for _scheme, _component, repo in fam.te_prequant_repos)
    companions.update(repo for repo, _f, _k in _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
    return frozenset(r.strip().lower() for r in companions - loadable if r)


def sd_cpp_text_encoder_candidates(fam: DiffusionFamily) -> tuple[tuple[str, str, str], ...]:
    """EVERY text-encoder set an sd.cpp load of *fam* could pick, unioned.

    For the guard, not for a load. A load reads the GGUF header and picks one; a guard
    reconstructing a checkpoint it cannot open has no header, and for FLUX.2-klein a renamed 9B
    file carries no size token either, so the string fallback answers 4B and the 9B encoder the
    load actually fetched is left unprotected. Naming both costs a delete that is refused and
    saves one that strands an installed model.
    """
    sets = [fam.sd_cpp_text_encoders]
    if fam.name == "flux.2-klein":
        sets.append(_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
    return tuple(dict.fromkeys(entry for group in sets for entry in group or ()))


def sd_cpp_text_encoders_for(
    fam: DiffusionFamily,
    repo_id: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    inner_dim: Optional[int] = None,
) -> tuple[tuple[str, str, str], ...]:
    """The sd.cpp text encoders for a specific load.

    FLUX.2-klein picks by variant (9B needs Qwen3-8B, 4B the family default); every other family
    returns its static table.

    ``inner_dim`` is the checkpoint's own answer, read from the GGUF header
    (``gguf_flux2_inner_dim``): it decides whenever the caller has it, because a renamed or
    hand-picked file makes the load identity say nothing. The repo id + filename string match is
    the fallback for the callers that have no header to read (the delete guard reconstructs a
    committed load; the plan runs before a byte is fetched)."""
    if fam.name == "flux.2-klein":
        # A dim this family does not have (a FLUX.2-dev file, a misread) falls through to the
        # strings rather than guessing, exactly as the base-size guard fails open on one.
        if inner_dim == _FLUX2_KLEIN_9B_INNER_DIM:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
        if inner_dim == _FLUX2_KLEIN_4B_INNER_DIM:
            return fam.sd_cpp_text_encoders
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
_FLUX2_INNER_DIMS = {
    3072: "FLUX.2-klein-4B / klein-base-4B",
    4096: "FLUX.2-klein-9B / klein-base-9B",
    6144: "FLUX.2-dev",
}
_FLUX2_KLEIN_4B_INNER_DIM = 3072
_FLUX2_KLEIN_9B_INNER_DIM = 4096
_FLUX2_BASE_INNER_DIM = {
    "black-forest-labs/flux.2-klein-4b": 3072,
    "black-forest-labs/flux.2-klein-base-4b": 3072,
    "black-forest-labs/flux.2-klein-9b": 4096,
    "black-forest-labs/flux.2-klein-base-9b": 4096,
    "black-forest-labs/flux.2-dev": 6144,
}


class _HeaderTensor(NamedTuple):
    """The two fields a FLUX.2 size probe reads off a GGUF tensor table entry."""

    name: str
    shape: Sequence[int]


def flux2_base_inner_dim(base_repo: Optional[str]) -> Optional[int]:
    """The ``inner_dim`` a FLUX.2 base config expects, or None when the repo is not one we map.

    Keyed on UPSTREAM ids, reached through ``canonical_base``: a known ungated mirror is
    byte-identical to what it copies, so it maps back and is checked exactly like its upstream.
    Anything else -- a local path, a third-party repack, a base we do not ship -- misses, and
    every caller fails OPEN on the None rather than guessing."""
    return _FLUX2_BASE_INNER_DIM.get(canonical_base(base_repo or "").lower())


def _flux2_inner_dim_from_tensors(tensors) -> Optional[int]:
    """``inner_dim`` from a parsed GGUF tensor table, or None when the probe tensor is absent."""
    for t in tensors:
        if t.name == _FLUX2_PROBE_TENSOR or t.name.endswith("." + _FLUX2_PROBE_TENSOR):
            # GGUF stores dims reversed relative to torch, so the input dim leads. A missing or
            # non-positive dim is a parse that went wrong, and "0" would be compared against the
            # base as a real answer and refuse a valid pick; say nothing instead.
            dim = int(t.shape[0]) if len(t.shape) else 0
            return dim if dim > 0 else None
    return None


def gguf_flux2_inner_dim(path) -> Optional[int]:
    """``inner_dim`` of a FLUX.2 GGUF, read from its header, or None if it cannot be determined."""
    try:
        from gguf import GGUFReader
        return _flux2_inner_dim_from_tensors(GGUFReader(str(path)).tensors)
    except Exception:
        return None


def gguf_flux2_inner_dim_from_header(header: bytes) -> Optional[int]:
    """``inner_dim`` read from the leading bytes of a FLUX.2 GGUF, or None.

    Lets the selection-time preflight range-read a few hundred KiB over HTTP instead of pulling
    the whole multi-GB checkpoint: the tensor table it needs sits in the first ~15 KiB. Same
    parser as ``gguf_flux2_inner_dim``, with the two things a PREFIX changes:

    * ``_build_tensors`` is skipped. The base class builds a numpy view over every tensor's DATA,
      which a prefix does not carry -- and satisfying those reads would allocate the whole
      declared checkpoint (tens of GiB), which is exactly the cost this avoids. Only the name and
      shape from the table are wanted, and both are already parsed.
    * ``_get`` REFUSES a read past the end instead of returning short or zero-filled data. The
      table is read field by field, so a prefix cutting between a tensor's name and its dims
      would otherwise hand back a zero shape -- a wrong answer, not a missing one, and one that
      would refuse a perfectly valid pick.

    None on anything unreadable: too short a prefix, a non-GGUF file, an absent probe tensor."""
    if not header:
        return None
    tmp_path = None
    reader = None
    try:
        import numpy as np
        from gguf import GGUFReader

        class _HeaderOnlyGGUFReader(GGUFReader):  # type: ignore[misc, valid-type]
            """``GGUFReader`` over a file that holds only the header."""

            def _get(
                self,
                offset,
                dtype,
                count = 1,
                override_order = None,
            ):
                itemsize = int(np.empty([], dtype = dtype).itemsize)
                if int(offset) < 0 or int(offset) + itemsize * int(count) > len(self.data):
                    raise ValueError("GGUF header is truncated")
                return super()._get(offset, dtype, count, override_order)

            def _build_tensors(self, start_offs, fields):
                # ``field.name`` is the decoded tensor name and parts[3] its dims, both already
                # read from real bytes by ``_get_tensor_info_field``.
                self.tensors = [_HeaderTensor(field.name, field.parts[3]) for field in fields]

        # GGUFReader memory-maps a path, so the prefix has to land on disk. Bounded by the
        # caller's range request, so this is KiB, never the checkpoint.
        fd, tmp_path = tempfile.mkstemp(suffix = ".gguf-header")
        with os.fdopen(fd, "wb") as fh:
            fh.write(header)
        reader = _HeaderOnlyGGUFReader(tmp_path)
        # Belt and braces on top of the refusing ``_get``: ``data_offset`` is where the table
        # ends, so a table that ran past the prefix is not one to read a shape out of. The
        # alignment slack is for a header-only file, whose padding was never written.
        if reader.data_offset > len(header) + min(int(reader.alignment), 4096):
            return None
        return _flux2_inner_dim_from_tensors(reader.tensors)
    except Exception:  # noqa: BLE001 — an unreadable header is not a verdict
        return None
    finally:
        # Drop the mmap before unlinking: Windows refuses to delete a mapped file.
        del reader
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def flux2_mismatch_reason(
    gguf_name: str, base_repo: str, got: Optional[int], want: Optional[int]
) -> Optional[str]:
    """Why this FLUX.2 GGUF cannot load against this base, or None when they agree.

    One message for all three checks on the pairing (the plan, the pre-eviction preflight and the
    loader's backstop), so the user reads the same sentence wherever it is caught. ``gguf_name`` is
    a display name the caller has already reduced to a basename. Returns None on any unknown -- an
    unreadable header, an unmapped base -- so every caller fails open."""
    if want is None or got is None or want == got:
        return None
    return (
        f"'{gguf_name}' is a "
        f"{_FLUX2_INNER_DIMS.get(got, f'FLUX.2 variant with inner_dim {got}')} "
        f"checkpoint, but it is being loaded against '{base_repo}', which is "
        f"{_FLUX2_INNER_DIMS.get(want, f'inner_dim {want}')}. Pass base_repo for the matching "
        f"variant, or pick the GGUF that matches the selected model."
    )


def assert_flux2_gguf_matches_base(fam, base_repo: str, gguf_path) -> None:
    """Fail early, and legibly, when a FLUX.2 GGUF is paired with a different-size base config.

    Without this the mismatch surfaces from inside the GGUF quantizer as a bare shape error
    ("expected torch.Size([18432, 3072]), decodes to (24576, 4096)") that names neither the file
    nor the repo. Fail-open by construction: any unreadable file, non-FLUX.2 tensor set, or
    unmapped base leaves the load exactly as it was.

    The LAST of three checks on the same pairing, and the only one that opens the downloaded file:
    ``diffusion_compat`` runs the same comparison off a range-read header at plan time and again
    before the resident pipeline is torn down, so a mismatch normally never reaches here."""
    if gguf_path is None or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return
    # Short-circuit on the free half: an unmapped base fails open anyway, so a mirror id must not
    # cost a memory-map of the whole checkpoint.
    want = flux2_base_inner_dim(base_repo)
    if want is None:
        return
    reason = flux2_mismatch_reason(
        Path(str(gguf_path)).name, base_repo, gguf_flux2_inner_dim(gguf_path), want
    )
    if reason is not None:
        raise ValueError(reason)


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

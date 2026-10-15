# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for diffusion model identification. No torch/diffusers imports: everything here is a
pure function of its string/path arguments, so it can be unit-tested without the heavy runtime. A
diffusion checkpoint published as a single-file GGUF only carries the transformer weights; the
matching VAE / text encoders / scheduler come from a companion ``diffusers`` base repo, and
``DiffusionFamily`` maps a checkpoint to the diffusers classes and base repo needed to assemble
the full pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, Optional, Sequence
from utils.paths.path_utils import is_appledouble_metadata

from .diffusion_flow_shift import flux_mu_shift
from .family_name_match import (
    name_key_in,
    normalize_family_name,
    token_in_name,
    token_length,
)
from .diffusion_nvfp4_flag import nvfp4_blocked
from .family_profile import AutoPrefer, ConvRotSpec, FamilyProfile


# The route matches these messages EXACTLY for a 409, so both engines raise them verbatim.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class LoadIdentity:
    """What a caller's derived request parameters depend on, as one comparable value.

    ``repo_id`` alone is not one: /images/load takes ``base_repo`` and ``family_override``
    independently of the path, so a local checkpoint reloads as a different model while the path
    stays put, and the images route derives steps/guidance from ``base_repo`` and its edit-only
    verdict from the family (#9448). Loads agreeing on all three derive identical parameters, which
    is exactly when accepting one for the other is correct.

    A type rather than a tuple, so pinning a bare repo id compares unequal and is refused instead of
    matching some other shape by accident.
    """

    repo_id: str
    base_repo: str
    family: str


def load_identity(repo_id, base_repo, family) -> LoadIdentity:
    """``LoadIdentity`` for one load. None and "" describe the same absent field."""
    return LoadIdentity(str(repo_id or ""), str(base_repo or ""), str(family or ""))


class DiffusionModelReplacedError(RuntimeError):
    """Both engines' ``generate`` refusing a ``LoadIdentity`` that is no longer loaded. Keeps a
    caller's per-model steps/guidance and workflow verdict, taken from an earlier ``status()``
    read, off a model they never validated (#9448). Here rather than in ``diffusion`` so the
    native engine can raise it without importing the torch backend."""

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
    cfg_kwarg: str = "guidance_scale"
    # False when the diffusers pipeline ignores a negative prompt; the native engine decides on its own.
    uses_negative_prompt: bool = True
    # The pipe attribute holding the denoiser: ``pipe.transformer`` for DiT families, ``pipe.unet`` for SDXL.
    denoiser_attr: str = "transformer"
    single_file_is_pipeline: bool = False
    pipeline_only: bool = False
    img2img_pipeline_class: Optional[str] = None
    inpaint_pipeline_class: Optional[str] = None
    controlnet_pipeline_class: Optional[str] = None
    controlnet_model_class: Optional[str] = None
    inpaint_preserves_size: bool = True
    edit: bool = False
    reference: bool = False
    unified_edit: bool = False
    layer_count: int = 0
    layer_resolution: int = 640
    max_condition_images: int = 4
    condition_image_mode: str = "RGB"
    dimension_multiple: int = 16
    max_output_side: int = 2048
    max_output_pixels: int = 2048 * 2048
    reference_resolutions: tuple[int, ...] = field(default_factory = tuple)
    comfy_flow_shift: Optional[float] = None
    # (lowercased id substring, shift or None = shipped) for checkpoints whose template differs; first match wins.
    comfy_flow_shift_variants: tuple[tuple[str, Optional[float]], ...] = field(
        default_factory = tuple
    )
    # (lowercased id substring, ((key, value), ...)) overriding ``base_repo``'s transformer config; first match wins.
    transformer_config_variants: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = field(
        default_factory = tuple
    )
    # Same config, different weights: a GGUF of one must never get the base's transformer of another.
    checkpoint_variants: tuple[str, ...] = field(default_factory = tuple)
    # Activation-guard cost of one condition pixel relative to one output pixel.
    condition_pixel_weight: float = 1.0
    aliases: tuple[str, ...] = field(default_factory = tuple)
    fp16_incompatible: bool = False
    fp16_guard: Optional[str] = None
    supports_torch_compile: bool = True
    cudnn_benchmark: bool = True
    # Pin inductor's reduction-config filter on every arch (``filter_reduction_configs``) or only on the listed (major,
    # minor) archs (``filter_reduction_configs_archs``); both unset = inductor's pick. Same two fields on VideoFamily.
    filter_reduction_configs: bool = False
    filter_reduction_configs_archs: tuple[tuple[int, int], ...] = field(default_factory = tuple)
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Lowercased bases with different weights: never inherit ``prequant_repos``; their variant rows still win.
    prequant_excluded_bases: tuple[str, ...] = field(default_factory = tuple)
    # Variant bases a repo id or GGUF name can select when no card ``base_model`` tag resolves one.
    named_variant_bases: tuple[str, ...] = field(default_factory = tuple)
    # Preferred filename rows; the derived name stays as fallback so older builds keep their
    # artifact.
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Opt-in per family: fp8 tolerance differs per encoder, so only after measuring the delta.
    te_quant_auto: Optional[str] = None
    sd_cpp_vae: Optional[tuple[str, str]] = None
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # mflux ModelConfig name (None = no MLX); ``_by_base`` maps a lowercase base repo to another config.
    mlx_model_config: Optional[str] = None
    mlx_model_config_by_base: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Family-specific sd-cli sampler settings so native output matches the model's supported invocation. None leaves
    # sd-cli defaults.
    sd_cpp_sampling_method: Optional[str] = None
    sd_cpp_flow_shift: Optional[float] = None
    # Literal an sd.cpp build carries once it can run this family; matched against binary bytes
    # since old runnable builds are never upgraded and release tags cannot answer it.
    sd_cpp_arch_marker: Optional[str] = None
    sd_cpp_edit_marker: Optional[str] = None
    trainable: bool = False
    train_base_repos: tuple[str, ...] = field(default_factory = tuple)
    deploy_base_repo: Optional[str] = None
    deploy_base_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Display label (Train UI); None = the family name.
    label: Optional[str] = None
    # Per-family policy data (quant excludes / ConvRot / denies, TE int8 schedule, preview, step skip, trusted
    # repos); see family_profile.FamilyProfile.
    profile: FamilyProfile = field(default_factory = FamilyProfile)

    def deploy_base_for(self, trained_base: str) -> str:
        """The inference checkpoint paired with ``trained_base``, or the input unchanged."""
        key = canonical_base(trained_base).lower()
        for training_repo, inference_repo in self.deploy_base_repos:
            if canonical_base(training_repo).lower() == key:
                return inference_repo
        return self.deploy_base_repo or trained_base


# Near-tied norm-reduction configs, benchmarked per process, made renders nondeterministic.
_REDUCTION_RACE_ARCHS: tuple[tuple[int, int], ...] = ((8, 0), (8, 9), (12, 0))

# --- Shared FamilyProfile data (see family_profile.FamilyProfile) ---

# int8 PER-FAMILY exclusions, on top of diffusion_transformer_quant._INT8_EXCLUDE_NAME_TOKENS. Qwen-Image MMDiT runs
# every TEXT-stream Linear at M = actual prompt tokens (unpadded, unlike FLUX's 512-token T5), so a short prompt falls
# under _int_mm's M floor of 16 and the denoise crashes. bf16 there costs ~nothing. Baked into the published int8
# artifacts of qwen-image / qwen-image-edit, so it moves only with a rebuild and republish.
_QWENIMAGE_INT8_EXCLUDES = (
    "txt_in",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "txt_mlp",
)
# Families whose activation ranges break specific schemes at the MODEL level (the smoke probe only proves the GEMM
# runs). Measured with the 28-pair prequant accuracy gate on B200: qwen-image + mxfp8 does semantic damage at 1024px,
# + nvfp4 is unusable (LPIPS 0.51). Neither has a known fix, so both stay denied and auto falls through to int8
# (excellent on Qwen). The deny also applies to an EXPLICIT request.
# fp8 was denied here too, for black frames. That cause is gone: it was torchao's fp8 scale chooser having no eps
# clamp, so qwen's all-zero text rows gave scale 0 and NaN, and ``_make_quant_config`` now floors it with
# ``activation_value_lb``. Re-measured on B200 with the floor, and BOTH paths the deny governs clear the 0.10 LPIPS
# bar: prequant checkpoint, full 28-pair gate -> 28/28 PASS (SSIM 0.87-0.99, LPIPS 0.027-0.228, CLIP delta <= 0.019);
# on-the-fly quantize_, gate's simple_object prompt at its own seeds -> LPIPS 0.048 / 0.033 / 0.027 / 0.063, against
# pre-floor plain fp8 at LPIPS 0.712 with SSIM 0.016 and mean luma 0. Keeping the deny cost real speed: it pinned qwen
# to int8 at 1.10x bf16 when compiled fp8 measures 1.21x.
_QWENIMAGE_SCHEME_DENY = frozenset({"mxfp8", "nvfp4"})
# Schemes denied for TRAINING on top of the inference deny. Training holds a stricter bar because the evidence above is
# rendering evidence: it says a frozen fp8 forward reconstructs the bf16 image, not that a LoRA converges when its
# frozen linears are fp8. Nobody has run that, so qwen fp8 stays out of the Train UI until someone does. Delete the
# entry once a training run is measured.
_QWENIMAGE_TRAIN_SCHEME_DENY = frozenset({"fp8"})
# Text-encoder int8 keep-bf16 schedule (skip_first, skip_last) from measured hidden-state fidelity: qwen-image
# (Qwen2.5-VL-7B) first+last 6 gives ~0.997 cosine; flux.2-dev (Mistral-Small-24B) first 3 gives ~0.98 (early-layer
# seeding). A family without one has no schedule clearing the bar, so int8 falls back to fp8.
_QWENIMAGE_TE_INT8_SKIP = (6, 6)
# Gated NVFP4 auto head (int8 first; nvfp4 only with a passing gate record for this base and device).
_GATED_AUTO_PREFER = AutoPrefer(
    floor = (10, 0), schemes = ("int8", "fp8", "nvfp4", "mxfp8"), gated = True
)
# AUTO keeps bf16 while it fits for these (measured with both sides compiled: INT8 was at most ~5% faster and failed
# the default-on LPIPS bar); when bf16 would offload, INT8 is still picked.
_LUMINA2_BF16_WHEN_RESIDENT = (
    "INT8 is no faster than compiled bf16 on a card that holds bf16 and changes image detail"
)
_HIDREAM_BF16_WHEN_RESIDENT = (
    "INT8 is barely faster than compiled bf16 on a card that holds bf16 and changes image detail"
)

# Keyed by architecture, not per variant: the base repo is read from the HF base_model tag at load time, so one entry
# covers Turbo/full, schnell/dev.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        label = "FLUX.1-dev",
        profile = FamilyProfile(
            auto_prefer = _GATED_AUTO_PREFER,
            preview_factors = "flux1",
            # Auto static step skip, measured vs the no-skip render (PR #12652); dev: every 2 is 0.29 LPIPS on a
            # dense-texture prompt, so max only.
            auto_static_skip = (
                ("black-forest-labs/flux.1-krea-dev", (("default", 2), ("max", 3), ("steps", 28))),
                ("black-forest-labs/flux.1-dev", (("max", 3), ("steps", 28))),
            ),
        ),
        checkpoint_variants = ("flux1-schnell", "flux1-krea-dev", "flux1-dev"),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        cudnn_benchmark = False,
        pipeline_class = "FluxPipeline",
        uses_negative_prompt = False,
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        # ComfyUI fixed mu 1.15 for dev / Krea; schnell first (a dev GGUF may resolve to its base). Keys name the model: paths match too.
        comfy_flow_shift_variants = tuple(
            (f"{prefix}-{model}", shift)
            for model, shift in (
                ("schnell", None),
                ("krea-dev", flux_mu_shift(1.15)),
                ("dev", flux_mu_shift(1.15)),
            )
            for prefix in ("flux.1", "flux1", "flux-1", "flux")
        ),
        prequant_repos = (
            ("int8", "unsloth/FLUX.1-schnell-FP8"),
            ("fp8", "unsloth/FLUX.1-schnell-FP8"),
        ),
        prequant_variant_repos = (
            ("black-forest-labs/flux.1-dev", "int8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-dev", "fp8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "int8", "unsloth/FLUX.1-Krea-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "fp8", "unsloth/FLUX.1-Krea-dev-FP8"),
            # schnell ONLY: dev / Krea-dev would download it just for _validate_checkpoint to refuse.
            ("black-forest-labs/flux.1-schnell", "nvfp4", "unsloth/FLUX.1-schnell-NVFP4"),
        ),
        te_prequant_repos = (("fp8", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8"),),
        aliases = ("flux1", "flux-1"),
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.1-dev",),
        img2img_pipeline_class = "FluxImg2ImgPipeline",
        inpaint_pipeline_class = "FluxInpaintPipeline",
        controlnet_pipeline_class = "FluxControlNetPipeline",
        controlnet_model_class = "FluxControlNetModel",
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = (
            ("unsloth/flux-text-encoders", "clip_l.safetensors", "clip_l"),
            ("unsloth/flux-text-encoders", "t5xxl_fp16.safetensors", "t5xxl"),
        ),
    ),
    # Must precede a generic flux match.
    DiffusionFamily(
        name = "flux.2-klein",
        label = "FLUX.2 Klein",
        profile = FamilyProfile(
            preview_factors = "flux2",
            auto_static_skip = (
                (
                    "black-forest-labs/flux.2-klein-base-4b",
                    (("default", 2), ("max", 3), ("steps", 50)),
                ),
            ),
        ),
        pipeline_class = "Flux2KleinPipeline",
        uses_negative_prompt = False,
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-klein-4B-FP8"),
            ("fp8", "unsloth/FLUX.2-klein-4B-FP8"),
        ),
        aliases = ("flux2-klein",),
        # A local ComfyUI file has no card tag, so without these a 9B pick got the 4B base and was refused.
        named_variant_bases = (
            "black-forest-labs/FLUX.2-klein-9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        # Train the undistilled bases, then preview their adapters on the matching 4-step models. Both vendor ids
        # resolve through the ungated mirrors at fetch time.
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
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        inpaint_preserves_size = False,
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
    DiffusionFamily(
        name = "flux.2-dev",
        label = "FLUX.2-dev",
        profile = FamilyProfile(te_int8_skip = (3, 0), preview_factors = "flux2"),
        pipeline_class = "Flux2Pipeline",
        uses_negative_prompt = False,
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-dev-FP8"),
            ("fp8", "unsloth/FLUX.2-dev-FP8"),
        ),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/FLUX.2-dev-FP8"),),
        aliases = ("flux2-dev", "flux2dev"),
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
        # Specific aliases first so detect_family prefers this over flux.1.
        name = "flux.1-kontext",
        profile = FamilyProfile(preview_factors = "flux1"),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        comfy_flow_shift = flux_mu_shift(
            1.15
        ),  # ComfyUI ModelSamplingFlux fixed mu 1.15 (Kontext template)
        pipeline_class = "FluxKontextPipeline",
        uses_negative_prompt = False,
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        aliases = ("flux.1-kontext-dev", "flux1-kontext", "flux-kontext", "kontext"),
        edit = True,
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = (
            ("unsloth/flux-text-encoders", "clip_l.safetensors", "clip_l"),
            ("unsloth/flux-text-encoders", "t5xxl_fp16.safetensors", "t5xxl"),
        ),
    ),
    DiffusionFamily(
        # Specific aliases first so detect_family prefers this over qwen-image.
        name = "qwen-image-edit",
        # Same DiT class + unpadded text stream as qwen-image.
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            train_scheme_deny = _QWENIMAGE_TRAIN_SCHEME_DENY,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
            preview_factors = "qwen_image",
        ),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        comfy_flow_shift = 3.1,  # ComfyUI ModelSamplingAuraFlow 3.1 (Qwen-Image-Edit 2511 template)
        comfy_flow_shift_variants = (
            ("qwen-image-edit-2511", 3.1),
            ("qwen-image-edit-2509", 3.0),
            ("qwen-image-edit", 3.0),
        ),
        # Only the 2511 config sets zero_cond_t; on 2509 / original Edit it renders oversaturated, off-identity images.
        transformer_config_variants = (
            ("qwen-image-edit-2511", ()),
            ("qwen-image-edit-2509", (("zero_cond_t", False),)),
            ("qwen-image-edit", (("zero_cond_t", False),)),
        ),
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
        fp16_incompatible = True,
        sd_cpp_vae = ("unsloth/Qwen-Image-ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
            ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision"),
        ),
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 3.0,
    ),
    DiffusionFamily(
        # Listed before qwen-image so the name outranks it.
        name = "qwen-image-layered",
        # Same DiT class + unpadded text stream as qwen-image.
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
        ),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        comfy_flow_shift = 1.0,
        layer_count = 2,
        layer_resolution = 640,
        pipeline_class = "QwenImageLayeredPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Layered",
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image_layered", "qwenimagelayered"),
        edit = True,
        max_condition_images = 1,
        condition_image_mode = "RGBA",
        fp16_incompatible = True,
        # Native: its own 4-channel VAE (the RGB one cannot decode layers; pixel-identical to the ComfyUI repack) and
        # qwen-image's encoder, no projector (sd.cpp: enable_vision = version != VERSION_QWEN_IMAGE_LAYERED).
        sd_cpp_vae = ("Qwen/Qwen-Image-Layered", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
        ),
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 1.0,
        # Layered support landed upstream in master-744 (556f04b); an older reused build has no such literal.
        sd_cpp_arch_marker = "qwen_image_layers",
    ),
    DiffusionFamily(
        name = "qwen-image",
        label = "Qwen-Image",
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            train_scheme_deny = _QWENIMAGE_TRAIN_SCHEME_DENY,
            auto_prefer = _GATED_AUTO_PREFER,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
            preview_factors = "qwen_image",
            auto_static_skip = (("qwen/qwen-image", (("default", 2), ("max", 3), ("steps", 20))),),
        ),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        comfy_flow_shift = 3.1,
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        prequant_repos = (
            ("int8", "unsloth/Qwen-Image-FP8"),
            ("fp8", "unsloth/Qwen-Image-FP8"),
        ),
        prequant_variant_repos = (
            ("qwen/qwen-image-2512", "int8", "unsloth/Qwen-Image-2512-FP8"),
            ("qwen/qwen-image-2512", "fp8", "unsloth/Qwen-Image-2512-FP8"),
            ("qwen/qwen-image-2512", "nvfp4", "unsloth/Qwen-Image-2512-NVFP4"),
        ),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image", "qwenimage"),
        fp16_incompatible = True,
        cudnn_benchmark = False,
        trainable = True,
        train_base_repos = ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "Qwen/Qwen-Image"),
        img2img_pipeline_class = "QwenImageImg2ImgPipeline",
        inpaint_pipeline_class = "QwenImageInpaintPipeline",
        controlnet_pipeline_class = "QwenImageControlNetPipeline",
        controlnet_model_class = "QwenImageControlNetModel",
        sd_cpp_vae = ("unsloth/Qwen-Image-ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
        ),
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 3.0,
    ),
    DiffusionFamily(
        # Different architecture from qwen-image, so its own family.
        name = "qwen-image-2.1",
        profile = FamilyProfile(
            # 2.1 is a 32-block SINGLE-stream DiT: no add_* projections, no txt_mlp, so the 20B MMDiT's exclusion list
            # does not apply and this one was measured rather than inherited. ``txt_in`` is here as a QUALITY lever,
            # not the small-M crash guard it is on qwen-image: all four policy arms rendered a deliberately 2-token
            # prompt without tripping ``_int_mm``'s M floor, so nothing crashes either way. Excluding it scored LPIPS
            # 0.0642 against 0.1032 with the generic tokens alone, paired over 16 calibration prompts, -0.0390 CI95
            # [-0.0708, -0.0072], better on 14/16.
            int8_exclude_tokens = ("txt_in",),
            int8_convrot = ConvRotSpec(
                group = 256,
                suffixes = (
                    "attn.to_q",
                    "attn.to_k",
                    "attn.to_v",
                    "attn.to_out.0",
                    "img_mlp.gate_layer",
                    "img_mlp.proj",
                    "img_mlp.out",
                ),
                filename = "Qwen-Image-2.1-INT8-ConvRot.safetensors",
                repo = "unsloth/Qwen-Image-2.1-FP8",
                variant_repos = ("unsloth/Qwen-Image-2.1-Turbo-FP8",),
                default_on = True,
            ),
            auto_prefer = _GATED_AUTO_PREFER,
            preview_factors = "qwen_image_21",
            auto_static_skip = (
                ("qwen/qwen-image-2.1", (("default", 3), ("max", 3), ("steps", 25))),
            ),
        ),
        # ComfyUI QwenImage21: ModelSamplingFlux fixed mu 0.69, no terminal stretch.
        comfy_flow_shift = flux_mu_shift(0.69),
        pipeline_class = "QwenImage21Pipeline",
        transformer_class = "QwenImage21Transformer2DModel",
        base_repo = "Qwen/Qwen-Image-2.1",
        prequant_repos = (
            ("int8", "unsloth/Qwen-Image-2.1-FP8"),
            ("fp8", "unsloth/Qwen-Image-2.1-FP8"),
            ("nvfp4", "unsloth/Qwen-Image-2.1-NVFP4"),
        ),
        # Turbo's own checkpoints (a different distill); derived names (2.1's plus -Turbo) resolve them.
        prequant_variant_repos = (
            ("qwen/qwen-image-2.1-turbo", "int8", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
            ("qwen/qwen-image-2.1-turbo", "fp8", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
        ),
        # 2.1's artifacts are baked from 2.1's denoiser: for nvfp4 Turbo quantizes its own weights.
        prequant_excluded_bases = ("qwen/qwen-image-2.1-turbo",),
        named_variant_bases = ("Qwen/Qwen-Image-2.1-Turbo",),
        # The artifacts are safetensors, not the historical torch.save pickle, so the family has to
        # NAME them: every derived fallback ends in .pt, and without these rows the loader would ask
        # the Hub for a file that is not there and silently fall back to the dense bf16 download.
        prequant_filenames = (
            ("fp8", "Qwen-Image-2.1-FP8.safetensors"),
            ("int8", "Qwen-Image-2.1-INT8.safetensors"),
        ),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-2.1-FP8"),),
        # int8 ConvRot as ComfyUI's template (fp8 misspells rendered text); encoder dominates VRAM.
        te_quant_auto = "int8",
        cfg_kwarg = "true_cfg_scale",
        # Unified: t2i plus optional condition images, so reference, not edit.
        reference = True,
        unified_edit = True,
        max_condition_images = 10,
        condition_image_mode = "RGBA",
        dimension_multiple = 32,
        max_output_side = 2752,
        max_output_pixels = 2400 * 1792,
        reference_resolutions = (512, 1024, 2048),
        condition_pixel_weight = 0.32,
        aliases = ("qwen_image_21", "qwenimage21", "qwen-image-21"),
        # 2.1 has its own VAE class. Kept in the FP8 repo, not the GGUF repo, so the GGUF repo
        # is not classified as companion-only.
        sd_cpp_vae = ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors"),
        # Supplied via --llm, not --qwen2vl (wrong vision preprocessing).
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                "llm",
            ),
            ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision"),
        ),
        sd_cpp_sampling_method = "euler",
        # No flow shift: sd.cpp picks a resolution-dependent schedule; a fixed shift overrides it.
        sd_cpp_arch_marker = "qwen_image_2_1",
        sd_cpp_edit_marker = "Qwen Image 2.1 editing requires Qwen3-VL vision weights",
    ),
    DiffusionFamily(
        name = "z-image",
        label = "Z-Image",
        profile = FamilyProfile(
            # exactly the Linears the hosted INT8 checkpoint quantizes (all input axes 256-divisible)
            int8_convrot = ConvRotSpec(
                group = 256,
                suffixes = (
                    "attention.to_q",
                    "attention.to_k",
                    "attention.to_v",
                    "attention.to_out.0",
                    "feed_forward.w1",
                    "feed_forward.w2",
                    "feed_forward.w3",
                    "cap_embedder.1",
                ),
                filename = "Z-Image-Turbo-INT8-ConvRot.safetensors",
                repo = "unsloth/Z-Image-Turbo-FP8",
                default_on = True,
            ),
            auto_prefer = _GATED_AUTO_PREFER,
            preview_factors = "zimage",
        ),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        cudnn_benchmark = False,
        comfy_flow_shift = 3.0,
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Z-Image-Turbo-FP8"),
            ("fp8", "unsloth/Z-Image-Turbo-FP8"),
            ("nvfp4", "unsloth/Z-Image-Turbo-NVFP4"),
        ),
        prequant_excluded_bases = ("tongyi-mai/z-image",),
        # NOT shared with flux.2-klein-4B: klein retrained this encoder's layer 35 MLP.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Z-Image-Turbo-FP8"),),
        aliases = ("zimage", "z_image"),
        trainable = True,
        train_base_repos = (
            "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
            "Tongyi-MAI/Z-Image-Turbo",
            "Tongyi-MAI/Z-Image",
        ),
        img2img_pipeline_class = "ZImageImg2ImgPipeline",
        inpaint_pipeline_class = "ZImageInpaintPipeline",
        fp16_incompatible = True,
        fp16_guard = "rescale_post_norm",
        mlx_model_config = "z-image-turbo",
        mlx_model_config_by_base = (("Tongyi-MAI/Z-Image", "z-image"),),
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
    DiffusionFamily(
        name = "krea-2",
        label = "Krea 2",
        pipeline_class = "Krea2Pipeline",
        transformer_class = "Krea2Transformer2DModel",
        base_repo = "krea/Krea-2-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Krea-2-Turbo-FP8"),
            ("fp8", "unsloth/Krea-2-Turbo-FP8"),
        ),
        prequant_excluded_bases = ("krea/krea-2-raw",),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Krea-2-Turbo-FP8"),),
        aliases = ("krea2",),
        trainable = True,
        train_base_repos = ("krea/Krea-2-Raw", "krea/Krea-2-Turbo"),
        deploy_base_repo = "krea/Krea-2-Turbo",
        fp16_incompatible = True,
    ),
    DiffusionFamily(
        name = "lumina-2",
        profile = FamilyProfile(auto_bf16_when_resident = _LUMINA2_BF16_WHEN_RESIDENT),
        pipeline_class = "Lumina2Pipeline",
        transformer_class = "Lumina2Transformer2DModel",
        base_repo = "Alpha-VLLM/Lumina-Image-2.0",
        prequant_repos = (
            ("int8", "unsloth/Lumina-Image-2.0-FP8"),
            ("fp8", "unsloth/Lumina-Image-2.0-FP8"),
        ),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Lumina-Image-2.0-FP8"),),
        # Not bare "lumina": Lumina-Next checkpoints are a different arch.
        aliases = ("lumina-image-2.0", "lumina-image-2", "lumina2"),
        fp16_incompatible = True,
    ),
    DiffusionFamily(
        name = "hunyuanimage-2.1",
        profile = FamilyProfile(
            auto_static_skip = (
                ("hunyuanvideo-community/hunyuanimage-2.1-diffusers", (("max", 2), ("steps", 50))),
            ),
        ),
        # hosted checkpoints, verified bit-identical to on-the-fly quantize (the guider pipeline is not run-to-run
        # deterministic, so same-seed LPIPS mixes trajectory divergence with harness noise)
        prequant_repos = (
            ("int8", "unsloth/HunyuanImage-2.1-FP8"),
            ("fp8", "unsloth/HunyuanImage-2.1-FP8"),
        ),
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        pipeline_class = "HunyuanImagePipeline",
        transformer_class = "HunyuanImageTransformer2DModel",
        base_repo = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        cfg_kwarg = "distilled_guidance_scale",
        aliases = ("hunyuanimage-2.1-diffusers", "hunyuanimage2.1"),
        fp16_incompatible = True,
        # Distilled MeanFlow files: two extra embedders, shift 4. The catch-all row marks the base as another variant.
        transformer_config_variants = (
            ("distilled", (("guidance_embeds", True), ("use_meanflow", True))),
            ("hunyuanimage", ()),
        ),
        comfy_flow_shift_variants = (("distilled", 4.0),),
    ),
    DiffusionFamily(
        name = "hidream-i1",
        profile = FamilyProfile(
            auto_bf16_when_resident = _HIDREAM_BF16_WHEN_RESIDENT,
            trusted_repos = ("hidream-ai/hidream-i1-dev", "hidream-ai/hidream-i1-fast"),
        ),
        # Hosted checkpoints: 28/28 per-case gate pairs per scheme (LPIPS 0.291 int8 / 0.278 fp8); int8 bit-identical
        # to on-the-fly quantize.
        prequant_repos = (
            ("int8", "unsloth/HiDream-I1-Full-FP8"),
            ("fp8", "unsloth/HiDream-I1-Full-FP8"),
        ),
        # Dev and Fast are distillations; base_model_id refuses Full's artifact too late.
        prequant_excluded_bases = ("hidream-ai/hidream-i1-dev", "hidream-ai/hidream-i1-fast"),
        # The generic TE pass covers only text_encoder.._3, so TE4 engages via hidream_te4_kwargs.
        te_prequant_repos = (("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),),
        pipeline_class = "HiDreamImagePipeline",
        transformer_class = "HiDreamImageTransformer2DModel",
        base_repo = "HiDream-ai/HiDream-I1-Full",
        aliases = ("hidream", "hidream-i1-full", "hidream-i1-dev", "hidream-i1-fast"),
        fp16_incompatible = True,
    ),
    DiffusionFamily(
        name = "ideogram-4",
        profile = FamilyProfile(
            trusted_repos = ("ideogram-ai/ideogram-4-nf4", "ideogram-ai/ideogram-4-nf4-diffusers"),
        ),
        pipeline_class = "Ideogram4Pipeline",
        uses_negative_prompt = False,
        transformer_class = "Ideogram4Transformer2DModel",
        base_repo = "ideogram-ai/ideogram-4-fp8",
        aliases = ("ideogram4", "ideogram-v4", "ideogram"),
        pipeline_only = True,
    ),
    DiffusionFamily(
        name = "sdxl",
        label = "SDXL",
        profile = FamilyProfile(preview_factors = "sdxl"),
        # ~1.8% slower warm, but compiled-UNet renders then match across servers (ComfyUI's default)
        cudnn_benchmark = False,
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
        trainable = True,
        train_base_repos = (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
        ),
    ),
)


def trainable_family_names() -> tuple[str, ...]:
    """Names of families Unsloth can train a LoRA on, in registry order."""
    return tuple(fam.name for fam in _FAMILIES if fam.trainable)


# CFG uses a guidance_scale/guidance_schedule pair; named here so the two modules cannot drift
IDEOGRAM4_FAMILY_NAME = "ideogram-4"

# generate carries the CFG-truncation ratio; named here so the two modules cannot drift
LUMINA2_FAMILY_NAME = "lumina-2"


_EXCLUDED_MODELS: tuple[tuple[str, str], ...] = (
    (
        # "-3" scoped so a future HunyuanImage 2.x with a diffusers pipeline falls through
        "hunyuanimage-3",
        "HunyuanImage-3.0 has no diffusers pipeline (it is an 80B autoregressive MoE "
        "that requires trust_remote_code), so Unsloth does not support it.",
    ),
)


def excluded_model_reason(repo_id: str) -> Optional[str]:
    """The stated reason ``repo_id`` is unsupported, or None when it is simply unknown."""
    needle = (repo_id or "").lower()
    for token, reason in _EXCLUDED_MODELS:
        if _token_in_needle(token, needle):
            return reason
    return None


_EDIT_KEYWORDS = ("edit", "kontext", "inpaint", "layered")


def _token_in_needle(token: str, needle: str) -> bool:
    """True when ``token`` appears in ``needle`` as a whole segment (delimited by ``- _ . / \\`` or
    a boundary), not a raw substring, so 'qwen-image-edit' matches '...-2511' but 'kontext'
    doesn't match 'kontextual'. Separator-insensitive."""
    return token_in_name(token, needle)


def _best_family_match(needle: str) -> Optional[DiffusionFamily]:
    """The family whose name/alias is the LONGEST whole-segment token of ``needle`` (longest = most
    specific, so '...qwen-image-edit-2511...' matches 'qwen-image-edit', not 'qwen-image')."""
    best: Optional[tuple[DiffusionFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or token_length(token) > best[1]):
                best = (fam, token_length(token))
    return best[0] if best else None


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Most-specific name/alias substring wins; unsupported editing/inpaint/layered variants return None."""
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        norm = normalize_family_name(key)
        for fam in _FAMILIES:
            if any(normalize_family_name(t) == norm for t in (fam.name, *fam.aliases)):
                return fam
        return None
    needle = repo_id.lower()
    match = _best_family_match(needle)
    if match is not None:
        # Scoped to the last path component so a parent folder named edit does not reject a file.
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
    """Matches only the base _class_name: a variant tagged here would load through the wrong pipeline."""
    key = (class_name or "").strip()
    if not key:
        return None
    for fam in _FAMILIES:
        if fam.pipeline_class and fam.pipeline_class == key:
            return fam
    return None


def pipeline_class_from_index(path: Optional[str]) -> Optional[str]:
    """Reads model_index.json as utf-8-sig, since PowerShell writes a BOM that plain utf-8 would fail on."""
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
    """Shared by listing and loader so the picker and validate_load_request answer from the same
    evidence."""
    fam = detect_family_by_pipeline_class(pipeline_class_from_index(path))
    if fam is None or _index_family_ruled_out(fam, path):
        return None
    return fam


def _index_family_ruled_out(fam: DiffusionFamily, path: Optional[str]) -> bool:
    """True when the directory's NAME carries a variant keyword the index's family cannot run."""
    basename = re.split(r"[/\\]+", str(path).lower())[-1]
    matched_tokens = (fam.name, *fam.aliases)
    return any(
        _token_in_needle(kw, basename) and not any(kw in tok for tok in matched_tokens)
        for kw in _EDIT_KEYWORDS
    )


def pipeline_index_contradicts_name(path: Optional[str]) -> bool:
    """Index declares a family its directory name rules out (e.g. -layered), so the name must not answer."""
    fam = detect_family_by_pipeline_class(pipeline_class_from_index(path))
    return fam is not None and _index_family_ruled_out(fam, path)


def detect_family_for_pick(
    repo_id: str,
    gguf_filename: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[DiffusionFamily]:
    """Falls back to the pipeline index for a moved local pipeline whose directory name is a commit hash."""
    fam = None
    if not override:
        # The checkpoint's own pipeline index outranks any guess from ancestor path segments.
        fam = detect_family_by_pipeline_index(repo_id)
        if fam is None and pipeline_index_contradicts_name(repo_id):
            return None
    if fam is None:
        fam = detect_family(repo_id, override)
    if fam is None and gguf_filename and not override:
        fam = detect_family(f"{repo_id}/{gguf_filename}", override)
    if not override:
        fam = _family_from_content(fam, repo_id, gguf_filename)
    return fam


def _family_from_content(
    fam: Optional[DiffusionFamily], repo_id: str, gguf_filename: Optional[str]
) -> Optional[DiffusionFamily]:
    """Reconcile the name verdict with a LOCAL file's header: non-DiT / video DiT -> None, renamed DiT
    -> its header family; the name still picks same-architecture variants. Remote picks untouched."""
    from .diffusion_content import local_pick_file, resolve_family_with_content

    path = local_pick_file(repo_id, gguf_filename)
    if not path:
        return fam
    needles = [repo_id] + ([f"{repo_id}/{gguf_filename}"] if gguf_filename else [])
    vetoed = fam is None and any(_best_family_match(n.lower()) is not None for n in needles)
    name, _ = resolve_family_with_content(fam.name if fam else None, path, "image", vetoed)
    if name is None:
        return None
    if fam is not None and fam.name == name:
        return fam
    return detect_family("", override = name)


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


# Swapped at fetch sites only, never in resolve_base_repo. A mirror must be a complete copy.
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

# Only a gated upstream justifies overriding an existing cache, so ungated pairs are separate.
_UNGATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512"),
    ("Qwen/Qwen-Image-2.1", "unsloth/Qwen-Image-2.1"),
    ("Qwen/Qwen-Image", "unsloth/Qwen-Image"),
    ("Qwen/Qwen-Image-Edit-2511", "unsloth/Qwen-Image-Edit-2511"),
    ("black-forest-labs/FLUX.2-klein-4B", "unsloth/FLUX.2-klein-4B"),
    # Lookup is by exact id, so every variant needs its own row.
    ("black-forest-labs/FLUX.2-klein-base-4B", "unsloth/FLUX.2-klein-base-4B"),
    ("Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo"),
    ("Alpha-VLLM/Lumina-Image-2.0", "unsloth/Lumina-Image-2.0"),
    ("HiDream-ai/HiDream-I1-Full", "unsloth/HiDream-I1-Full"),
    ("HiDream-ai/HiDream-I1-Dev", "unsloth/HiDream-I1-Dev"),
    ("HiDream-ai/HiDream-I1-Fast", "unsloth/HiDream-I1-Fast"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "unsloth/stable-diffusion-xl-base-1.0"),
    ("stabilityai/sdxl-turbo", "unsloth/sdxl-turbo"),
)
_MIRROR_PAIRS: tuple[tuple[str, str], ...] = _GATED_MIRROR_PAIRS + _UNGATED_MIRROR_PAIRS
_GATED_MIRRORS: dict[str, str] = {u.lower(): m for u, m in _MIRROR_PAIRS}
_MIRROR_UPSTREAM: dict[str, str] = {m.lower(): u for u, m in _MIRROR_PAIRS}
_GATED_UPSTREAMS: frozenset[str] = frozenset(u.lower() for u, _m in _GATED_MIRROR_PAIRS)


def mirror_repo(repo_id: Optional[str]) -> Optional[str]:
    """The unsloth mirror of ``repo_id``, or None when it is not a mirrored vendor base."""
    return _GATED_MIRRORS.get((repo_id or "").strip().lower())


def upstream_is_gated(repo_id: Optional[str]) -> bool:
    """True when ``repo_id`` is a vendor base the Hub refuses without accepted terms. Distinct from
    "has a mirror": most of the mirror table is ungated and exists only to keep the fetch inside
    ``unsloth/*``. Only the gated half justifies overriding a user's cache."""
    return (repo_id or "").strip().lower() in _GATED_UPSTREAMS


def named_variant_base(fam: "DiffusionFamily", *names: Optional[str]) -> Optional[str]:
    """The longest ``fam.named_variant_bases`` entry ``names`` spell (separators folded), or None. The last, most
    specific name decides first; one spelling the plain base stops the search (a 2.1 file in a Turbo repo stays 2.1)."""

    def fold(text: Optional[str]) -> str:
        return "".join(c for c in (text or "").lower() if c.isalnum())

    variants = [
        (b, fold(b.rsplit("/", 1)[-1])) for b in getattr(fam, "named_variant_bases", ()) or ()
    ]
    plain = fold((getattr(fam, "base_repo", "") or "").rsplit("/", 1)[-1])
    for name in reversed([n for n in names if n]):
        identity = fold(name)
        hits = [b for b, key in variants if key and key in identity]
        if hits:
            return max(hits, key = len)
        if plain and plain in identity:
            return None
    return None


def canonical_base(repo_id: Optional[str]) -> str:
    """A mirror id mapped back to the upstream it copies, else ``repo_id`` unchanged. Base-keyed
    tables hold UPSTREAM ids, so every lookup normalises here: a mirror reaching
    ``_FLUX2_BASE_INNER_DIM`` misses, and that shape guard fails OPEN, so it goes silent."""
    base = (repo_id or "").strip()
    return _MIRROR_UPSTREAM.get(base.lower(), base)


_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".ckpt", ".gguf"})


def _cached_revisions(root: Path, repo_id: str) -> list[Path]:
    """The snapshot dirs a fetch pinned to ``root`` could resolve for ``repo_id``. Never raises.
    ``refs/main`` alone when it exists, the one commit a branch fetch falls back to on a failed
    HEAD; every snapshot when it does not, since a commit-pinned download leaves no ref."""
    try:
        repo = root / f"models--{repo_id.replace('/', '--')}"
        ref = repo / "refs" / "main"
        revs = (
            [repo / "snapshots" / ref.read_text(encoding = "utf-8").strip()]
            if ref.is_file()
            else sorted((repo / "snapshots").iterdir())
        )
        return [rev for rev in revs if rev.is_dir()]
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return []


def _root_revision_coverage(root: Path, repo_id: str, wanted: Sequence[str]) -> set[frozenset[str]]:
    """One entry per revision a root could resolve; a name is never borrowed from a superseded snapshot."""
    covers: set[frozenset[str]] = {frozenset()}
    for rev in _cached_revisions(root, repo_id):
        covers.add(frozenset(name for name in wanted if (rev / name).exists()))
    return covers


def _root_holds_upstream(root: Path, repo_id: str, wanted: Sequence[str]) -> bool:
    """``_upstream_is_cached`` for ONE cache root. Never raises."""
    try:
        for rev in _cached_revisions(root, repo_id):
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
    """Needs one revision holding all files, else a stray config would pin the load to the gated
    upstream."""
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        roots = [Path(active_hf_hub_cache())]
        if other_root:
            from huggingface_hub import constants
            fallback = Path(constants.HF_HUB_CACHE)
            if fallback != roots[0]:
                roots.append(fallback)
        wanted = tuple(files or ())
        if wanted and len(roots) > 1:
            reachable = {frozenset()}
            for root in roots:
                covers = _root_revision_coverage(root, repo_id, wanted)
                reachable = {have | cover for have in reachable for cover in covers}
            return any(set(wanted) <= have for have in reachable)
        return any(_root_holds_upstream(root, repo_id, wanted) for root in roots)
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return False


def cache_holds_files(repo_id: str, files: Sequence[str]) -> bool:
    """Checks the live root only: from_pretrained is pinned there and cannot see the other root."""
    return bool(files) and _upstream_is_cached(repo_id, tuple(files))


# HF cache is keyed by repo id: map back to the old repack so upgrades don't re-download.
_SD_CPP_LEGACY_SOURCES: dict[str, str] = {
    "unsloth/flux-text-encoders": "comfyanonymous/flux_text_encoders",
    "unsloth/qwen-image-comfyui": "Comfy-Org/Qwen-Image_ComfyUI",
    "unsloth/flux.2-vae": "Comfy-Org/flux2-dev",
    "unsloth/flux.2-dev-comfyui": "Comfy-Org/flux2-dev",
    "unsloth/z-image-turbo-comfyui": "Comfy-Org/z_image_turbo",
    "unsloth/flux.2-klein-9b-comfyui": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
    "unsloth/wan2.2-ti2v-5b-gguf": "QuantStack/Wan2.2-TI2V-5B-GGUF",
    "unsloth/minimax-h3-gguf": "Comfy-Org/MiniMax-H3",
    "unsloth/minimax-h3-fp8": "Comfy-Org/MiniMax-H3",
}


def legacy_source_repo(repo_id: Optional[str]) -> Optional[str]:
    """The community repack ``repo_id`` mirrors, or None when it is not one of the mirrors."""
    return _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())


def prefer_cached_legacy_source(repo_id: str, files: Optional[Sequence[str]] = None) -> str:
    """Keeps an old repack whose bytes are cached in either root, since the mirror id cannot reach them."""
    legacy = _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())
    if not legacy:
        return repo_id
    return legacy if _upstream_is_cached(legacy, files, other_root = True) else repo_id


def _is_local_path(base: str) -> bool:
    """Local directory check: a relative dir named like a vendor id must be treated as a local path."""
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
    """Fetch-only swap to the identical ungated mirror; the upstream id stays what is shown and saved."""
    del hf_token  # noqa: F841 -- signature stability only
    if os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip():
        return base if _is_local_path(base) else canonical_base(base)
    mirror = mirror_repo(base)
    if not mirror:
        return base
    if _is_local_path(base):
        return base
    return base if _upstream_is_cached(base, files) else mirror


# Matched by substring, most specific first; keep in sync with UI MODEL_DEFAULTS.
_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("z-image-turbo", 8, 0.0),
    # Must precede the generic "krea" key.
    ("flux.1-krea", 20, 3.5),
    # Must precede the generic "krea" key.
    ("krea-2-raw", 52, 3.5),
    # Krea 2 Turbo (distilled): 8 steps, no CFG. "krea" then covers Turbo and other krea ids but Raw.
    ("flux1-krea", 20, 3.5),  # before Krea-2's generic row
    ("krea", 8, 0.0),
    ("flux.1-schnell", 4, 0.0),
    ("flux1-schnell", 4, 0.0),
    ("kontext", 20, 2.5),  # editing: before the generic flux.1
    ("flux.1", 20, 3.5),
    ("flux1", 20, 3.5),
    # Undistilled base runs real CFG; keep before the generic distilled key.
    ("flux.2-klein-base", 20, 5.0),
    ("flux.2-klein", 4, 1.0),
    ("flux.2-dev", 20, 4.0),  # full (non-distilled)
    # Before the generic qwen-image key, Turbo before 2.1; qwenimage21 has no separator to fold, hence its own row.
    ("qwen-image-2.1-turbo", 8, 1.0),
    ("qwen-image-21-turbo", 8, 1.0),
    ("qwenimage21-turbo", 8, 1.0),
    ("qwenimage21turbo", 8, 1.0),
    ("qwen-image-2.1", 25, 1.0),
    ("qwen-image-21", 25, 1.0),
    ("qwen_image_21", 25, 1.0),
    ("qwenimage21", 25, 1.0),
    ("qwen-image-layered", 20, 2.5),
    ("qwen_image_layered", 20, 2.5),
    ("qwenimagelayered", 20, 2.5),
    ("qwen-image-edit-2509", 20, 4.0),
    ("qwen-image-edit", 40, 4.0),
    ("qwen-image-2512", 50, 4.0),
    ("qwen-image", 20, 4.0),
    # diffusers Z-Image g = ComfyUI cfg - 1.
    ("z-image", 25, 3.0),
    ("lumina", 50, 4.0),
    ("hunyuanimage", 50, 3.25),
    ("hidream-i1-dev", 28, 0.0),
    ("hidream-i1-fast", 16, 0.0),
    ("hidream", 50, 5.0),
    ("ideogram", 20, 7.0),
    ("sdxl-turbo", 3, 0.0),
    ("stable-diffusion-xl", 25, 7.0),
    ("sdxl", 25, 7.0),
)
_GENERATION_DEFAULT_FALLBACK = (9, 0.0)


def _first_variant(rows: Any, identifiers: tuple[Optional[str], ...]) -> Optional[tuple]:
    """First ``(key, ...)`` row whose key is in an identifier; ``_`` reads as ``-`` (ComfyUI file names)."""
    for identifier in identifiers:
        needle = (identifier or "").lower().replace("_", "-")
        for row in rows or ():
            if row[0] in needle:
                return row
    return None


def comfy_flow_shift_for(fam: Any, *identifiers: Optional[str]) -> Optional[float]:
    row = _first_variant(getattr(fam, "comfy_flow_shift_variants", ()), identifiers)
    return row[1] if row else getattr(fam, "comfy_flow_shift", None)


def transformer_config_overrides_for(fam: Any, *identifiers: Optional[str]) -> dict[str, Any]:
    row = _first_variant(getattr(fam, "transformer_config_variants", ()), identifiers)
    return dict(row[1]) if row else {}


def _first_checkpoint_variant(
    keys: tuple[str, ...], identifiers: tuple[Optional[str], ...]
) -> Optional[str]:
    # flux.1-dev / flux1_dev / FLUX-1-dev all read flux1-dev
    for identifier in identifiers:
        needle = re.sub(r"(?<=[a-z])-(?=\d)", "", normalize_family_name(identifier or ""))
        for key in keys:
            if key in needle:
                return key
    return None


def transformer_variant_differs_from_base(
    fam: Any, base: Optional[str], *identifiers: Optional[str]
) -> bool:
    """Checkpoint and ``base`` name different variants; a base naming none (a local dir) is unknown."""
    keys = getattr(fam, "checkpoint_variants", ())
    if keys:
        base_key = _first_checkpoint_variant(keys, (base,))
        return base_key is not None and _first_checkpoint_variant(keys, identifiers) not in (
            None,
            base_key,
        )
    rows = getattr(fam, "transformer_config_variants", ())
    base_row = _first_variant(rows, (base,))
    return base_row is not None and _first_variant(rows, identifiers) not in (None, base_row)


def named_generation_params(*identifiers: Optional[str]) -> Optional[tuple[int, float]]:
    """``(steps, guidance)`` of the first identifier naming a known model (a local path resolves via its base), or None."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, steps, guidance in _GENERATION_DEFAULTS:
            if name_key_in(key, needle):
                return steps, guidance
    return None


def default_generation_params(*identifiers: Optional[str]) -> tuple[int, float]:
    """:func:`named_generation_params`, else the generic fallback."""
    return named_generation_params(*identifiers) or _GENERATION_DEFAULT_FALLBACK


def generation_params_with_grid(
    grid: Optional[tuple[float, ...]], *identifiers: Optional[str]
) -> tuple[int, float]:
    """:func:`default_generation_params`, but an unnamed load with a shipped grid runs the grid's own step count."""
    named = named_generation_params(*identifiers)
    if named is not None:
        return named
    if grid:
        return len(grid), _GENERATION_DEFAULT_FALLBACK[1]
    return _GENERATION_DEFAULT_FALLBACK


def family_prequant_repo(
    fam: DiffusionFamily,
    scheme: str,
    base_repo: Optional[str] = None,
) -> Optional[str]:
    """A variant checkpoint wins for its base; an excluded base gets None, since its weights differ."""
    if nvfp4_blocked(scheme):
        return None
    base = canonical_base(base_repo).lower()
    named = named_variant_base(fam, base_repo) if base else None
    if named and named.lower() != base:
        # A local copy naming a variant: its row if the loader's tail compare accepts it, else none (never 2.1's).
        if (
            base.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            != named.lower().rsplit("/", 1)[-1]
        ):
            return None
        base = named.lower()
    if base:
        # getattr, since video families have no such field; a plain read would silently drop their
        # prequant.
        for entry_base, entry_scheme, repo_id in fam.prequant_variant_repos:
            if entry_base == base and entry_scheme == scheme:
                return repo_id
        if base in (getattr(fam, "prequant_excluded_bases", ()) or ()):
            return None
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def family_prequant_filename(
    fam: DiffusionFamily,
    scheme: str,
    task: Optional[str] = None,
) -> Optional[str]:
    """Task rows win over agnostic ones, as sibling partitions are indistinguishable by any later check."""
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


# diffusers 0.37.0 requires Python >= 3.10
_DIFFUSERS_DROPPED_PY39 = "0.37.0"
_MAX_PIPELINE_MANIFEST_BYTES = 1 << 20

# First diffusers release exporting each class; every class the listing probes belongs here.
_PIPELINE_MIN_DIFFUSERS: dict[str, str] = {
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
    "QwenImageLayeredPipeline": "0.37.0",
    "Flux2KleinInpaintPipeline": "0.38.0",
    "Ideogram4Pipeline": "0.39.0",
    # _version_tuple stops at non-numeric parts, so 0.41.0.dev0 satisfies this.
    "QwenImage21Pipeline": "0.41.0",
    "Krea2Pipeline": "0.39.0",
    "QwenImagePipeline": "0.35.0",
    "QwenImageImg2ImgPipeline": "0.35.0",
    "QwenImageInpaintPipeline": "0.35.0",
    "FluxKontextPipeline": "0.35.0",
    "HiDreamImagePipeline": "0.34.0",
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
    """Unlisted classes return None: every release in play has them, so no Python upgrade is implied."""
    minimum = _PIPELINE_MIN_DIFFUSERS.get(pipeline_class)
    if minimum is None:
        return None, False
    return minimum, _version_tuple(minimum) >= _version_tuple(_DIFFUSERS_DROPPED_PY39)


def _json_dict(path: Path, max_bytes: int = _MAX_PIPELINE_MANIFEST_BYTES) -> Optional[dict]:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        # PowerShell writes JSON with a UTF-8 BOM.
        payload = json.loads(path.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


_CALLER_SUPPLIED_COMPONENTS = {"HiDreamImagePipeline": frozenset({"text_encoder_4", "tokenizer_4"})}
# The load uses variant=None, which cannot open *.fp16.safetensors.
_LOCAL_PIPELINE_WEIGHT_FORMATS = (
    (("diffusion_pytorch_model", "model"), "safetensors"),
    (("diffusion_pytorch_model", "pytorch_model"), "bin"),
)
_MAX_PIPELINE_WEIGHT_INDEX_BYTES = 64 * 1024 * 1024
_LOCAL_PIPELINE_METADATA_CONFIGS = (
    (("tokenizer",), ("tokenizer_config.json",)),
    (("scheduler",), ("scheduler_config.json",)),
    (("guider", "guidance"), ("guider_config.json",)),
    (("featureextractor", "imageprocessor"), ("preprocessor_config.json",)),
    (("processor",), ("processor_config.json", "preprocessor_config.json")),
)
_SELF_CONTAINED_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "vocab.txt",
    "spiece.model",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _safe_relative_parts(text: str) -> Optional[tuple[str, ...]]:
    # Reject other separators and drive prefixes, or ..\\ / C: escape on Windows.
    relative = PurePosixPath(text)
    if "\\" in text or ":" in text or relative.is_absolute() or ".." in relative.parts:
        return None
    return relative.parts


def _local_weights_are_complete(component: Path, library_name: str) -> bool:
    # The first format with any weights present decides: a leftover .bin index cannot veto safetensors.
    for stems, ext in _LOCAL_PIPELINE_WEIGHT_FORMATS:
        if library_name == "transformers":
            stems = tuple(s for s in stems if s != "diffusion_pytorch_model")
            if any(_nonempty_file(component / f"{s}.{ext}") for s in stems):
                return True
        index = next(
            (p for s in stems if (p := component / f"{s}.{ext}.index.json").exists()), None
        )
        if index is not None:
            weight_map = (_json_dict(index, _MAX_PIPELINE_WEIGHT_INDEX_BYTES) or {}).get(
                "weight_map"
            )
            shards = (
                {str(v) for v in weight_map.values() if v} if isinstance(weight_map, dict) else ()
            )
            parts = [_safe_relative_parts(shard) for shard in shards]
            return bool(parts) and all(
                p is not None and _nonempty_file(component.joinpath(*p)) for p in parts
            )
        sizes = [w.stat().st_size for s in stems if (w := component / f"{s}.{ext}").is_file()]
        if sizes:
            return any(sizes)
    return False


def _local_pipeline_component_is_complete(
    component: Path, library_name: str, class_name: str, config_only_model_components: bool
) -> bool:
    if not component.is_dir():
        return False
    if library_name not in {"diffusers", "transformers"}:
        return any(_nonempty_file(child) for child in component.iterdir())
    identity = class_name.replace("_", "").lower()
    for tokens, config_names in _LOCAL_PIPELINE_METADATA_CONFIGS:
        if not any(token in identity for token in tokens):
            continue
        if not any(
            _json_dict(component / name, _MAX_PIPELINE_WEIGHT_INDEX_BYTES) is not None
            for name in config_names
        ):
            return False
        if tokens[0] not in ("tokenizer", "processor") or "byt5tokenizer" in identity:
            return True
        return any(_nonempty_file(component / a) for a in _SELF_CONTAINED_TOKENIZER_ASSETS) or all(
            _nonempty_file(component / a) for a in ("vocab.json", "merges.txt")
        )
    return _json_dict(component / "config.json") is not None and (
        config_only_model_components or _local_weights_are_complete(component, library_name)
    )


def local_pipeline_components_are_complete(
    root: Path | str,
    filename: str,
    *,
    excluded_components: Sequence[str] = (),
    config_only_model_components: bool = False,
) -> bool:
    """Check local component presence and known Diffusers/Transformers serialization layouts."""
    if filename not in {"model_index.json", "modular_model_index.json"}:
        return False
    base = Path(root).expanduser()
    payload = _json_dict(base / filename) or {}
    class_name = payload.get("_class_name")
    if not isinstance(class_name, str) or not class_name.strip():
        return False
    caller_supplied = _CALLER_SUPPLIED_COMPONENTS.get(class_name, frozenset())
    declared = False
    try:
        for name, spec in payload.items():
            if (
                name.startswith("_")
                or not isinstance(spec, (list, tuple))
                or len(spec) < 2
                or not (isinstance(spec[0], str) and isinstance(spec[1], str))
            ):
                continue
            if name in {"", ".."} or "\\" in name or Path(name).name != name:
                return False
            if name in excluded_components or (
                name in caller_supplied and not (base / name).exists()
            ):
                continue
            declared = True
            component = base / name
            source = spec[2] if filename == "modular_model_index.json" and len(spec) >= 3 else None
            source = source if isinstance(source, dict) else {}
            repo = source.get("pretrained_model_name_or_path") or source.get("repo")
            if isinstance(repo, str) and repo.strip():
                repo = repo.strip()
                subfolder = source.get("subfolder")
                parts = (
                    _safe_relative_parts((subfolder or "").strip())
                    if subfolder is None or isinstance(subfolder, str)
                    else None
                )
                if parts is None:
                    return False
                rooted = Path(repo).expanduser()
                rooted = rooted if rooted.is_absolute() else base / rooted
                if not rooted.exists():
                    if repo.startswith(("/", "\\", "~", ".")) or "\\" in repo or ":" in repo:
                        return False
                    continue
                component = rooted.joinpath(*parts)
            if not _local_pipeline_component_is_complete(
                component, spec[0], spec[1], config_only_model_components
            ):
                return False
    except OSError:
        return False
    return declared


# Entries name releases not yet published; the remedy is diffusers-main.txt, not a pip upgrade.
_UNRELEASED_MIN_DIFFUSERS: frozenset = frozenset()


_DIFFUSERS_MAIN_PIN = Path(__file__).resolve().parents[2] / "requirements" / "diffusers-main.txt"
_DIFFUSERS_MAIN_COMMIT_RE = re.compile(
    r"github\.com/(?P<repo>[^/\s]+/[^/@\s]+?)(?:\.git)?@(?P<commit>[0-9a-fA-F]{40})\b"
)


DIFFUSERS_UPDATE_REMEDY = (
    "Update Unsloth to install it (in the desktop app: Settings, Check for updates; from a "
    "terminal: unsloth studio update), then restart Unsloth."
)
DIFFUSERS_MAIN_RESTART_REMEDY = (
    "Restart Unsloth: on start it installs this pinned build by itself when it can reach "
    "github.com. If that still fails, update Unsloth (in the desktop app: Settings, Check for "
    "updates)."
)


def _diffusers_main_archive_remedy() -> str:
    """Names the pinned commit and a zip install, since the git+https pin fails on hosts without git."""
    generic = (
        f"{DIFFUSERS_MAIN_RESTART_REMEDY} On a pip or server install, re-run the Unsloth installer "
        "(leaving UNSLOTH_DIFFUSERS_MAIN unset), which installs it from a zip archive when git is "
        "missing."
    )
    try:
        text = _DIFFUSERS_MAIN_PIN.read_text(encoding = "utf-8-sig")
    except (OSError, ValueError):
        return generic
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        found = _DIFFUSERS_MAIN_COMMIT_RE.search(stripped)
        if found is None:
            continue
        url = (
            f"https://github.com/{found.group('repo')}/archive/"
            f"{found.group('commit').lower()}.zip"
        )
        return (
            f"{generic} To install it by hand from a terminal with no git at all: "
            f'pip install "diffusers @ {url}"'
        )
    return generic


def _too_old_message(pipeline_class: str, family_name: str, installed: str) -> str:
    """The refusal text: what is missing, what is installed, and a remedy this interpreter can
    actually carry out."""
    minimum, needs_py310 = pipeline_class_requirement(pipeline_class)
    if minimum is None:
        return (
            f"'{family_name}' needs a newer diffusers ({pipeline_class}); this environment has "
            f"diffusers {installed}. {DIFFUSERS_UPDATE_REMEDY} On a plain pip install: "
            "pip install -U diffusers."
        )
    if minimum in _UNRELEASED_MIN_DIFFUSERS:
        remedy = _diffusers_main_archive_remedy()
        return (
            f"'{family_name}' needs diffusers >= {minimum} ({pipeline_class}), which has not been "
            f"released yet; this environment has diffusers {installed}. Unsloth installs a pinned "
            "build of diffusers main for this and that build is not here, which almost always "
            "means the install had no working git (check with: git --version) or could not reach "
            f"github.com. {remedy}"
        )
    remedy = (
        f"{DIFFUSERS_UPDATE_REMEDY} On a plain pip install: pip install -U 'diffusers>={minimum}'."
    )
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
    """Backends a diffusers placeholder requires; hasattr wrongly answers True for these classes."""
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
    """Raises ValueError before any download if diffusers lacks the class; routes map it to 400."""
    # Request threads may race the background torch warm; importing diffusers pulls torch._dynamo.
    try:
        from loggers import get_logger
        from utils.torch_warmup import close_dynamo_import_window
        close_dynamo_import_window(get_logger(__name__))
    except Exception:  # noqa: BLE001, S110 - optimisation only, and this module has no logger
        pass

    try:
        from .ltx2_import_compat import ensure_ltx2_pipelines_importable, is_ltx2_pipeline_class

        if is_ltx2_pipeline_class(pipeline_class):
            ensure_ltx2_pipelines_importable()
        import diffusers

        present = hasattr(diffusers, pipeline_class)
        dummy_backends = _dummy_required_backends(getattr(diffusers, pipeline_class, None))
    except Exception as exc:  # noqa: BLE001 -- see below: this check must never raise anything but its own ValueError
        # Must raise ValueError, not ModuleNotFoundError/RuntimeError (hasattr imports submodules).
        if strict:
            raise ValueError(
                f"'{family_name}' needs diffusers ({pipeline_class}), which this environment "
                f"cannot import: {exc}. {DIFFUSERS_UPDATE_REMEDY} On a plain pip install, repair it "
                "with: pip install -U diffusers."
            ) from None
        return

    if present and dummy_backends:
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
    """Compares release numbers only: local suffixes like +dfsg and .dev0 git builds still have the
    class."""
    try:
        from packaging.version import Version
        return Version(Version(installed).base_version) >= Version(minimum)
    except Exception:  # noqa: BLE001 -- an unparseable version must not hide a model
        return True


def family_probe_class(fam: Any) -> str:
    """ModularPipeline is generic and predates families, so probe the family's transformer class instead."""
    name = str(getattr(fam, "pipeline_class", "") or "")
    if name == "ModularPipeline":
        return str(getattr(fam, "transformer_class", None) or name)
    return name


def family_pipeline_available(fam: Optional[DiffusionFamily]) -> bool:
    """Checks installed-version metadata, since probing lazy attributes imports pipeline deps; fails
    open."""
    if fam is None:
        return False
    name = family_probe_class(fam)
    # No class name: answer open; hasattr(diffusers, "") is False and would hide the model.
    if not name:
        return True
    if "diffusers" in sys.modules:
        module = sys.modules["diffusers"]
        if module is None:
            return True
        if not _module_namespace_is_unreadable(module):
            try:
                return hasattr(module, name)
            except Exception:  # noqa: BLE001 -- a probe failure must not hide a model
                return True
    minimum, _needs_py310 = pipeline_class_requirement(name)
    if minimum is None:
        return True
    installed = _installed_diffusers_version()
    if installed is None:
        return True
    return _installed_at_least(installed, minimum)


def _family_override_resolved(family_override: Optional[str], fam) -> tuple:
    reason = "detected from the model" if family_override is None else "requested"
    return (family_override, fam.name, reason)


def family_selectable(fam) -> bool:
    """Import-free: a status poll importing diffusers raced the loader's own diffusers import."""
    module = sys.modules.get("diffusers", False)
    if module is False:
        try:
            module = importlib.util.find_spec("diffusers")
        except (ImportError, ValueError):
            module = None
    return module is not None and family_pipeline_available(fam)


def pipeline_available_family_names() -> tuple[str, ...]:
    return tuple(fam.name for fam in _FAMILIES if family_selectable(fam))


def family_gguf_loadable(fam: DiffusionFamily) -> bool:
    """A single-file-pipeline family or a multi-denoiser family has no transformer-only GGUF to load."""
    return not fam.single_file_is_pipeline and not fam.pipeline_only


def family_mlx_supported(fam: Optional[DiffusionFamily]) -> bool:
    return fam is not None and bool(getattr(fam, "mlx_model_config", None))


def mlx_model_config_for(fam: DiffusionFamily, base_repo: Optional[str]) -> Optional[str]:
    if not family_mlx_supported(fam):
        return None
    key = (base_repo or "").strip().lower()
    for base, name in getattr(fam, "mlx_model_config_by_base", ()) or ():
        if key == base.lower():
            return name
    return fam.mlx_model_config


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """Only the VAE and text-encoder mapping; sd_cpp_arch_marker families also need a runnable binary."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS = (
    (
        "unsloth/FLUX.2-klein-9B-ComfyUI",
        "split_files/text_encoders/qwen_3_8b.safetensors",
        "llm",
    ),
)


def sd_cpp_companion_only_repo_ids() -> frozenset[str]:
    """Diffusion-only: repos with just sd.cpp's VAE or text encoder and no denoiser; not for chat
    listings."""
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


def prequant_only_repo_ids() -> frozenset[str]:
    """Repos hosting only prequant checkpoints (no model_index.json), never a base or mirror."""
    hosted: set[str] = set()
    bases: set[str] = set()
    for fam in _FAMILIES:
        hosted.update(repo for _scheme, repo in fam.prequant_repos)
        hosted.update(repo for _base, _scheme, repo in fam.prequant_variant_repos)
        hosted.update(repo for _scheme, _component, repo in fam.te_prequant_repos)
        bases.add(fam.base_repo)
        bases.update(fam.train_base_repos)
        if fam.deploy_base_repo:
            bases.add(fam.deploy_base_repo)
    bases.update(rid for pair in _MIRROR_PAIRS for rid in pair)
    lowered = {b.strip().lower() for b in bases if b}
    return frozenset(r.strip().lower() for r in hosted if r and r.strip().lower() not in lowered)


def prequant_repo_role(
    fam: DiffusionFamily, repo_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    key = (repo_id or "").strip().lower()
    known = {
        b.lower(): b
        for b in (fam.base_repo, *fam.train_base_repos, fam.deploy_base_repo or "")
        if b
    }
    bases: list[str] = []
    schemes: list[str] = []
    for entry_base, scheme, repo in fam.prequant_variant_repos:
        if repo.strip().lower() == key:
            bases.append(known.get(entry_base.lower(), entry_base))
            schemes.append(scheme)
    for scheme, repo in fam.prequant_repos:
        if repo.strip().lower() == key:
            bases.append(fam.base_repo)
            schemes.append(scheme)
    te = sorted(
        {
            scheme
            for scheme, _component, repo in fam.te_prequant_repos
            if repo.strip().lower() == key
        }
    )
    if te and not bases:
        bases.append(fam.base_repo)
    return tuple(dict.fromkeys(bases)), tuple(sorted(set(schemes))), tuple(te)


def sd_cpp_text_encoder_candidates(fam: DiffusionFamily) -> tuple[tuple[str, str, str], ...]:
    """Every encoder set a load could pick, for the delete guard, since a renamed file has no size token."""
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
    """FLUX.2-klein picks its encoder by GGUF inner_dim when known, not by a renamable filename."""
    if fam.name == "flux.2-klein":
        if inner_dim == _FLUX2_KLEIN_9B_INNER_DIM:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
        if inner_dim == _FLUX2_KLEIN_4B_INNER_DIM:
            return fam.sd_cpp_text_encoders
        identity = f"{repo_id or ''}/{gguf_filename or ''}".lower()
        # Match the size token alone: klein-BASE-9B is 9B too.
        if _token_in_needle("9b", identity) or "klein9b" in identity:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
    return fam.sd_cpp_text_encoders


# inner_dim is the only dim differing between klein sizes and dev; GGUF metadata cannot tell.
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
    """Keyed on upstream ids via canonical_base, so ungated mirrors match; None means fail open."""
    return _FLUX2_BASE_INNER_DIM.get(canonical_base(base_repo or "").lower())


def _flux2_inner_dim_from_tensors(tensors) -> Optional[int]:
    """``inner_dim`` from a parsed GGUF tensor table, or None when the probe tensor is absent."""
    for t in tensors:
        if t.name == _FLUX2_PROBE_TENSOR or t.name.endswith("." + _FLUX2_PROBE_TENSOR):
            # GGUF stores dims reversed. A missing dim answers nothing rather than 0.
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
    """Parses a range-read prefix; reads past its end are refused, so a cut shape is never taken as zero."""
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
                self.tensors = [_HeaderTensor(field.name, field.parts[3]) for field in fields]

        # GGUFReader memory-maps a path, so the prefix must land on disk.
        fd, tmp_path = tempfile.mkstemp(suffix = ".gguf-header")
        with os.fdopen(fd, "wb") as fh:
            fh.write(header)
        reader = _HeaderOnlyGGUFReader(tmp_path)
        if reader.data_offset > len(header) + min(int(reader.alignment), 4096):
            return None
        return _flux2_inner_dim_from_tensors(reader.tensors)
    except Exception:  # noqa: BLE001 - an unreadable header is not a verdict
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
    """One message shared by the plan, preflight and loader checks, so the user sees the same reason."""
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
    """Refuses a FLUX.2 GGUF paired with a mismatched base, before the quantizer's bare shape error."""
    if gguf_path is None or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return
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
    # resolve symlinks before the containment check (the lexical guards miss a symlink escape)
    repo_real = repo_root.resolve()
    child = repo_root.joinpath(*rel.parts).resolve()
    if child != repo_real and repo_real not in child.parents:
        raise ValueError("gguf_filename must resolve to a file inside the repo.")
    if not child.is_file():
        raise FileNotFoundError(f"'{gguf_filename}' is not a file under {repo_root}.")
    return child

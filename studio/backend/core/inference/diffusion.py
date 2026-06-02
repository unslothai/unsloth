# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion image generation backend.

Loads Hugging Face diffusion checkpoints in either the standard
``diffusers`` layout or the single-file GGUF layout published under
``unsloth/*-GGUF`` (Flux 2, Flux 2 Klein, Qwen-Image, SD3, SDXL, ...).
GGUF files are dynamically dequantised on-device via
``diffusers.GGUFQuantizationConfig``, then the rest of the pipeline
(VAE, text encoders, scheduler) is pulled from the matching ``diffusers``
repo so end users only ever need one local file plus the metadata repo.

The module is intentionally torch-only: it never spawns a subprocess and
shares the active CUDA / MPS device with the rest of Studio. The cost of
not having a separate process is that loading a diffusion model and a
GGUF chat model at the same time can OOM on consumer GPUs; the routes
layer must therefore swap between the two as needed (the orchestrator
unloads llama-server before any diffusion load on hosts with < 24 GB).

The class deliberately exposes a small, llama-cpp-style surface:

    load_model(repo_id, ...)
    generate_image(prompt, ...) -> PIL.Image
    unload_model()
    status() -> dict

so the route layer at ``studio/backend/routes/inference.py`` can mirror
the existing llama-server lifecycle (probe + load + generate + unload)
without learning a second API.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib.util
import io
import os
import re
import sys
import threading
import time
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE = "aggressive"
DIFFUSION_OFFLOAD_POLICY_BALANCED = "balanced"
DIFFUSION_OFFLOAD_POLICY_NONE = "none"
DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE = "less_aggressive"
DIFFUSION_OFFLOAD_POLICY_HYBRID = "hybrid"
DIFFUSION_OFFLOAD_POLICIES = {
    DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE,
    DIFFUSION_OFFLOAD_POLICY_BALANCED,
    DIFFUSION_OFFLOAD_POLICY_NONE,
    DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    DIFFUSION_OFFLOAD_POLICY_HYBRID,
}
DIFFUSION_SAFETENSORS_QUANT_NONE = "none"
DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT = "bitsandbytes_4bit"
DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4 = "bitsandbytes_4bit_nf4"
DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT = "bitsandbytes_8bit"
DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY = "torchao_int8_weight_only"
DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY = "torchao_int4_weight_only"
DIFFUSION_SAFETENSORS_QUANTS = {
    DIFFUSION_SAFETENSORS_QUANT_NONE,
    DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT,
    DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
    DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
    DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY,
    DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY,
}
DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS = (
    "transformer",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "pe",
)
DIFFUSION_DIFFUSERS_QUANT_COMPONENTS = {"transformer", "unet", "vae"}
DIFFUSION_TRANSFORMERS_QUANT_COMPONENTS = {
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "pe",
}
MIN_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB = 24 * 1024
MID_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB = 32 * 1024
HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB = 64 * 1024
VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB = 96 * 1024
MID_BALANCED_GGUF_CUDA_CACHE_MIB = 2048
DEFAULT_BALANCED_GGUF_CUDA_CACHE_MIB = 4096
HIGH_BALANCED_GGUF_CUDA_CACHE_MIB = 8192
VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_MIB = 16 * 1024
BALANCED_GGUF_CUDA_CACHE_HEADROOM_MIB = 8 * 1024


# ─── Pipeline registry ────────────────────────────────────────────────
#
# Keep this list narrow on purpose: only ship the small text-to-image
# families with first-class GGUF coverage on the Hub. Anything else is
# either video (LTX*, Wan) or research-grade (Sana, SD3.5) and can be
# added once it has a working GGUF release plus a smoke test.
#
# Each entry maps a substring of the loaded repo id (case-insensitive)
# to the (pipeline_class_name, transformer_class_name, default base
# repo for missing pieces). ``base_repo`` is what we pass to
# ``Pipeline.from_pretrained`` to pick up the VAE + text encoders when
# the user gave us a GGUF-only repo. The base_repo is documented to the
# user via ``status()`` so they understand why a second download fires.


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    media_kind: str = "image"
    guidance_kwarg: str = "guidance_scale"
    default_steps: int = 24
    default_guidance_scale: float = 3.5
    default_width: int = 1024
    default_height: int = 1024
    default_num_frames: Optional[int] = None
    default_frame_rate: Optional[float] = None
    default_negative_prompt: Optional[str] = None
    requires_image_input: bool = False
    default_call_kwargs: dict[str, Any] = field(default_factory = dict)
    supports_gguf_single_file: bool = True
    # Optional: list of HF "trigger" substrings besides ``name`` that map
    # to this family (e.g. "flux1-dev" plus "flux.1-dev"). Lowercased.
    aliases: tuple[str, ...] = field(default_factory = tuple)


@dataclass(frozen = True)
class DiffusionBaseRepoResolution:
    base_repo: str
    source: str
    confidence: str
    variant: Optional[str] = None
    warning: Optional[str] = None


@dataclass(frozen = True)
class DiffusionVariant:
    family: str
    variant: str
    base_repo: str
    description: str
    default_steps: Optional[int] = None
    default_guidance_scale: Optional[float] = None
    default_call_kwargs: dict[str, Any] = field(default_factory = dict)


@dataclass(frozen = True)
class DiffusionSamplingDefaults:
    default_steps: int
    default_guidance_scale: float
    default_call_kwargs: dict[str, Any] = field(default_factory = dict)


@dataclass(frozen = True)
class CuratedDiffusionGGUF:
    repo_id: str
    family: str
    base_repo: str
    filename_prefixes: tuple[str, ...]
    variant: Optional[str] = None
    recommended_offload_policy: str = DIFFUSION_OFFLOAD_POLICY_BALANCED


@dataclass(frozen = True)
class DiffusionLoadPreset:
    id: str
    display_name: str
    family: str
    pipeline_repo: str
    transformer_gguf_repo: str
    transformer_filename_prefixes: tuple[str, ...]
    variant: Optional[str] = None
    recommended_offload_policy: str = DIFFUSION_OFFLOAD_POLICY_BALANCED


@dataclass(frozen = True)
class DiffusionSamplingContract:
    family: Optional[str]
    media_kind: Optional[str]
    pipeline_class: Optional[str]
    transformer_class: Optional[str]
    base_repo: Optional[str]
    base_repo_source: Optional[str]
    base_repo_confidence: Optional[str]
    base_repo_variant: Optional[str]
    gguf: bool
    scheduler_class: Optional[str]
    scheduler_config_class: Optional[str]
    pipeline_is_distilled: Optional[bool]
    guidance_kwarg: str
    default_guidance_scale: float
    default_steps: int
    guidance_semantics: str
    default_width: int
    default_height: int
    requires_image_input: bool
    has_default_negative_prompt: bool
    default_call_kwargs: dict[str, Any]

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "media_kind": self.media_kind,
            "pipeline_class": self.pipeline_class,
            "transformer_class": self.transformer_class,
            "base_repo": _display_repo_id(self.base_repo),
            "base_repo_source": self.base_repo_source,
            "base_repo_confidence": self.base_repo_confidence,
            "base_repo_variant": self.base_repo_variant,
            "gguf": self.gguf,
            "scheduler_class": self.scheduler_class,
            "scheduler_config_class": self.scheduler_config_class,
            "pipeline_is_distilled": self.pipeline_is_distilled,
            "guidance_kwarg": self.guidance_kwarg,
            "default_guidance_scale": self.default_guidance_scale,
            "default_steps": self.default_steps,
            "guidance_semantics": self.guidance_semantics,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "requires_image_input": self.requires_image_input,
            "has_default_negative_prompt": self.has_default_negative_prompt,
            "default_call_kwargs": dict(self.default_call_kwargs),
        }


@dataclass(frozen = True)
class DiffusionLoraState:
    repo: str
    weight_name: Optional[str]
    adapter_name: str
    scale: float
    fused: bool

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "repo": _display_repo_id(self.repo),
            "weight_name": self.weight_name,
            "adapter_name": self.adapter_name,
            "scale": self.scale,
            "fused": self.fused,
        }


@dataclass(frozen = True)
class DiffusionGGUFInspection:
    """Lightweight GGUF identity signal used before choosing a base repo.

    This deliberately does not try to be a ComfyUI clone. ComfyUI-GGUF
    converts a GGUF into a Comfy state dict and then lets Comfy build a
    native model from key/shape signatures. Studio still loads through
    Diffusers, so this object only answers the questions we need before
    calling ``from_pretrained``: what architecture/layout does the file
    appear to use, which family variants are hinted, and are those hints
    strong enough to select a companion repo safely.
    """

    architecture: Optional[str]
    layout: Optional[str]
    family_hints: tuple[str, ...] = ()
    variant_hints: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, Any] = field(default_factory = dict)
    matched_signatures: tuple[str, ...] = ()
    tensor_count: int = 0
    warnings: tuple[str, ...] = ()

    def variant_values(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(variant for _, variant in self.variant_hints))


_FAMILIES: tuple[DiffusionFamily, ...] = (
    # The "9b" alias is checked first so a "flux-2-klein-9b" GGUF picks
    # the 9B companion repo instead of the 4B one when the user does not
    # pass an explicit base_repo. The common 4B GGUF path is distilled
    # and should default to its 4-step companion repo.
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        # Default for klein when no explicit base_repo: Apache-2.0
        # distilled 4B. Base repos are selected by _smart_base_repo when
        # the repo name contains "base".
        # The frontend curated picker always passes base_repo explicitly,
        # so this default only fires for "custom HF repo" mode.
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        default_steps = 4,
        default_guidance_scale = 1.0,
        aliases = ("flux2-klein", "flux-2-klein", "flux.2.klein"),
    ),
    DiffusionFamily(
        name = "flux.2",
        pipeline_class = "Flux2Pipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        default_steps = 50,
        default_guidance_scale = 4.0,
        aliases = ("flux2-dev", "flux-2-dev", "flux.2.dev"),
    ),
    DiffusionFamily(
        name = "flux.1-kontext",
        pipeline_class = "FluxKontextPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        default_steps = 28,
        default_guidance_scale = 2.5,
        requires_image_input = True,
        aliases = (
            "flux1-kontext",
            "flux1-kontext-dev",
            "flux-1-kontext",
            "flux-1-kontext-dev",
            "flux.1.kontext",
            "flux.1.kontext.dev",
        ),
    ),
    DiffusionFamily(
        name = "flux.1-schnell",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        default_steps = 4,
        default_guidance_scale = 0.0,
        aliases = (
            "flux1-schnell",
            "flux-1-schnell",
            "flux.1.schnell",
            "flux-schnell",
        ),
    ),
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-dev",
        default_steps = 50,
        default_guidance_scale = 3.5,
        aliases = ("flux1-dev", "flux-1-dev", "flux.1.dev", "flux-dev"),
    ),
    DiffusionFamily(
        name = "qwen-image-2512",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-2512",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
        aliases = ("qwenimage2512", "qwen_image_2512", "qwen-image-2512"),
    ),
    DiffusionFamily(
        name = "qwen-image-edit-2511",
        pipeline_class = "QwenImageEditPlusPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 40,
        default_guidance_scale = 4.0,
        default_negative_prompt = " ",
        default_call_kwargs = {"guidance_scale": 1.0},
        requires_image_input = True,
        aliases = ("qwenimageedit2511", "qwen_image_edit_2511", "qwen-image-edit-2511"),
    ),
    DiffusionFamily(
        name = "qwen-image-edit-2509",
        pipeline_class = "QwenImageEditPlusPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Edit-2509",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 40,
        default_guidance_scale = 4.0,
        default_negative_prompt = " ",
        default_call_kwargs = {"guidance_scale": 1.0},
        requires_image_input = True,
        aliases = ("qwenimageedit2509", "qwen_image_edit_2509", "qwen-image-edit-2509"),
    ),
    DiffusionFamily(
        name = "qwen-image-edit",
        pipeline_class = "QwenImageEditPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Edit",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_negative_prompt = " ",
        requires_image_input = True,
        aliases = ("qwenimageedit", "qwen_image_edit", "qwen-image-edit"),
    ),
    DiffusionFamily(
        name = "qwen-image-layered",
        pipeline_class = "QwenImageLayeredPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Layered",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_negative_prompt = " ",
        default_width = 640,
        default_height = 640,
        default_call_kwargs = {
            "layers": 4,
            "cfg_normalize": True,
            "use_en_prompt": True,
        },
        requires_image_input = True,
        aliases = ("qwenimagelayered", "qwen_image_layered", "qwen-image-layered"),
    ),
    DiffusionFamily(
        name = "qwen-image",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        guidance_kwarg = "true_cfg_scale",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_negative_prompt = " ",
        aliases = ("qwenimage", "qwen_image"),
    ),
    DiffusionFamily(
        name = "z-image-turbo",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        default_steps = 9,
        default_guidance_scale = 0.0,
        aliases = ("zimage-turbo", "z_image_turbo", "z-image-turbo"),
    ),
    DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_call_kwargs = {"cfg_normalization": False},
        aliases = ("zimage", "z_image", "z-image"),
    ),
    DiffusionFamily(
        name = "ernie-image-turbo",
        pipeline_class = "ErnieImagePipeline",
        transformer_class = "ErnieImageTransformer2DModel",
        base_repo = "baidu/ERNIE-Image-Turbo",
        default_steps = 8,
        default_guidance_scale = 1.0,
        default_call_kwargs = {"use_pe": True},
        supports_gguf_single_file = True,
        aliases = ("ernieimage-turbo", "ernie_image_turbo", "ernie-image-turbo"),
    ),
    DiffusionFamily(
        name = "ernie-image",
        pipeline_class = "ErnieImagePipeline",
        transformer_class = "ErnieImageTransformer2DModel",
        base_repo = "baidu/ERNIE-Image",
        default_steps = 50,
        default_guidance_scale = 4.0,
        default_call_kwargs = {"use_pe": True},
        supports_gguf_single_file = True,
        aliases = ("ernieimage", "ernie_image", "ernie-image"),
    ),
    DiffusionFamily(
        name = "stable-diffusion-3",
        pipeline_class = "StableDiffusion3Pipeline",
        transformer_class = "SD3Transformer2DModel",
        base_repo = "stabilityai/stable-diffusion-3-medium-diffusers",
        default_steps = 28,
        default_guidance_scale = 7.0,
        # Intentionally NOT including "sd3.5" / "stable-diffusion-3.5"
        # here: the SD3.5 family uses a different transformer config and
        # base repo than SD3 Medium, and silently pairing SD3.5 GGUFs
        # with the Medium base produces a misleading load. Add a
        # dedicated SD3.5 family with its own base_repo when we ship
        # smoke coverage for it.
        aliases = ("sd3-medium", "stable-diffusion-3-medium"),
    ),
    # SDXL: full diffusers path only (no GGUF). SDXL uses a UNet (not a
    # transformer) and wiring UNet2DConditionModel.from_single_file +
    # GGUF is a separate code path the rest of this module does not
    # exercise. The family is intentionally NOT in _FAMILIES so the
    # frontend status panel does not advertise GGUF support we do not
    # implement; callers wanting SDXL full repos can still do so by
    # passing the diffusers repo with no gguf_filename and
    # family_override = "stable-diffusion-xl" via the route, which uses
    # the lookup in _FULL_REPO_FAMILIES.
)


# Families available via family_override on the routes layer when the
# user is loading a full diffusers checkpoint (no GGUF). Kept separate
# from _FAMILIES so the GGUF-only status panel does not over-advertise.
_FULL_REPO_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "ltx2-3-distilled",
        pipeline_class = "LTX2Pipeline",
        transformer_class = "LTX2VideoTransformer3DModel",
        base_repo = "diffusers/LTX-2.3-Distilled-Diffusers",
        media_kind = "video",
        default_steps = 8,
        default_guidance_scale = 1.0,
        default_width = 768,
        default_height = 512,
        default_num_frames = 121,
        default_frame_rate = 24.0,
        supports_gguf_single_file = False,
        aliases = (
            "ltx2",
            "ltx-2",
            "ltx2-3",
            "ltx-2-3",
            "ltx2.3",
            "ltx-2.3",
            "ltx-2.3-distilled",
        ),
    ),
    DiffusionFamily(
        name = "wan2-2-t2v",
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        media_kind = "video",
        default_steps = 40,
        default_guidance_scale = 4.0,
        default_width = 1280,
        default_height = 720,
        default_num_frames = 81,
        default_frame_rate = 16.0,
        supports_gguf_single_file = False,
        aliases = (
            "wan",
            "wan2",
            "wan2-2",
            "wan-2-2",
            "wan2.2",
            "wan-2.2",
            "wan2-2-t2v",
            "wan2.2-t2v",
        ),
    ),
    DiffusionFamily(
        name = "stable-diffusion-xl",
        pipeline_class = "StableDiffusionXLPipeline",
        transformer_class = "",
        base_repo = "stabilityai/stable-diffusion-xl-base-1.0",
        default_steps = 40,
        default_guidance_scale = 5.0,
        supports_gguf_single_file = False,
        aliases = ("sdxl",),
    ),
)

_TEXT_ENCODER_GGUF_COMPONENTS = frozenset(
    {
        "pe",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
    }
)

_FLUX2_DEFAULT_TEXT_ENCODER_GGUF_REPO = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
_ERNIE_DEFAULT_TEXT_ENCODER_GGUF_REPO = "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
_ERNIE_DEFAULT_PROMPT_ENHANCER_GGUF_REPO = "Green-Sky/Ernie-Image-Prompt-Enhancer-Ministral-3B-GGUF"
_QWEN2VL_DEFAULT_TEXT_ENCODER_GGUF_REPO = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
_QWEN3_DEFAULT_TEXT_ENCODER_GGUF_REPO = "unsloth/Qwen3-4B-GGUF"


@dataclass(frozen = True)
class _TextEncoderGgufPlan:
    loader_name: str
    component_name: str


def _default_text_encoder_gguf_repo(fam: DiffusionFamily) -> str:
    if fam.name.startswith("ernie-image"):
        return _ERNIE_DEFAULT_TEXT_ENCODER_GGUF_REPO
    if fam.name.startswith("qwen-image"):
        return _QWEN2VL_DEFAULT_TEXT_ENCODER_GGUF_REPO
    if fam.name.startswith("z-image") or fam.name == "flux.2-klein":
        return _QWEN3_DEFAULT_TEXT_ENCODER_GGUF_REPO
    return _FLUX2_DEFAULT_TEXT_ENCODER_GGUF_REPO


def _default_prompt_enhancer_gguf_repo(fam: DiffusionFamily) -> Optional[str]:
    if fam.name.startswith("ernie-image"):
        return _ERNIE_DEFAULT_PROMPT_ENHANCER_GGUF_REPO
    return None


def _resolve_text_encoder_gguf_plan(
    fam: DiffusionFamily,
    *,
    architecture: Optional[str],
    requested_component: Optional[str],
) -> Optional[_TextEncoderGgufPlan]:
    component_name = requested_component or "text_encoder"

    # FLUX.2 dev uses a Mistral3 text encoder. Some Mistral-family GGUFs
    # report a llama-like architecture, so the pipeline family must win
    # over the generic architecture mapping here.
    if fam.name == "flux.2":
        if component_name != "text_encoder":
            raise RuntimeError(
                "FLUX.2 text_encoder_gguf_filename can only replace "
                "the text_encoder component."
            )
        return _TextEncoderGgufPlan(
            loader_name = "LazyFlux2MistralTextEncoder",
            component_name = "text_encoder",
        )

    if fam.name.startswith("ernie-image"):
        if component_name == "text_encoder":
            return _TextEncoderGgufPlan(
                loader_name = "LazyMistral3TextEncoder",
                component_name = "text_encoder",
            )
        if component_name == "pe":
            return _TextEncoderGgufPlan(
                loader_name = "LazyMinistral3PromptEnhancer",
                component_name = "pe",
            )

    if architecture == "qwen2vl" and fam.name.startswith("qwen-image"):
        return _TextEncoderGgufPlan(
            loader_name = "LazyQwen2VLTextEncoder",
            component_name = component_name,
        )

    if architecture == "qwen3" and (
        fam.name.startswith("z-image") or fam.name == "flux.2-klein"
    ):
        return _TextEncoderGgufPlan(
            loader_name = "LazyQwen3TextEncoder",
            component_name = component_name,
        )

    if architecture in {"t5", "t5encoder"} and fam.name in {
        "flux.1",
        "stable-diffusion-3",
    }:
        if requested_component is None:
            component_name = (
                "text_encoder_3"
                if fam.name == "stable-diffusion-3"
                else "text_encoder_2"
            )
        return _TextEncoderGgufPlan(
            loader_name = "LazyT5TextEncoder",
            component_name = component_name,
        )

    if requested_component is None:
        return None

    generic_loader_by_architecture = {
        "mistral3": "LazyMistral3TextEncoder",
        "llama": "LazyLlamaTextEncoder",
        "qwen2vl": "LazyQwen2VLTextEncoder",
        "qwen3": "LazyQwen3TextEncoder",
        "qwen3vl": "LazyQwen3VLTextEncoder",
        "gemma3": "LazyGemma3TextEncoder",
        "t5": "LazyT5TextEncoder",
        "t5encoder": "LazyT5TextEncoder",
    }
    loader_name = generic_loader_by_architecture.get(architecture or "")
    if loader_name is None:
        return None
    return _TextEncoderGgufPlan(
        loader_name = loader_name,
        component_name = component_name,
    )


def _load_text_encoder_gguf_from_plan(
    gguf_text_encoder_mod: Any,
    plan: _TextEncoderGgufPlan,
    gguf_path: str,
    *,
    base_repo_or_path: str,
    mmproj_gguf_path: Any,
    compute_dtype: Any,
    resident_device: Any,
    token: Optional[str],
) -> Any:
    loader = getattr(gguf_text_encoder_mod, plan.loader_name)
    common_kwargs = {
        "base_repo_or_path": base_repo_or_path,
        "compute_dtype": compute_dtype,
        "resident_device": resident_device,
        "token": token,
    }
    if plan.loader_name == "LazyQwen2VLTextEncoder":
        return loader.from_gguf(
            gguf_path,
            mmproj_gguf_path = mmproj_gguf_path,
            **common_kwargs,
        )
    if plan.loader_name in {
        "LazyLlamaTextEncoder",
        "LazyQwen3VLTextEncoder",
        "LazyGemma3TextEncoder",
        "LazyT5TextEncoder",
        "LazyMinistral3PromptEnhancer",
    }:
        return loader.from_gguf(
            gguf_path,
            subfolder = plan.component_name,
            **common_kwargs,
        )
    return loader.from_gguf(gguf_path, **common_kwargs)


def _repo_leaf(value: str) -> str:
    cleaned = (value or "").rstrip("/\\")
    return re.split(r"[\\/]+", cleaned)[-1].lower() if cleaned else ""


@dataclass(frozen = True)
class _DiffusionGGUFKeySignature:
    name: str
    architecture: str
    layout: str
    family_hints: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


_DIFFUSION_GGUF_IMAGE_ARCHITECTURES = frozenset(
    {
        "ernie_image",
        "flux",
        "qwen_image",
        "sd3",
        "sdxl",
        "z_image",
        "wan",
        "ltxv",
    }
)

_DIFFUSION_GGUF_SIGNATURES: tuple[_DiffusionGGUFKeySignature, ...] = (
    _DiffusionGGUFKeySignature(
        name = "flux_comfy",
        architecture = "flux",
        layout = "comfy",
        family_hints = (
            "flux.1",
            "flux.1-kontext",
            "flux.1-schnell",
            "flux.2",
            "flux.2-klein",
        ),
        required = ("double_blocks.0.img_attn.proj.weight",),
    ),
    _DiffusionGGUFKeySignature(
        name = "flux_diffusers",
        architecture = "flux",
        layout = "diffusers",
        family_hints = (
            "flux.1",
            "flux.1-kontext",
            "flux.1-schnell",
            "flux.2",
            "flux.2-klein",
        ),
        required = ("transformer_blocks.0.attn.norm_added_k.weight",),
    ),
    _DiffusionGGUFKeySignature(
        name = "z_image",
        architecture = "z_image",
        layout = "comfy",
        family_hints = ("z-image", "z-image-turbo"),
        required = ("noise_refiner.0.attention.norm_k.weight",),
    ),
    _DiffusionGGUFKeySignature(
        name = "ernie_image",
        architecture = "ernie_image",
        layout = "diffusers",
        family_hints = ("ernie-image", "ernie-image-turbo"),
        required = (
            "adaLN_modulation.1.weight",
            "layers.0.self_attention.to_q.weight",
            "layers.0.self_attention.to_out.0.weight",
        ),
    ),
    _DiffusionGGUFKeySignature(
        name = "sd3_comfy",
        architecture = "sd3",
        layout = "comfy",
        family_hints = ("stable-diffusion-3",),
        required = ("joint_blocks.0.x_block.attn.qkv.weight",),
    ),
    _DiffusionGGUFKeySignature(
        name = "sd3_diffusers",
        architecture = "sd3",
        layout = "diffusers",
        family_hints = ("stable-diffusion-3",),
        required = (
            "transformer_blocks.0.attn.add_q_proj.weight",
            "pos_embed.proj.weight",
        ),
    ),
    _DiffusionGGUFKeySignature(
        name = "sdxl_diffusers",
        architecture = "sdxl",
        layout = "diffusers_unet",
        family_hints = ("stable-diffusion-xl",),
        required = (
            "down_blocks.0.downsamplers.0.conv.weight",
            "add_embedding.linear_1.weight",
        ),
    ),
    _DiffusionGGUFKeySignature(
        name = "sdxl_comfy",
        architecture = "sdxl",
        layout = "comfy_ldm",
        family_hints = ("stable-diffusion-xl",),
        required = (
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight",
            "output_blocks.5.2.conv.weight",
        ),
    ),
)


_GGUF_ARCHITECTURE_FAMILY_HINTS: dict[str, tuple[str, ...]] = {
    "flux": (
        "flux.1",
        "flux.1-kontext",
        "flux.1-schnell",
        "flux.2",
        "flux.2-klein",
    ),
    "qwen_image": (
        "qwen-image",
        "qwen-image-2512",
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
        "qwen-image-layered",
    ),
    "sd3": ("stable-diffusion-3",),
    "sdxl": ("stable-diffusion-xl",),
    "z_image": ("z-image", "z-image-turbo"),
    "ernie_image": ("ernie-image", "ernie-image-turbo"),
    "wan": ("wan2-2-t2v",),
    "ltxv": ("ltx2-3-distilled",),
}


def _metadata_string_values(metadata: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for key, value in sorted(metadata.items()):
        if isinstance(value, str) and value.strip():
            values.append((key, value.strip()))
    return tuple(values)


def _inspect_diffusion_gguf_tensor_names(
    tensor_names: set[str],
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> DiffusionGGUFInspection:
    metadata = dict(metadata or {})
    metadata_arch = metadata.get("general.architecture")
    architecture = (
        str(metadata_arch).strip().lower()
        if isinstance(metadata_arch, str) and metadata_arch.strip()
        else None
    )
    matched_signatures: list[str] = []
    family_hints: list[str] = []
    layouts: list[str] = []
    signature_arches: list[str] = []
    warnings: list[str] = []

    if architecture in _GGUF_ARCHITECTURE_FAMILY_HINTS:
        family_hints.extend(_GGUF_ARCHITECTURE_FAMILY_HINTS[architecture])
    elif architecture and architecture not in _DIFFUSION_GGUF_IMAGE_ARCHITECTURES:
        warnings.append(f"unrecognized_gguf_architecture:{architecture}")

    for signature in _DIFFUSION_GGUF_SIGNATURES:
        if not all(key in tensor_names for key in signature.required):
            continue
        if any(key in tensor_names for key in signature.forbidden):
            continue
        matched_signatures.append(signature.name)
        layouts.append(signature.layout)
        signature_arches.append(signature.architecture)
        family_hints.extend(signature.family_hints)

    unique_signature_arches = tuple(dict.fromkeys(signature_arches))
    if architecture is None and len(unique_signature_arches) == 1:
        architecture = unique_signature_arches[0]
    elif architecture is not None:
        for signature_arch in unique_signature_arches:
            if signature_arch != architecture:
                warnings.append(
                    f"architecture_conflict:metadata={architecture},signature={signature_arch}"
                )
        if len(unique_signature_arches) == 1 and unique_signature_arches[0] != architecture:
            signature_arch = unique_signature_arches[0]
            should_prefer_signature = (
                architecture == "wan"
                and signature_arch == "ernie_image"
            ) or architecture not in _DIFFUSION_GGUF_IMAGE_ARCHITECTURES
            if should_prefer_signature:
                metadata_hints = _GGUF_ARCHITECTURE_FAMILY_HINTS.get(architecture, ())
                if metadata_hints:
                    family_hints = [
                        hint
                        for hint in family_hints
                        if hint not in metadata_hints
                    ]
                architecture = signature_arch
                family_hints.extend(
                    _GGUF_ARCHITECTURE_FAMILY_HINTS.get(architecture, ())
                )
            else:
                signature_hints = _GGUF_ARCHITECTURE_FAMILY_HINTS.get(signature_arch, ())
                if signature_hints:
                    family_hints = [
                        hint
                        for hint in family_hints
                        if hint not in signature_hints
                    ]

    unique_layouts = tuple(dict.fromkeys(layouts))
    layout = unique_layouts[0] if len(unique_layouts) == 1 else None
    if len(unique_layouts) > 1:
        warnings.append("layout_conflict:" + ",".join(unique_layouts))

    variant_hints: list[tuple[str, str]] = []
    for key, value in _metadata_string_values(metadata):
        for family in _DIFFUSION_VARIANTS_BY_FAMILY:
            if variant := _variant_from_text_for_family(family, value):
                variant_hints.append((f"metadata:{key}", variant))

    return DiffusionGGUFInspection(
        architecture = architecture,
        layout = layout,
        family_hints = tuple(dict.fromkeys(family_hints)),
        variant_hints = tuple(dict.fromkeys(variant_hints)),
        metadata = metadata,
        matched_signatures = tuple(matched_signatures),
        tensor_count = len(tensor_names),
        warnings = tuple(dict.fromkeys(warnings)),
    )


def _read_diffusion_gguf_metadata(reader: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        field_names = list(getattr(reader, "fields", {}) or {})
    except Exception:
        return metadata
    for field_name in field_names:
        try:
            field = reader.get_field(field_name)
            field_types = getattr(field, "types", ())
            if len(field_types) != 1:
                continue
            field_type = field_types[0]
            # Avoid importing gguf enum names at module import time.
            type_name = getattr(field_type, "name", str(field_type)).upper()
            value = field.parts[field.data[-1]]
            if type_name.endswith("STRING"):
                metadata[field_name] = str(value, "utf-8")
            elif type_name.endswith("INT32") or type_name.endswith("UINT32"):
                metadata[field_name] = int(value.item())
            elif type_name.endswith("F32") or type_name.endswith("FLOAT32"):
                metadata[field_name] = float(value.item())
            elif type_name.endswith("BOOL"):
                metadata[field_name] = bool(value.item())
        except Exception:
            continue
    return metadata


def _inspect_diffusion_gguf_file(path: Path) -> DiffusionGGUFInspection:
    try:
        import gguf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Diffusion GGUF inspection requires the gguf runtime package. "
            "Re-run Studio setup before loading an image GGUF."
        ) from exc

    reader = gguf.GGUFReader(str(path))
    tensor_names = {str(tensor.name) for tensor in getattr(reader, "tensors", ())}
    metadata = _read_diffusion_gguf_metadata(reader)
    return _inspect_diffusion_gguf_tensor_names(tensor_names, metadata = metadata)


def _gguf_inspection_matches_family(
    inspection: Optional[DiffusionGGUFInspection],
    fam: DiffusionFamily,
) -> bool:
    if inspection is None or not inspection.family_hints:
        return True
    return fam.name in inspection.family_hints


_DIFFUSION_VARIANTS: tuple[DiffusionVariant, ...] = (
    DiffusionVariant(
        family = "flux.2-klein",
        variant = "distilled-4b",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        description = "distilled 4B",
        default_steps = 4,
        default_guidance_scale = 1.0,
    ),
    DiffusionVariant(
        family = "flux.2-klein",
        variant = "distilled-9b",
        base_repo = "black-forest-labs/FLUX.2-klein-9B",
        description = "distilled 9B",
        default_steps = 4,
        default_guidance_scale = 1.0,
    ),
    DiffusionVariant(
        family = "flux.2-klein",
        variant = "base-4b",
        base_repo = "black-forest-labs/FLUX.2-klein-base-4B",
        description = "base 4B",
        default_steps = 50,
        default_guidance_scale = 4.0,
    ),
    DiffusionVariant(
        family = "flux.2-klein",
        variant = "base-9b",
        base_repo = "black-forest-labs/FLUX.2-klein-base-9B",
        description = "base 9B",
        default_steps = 50,
        default_guidance_scale = 4.0,
    ),
)

_DIFFUSION_VARIANTS_BY_FAMILY: dict[str, tuple[DiffusionVariant, ...]] = {
    family: tuple(variant for variant in _DIFFUSION_VARIANTS if variant.family == family)
    for family in tuple(dict.fromkeys(variant.family for variant in _DIFFUSION_VARIANTS))
}

_DIFFUSION_VARIANT_BY_FAMILY_AND_ID: dict[tuple[str, str], DiffusionVariant] = {
    (variant.family, variant.variant): variant
    for variant in _DIFFUSION_VARIANTS
}

_CURATED_UNSLOTH_DIFFUSION_GGUFS: tuple[CuratedDiffusionGGUF, ...] = (
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.1-Kontext-dev-GGUF",
        family = "flux.1-kontext",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        filename_prefixes = ("flux1-kontext-dev-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        family = "flux.1",
        base_repo = "black-forest-labs/FLUX.1-dev",
        filename_prefixes = ("flux1-dev-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.1-schnell-GGUF",
        family = "flux.1-schnell",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        filename_prefixes = ("flux1-schnell-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.2-dev-GGUF",
        family = "flux.2",
        base_repo = "black-forest-labs/FLUX.2-dev",
        filename_prefixes = ("flux2-dev-",),
        recommended_offload_policy = DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.2-klein-4B-GGUF",
        family = "flux.2-klein",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        filename_prefixes = ("flux-2-klein-4b-",),
        variant = "distilled-4b",
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.2-klein-9B-GGUF",
        family = "flux.2-klein",
        base_repo = "black-forest-labs/FLUX.2-klein-9B",
        filename_prefixes = ("flux-2-klein-9b-",),
        variant = "distilled-9b",
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.2-klein-base-4B-GGUF",
        family = "flux.2-klein",
        base_repo = "black-forest-labs/FLUX.2-klein-base-4B",
        filename_prefixes = ("flux-2-klein-base-4b-",),
        variant = "base-4b",
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/FLUX.2-klein-base-9B-GGUF",
        family = "flux.2-klein",
        base_repo = "black-forest-labs/FLUX.2-klein-base-9B",
        filename_prefixes = ("flux-2-klein-base-9b-",),
        variant = "base-9b",
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Z-Image-GGUF",
        family = "z-image",
        base_repo = "Tongyi-MAI/Z-Image",
        filename_prefixes = ("z-image-",),
        recommended_offload_policy = DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        family = "z-image-turbo",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        filename_prefixes = ("z-image-turbo-",),
        recommended_offload_policy = DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/ERNIE-Image-Turbo-GGUF",
        family = "ernie-image-turbo",
        base_repo = "baidu/ERNIE-Image-Turbo",
        filename_prefixes = ("ernie-image-turbo-",),
        recommended_offload_policy = DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/ERNIE-Image-GGUF",
        family = "ernie-image",
        base_repo = "baidu/ERNIE-Image",
        filename_prefixes = ("ernie-image-",),
        recommended_offload_policy = DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-GGUF",
        family = "qwen-image",
        base_repo = "Qwen/Qwen-Image",
        filename_prefixes = ("qwen-image-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-Edit-GGUF",
        family = "qwen-image-edit",
        base_repo = "Qwen/Qwen-Image-Edit",
        filename_prefixes = ("qwen-image-edit-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-Edit-2509-GGUF",
        family = "qwen-image-edit-2509",
        base_repo = "Qwen/Qwen-Image-Edit-2509",
        filename_prefixes = ("qwen-image-edit-2509-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-2512-GGUF",
        family = "qwen-image-2512",
        base_repo = "Qwen/Qwen-Image-2512",
        filename_prefixes = ("qwen-image-2512-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-Edit-2511-GGUF",
        family = "qwen-image-edit-2511",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        filename_prefixes = ("qwen-image-edit-2511-",),
    ),
    CuratedDiffusionGGUF(
        repo_id = "unsloth/Qwen-Image-Layered-GGUF",
        family = "qwen-image-layered",
        base_repo = "Qwen/Qwen-Image-Layered",
        filename_prefixes = ("qwen-image-layered-",),
    ),
)

_CURATED_UNSLOTH_DIFFUSION_GGUFS_BY_REPO: dict[str, CuratedDiffusionGGUF] = {
    spec.repo_id.lower(): spec
    for spec in _CURATED_UNSLOTH_DIFFUSION_GGUFS
}


def _curated_gguf_recommended_offload_policy(
    *,
    repo_id: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    transformer_gguf_repo: Optional[str] = None,
    transformer_gguf_filename: Optional[str] = None,
    device: Optional[str] = "cuda",
    free_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
) -> Optional[str]:
    """Return Studio's curated policy for recognized GGUF component loads."""

    spec: Optional[CuratedDiffusionGGUF] = None
    if transformer_gguf_repo and transformer_gguf_filename:
        spec = _CURATED_UNSLOTH_DIFFUSION_GGUFS_BY_REPO.get(
            str(transformer_gguf_repo).lower()
        )
    if spec is None and repo_id and gguf_filename:
        spec = _CURATED_UNSLOTH_DIFFUSION_GGUFS_BY_REPO.get(str(repo_id).lower())
    if spec is None:
        return None

    return spec.recommended_offload_policy


def _preset_id_from_curated_diffusion_gguf(spec: CuratedDiffusionGGUF) -> str:
    leaf = _repo_leaf(spec.repo_id)
    if leaf.endswith("-gguf"):
        leaf = leaf[:-5]
    return leaf.replace("_", "-")


def _title_from_preset_id(preset_id: str) -> str:
    special = {
        "flux": "FLUX",
        "qwen": "Qwen",
        "z": "Z",
        "ernie": "ERNIE",
        "sd3": "SD3",
        "sdxl": "SDXL",
    }
    words = []
    for raw in re.split(r"[-_.]+", preset_id):
        if not raw:
            continue
        words.append(special.get(raw, raw.upper() if raw in {"gguf"} else raw.capitalize()))
    return " ".join(words)


def _diffusion_preset_from_curated_gguf(
    spec: CuratedDiffusionGGUF,
) -> DiffusionLoadPreset:
    preset_id = _preset_id_from_curated_diffusion_gguf(spec)
    return DiffusionLoadPreset(
        id = preset_id,
        display_name = _title_from_preset_id(preset_id),
        family = spec.family,
        pipeline_repo = spec.base_repo,
        transformer_gguf_repo = spec.repo_id,
        transformer_filename_prefixes = spec.filename_prefixes,
        variant = spec.variant,
        recommended_offload_policy = spec.recommended_offload_policy,
    )


_CURATED_DIFFUSION_PRESETS: tuple[DiffusionLoadPreset, ...] = tuple(
    _diffusion_preset_from_curated_gguf(spec)
    for spec in _CURATED_UNSLOTH_DIFFUSION_GGUFS
)

_CURATED_DIFFUSION_PRESETS_BY_ID: dict[str, DiffusionLoadPreset] = {
    key: preset
    for preset in _CURATED_DIFFUSION_PRESETS
    for key in (
        preset.id.lower(),
        preset.transformer_gguf_repo.lower(),
        _repo_leaf(preset.transformer_gguf_repo).lower(),
    )
}


def _contains_label(text: str, label: str) -> bool:
    return re.search(rf"(^|[^a-z0-9]){re.escape(label)}([^a-z0-9]|$)", text) is not None


def _flux2_klein_variant_from_text(text: str) -> Optional[str]:
    """Infer the Flux2 Klein variant from one repo/file leaf.

    The parser is intentionally conservative: it only looks at the
    repo/file leaf (callers strip parent dirs first) and treats
    ``base`` as a token so names like ``database`` do not accidentally
    select a base model. GGUF filenames often carry the missing variant
    clue for third-party finetunes, so this helper is used for both the
    repo leaf and GGUF filename leaf.
    """

    value = text.lower()
    is_9b = re.search(r"(^|[^a-z0-9])9b([^a-z0-9]|$)", value) is not None
    is_4b = re.search(r"(^|[^a-z0-9])4b([^a-z0-9]|$)", value) is not None
    is_base = (
        _contains_label(value, "base")
        or re.search(r"base[-_. ]?(4b|9b)", value) is not None
    )
    is_distilled = (
        "distill" in value
        or _contains_label(value, "schnell")
    )
    if is_9b and is_base:
        return "base-9b"
    if is_9b:
        return "distilled-9b"
    if is_4b and is_base:
        return "base-4b"
    if is_base:
        return "base-4b"
    if is_4b or is_distilled:
        return "distilled-4b"
    return None


_DIFFUSION_VARIANT_TEXT_DETECTORS: dict[str, Callable[[str], Optional[str]]] = {
    "flux.2-klein": _flux2_klein_variant_from_text,
}


def _family_variant_display_name(family: str) -> str:
    if family == "flux.2-klein":
        return "FLUX.2 Klein"
    return family


def _variant_from_text_for_family(family: str, text: str) -> Optional[str]:
    detector = _DIFFUSION_VARIANT_TEXT_DETECTORS.get(family)
    if detector is None:
        return None
    variant = detector(text)
    if variant is None:
        return None
    if (family, variant) not in _DIFFUSION_VARIANT_BY_FAMILY_AND_ID:
        return None
    return variant


def _candidate_base_repo_message(family: str) -> str:
    candidates = _DIFFUSION_VARIANTS_BY_FAMILY.get(family, ())
    if not candidates:
        return ""
    joined = ", ".join(
        f"{candidate.base_repo} ({candidate.description})"
        for candidate in candidates
    )
    return f"base_repo candidates: {joined}."


def _curated_unsloth_diffusion_gguf(repo_id: str) -> Optional[CuratedDiffusionGGUF]:
    return _CURATED_UNSLOTH_DIFFUSION_GGUFS_BY_REPO.get(str(repo_id).lower())


def _filename_matches_curated_diffusion_gguf(
    spec: CuratedDiffusionGGUF,
    gguf_filename: Optional[str],
) -> bool:
    leaf = _repo_leaf(gguf_filename or "").lower()
    return bool(leaf) and any(
        leaf.startswith(prefix.lower())
        for prefix in spec.filename_prefixes
    )


def _variant_hint_matches_for_family(
    *,
    family: str,
    repo_id: str,
    gguf_filename: Optional[str],
    gguf_inspection: Optional[DiffusionGGUFInspection] = None,
) -> list[tuple[str, str, str]]:
    labels = [
        ("repo/path", _repo_leaf(repo_id), "name_heuristic"),
        ("gguf_filename", _repo_leaf(gguf_filename or ""), "filename_heuristic"),
    ]
    matches = [
        (label_name, variant, source)
        for label_name, label, source in labels
        if (variant := _variant_from_text_for_family(family, label)) is not None
    ]
    if gguf_inspection is not None:
        matches.extend(
            (source, variant, "gguf_metadata")
            for source, variant in gguf_inspection.variant_hints
            if (family, variant) in _DIFFUSION_VARIANT_BY_FAMILY_AND_ID
        )
    return matches


def _resolve_variant_base_repo_from_hints(
    *,
    fam: DiffusionFamily,
    repo_id: str,
    gguf_filename: Optional[str],
    gguf_inspection: Optional[DiffusionGGUFInspection] = None,
) -> DiffusionBaseRepoResolution:
    matches = _variant_hint_matches_for_family(
        family = fam.name,
        repo_id = repo_id,
        gguf_filename = gguf_filename,
        gguf_inspection = gguf_inspection,
    )
    variants = {variant for _, variant, _ in matches}
    candidate_lines = _candidate_base_repo_message(fam.name)
    family_display = _family_variant_display_name(fam.name)
    if not variants:
        raise RuntimeError(
            f"Ambiguous {family_display} GGUF: could not infer the model "
            "variant from repo/path "
            f"'{_display_repo_id(repo_id)}' or GGUF filename "
            f"'{Path(gguf_filename or '').name}'. Pass base_repo explicitly. "
            f"{candidate_lines} If this is a finetune, use the original "
            "base repo it was trained from."
        )
    if len(variants) > 1:
        seen = ", ".join(f"{label}={variant}" for label, variant, _ in matches)
        raise RuntimeError(
            f"Conflicting {family_display} GGUF variant hints: "
            f"{seen}. Pass base_repo explicitly so Studio does not use the "
            f"wrong sampling contract. {candidate_lines}"
        )
    variant = next(iter(variants))
    variant_config = _DIFFUSION_VARIANT_BY_FAMILY_AND_ID[(fam.name, variant)]
    source = matches[0][2]
    return DiffusionBaseRepoResolution(
        base_repo = variant_config.base_repo,
        source = source,
        confidence = "heuristic",
        variant = variant,
    )


def _resolve_diffusion_base_repo(
    *,
    fam: DiffusionFamily,
    repo_id: str,
    gguf_filename: Optional[str],
    base_repo: Optional[str],
    gguf_inspection: Optional[DiffusionGGUFInspection] = None,
) -> DiffusionBaseRepoResolution:
    """Resolve the companion Diffusers repo for a diffusion load.

    Name-based inference is kept as a compatibility fallback for curated
    and obvious GGUF repos, but the returned metadata makes that decision
    auditable. Flux2 Klein is intentionally stricter than the older
    ``_smart_base_repo`` helper: base and distilled variants share a
    family but require different sampling semantics, so ambiguous custom
    fine-tunes must pass an explicit ``base_repo`` instead of silently
    inheriting the distilled default.
    """

    if not gguf_filename:
        variant = None
        if fam.name in _DIFFUSION_VARIANTS_BY_FAMILY:
            variant = _variant_from_text_for_family(fam.name, _repo_leaf(repo_id))
        return DiffusionBaseRepoResolution(
            base_repo = _expand_existing_local_path(repo_id),
            source = "full_repo",
            confidence = "explicit",
            variant = variant,
        )
    if base_repo:
        variant = None
        if fam.name in _DIFFUSION_VARIANTS_BY_FAMILY:
            variant = _variant_from_text_for_family(fam.name, _repo_leaf(base_repo))
        return DiffusionBaseRepoResolution(
            base_repo = _expand_existing_local_path(base_repo),
            source = "explicit",
            confidence = "explicit",
            variant = variant,
        )
    curated = _curated_unsloth_diffusion_gguf(repo_id)
    if (
        curated is not None
        and curated.family == fam.name
        and _filename_matches_curated_diffusion_gguf(curated, gguf_filename)
    ):
        return DiffusionBaseRepoResolution(
            base_repo = curated.base_repo,
            source = "name_heuristic" if curated.variant else "family_default",
            confidence = "heuristic",
            variant = curated.variant,
        )
    if fam.name not in _DIFFUSION_VARIANTS_BY_FAMILY:
        return DiffusionBaseRepoResolution(
            base_repo = fam.base_repo,
            source = "family_default",
            confidence = "heuristic",
        )

    return _resolve_variant_base_repo_from_hints(
        fam = fam,
        repo_id = repo_id,
        gguf_filename = gguf_filename,
        gguf_inspection = gguf_inspection,
    )


def _smart_base_repo(fam: DiffusionFamily, repo_id: str) -> str:
    """Pick the best matching base diffusers repo for a given GGUF repo
    when the caller did not pass an explicit base_repo.

    Currently only specialises the flux.2-klein family: explicit
    variant markers in the repo leaf choose the matching 4B / 9B and
    base / distilled companion repo. Ambiguous names raise so callers
    do not silently get the wrong sampling contract.

    Only the LAST segment of the repo id / path is inspected so a
    namespace or parent directory like ``baseorg/...`` or
    ``/home/me/.cache/base/...`` does not falsely select the Base
    variant (round 12 review #9). Splits on BOTH ``/`` and ``\\`` so
    Windows local paths like ``C:\\Users\\me\\base\\FLUX.2-klein-4B``
    do not get scored as "base" via the parent directory either
    (round 13 P2 #13).
    """
    return _resolve_diffusion_base_repo(
        fam = fam,
        repo_id = repo_id,
        gguf_filename = "model.gguf",
        base_repo = None,
        gguf_inspection = None,
    ).base_repo


def _optional_class_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return type(value).__name__


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _pipeline_is_distilled(pipe: Any) -> Optional[bool]:
    config = getattr(pipe, "config", None)
    if config is None:
        return None
    if isinstance(config, dict):
        return _optional_bool(config.get("is_distilled"))
    return _optional_bool(getattr(config, "is_distilled", None))


def _sampling_defaults_for_family(
    fam: DiffusionFamily,
    *,
    base_repo_variant: Optional[str] = None,
) -> DiffusionSamplingDefaults:
    default_steps = fam.default_steps
    default_guidance_scale = fam.default_guidance_scale
    default_call_kwargs = dict(fam.default_call_kwargs)

    if base_repo_variant is not None:
        variant = _DIFFUSION_VARIANT_BY_FAMILY_AND_ID.get(
            (fam.name, base_repo_variant)
        )
        if variant is not None:
            if variant.default_steps is not None:
                default_steps = variant.default_steps
            if variant.default_guidance_scale is not None:
                default_guidance_scale = variant.default_guidance_scale
            default_call_kwargs.update(variant.default_call_kwargs)

    return DiffusionSamplingDefaults(
        default_steps = int(default_steps),
        default_guidance_scale = float(default_guidance_scale),
        default_call_kwargs = default_call_kwargs,
    )


def _family_by_name(name: str) -> Optional[DiffusionFamily]:
    for fam in _FAMILIES + _FULL_REPO_FAMILIES:
        if fam.name == name:
            return fam
    return None


def _preset_default_text_encoder_gguf_repo(fam: DiffusionFamily) -> Optional[str]:
    if (
        fam.name.startswith("ernie-image")
        or fam.name.startswith("qwen-image")
        or fam.name.startswith("z-image")
        or fam.name in {"flux.2", "flux.2-klein"}
    ):
        return _default_text_encoder_gguf_repo(fam)
    return None


def _public_diffusion_preset(preset: DiffusionLoadPreset) -> dict[str, Any]:
    fam = _family_by_name(preset.family)
    defaults = (
        _sampling_defaults_for_family(fam, base_repo_variant = preset.variant)
        if fam is not None
        else None
    )
    return {
        "id": preset.id,
        "display_name": preset.display_name,
        "family": preset.family,
        "variant": preset.variant,
        "pipeline_repo": preset.pipeline_repo,
        "transformer_gguf_repo": preset.transformer_gguf_repo,
        "transformer_filename_prefixes": list(preset.transformer_filename_prefixes),
        "recommended_offload_policy": preset.recommended_offload_policy,
        "default_steps": defaults.default_steps if defaults is not None else None,
        "default_guidance_scale": (
            defaults.default_guidance_scale if defaults is not None else None
        ),
        "default_call_kwargs": (
            dict(defaults.default_call_kwargs) if defaults is not None else {}
        ),
        "default_width": fam.default_width if fam is not None else None,
        "default_height": fam.default_height if fam is not None else None,
        "requires_image_input": bool(fam.requires_image_input) if fam is not None else False,
        "default_text_encoder_gguf_repo": (
            _preset_default_text_encoder_gguf_repo(fam) if fam is not None else None
        ),
        "default_prompt_enhancer_gguf_repo": (
            _default_prompt_enhancer_gguf_repo(fam) if fam is not None else None
        ),
    }


def curated_diffusion_presets() -> list[dict[str, Any]]:
    """Return Studio's first-class image presets.

    A preset is only a load plan: it selects the normal Diffusers
    pipeline repo and the compatible Unsloth transformer GGUF repo.
    The caller still chooses the actual GGUF filename/quant, so adding
    new quant files does not require backend code changes.
    """

    return [
        _public_diffusion_preset(preset)
        for preset in _CURATED_DIFFUSION_PRESETS
    ]


def _resolve_diffusion_preset(preset_id: str) -> DiffusionLoadPreset:
    preset = _CURATED_DIFFUSION_PRESETS_BY_ID.get(str(preset_id).lower())
    if preset is None:
        known = ", ".join(preset.id for preset in _CURATED_DIFFUSION_PRESETS)
        raise ValueError(
            f"Unknown diffusion preset_id '{preset_id}'. Known presets: {known}."
        )
    return preset


def _quant_label_to_gguf_filename(
    preset: DiffusionLoadPreset,
    quant: Optional[str],
) -> Optional[str]:
    if quant is None:
        return None
    label = str(quant).strip()
    if not label:
        return None
    if "/" in label or "\\" in label:
        raise ValueError("transformer_quant must be a quant label, not a path.")
    if label.lower().endswith(".gguf"):
        label = label[:-5]
    if label.lower().endswith(".safetensors"):
        raise ValueError("transformer_quant must name a GGUF quant, not safetensors.")
    return f"{preset.transformer_filename_prefixes[0]}{label}.gguf"


def resolve_diffusion_load_plan(
    *,
    preset_id: Optional[str] = None,
    repo_id: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    transformer_gguf_repo: Optional[str] = None,
    transformer_gguf_filename: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    base_repo: Optional[str] = None,
    text_encoder_gguf_repo: Optional[str] = None,
    text_encoder_gguf_filename: Optional[str] = None,
    text_encoder_gguf_component: Optional[str] = None,
    prompt_enhancer_gguf_repo: Optional[str] = None,
    prompt_enhancer_gguf_filename: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_weight_name: Optional[str] = None,
    lora_adapter_name: Optional[str] = None,
    lora_scale: Optional[float] = None,
    lora_fuse: bool = False,
    family_override: Optional[str] = None,
    offload_policy: Optional[str] = None,
    safetensors_quantization: Optional[str] = None,
    safetensors_quantization_components: Optional[list[str]] = None,
    require_loadable: bool = False,
) -> dict[str, Any]:
    """Expand a Studio preset into concrete DiffusionBackend kwargs.

    Direct repo loading remains generic. This helper only applies the
    curated Studio opinion when ``preset_id`` is supplied: use the
    preset's normal Diffusers pipeline repo, swap the transformer from
    the paired GGUF repo, and fill known component repos when the caller
    selected a text/prompt-enhancer GGUF filename.
    """

    if preset_id is None:
        if not repo_id:
            raise ValueError("repo_id is required when preset_id is not set.")
        normalized_safetensors_quantization = _normalize_safetensors_quantization(
            safetensors_quantization
        )
        normalized_safetensors_quantization_components = (
            _normalize_safetensors_quantization_components(
                safetensors_quantization_components
            )
        )
        if (
            normalized_safetensors_quantization
            and normalized_safetensors_quantization
            != DIFFUSION_SAFETENSORS_QUANT_NONE
            and (
                gguf_filename
                or transformer_gguf_filename
                or text_encoder_gguf_filename
                or prompt_enhancer_gguf_filename
            )
        ):
            raise ValueError(
                "safetensors_quantization is only supported for regular "
                "Diffusers safetensors repos. Omit GGUF component filenames "
                "or omit safetensors_quantization."
            )
        load_kwargs = {
            "repo_id": repo_id,
            "gguf_filename": gguf_filename,
            "transformer_gguf_repo": transformer_gguf_repo,
            "transformer_gguf_filename": transformer_gguf_filename,
            "base_repo": base_repo,
            "text_encoder_gguf_repo": text_encoder_gguf_repo,
            "text_encoder_gguf_filename": text_encoder_gguf_filename,
            "text_encoder_gguf_component": text_encoder_gguf_component,
            "prompt_enhancer_gguf_repo": prompt_enhancer_gguf_repo,
            "prompt_enhancer_gguf_filename": prompt_enhancer_gguf_filename,
            "lora_repo": lora_repo,
            "lora_weight_name": lora_weight_name,
            "lora_adapter_name": lora_adapter_name,
            "lora_scale": lora_scale,
            "lora_fuse": lora_fuse,
            "family_override": family_override,
            "offload_policy": offload_policy,
            "safetensors_quantization": normalized_safetensors_quantization,
            "safetensors_quantization_components": (
                normalized_safetensors_quantization_components
            ),
        }
        return {
            "preset": None,
            "ready_to_load": True,
            "load_kwargs": load_kwargs,
            "component_sources": {},
            "warnings": [],
        }

    preset = _resolve_diffusion_preset(preset_id)
    normalized_safetensors_quantization = _normalize_safetensors_quantization(
        safetensors_quantization
    )
    normalized_safetensors_quantization_components = (
        _normalize_safetensors_quantization_components(
            safetensors_quantization_components
        )
    )
    fam = _family_by_name(preset.family)
    if fam is None:
        raise ValueError(
            f"Diffusion preset '{preset_id}' references unknown family "
            f"'{preset.family}'."
        )

    planned_repo_id = repo_id or preset.pipeline_repo
    if base_repo and _expand_existing_local_path(base_repo) != _expand_existing_local_path(planned_repo_id):
        raise ValueError(
            "Preset loads use repo_id as the normal Diffusers pipeline repo. "
            "Omit base_repo or set it to the same repo_id."
        )

    planned_transformer_repo = transformer_gguf_repo or preset.transformer_gguf_repo
    planned_transformer_filename = (
        transformer_gguf_filename
        or gguf_filename
        or _quant_label_to_gguf_filename(preset, transformer_quant)
    )
    if require_loadable and not planned_transformer_filename:
        raise ValueError(
            "Preset diffusion loads require transformer_gguf_filename, "
            "gguf_filename, or transformer_quant."
        )
    if (
        planned_transformer_filename
        and planned_transformer_repo.lower() == preset.transformer_gguf_repo.lower()
        and not any(
            _repo_leaf(planned_transformer_filename).lower().startswith(prefix.lower())
            for prefix in preset.transformer_filename_prefixes
        )
    ):
        prefixes = ", ".join(preset.transformer_filename_prefixes)
        raise ValueError(
            f"GGUF filename '{planned_transformer_filename}' does not match "
            f"preset '{preset.id}' prefixes: {prefixes}."
        )

    planned_text_repo = text_encoder_gguf_repo
    if text_encoder_gguf_filename and not planned_text_repo:
        planned_text_repo = _default_text_encoder_gguf_repo(fam)
    planned_pe_repo = prompt_enhancer_gguf_repo
    if prompt_enhancer_gguf_filename and not planned_pe_repo:
        planned_pe_repo = _default_prompt_enhancer_gguf_repo(fam)
    if (
        normalized_safetensors_quantization
        and normalized_safetensors_quantization != DIFFUSION_SAFETENSORS_QUANT_NONE
        and (
            planned_transformer_filename
            or text_encoder_gguf_filename
            or prompt_enhancer_gguf_filename
        )
    ):
        raise ValueError(
            "safetensors_quantization is only supported for regular "
            "Diffusers safetensors repos. Studio diffusion presets currently "
            "swap GGUF components, so omit safetensors_quantization for preset "
            "loads."
        )
    defaults = _sampling_defaults_for_family(fam, base_repo_variant = preset.variant)
    effective_offload_policy = offload_policy
    if effective_offload_policy is None:
        effective_offload_policy = (
            _curated_gguf_recommended_offload_policy(
                repo_id = planned_repo_id,
                transformer_gguf_repo = planned_transformer_repo,
                transformer_gguf_filename = planned_transformer_filename,
            )
            or preset.recommended_offload_policy
        )

    load_kwargs = {
        "repo_id": planned_repo_id,
        "gguf_filename": None,
        "transformer_gguf_repo": planned_transformer_repo,
        "transformer_gguf_filename": planned_transformer_filename,
        "base_repo": None,
        "text_encoder_gguf_repo": planned_text_repo,
        "text_encoder_gguf_filename": text_encoder_gguf_filename,
        "text_encoder_gguf_component": text_encoder_gguf_component,
        "prompt_enhancer_gguf_repo": planned_pe_repo,
        "prompt_enhancer_gguf_filename": prompt_enhancer_gguf_filename,
        "lora_repo": lora_repo,
        "lora_weight_name": lora_weight_name,
        "lora_adapter_name": lora_adapter_name,
        "lora_scale": lora_scale,
        "lora_fuse": lora_fuse,
        "family_override": family_override or preset.family,
        "offload_policy": effective_offload_policy,
        "safetensors_quantization": normalized_safetensors_quantization,
        "safetensors_quantization_components": (
            normalized_safetensors_quantization_components
        ),
    }
    ready_to_load = bool(planned_transformer_filename)
    return {
        "preset": _public_diffusion_preset(preset),
        "ready_to_load": ready_to_load,
        "load_kwargs": load_kwargs,
        "component_sources": _build_diffusion_component_sources(
            pipeline_repo = planned_repo_id,
            diffusion_gguf_repo = planned_transformer_repo,
            diffusion_gguf_filename = planned_transformer_filename,
            text_encoder_gguf_repo = planned_text_repo,
            text_encoder_gguf_filename = text_encoder_gguf_filename,
            text_encoder_component = text_encoder_gguf_component,
            prompt_enhancer_gguf_repo = planned_pe_repo,
            prompt_enhancer_gguf_filename = prompt_enhancer_gguf_filename,
            lora_state = None,
        ),
        "sampling_defaults": {
            "num_inference_steps": defaults.default_steps,
            "guidance_scale": defaults.default_guidance_scale,
            "call_kwargs": dict(defaults.default_call_kwargs),
            "width": fam.default_width,
            "height": fam.default_height,
        },
        "warnings": [],
    }


def _pipe_call_default(pipe: Any, name: str) -> Any:
    import inspect

    try:
        param = inspect.signature(pipe.__call__).parameters.get(name)
    except (TypeError, ValueError):
        return None
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


def _guidance_kwarg_for_pipe(pipe: Any, fam: Optional[DiffusionFamily]) -> str:
    if fam is not None and fam.name != "diffusers":
        return fam.guidance_kwarg
    if _pipe_accepts_kwarg(pipe, "guidance_scale"):
        return "guidance_scale"
    if _pipe_accepts_kwarg(pipe, "true_cfg_scale"):
        return "true_cfg_scale"
    return "guidance_scale"


def _sampling_defaults_for_loaded_pipeline(
    pipe: Any,
    fam: Optional[DiffusionFamily],
    *,
    base_repo_variant: Optional[str] = None,
) -> DiffusionSamplingDefaults:
    if fam is not None and fam.name != "diffusers":
        return _sampling_defaults_for_family(
            fam,
            base_repo_variant = base_repo_variant,
        )

    steps = _pipe_call_default(pipe, "num_inference_steps")
    guidance_kwarg = _guidance_kwarg_for_pipe(pipe, fam)
    guidance = _pipe_call_default(pipe, guidance_kwarg)
    return DiffusionSamplingDefaults(
        default_steps = int(steps) if isinstance(steps, int) else 24,
        default_guidance_scale = (
            float(guidance)
            if isinstance(guidance, (int, float))
            else 3.5
        ),
        default_call_kwargs = {},
    )


def _guidance_semantics(
    fam: Optional[DiffusionFamily],
    *,
    is_distilled: Optional[bool],
    base_repo_variant: Optional[str],
) -> str:
    if fam is None:
        return "unknown"
    if fam.guidance_kwarg == "true_cfg_scale":
        return "true_classifier_free_guidance"
    if fam.name == "flux.2-klein":
        if is_distilled is True or (
            base_repo_variant is not None and base_repo_variant.startswith("distilled")
        ):
            return "distilled_single_pass"
        if is_distilled is False or (
            base_repo_variant is not None and base_repo_variant.startswith("base")
        ):
            return "classifier_free_guidance"
        return "flux2_klein_unknown_variant"
    if fam.default_guidance_scale <= 1.0:
        return "distilled_or_guidance_disabled"
    return "guidance_scale"


def _build_sampling_contract(
    *,
    pipe: Any,
    fam: Optional[DiffusionFamily],
    base_repo: Optional[str],
    base_repo_source: Optional[str],
    base_repo_confidence: Optional[str],
    base_repo_variant: Optional[str],
    gguf_filename: Optional[str],
) -> Optional[DiffusionSamplingContract]:
    if fam is None:
        return None
    scheduler = getattr(pipe, "scheduler", None)
    scheduler_config = getattr(scheduler, "config", None)
    is_distilled = _pipeline_is_distilled(pipe)
    defaults = _sampling_defaults_for_loaded_pipeline(
        pipe,
        fam,
        base_repo_variant = base_repo_variant,
    )
    guidance_kwarg = _guidance_kwarg_for_pipe(pipe, fam)
    return DiffusionSamplingContract(
        family = fam.name,
        media_kind = fam.media_kind,
        pipeline_class = fam.pipeline_class,
        transformer_class = fam.transformer_class,
        base_repo = base_repo,
        base_repo_source = base_repo_source,
        base_repo_confidence = base_repo_confidence,
        base_repo_variant = base_repo_variant,
        gguf = bool(gguf_filename),
        scheduler_class = _optional_class_name(scheduler),
        scheduler_config_class = _optional_class_name(scheduler_config),
        pipeline_is_distilled = is_distilled,
        guidance_kwarg = guidance_kwarg,
        default_guidance_scale = defaults.default_guidance_scale,
        default_steps = defaults.default_steps,
        guidance_semantics = _guidance_semantics(
            fam,
            is_distilled = is_distilled,
            base_repo_variant = base_repo_variant,
        ),
        default_width = int(fam.default_width),
        default_height = int(fam.default_height),
        requires_image_input = bool(fam.requires_image_input),
        has_default_negative_prompt = fam.default_negative_prompt is not None,
        default_call_kwargs = defaults.default_call_kwargs,
    )


def _apply_diffusion_lora(
    pipe: Any,
    *,
    lora_repo: str,
    lora_weight_name: Optional[str],
    lora_adapter_name: Optional[str],
    lora_scale: Optional[float],
    lora_fuse: bool,
    hf_token: Optional[str],
    gguf_filename: Optional[str],
    uses_studio_lazy_gguf_modules: bool,
) -> DiffusionLoraState:
    """Attach a Diffusers LoRA adapter to a loaded pipeline.

    This is intentionally a no-op unless the caller explicitly provides
    a LoRA repo/path, so normal image loads keep the exact same execution
    path. Unfused LoRA is attempted for both full and GGUF-backed
    pipelines via Diffusers' own loader. Fusion is restricted to
    non-GGUF loads because fusing into Studio's lazy/quantized GGUF
    modules would mutate temporary dequantized weights rather than a
    normal dense parameter tree.
    """

    if lora_fuse and gguf_filename:
        raise RuntimeError(
            "Fusing LoRA into a GGUF diffusion model is not supported. "
            "Load the adapter unfused, or use a non-GGUF Diffusers model."
        )
    if gguf_filename and uses_studio_lazy_gguf_modules:
        raise RuntimeError(
            "LoRA adapters for this GGUF diffusion model are not supported in "
            "the current low-VRAM path because Studio replaced Diffusers "
            "GGUFLinear modules with lazy quantized modules that PEFT cannot "
            "inject into. Use a non-GGUF Diffusers model for LoRA, or run a "
            "Diffusers GGUF build that keeps upstream GGUFLinear modules."
        )
    load_lora = getattr(pipe, "load_lora_weights", None)
    if not callable(load_lora):
        raise RuntimeError(
            f"{type(pipe).__name__} does not support Diffusers LoRA loading."
        )
    adapter_name = (lora_adapter_name or "default").strip() or "default"
    scale = float(1.0 if lora_scale is None else lora_scale)
    if not (0.0 <= scale <= 10.0):
        raise ValueError("lora_scale must be in [0, 10].")
    if lora_weight_name and not str(lora_weight_name).lower().endswith(".safetensors"):
        raise RuntimeError(
            "Diffusers LoRA loading in Studio only accepts safetensors "
            "weights. Choose a .safetensors LoRA file."
        )
    kwargs: dict[str, Any] = {
        "adapter_name": adapter_name,
        "use_safetensors": True,
    }
    if lora_weight_name:
        kwargs["weight_name"] = lora_weight_name
    if hf_token:
        kwargs["token"] = hf_token
    _guard_peft_optional_bitsandbytes()
    load_lora(lora_repo, **kwargs)
    set_adapters = getattr(pipe, "set_adapters", None)
    if callable(set_adapters):
        set_adapters(adapter_name, adapter_weights = scale)
    if lora_fuse:
        fuse_lora = getattr(pipe, "fuse_lora", None)
        if not callable(fuse_lora):
            raise RuntimeError(
                f"{type(pipe).__name__} does not support LoRA fusion."
            )
        fuse_lora(lora_scale = scale, adapter_names = [adapter_name])
    return DiffusionLoraState(
        repo = lora_repo,
        weight_name = lora_weight_name,
        adapter_name = adapter_name,
        scale = scale,
        fused = bool(lora_fuse),
    )


def _expand_existing_local_path(value: str) -> str:
    """Expand ``~`` in ``value`` when the expanded path exists locally.

    Round 14 P2 #11: the GGUF local path branch already calls
    ``Path(repo_id).expanduser()``, but the full-diffusers-repo and
    base-companion-repo paths passed the literal ``~/...`` straight
    into ``from_pretrained``, which treated it as a Hub id and tried
    to download. Keep behaviour identical for Hub ids (no leading
    ``~`` -> return as-is) and for non-existent expansions (the
    diffusers loader will surface its own ``not found`` error).
    """
    if not value or not isinstance(value, str) or not value.startswith("~"):
        return value
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate)
    return value


def _preflight_diffusers_subfolder_config(
    repo: str,
    subfolder: str,
    hf_token: Optional[str],
) -> None:
    """Round 21 P2 #6: also probe ``{subfolder}/config.json``.

    The full-repo preflight at ``_preflight_full_diffusers_repo``
    only proves ``model_index.json`` exists. For GGUF loads the
    follow-up ``from_single_file(..., config=effective_base,
    subfolder="transformer")`` still needs a matching
    ``transformer/config.json`` on the base companion. Without
    this second probe a base that has model_index.json but no
    transformer config would still unload chat before the load
    failed.
    """
    if not repo or not subfolder:
        return
    try:
        local = Path(repo).expanduser()
    except (OSError, ValueError):
        local = None
    if local is not None and local.exists():
        config_path = local / subfolder / "config.json"
        if not config_path.is_file():
            raise RuntimeError(
                f"Diffusion repo '{_display_repo_id(repo)}' is missing "
                f"{subfolder}/config.json."
            )
        return
    if (local is not None and local.is_absolute()) or repo.startswith("~"):
        # Local-only path that does not exist -- _preflight_full_diffusers_repo
        # already raised for the absent directory, so reaching here means the
        # caller is loading a Hub id that just looks like a path. Fall through
        # to the network probe.
        pass
    try:
        from huggingface_hub import hf_hub_download as _hf_hub_download
    except Exception:
        return
    try:
        _hf_hub_download(
            repo_id = repo,
            filename = "config.json",
            subfolder = subfolder,
            token = hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not access diffusion repo '{_display_repo_id(repo)}' "
            f"{subfolder}/config.json before unloading the current model."
        ) from exc


def _preflight_full_diffusers_repo(repo: str, hf_token: Optional[str]) -> None:
    """Prove a full diffusers repo is accessible before any unloads.

    Round 19 P1 #3: the GGUF path's ``hf_hub_download(gguf_filename)``
    above this function fails fast on a bad / private / gated /
    typo'd repo before we touch the chat backend. The full diffusers
    path used to skip that round-trip and only discover the issue
    inside ``from_pretrained`` AFTER the user's chat model was
    already unloaded. Add the same one-file probe (``model_index.json``
    is the diffusers manifest; every diffusers repo has one).

    Local paths are checked structurally so we do not hit the network
    for a missing on-disk directory; both branches raise RuntimeError
    so the surrounding load_model bails out before the chat unload.
    The display label is collapsed via ``_display_repo_id`` so an
    absolute filesystem path in the error message does not leak the
    operator's layout (see round 17 P2 #9).
    """
    if not repo:
        return
    try:
        local = Path(repo).expanduser()
    except (OSError, ValueError):
        local = None
    if local is not None and local.exists():
        if not local.is_dir():
            raise RuntimeError(
                f"Diffusion repo '{_display_repo_id(repo)}' is not a directory."
            )
        if not (local / "model_index.json").is_file():
            raise RuntimeError(
                f"Diffusion repo '{_display_repo_id(repo)}' is missing "
                "model_index.json."
            )
        return
    if (local is not None and local.is_absolute()) or repo.startswith("~"):
        raise RuntimeError(
            f"Local diffusion repo '{_display_repo_id(repo)}' does not exist."
        )
    try:
        from huggingface_hub import hf_hub_download as _hf_hub_download
    except Exception:
        # diffusers is installed but huggingface_hub is missing -- let
        # the downstream loader produce the canonical error.
        return
    try:
        _hf_hub_download(
            repo_id = repo,
            filename = "model_index.json",
            token = hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not access diffusion repo '{_display_repo_id(repo)}' "
            "before unloading the current model."
        ) from exc


def _display_repo_id(value: Any) -> Any:
    """Return a public-facing label for a repo_id / base_repo.

    For Hub-style identifiers (``owner/repo``) the value passes
    through unchanged so the Images panel and result figcaption
    stay informative. Absolute local paths (``/home/me/exports/...``
    or ``C:\\Users\\...``) collapse to the leaf name so
    ``/images/status`` does not leak the user's filesystem layout
    to other authenticated browser sessions (round 15 P2 #6). HF
    tokens are scrubbed defensively in case they slipped past the
    request-side validator.
    """
    if not isinstance(value, str) or not value:
        return value
    try:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() or candidate.exists():
            # Defense-in-depth: redact any hf_... pattern that survives
            # in the leaf name before returning it to the UI / log line.
            return _redact_hf_tokens(candidate.name or value)
    except (OSError, ValueError):
        pass
    return _redact_hf_tokens(value)


_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _redact_hf_tokens(value: Any) -> Any:
    """Scrub embedded ``hf_xxxxxxxx`` tokens out of a string before
    logging. Round 14 P2 #9: callers can wrap an authenticated URL
    (``https://hf_token@huggingface.co/...``) into ``repo_id`` /
    ``base_repo`` / paths; the token would otherwise reach
    structured-log sinks via the load-info / load-failure log lines.
    Non-strings are returned unchanged so the helper is safe to
    sprinkle through ``logger.info`` / ``logger.error`` argument
    lists.
    """
    if not isinstance(value, str):
        return value
    return _HF_TOKEN_RE.sub("<redacted>", value)


def _resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve a GGUF filename inside a local repo directory safely.

    Returns the resolved absolute path or raises ``RuntimeError`` if:
    - ``gguf_filename`` is absolute (``/etc/passwd``) or contains a
      Windows separator (``..\\..\\secret.gguf``);
    - the parts contain ``""`` / ``.`` / ``..`` (``../other.gguf``);
    - the resolved candidate escapes ``repo_root`` after symlinks /
      ``..`` collapse;
    - the resolved candidate is not a regular file.

    This is the only path that bridges a user-supplied ``gguf_filename``
    string into ``Path``s the loader opens, so confining it to the
    chosen repo here protects the delete-ownership guards downstream
    (round 13 P1 #2). ``hf_hub_download`` already enforces the same
    invariant for Hub repos.
    """
    # ``Path("/etc/passwd").is_absolute()`` is False on Windows (POSIX
    # absolute paths read as drive-relative), so check both pathlib
    # flavours plus a leading separator so the rejection is portable.
    if (
        Path(gguf_filename).is_absolute()
        or PurePosixPath(gguf_filename).is_absolute()
        or gguf_filename.startswith(("/", "\\"))
        or "\\" in gguf_filename
    ):
        raise RuntimeError("gguf_filename must be a relative file path inside repo_id.")
    rel = PurePosixPath(gguf_filename)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise RuntimeError(
            "gguf_filename must not contain empty, '.', or '..' segments."
        )
    root = repo_root.expanduser().resolve(strict = True)
    try:
        candidate = (root / Path(*rel.parts)).resolve(strict = True)
    except (OSError, FileNotFoundError) as exc:
        # strict=True raises FileNotFoundError on a missing leaf or
        # parent component, and OSError on a malformed Windows path
        # (e.g. drive letters injected through the user-supplied
        # string). Either way the candidate does not exist inside the
        # chosen repo, which is exactly the "file not in repo" failure
        # mode the caller cares about.
        raise RuntimeError(
            f"Local repo path '{repo_root}' does not contain '{gguf_filename}'."
        ) from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "gguf_filename must stay inside the local repo_id directory."
        ) from exc
    if not candidate.is_file():
        raise RuntimeError(
            f"Local repo path '{repo_root}' does not contain '{gguf_filename}'."
        )
    return candidate


def _strip_gguf_quant_suffix_for_mmproj(name: str) -> str:
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        return name[: match.start()]
    return name


def _candidate_text_encoder_mmproj_filenames(text_encoder_gguf_filename: str) -> list[str]:
    """Return likely sibling mmproj GGUF filenames for a text GGUF."""

    rel = PurePosixPath(text_encoder_gguf_filename)
    stem = _strip_gguf_quant_suffix_for_mmproj(Path(rel.name).stem)
    parent = rel.parent
    names: list[str] = []
    if stem:
        names.extend(
            [
                f"{stem}-mmproj-BF16.gguf",
                f"{stem}-mmproj.gguf",
            ]
        )
    names.extend(["mmproj-BF16.gguf", "mmproj.gguf"])

    candidates: list[str] = []
    seen: set[str] = set()
    for name in names:
        candidate = str(parent / name) if str(parent) not in ("", ".") else name
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _download_text_encoder_mmproj_from_hub(
    hf_hub_download: Any,
    *,
    repo_id: str,
    text_encoder_gguf_filename: str,
    token: Optional[str],
) -> Path | None:
    """Download a likely sibling Qwen2VL mmproj GGUF from the same Hub repo."""

    for filename in _candidate_text_encoder_mmproj_filenames(text_encoder_gguf_filename):
        with contextlib.suppress(Exception):
            return Path(
                hf_hub_download(
                    repo_id = repo_id,
                    filename = filename,
                    token = token,
                )
            )
    return None


# Negative substrings that disqualify a candidate family even when its
# name appears as a substring of the repo id. Prevents
# "stable-diffusion-3" matching SD3.5 and "qwen-image" matching
# Qwen-Image-Edit. Each entry maps a family name to substrings that
# must NOT appear anywhere in the repo id.
_FAMILY_EXCLUDE: dict[str, tuple[str, ...]] = {
    "stable-diffusion-3": (
        "3.5",
        "3-5",
        "3_5",
        "stable-diffusion-3.5",
        "stable_diffusion_3_5",
    ),
    # All underscore / hyphen spellings that appear in Hub repo ids for
    # the *-Edit family must exclude Qwen-Image, otherwise
    # ``unsloth/qwen_image_edit-GGUF`` matches the Qwen-Image base.
    "qwen-image": (
        "qwen-image-edit",
        "qwenimage-edit",
        "qwen_image_edit",
        "qwenimageedit",
    ),
}


def detect_family(
    repo_id: str, *, override_family: Optional[str] = None
) -> Optional[DiffusionFamily]:
    """Return the diffusion family matching ``repo_id``.

    Matching is substring-based and case-insensitive, with a small
    deny list (``_FAMILY_EXCLUDE``) for known false positives such as
    SD3.5 (would otherwise match SD3 Medium) and Qwen-Image-Edit
    (would otherwise match Qwen-Image). ``override_family`` bypasses
    substring matching and looks up by ``DiffusionFamily.name`` or
    (when explicitly asked) by ``_FULL_REPO_FAMILIES.name``. Returns
    ``None`` when no family applies so callers can surface a clear
    "unsupported model" error rather than guessing wrong.
    """
    if override_family:
        wanted = override_family.strip().lower()
        for fam in _FAMILIES + _FULL_REPO_FAMILIES:
            if fam.name == wanted:
                return fam
        return None
    needle = (repo_id or "").lower()
    if not needle:
        return None
    # Round 17 P2 #10: if repo_id is an absolute local path, the
    # whole path goes into ``needle`` and the _FAMILY_EXCLUDE deny
    # lists match against parent-directory names too. That means
    # ``/home/me/qwen-image-edit-cache/flux-2-klein-4b`` would be
    # excluded from the Flux family because the parent contains
    # ``qwen-image-edit``. Reduce to the leaf when the candidate
    # looks like a filesystem path so excludes only consider the
    # model directory itself.
    if "/" in needle or "\\" in needle:
        try:
            candidate = Path(repo_id).expanduser()
            if candidate.is_absolute() or candidate.exists():
                leaf = candidate.name
                if leaf:
                    needle = leaf.lower()
        except (OSError, ValueError):
            pass
    # Normalise mixed separator spellings (``Qwen_Image-Edit-GGUF``,
    # ``Qwen-Image_Edit-GGUF``, ``Qwen.Image.Edit-GGUF``) and the
    # compact concatenation (``QwenImageEdit-GGUF``) so the
    # _FAMILY_EXCLUDE deny lists do not need every permutation of
    # ``-``, ``_``, ``.`` and run-together spellings to keep
    # Qwen-Image-Edit out of the base Qwen-Image family (round 14
    # P2 #8).
    needle_norm = re.sub(r"[^a-z0-9]+", "-", needle).strip("-")
    needle_compact = re.sub(r"[^a-z0-9]+", "", needle)
    # Per-token compact strings let ``unsloth/Flux2Klein-GGUF`` match
    # the ``flux2klein`` alias: the whole-needle compact is
    # ``unslothflux2kleingguf`` and the regex boundary check rejects
    # the embedded match, but the token ``Flux2Klein`` (between the
    # ``/`` and the ``-``) compacts to exactly ``flux2klein`` (round
    # 16 P2 #9).
    needle_compact_tokens = {
        re.sub(r"[^a-z0-9]+", "", token)
        for token in re.split(r"[^a-z0-9]+", needle)
        if token
    }

    def _matches_family_token(term: str) -> bool:
        """Token-boundary match on the normalised needle. Prevents
        ``owner/flux.20-model`` from matching ``flux.2`` because
        ``flux.20`` does not have a separator after ``flux-2``
        (round 15 P2 #8). Compact spellings (``flux2klein``) match
        only when they appear as a complete repo-name token, not
        as a substring of a longer token (round 16 P2 #9)."""
        term_norm = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
        if not term_norm:
            return False
        if re.search(rf"(^|-){re.escape(term_norm)}($|-)", needle_norm):
            return True
        term_compact = re.sub(r"[^a-z0-9]+", "", term.lower())
        if not term_compact:
            return False
        return term_compact in needle_compact_tokens or term_compact == needle_compact

    # Scan _FAMILIES first (GGUF-supported), then _FULL_REPO_FAMILIES
    # so a repo like ``stabilityai/stable-diffusion-xl-base-1.0`` is
    # auto-detected as SDXL instead of returning None.
    for fam in _FAMILIES + _FULL_REPO_FAMILIES:
        excludes = _FAMILY_EXCLUDE.get(fam.name, ())
        if any(
            e in needle
            or re.sub(r"[^a-z0-9]+", "-", e).strip("-") in needle_norm
            or re.sub(r"[^a-z0-9]+", "", e) in needle_compact
            for e in excludes
        ):
            continue
        if _matches_family_token(fam.name):
            return fam
        for alias in fam.aliases:
            if alias and _matches_family_token(alias):
                return fam
    return None


def supported_families() -> list[dict[str, Any]]:
    """Public-facing list of families for ``/api/inference/images/status``."""
    return [
        {
            "name": fam.name,
            "pipeline_class": fam.pipeline_class,
            "base_repo": fam.base_repo,
            "media_kind": fam.media_kind,
            "guidance_kwarg": fam.guidance_kwarg,
            "default_steps": fam.default_steps,
            "default_guidance_scale": fam.default_guidance_scale,
            "default_width": fam.default_width,
            "default_height": fam.default_height,
            "default_num_frames": fam.default_num_frames,
            "default_frame_rate": fam.default_frame_rate,
            "requires_image_input": fam.requires_image_input,
            "supports_gguf_single_file": fam.supports_gguf_single_file,
        }
        for fam in _FAMILIES + _FULL_REPO_FAMILIES
    ]


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def supported_optimization_options() -> dict[str, Any]:
    """Public-facing optimization choices for image diffusion loads.

    Keep this data close to the backend constants so the frontend can
    render available choices without hardcoding backend-only strings.
    """

    bnb_available = _module_available("bitsandbytes")
    torchao_available = _module_available("torchao")
    mslk_available = _module_available("mslk")
    torch_compile_available = False
    try:
        import torch

        torch_compile_available = callable(getattr(torch, "compile", None))
    except Exception:
        torch_compile_available = False

    return {
        "offload_policies": [
            {
                "name": DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE,
                "media_kind": "image",
                "uses_diffusers_model_cpu_offload": True,
                "keeps_gguf_weights_cpu_resident": True,
                "recommended_for": "lowest image VRAM when throughput can drop",
            },
            {
                "name": DIFFUSION_OFFLOAD_POLICY_BALANCED,
                "media_kind": "image",
                "uses_diffusers_model_cpu_offload": False,
                "keeps_gguf_weights_cpu_resident": True,
                "recommended_for": "default GGUF image tradeoff on CUDA",
            },
            {
                "name": DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
                "media_kind": "image",
                "uses_diffusers_model_cpu_offload": False,
                "keeps_gguf_weights_cpu_resident": "text_encoder_only",
                "recommended_for": "more throughput when diffusion GGUF fits in VRAM",
            },
            {
                "name": DIFFUSION_OFFLOAD_POLICY_HYBRID,
                "media_kind": "image",
                "alias_of": DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
                "uses_diffusers_model_cpu_offload": False,
                "keeps_gguf_weights_cpu_resident": "text_encoder_only",
                "recommended_for": "compatibility alias for less_aggressive",
            },
            {
                "name": DIFFUSION_OFFLOAD_POLICY_NONE,
                "media_kind": "image",
                "uses_diffusers_model_cpu_offload": False,
                "keeps_gguf_weights_cpu_resident": False,
                "recommended_for": "highest residency when model fits in VRAM",
            },
        ],
        "safetensors_quantizations": [
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_NONE,
                "backend": None,
                "available": True,
                "requires": [],
                "default_components": None,
            },
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT,
                "backend": "bitsandbytes",
                "available": bnb_available,
                "requires": ["bitsandbytes"],
                "default_components": list(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS),
            },
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
                "backend": "bitsandbytes",
                "available": bnb_available,
                "requires": ["bitsandbytes"],
                "default_components": list(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS),
            },
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
                "backend": "bitsandbytes",
                "available": bnb_available,
                "requires": ["bitsandbytes"],
                "default_components": list(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS),
            },
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY,
                "backend": "torchao",
                "available": torchao_available,
                "requires": ["torchao"],
                "default_components": list(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS),
            },
            {
                "name": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY,
                "backend": "torchao",
                "available": torchao_available and mslk_available,
                "requires": ["torchao", "mslk"],
                "default_components": list(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS),
            },
        ],
        "safetensors_quantization_components": list(
            DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS
        ),
        "compile": {
            "torch_compile_available": torch_compile_available,
            "gguf_balanced_dequant_compile": {
                "default_enabled": True,
                "automatic_when": (
                    "CUDA load with offload_policy=balanced and CPU-resident "
                    "diffusion GGUF weights"
                ),
                "status_counter": (
                    "gguf_prepared_module_counts.diffusion_compiled_dequant_modules"
                ),
            },
            "gguf_balanced_cuda_cache": {
                "default_enabled": True,
                "automatic_when": (
                    "CUDA load with offload_policy=balanced and CPU-resident "
                    "diffusion GGUF weights"
                ),
                "env_override": "UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB",
                "free_memory_headroom_mib": BALANCED_GGUF_CUDA_CACHE_HEADROOM_MIB,
                "tiers": [
                    {
                        "min_total_vram_mib": MIN_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB,
                        "max_total_vram_mib": MID_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB,
                        "cache_budget_mib": MID_BALANCED_GGUF_CUDA_CACHE_MIB,
                    },
                    {
                        "min_total_vram_mib": MID_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB,
                        "max_total_vram_mib": HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB,
                        "cache_budget_mib": DEFAULT_BALANCED_GGUF_CUDA_CACHE_MIB,
                    },
                    {
                        "min_total_vram_mib": HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB,
                        "max_total_vram_mib": (
                            VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB
                        ),
                        "cache_budget_mib": HIGH_BALANCED_GGUF_CUDA_CACHE_MIB,
                    },
                    {
                        "min_total_vram_mib": (
                            VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB
                        ),
                        "max_total_vram_mib": None,
                        "cache_budget_mib": VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_MIB,
                    },
                ],
                "status_counters": {
                    "modules": "gguf_prepared_module_counts.diffusion_cuda_cache_modules",
                    "budget_mib": (
                        "gguf_prepared_module_counts.diffusion_cuda_cache_budget_mib"
                    ),
                    "candidate_mib": (
                        "gguf_prepared_module_counts.diffusion_cuda_cache_candidate_mib"
                    ),
                    "selected_mib": (
                        "gguf_prepared_module_counts.diffusion_cuda_cache_selected_mib"
                    ),
                },
            },
            "denoiser_torch_compile": {
                "default_enabled": False,
                "recommended_scope": "denoiser_or_repeated_blocks_only",
                "reason": (
                    "Useful for long resident sessions, but cold-start cost and "
                    "CUDA graph compatibility need per-model benchmarking."
                ),
            },
            "group_offload": {
                "image_default": False,
                "media_kind": "video",
                "reason": (
                    "Diffusers group offload is a video-oriented memory strategy; "
                    "image defaults use model CPU offload or GGUF residency policies."
                ),
            },
        },
    }


def _enable_flux2_klein_embedded_guidance(pipe: Any, fam: Optional[DiffusionFamily]) -> bool:
    """Optionally make Flux2 Klein use Flux-style single-pass guidance.

    Diffusers' Flux2KleinPipeline currently treats non-distilled Klein
    guidance_scale > 1 as classifier-free guidance, which runs the
    transformer twice per denoising step. The Flux-style path
    sends the scalar guidance into Flux2Transformer2DModel.forward as a
    conditioning embedding and keeps denoising single-pass. Keep this
    experimental path opt-in because it is not activation-equivalent to
    Diffusers' official two-pass CFG sampling.
    """

    if fam is None or fam.name != "flux.2-klein":
        return False
    if (
        os.environ.get("UNSLOTH_STUDIO_FLUX2_KLEIN_SINGLE_PASS_GUIDANCE", "")
        .strip()
        .lower()
        not in {"1", "true", "yes", "on"}
    ):
        return False
    transformer = getattr(pipe, "transformer", None)
    original_forward = getattr(transformer, "forward", None)
    if transformer is None or original_forward is None:
        return False
    if getattr(transformer, "_unsloth_flux2_klein_guidance_patched", False):
        setattr(pipe, "_unsloth_flux2_klein_embedded_guidance", True)
        return True

    def _set_distilled_flag(value: bool) -> None:
        if hasattr(pipe, "register_to_config"):
            pipe.register_to_config(is_distilled = value)
        elif hasattr(pipe, "config"):
            with contextlib.suppress(Exception):
                setattr(pipe.config, "is_distilled", value)

    def _forward_with_embedded_guidance(*args: Any, guidance: Any = None, **kwargs: Any) -> Any:
        if (
            guidance is None
            and len(args) < 6
            and getattr(pipe, "_unsloth_flux2_klein_embedded_guidance", False)
        ):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            guidance_scale = getattr(pipe, "_guidance_scale", None)
            if hidden_states is not None and guidance_scale is not None:
                import torch

                batch = int(getattr(hidden_states, "shape", [1])[0])
                guidance = torch.full(
                    [1],
                    float(guidance_scale),
                    device = hidden_states.device,
                    dtype = torch.float32,
                ).expand(batch)
        return original_forward(*args, guidance = guidance, **kwargs)

    transformer.forward = _forward_with_embedded_guidance
    setattr(transformer, "_unsloth_flux2_klein_guidance_patched", True)
    setattr(pipe, "_unsloth_flux2_klein_embedded_guidance", True)

    # Flux2KleinPipeline.do_classifier_free_guidance is derived from
    # config.is_distilled. Keep the flag false while check_inputs runs
    # so Diffusers does not emit its "guidance ignored" warning, then
    # mark this instance distilled before the denoising loop so the
    # unconditional CFG pass is skipped.
    original_check_inputs = getattr(pipe, "check_inputs", None)
    if original_check_inputs is not None and not getattr(
        pipe,
        "_unsloth_flux2_klein_check_inputs_patched",
        False,
    ):

        def _check_inputs_then_disable_cfg(*args: Any, **kwargs: Any) -> Any:
            _set_distilled_flag(False)
            try:
                return original_check_inputs(*args, **kwargs)
            finally:
                _set_distilled_flag(True)

        pipe.check_inputs = _check_inputs_then_disable_cfg
        setattr(pipe, "_unsloth_flux2_klein_check_inputs_patched", True)
        _set_distilled_flag(False)
    else:
        _set_distilled_flag(True)
    return True


def _env_flux2_klein_batched_cfg_enabled() -> bool:
    value = os.environ.get("UNSLOTH_STUDIO_FLUX2_KLEIN_BATCHED_CFG")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _enable_flux2_klein_batched_cfg(pipe: Any, fam: Optional[DiffusionFamily]) -> bool:
    """Batch Flux2 Klein CFG branches while preserving Diffusers' CFG formula."""

    if fam is None or fam.name != "flux.2-klein":
        return False
    if not _env_flux2_klein_batched_cfg_enabled():
        return False
    transformer = getattr(pipe, "transformer", None)
    original_forward = getattr(transformer, "forward", None)
    if transformer is None or original_forward is None:
        return False
    if getattr(transformer, "_unsloth_flux2_klein_batched_cfg_patched", False):
        setattr(pipe, "_unsloth_flux2_klein_batched_cfg", True)
        return True

    import torch

    pending: dict[str, Any] = {}

    def _batched_cfg_forward(*args: Any, **kwargs: Any) -> Any:
        if (
            args
            or kwargs.get("guidance", None) is not None
            or kwargs.get("return_dict", False)
            or not getattr(pipe, "do_classifier_free_guidance", False)
        ):
            pending.clear()
            return original_forward(*args, **kwargs)

        hidden_states = kwargs.get("hidden_states")
        timestep = kwargs.get("timestep")
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        txt_ids = kwargs.get("txt_ids")
        img_ids = kwargs.get("img_ids")
        required = (hidden_states, timestep, encoder_hidden_states, txt_ids, img_ids)
        if not all(hasattr(value, "shape") for value in required):
            pending.clear()
            return original_forward(*args, **kwargs)

        if not pending:
            placeholder = torch.empty_like(hidden_states)
            pending["kwargs"] = dict(kwargs)
            pending["placeholder"] = placeholder
            pending["batch"] = int(hidden_states.shape[0])
            return (placeholder,)

        previous_kwargs = pending.pop("kwargs")
        placeholder = pending.pop("placeholder")
        cond_batch = int(pending.pop("batch"))
        if int(hidden_states.shape[0]) != cond_batch:
            pending.clear()
            return original_forward(*args, **kwargs)

        batched_kwargs = dict(previous_kwargs)
        for key in ("hidden_states", "timestep", "encoder_hidden_states", "txt_ids", "img_ids"):
            batched_kwargs[key] = torch.cat([previous_kwargs[key], kwargs[key]], dim = 0)
        batched_kwargs["guidance"] = None
        batched_kwargs["return_dict"] = False
        output = original_forward(**batched_kwargs)
        if not isinstance(output, tuple) or not output:
            return output
        batched_noise = output[0]
        cond_noise, uncond_noise = batched_noise.split(
            [cond_batch, int(hidden_states.shape[0])],
            dim = 0,
        )
        placeholder.copy_(cond_noise)
        return (uncond_noise, *output[1:])

    transformer.forward = _batched_cfg_forward
    setattr(transformer, "_unsloth_flux2_klein_batched_cfg_patched", True)
    setattr(pipe, "_unsloth_flux2_klein_batched_cfg", True)
    return True


def _apply_diffusion_memory_policy(pipe: Any, offload_policy: Optional[str]) -> None:
    """Apply small per-pipeline memory knobs for Studio's VRAM presets."""

    if offload_policy != DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE:
        return
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    for method_name in ("enable_slicing", "enable_tiling"):
        method = getattr(vae, method_name, None)
        if method is not None:
            method()


def _clone_prompt_embeds_to_device(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [
            item.to(device, non_blocking = True) if hasattr(item, "to") else item
            for item in value
        ]
    if hasattr(value, "to"):
        return value.to(device, non_blocking = True)
    return value


def _store_prompt_embeds_on_cpu(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_store_prompt_embeds_on_cpu(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "to"):
        return value.detach().to("cpu", copy = True)
    return value


def _primary_prompt_embeds(value: Any) -> Any:
    """Unwrap Diffusers encode_prompt auxiliary outputs.

    Some pipelines return only prompt embeddings, while Flux2/Klein returns
    ``(prompt_embeds, text_ids)``. The pipeline call accepts only
    ``prompt_embeds`` and recomputes text ids internally, so cache the tensor
    payload rather than the auxiliary tuple.
    """
    if isinstance(value, tuple) and value and hasattr(value[0], "shape"):
        return value[0]
    return value


def _prompt_embeds_and_optional_mask(
    value: Any,
    *,
    accepts_mask: bool,
) -> tuple[Any, Any]:
    if (
        accepts_mask
        and isinstance(value, tuple)
        and len(value) >= 2
        and hasattr(value[0], "shape")
    ):
        prompt_embeds = value[0]
        prompt_embeds_mask = value[1]
    else:
        prompt_embeds = _primary_prompt_embeds(value)
        prompt_embeds_mask = None
    if accepts_mask and prompt_embeds_mask is None and hasattr(prompt_embeds, "shape"):
        try:
            import torch

            if len(prompt_embeds.shape) >= 2:
                prompt_embeds_mask = torch.ones(
                    (int(prompt_embeds.shape[0]), int(prompt_embeds.shape[1])),
                    dtype = torch.long,
                    device = prompt_embeds.device,
                )
        except Exception:
            prompt_embeds_mask = None
    return prompt_embeds, prompt_embeds_mask


def _env_pin_cpu_resident_gguf() -> bool:
    return (
        os.environ.get("UNSLOTH_STUDIO_GGUF_PIN_CPU_RESIDENT", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _balanced_gguf_cuda_cache_bytes(
    *,
    device: str | None = "cuda",
    free_bytes: int | None = None,
    total_bytes: int | None = None,
) -> int:
    value = os.environ.get("UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB")
    if value is None or not value.strip():
        if device != "cuda":
            return 0
        if free_bytes is None or total_bytes is None:
            try:
                import torch

                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                return 0
        free_mib = int(free_bytes) // (1024 * 1024)
        total_mib = int(total_bytes) // (1024 * 1024)
        if total_mib < MIN_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB:
            return 0
        if total_mib >= VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB:
            cap_mib = VERY_HIGH_BALANCED_GGUF_CUDA_CACHE_MIB
        elif total_mib >= HIGH_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB:
            cap_mib = HIGH_BALANCED_GGUF_CUDA_CACHE_MIB
        elif total_mib >= MID_BALANCED_GGUF_CUDA_CACHE_TOTAL_MIB:
            cap_mib = DEFAULT_BALANCED_GGUF_CUDA_CACHE_MIB
        else:
            cap_mib = MID_BALANCED_GGUF_CUDA_CACHE_MIB
        budget_mib = min(
            cap_mib,
            max(0, free_mib - BALANCED_GGUF_CUDA_CACHE_HEADROOM_MIB),
        )
        return budget_mib * 1024 * 1024
    try:
        mib = int(value.strip())
    except ValueError:
        logger.warning(
            "Ignoring invalid UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB=%r; disabling "
            "the optional balanced GGUF CUDA cache.",
            value,
        )
        mib = 0
    return max(0, mib) * 1024 * 1024


def _guard_diffusers_optional_bitsandbytes() -> None:
    """Keep broken optional bitsandbytes installs from blocking diffusion.

    Diffusers imports its bitsandbytes quantizer module while importing
    generic model/LoRA utilities, even for dense and GGUF-only loads. A
    mismatched local bitsandbytes wheel can therefore make every image
    pipeline fail at import time. If importing bitsandbytes itself fails,
    provide a tiny Diffusers quantizer-module stub so non-BnB loads keep
    working; actual BnB quantizer use still raises a direct error.
    """

    if "diffusers.quantizers.bitsandbytes" in sys.modules:
        return
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            import bitsandbytes  # noqa: F401
        return
    except Exception as exc:
        bnb_error = exc

    def _raise_unavailable_bnb(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Diffusers bitsandbytes quantization is unavailable because "
            f"bitsandbytes failed to import: {bnb_error}"
        )

    class _UnavailableBnBDiffusersQuantizer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _raise_unavailable_bnb()

    stub = types.ModuleType("diffusers.quantizers.bitsandbytes")
    stub.__path__ = []
    stub.BnB4BitDiffusersQuantizer = _UnavailableBnBDiffusersQuantizer
    stub.BnB8BitDiffusersQuantizer = _UnavailableBnBDiffusersQuantizer
    utils_stub = types.ModuleType("diffusers.quantizers.bitsandbytes.utils")
    utils_stub.dequantize_and_replace = _raise_unavailable_bnb
    utils_stub.dequantize_bnb_weight = _raise_unavailable_bnb
    utils_stub.replace_with_bnb_linear = _raise_unavailable_bnb
    utils_stub._check_bnb_status = lambda _module: (False, False, False)
    bnb_quantizer_stub = types.ModuleType(
        "diffusers.quantizers.bitsandbytes.bnb_quantizer"
    )
    bnb_quantizer_stub.BnB4BitDiffusersQuantizer = _UnavailableBnBDiffusersQuantizer
    bnb_quantizer_stub.BnB8BitDiffusersQuantizer = _UnavailableBnBDiffusersQuantizer
    sys.modules["diffusers.quantizers.bitsandbytes"] = stub
    sys.modules["diffusers.quantizers.bitsandbytes.utils"] = utils_stub
    sys.modules[
        "diffusers.quantizers.bitsandbytes.bnb_quantizer"
    ] = bnb_quantizer_stub


def _require_bitsandbytes_for_safetensors_quantization() -> None:
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Diffusers bitsandbytes safetensors quantization is unavailable "
            f"because bitsandbytes failed to import: {exc}"
        ) from exc


def _guard_peft_optional_bitsandbytes() -> None:
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            import bitsandbytes  # noqa: F401
        return
    except Exception:
        pass

    try:
        import peft.import_utils as peft_import_utils
    except Exception:
        return

    peft_import_utils.is_bnb_available = lambda: False
    peft_import_utils.is_bnb_4bit_available = lambda: False
    for module_name in (
        "peft.tuners.lora.model",
        "peft.tuners.adalora.model",
        "peft.tuners.ia3.model",
        "peft.tuners.oft.model",
        "peft.tuners.vera.model",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "is_bnb_available", lambda: False)
            setattr(module, "is_bnb_4bit_available", lambda: False)


def _normalize_safetensors_quantization(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": None,
        "false": None,
        "off": None,
        "none": DIFFUSION_SAFETENSORS_QUANT_NONE,
        "bnb4": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
        "bnb_4bit": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
        "bitsandbytes4bit": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
        "bitsandbytes_4bit": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT,
        "bitsandbytes_4bit_nf4": DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
        "bnb8": DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
        "bnb_8bit": DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
        "bitsandbytes8bit": DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
        "bitsandbytes_8bit": DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
        "torchao_int8": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY,
        "torchao_int8_weight_only": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY,
        "torchao_int4": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY,
        "torchao_int4_weight_only": DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized is None:
        return None
    if normalized not in DIFFUSION_SAFETENSORS_QUANTS:
        allowed = ", ".join(sorted(DIFFUSION_SAFETENSORS_QUANTS))
        raise ValueError(f"safetensors_quantization must be one of: {allowed}")
    return normalized


def _normalize_safetensors_quantization_components(
    components: Optional[list[str] | tuple[str, ...]],
) -> Optional[list[str]]:
    if components is None:
        return None
    normalized: list[str] = []
    for component in components:
        item = str(component).strip()
        if not item:
            continue
        if item not in DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS:
            allowed = ", ".join(DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS)
            raise ValueError(
                "safetensors_quantization_components entries must be one of: "
                f"{allowed}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized or None


def _safetensors_quantization_components(
    components: Optional[list[str] | tuple[str, ...]],
) -> list[str]:
    return list(
        _normalize_safetensors_quantization_components(components)
        or DIFFUSION_SAFETENSORS_QUANT_DEFAULT_COMPONENTS
    )


def _torchao_weight_only_config(quantization: str) -> Any:
    if quantization == DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY:
        from torchao.quantization import Int8WeightOnlyConfig

        return Int8WeightOnlyConfig()
    if quantization == DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY:
        if importlib.util.find_spec("mslk") is None:
            raise RuntimeError(
                "TorchAO int4 safetensors quantization requires mslk >= 1.0.0. "
                "Install mslk or use torchao_int8_weight_only / bitsandbytes_4bit_nf4."
            )
        from torchao.quantization import Int4WeightOnlyConfig

        return Int4WeightOnlyConfig()
    raise ValueError(f"Unsupported torchao safetensors quantization: {quantization}")


def _build_safetensors_pipeline_quantization_config(
    diffusers_mod: Any,
    quantization: Optional[str],
    components: Optional[list[str] | tuple[str, ...]],
    dtype: Any,
) -> tuple[Any, Optional[str], Optional[list[str]]]:
    """Build a Diffusers pipeline-level quantization config for full repos."""

    normalized = _normalize_safetensors_quantization(quantization)
    if normalized is None or normalized == DIFFUSION_SAFETENSORS_QUANT_NONE:
        return None, normalized, None

    selected_components = _safetensors_quantization_components(components)
    pipeline_quant_config_cls = getattr(diffusers_mod, "PipelineQuantizationConfig", None)
    if pipeline_quant_config_cls is None:
        try:
            from diffusers.quantizers import PipelineQuantizationConfig
        except Exception as exc:
            raise RuntimeError(
                "Diffusers pipeline quantization is unavailable in this runtime. "
                "Upgrade the Studio torch runtime before loading a quantized "
                "safetensors diffusion model."
            ) from exc
        pipeline_quant_config_cls = PipelineQuantizationConfig

    if normalized in {
        DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT,
        DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4,
        DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT,
    }:
        _require_bitsandbytes_for_safetensors_quantization()
        if normalized == DIFFUSION_SAFETENSORS_QUANT_BNB_8BIT:
            return (
                pipeline_quant_config_cls(
                    quant_backend = "bitsandbytes_8bit",
                    quant_kwargs = {"load_in_8bit": True},
                    components_to_quantize = selected_components,
                ),
                normalized,
                selected_components,
            )
        quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": dtype,
        }
        if normalized == DIFFUSION_SAFETENSORS_QUANT_BNB_4BIT_NF4:
            quant_kwargs["bnb_4bit_quant_type"] = "nf4"
        return (
            pipeline_quant_config_cls(
                quant_backend = "bitsandbytes_4bit",
                quant_kwargs = quant_kwargs,
                components_to_quantize = selected_components,
            ),
            normalized,
            selected_components,
        )

    if normalized in {
        DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT8_WEIGHT_ONLY,
        DIFFUSION_SAFETENSORS_QUANT_TORCHAO_INT4_WEIGHT_ONLY,
    }:
        try:
            from transformers import TorchAoConfig as TransformersTorchAoConfig
        except Exception as exc:
            raise RuntimeError(
                "TorchAO safetensors quantization needs transformers with "
                "TorchAoConfig support."
            ) from exc
        diffusers_torchao_config_cls = getattr(diffusers_mod, "TorchAoConfig", None)
        if diffusers_torchao_config_cls is None:
            try:
                from diffusers import TorchAoConfig as DiffusersTorchAoConfig
            except Exception as exc:
                raise RuntimeError(
                    "TorchAO safetensors quantization needs diffusers with "
                    "TorchAoConfig support."
                ) from exc
            diffusers_torchao_config_cls = DiffusersTorchAoConfig

        quant_mapping: dict[str, Any] = {}
        for component in selected_components:
            if component in DIFFUSION_DIFFUSERS_QUANT_COMPONENTS:
                quant_mapping[component] = diffusers_torchao_config_cls(
                    _torchao_weight_only_config(normalized)
                )
            elif component in DIFFUSION_TRANSFORMERS_QUANT_COMPONENTS:
                quant_mapping[component] = TransformersTorchAoConfig(
                    _torchao_weight_only_config(normalized)
                )
        return (
            pipeline_quant_config_cls(quant_mapping = quant_mapping),
            normalized,
            selected_components,
        )

    raise ValueError(f"Unsupported safetensors quantization: {normalized}")


def _guard_transformers_tokenizers_backend() -> None:
    """Provide ERNIE compatibility shims for current Transformers builds."""

    try:
        import transformers
        from transformers import PreTrainedTokenizerFast
    except Exception:
        return
    if "TokenizersBackend" not in getattr(transformers, "__dict__", {}):

        class TokenizersBackend(PreTrainedTokenizerFast):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                extra_special_tokens = kwargs.pop("extra_special_tokens", None)
                if isinstance(extra_special_tokens, list):
                    existing = list(kwargs.get("additional_special_tokens") or [])
                    kwargs["additional_special_tokens"] = list(
                        dict.fromkeys(existing + extra_special_tokens)
                    )
                elif extra_special_tokens is not None:
                    kwargs["extra_special_tokens"] = extra_special_tokens
                super().__init__(*args, **kwargs)

        TokenizersBackend.__name__ = "TokenizersBackend"
        TokenizersBackend.__qualname__ = "TokenizersBackend"
        TokenizersBackend.__module__ = "transformers"
        transformers.TokenizersBackend = TokenizersBackend
    try:
        from transformers import MinistralConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        CONFIG_MAPPING["ministral3"]
    except KeyError:
        CONFIG_MAPPING.register("ministral3", MinistralConfig)
    except Exception:
        return


def _transformers_tokenizers_backend_cls() -> Any:
    _guard_transformers_tokenizers_backend()
    import transformers

    tokenizer_cls = getattr(transformers, "__dict__", {}).get("TokenizersBackend")
    if tokenizer_cls is None:
        raise RuntimeError(
            "ERNIE tokenizers require a Transformers TokenizersBackend "
            "compatibility class, but Studio could not install the fallback."
        )
    return tokenizer_cls


def _normalize_diffusion_offload_policy(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "legacy", "auto"}:
        return None
    if normalized not in DIFFUSION_OFFLOAD_POLICIES:
        allowed = ", ".join(sorted(DIFFUSION_OFFLOAD_POLICIES))
        raise ValueError(f"offload_policy must be one of: {allowed}")
    return normalized


def _resolve_diffusion_offload_policy(
    *,
    offload_policy: Optional[str],
    enable_model_cpu_offload: bool,
    gguf_quantized_cpu_resident: Optional[bool],
    gguf_pin_cpu_resident: Optional[bool],
) -> tuple[Optional[str], bool, bool, bool]:
    """Resolve Studio's user-facing VRAM policy to backend switches."""

    normalized = _normalize_diffusion_offload_policy(offload_policy)
    if normalized == DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE:
        return normalized, True, True, True
    if normalized == DIFFUSION_OFFLOAD_POLICY_BALANCED:
        return normalized, False, True, True
    if normalized in {
        DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
        DIFFUSION_OFFLOAD_POLICY_HYBRID,
    }:
        return normalized, False, False, True
    if normalized == DIFFUSION_OFFLOAD_POLICY_NONE:
        return normalized, False, False, False

    if gguf_quantized_cpu_resident is None:
        gguf_quantized_cpu_resident = bool(enable_model_cpu_offload)
    if gguf_pin_cpu_resident is None:
        gguf_pin_cpu_resident = bool(
            gguf_quantized_cpu_resident
            and enable_model_cpu_offload
        ) or _env_pin_cpu_resident_gguf()
    return (
        None,
        bool(enable_model_cpu_offload),
        bool(gguf_quantized_cpu_resident),
        bool(gguf_pin_cpu_resident),
    )


def _component_source_repo(repo: Optional[str]) -> Optional[str]:
    return _display_repo_id(repo) if repo else None


def _build_diffusion_component_sources(
    *,
    pipeline_repo: str,
    diffusion_gguf_repo: Optional[str],
    diffusion_gguf_filename: Optional[str],
    text_encoder_gguf_repo: Optional[str],
    text_encoder_gguf_filename: Optional[str],
    text_encoder_component: Optional[str],
    prompt_enhancer_gguf_repo: Optional[str],
    prompt_enhancer_gguf_filename: Optional[str],
    lora_state: Optional[DiffusionLoraState],
) -> dict[str, Any]:
    """Describe which repo owns each loaded pipeline component."""

    sources: dict[str, Any] = {
        "pipeline": {
            "source": "repo",
            "repo": _component_source_repo(pipeline_repo),
        },
        "scheduler": {
            "source": "pipeline_repo",
            "repo": _component_source_repo(pipeline_repo),
        },
        "vae": {
            "source": "pipeline_repo",
            "repo": _component_source_repo(pipeline_repo),
        },
        "tokenizers": {
            "source": "pipeline_repo",
            "repo": _component_source_repo(pipeline_repo),
        },
        "transformer": {
            "source": "pipeline_repo",
            "repo": _component_source_repo(pipeline_repo),
        },
    }
    if diffusion_gguf_filename:
        sources["transformer"] = {
            "source": "gguf",
            "repo": _component_source_repo(diffusion_gguf_repo),
            "filename": Path(diffusion_gguf_filename).name,
        }
    if text_encoder_gguf_filename:
        component = text_encoder_component or "text_encoder"
        sources[component] = {
            "source": "gguf",
            "repo": _component_source_repo(text_encoder_gguf_repo),
            "filename": Path(text_encoder_gguf_filename).name,
        }
    if prompt_enhancer_gguf_filename:
        sources["pe"] = {
            "source": "gguf",
            "repo": _component_source_repo(prompt_enhancer_gguf_repo),
            "filename": Path(prompt_enhancer_gguf_filename).name,
        }
    if lora_state is not None:
        sources["lora"] = lora_state.as_public_dict()
    return sources


# ─── Backend ──────────────────────────────────────────────────────────


class DiffusionBackend:
    """Singleton-style diffusion backend.

    One pipeline at a time; ``load_model`` swaps the previous one out.
    Generation is mutex'd so concurrent requests serialise rather than
    racing GPU memory.
    """

    def __init__(self) -> None:
        self._pipe: Any = None
        # `_lock` protects mutations to the small state fields and is
        # the only lock taken by status(). It is intentionally NOT held
        # for the long pipeline forward pass: holding it for the whole
        # generate would block status() polls (frontend at 1 Hz) and
        # any concurrent unload requests for minutes at a time.
        #
        # `_load_lock` serialises the entire load_model call so two
        # concurrent /images/load requests cannot both reach
        # pipeline_cls.from_pretrained at the same time (which would
        # double-spend VRAM and corrupt _pipe).
        #
        # `_generate_lock` serialises pipeline __call__ since diffusers
        # pipelines are not thread-safe; overlapping forwards on the
        # shared pipe corrupt internal scheduler state.
        #
        # Lock order is load -> state and generate -> state (never
        # state -> load/generate) so a forward in flight cannot
        # deadlock the next load or a status poll.
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._family: Optional[DiffusionFamily] = None
        self._repo_id: Optional[str] = None
        self._diffusion_gguf_repo: Optional[str] = None
        self._gguf_path: Optional[str] = None
        # Original ``gguf_filename`` the caller passed in, preserved
        # so delete guards can compare against subdirectory variants
        # like ``BF16/model.gguf`` or ``Q4_K_M/model.gguf`` instead
        # of the collapsed basename (round 14 P1 #4). The basename
        # alone (``model.gguf``) loses the quant directory and lets
        # /delete-cached unlink the wrong file.
        self._gguf_filename: Optional[str] = None
        self._text_encoder_gguf_repo: Optional[str] = None
        self._text_encoder_gguf_path: Optional[str] = None
        self._text_encoder_gguf_filename: Optional[str] = None
        self._prompt_enhancer_gguf_repo: Optional[str] = None
        self._prompt_enhancer_gguf_path: Optional[str] = None
        self._prompt_enhancer_gguf_filename: Optional[str] = None
        self._lora_state: Optional[DiffusionLoraState] = None
        self._component_sources: dict[str, Any] = {}
        self._base_repo: Optional[str] = None
        self._base_repo_source: Optional[str] = None
        self._base_repo_confidence: Optional[str] = None
        self._base_repo_variant: Optional[str] = None
        self._base_repo_warning: Optional[str] = None
        self._sampling_contract: Optional[DiffusionSamplingContract] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        # True when ``enable_model_cpu_offload()`` was applied on the
        # loaded pipeline. Diffusers' offload moves the active
        # submodule between CPU and GPU on each step, so a CUDA
        # ``torch.Generator`` mismatches the CPU-resident embeddings
        # and generation crashes mid-forward (round 14 P1 #6). When
        # this is True, seeded generation has to use a CPU generator
        # regardless of self._device.
        self._cpu_offload_enabled: bool = False
        self._offload_policy: Optional[str] = None
        self._gguf_quantized_cpu_resident: bool = False
        self._gguf_pin_cpu_resident: bool = False
        self._gguf_execution_backend: Optional[str] = None
        self._gguf_prepared_module_counts: dict[str, int] = {}
        self._safetensors_quantization: Optional[str] = None
        self._safetensors_quantization_components: Optional[list[str]] = None
        self._load_timings: dict[str, float] = {}
        self._prompt_embedding_cache_key: Optional[tuple[Any, ...]] = None
        self._prompt_embedding_cache_value: Optional[
            tuple[Any, Any, Any, Any]
        ] = None
        self._loaded_at: Optional[float] = None
        self._loading: bool = False
        self._last_error: Optional[str] = None
        # `_pending_*` fields advertise the target of an in-flight load
        # so cache- and finetuned-delete guards can refuse to rmtree a
        # repo while it is being downloaded / read. They are set under
        # _lock at the start of load_model and cleared on success or
        # in the finally block. The route layer reads them via
        # status() under _lock.
        self._pending_repo_id: Optional[str] = None
        self._pending_diffusion_gguf_repo: Optional[str] = None
        self._pending_base_repo: Optional[str] = None
        self._pending_base_repo_source: Optional[str] = None
        self._pending_base_repo_confidence: Optional[str] = None
        self._pending_base_repo_variant: Optional[str] = None
        self._pending_base_repo_warning: Optional[str] = None
        self._pending_gguf_filename: Optional[str] = None
        self._pending_text_encoder_gguf_repo: Optional[str] = None
        self._pending_text_encoder_gguf_filename: Optional[str] = None
        self._pending_prompt_enhancer_gguf_repo: Optional[str] = None
        self._pending_prompt_enhancer_gguf_filename: Optional[str] = None
        self._pending_lora_repo: Optional[str] = None
        self._pending_lora_weight_name: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def repo_id(self) -> Optional[str]:
        return self._repo_id

    def status(self, *, include_internal: bool = False) -> dict[str, Any]:
        # Take _lock so the snapshot cannot observe a torn state where
        # _pipe was already swapped but _family/_repo_id haven't been
        # updated yet (or vice versa). Frontend polling at 1 Hz would
        # otherwise render impossible "loaded but no repo_id" states.
        # Only echo the GGUF basename; full absolute path leaks the
        # local HF cache layout (and the system username on default
        # POSIX layouts) to any authenticated Studio session.
        #
        # Round 16 P1 #5: the guard-facing ``active_*`` / ``pending_*``
        # fields hold the EXACT raw path (so /delete-cached can match
        # an HF snapshot mmap) but are NOT safe to surface to the
        # browser. Callers that need the raw path (route-internal
        # delete guards) pass ``include_internal=True``; the public
        # ``/api/inference/images/status`` route always uses the
        # public payload.
        with self._lock:
            # Expose BOTH the resident pipeline's id AND the pending
            # load target. Delete guards must check both: when model A
            # is already loaded and a swap to model B is in flight,
            # only checking one would let the user rmtree whichever
            # repo the guard ignored. UI-facing ``repo_id`` /
            # ``base_repo`` / ``gguf_filename`` still prefer pending
            # during a swap so the panel shows the load target the
            # user just clicked.
            active_repo = self._repo_id
            active_diffusion_gguf_repo = self._diffusion_gguf_repo
            active_base = self._base_repo
            active_base_source = self._base_repo_source
            active_base_confidence = self._base_repo_confidence
            active_base_variant = self._base_repo_variant
            active_base_warning = self._base_repo_warning
            active_sampling_contract = self._sampling_contract
            active_gguf = self._gguf_filename
            active_te_repo = self._text_encoder_gguf_repo
            active_te_gguf = self._text_encoder_gguf_filename
            active_pe_repo = self._prompt_enhancer_gguf_repo
            active_pe_gguf = self._prompt_enhancer_gguf_filename
            active_lora_state = self._lora_state
            active_component_sources = dict(self._component_sources)
            pending_repo = self._pending_repo_id if self._loading else None
            pending_diffusion_gguf_repo = (
                self._pending_diffusion_gguf_repo if self._loading else None
            )
            pending_base = self._pending_base_repo if self._loading else None
            pending_base_source = self._pending_base_repo_source if self._loading else None
            pending_base_confidence = self._pending_base_repo_confidence if self._loading else None
            pending_base_variant = self._pending_base_repo_variant if self._loading else None
            pending_base_warning = self._pending_base_repo_warning if self._loading else None
            pending_gguf = self._pending_gguf_filename if self._loading else None
            pending_te_repo = self._pending_text_encoder_gguf_repo if self._loading else None
            pending_te_gguf = self._pending_text_encoder_gguf_filename if self._loading else None
            pending_pe_repo = self._pending_prompt_enhancer_gguf_repo if self._loading else None
            pending_pe_gguf = self._pending_prompt_enhancer_gguf_filename if self._loading else None
            pending_lora_repo = self._pending_lora_repo if self._loading else None
            pending_lora_weight_name = self._pending_lora_weight_name if self._loading else None
            # When a swap is in flight, the UI-facing repo_id /
            # base_repo / gguf_filename advertise the PENDING model
            # but ``self._family`` still points at the previously
            # loaded pipeline. Reporting them together produces a
            # repo/family pair that never existed (round 11 #6).
            # Null the family / pipeline_class while a swap is in
            # flight; the frontend can fall back to "unknown".
            ui_family = self._family.name if self._family else None
            ui_pipeline_class = self._family.pipeline_class if self._family else None
            ui_media_kind = self._family.media_kind if self._family else None
            if pending_repo and pending_repo != active_repo:
                ui_family = None
                ui_pipeline_class = None
                ui_media_kind = None
            # UI-facing ``gguf_filename`` collapses to the basename
            # so the Images panel does not surface internal cache /
            # variant directory names. Guard-facing ``active_*`` /
            # ``pending_*`` retain the full caller-supplied filename
            # so /delete-cached can compare against subdirectory
            # variants like ``BF16/model.gguf`` (round 14 P1 #4-5).
            ui_gguf = pending_gguf or active_gguf
            ui_gguf_basename = Path(ui_gguf).name if ui_gguf else None
            ui_te_gguf = pending_te_gguf or active_te_gguf
            ui_te_gguf_basename = Path(ui_te_gguf).name if ui_te_gguf else None
            ui_pe_gguf = pending_pe_gguf or active_pe_gguf
            ui_pe_gguf_basename = Path(ui_pe_gguf).name if ui_pe_gguf else None
            active_lora_public = (
                active_lora_state.as_public_dict()
                if active_lora_state is not None
                else None
            )
            if pending_lora_repo:
                active_lora_public = {
                    "repo": _display_repo_id(pending_lora_repo),
                    "weight_name": pending_lora_weight_name,
                    "adapter_name": None,
                    "scale": None,
                    "fused": None,
                }
            # UI-facing ``repo_id`` / ``base_repo`` collapse absolute
            # local paths to their leaf name so ``/images/status``
            # does not leak the user's filesystem layout to other
            # authenticated browser sessions (round 15 P2 #6). The
            # guard-facing ``active_*`` / ``pending_*`` fields below
            # preserve the exact value so delete guards still match
            # against the snapshot path.
            payload: dict[str, Any] = {
                "is_loaded": self._pipe is not None,
                "is_loading": self._loading,
                "repo_id": _display_repo_id(pending_repo or active_repo),
                "pipeline_repo": _display_repo_id(pending_base or active_base),
                "family": ui_family,
                "pipeline_class": ui_pipeline_class,
                "media_kind": ui_media_kind,
                "base_repo": _display_repo_id(pending_base or active_base),
                "base_repo_source": pending_base_source or active_base_source,
                "base_repo_confidence": pending_base_confidence or active_base_confidence,
                "base_repo_variant": pending_base_variant or active_base_variant,
                "base_repo_warning": pending_base_warning or active_base_warning,
                "sampling_contract": (
                    active_sampling_contract.as_public_dict()
                    if active_sampling_contract is not None and not pending_repo
                    else None
                ),
                "gguf_filename": ui_gguf_basename,
                "transformer_gguf_repo": _display_repo_id(
                    pending_diffusion_gguf_repo or active_diffusion_gguf_repo
                ),
                "transformer_gguf_filename": ui_gguf_basename,
                "text_encoder_gguf_repo": _display_repo_id(pending_te_repo or active_te_repo),
                "text_encoder_gguf_filename": ui_te_gguf_basename,
                "prompt_enhancer_gguf_repo": _display_repo_id(pending_pe_repo or active_pe_repo),
                "prompt_enhancer_gguf_filename": ui_pe_gguf_basename,
                "lora": active_lora_public,
                "component_sources": (
                    active_component_sources
                    if active_component_sources and not pending_repo
                    else None
                ),
                "gguf_quantized_cpu_resident": self._gguf_quantized_cpu_resident,
                "gguf_pin_cpu_resident": self._gguf_pin_cpu_resident,
                "offload_policy": self._offload_policy,
                "gguf_execution_backend": self._gguf_execution_backend,
                "gguf_prepared_module_counts": dict(self._gguf_prepared_module_counts),
                "safetensors_quantization": self._safetensors_quantization,
                "safetensors_quantization_components": (
                    list(self._safetensors_quantization_components)
                    if self._safetensors_quantization_components is not None
                    else None
                ),
                "load_timings": dict(self._load_timings),
                "device": self._device,
                "dtype": self._dtype,
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "supported_families": supported_families(),
                "optimization_options": supported_optimization_options(),
            }
            if include_internal:
                # Guard-facing fields: every repo / path / GGUF
                # filename the backend owns RIGHT NOW. Delete routes
                # iterate both, paired so the variant-filename check
                # is compared against the SAME repo that owns it
                # (round 13 P1 #3-5). Round 16 P1 #5: never returned
                # by the public /images/status route.
                payload.update(
                    {
                        "active_repo_id": active_repo,
                        "active_diffusion_gguf_repo": active_diffusion_gguf_repo,
                        "active_base_repo": active_base,
                        "active_base_repo_source": active_base_source,
                        "active_base_repo_confidence": active_base_confidence,
                        "active_base_repo_variant": active_base_variant,
                        "active_base_repo_warning": active_base_warning,
                        "active_gguf_filename": active_gguf,
                        "active_text_encoder_gguf_repo": active_te_repo,
                        "active_text_encoder_gguf_filename": active_te_gguf,
                        "active_prompt_enhancer_gguf_repo": active_pe_repo,
                        "active_prompt_enhancer_gguf_filename": active_pe_gguf,
                        "active_lora_repo": active_lora_state.repo if active_lora_state else None,
                        "active_lora_weight_name": (
                            active_lora_state.weight_name if active_lora_state else None
                        ),
                        "pending_repo_id": pending_repo,
                        "pending_diffusion_gguf_repo": pending_diffusion_gguf_repo,
                        "pending_base_repo": pending_base,
                        "pending_base_repo_source": pending_base_source,
                        "pending_base_repo_confidence": pending_base_confidence,
                        "pending_base_repo_variant": pending_base_variant,
                        "pending_base_repo_warning": pending_base_warning,
                        "pending_gguf_filename": pending_gguf,
                        "pending_text_encoder_gguf_repo": pending_te_repo,
                        "pending_text_encoder_gguf_filename": pending_te_gguf,
                        "pending_prompt_enhancer_gguf_repo": pending_pe_repo,
                        "pending_prompt_enhancer_gguf_filename": pending_pe_gguf,
                        "pending_lora_repo": pending_lora_repo,
                        "pending_lora_weight_name": pending_lora_weight_name,
                    }
                )
            return payload

    def generation_defaults(self) -> dict[str, Any]:
        """Return generation defaults for the currently loaded family."""
        with self._lock:
            fam = self._family
            pipe = self._pipe
            base_repo_variant = self._base_repo_variant
        defaults = (
            _sampling_defaults_for_loaded_pipeline(
                pipe,
                fam,
                base_repo_variant = base_repo_variant,
            )
            if pipe is not None
            else None
        )
        return {
            "num_inference_steps": (
                defaults.default_steps if defaults is not None else 24
            ),
            "guidance_scale": (
                defaults.default_guidance_scale if defaults is not None else 3.5
            ),
            "width": fam.default_width if fam is not None else 1024,
            "height": fam.default_height if fam is not None else 1024,
        }

    def _pick_device_and_dtype(self) -> tuple[str, "Any"]:
        """Pick (device, dtype) for the current host.

        CUDA-first because that is the only path our diffusion GGUFs are
        validated on. On macOS we use MPS in float16 to keep the pipeline
        on the Metal GPU. CPU is allowed only as a last resort because
        running FLUX on CPU is unusably slow (> 10 minutes per image).

        BF16 is gated on ``torch.cuda.is_bf16_supported`` because the
        Pascal / Turing class (sm_60 / sm_70 / sm_75) reports
        ``is_available() == True`` but lacks BF16 ALUs; FLUX kernels
        then fail inside ``from_pretrained`` or at the first denoise
        step. Those cards still work on FP16, so fall back rather than
        refuse to load.
        """
        import torch

        if torch.cuda.is_available():
            bf16_ok = False
            try:
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_ok = False
            return "cuda", torch.bfloat16 if bf16_ok else torch.float16
        if (
            hasattr(torch, "backends")
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "mps", torch.float16
        return "cpu", torch.float32

    def load_model(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        transformer_gguf_repo: Optional[str] = None,
        transformer_gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        text_encoder_gguf_repo: Optional[str] = None,
        text_encoder_gguf_filename: Optional[str] = None,
        text_encoder_gguf_component: Optional[str] = None,
        prompt_enhancer_gguf_repo: Optional[str] = None,
        prompt_enhancer_gguf_filename: Optional[str] = None,
        lora_repo: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
        lora_adapter_name: Optional[str] = None,
        lora_scale: Optional[float] = None,
        lora_fuse: bool = False,
        hf_token: Optional[str] = None,
        family_override: Optional[str] = None,
        enable_model_cpu_offload: bool = True,
        offload_policy: Optional[str] = None,
        gguf_quantized_cpu_resident: Optional[bool] = None,
        gguf_pin_cpu_resident: Optional[bool] = None,
        safetensors_quantization: Optional[str] = None,
        safetensors_quantization_components: Optional[list[str]] = None,
        ignore_public_load_pending_workload: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a diffusion model.

        ``repo_id`` is the Hugging Face repo id of either a GGUF-only
        repo (e.g. ``unsloth/FLUX.2-klein-4B-GGUF``) or a full diffusers
        repo (e.g. ``black-forest-labs/FLUX.2-klein``). When the repo
        contains a GGUF, ``gguf_filename`` picks which quant to load;
        otherwise diffusers' standard config-driven load runs.

        ``base_repo`` overrides the auto-detected diffusers base used
        for VAE / text encoders. ``family_override`` short-circuits the
        substring matcher when an exotic repo name confuses it.

        ``offload_policy`` is the preferred VRAM control for GGUF image
        models. ``aggressive`` installs Diffusers CPU offload hooks and
        keeps packed GGUF tensors CPU-resident; ``balanced`` keeps all
        packed GGUF tensors CPU-resident without full Diffusers CPU
        offload hooks; ``less_aggressive`` keeps only text-encoder GGUF
        tensors CPU-resident while the diffusion transformer stays on
        GPU; ``none`` keeps the pipeline resident on the selected device
        and does not force GGUF tensors to stay on CPU. When omitted for
        a curated Studio GGUF, the curated recommendation is used. The
        lower-level booleans remain accepted for existing callers. When
        ``offload_policy`` is supplied, it owns those booleans.

        ``gguf_quantized_cpu_resident`` controls whether packed GGUF
        tensors stay on CPU and are copied/dequantized on demand. When
        omitted, it follows ``enable_model_cpu_offload`` for backwards
        compatibility. Setting it true while disabling full model CPU
        offload enables a hybrid benchmarking path: quantized GGUF
        weights stay off GPU, but Diffusers does not install full
        Accelerate offload hooks for every pipeline component.
        ``gguf_pin_cpu_resident`` pins CPU-resident packed GGUF tensors
        so host-to-device copies can use non-blocking transfer. When
        omitted, it follows UNSLOTH_STUDIO_GGUF_PIN_CPU_RESIDENT.

        ``safetensors_quantization`` is only for regular Diffusers
        safetensors repos. It uses Diffusers pipeline-level
        quantization to quantize selected components with bitsandbytes
        or torchao. It is rejected when any GGUF component swap is
        active because GGUF already owns the transformer/text precision.

        ``lora_repo`` optionally points at a Diffusers LoRA adapter repo
        or local path. When provided, the adapter is attached after the
        pipeline is constructed and before device placement/offload. LoRA
        fusion is restricted to non-GGUF models; unfused adapters are
        attempted for both full and GGUF-backed pipelines via Diffusers'
        native loader.

        Raises ``RuntimeError`` on failure with a user-facing message.
        On a failed swap the previous pipeline is also released to
        keep peak VRAM bounded; status() reports is_loaded=false with
        last_error set so the caller can react.
        """
        # Surface a friendly load error when the no-torch / partial
        # install path is active: the user clicked Load on the Images
        # page but the runtime never installed torch + diffusers (round
        # 13 P2 #12). Without this wrapper the import surfaces as a
        # raw ``ModuleNotFoundError`` -> 500 instead of a 400 the UI
        # can display.
        try:
            from huggingface_hub import hf_hub_download
            import diffusers
            import torch
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "Diffusion image generation requires the torch / diffusers "
                f"runtime. Missing dependency: {missing}. Install the Studio "
                "torch runtime (re-run setup.sh / install.ps1) before "
                "loading an image model."
            ) from exc
        _guard_diffusers_optional_bitsandbytes()

        # Round 30 P1 #11: also preflight transformers BEFORE any
        # destructive unload. Diffusers can expose stub pipeline
        # classes when transformers is missing or broken, so the load
        # would otherwise tear down chat first and fail later inside
        # from_pretrained. Use find_spec (no module execution) so test
        # environments that stub these modules still pass the preflight
        # without us actually importing them.
        # Round 34: accelerate is pulled in transitively by every
        # supported transformers install path (it is a hard runtime
        # dep of transformers' PyTorch backend), so a separate
        # find_spec("accelerate") guard is redundant in practice and
        # broke the CI test matrix where the test env ships
        # transformers without accelerate. The offload code path
        # (``enable_model_cpu_offload`` / ``device_map="auto"``)
        # will surface a clean ModuleNotFoundError if a user somehow
        # arrives at an offload-needed load without it.
        import importlib.util as _ilu

        if _ilu.find_spec("transformers") is None:
            raise RuntimeError(
                "Diffusion image generation requires the Studio torch "
                "runtime. Missing dependency: transformers. Install the "
                "Studio torch runtime (re-run setup.sh / install.ps1) "
                "before loading an image model."
            )
        _guard_transformers_tokenizers_backend()
        load_started = time.perf_counter()
        load_timings: dict[str, float] = {}

        @contextlib.contextmanager
        def _load_phase(name: str):
            phase_started = time.perf_counter()
            try:
                yield
            finally:
                elapsed = time.perf_counter() - phase_started
                load_timings[name] = load_timings.get(name, 0.0) + elapsed

        if transformer_gguf_filename and gguf_filename and transformer_gguf_filename != gguf_filename:
            raise ValueError(
                "Use either gguf_filename or transformer_gguf_filename for the "
                "diffusion transformer quant, not both with different values."
            )
        explicit_transformer_swap = bool(
            transformer_gguf_repo or transformer_gguf_filename
        )
        diffusion_gguf_filename = transformer_gguf_filename or gguf_filename
        diffusion_gguf_repo = transformer_gguf_repo
        if explicit_transformer_swap:
            if not diffusion_gguf_repo:
                raise ValueError(
                    "transformer_gguf_filename requires transformer_gguf_repo "
                    "because repo_id is reserved for the normal Diffusers pipeline repo."
                )
            if not diffusion_gguf_filename:
                raise ValueError(
                    "transformer_gguf_repo requires transformer_gguf_filename."
                )
            if base_repo and _expand_existing_local_path(base_repo) != _expand_existing_local_path(repo_id):
                raise ValueError(
                    "When transformer_gguf_repo is set, repo_id is already the "
                    "Diffusers pipeline repo. Omit base_repo or set it to the "
                    "same repo_id."
                )
            pipeline_repo = repo_id
        else:
            diffusion_gguf_repo = repo_id if diffusion_gguf_filename else None
            pipeline_repo = base_repo if diffusion_gguf_filename and base_repo else repo_id

        resolved_safetensors_quantization = _normalize_safetensors_quantization(
            safetensors_quantization
        )
        resolved_safetensors_quantization_components = (
            _normalize_safetensors_quantization_components(
                safetensors_quantization_components
            )
        )
        if (
            resolved_safetensors_quantization
            and resolved_safetensors_quantization != DIFFUSION_SAFETENSORS_QUANT_NONE
            and (
                diffusion_gguf_filename
                or text_encoder_gguf_filename
                or prompt_enhancer_gguf_filename
            )
        ):
            raise ValueError(
                "safetensors_quantization is only supported for regular "
                "Diffusers safetensors repos. Omit it when loading GGUF "
                "transformer, text encoder, or prompt-enhancer components."
            )

        family_probe_repo = pipeline_repo if explicit_transformer_swap else repo_id
        fam = detect_family(family_probe_repo, override_family = family_override)
        if fam is None and diffusion_gguf_filename and not explicit_transformer_swap:
            fam = detect_family(repo_id, override_family = family_override)
        if fam is None and not diffusion_gguf_filename and family_override is None:
            fam = DiffusionFamily(
                name = "diffusers",
                pipeline_class = "DiffusionPipeline",
                transformer_class = "",
                base_repo = _expand_existing_local_path(pipeline_repo),
                supports_gguf_single_file = False,
                aliases = (),
            )
        if fam is None:
            # Round 22 P2 #4: route the repo label through
            # ``_display_repo_id`` so a local absolute path that did
            # not match any family does not leak the operator's
            # filesystem layout via the error message / last_error
            # / 400 response body.
            raise RuntimeError(
                f"Could not infer a diffusion family for '{_display_repo_id(pipeline_repo)}'. "
                "Pass family_override = 'flux.2-klein' / 'flux.2' / "
                "'flux.1-kontext' / 'flux.1-schnell' / 'flux.1' / "
                "'qwen-image' / 'qwen-image-2512' / "
                "'qwen-image-edit' / 'qwen-image-edit-2509' / "
                "'qwen-image-edit-2511' / 'qwen-image-layered' / "
                "'z-image' / 'z-image-turbo' / 'ernie-image' / "
                "'ernie-image-turbo' / 'ltx2-3-distilled' / "
                "'wan2-2-t2v' / 'stable-diffusion-3' / "
                "'stable-diffusion-xl' to disambiguate."
            )
        if (
            text_encoder_gguf_component is not None
            and text_encoder_gguf_component not in _TEXT_ENCODER_GGUF_COMPONENTS
        ):
            allowed_components = ", ".join(sorted(_TEXT_ENCODER_GGUF_COMPONENTS))
            raise ValueError(
                "text_encoder_gguf_component must be one of: "
                f"{allowed_components}."
            )
        if prompt_enhancer_gguf_filename and text_encoder_gguf_component == "pe":
            raise ValueError(
                "Use either text_encoder_gguf_component='pe' or "
                "prompt_enhancer_gguf_filename, not both."
            )

        device, dtype = self._pick_device_and_dtype()

        if (
            offload_policy is None
            and gguf_quantized_cpu_resident is None
            and gguf_pin_cpu_resident is None
        ):
            offload_policy = _curated_gguf_recommended_offload_policy(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                transformer_gguf_repo = transformer_gguf_repo,
                transformer_gguf_filename = transformer_gguf_filename,
                device = device,
            )

        (
            resolved_offload_policy,
            enable_model_cpu_offload,
            gguf_quantized_cpu_resident,
            gguf_pin_cpu_resident,
        ) = _resolve_diffusion_offload_policy(
            offload_policy = offload_policy,
            enable_model_cpu_offload = enable_model_cpu_offload,
            gguf_quantized_cpu_resident = gguf_quantized_cpu_resident,
            gguf_pin_cpu_resident = gguf_pin_cpu_resident,
        )

        # Round 32 P1 #3: track whether the backend-side
        # helper-busy check published a "diffusion-backend" pending
        # entry so the outer finally clears the matching publish
        # exactly once. Set inside the try below right after the
        # snapshot succeeds.
        backend_pending_published = False

        # _load_lock serialises the entire load so two concurrent calls
        # cannot both kick off a multi-GB download + GPU upload at once.
        # The second caller waits behind the first and then loads on top
        # of the now-populated state via the normal swap path.
        # _generate_lock is also taken so we do not start swapping the
        # pipeline (release old + allocate new) while a previous
        # generation is still iterating denoising steps; releasing the
        # pipe out from under an in-flight forward corrupts scheduler
        # state. Order: _load_lock -> _generate_lock -> _lock so a
        # forward (which only takes _generate_lock + briefly _lock)
        # cannot block a queued load forever.
        with self._load_lock, self._generate_lock:
            with self._lock:
                self._loading = True
                self._last_error = None
                # Publish the pending target so cache / finetuned
                # delete guards can see what is mid-download even
                # before _repo_id / _base_repo are populated on
                # success.
                self._pending_repo_id = repo_id
                self._pending_diffusion_gguf_repo = diffusion_gguf_repo
                self._pending_base_repo = pipeline_repo
                self._pending_base_repo_source = (
                    "pipeline_repo" if explicit_transformer_swap else ("explicit" if base_repo else None)
                )
                self._pending_base_repo_confidence = (
                    "explicit" if (explicit_transformer_swap or base_repo) else None
                )
                self._pending_base_repo_variant = None
                self._pending_base_repo_warning = None
                # Store the caller's full ``gguf_filename`` (e.g.
                # ``BF16/model.gguf``) so the variant-aware delete
                # guards have the subdirectory info. The UI side of
                # status() still collapses to the basename for display.
                self._pending_gguf_filename = (
                    diffusion_gguf_filename if diffusion_gguf_filename else None
                )
                self._pending_text_encoder_gguf_repo = text_encoder_gguf_repo
                self._pending_text_encoder_gguf_filename = text_encoder_gguf_filename
                self._pending_prompt_enhancer_gguf_repo = prompt_enhancer_gguf_repo
                self._pending_prompt_enhancer_gguf_filename = prompt_enhancer_gguf_filename
                self._pending_lora_repo = lora_repo
                self._pending_lora_weight_name = lora_weight_name
            try:
                pipeline_cls = getattr(diffusers, fam.pipeline_class, None)
                if pipeline_cls is None:
                    raise RuntimeError(
                        f"diffusers {diffusers.__version__} has no "
                        f"{fam.pipeline_class}; upgrade diffusers and retry."
                    )
                transformer_cls = (
                    getattr(diffusers, fam.transformer_class, None)
                    if fam.transformer_class
                    else None
                )

                # Resolution rules for the "what repo to call
                # from_pretrained on" question:
                #   1. no GGUF file -> caller is loading a full
                #      diffusers repo; use repo_id directly so we do
                #      not silently substitute the family default
                #      AND ignore any base_repo input (it is only
                #      meaningful as a GGUF companion override). The
                #      old order let ``base_repo`` swap a fine-tuned
                #      ``owner/my-flux.1-finetune`` for
                #      ``black-forest-labs/FLUX.1-dev`` while status
                #      still advertised the user's repo (round 13
                #      P2 #10).
                #   2. otherwise prefer caller-supplied base_repo for
                #      the missing VAE / text encoder components.
                #   3. otherwise use the family + repo_id resolver so
                #      obvious curated GGUFs keep working, while ambiguous
                #      Flux2 Klein fine-tunes fail before using the wrong
                #      sampling contract.
                if not diffusion_gguf_filename:
                    # Guard: a repo that ends in "-GGUF" (the unsloth
                    # convention) is GGUF-only and will 500 on
                    # from_pretrained; surface a clear error instead of
                    # letting diffusers raise a confusing model-index
                    # failure deep in the loader.
                    if pipeline_repo.lower().endswith("-gguf"):
                        raise RuntimeError(
                            f"'{pipeline_repo}' looks like a GGUF-only repo. "
                            "Either provide gguf_filename to pick a quant, "
                            "or load a full diffusers repo (base_repo only "
                            "applies when picking a GGUF quant)."
                        )
                    # The resolver below expands ``~/models/my-flux`` so
                    # diffusers' from_pretrained does not pass the literal
                    # tilde through to ``os.path.isdir`` and fall back to
                    # the Hub (round 14 P2 #11).
                base_resolution: Optional[DiffusionBaseRepoResolution] = None
                effective_base: Optional[str] = None
                # Full Diffusers repos and explicit GGUF companion repos
                # can resolve immediately. Ambiguous GGUF companion
                # selection waits until after the GGUF file is local so
                # the resolver can use metadata and tensor-key signatures
                # instead of failing on repo names alone.
                if not diffusion_gguf_filename or base_repo or explicit_transformer_swap:
                    base_resolution = _resolve_diffusion_base_repo(
                        fam = fam,
                        repo_id = pipeline_repo,
                        gguf_filename = None if explicit_transformer_swap else diffusion_gguf_filename,
                        base_repo = None if explicit_transformer_swap else base_repo,
                    )
                    effective_base = base_resolution.base_repo
                    with self._lock:
                        self._pending_base_repo = effective_base
                        self._pending_base_repo_source = base_resolution.source
                        self._pending_base_repo_confidence = base_resolution.confidence
                        self._pending_base_repo_variant = base_resolution.variant
                        self._pending_base_repo_warning = base_resolution.warning

                transformer = None
                local_gguf_path: Optional[str] = None
                diffusion_gguf_inspection: Optional[DiffusionGGUFInspection] = None
                text_encoder = None
                prompt_enhancer = None
                lora_state: Optional[DiffusionLoraState] = None
                local_text_encoder_gguf_path: Optional[str] = None
                effective_text_encoder_gguf_repo: Optional[str] = None
                text_encoder_gguf_info: Any = None
                text_encoder_mmproj_path: Any = None
                text_encoder_gguf_plan: Optional[_TextEncoderGgufPlan] = None
                local_prompt_enhancer_gguf_path: Optional[str] = None
                effective_prompt_enhancer_gguf_repo: Optional[str] = None
                prompt_enhancer_gguf_plan: Optional[_TextEncoderGgufPlan] = None
                prepared_gguf_module_counts: dict[str, int] = {}
                text_encoder_component_name = (
                    text_encoder_gguf_component or "text_encoder"
                )
                diffusion_gguf_resident_device = (
                    "cpu"
                    if gguf_quantized_cpu_resident and device == "cuda"
                    else None
                )
                text_encoder_resident_device = diffusion_gguf_resident_device
                if (
                    resolved_offload_policy
                    in {
                        DIFFUSION_OFFLOAD_POLICY_LESS_AGGRESSIVE,
                        DIFFUSION_OFFLOAD_POLICY_HYBRID,
                    }
                    and device == "cuda"
                ):
                    text_encoder_resident_device = "cpu"
                gguf_pin_cpu_resident_active = bool(
                    gguf_pin_cpu_resident
                    and (
                        diffusion_gguf_resident_device == "cpu"
                        or text_encoder_resident_device == "cpu"
                    )
                )
                if diffusion_gguf_filename:
                    if transformer_cls is None:
                        raise RuntimeError(
                            f"Family {fam.name} does not have a GGUF transformer "
                            "path wired in this build; load the full repo instead."
                        )
                    has_single_file_loader = hasattr(transformer_cls, "from_single_file")
                    has_state_dict_loader = _transformer_supports_gguf_state_dict_load(
                        transformer_cls
                    )
                    if not fam.supports_gguf_single_file or (
                        not has_single_file_loader and not has_state_dict_loader
                    ):
                        raise RuntimeError(
                            f"Family {fam.name} cannot load GGUF single-file "
                            f"transformers with diffusers {diffusers.__version__}; "
                            "load the full diffusers repo instead."
                        )
                    # DiffusionLoadRequest.repo_id is documented to
                    # accept either a Hub repo id OR a local path
                    # (Studio export, downloaded HF snapshot, etc.).
                    # We accept BOTH absolute and relative local
                    # directories: Studio exports surface as relative
                    # paths like ``exports/my-flux`` and earlier
                    # versions only accepted absolute paths, falling
                    # through to ``hf_hub_download`` which then
                    # raised HFValidationError on the relative path
                    # (round 13 P1 #2). For local paths we route the
                    # gguf_filename through ``_resolve_local_gguf_child``
                    # so traversal (``../secret.gguf``) and absolute
                    # filename escapes (``/etc/passwd``) are rejected
                    # BEFORE the file is opened, which also keeps the
                    # delete-ownership guards aligned with what was
                    # actually loaded.
                    with _load_phase("resolve_diffusion_gguf"):
                        diffusion_gguf_repo_path = Path(diffusion_gguf_repo or "").expanduser()
                        if diffusion_gguf_repo_path.is_dir():
                            local_gguf_path = str(
                                _resolve_local_gguf_child(
                                    diffusion_gguf_repo_path,
                                    diffusion_gguf_filename,
                                )
                            )
                        else:
                            local_gguf_path = hf_hub_download(
                                repo_id = diffusion_gguf_repo,
                                filename = diffusion_gguf_filename,
                                token = hf_token,
                            )
                    local_diffusion_gguf = Path(local_gguf_path)
                    if (
                        local_diffusion_gguf.is_file()
                        and base_repo is None
                        and not explicit_transformer_swap
                    ):
                        with _load_phase("inspect_diffusion_gguf"):
                            diffusion_gguf_inspection = _inspect_diffusion_gguf_file(
                                local_diffusion_gguf
                            )
                        if not _gguf_inspection_matches_family(
                            diffusion_gguf_inspection,
                            fam,
                        ):
                            hints = ", ".join(diffusion_gguf_inspection.family_hints)
                            raise RuntimeError(
                                "The selected diffusion family does not match the "
                                "GGUF architecture signals. Detected "
                                f"architecture={diffusion_gguf_inspection.architecture or 'unknown'}, "
                                f"layout={diffusion_gguf_inspection.layout or 'unknown'}, "
                                f"family_hints=[{hints or 'none'}], but load requested "
                                f"family={fam.name}."
                            )

                if base_resolution is None:
                    base_resolution = _resolve_diffusion_base_repo(
                        fam = fam,
                        repo_id = diffusion_gguf_repo or repo_id,
                        gguf_filename = diffusion_gguf_filename,
                        base_repo = base_repo,
                        gguf_inspection = diffusion_gguf_inspection,
                    )
                    effective_base = base_resolution.base_repo
                    with self._lock:
                        self._pending_base_repo = effective_base
                        self._pending_base_repo_source = base_resolution.source
                        self._pending_base_repo_confidence = base_resolution.confidence
                        self._pending_base_repo_variant = base_resolution.variant
                        self._pending_base_repo_warning = base_resolution.warning

                if effective_base is None:
                    raise RuntimeError("Internal error: missing diffusion base repo.")
                # ``repo_id`` / ``effective_base`` are user-supplied
                # strings that can embed an ``hf_xxxxx`` token via a
                # URL-style path (``https://hf_token@huggingface.co/...``).
                # Scrub them BEFORE the logger formats the line so the
                # token never reaches structured-log sinks (round 14
                # P2 #9).
                # Round 23 P2 #11: ``_redact_hf_tokens`` only scrubs
                # ``hf_xxxxx`` substrings, so an absolute local
                # path like ``/home/alice/private/FLUX.2-klein-GGUF``
                # used to land in this log line verbatim. Route
                # through ``_display_repo_id`` so the leaf is
                # logged when the value is a filesystem path, with
                # the token-redaction step inside that helper as a
                # belt-and-braces defence.
                logger.info(
                    "Loading diffusion model %s (family=%s, device=%s, dtype=%s, base=%s)",
                    _display_repo_id(repo_id),
                    fam.name,
                    device,
                    dtype,
                    _display_repo_id(effective_base),
                )
                if text_encoder_gguf_filename:
                    effective_text_encoder_gguf_repo = (
                        text_encoder_gguf_repo or _default_text_encoder_gguf_repo(fam)
                    )
                    with self._lock:
                        self._pending_text_encoder_gguf_repo = effective_text_encoder_gguf_repo
                    with _load_phase("resolve_text_encoder_gguf"):
                        te_repo_path = Path(effective_text_encoder_gguf_repo).expanduser()
                        if te_repo_path.is_dir():
                            local_text_encoder_gguf_path = str(
                                _resolve_local_gguf_child(te_repo_path, text_encoder_gguf_filename)
                            )
                        else:
                            local_text_encoder_gguf_path = hf_hub_download(
                                repo_id = effective_text_encoder_gguf_repo,
                                filename = text_encoder_gguf_filename,
                                token = hf_token,
                            )
                    local_te_path = Path(local_text_encoder_gguf_path)
                    if local_te_path.is_file():
                        from .gguf_text_encoder import inspect_text_encoder_gguf

                        with _load_phase("inspect_text_encoder_gguf"):
                            text_encoder_gguf_info = inspect_text_encoder_gguf(local_te_path)
                        text_encoder_mmproj_path = getattr(
                            text_encoder_gguf_info,
                            "mmproj_path",
                            None,
                        )
                    text_encoder_gguf_arch = (
                        getattr(text_encoder_gguf_info, "architecture", None)
                        if text_encoder_gguf_info is not None
                        else None
                    )
                    if (
                        text_encoder_gguf_arch == "qwen2vl"
                        and text_encoder_mmproj_path is None
                        and not te_repo_path.is_dir()
                    ):
                        with _load_phase("resolve_text_encoder_mmproj"):
                            text_encoder_mmproj_path = _download_text_encoder_mmproj_from_hub(
                                hf_hub_download,
                                repo_id = effective_text_encoder_gguf_repo,
                                text_encoder_gguf_filename = text_encoder_gguf_filename,
                                token = hf_token,
                            )
                    text_encoder_gguf_plan = _resolve_text_encoder_gguf_plan(
                        fam,
                        architecture = text_encoder_gguf_arch,
                        requested_component = text_encoder_gguf_component,
                    )
                    if text_encoder_gguf_plan is None:
                        arch_note = ""
                        if text_encoder_gguf_info is not None:
                            arch_note = (
                                f" Detected text GGUF architecture: "
                                f"{text_encoder_gguf_arch or 'unknown'}."
                            )
                        raise RuntimeError(
                            "text_encoder_gguf_filename is currently supported for FLUX.2 dev "
                            "Mistral text encoders, Qwen-Image Qwen2VL text encoders, "
                            "Z-Image Qwen3 text encoders, FLUX.1 T5 encoders, "
                            "SD3 T5 encoders, or any supported lazy text architecture "
                            "when text_encoder_gguf_component is explicitly set."
                            + arch_note
                        )
                    text_encoder_component_name = text_encoder_gguf_plan.component_name
                if prompt_enhancer_gguf_filename:
                    effective_prompt_enhancer_gguf_repo = (
                        prompt_enhancer_gguf_repo
                        or _default_prompt_enhancer_gguf_repo(fam)
                    )
                    if not effective_prompt_enhancer_gguf_repo:
                        raise RuntimeError(
                            "prompt_enhancer_gguf_filename is only supported for "
                            "families with a prompt-enhancer component."
                        )
                    with self._lock:
                        self._pending_prompt_enhancer_gguf_repo = (
                            effective_prompt_enhancer_gguf_repo
                        )
                    pe_repo_path = Path(effective_prompt_enhancer_gguf_repo).expanduser()
                    if pe_repo_path.is_dir():
                        local_prompt_enhancer_gguf_path = str(
                            _resolve_local_gguf_child(
                                pe_repo_path,
                                prompt_enhancer_gguf_filename,
                            )
                        )
                    else:
                        local_prompt_enhancer_gguf_path = hf_hub_download(
                            repo_id = effective_prompt_enhancer_gguf_repo,
                            filename = prompt_enhancer_gguf_filename,
                            token = hf_token,
                        )
                    prompt_enhancer_gguf_plan = _resolve_text_encoder_gguf_plan(
                        fam,
                        architecture = "mistral3",
                        requested_component = "pe",
                    )
                    if prompt_enhancer_gguf_plan is None:
                        raise RuntimeError(
                            f"Family {fam.name} does not support a prompt-enhancer "
                            "GGUF component."
                        )

                # Round 20 P1 #1: every load mode (full diffusers
                # repo, GGUF + explicit base_repo, GGUF + auto-picked
                # base_repo) feeds ``effective_base`` into
                # ``from_pretrained`` further down. The round 19
                # preflight only ran for the first two, so an
                # auto-picked GGUF companion that turned out to be
                # gated / private / missing still unloaded chat
                # before the load failed. Always preflight
                # ``effective_base`` so a bad companion repo is
                # caught BEFORE chat / export are released.
                with _load_phase("preflight_pipeline_repo"):
                    _preflight_full_diffusers_repo(effective_base, hf_token)
                # Round 21 P2 #6: the GGUF transformer path also
                # consumes ``effective_base`` via
                # ``from_single_file(config=effective_base,
                # subfolder="transformer")``. A base that has
                # ``model_index.json`` but lacks
                # ``transformer/config.json`` would pass the
                # round-19 preflight and only fail AFTER the chat
                # unload. Run the subfolder probe too so the
                # second cheap failure mode is also caught early.
                if diffusion_gguf_filename and fam.transformer_class:
                    with _load_phase("preflight_transformer_config"):
                        _preflight_diffusers_subfolder_config(
                            effective_base,
                            "transformer",
                            hf_token,
                        )

                # Round 20 P1 #2: ``diffusers.GGUFQuantizationConfig``
                # imports the ``gguf`` package lazily at construction
                # time. Partial Studio installs (``diffusers`` present,
                # ``gguf`` not) used to discover that AFTER the chat /
                # export release calls. Build the quant config up
                # front so the missing-dependency surface raises
                # while the user's chat model is still resident.
                quant_config = None
                if diffusion_gguf_filename:
                    try:
                        quant_config = diffusers.GGUFQuantizationConfig(
                            compute_dtype = dtype
                        )
                    except ModuleNotFoundError as exc:
                        missing = exc.name or str(exc)
                        raise RuntimeError(
                            "Diffusion GGUF loading requires the gguf "
                            "runtime package. Missing dependency: "
                            f"{missing}. Re-run Studio setup before "
                            "loading an image GGUF."
                        ) from exc

                pipeline_quant_config = None
                if (
                    resolved_safetensors_quantization
                    and resolved_safetensors_quantization
                    != DIFFUSION_SAFETENSORS_QUANT_NONE
                ):
                    with _load_phase("build_safetensors_quantization_config"):
                        (
                            pipeline_quant_config,
                            resolved_safetensors_quantization,
                            resolved_safetensors_quantization_components,
                        ) = _build_safetensors_pipeline_quantization_config(
                            diffusers,
                            resolved_safetensors_quantization,
                            resolved_safetensors_quantization_components,
                            dtype,
                        )

                # All cheap failure points (bad gguf_filename, missing
                # pipeline / transformer class, gated download token,
                # transient Hub error on the GGUF download) have now
                # been validated. Anything past this line allocates
                # GPU memory, so:
                #   1. Verify training is idle and the export job (if
                #      any) is also idle. ``_release_other_gpu_owners
                #      _for_diffusion`` RAISES on conflict, so it must
                #      run BEFORE we unload chat (round 16 P1 #2): a
                #      route precheck -> worker race could otherwise
                #      drop the user's chat model only to bail out
                #      because training started in between, and a
                #      direct ``DiffusionBackend.load_model`` caller
                #      that did not run the route prechecks would also
                #      leave chat unloaded for nothing.
                #   2. Release the chat backend (llama-server + the
                #      safetensors orchestrator) now that we know the
                #      load can actually proceed.
                #   3. Release any *previous* diffusion pipeline so the
                #      new transformer / new from_pretrained does not
                #      race the old pipe for VRAM. Switching between
                #      FLUX.2 klein 4B and 9B on a 16-24 GB GPU OOMs
                #      otherwise: from_single_file allocates the new
                #      transformer while the old pipeline still owns
                #      its weights.
                #   4. THEN call from_single_file / from_pretrained.
                # Round 29 P1 #1: do ALL cheap conflict checks BEFORE
                # any destructive unload, so a training/export conflict
                # caught inside _release_other_gpu_owners_for_diffusion
                # does NOT leave the user with no chat model after we
                # already unloaded it. The helper-busy check is
                # split out of _release_chat_backend_for_diffusion;
                # _release_other_gpu_owners_for_diffusion raises
                # RuntimeError early when training/export is active
                # without touching the chat backend.
                # Round 32 P1 #3: publish a backend-side pending
                # entry under the helper-advisor start lock so a
                # direct / test / future caller of this method is
                # symmetric with the route layer's
                # _raise_if_helper_advisor_busy("diffusion"). The
                # route's "diffusion" tag and this "diffusion-
                # backend" tag refcount independently; both
                # contribute to public_load_pending().
                with _load_phase("release_other_backends"):
                    backend_pending_published = _raise_if_helper_advisor_busy_for_diffusion(
                        publish_pending = True,
                        ignore_pending_workload = (ignore_public_load_pending_workload),
                    )
                    _release_other_gpu_owners_for_diffusion()
                    _release_chat_backend_for_diffusion(check_helper_advisor = False)

                old = self._pipe
                if old is not None:
                    with self._lock:
                        # Clear ALL metadata together so a failed swap
                        # cannot leave status() reporting the previous
                        # repo / family / base_repo on top of an empty
                        # pipe. The except block below will restore
                        # last_error so the caller knows what happened.
                        self._pipe = None
                        self._family = None
                        self._repo_id = None
                        self._diffusion_gguf_repo = None
                        self._gguf_path = None
                        self._gguf_filename = None
                        self._text_encoder_gguf_repo = None
                        self._text_encoder_gguf_path = None
                        self._text_encoder_gguf_filename = None
                        self._prompt_enhancer_gguf_repo = None
                        self._prompt_enhancer_gguf_path = None
                        self._prompt_enhancer_gguf_filename = None
                        self._lora_state = None
                        self._component_sources = {}
                        self._base_repo = None
                        self._base_repo_source = None
                        self._base_repo_confidence = None
                        self._base_repo_variant = None
                        self._base_repo_warning = None
                        self._sampling_contract = None
                        self._device = None
                        self._dtype = None
                        self._cpu_offload_enabled = False
                        self._offload_policy = None
                        self._gguf_quantized_cpu_resident = False
                        self._gguf_pin_cpu_resident = False
                        self._gguf_execution_backend = None
                        self._gguf_prepared_module_counts = {}
                        self._safetensors_quantization = None
                        self._safetensors_quantization_components = None
                        self._load_timings = {}
                        self._prompt_embedding_cache_key = None
                        self._prompt_embedding_cache_value = None
                        self._loaded_at = None
                    with _load_phase("release_previous_pipeline"):
                        _release(old)
                        old = None
                        # Now that both the attribute and the local
                        # have been nulled, the pipeline is unreachable;
                        # ask the CUDA allocator to release its slabs so
                        # the next from_pretrained does not OOM behind
                        # an already-freed-but-cached arena.
                        _drain_cuda_cache()

                if diffusion_gguf_filename:
                    # ``quant_config`` was already constructed above
                    # (round 20 P1 #2 pre-release fail-fast).
                    if hasattr(transformer_cls, "from_single_file"):
                        # Diffusers-format GGUFs (FLUX.2 klein / Qwen-Image /
                        # SD3) need the matching base repo's component config
                        # at config=<base_repo>, subfolder="transformer".
                        # Older converted GGUFs ignore those kwargs. The
                        # token is also passed because gated GGUF repos
                        # require it both at download and at config read time.
                        single_file_kwargs: dict[str, Any] = {
                            "quantization_config": quant_config,
                            "torch_dtype": dtype,
                            "config": effective_base,
                            "subfolder": "transformer",
                        }
                        if hf_token:
                            single_file_kwargs["token"] = hf_token
                        with _load_phase("load_diffusion_transformer"):
                            with _patch_diffusers_gguf_checkpoint_loader_no_copy():
                                transformer = transformer_cls.from_single_file(
                                    local_gguf_path,
                                    **single_file_kwargs,
                                )
                    else:
                        with _load_phase("load_diffusion_transformer"):
                            transformer = _load_transformer_gguf_from_state_dict(
                                transformer_cls,
                                local_gguf_path,
                                base_repo = effective_base,
                                dtype = dtype,
                                quant_config = quant_config,
                                token = hf_token,
                            )
                    with _load_phase("prepare_diffusion_gguf_linear"):
                        lazy_linear_count = _replace_diffusers_gguf_linear_parameters(
                            transformer,
                            dtype,
                            resident_device = diffusion_gguf_resident_device,
                        )
                    if lazy_linear_count:
                        prepared_gguf_module_counts["diffusion_linear_lazy"] = lazy_linear_count
                        logger.info(
                            "Replaced %d diffusion GGUFLinear modules with Studio "
                            "lazy GGUF linear modules.",
                            lazy_linear_count,
                        )
                    with _load_phase("prepare_diffusion_gguf_non_linear"):
                        prepared_gguf_non_linear = _prepare_gguf_non_linear_parameters(
                            transformer,
                            dtype,
                            resident_device = diffusion_gguf_resident_device,
                        )
                    for key, count in prepared_gguf_non_linear.items():
                        prepared_gguf_module_counts[f"diffusion_{key}"] = count
                    if prepared_gguf_non_linear:
                        logger.info(
                            "Prepared diffusion GGUF non-linear parameters that "
                            "Diffusers does not execute lazily: %s.",
                            prepared_gguf_non_linear,
                        )
                    if diffusion_gguf_resident_device is not None:
                        with _load_phase("patch_diffusion_cpu_resident"):
                            patched_gguf_modules = _patch_gguf_modules_for_resident_device(
                                transformer,
                                diffusion_gguf_resident_device,
                                pin_memory = gguf_pin_cpu_resident_active,
                            )
                        if patched_gguf_modules:
                            prepared_gguf_module_counts[
                                "diffusion_cpu_resident_modules"
                            ] = patched_gguf_modules
                            logger.info(
                                "Patched %d diffusion GGUF modules for CPU-resident "
                                "quantized weights.",
                                patched_gguf_modules,
                            )
                    if (
                        resolved_offload_policy == DIFFUSION_OFFLOAD_POLICY_BALANCED
                        and diffusion_gguf_resident_device == "cpu"
                        and device == "cuda"
                    ):
                        cuda_cache_bytes = _balanced_gguf_cuda_cache_bytes(device = device)
                        if cuda_cache_bytes > 0:
                            try:
                                from .gguf_text_encoder import (
                                    configure_lazy_gguf_cuda_cache,
                                    install_compiled_lazy_gguf_linear_dequant,
                                )
                            except Exception as exc:
                                logger.debug(
                                    "Skipping optional balanced GGUF CUDA cache: %s",
                                    exc,
                                )
                            else:
                                with _load_phase("configure_balanced_cuda_cache"):
                                    cache_stats = configure_lazy_gguf_cuda_cache(
                                        transformer,
                                        cuda_cache_bytes,
                                    )
                                if cache_stats.get("modules", 0):
                                    prepared_gguf_module_counts[
                                        "diffusion_cuda_cache_modules"
                                    ] = int(cache_stats["modules"])
                                    prepared_gguf_module_counts[
                                        "diffusion_cuda_cache_budget_mib"
                                    ] = int(cache_stats["budget_bytes"] // (1024 * 1024))
                                    prepared_gguf_module_counts[
                                        "diffusion_cuda_cache_candidate_mib"
                                    ] = int(cache_stats["candidate_bytes"] // (1024 * 1024))
                                    prepared_gguf_module_counts[
                                        "diffusion_cuda_cache_selected_mib"
                                    ] = int(cache_stats["selected_bytes"] // (1024 * 1024))
                                    logger.info(
                                        "Configured balanced diffusion GGUF CUDA cache "
                                        "for %d CPU-resident modules with budget %d MiB "
                                        "(selected %d MiB of %d MiB candidates).",
                                        cache_stats["modules"],
                                        cache_stats["budget_bytes"] // (1024 * 1024),
                                        cache_stats["selected_bytes"] // (1024 * 1024),
                                        cache_stats["candidate_bytes"] // (1024 * 1024),
                                    )
                                with _load_phase("compile_balanced_gguf_dequant"):
                                    compile_stats = install_compiled_lazy_gguf_linear_dequant(
                                        transformer
                                    )
                                if compile_stats.get("modules", 0):
                                    prepared_gguf_module_counts[
                                        "diffusion_compiled_dequant_modules"
                                    ] = int(compile_stats["modules"])
                                    logger.info(
                                        "Installed compiled balanced diffusion GGUF "
                                        "dequant for %d lazy Linear modules.",
                                        compile_stats["modules"],
                                    )
                if local_text_encoder_gguf_path:
                    from . import gguf_text_encoder as gguf_text_encoder_mod

                    if text_encoder_gguf_plan is None:
                        raise RuntimeError(
                            "Internal error: missing text GGUF load plan."
                        )
                    with _load_phase("load_text_encoder_gguf"):
                        text_encoder = _load_text_encoder_gguf_from_plan(
                            gguf_text_encoder_mod,
                            text_encoder_gguf_plan,
                            local_text_encoder_gguf_path,
                            base_repo_or_path = effective_base,
                            mmproj_gguf_path = text_encoder_mmproj_path,
                            compute_dtype = dtype,
                            resident_device = text_encoder_resident_device,
                            token = hf_token,
                        )
                    if text_encoder_resident_device is not None:
                        patch_text_encoder = getattr(
                            gguf_text_encoder_mod,
                            "patch_gguf_text_encoder_for_resident_device",
                            None,
                        )
                        if patch_text_encoder is not None:
                            with _load_phase("patch_text_encoder_cpu_resident"):
                                patched_text_modules = patch_text_encoder(
                                text_encoder,
                                text_encoder_resident_device,
                                pin_memory = gguf_pin_cpu_resident_active,
                                )
                            if patched_text_modules:
                                prepared_gguf_module_counts[
                                    "text_cpu_resident_modules"
                                ] = patched_text_modules
                                logger.info(
                                    "Patched %d text-encoder GGUF modules for "
                                    "CPU-resident quantized weights.",
                                    patched_text_modules,
                                )
                if local_prompt_enhancer_gguf_path:
                    from . import gguf_text_encoder as gguf_text_encoder_mod

                    if prompt_enhancer_gguf_plan is None:
                        raise RuntimeError(
                            "Internal error: missing prompt-enhancer GGUF load plan."
                        )
                    with _load_phase("load_prompt_enhancer_gguf"):
                        prompt_enhancer = _load_text_encoder_gguf_from_plan(
                            gguf_text_encoder_mod,
                            prompt_enhancer_gguf_plan,
                            local_prompt_enhancer_gguf_path,
                            base_repo_or_path = effective_base,
                            mmproj_gguf_path = None,
                            compute_dtype = dtype,
                            resident_device = text_encoder_resident_device,
                            token = hf_token,
                        )
                    if text_encoder_resident_device is not None:
                        patch_text_encoder = getattr(
                            gguf_text_encoder_mod,
                            "patch_gguf_text_encoder_for_resident_device",
                            None,
                        )
                        if patch_text_encoder is not None:
                            with _load_phase("patch_prompt_enhancer_cpu_resident"):
                                patched_pe_modules = patch_text_encoder(
                                    prompt_enhancer,
                                    text_encoder_resident_device,
                                    pin_memory = gguf_pin_cpu_resident_active,
                                )
                            if patched_pe_modules:
                                prepared_gguf_module_counts[
                                    "prompt_enhancer_cpu_resident_modules"
                                ] = patched_pe_modules
                                logger.info(
                                    "Patched %d prompt-enhancer GGUF modules for "
                                    "CPU-resident quantized weights.",
                                    patched_pe_modules,
                                )

                with _load_phase("load_family_components"):
                    extra_components = _family_load_components(
                        diffusers,
                        fam,
                        effective_base,
                        dtype,
                        hf_token,
                    )
                pipe_kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    # use_safetensors=True refuses pickle-backed .bin
                    # weights at load time. Diffusers will fall back to
                    # safetensors variants on repos that publish both,
                    # and hard-error on repos that only ship .bin (which
                    # is the threat model we want to block since pickle
                    # files can execute arbitrary code in this process).
                    "use_safetensors": True,
                }
                if transformer is not None:
                    pipe_kwargs["transformer"] = transformer
                if text_encoder is not None:
                    pipe_kwargs[text_encoder_component_name] = text_encoder
                if (
                    fam.name.startswith("ernie-image")
                    and getattr(diffusers, "__version__", "") != "fake"
                ):
                    tokenizers_backend_cls = _transformers_tokenizers_backend_cls()
                    pipe_kwargs["tokenizer"] = (
                        tokenizers_backend_cls.from_pretrained(
                            effective_base,
                            subfolder = "tokenizer",
                            token = hf_token,
                        )
                    )
                if prompt_enhancer is not None:
                    pipe_kwargs["pe"] = prompt_enhancer
                    if (
                        fam.name.startswith("ernie-image")
                        and getattr(diffusers, "__version__", "") != "fake"
                    ):
                        tokenizers_backend_cls = _transformers_tokenizers_backend_cls()
                        pipe_kwargs["pe_tokenizer"] = (
                            tokenizers_backend_cls.from_pretrained(
                                effective_base,
                                subfolder = "pe_tokenizer",
                                token = hf_token,
                            )
                        )
                elif fam.name.startswith("ernie-image"):
                    pipe_kwargs["pe"] = None
                    pipe_kwargs["pe_tokenizer"] = None
                if pipeline_quant_config is not None:
                    pipe_kwargs["quantization_config"] = pipeline_quant_config
                pipe_kwargs.update(extra_components)
                if hf_token:
                    pipe_kwargs["token"] = hf_token

                pipe = None
                cpu_offload_enabled = bool(
                    enable_model_cpu_offload and device == "cuda"
                )
                try:
                    with _load_phase("pipeline_from_pretrained"):
                        pipe = pipeline_cls.from_pretrained(effective_base, **pipe_kwargs)
                    if lora_repo:
                        with _load_phase("apply_lora"):
                            lora_state = _apply_diffusion_lora(
                                pipe,
                                lora_repo = lora_repo,
                                lora_weight_name = lora_weight_name,
                                lora_adapter_name = lora_adapter_name,
                                lora_scale = lora_scale,
                                lora_fuse = bool(lora_fuse),
                                hf_token = hf_token,
                                gguf_filename = diffusion_gguf_filename,
                                uses_studio_lazy_gguf_modules = bool(
                                    prepared_gguf_module_counts.get("diffusion_linear_lazy")
                                    or text_encoder_gguf_filename
                                    or prompt_enhancer_gguf_filename
                                ),
                            )
                    # Device placement / offload can ALSO raise after
                    # from_pretrained succeeded (OOM at the .to(device)
                    # copy, accelerate offload hook misconfigured, etc.).
                    # If we let the exception escape now, the local
                    # ``pipe`` lives on the traceback frame until the
                    # caller drops it, holding multi-GB of VRAM behind
                    # the next load attempt. Explicitly release both
                    # pipe and transformer in the same try (round 13
                    # P2 #11).
                    with _load_phase("device_placement"):
                        if cpu_offload_enabled:
                            pipe.enable_model_cpu_offload()
                        else:
                            pipe.to(device)
                    if _enable_flux2_klein_embedded_guidance(pipe, fam):
                        logger.info(
                            "Enabled single-pass embedded guidance for Flux2 Klein."
                        )
                    elif _enable_flux2_klein_batched_cfg(pipe, fam):
                        logger.info(
                            "Enabled batched classifier-free guidance for Flux2 Klein."
                        )
                    with _load_phase("apply_memory_policy"):
                        _apply_diffusion_memory_policy(pipe, resolved_offload_policy)
                except Exception:
                    if pipe is not None:
                        _release(pipe)
                        pipe = None
                    if transformer is not None:
                        _release(transformer)
                        transformer = None
                    if text_encoder is not None:
                        _release(text_encoder)
                        text_encoder = None
                    if prompt_enhancer is not None:
                        _release(prompt_enhancer)
                        prompt_enhancer = None
                    for component in (extra_components or {}).values():
                        _release(component)
                    extra_components = {}
                    _drain_cuda_cache()
                    raise

                with self._lock:
                    self._pipe = pipe
                    self._family = fam
                    self._repo_id = repo_id
                    self._diffusion_gguf_repo = (
                        diffusion_gguf_repo if diffusion_gguf_filename else None
                    )
                    self._gguf_path = local_gguf_path
                    # Preserve the full caller-supplied filename, not
                    # just the basename, so per-variant delete guards
                    # see ``BF16/model.gguf`` (round 14 P1 #4).
                    self._gguf_filename = (
                        diffusion_gguf_filename if diffusion_gguf_filename else None
                    )
                    self._text_encoder_gguf_repo = effective_text_encoder_gguf_repo
                    self._text_encoder_gguf_path = local_text_encoder_gguf_path
                    self._text_encoder_gguf_filename = (
                        text_encoder_gguf_filename if text_encoder_gguf_filename else None
                    )
                    self._prompt_enhancer_gguf_repo = effective_prompt_enhancer_gguf_repo
                    self._prompt_enhancer_gguf_path = local_prompt_enhancer_gguf_path
                    self._prompt_enhancer_gguf_filename = (
                        prompt_enhancer_gguf_filename
                        if prompt_enhancer_gguf_filename
                        else None
                    )
                    self._lora_state = lora_state
                    self._component_sources = _build_diffusion_component_sources(
                        pipeline_repo = effective_base,
                        diffusion_gguf_repo = diffusion_gguf_repo,
                        diffusion_gguf_filename = diffusion_gguf_filename,
                        text_encoder_gguf_repo = effective_text_encoder_gguf_repo,
                        text_encoder_gguf_filename = text_encoder_gguf_filename,
                        text_encoder_component = (
                            text_encoder_component_name
                            if text_encoder_gguf_filename
                            else None
                        ),
                        prompt_enhancer_gguf_repo = effective_prompt_enhancer_gguf_repo,
                        prompt_enhancer_gguf_filename = prompt_enhancer_gguf_filename,
                        lora_state = lora_state,
                    )
                    self._base_repo = effective_base
                    self._base_repo_source = base_resolution.source
                    self._base_repo_confidence = base_resolution.confidence
                    self._base_repo_variant = base_resolution.variant
                    self._base_repo_warning = base_resolution.warning
                    self._sampling_contract = _build_sampling_contract(
                        pipe = pipe,
                        fam = fam,
                        base_repo = effective_base,
                        base_repo_source = base_resolution.source,
                        base_repo_confidence = base_resolution.confidence,
                        base_repo_variant = base_resolution.variant,
                        gguf_filename = diffusion_gguf_filename,
                    )
                    self._device = device
                    self._dtype = str(dtype).replace("torch.", "")
                    self._cpu_offload_enabled = cpu_offload_enabled
                    self._offload_policy = resolved_offload_policy
                    cpu_resident_gguf_modules = sum(
                        int(prepared_gguf_module_counts.get(key, 0) or 0)
                        for key in (
                            "diffusion_cpu_resident_modules",
                            "text_cpu_resident_modules",
                            "prompt_enhancer_cpu_resident_modules",
                        )
                    )
                    self._gguf_quantized_cpu_resident = cpu_resident_gguf_modules > 0
                    self._gguf_pin_cpu_resident = bool(
                        gguf_pin_cpu_resident_active and cpu_resident_gguf_modules > 0
                    )
                    self._gguf_execution_backend = (
                        _detect_gguf_execution_backend(device)
                        if (
                            diffusion_gguf_filename
                            or text_encoder_gguf_filename
                            or prompt_enhancer_gguf_filename
                        )
                        else None
                    )
                    self._gguf_prepared_module_counts = dict(
                        prepared_gguf_module_counts
                    )
                    self._safetensors_quantization = (
                        resolved_safetensors_quantization
                        if resolved_safetensors_quantization
                        != DIFFUSION_SAFETENSORS_QUANT_NONE
                        else None
                    )
                    self._safetensors_quantization_components = (
                        list(resolved_safetensors_quantization_components)
                        if (
                            resolved_safetensors_quantization
                            and resolved_safetensors_quantization
                            != DIFFUSION_SAFETENSORS_QUANT_NONE
                            and resolved_safetensors_quantization_components
                        )
                        else None
                    )
                    load_timings["total"] = time.perf_counter() - load_started
                    self._load_timings = {
                        key: round(value, 6)
                        for key, value in load_timings.items()
                    }
                    self._prompt_embedding_cache_key = None
                    self._prompt_embedding_cache_value = None
                    self._loaded_at = time.time()
                    # Clear loading + pending here, BEFORE returning,
                    # so the response payload reports the resident
                    # pipeline cleanly (is_loading=false, no pending_*).
                    # The ``finally`` block below is idempotent and
                    # still clears on error / early raise paths.
                    self._loading = False
                    self._pending_repo_id = None
                    self._pending_diffusion_gguf_repo = None
                    self._pending_base_repo = None
                    self._pending_base_repo_source = None
                    self._pending_base_repo_confidence = None
                    self._pending_base_repo_variant = None
                    self._pending_base_repo_warning = None
                    self._pending_gguf_filename = None
                    self._pending_text_encoder_gguf_repo = None
                    self._pending_text_encoder_gguf_filename = None
                    self._pending_prompt_enhancer_gguf_repo = None
                    self._pending_prompt_enhancer_gguf_filename = None
                    self._pending_lora_repo = None
                    self._pending_lora_weight_name = None

                return self.status()
            except Exception as exc:
                # Scrub hf_token and pipe_kwargs from frame locals BEFORE
                # logger.exception() captures them. Rich tracebacks and
                # some structlog formatters render frame locals, which
                # would otherwise echo the raw hf_... token into logs
                # and any error reporting sink the user has wired up.
                # ALSO scrub the exception message itself: huggingface_hub
                # / diffusers can include the bearer token verbatim in
                # 401 / 403 messages, which would propagate through
                # ``_last_error`` (rendered in status()) and the
                # user-facing RuntimeError (rendered in route responses).
                scrub_token = hf_token
                hf_token = None  # noqa: F841
                pipe_kwargs = None  # noqa: F841
                single_file_kwargs = None  # noqa: F841
                extra_components = None  # noqa: F841
                exc_msg = str(exc)
                if scrub_token:
                    exc_msg = exc_msg.replace(scrub_token, "<redacted>")
                # Hugging Face tokens are prefixed ``hf_``; replace any
                # leftover ``hf_...`` substrings to catch tokens we did
                # not store as ``scrub_token`` (e.g. cached tokens that
                # huggingface_hub picked up on its own).
                import re

                exc_msg = re.sub(r"hf_[A-Za-z0-9]{20,}", "<redacted>", exc_msg)

                # Round 17 P2 #9: diffusers / safetensors raise errors
                # like ``FileNotFoundError: /home/alice/models/foo.gguf``
                # or ``OSError: Error while loading state dict from
                # C:\\Users\\bob\\repos\\flux``. These messages flow
                # into ``_last_error`` (rendered by status() to every
                # authenticated browser tab) and the user-facing
                # RuntimeError, which would leak the operator's
                # filesystem layout to other sessions. Collapse the
                # known repo / base / gguf paths to their leaf name
                # using the same convention as _display_repo_id().
                def _collapse_local(msg: str, candidate: Optional[str]) -> str:
                    if not candidate or not isinstance(candidate, str):
                        return msg
                    try:
                        p = Path(candidate).expanduser()
                    except (OSError, ValueError):
                        return msg
                    leaf = p.name or candidate
                    needles: set[str] = set()
                    # Round 20 P2 #6: a relative candidate like
                    # ``exports/my-flux`` used to collapse only the
                    # exact ``exports/my-flux`` substring, but
                    # downstream libraries (diffusers / safetensors)
                    # resolve and emit ``/mnt/disks/.../exports/my-flux/...``
                    # absolute strings that leaked the operator's
                    # filesystem layout. Also scrub the resolved
                    # absolute form so the leaf is the only path
                    # fragment that survives.
                    try:
                        if p.exists():
                            needles.add(str(p.resolve()))
                        elif p.is_absolute():
                            needles.add(str(p))
                    except (OSError, ValueError):
                        pass
                    if "/" in candidate or "\\" in candidate:
                        needles.add(candidate)
                    # Replace longest first so a parent-directory
                    # substring does not blank out the leaf-only
                    # context the user needs.
                    for needle in sorted(
                        (n for n in needles if n and n != leaf),
                        key = len,
                        reverse = True,
                    ):
                        msg = msg.replace(needle, leaf)
                    return msg

                # ``effective_base`` and ``gguf_filename`` are local
                # to the try block above and may be unbound if the
                # exception fired before assignment (e.g. the GGUF
                # repo / filename validation raises before
                # ``effective_base`` is computed). ``locals().get``
                # keeps the scrub a no-op in that case.
                # Round 18 P2 #9: also scrub ``local_gguf_path``. The
                # GGUF quant is loaded via
                # ``transformer_cls.from_single_file(local_gguf_path)``,
                # and diffusers / safetensors errors include the
                # resolved absolute HF cache path
                # (``/home/alice/.cache/huggingface/hub/.../flux.gguf``).
                # Without this the cache path would leak into
                # ``_last_error`` (and therefore status() / log lines).
                _locals = locals()
                exc_msg = _collapse_local(exc_msg, repo_id)
                exc_msg = _collapse_local(exc_msg, _locals.get("effective_base"))
                exc_msg = _collapse_local(exc_msg, _locals.get("gguf_filename"))
                exc_msg = _collapse_local(exc_msg, _locals.get("local_gguf_path"))
                exc_msg = _collapse_local(exc_msg, _locals.get("effective_text_encoder_gguf_repo"))
                exc_msg = _collapse_local(exc_msg, _locals.get("text_encoder_gguf_filename"))
                exc_msg = _collapse_local(exc_msg, _locals.get("local_text_encoder_gguf_path"))
                exc_msg = _collapse_local(exc_msg, _locals.get("effective_prompt_enhancer_gguf_repo"))
                exc_msg = _collapse_local(exc_msg, _locals.get("prompt_enhancer_gguf_filename"))
                exc_msg = _collapse_local(exc_msg, _locals.get("local_prompt_enhancer_gguf_path"))
                exc_msg = _collapse_local(exc_msg, _locals.get("lora_repo"))
                exc_msg = _collapse_local(exc_msg, _locals.get("lora_weight_name"))
                with self._lock:
                    self._last_error = exc_msg
                # ``logger.exception`` would emit the raw exception
                # (including any unredacted ``hf_...`` token inside
                # the message OR traceback locals on rich loggers).
                # Use ``logger.error`` with the already-scrubbed
                # message and exc_info=False so the bearer token
                # cannot leak through structured logging sinks.
                # Round 23 P2 #12: same fix as the start-of-load
                # log above. ``_redact_hf_tokens`` alone left
                # absolute local repo paths in this failure line.
                logger.error(
                    "Diffusion load failed for %s: %s",
                    _display_repo_id(repo_id),
                    exc_msg,
                )
                raise RuntimeError(
                    f"Failed to load diffusion model: {exc_msg}"
                ) from exc
            finally:
                with self._lock:
                    self._loading = False
                    # Clear pending so status() falls back to publishing
                    # the resident pipeline (or nothing, on a failed
                    # swap). Keeping pending alive after the load
                    # finishes would falsely block deletes forever.
                    self._pending_repo_id = None
                    self._pending_diffusion_gguf_repo = None
                    self._pending_base_repo = None
                    self._pending_base_repo_source = None
                    self._pending_base_repo_confidence = None
                    self._pending_base_repo_variant = None
                    self._pending_base_repo_warning = None
                    self._pending_gguf_filename = None
                    self._pending_text_encoder_gguf_repo = None
                    self._pending_text_encoder_gguf_filename = None
                    self._pending_prompt_enhancer_gguf_repo = None
                    self._pending_prompt_enhancer_gguf_filename = None
                    self._pending_lora_repo = None
                    self._pending_lora_weight_name = None
                # Round 32 P1 #3: clear the backend-side public-load
                # pending publish if it was set. Skipped when the
                # helper-busy snapshot raised (no publish to clear)
                # so the counter stays in sync with publishes.
                if backend_pending_published:
                    _clear_diffusion_backend_pending()

    def unload_model(self) -> dict[str, Any]:
        # Take the load lock and the generate lock so unload cannot:
        #   * race with an in-flight load_model and have the load
        #     thread overwrite the cleared state after we already
        #     returned {"is_loaded": false}.
        #   * return is_loaded=false while a forward pass is still
        #     iterating denoising steps on the soon-to-be-freed pipe.
        # The generate forward only holds _generate_lock (briefly
        # _lock), so acquiring _generate_lock here blocks until any
        # in-flight generation completes.
        with self._load_lock, self._generate_lock:
            with self._lock:
                old = self._pipe
                # Mark the slot as busy BEFORE clearing _pipe so a
                # concurrent helper-busy check (which treats either
                # is_loaded OR is_loading as busy) does not see a
                # ``free`` GPU during the release + cache-drain window.
                # is_loading is cleared in finally once the VRAM is
                # actually freed.
                self._loading = True
                self._pipe = None
                self._family = None
                self._repo_id = None
                self._diffusion_gguf_repo = None
                self._gguf_path = None
                self._gguf_filename = None
                self._text_encoder_gguf_repo = None
                self._text_encoder_gguf_path = None
                self._text_encoder_gguf_filename = None
                self._prompt_enhancer_gguf_repo = None
                self._prompt_enhancer_gguf_path = None
                self._prompt_enhancer_gguf_filename = None
                self._lora_state = None
                self._component_sources = {}
                self._pending_lora_repo = None
                self._pending_lora_weight_name = None
                self._base_repo = None
                self._base_repo_source = None
                self._base_repo_confidence = None
                self._base_repo_variant = None
                self._base_repo_warning = None
                self._sampling_contract = None
                self._device = None
                self._dtype = None
                self._cpu_offload_enabled = False
                self._offload_policy = None
                self._gguf_quantized_cpu_resident = False
                self._gguf_pin_cpu_resident = False
                self._gguf_execution_backend = None
                self._gguf_prepared_module_counts = {}
                self._safetensors_quantization = None
                self._safetensors_quantization_components = None
                self._load_timings = {}
                self._prompt_embedding_cache_key = None
                self._prompt_embedding_cache_value = None
                self._loaded_at = None
            try:
                _release(old)
                old = None  # noqa: F841
                _drain_cuda_cache()
            finally:
                with self._lock:
                    self._loading = False
        return {"is_loaded": False}

    # ── generation ────────────────────────────────────────────────

    def _apply_prompt_embedding_cache_unlocked(
        self,
        *,
        pipe: Any,
        fam: Optional[DiffusionFamily],
        call_kwargs: dict[str, Any],
        prompt: str,
        negative_prompt: Optional[str],
        guidance_scale: float,
        device: str,
    ) -> None:
        """Reuse prompt embeddings for repeated generations on one pipeline.

        Graph/workflow engines naturally cache upstream text-encoder nodes
        when only the seed changes. Studio's direct pipeline path has no graph,
        so repeated warm generations used to re-run the text encoder every
        time. Keep the cache tiny and model-agnostic: only activate for
        pipelines that expose Diffusers' `encode_prompt` plus
        `prompt_embeds`/`negative_prompt_embeds` call kwargs, and store the
        cached tensors on CPU so low-VRAM policies do not inherit persistent
        CUDA embeddings.
        """

        input_images = call_kwargs.get("image")
        if not prompt or input_images is not None:
            # Image-to-image/edit pipelines can have additional conditioning
            # semantics. Keep this cache to text-to-image until those APIs are
            # benchmarked separately.
            return
        if not _pipe_accepts_kwarg(pipe, "prompt_embeds"):
            return
        encode_prompt = getattr(pipe, "encode_prompt", None)
        if not callable(encode_prompt):
            return

        do_classifier_free_guidance = bool(guidance_scale > 1.0)
        if do_classifier_free_guidance and not _pipe_accepts_kwarg(
            pipe,
            "negative_prompt_embeds",
        ):
            return
        accepts_prompt_mask = _pipe_accepts_kwarg(pipe, "prompt_embeds_mask")
        accepts_negative_prompt_mask = _pipe_accepts_kwarg(
            pipe,
            "negative_prompt_embeds_mask",
        )

        max_sequence_length = int(call_kwargs.get("max_sequence_length", 512))
        num_images_per_prompt = int(call_kwargs.get("num_images_per_prompt", 1))
        use_pe = bool(call_kwargs.get("use_pe", False))
        width = int(call_kwargs.get("width", 0) or 0)
        height = int(call_kwargs.get("height", 0) or 0)
        negative_key = negative_prompt if do_classifier_free_guidance else None
        cache_key = (
            id(pipe),
            fam.name if fam is not None else None,
            prompt,
            negative_key or "",
            do_classifier_free_guidance,
            max_sequence_length,
            num_images_per_prompt,
            use_pe,
            width,
            height,
        )
        cached = (
            self._prompt_embedding_cache_value
            if self._prompt_embedding_cache_key == cache_key
            else None
        )

        if cached is None:
            execution_device = getattr(pipe, "_execution_device", None) or device
            try:
                prompt_embeds, negative_prompt_embeds = encode_prompt(
                    prompt = prompt,
                    negative_prompt = negative_prompt,
                    do_classifier_free_guidance = do_classifier_free_guidance,
                    device = execution_device,
                    max_sequence_length = max_sequence_length,
                )
                prompt_embeds, prompt_embeds_mask = _prompt_embeds_and_optional_mask(
                    prompt_embeds,
                    accepts_mask = accepts_prompt_mask,
                )
                if do_classifier_free_guidance:
                    (
                        negative_prompt_embeds,
                        negative_prompt_embeds_mask,
                    ) = _prompt_embeds_and_optional_mask(
                        negative_prompt_embeds,
                        accepts_mask = accepts_negative_prompt_mask,
                    )
                else:
                    negative_prompt_embeds = None
                    negative_prompt_embeds_mask = None
            except TypeError:
                try:
                    prompt_for_encode: Any = prompt
                    if (
                        use_pe
                        and callable(getattr(pipe, "_enhance_prompt_with_pe", None))
                        and getattr(pipe, "pe", None) is not None
                        and getattr(pipe, "pe_tokenizer", None) is not None
                    ):
                        prompt_values = [prompt] if isinstance(prompt, str) else list(prompt)
                        prompt_for_encode = [
                            pipe._enhance_prompt_with_pe(
                                item,
                                execution_device,
                                width = width or 1024,
                                height = height or 1024,
                            )
                            for item in prompt_values
                        ]
                    prompt_embeds = encode_prompt(
                        prompt_for_encode,
                        execution_device,
                        1,
                    )
                    prompt_embeds, prompt_embeds_mask = _prompt_embeds_and_optional_mask(
                        prompt_embeds,
                        accepts_mask = accepts_prompt_mask,
                    )
                    if do_classifier_free_guidance:
                        negative_values: Any = negative_prompt or ""
                        prompt_count = (
                            len(prompt_for_encode)
                            if isinstance(prompt_for_encode, list)
                            else 1
                        )
                        if isinstance(negative_values, str):
                            negative_values = [negative_values] * prompt_count
                        negative_prompt_embeds = encode_prompt(
                            negative_values,
                            execution_device,
                            1,
                        )
                        (
                            negative_prompt_embeds,
                            negative_prompt_embeds_mask,
                        ) = _prompt_embeds_and_optional_mask(
                            negative_prompt_embeds,
                            accepts_mask = accepts_negative_prompt_mask,
                        )
                    else:
                        negative_prompt_embeds = None
                        negative_prompt_embeds_mask = None
                except TypeError:
                    return
            cached = (
                _store_prompt_embeds_on_cpu(prompt_embeds),
                _store_prompt_embeds_on_cpu(prompt_embeds_mask),
                _store_prompt_embeds_on_cpu(negative_prompt_embeds),
                _store_prompt_embeds_on_cpu(negative_prompt_embeds_mask),
            )
            self._prompt_embedding_cache_key = cache_key
            self._prompt_embedding_cache_value = cached

        execution_device = getattr(pipe, "_execution_device", None) or device
        call_kwargs["prompt"] = None
        call_kwargs["prompt_embeds"] = _clone_prompt_embeds_to_device(
            cached[0],
            execution_device,
        )
        if accepts_prompt_mask and cached[1] is not None:
            call_kwargs["prompt_embeds_mask"] = _clone_prompt_embeds_to_device(
                cached[1],
                execution_device,
            )
        if _pipe_accepts_kwarg(pipe, "negative_prompt_embeds"):
            call_kwargs["negative_prompt_embeds"] = _clone_prompt_embeds_to_device(
                cached[2],
                execution_device,
            )
        if accepts_negative_prompt_mask and cached[3] is not None:
            call_kwargs["negative_prompt_embeds_mask"] = _clone_prompt_embeds_to_device(
                cached[3],
                execution_device,
            )
        call_kwargs.pop("negative_prompt", None)

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        input_images: Optional[list[Any]] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "Any":
        """Generate a single PIL image and return it.

        Concurrent generations are serialised by ``_generate_lock`` so
        diffusion pipelines (not thread-safe; overlapping ``__call__``s
        corrupt internal scheduler state) only ever run one at a time.
        The state ``_lock`` is taken only to snapshot ``_pipe`` /
        ``_device`` and immediately released: holding it for the whole
        forward pass blocked ``status()`` polls and concurrent unload
        requests for the entire (minutes-long) generation, which made
        the UI feel frozen.
        """
        # Take _generate_lock FIRST so a concurrent unload/load that
        # observes us holding it will queue behind this generation
        # (and `unload_model` then waits its turn before clearing
        # state). Snapshotting `self._pipe` outside the lock and then
        # taking the lock let a load/unload race in between, so the
        # forward could run against a freed or swapped pipeline.
        with self._generate_lock:
            return self._generate_image_unlocked(
                prompt = prompt,
                negative_prompt = negative_prompt,
                input_images = input_images,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                width = width,
                height = height,
                seed = seed,
            )

    def _generate_image_unlocked(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        input_images: Optional[list[Any]] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        return_all_images: bool = False,
    ) -> "Any":
        """Inner body of ``generate_image`` that ASSUMES the caller
        already holds ``_generate_lock``. Lets
        ``generate_image_with_metadata`` snapshot metadata under the
        same lock without deadlocking on a non-reentrant
        ``threading.Lock`` (round 13 P2 #9)."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")

        import torch

        with self._lock:
            if self._pipe is None:
                raise RuntimeError("No diffusion model is loaded.")
            pipe = self._pipe
            fam = self._family
            base_repo_variant = self._base_repo_variant
            device = self._device or "cpu"
            cpu_offload_enabled = self._cpu_offload_enabled
            drain_cache_after_generation = (
                bool(self._gguf_quantized_cpu_resident) and device == "cuda"
            )
        defaults = (
            _sampling_defaults_for_loaded_pipeline(
                pipe,
                fam,
                base_repo_variant = base_repo_variant,
            )
            if fam is not None
            else None
        )
        resolved_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else (defaults.default_steps if defaults is not None else 24)
        )
        resolved_guidance = float(
            guidance_scale
            if guidance_scale is not None
            else (defaults.default_guidance_scale if defaults is not None else 3.5)
        )
        resolved_width = int(
            width if width is not None else (fam.default_width if fam is not None else 1024)
        )
        resolved_height = int(
            height if height is not None else (fam.default_height if fam is not None else 1024)
        )
        if fam is not None and fam.media_kind != "image":
            raise RuntimeError(
                f"{fam.name} is a {fam.media_kind} generation family and cannot be "
                "used with the image generation route."
            )
        if resolved_steps < 1 or resolved_steps > 200:
            raise ValueError("num_inference_steps must be in [1, 200]")
        if (
            resolved_width <= 0
            or resolved_height <= 0
            or resolved_width > 2048
            or resolved_height > 2048
        ):
            raise ValueError("width and height must be in (0, 2048]")
        # Snap to a multiple of 8: Flux / SD pipelines require it and a
        # silent crash deep in the VAE is much worse than a clear error
        # message up front.
        if resolved_width % 8 or resolved_height % 8:
            raise ValueError("width and height must be multiples of 8")
        if fam is not None and fam.requires_image_input:
            if not input_images:
                raise RuntimeError(f"{fam.name} requires image input.")
        elif input_images:
            raise RuntimeError(
                f"{fam.name if fam is not None else 'This diffusion model'} "
                "does not accept image input."
            )
        generator = None
        if seed is not None:
            # Match the device of the pipeline so determinism holds
            # across reload cycles. When CPU offload is enabled
            # (the default on CUDA hosts), diffusers shuttles each
            # submodule between CPU and GPU on every step. A CUDA
            # torch.Generator then mismatches the CPU-resident
            # embeddings at the start of the forward and the run
            # crashes (round 14 P1 #6). Use a CPU generator in that
            # case; numerical determinism for the same seed is
            # preserved because the seed feeds an int rather than a
            # device-local RNG state.
            if cpu_offload_enabled:
                gen_device = "cpu"
            else:
                gen_device = (
                    "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                )
            generator = torch.Generator(device = gen_device).manual_seed(int(seed))

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": resolved_steps,
            "width": resolved_width,
            "height": resolved_height,
        }
        if defaults is not None and defaults.default_call_kwargs:
            call_kwargs.update(
                {
                    key: value
                    for key, value in defaults.default_call_kwargs.items()
                    if _pipe_accepts_kwarg(pipe, key)
                }
            )
        if input_images:
            if _pipe_accepts_kwarg(pipe, "image"):
                prepared_input_images = list(input_images)
                if fam is not None and fam.name == "qwen-image-layered":
                    prepared_input_images = [
                        image.convert("RGBA")
                        if hasattr(image, "convert") and getattr(image, "mode", None) != "RGBA"
                        else image
                        for image in prepared_input_images
                    ]
                call_kwargs["image"] = (
                    prepared_input_images[0]
                    if len(prepared_input_images) == 1
                    else prepared_input_images
                )
            else:
                raise RuntimeError(
                    f"{type(pipe).__name__} does not accept image input."
                )
        guidance_kwarg = _guidance_kwarg_for_pipe(pipe, fam)
        if _pipe_accepts_kwarg(pipe, guidance_kwarg):
            call_kwargs[guidance_kwarg] = resolved_guidance
        elif _pipe_accepts_kwarg(pipe, "guidance_scale"):
            call_kwargs["guidance_scale"] = resolved_guidance
        # FLUX.2 / FLUX.2 klein pipelines do NOT accept
        # negative_prompt and 500 if you pass it in. Inspect the
        # signature and only forward when supported; warn otherwise
        # so the UI can disable the field for incompatible families.
        effective_negative_prompt = negative_prompt
        if (
            (effective_negative_prompt is None or not effective_negative_prompt.strip())
            and fam is not None
            and fam.default_negative_prompt is not None
            and guidance_kwarg == "true_cfg_scale"
        ):
            effective_negative_prompt = fam.default_negative_prompt
        should_forward_negative = (
            effective_negative_prompt is not None
            and (
                bool(effective_negative_prompt.strip())
                or (
                    fam is not None
                    and fam.default_negative_prompt is not None
                    and effective_negative_prompt == fam.default_negative_prompt
                )
            )
        )
        if should_forward_negative:
            if _pipe_accepts_kwarg(pipe, "negative_prompt"):
                call_kwargs["negative_prompt"] = effective_negative_prompt
                # QwenImagePipeline and FluxPipeline treat
                # guidance_scale as distilled CFG and use
                # true_cfg_scale as the real classifier-free
                # guidance knob; the negative prompt is only
                # effective when true_cfg_scale > 1. Forward the
                # user-supplied guidance_scale through both so the
                # negative prompt actually steers generation.
                if _pipe_accepts_kwarg(pipe, "true_cfg_scale"):
                    call_kwargs["true_cfg_scale"] = resolved_guidance
            else:
                logger.info(
                    "Dropping negative_prompt: %s does not accept it",
                    type(pipe).__name__,
                )
        if generator is not None:
            call_kwargs["generator"] = generator

        self._apply_prompt_embedding_cache_unlocked(
            pipe = pipe,
            fam = fam,
            call_kwargs = call_kwargs,
            prompt = prompt,
            negative_prompt = (
                effective_negative_prompt if should_forward_negative else None
            ),
            guidance_scale = resolved_guidance,
            device = device,
        )

        # QwenImageLayeredPipeline takes a fixed layer resolution rather
        # than width/height in its public signature. The UI still sends
        # width/height for consistency with image generation, so map the
        # requested square size to `resolution` when the pipeline wants it.
        if (
            fam is not None
            and fam.name == "qwen-image-layered"
            and _pipe_accepts_kwarg(pipe, "resolution")
        ):
            call_kwargs.pop("width", None)
            call_kwargs.pop("height", None)
            call_kwargs.setdefault("resolution", min(resolved_width, resolved_height))
            call_kwargs.setdefault("layers", 4)
            call_kwargs.setdefault("cfg_normalize", True)
            call_kwargs.setdefault("use_en_prompt", True)

        try:
            out = pipe(**call_kwargs)
            images = _extract_pipeline_images(out)
        finally:
            if drain_cache_after_generation:
                _drain_cuda_cache()
        if not images:
            raise RuntimeError("Diffusion pipeline returned no images.")
        if return_all_images:
            return images
        return images[0]

    def generate_images_with_metadata(
        self,
        **kwargs: Any,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Generate one or more images and snapshot producer metadata.

        Most pipelines return one image, but QwenImageLayeredPipeline
        returns a list of RGBA layers nested under `.images[0]`. Keep a
        list-returning API in the backend so the route can expose those
        layers without special-casing the pipeline class.
        """
        with self._generate_lock:
            images = self._generate_image_unlocked(
                **kwargs,
                return_all_images = True,
            )
            with self._lock:
                meta = {
                    "model": _display_repo_id(self._repo_id),
                    "family": self._family.name if self._family else None,
                    "output_count": len(images),
                }
        return images, meta

    def generate_image_with_metadata(
        self,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate a single image AND snapshot its identifying metadata.

        Returns ``(pil_image, {"model": <repo_id>, "family": <name>})``
        where the metadata reflects the pipeline that produced the
        image. Snapshotted under ``_generate_lock + _lock`` so a
        queued unload / load that promotes a different pipeline
        cannot replace ``self._repo_id`` / ``self._family`` between
        the forward returning and the route reading status (round
        13 P2 #9). The route uses these values directly in the
        response instead of re-calling ``status()``.
        """
        with self._generate_lock:
            image = self._generate_image_unlocked(**kwargs)
            with self._lock:
                # Round 16 P1 #6: route ``model`` through
                # _display_repo_id so a generation response for a
                # locally-loaded model cannot echo back an absolute
                # filesystem path to the browser.
                meta = {
                    "model": _display_repo_id(self._repo_id),
                    "family": self._family.name if self._family else None,
                }
        return image, meta

    def generate_video_with_metadata(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate video frames and snapshot producer metadata.

        This is intentionally family-metadata-driven rather than LTX/Wan
        route-specific. The family registry owns dimensions, frame
        counts, and guidance defaults; this method only maps that
        normalized request into the currently loaded pipeline signature.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")

        import torch

        with self._generate_lock:
            with self._lock:
                if self._pipe is None:
                    raise RuntimeError("No diffusion model is loaded.")
                pipe = self._pipe
                fam = self._family
                base_repo_variant = self._base_repo_variant
                device = self._device or "cpu"
                cpu_offload_enabled = self._cpu_offload_enabled
            if fam is None or fam.media_kind != "video":
                raise RuntimeError(
                    f"{fam.name if fam is not None else 'This diffusion model'} "
                    "is not a video generation family."
                )

            defaults = _sampling_defaults_for_loaded_pipeline(
                pipe,
                fam,
                base_repo_variant = base_repo_variant,
            )
            resolved_steps = int(
                num_inference_steps
                if num_inference_steps is not None
                else defaults.default_steps
            )
            resolved_guidance = float(
                guidance_scale
                if guidance_scale is not None
                else defaults.default_guidance_scale
            )
            resolved_width = int(width if width is not None else fam.default_width)
            resolved_height = int(height if height is not None else fam.default_height)
            resolved_frames = int(
                num_frames
                if num_frames is not None
                else (fam.default_num_frames or 1)
            )
            resolved_frame_rate = float(
                frame_rate
                if frame_rate is not None
                else (fam.default_frame_rate or 16.0)
            )
            if resolved_steps < 1 or resolved_steps > 200:
                raise ValueError("num_inference_steps must be in [1, 200]")
            if (
                resolved_width <= 0
                or resolved_height <= 0
                or resolved_width > 2048
                or resolved_height > 2048
            ):
                raise ValueError("width and height must be in (0, 2048]")
            if resolved_width % 8 or resolved_height % 8:
                raise ValueError("width and height must be multiples of 8")
            if resolved_frames < 1 or resolved_frames > 513:
                raise ValueError("num_frames must be in [1, 513]")
            if resolved_frame_rate <= 0 or resolved_frame_rate > 240:
                raise ValueError("frame_rate must be in (0, 240]")

            generator = None
            if seed is not None:
                if cpu_offload_enabled:
                    gen_device = "cpu"
                else:
                    gen_device = (
                        "cuda"
                        if device == "cuda" and torch.cuda.is_available()
                        else "cpu"
                    )
                generator = torch.Generator(device = gen_device).manual_seed(int(seed))

            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": resolved_steps,
                "width": resolved_width,
                "height": resolved_height,
                "num_frames": resolved_frames,
            }
            if defaults.default_call_kwargs:
                call_kwargs.update(
                    {
                        key: value
                        for key, value in defaults.default_call_kwargs.items()
                        if _pipe_accepts_kwarg(pipe, key)
                    }
                )
            call_kwargs.update(_video_family_call_defaults(fam))
            if _pipe_accepts_kwarg(pipe, "guidance_scale"):
                call_kwargs["guidance_scale"] = resolved_guidance
            if guidance_scale_2 is not None and _pipe_accepts_kwarg(
                pipe,
                "guidance_scale_2",
            ):
                call_kwargs["guidance_scale_2"] = float(guidance_scale_2)
            if _pipe_accepts_kwarg(pipe, "frame_rate"):
                call_kwargs["frame_rate"] = resolved_frame_rate
            effective_negative_prompt = negative_prompt
            if (
                effective_negative_prompt is None
                and "negative_prompt" not in call_kwargs
                and fam.default_negative_prompt is not None
            ):
                effective_negative_prompt = fam.default_negative_prompt
            if effective_negative_prompt is not None and _pipe_accepts_kwarg(
                pipe,
                "negative_prompt",
            ):
                call_kwargs["negative_prompt"] = effective_negative_prompt
            if generator is not None:
                call_kwargs["generator"] = generator

            out = pipe(**call_kwargs)
            video = _extract_pipeline_video(out)
            with self._lock:
                meta = {
                    "model": _display_repo_id(self._repo_id),
                    "family": self._family.name if self._family else None,
                    "width": resolved_width,
                    "height": resolved_height,
                    "num_frames": resolved_frames,
                    "frame_rate": resolved_frame_rate,
                    "num_inference_steps": resolved_steps,
                    "guidance_scale": resolved_guidance,
                    "guidance_scale_2": guidance_scale_2,
                }
            return video, meta


def _pipe_accepts_kwarg(pipe: Any, name: str) -> bool:
    """True if ``pipe.__call__`` advertises a kwarg called ``name``.

    Cheap inspect-based probe so we do not have to maintain a manual
    list of which pipeline classes accept negative_prompt. Returns
    False on any introspection error so callers stay on the safe path.
    """
    import inspect

    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _video_family_call_defaults(fam: DiffusionFamily) -> dict[str, Any]:
    """Return family-specific call kwargs for video pipelines.

    These live behind a helper so imports for optional pipeline utility
    modules happen only when the video family is actually used.
    """
    if fam.name == "ltx2-3-distilled":
        from diffusers.pipelines.ltx2.utils import (
            DEFAULT_NEGATIVE_PROMPT,
            DISTILLED_SIGMA_VALUES,
        )

        return {
            "sigmas": DISTILLED_SIGMA_VALUES,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "output_type": "np",
            "return_dict": False,
        }
    if fam.name == "wan2-2-t2v":
        return {
            "guidance_scale_2": 3.0,
            "negative_prompt": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
            "output_type": "np",
            "return_dict": True,
        }
    return {}


def _family_load_components(
    diffusers_module: Any,
    fam: DiffusionFamily,
    effective_base: str,
    dtype: Any,
    hf_token: Optional[str],
) -> dict[str, Any]:
    """Load auxiliary components that a family needs at pipeline load.

    Most families can use the pipeline repo's default components. Wan is
    the current exception from the official Diffusers recipe: the VAE is
    intentionally loaded in FP32 while the transformer/text stack uses
    the selected runtime dtype.
    """
    if fam.name != "wan2-2-t2v":
        return {}
    autoencoder_cls = getattr(diffusers_module, "AutoencoderKLWan", None)
    if autoencoder_cls is None:
        raise RuntimeError(
            "Wan video generation requires diffusers.AutoencoderKLWan; "
            "upgrade diffusers and retry."
        )
    import torch

    kwargs: dict[str, Any] = {
        "subfolder": "vae",
        "torch_dtype": torch.float32,
    }
    if hf_token:
        kwargs["token"] = hf_token
    return {"vae": autoencoder_cls.from_pretrained(effective_base, **kwargs)}


def _patch_gguf_modules_for_resident_device(
    root: Any,
    resident_device: Any,
    *,
    pin_memory: bool | None = None,
) -> int:
    """Patch GGUF modules with the shared native residency helper."""

    from .gguf_text_encoder import patch_gguf_text_encoder_for_resident_device

    return patch_gguf_text_encoder_for_resident_device(
        root,
        resident_device,
        pin_memory = pin_memory,
    )


def _read_gguf_orig_shape_metadata(reader: Any, tensor_name: str) -> tuple[int, ...] | None:
    """Read optional original-shape metadata without importing text loaders."""

    get_field = getattr(reader, "get_field", None)
    if not callable(get_field) or not tensor_name:
        return None
    field = get_field(f"comfy.gguf.orig_shape.{tensor_name}")
    if field is None:
        return None
    try:
        dims: list[int] = []
        for part_idx in field.data:
            value = field.parts[part_idx]
            if hasattr(value, "item"):
                value = value.item()
            elif isinstance(value, (list, tuple)):
                value = value[0]
                if hasattr(value, "item"):
                    value = value.item()
            else:
                try:
                    value = value[0]
                    if hasattr(value, "item"):
                        value = value.item()
                except Exception:
                    pass
            dims.append(int(value))
        return tuple(dims)
    except Exception:
        return None


def _load_gguf_checkpoint_no_copy(gguf_checkpoint_path: str, return_tensors: bool = False) -> dict[str, Any]:
    """Load a GGUF checkpoint like Diffusers, but without eager data copies.

    Upstream Diffusers currently builds every GGUF tensor with
    ``torch.from_numpy(tensor.data.copy())``. That preserves quantized
    storage, but still performs a full CPU copy of the file before the
    quantizer can replace modules or Studio can pin/offload them. This
    loader keeps the GGUF arrays mmap-backed so transfer and
    dequantization can be deferred to the layer op while retaining
    Diffusers' expected state-dict shape.
    """

    del return_tensors  # Diffusers' implementation accepts but ignores it.

    import warnings

    import torch

    try:
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import (
            GGUFParameter,
            SUPPORTED_GGUF_QUANT_TYPES,
        )
    except Exception as exc:
        raise ImportError(
            "Loading a GGUF checkpoint in PyTorch requires torch, gguf, "
            "and diffusers GGUF quantizer utilities."
        ) from exc

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters: dict[str, Any] = {}
    native_dequant_types = set(
        getattr(getattr(gguf, "quants", None), "_type_traits", {}).keys()
    )
    native_bf16 = getattr(gguf.GGMLQuantizationType, "BF16", None)
    if native_bf16 is not None:
        native_dequant_types.add(native_bf16)
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type
        logical_shape = _read_gguf_orig_shape_metadata(reader, name)
        if logical_shape is None:
            logical_shape = tuple(int(v) for v in reversed(tensor.shape))
        is_gguf_quant = quant_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        )
        if (
            is_gguf_quant
            and quant_type not in SUPPORTED_GGUF_QUANT_TYPES
            and quant_type not in native_dequant_types
        ):
            supported = "\n".join(str(qtype) for qtype in SUPPORTED_GGUF_QUANT_TYPES)
            raise ValueError(
                f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                "\n\nCurrently the following quantization types are supported: \n\n"
                f"{supported}"
                "\n\nTo request support for this quantization type please open an issue "
                "here: https://github.com/huggingface/diffusers"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message = "The given NumPy array is not writable",
            )
            weights = torch.from_numpy(tensor.data)
        if not is_gguf_quant:
            numel = 1
            for dim in logical_shape:
                numel *= dim
            if weights.numel() == numel:
                weights = weights.reshape(logical_shape)
        try:
            # Keep the reader alive for mmap-backed arrays after this
            # local function returns. Tensor subclasses used by Diffusers
            # carry regular Python attrs, and plain tensors also allow
            # attributes in current PyTorch.
            weights._unsloth_gguf_reader = reader
        except Exception:
            pass
        parsed_parameters[name] = (
            GGUFParameter(weights, quant_type = quant_type)
            if is_gguf_quant
            else weights
        )
        if is_gguf_quant:
            try:
                parsed_parameters[name].quant_shape = tuple(logical_shape)
            except Exception:
                pass
        try:
            parsed_parameters[name]._unsloth_gguf_reader = reader
        except Exception:
            pass
    return parsed_parameters


@contextlib.contextmanager
def _patch_diffusers_gguf_checkpoint_loader_no_copy():
    """Temporarily route Diffusers single-file GGUF loads through no-copy IO."""

    try:
        import diffusers.models.model_loading_utils as model_loading_utils
    except Exception:
        yield
        return

    original = getattr(model_loading_utils, "load_gguf_checkpoint", None)
    if original is None:
        yield
        return
    model_loading_utils.load_gguf_checkpoint = _load_gguf_checkpoint_no_copy
    try:
        yield
    finally:
        model_loading_utils.load_gguf_checkpoint = original


def _transformer_supports_gguf_state_dict_load(transformer_cls: Any) -> bool:
    return bool(
        hasattr(transformer_cls, "load_config")
        and hasattr(transformer_cls, "from_config")
    )


def _set_module_tensor_no_copy(root: Any, tensor_name: str, value: Any, dtype: Any) -> None:
    """Attach a checkpoint tensor to an already-created module.

    This mirrors the small assignment part of Diffusers' quantized loader, but
    keeps it local so model classes without ``from_single_file`` can still
    share Studio's GGUF lazy/offload path.
    """

    import torch

    module, leaf = _module_and_leaf(root, tensor_name)
    if leaf in getattr(module, "_parameters", {}):
        if hasattr(value, "quant_type"):
            parameter = value.to("cpu")
            try:
                parameter.requires_grad_(False)
            except Exception:
                pass
        else:
            tensor = value.detach()
            if tensor.is_floating_point() and isinstance(dtype, torch.dtype):
                tensor = tensor.to(dtype = dtype)
            parameter = torch.nn.Parameter(tensor.contiguous(), requires_grad = False)
        module._parameters[leaf] = parameter
        return
    if leaf in getattr(module, "_buffers", {}):
        tensor = value.to("cpu") if hasattr(value, "to") else value
        if (
            hasattr(tensor, "is_floating_point")
            and tensor.is_floating_point()
            and isinstance(dtype, torch.dtype)
        ):
            tensor = tensor.to(dtype = dtype)
        module._buffers[leaf] = tensor
        return
    raise RuntimeError(f"{tensor_name} is not a parameter or buffer on {type(module).__name__}.")


def _load_transformer_gguf_from_state_dict(
    transformer_cls: Any,
    gguf_path: str,
    *,
    base_repo: str,
    dtype: Any,
    quant_config: Any,
    token: Optional[str],
) -> Any:
    """Load a GGUF transformer for classes that lack ``from_single_file``.

    ERNIE's current Diffusers implementation has a normal config/from_config
    path but no single-file GGUF convenience method. The GGUF files already use
    the Diffusers state-dict keyspace, so the clean fallback is:

    1. read the GGUF state dict with Studio's mmap/no-copy reader;
    2. instantiate the transformer on meta tensors from the official config;
    3. ask Diffusers' GGUF quantizer to replace matching linear modules;
    4. attach GGUFParameter objects directly, preserving quant metadata.
    """

    if not _transformer_supports_gguf_state_dict_load(transformer_cls):
        raise RuntimeError(
            f"{transformer_cls.__name__} cannot be constructed from a Diffusers config."
        )

    try:
        from accelerate import init_empty_weights
        from diffusers.quantizers.gguf import GGUFQuantizer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Fallback diffusion GGUF loading requires accelerate and Diffusers "
            "GGUFQuantizer support."
        ) from exc

    config_kwargs: dict[str, Any] = {"subfolder": "transformer"}
    if token:
        config_kwargs["token"] = token
    config = transformer_cls.load_config(base_repo, **config_kwargs)

    state_dict = _load_gguf_checkpoint_no_copy(gguf_path)
    with init_empty_weights():
        transformer = transformer_cls.from_config(config)

    quantizer = GGUFQuantizer(quant_config)
    quantizer._process_model_before_weight_loading(
        transformer,
        device_map = None,
        state_dict = state_dict,
        keep_in_fp32_modules = [],
    )
    for name, value in state_dict.items():
        _set_module_tensor_no_copy(transformer, name, value, dtype)
    transformer.requires_grad_(False)
    return transformer


def _materialize_gguf_embedding_parameters(root: Any, dtype: Any = None) -> int:
    """Materialize GGUF ``nn.Embedding`` weights left untouched by Diffusers.

    Diffusers' GGUF quantizer replaces linear layers with GGUF-aware
    modules, but embeddings can remain raw ``GGUFParameter`` instances.
    For BF16 GGUF embeddings the raw storage has byte-shaped trailing
    dimensions, e.g. Qwen-Image-Layered's logical ``(2, 3072)``
    ``addition_t_embedding`` appears as ``(2, 6144)``. Running the
    regular ``nn.Embedding`` forward on that raw parameter then produces
    a width-6144 tensor and breaks the model. In the Diffusers path we
    keep the scope conservative and materialize only embedding weights.
    """

    if not hasattr(root, "modules"):
        return 0

    import torch

    target_dtype = dtype if isinstance(dtype, torch.dtype) else None

    materialized = 0
    for module in root.modules():
        if not isinstance(module, torch.nn.Embedding):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "quant_type"):
            continue
        dense_weight = _dequantize_diffusers_gguf_parameter(weight, target_dtype)
        module.weight = torch.nn.Parameter(
            dense_weight.contiguous(),
            requires_grad = False,
        )
        materialized += 1
    return materialized


def _prepare_gguf_non_linear_parameters(
    root: Any,
    dtype: Any = None,
    *,
    resident_device: Any = None,
) -> dict[str, int]:
    """Prepare non-linear GGUF parameters left raw by Diffusers.

    Upstream Diffusers replaces GGUF linear layers, but other modules can
    retain raw byte-shaped ``GGUFParameter`` weights. Studio handles the
    cases that matter for memory by wrapping quantized ``Embedding``,
    ``Conv2d``, and norm weights lazily where their GGUF logical shapes
    are representable.
    """

    stats = {
        "embeddings_lazy": _replace_gguf_embedding_parameters(
            root,
            dtype,
            resident_device = resident_device,
        ),
        "conv2d": _replace_gguf_conv2d_parameters(
            root,
            dtype,
            resident_device = resident_device,
        ),
        "norms_lazy": _replace_gguf_norm_parameters(
            root,
            dtype,
            resident_device = resident_device,
        ),
    }
    return {key: value for key, value in stats.items() if value}


def _diffusers_gguf_fused_cuda_available() -> bool:
    try:
        from diffusers.quantizers.gguf import utils as gguf_utils
    except Exception:
        return False
    return getattr(gguf_utils, "ops", None) is not None


def _detect_gguf_execution_backend(device: str | None) -> str:
    if device != "cuda":
        return "torch_dequant"
    return (
        "diffusers_fused_cuda"
        if _diffusers_gguf_fused_cuda_available()
        else "torch_dequant_cuda"
    )


def _replace_diffusers_gguf_linear_parameters(
    root: Any,
    dtype: Any = None,
    *,
    resident_device: Any = None,
) -> int:
    """Replace upstream GGUFLinear modules when no fused CUDA op is present.

    Diffusers' fallback GGUFLinear path dequantizes with regular Torch ops,
    and Studio's CPU-resident wrapper has to swap temporary GGUFParameter
    objects into the module before every forward.  Our lazy module stores the
    same packed bytes as buffers, so CPU-resident execution keeps the
    native transfer/dequant contract without per-forward parameter
    mutation.  When Diffusers' fused CUDA extension is available, keep the
    upstream module so the fused matmul path remains usable.
    """

    if _diffusers_gguf_fused_cuda_available() or not hasattr(root, "named_modules"):
        return 0

    import torch

    try:
        from diffusers.quantizers.gguf.utils import GGUFLinear
        from .gguf_text_encoder import LazyGGUFLinear
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Diffusion GGUF Linear wrapping requires GGUFLinear and LazyGGUFLinear.") from exc

    replacements: list[tuple[str, Any]] = []
    for name, module in root.named_modules():
        if not name or not isinstance(module, GGUFLinear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "quant_type"):
            continue
        replacements.append((name, module))

    replaced = 0
    for name, module in replacements:
        weight = getattr(module, "weight")
        parent, leaf = _module_and_leaf(root, name)
        module_dtype = getattr(module, "compute_dtype", None)
        compute_dtype = (
            module_dtype
            if isinstance(module_dtype, torch.dtype)
            else (dtype if isinstance(dtype, torch.dtype) else torch.float32)
        )
        bias = getattr(module, "bias", None)
        qbias = None
        bias_quant_type = None
        bias_logical_shape = None
        if bias is not None and hasattr(bias, "quant_type"):
            qbias = _plain_tensor_from_gguf_parameter(bias)
            bias_quant_type = getattr(bias, "quant_type")
            bias_logical_shape = _gguf_parameter_logical_shape(bias)
            bias = None
        elif bias is not None:
            bias = bias.detach().to(dtype = compute_dtype)
        logical_shape = _gguf_parameter_logical_shape(weight)
        out_features = int(
            getattr(module, "out_features", logical_shape[0] if logical_shape else 0)
        )
        in_features = int(
            getattr(
                module,
                "in_features",
                logical_shape[1] if logical_shape and len(logical_shape) > 1 else 0,
            )
        )
        lazy = LazyGGUFLinear(
            _plain_tensor_from_gguf_parameter(weight),
            getattr(weight, "quant_type"),
            in_features = in_features,
            out_features = out_features,
            compute_dtype = compute_dtype,
            resident_device = resident_device,
            bias = bias,
            qbias = qbias,
            bias_quant_type = bias_quant_type,
            bias_logical_shape = bias_logical_shape,
        )
        setattr(parent, leaf, lazy)
        replaced += 1
    return replaced


def _plain_tensor_from_gguf_parameter(weight: Any) -> torch.Tensor:
    as_tensor = getattr(weight, "as_tensor", None)
    if callable(as_tensor):
        try:
            return as_tensor().detach()
        except Exception:
            pass
    return weight.detach()


def _gguf_parameter_logical_shape(weight: Any) -> tuple[int, ...] | None:
    quant_shape = getattr(weight, "quant_shape", None)
    if quant_shape is None:
        return None
    try:
        return tuple(int(dim) for dim in quant_shape)
    except Exception:
        return None


def _module_and_leaf(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _replace_gguf_conv2d_parameters(
    root: Any,
    dtype: Any = None,
    *,
    resident_device: Any = None,
) -> int:
    """Replace raw GGUF Conv2d weights with a lazy dequantizing module."""

    if not hasattr(root, "named_modules"):
        return 0

    import torch

    target_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    try:
        from .gguf_text_encoder import LazyGGUFConv2d
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Diffusion GGUF Conv2d wrapping requires LazyGGUFConv2d.") from exc

    replacements: list[tuple[str, torch.nn.Conv2d, Any]] = []
    for name, module in root.named_modules():
        if not name or not isinstance(module, torch.nn.Conv2d):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "quant_type"):
            continue
        replacements.append((name, module, weight))

    replaced = 0
    for name, module, weight in replacements:
        parent, leaf = _module_and_leaf(root, name)
        bias = getattr(module, "bias", None)
        qbias = None
        bias_quant_type = None
        bias_logical_shape = None
        if bias is not None and hasattr(bias, "quant_type"):
            qbias = _plain_tensor_from_gguf_parameter(bias)
            bias_quant_type = getattr(bias, "quant_type")
            bias_logical_shape = _gguf_parameter_logical_shape(bias)
            bias = None
        elif bias is not None:
            bias = bias.detach().to(dtype = target_dtype)
        lazy = LazyGGUFConv2d(
            _plain_tensor_from_gguf_parameter(weight),
            getattr(weight, "quant_type"),
            in_channels = module.in_channels,
            out_channels = module.out_channels,
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            dilation = module.dilation,
            groups = module.groups,
            compute_dtype = target_dtype,
            padding_mode = module.padding_mode,
            resident_device = resident_device,
            bias = bias,
            qbias = qbias,
            bias_quant_type = bias_quant_type,
            bias_logical_shape = bias_logical_shape,
            logical_shape = _gguf_parameter_logical_shape(weight),
        )
        setattr(parent, leaf, lazy)
        replaced += 1
    return replaced


def _replace_gguf_embedding_parameters(
    root: Any,
    dtype: Any = None,
    *,
    resident_device: Any = None,
) -> int:
    """Replace raw GGUF Embedding weights with lazy row-wise dequant modules."""

    if not hasattr(root, "named_modules"):
        return 0

    import torch

    target_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    try:
        from .gguf_text_encoder import LazyGGUFEmbedding
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Diffusion GGUF Embedding wrapping requires LazyGGUFEmbedding.") from exc

    replacements: list[tuple[str, torch.nn.Embedding, Any]] = []
    for name, module in root.named_modules():
        if not name or not isinstance(module, torch.nn.Embedding):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "quant_type"):
            continue
        replacements.append((name, module, weight))

    replaced = 0
    for name, module, weight in replacements:
        logical_shape = _gguf_parameter_logical_shape(weight)
        if logical_shape is None or len(logical_shape) != 2:
            # Fallback to the prior correctness path if a future GGUF
            # embedding layout cannot be represented as rows.
            dense_weight = _dequantize_diffusers_gguf_parameter(weight, target_dtype)
            module.weight = torch.nn.Parameter(dense_weight, requires_grad = False)
            replaced += 1
            continue
        parent, leaf = _module_and_leaf(root, name)
        lazy = LazyGGUFEmbedding(
            _plain_tensor_from_gguf_parameter(weight),
            getattr(weight, "quant_type"),
            num_embeddings = int(logical_shape[0]),
            embedding_dim = int(logical_shape[1]),
            compute_dtype = target_dtype,
            resident_device = resident_device,
            padding_idx = module.padding_idx,
            max_norm = module.max_norm,
            norm_type = module.norm_type,
            scale_grad_by_freq = module.scale_grad_by_freq,
            sparse = module.sparse,
        )
        setattr(parent, leaf, lazy)
        replaced += 1
    return replaced


def _dequantize_diffusers_gguf_parameter(weight: Any, dtype: Any = None) -> torch.Tensor:
    import torch

    try:
        from .gguf_text_encoder import _dequantize_gguf_bytes
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Diffusion GGUF parameter materialization requires Studio GGUF helpers.") from exc

    target_dtype = dtype if isinstance(dtype, torch.dtype) else None
    return _dequantize_gguf_bytes(
        _plain_tensor_from_gguf_parameter(weight),
        getattr(weight, "quant_type"),
        dtype = target_dtype,
        logical_shape = _gguf_parameter_logical_shape(weight),
    ).contiguous()


def _replace_gguf_norm_parameters(
    root: Any,
    dtype: Any = None,
    *,
    resident_device: Any = None,
) -> int:
    """Replace raw GGUF norm parameters with lazy dequantizing modules."""

    if not hasattr(root, "named_modules"):
        return 0

    import torch

    target_dtype = dtype if isinstance(dtype, torch.dtype) else None
    compute_dtype = target_dtype or torch.float32
    try:
        from .gguf_text_encoder import LazyGGUFGroupNorm, LazyGGUFLayerNorm
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Diffusion GGUF norm wrapping requires lazy GGUF norm modules.") from exc

    replacements: list[tuple[str, torch.nn.Module]] = []
    for name, module in root.named_modules():
        if not name or not isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            continue
        if not any(
            getattr(getattr(module, parameter_name, None), "quant_type", None) is not None
            for parameter_name in ("weight", "bias")
        ):
            continue
        replacements.append((name, module))

    replaced = 0
    for name, module in replacements:
        parent, leaf = _module_and_leaf(root, name)
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        kwargs = {
            "compute_dtype": compute_dtype,
            "resident_device": resident_device,
            "qweight": (
                _plain_tensor_from_gguf_parameter(weight)
                if weight is not None and hasattr(weight, "quant_type")
                else None
            ),
            "weight_quant_type": (
                getattr(weight, "quant_type")
                if weight is not None and hasattr(weight, "quant_type")
                else None
            ),
            "weight": (
                weight.detach().to(dtype = compute_dtype)
                if weight is not None and not hasattr(weight, "quant_type")
                else None
            ),
            "weight_logical_shape": (
                _gguf_parameter_logical_shape(weight)
                if weight is not None and hasattr(weight, "quant_type")
                else None
            ),
            "qbias": (
                _plain_tensor_from_gguf_parameter(bias)
                if bias is not None and hasattr(bias, "quant_type")
                else None
            ),
            "bias_quant_type": (
                getattr(bias, "quant_type")
                if bias is not None and hasattr(bias, "quant_type")
                else None
            ),
            "bias": (
                bias.detach().to(dtype = compute_dtype)
                if bias is not None and not hasattr(bias, "quant_type")
                else None
            ),
            "bias_logical_shape": (
                _gguf_parameter_logical_shape(bias)
                if bias is not None and hasattr(bias, "quant_type")
                else None
            ),
        }
        if isinstance(module, torch.nn.LayerNorm):
            lazy = LazyGGUFLayerNorm(
                normalized_shape = module.normalized_shape,
                eps = module.eps,
                **kwargs,
            )
        else:
            lazy = LazyGGUFGroupNorm(
                num_groups = module.num_groups,
                num_channels = module.num_channels,
                eps = module.eps,
                **kwargs,
            )
        setattr(parent, leaf, lazy)
        replaced += 1
    return replaced


def _materialize_gguf_norm_parameters(root: Any, dtype: Any = None) -> int:
    """Backward-compatible alias for older focused tests/helpers."""

    return _replace_gguf_norm_parameters(root, dtype)


def _extract_pipeline_images(out: Any) -> list[Any]:
    """Flatten common diffusers image output shapes.

    Standard image pipelines return `out.images == [PIL.Image]`.
    QwenImageLayeredPipeline returns layer groups, e.g.
    `out.images == [[rgba_layer_0, rgba_layer_1, ...]]`. Keep this
    deliberately structural so future multi-image pipelines work
    without a family-specific branch.
    """
    images = getattr(out, "images", None) or []
    flattened: list[Any] = []
    for item in images:
        if isinstance(item, (list, tuple)):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _extract_pipeline_video(out: Any) -> Any:
    """Extract frames from common diffusers video output shapes."""
    if isinstance(out, tuple):
        candidate = out[0] if out else None
    else:
        candidate = getattr(out, "frames", None)
        if candidate is None:
            candidate = getattr(out, "videos", None)
    if candidate is None:
        raise RuntimeError(f"Diffusion video pipeline returned no video frames.")
    ndim = getattr(candidate, "ndim", None)
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise RuntimeError("Diffusion video pipeline returned no video frames.")
        return candidate[0]
    if ndim == 5:
        return candidate[0]
    return candidate


def encode_png_base64(pil_image: "Any") -> str:
    """Encode a PIL image to base64-encoded PNG."""
    import base64

    buf = io.BytesIO()
    pil_image.save(buf, format = "PNG", optimize = True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_mp4_base64(frames: "Any", *, fps: float) -> str:
    """Encode video frames to a base64 MP4."""
    import base64
    import tempfile
    from diffusers.utils import export_to_video

    tmp_root = Path.cwd() / "temp" / "diffusion_videos"
    tmp_root.mkdir(parents = True, exist_ok = True)
    path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix = ".mp4",
            dir = str(tmp_root),
            delete = False,
        ) as handle:
            path = Path(handle.name)
        export_to_video(frames, str(path), fps = int(round(fps)))
        return base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        if path is not None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


# ─── Helpers ──────────────────────────────────────────────────────────


def _raise_if_helper_advisor_busy_for_diffusion(
    *,
    publish_pending: bool = False,
    ignore_pending_workload: Optional[str] = None,
) -> bool:
    """Round 29 P1 #1: split the helper-busy check out of
    _release_chat_backend_for_diffusion so the diffusion load can
    check ALL conflicts (helper, training, export) BEFORE doing ANY
    destructive unloads. Otherwise a route-precheck race or a direct
    backend call would unload the user's chat while training was
    active, then 409 with the user holding no model at all.

    Round 32 P1 #3: when ``publish_pending=True`` also takes
    ``_HELPER_ADVISOR_START_LOCK`` and publishes a
    ``diffusion-backend`` public-load pending entry so a concurrent
    AI Assist helper / advisor start that wins the start lock sees
    the pending public owner and refuses VRAM. The route layer
    publishes its own ``diffusion`` tag (refcount semantics, so the
    two publishes coexist without erasing each other). Returns True
    when a pending entry was actually published so the caller can
    pair it with ``_clear_diffusion_backend_pending`` in finally.
    Direct callers (tests, scripts) opt in with ``publish_pending=
    True`` to get the same atomic check + publish the route gets.
    The ``check_helper_advisor`` callback in
    ``_release_chat_backend_for_diffusion`` keeps the default False
    so legacy callers do not double-publish or leak pending entries.
    """
    try:
        from utils.datasets.llm_assist import (
            _HELPER_ADVISOR_START_LOCK,
            _publish_public_load_pending,
            helper_advisor_busy,
            public_load_pending,
        )
    except Exception:
        return False
    with _HELPER_ADVISOR_START_LOCK:
        if helper_advisor_busy():
            raise RuntimeError(
                "AI Assist (helper / advisor GGUF) is still using the GPU. "
                "Wait for it to finish before loading a diffusion image model."
            )
        # Round 38 P1: mirror the route-side _raise_if_helper_advisor_busy
        # public_load_pending parity check. When publishing, refuse if
        # ANOTHER public workload is already mid-handoff. Route-wrapped
        # calls pass ignore_pending_workload="diffusion" so the
        # route's own publish (which happened just before
        # backend.load_model) does not cause the backend's atomic
        # check to self-block.
        if publish_pending and public_load_pending(excluding = ignore_pending_workload):
            raise RuntimeError(
                "Another GPU workload is mid-handoff. Wait for it to "
                "finish before loading a diffusion image model."
            )
        if publish_pending:
            _publish_public_load_pending("diffusion-backend")
            return True
    return False


def _clear_diffusion_backend_pending() -> None:
    """Round 32 P1 #3: paired clear for
    ``_raise_if_helper_advisor_busy_for_diffusion(publish_pending=True)``.
    Safe to call when the helpers module is unavailable (no-op)."""
    try:
        from utils.datasets.llm_assist import _release_public_load_pending
    except Exception:
        return
    try:
        _release_public_load_pending("diffusion-backend")
    except Exception:
        pass


_MISSING_BACKEND = object()


def _existing_runtime_backend(
    module_names: tuple[str, ...],
    *,
    singleton_name: str,
    getter_name: str,
) -> Any:
    """Return an already-created Studio runtime backend, if one exists.

    Diffusion loads need to release active GPU owners, but merely
    checking that state should not create training/export/chat
    orchestrators. Those constructors can start subprocess-side setup
    and cache/model-list work that is unrelated to image loading.
    """

    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        backend = getattr(module, singleton_name, _MISSING_BACKEND)
        if backend is not _MISSING_BACKEND:
            if backend is not None:
                return backend
            continue
        getter = getattr(module, getter_name, None)
        if callable(getter) and not hasattr(module, "__path__"):
            return getter()
    return None


def _release_chat_backend_for_diffusion(*, check_helper_advisor: bool = True) -> None:
    """Unload any running chat backend before a diffusion load.

    Diffusion pipelines on FLUX-class models can eat 12-24 GB of VRAM,
    and the chat backends (llama-server for GGUF, the safetensors
    Inference orchestrator for HF / Unsloth) typically hold onto their
    loaded weights until told to drop them. Asking both to release
    their weights first means a typical 24 GB consumer GPU can host
    one chat model OR one diffusion model without manual unload steps.

    A missing chat backend module is a silent no-op (fresh install /
    no GGUF use). An unload that ACTUALLY fails (raises or leaves
    the backend resident) raises ``RuntimeError`` so the surrounding
    diffusion ``load_model`` bails out instead of double-owning VRAM
    (round 17 P1 #2).
    """
    # Round 27 P1 #2 / round 29 P1 #1: helper / advisor GGUF loads
    # run on a PRIVATE LlamaCppBackend so the global llama check below
    # cannot see them. The actual busy check now lives in
    # _raise_if_helper_advisor_busy_for_diffusion so the caller can do
    # ALL conflict checks BEFORE any destructive unload. Kept here as
    # a default-on safety net for callers that did not run the
    # standalone check.
    if check_helper_advisor:
        _raise_if_helper_advisor_busy_for_diffusion()
    # 1. GGUF chat backend (llama-server subprocess). We unload when
    #    EITHER is_loaded is True (resident model) OR is_active is
    #    True (mid-download / startup) OR loading_model_identifier is
    #    populated (HF GGUF download in progress, before is_active /
    #    is_loaded flip). The last case is what round 13 P1 #8 flagged.
    backend = _existing_runtime_backend(
        ("routes.inference",),
        singleton_name = "_llama_cpp_backend",
        getter_name = "get_llama_cpp_backend",
    )
    if backend is None:
        logger.debug("llama-server backend is not initialized before diffusion load")
    else:
        is_loaded = bool(getattr(backend, "is_loaded", False))
        is_active = bool(getattr(backend, "is_active", False))
        is_loading = bool(getattr(backend, "loading_model_identifier", None))
        if is_loaded or is_active or is_loading:
            logger.info(
                "Unloading llama-server (loaded=%s active=%s loading=%s) before diffusion load",
                is_loaded,
                is_active,
                is_loading,
            )
            try:
                ok = backend.unload_model()
            except Exception as exc:
                raise RuntimeError(
                    "Could not unload the existing GGUF chat model before "
                    "loading a diffusion image model."
                ) from exc
            # Round 28 P1 #12: a cancelled pending GGUF download takes
            # up to a few seconds to clear loading_model_identifier in
            # its finally block. Wait briefly so the same retryable
            # cancel path used by the unload route does not 503 us.
            deadline = time.monotonic() + 5.0
            while (
                getattr(backend, "loading_model_identifier", None)
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            # Round 18 P1 #4: also reject when ``loading_model_identifier``
            # is still set after the unload call. Without this, a GGUF
            # download / startup that was already in flight before the
            # diffusion handoff (and which never flipped is_active to
            # True before the unload landed) keeps allocating into VRAM
            # while diffusion proceeds, double-owning the GPU.
            if (
                ok is False
                or getattr(backend, "is_loaded", False)
                or getattr(backend, "is_active", False)
                or getattr(backend, "loading_model_identifier", None)
            ):
                raise RuntimeError(
                    "The existing GGUF chat model is still active or loading "
                    "after unload; retry before loading a diffusion image model."
                )

    # 2. Safetensors / HF chat backend (the InferenceOrchestrator that
    #    serves FastVisionModel / FastLanguageModel weights). When this
    #    backend has a model resident on the same GPU, a diffusion load
    #    will OOM the same way. We also flush any loading_models set so
    #    a chat load that is mid-download cannot race the diffusion
    #    allocation.
    backend = _existing_runtime_backend(
        ("core.inference", "core.inference.orchestrator"),
        singleton_name = "_inference_backend",
        getter_name = "get_inference_backend",
    )
    if backend is None:
        logger.debug("safetensors backend is not initialized before diffusion load")
        return

    active_model_name = getattr(backend, "active_model_name", None)
    loading_models = set(getattr(backend, "loading_models", set()) or set())

    def _require_unload(model_name: str) -> None:
        try:
            ok = backend.unload_model(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not unload safetensors chat model '{model_name}' "
                "before loading a diffusion image model."
            ) from exc
        if ok is False:
            raise RuntimeError(
                f"Safetensors backend refused to unload '{model_name}' "
                "before loading a diffusion image model."
            )
        # Round 19 P1 #2: per-name post-state check. ``unload_model``
        # returning ``True`` does not guarantee the orchestrator
        # actually dropped the weights; the worker may have responded
        # while still holding them, or a concurrent ``load`` may have
        # repopulated the tracker. Verify the specific name is gone
        # so the surrounding diffusion load bails out instead of
        # silently double-owning VRAM.
        active_after = getattr(backend, "active_model_name", None)
        loading_after = set(getattr(backend, "loading_models", set()) or set())
        if active_after == model_name or model_name in loading_after:
            raise RuntimeError(
                f"Safetensors chat model '{model_name}' is still active "
                "or loading after unload; retry before loading a diffusion image model."
            )

    if active_model_name:
        logger.info(
            "Unloading safetensors chat backend '%s' before diffusion load",
            active_model_name,
        )
        _require_unload(active_model_name)
    for loading in loading_models:
        if loading == active_model_name:
            continue
        logger.info(
            "Unloading in-flight safetensors chat load '%s' before diffusion",
            loading,
        )
        _require_unload(loading)

    # Round 21 P1 #5: final sweep without the owned_names filter.
    # A concurrent ``/load`` that appeared AFTER the initial
    # snapshot was previously ignored, so a chat model that started
    # loading during the diffusion handoff slipped through and
    # raced the diffusion allocation for VRAM. Treat ANY surviving
    # active / loading entry as a failure so the surrounding
    # load_model raises and the caller retries.
    remaining_loading = set(getattr(backend, "loading_models", set()) or set())
    remaining_active = getattr(backend, "active_model_name", None)
    if remaining_loading or remaining_active:
        raise RuntimeError(
            "A safetensors chat model is still active or loading "
            "after unload; retry before loading a diffusion image model."
        )


def _release_other_gpu_owners_for_diffusion() -> None:
    """Best-effort: shut down export subprocess + active training before
    a diffusion load. Both can hold multi-GB of VRAM and would OOM the
    diffusion allocation on consumer GPUs."""
    # Export resident checkpoint. We tear down a SETTLED export
    # (current_checkpoint populated AND is_export_active() False)
    # because that means the export ran to completion and the user
    # can re-load the result. An in-flight export job
    # (is_export_active() True) is NEVER touched here: terminating
    # it would corrupt the user's partial output artifact.
    #
    # The route layer also rejects /images/load with HTTP 409 via
    # _raise_if_export_active when is_export_active() is True. This
    # helper repeats the local check anyway so that direct backend
    # callers (tests, scripts, future routes that forget the
    # higher-level guard) cannot still kill an active export.
    # Training-active check runs FIRST so direct backend callers
    # (tests, scripts, future routes) cannot bypass the route layer's
    # 409 by calling ``load_model`` directly while a training run is
    # active (round 15 P1 #3). The route layer's
    # ``_raise_if_training_active`` still runs ahead of the load to
    # surface the conflict as 409; this helper re-raises so direct
    # callers see the same RuntimeError the export-active path raises.
    training_backend = _existing_runtime_backend(
        ("core.training", "core.training.training"),
        singleton_name = "_training_backend",
        getter_name = "get_training_backend",
    )
    if training_backend is None:
        logger.debug("training backend is not initialized before diffusion load")
    else:
        try:
            training_active = bool(training_backend.is_training_active())
        except Exception as exc:
            # Unverifiable status -> fail closed (might be active).
            raise RuntimeError(
                "Could not verify training status before loading a "
                "diffusion image model."
            ) from exc
        if training_active:
            raise RuntimeError(
                "Training is currently active. Stop the training run "
                "before loading a diffusion image model."
            )

    exp = _existing_runtime_backend(
        ("core.export", "core.export.orchestrator"),
        singleton_name = "_export_backend",
        getter_name = "get_export_backend",
    )
    if exp is None:
        logger.debug("export backend is not initialized before diffusion load")
        return

    is_export_active_fn = getattr(exp, "is_export_active", None)
    if is_export_active_fn is not None:
        try:
            export_is_active = bool(is_export_active_fn())
        except Exception as exc:
            # Round 16 P2 #8: distinguish unverifiable status from
            # active export. The previous "treat as active" mapping
            # surfaced as a misleading 409 conflict; raise a
            # "Could not verify" RuntimeError so the route layer
            # maps it to 503 (retryable) instead.
            raise RuntimeError(
                "Could not verify export status before loading a "
                "diffusion image model."
            ) from exc
        if export_is_active:
            # Round 14 P2 #10: the prior behaviour logged a warning
            # and continued, so direct ``DiffusionBackend.load_model``
            # callers (tests, scripts) silently bypassed the route
            # layer's 409. Hard-refuse instead so any code path that
            # reaches this helper while an export is active sees the
            # same failure mode the route returns.
            raise RuntimeError(
                "An export job is currently active. Stop the export "
                "job before loading a diffusion image model."
            )

    if getattr(exp, "current_checkpoint", None):
        # Round 18 P1 #2: a wedged ``_shutdown_subprocess`` used to log
        # at debug level and continue, so direct backend callers could
        # allocate diffusion VRAM on top of an export checkpoint that
        # still owned the GPU. Mirror the route-level helper and raise
        # so the surrounding ``load_model`` bails out with a clean
        # RuntimeError that the route layer maps to HTTP 503.
        try:
            logger.info("Shutting down idle export subprocess before diffusion load")
            exp._shutdown_subprocess()
        except Exception as exc:
            raise RuntimeError(
                "Could not unload the idle export checkpoint before "
                "loading a diffusion image model."
            ) from exc
        exp.current_checkpoint = None
        exp.is_vision = False
        exp.is_peft = False

    # Note: active training is *not* stopped here. The route layer
    # (`_raise_if_training_active` in routes/inference.py) refuses
    # /images/load with HTTP 409 before this helper runs, so reaching
    # this point with training still active would only happen in
    # programmatic backend calls (tests, scripts). Silently terminating
    # someone's training run when the diffusion load might still fail
    # is worse than letting the load OOM and surfacing it explicitly.


def _release(obj: Any) -> None:
    """Best-effort GPU-memory release for a pipeline being swapped out.

    Only drops the local reference (which the caller has already
    nulled in its own scope) and runs ``gc.collect()`` so __del__
    fires. Does NOT call ``torch.cuda.empty_cache()`` here because
    when the caller still holds the actual reference in a local /
    attribute, ``empty_cache()`` would run before __del__ released
    the weights and would not actually free GPU memory. Use
    ``_drain_cuda_cache()`` AFTER the last reference has been nulled.
    """
    if obj is None:
        return
    try:
        del obj
    except Exception:
        pass
    gc.collect()


def _drain_cuda_cache() -> None:
    """Hand freed weights back to the active accelerator's allocator.

    Call this AFTER every reference to the freed object has been
    dropped (caller's local + attribute) and a ``gc.collect()`` has
    fired __del__. Calling earlier would empty an already-pinned
    cache and not actually release the memory.

    Handles CUDA *and* MPS (Apple Silicon) so a diffusion swap on
    macOS actually returns VRAM to the Metal allocator.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import torch

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_module = getattr(torch, "mps", None)
            empty_cache = (
                getattr(mps_module, "empty_cache", None) if mps_module else None
            )
            if empty_cache is not None:
                empty_cache()
    except Exception:
        pass


# ─── Module-level singleton ───────────────────────────────────────────


_singleton: Optional[DiffusionBackend] = None
_singleton_lock = threading.Lock()


def get_diffusion_backend() -> DiffusionBackend:
    """Return the process-wide diffusion backend (lazy-instantiated)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DiffusionBackend()
    return _singleton


async def async_generate(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> "Any":
    """Run ``generate_image`` in the default executor so route handlers
    do not block the event loop for the 5-30 s a diffusion step takes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: backend.generate_image(**kwargs))


async def async_generate_with_metadata(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``generate_image_with_metadata`` in the default executor.

    Used by the /images/generate route so the response model / family
    fields reflect the pipeline that actually produced the image, even
    if an unload races the route between the forward returning and the
    response being assembled (round 13 P2 #9)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: backend.generate_image_with_metadata(**kwargs),
    )


async def async_generate_images_with_metadata(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Run ``generate_images_with_metadata`` in the default executor.

    Multi-output pipelines such as Qwen Image Layered return all images
    here, while single-image pipelines return a one-item list.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: backend.generate_images_with_metadata(**kwargs),
    )


async def async_generate_video_with_metadata(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``generate_video_with_metadata`` in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: backend.generate_video_with_metadata(**kwargs),
    )

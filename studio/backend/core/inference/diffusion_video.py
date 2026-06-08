# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video-model helpers for the Studio diffusion backend."""

from __future__ import annotations

import json
import os
from typing import Any, Optional


LTX2_3_OFFICIAL_REPO = "Lightricks/LTX-2.3"
LTX2_3_SPATIAL_UPSCALER_X2_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX2_3_DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LTX2_3_BASE_STAGE_2_LORA_ADAPTER = "ltx2_3_distilled_stage2"
LTX2_3_BASE_STAGE_2_LORA_SCALE = 0.8
LTX2_3_DISTILLED_STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875]
LTX2_3_DISTILLED_STAGE_2_OFFICIAL_SIGMAS = [
    *LTX2_3_DISTILLED_STAGE_2_SIGMAS,
    0.0,
]
LTX2_3_BASE_TWO_STAGE_PROFILE = "official-base-two-stage"
LTX2_3_DISTILLED_TWO_STAGE_PROFILE = "official-distilled-two-stage"


def video_family_call_defaults(fam: Any) -> dict[str, Any]:
    """Return family-specific call kwargs for video pipelines.

    Imports for optional pipeline utility modules happen only when the
    video family is actually used.
    """
    if fam.name == "ltx2-3-base":
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

        return {
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "output_type": "np",
            "return_dict": False,
        }
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
    if fam.name in {"wan2-2-t2v", "wan2-2-i2v", "wan2-2-ti2v-5b"}:
        defaults = {
            "negative_prompt": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
            "output_type": "np",
            "return_dict": True,
        }
        if fam.name in {"wan2-2-t2v", "wan2-2-i2v"}:
            defaults["guidance_scale_2"] = 3.0
        return defaults
    return {}


def family_load_components(
    diffusers_module: Any,
    fam: Any,
    effective_base: str,
    dtype: Any,
    hf_token: Optional[str],
) -> dict[str, Any]:
    """Load auxiliary components that a video family needs at pipeline load."""
    if fam.name not in {"wan2-2-t2v", "wan2-2-i2v", "wan2-2-ti2v-5b"}:
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


def get_ltx2_latent_upsampler(
    backend: Any,
    *,
    dtype: Any,
    device: Any,
    release: Any,
    drain_cuda_cache: Any,
) -> Any:
    """Load and cache the official LTX-2.3 x2 latent upsampler."""

    cache_key = (
        LTX2_3_OFFICIAL_REPO,
        LTX2_3_SPATIAL_UPSCALER_X2_FILENAME,
        str(dtype),
        str(device),
    )
    if (
        backend._ltx2_latent_upsampler is not None
        and backend._ltx2_latent_upsampler_cache_key == cache_key
    ):
        return backend._ltx2_latent_upsampler

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import load_file
    from diffusers.pipelines.ltx2.latent_upsampler import (
        LTX2LatentUpsamplerModel,
    )

    token = os.environ.get("HF_TOKEN") or None
    path = hf_hub_download(
        LTX2_3_OFFICIAL_REPO,
        filename = LTX2_3_SPATIAL_UPSCALER_X2_FILENAME,
        token = token,
    )
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        metadata = handle.metadata() or {}
    config = json.loads(metadata.get("config") or "{}")
    upsampler = LTX2LatentUpsamplerModel(
        in_channels = int(config.get("in_channels", 128)),
        mid_channels = int(config.get("mid_channels", 1024)),
        num_blocks_per_stage = int(config.get("num_blocks_per_stage", 4)),
        dims = int(config.get("dims", 3)),
        spatial_upsample = bool(config.get("spatial_upsample", True)),
        temporal_upsample = bool(config.get("temporal_upsample", False)),
        rational_spatial_scale = float(
            config.get("spatial_scale", config.get("rational_spatial_scale", 2.0))
        ),
        use_rational_resampler = bool(
            config.get("rational_resampler", config.get("use_rational_resampler", True))
        ),
    )
    state_dict = load_file(path, device = "cpu")
    upsampler.load_state_dict(state_dict, strict = True)
    upsampler.to(device = device, dtype = dtype)
    upsampler.eval()

    old = backend._ltx2_latent_upsampler
    backend._ltx2_latent_upsampler = upsampler
    backend._ltx2_latent_upsampler_cache_key = cache_key
    release(old)
    drain_cuda_cache()
    return upsampler


def ensure_ltx2_distilled_lora_adapter(backend: Any, pipe: Any) -> None:
    """Load the official distilled LoRA used by LTX-2.3 stage 2."""

    cache_key = (
        id(pipe),
        LTX2_3_OFFICIAL_REPO,
        LTX2_3_DISTILLED_LORA_FILENAME,
        LTX2_3_BASE_STAGE_2_LORA_ADAPTER,
    )
    if backend._ltx2_distilled_lora_cache_key == cache_key:
        return

    load_lora = getattr(pipe, "load_lora_weights", None)
    if not callable(load_lora):
        raise RuntimeError(
            f"{type(pipe).__name__} does not support loading the official "
            "LTX-2.3 distilled LoRA required for two-stage base video."
        )
    token = os.environ.get("HF_TOKEN") or None
    kwargs: dict[str, Any] = {
        "weight_name": LTX2_3_DISTILLED_LORA_FILENAME,
        "adapter_name": LTX2_3_BASE_STAGE_2_LORA_ADAPTER,
    }
    if token:
        kwargs["token"] = token
    load_lora(LTX2_3_OFFICIAL_REPO, **kwargs)
    backend._ltx2_distilled_lora_cache_key = cache_key


def disable_ltx2_distilled_lora_adapter(pipe: Any) -> None:
    disable_lora = getattr(pipe, "disable_lora", None)
    if callable(disable_lora):
        disable_lora()


def enable_ltx2_distilled_lora_adapter(
    backend: Any,
    pipe: Any,
    *,
    scale: float,
) -> None:
    backend._ensure_ltx2_distilled_lora_adapter(pipe)
    set_adapters = getattr(pipe, "set_adapters", None)
    if not callable(set_adapters):
        raise RuntimeError(
            f"{type(pipe).__name__} cannot activate the official LTX-2.3 "
            "distilled LoRA adapter."
        )
    set_adapters(
        [LTX2_3_BASE_STAGE_2_LORA_ADAPTER],
        adapter_weights = [float(scale)],
    )
    enable_lora = getattr(pipe, "enable_lora", None)
    if callable(enable_lora):
        enable_lora()


def generate_ltx2_base_two_stage_video(
    backend: Any,
    *,
    pipe: Any,
    base_call_kwargs: dict[str, Any],
    resolved_width: int,
    resolved_height: int,
    resolved_steps: int,
    resolved_guidance: float,
    resolved_frame_rate: float,
    resolved_frames: int,
    generator: Any,
    device: str,
    extract_pipeline_video: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run the clean-room LTX-2.3 base two-stage production workflow."""

    if resolved_width % 64 or resolved_height % 64:
        raise ValueError(
            "LTX-2.3 two-stage base generation requires final width "
            "and height to be multiples of 64."
        )

    import torch

    execution_device = getattr(pipe, "_execution_device", None) or device
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        try:
            dtype = next(transformer.parameters()).dtype
        except StopIteration:
            dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    upsampler = backend._get_ltx2_latent_upsampler(
        dtype = dtype,
        device = execution_device,
    )

    stage_1_width = resolved_width // 2
    stage_1_height = resolved_height // 2
    backend._disable_ltx2_distilled_lora_adapter(pipe)
    stage_1_kwargs = dict(base_call_kwargs)
    stage_1_kwargs.update(
        {
            "width": stage_1_width,
            "height": stage_1_height,
            "num_inference_steps": resolved_steps,
            "guidance_scale": resolved_guidance,
            "output_type": "latent",
            "return_dict": False,
        }
    )
    if generator is not None:
        stage_1_kwargs["generator"] = generator

    stage_1_out = pipe(**stage_1_kwargs)
    if not isinstance(stage_1_out, tuple) or len(stage_1_out) < 2:
        raise RuntimeError(
            "LTX-2.3 base stage 1 did not return video and audio latents."
        )
    video_latents, audio_latents = stage_1_out[0], stage_1_out[1]
    upsampler_param = next(upsampler.parameters())
    video_latents = video_latents.to(
        device = upsampler_param.device,
        dtype = upsampler_param.dtype,
    )
    with torch.no_grad():
        upscaled_video_latents = upsampler(video_latents)

    stage_2_kwargs = dict(base_call_kwargs)
    for key in (
        "stg_scale",
        "modality_scale",
        "guidance_rescale",
        "audio_guidance_scale",
        "audio_stg_scale",
        "audio_modality_scale",
        "audio_guidance_rescale",
        "spatio_temporal_guidance_blocks",
        "use_cross_timestep",
    ):
        stage_2_kwargs.pop(key, None)
    stage_2_kwargs.update(
        {
            "width": resolved_width,
            "height": resolved_height,
            "num_frames": resolved_frames,
            "frame_rate": resolved_frame_rate,
            "num_inference_steps": len(LTX2_3_DISTILLED_STAGE_2_SIGMAS),
            "guidance_scale": 1.0,
            "sigmas": LTX2_3_DISTILLED_STAGE_2_SIGMAS,
            "noise_scale": LTX2_3_DISTILLED_STAGE_2_SIGMAS[0],
            "latents": upscaled_video_latents,
            "audio_latents": audio_latents,
            "output_type": "np",
            "return_dict": False,
        }
    )
    if generator is not None:
        stage_2_kwargs["generator"] = generator

    try:
        backend._enable_ltx2_distilled_lora_adapter(
            pipe,
            scale = LTX2_3_BASE_STAGE_2_LORA_SCALE,
        )
        out = pipe(**stage_2_kwargs)
    finally:
        backend._disable_ltx2_distilled_lora_adapter(pipe)
    return extract_pipeline_video(out), {
        "sampling_profile": LTX2_3_BASE_TWO_STAGE_PROFILE,
        "stage_1_width": stage_1_width,
        "stage_1_height": stage_1_height,
        "stage_2_sigmas": list(LTX2_3_DISTILLED_STAGE_2_OFFICIAL_SIGMAS),
        "stage_2_lora": LTX2_3_DISTILLED_LORA_FILENAME,
        "stage_2_lora_scale": LTX2_3_BASE_STAGE_2_LORA_SCALE,
        "latent_upsampler": LTX2_3_SPATIAL_UPSCALER_X2_FILENAME,
    }


def generate_ltx2_distilled_two_stage_video(
    backend: Any,
    *,
    pipe: Any,
    base_call_kwargs: dict[str, Any],
    resolved_width: int,
    resolved_height: int,
    resolved_steps: int,
    resolved_guidance: float,
    resolved_frame_rate: float,
    resolved_frames: int,
    generator: Any,
    device: str,
    extract_pipeline_video: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run the clean-room LTX-2.3 distilled two-stage workflow."""

    if resolved_width % 64 or resolved_height % 64:
        raise ValueError(
            "LTX-2.3 two-stage distilled generation requires final "
            "width and height to be multiples of 64."
        )

    import torch
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    execution_device = getattr(pipe, "_execution_device", None) or device
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        try:
            dtype = next(transformer.parameters()).dtype
        except StopIteration:
            dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    upsampler = backend._get_ltx2_latent_upsampler(
        dtype = dtype,
        device = execution_device,
    )

    stage_1_width = resolved_width // 2
    stage_1_height = resolved_height // 2
    stage_1_kwargs = dict(base_call_kwargs)
    stage_1_kwargs.update(
        {
            "width": stage_1_width,
            "height": stage_1_height,
            "num_inference_steps": resolved_steps,
            "guidance_scale": resolved_guidance,
            "sigmas": DISTILLED_SIGMA_VALUES,
            "output_type": "latent",
            "return_dict": False,
        }
    )
    if generator is not None:
        stage_1_kwargs["generator"] = generator

    stage_1_out = pipe(**stage_1_kwargs)
    if not isinstance(stage_1_out, tuple) or len(stage_1_out) < 2:
        raise RuntimeError(
            "LTX-2.3 distilled stage 1 did not return video and audio latents."
        )
    video_latents, audio_latents = stage_1_out[0], stage_1_out[1]
    upsampler_param = next(upsampler.parameters())
    video_latents = video_latents.to(
        device = upsampler_param.device,
        dtype = upsampler_param.dtype,
    )
    with torch.no_grad():
        upscaled_video_latents = upsampler(video_latents)

    stage_2_kwargs = dict(base_call_kwargs)
    stage_2_kwargs.update(
        {
            "width": resolved_width,
            "height": resolved_height,
            "num_frames": resolved_frames,
            "frame_rate": resolved_frame_rate,
            "num_inference_steps": len(LTX2_3_DISTILLED_STAGE_2_SIGMAS),
            "guidance_scale": 1.0,
            "sigmas": LTX2_3_DISTILLED_STAGE_2_SIGMAS,
            "noise_scale": LTX2_3_DISTILLED_STAGE_2_SIGMAS[0],
            "latents": upscaled_video_latents,
            "audio_latents": audio_latents,
            "output_type": "np",
            "return_dict": False,
        }
    )
    if generator is not None:
        stage_2_kwargs["generator"] = generator

    out = pipe(**stage_2_kwargs)
    return extract_pipeline_video(out), {
        "sampling_profile": LTX2_3_DISTILLED_TWO_STAGE_PROFILE,
        "stage_1_width": stage_1_width,
        "stage_1_height": stage_1_height,
        "stage_1_sigmas": list(DISTILLED_SIGMA_VALUES),
        "stage_2_sigmas": list(LTX2_3_DISTILLED_STAGE_2_OFFICIAL_SIGMAS),
        "latent_upsampler": LTX2_3_SPATIAL_UPSCALER_X2_FILENAME,
    }

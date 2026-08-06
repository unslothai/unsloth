# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 helpers shared by the Diffusers and stable-diffusion.cpp paths."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

H3_GGUF_REPO = "leejet/MiniMax-H3-GGUF"
H3_COMPONENT_REPO = "Comfy-Org/MiniMax-H3"
H3_VIDEO_VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "vae/minimax_h3_audio_vae_fp32.safetensors"
H3_QWEN_Q2 = "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
H3_QWEN_Q4 = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"

# Measured with the merged Diffusers T2VA workflow and component-level CPU
# offload. The base is the largest component plus runtime overhead; activation
# memory scales with spatiotemporal volume across the tested 960x544 and
# 1344x768, 124-345 frame matrix. The guard covers allocator variation around
# the measured success and OOM boundaries.
H3_DIFFUSERS_VRAM_BASE_GB = 68.5
H3_DIFFUSERS_VRAM_GB_PER_MPIXEL_FRAME = 0.08


def estimate_h3_diffusers_vram_gb(width: int, height: int, num_frames: int) -> float:
    """Measured available-VRAM floor for an H3 Diffusers generation."""
    volume_mpixel_frames = width * height * num_frames / 1_000_000
    return H3_DIFFUSERS_VRAM_BASE_GB + (
        H3_DIFFUSERS_VRAM_GB_PER_MPIXEL_FRAME * volume_mpixel_frames
    )


def estimate_h3_diffusers_host_ram_gb(available_vram_gb: float) -> float:
    """Host-RAM floor for the offload tier selected at the available VRAM."""
    return 85.0 if available_vram_gb >= 132.0 else 150.0


def is_h3_native(family: Any, kind: str) -> bool:
    return getattr(family, "name", None) == "minimax-h3" and kind == "gguf"


def h3_text_encoder_filename(transformer_filename: str) -> str:
    return H3_QWEN_Q2 if "-q2_" in transformer_filename.lower() else H3_QWEN_Q4


def validate_h3_transformer_filename(filename: str) -> None:
    name = Path(filename).name.lower()
    if not name.startswith("minimax_h3_fl2va") or not name.endswith(".gguf"):
        raise ValueError(
            "MiniMax-H3 text-to-video needs a minimax_h3_fl2va*.gguf transformer. "
            "The Qwen encoder and Ref2VA checkpoints are companion models, not T2VA picks."
        )


def h3_native_hub_files(transformer_filename: str) -> tuple[tuple[str, str], ...]:
    validate_h3_transformer_filename(transformer_filename)
    return (
        (H3_GGUF_REPO, transformer_filename),
        (H3_GGUF_REPO, h3_text_encoder_filename(transformer_filename)),
        (H3_COMPONENT_REPO, H3_VIDEO_VAE),
        (H3_COMPONENT_REPO, H3_AUDIO_VAE),
    )


@dataclass(frozen = True)
class MiniMaxH3NativeRuntime:
    engine: Any
    files: Any
    offload_flags: tuple[str, ...]


def transcode_video_to_mp4(source: Path, *, fps: int) -> bytes:
    """Convert an sd.cpp WebM into gallery-compatible H.264/AAC MP4 bytes."""
    import av
    import numpy as np
    import torch
    from diffusers.utils.export_utils import encode_video

    tmp = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
    tmp.close()
    target = Path(tmp.name)
    try:
        with av.open(str(source)) as src:
            video_frames = [frame.to_ndarray(format = "rgb24") for frame in src.decode(video = 0)]

        audio_waveform = None
        sample_rate = None
        with av.open(str(source)) as src:
            if src.streams.audio:
                source_audio = src.streams.audio[0]
                sample_rate = int(source_audio.codec_context.sample_rate or 32000)
                resampler = av.AudioResampler(format = "fltp", layout = "stereo", rate = sample_rate)
                chunks = []
                for frame in src.decode(audio = 0):
                    chunks.extend(resampled.to_ndarray() for resampled in resampler.resample(frame))
                chunks.extend(resampled.to_ndarray() for resampled in resampler.resample(None))
                if chunks:
                    audio_waveform = torch.from_numpy(np.concatenate(chunks, axis = 1))

        encode_video(
            torch.from_numpy(np.stack(video_frames)),
            fps,
            str(target),
            audio = audio_waveform,
            audio_sample_rate = sample_rate,
        )
        return target.read_bytes()
    finally:
        target.unlink(missing_ok = True)


def inspect_video(path: Path) -> tuple[int, int, int, bool]:
    """Return width, height, decoded frame count, and audio presence."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        frames = int(stream.frames or 0)
        has_audio = bool(container.streams.audio)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(video = 0))
    return width, height, frames, has_audio

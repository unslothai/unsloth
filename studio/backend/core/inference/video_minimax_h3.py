# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 helpers shared by the Diffusers and stable-diffusion.cpp paths."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Must stay equal to the minimax-h3 family's `gguf_repo`. They are the same one-click pick, and
# main's test_curated_gguf_repos_are_unsloth_mirrors only checks the family field, so a divergence
# here would let that test pass while the actual download still came from a community repack.
# tests/test_video_backend.py::test_the_h3_native_repo_matches_the_family_gguf_repo pins the pair.
# The mirror carries the Qwen3-VL encoder quants as well as the denoisers, so this repo alone
# satisfies h3_native_hub_files' first two entries.
H3_GGUF_REPO = "unsloth/MiniMax-H3-GGUF"
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


def h3_download_error(repo_id: str, filename: str, exc: Exception) -> Exception:
    """Turn a Hub download failure on an H3 component into something a user can act on.

    The Hub says "Repository Not Found ... make sure you are authenticated" for a repo that is
    private or gated as well as for one that genuinely does not exist. For H3 that message is
    actively misleading in both directions: the mirror is real, and the user's own token is
    usually fine. Name the repo, say which of the four components it was, and say what to do.

    Returns the exception to raise (never raises), so the caller keeps ``raise ... from exc`` and
    the original traceback survives. Anything that is not a recognised access error is passed back
    unchanged rather than reworded, so a timeout or a disk-full still reads as itself.
    """
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if not isinstance(exc, (RepositoryNotFoundError, GatedRepoError)):
        return exc

    role = {
        H3_VIDEO_VAE: "video VAE",
        H3_AUDIO_VAE: "audio VAE",
        H3_QWEN_Q2: "text encoder",
        H3_QWEN_Q4: "text encoder",
    }.get(filename, "denoiser")
    gated = isinstance(exc, GatedRepoError)
    detail = (
        "accept its licence on the Hub, then set a token in Settings"
        if gated
        else "it may be private or not published yet, or your token may not cover it"
    )
    return RuntimeError(
        f"MiniMax-H3 could not download its {role} ({filename}) from {repo_id}: {detail}. "
        f"The other H3 components are unaffected."
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
    """Convert an sd.cpp WebM into a gallery-compatible H.264/AAC MP4.

    The native backend is available in Studio's no-torch runtime, so keep this
    export entirely in PyAV rather than routing decoded frames through Diffusers.
    """
    import av

    tmp = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
    tmp.close()
    target = Path(tmp.name)
    try:
        with av.open(str(source)) as src, av.open(str(target), mode = "w", format = "mp4") as dst:
            source_video = src.streams.video[0]
            output_video = dst.add_stream("libx264", rate = fps)
            output_video.width = int(source_video.codec_context.width)
            output_video.height = int(source_video.codec_context.height)
            output_video.pix_fmt = "yuv420p"

            source_audio = src.streams.audio[0] if src.streams.audio else None
            output_audio = None
            audio_resampler = None
            if source_audio is not None:
                sample_rate = int(source_audio.codec_context.sample_rate or 32000)
                output_audio = dst.add_stream("aac", rate = sample_rate)
                output_audio.layout = "stereo"
                audio_resampler = av.AudioResampler(
                    format = "fltp", layout = "stereo", rate = sample_rate
                )

            selected = (source_video,) if source_audio is None else (source_video, source_audio)
            for packet in src.demux(*selected):
                for frame in packet.decode():
                    if packet.stream.type == "video":
                        for encoded in output_video.encode(frame):
                            dst.mux(encoded)
                    elif output_audio is not None and audio_resampler is not None:
                        for resampled in audio_resampler.resample(frame):
                            for encoded in output_audio.encode(resampled):
                                dst.mux(encoded)

            if output_audio is not None and audio_resampler is not None:
                for resampled in audio_resampler.resample(None):
                    for encoded in output_audio.encode(resampled):
                        dst.mux(encoded)
            for encoded in output_video.encode():
                dst.mux(encoded)
            if output_audio is not None:
                for encoded in output_audio.encode():
                    dst.mux(encoded)
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

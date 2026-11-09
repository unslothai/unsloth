# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frame-threaded x264 encode (SLICE stalls each frame); UNSLOTH_STUDIO_VIDEO_FRAME_THREADS=0 reverts."""

from __future__ import annotations

import functools
import os
import platform
from typing import Any

FRAME_THREADS_ENV = "UNSLOTH_STUDIO_VIDEO_FRAME_THREADS"
# Kill switch: "0" lets libx264 use its AVX-512 kernels again (output then depends on uninitialised heap memory).
X264_ASM_CAP_ENV = "UNSLOTH_STUDIO_VIDEO_X264_ASM_CAP"


def frame_threads_enabled() -> bool:
    return os.environ.get(FRAME_THREADS_ENV, "").strip().lower() not in ("0", "false", "no", "off")


@functools.lru_cache(maxsize = 1)
def _cpu_runs_x264_avx512() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64", "x64"):
        return False
    try:
        import numpy as np
        core = getattr(np, "_core", None) or getattr(np, "core")
        features = core._multiarray_umath.__cpu_features__
    except Exception:  # noqa: BLE001 - unknown CPU: keep libx264's own dispatch
        return False
    return all(features.get(name) for name in ("AVX512_SKX", "AVX2", "FMA3"))


def x264_options() -> dict:
    """libx264 options for both encode paths: its AVX-512 kernels (core 165, PyAV) read uninitialised heap memory, so
    later encodes in a process are not reproducible; capping at AVX2 makes output depend only on the frames."""
    if os.environ.get(X264_ASM_CAP_ENV, "").strip().lower() in ("0", "false", "no", "off"):
        return {}
    return {"x264-params": "asm=AVX2"} if _cpu_runs_x264_avx512() else {}


def add_x264_stream(container: Any, fps: int, width: int, height: int) -> Any:
    stream = container.add_stream("libx264", rate = int(fps), options = x264_options() or None)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    stream.codec_context.thread_type = "FRAME"
    return stream


def encode_x264(
    video: Any,
    fps: int,
    path: str,
    audio: Any = None,
    audio_sample_rate: Any = None,
) -> bool:
    """Write ``path`` with frame-threaded libx264; False (caller runs ``encode_video``) when anything is unsure."""
    if not frame_threads_enabled():
        return False
    from .video_nvenc import _uint8_frames

    frames = _uint8_frames(video)
    if frames is None or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        return False
    if audio is not None and audio_sample_rate is None:
        return False
    try:
        import av
        from diffusers.utils.export_utils import _prepare_audio_stream, _write_audio
    except Exception:  # noqa: BLE001 - no PyAV or an older diffusers: the stock exporter decides
        return False
    try:
        with av.open(path, mode = "w", format = "mp4") as container:
            stream = add_x264_stream(container, fps, frames.shape[2], frames.shape[1])
            audio_stream = (
                _prepare_audio_stream(container, audio_sample_rate) if audio is not None else None
            )
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format = "rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                _write_audio(container, audio_stream, audio, audio_sample_rate, av)
        return True
    except Exception:  # noqa: BLE001 - a partial file is overwritten by the stock exporter
        return False

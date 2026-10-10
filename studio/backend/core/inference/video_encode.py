# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusers' ``encode_video`` stream (libx264 defaults, yuv420p, same audio helpers) with x264 frame threads:
PyAV opens codecs with ``thread_type = SLICE``, so every frame waits for its slowest slice.
``UNSLOTH_STUDIO_VIDEO_FRAME_THREADS=0`` restores the diffusers call."""

from __future__ import annotations

import os
from typing import Any

FRAME_THREADS_ENV = "UNSLOTH_STUDIO_VIDEO_FRAME_THREADS"


def frame_threads_enabled() -> bool:
    return os.environ.get(FRAME_THREADS_ENV, "").strip().lower() not in ("0", "false", "no", "off")


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
            stream = container.add_stream("libx264", rate = int(fps))
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_type = "FRAME"
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

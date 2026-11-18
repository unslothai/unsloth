# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frame-threaded x264 encode (SLICE stalls each frame); UNSLOTH_STUDIO_VIDEO_FRAME_THREADS=0 reverts.

``reference=True`` (MiniMax-H3) writes the file the way the reference workflow's video save does: H.264 at CRF 18,
RGB converted with the BT.709 matrix and tagged BT.709 / sRGB transfer / limited range, and the soundtrack as float
AAC."""

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


# The reference save's libx264 quality (its default CRF); libx264's own default is 23.
REFERENCE_X264_CRF = 18
# AVColorSpace / AVColorPrimaries BT.709, AVColorTransferCharacteristic IEC 61966-2-1 (sRGB), AVColorRange MPEG
# (limited): the reference save's tags for sRGB frames
_BT709 = 1
_TRC_SRGB = 13
_RANGE_MPEG = 1


def _tag_bt709(target: Any) -> None:
    # PyAV 15 frames carry only colorspace / color_range; the stream's codec context carries all four
    for name, value in (
        ("color_primaries", _BT709),
        ("color_trc", _TRC_SRGB),
        ("colorspace", _BT709),
        ("color_range", _RANGE_MPEG),
    ):
        try:
            setattr(target, name, value)
        except AttributeError:
            pass


def add_x264_stream(
    container: Any,
    fps: int,
    width: int,
    height: int,
    *,
    reference: bool = False,
) -> Any:
    options = dict(x264_options())
    if reference:
        options["crf"] = str(REFERENCE_X264_CRF)
    stream = container.add_stream("libx264", rate = int(fps), options = options or None)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    stream.codec_context.thread_type = "FRAME"
    if reference:
        _tag_bt709(stream.codec_context)
    return stream


def x264_reformatter(av: Any, *, reference: bool = False) -> Any:
    """One reformatter for a whole encode (``reference`` only): a per-frame ``frame.reformat`` with a colorspace
    rebuilds the swscale context every frame, 2.2x the conversion the encoder does on its own."""
    if not reference:
        return None
    from av.video.reformatter import VideoReformatter

    return VideoReformatter()


def x264_frame(
    av: Any,
    rgb: Any,
    *,
    reference: bool = False,
    reformatter: Any = None,
) -> Any:
    """One uint8 RGB frame for ``add_x264_stream``. ``reference``: converted to yuv420p with the BT.709 matrix and
    tagged, as the reference save does (left to the encoder, swscale would use its BT.601 default, untagged)."""
    frame = av.VideoFrame.from_ndarray(rgb, format = "rgb24")
    if not reference:
        return frame
    if reformatter is None:
        frame = frame.reformat(format = "yuv420p", dst_colorspace = _BT709)
    else:
        frame = reformatter.reformat(frame, format = "yuv420p", dst_colorspace = _BT709)
    _tag_bt709(frame)
    return frame


def write_reference_audio(
    container: Any, stream: Any, samples: Any, sample_rate: int, av: Any
) -> None:
    """The soundtrack as the reference save writes it: the float waveform as one ``fltp`` frame (no int16 step)."""
    import torch

    wave = torch.as_tensor(samples).detach().float().cpu()
    if wave.ndim == 1:
        wave = wave[None]
    if wave.shape[0] not in (1, 2) and wave.shape[-1] in (1, 2):
        wave = wave.T
    if wave.shape[0] == 1:
        # the stream is stereo (diffusers' _prepare_audio_stream); a mono track plays on both channels
        wave = wave.expand(2, -1)
    frame = av.AudioFrame.from_ndarray(wave.contiguous().numpy(), format = "fltp", layout = "stereo")
    frame.sample_rate = int(sample_rate)
    frame.pts = 0
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def encode_x264(
    video: Any,
    fps: int,
    path: str,
    audio: Any = None,
    audio_sample_rate: Any = None,
    *,
    reference: bool = False,
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
            stream = add_x264_stream(
                container,
                fps,
                frames.shape[2],
                frames.shape[1],
                **({"reference": True} if reference else {}),
            )
            audio_stream = (
                _prepare_audio_stream(container, audio_sample_rate) if audio is not None else None
            )
            reformatter = x264_reformatter(av, reference = reference)
            for frame in frames:
                for packet in stream.encode(
                    x264_frame(av, frame, reference = reference, reformatter = reformatter)
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                (write_reference_audio if reference else _write_audio)(
                    container, audio_stream, audio, audio_sample_rate, av
                )
        return True
    except Exception:  # noqa: BLE001 - a partial file is overwritten by the stock exporter
        return False

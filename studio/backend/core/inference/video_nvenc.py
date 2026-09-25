# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in NVENC H.264 for the video mp4 export: ``UNSLOTH_STUDIO_VIDEO_ENCODER=nvenc``.

libx264 (diffusers ``encode_video``) stays the default. NVENC output is a different, larger file (20-29% more bytes at
+0.17 to +0.37 dB PSNR against libx264's defaults with the settings below, on a 960x544x124 H3 clip with audio), so it
is never picked automatically. It pays off only where the CPU is the bottleneck: 3.79 s -> 1.02 s on a 2-vCPU T4 host,
1.59 s -> 0.91 s on a 12-vCPU L4, while a 48-vCPU RTX PRO 6000 host is faster on libx264 (0.54 s vs 0.63 s).

Anything that stops NVENC (no encoder engine: A100 / H100 / B200 class; driver older than the encoder API the PyAV
wheel needs; a container without the video driver capability; PyAV built without it; the GeForce session limit; any
error mid-encode) falls back to libx264 for that clip. Only CUDA builds are tried; ROCm, MPS and CPU never are.
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
from typing import Any, Optional

ENCODER_ENV = "UNSLOTH_STUDIO_VIDEO_ENCODER"
CODEC = "h264_nvenc"
# preset p5 + VBR cq 24 with AQ: the setting closest to libx264's defaults in quality at 2-10x the speed. b=0 lifts
# FFmpeg's 2 Mbit/s default bitrate cap, which otherwise overrides the quality target.
NVENC_OPTIONS = {
    "preset": "p5",
    "tune": "hq",
    "rc": "vbr",
    "cq": "24",
    "b": "0",
    "spatial-aq": "1",
    "temporal-aq": "1",
    "bf": "3",
}
# No NVENC engine at all (NVIDIA's encode support matrix). A failed open costs 2-12 s, so these skip the probe.
_NO_NVENC_GPUS = (
    "A100",
    "A30",
    "A800",
    "H100",
    "H800",
    "H200",
    "H20",
    "B200",
    "B300",
    "GB200",
    "GB300",
    "GH200",
)
# A failed probe may be transient (another process holding the GeForce session limit), so it is retried after this.
_PROBE_RETRY_S = 600.0

_probe_lock = threading.Lock()
_probed: dict = {}


def nvenc_requested() -> bool:
    return os.environ.get(ENCODER_ENV, "").strip().lower() == "nvenc"


def _encode(
    container: Any,
    frames: Any,
    fps: int,
    gpu: int,
    audio: Any = None,
    audio_sample_rate: Any = None,
) -> None:
    import av

    stream = container.add_stream(CODEC, rate = int(fps), options = {**NVENC_OPTIONS, "gpu": str(gpu)})
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    audio_stream = None
    if audio is not None:
        from diffusers.utils.export_utils import _prepare_audio_stream
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format = "rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    if audio_stream is not None:
        from diffusers.utils.export_utils import _write_audio
        _write_audio(container, audio_stream, audio, audio_sample_rate, av)


def _probe(gpu: int) -> bool:
    import av
    import numpy as np
    try:
        with av.open(io.BytesIO(), mode = "w", format = "mp4") as container:
            _encode(container, np.zeros((1, 256, 256, 3), dtype = np.uint8), 24, gpu)
        return True
    except Exception:  # noqa: BLE001 - any failure means "use libx264"
        return False


def nvenc_gpu(device: Any = None, logger: Any = None) -> Optional[int]:
    """The CUDA ordinal to encode on when NVENC was asked for and works here, else None. Probed once per GPU; a failed
    probe is retried after ``_PROBE_RETRY_S``."""
    if not nvenc_requested():
        return None
    try:
        import torch

        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return None
        index = torch.device(device).index if device is not None else None
        if index is None:
            index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
    except Exception:  # noqa: BLE001
        return None
    key = (index, name, os.environ.get("CUDA_VISIBLE_DEVICES"))
    with _probe_lock:
        cached = _probed.get(key)
        if cached is None or (cached[1] is not None and time.monotonic() - cached[1] >= _PROBE_RETRY_S):
            if set(re.split(r"[\s\-_/]+", name.upper())) & set(_NO_NVENC_GPUS):
                ok, failed_at = False, None
            else:
                ok = _probe(index)
                failed_at = None if ok else time.monotonic()
            _probed[key] = (ok, failed_at)
            if logger is not None and (cached is None or cached[0] != ok):
                logger.info(
                    "video.encoder: %s on %s (%s)",
                    "h264_nvenc" if ok else "libx264",
                    name,
                    f"{ENCODER_ENV}=nvenc" + ("" if ok else "; NVENC unavailable, keeping libx264"),
                )
        return index if _probed[key][0] else None


def _uint8_frames(video: Any) -> Any:
    """What diffusers' encode_video would hand the encoder, as an (F, H, W, 3) uint8 array; None when unsure."""
    import numpy as np

    if isinstance(video, list):
        return np.stack([np.asarray(frame) for frame in video]) if video else None
    if isinstance(video, np.ndarray):
        if video.dtype == np.uint8:
            return video
        if not (np.all(video >= 0) and np.all(video <= 1)):
            return None
        return (video * 255).round().astype("uint8")
    try:
        import torch
        if isinstance(video, torch.Tensor) and video.dtype == torch.uint8:
            return video.cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return None


def encode_nvenc(
    video: Any,
    fps: int,
    path: str,
    gpu: int,
    audio: Any = None,
    audio_sample_rate: Any = None,
) -> bool:
    """Write ``path`` with NVENC; False (nothing usable written) when the caller should encode with libx264."""
    frames = _uint8_frames(video)
    if frames is None or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        return False
    if audio is not None and audio_sample_rate is None:
        return False
    try:
        import av
        with av.open(path, mode = "w") as container:
            _encode(container, frames, fps, gpu, audio, audio_sample_rate)
        return True
    except Exception:  # noqa: BLE001 - e.g. the GeForce session limit; the caller re-encodes with libx264
        return False

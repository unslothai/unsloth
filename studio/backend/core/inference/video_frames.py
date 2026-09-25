# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a decoded clip into the uint8 RGB frames the mp4 encoder takes, on the device that decoded it.

A video pipeline's np / pil postprocess copies the float clip to pageable host memory and rounds it on the CPU (PIL
also builds one image per frame), then ``encode_video`` stacks and converts it again: about 1.4-1.9 s at 960x544x124
and 3-7 s at 1344x768x141 on MiniMax-H3. Around the pipeline call, the processor's np / pil request instead runs the
same ``(x * 255).round()`` in float32 on the GPU, slice by slice into pinned host memory, and hands back a CPU uint8
(B, F, H, W, C) tensor, which ``encode_video`` takes as is. The frames are bit-identical: float32 multiply and
round-half-to-even agree between numpy and torch, and the clip is range-checked first. A clip with any value outside
[0, 1] (NaN included), or on a device other than CUDA / CPU, goes through the original postprocess unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

_SLICE_BYTES = 64 << 20


def device_uint8(frames: Any) -> Optional[Any]:
    """(F, C, H, W) float frames in [0, 1] -> CPU uint8 (F, H, W, C); None when the fast path does not apply."""
    import torch

    device = frames.device
    if device.type not in ("cuda", "cpu"):
        return None
    try:
        lo, hi = torch.aminmax(frames)
        if not (bool(lo >= 0) and bool(hi <= 1)):
            return None
    except Exception:  # noqa: BLE001 - empty clip, or a dtype aminmax lacks
        return None
    count, channels, height, width = frames.shape
    host = None
    if device.type == "cuda":
        try:
            host = torch.empty((count, height, width, channels), dtype = torch.uint8, pin_memory = True)
        except Exception:  # noqa: BLE001 - pinned allocation refused (host limits); a pageable copy is still exact
            host = None
    pinned = host is not None
    if host is None:
        host = torch.empty((count, height, width, channels), dtype = torch.uint8)
    step = max(1, _SLICE_BYTES // max(1, channels * height * width * 4))
    for start in range(0, count, step):
        piece = frames[start : start + step].to(torch.float32) * 255
        piece = piece.round_().to(torch.uint8).permute(0, 2, 3, 1)
        host[start : start + step].copy_(piece, non_blocking = pinned)
    if pinned:
        torch.cuda.current_stream(device).synchronize()
    return host


@contextlib.contextmanager
def uint8_video_frames(pipe: Any) -> Iterator[None]:
    """While active, ``pipe.video_processor``'s np / pil ``postprocess_video`` returns CPU uint8 (B, F, H, W, C)."""
    proc = getattr(pipe, "video_processor", None)
    original = getattr(proc, "postprocess_video", None)
    postprocess = getattr(proc, "postprocess", None)
    if not callable(original) or not callable(postprocess):
        yield
        return
    import torch

    had_own = "postprocess_video" in vars(proc)

    def postprocess_video(
        video: Any,
        output_type: str = "np",
        **kwargs: Any,
    ) -> Any:
        if (
            output_type not in ("np", "pil")
            or not isinstance(video, torch.Tensor)
            or video.ndim != 5
        ):
            return original(video, output_type, **kwargs)
        clips = []
        for batch in range(video.shape[0]):
            # "pt" stops right after the denormalize the np / pil paths apply, on the same view they use.
            clip = device_uint8(postprocess(video[batch].permute(1, 0, 2, 3), "pt", **kwargs))
            if clip is None:
                return original(video, output_type, **kwargs)
            clips.append(clip)
        return clips[0].unsqueeze(0) if len(clips) == 1 else torch.stack(clips)

    try:
        proc.postprocess_video = postprocess_video
    except Exception:  # noqa: BLE001 - a processor that refuses instance attributes keeps its own path
        yield
        return
    try:
        yield
    finally:
        if had_own:
            proc.postprocess_video = original
        else:
            try:
                del proc.postprocess_video
            except AttributeError:
                pass

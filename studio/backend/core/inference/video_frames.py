# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a decoded clip into the uint8 RGB frames the mp4 encoder takes, on the device that decoded it.

diffusers' ``output_type="np"`` / ``"pil"`` copy the float clip to pageable host memory and round it on the CPU, and
``encode_video`` range-checks and rounds an np clip again: about 3 s at 960x544x124 and 4 s at 1344x768x129. Video
pipelines are asked for ``"pt"`` instead and the same ``(x * 255).round()`` runs in float32 on the GPU, then one
uint8 copy goes through pinned memory. The frames are bit-identical: float32 multiply and round-half-to-even agree
between numpy and torch. A clip with any value outside [0, 1] (NaN included), or on a device other than CUDA / CPU,
is rebuilt as exactly the np or PIL object the pipeline would have returned, so it warns or fails as before.
"""

from __future__ import annotations

from typing import Any, Optional

_SLICE_BYTES = 64 << 20


def legacy_output_type(pipe: Any, call_params: Any, modular: bool) -> Optional[str]:
    """What the pipeline returns when Studio passes no ``output_type`` ('np' or 'pil'); None when unknown, in which
    case the caller keeps today's call."""
    if modular:
        try:
            for param in getattr(getattr(pipe, "blocks", None), "inputs", None) or ():
                if getattr(param, "name", None) == "output_type":
                    default = getattr(param, "default", None)
                    return default if default in ("np", "pil") else None
        except Exception:  # noqa: BLE001
            return None
        return None
    param = call_params.get("output_type") if hasattr(call_params, "get") else None
    default = getattr(param, "default", None)
    return default if default in ("np", "pil") else None


def _legacy(frames: Any, output_type: str) -> Any:
    from diffusers.image_processor import VaeImageProcessor

    # The exact np / pil tail of VaeImageProcessor.postprocess after its denormalize, which "pt" already applied.
    arr = VaeImageProcessor.pt_to_numpy(frames)
    return VaeImageProcessor.numpy_to_pil(arr) if output_type == "pil" else arr


def to_uint8_frames(frames: Any, output_type: str) -> Any:
    """``frames`` is one clip from an ``output_type="pt"`` pipeline, (F, C, H, W). Returns a CPU uint8 (F, H, W, C)
    tensor, or the legacy ``output_type`` object when the fast path does not apply."""
    import torch

    tensor_cls = getattr(torch, "Tensor", None)
    if tensor_cls is None or not isinstance(frames, tensor_cls):
        return frames
    device = frames.device
    if device.type not in ("cuda", "cpu") or frames.numel() == 0:
        return _legacy(frames, output_type)
    try:
        lo, hi = torch.aminmax(frames)
        in_range = bool(lo >= 0) and bool(hi <= 1)
    except Exception:  # noqa: BLE001
        in_range = False
    if not in_range:
        return _legacy(frames, output_type)
    count, channels, height, width = frames.shape
    step = max(1, _SLICE_BYTES // max(1, channels * height * width * 4))
    out = torch.empty((count, height, width, channels), dtype = torch.uint8, device = device)
    for start in range(0, count, step):
        piece = frames[start : start + step].to(torch.float32) * 255
        out[start : start + step] = piece.round_().to(torch.uint8).permute(0, 2, 3, 1)
    if device.type == "cpu":
        return out
    try:
        host = torch.empty(out.shape, dtype = torch.uint8, pin_memory = True)
        host.copy_(out, non_blocking = True)
        torch.cuda.current_stream(device).synchronize()
        return host
    except Exception:  # noqa: BLE001 - pinned allocation refused (host limits); a pageable copy is still exact
        return out.cpu()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrated GPUs skip pinned host memory: their VRAM is system RAM, so a pinned copy only takes RAM from the GPU
(ComfyUI does the same). An unclassifiable device answers discrete (pinning allowed)."""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_CACHE: dict[int, bool] = {}


def _device_index(torch: Any, device: Any) -> Optional[int]:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return int(dev.index) if dev.index is not None else int(torch.cuda.current_device())


def _classify(torch: Any, index: int) -> bool:
    props = torch.cuda.get_device_properties(index)
    from .rocm_bf16 import is_rocm_torch

    if is_rocm_torch(torch):
        from utils.hardware.hardware import _rocm_props_unified_status
        return _rocm_props_unified_status(props) is True
    from utils.hardware.hardware import _cuda_props_are_integrated

    return bool(_cuda_props_are_integrated(props))


def device_is_integrated(device: Any = None) -> bool:
    """Never raises: no GPU, a non-CUDA device or an unreadable property answers False (discrete)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        index = _device_index(torch, device)
        if index is None:
            return False
        with _LOCK:
            hit = _CACHE.get(index)
        if hit is not None:
            return hit
        answer = bool(_classify(torch, index))
        with _LOCK:
            _CACHE[index] = answer
        return answer
    except Exception:  # noqa: BLE001 - unclassifiable: keep the discrete path
        return False


def host_pinning_allowed(device: Any = None) -> bool:
    return not device_is_integrated(device)


def reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()

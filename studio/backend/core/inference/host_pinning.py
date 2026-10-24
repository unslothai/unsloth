# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether page-locked host memory is worth taking for host-to-device copies on a device.

An integrated GPU (AMD APU such as gfx1151 Strix Halo, Jetson, DGX Spark) has no PCIe link: its "VRAM" is the same
system RAM, so a pinned staging copy buys no faster transfer and only takes RAM the GPU itself allocates from. ComfyUI
turns pinned memory off for such devices (``comfy/model_management.py``, ``is_integrated_gpu()`` sets
``DISABLE_PINNED_MEMORY``); every diffusion / video pinning site here asks ``host_pinning_allowed`` and keeps plain
pageable host tensors when it answers False. A discrete GPU, or a device that cannot be classified, answers True, so
those keep exactly the path they had.

The classifier is the one Studio's memory policy already uses: ``_cuda_props_are_integrated`` for CUDA (the driver's
``integrated`` flag), ``_rocm_props_unified_status`` for ROCm (the driver flag, else the known APU arches).
"""

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
    """True when ``device`` (default: the current CUDA / HIP device) is an integrated GPU sharing system RAM.
    Never raises: no GPU, a non-CUDA device or an unreadable property answers False (the discrete behaviour)."""
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
    """False on an integrated GPU (pinning only doubles host RAM pressure there); True otherwise."""
    return not device_is_integrated(device)


def reset_cache() -> None:
    """Forget the per-device answers (tests stub the device properties)."""
    with _LOCK:
        _CACHE.clear()

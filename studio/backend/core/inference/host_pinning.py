# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrated GPUs skip pinned host memory (VRAM is RAM); unclassifiable devices count as discrete"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_CACHE: dict[int, Optional[bool]] = {}


def _device_index(torch: Any, device: Any) -> Optional[int]:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return int(dev.index) if dev.index is not None else int(torch.cuda.current_device())


def _classify(torch: Any, index: int) -> Optional[bool]:
    props = torch.cuda.get_device_properties(index)
    from .rocm_bf16 import is_rocm_torch

    if is_rocm_torch(torch):
        from utils.hardware.hardware import _rocm_props_unified_status
        return _rocm_props_unified_status(props)
    from utils.hardware.hardware import _cuda_props_are_integrated

    return bool(_cuda_props_are_integrated(props))


def device_integrated_status(device: Any = None) -> Optional[bool]:
    """True integrated, False discrete, None unknown (no GPU, a non-CUDA device, an unreadable property). Never raises."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = _device_index(torch, device)
        if index is None:
            return None
        with _LOCK:
            if index in _CACHE:
                return _CACHE[index]
        answer = _classify(torch, index)
        answer = None if answer is None else bool(answer)
        with _LOCK:
            _CACHE[index] = answer
        return answer
    except Exception:  # noqa: BLE001 - unclassifiable
        return None


def device_is_integrated(device: Any = None) -> bool:
    """Never raises: no GPU, a non-CUDA device or an unreadable property answers False (discrete)."""
    return device_integrated_status(device) is True


def xpu_is_integrated() -> bool:
    """Whether the current XPU is a Core Ultra Arc iGPU, by PCI id. Never raises."""
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return False
        device_id = getattr(xpu.get_device_properties(), "device_id", None)
        from utils.hardware.hardware import _INTEL_XPU_IGPU_PCI_IDS

        return isinstance(device_id, int) and device_id in _INTEL_XPU_IGPU_PCI_IDS
    except Exception:  # noqa: BLE001 - unclassifiable: keep the discrete path
        return False


def host_pinning_allowed(device: Any = None) -> bool:
    return not device_is_integrated(device)


def reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()


_get_last_error: Any = None


def _resolve_get_last_error() -> Any:
    """``cudaGetLastError`` (``hipGetLastError`` on ROCm) of the runtime torch already loaded."""
    import ctypes
    import sys

    import torch

    hip = bool(getattr(torch.version, "hip", None))
    sym = "hipGetLastError" if hip else "cudaGetLastError"
    if sys.platform == "win32":
        import glob
        import os

        # torch's wheels ship the runtime DLL next to torch's own; loading it again returns the mapped module
        pattern = "amdhip64*.dll" if hip else "cudart64_*.dll"
        for path in sorted(
            glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", pattern))
        ):
            fn = getattr(ctypes.CDLL(path), sym, None)
            if fn is not None:
                return fn
        return None
    import os

    # The runtime torch itself links: another package (or LD_PRELOAD) can map a second libcudart, a .so.12
    # beside torch's .so.13, and the global namespace then resolves the symbol to it, clearing nothing torch
    # sees. A lookup through torch's own extension searches only its dependency tree.
    fn = getattr(ctypes.CDLL(torch._C.__file__), sym, None)
    if fn is not None:
        return fn
    lib = "libamdhip64.so" if hip else "libcudart.so"
    major = ((torch.version.hip if hip else torch.version.cuda) or "").split(".")[0]
    mapped = []
    try:
        with open("/proc/self/maps", encoding = "utf-8") as maps:
            for ln in maps:
                path = ln.split()[-1]
                if lib in os.path.basename(path) and path not in mapped:
                    mapped.append(path)
    except OSError:
        pass
    own = [p for p in mapped if os.path.basename(p).startswith(f"{lib}.{major}")] if major else []
    for path in own or mapped:
        fn = getattr(ctypes.CDLL(path), sym, None)
        if fn is not None:
            return fn
    # no /proc (or a static runtime): torch usually loads the runtime globally
    return getattr(ctypes.CDLL(None), sym, None)


def consume_cuda_error() -> None:
    """Clear this thread's pending CUDA / HIP runtime error after a handled failure (a refused
    ``cudaHostRegister``): left pending, the thread's next kernel launch raises it. Never raises."""
    global _get_last_error
    try:
        if _get_last_error is None:
            import ctypes

            fn = _resolve_get_last_error()
            if fn is None:
                return
            fn.argtypes = []
            fn.restype = ctypes.c_int
            _get_last_error = fn
        _get_last_error()
    except Exception:  # noqa: BLE001 - no runtime symbol (CPU build): nothing is pending
        pass

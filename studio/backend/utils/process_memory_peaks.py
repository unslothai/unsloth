# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Peak memory of the current process, for the training run record and status API.

Every value is in bytes, or None when this host cannot measure it:

- ``gpu_peak_allocated_bytes``: ``torch.cuda.max_memory_allocated`` summed over every device
  visible to this process (tensors the caching allocator handed out).
- ``gpu_peak_reserved_bytes``: ``torch.cuda.max_memory_reserved`` summed the same way (what the
  allocator held from the driver, so it is at least the allocated peak).
- ``gpu_device_count``: how many devices the two sums cover.
- ``host_peak_rss_bytes``: peak resident set (Linux / macOS ``ru_maxrss``; Windows peak working set).
- ``host_peak_commit_bytes``: Windows only, peak committed (pagefile-backed) private memory.

Training runs in its own spawned worker per run, so a process peak is that run's peak. The CUDA
peak counters are never reset here: the live offload panel reads the same counters.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

MEMORY_PEAK_FIELDS = (
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "host_peak_rss_bytes",
    "host_peak_commit_bytes",
)


def _gpu_peaks(torch_module: Any = None) -> tuple[Optional[int], Optional[int], Optional[int]]:
    # Never import torch just to measure: the MLX worker has no torch, and an import costs seconds.
    torch = torch_module if torch_module is not None else sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is None or not cuda.is_available():
            return None, None, None
        count = int(cuda.device_count())
        if count <= 0:
            return None, None, None
        allocated = sum(int(cuda.max_memory_allocated(i)) for i in range(count))
        reserved = sum(int(cuda.max_memory_reserved(i)) for i in range(count))
        return allocated, reserved, count
    except Exception:
        return None, None, None


def _windows_peaks() -> tuple[Optional[int], Optional[int]]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        wset = getattr(info, "peak_wset", None)
        commit = getattr(info, "peak_pagefile", None)
        if wset is not None or commit is not None:
            return (
                int(wset) if wset is not None else None,
                int(commit) if commit is not None else None,
            )
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize), int(counters.PeakPagefileUsage)
    except Exception:
        pass
    return None, None


def _posix_peak_rss() -> Optional[int]:
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if peak <= 0:
        return None
    # ru_maxrss is bytes on macOS, KiB on Linux and the other Unixes.
    return peak if sys.platform == "darwin" else peak * 1024


def host_peaks() -> tuple[Optional[int], Optional[int]]:
    """(peak RSS bytes, peak commit bytes); commit is only measured on Windows."""
    if sys.platform == "win32":
        return _windows_peaks()
    return _posix_peak_rss(), None


def process_memory_peaks(torch_module: Any = None) -> Optional[dict]:
    """This process's memory peaks so far, or None when nothing could be measured."""
    allocated, reserved, count = _gpu_peaks(torch_module)
    rss, commit = host_peaks()
    peaks = {
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
        "gpu_device_count": count,
        "host_peak_rss_bytes": rss,
        "host_peak_commit_bytes": commit,
    }
    if all(peaks[k] is None for k in MEMORY_PEAK_FIELDS):
        return None
    return peaks


def merge_memory_peaks(current: Optional[dict], incoming: Any) -> Optional[dict]:
    """Field-wise max of two samples, so a late or partial sample never lowers a recorded peak."""
    if not isinstance(incoming, dict):
        return current
    merged = dict(current or {})
    for key in (*MEMORY_PEAK_FIELDS, "gpu_device_count"):
        value = incoming.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            continue
        value = int(value)
        previous = merged.get(key)
        merged[key] = value if previous is None else max(previous, value)
    for key in (*MEMORY_PEAK_FIELDS, "gpu_device_count"):
        merged.setdefault(key, None)
    return merged

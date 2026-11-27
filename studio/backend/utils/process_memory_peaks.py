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
peak counters are never reset here: the live offload panel reads the same counters. Code that does
reset them (the diffusion trainer excludes load and preview rounds from its own figure) calls
``bank_gpu_peaks`` first, so the process peak still covers the whole run.

On Apple MPS torch keeps no peak counter, so ``current_allocated_memory`` (allocated) and
``driver_allocated_memory`` (reserved) are sampled on every call and the running max is reported.
ROCm builds answer through ``torch.cuda``. ``include_children`` adds the resident sets of the
process's live descendants (the diffusion trainer's multi-GPU ranks) to a sampled tree total.
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


# Running maxima this module keeps itself: CUDA peaks banked before a counter reset, sampled MPS
# values, and the sampled process-tree RSS. Per process, so per run (one spawned worker per run).
_banked_cuda: dict[int, tuple[int, int]] = {}
_mps_peak: list[Optional[int]] = [None, None]
_tree_rss_peak: list[Optional[int]] = [None]


def _max_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    return a if b is None else max(a, b)


def _resolve_torch(torch_module: Any) -> Any:
    # Never import torch just to measure: the MLX worker has no torch, and an import costs seconds.
    return torch_module if torch_module is not None else sys.modules.get("torch")


def _cuda_device_peaks(cuda: Any) -> Optional[list[tuple[int, int]]]:
    if cuda is None or not cuda.is_available():
        return None
    count = int(cuda.device_count())
    if count <= 0:
        return None
    peaks = []
    for i in range(count):
        allocated = int(cuda.max_memory_allocated(i))
        reserved = int(cuda.max_memory_reserved(i))
        banked = _banked_cuda.get(i)
        if banked is not None:
            allocated, reserved = max(allocated, banked[0]), max(reserved, banked[1])
        peaks.append((allocated, reserved))
    return peaks


def bank_gpu_peaks(torch_module: Any = None) -> None:
    """Remember the CUDA peaks so far; call before ``reset_peak_memory_stats`` so they survive it."""
    try:
        peaks = _cuda_device_peaks(getattr(_resolve_torch(torch_module), "cuda", None))
    except Exception:
        return
    for i, value in enumerate(peaks or ()):
        _banked_cuda[i] = value


def _mps_peaks(torch: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps = getattr(torch, "mps", None)
    if backend is None or mps is None or not backend.is_available():
        return None, None, None
    allocated = int(mps.current_allocated_memory())
    reserved = int(mps.driver_allocated_memory())
    _mps_peak[0] = _max_opt(_mps_peak[0], allocated)
    _mps_peak[1] = _max_opt(_mps_peak[1], reserved)
    return _mps_peak[0], _mps_peak[1], 1


def _gpu_peaks(torch_module: Any = None) -> tuple[Optional[int], Optional[int], Optional[int]]:
    torch = _resolve_torch(torch_module)
    try:
        peaks = _cuda_device_peaks(getattr(torch, "cuda", None))
        if peaks is not None:
            return sum(a for a, _ in peaks), sum(r for _, r in peaks), len(peaks)
    except Exception:
        return None, None, None
    try:
        return _mps_peaks(torch)
    except Exception:
        return None, None, None


def _tree_rss() -> Optional[int]:
    """Sampled resident set of this process plus its live descendants (running max), or None."""
    try:
        import psutil

        me = psutil.Process()
        total = int(me.memory_info().rss)
        for child in me.children(recursive = True):
            try:
                total += int(child.memory_info().rss)
            except Exception:  # exited between the listing and the read
                continue
    except Exception:
        return _tree_rss_peak[0]
    _tree_rss_peak[0] = _max_opt(_tree_rss_peak[0], total)
    return _tree_rss_peak[0]


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


def process_memory_peaks(
    torch_module: Any = None, *, include_children: bool = False
) -> Optional[dict]:
    """This process's memory peaks so far, or None when nothing could be measured.

    ``include_children``: the host RSS peak covers the process tree, the larger of this process's
    own peak and the sampled sum over it and its live descendants.
    """
    allocated, reserved, count = _gpu_peaks(torch_module)
    rss, commit = host_peaks()
    if include_children:
        rss = _max_opt(rss, _tree_rss())
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


# Terminal events: the last chance to record a peak, e.g. the OOM that ends a run or a final save.
_TERMINAL_EVENT_TYPES = frozenset({"error", "complete"})


class PeakStampingQueue:
    """Wraps the worker's event queue so every terminal event (or every ``event_types`` event)
    carries a fresh sample.

    Error events are emitted from many places (load failures, OOMs, security refusals) and the MLX
    completion path sends its own events, so stamping them here keeps every exit covered. Events that
    already carry ``memory_peaks`` are passed through unchanged.
    """

    def __init__(
        self,
        queue: Any,
        *,
        event_types: frozenset = _TERMINAL_EVENT_TYPES,
        include_children: bool = False,
    ):
        self._queue = queue
        self._event_types = event_types
        self._include_children = include_children

    def put(self, event: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(event, dict)
            and event.get("type") in self._event_types
            and event.get("memory_peaks") is None
        ):
            try:
                peaks = (
                    process_memory_peaks(include_children = True)
                    if self._include_children
                    else process_memory_peaks()
                )
            except Exception:
                peaks = None
            if peaks is not None:
                event = {**event, "memory_peaks": peaks}
        return self._queue.put(event, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in (
            "_queue",
            "_event_types",
            "_include_children",
        ):  # not set yet (e.g. while unpickling): avoid recursing
            raise AttributeError(name)
        return getattr(self._queue, name)

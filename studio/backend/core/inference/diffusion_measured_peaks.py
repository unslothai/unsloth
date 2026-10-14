# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Learned activation peaks: the real CUDA peak above the resident weights, recorded per
(family, denoiser quant, speed tier, resolution bucket, frames, GPU name) at a successful render and persisted under
the Studio cache dir, so later loads of the same model place it from a measurement instead of the flat
8192 MiB/MP estimate.

The static tables in ``diffusion_memory`` stay the seed: a learned value only RAISES a seeded peak (a local
measurement above the seed means this machine needs more) and supplies one where no seed exists. It never lowers a
seed, because one render does not exercise every phase (a prompt-cache hit skips the encoder, a smaller request
decodes less), and the seed was measured on the worst phase.

``UNSLOTH_DIFFUSION_LEARNED_PEAKS=0`` turns lookups and recording off; ``UNSLOTH_DIFFUSION_LEARNED_PEAKS_FILE``
names the JSON file (default ``<studio cache>/diffusion_measured_peaks.json``).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

LEARNED_PEAKS_ENV = "UNSLOTH_DIFFUSION_LEARNED_PEAKS"
LEARNED_PEAKS_FILE_ENV = "UNSLOTH_DIFFUSION_LEARNED_PEAKS_FILE"
_FILE_NAME = "diffusion_measured_peaks.json"
_VERSION = 1
_MAX_ENTRIES = 512
_MEGAPIXEL = 1024 * 1024
# Smallest bucket: a quarter megapixel. Below it the activation is dominated by fixed costs.
_MIN_BUCKET_PIXELS = _MEGAPIXEL // 4

_lock = threading.Lock()
_cache: dict[str, Any] = {"path": None, "mtime": None, "data": None}


def enabled() -> bool:
    return (os.environ.get(LEARNED_PEAKS_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def store_path() -> Optional[Path]:
    explicit = (os.environ.get(LEARNED_PEAKS_FILE_ENV) or "").strip()
    if explicit:
        return Path(explicit)
    try:
        from utils.paths.storage_roots import cache_root
    except Exception:  # noqa: BLE001 - imported without studio/backend on sys.path: nowhere safe to write
        return None
    try:
        return cache_root() / _FILE_NAME
    except Exception:  # noqa: BLE001
        return None


def resolution_bucket(
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
) -> int:
    """Pixels of the power-of-two megapixel bucket covering ``width x height x batch`` (at least 1/4 MP)."""
    pixels = max(1, int(width or 1024)) * max(1, int(height or 1024)) * max(1, int(batch_size or 1))
    if pixels <= _MIN_BUCKET_PIXELS:
        return _MIN_BUCKET_PIXELS
    return int(_MIN_BUCKET_PIXELS * 2 ** math.ceil(math.log2(pixels / _MIN_BUCKET_PIXELS)))


def model_key(
    family: Optional[str],
    quant: Optional[str],
    speed_mode: Optional[str],
    frames: int = 1,
    device: Optional[str] = None,
) -> str:
    """``device``: the GPU's name. Kernels and workspaces differ by card, so a heterogeneous host keeps one set of
    peaks per card model instead of placing one GPU from another's measurement."""
    return "|".join(
        (
            str(family or "unknown").lower(),
            str(quant or "unknown"),
            str(speed_mode or "unknown"),
            f"f{max(1, int(frames or 1))}",
            str(device or "unknown"),
        )
    )


_runtime: list = []
_KERNEL_DISTS = (
    "triton",
    "torchao",
    "flash-attn",
    "flash_attn_3",
    "xformers",
    "sageattention",
    "kernels",
)


def runtime_stamp() -> str:
    """The runtimes a peak depends on (kernel choice, workspaces): a store written under others is ignored."""
    if not _runtime:
        from importlib.metadata import version

        parts = []
        for dist in ("torch", "diffusers"):
            try:
                parts.append(f"{dist}={version(dist)}")
            except Exception:  # noqa: BLE001
                parts.append(f"{dist}=?")
        # attention and quantization kernels (their workspaces are part of a peak); an absent one is left out
        for dist in _KERNEL_DISTS:
            try:
                parts.append(f"{dist}={version(dist)}")
            except Exception:  # noqa: BLE001
                pass
        try:
            import torch
            parts.append(f"cuda={torch.version.cuda or torch.version.hip or '?'}")
        except Exception:  # noqa: BLE001
            parts.append("cuda=?")
        _runtime.append("|".join(parts))
    return _runtime[0]


def _empty() -> dict:
    return {"version": _VERSION, "runtime": runtime_stamp(), "entries": {}}


def _read(path: Path, fresh: bool = False) -> dict:
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return _empty()
    if (
        not fresh
        and _cache["path"] == str(path)
        and _cache["mtime"] == mtime
        and _cache["data"] is not None
    ):
        return _cache["data"]
    try:
        data = json.loads(path.read_text())
        if (
            not isinstance(data, dict)
            or data.get("version") != _VERSION
            or data.get("runtime") != runtime_stamp()
            or not isinstance(data.get("entries"), dict)
        ):
            data = _empty()
    except Exception:  # noqa: BLE001 - a corrupt file is an empty store, never a failed load
        data = _empty()
    _cache.update(path = str(path), mtime = mtime, data = data)
    return data


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive lock on ``<store>.lock`` across processes sharing the cache root (two Studio servers, a worker), so a
    read / merge / replace cannot overwrite another process's higher peak. Advisory; held only for the merge."""
    path.parent.mkdir(parents = True, exist_ok = True)
    handle = open(str(path) + ".lock", "a+b")
    try:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # Windows
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        handle.close()  # closing releases either lock


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    fd, tmp = tempfile.mkstemp(prefix = ".peaks.", dir = str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent = 1)  # insertion order is the eviction order
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _cache.update(path = None, mtime = None, data = None)


def record(
    *,
    family: Optional[str],
    quant: Optional[str],
    speed_mode: Optional[str],
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    frames: int = 1,
    peak_mib: int,
    device_name: Optional[str] = None,
    quadratic: bool = False,
    held: Optional[list] = None,
) -> bool:
    """Keep the max peak seen for this key and bucket, scaled to the bucket's ceiling so a smaller render in the bucket
    still answers a full-size one (squared with ``quadratic``: SDPA math attention grows with pixels squared, and the
    key's entry keeps scaling that way). Returns True when the stored value rose. ``held`` (a list) gets True appended
    when the store holds at least this peak afterwards (raised, or already covered): False when nothing was persisted."""
    if (
        not enabled()
        or peak_mib is None
        or int(peak_mib) <= 0
        or not family
        or not identified(quant)
    ):
        return False
    if quadratic and int(batch_size or 1) > 1:
        return False  # a math-attention entry holds single images: it scales squared in pixels, linearly in batch
    path = store_path()
    if path is None:
        return False
    device_name = device_name or current_device_name()
    key = model_key(family, quant, speed_mode, frames, device_name)
    bucket_pixels = resolution_bucket(width, height, batch_size)
    bucket = str(bucket_pixels)
    pixels = max(1, int(width or 1024)) * max(1, int(height or 1024)) * max(1, int(batch_size or 1))
    peak_mib = int(
        math.ceil(
            int(peak_mib)
            * (bucket_pixels / float(max(pixels, bucket_pixels // 2))) ** (2 if quadratic else 1)
        )
    )
    try:
        with _lock, _file_lock(path):
            data = _read(path, fresh = True)
            entries = data.setdefault("entries", {})
            entry = entries.pop(key, None) or {"peaks": {}}
            entries[key] = entry  # most recently raised last: the cap evicts from the front
            peaks = entry.setdefault("peaks", {})
            old = int(peaks.get(bucket) or 0)
            if int(peak_mib) <= old:
                if held is not None:
                    held.append(True)
                return False
            peaks[bucket] = int(peak_mib)
            if quadratic:
                entry["quadratic"] = True
            if device_name:
                entry["device"] = str(device_name)  # also in the key; kept for readers of the file
            if len(entries) > _MAX_ENTRIES:
                for stale in list(entries)[: len(entries) - _MAX_ENTRIES]:
                    entries.pop(stale, None)
            _write(path, data)
            if held is not None:
                held.append(True)
            return True
    except Exception:  # noqa: BLE001 - persistence is best-effort
        return False


def lookup(
    *,
    family: Optional[str],
    quant: Optional[str],
    speed_mode: Optional[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    frames: int = 1,
    device: Optional[str] = None,
) -> Optional[int]:
    """Learned peak MiB (no margin) for a request of this size, or None when nothing was recorded.
    ``device`` (GPU name) defaults to the current CUDA device, the one the planner snapshots.

    The smallest recorded bucket at or above the request's bucket answers as is (an overestimate); with only smaller
    buckets recorded, the largest one is scaled linearly by pixel count."""
    found = lookup_bucket(
        family = family,
        quant = quant,
        speed_mode = speed_mode,
        width = width,
        height = height,
        batch_size = batch_size,
        frames = frames,
        device = device,
    )
    return None if found is None else found[0]


def identified(quant: Optional[str]) -> bool:
    """False when the label or a part of it could not be identified (empty, a bare ``unknown`` format,
    ``name:unknown``): such a load neither reads nor records a learned peak and keeps the static estimate."""
    parts = str(quant or "").split("|")
    return bool(parts[0]) and not any(
        part == "unknown" or part.endswith(":unknown") for part in parts
    )


def lookup_bucket(
    *,
    family: Optional[str],
    quant: Optional[str],
    speed_mode: Optional[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    frames: int = 1,
    device: Optional[str] = None,
) -> Optional[tuple[int, int]]:
    """``lookup`` plus the pixels the peak covers and whether it scales squared: (peak MiB, bucket pixels,
    quadratic), or None."""
    if not enabled() or not family or not identified(quant):
        return None
    path = store_path()
    if path is None:
        return None
    try:
        with _lock:
            key = model_key(family, quant, speed_mode, frames, device or current_device_name())
            entry = (_read(path).get("entries") or {}).get(key)
        peaks = {
            int(k): int(v) for k, v in ((entry or {}).get("peaks") or {}).items() if int(v) > 0
        }
    except Exception:  # noqa: BLE001
        return None
    if not peaks:
        return None
    quadratic = bool((entry or {}).get("quadratic"))
    batch = max(1, int(batch_size or 1))
    # a math-attention entry holds single images (``record``): its buckets are per image, the batch scales linearly
    want = resolution_bucket(width, height, 1 if quadratic else batch)
    times = batch if quadratic else 1
    above = [b for b in peaks if b >= want]
    if above:
        return peaks[min(above)] * times, min(above), quadratic
    largest = max(peaks)
    power = 2 if quadratic else 1
    return (
        int(math.ceil(peaks[largest] * (want / float(largest)) ** power)) * times,
        want,
        quadratic,
    )


def record_work(key: Any, work: int, peak_mib: int, held: Optional[list] = None) -> bool:
    """Persist a peak under an opaque calibration key (the video path's work = pixels x frames table); max wins.
    ``held`` as in ``record``."""
    if not enabled() or key is None or peak_mib is None or int(peak_mib) <= 0 or int(work) <= 0:
        return False
    path = store_path()
    if path is None:
        return False
    try:
        with _lock, _file_lock(path):
            data = _read(path, fresh = True)
            table = data.setdefault("work", {})
            peaks = table.pop(repr(key), None) or {}
            table[repr(key)] = peaks
            old = int(peaks.get(str(int(work))) or 0)
            if int(peak_mib) <= old:
                if held is not None:
                    held.append(True)
                return False
            peaks[str(int(work))] = int(peak_mib)
            if len(table) > _MAX_ENTRIES:
                for stale in list(table)[: len(table) - _MAX_ENTRIES]:
                    table.pop(stale, None)
            _write(path, data)
            if held is not None:
                held.append(True)
            return True
    except Exception:  # noqa: BLE001 - persistence is best-effort
        return False


def work_peaks(key: Any) -> dict[int, int]:
    """Persisted {work: peak MiB} for an opaque calibration key ({} when none or disabled)."""
    if not enabled() or key is None:
        return {}
    path = store_path()
    if path is None:
        return {}
    try:
        with _lock:
            peaks = (_read(path).get("work") or {}).get(repr(key)) or {}
        return {int(w): int(v) for w, v in peaks.items() if int(v) > 0}
    except Exception:  # noqa: BLE001
        return {}


class _Window:
    overlapped = False


_windows_lock = threading.Lock()
_windows: dict[str, "weakref.WeakSet[_Window]"] = {}


def open_window(device: Any) -> _Window:
    """Register a peak measurement on ``device``. Resetting the peak is allocator-wide, so a second render arming on
    the same card (image and video run on separate threads) spoils both: each one's window reads as overlapped. An
    abandoned window drops out once nothing references it."""
    window = _Window()
    with _windows_lock:
        live = _windows.setdefault(str(device), weakref.WeakSet())
        if len(live):
            window.overlapped = True
            for other in live:
                other.overlapped = True
        live.add(window)
    return window


def close_window(device: Any, window: Optional[_Window]) -> bool:
    """End a measurement; True when no other measurement on the card overlapped it."""
    if window is None:
        return True
    with _windows_lock:
        live = _windows.get(str(device))
        if live is not None:
            live.discard(window)
    return not window.overlapped


class PeakProbe:
    """CUDA peak above the allocation at ``start``; ``finish`` returns MiB or None (non-CUDA / unreadable, or another
    measurement on the card reset the peak meanwhile)."""

    def __init__(self, device: Any) -> None:
        self.device = None
        self.base = None
        try:
            import torch

            dev = torch.device(device)
            if dev.type != "cuda" or not torch.cuda.is_available():
                return
            if dev.index is None:
                dev = torch.device("cuda", torch.cuda.current_device())
            self.window = open_window(dev)
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            self.base = int(torch.cuda.memory_allocated(dev))
            self.device = dev
        except Exception:  # noqa: BLE001 - measurement only
            self.device = None

    def finish(self) -> Optional[int]:
        if self.device is None or self.base is None:
            return None
        try:
            import torch
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device)) - self.base
        except Exception:  # noqa: BLE001
            peak = None
        # closed only after the read: a render arming in between would otherwise reset the peak unseen
        if not close_window(self.device, getattr(self, "window", None)) or peak is None:
            return None
        return -(-peak // (1024 * 1024)) if peak > 0 else None


def current_device_name() -> Optional[str]:
    """Name of the current CUDA device (None without CUDA); the planner's snapshot reads this device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001
        return None
    return device_name("cuda")


def device_name(device: Any) -> Optional[str]:
    try:
        import torch
        return str(torch.cuda.get_device_name(torch.device(device)))
    except Exception:  # noqa: BLE001
        return None

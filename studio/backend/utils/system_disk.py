# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The volume holding downloaded models, when it is not the system disk (#9259)."""

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional


_SAME_POOL_SLACK = 1 << 30
_TTL_S = 30.0
_FIRST_WAIT_S = 0.5


def _hub_cache() -> Optional[Path]:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return get_hf_cache_paths().hub_cache
    except Exception:  # noqa: BLE001 - a settings read must not fail the system poll
        return None


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _locate(path: Path) -> tuple[Path, int]:
    """(nearest existing ancestor, its device). Climbs only past MISSING paths: a fresh install
    has no cache dir yet, but an unreadable one must not report its parent's volume."""
    for candidate in (path, *path.parents):
        try:
            return candidate, _device(candidate)
        except NotADirectoryError:
            continue
        except FileNotFoundError:
            # Windows reports an untraversable parent as FileNotFoundError too.
            from utils.hf_cache_settings import _absence_is_real
            if not _absence_is_real(candidate):
                raise
    raise FileNotFoundError(path)


def models_disk_usage(cache: Optional[Path] = None) -> Optional[dict]:
    """Resolve symlinks first: a missing cache behind a symlink would climb the target's parents."""
    cache = _hub_cache() if cache is None else cache
    if cache is None:
        return None
    try:
        target, device = _locate(Path(os.path.realpath(cache)))
        root = Path(os.path.abspath(os.sep))
        if device == _device(root):
            return None
        usage = shutil.disk_usage(target)
        system = shutil.disk_usage(root)
    except (OSError, ValueError):
        return None
    # APFS Data volumes and btrfs subvolumes have their own st_dev but share `/`'s pool.
    if usage.total == system.total and abs(usage.free - system.free) < _SAME_POOL_SLACK:
        return None
    seen = usage.used + usage.free
    return {
        "total_gb": round(usage.total / 1e9, 2),
        "free_gb": round(usage.free / 1e9, 2),
        "percent_used": round(usage.used / seen * 100, 1) if seen else 0,
    }


_lock = threading.Lock()
_readings: dict = {}
_probes: dict = {}


def _cache_key() -> str:
    try:
        from utils.hf_cache_settings import configured_cache_key
        return configured_cache_key()
    except Exception:  # noqa: BLE001 - a settings read must not fail the system poll
        return "default"


def _probe(key: str) -> None:
    # Resolving the configured path can block on a dead mount, so do it on this thread.
    reading = models_disk_usage()
    with _lock:
        _probes.pop(key, None)
        _readings[key] = (time.monotonic(), reading)


def cached_models_disk_usage() -> Optional[dict]:
    """models_disk_usage for the 3 s /api/system poll: statvfs on a hung NFS/SMB cache blocks, so
    the probe runs off the request thread, at most one per cache path, refreshed every _TTL_S.
    """
    key = _cache_key()
    with _lock:
        at, reading = _readings.get(key, (None, None))
        if key in _probes or (at is not None and time.monotonic() - at < _TTL_S):
            return reading
        probe = threading.Thread(target = _probe, args = (key,), daemon = True)
        _probes[key] = probe
    probe.start()
    probe.join(_FIRST_WAIT_S)
    with _lock:
        return _readings.get(key, (None, None))[1]

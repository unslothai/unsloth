# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk usage for the Live Monitor: the system disk plus the volume holding the HF cache (#9259)."""

import os
import shutil
from pathlib import Path
from typing import Iterable, NamedTuple, Optional


class DiskUsage(NamedTuple):
    total: int
    free: int
    percent: float
    filesystems: int


def _hub_cache() -> Optional[Path]:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return get_hf_cache_paths().hub_cache
    except Exception:  # noqa: BLE001 - a settings read must not cost the root reading
        return None


def _nearest_existing(path: Path) -> Path:
    # A fresh install has no cache dir yet; its volume is its nearest existing parent.
    return next((p for p in (path, *path.parents) if p.exists()), path)


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def system_disk_usage(extra_roots: Optional[Iterable] = None) -> Optional[DiskUsage]:
    """Union of the root filesystem and the hub cache's, deduplicated by device.

    realpath so a symlinked cache reports the mount holding the bytes, not the link's. Keyed on
    st_dev, not (total, free): free moves between two reads while a download writes.
    """
    roots = [Path(os.path.abspath(os.sep))]
    if extra_roots is None:
        cache = _hub_cache()
        extra_roots = [cache] if cache is not None else []
    for root in extra_roots:
        try:
            roots.append(_nearest_existing(Path(os.path.realpath(root))))
        except (OSError, ValueError):
            continue

    readings = {}
    for root in roots:
        try:
            device = _device(root)
            if device not in readings:
                readings[device] = shutil.disk_usage(root)
        except (OSError, ValueError):
            continue
    if not readings:
        return None
    total = sum(u.total for u in readings.values())
    free = sum(u.free for u in readings.values())
    used = sum(u.used for u in readings.values())
    # psutil's formula (root-reserved blocks excluded), so a one-disk host reads as before.
    percent = round(used / (used + free) * 100, 1) if used + free else 0.0
    return DiskUsage(total, free, percent, len(readings))

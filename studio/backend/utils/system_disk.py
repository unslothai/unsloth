# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The volume holding downloaded models, when it is not the system disk (#9259)."""

import os
import shutil
from pathlib import Path
from typing import Optional


def _hub_cache() -> Optional[Path]:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return get_hf_cache_paths().hub_cache
    except Exception:  # noqa: BLE001 - a settings read must not fail the system poll
        return None


def _nearest_existing(path: Path) -> Path:
    # A fresh install has no cache dir yet; its volume is its nearest existing parent.
    return next((p for p in (path, *path.parents) if p.exists()), path)


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def models_disk_usage(cache: Optional[Path] = None) -> Optional[dict]:
    """Usage of the HF hub cache's volume, or None when it is the system disk or unreadable.

    realpath first, so a cache path that does not exist yet behind a symlink climbs the target's
    parents (the drive that will hold the bytes), not the link's.
    """
    cache = _hub_cache() if cache is None else cache
    if cache is None:
        return None
    try:
        target = _nearest_existing(Path(os.path.realpath(cache)))
        if _device(target) == _device(Path(os.path.abspath(os.sep))):
            return None
        usage = shutil.disk_usage(target)
    except (OSError, ValueError):
        return None
    # psutil's percent (root-reserved blocks excluded), matching the system disk reading.
    seen = usage.used + usage.free
    return {
        "total_gb": round(usage.total / 1e9, 2),
        "free_gb": round(usage.free / 1e9, 2),
        "percent_used": round(usage.used / seen * 100, 1) if seen else 0,
    }

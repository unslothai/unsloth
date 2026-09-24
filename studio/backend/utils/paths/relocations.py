# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Folders the installation owner moved a kind of file to (Settings > Library > Files on disk).

Only the owner's folders move: a managed account keeps every file in its private workspace, so a
choice is read in the owner context alone. Kept in the owner's app settings and cached, since the
galleries resolve their folder on every file lookup.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from utils.account_context import is_owner_context

# Fine-tunes and exports are not here: training history, chats and the model picker keep their
# absolute paths, so moving them would strand those references.
MOVABLE = ("uploads", "images", "videos", "audio")

_SETTING = "library.locations"
_lock = threading.Lock()
# Keyed by the owner's database, so a test or a relaunch on another root never reads a stale map.
_cache: dict[str, dict[str, str]] = {}


def _db_key() -> str:
    from utils.paths.storage_roots import studio_db_path

    return str(studio_db_path())


def _load() -> dict[str, str]:
    key = _db_key()
    with _lock:
        if key in _cache:
            return _cache[key]
    from storage.studio_db import get_app_setting

    raw = get_app_setting(_SETTING, {})
    chosen = (
        {k: v for k, v in raw.items() if k in MOVABLE and isinstance(v, str) and v}
        if isinstance(raw, dict)
        else {}
    )
    with _lock:
        _cache[key] = chosen
    return chosen


def chosen(key: str) -> Optional[Path]:
    """The folder the owner picked for `key`, or None for the default."""
    if not is_owner_context():
        return None
    try:
        value = _load().get(key)
    except (sqlite3.Error, OSError):
        # No settings table yet (a fresh or test database): nothing was ever moved.
        return None
    return Path(value) if value else None


def relocated(key: str, default: Path) -> Path:
    """Where `key`'s files live: the owner's chosen folder, else `default`."""
    return chosen(key) or default


def set_chosen(key: str, path: Optional[Path]) -> None:
    """Record `path` for `key`; None goes back to the default. Owner context only."""
    if key not in MOVABLE:
        raise ValueError(f"{key} files cannot move")
    from storage.studio_db import upsert_app_settings

    updated = dict(_load())
    if path is None:
        updated.pop(key, None)
    else:
        updated[key] = str(path)
    upsert_app_settings({_SETTING: updated}, read_back = False)
    with _lock:
        _cache[_db_key()] = updated


def forget_cache() -> None:
    """For tests that swap the database under a live process."""
    with _lock:
        _cache.clear()

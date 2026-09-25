# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Folders the installation owner moved a kind of file to (Settings > Library > Files on disk).

Only the owner's folders move: a managed account keeps every file in its private workspace, so a
choice is read in the owner context alone. Kept in the owner's app settings and cached, since the
galleries resolve their folder on every file lookup.
"""

from __future__ import annotations

import os
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
# Each choice is {"path", "mount"}: `mount` is the drive or share the folder was on when picked.
# One saved before mount points were recorded is its bare path, and is marked `bare` until its
# folder is there to ask (while it is not, the drive is what it would name).
_cache: dict[str, dict[str, dict]] = {}
# Resolving the database path walks the filesystem, and the galleries ask on every file lookup.
# It only moves with the variables that place Studio's home (or a test swapping the function).
_HOME_VARIABLES = ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_HOME")
_db_keys: dict[tuple, str] = {}


def _db_key() -> str:
    from utils.paths import storage_roots

    placed = (
        *(os.environ.get(name) for name in _HOME_VARIABLES),
        id(storage_roots.studio_db_path),
    )
    if placed not in _db_keys:
        _db_keys.clear()
        _db_keys[placed] = str(storage_roots.studio_db_path())
    return _db_keys[placed]


def _entry(value) -> Optional[dict]:
    if isinstance(value, str) and value:
        if not Path(value).is_dir():
            return {"path": value, "mount": None, "bare": True}
        return {"path": value, "mount": mount_point(Path(value))}
    if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
        mount = value.get("mount")
        return {"path": value["path"], "mount": mount if isinstance(mount, str) and mount else None}
    return None


def _save(chosen: dict[str, dict]) -> None:
    from storage.studio_db import upsert_app_settings
    stored = {
        kind: entry["path"]
        if entry.get("bare")
        else {"path": entry["path"], "mount": entry["mount"]}
        for kind, entry in chosen.items()
    }
    upsert_app_settings({_SETTING: stored}, read_back = False)


def _load() -> dict[str, dict]:
    key = _db_key()
    with _lock:
        if key in _cache:
            return _cache[key]
    from storage.studio_db import get_app_setting

    raw = get_app_setting(_SETTING, {})
    raw = raw if isinstance(raw, dict) else {}
    chosen = {kind: _entry(value) for kind, value in raw.items() if kind in MOVABLE}
    chosen = {kind: entry for kind, entry in chosen.items() if entry is not None}
    # A bare path whose folder is there now learned its mount: saved, so it is asked once.
    if any(isinstance(raw[kind], str) and not entry.get("bare") for kind, entry in chosen.items()):
        try:
            _save(chosen)
        except (sqlite3.Error, OSError):
            pass
    with _lock:
        _cache[key] = chosen
    return chosen


def _chosen_entry(key: str) -> Optional[dict]:
    if not is_owner_context():
        return None
    try:
        return _load().get(key)
    except (sqlite3.Error, OSError):
        # No settings table yet (a fresh or test database): nothing was ever moved.
        return None


def chosen(key: str) -> Optional[Path]:
    """The folder the owner picked for `key`, or None for the default."""
    entry = _chosen_entry(key)
    return Path(entry["path"]) if entry else None


class LocationUnavailable(OSError):
    """A chosen folder that is not there now, its drive unplugged or unmounted."""


def mount_point(path: Path) -> Optional[str]:
    """The mount point `path` is under when that is not the system disk's root: a drive or share
    mounted at a fixed folder (/mnt/usb, /media/me/Drive, a Windows mounted folder). None for a
    folder on the system disk, or on a drive with a letter or volume of its own, which goes away
    whole when it is unplugged."""
    for candidate in (path, *path.parents):
        try:
            if os.path.ismount(candidate):
                return None if candidate.parent == candidate else str(candidate)
        except (OSError, ValueError):
            return None
    return None


def _unavailable(entry: dict) -> bool:
    folder = Path(entry["path"])
    if not folder.is_dir():
        return True
    # A fixed mount point stays behind as an empty folder once its drive is gone, and saves would
    # land on the disk beneath it. The device number is not compared: it changes when a drive is
    # plugged into another port, and the folder is still there then.
    mount = entry.get("mount")
    return bool(mount) and not os.path.ismount(mount)


def location_dir(key: str, default: Path) -> Path:
    """`key`'s folder, ready to use. Only the default is created: a chosen folder that has gone is
    not made again, or files would land on the disk beneath its mount point and vanish once the
    drive is back."""
    from utils.paths.storage_roots import ensure_dir

    entry = _chosen_entry(key)
    if entry is None:
        return ensure_dir(default)
    folder = Path(entry["path"])
    if _unavailable(entry):
        raise LocationUnavailable(
            f"{folder} is not available. Reconnect its drive, or reset the folder in Settings."
        )
    return folder


def is_available(key: str) -> bool:
    """False while `key`'s chosen folder is on a drive that is not there."""
    entry = _chosen_entry(key)
    return entry is None or not _unavailable(entry)


def set_chosen(key: str, path: Optional[Path]) -> None:
    """Record `path` for `key`; None goes back to the default. Owner context only."""
    if key not in MOVABLE:
        raise ValueError(f"{key} files cannot move")
    updated = dict(_load())
    if path is None:
        updated.pop(key, None)
    else:
        updated[key] = {"path": str(path), "mount": mount_point(path)}
    _save(updated)
    with _lock:
        _cache[_db_key()] = updated


def forget_cache() -> None:
    """For tests that swap the database under a live process."""
    with _lock:
        _cache.clear()
        _db_keys.clear()

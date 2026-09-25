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
from typing import Callable, Optional

from utils.account_context import is_owner_context

# Fine-tunes and exports are not here: training history, chats and the model picker keep their
# absolute paths, so moving them would strand those references.
MOVABLE = ("uploads", "images", "videos", "audio")

_SETTING = "library.locations"
_lock = threading.Lock()
# Keyed by the owner's database, so a test or a relaunch on another root never reads a stale map.
# Each choice is {"path", "mount"}: `mount` is the drive or share the folder was on when picked.
# One saved before mount points were recorded is its bare path, and is marked `bare` until its
# folder is there to ask (while it is not, the drive is what it would name). A move under way also
# keeps `moving_from`, the folder its files are leaving, until they have all left it, and
# `moving_mount`, the drive or share that folder is on.
_cache: dict[str, dict[str, dict]] = {}
# Bumped by each write, so a load that read the database before one never caches what it read.
_generation = 0
# Resolving the database path walks the filesystem, and the galleries ask on every file lookup.
# It only moves with the variables that place Studio's home (or a test swapping the function).
_HOME_VARIABLES = ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_HOME")
_db_keys: dict[tuple, str] = {}
# Set by core.library: finishes the move of a kind whose `moving_from` a crash left behind.
resume_move: Optional[Callable[[str], None]] = None


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
        mount, source = value.get("mount"), value.get("moving_from")
        entry = {
            "path": value["path"],
            "mount": mount if isinstance(mount, str) and mount else None,
        }
        if isinstance(source, str) and source:
            entry["moving_from"] = source
            source_mount = value.get("moving_mount")
            entry["moving_mount"] = (
                source_mount if isinstance(source_mount, str) and source_mount else None
            )
        return entry
    return None


def _save(chosen: dict[str, dict]) -> None:
    from storage.studio_db import upsert_app_settings
    stored = {
        kind: entry["path"]
        if entry.get("bare")
        else {
            key: entry[key]
            for key in ("path", "mount", "moving_from", "moving_mount")
            if key in entry
        }
        for kind, entry in chosen.items()
    }
    upsert_app_settings({_SETTING: stored}, read_back = False)


def _load() -> dict[str, dict]:
    key = _db_key()
    with _lock:
        if key in _cache:
            return _cache[key]
        generation = _generation
    from storage.studio_db import get_app_setting

    raw = get_app_setting(_SETTING, {})
    raw = raw if isinstance(raw, dict) else {}
    chosen = {kind: _entry(value) for kind, value in raw.items() if kind in MOVABLE}
    chosen = {kind: entry for kind, entry in chosen.items() if entry is not None}
    learned = any(
        isinstance(raw[kind], str) and not entry.get("bare") for kind, entry in chosen.items()
    )
    with _lock:
        # Written meanwhile: what was read is older than what that write cached.
        if _generation != generation:
            return _cache.get(key, chosen)
        # A bare path whose folder is there now learned its mount: saved, so it is asked once.
        if learned:
            try:
                _save(chosen)
            except (sqlite3.Error, OSError):
                pass
        _cache[key] = chosen
    # Loaded once per database, so a move cut short is finished before its folder is used.
    interrupted = [kind for kind, entry in chosen.items() if entry.get("moving_from")]
    if interrupted and resume_move is not None:
        for kind in interrupted:
            resume_move(kind)
        with _lock:
            return _cache.get(key, chosen)
    return chosen


def _chosen_entry(key: str) -> Optional[dict]:
    if not is_owner_context():
        return None
    try:
        return _load().get(key)
    except sqlite3.OperationalError as exc:
        # No settings table (a database made without Studio's schema): nothing was ever moved.
        # Any other failure, a lock held too long among them, is raised: falling back would save
        # into the default folder, where the file vanishes once the chosen one is read again.
        if "no such table" in str(exc):
            return None
        raise


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


def _source_available(entry: dict) -> bool:
    """Whether the drive of the folder a move is leaving is there."""
    mount = entry.get("moving_mount")
    if mount:
        return os.path.ismount(mount)
    # A drive with a letter or volume of its own goes away whole.
    anchor = Path(entry["moving_from"]).anchor
    return not anchor or os.path.isdir(anchor)


def _in_use(entry: dict) -> Optional[Path]:
    """The folder `entry` saves into now: its own, or, while a move cut short waits for the drive
    it was moving onto, the folder the files were leaving, where those not moved yet still are.
    That stays in use until the move is taken up again, even once the drive is back: the chosen
    folder alone would hide what the folder standing in holds. None while neither is there."""
    source = entry.get("moving_from")
    standing_in = bool(source) and _source_available(entry) and Path(source).is_dir()
    if standing_in and entry.get("waiting"):
        return Path(source)
    if not _unavailable(entry):
        return Path(entry["path"])
    return Path(source) if standing_in else None


def location_dir(key: str, default: Path) -> Path:
    """`key`'s folder, ready to use. Only the default is created: a chosen folder that has gone is
    not made again, or files would land on the disk beneath its mount point and vanish once the
    drive is back."""
    from utils.paths.storage_roots import ensure_dir

    entry = _chosen_entry(key)
    if entry is None:
        return ensure_dir(default)
    folder = _in_use(entry)
    if folder is None:
        raise LocationUnavailable(
            f"{entry['path']} is not available. Reconnect its drive, or reset the folder in "
            "Settings."
        )
    return folder


def is_available(key: str) -> bool:
    """False while the folder `key` saves into is on a drive that is not there."""
    entry = _chosen_entry(key)
    return entry is None or _in_use(entry) is not None


def chosen_available(key: str) -> bool:
    """False while `key`'s chosen folder itself is on a drive that is not there, even while the
    folder a move was leaving stands in for it."""
    entry = _chosen_entry(key)
    return entry is None or not _unavailable(entry)


def wait_for_move(key: str) -> None:
    """Mark `key`'s move cut short as waiting for its chosen folder's drive, which keeps the folder
    it was leaving in use until the move is taken up again. For this process only: the next load
    tries the move again."""
    with _lock:
        entry = _cache.get(_db_key(), {}).get(key)
        if entry is not None and entry.get("moving_from"):
            entry["waiting"] = True


def moving_from(key: str) -> Optional[Path]:
    """The folder `key`'s files are still leaving, while a move is under way (or was cut short)."""
    entry = _chosen_entry(key)
    return Path(entry["moving_from"]) if entry and entry.get("moving_from") else None


def moving_from_available(key: str) -> bool:
    """False while the folder `key`'s files are leaving is on a drive that is not there: a folder
    missing from it then is not one the move emptied."""
    entry = _chosen_entry(key)
    return not entry or not entry.get("moving_from") or _source_available(entry)


def set_chosen(
    key: str,
    path: Optional[Path],
    moving_from: Optional[Path] = None,
) -> None:
    """Record `path` for `key`; None goes back to the default. `moving_from` marks a move under
    way, to be finished on the next load if it never is. Owner context only."""
    if key not in MOVABLE:
        raise ValueError(f"{key} files cannot move")
    updated = dict(_load())
    if path is None:
        updated.pop(key, None)
    else:
        updated[key] = {"path": str(path), "mount": mount_point(path)}
        if moving_from is not None:
            updated[key]["moving_from"] = str(moving_from)
            updated[key]["moving_mount"] = mount_point(moving_from)
    global _generation
    with _lock:
        _save(updated)
        _generation += 1
        _cache[_db_key()] = updated


def forget_cache() -> None:
    """For tests that swap the database under a live process."""
    global _generation
    with _lock:
        _generation += 1
        _cache.clear()
        _db_keys.clear()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime-local ``studio.db`` restore and persistent, consistent snapshots."""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
_DEBOUNCE_SECONDS = 0.5
_lock = threading.Lock()
_timer: threading.Timer | None = None
_pending_live_path: Path | None = None


def resolve_snapshot_path() -> Path | None:
    raw = (os.environ.get("UNSLOTH_STUDIO_DB_BACKUP") or "").strip()
    if not raw:
        return None
    if "\x00" in raw:
        raise ValueError("UNSLOTH_STUDIO_DB_BACKUP may not contain null bytes")
    path = Path(raw).expanduser()
    if not path.is_absolute() or ".." in raw.replace("\\", "/").split("/"):
        raise ValueError("UNSLOTH_STUDIO_DB_BACKUP must be an absolute path without '..'")
    return path.resolve(strict = False)


def _validate_database(path: Path) -> None:
    if not path.is_file():
        raise ValueError("snapshot is not a file")
    with sqlite3.connect(f"file:{path}?mode=ro", uri = True) as conn:
        result = conn.execute("PRAGMA quick_check").fetchone()
        if not result or result[0] != "ok":
            raise ValueError(f"SQLite quick_check failed: {result!r}")


def restore_snapshot_if_needed(live_path: Path) -> bool:
    """Restore a valid snapshot only when no runtime-local database exists."""
    try:
        snapshot = resolve_snapshot_path()
        if snapshot is None or live_path.exists() or not snapshot.exists():
            return False
        _validate_database(snapshot)
        live_path.parent.mkdir(parents = True, exist_ok = True)
        fd, temporary = tempfile.mkstemp(prefix = f".{live_path.name}.restore-", dir = live_path.parent)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            with sqlite3.connect(str(snapshot)) as source, sqlite3.connect(str(temporary_path)) as target:
                source.backup(target)
            _validate_database(temporary_path)
            os.replace(temporary_path, live_path)
        finally:
            temporary_path.unlink(missing_ok = True)
        return True
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("Studio DB snapshot restore skipped; using a fresh local DB: %s", exc)
        return False


def create_snapshot(live_path: Path) -> bool:
    """Publish an atomic SQLite backup without exposing WAL/SHM files."""
    temporary_path: Path | None = None
    try:
        destination = resolve_snapshot_path()
        if destination is None:
            return False
        if destination == live_path.resolve(strict = False):
            raise ValueError("snapshot destination must differ from the live database")
        destination.parent.mkdir(parents = True, exist_ok = True)
        fd, temporary = tempfile.mkstemp(prefix = f".{destination.name}.snapshot-", dir = destination.parent)
        os.close(fd)
        temporary_path = Path(temporary)
        with sqlite3.connect(str(live_path)) as source, sqlite3.connect(str(temporary_path)) as target:
            source.backup(target)
        _validate_database(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        return True
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("Could not persist Studio DB snapshot; last successful snapshot retained: %s", exc)
        return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok = True)


def _run_pending() -> None:
    global _timer, _pending_live_path
    with _lock:
        live_path = _pending_live_path
        _pending_live_path = None
        _timer = None
    if live_path is not None:
        create_snapshot(live_path)


def request_snapshot(live_path: Path) -> None:
    """Coalesce nearby metadata transitions; never fail their caller."""
    global _timer, _pending_live_path
    try:
        if resolve_snapshot_path() is None:
            return
    except ValueError as exc:
        logger.warning("Studio DB snapshot configuration ignored: %s", exc)
        return
    with _lock:
        _pending_live_path = live_path
        if _timer is not None:
            _timer.cancel()
        _timer = threading.Timer(_DEBOUNCE_SECONDS, _run_pending)
        _timer.daemon = True
        _timer.start()


def flush_snapshot(live_path: Path) -> bool:
    """Cancel a pending request and synchronously snapshot during shutdown."""
    global _timer, _pending_live_path
    with _lock:
        if _timer is not None:
            _timer.cancel()
        _timer = None
        _pending_live_path = None
    return create_snapshot(live_path)

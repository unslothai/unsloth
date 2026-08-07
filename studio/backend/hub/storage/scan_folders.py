# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistence for user-registered custom model scan folders.

Self-bootstrapping table inside the existing studio SQLite so the Hub module
doesn't have to modify upstream studio_db.py's schema init."""

from __future__ import annotations

import os
import platform
import sqlite3
import threading
from datetime import datetime, timezone

from storage.studio_db import get_connection
from hub.utils.paths import normalize_path
from utils.paths.external_media import is_linux_run_media_path, is_local_filesystem_root
from utils.paths.sensitive import (
    contains_sensitive_path_component as _shared_contains_sensitive_path_component,
)


_schema_lock = threading.Lock()
_schema_ready = False


def _denied_path_prefixes() -> list[str]:
    system = platform.system()
    if system == "Linux":
        return ["/proc", "/sys", "/dev", "/etc", "/boot", "/run"]
    if system == "Darwin":
        # realpath() resolves /etc -> /private/etc, /tmp -> /private/tmp on macOS,
        # so include the /private variants to avoid bypasses.
        return [
            "/System",
            "/Library",
            "/dev",
            "/etc",
            "/private/etc",
            "/tmp",
            "/private/tmp",
            "/var",
            "/private/var",
        ]
    if system == "Windows":
        win = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        return [os.path.normcase(p) for p in [win, pf, pf86]]
    return []


def is_denied_system_path(path: str) -> bool:
    """True if *path* is, or descends from, a denied system directory.

    Mirrors the denylist add_scan_folder() enforces at registration so the
    browser refuses /etc, /proc, C:\\Windows, etc. even when the allowlist holds
    a broad root (a Windows drive root C:\\ or a legacy-registered / root). The
    /run carve-out keeps Linux removable-media mounts browseable. Expects an
    already-resolved (realpath) path so symlinks cannot escape into a denied subtree.
    """
    is_win = platform.system() == "Windows"
    check = os.path.normcase(path) if is_win else path
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            if prefix == "/run" and is_linux_run_media_path(check):
                continue
            return True
    return False


def _contains_sensitive_path_component(path: str) -> bool:
    return _shared_contains_sensitive_path_component(path)


def contains_sensitive_path_component(path: str) -> bool:
    """Public predicate for the credential/config denylist (.ssh, .aws, ...).

    Shared with the folder browser so browse and register enforce one policy."""
    return _contains_sensitive_path_component(path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        collation = "COLLATE NOCASE" if platform.system() == "Windows" else ""
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS scan_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE {collation},
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        _schema_ready = True


def list_scan_folders() -> list[dict]:
    conn = get_connection()
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT id, path, created_at FROM scan_folders ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_scan_folder(path: str) -> dict:
    """Add a readable directory for the local OS user; not a multi-user sandbox."""
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")
    normalized = os.path.realpath(os.path.expanduser(normalize_path(path.strip())))

    if not os.path.exists(normalized):
        raise ValueError("Path does not exist")
    if not os.path.isdir(normalized):
        raise ValueError("Path must be a directory, not a file")
    if not os.access(normalized, os.R_OK | os.X_OK):
        raise ValueError("Path is not readable")
    if is_local_filesystem_root(normalized):
        # A local fs root ("/", "C:\\") would expose denied system dirs via browse;
        # a UNC share root (\\server\share) has none under it and stays registerable.
        raise ValueError("The filesystem root cannot be registered")
    if _contains_sensitive_path_component(normalized):
        raise ValueError("Credential or configuration directories are not allowed")

    is_win = platform.system() == "Windows"
    check = os.path.normcase(normalized) if is_win else normalized
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            if prefix == "/run" and is_linux_run_media_path(check):
                continue
            raise ValueError(f"Path under {prefix} is not allowed")

    conn = get_connection()
    try:
        _ensure_schema(conn)
        now = datetime.now(timezone.utc).isoformat()
        if is_win:
            existing = conn.execute(
                "SELECT id, path, created_at FROM scan_folders WHERE path = ? COLLATE NOCASE",
                (normalized,),
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id, path, created_at FROM scan_folders WHERE path = ?",
                (normalized,),
            ).fetchone()
        if existing is not None:
            return dict(existing)
        try:
            conn.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)",
                (normalized, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        fallback_sql = (
            "SELECT id, path, created_at FROM scan_folders WHERE path = ? COLLATE NOCASE"
            if is_win
            else "SELECT id, path, created_at FROM scan_folders WHERE path = ?"
        )
        row = conn.execute(fallback_sql, (normalized,)).fetchone()
        if row is None:
            raise ValueError("Folder was concurrently removed")
        return dict(row)
    finally:
        conn.close()


def remove_scan_folder(id: int) -> None:
    # sqlite INTEGER is signed 64-bit; ids outside that range cannot exist.
    if not -(2**63) <= id < 2**63:
        return
    conn = get_connection()
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM scan_folders WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()

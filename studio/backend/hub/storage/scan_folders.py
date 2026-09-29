# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistence for user-registered custom model scan folders.

Self-bootstrapping table inside the existing studio SQLite so the Hub module
doesn't have to modify upstream studio_db.py's schema init."""

from __future__ import annotations

import os
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.studio_db import get_connection
from hub.utils.paths import normalize_path
from utils.paths.external_media import is_linux_run_media_path, is_local_filesystem_root
from utils.paths.path_utils import macos_volume_ignores_case
from utils.paths.scan_folder_health import is_readable_dir
from utils.paths.sensitive import (
    contains_sensitive_path_component as _shared_contains_sensitive_path_component,
)


def _denied_path_prefixes() -> list[str]:
    system = platform.system()
    if system == "Linux":
        return ["/proc", "/sys", "/dev", "/etc", "/boot", "/run"]
    if system == "Darwin":
        # realpath() resolves /etc -> /private/etc and /tmp -> /private/tmp on macOS, so include the
        # /private variants to avoid bypasses.
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
    return _denied_prefix(path) is not None


def _denied_prefix(path: str) -> str | None:
    check = _comparable_path(path)
    for prefix in _denied_path_prefixes():
        if is_within_any(path, [prefix]) and not (
            prefix == "/run" and is_linux_run_media_path(check)
        ):
            return prefix
    return None


# Longest first: \\?\UNC\server\share is the share \\server\share. After normcase, so lower case.
_EXTENDED_PREFIXES = (
    ("\\\\?\\unc\\", "\\\\"),
    ("\\\\.\\unc\\", "\\\\"),
    ("\\\\?\\", ""),
    ("\\\\.\\", ""),
)


def _comparable_path(path: str, fold: bool | None = None) -> str:
    """``path`` spelled the way the filesystem compares it, for prefix checks. Windows ignores
    case, and so does macOS unless the volume is case-sensitive (``fold`` says, else it is asked:
    /LIBRARY is /Library on a default volume only). realpath() keeps a Windows extended-length
    prefix (``\\\\?\\C:\\Windows``) that would otherwise hide the folder behind it."""
    system = platform.system()
    if system == "Windows":
        check = os.path.normcase(path)
        for extended, plain in _EXTENDED_PREFIXES:
            if check.startswith(extended):
                return plain + check[len(extended) :]
        return check
    if system == "Darwin":
        if fold is None:
            fold = macos_volume_ignores_case(path)
        return path.casefold() if fold else path
    return path


def is_within_any(path: str, prefixes) -> bool:
    """True if resolved ``path`` is, or is inside, one of ``prefixes``, compared the way the
    filesystem holding ``path`` compares names (see ``_comparable_path``)."""
    fold = platform.system() == "Darwin" and macos_volume_ignores_case(path)
    check = _comparable_path(path, fold)
    for prefix in prefixes:
        prefix = _comparable_path(str(prefix), fold).rstrip(os.sep) or os.sep
        if check == prefix or check.startswith(
            prefix if prefix.endswith(os.sep) else prefix + os.sep
        ):
            return True
    return False


def _contains_sensitive_path_component(path: str) -> bool:
    return _shared_contains_sensitive_path_component(path)


def contains_sensitive_path_component(path: str) -> bool:
    """Public predicate for the credential/config denylist (.ssh, .aws, ...).

    Shared with the folder browser so browse and register enforce one policy."""
    return _contains_sensitive_path_component(path)


def list_scan_folders() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, path, created_at FROM scan_folders ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_scan_folder_with_status(path: str) -> tuple[dict, bool]:
    """Add a readable scan folder and return its row plus whether it was inserted."""
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")
    normalized = os.path.realpath(os.path.expanduser(normalize_path(path.strip())))

    if not os.path.exists(normalized):
        raise ValueError("Path does not exist")
    if not os.path.isdir(normalized):
        raise ValueError("Path must be a directory, not a file")
    if is_local_filesystem_root(normalized):
        # A local fs root would expose denied system dirs via browse; a UNC share root has none under it and
        # stays registerable.
        raise ValueError("The filesystem root cannot be registered")
    if _contains_sensitive_path_component(normalized):
        raise ValueError("Credential or configuration directories are not allowed")
    from utils.paths.storage_roots import within_account

    if not within_account(Path(normalized)):
        raise ValueError("Path is outside this account's workspace")

    is_win = platform.system() == "Windows"
    denied = _denied_prefix(normalized)
    if denied is not None:
        raise ValueError(f"Path under {denied} is not allowed")

    # Last, so a denied path is never opened. Mirrors studio_db.py.
    if not is_readable_dir(normalized):
        raise ValueError("Path is not readable")

    conn = get_connection()
    try:
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
            return dict(existing), False
        inserted = False
        try:
            conn.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)",
                (normalized, now),
            )
            conn.commit()
            inserted = True
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
        return dict(row), inserted
    finally:
        conn.close()


def add_scan_folder(path: str) -> dict:
    """Add a readable directory for the local OS user; not a multi-user sandbox."""
    row, _ = add_scan_folder_with_status(path)
    return row


def remove_scan_folder(id: int) -> bool:
    # sqlite INTEGER is signed 64-bit; ids outside that range cannot exist.
    if not -(2**63) <= id < 2**63:
        return False
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM scan_folders WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

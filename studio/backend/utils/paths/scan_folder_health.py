# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Readability of user-registered model scan folders.

Two halves of one problem: prove a folder can be listed before registering it,
and remember the folders a later scan could not read so the UI can say so
instead of showing an empty model list.
"""

from __future__ import annotations

import errno
import os

# Readable, or not scanned yet.
STATUS_OK = "ok"
# Refused by the OS: mode bits, an ACL, macOS TCC, Windows Controlled Folder Access.
STATUS_PERMISSION_DENIED = "permission_denied"
# Deleted, renamed, or on an unmounted volume.
STATUS_MISSING = "missing"
# Any other OSError: I/O error, dead network mount, symlink loop.
STATUS_UNREADABLE = "unreadable"


def is_readable_dir(path: str) -> bool:
    """True only if ``path`` can actually be listed.

    ``os.access`` answers from mode bits alone, so it returns True for a
    directory macOS TCC or a Windows ACL still refuses. Opening the directory is
    the authoritative answer and costs one syscall on a path the user just picked.
    """
    if not os.access(path, os.R_OK | os.X_OK):
        return False
    try:
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError:
        return False
    return True


def classify_scan_error(error: OSError) -> str:
    """Map an error the scan already raised onto a status the UI can render."""
    if isinstance(error, PermissionError) or error.errno in (errno.EACCES, errno.EPERM):
        return STATUS_PERMISSION_DENIED
    if isinstance(error, FileNotFoundError) or error.errno in (errno.ENOENT, errno.ENOTDIR):
        return STATUS_MISSING
    return STATUS_UNREADABLE


# Written only when a scan fails and read only when the folder list is rendered,
# so a healthy scan pays nothing. Bounded by the registered folder count.
_MAX_TRACKED = 256
_failed: dict[str, str] = {}


def record_scan_failure(path: str, error: OSError) -> None:
    """Remember why a scan skipped ``path``."""
    if len(_failed) >= _MAX_TRACKED:
        _failed.clear()
    _failed[path] = classify_scan_error(error)


def clear_scan_failure(path: str) -> None:
    """Forget a past failure once the folder reads again."""
    if _failed:
        _failed.pop(path, None)


def note_scan_folder_scanned(path: str, *, found: bool) -> None:
    """Record the outcome of a scan that did not raise.

    A folder that yielded models is healthy. One that yielded nothing is either
    genuinely empty or gone (renamed, deleted, unmounted volume), and the
    scanners return an empty list for both. One stat separates them, and only
    runs for a folder that had no scan work to do.
    """
    if found or os.path.isdir(path):
        clear_scan_failure(path)
        return
    if len(_failed) >= _MAX_TRACKED:
        _failed.clear()
    _failed[path] = STATUS_MISSING


def scan_folder_status(path: str) -> str:
    """Last known status for ``path``. A dict lookup, never touches the disk."""
    if not _failed:
        return STATUS_OK
    return _failed.get(path, STATUS_OK)


def annotate_scan_folders(folders: list[dict]) -> list[dict]:
    """Copy each folder row with its status attached."""
    if not _failed:
        return [{**folder, "status": STATUS_OK} for folder in folders]
    return [
        {**folder, "status": _failed.get(str(folder.get("path", "")), STATUS_OK)}
        for folder in folders
    ]

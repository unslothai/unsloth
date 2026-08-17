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


# Enough to catch a folder whose model dirs are all refused without walking a
# large one. The scan that got here already returned nothing.
_CHILD_PROBE_LIMIT = 64


def probe_status(path: str, *, children: bool = False) -> str:
    """Open ``path`` and report what the OS says. No walking.

    With ``children``, also open up to ``_CHILD_PROBE_LIMIT`` subdirectories and
    report the first refusal. A root can list fine while every model under it is
    denied, and the scanners skip unreadable children silently, so both arrive as
    the same empty list. Stops at the first bad child, so the denied-everything
    case costs one extra open.
    """
    try:
        with os.scandir(path) as entries:
            if not children:
                next(entries, None)
                return STATUS_OK
            probed = 0
            for entry in entries:
                if probed >= _CHILD_PROBE_LIMIT:
                    break
                try:
                    if not entry.is_dir():
                        continue
                except OSError as error:
                    return classify_scan_error(error)
                probed += 1
                child_status = probe_status(entry.path)
                if child_status != STATUS_OK:
                    return child_status
    except OSError as error:
        return classify_scan_error(error)
    return STATUS_OK


def is_readable_dir(path: str) -> bool:
    """True only if ``path`` can actually be listed.

    ``os.access`` answers from mode bits alone, so it returns True for a
    directory macOS TCC or a Windows ACL still refuses. Opening the directory is
    the authoritative answer and costs one syscall on a path the user just picked.
    """
    if not os.access(path, os.R_OK | os.X_OK):
        return False
    return probe_status(path) == STATUS_OK


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
    """Record the outcome of a scan of ``path``.

    A folder that yielded models is healthy, and costs nothing here. One that
    yielded nothing is empty, gone, or refused, and the scanners return an empty
    list for all three: some raise, some swallow the error. So ask the OS once,
    and only for a folder that had no scan work to do anyway.
    """
    if found:
        clear_scan_failure(path)
        return
    status = probe_status(path, children = True)
    if status == STATUS_OK:
        clear_scan_failure(path)
        return
    if len(_failed) >= _MAX_TRACKED:
        _failed.clear()
    _failed[path] = status


def scan_folder_status(path: str) -> str:
    """Last known status for ``path``. A dict lookup, never touches the disk."""
    if not _failed:
        return STATUS_OK
    return _failed.get(path, STATUS_OK)


def refresh_failed_scan_folders(folders: list[dict]) -> None:
    """Re-check the folders currently marked bad, and only those.

    The row tells the user to fix permissions and reopen the dialog, so reopening
    has to be able to clear it. Nothing else rechecks between inventory scans.
    A healthy folder is not in the registry, so it is never opened here.
    """
    if not _failed:
        return
    for folder in folders:
        path = str(folder.get("path", ""))
        previous = _failed.get(path)
        if previous is None:
            continue
        status = probe_status(path, children = True)
        if status == STATUS_OK:
            _failed.pop(path, None)
        elif status != previous:
            _failed[path] = status


def annotate_scan_folders(folders: list[dict]) -> list[dict]:
    """Copy each folder row with its status attached."""
    if not _failed:
        return [{**folder, "status": STATUS_OK} for folder in folders]
    return [
        {**folder, "status": _failed.get(str(folder.get("path", "")), STATUS_OK)}
        for folder in folders
    ]

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


# Deep enough for <root>/<publisher>/<model>, the LM Studio and HF cache shape.
_PROBE_DEPTH = 2
# Total opens for one probe, whatever the shape. Bounds the cost; the depth does not.
_PROBE_OPEN_LIMIT = 64


def _probe_dir(path: str, *, depth: int, budget: list[int]) -> str:
    """Open ``path``, then its subdirectories down to ``depth``, depth first.

    Depth first so a denied model is found in three opens rather than after every
    publisher. ``budget`` is shared across the whole walk and decremented per open.
    """
    if budget[0] <= 0:
        return STATUS_OK
    budget[0] -= 1
    subdirs: list[str] = []
    try:
        with os.scandir(path) as entries:
            if depth <= 0:
                # Only need to know it opens.
                next(entries, None)
                return STATUS_OK
            for entry in entries:
                if len(subdirs) >= budget[0]:
                    break
                try:
                    if entry.is_dir():
                        subdirs.append(entry.path)
                except OSError as error:
                    return classify_scan_error(error)
    except OSError as error:
        return classify_scan_error(error)
    for subdir in subdirs:
        status = _probe_dir(subdir, depth = depth - 1, budget = budget)
        if status != STATUS_OK:
            return status
        if budget[0] <= 0:
            break
    return STATUS_OK


def probe_status(path: str, *, children: bool = False) -> str:
    """Open ``path`` and report what the OS says. No walking.

    With ``children``, also open what is under it. A root can list fine while the
    models below it are denied, and the scanners skip an unreadable entry
    silently, so both arrive as the same empty list.
    """
    if not children:
        return _probe_dir(path, depth = 0, budget = [1])
    return _probe_dir(path, depth = _PROBE_DEPTH, budget = [_PROBE_OPEN_LIMIT])


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

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
from typing import Optional

# Readable, or not scanned yet.
STATUS_OK = "ok"
# Refused by the OS: mode bits, an ACL, macOS TCC, Windows Controlled Folder Access.
STATUS_PERMISSION_DENIED = "permission_denied"
# Deleted, renamed, or on an unmounted volume.
STATUS_MISSING = "missing"
# Any other OSError: I/O error, dead network mount, symlink loop.
STATUS_UNREADABLE = "unreadable"
# The folder works, but something inside it is refused.
STATUS_PARTIAL = "partial"
# Internal only: the probe ran out of budget before it saw everything
STATUS_UNKNOWN = "unknown"


# Deep enough for <root>/<publisher>/<model>, the LM Studio and HF cache shape.
_PROBE_DEPTH = 2
# Total opens for one probe, whatever the shape. Bounds the cost; the depth does not.
_PROBE_OPEN_LIMIT = 64
# A huggingface_hub cache keeps the weights only in <root>/models--org--name/snapshots/<commit>/, so refusing that
# directory leaves the folder looking healthy and empty at once.
_HF_SNAPSHOTS_DIR = "snapshots"


def _probe_dir(path: str, *, depth: int, budget: list[int]) -> tuple[str, Optional[str]]:
    """Open ``path``, then its subdirectories down to ``depth``, depth first.

    Returns the status and the directory that refused. Depth first so a denied
    model is found in three opens rather than after every publisher. ``budget``
    is shared across the walk and spent one per open. Running out returns
    ``STATUS_UNKNOWN``: the tail was never looked at, which is not the same as
    finding it healthy.
    """
    if budget[0] <= 0:
        return STATUS_UNKNOWN, None
    budget[0] -= 1
    subdirs: list[tuple[str, str]] = []
    truncated = False
    try:
        with os.scandir(path) as entries:
            if depth <= 0:
                # Only need to know it opens.
                next(entries, None)
                return STATUS_OK, None
            for entry in entries:
                if len(subdirs) >= budget[0]:
                    truncated = True
                    break
                try:
                    if entry.is_dir():
                        subdirs.append((entry.name, entry.path))
                except OSError as error:
                    return classify_scan_error(error), entry.path
    except OSError as error:
        return classify_scan_error(error), path
    for name, subdir in subdirs:
        child_depth = depth - 1
        if child_depth <= 0 and name == _HF_SNAPSHOTS_DIR:
            # Spend one more level here rather than raising the depth everywhere: a diffusers pipeline's component
            # directories would otherwise burn the budget.
            child_depth = 1
        status, cause = _probe_dir(subdir, depth = child_depth, budget = budget)
        if status == STATUS_MISSING:
            # It was in the listing a moment ago and is gone now (a model being deleted, or a download renaming its temp
            # directory), which says nothing about the folder the user registered.
            continue
        if status != STATUS_OK:
            return status, cause or subdir
    return (STATUS_UNKNOWN, None) if truncated else (STATUS_OK, None)


def probe_folder(path: str, *, children: bool = False) -> tuple[str, Optional[str]]:
    """Status of ``path`` plus the directory that refused, if any."""
    if not children:
        return _probe_dir(path, depth = 0, budget = [1])
    return _probe_dir(path, depth = _PROBE_DEPTH, budget = [_PROBE_OPEN_LIMIT])


def probe_status(path: str, *, children: bool = False) -> str:
    """Open ``path`` and report what the OS says. No walking.

    With ``children``, also open what is under it. A root can list fine while the
    models below it are denied, and the scanners skip an unreadable entry
    silently, so both arrive as the same empty list.
    """
    return probe_folder(path, children = children)[0]


def is_readable_dir(path: str) -> bool:
    """True only if ``path`` can actually be listed.

    ``os.access`` answers from mode bits alone, so it returns True for a
    directory macOS TCC or a Windows ACL still refuses. Opening the directory is
    the authoritative answer and costs one syscall on a path the user just picked.
    """
    if not os.access(path, os.R_OK | os.X_OK):
        return False
    return probe_status(path) == STATUS_OK


# Windows reports the real reason in ``winerror``: CPython folds ERROR_NOT_READY, ERROR_CRC and every media failure onto
# EACCES, so an ejected card reader reads as a permissions problem.
# ``errno`` is a lossy translation, and CPython's PC/errmap.h folds ERROR_GEN_FAILURE onto EACCES as well.
_WINDOWS_PERMISSION = frozenset((5, 65, 1314))
_WINDOWS_MISSING = frozenset((2, 3, 15, 20, 21, 53, 55, 67, 161, 206, 267))


def classify_scan_error(error: OSError) -> str:
    """Map an error the scan already raised onto a status the UI can render."""
    # Absent on POSIX, and None unless CPython set it, so this never shadows errno.
    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        if winerror in _WINDOWS_PERMISSION:
            return STATUS_PERMISSION_DENIED
        return STATUS_MISSING if winerror in _WINDOWS_MISSING else STATUS_UNREADABLE
    if isinstance(error, PermissionError) or error.errno in (errno.EACCES, errno.EPERM):
        return STATUS_PERMISSION_DENIED
    if isinstance(error, FileNotFoundError) or error.errno in (errno.ENOENT, errno.ENOTDIR):
        return STATUS_MISSING
    return STATUS_UNREADABLE


# Read on every folder list, written only when a probe finds something wrong.
# Each value is (status, the directory that refused), so a recheck can settle a folder in one open instead of walking it
# again. Bounded by the folder count.
_MAX_TRACKED = 256
_failed: dict[str, tuple[str, str]] = {}


def _record(path: str, status: str, cause: str) -> None:
    if len(_failed) >= _MAX_TRACKED:
        _failed.clear()
    _failed[path] = (status, cause)


def record_scan_failure(path: str, error: OSError) -> None:
    """Remember why a scan skipped ``path``."""
    _record(path, classify_scan_error(error), path)


def clear_scan_failure(path: str) -> None:
    """Forget a past failure once the folder reads again."""
    if _failed:
        _failed.pop(path, None)


def note_scan_folder_scanned(path: str, *, found: bool) -> None:
    """Record the outcome of a scan of ``path``.

    Empty, gone, refused, or working with one model refused: the scanners return
    the same list for all of them, because they swallow the error per entry. So
    ask the OS instead of trying to read it back out of them. Bounded by
    ``_PROBE_OPEN_LIMIT`` opens per folder.
    """
    status, cause = probe_folder(path, children = True)
    if status == STATUS_UNKNOWN:
        # Budget gone before the tail was reached, so this proves nothing either
        # way. Settle it on the one directory that refused last time.
        _recheck_cause(path)
        return
    if status == STATUS_OK:
        clear_scan_failure(path)
        return
    # Models came back, so the folder itself is fine and only part of it is
    # refused. Saying it cannot be read would contradict the rows on screen.
    if found:
        status = STATUS_PARTIAL
    _record(path, status, cause or path)


def _recheck_cause(path: str) -> None:
    """Clear ``path`` if the directory that refused last time opens now.

    One open, and it does not depend on the probe reaching that directory again,
    so recovery works on a folder too wide to walk inside the budget.
    """
    entry = _failed.get(path)
    if entry is None:
        return
    if probe_status(entry[1]) == STATUS_OK:
        _failed.pop(path, None)


def scan_folder_status(path: str) -> str:
    """Last known status for ``path``. A dict lookup, never touches the disk."""
    if not _failed:
        return STATUS_OK
    entry = _failed.get(path)
    return STATUS_OK if entry is None else entry[0]


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
        entry = _failed.get(path)
        if entry is None:
            continue
        previous = entry[0]
        status, cause = probe_folder(path, children = True)
        if status == STATUS_UNKNOWN:
            _recheck_cause(path)
            continue
        if status == STATUS_OK:
            _failed.pop(path, None)
            continue
        if (
            previous == STATUS_PARTIAL
            and status == STATUS_PERMISSION_DENIED
            and cause is not None
            and cause != path
        ):
            # The probe cannot see that models were found.
            status = STATUS_PARTIAL
        _record(path, status, cause or path)


def annotate_scan_folders(folders: list[dict]) -> list[dict]:
    """Copy each folder row with its status attached."""
    if not _failed:
        return [{**folder, "status": STATUS_OK} for folder in folders]
    return [
        {**folder, "status": scan_folder_status(str(folder.get("path", "")))} for folder in folders
    ]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External media path helpers."""

from __future__ import annotations

import getpass
import os
import platform
import string
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from utils.paths.sensitive import (
    contains_sensitive_path_component,
    is_sensitive_path_component,
)


def is_local_filesystem_root(path: str, *, _pathmod = os.path) -> bool:
    """True for a bare local filesystem root -- POSIX ``/``, a drive root ``C:\\``,
    or a device-namespace volume root like ``\\\\?\\C:\\`` or
    ``\\\\?\\Volume{GUID}\\`` -- which sit above denied system dirs, but NOT a UNC
    share root (``\\\\server\\share`` or its ``\\\\?\\UNC\\...`` form), which has
    none under it and was registerable before this guard. ``splitdrive`` is empty
    on POSIX servers, so this reduces to the plain ``dirname == self`` test there.
    ``_pathmod`` lets tests drive ``ntpath`` semantics on a POSIX CI.
    """
    # Resolve the Windows device / extended-length namespace, where \\?\C:\,
    # \\.\C:\ and \\?\Volume{GUID}\ are all bare LOCAL volume roots (rejected)
    # while only \\?\UNC\server\share is a UNC share (handled like \\server\share).
    if path[:4].lower() in ("\\\\?\\", "\\\\.\\"):
        rest = path[4:]
        if rest[:4].lower() == "unc\\":
            path = "\\\\" + rest[4:]
        else:
            # A device volume root is just the volume specifier (C:, Volume{GUID})
            # with no further component; a deeper path is an ordinary folder.
            core = rest.rstrip("\\/")
            return "\\" not in core and "/" not in core
    if _pathmod.dirname(path) != path:
        return False
    drive, _ = _pathmod.splitdrive(path)
    return drive[:2] not in ("\\\\", "//")


def _is_linux_media_mount_path(path: str, media_root: Path | str) -> bool:
    normalized = os.path.normpath(os.path.realpath(os.path.expanduser(path)))
    root = os.path.normpath(os.path.realpath(os.path.expanduser(str(media_root))))
    try:
        rel = os.path.relpath(normalized, root)
    except ValueError:
        return False
    if rel == "." or rel == ".." or rel.startswith(f"..{os.sep}"):
        return False
    parts = [part for part in rel.split(os.sep) if part]
    return len(parts) >= 2 and all(part not in (".", "..") for part in parts[:2])


def is_linux_run_media_path(path: str) -> bool:
    """True for Linux removable-media paths under /run/media/<user>/<volume>."""
    if platform.system() != "Linux":
        return False
    return _is_linux_media_mount_path(path, "/run/media")


def _current_username() -> str | None:
    try:
        user = getpass.getuser().strip()
    except Exception:
        return None
    return user or None


def _contains_sensitive_media_component(path: Path, media_root: Path) -> bool:
    try:
        rel = path.relative_to(media_root)
    except ValueError:
        rel = path
    return contains_sensitive_path_component(str(rel))


def linux_run_media_mount_roots(
    base: Path | str = "/run/media", *, user: str | None = None
) -> list[Path]:
    """Readable /run/media/<user>/<volume> roots for the folder browser."""
    if platform.system() != "Linux":
        return []
    user = user or _current_username()
    if not user or user in (".", "..") or os.sep in user:
        return []
    base_path = Path(base)
    try:
        resolved_base = base_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return []

    roots: list[Path] = []
    seen: set[str] = set()
    user_dir = base_path / user
    try:
        if not user_dir.is_dir():
            return []
        volume_dirs = list(user_dir.iterdir())
    except (OSError, RuntimeError, ValueError):
        return []
    for volume_dir in volume_dirs:
        if is_sensitive_path_component(volume_dir.name):
            continue
        try:
            resolved = volume_dir.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if not _is_linux_media_mount_path(str(resolved), resolved_base):
            continue
        if _contains_sensitive_media_component(resolved, resolved_base):
            continue
        key = os.path.normcase(os.path.realpath(str(resolved)))
        if key in seen:
            continue
        try:
            is_dir = resolved.is_dir()
        except OSError:
            continue
        if is_dir and os.access(resolved, os.R_OK | os.X_OK):
            seen.add(key)
            roots.append(resolved)
    return roots


def _active_windows_drive_bitmask() -> int:
    """Active-logical-drive bitmask from ``GetLogicalDrives`` (bit 0 = ``A:``), or ``0`` when unavailable.

    A fast non-blocking call that lets :func:`windows_drive_roots` skip the
    ``os.path.isdir`` probe on unmapped letters. A disconnected network mapping
    stays set here, so it does not guard the reconnect stall on its own;
    :func:`windows_drive_roots` bounds each surviving probe too. Returns ``0``
    (probe every letter) when ctypes/``windll`` is missing.
    """
    try:
        import ctypes
        return int(ctypes.windll.kernel32.GetLogicalDrives())
    except Exception:  # noqa: BLE001 -- best-effort; fall back to probing all letters
        return 0


# A disconnected mapped drive stays set in the GetLogicalDrives bitmask, so
# ``os.path.isdir`` on it can block for tens of seconds. Bound each drive probe
# so one stale mapping cannot stall a whole folder-browser request.
_DRIVE_PROBE_TIMEOUT_S = 2.0


def _readable_dirs_within(paths: Iterable[str], timeout: float) -> set[str]:
    """Which of *paths* are readable directories, probed concurrently under one overall *timeout* (seconds).

    Each path is checked (``os.path.isdir`` + ``os.access(R_OK)``) in its own
    daemon thread and the call waits at most *timeout* total, not per path, so N
    stalled network drives add ~timeout instead of N*timeout. A path not
    answering ``True`` by the deadline is treated as unreadable. The daemon
    threads are never joined past the deadline, so a stuck OS call cannot delay
    interpreter exit or block the caller (``os.path.isdir`` releases the GIL).
    """
    paths = list(paths)  # fixed input we can iterate twice; one probe per path
    results: dict[str, bool] = {}

    def _probe(path: str) -> None:
        try:
            results[path] = os.path.isdir(path) and os.access(path, os.R_OK)
        except OSError:
            results[path] = False

    threads: list[threading.Thread] = []
    for path in paths:
        thread = threading.Thread(target = _probe, args = (path,), daemon = True)
        thread.start()
        threads.append(thread)

    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))

    # Iterate the fixed input, not results.items(): a probe that timed out is
    # still alive and may insert its key here, which would raise "dictionary
    # changed size during iteration". results.get() is an atomic read.
    return {path for path in paths if results.get(path)}


def _readable_dir_within(path: str, timeout: float) -> bool:
    """``os.path.isdir(path) and os.access(path, R_OK)``, bounded by *timeout* seconds; single-path wrapper over :func:`_readable_dirs_within`."""
    return path in _readable_dirs_within((path,), timeout)


def windows_drive_roots(drive_letters: Iterable[str] = string.ascii_uppercase) -> list[Path]:
    """Readable logical drive roots (``C:\\``, ``D:\\`` ...) for the folder browser; the Windows analog of :func:`linux_run_media_mount_roots`.

    Without it the allowlist and chips only reach the home drive, so a user
    cannot navigate from ``C:`` to ``D:``/``E:``. ``GetLogicalDrives`` drops
    unmapped letters; the rest are probed concurrently under a single timeout
    and kept only if readable in time. A disconnected mapped drive stays active
    in the bitmask and its ``os.path.isdir`` can hang for tens of seconds, so
    parallel probing bounds the added delay at ~one timeout rather than one per
    drive. Returns ``[]`` off Windows.
    """
    if platform.system() != "Windows":
        return []

    active_mask = _active_windows_drive_bitmask()
    candidates: list[str] = []
    seen: set[str] = set()
    for letter in drive_letters:
        letter = letter.strip().rstrip(":").upper()
        if len(letter) != 1 or letter not in string.ascii_uppercase:
            continue
        if active_mask and not active_mask & (1 << (ord(letter) - ord("A"))):
            continue
        root_text = f"{letter}:\\"
        key = os.path.normcase(root_text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(root_text)

    # Bounded concurrent probe: an active bitmask bit can still be a
    # disconnected mapping whose os.path.isdir blocks, so probe all at once.
    readable = _readable_dirs_within(candidates, _DRIVE_PROBE_TIMEOUT_S)
    return [Path(root_text) for root_text in candidates if root_text in readable]

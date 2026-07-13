# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External media path helpers."""

from __future__ import annotations

import getpass
import os
import platform
import string
import threading
from collections.abc import Iterable
from pathlib import Path

from utils.paths.sensitive import (
    contains_sensitive_path_component,
    is_sensitive_path_component,
)


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
    """Bitmask of active logical drives from ``GetLogicalDrives`` (bit 0 = ``A:``),
    or ``0`` when the call is unavailable.

    This is a fast, non-blocking OS call. It lets :func:`windows_drive_roots`
    skip the ``os.path.isdir`` probe on drive letters with no mapping at all.
    A letter mapped to a *disconnected* network share stays set in this bitmask
    (``GetLogicalDrives`` includes mapped network drives), so it does not guard
    against the reconnect stall on its own — :func:`windows_drive_roots` bounds
    each surviving probe as well (see :func:`_readable_dir_within`). Returns
    ``0`` (probe every letter) when ctypes/``windll`` is unavailable, so the
    helper degrades gracefully.
    """
    try:
        import ctypes
        return int(ctypes.windll.kernel32.GetLogicalDrives())
    except Exception:  # noqa: BLE001 -- best-effort; fall back to probing all letters
        return 0


# A disconnected but still-mapped network drive stays set in the
# GetLogicalDrives bitmask, so ``os.path.isdir`` on it can block for tens of
# seconds while Windows tries to reconnect. Bound each drive probe with this
# timeout so one stale mapping cannot stall a whole folder-browser request.
_DRIVE_PROBE_TIMEOUT_S = 2.0


def _readable_dir_within(path: str, timeout: float) -> bool:
    """``os.path.isdir(path) and os.access(path, R_OK)``, bounded by *timeout*
    seconds.

    Runs the probe in a daemon thread and reports ``False`` if it does not
    answer in time, so a hung drive (e.g. a disconnected-but-mapped network
    share) is skipped instead of blocking the caller. The thread is never
    joined past the timeout and is a daemon, so the stuck OS call cannot delay
    interpreter exit. ``os.path.isdir`` releases the GIL during the syscall, so
    the still-running probe does not block the caller after the timeout either.
    """
    result: dict[str, bool] = {}

    def _probe() -> None:
        try:
            result["ok"] = os.path.isdir(path) and os.access(path, os.R_OK)
        except OSError:
            result["ok"] = False

    thread = threading.Thread(target = _probe, daemon = True)
    thread.start()
    thread.join(timeout)
    return result.get("ok", False)


def windows_drive_roots(drive_letters: Iterable[str] = string.ascii_uppercase) -> list[Path]:
    """Readable logical drive roots (``C:\\``, ``D:\\`` ...) for the folder browser.

    The Windows analog of :func:`linux_run_media_mount_roots`. Without it the
    browser's allowlist and suggestion chips only reach roots on the home drive,
    so a user cannot navigate from ``C:`` to ``D:``/``E:`` to pick a model
    directory. ``GetLogicalDrives`` first drops letters with no mapping; each
    remaining candidate is then probed under a short timeout and included only
    if it resolves to a readable directory in time. The timeout matters because
    a mapped-but-disconnected network drive stays active in the bitmask and its
    ``os.path.isdir`` can otherwise hang for tens of seconds, stalling every
    folder-browser request. Returns ``[]`` off Windows, so callers on
    Linux/macOS are unaffected.
    """
    if platform.system() != "Windows":
        return []

    active_mask = _active_windows_drive_bitmask()
    roots: list[Path] = []
    seen: set[str] = set()
    for letter in drive_letters:
        letter = letter.strip().rstrip(":").upper()
        if len(letter) != 1 or letter not in string.ascii_uppercase:
            continue
        if active_mask and not active_mask & (1 << (ord(letter) - ord("A"))):
            continue
        root_text = f"{letter}:\\"
        # Bounded probe: an active bitmask bit can still be a disconnected
        # network mapping, whose os.path.isdir blocks for tens of seconds.
        if not _readable_dir_within(root_text, _DRIVE_PROBE_TIMEOUT_S):
            continue
        key = os.path.normcase(root_text)
        if key in seen:
            continue
        seen.add(key)
        roots.append(Path(root_text))
    return roots

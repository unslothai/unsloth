# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External media path helpers."""

from __future__ import annotations

import getpass
import os
import platform
import string
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
    skip the ``os.path.isdir`` probe on inactive drive letters — probing a
    drive letter mapped to a disconnected network share can otherwise block the
    caller for tens of seconds each. Returns ``0`` (probe every letter) when
    ctypes/``windll`` is unavailable, so the helper degrades gracefully.
    """
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetLogicalDrives())
    except Exception:  # noqa: BLE001 -- best-effort; fall back to probing all letters
        return 0


def windows_drive_roots(drive_letters: Iterable[str] = string.ascii_uppercase) -> list[Path]:
    """Readable logical drive roots (``C:\\``, ``D:\\`` ...) for the folder browser.

    The Windows analog of :func:`linux_run_media_mount_roots`. Without it the
    browser's allowlist and suggestion chips only reach roots on the home drive,
    so a user cannot navigate from ``C:`` to ``D:``/``E:`` to pick a model
    directory. Active drives are resolved from ``GetLogicalDrives`` first so
    disconnected/unmapped letters are never probed (a stray ``os.path.isdir`` on
    a dead network drive can hang for tens of seconds); each surviving candidate
    is then included only if it resolves to a readable directory, so absent or
    empty drives never show as dead entries. Returns ``[]`` off Windows, so
    callers on Linux/macOS are unaffected.
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
        try:
            if not os.path.isdir(root_text):
                continue
        except OSError:
            continue
        if not os.access(root_text, os.R_OK):
            continue
        key = os.path.normcase(root_text)
        if key in seen:
            continue
        seen.add(key)
        roots.append(Path(root_text))
    return roots

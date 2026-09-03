# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared allowlist helpers for secure local model file access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loggers import get_logger

from hub.utils.paths import (
    exports_root,
    hf_default_cache_dir,
    legacy_hf_cache_dir,
    normalize_path,
    outputs_root,
    studio_root,
    well_known_model_dirs,
)
from utils.paths.external_media import (
    linux_run_media_mount_roots,
    macos_volume_roots,
    windows_drive_roots,
)
from hub.services.models.common import _safe_is_dir
from hub.services.models.local_inventory import _resolve_hf_cache_dir

logger = get_logger(__name__)


def _build_browse_allowlist(
    media_roots: Optional[list[Path]] = None, drive_roots: Optional[list[Path]] = None
) -> list[Path]:
    """Root directories the browser may walk (also seeds the suggestion chips): HOME, resolved HF cache dirs, Unsloth outputs/exports/root, registered scan folders, and well-known local-LLM dirs. Each is added only if it resolves to a real directory so the sandbox has no dead boundary.

    *media_roots* / *drive_roots* let the caller pass already-probed
    removable-media and Windows drive roots so they aren't scanned again (a
    disconnected mapped drive can make each probe slow); probed here when ``None``."""
    from hub.storage.scan_folders import list_scan_folders

    candidates: list[Path] = []

    def _add(p: Optional[Path | str]) -> None:
        if p is None:
            return
        try:
            p = Path(normalize_path(str(p))).expanduser()
            resolved = p.resolve()
        except (OSError, RuntimeError, ValueError):
            return
        if _safe_is_dir(resolved):
            candidates.append(resolved)

    _add(Path.home())
    if media_roots is None:
        media_roots = [*linux_run_media_mount_roots(), *macos_volume_roots()]
    if drive_roots is None:
        drive_roots = windows_drive_roots()
    for p in media_roots:
        _add(p)
    for p in drive_roots:
        _add(p)
    _add(_resolve_hf_cache_dir())
    try:
        from utils.hf_cache_settings import known_hf_cache_homes
        for cache_home in known_hf_cache_homes():
            _add(cache_home)
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    try:
        _add(hf_default_cache_dir())
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    try:
        _add(legacy_hf_cache_dir())
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    try:
        _add(studio_root())
        _add(outputs_root())
        _add(exports_root())
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: studio roots unavailable: %s", exc)
    try:
        for folder in list_scan_folders():
            p = folder.get("path")
            if p:
                _add(p)
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: could not load scan folders: %s", exc)
    try:
        for p in well_known_model_dirs():
            _add(p)
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: well-known dirs unavailable: %s", exc)

    seen: set[str] = set()
    deduped: list[Path] = []
    for p in candidates:
        key = os.path.normcase(os.path.realpath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _is_path_inside_allowlist(target: Path, allowed_roots: list[Path]) -> bool:
    """True if *target* equals or descends from any allowed root; uses ``os.path.realpath`` so symlinks cannot escape the sandbox.

    A Windows drive root (``D:\\``) authorizes its descendants, but a bare POSIX
    root (``/``) must NOT: a single ``/`` allowlist entry (e.g. a legacy scan
    folder) would otherwise authorize every absolute path, reaching ``/var``,
    ``/root``, etc. the denylist does not cover. Mirrors the legacy browser so
    both treat ``/`` identically.
    """
    try:
        target_real = os.path.normcase(os.path.realpath(str(target)))
    except OSError:
        return False
    for root in allowed_roots:
        try:
            root_real = os.path.normcase(os.path.realpath(str(root)))
        except OSError:
            continue
        if target_real == root_real:
            return True
        drive, tail = os.path.splitdrive(root_real)
        if os.path.dirname(root_real) == root_real and not drive:
            # Bare POSIX filesystem root: the equality above is the only match, so it must not authorize
            # arbitrary descendants.
            continue
        if drive.startswith(("\\\\", "//")) and not tail:
            # os.path.commonpath raises "can't mix absolute and relative" on a bare UNC share root, so authorize
            # its descendants with a boundary-safe prefix test.
            if target_real.startswith(root_real.rstrip("\\/") + os.sep):
                return True
            continue
        try:
            if os.path.commonpath([target_real, root_real]) == root_real:
                return True
        except ValueError:
            continue
    return False

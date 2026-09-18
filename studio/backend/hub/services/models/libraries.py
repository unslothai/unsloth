# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP response builders for model libraries: listing, registering, default
selection and moving a cached model between libraries without re-downloading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.storage.model_libraries import (
    add_model_library,
    list_model_libraries,
    remove_model_library,
    set_default_model_library,
)
from hub.utils.paths import path_is_same_or_child

logger = get_logger(__name__)


def _library_status(home: Path) -> dict:
    entry = {
        "hub_cache": str(home / "hub"),
        "xet_cache": str(home / "xet"),
        "available": False,
        "writable": False,
        "free_bytes": None,
        "total_bytes": None,
    }
    try:
        entry["available"] = home.is_dir()
    except OSError:
        return entry
    if not entry["available"]:
        return entry
    try:
        entry["writable"] = os.access(home, os.W_OK | os.X_OK)
    except OSError:
        pass
    try:
        usage = shutil.disk_usage(home)
        entry["free_bytes"] = int(usage.free)
        entry["total_bytes"] = int(usage.total)
    except OSError:
        pass
    return entry


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def list_libraries_response() -> dict:
    from utils.hf_cache_settings import get_hf_cache_paths

    active = get_hf_cache_paths()
    entries = [
        {
            "id": None,
            "path": str(active.cache_home),
            "label": None,
            "is_default": True,
            "created_at": None,
            **_library_status(active.cache_home),
        }
    ]
    seen = {_path_key(active.cache_home)}
    for row in list_model_libraries():
        path = Path(row["path"])
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "id": row["id"],
                "path": str(path),
                "label": row.get("label"),
                "is_default": False,
                "created_at": row.get("created_at"),
                **_library_status(path),
            }
        )
    return {"libraries": entries}


def _row_is_default(row: dict) -> bool:
    from utils.hf_cache_settings import get_hf_cache_paths
    return _path_key(row["path"]) == _path_key(get_hf_cache_paths().cache_home)


def add_library_response(path: str, label: Optional[str] = None) -> dict:
    try:
        row = add_model_library(path, label)
    except ValueError as exc:
        logger.warning("Model library rejected: %s (path=%s)", exc, path)
        raise HTTPException(status_code = 400, detail = str(exc))
    home = Path(row["path"])
    return {
        "id": row["id"],
        "path": row["path"],
        "label": row.get("label"),
        "is_default": _row_is_default(row),
        "created_at": row.get("created_at"),
        **_library_status(home),
    }


def remove_library_response(library_id: int) -> dict:
    try:
        removed = remove_model_library(library_id)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    if removed:
        from hub.utils.inventory_scan import invalidate_hf_cache_scans
        invalidate_hf_cache_scans()
    return {"ok": removed}


def set_default_library_response(library_id: int) -> dict:
    try:
        set_default_model_library(library_id)
    except ValueError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))
    from utils.hf_cache_settings import get_hf_cache_paths
    return {"ok": True, "path": str(get_hf_cache_paths().cache_home)}


def library_cache_paths(library_id: Optional[str]):
    """Resolve ``library_id`` ('default', empty or a row id) to cache paths for
    download targeting. Raises HTTPException 404 for an unknown id."""
    from utils.hf_cache_settings import HuggingFaceCachePaths, get_hf_cache_paths

    choice = (library_id or "").strip() or "default"
    if choice == "default":
        return get_hf_cache_paths()
    try:
        row_id = int(choice)
    except ValueError:
        raise HTTPException(status_code = 404, detail = "Library not found")
    for row in list_model_libraries():
        if row["id"] == row_id:
            home = Path(row["path"])
            return HuggingFaceCachePaths(home, home / "hub", home / "xet", "studio")
    raise HTTPException(status_code = 404, detail = "Library not found")


def _remove_failed_move(dest: Path) -> None:
    """Best-effort removal of a half-copied destination after a failed move;
    the source is still intact there."""
    try:
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
    except OSError as exc:
        logger.warning("Could not remove partial move destination %s: %s", dest, exc)


def _xet_skip_note(source_home: Path) -> Optional[str]:
    """Note for the response when the library's shared, content-addressed xet/
    store stays behind."""
    try:
        xet = source_home / "xet"
        has_data = xet.is_dir() and any(xet.iterdir())
    except OSError:
        return None
    if not has_data:
        return None
    return (
        "Moved the repo's hub/ directory only; the library's content-addressed "
        "xet/ store was left in place because it is shared by all its repos. "
        "The moved files are self-contained."
    )


def move_model_response(repo_id: str, variant: Optional[str], target_library_id: str) -> dict:
    if variant is not None:
        raise HTTPException(
            status_code = 400,
            detail = (
                "Model moves are whole-repo only; a variant-scoped move is not "
                "supported yet. Retry without a variant to move the whole cached repo."
            ),
        )

    from hub.utils import download_manifest
    from hub.utils.hf_cache_state import (
        iter_repo_cache_dirs,
        repo_cache_dir_name,
        same_existing_path,
    )
    from hub.utils.inventory_scan import invalidate_hf_cache_scans

    target_paths = library_cache_paths(target_library_id)
    target_hub = target_paths.hub_cache

    from hub.utils import download_registry

    if download_registry.get_models_registry().active_jobs(repo_id):
        raise HTTPException(
            status_code = 409,
            detail = "This model is still downloading; wait for it to finish or cancel it.",
        )
    try:
        from hub.services.models.downloads import _load_in_flight
        if _load_in_flight(repo_id):
            raise HTTPException(
                status_code = 409,
                detail = "This model is loading or staging right now; try when it is idle.",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    target_name = repo_cache_dir_name("model", repo_id)
    already_in_target = (target_hub / target_name).exists()
    source_dirs = [p for p in iter_repo_cache_dirs("model", repo_id) if p is not None]
    if not source_dirs and not already_in_target:
        raise HTTPException(status_code = 404, detail = "This model is not cached in any library.")
    if already_in_target:
        return {
            "ok": True,
            "repo_id": repo_id,
            "variant": None,
            "target_home": str(target_paths.cache_home),
            "already_in_library": True,
            "note": "A copy of this model already exists in the target library; nothing was moved.",
        }

    moved_any = False
    xet_note: Optional[str] = None
    for source in source_dirs:
        if same_existing_path(source.parent, target_hub):
            continue
        dest = target_hub / source.name
        if dest.exists():
            logger.warning(
                "Move of %s skipped a duplicate copy in %s: target %s already holds the repo.",
                repo_id,
                source.parent,
                dest,
            )
            continue
        try:
            shutil.move(str(source), str(dest))
        except OSError as exc:
            _remove_failed_move(dest)
            logger.error("Failed to move model %s to %s: %s", repo_id, target_hub, exc)
            raise HTTPException(
                status_code = 500,
                detail = f"Failed to move the model: {exc}",
            )
        download_manifest.purge_all_state_for_repo("model", repo_id, hub_cache = str(source.parent))
        if xet_note is None:
            xet_note = _xet_skip_note(source.parent.parent)
            if xet_note is not None:
                logger.info("%s (repo %s)", xet_note, repo_id)
        moved_any = True

    if moved_any:
        invalidate_hf_cache_scans()
    return {
        "ok": True,
        "repo_id": repo_id,
        "variant": None,
        "target_home": str(target_paths.cache_home),
        "already_in_library": not moved_any,
        "note": xet_note,
    }

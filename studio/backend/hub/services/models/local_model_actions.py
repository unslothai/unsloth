# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reveal and delete for on-device models outside the Hugging Face cache (LM Studio, custom
folders, ./models). Only a path the local inventory lists is accepted, so callers cannot probe or
remove arbitrary files."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.inventory import LocalModelInfo
from hub.services.models import account_access, deletion, local_inventory
from hub.utils.host_paths import scrub_paths

logger = get_logger(__name__)

REVEALABLE_SOURCES = frozenset({"models_dir", "lmstudio", "custom", "ollama", "hermes", "omlx"})
# Ollama stores content-addressed blobs shared between tags, and Hermes / oMLX own their folders.
DELETABLE_SOURCES = frozenset({"models_dir", "lmstudio", "custom"})


def _identity(raw: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.path.expanduser(str(raw))))


async def _listed_row(path: str, sources: frozenset[str]) -> LocalModelInfo:
    if not path or not path.strip():
        raise HTTPException(status_code = 400, detail = "path is required")
    wanted = _identity(path)
    listing = await local_inventory.list_local_models_response()
    for row in listing.models:
        if row.source in sources and row.path and _identity(row.path) == wanted:
            return row
    raise HTTPException(status_code = 404, detail = "Model not found on device")


def _protected_roots(custom_folders: list[dict]) -> set[str]:
    sources = local_inventory._local_inventory_sources()
    roots: list[Path | str] = [
        Path("./models"),
        sources.hf_cache_dir,
        sources.legacy_hf,
        sources.hf_default,
        *sources.lm_dirs,
        *sources.ollama_dirs,
        *sources.hermes_dirs,
        *sources.known_hf_caches,
        *sources.omlx_dirs,
        *(str(folder.get("path") or "") for folder in custom_folders),
    ]
    return {_identity(root) for root in roots if str(root)}


def _delete_blocking(target: Path, protected: set[str]) -> None:
    resolved = _identity(target)
    # A row whose path IS a scan root (a custom folder holding one model) would take the
    # folder itself, and an ancestor of a root takes everything under it.
    for root in protected:
        if resolved == root or root.startswith(resolved.rstrip(os.sep) + os.sep):
            raise HTTPException(
                status_code = 400,
                detail = "This model is a scanned folder itself; remove the folder instead",
            )
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    else:
        raise HTTPException(status_code = 404, detail = "Model not found on device")
    # LM Studio nests publisher/repo/: drop folders the delete left empty, up to the scan root.
    parent = Path(os.path.realpath(target.parent))
    while _identity(parent) not in protected and parent != parent.parent:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _load_state_blocks_delete(paths: tuple[str, ...]) -> str | None:
    for path in paths:
        if deletion._llama_cpp_blocks_delete(path, None) or (
            deletion._inference_backend_blocks_delete(path)
        ):
            return "Unload the model before deleting"
        detail = deletion._diffusion_blocks_delete(path) or deletion._video_blocks_delete(path)
        if detail:
            return detail
    return None


async def reveal_local_model_response(path: str) -> dict:
    account_access.require_installation_owner()
    from utils.paths.path_utils import reveal_in_file_manager

    row = await _listed_row(path, REVEALABLE_SOURCES)
    target = Path(row.path)
    try:
        await asyncio.to_thread(reveal_in_file_manager, target)
    except FileNotFoundError:
        raise HTTPException(status_code = 404, detail = "Model not found on device")
    except Exception as e:
        logger.error("Failed to reveal %s: %s", scrub_paths(str(target)), scrub_paths(e))
        raise HTTPException(status_code = 500, detail = "Failed to open file manager")
    return {"status": "ok"}


async def delete_local_model_response(path: str) -> dict:
    account_access.require_installation_owner()
    row = await _listed_row(path, DELETABLE_SOURCES)
    target = Path(row.path)
    # Loaded ids are whatever path the load was given: check the listed and the resolved spelling.
    spellings = tuple(dict.fromkeys((row.path, os.path.realpath(row.path))))
    try:
        blocks = await asyncio.to_thread(_load_state_blocks_delete, spellings)
    except Exception as e:
        logger.warning("Load-state verification failed for a local delete: %s", scrub_paths(e))
        raise HTTPException(status_code = 503, detail = deletion._LOAD_STATE_UNVERIFIABLE_DETAIL)
    if blocks:
        raise HTTPException(status_code = 400, detail = blocks)

    custom_folders = await local_inventory._load_custom_folders()
    protected = await asyncio.to_thread(_protected_roots, custom_folders)
    try:
        await asyncio.to_thread(_delete_blocking, target, protected)
    except HTTPException:
        raise
    except OSError as e:
        logger.error("Failed to delete %s: %s", scrub_paths(str(target)), scrub_paths(e))
        raise HTTPException(status_code = 500, detail = "Failed to delete the model files")
    finally:
        from core.inference.local_model_resolver import invalidate_index
        from hub.services.models import cache_inventory

        cache_inventory.invalidate_hf_cache_scans()
        invalidate_index()
    return {"status": "deleted"}

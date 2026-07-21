# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local model, HF cache, LM Studio, and Ollama inventory services.

Ollama logic lives in :mod:`hub.services.models.ollama`; this module
orchestrates all on-device sources and exposes the route handlers.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.inventory import LocalModelInfo, LocalModelListResponse, ModelFormat
from hub.storage.scan_folders import (
    add_scan_folder,
    list_scan_folders,
    remove_scan_folder,
)
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.paths import (
    hf_default_cache_dir,
    legacy_hf_cache_dir,
    lmstudio_model_dirs,
    normalize_path,
    ollama_model_dirs,
    outputs_root,
    path_is_same_or_child,
    studio_root,
)
from hub.services.models import common as model_common
from hub.services.models.ollama import scan_ollama_dir
from utils.hidden_models import is_hidden_model

logger = get_logger(__name__)
_MAX_MODELS_PER_CUSTOM_FOLDER = 200
_MAX_CUSTOM_FOLDER_ENTRIES = 2000
_MODEL_SIGNAL_PROBE_LIMIT = 200

# Local aliases keep the extracted code close to the original implementation.
_is_model_directory = model_common._is_model_directory
_local_inventory_id = model_common._local_inventory_id
_local_model_info = model_common._local_model_info
_capabilities_for_format = model_common._capabilities_for_format
_apply_format_aware_partial = model_common._apply_format_aware_partial
_classify_local_path = model_common._classify_local_path
_is_main_gguf_filename = model_common._is_main_gguf_filename
_is_transformers_bin_weight_file = model_common._is_transformers_bin_weight_file
_prefer_complete_larger = model_common._prefer_complete_larger
_gguf_variant_state_summary = model_common._gguf_variant_state_summary


def _is_immediate_model_weight_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return True
    if suffix == ".gguf":
        return _is_main_gguf_filename(path.name)
    if suffix == ".bin":
        return _is_transformers_bin_weight_file(path)
    return False


def _has_immediate_model_weight(
    path: Path, *, probe_limit: int = _MODEL_SIGNAL_PROBE_LIMIT
) -> bool:
    try:
        for index, entry in enumerate(path.iterdir(), start = 1):
            if index > probe_limit:
                break
            try:
                if entry.is_file() and _is_immediate_model_weight_file(entry):
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _has_immediate_model_signal(
    path: Path, *, probe_limit: int = _MODEL_SIGNAL_PROBE_LIMIT
) -> bool:
    try:
        if (path / "config.json").exists() or (path / "adapter_config.json").exists():
            return True
    except OSError:
        return False
    return _has_immediate_model_weight(path, probe_limit = probe_limit)


def _is_model_directory_for_scan(path: Path, *, entry_limit: int | None) -> bool:
    if entry_limit is None:
        return _is_model_directory(path)
    try:
        has_config = (path / "config.json").exists() or (path / "adapter_config.json").exists()
    except OSError:
        return False
    return has_config and _has_immediate_model_weight(path)


def _resolve_hf_cache_dir() -> Path:
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


def _scan_models_dir(
    models_dir: Path,
    *,
    limit: int | None = None,
    entry_limit: int | None = None,
) -> List[LocalModelInfo]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    _is_self_model = _is_model_directory_for_scan(
        models_dir,
        entry_limit = entry_limit,
    )

    if _is_self_model:
        try:
            updated_at = models_dir.stat().st_mtime
        except OSError:
            updated_at = None
        return _classify_local_path(
            models_dir,
            "models_dir",
            updated_at = updated_at,
        )

    found: List[LocalModelInfo] = []
    visited = 0
    try:
        children = models_dir.iterdir()
    except OSError:
        return found
    for child in children:
        if limit is not None and len(found) >= limit:
            break
        visited += 1
        if entry_limit is not None and visited > entry_limit:
            break
        try:
            is_dir = child.is_dir()
            is_gguf_file = not is_dir and child.suffix.lower() == ".gguf" and child.is_file()
            if not is_dir and not is_gguf_file:
                continue
            has_model_files = is_gguf_file or _has_immediate_model_signal(child)
        except OSError:
            # Skip individual children that are unreadable (permissions, broken
            # symlinks, etc.) rather than failing the entire scan.
            continue
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        rows = _classify_local_path(
            child,
            "models_dir",
            updated_at = updated_at,
        )
        if limit is not None:
            rows = rows[: max(0, limit - len(found))]
        found.extend(rows)

    return found


def _safe_is_dir(path: Path) -> bool:
    """``Path.is_dir()`` treating an unreadable path (``PermissionError`` /
    ``OSError`` on a restricted ``~/.cache/huggingface/hub``) as "not a
    directory", so the inventory skips that source instead of 500ing the Hub page.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def _hf_repo_dir_has_content(repo_dir: Path) -> bool:
    blobs_dir = repo_dir / "blobs"
    if not blobs_dir.is_dir():
        return False
    try:
        for entry in blobs_dir.iterdir():
            if entry.is_file() or entry.is_symlink():
                return True
    except OSError:
        return False
    return False


def _scan_hf_cache(
    cache_dir: Path,
    *,
    entry_limit: int | None = None,
    active_cache: bool = True,
) -> List[LocalModelInfo]:
    if not _safe_is_dir(cache_dir):
        return []

    discovered: List[tuple[Path, str, Optional[float]]] = []
    visited = 0
    try:
        entries = cache_dir.iterdir()
    except OSError:
        return []
    for repo_dir in entries:
        visited += 1
        if entry_limit is not None and visited > entry_limit:
            break
        if not repo_dir.name.startswith("models--"):
            continue
        if not repo_dir.is_dir():
            continue
        if not _hf_repo_dir_has_content(repo_dir):
            continue
        repo_name = repo_dir.name[len("models--") :]
        if not repo_name:
            continue
        model_id = repo_name.replace("--", "/")
        try:
            updated_at = repo_dir.stat().st_mtime
        except OSError:
            updated_at = None
        discovered.append((repo_dir, model_id, updated_at))

    found: list[LocalModelInfo] = []
    for repo_dir, model_id, updated_at in discovered:
        snapshot_partial = hf_cache_scan.is_snapshot_partial(
            "model",
            model_id,
            repo_dir,
        )
        gguf_partial = hf_cache_scan.is_gguf_repo_partial(model_id, repo_dir)
        has_gguf_variant_state, gguf_variant_state_size = _gguf_variant_state_summary(model_id)
        snapshot_partial_transport = (
            hf_cache_scan.partial_transport_for(
                "model",
                model_id,
                repo_cache_dir = repo_dir,
            )
            if snapshot_partial
            else None
        )
        resolved = hf_cache_scan.resolve_hf_cache_realpath(repo_dir)
        scan_path = Path(resolved) if resolved else repo_dir
        load_path = repo_dir if active_cache else scan_path
        # partial=False here; _apply_format_aware_partial below rewrites per-row
        # so a hybrid repo's gguf row doesn't taint its safetensors row.
        rows = _classify_local_path(
            scan_path,
            "hf_cache",
            load_path = load_path,
            display_name = model_id.split("/")[-1],
            model_id = model_id,
            updated_at = updated_at,
            partial = False,
            active_cache = active_cache,
        )
        if not rows:
            if has_gguf_variant_state and gguf_partial:
                rows = [
                    _local_model_info(
                        scan_path = repo_dir,
                        load_path = load_path,
                        source = "hf_cache",
                        model_format = "gguf",
                        display_name = model_id.split("/")[-1],
                        model_id = model_id,
                        updated_at = updated_at,
                        partial = True,
                        requires_variant = True,
                        size_bytes = gguf_variant_state_size,
                        active_cache = active_cache,
                    )
                ]
            else:
                # Fallback row's model_format is "unknown"; either signal
                # applies because we can't dispatch to a specific predicate.
                rows = [
                    _local_model_info(
                        scan_path = repo_dir,
                        load_path = load_path,
                        source = "hf_cache",
                        model_format = "unknown",
                        display_name = model_id.split("/")[-1],
                        model_id = model_id,
                        updated_at = updated_at,
                        partial = snapshot_partial or gguf_partial,
                        active_cache = active_cache,
                    )
                ]
        elif (
            has_gguf_variant_state
            and gguf_partial
            and not any(row.model_format == "gguf" for row in rows)
        ):
            rows.append(
                _local_model_info(
                    scan_path = repo_dir,
                    load_path = load_path,
                    source = "hf_cache",
                    model_format = "gguf",
                    display_name = model_id.split("/")[-1],
                    model_id = model_id,
                    updated_at = updated_at,
                    partial = True,
                    requires_variant = True,
                    size_bytes = gguf_variant_state_size,
                    active_cache = active_cache,
                )
            )
        rows = _apply_format_aware_partial(
            rows,
            snapshot_partial = snapshot_partial,
            gguf_partial = gguf_partial,
            snapshot_partial_transport = snapshot_partial_transport,
        )
        found.extend(rows)
    return found


def _scan_lmstudio_dir(lm_dir: Path, *, entry_limit: int | None = None) -> List[LocalModelInfo]:
    """Scan an LM Studio models dir (``publisher/model-name`` folders of GGUFs, or top-level standalone GGUFs)."""
    if not lm_dir.exists() or not lm_dir.is_dir():
        return []

    # If the dir is itself a model dir (config + weights), it's not an LM Studio
    # publisher structure -- return it as a single entry rather than descend.
    if _is_model_directory(lm_dir):
        try:
            updated_at = lm_dir.stat().st_mtime
        except OSError:
            updated_at = None
        return _classify_local_path(
            lm_dir,
            "lmstudio",
            updated_at = updated_at,
        )

    found: List[LocalModelInfo] = []
    visited = 0
    exhausted = False

    def _consume_visit() -> bool:
        nonlocal visited
        visited += 1
        return entry_limit is not None and visited > entry_limit

    try:
        children = lm_dir.iterdir()
    except OSError:
        return found
    for child in children:
        if _consume_visit():
            break
        try:
            if not child.is_dir():
                if child.suffix == ".gguf" and child.is_file():
                    try:
                        updated_at = child.stat().st_mtime
                    except OSError:
                        updated_at = None
                    found.extend(
                        _classify_local_path(
                            child,
                            "lmstudio",
                            updated_at = updated_at,
                        )
                    )
                continue

            # Child is itself a model dir: surface it directly, not as a publisher.
            if _is_model_directory(child):
                try:
                    updated_at = child.stat().st_mtime
                except OSError:
                    updated_at = None
                found.extend(
                    _classify_local_path(
                        child,
                        "lmstudio",
                        updated_at = updated_at,
                    )
                )
                continue

            # child is a publisher directory -- scan its sub-directories
            for model_dir in child.iterdir():
                if _consume_visit():
                    exhausted = True
                    break
                try:
                    if model_dir.is_dir():
                        has_model = _has_immediate_model_signal(model_dir)
                        if not has_model:
                            continue
                        model_id = f"{child.name}/{model_dir.name}"
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        found.extend(
                            _classify_local_path(
                                model_dir,
                                "lmstudio",
                                display_name = model_dir.name,
                                model_id = model_id,
                                updated_at = updated_at,
                            )
                        )
                    elif model_dir.suffix == ".gguf" and model_dir.is_file():
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        found.extend(
                            _classify_local_path(
                                model_dir,
                                "lmstudio",
                                model_id = f"{child.name}/{model_dir.stem}",
                                updated_at = updated_at,
                            )
                        )
                except OSError:
                    continue
            if exhausted:
                break
        except OSError:
            continue
    return found


def _resolve_allowed_models_dir(models_dir: str, allowed_roots: list[Path]) -> Path:
    """Resolve a requested model scan directory without widening subpaths."""
    if not models_dir or not models_dir.strip():
        raise ValueError("Directory not allowed")

    requested = Path(os.path.realpath(os.path.expanduser(normalize_path(models_dir.strip()))))
    if any(path_is_same_or_child(requested, root) for root in allowed_roots):
        return requested

    raise ValueError("Directory not allowed")


def _coerce_scan_folder_path(raw_path: str) -> str:
    """Normalize a scan registration target; the registry stores directories, so a pasted weight-file path is reduced to its parent folder."""
    if not raw_path or not raw_path.strip():
        raise ValueError("Path cannot be empty")
    raw = raw_path.strip()
    if "\x00" in raw:
        raise ValueError("Path cannot contain null bytes")

    def normalize(value: str) -> Path:
        return Path(os.path.realpath(os.path.expanduser(normalize_path(value))))

    try:
        normalized = normalize(raw)
    except (OSError, ValueError) as e:
        raise ValueError(f"Path is not readable: {e}") from e
    try:
        exists = normalized.exists()
        is_dir = normalized.is_dir()
        is_file = normalized.is_file()
    except (OSError, ValueError) as e:
        raise ValueError(f"Path is not readable: {e}") from e

    if not exists and "\\" in raw:
        try:
            slash_normalized = normalize(raw.replace("\\", "/"))
            slash_exists = slash_normalized.exists()
        except (OSError, ValueError) as e:
            raise ValueError(f"Path is not readable: {e}") from e
        if slash_exists:
            normalized = slash_normalized
            try:
                is_dir = normalized.is_dir()
                is_file = normalized.is_file()
            except (OSError, ValueError) as e:
                raise ValueError(f"Path is not readable: {e}") from e
            exists = True

    if not exists:
        return str(normalized)
    if is_dir:
        return str(normalized)
    if is_file:
        suffix = normalized.suffix.lower()
        if suffix not in {".gguf", ".safetensors", ".bin"}:
            raise ValueError("Path must be a folder or model weight file")
        return str(normalized.parent)
    return str(normalized)


async def _scan_source(label: str, scanner, path: Path) -> List[LocalModelInfo]:
    try:
        return await asyncio.to_thread(scanner, path)
    except Exception as e:
        logger.warning("Skipping %s scan for %s: %s", label, path, e)
        return []


async def _collect_models_from_default_sources(
    models_root: Path,
    hf_cache_dir: Path,
    legacy_hf: Path,
    hf_default: Path,
    lm_dirs: list[Path],
    ollama_dirs: list[Path],
) -> List[LocalModelInfo]:
    local_models = await _scan_source("models directory", _scan_models_dir, models_root)
    local_models += await _scan_source("HF cache", _scan_hf_cache, hf_cache_dir)

    if _safe_is_dir(legacy_hf) and legacy_hf.resolve() != hf_cache_dir.resolve():
        local_models += await _scan_source(
            "legacy HF cache",
            lambda path: _scan_hf_cache(path, active_cache = False),
            legacy_hf,
        )

    if (
        _safe_is_dir(hf_default)
        and hf_default.resolve() != hf_cache_dir.resolve()
        and hf_default.resolve() != legacy_hf.resolve()
    ):
        local_models += await _scan_source(
            "default HF cache",
            lambda path: _scan_hf_cache(path, active_cache = False),
            hf_default,
        )

    from utils.hf_cache_settings import known_hf_hub_caches

    seen_hf = {
        os.path.normcase(str(path.resolve(strict = False)))
        for path in (hf_cache_dir, legacy_hf, hf_default)
    }
    for previous_cache in known_hf_hub_caches():
        key = os.path.normcase(str(previous_cache.resolve(strict = False)))
        if key in seen_hf:
            continue
        seen_hf.add(key)
        local_models += await _scan_source(
            "previous HF cache",
            lambda path: _scan_hf_cache(path, active_cache = False),
            previous_cache,
        )

    for lm_dir in lm_dirs:
        local_models += await _scan_source("LM Studio", _scan_lmstudio_dir, lm_dir)

    for ollama_dir in ollama_dirs:
        local_models += await _scan_source("Ollama", scan_ollama_dir, ollama_dir)

    return local_models


def _scan_custom_folder(folder_path: Path) -> List[LocalModelInfo]:
    supported_formats: set[ModelFormat] = {"gguf", "safetensors", "adapter"}
    generic = [
        m
        for m in (
            _scan_models_dir(
                folder_path,
                limit = _MAX_MODELS_PER_CUSTOM_FOLDER,
                entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
            )
            + _scan_hf_cache(
                folder_path,
                entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
                active_cache = False,
            )
            + _scan_lmstudio_dir(folder_path, entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES)
        )
        if m.model_format in supported_formats
        if not any(p in (".studio_links", "ollama_links") for p in Path(m.path).parts)
    ]
    return generic[:_MAX_MODELS_PER_CUSTOM_FOLDER]


def _promote_to_custom_source(model: LocalModelInfo) -> LocalModelInfo:
    if model.source == "hf_cache":
        return model
    return model.model_copy(
        update = {
            "source": "custom",
            "model_id": None,
            "inventory_id": _local_inventory_id(
                "custom",
                model.model_format,
                model.path,
                model.format_variant,
            ),
            "capabilities": _capabilities_for_format(
                model.model_format,
                "custom",
                partial = model.partial,
                requires_variant = model.capabilities.requires_variant,
            ),
        }
    )


async def _collect_models_from_custom_folders() -> List[LocalModelInfo]:
    try:
        custom_folders = await asyncio.to_thread(list_scan_folders)
    except Exception as e:
        logger.warning("Could not load custom scan folders: %s", e)
        return []

    local_models: List[LocalModelInfo] = []
    for folder in custom_folders:
        folder_path = Path(normalize_path(folder["path"])).expanduser()
        try:
            custom_models = await asyncio.to_thread(_scan_custom_folder, folder_path)
        except Exception as e:
            logger.warning("Skipping unreadable scan folder %s: %s", folder_path, e)
            continue
        local_models.extend(_promote_to_custom_source(m) for m in custom_models)
    return local_models


def _dedupe_local_models(local_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    deduped: dict[str, LocalModelInfo] = {}
    for model in local_models:
        if model.source == "hf_cache" and model.model_id:
            key = "\x00".join(
                (
                    "hf_cache",
                    model.model_id.strip().lower(),
                    model.model_format,
                    model.format_variant or "",
                )
            )
        else:
            row_key = model.inventory_id or model.id
            key = f"{row_key}\x00custom" if model.source == "custom" else row_key
        existing = deduped.get(key)
        if (
            existing is None
            or (model.active_cache is True and existing.active_cache is not True)
            or (
                model.active_cache == existing.active_cache
                and _prefer_complete_larger(
                    model.partial,
                    model.size_bytes,
                    existing.partial,
                    existing.size_bytes,
                )
            )
        ):
            deduped[key] = model
    return sorted(
        deduped.values(),
        key = lambda item: (item.updated_at or 0),
        reverse = True,
    )


def _filter_hidden_models(local_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    """Remove infrastructure-only models from the shared local inventory."""
    visible: list[LocalModelInfo] = []
    for model in local_models:
        resolved_cache_path = (
            hf_cache_scan.resolve_hf_cache_realpath(Path(model.path))
            if model.source == "hf_cache"
            else None
        )
        if not is_hidden_model(model.id, model.model_id, model.path, resolved_cache_path):
            visible.append(model)
    return visible


async def list_local_models_response(models_dir: str = "./models") -> LocalModelListResponse:
    """List local model candidates from every supported on-device source."""
    hf_cache_dir = _resolve_hf_cache_dir()
    legacy_hf = legacy_hf_cache_dir()
    hf_default = hf_default_cache_dir()
    lm_dirs = lmstudio_model_dirs()
    ollama_dirs = ollama_model_dirs()

    allowed_roots: list[Path] = [Path("./models").resolve(), hf_cache_dir]
    if _safe_is_dir(legacy_hf):
        allowed_roots.append(legacy_hf)
    if _safe_is_dir(hf_default):
        allowed_roots.append(hf_default)
    allowed_roots.extend([studio_root(), outputs_root()])

    try:
        models_root = _resolve_allowed_models_dir(models_dir, allowed_roots)
    except ValueError:
        raise HTTPException(status_code = 403, detail = "Directory not allowed")

    try:
        local_models = await _collect_models_from_default_sources(
            models_root,
            hf_cache_dir,
            legacy_hf,
            hf_default,
            lm_dirs,
            ollama_dirs,
        )
        local_models += await _collect_models_from_custom_folders()
        models = _dedupe_local_models(_filter_hidden_models(local_models))

        return LocalModelListResponse(
            models_dir = str(models_root),
            hf_cache_dir = str(hf_cache_dir),
            lmstudio_dirs = [str(d) for d in lm_dirs],
            ollama_dirs = [str(d) for d in ollama_dirs],
            models = models,
        )
    except Exception as e:
        logger.error(f"Error listing local models: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list local models: {str(e)}",
        )


def get_models_folder_response() -> dict:
    """Return the directory where downloaded models are stored.

    This is the active HF hub cache (honors ``HF_HOME`` / ``HF_HUB_CACHE``);
    the desktop app reveals it in the OS file manager.
    """
    path = _resolve_hf_cache_dir()
    # Create it if missing so "Open folder" works before the first download:
    # HF builds the cache lazily, and studio only pre-creates the *default*
    # dir, not a user's explicit HF_HOME / HF_HUB_CACHE.
    try:
        path.mkdir(parents = True, exist_ok = True)
    except OSError as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to create models folder: {path}: {e}",
        ) from e
    if not path.is_dir():
        raise HTTPException(
            status_code = 500,
            detail = f"Models folder path is not a directory: {path}",
        )
    return {"path": str(path)}


def get_scan_folders_response() -> dict:
    return {"folders": list_scan_folders()}


def add_scan_folder_response(path: str) -> dict:
    try:
        folder = add_scan_folder(_coerce_scan_folder_path(path))
    except ValueError as e:
        logger.warning("Scan folder rejected: %s (path=%s)", e, path)
        raise HTTPException(status_code = 400, detail = str(e))
    logger.info("Scan folder added: %s", folder.get("path"))
    return folder


def remove_scan_folder_response(folder_id: int) -> dict:
    remove_scan_folder(folder_id)
    logger.info("Scan folder removed: id=%s", folder_id)
    return {"ok": True}

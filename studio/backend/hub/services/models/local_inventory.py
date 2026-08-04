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
from hub.utils import download_manifest, inventory_scan as hf_cache_scan
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
_LocalInventoryKey = tuple[
    str, int, Optional[str], tuple[str, ...], tuple[tuple[str, str, str], ...]
]
_local_inventory_flights: dict[_LocalInventoryKey, asyncio.Task[LocalModelListResponse]] = {}

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


def _discover_hf_cache(
    cache_dir: Path, *, entry_limit: int | None = None
) -> list[tuple[Path, str, Optional[float]]]:
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
    return discovered


def _scan_hf_cache(
    cache_dir: Path,
    *,
    entry_limit: int | None = None,
    active_cache: bool = True,
    discovered: Optional[list[tuple[Path, str, Optional[float]]]] = None,
    variant_states: Optional[download_manifest.VariantStateIndex] = None,
    active_hub_cache: Optional[Path] = None,
) -> List[LocalModelInfo]:
    if discovered is None:
        discovered = _discover_hf_cache(cache_dir, entry_limit = entry_limit)
    if not discovered:
        return []
    if variant_states is None:
        variant_states = download_manifest.build_variant_state_index(
            [("model", model_id, cache_dir) for _repo, model_id, _updated in discovered],
            active_hub_cache = active_hub_cache
            or (cache_dir if active_cache else _resolve_hf_cache_dir()),
        )

    found: list[LocalModelInfo] = []
    for repo_dir, model_id, updated_at in discovered:
        variant_state = variant_states.for_repo("model", model_id, hub_cache = cache_dir)
        snapshot_partial = hf_cache_scan.is_snapshot_partial(
            "model",
            model_id,
            repo_dir,
            variant_state = variant_state,
        )
        gguf_partial = hf_cache_scan.is_gguf_repo_partial(
            model_id,
            repo_dir,
            variant_state = variant_state,
        )
        has_gguf_variant_state, gguf_variant_state_size = _gguf_variant_state_summary(
            model_id,
            hub_cache = cache_dir,
            variant_state = variant_state,
        )
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
    hf_sources = [("HF cache", hf_cache_dir, True)]

    if _safe_is_dir(legacy_hf) and legacy_hf.resolve() != hf_cache_dir.resolve():
        hf_sources.append(("legacy HF cache", legacy_hf, False))

    if (
        _safe_is_dir(hf_default)
        and hf_default.resolve() != hf_cache_dir.resolve()
        and hf_default.resolve() != legacy_hf.resolve()
    ):
        hf_sources.append(("default HF cache", hf_default, False))

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
        hf_sources.append(("previous HF cache", previous_cache, False))

    discovered_sources = []
    state_repositories = []
    for label, cache_dir, active_cache in hf_sources:
        discovered = await _scan_source(label, _discover_hf_cache, cache_dir)
        discovered_sources.append((label, cache_dir, active_cache, discovered))
        state_repositories.extend(
            ("model", model_id, cache_dir) for _repo, model_id, _updated in discovered
        )
    try:
        variant_states = await asyncio.to_thread(
            download_manifest.build_variant_state_index,
            state_repositories,
            active_hub_cache = hf_cache_dir,
        )
    except Exception as e:
        logger.warning("Could not build shared Hub-state index: %s", e)
        variant_states = None
    for label, cache_dir, active_cache, discovered in discovered_sources:
        local_models += await _scan_source(
            label,
            lambda path, rows = discovered, active = active_cache: _scan_hf_cache(
                path,
                active_cache = active,
                discovered = rows,
                variant_states = variant_states,
                active_hub_cache = hf_cache_dir,
            ),
            cache_dir,
        )

    for lm_dir in lm_dirs:
        local_models += await _scan_source("LM Studio", _scan_lmstudio_dir, lm_dir)

    for ollama_dir in ollama_dirs:
        local_models += await _scan_source("Ollama", scan_ollama_dir, ollama_dir)

    return local_models


def _scan_custom_folder(folder_path: Path) -> List[LocalModelInfo]:
    from utils.models.model_config import detect_gguf_model

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
    selectable = []
    for model in generic:
        if model.model_format != "gguf" or model.partial:
            selectable.append(model)
            continue
        path = Path(model.path)
        if path.is_dir():
            if any(
                detect_gguf_model(str(file), model_root = str(folder_path)) is not None
                for file in path.glob("*")
                if not _safe_is_dir(file) and file.suffix.lower() == ".gguf"
            ):
                selectable.append(model)
        elif detect_gguf_model(model.path, model_root = str(folder_path)) is not None:
            selectable.append(model)
    return selectable[:_MAX_MODELS_PER_CUSTOM_FOLDER]


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


async def _load_custom_folders() -> list[dict]:
    try:
        return await asyncio.to_thread(list_scan_folders)
    except Exception as e:
        logger.warning("Could not load custom scan folders: %s", e)
        return []


async def _collect_models_from_custom_folders(
    custom_folders: Optional[list[dict]] = None,
) -> List[LocalModelInfo]:
    if custom_folders is None:
        custom_folders = await _load_custom_folders()
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
        prefer_candidate = existing is None
        if existing is not None:
            if model.partial != existing.partial:
                prefer_candidate = not model.partial
            elif (model.active_cache is True) != (existing.active_cache is True):
                prefer_candidate = model.active_cache is True
            else:
                prefer_candidate = _prefer_complete_larger(
                    model.partial,
                    model.size_bytes,
                    existing.partial,
                    existing.size_bytes,
                )
        if prefer_candidate:
            deduped[key] = model
    return sorted(
        deduped.values(),
        key = lambda item: item.updated_at or 0,
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


async def _scan_local_models_response(
    models_dir: str, custom_folders: list[dict]
) -> LocalModelListResponse:
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
        local_models += await _collect_models_from_custom_folders(custom_folders)
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


def _clear_local_inventory_flight(
    key: _LocalInventoryKey, task: asyncio.Task[LocalModelListResponse]
) -> None:
    if _local_inventory_flights.get(key) is task:
        _local_inventory_flights.pop(key, None)
    if not task.cancelled():
        task.exception()


async def list_local_models_response(models_dir: str = "./models") -> LocalModelListResponse:
    """Coalesce overlapping local inventory requests for the same models root."""
    custom_folders = await _load_custom_folders()

    async def scan_and_classify() -> LocalModelListResponse:
        response = await _scan_local_models_response(models_dir, custom_folders)
        # These rows feed the same pickers as /api/models/local. Classify inside the shared
        # worker so retrying waiters do not repeat GGUF metadata reads.
        try:
            from routes.models import _local_model_task
            models = [
                model.model_copy(update = {"task": _local_model_task(model)})
                for model in response.models
            ]
            return response.model_copy(update = {"models": models})
        except Exception as e:  # noqa: BLE001 -- classification never breaks the listing
            logger.warning("Could not classify local model tasks: %s", e)
            return response

    custom_generation = tuple(
        (
            str(folder.get("id", "")),
            str(folder.get("path", "")),
            str(folder.get("created_at", "")),
        )
        for folder in custom_folders
    )
    mutations = download_manifest.variant_state_mutation_snapshot()
    while mutations.in_progress:
        await asyncio.sleep(0.01)
        mutations = download_manifest.variant_state_mutation_snapshot()
    key = (
        os.path.normcase(models_dir),
        hf_cache_scan.hf_cache_scans_epoch(),
        download_manifest.variant_state_generation(),
        mutations.markers,
        custom_generation,
    )
    flight = _local_inventory_flights.get(key)
    if flight is None or flight.done():
        flight = asyncio.create_task(scan_and_classify())
        _local_inventory_flights[key] = flight
        flight.add_done_callback(
            lambda task, flight_key = key: _clear_local_inventory_flight(flight_key, task)
        )
    return await asyncio.shield(flight)


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

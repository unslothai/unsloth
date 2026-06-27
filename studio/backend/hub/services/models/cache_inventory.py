# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model inventory."""

from __future__ import annotations

import json
import asyncio
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.inventory import ModelFormat
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils import download_registry
from hub.utils.snapshot_filters import (
    blob_hashes_for_siblings,
    snapshot_download_blob_hashes,
    snapshot_download_siblings,
    snapshot_download_size,
)
from hub.services.models.common import (
    _capabilities_for_format,
    _classify_non_gguf_model_format,
    _gguf_variant_state_summary,
    _is_adapter_weight_name,
    _is_checkpoint_weight_name,
    _is_gguf_filename,
    _is_main_gguf_filename,
    _is_transformers_safetensors_weight_name,
    _local_inventory_id,
    _prefer_complete_larger,
    _runtime_for_format,
)
from hub.utils.gguf import extract_quant_label
from hub.utils.gguf_plan import build_gguf_variant_plans

logger = get_logger(__name__)


def _is_non_gguf_weight_filename(name: str) -> bool:
    lower = str(name).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if not lower or _is_gguf_filename(lower):
        return False
    return (
        _is_adapter_weight_name(lower)
        or lower.endswith(".safetensors")
        or _is_checkpoint_weight_name(lower)
    )


def _non_gguf_weight_siblings(siblings) -> list:
    return [
        sibling
        for sibling in snapshot_download_siblings(siblings)
        if _is_non_gguf_weight_filename(getattr(sibling, "rfilename", ""))
    ]


_repo_size_cache: "OrderedDict[tuple[str, str, str], tuple[int, frozenset[str], float]]" = (
    OrderedDict()
)
_repo_size_neg_cache: "OrderedDict[tuple[str, str, str], float]" = OrderedDict()
_REPO_SIZE_CACHE_MAX = 256
_REPO_SIZE_POS_TTL = 60.0
_REPO_SIZE_NEG_TTL = 60.0
_MODEL_METADATA_TIMEOUT_SECONDS = 5.0
_repo_size_cache_lock = threading.Lock()

# Short in-process TTL cache for per-GGUF remote update checks. The key includes
# repo, variant, and a token fingerprint; the value's local-blob fingerprint
# invalidates a cached result when local MAIN GGUF blobs change inside the TTL.
_GGUF_UPDATE_CHECK_TTL = 60.0
_GGUF_UPDATE_CHECK_MAX = 256
_gguf_update_check_cache: "dict[tuple[str, str, str], tuple[float, bool, frozenset]]" = {}
_gguf_update_check_lock = threading.Lock()


def get_repo_snapshot_metadata_cached(
    repo_id: str,
    hf_token: Optional[str] = None,
    *,
    weight_only: bool = False,
) -> tuple[int, frozenset[str]]:
    token_fp = hf_cache_scan.token_fingerprint(hf_token)
    cache_key = (repo_id, token_fp, "weights" if weight_only else "snapshot")
    with _repo_size_cache_lock:
        cached = _repo_size_cache.get(cache_key)
        if cached is not None:
            total, blob_hashes, ts = cached
            if (time.monotonic() - ts) < _REPO_SIZE_POS_TTL:
                _repo_size_cache.move_to_end(cache_key)
                return total, blob_hashes
            del _repo_size_cache[cache_key]
        neg_ts = _repo_size_neg_cache.get(cache_key)
        if neg_ts is not None and (time.monotonic() - neg_ts) < _REPO_SIZE_NEG_TTL:
            return 0, frozenset()
    try:
        from huggingface_hub import HfApi
        info = HfApi(token = hf_token).model_info(
            repo_id,
            files_metadata = True,
            timeout = _MODEL_METADATA_TIMEOUT_SECONDS,
        )
        if weight_only:
            siblings = _non_gguf_weight_siblings(info.siblings)
            total = sum(int(getattr(sibling, "size", 0) or 0) for sibling in siblings)
            blob_hashes = blob_hashes_for_siblings(siblings)
        else:
            total = snapshot_download_size(info.siblings)
            blob_hashes = snapshot_download_blob_hashes(info.siblings)
    except Exception as e:
        logger.warning(
            "Failed to get repo size for %s: %s",
            repo_id,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        with _repo_size_cache_lock:
            _repo_size_neg_cache[cache_key] = time.monotonic()
            _repo_size_neg_cache.move_to_end(cache_key)
            while len(_repo_size_neg_cache) > _REPO_SIZE_CACHE_MAX:
                _repo_size_neg_cache.popitem(last = False)
        return 0, frozenset()
    with _repo_size_cache_lock:
        _repo_size_cache[cache_key] = (total, blob_hashes, time.monotonic())
        _repo_size_cache.move_to_end(cache_key)
        _repo_size_neg_cache.pop(cache_key, None)
        while len(_repo_size_cache) > _REPO_SIZE_CACHE_MAX:
            _repo_size_cache.popitem(last = False)
    return total, blob_hashes


def all_hf_cache_scans():
    return hf_cache_scan.all_hf_cache_scans()


def _repo_gguf_size_bytes(repo_info) -> int:
    """Sum primary GGUF blob sizes across revisions, deduped by blob path (HF hardlinks shared blobs); mmproj is excluded so a vision-adapter-only repo isn't classed as GGUF."""
    unique_blobs: dict[str, int] = {}
    for revision in repo_info.revisions:
        rev_id = getattr(revision, "commit_hash", None) or str(id(revision))
        for f in revision.files:
            if _is_main_gguf_filename(f.file_name):
                blob_path = getattr(f, "blob_path", None)
                size = f.size_on_disk or 0
                if blob_path:
                    unique_blobs[str(blob_path)] = size
                else:
                    unique_blobs[f"{rev_id}:{f.file_name}"] = size
    return sum(unique_blobs.values())


def _repo_has_gguf_files(repo_info) -> bool:
    return _repo_gguf_size_bytes(repo_info) > 0


def _cached_repo_file_name(file_obj) -> str:
    file_path = getattr(file_obj, "file_path", None)
    if file_path:
        try:
            path = Path(file_path)
            parts = path.parts
            snapshots_idx = max(i for i, part in enumerate(parts) if part == "snapshots")
            if len(parts) > snapshots_idx + 2:
                return Path(*parts[snapshots_idx + 2 :]).as_posix()
        except Exception:
            pass
    return str(getattr(file_obj, "file_name", "")).replace("\\", "/")


def _repo_gguf_blob_map(repo_info) -> dict[str, set[str]]:
    """Map each cached MAIN gguf file's repo-relative name to the SET of its local
    blob hashes across all cached revisions.

    HF names each local cache blob FILE by the file's etag (lfs.sha256 else
    blob_id), so a local file's blob hash == ``Path(blob_path).name``. An updated
    repo keeps BOTH the old and new revision snapshots until HF garbage-collects
    them, so the same file resolves to several blobs; collecting them ALL (not
    just the first one seen, since ``repo_info.revisions`` is a frozenset and
    yields them in arbitrary order) lets the remote-vs-local diff treat the file
    as current when the remote (``main``) blob is present in any cached revision.
    Mirrors the ``cached_blob_ids`` membership test in routes/models.py."""
    blob_map: dict[str, set[str]] = {}
    for revision in repo_info.revisions:
        for f in revision.files:
            if not _is_main_gguf_filename(f.file_name):
                continue
            blob_path = getattr(f, "blob_path", None)
            if not blob_path:
                continue
            name = _cached_repo_file_name(f)
            blob_map.setdefault(name, set()).add(Path(blob_path).name)
    return blob_map


def _filter_gguf_blob_map_by_variant(
    local_blobs: dict[str, set[str]], gguf_variant: Optional[str]
) -> dict[str, set[str]]:
    variant = (gguf_variant or "").strip().lower()
    if not variant:
        return local_blobs
    return {
        name: blobs
        for name, blobs in local_blobs.items()
        if extract_quant_label(name).lower() == variant
    }


def _prefer_cache_row(candidate: dict, existing: Optional[dict]) -> bool:
    if existing is None:
        return True
    return _prefer_complete_larger(
        bool(candidate.get("partial")),
        int(candidate.get("size_bytes") or 0),
        bool(existing.get("partial")),
        int(existing.get("size_bytes") or 0),
    )


def _cache_inventory_fields(
    repo_id: str,
    model_format: ModelFormat,
    *,
    partial: bool = False,
    requires_variant: bool = False,
) -> dict:
    return {
        "inventory_id": _local_inventory_id("cache", model_format, repo_id),
        "load_id": repo_id,
        "model_format": model_format,
        "runtime": _runtime_for_format(model_format),
        "format_variant": None,
        "capabilities": _capabilities_for_format(
            model_format,
            "hf_cache",
            partial = partial,
            requires_variant = requires_variant,
        ).model_dump(),
    }


def invalidate_hf_cache_scans() -> None:
    hf_cache_scan.invalidate_hf_cache_scans()


def _scan_cached_gguf() -> list[dict]:
    """Synchronous HF-cache disk walk for GGUF repos; runs in a worker thread."""
    cache_scans = all_hf_cache_scans()

    seen_lower: dict[str, dict] = {}
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            try:
                if str(repo_info.repo_type) != "model":
                    continue
                repo_id = repo_info.repo_id
                total_size = _repo_gguf_size_bytes(repo_info)
                has_variant_state, variant_state_size = _gguf_variant_state_summary(repo_id)
                if total_size == 0 and not has_variant_state:
                    continue
                partial = hf_cache_scan.is_gguf_repo_partial(
                    repo_id,
                    Path(repo_info.repo_path),
                )
                if total_size == 0 and not partial:
                    continue
                key = repo_id.lower()
                existing = seen_lower.get(key)
                row = {
                    "repo_id": repo_id,
                    "size_bytes": max(total_size, variant_state_size),
                    "cache_path": str(repo_info.repo_path),
                    "partial": partial,
                    # GGUF row-level transport is ambiguous (variants may differ);
                    # per-variant detail lives on GgufVariantDetail.
                    "partial_transport": None,
                }
                row.update(
                    _cache_inventory_fields(
                        repo_id,
                        "gguf",
                        partial = bool(row["partial"]),
                        requires_variant = True,
                    )
                )
                if _prefer_cache_row(row, existing):
                    seen_lower[key] = row
            except Exception as e:
                repo_label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning(f"Skipping cached GGUF repo {repo_label}: {e}")
                continue
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _gguf_variant_update_available_from_remote(
    local_blobs: dict[str, set[str]], remote_paths: set[str], remote_hashes: frozenset[str]
) -> bool:
    local_by_posix = {key.replace("\\", "/"): value for key, value in local_blobs.items()}
    local_hashes = frozenset(blob for blobs in local_by_posix.values() for blob in blobs)
    remote_paths = {path.replace("\\", "/") for path in remote_paths}
    return bool(remote_paths - set(local_by_posix) or remote_hashes - local_hashes)


def _remote_blob_id(path_info) -> Optional[str]:
    lfs = getattr(path_info, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    value = value or getattr(path_info, "blob_id", None)
    return str(value) if value else None


def gguf_variant_update_statuses(
    repo_id: str,
    local_blobs_by_variant: dict[str, dict[str, set[str]]],
    hf_token: Optional[str] = None,
    *,
    remote_paths_by_variant: Optional[dict[str, list[str]]] = None,
) -> dict[str, bool]:
    """Return update status for selected GGUF variants using one shared predicate.

    ``local_blobs_by_variant`` maps quant label -> cached MAIN GGUF repo paths
    and their local blob hashes. When ``remote_paths_by_variant`` is provided,
    callers already know the current remote filenames and this function resolves
    those paths with one ``get_paths_info`` call. Otherwise it resolves the
    current remote GGUF variant plans from model metadata. The comparison itself
    is intentionally main-GGUF-only; companion-only mmproj/MTP changes do not
    trigger an update cue.
    """
    local_by_variant = {
        str(variant).strip().lower(): blobs
        for variant, blobs in local_blobs_by_variant.items()
        if str(variant).strip() and blobs
    }
    result = {variant: False for variant in local_by_variant}
    if not local_by_variant:
        return result

    if remote_paths_by_variant is not None:
        from huggingface_hub import get_paths_info

        path_to_variants: dict[str, set[str]] = {}
        for variant, paths in remote_paths_by_variant.items():
            variant_key = str(variant).strip().lower()
            if variant_key not in local_by_variant:
                continue
            for path in paths:
                path_key = str(path).replace("\\", "/")
                if not path_key:
                    continue
                path_to_variants.setdefault(path_key, set()).add(variant_key)
        if not path_to_variants:
            return result

        remote_paths: dict[str, set[str]] = {variant: set() for variant in local_by_variant}
        remote_hashes: dict[str, set[str]] = {variant: set() for variant in local_by_variant}
        for path, variants in path_to_variants.items():
            for variant in variants:
                remote_paths[variant].add(path)
        for path_info in get_paths_info(
            repo_id = repo_id,
            paths = list(path_to_variants),
            token = hf_token,
        ):
            path = str(path_info.path).replace("\\", "/")
            remote_blob = _remote_blob_id(path_info)
            for variant in path_to_variants.get(path, set()):
                if remote_blob:
                    remote_hashes[variant].add(remote_blob)
        for variant, local_blobs in local_by_variant.items():
            result[variant] = _gguf_variant_update_available_from_remote(
                local_blobs,
                remote_paths.get(variant, set()),
                frozenset(remote_hashes.get(variant, set())),
            )
        return result

    from huggingface_hub import HfApi

    info = HfApi(token = hf_token).model_info(
        repo_id,
        files_metadata = True,
        token = hf_token,
    )
    plans = build_gguf_variant_plans(list(getattr(info, "siblings", []) or []))
    for variant, local_blobs in local_by_variant.items():
        plan = plans.get(variant)
        if plan is None:
            continue
        result[variant] = _gguf_variant_update_available_from_remote(
            local_blobs,
            {path.replace("\\", "/") for path in plan.main_filenames},
            plan.main_hashes,
        )
    return result


def _gguf_remote_update(
    repo_id: str,
    local_blobs: dict[str, set[str]],
    gguf_variant: Optional[str] = None,
    hf_token = None,
) -> bool:
    """True iff the selected remote MAIN GGUF files are absent from the local cache.

    Compares per-file blob SHAs (HF names blobs by lfs.sha256, else git blob id)
    so a metadata-only commit does NOT flag an update. ``local_blobs`` maps each
    file to the SET of blobs cached across revisions. A model that was already
    updated still holds the pre-update revision next to the new one, so it counts
    as current the moment the remote blob is one of them (a blob equality against
    a single arbitrary revision would report a phantom update). For a selected
    variant, resolve the CURRENT remote GGUF plan first so filename changes and
    re-shards are detected instead of querying stale local paths. Best-effort:
    callers swallow exceptions and keep update_available False."""
    from huggingface_hub import get_paths_info

    variant = (gguf_variant or "").strip().lower()
    if variant:
        return gguf_variant_update_statuses(
            repo_id,
            {variant: local_blobs},
            hf_token,
        ).get(variant, False)

    # Repo-level fallback only; selected variants resolve the current remote
    # plan above so renamed/re-sharded quants are compared against current files.
    local_by_posix = {key.replace("\\", "/"): value for key, value in local_blobs.items()}
    remote_path_infos = get_paths_info(repo_id = repo_id, paths = list(local_by_posix), token = hf_token)
    for path_info in remote_path_infos:
        remote_blob = _remote_blob_id(path_info)
        local_set = local_by_posix.get(path_info.path.replace("\\", "/"))
        if not local_set or remote_blob not in local_set:
            return True
    return False


async def list_cached_gguf_response(hf_token: Optional[str] = None):
    """List GGUF repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cached = await asyncio.to_thread(_scan_cached_gguf)
        return {"cached": cached}
    except Exception as e:
        logger.error(
            "Error listing cached GGUF repos: %s",
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        raise HTTPException(
            status_code = 500,
            detail = "Failed to read the local model cache.",
        ) from e


class _CachedNonGgufPayload(NamedTuple):
    size_bytes: int
    has_runnable_weights: bool
    model_format: ModelFormat


def _repo_non_gguf_model_payload(repo_info) -> _CachedNonGgufPayload:
    all_weight_blobs: dict[str, int] = {}
    adapter_blobs: dict[str, int] = {}
    safetensors_blobs: dict[str, int] = {}
    checkpoint_blobs: dict[str, int] = {}
    has_config = False
    has_adapter_config = False
    has_adapter_weights = False
    has_safetensors = False
    has_transformers_safetensors = False
    has_checkpoint = False

    def _record_blob(target: dict[str, int], file_obj, rev_id: str, file_name: str) -> None:
        blob_path = getattr(file_obj, "blob_path", None)
        size = int(file_obj.size_on_disk or 0)
        key = str(blob_path) if blob_path else f"{rev_id}:{file_name}"
        target[key] = size
        all_weight_blobs[key] = size

    for revision in repo_info.revisions:
        rev_id = getattr(revision, "commit_hash", None) or str(id(revision))
        for f in revision.files:
            file_name = str(f.file_name)
            lower = file_name.lower()
            name = lower.replace("\\", "/").rsplit("/", 1)[-1]
            if _is_gguf_filename(lower):
                continue
            if name == "config.json":
                has_config = True
                continue
            if name == "adapter_config.json":
                has_adapter_config = True
                continue
            is_adapter = _is_adapter_weight_name(name)
            is_safetensors = name.endswith(".safetensors") and not is_adapter
            is_checkpoint = _is_checkpoint_weight_name(name)
            if is_adapter:
                has_adapter_weights = True
                _record_blob(adapter_blobs, f, rev_id, file_name)
            if is_safetensors:
                has_safetensors = True
                if _is_transformers_safetensors_weight_name(name):
                    has_transformers_safetensors = True
                _record_blob(safetensors_blobs, f, rev_id, file_name)
            if is_checkpoint:
                has_checkpoint = True
                _record_blob(checkpoint_blobs, f, rev_id, file_name)

    model_format = (
        _classify_non_gguf_model_format(
            has_config = has_config,
            has_adapter_config = has_adapter_config,
            has_adapter_weights = has_adapter_weights,
            has_safetensors = has_safetensors,
            has_transformers_safetensors = has_transformers_safetensors,
            has_checkpoint_weights = has_checkpoint,
            trusted_hf_cache_repo = True,
        )
        or "unknown"
    )
    if model_format == "adapter":
        size_bytes = sum(adapter_blobs.values())
    elif model_format == "safetensors":
        size_bytes = sum(safetensors_blobs.values())
    elif model_format == "checkpoint":
        size_bytes = sum(checkpoint_blobs.values())
    else:
        size_bytes = sum(all_weight_blobs.values())

    return _CachedNonGgufPayload(
        size_bytes = size_bytes,
        has_runnable_weights = model_format != "unknown",
        model_format = model_format,
    )


def _repo_local_weight_blob_hashes(repo_info) -> frozenset[str]:
    """Local-side equivalent of remote non-GGUF weight blob hashes.

    HF names each local cache blob FILE by the file's etag (lfs.sha256 else
    blob_id), so a local file's blob hash == ``Path(blob_path).name``. Only
    runnable weight files participate so metadata-only remote changes do not
    create phantom model updates."""
    hashes: set[str] = set()
    for revision in repo_info.revisions:
        for f in revision.files:
            if not _is_non_gguf_weight_filename(f.file_name):
                continue
            blob_path = getattr(f, "blob_path", None)
            if blob_path:
                hashes.add(Path(blob_path).name)
    return frozenset(hashes)


def _cached_model_snapshot_path(repo_path: Path) -> Optional[Path]:
    resolved = hf_cache_scan.resolve_hf_cache_realpath(repo_path)
    if not resolved:
        return None
    path = Path(resolved)
    return path if path.is_dir() else None


def _read_json_object(path: Path) -> dict:
    try:
        with open(path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_model_card_frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding = "utf-8")
    except Exception:
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        body.append(line)
    if not body:
        return {}
    try:
        import yaml
        data = yaml.safe_load("\n".join(body)) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cached_model_local_metadata(repo_path: Path) -> dict:
    snapshot = _cached_model_snapshot_path(repo_path)
    if snapshot is None:
        return {}

    result: dict = {}
    config = _read_json_object(snapshot / "config.json")
    quant_method = (
        config.get("quantization_config", {}).get("quant_method")
        if isinstance(config.get("quantization_config"), dict)
        else None
    )
    if isinstance(quant_method, str) and quant_method.strip():
        result["quant_method"] = quant_method.strip()

    card = _read_model_card_frontmatter(snapshot / "README.md")
    pipeline_tag = card.get("pipeline_tag")
    if isinstance(pipeline_tag, str) and pipeline_tag.strip():
        result["pipeline_tag"] = pipeline_tag.strip()
    library_name = card.get("library_name")
    if isinstance(library_name, str) and library_name.strip():
        result["library_name"] = library_name.strip()
    tags = card.get("tags")
    if isinstance(tags, list):
        clean_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        if clean_tags:
            result["tags"] = clean_tags
    return result


def _scan_cached_models() -> list[dict]:
    """Synchronous HF-cache disk walk for non-GGUF model repos; runs in a worker thread."""
    cache_scans = all_hf_cache_scans()

    seen_lower: dict[str, dict] = {}
    inspected = 0
    skipped_gguf = 0
    skipped_no_weights = 0
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            inspected += 1
            try:
                if str(repo_info.repo_type) != "model":
                    continue
                repo_id = repo_info.repo_id
                has_main_gguf = _repo_has_gguf_files(repo_info)
                payload = _repo_non_gguf_model_payload(repo_info)
                if payload.size_bytes == 0:
                    if has_main_gguf:
                        skipped_gguf += 1
                    continue
                if not payload.has_runnable_weights:
                    skipped_no_weights += 1
                    continue
                key = repo_id.lower()
                existing = seen_lower.get(key)
                repo_path = Path(repo_info.repo_path)
                snapshot_partial = hf_cache_scan.is_snapshot_partial(
                    "model",
                    repo_id,
                    repo_path,
                )
                row = {
                    "repo_id": repo_id,
                    "size_bytes": payload.size_bytes,
                    "cache_path": str(repo_info.repo_path),
                    "partial": snapshot_partial,
                    "partial_transport": (
                        hf_cache_scan.partial_transport_for(
                            "model",
                            repo_id,
                            repo_cache_dir = repo_path,
                        )
                        if snapshot_partial
                        else None
                    ),
                    **_cached_model_local_metadata(repo_path),
                }
                row.update(
                    _cache_inventory_fields(
                        repo_id,
                        payload.model_format,
                        partial = bool(row["partial"]),
                    )
                )
                if _prefer_cache_row(row, existing):
                    seen_lower[key] = row
            except Exception as e:
                repo_label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning(f"Skipping cached model repo {repo_label}: {e}")
                continue
    cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
    logger.info(
        "Cached model scan: inspected=%d skipped_gguf=%d skipped_no_weights=%d returned=%d",
        inspected,
        skipped_gguf,
        skipped_no_weights,
        len(cached),
    )
    return cached


def _repo_update_status_candidate_row(repo_info, gguf_variant: Optional[str]) -> Optional[dict]:
    repo_id = repo_info.repo_id
    repo_path = Path(repo_info.repo_path)
    if gguf_variant or _repo_has_gguf_files(repo_info):
        total_size = _repo_gguf_size_bytes(repo_info)
        has_variant_state, variant_state_size = _gguf_variant_state_summary(repo_id)
        if total_size == 0 and not has_variant_state:
            return None
        partial = hf_cache_scan.is_gguf_repo_partial(repo_id, repo_path)
        if total_size == 0 and not partial:
            return None
        return {
            "repo_id": repo_id,
            "size_bytes": max(total_size, variant_state_size),
            "partial": partial,
        }

    payload = _repo_non_gguf_model_payload(repo_info)
    if payload.size_bytes == 0 or not payload.has_runnable_weights:
        return None
    return {
        "repo_id": repo_id,
        "size_bytes": payload.size_bytes,
        "partial": hf_cache_scan.is_snapshot_partial("model", repo_id, repo_path),
    }


def _preferred_update_status_repo_info(repo_id: str, gguf_variant: Optional[str]):
    target_lower = repo_id.lower()
    selected_info = None
    selected_row: Optional[dict] = None
    for hf_cache in all_hf_cache_scans():
        for candidate in hf_cache.repos:
            if str(candidate.repo_type) != "model":
                continue
            if candidate.repo_id.lower() != target_lower:
                continue
            row = _repo_update_status_candidate_row(candidate, gguf_variant)
            if row is None:
                continue
            if _prefer_cache_row(row, selected_row):
                selected_info = candidate
                selected_row = row
    return selected_info


async def list_cached_models_response(hf_token: Optional[str] = None):
    """List non-GGUF model repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cached = await asyncio.to_thread(_scan_cached_models)
        return {"cached": cached}
    except Exception as e:
        logger.error(
            "Error listing cached models: %s",
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        raise HTTPException(
            status_code = 500,
            detail = "Failed to read the local model cache.",
        ) from e


async def repo_update_status_response(
    repo_id: str, gguf_variant: Optional[str], hf_token: Optional[str]
) -> dict:
    """On-demand update check for a single cached repo.

    Resolves the cached ``repo_info`` for ``repo_id``, then compares local blobs
    against the remote (per-file GGUF blob SHAs via get_paths_info, or the full
    non-GGUF snapshot blob hashes via model_info). Anonymous-capable: a None
    ``hf_token`` resolves public repos (matching the picker). Best-effort: any
    failure (not cached, network error, timeout) degrades to update_available
    False so the caller never blocks on a flaky remote."""
    try:
        repo_info = _preferred_update_status_repo_info(repo_id, gguf_variant)
        if repo_info is None:
            # Not cached locally => nothing to update.
            return {"update_available": False}

        is_gguf = bool(gguf_variant) or _repo_has_gguf_files(repo_info)
        if is_gguf:
            local_blobs = _filter_gguf_blob_map_by_variant(
                _repo_gguf_blob_map(repo_info), gguf_variant
            )
            if not local_blobs:
                return {"update_available": False}
            now = time.monotonic()
            fingerprint = frozenset(blob for blobs in local_blobs.values() for blob in blobs)
            token_fp = hf_cache_scan.token_fingerprint(hf_token)
            variant_key = (gguf_variant or "").strip().lower()
            cache_key = (repo_id.lower(), variant_key, token_fp)
            with _gguf_update_check_lock:
                # Prune expired entries so the cache stays bounded to recently-seen repos.
                for stale_key in [
                    k
                    for k, v in _gguf_update_check_cache.items()
                    if now - v[0] >= _GGUF_UPDATE_CHECK_TTL
                ]:
                    _gguf_update_check_cache.pop(stale_key, None)
                cached_entry = _gguf_update_check_cache.get(cache_key)
            # Reuse only if fresh AND the local blobs are unchanged since the
            # cached result (a local download/delete must invalidate it).
            if (
                cached_entry is not None
                and now - cached_entry[0] < _GGUF_UPDATE_CHECK_TTL
                and cached_entry[2] == fingerprint
            ):
                return {"update_available": cached_entry[1]}
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _gguf_remote_update, repo_id, local_blobs, gguf_variant, hf_token
                ),
                timeout = 6.0,
            )
            with _gguf_update_check_lock:
                _gguf_update_check_cache[cache_key] = (
                    time.monotonic(),
                    result,
                    fingerprint,
                )
                while len(_gguf_update_check_cache) > _GGUF_UPDATE_CHECK_MAX:
                    _gguf_update_check_cache.pop(next(iter(_gguf_update_check_cache)))
            return {"update_available": result}

        local_hashes = _repo_local_weight_blob_hashes(repo_info)
        _total, remote_hashes = await asyncio.wait_for(
            asyncio.to_thread(
                get_repo_snapshot_metadata_cached,
                repo_id,
                hf_token,
                weight_only = True,
            ),
            timeout = 6.0,
        )
        # Remote has a weight blob the local snapshot lacks => an update exists.
        result = bool(remote_hashes - local_hashes)
        return {"update_available": result}
    except Exception:
        return {"update_available": False}  # best-effort: degrade to no-update

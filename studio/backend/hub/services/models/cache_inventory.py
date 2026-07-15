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
    snapshot_download_blob_hashes,
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

logger = get_logger(__name__)

_repo_size_cache: "OrderedDict[tuple[str, str, str], tuple[int, frozenset[str], float]]" = (
    OrderedDict()
)
_repo_size_neg_cache: "OrderedDict[tuple[str, str, str], float]" = OrderedDict()
_REPO_SIZE_CACHE_MAX = 256
_REPO_SIZE_POS_TTL = 60.0
_REPO_SIZE_NEG_TTL = 60.0
_MODEL_METADATA_TIMEOUT_SECONDS = 5.0
_repo_size_cache_lock = threading.Lock()

# Identity for a cached file with no HF blob (Windows without Developer Mode: hf
# moves the blob into snapshots/ and leaves blobs/ empty).
_LOCAL_SIZE_IDENTITY_PREFIX = "size:"


def get_repo_snapshot_metadata_cached(
    repo_id: str, hf_token: Optional[str] = None
) -> tuple[int, frozenset[str]]:
    token_fp = hf_cache_scan.token_fingerprint(hf_token)
    cache_key = (repo_id, token_fp, "snapshot")
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


def _is_real_cache_blob(blob: Optional[Path], repo_dir: Optional[Path]) -> bool:
    """True only for a real cache blob at ``<repo_dir>/blobs/<etag>``.

    A no-symlink ``snapshots/`` file (name is the filename, not an etag) or a
    repo's own ``blobs/`` subdir is not the cache blob store.
    """
    if blob is None or repo_dir is None:
        return False
    try:
        return blob.parent.resolve(strict = False) == (repo_dir / "blobs").resolve(strict = False)
    except OSError:
        return False


def _cached_blob_hash(blob_path, repo_path = None) -> Optional[str]:
    """The cache blob hash (etag) for a cached file, or None when there is no blob.

    Only a real blob under the repo's ``blobs/`` dir has name == hash; a moved
    no-symlink ``snapshots/`` file is "no blob", so the caller uses a size identity.
    """
    path = Path(blob_path)
    repo_dir = Path(repo_path) if repo_path is not None else None
    return path.name if _is_real_cache_blob(path, repo_dir) else None


def local_size_identity(size: int) -> str:
    """Identity for a cached file whose blob hash is unknowable: its size.

    Re-hashing multi-GB GGUFs on the inventory hot path is not viable, and a
    ``size:`` token never collides with a hex hash.
    """
    return f"{_LOCAL_SIZE_IDENTITY_PREFIX}{int(size)}"


def _repo_gguf_blob_map(repo_info, *, include_companions: bool = False) -> dict[str, set[str]]:
    """Map each cached GGUF file's repo-relative name to the SET of its local
    identities across all revisions.

    An identity is the file's blob hash, or a size identity when the cache holds no
    blob (Windows without Developer Mode). BOTH old and new revision blobs are kept
    (a set), so the diff treats the file as current when the remote ``main`` blob is
    in any cached revision. Main GGUF only by default; update checks opt into
    companions to compare a shared mmproj/MTP blob too.
    """
    blob_map: dict[str, set[str]] = {}
    repo_path = getattr(repo_info, "repo_path", None)
    for revision in repo_info.revisions:
        for f in revision.files:
            if include_companions:
                if not _is_gguf_filename(f.file_name):
                    continue
            elif not _is_main_gguf_filename(f.file_name):
                continue
            blob_path = getattr(f, "blob_path", None)
            if not blob_path:
                continue
            name = _cached_repo_file_name(f)
            identity = _cached_blob_hash(blob_path, repo_path)
            if identity is None:
                size = int(getattr(f, "size_on_disk", 0) or 0)
                if size <= 0:
                    continue
                identity = local_size_identity(size)
            blob_map.setdefault(name, set()).add(identity)
    return blob_map


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

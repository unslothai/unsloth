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
from hub.utils import download_manifest, download_registry
from hub.utils.hf_cache_state import snapshot_selection_key
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
    _is_training_artefact_name,
    _is_gguf_filename,
    _is_main_gguf_filename,
    _is_mmproj_filename,
    _is_transformers_safetensors_weight_name,
    _local_inventory_id,
    _runtime_for_format,
)

# Imported at module scope (not inside the per-repo scan loop) so a broken
# import surfaces at startup instead of silently emptying the inventory: the
# scan loop swallows per-repo exceptions and would drop every repo. Lives under
# ``utils`` (not ``utils.models``) to avoid the eager model-config/checkpoint
# imports in ``utils/models/__init__.py``.
from utils.hidden_models import is_hidden_model

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


_cached_inventory_flights: dict[
    tuple[asyncio.AbstractEventLoop, tuple[str, int]], asyncio.Task[list[dict]]
] = {}

# Retrying a superseded scan is only worth it while invalidations are occasional;
# past this the endpoint has to answer instead of restarting the walk forever.
_INVENTORY_SCAN_MAX_ATTEMPTS = 8
# Last scan per inventory name that confirmed its epoch, served when the cap is hit.
_last_confirmed_inventory: dict[str, list[dict]] = {}

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
            # Snapshot-relative: only the directory marks an MTP/ drafter as a companion.
            name = _cached_repo_file_name(f)
            if _is_main_gguf_filename(name):
                blob_path = getattr(f, "blob_path", None)
                size = f.size_on_disk or 0
                if blob_path:
                    unique_blobs[str(blob_path)] = size
                else:
                    unique_blobs[f"{rev_id}:{name}"] = size
    return sum(unique_blobs.values())


def _repo_has_gguf_files(repo_info) -> bool:
    return _repo_gguf_size_bytes(repo_info) > 0


def _blob_mtime(file_obj) -> float:
    ts = getattr(file_obj, "blob_last_modified", None)
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    blob_path = getattr(file_obj, "blob_path", None)
    if blob_path:
        try:
            return float(Path(blob_path).stat().st_mtime)
        except OSError:
            pass
    return 0.0


def _repo_gguf_last_modified(repo_info) -> float:
    latest = 0.0
    for revision in repo_info.revisions:
        for f in revision.files:
            if _is_main_gguf_filename(_cached_repo_file_name(f)):
                latest = max(latest, _blob_mtime(f))
    return latest


def _repo_has_mmproj(repo_info) -> bool:
    # An mmproj file only makes a repo vision-capable when it is an actual GGUF
    # projector; a non-GGUF sidecar (e.g. mmproj_config.json) does not, and the
    # runtime's projector detection is GGUF-only.
    return any(
        _is_gguf_filename(f.file_name) and _is_mmproj_filename(f.file_name)
        for revision in repo_info.revisions
        for f in revision.files
    )


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
            name = _cached_repo_file_name(f)
            if include_companions:
                if not _is_gguf_filename(name):
                    continue
            elif not _is_main_gguf_filename(name):
                continue
            blob_path = getattr(f, "blob_path", None)
            if not blob_path:
                continue
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
    candidate_partial = bool(candidate.get("partial"))
    existing_partial = bool(existing.get("partial"))
    if candidate_partial != existing_partial:
        return not candidate_partial
    candidate_active = bool(candidate.get("active_cache"))
    existing_active = bool(existing.get("active_cache"))
    if candidate_active != existing_active:
        return candidate_active
    return int(candidate.get("size_bytes") or 0) > int(existing.get("size_bytes") or 0)


class _LoadIdentity(NamedTuple):
    """A row's load target and the directory it lands in.

    *load_snapshot* is not always ``Path(load_id)``: a *load_id* left as the repo id resolves
    through ``refs/main``, so the row describes THAT snapshot.
    """

    load_id: str
    active_cache: bool
    load_snapshot: Optional[Path]


def _resolve_load_identity(
    repo_id: str,
    *,
    repo_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    active_hub_cache: Optional[Path] = None,
    payload_snapshots: Optional[frozenset[str]] = None,
) -> _LoadIdentity:
    """Single answer to "what will this row load, and from which directory".

    The partial flag, the metadata probe and the load id must agree on one directory, so resolve it
    once here. *snapshot_path* becomes the load identity whenever the repo id will not resolve, so
    pass a snapshot holding this row's payload, not merely the newest. *payload_snapshots* is every
    snapshot that does; None means the caller does not track them, so *snapshot_path* is trusted.
    """
    load_id = repo_id
    active_cache = True
    if repo_path is not None:
        try:
            if active_hub_cache is None:
                from utils.hf_cache_settings import get_hf_cache_paths
                active_hub_cache = get_hf_cache_paths().hub_cache
            active_root = active_hub_cache.resolve(strict = False)
            cached_root = repo_path.parent.resolve(strict = False)
            if cached_root != active_root:
                active_cache = False
                load_id = str(snapshot_path or repo_path.resolve(strict = False))
        except (OSError, RuntimeError, ValueError):
            active_cache = False
            load_id = str(snapshot_path or repo_path)
    # Only pin a snapshot known to hold the payload; the newest may be unusable.
    default_snapshot: Optional[Path] = None
    if (
        load_id == repo_id
        and snapshot_path is not None
        and repo_path is not None
        and (payload_snapshots is None or str(snapshot_path) in payload_snapshots)
    ):
        default_snapshot = hf_cache_scan.default_ref_snapshot(repo_path)
        # No usable refs/main: from_pretrained would fail offline or fetch HEAD, so pin the payload.
        if (
            default_snapshot is None
            or (payload_snapshots and str(default_snapshot) not in payload_snapshots)
            # refs/main can land on a torn revision, so keep the id only if it lands on a complete one.
            or (
                default_snapshot != snapshot_path
                and not hf_cache_scan.snapshot_holds_a_complete_payload(
                    default_snapshot, quants = False
                )
                and hf_cache_scan.snapshot_holds_a_complete_payload(snapshot_path, quants = False)
            )
        ):
            load_id = str(snapshot_path)
    # Keeping the repo id lets refs/main decide, possibly an older payload snapshot.
    load_snapshot = (default_snapshot or snapshot_path) if load_id == repo_id else snapshot_path
    return _LoadIdentity(load_id, active_cache, load_snapshot)


def _cache_inventory_fields(
    repo_id: str,
    model_format: ModelFormat,
    *,
    repo_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    active_hub_cache: Optional[Path] = None,
    partial: bool = False,
    requires_variant: bool = False,
    payload_snapshots: Optional[frozenset[str]] = None,
    identity: Optional[_LoadIdentity] = None,
    gguf_snapshot: Optional[Path] = None,
    repo_info = None,
    hidden_infra: bool = False,
    companion: bool = False,
) -> dict:
    """Load identity plus the capability block for one cache row.

    The SOLE producer of a row's ``capabilities``: every flag is derived from the snapshot this row
    describes, so add new flags here rather than patching them on afterwards. *identity* is accepted
    already resolved so it cannot be resolved twice to different answers.
    """
    if identity is None:
        identity = _resolve_load_identity(
            repo_id,
            repo_path = repo_path,
            snapshot_path = snapshot_path,
            active_hub_cache = active_hub_cache,
            payload_snapshots = payload_snapshots,
        )
    capabilities = _capabilities_for_format(
        model_format,
        "hf_cache",
        partial = partial,
        requires_variant = requires_variant,
    ).model_dump()
    # The loader's companion search never leaves the quants' snapshot.
    if model_format == "gguf" and (
        hf_cache_scan.snapshot_has_gguf_projector(gguf_snapshot)
        if gguf_snapshot is not None
        else repo_info is not None and _repo_has_mmproj(repo_info)
    ):
        capabilities["supports_vision"] = True
    if hidden_infra:
        capabilities["can_chat"] = False
    # A VAE / text-encoder mirror holds no language model, so it cannot chat whatever its weight
    # format says. Set HERE rather than left to the row's companion flag alone: startup auto-load
    # filters on capabilities.can_chat (isChattableCachedRepo), not on that flag, so a row that
    # only carried the flag was still auto-loadable as a chat model.
    if companion:
        capabilities["can_chat"] = False
    return {
        "inventory_id": _local_inventory_id("cache", model_format, repo_id),
        "load_id": identity.load_id,
        "active_cache": identity.active_cache,
        "model_format": model_format,
        "runtime": _runtime_for_format(model_format),
        "format_variant": None,
        "capabilities": capabilities,
    }


def invalidate_hf_cache_scans() -> None:
    hf_cache_scan.invalidate_hf_cache_scans()


def _is_hidden_infra_repo(*values: str | None) -> bool:
    """True for infra-only repos (the RAG embedder and the llama.cpp install
    validation probe) that are cached as a side effect of Studio itself and are
    not usable chat models."""
    return is_hidden_model(*values)


def _cached_row_companion(repo_id: str) -> bool:
    """Whether this row is an sd.cpp companion mirror (VAE / text encoders, no denoiser).

    Same classifier the models API uses. The chat picker is backed by THIS endpoint, so a flag set
    only on the legacy route arrives as undefined here and the filter never fires -- the same trap
    ``single_file`` fell into below. Best-effort: a classification failure never hides a row.
    """
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        return (repo_id or "").strip().lower() in sd_cpp_companion_only_repo_ids()
    except Exception:  # noqa: BLE001 -- a classification failure never hides a row
        return False


def _cached_row_task(repo_info, *, gguf: bool) -> Optional[str]:
    """Pipeline task for a cached row, from the same classifiers the models API uses.

    The Images/Video pickers filter On Device rows on this and the chat picker routes a diffusion
    pick by it, so a row that arrives without one is dropped from those lists entirely. Imported
    lazily (routes.models imports this package) and best-effort: an unreadable repo just has no
    task, exactly as before.
    """
    try:
        from routes.models import _cached_repo_task, _repo_gguf_task
        return _repo_gguf_task(repo_info) if gguf else _cached_repo_task(repo_info)
    except Exception:  # noqa: BLE001 -- a classification failure never hides a row
        return None


def _variant_state_repositories(cache_scans):
    for cache_scan in cache_scans:
        for repo in cache_scan.repos:
            try:
                if str(repo.repo_type) == "model":
                    yield "model", repo.repo_id, Path(repo.repo_path).parent
            except Exception:
                continue


def _scan_cached_gguf(
    *, cache_scans: Optional[list] = None, active_hub_cache: Optional[Path] = None
) -> list[dict]:
    """Synchronous HF-cache disk walk for GGUF repos; runs in a worker thread."""
    if cache_scans is None:
        cache_scans = all_hf_cache_scans()
    if active_hub_cache is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        active_hub_cache = get_hf_cache_paths().hub_cache
    variant_states = download_manifest.build_variant_state_index(
        _variant_state_repositories(cache_scans),
        active_hub_cache = active_hub_cache,
    )

    seen_lower: dict[str, dict] = {}
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            try:
                if str(repo_info.repo_type) != "model":
                    continue
                repo_id = repo_info.repo_id
                repo_path = Path(repo_info.repo_path)
                variant_state = variant_states.for_repo(
                    "model",
                    repo_id,
                    hub_cache = repo_path.parent,
                )
                snapshot_path = _cached_model_snapshot_path(repo_path)
                total_size = _repo_gguf_size_bytes(repo_info)
                has_variant_state, variant_state_size = _gguf_variant_state_summary(
                    repo_id,
                    hub_cache = repo_path.parent,
                    variant_state = variant_state,
                )
                is_hidden_infra = _is_hidden_infra_repo(
                    repo_id,
                    str(repo_path),
                    str(snapshot_path) if snapshot_path is not None else None,
                )
                # Hide infra repos unless the user downloaded a variant via
                # the Hub; variant state only exists for user downloads.
                if is_hidden_infra and not has_variant_state:
                    continue
                if total_size == 0 and not has_variant_state:
                    continue
                # Must run after the skips above and before the partial walk it scopes.
                gguf_snapshot, gguf_payload_snapshots = _repo_gguf_payload_snapshots(repo_info)
                partial = hf_cache_scan.is_gguf_repo_partial(
                    repo_id,
                    repo_path,
                    snapshot_dir = gguf_snapshot,
                    variant_state = variant_state,
                )
                if total_size == 0 and not partial:
                    continue
                key = repo_id.lower()
                existing = seen_lower.get(key)
                last_modified = _repo_gguf_last_modified(repo_info)
                row = {
                    "repo_id": repo_id,
                    "size_bytes": max(total_size, variant_state_size),
                    "cache_path": str(repo_info.repo_path),
                    "task": _cached_row_task(repo_info, gguf = True),
                    "partial": partial,
                    # A marker-only sibling moves neither size nor mtime.
                    "has_variant_state": has_variant_state,
                    # GGUF row-level transport is ambiguous (variants may differ);
                    # per-variant detail lives on GgufVariantDetail.
                    "partial_transport": None,
                }
                last_modified = max(last_modified, (existing or {}).get("last_modified", 0.0))
                if last_modified > 0:
                    row["last_modified"] = last_modified
                row.update(
                    _cache_inventory_fields(
                        repo_id,
                        "gguf",
                        repo_path = repo_path,
                        snapshot_path = gguf_snapshot or snapshot_path,
                        active_hub_cache = active_hub_cache,
                        partial = bool(row["partial"]),
                        requires_variant = True,
                        payload_snapshots = gguf_payload_snapshots,
                        # Scopes the row's vision flag to one directory.
                        gguf_snapshot = gguf_snapshot,
                        repo_info = repo_info,
                        # Visible infra variants remain management-only.
                        hidden_infra = is_hidden_infra,
                    )
                )
                # Only the winning cache root loads, so the loser's vision flag must not carry over.
                if _prefer_cache_row(row, existing):
                    seen_lower[key] = row
                elif last_modified > existing.get("last_modified", 0.0):
                    existing["last_modified"] = last_modified
            except Exception as e:
                repo_label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning(f"Skipping cached GGUF repo {repo_label}: {e}")
                continue
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


class _CacheSourceChanged(RuntimeError):
    pass


def _scan_cached_inventory_snapshot(scanner, expected_epoch: int) -> list[dict]:
    from utils.hf_cache_settings import get_hf_cache_paths

    active_hub_cache = get_hf_cache_paths().hub_cache
    cache_scans = all_hf_cache_scans()
    if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
        raise _CacheSourceChanged
    rows = scanner(cache_scans = cache_scans, active_hub_cache = active_hub_cache)
    # The walk itself takes seconds, so a delete or a finished download landing
    # during it supersedes these rows just as surely as one landing before it:
    # without this a repo deleted mid-scan is still listed in the response.
    if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
        raise _CacheSourceChanged
    return rows


async def _shared_cached_inventory_scan(name: str, scanner) -> list[dict]:
    for _attempt in range(_INVENTORY_SCAN_MAX_ATTEMPTS):
        epoch = hf_cache_scan.hf_cache_scans_epoch()
        try:
            rows = await hf_cache_scan.shared_scan(
                _cached_inventory_flights,
                (name, epoch),
                lambda expected_epoch = epoch: asyncio.to_thread(
                    _scan_cached_inventory_snapshot, scanner, expected_epoch
                ),
            )
        except _CacheSourceChanged:
            continue
        _last_confirmed_inventory[name] = rows
        return rows
    # Invalidations are arriving faster than the walk completes, so no scan will
    # confirm as current. Answer with the last one that did rather than spin a
    # full cache walk per epoch forever.
    logger.warning(
        "Cached %s inventory kept racing cache invalidations; serving the last confirmed scan",
        name,
    )
    return _last_confirmed_inventory.get(name, [])


async def list_cached_gguf_response(hf_token: Optional[str] = None):
    """List GGUF repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cached = await _shared_cached_inventory_scan("gguf", _scan_cached_gguf)
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
    last_modified: float
    payload_snapshot: Optional[Path]
    payload_snapshots: frozenset[str]


# Keys mirror _classify_non_gguf_model_format's kwargs, so a revision classifies like the repo.
_PAYLOAD_FLAGS = (
    "has_config",
    "has_adapter_config",
    "has_adapter_weights",
    "has_safetensors",
    "has_transformers_safetensors",
    "has_checkpoint_weights",
)


def _newest_snapshot_dir(candidates) -> Optional[Path]:
    """Newest of *candidates*, or None when there are none.

    Ordered by ``snapshot_selection_key``, shared with ``latest_snapshot_dir`` and
    ``iter_hf_cache_snapshots`` so every consumer agrees; mtime alone left frozenset ties unbroken.
    """
    paths = [Path(candidate) for candidate in candidates]
    if not paths:
        return None
    best = max(paths, key = snapshot_selection_key)
    try:
        return best.resolve()
    except OSError:
        return best


def _resolved_snapshot_ids(candidates) -> frozenset[str]:
    """The same strings ``_newest_snapshot_dir`` would return, for membership."""
    resolved: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        try:
            resolved.add(str(path.resolve()))
        except OSError:
            resolved.add(str(path))
    return frozenset(resolved)


def _repo_gguf_payload_snapshots(repo_info) -> tuple[Optional[Path], frozenset[str]]:
    """Snapshot dirs a GGUF load can actually use, plus the newest of them.

    Size sums quants over every revision but variant resolution reads only the ``load_id`` directory,
    so they must agree or an advertised quant resolves to nothing. Prefer a snapshot holding a whole
    quant (a mixed one counts; the lister trims to the completed subset), else any primary GGUF.
    """
    # Snapshot-relative: only the directory marks an ``MTP/`` drafter as a companion.
    with_gguf = [
        snapshot
        for revision in repo_info.revisions
        if (snapshot := getattr(revision, "snapshot_path", None)) is not None
        and any(_is_main_gguf_filename(_cached_repo_file_name(f)) for f in revision.files)
    ]
    complete = [
        snapshot
        for snapshot in with_gguf
        if hf_cache_scan.snapshot_has_complete_variants(str(snapshot))
    ]
    usable = complete or with_gguf
    return _newest_snapshot_dir(usable), _resolved_snapshot_ids(usable)


def _repo_non_gguf_model_payload(repo_info) -> _CachedNonGgufPayload:
    all_weight_blobs: dict[str, tuple[int, float]] = {}
    adapter_blobs: dict[str, tuple[int, float]] = {}
    safetensors_blobs: dict[str, tuple[int, float]] = {}
    checkpoint_blobs: dict[str, tuple[int, float]] = {}
    repo_flags = dict.fromkeys(_PAYLOAD_FLAGS, False)
    revision_flags: list[tuple[Path, dict[str, bool]]] = []

    def _record_blob(
        target: dict[str, tuple[int, float]], file_obj, rev_id: str, file_name: str
    ) -> None:
        blob_path = getattr(file_obj, "blob_path", None)
        size = int(file_obj.size_on_disk or 0)
        key = str(blob_path) if blob_path else f"{rev_id}:{file_name}"
        value = (size, _blob_mtime(file_obj))
        target[key] = value
        all_weight_blobs[key] = value

    for revision in repo_info.revisions:
        rev_id = getattr(revision, "commit_hash", None) or str(id(revision))
        flags = dict.fromkeys(_PAYLOAD_FLAGS, False)
        for f in revision.files:
            file_name = str(f.file_name)
            lower = file_name.lower()
            name = lower.replace("\\", "/").rsplit("/", 1)[-1]
            if _is_gguf_filename(lower):
                continue
            # Configs are opened by exact name at the snapshot root: probed below, not here.
            if name in ("config.json", "adapter_config.json"):
                continue
            is_adapter = _is_adapter_weight_name(name)
            is_safetensors = (
                name.endswith(".safetensors")
                and not is_adapter
                # Trainer state is not the model, so it cannot classify a revision as one.
                and not _is_training_artefact_name(name)
            )
            is_checkpoint = _is_checkpoint_weight_name(name) and not _is_training_artefact_name(
                name
            )
            if is_adapter:
                flags["has_adapter_weights"] = True
                _record_blob(adapter_blobs, f, rev_id, file_name)
            if is_safetensors:
                flags["has_safetensors"] = True
                if _is_transformers_safetensors_weight_name(name):
                    flags["has_transformers_safetensors"] = True
                _record_blob(safetensors_blobs, f, rev_id, file_name)
            if is_checkpoint:
                flags["has_checkpoint_weights"] = True
                _record_blob(checkpoint_blobs, f, rev_id, file_name)
        snapshot = getattr(revision, "snapshot_path", None)
        if snapshot is not None:
            for config_name, key in (
                ("config.json", "has_config"),
                # A diffusers pipeline's root manifest, which plays exactly the role config.json
                # plays for transformers: from_pretrained reads it at the snapshot root to learn
                # the component layout. Without it here no pure pipeline snapshot could classify
                # (real repos ship model_index.json and per-component configs, never a root
                # config.json), payload_snapshots came back empty, and every cached diffusion base
                # pipeline was force-flagged partial and dropped from the On Device lists.
                ("model_index.json", "has_config"),
                ("adapter_config.json", "has_adapter_config"),
            ):
                try:
                    if (Path(snapshot) / config_name).is_file():
                        flags[key] = True
                except OSError:
                    continue
            revision_flags.append((Path(snapshot), flags))
        for key, seen in flags.items():
            if seen:
                repo_flags[key] = True

    model_format = (
        _classify_non_gguf_model_format(**repo_flags, trusted_hf_cache_repo = True) or "unknown"
    )

    def _revisions_of(fmt: str) -> tuple[list, list]:
        # Weights pool across revisions, so the pinned snapshot must classify alone; trusted=False
        # because from_pretrained needs config.json in that one directory. Filename classification
        # also matches torn revisions, so the whole ones are tracked separately.
        snapshots = [
            snapshot
            for snapshot, flags in revision_flags
            if _classify_non_gguf_model_format(**flags, trusted_hf_cache_repo = False) == fmt
        ]
        return snapshots, [
            s for s in snapshots if hf_cache_scan.snapshot_holds_a_complete_payload(s, quants = False)
        ]

    payload_snapshots, complete = _revisions_of(model_format)
    if not complete:
        # The repo-wide flags are OR-ed across revisions, so they can name a format whose every
        # revision is torn while another format has a whole one sitting right there. Load what
        # loads: an interrupted safetensors attempt must not hide a complete checkpoint.
        for candidate in ("safetensors", "checkpoint", "adapter"):
            if candidate == model_format:
                continue
            candidate_snapshots, candidate_complete = _revisions_of(candidate)
            if candidate_complete:
                model_format = candidate
                payload_snapshots, complete = candidate_snapshots, candidate_complete
                break

    if model_format == "adapter":
        selected_blobs = adapter_blobs
    elif model_format == "safetensors":
        selected_blobs = safetensors_blobs
    elif model_format == "checkpoint":
        selected_blobs = checkpoint_blobs
    else:
        selected_blobs = all_weight_blobs
    return _CachedNonGgufPayload(
        size_bytes = sum(size for size, _mtime in selected_blobs.values()),
        has_runnable_weights = model_format != "unknown",
        model_format = model_format,
        last_modified = max((mtime for _size, mtime in selected_blobs.values()), default = 0.0),
        payload_snapshot = _newest_snapshot_dir(complete or payload_snapshots),
        payload_snapshots = _resolved_snapshot_ids(payload_snapshots),
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


def _is_whisper_model_config(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip().lower() == "whisper":
        return True
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(
        isinstance(name, str) and name == "WhisperForConditionalGeneration"
        for name in architectures
    )


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


def _cached_model_local_metadata(repo_path: Path, snapshot: Optional[Path] = None) -> dict:
    # Describe the directory the row hands out, not merely the newest.
    if snapshot is None:
        snapshot = _cached_model_snapshot_path(repo_path)
    if snapshot is None:
        return {}

    result: dict = {}
    config = _read_json_object(snapshot / "config.json")
    if _is_whisper_model_config(config):
        result["_hidden_stt"] = True
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


def _scan_cached_models(
    *, cache_scans: Optional[list] = None, active_hub_cache: Optional[Path] = None
) -> list[dict]:
    """Synchronous HF-cache disk walk for non-GGUF model repos; runs in a worker thread."""
    if cache_scans is None:
        cache_scans = all_hf_cache_scans()
    if active_hub_cache is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        active_hub_cache = get_hf_cache_paths().hub_cache
    try:
        variant_states = download_manifest.build_variant_state_index(
            _variant_state_repositories(cache_scans),
            active_hub_cache = active_hub_cache,
        )
    except Exception as e:
        logger.warning("Could not build shared cached-model state index: %s", e)
        variant_states = None

    seen_lower: dict[str, dict] = {}
    inspected = 0
    skipped_gguf = 0
    skipped_no_weights = 0
    skipped_stt = 0
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            inspected += 1
            try:
                if str(repo_info.repo_type) != "model":
                    continue
                repo_id = repo_info.repo_id
                repo_path = Path(repo_info.repo_path)
                snapshot_path = _cached_model_snapshot_path(repo_path)
                # The non-GGUF embedder has no variant downloads; always hide.
                if _is_hidden_infra_repo(
                    repo_id,
                    str(repo_path),
                    str(snapshot_path) if snapshot_path is not None else None,
                ):
                    continue
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
                # Resolved once so the metadata probe, partial walk and load id agree.
                identity = _resolve_load_identity(
                    repo_id,
                    repo_path = repo_path,
                    snapshot_path = payload.payload_snapshot or snapshot_path,
                    active_hub_cache = active_hub_cache,
                    payload_snapshots = payload.payload_snapshots,
                )
                load_snapshot = identity.load_snapshot
                local_metadata = _cached_model_local_metadata(repo_path, load_snapshot)
                if local_metadata.pop("_hidden_stt", False):
                    skipped_stt += 1
                    continue
                # Scoped to the row's snapshot, so an incomplete newer revision cannot flip can_chat.
                download_partial = hf_cache_scan.is_snapshot_partial(
                    "model",
                    repo_id,
                    repo_path,
                    snapshot_dir = load_snapshot,
                    variant_state = (
                        variant_states.for_repo(
                            "model",
                            repo_id,
                            hub_cache = repo_path.parent,
                        )
                        if variant_states is not None
                        else None
                    ),
                )
                # A companion-only prefetch (pipeline manifest + VAE / text-encoder but no transformer shards, left by a GGUF load) passes
                # the download check yet cannot from_pretrained, so mark it partial and the pickers stop advertising it as on-device.
                # Scoped to the row's own snapshot, like the download check above it: the repo-wide twin picks the newest revision for
                # itself, so a newer companion-only snapshot beside the complete one refs/main resolves to marked this row partial.
                companion_only = hf_cache_scan.snapshot_pipeline_missing_denoiser(load_snapshot)
                snapshot_partial = download_partial or companion_only
                # Flags are OR-ed over revisions, so no payload snapshot means no directory
                # serves the row and it would reach for the Hub.
                if not payload.payload_snapshots:
                    snapshot_partial = True
                row_task = _cached_row_task(repo_info, gguf = False)
                row = {
                    "repo_id": repo_id,
                    "size_bytes": payload.size_bytes,
                    "cache_path": str(repo_info.repo_path),
                    "task": row_task,
                    "partial": snapshot_partial,
                    "partial_transport": (
                        hf_cache_scan.partial_transport_for(
                            "model",
                            repo_id,
                            repo_cache_dir = repo_path,
                        )
                        # Only a genuine download partial has a transport; a companion-only snapshot arrived intact and has no Resume story.
                        if download_partial
                        else None
                    ),
                    # Diffusion repos with no pipeline index load only via from_single_file, so the task pickers must not offer them as
                    # pipeline loads. Without this the picker's single_file gate sees undefined on every hub-sourced row. Read from the
                    # row's snapshot: the manifest a sibling revision carries is not the one this row's load_id opens.
                    "single_file": bool(
                        row_task is not None
                        and not hf_cache_scan.snapshot_has_pipeline_index(load_snapshot)
                    ),
                    # Listed so tens of GB of companion weights stay visible and deletable, but
                    # flagged so no picker offers a denoiser-less repo as a load.
                    "companion": _cached_row_companion(repo_id),
                    **local_metadata,
                }
                last_modified = max(
                    payload.last_modified,
                    (existing or {}).get("last_modified", 0.0),
                )
                if last_modified > 0:
                    row["last_modified"] = last_modified
                row.update(
                    _cache_inventory_fields(
                        repo_id,
                        payload.model_format,
                        identity = identity,
                        partial = bool(row["partial"]),
                        companion = bool(row["companion"]),
                    )
                )
                if _prefer_cache_row(row, existing):
                    seen_lower[key] = row
                elif last_modified > existing.get("last_modified", 0.0):
                    existing["last_modified"] = last_modified
            except Exception as e:
                repo_label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning(f"Skipping cached model repo {repo_label}: {e}")
                continue
    cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
    logger.info(
        "Cached model scan: inspected=%d skipped_gguf=%d skipped_no_weights=%d "
        "skipped_stt=%d returned=%d",
        inspected,
        skipped_gguf,
        skipped_no_weights,
        skipped_stt,
        len(cached),
    )
    return cached


async def list_cached_models_response(hf_token: Optional[str] = None):
    """List non-GGUF model repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cached = await _shared_cached_inventory_scan("models", _scan_cached_models)
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

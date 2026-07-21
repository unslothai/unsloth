# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Start, cancel, and report progress for dataset downloads."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.downloads import (
    ActiveDownloadsResponse,
    CancelDatasetDownloadRequest,
    DatasetDownloadJobStatus,
    DownloadDatasetRequest,
)
from hub.services import snapshot_progress
from hub.services import download_lifecycle
from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.hf_cache_state import has_active_incomplete_blobs
from hub.utils.paths import (
    is_valid_repo_id as _is_valid_repo_id,
    resolve_cached_repo_id_case,
)
from hub.utils.snapshot_filters import (
    blob_hashes_for_siblings,
    total_size_for_siblings,
)

logger = get_logger(__name__)

_dataset_size_cache: "OrderedDict[str, tuple[int, frozenset[str], bool, str, float]]" = (
    OrderedDict()
)
_dataset_size_neg_cache: "OrderedDict[tuple[str, str], float]" = OrderedDict()
_DATASET_SIZE_CACHE_MAX = 256
_DATASET_SIZE_POS_TTL = 60.0
_DATASET_SIZE_NEG_TTL = 60.0
_DATASET_SIZE_TIMEOUT_SECONDS = 5.0
_dataset_size_cache_lock = threading.Lock()

_registry = download_registry.get_datasets_registry()


def _download_job_key(repo_id: str) -> str:
    return download_registry.normalize_repo_key(repo_id)


def get_dataset_snapshot_metadata_cached(
    repo_id: str, hf_token: Optional[str] = None
) -> tuple[int, frozenset[str]]:
    """Raw snapshot size + expected blob hashes for a dataset repo.

    The dataset worker downloads every sibling, so the denominator is the full
    sibling-size sum and the hashes cover every file. Consumed by the shared
    ``snapshot_progress`` accounting."""
    token_fp = hf_cache_scan.token_fingerprint(hf_token)
    cache_key = (repo_id, token_fp)
    with _dataset_size_cache_lock:
        cached = _dataset_size_cache.get(repo_id)
        if cached is not None:
            size, hashes, restricted, cached_fp, ts = cached
            if (time.monotonic() - ts) >= _DATASET_SIZE_POS_TTL:
                del _dataset_size_cache[repo_id]
            # A gated/private repo's metadata is only served back to the token
            # that fetched it; another token may have no access at all.
            elif not restricted or cached_fp == token_fp:
                _dataset_size_cache.move_to_end(repo_id)
                return size, hashes
        neg_ts = _dataset_size_neg_cache.get(cache_key)
        if neg_ts is not None and (time.monotonic() - neg_ts) < _DATASET_SIZE_NEG_TTL:
            return 0, frozenset()
    try:
        from huggingface_hub import HfApi

        info = HfApi(token = hf_token).dataset_info(
            repo_id,
            files_metadata = True,
            timeout = _DATASET_SIZE_TIMEOUT_SECONDS,
        )
        total = total_size_for_siblings(info.siblings)
        hashes = blob_hashes_for_siblings(info.siblings)
        restricted = bool(getattr(info, "private", False) or getattr(info, "gated", False))
    except Exception:
        with _dataset_size_cache_lock:
            _dataset_size_neg_cache[cache_key] = time.monotonic()
            _dataset_size_neg_cache.move_to_end(cache_key)
            while len(_dataset_size_neg_cache) > _DATASET_SIZE_CACHE_MAX:
                _dataset_size_neg_cache.popitem(last = False)
        return 0, frozenset()
    with _dataset_size_cache_lock:
        _dataset_size_cache[repo_id] = (
            total,
            hashes,
            restricted,
            token_fp,
            time.monotonic(),
        )
        _dataset_size_cache.move_to_end(repo_id)
        _dataset_size_neg_cache.pop(cache_key, None)
        while len(_dataset_size_cache) > _DATASET_SIZE_CACHE_MAX:
            _dataset_size_cache.popitem(last = False)
    return total, hashes


async def get_dataset_download_progress_response(
    repo_id: str,
    expected_bytes: int = 0,
    hf_token: Optional[str] = None,
) -> dict:
    """Return download progress for a HuggingFace dataset repo.

    Scans the ``datasets--owner--name`` cache dir and shares the blob accounting
    with the model path via ``snapshot_progress``. Returns ``cache_path`` for the
    UI."""
    return await snapshot_progress.snapshot_progress_response(
        repo_type = "dataset",
        repo_id = repo_id,
        job_key = _download_job_key(repo_id),
        expected_bytes = expected_bytes,
        hf_token = hf_token,
        registry = _registry,
        metadata_resolver = get_dataset_snapshot_metadata_cached,
    )


def _dataset_status(key: str, *, repo_id: Optional[str] = None) -> DatasetDownloadJobStatus:
    state, error, generation = download_lifecycle.idle_status(
        _registry,
        key,
        repo_type = "dataset",
        repo_id = repo_id,
        variant = None,
    )
    return DatasetDownloadJobStatus(state = state, error = error, generation = generation)


async def download_dataset_response(
    body: DownloadDatasetRequest, hf_token: Optional[str] = None
) -> dict:
    """Start a background download for a HuggingFace dataset."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid repo_id: {repo_id!r}",
        )
    # Canonicalize so two different-cased paste-ins share one job + cache dir.
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "dataset")
    key = _download_job_key(repo_id)

    use_xet = download_lifecycle.resolve_effective_use_xet(body.use_xet)
    transport = download_lifecycle.resolve_transport(use_xet)
    from utils.hf_cache_settings import get_hf_cache_paths

    cache_paths = get_hf_cache_paths()
    cache_env = cache_paths.child_env({})

    claimed, claim_state = _registry.claim(
        key,
        transport,
        repo_type = "dataset",
        repo_id = repo_id,
        hub_cache = str(cache_paths.hub_cache),
        xet_cache = str(cache_paths.xet_cache),
    )
    generation = _registry.current_generation(key)
    if not claimed:
        # Pollable when rejected by this repo's own in-flight job; an
        # in-progress delete leaves no job, so flag it via ``adoptable``.
        return {
            "repo_id": repo_id,
            "state": claim_state,
            "accepted": _registry.adoptable(key),
            "generation": generation,
        }
    download_manifest.clear_cancel_marker("dataset", repo_id, None)

    state = download_lifecycle.launch_worker(
        _registry,
        key,
        spawn = lambda: download_lifecycle.spawn_worker(
            ["--repo-id", repo_id, "--dataset"],
            hf_token,
            use_xet = use_xet,
            cache_env = cache_env,
        ),
        hf_token = hf_token,
        label = repo_id,
        log_prefix = "Dataset download",
        logger = logger,
        repo_type = "dataset",
        repo_id = repo_id,
        transport = transport,
        watch_name = f"hf-dataset-download-watch-{repo_id}",
    )

    return {
        "repo_id": repo_id,
        "state": state,
        "accepted": True,
        "generation": generation,
    }


async def cancel_dataset_download_response(body: CancelDatasetDownloadRequest) -> dict:
    """Cancel an in-flight dataset download (SIGKILL; HF cache resumes on next download)."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid repo_id: {repo_id!r}",
        )
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "dataset")
    key = _download_job_key(repo_id)

    state = download_lifecycle.cancel_worker(
        _registry,
        key,
        generation = body.generation,
        label = f"dataset {repo_id}",
        logger = logger,
    )
    return {"repo_id": repo_id, "state": state}


async def get_dataset_download_status_response(repo_id: str) -> DatasetDownloadJobStatus:
    """Return the latest state of a background dataset download job."""
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return DatasetDownloadJobStatus(state = "idle")
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "dataset")
    return _dataset_status(_download_job_key(repo_id), repo_id = repo_id)


async def get_active_dataset_downloads_response(repo_id: str = "") -> ActiveDownloadsResponse:
    repo_id = repo_id.strip()
    if repo_id and not _is_valid_repo_id(repo_id):
        return ActiveDownloadsResponse(downloads = [])
    canonical_repo_id = (
        await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "dataset")
        if repo_id
        else None
    )
    return ActiveDownloadsResponse(
        downloads = download_lifecycle.active_download_refs(
            _registry,
            canonical_repo_id,
            with_variant = False,
        )
    )


async def get_dataset_transport_status_response(repo_id: str) -> dict:
    """Last transport used, whether partial blobs exist, and whether they
    support byte-level resume. XET partials show via ``has_partial`` but are not
    byte-level resumable (see ``models.get_model_transport_status``)."""
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return {"has_partial": False, "last_transport": None, "resumable": False}
    return {
        "has_partial": has_active_incomplete_blobs("dataset", repo_id),
        "last_transport": download_registry.read_active_transport_marker("dataset", repo_id),
        "resumable": download_registry.is_resumable_partial("dataset", repo_id),
    }


registry = _registry

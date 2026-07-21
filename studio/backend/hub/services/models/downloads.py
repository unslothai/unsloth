# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download orchestration."""

from __future__ import annotations

import asyncio
from typing import Optional, TYPE_CHECKING

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.downloads import (
    ActiveDownloadsResponse,
    CancelDownloadRequest,
    DownloadJobStatus,
    DownloadModelRequest,
)
from hub.utils import download_registry
from hub.utils import download_manifest
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.hf_cache_state import has_active_incomplete_blobs
from hub.utils.paths import (
    is_valid_gguf_variant as _is_valid_gguf_variant,
    is_valid_repo_id as _is_valid_repo_id,
    resolve_cached_repo_id_case,
)
from hub.services import snapshot_progress
from hub.services import download_lifecycle
from hub.services.models import cache_inventory, gguf_variants

logger = get_logger(__name__)

if TYPE_CHECKING:
    import subprocess

_registry = download_registry.get_models_registry()


def _download_job_key(repo_id: str, variant: Optional[str]) -> str:
    return download_registry.normalize_job_key(
        f"{download_registry.normalize_repo_key(repo_id)}::{variant or ''}"
    )


def _job_status(
    key: str,
    *,
    repo_id: Optional[str] = None,
    variant: Optional[str] = None,
) -> DownloadJobStatus:
    state, error, generation = download_lifecycle.idle_status(
        _registry,
        key,
        repo_type = "model",
        repo_id = repo_id,
        variant = variant,
    )
    return DownloadJobStatus(state = state, error = error, generation = generation)


def _load_in_flight(repo_id: str) -> bool:
    try:
        from core.inference.llama_cpp import hf_gguf_load_in_flight
        return hf_gguf_load_in_flight(repo_id)
    except Exception:
        return False


def _load_in_flight_error(repo_id: str) -> HTTPException:
    return HTTPException(
        status_code = 409,
        detail = (
            f"A model load for '{repo_id}' is in progress and may be "
            "downloading it. Wait for the load to finish (or cancel it), "
            "then start the download."
        ),
    )


def _reject_if_load_in_flight(repo_id: str) -> None:
    if _load_in_flight(repo_id):
        raise _load_in_flight_error(repo_id)


def _spawn_download_worker(
    repo_id: str,
    variant: Optional[str],
    hf_token: Optional[str],
    use_xet: bool = True,
    protected_blob_hashes: Optional[frozenset[str]] = None,
    cache_env: Optional[dict[str, str]] = None,
) -> subprocess.Popen:
    args = ["--repo-id", repo_id]
    if variant:
        args.extend(["--variant", variant])
    return download_lifecycle.spawn_worker(
        args,
        hf_token,
        use_xet = use_xet,
        protected_blob_hashes = protected_blob_hashes,
        cache_env = cache_env,
    )


async def download_model_response(body: DownloadModelRequest, hf_token: Optional[str] = None):
    """Start a background download for a HuggingFace model."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid repo_id: {repo_id!r}",
        )
    # Canonicalize so two different-cased paste-ins share one job + cache dir.
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")

    # Avoid concurrent writers to the same HF cache files.
    _reject_if_load_in_flight(repo_id)

    variant = (body.gguf_variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid gguf_variant: {variant!r}",
        )
    key = _download_job_key(repo_id, variant)
    use_xet = download_lifecycle.resolve_effective_use_xet(body.use_xet)
    transport = download_lifecycle.resolve_transport(use_xet)
    from utils.hf_cache_settings import get_hf_cache_paths

    cache_paths = get_hf_cache_paths()
    cache_env = cache_paths.child_env({})
    variant_blob_hashes = frozenset()
    variant_progress_blob_hashes = frozenset()
    completed_baseline_bytes = 0
    if variant is not None:
        try:
            variant_blob_hashes = await asyncio.to_thread(
                gguf_variants.gguf_variant_blob_hashes,
                repo_id,
                variant,
                hf_token,
                include_companions = False,
            )
            variant_progress_blob_hashes = await asyncio.to_thread(
                gguf_variants.gguf_variant_blob_hashes,
                repo_id,
                variant,
                hf_token,
                include_companions = True,
            )
        except Exception as e:
            logger.warning(
                "GGUF hash pre-resolution failed for %s [%s]; continuing without "
                "a completed-bytes baseline or peer-protection hashes (the worker "
                "re-resolves its own blobs before purging): %s",
                repo_id,
                variant,
                download_registry.scrub_secrets(str(e), hf_token = hf_token),
            )
        has_variant_resume_state = (
            download_manifest.has_cancel_marker("model", repo_id, variant)
            or download_manifest.read_manifest("model", repo_id, variant) is not None
        )
        if variant_progress_blob_hashes and not has_variant_resume_state:
            completed_baseline_bytes = await asyncio.to_thread(
                download_registry.completed_blob_bytes,
                "model",
                repo_id,
                variant_progress_blob_hashes,
            )

    claimed, claim_state = _registry.claim(
        key,
        transport,
        repo_type = "model",
        repo_id = repo_id,
        variant = variant,
        blob_hashes = variant_blob_hashes,
        progress_blob_hashes = variant_progress_blob_hashes,
        completed_baseline_bytes = completed_baseline_bytes,
        admission_check = lambda: not _load_in_flight(repo_id),
        hub_cache = str(cache_paths.hub_cache),
        xet_cache = str(cache_paths.xet_cache),
    )
    generation = _registry.current_generation(key)
    if not claimed:
        if claim_state == "admission_blocked":
            raise _load_in_flight_error(repo_id)
        # claim_state is the blocking job's state. The client can attach only
        # when the blocker is this key's own in-flight job (adoptable); a
        # cross-variant conflict or in-progress delete is not accepted.
        return {
            "job_key": key,
            "state": claim_state,
            "accepted": _registry.adoptable(key),
            "generation": generation,
        }
    download_manifest.clear_cancel_marker("model", repo_id, variant)
    # Blobs a concurrent same-repo variant is already writing (e.g. a shared
    # mmproj). The worker must not purge these during cache preparation.
    protected_blob_hashes = _registry.peer_blob_hashes(key) if variant else frozenset()

    label = f"{repo_id}{f' [{variant}]' if variant else ''}"
    state = download_lifecycle.launch_worker(
        _registry,
        key,
        spawn = lambda: _spawn_download_worker(
            repo_id,
            variant,
            hf_token,
            use_xet = use_xet,
            protected_blob_hashes = protected_blob_hashes,
            cache_env = cache_env,
        ),
        hf_token = hf_token,
        label = label,
        log_prefix = "Download",
        logger = logger,
        repo_type = "model",
        repo_id = repo_id,
        transport = transport,
        watch_name = f"hf-download-watch-{repo_id}",
    )

    return {
        "job_key": key,
        "state": state,
        "accepted": True,
        "generation": generation,
    }


async def cancel_download_model_response(body: CancelDownloadRequest):
    """Cancel an in-flight model download (SIGKILL; HF cache resumes on next download)."""
    repo_id = body.repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid repo_id: {repo_id!r}",
        )
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")
    variant = (body.gguf_variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid gguf_variant: {variant!r}",
        )
    key = _download_job_key(repo_id, variant)

    state = download_lifecycle.cancel_worker(
        _registry,
        key,
        generation = body.generation,
        label = repo_id,
        logger = logger,
    )
    return {"job_key": key, "state": state}


async def get_download_status_response(repo_id: str, gguf_variant: str = "") -> DownloadJobStatus:
    """Return the latest state of a background download job."""
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return DownloadJobStatus(state = "idle")
    repo_id = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")
    variant = (gguf_variant or "").strip() or None
    key = _download_job_key(repo_id, variant)
    return _job_status(key, repo_id = repo_id, variant = variant)


async def get_active_downloads_response(repo_id: str = "") -> ActiveDownloadsResponse:
    """Return every in-flight download for a repo in a single call."""
    repo_id = repo_id.strip()
    if repo_id and not _is_valid_repo_id(repo_id):
        return ActiveDownloadsResponse(downloads = [])
    canonical_repo_id = (
        await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")
        if repo_id
        else None
    )
    return ActiveDownloadsResponse(
        downloads = download_lifecycle.active_download_refs(
            _registry,
            canonical_repo_id,
            with_variant = True,
        )
    )


def _variant_transport_status(repo_id: str, variant: str, hf_token: Optional[str]) -> dict:
    incomplete_hashes = download_registry.incomplete_blob_hashes(
        "model",
        repo_id,
        active_only = True,
    )
    variant_hashes = gguf_variants.gguf_variant_blob_hashes(
        repo_id,
        variant,
        hf_token,
        allow_remote = False,
    )
    has_partial = hf_cache_scan.is_variant_partial(
        repo_id,
        variant,
        incomplete_blob_hashes = incomplete_hashes,
        variant_blob_hashes = variant_hashes,
    )
    last_transport = hf_cache_scan.partial_transport_for("model", repo_id, variant)
    if (
        last_transport is None
        and has_partial
        and incomplete_hashes
        and variant_hashes
        and incomplete_hashes.intersection(variant_hashes)
    ):
        last_transport = download_registry.read_active_transport_marker(
            "model",
            repo_id,
            variant,
        )
    has_matching_incomplete = bool(
        incomplete_hashes and variant_hashes and incomplete_hashes.intersection(variant_hashes)
    )
    return {
        "has_partial": has_partial,
        "last_transport": last_transport,
        "resumable": (
            has_matching_incomplete and last_transport == download_registry.TRANSPORT_HTTP
        ),
    }


async def get_model_transport_status_response(
    repo_id: str,
    gguf_variant: str = "",
    hf_token: Optional[str] = None,
) -> dict:
    """Return last transport used for this repo + whether any partial blobs
    exist + whether that partial supports byte-level resume.

    ``resumable`` is True only when an HTTP partial exists. XET partials
    are reported via ``has_partial`` but always have ``resumable=False``
    because ``hf_xet`` rewrites the destination from scratch on every
    call (network resume happens transparently via its chunk cache).
    """
    repo_id = repo_id.strip()
    if not _is_valid_repo_id(repo_id):
        return {"has_partial": False, "last_transport": None, "resumable": False}
    variant = (gguf_variant or "").strip()
    if variant:
        if not _is_valid_gguf_variant(variant):
            return {"has_partial": False, "last_transport": None, "resumable": False}
        return _variant_transport_status(repo_id, variant, hf_token)
    return {
        "has_partial": has_active_incomplete_blobs("model", repo_id),
        "last_transport": download_registry.read_active_transport_marker("model", repo_id),
        "resumable": download_registry.is_resumable_partial("model", repo_id),
    }


async def get_gguf_download_progress_response(
    repo_id: str,
    variant: str = "",
    expected_bytes: int = 0,
    hf_token: Optional[str] = None,
) -> dict:
    """Return download progress for a specific GGUF variant."""
    expected_total = max(expected_bytes, 0)
    progress_variant = variant.strip() or None
    if progress_variant is not None and not _is_valid_gguf_variant(progress_variant):
        return {
            "downloaded_bytes": 0,
            "completed_bytes": 0,
            "complete_on_disk": False,
            "expected_bytes": expected_total,
            "progress": 0,
            "cache_path": None,
        }

    def _metadata_resolver(
        resolved_repo_id: str, token: Optional[str]
    ) -> tuple[int, frozenset[str]]:
        if progress_variant is None:
            return expected_total, frozenset()
        requirement = gguf_variants.gguf_variant_requirements(
            resolved_repo_id,
            progress_variant,
            token,
        )
        if requirement is not None:
            return requirement.download_size_bytes, requirement.required_hashes
        manifest = download_manifest.read_manifest(
            "model",
            resolved_repo_id,
            progress_variant,
        )
        if manifest is not None:
            return (
                sum(max(0, int(file.size or 0)) for file in manifest.expected_files),
                frozenset(file.sha256 for file in manifest.expected_files if file.sha256),
            )
        return (
            expected_total,
            gguf_variants.gguf_variant_blob_hashes(
                resolved_repo_id,
                progress_variant,
                token,
                allow_remote = False,
            ),
        )

    return await snapshot_progress.snapshot_progress_response(
        repo_type = "model",
        repo_id = repo_id,
        job_key = _download_job_key(repo_id, progress_variant),
        expected_bytes = expected_total,
        hf_token = hf_token,
        registry = _registry,
        metadata_resolver = _metadata_resolver,
        variant = progress_variant,
    )


async def get_download_progress_response(
    repo_id: str,
    expected_bytes: int = 0,
    hf_token: Optional[str] = None,
) -> dict:
    """Return download progress for any HuggingFace model repo.

    Checks the local HF cache for completed blobs and in-progress
    (.incomplete) downloads. Uses the caller-supplied expected total
    when available; otherwise queries HF metadata and caches it.
    Also returns ``cache_path``: the realpath of the snapshot directory
    (or the cache repo root if no snapshot exists yet) so the UI can
    show users where the weights actually live on disk.
    """
    return await snapshot_progress.snapshot_progress_response(
        repo_type = "model",
        repo_id = repo_id,
        job_key = _download_job_key(repo_id, None),
        expected_bytes = expected_bytes,
        hf_token = hf_token,
        registry = _registry,
        metadata_resolver = cache_inventory.get_repo_snapshot_metadata_cached,
    )


registry = _registry

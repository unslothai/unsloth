# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

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
from hub.utils import gguf_plan
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.hf_cache_state import has_active_incomplete_blobs, preferred_repo_cache_dirs
from hub.utils.snapshot_filters import blob_hashes_for_siblings
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


# A scope rides the variant slot as "@name". No GGUF quant label starts with "@", so a scoped job
# never collides with a real variant or the full snapshot.
_SCOPE_PREFIX = "@"


def _scope_variant(scope_id: Optional[str]) -> Optional[str]:
    scope = (scope_id or "").strip()
    return f"{_SCOPE_PREFIX}{scope}" if scope else None


def scoped_file_blob_hashes(
    repo_id: str, files: Sequence[str], hf_token: Optional[str]
) -> frozenset[str]:
    """Blob hashes for exactly ``files``, so a scoped job's progress, purge and peer
    protection cover its own files and nothing else in the repo."""
    from huggingface_hub import HfApi

    wanted = set(files)
    info = HfApi().model_info(repo_id, files_metadata = True, token = hf_token)
    return blob_hashes_for_siblings(
        [s for s in info.siblings if getattr(s, "rfilename", None) in wanted]
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


def _diffusion_load_in_flight(repo_id: str) -> bool:
    """Whether the Images or Video backend is currently STAGING *repo_id* (or its companion
    base repo) for a load. Both stage through the same HF cache as the download worker, so a
    download started now would put two writers on the same blobs -- the exact race the
    llama.cpp guard below prevents for chat. ``loading_repo_ids`` is the same signal the
    delete-cached guard uses. Best-effort: an unavailable backend reports not-in-flight so a
    probe failure never blocks a legitimate download."""
    key = download_registry.normalize_repo_key(repo_id)
    getters = []
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        getters.append(get_active_diffusion_engine)
    except Exception:
        pass
    try:
        from core.inference.video import get_video_backend
        getters.append(get_video_backend)
    except Exception:
        pass
    for get_backend in getters:
        try:
            backend = get_backend()
            for lid in getattr(backend, "loading_repo_ids", tuple)():
                if download_registry.normalize_repo_key(str(lid)) == key:
                    return True
        except Exception as e:
            logger.debug(f"Load-in-flight probe failed for {repo_id}: {e}")
            continue
    return False


def _load_in_flight(repo_id: str) -> bool:
    """Whether ANY loader is already fetching *repo_id*. Chat is not the only loader that
    downloads on the load path: the Images and Video backends stage their snapshots the same
    way, so both are consulted."""
    try:
        from core.inference.llama_cpp import hf_gguf_load_in_flight
        if hf_gguf_load_in_flight(repo_id):
            return True
    except Exception:
        pass
    return _diffusion_load_in_flight(repo_id)


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
    files: Optional[Sequence[str]] = None,
    allow_ambient_token: bool = True,
) -> subprocess.Popen:
    args = ["--repo-id", repo_id]
    if variant:
        args.extend(["--variant", variant])
    if files:
        # Via a temp file, not argv: a pipeline repo's list runs to hundreds of names.
        args.extend(["--files-json", download_lifecycle.write_files_manifest(files)])
    return download_lifecycle.spawn_worker(
        args,
        hf_token,
        use_xet = use_xet,
        protected_blob_hashes = protected_blob_hashes,
        cache_env = cache_env,
        allow_ambient_token = allow_ambient_token,
    )


async def download_model_response(
    body: DownloadModelRequest,
    hf_token: Optional[str] = None,
    *,
    allow_ambient_token: bool = True,
):
    """Start a background download for a HuggingFace model.

    ``allow_ambient_token=False`` keeps the worker anonymous when the caller sent
    no token, for repos named over the API rather than chosen here.
    """
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
    # A scoped job fetches only `files` and keys itself apart from the repo's full snapshot.
    scoped_files = [f for f in (body.files or []) if f and f.strip()]
    scope_variant = _scope_variant(body.scope_id)
    if scope_variant is not None:
        if variant is not None:
            raise HTTPException(
                status_code = 400,
                detail = "scope_id and gguf_variant are mutually exclusive.",
            )
        if not scoped_files:
            raise HTTPException(status_code = 400, detail = "scope_id requires a non-empty files list.")
        if not _is_valid_gguf_variant(scope_variant):
            raise HTTPException(status_code = 400, detail = f"Invalid scope_id: {body.scope_id!r}")
        variant = scope_variant
    key = _download_job_key(repo_id, variant)
    # Off the event loop: resolving "auto" can run the Xet reachability probe, and a blackholed DNS
    # makes that outlast its 3s budget while every other request waits behind it.
    use_xet, transport_reason = await asyncio.to_thread(
        download_lifecycle.resolve_requested_use_xet,
        getattr(body, "transport_mode", None),
        body.use_xet,
    )
    transport = download_lifecycle.resolve_transport(use_xet)
    logger.info("Download transport for %s: %s (%s)", repo_id, transport, transport_reason)
    from utils.hf_cache_settings import get_hf_cache_paths

    cache_paths = get_hf_cache_paths()
    cache_env = cache_paths.child_env({})
    variant_blob_hashes = frozenset()
    variant_progress_blob_hashes = frozenset()
    completed_baseline_bytes = 0
    if variant is not None:
        try:
            if scope_variant is not None:
                # A scope owns exactly its own files: same set for purge and for progress.
                variant_blob_hashes = await asyncio.to_thread(
                    scoped_file_blob_hashes, repo_id, scoped_files, hf_token
                )
                variant_progress_blob_hashes = variant_blob_hashes
            else:
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
        scoped_files = scoped_files if scope_variant is not None else None,
    )
    generation = _registry.current_generation(key)
    if not claimed:
        if claim_state == "admission_blocked":
            raise _load_in_flight_error(repo_id)
        if claim_state == "scope_file_mismatch":
            raise HTTPException(
                status_code = 409,
                detail = (
                    f"Another download for '{repo_id}' is already fetching a different "
                    "set of files. Wait for it to finish (or cancel it), then start "
                    "this one."
                ),
            )
        # claim_state is the blocking job's state. Attaching and accepting are one verdict: only this
        # key's own in-flight job can be joined, and a cross-variant conflict or in-progress delete joined
        # nothing.
        adoptable = _registry.adoptable(key)
        return {
            "job_key": key,
            "state": claim_state,
            "accepted": adoptable,
            "attached": adoptable,
            "generation": generation,
            # An adopted job keeps the transport it started on, so report it rather than let the caller assume
            # the one it asked for.
            "transport": _registry.job_transport(key),
            # And its cancel marker: a run that fell back from Xet to HTTP still cancels into a restart-only
            # partial.
            "cancel_transport": _registry.job_cancel_transport(key),
        }
    download_manifest.clear_cancel_marker(
        "model",
        repo_id,
        variant,
        hub_cache = cache_paths.hub_cache,
    )
    # Blobs a concurrent same-repo variant is already writing, such as a shared mmproj: the worker must
    # not purge these during cache preparation.
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
            files = scoped_files if scope_variant is not None else None,
            allow_ambient_token = allow_ambient_token,
        ),
        hf_token = hf_token,
        allow_ambient_token = allow_ambient_token,
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
        "attached": False,
        "generation": generation,
        # The transport that was actually resolved: an explicit "xet" is downgraded to HTTP where hf_xet is
        # unavailable, and a client that assumed its request stood would offer the wrong stop control.
        "transport": transport,
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
    # A partial counts toward "resumable" only while a writer that reopens it is installed, since the
    # next download start sweeps whatever cannot be reopened.
    resumable_hashes = download_registry.incomplete_blob_hashes(
        "model",
        repo_id,
        active_only = True,
        resumable_only = True,
    )
    has_resumable_incomplete = bool(
        resumable_hashes and variant_hashes and resumable_hashes.intersection(variant_hashes)
    )
    return {
        "has_partial": has_partial,
        "last_transport": last_transport,
        "resumable": (
            has_resumable_incomplete and last_transport == download_registry.TRANSPORT_HTTP
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


def _variant_manifest_in_any_cache(
    repo_id: str,
    variant: str,
    *,
    force_active: bool = False,
    active_root: Optional[Path] = None,
) -> Optional[download_manifest.Manifest]:
    """The manifest half of :func:`_variant_manifest_decision`."""
    return _variant_manifest_decision(
        repo_id, variant, force_active = force_active, active_root = active_root
    )[1]


def _variant_manifest_decision(
    repo_id: str,
    variant: str,
    *,
    force_active: bool = False,
    active_root: Optional[Path] = None,
) -> "tuple[str, Optional[download_manifest.Manifest]]":
    """The variant's manifest from whichever cache dir on disk holds it, and why.

    The verdict is ``"found"``, ``"absent"`` (no cache on disk has one) or ``"refused"`` (one
    exists but applying it across the scanned caches would be wrong). Callers have to tell those
    last two apart: a refusal is a decision that NO manifest may speak here, so re-reading one by
    another route walks straight back into the answer this function just rejected.

    snapshot_progress reads manifests per scanned cache entry (``entry.parent``)
    while this resolver only ever asked the active cache, so the two could
    disagree about whether a manifest exists at all. When it lost, the expected
    file set came back empty and the hash filter then dropped every blob in the
    shared ``blobs/`` dir -- a finished variant reporting 0 bytes against the
    caller's catalog-hinted total. Active cache first, so the common case is one
    lookup; every candidate found has to agree before one is returned.
    """
    # The active cache's manifest is a candidate like any other, NOT an early return: its repo dir can
    # be gone while its scoped state holds an old manifest, and applying those hashes to a remembered
    # cache that has the complete variant filters out every blob of it.
    found: list[download_manifest.Manifest] = []
    # active_root is the root the job records, which is the one snapshot_progress scans and not
    # necessarily the configured default.
    active_manifest = download_manifest.read_manifest(
        "model", repo_id, variant, hub_cache = active_root
    )
    if active_manifest is not None:
        found.append(active_manifest)
    # The active cache was just probed by the call above and a state-dir miss is not free, so skip the
    # entry that repeats it; in the common case preferred_repo_cache_dirs returns only that entry.
    active = download_manifest._canonical_hub_cache(active_root)
    # The SAME cache dirs snapshot_progress will scan: a remembered cache's manifest for the same
    # variant would have its hashes applied to the active root's blobs, leaving the card at 0 B.
    for entry in preferred_repo_cache_dirs(
        "model", repo_id, force_active = force_active, active_root = active_root
    ):
        if active is not None and download_manifest._canonical_hub_cache(entry.parent) == active:
            if active_manifest is None:
                # Anything returned for a cache with no manifest of its own would be another cache's answer applied
                # to its blobs.
                return ("refused", None)
            continue
        manifest = download_manifest.read_manifest(
            "model",
            repo_id,
            variant,
            hub_cache = entry.parent,
        )
        if manifest is None:
            # A scanned cache that contributed NOTHING may hold the complete snapshot, since a manifest can be
            # deleted or never written, and another cache's hashes would filter out every blob AND disable the
            # name-based fallback.
            return ("refused", None)
        found.append(manifest)
    if not found:
        return ("absent", None)
    # One answer, or several that agree: safe to apply to every scanned entry, which is what
    # snapshot_progress does with this hash set.
    # Several that DISAGREE must be refused: snapshot_progress picks its reading by bytes across all
    # preferred cache dirs while the hashes come from one lookup, so the first cache's older revision
    # filters out every blob of a later complete one. None degrades to the name-based fallback.
    first = _manifest_hashes(found[0])
    if any(_manifest_hashes(m) != first for m in found[1:]):
        return ("refused", None)
    return ("found", found[0])


def _manifest_hashes(manifest: download_manifest.Manifest) -> frozenset[str]:
    """The manifest's expected-file identity, for comparing two caches' answers."""
    return frozenset(f"{f.sha256 or ''}:{f.path}:{f.size}" for f in (manifest.expected_files or ()))


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
        job_key = _download_job_key(resolved_repo_id, progress_variant)
        job = _registry.get_job(job_key)
        # getattr, the same way snapshot_progress reads it: a registry without the accessor simply has no recorded root.
        get_job_metadata = getattr(_registry, "get_job_metadata", None)
        job_metadata = get_job_metadata(job_key) if callable(get_job_metadata) else None
        hub_cache = getattr(job_metadata, "hub_cache", None)
        verdict, manifest = _variant_manifest_decision(
            resolved_repo_id,
            progress_variant,
            # Same scoping rule snapshot_progress applies to its own scan.
            force_active = job.state in {"running", "cancelling"},
            active_root = Path(hub_cache) if hub_cache else None,
        )
        if manifest is not None:
            return (
                sum(max(0, int(file.size or 0)) for file in manifest.expected_files),
                frozenset(file.sha256 for file in manifest.expected_files if file.sha256),
            )
        if verdict == "refused":
            # A refusal is not a miss: the blob-hash helper reads the DEFAULT cache's manifest with none of
            # this scoping, so falling through reinstates the hashes just rejected. An empty set degrades to
            # the per-entry name-based fallback.
            return (expected_total, frozenset())
        return (
            expected_total,
            gguf_variants.gguf_variant_blob_hashes(
                resolved_repo_id,
                progress_variant,
                token,
                allow_remote = False,
            ),
        )

    def _expected_files_resolver(
        resolved_repo_id: str, token: Optional[str]
    ) -> Sequence[download_manifest.ExpectedFile]:
        """What HF says this variant should contain, paths and declared sizes.

        The only thing that lets a finished variant whose manifest is missing
        settle terminal instead of staying partial forever, so it has to be the
        metadata's own file list: a byte tally taken from the shared blobs/ dir
        cannot tell this quant's bytes from a sibling's. The requirement lookup
        is cached, and snapshot_progress only calls this once a reading has
        otherwise passed for complete.
        """
        if progress_variant is None:
            return ()
        requirement = gguf_variants.gguf_variant_requirements(
            resolved_repo_id,
            progress_variant,
            token,
        )
        return requirement.expected_files if requirement is not None else ()

    def _variant_file_matcher(path: str, *, companions: bool = True) -> bool:
        # Main shards are matched by quant label; mmproj and the MTP drafter are downloaded with every
        # variant, so they belong to whichever one is being polled.
        # companions=False asks the narrower question, whether this path proves the quant ITSELF is here:
        # shared companions belong to every quant, so counting them reported bytes for a deleted file.
        if progress_variant is None:
            return False
        if gguf_plan.is_main_gguf_variant_path(path, progress_variant):
            return True
        return companions and gguf_plan.is_companion_gguf_path(path)

    return await snapshot_progress.snapshot_progress_response(
        repo_type = "model",
        repo_id = repo_id,
        job_key = _download_job_key(repo_id, progress_variant),
        expected_bytes = expected_total,
        hf_token = hf_token,
        registry = _registry,
        metadata_resolver = _metadata_resolver,
        variant = progress_variant,
        variant_file_matcher = _variant_file_matcher,
        expected_files_resolver = _expected_files_resolver,
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model deletion services."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.gguf import extract_quant_label
from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    purge_partial_repo,
    purge_repo_cache_dirs,
)
from hub.utils.paths import (
    is_valid_gguf_variant as _is_valid_gguf_variant,
    is_valid_repo_id as _is_valid_repo_id,
    resolve_cached_repo_id_case,
)
from hub.services import resolve_destructive_repo_ids
from hub.services.models import cache_inventory, downloads, gguf_variants
from hub.services.models.common import (
    _is_gguf_filename,
    _is_main_gguf_filename,
    _is_mmproj_filename,
)

logger = get_logger(__name__)


def _snapshot_blob_reference_counts(repo_dir: Optional[Path]) -> dict[Path, int]:
    """Map each blob's realpath to how many snapshot symlinks still point at it.

    Used by per-variant deletion to avoid unlinking a content-addressed blob
    that another revision (or a differently-named file) still references — which
    would dangle that symlink and make the scanner report the repo as partial.
    Computed *after* the target variant's own symlinks are removed so they don't
    self-count. Platforms without symlinks (some Windows setups) yield an empty
    map, degrading to "no sharing detected"."""
    counts: dict[Path, int] = {}
    if repo_dir is None:
        return counts
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return counts
    try:
        entries = list(snapshots.rglob("*"))
    except OSError:
        return counts
    for link in entries:
        try:
            if not link.is_symlink():
                continue
            target = link.resolve()
        except OSError:
            continue
        counts[target] = counts.get(target, 0) + 1
    return counts


def _blob_hash_from_path(blob: Path) -> Optional[str]:
    name = blob.name
    if not name or name.endswith(INCOMPLETE_SUFFIX):
        return None
    return name


def _path_exists_or_symlink(path: Path) -> bool:
    try:
        return path.is_symlink() or path.exists()
    except OSError:
        return False


def _repo_file_matches(
    target_repo, predicate
) -> list[tuple[Path, Optional[Path], str]]:
    matches: list[tuple[Path, Optional[Path], str]] = []
    for rev in getattr(target_repo, "revisions", ()):
        for f in getattr(rev, "files", ()):
            name = str(getattr(f, "file_name", ""))
            if not predicate(name):
                continue
            file_path = getattr(f, "file_path", None)
            if not file_path:
                continue
            blob_path = getattr(f, "blob_path", None)
            matches.append(
                (
                    Path(file_path),
                    Path(blob_path) if blob_path else None,
                    name,
                )
            )
    return matches


def _has_remaining_main_gguf(target_repo) -> bool:
    return any(
        _path_exists_or_symlink(snap)
        for snap, _blob, _name in _repo_file_matches(
            target_repo,
            _is_main_gguf_filename,
        )
    )


def _delete_gguf_variant_from_repos(
    repo_id: str,
    variant: str,
    target_repos: list,
    hf_token: Optional[str],
    *,
    sibling_active: bool = False,
) -> dict:
    failures: list[str] = []
    removed_snapshots = 0
    deleted_bytes = 0
    deleted_blobs = 0
    completed_hashes: set[str] = set()

    for target_repo in target_repos:
        repo_dir = (
            Path(target_repo.repo_path)
            if getattr(target_repo, "repo_path", None)
            else None
        )
        matched = _repo_file_matches(
            target_repo,
            lambda name: _is_main_gguf_filename(name)
            and extract_quant_label(name).lower() == variant.lower(),
        )

        for snap, _blob, name in matched:
            try:
                if _path_exists_or_symlink(snap):
                    snap.unlink()
                    removed_snapshots += 1
            except OSError as e:
                failures.append(f"{name}: {e}")

        companion_matches: list[tuple[Path, Optional[Path], str]] = []
        if matched and not sibling_active and not _has_remaining_main_gguf(target_repo):
            companion_matches = _repo_file_matches(
                target_repo,
                lambda name: _is_gguf_filename(name) and _is_mmproj_filename(name),
            )
            for snap, _blob, name in companion_matches:
                try:
                    if _path_exists_or_symlink(snap):
                        snap.unlink()
                        removed_snapshots += 1
                except OSError as e:
                    failures.append(f"{name}: {e}")

        ref_counts = _snapshot_blob_reference_counts(repo_dir)
        seen_blobs: set[Path] = set()
        for _snap, blob, name in [*matched, *companion_matches]:
            if blob is None:
                continue
            blob_hash = _blob_hash_from_path(blob)
            if blob_hash:
                completed_hashes.add(blob_hash)
            try:
                blob_key = blob.resolve()
            except OSError:
                blob_key = blob
            if blob_key in seen_blobs:
                continue
            seen_blobs.add(blob_key)
            if ref_counts.get(blob_key, 0) > 0:
                continue
            try:
                if blob.exists():
                    deleted_bytes += blob.stat().st_size
                    blob.unlink()
                    deleted_blobs += 1
            except OSError as e:
                failures.append(f"{name}: {e}")

    if failures:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: "
                f"{len(failures)} file(s) are in use. "
                "Unload the model and try again."
            ),
        )

    incomplete_result = gguf_variants.delete_variant_incomplete_blobs_result(
        repo_id,
        variant,
        hf_token,
        extra_hashes = frozenset(completed_hashes),
        companions = not sibling_active,
    )
    if incomplete_result.unresolved:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: partial "
                "download bytes exist but this variant's blob hashes are unavailable. "
                "Reconnect or provide access to the repo, then try again."
            ),
        )

    state_purged = download_manifest.purge_state("model", repo_id, variant)
    if (
        removed_snapshots == 0
        and deleted_blobs == 0
        and incomplete_result.deleted == 0
        and not state_purged
    ):
        raise HTTPException(
            status_code = 404,
            detail = f"Variant {variant} not found in cache for {repo_id}",
        )

    freed_mb = deleted_bytes / (1024 * 1024)
    logger.info(
        f"Deleted {removed_snapshots} file(s) for {repo_id} variant {variant}: "
        f"{freed_mb:.1f} MB freed"
    )
    return {"status": "deleted", "repo_id": repo_id, "variant": variant}


def _loaded_id_matches_repo(loaded_id: str, repo_id: str) -> bool:
    """True when *loaded_id* is *repo_id* itself or a file within it. Boundary
    aware on the ``/`` separator so a sibling repo that merely shares a name
    prefix (``org/model`` vs ``org/model-v2``) does not falsely match."""
    rid = repo_id.lower()
    lid = loaded_id.lower()
    return lid == rid or lid.startswith(f"{rid}/")


def _loaded_repo_variant_blocks_delete(
    loaded_id: str,
    repo_id: str,
    delete_variant: Optional[str],
    loaded_variant: Optional[str],
) -> bool:
    if not _loaded_id_matches_repo(loaded_id, repo_id):
        return False
    if not delete_variant:
        return True
    if not loaded_variant:
        return True
    return loaded_variant.lower() == delete_variant.lower()


_LOAD_STATE_UNVERIFIABLE_DETAIL = (
    "Couldn't verify whether this model is still loaded for inference. "
    "Unload it if it is active, then try deleting again."
)


def _llama_cpp_blocks_delete(repo_id: str, variant: Optional[str]) -> bool:
    """Whether the llama.cpp backend currently holds *repo_id* (/variant).

    Acquiring the backend is failure-tolerant: an import/construction error
    means no llama-server runs in this process, so nothing is loaded and the
    delete is safe. Reading the load state of an *acquired* backend is not
    guarded here: if it raises, the exception propagates so the caller can
    fail closed rather than delete weights out from under a live process."""
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
    except Exception as e:
        logger.debug(
            f"llama.cpp backend unavailable during delete guard for {repo_id}: {e}"
        )
        return False
    loaded_id = backend.model_identifier
    loaded_variant = getattr(backend, "hf_variant", None)
    if backend.is_active and not backend.is_loaded and loaded_id:
        return _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
        )
    if backend.is_loaded and loaded_id:
        return _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
        )
    return False


def _inference_backend_blocks_delete(repo_id: str) -> bool:
    """Whether the subprocess inference backend currently holds *repo_id*.

    Same fail-open-on-acquire / surface-on-query contract as
    :func:`_llama_cpp_blocks_delete`."""
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
    except Exception as e:
        logger.debug(
            f"Inference backend unavailable during delete guard for {repo_id}: {e}"
        )
        return False
    active_name = backend.active_model_name
    return bool(active_name) and _loaded_id_matches_repo(active_name, repo_id)


async def delete_cached_model_response(
    repo_id: str, variant: Optional[str] = None, hf_token: Optional[str] = None
):
    """Delete a cached model repo (or a specific GGUF variant) from the HF cache.

    When *variant* is provided, only the GGUF files matching that quant label
    are removed (e.g. ``UD-Q4_K_XL``).  Otherwise the entire repo is deleted.
    Refuses if the model is currently loaded for inference.
    """
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid gguf_variant: {variant!r}",
        )

    # Refuse if the model is currently loaded for inference. This guard fails
    # closed: when a live backend's load state can't be read, abort with 503
    # rather than risk unlinking weights under a running inference process.
    # (An absent backend is not an error — nothing can be loaded through one
    # that isn't running — so the helpers return False in that case.)
    try:
        blocks_delete = _llama_cpp_blocks_delete(repo_id, variant) or (
            _inference_backend_blocks_delete(repo_id)
        )
    except Exception as e:
        logger.warning(
            f"Load-state verification failed for {repo_id}; refusing delete: {e}"
        )
        raise HTTPException(
            status_code = 503,
            detail = _LOAD_STATE_UNVERIFIABLE_DETAIL,
        )
    if blocks_delete:
        raise HTTPException(
            status_code = 400,
            detail = "Unload the model before deleting",
        )

    repo_key = resolve_cached_repo_id_case(repo_id, repo_type = "model")
    if not downloads.registry.begin_delete(repo_key, variant):
        detail = (
            f"Cancel the {variant} download before deleting it."
            if variant is not None
            else "Cancel the active downloads before deleting."
        )
        raise HTTPException(status_code = 400, detail = detail)
    try:
        return await asyncio.to_thread(
            _delete_cached_model_blocking, repo_id, variant, hf_token
        )
    finally:
        downloads.registry.end_delete(repo_key, variant)
        cache_inventory.invalidate_hf_cache_scans()


def _delete_cached_model_blocking(
    repo_id: str, variant: Optional[str], hf_token: Optional[str]
) -> dict:
    try:
        # A sibling quantization may be downloading concurrently (a per-variant
        # delete does not block it). When one is, restrict this delete to the
        # variant's own files so the shared mmproj companion stays intact for the
        # live sibling.
        sibling_active = bool(
            variant and downloads.registry.has_active_peer_variant(repo_id, variant)
        )

        cache_scans = cache_inventory.all_hf_cache_scans()

        candidate_entries = []
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                if str(repo_info.repo_type) != "model":
                    continue
                if repo_info.repo_id.lower() == repo_id.lower():
                    candidate_entries.append((hf_cache, repo_info))

        matched_repo_ids = resolve_destructive_repo_ids(
            repo_id,
            [str(repo_info.repo_id) for _hf_cache, repo_info in candidate_entries],
            noun = "models",
        )
        target_entries = [
            (hf_cache, repo_info)
            for hf_cache, repo_info in candidate_entries
            if str(repo_info.repo_id) in matched_repo_ids
        ]

        if not target_entries:
            if variant is None:
                cache_purged = purge_repo_cache_dirs(
                    "model", repo_id
                ) or purge_partial_repo("model", repo_id)
                state_purged = (
                    download_manifest.purge_all_state_for_repo("model", repo_id) > 0
                )
                if cache_purged or state_purged:
                    return {"status": "deleted", "repo_id": repo_id}
            if variant:
                incomplete_result = (
                    gguf_variants.delete_variant_incomplete_blobs_result(
                        repo_id,
                        variant,
                        hf_token,
                        companions = not sibling_active,
                    )
                )
                if incomplete_result.unresolved:
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            f"Couldn't fully delete {variant} for {repo_id}: partial "
                            "download bytes exist but this variant's blob hashes are unavailable. "
                            "Reconnect or provide access to the repo, then try again."
                        ),
                    )
                state_purged = download_manifest.purge_state(
                    "model",
                    repo_id,
                    variant,
                )
                if incomplete_result.deleted > 0 or state_purged:
                    return {
                        "status": "deleted",
                        "repo_id": repo_id,
                        "variant": variant,
                    }
            raise HTTPException(status_code = 404, detail = "Model not found in cache")

        if variant:
            return _delete_gguf_variant_from_repos(
                repo_id,
                variant,
                [repo for _cache, repo in target_entries],
                hf_token,
                sibling_active = sibling_active,
            )

        deleted_revisions = False
        for hf_cache, repo_info in target_entries:
            revision_hashes = [
                rev.commit_hash
                for rev in repo_info.revisions
                if getattr(rev, "commit_hash", None)
            ]
            if not revision_hashes:
                continue
            delete_strategy = hf_cache.delete_revisions(*revision_hashes)
            logger.info(
                f"Deleting cached model {repo_id} from "
                f"{getattr(hf_cache, 'cache_dir', '<unknown>')}: "
                f"{delete_strategy.expected_freed_size_str} will be freed"
            )
            delete_strategy.execute()
            deleted_revisions = True

        cache_purged = purge_repo_cache_dirs("model", repo_id)
        partial_purged = purge_partial_repo("model", repo_id)
        state_purged = download_manifest.purge_all_state_for_repo("model", repo_id) > 0

        if not (deleted_revisions or cache_purged or partial_purged or state_purged):
            raise HTTPException(status_code = 404, detail = "No revisions found for model")

        return {"status": "deleted", "repo_id": repo_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error deleting cached model %s: %s",
            repo_id,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        raise HTTPException(
            status_code = 500,
            detail = "Failed to delete cached model: "
            + download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )

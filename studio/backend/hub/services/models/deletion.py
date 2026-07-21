# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model deletion."""

from __future__ import annotations

import asyncio
import errno
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.gguf import extract_quant_label, extract_quant_token
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
    _is_mtp_drafter_path,
)

logger = get_logger(__name__)


def _snapshot_blob_reference_counts(repo_dir: Optional[Path]) -> dict[Path, int]:
    """Map each blob's realpath to its live snapshot symlink count, so per-variant deletion never unlinks a blob another revision still references (call after the target variant's own symlinks are removed)."""
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


def _remove_empty_variant_dirs(
    target_repos: list, variant: str
) -> tuple[int, list[str]]:
    """Remove now-empty ``snapshots/<rev>/<quant>/`` folders for *variant* (the
    quant label names the folder); only empty dirs go, so siblings are safe.
    Returns (count removed, removal failures other than a concurrent refill)."""
    variant_key = (extract_quant_token(variant) or variant).lower()
    removed = 0
    failures: list[str] = []
    for target_repo in target_repos:
        repo_path = getattr(target_repo, "repo_path", None)
        if not repo_path:
            continue
        snapshots = Path(repo_path) / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap_dirs = [
                s for s in snapshots.iterdir() if s.is_dir() and not s.is_symlink()
            ]
        except OSError:
            continue
        for snap in snap_dirs:
            try:
                subs = list(snap.iterdir())
            except OSError:
                continue
            for sub in subs:
                try:
                    if sub.is_symlink() or not sub.is_dir():
                        continue
                    folder_quant = extract_quant_token(sub.name)
                    matches = (
                        folder_quant is not None and folder_quant.lower() == variant_key
                    ) or sub.name.lower() == variant.lower()
                    if not matches or any(sub.iterdir()):
                        continue
                except OSError:
                    continue
                try:
                    sub.rmdir()
                    removed += 1
                except OSError as e:
                    # A concurrent download refilling the dir (ENOTEMPTY) is not a
                    # failure; a read-only cache or locked dir is, so surface it.
                    if e.errno != errno.ENOTEMPTY:
                        failures.append(f"{sub.name}: {e}")
    return removed, failures


def _remove_empty_snapshot_dirs(target_repos: list) -> tuple[int, list[str]]:
    removed = 0
    failures: list[str] = []
    for target_repo in target_repos:
        repo_path = getattr(target_repo, "repo_path", None)
        if not repo_path:
            continue
        snapshots = Path(repo_path) / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap_dirs = [
                s for s in snapshots.iterdir() if s.is_dir() and not s.is_symlink()
            ]
        except OSError:
            continue
        for snap in snap_dirs:
            try:
                snap.rmdir()
                removed += 1
            except OSError as e:
                if e.errno != errno.ENOTEMPTY:
                    failures.append(f"{snap.name}: {e}")
    return removed, failures


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
                # Companions: mmproj and the MTP drafter -- downloaded with
                # every variant, so the last variant's delete reclaims them.
                lambda name: _is_gguf_filename(name)
                and (_is_mmproj_filename(name) or _is_mtp_drafter_path(name)),
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
    # Reclaim the empty quant folder so it stops 404ing on delete.
    removed_dirs, dir_failures = _remove_empty_variant_dirs(target_repos, variant)
    removed_snap_dirs, snap_dir_failures = _remove_empty_snapshot_dirs(target_repos)
    removed_dirs += removed_snap_dirs
    dir_failures.extend(snap_dir_failures)
    if dir_failures:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: "
                f"{len(dir_failures)} folder(s) could not be removed "
                "(read-only cache or in use). Try again."
            ),
        )
    if (
        removed_snapshots == 0
        and deleted_blobs == 0
        and incomplete_result.deleted == 0
        and not state_purged
        and removed_dirs == 0
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


def reclaim_replaced_gguf_variant(
    repo_id: str,
    variant: str,
    keep_main_hashes: frozenset[str],
    hf_token: Optional[str] = None,
) -> dict:
    """Prune stale main-GGUF files for a variant after a replacement verified.

    This is intentionally narrower than user-driven delete: it removes only
    same-variant main files whose local blob hash is not in *keep_main_hashes*,
    then unlinks their blobs only if no remaining snapshot references them.
    Shared companions and sibling variants are left intact.
    """
    if not keep_main_hashes:
        logger.info(
            "Skipping stale GGUF reclaim for %s [%s]: current main hashes unresolved",
            repo_id,
            variant,
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "unresolved_hashes",
        }
    if not _is_valid_repo_id(repo_id) or not _is_valid_gguf_variant(variant):
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "invalid_target",
        }

    failures: list[str] = []
    removed_snapshots = 0
    deleted_blobs = 0
    deleted_bytes = 0
    variant_key = variant.lower()

    try:
        cache_scans = cache_inventory.all_hf_cache_scans()
    except Exception as e:
        logger.warning(
            "Skipping stale GGUF reclaim for %s [%s]: cache scan failed: %s",
            repo_id,
            variant,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "scan_failed",
        }

    candidate_repos = [
        repo_info
        for hf_cache in cache_scans
        for repo_info in hf_cache.repos
        if str(getattr(repo_info, "repo_type", "")) == "model"
        and str(getattr(repo_info, "repo_id", "")).lower() == repo_id.lower()
    ]
    try:
        matched_repo_ids = resolve_destructive_repo_ids(
            repo_id,
            [str(getattr(repo_info, "repo_id", "")) for repo_info in candidate_repos],
            noun = "models",
        )
    except HTTPException as e:
        detail = getattr(e, "detail", str(e))
        logger.warning(
            "Skipping stale GGUF reclaim for %s [%s]: %s",
            repo_id,
            variant,
            download_registry.scrub_secrets(str(detail), hf_token = hf_token),
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "ambiguous_repo",
        }
    target_repos = [
        repo_info
        for repo_info in candidate_repos
        if str(getattr(repo_info, "repo_id", "")) in matched_repo_ids
    ]

    for target_repo in target_repos:
        repo_dir = (
            Path(target_repo.repo_path)
            if getattr(target_repo, "repo_path", None)
            else None
        )
        stale_matches: list[tuple[Path, Optional[Path], str]] = []
        matches = _repo_file_matches(
            target_repo,
            lambda name: _is_main_gguf_filename(name)
            and extract_quant_label(name).lower() == variant_key,
        )
        for snap, blob, name in matches:
            # Prune only a file we can identify as a real, stale cache blob. A
            # no-symlink snapshot file has no identifiable blob hash, so keep it.
            blob_hash = (
                _blob_hash_from_path(blob)
                if cache_inventory._is_real_cache_blob(blob, repo_dir)
                else None
            )
            if blob_hash is None or blob_hash in keep_main_hashes:
                continue
            stale_matches.append((snap, blob, name))

        if not stale_matches:
            continue

        for snap, _blob, name in stale_matches:
            try:
                if _path_exists_or_symlink(snap):
                    snap.unlink()
                    removed_snapshots += 1
            except OSError as e:
                failures.append(f"{name}: {e}")

        ref_counts = _snapshot_blob_reference_counts(repo_dir)
        seen_blobs: set[Path] = set()
        for _snap, blob, name in stale_matches:
            if blob is None:
                continue
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

    removed_dirs = 0
    dir_failures: list[str] = []
    if target_repos:
        removed_dirs, dir_failures = _remove_empty_variant_dirs(target_repos, variant)
        removed_snap_dirs, snap_dir_failures = _remove_empty_snapshot_dirs(target_repos)
        removed_dirs += removed_snap_dirs
        dir_failures.extend(snap_dir_failures)
        failures.extend(dir_failures)

    if failures:
        logger.warning(
            "Stale GGUF reclaim for %s [%s] left %d failure(s): %s",
            repo_id,
            variant,
            len(failures),
            "; ".join(failures[:3]),
        )

    if removed_snapshots or deleted_blobs or removed_dirs:
        cache_inventory.invalidate_hf_cache_scans()
        logger.info(
            "Reclaimed stale GGUF %s [%s]: snapshots=%d blobs=%d dirs=%d freed=%.1f MB",
            repo_id,
            variant,
            removed_snapshots,
            deleted_blobs,
            removed_dirs,
            deleted_bytes / (1024 * 1024),
        )

    return {
        "status": "reclaimed",
        "repo_id": repo_id,
        "variant": variant,
        "removed_snapshots": removed_snapshots,
        "deleted_blobs": deleted_blobs,
        "removed_dirs": removed_dirs,
    }


def _loaded_id_matches_repo(loaded_id: str, repo_id: str) -> bool:
    """True when *loaded_id* is *repo_id* or a file within it; ``/``-boundary aware so ``org/model`` doesn't match sibling ``org/model-v2``."""
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
    """Whether the llama.cpp backend holds *repo_id* (/variant). Acquiring fails open (import error means nothing loaded); reading load state is unguarded so a raise propagates and the caller fails closed rather than delete a live model."""
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
    """Whether the subprocess inference backend holds *repo_id*; same fail-open-on-acquire / surface-on-query contract as :func:`_llama_cpp_blocks_delete`."""
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
    repo_id: str,
    variant: Optional[str] = None,
    hf_token: Optional[str] = None,
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

    # Guard fails closed: if a live backend's load state can't be read, abort
    # with 503 rather than risk unlinking weights under a running process.
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

    repo_key = await asyncio.to_thread(
        resolve_cached_repo_id_case, repo_id, repo_type = "model"
    )
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
        # If a sibling quant is downloading concurrently, restrict this delete to
        # the variant's own files and leave the shared mmproj companion for it.
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

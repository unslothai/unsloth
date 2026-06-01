# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared snapshot download-progress computation for models and datasets.

Both repo types download a full HF snapshot and report progress the same way:
scan the cache's ``blobs/`` dir, split finalized vs ``.incomplete`` bytes,
filter to the target revision's expected blob hashes, and divide by the
revision's total size. The only thing that differs is how the
``(expected_total, expected_hashes)`` metadata is resolved, so callers inject
that as ``metadata_resolver``. Keeping the scan/accounting in one place stops
the two copies from drifting (a prior hash-filter fix had landed only on the
model copy, leaving datasets summing stale blobs against the wrong total)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.state_dir import RepoType
from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    blob_bytes_present,
    latest_snapshot_dir,
    preferred_repo_cache_dirs,
)
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id

logger = get_logger(__name__)

# (repo_id, hf_token) -> (expected_total_bytes, expected_blob_hashes)
SnapshotMetadataResolver = Callable[[str, Optional[str]], "tuple[int, frozenset[str]]"]


def _empty_progress(expected_bytes: int) -> dict:
    return {
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": max(expected_bytes, 0),
        "progress": 0,
        "cache_path": None,
    }


def _snapshot_complete_on_disk(
    *,
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    entry: Path,
    expected_total: int,
    completed_bytes: int,
    in_progress_bytes: int,
) -> bool:
    if expected_total <= 0 or completed_bytes < expected_total or in_progress_bytes > 0:
        return False
    snapshot_dir = latest_snapshot_dir(entry)
    if snapshot_dir is None:
        return False
    if variant is None and hf_cache_scan.repo_cache_dir_has_incomplete_blobs(entry):
        return False
    if download_manifest.has_cancel_marker(repo_type, repo_id, variant):
        return False
    manifest = download_manifest.read_manifest(repo_type, repo_id, variant)
    if manifest is None:
        return False
    return download_manifest.verify_against_disk(manifest, snapshot_dir).ok


def compute_snapshot_progress(
    *,
    repo_type: RepoType,
    repo_id: str,
    job_key: str,
    expected_bytes: int,
    hf_token: Optional[str],
    registry,
    metadata_resolver: SnapshotMetadataResolver,
    variant: Optional[str] = None,
    use_metadata_total_max: bool = False,
) -> dict:
    """Synchronous progress reading. Safe to run under ``asyncio.to_thread``."""
    empty = _empty_progress(expected_bytes)
    if not _is_valid_repo_id(repo_id):
        return empty

    job_state = registry.get_job(job_key).state
    force_active = job_state in {"running", "cancelling"}
    get_job_metadata = getattr(registry, "get_job_metadata", None)
    metadata = get_job_metadata(job_key) if callable(get_job_metadata) else None
    completed_baseline_bytes = max(
        0,
        int(getattr(metadata, "completed_baseline_bytes", 0) or 0),
    )

    expected_total = max(expected_bytes, 0)
    # Always resolve the revision's blob hashes (cached) so stale blobs from a
    # superseded revision can't inflate the count even when the caller supplied
    # a total. Hashes degrade to empty (count-all) only when metadata is
    # unavailable, e.g. offline. ``expected_total`` falls back to the resolved
    # total only when the caller didn't supply a trustworthy one.
    meta_total, expected_hashes = metadata_resolver(repo_id, hf_token)
    if expected_total <= 0 or use_metadata_total_max:
        expected_total = max(expected_total, meta_total)

    # When hashes aren't resolved yet, a variant must not count finalized blobs
    # unscoped: sibling quants share one blobs/ dir and would inflate a fresh
    # variant. In-progress (.incomplete) blobs are always counted (own writes).
    count_finalized_unscoped = variant is None

    readings: list[tuple[int, int, Optional[str], bool]] = []
    for entry in preferred_repo_cache_dirs(
        repo_type,
        repo_id,
        force_active = force_active,
    ):
        completed_bytes = 0
        in_progress_bytes = 0
        cache_path = hf_cache_scan.resolve_hf_cache_realpath(entry)
        blobs_dir = entry / "blobs"
        if blobs_dir.is_dir():
            try:
                blob_entries = list(blobs_dir.iterdir())
            except OSError:
                blob_entries = []
            for f in blob_entries:
                # Skip a blob that vanished mid-poll (renamed out of
                # *.incomplete) rather than zeroing the whole reading.
                try:
                    if not f.is_file():
                        continue
                    if f.name.endswith(INCOMPLETE_SUFFIX):
                        blob_hash = f.name[: -len(INCOMPLETE_SUFFIX)]
                        if expected_hashes and blob_hash not in expected_hashes:
                            continue
                        in_progress_bytes += blob_bytes_present(f)
                    else:
                        if expected_hashes:
                            if f.name not in expected_hashes:
                                continue
                        elif not count_finalized_unscoped:
                            continue
                        completed_bytes += f.stat().st_size
                except OSError:
                    continue
        readings.append(
            (
                completed_bytes,
                in_progress_bytes,
                cache_path,
                _snapshot_complete_on_disk(
                    repo_type = repo_type,
                    repo_id = repo_id,
                    variant = variant,
                    entry = entry,
                    expected_total = expected_total,
                    completed_bytes = completed_bytes,
                    in_progress_bytes = in_progress_bytes,
                ),
            )
        )

    selected = max(
        readings,
        key = lambda item: (item[0] + item[1], item[0]),
        default = None,
    )
    if selected is None:
        return empty

    completed_bytes, in_progress_bytes, cache_path, complete_on_disk = selected
    downloaded_bytes = completed_bytes + in_progress_bytes
    # While the variant is still downloading, the baseline hides companion bytes
    # that were already on disk before this job started. Once it is verified
    # complete, report the full figures instead: a baseline equal to the total
    # would zero out expected/completed/downloaded and make the frontend read a
    # finished variant as 0-byte and evict it as gone.
    effective_baseline_bytes = (
        0 if complete_on_disk else min(completed_baseline_bytes, completed_bytes)
    )
    display_completed_bytes = max(0, completed_bytes - effective_baseline_bytes)
    display_downloaded_bytes = max(0, downloaded_bytes - effective_baseline_bytes)

    if expected_total <= 0:
        # Cannot determine total; report bytes only, no percentage.
        return {
            "downloaded_bytes": display_downloaded_bytes,
            "completed_bytes": display_completed_bytes,
            "complete_on_disk": False,
            "expected_bytes": 0,
            "progress": 0,
            "cache_path": cache_path,
        }

    display_expected_total = max(0, expected_total - effective_baseline_bytes)
    if downloaded_bytes == 0:
        return {
            **empty,
            "expected_bytes": display_expected_total,
            "cache_path": cache_path,
        }

    # Cap at 0.99 during active polling unless the manifest-backed disk check
    # has verified completion. A byte ratio alone is not enough: on resume of a
    # near-complete download, completed bytes can already sit above the
    # threshold while remaining files are still downloading.
    progress = (
        1.0
        if complete_on_disk
        else (
            min(display_downloaded_bytes / display_expected_total, 0.99)
            if display_expected_total > 0
            else 0
        )
    )
    return {
        "downloaded_bytes": display_downloaded_bytes,
        "completed_bytes": display_completed_bytes,
        "complete_on_disk": complete_on_disk,
        "expected_bytes": display_expected_total,
        "progress": round(progress, 3),
        "cache_path": cache_path,
    }


async def snapshot_progress_response(
    *,
    repo_type: RepoType,
    repo_id: str,
    job_key: str,
    expected_bytes: int,
    hf_token: Optional[str],
    registry,
    metadata_resolver: SnapshotMetadataResolver,
    variant: Optional[str] = None,
    use_metadata_total_max: bool = False,
) -> dict:
    """Async wrapper: offloads the blocking cache walk and never raises."""
    try:
        return await asyncio.to_thread(
            compute_snapshot_progress,
            repo_type = repo_type,
            repo_id = repo_id,
            job_key = job_key,
            expected_bytes = expected_bytes,
            hf_token = hf_token,
            registry = registry,
            metadata_resolver = metadata_resolver,
            variant = variant,
            use_metadata_total_max = use_metadata_total_max,
        )
    except Exception as e:
        logger.warning(
            "Error checking %s download progress for %s: %s: %s",
            repo_type,
            repo_id,
            type(e).__name__,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        return _empty_progress(expected_bytes)

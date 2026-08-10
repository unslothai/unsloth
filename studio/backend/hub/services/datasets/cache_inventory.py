# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached dataset inventory and deletion services."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.services import resolve_destructive_repo_ids
from hub.services.datasets import downloads
from hub.utils import download_manifest
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.dataset_cache import (
    hf_datasets_cache_roots,
    processed_dataset_cache_has_artifacts,
)
from hub.utils.dataset_processed_cache import (
    app_processed_dataset_cache_from_path,
    delete_app_processed_dataset_caches,
    iter_app_processed_dataset_caches,
)
from hub.utils.hf_cache_state import (
    purge_partial_repo,
    purge_repo_cache_dirs,
    resolve_delete_target_root,
    resolve_destructive_case_matches,
)
from hub.utils.paths import (
    is_valid_repo_id as _is_valid_repo_id,
    resolve_cached_repo_id_case,
)

logger = get_logger(__name__)


def _collect_hf_cache_scans() -> tuple[list, set[str]]:
    scans = hf_cache_scan.all_hf_cache_scans()
    seen_roots = {
        str(cache_dir)
        for cache_dir in (getattr(scan, "cache_dir", None) for scan in scans)
        if cache_dir is not None
    }
    return scans, seen_roots


def _hf_hub_cache_roots() -> list[Path]:
    from hub.utils.hf_cache_state import hf_cache_roots
    return hf_cache_roots()


def _repo_id_from_hub_dataset_dir(name: str) -> str | None:
    if not name.startswith("datasets--"):
        return None
    encoded = name.removeprefix("datasets--")
    owner, sep, repo = encoded.partition("--")
    if not sep or not owner or not repo:
        return None
    repo_id = f"{owner}/{repo}"
    return repo_id if _is_valid_repo_id(repo_id) else None


def _directory_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _prefer_dataset_cache_row(candidate: dict, existing: Optional[dict]) -> bool:
    if existing is None:
        return True
    candidate_partial = bool(candidate.get("partial"))
    existing_partial = bool(existing.get("partial"))
    if candidate_partial != existing_partial:
        return not candidate_partial
    return int(candidate.get("size_bytes") or 0) > int(existing.get("size_bytes") or 0)


def _raw_row_hub_cache(row: dict) -> Optional[Path]:
    cache_path = row.get("cache_path")
    if not isinstance(cache_path, str):
        return None
    try:
        path = Path(cache_path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return None
    return path.parent if path.name.lower().startswith("datasets--") else None


def _hub_dataset_snapshot_count(path: Path) -> int:
    snapshots = path / "snapshots"
    try:
        return sum(1 for entry in snapshots.iterdir() if entry.is_dir())
    except OSError:
        return 0


def _scan_hub_dataset_cache_dirs() -> list[dict]:
    """Fallback scanner: ``scan_cache_dir()`` skips repos when one cache entry is partially corrupt, so this keeps On Device matching disk."""
    seen_lower: dict[str, dict] = {}
    for root in _hf_hub_cache_roots():
        try:
            entries = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            repo_id = _repo_id_from_hub_dataset_dir(entry.name)
            if repo_id is None:
                continue
            size_bytes = _directory_size(entry / "blobs")
            if size_bytes <= 0:
                size_bytes = _directory_size(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            snapshot_partial = _hub_dataset_snapshot_count(
                entry
            ) == 0 or hf_cache_scan.is_snapshot_partial("dataset", repo_id, entry)
            row = {
                "repo_id": repo_id,
                "size_bytes": size_bytes,
                "cache_path": str(entry.resolve()),
                # snapshot_count == 0 catches blobs-but-no-snapshot; is_snapshot_partial adds row-state checks.
                "partial": snapshot_partial,
                "partial_transport": (
                    hf_cache_scan.partial_transport_for(
                        "dataset",
                        repo_id,
                        repo_cache_dir = entry,
                    )
                    if snapshot_partial
                    else None
                ),
            }
            if _prefer_dataset_cache_row(row, existing):
                seen_lower[key] = row
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _hf_datasets_cache_roots() -> list[Path]:
    return hf_datasets_cache_roots()


def _repo_id_from_datasets_cache_dir(name: str) -> str | None:
    if "___" not in name:
        return None
    owner, repo = name.split("___", 1)
    repo_id = f"{owner}/{repo}"
    return repo_id if _is_valid_repo_id(repo_id) else None


def _is_processed_dataset_cache_path(repo_id: str, cache_path: str) -> bool:
    """True when *cache_path* is this repo's processed Arrow cache dir
    (``<owner>___<repo>`` directly under an HF_DATASETS_CACHE root). Such rows
    have no Hub ``datasets--`` layout, so they are deleted via the processed
    path and must not be rejected as an invalid cache_path."""
    try:
        resolved = Path(cache_path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return False
    if resolved.name.lower() != repo_id.replace("/", "___").lower():
        return False
    roots = {r.resolve(strict = False) for r in _hf_datasets_cache_roots()}
    return resolved.parent.resolve(strict = False) in roots


def _processed_dataset_cache_size(path: Path) -> int:
    total = 0
    try:
        for directory, dirnames, filenames in os.walk(path, followlinks = False):
            base = Path(directory)
            dirnames[:] = [name for name in dirnames if not (base / name).is_symlink()]
            for filename in filenames:
                entry = base / filename
                try:
                    if entry.is_file() and not entry.is_symlink():
                        total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _scan_processed_dataset_caches() -> list[dict]:
    """`load_dataset()` stores processed Arrow caches separately from the Hub snapshot cache, so they're usable on-device but invisible to `scan_cache_dir()`."""
    seen_lower: dict[str, dict] = {}
    for root in _hf_datasets_cache_roots():
        try:
            entries = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            repo_id = _repo_id_from_datasets_cache_dir(entry.name)
            if repo_id is None:
                continue
            if not processed_dataset_cache_has_artifacts(entry):
                continue
            size_bytes = _processed_dataset_cache_size(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            if existing is None or size_bytes > existing["size_bytes"]:
                seen_lower[key] = {
                    "repo_id": repo_id,
                    "size_bytes": size_bytes,
                    "cache_path": str(entry.resolve()),
                    "processed_cache": True,
                    "partial": False,
                }
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _scan_app_processed_dataset_caches() -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for entry in iter_app_processed_dataset_caches():
        size_bytes = _processed_dataset_cache_size(entry.path)
        key = (
            entry.repo_id.casefold(),
            os.path.normcase(str(entry.hub_cache)),
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "repo_id": entry.repo_id,
                "size_bytes": size_bytes,
                "cache_path": str(entry.path),
                "processed_cache": True,
                "app_processed_cache": True,
                "app_processed_hub_cache": str(entry.hub_cache),
                "partial": True,
            }
        else:
            existing["size_bytes"] += size_bytes
    return sorted(
        grouped.values(),
        key = lambda row: (row["repo_id"].casefold(), row["app_processed_hub_cache"]),
    )


def _scan_hf_dataset_caches() -> list[dict]:
    scans, seen_roots = _collect_hf_cache_scans()

    seen_lower: dict[str, dict] = {}
    inspected = 0
    for hf_cache in scans:
        for repo_info in hf_cache.repos:
            inspected += 1
            try:
                # str(...) guards against the library switching repo_type to an Enum.
                if str(repo_info.repo_type) != "dataset":
                    continue
                total_size = int(getattr(repo_info, "size_on_disk", 0) or 0)
                if total_size == 0:
                    unique_blobs: dict[str, int] = {}
                    for rev in repo_info.revisions:
                        rev_id = getattr(rev, "commit_hash", None) or str(id(rev))
                        for f in rev.files:
                            blob_path = getattr(f, "blob_path", None)
                            key = str(blob_path) if blob_path else f"{rev_id}:{f.file_name}"
                            unique_blobs[key] = int(f.size_on_disk or 0)
                    total_size = sum(unique_blobs.values())
                key = repo_info.repo_id.lower()
                existing = seen_lower.get(key)
                cache_dir = Path(repo_info.repo_path)
                snapshot_partial = hf_cache_scan.is_snapshot_partial(
                    "dataset",
                    repo_info.repo_id,
                    cache_dir,
                )
                row = {
                    "repo_id": repo_info.repo_id,
                    "size_bytes": total_size,
                    "cache_path": str(repo_info.repo_path),
                    "partial": snapshot_partial,
                    "partial_transport": (
                        hf_cache_scan.partial_transport_for(
                            "dataset",
                            repo_info.repo_id,
                            repo_cache_dir = cache_dir,
                        )
                        if snapshot_partial
                        else None
                    ),
                }
                if _prefer_dataset_cache_row(row, existing):
                    seen_lower[key] = row
            except Exception as exc:
                label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning("Skipping cached dataset repo %s: %s", label, exc)
    for row in _scan_hub_dataset_cache_dirs():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if _prefer_dataset_cache_row(row, existing):
            seen_lower[key] = row
        elif existing is not None and bool(existing.get("partial")) == bool(row.get("partial")):
            existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
            existing["cache_path"] = existing.get("cache_path") or row.get("cache_path")
            if (
                existing.get("partial")
                and not existing.get("partial_transport")
                and row.get("partial_transport")
            ):
                existing["partial_transport"] = row["partial_transport"]
    for row in _scan_processed_dataset_caches():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if existing is None or (bool(existing.get("partial")) and not bool(row.get("partial"))):
            seen_lower[key] = row
        else:
            existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
            # Preserve the raw path for scoped deletion and expose the processed Arrow path separately.
            if row.get("processed_cache"):
                existing["processed_cache"] = True
                existing["load_cache_path"] = row.get("cache_path")
    for row in _scan_app_processed_dataset_caches():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if existing is None:
            seen_lower[key] = row
            continue
        raw_hub_cache = _raw_row_hub_cache(existing)
        try:
            app_hub_cache = Path(row["app_processed_hub_cache"]).resolve(strict = False)
        except (OSError, RuntimeError, ValueError):
            app_hub_cache = None
        if raw_hub_cache is not None and raw_hub_cache == app_hub_cache:
            existing["size_bytes"] = int(existing.get("size_bytes") or 0) + int(
                row.get("size_bytes") or 0
            )
            existing["processed_cache"] = True
            existing["app_processed_cache"] = True
    logger.info(
        "Cached dataset scan: roots=%d inspected=%d returned=%d",
        len(seen_roots) or len(scans),
        inspected,
        len(seen_lower),
    )
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


async def list_cached_datasets_response() -> dict:
    """List dataset repos already downloaded into the HF cache."""
    try:
        return {"cached": await asyncio.to_thread(_scan_hf_dataset_caches)}
    except Exception as exc:
        logger.error("Error listing cached datasets: %s", exc, exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to read the local dataset cache.",
        ) from exc


async def delete_cached_dataset_response(repo_id: str, cache_path: Optional[str] = None) -> dict:
    """Remove a cached dataset repo from the HF cache."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")

    repo_key = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "dataset")
    if not downloads.registry.begin_delete(repo_key):
        raise HTTPException(
            status_code = 400,
            detail = "Cancel the active download before deleting.",
        )
    try:
        return await asyncio.to_thread(_delete_cached_dataset_blocking, repo_key, cache_path)
    finally:
        downloads.registry.end_delete(repo_key)
        hf_cache_scan.invalidate_hf_cache_scans()


def _delete_cached_dataset_blocking(repo_id: str, cache_path: Optional[str] = None) -> dict:
    scans, _seen_roots = _collect_hf_cache_scans()
    app_entry = app_processed_dataset_cache_from_path(repo_id, cache_path) if cache_path else None

    # Group this dataset's copies by owning cache root, then target exactly one cache.
    owners: dict = {}
    for hf_cache in scans:
        for repo_info in hf_cache.repos:
            if str(repo_info.repo_type) != "dataset":
                continue
            if repo_info.repo_id.lower() != repo_id.lower():
                continue
            try:
                owner = Path(repo_info.repo_path).parent.resolve(strict = False)
            except (OSError, RuntimeError, ValueError):
                continue
            owners.setdefault(owner, []).append((hf_cache, repo_info))

    target_root = resolve_delete_target_root("dataset", repo_id, cache_path, owners.keys())
    # A processed-only dataset row sends its Arrow cache path, which is not a Hub datasets-- dir, so
    # resolve_delete_target_root returns None. Accept it and fall through to the processed delete.
    if target_root is None and not (
        cache_path
        and (_is_processed_dataset_cache_path(repo_id, cache_path) or app_entry is not None)
    ):
        raise HTTPException(status_code = 400, detail = "Invalid cache_path")
    candidate_entries = owners.get(target_root, []) if target_root is not None else []
    matched_repo_ids = resolve_destructive_repo_ids(
        repo_id,
        [str(repo_info.repo_id) for _hf_cache, repo_info in candidate_entries],
        noun = "datasets",
    )

    deleted = False
    failures: list[str] = []
    for hf_cache, repo_info in candidate_entries:
        if str(repo_info.repo_id) not in matched_repo_ids:
            continue
        try:
            strategy = hf_cache.delete_revisions(*(rev.commit_hash for rev in repo_info.revisions))
            strategy.execute()
            deleted = True
        except Exception as exc:
            failures.append(str(exc))
            logger.error(
                "Failed deleting cached dataset %s from %s: %s",
                repo_id,
                getattr(hf_cache, "cache_dir", "<unknown>"),
                exc,
                exc_info = True,
            )

    # Restrict the processed Arrow-cache delete to the selected cache's datasets root. A processed
    # cache_path scopes to its own root; a Hub target to the datasets root sharing its cache home;
    # an unspecified cache_path stays global (legacy).
    processed_roots: Optional[set[Path]]
    if not cache_path:
        processed_roots = None
    elif _is_processed_dataset_cache_path(repo_id, cache_path):
        processed_roots = {Path(cache_path).expanduser().resolve(strict = False).parent}
    else:
        home = target_root.parent if target_root is not None else None
        processed_roots = {
            root.resolve(strict = False)
            for root in _hf_datasets_cache_roots()
            if home is not None and root.resolve(strict = False).parent == home
        }

    processed_deleted, processed_failures = _delete_processed_dataset_cache(
        repo_id, only_roots = processed_roots
    )
    failures.extend(processed_failures)
    delete_app_cache = not cache_path or app_entry is not None or target_root is not None
    app_hub_cache = app_entry.hub_cache if app_entry is not None else target_root
    app_deleted, app_failures = (
        _delete_app_processed_dataset_cache(
            repo_id,
            hub_cache = app_hub_cache,
        )
        if delete_app_cache
        else (False, [])
    )
    failures.extend(app_failures)
    if failures:
        raise HTTPException(
            status_code = 500,
            detail = (
                f"Failed to delete dataset from {len(failures)} cache "
                "location(s). Some files may remain."
            ),
        )

    # ``scan_cache_dir()`` skips blob-only/corrupt repos the revision delete can't touch, yet the
    # fallback scanner shows them, so purge the whole dir. Hub cache targets only.
    cache_purged = partial_purged = state_purged = False
    if target_root is not None:
        cache_purged = purge_repo_cache_dirs("dataset", repo_id, root = target_root)
        partial_purged = purge_partial_repo("dataset", repo_id, root = target_root)
        state_purged = (
            download_manifest.purge_all_state_for_repo("dataset", repo_id, hub_cache = target_root)
            > 0
        )
    if not (
        deleted
        or processed_deleted
        or app_deleted
        or cache_purged
        or partial_purged
        or state_purged
    ):
        raise HTTPException(status_code = 404, detail = "Dataset not found in cache")
    return {"status": "deleted", "repo_id": repo_id}


def _delete_processed_dataset_cache(
    repo_id: str, only_roots: Optional[set[Path]] = None
) -> tuple[bool, list[str]]:
    import shutil

    target = repo_id.replace("/", "___")
    folded_target = target.lower()
    deleted = False
    failures: list[str] = []
    for root in _hf_datasets_cache_roots():
        # Scope to the selected cache's datasets root(s) so copies under other cache homes survive.
        if only_roots is not None and root.resolve(strict = False) not in only_roots:
            continue
        try:
            entries = [
                entry
                for entry in root.iterdir()
                if entry.is_dir() and entry.name.lower() == folded_target
            ]
        except OSError:
            continue
        matched_names = resolve_destructive_case_matches(
            target,
            (entry.name for entry in entries),
        )
        if not matched_names:
            continue
        for entry in entries:
            if entry.name not in matched_names:
                continue
            try:
                shutil.rmtree(entry)
                deleted = True
            except Exception as exc:
                failures.append(str(exc))
                logger.error(
                    "Failed deleting processed dataset cache %s: %s",
                    repo_id,
                    exc,
                    exc_info = True,
                )
    return deleted, failures


def _delete_app_processed_dataset_cache(
    repo_id: str, *, hub_cache: Optional[Path] = None
) -> tuple[bool, list[str]]:
    deleted, failures = delete_app_processed_dataset_caches(
        repo_id,
        hub_cache = hub_cache,
    )
    for failure in failures:
        logger.error(
            "Failed deleting processed dataset cache %s: %s",
            repo_id,
            failure,
        )
    return deleted, failures

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
                # snapshot_count == 0 catches blobs-but-no-snapshot;
                # is_snapshot_partial adds active-row state checks.
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
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not path.is_dir():
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(path)

    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        _add(Path(env_cache).expanduser())

    try:
        from datasets import config as datasets_config
        _add(Path(datasets_config.HF_DATASETS_CACHE))
    except Exception:
        pass

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        _add(Path(hf_home).expanduser() / "datasets")

    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    _add(xdg_cache / "huggingface" / "datasets")
    return roots


def _repo_id_from_datasets_cache_dir(name: str) -> str | None:
    if "___" not in name:
        return None
    owner, repo = name.split("___", 1)
    repo_id = f"{owner}/{repo}"
    return repo_id if _is_valid_repo_id(repo_id) else None


def _processed_dataset_cache_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _looks_like_processed_dataset_cache(path: Path) -> bool:
    try:
        for entry in path.rglob("*"):
            if not entry.is_file():
                continue
            if entry.name in {"dataset_info.json", "state.json"}:
                return True
            if entry.suffix == ".arrow":
                return True
    except OSError:
        return False
    return False


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
            if not _looks_like_processed_dataset_cache(entry):
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
            # Keep the processed-cache marker when a repo is both snapshot and
            # processed Arrow cache; merging by size alone dropped it.
            if row.get("processed_cache"):
                existing["processed_cache"] = True
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

    # Group this dataset's copies by owning cache root, then target exactly one
    # cache so a delete never removes copies in other, previously selected caches.
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
    if target_root is None:
        raise HTTPException(status_code = 400, detail = "Invalid cache_path")
    candidate_entries = owners.get(target_root, [])
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

    processed_deleted, processed_failures = _delete_processed_dataset_cache(repo_id)
    failures.extend(processed_failures)
    if failures:
        raise HTTPException(
            status_code = 500,
            detail = (
                f"Failed to delete dataset from {len(failures)} cache "
                "location(s). Some files may remain."
            ),
        )

    # ``scan_cache_dir()`` skips blob-only/corrupt repos the revision delete
    # can't touch, yet the fallback scanner shows them; purge the whole dir.
    cache_purged = purge_repo_cache_dirs("dataset", repo_id, root = target_root)
    partial_purged = purge_partial_repo("dataset", repo_id, root = target_root)
    state_purged = (
        download_manifest.purge_all_state_for_repo("dataset", repo_id, hub_cache = target_root) > 0
    )
    if not (deleted or processed_deleted or cache_purged or partial_purged or state_purged):
        raise HTTPException(status_code = 404, detail = "Dataset not found in cache")
    return {"status": "deleted", "repo_id": repo_id}


def _delete_processed_dataset_cache(repo_id: str) -> tuple[bool, list[str]]:
    import shutil

    target = repo_id.replace("/", "___")
    folded_target = target.lower()
    deleted = False
    failures: list[str] = []
    for root in _hf_datasets_cache_roots():
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

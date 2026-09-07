# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached dataset inventory and deletion services."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.services import resolve_destructive_repo_ids
from hub.services.datasets import downloads, local_options
from hub.utils import download_manifest
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.dataset_cache import (
    dataset_snapshot_from_cache_path,
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


def _directory_stats(path: Path) -> tuple[int, float]:
    total = 0
    last_modified = 0.0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    stat = entry.stat()
                    total += stat.st_size
                    last_modified = max(last_modified, _usable_mtime(stat.st_mtime))
            except OSError:
                continue
    except OSError:
        return 0, 0.0
    return total, last_modified


def _usable_mtime(value) -> float:
    """a timestamp we are willing to publish, else 0.0 meaning "unknown".

    Finiteness is not paranoia about stat(): ``candidate`` is whatever
    huggingface_hub put on the object, and Starlette encodes with
    ``allow_nan = False``, so one inf 500s the whole response.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0 else 0.0


def _safe_mtime(path: Path) -> float:
    """directory mtime as POSIX seconds, or 0.0 when unreadable.

    mtime only, so it is portable across Windows, macOS and Linux. A broken
    symlink or a share with no clock lands on 0.0, which callers drop.
    """
    try:
        return _usable_mtime(path.stat().st_mtime)
    except OSError:
        return 0.0


def _dataset_last_modified(candidate, *paths: Optional[Path]) -> float:
    """newest change time for a cached dataset row, as POSIX seconds.

    Prefers huggingface_hub's own value, else stat()s the cache dirs. Same unit
    as the cached-model scan.
    """
    newest = _usable_mtime(candidate)
    for path in paths:
        if path is None:
            continue
        newest = max(newest, _safe_mtime(path))
    return newest


def _merge_last_modified(existing: dict, row: dict) -> None:
    """keep the newer timestamp when two scans describe one dataset."""
    newest = max(
        _usable_mtime(existing.get("last_modified")),
        _usable_mtime(row.get("last_modified")),
    )
    if newest > 0:
        existing["last_modified"] = newest


def _adopt_newer_last_modified(winner: dict, loser: Optional[dict]) -> None:
    """carry a discarded row's timestamp onto the row that replaces it.

    Winning is decided on completeness then size, neither of which is recency,
    so a bigger-but-older copy would otherwise bury the newer date and sink the
    dataset in Recent. The cached-model scan keeps the max the same way.
    """
    if loser is None:
        return
    newest = max(
        _usable_mtime(winner.get("last_modified")),
        _usable_mtime(loser.get("last_modified")),
    )
    if newest > 0:
        winner["last_modified"] = newest


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


# anything not named here counts as payload, so unknown formats are never read as an empty
# snapshot. the windows names are spelled out because huggingface_hub only added them after 0.36;
# the data-file names come from local_options so this and the resolver cannot drift apart.
_DATASET_NON_PAYLOAD_FILENAMES = (
    frozenset(
        {
            ".ds_store",
            ".gitattributes",
            ".huggingface.yaml",
            "desktop.ini",
            "license",
            "license.md",
            "license.txt",
            "thumbs.db",
        }
    )
    | {name.lower() for name in hf_cache_scan._CACHE_ENTRIES_TO_IGNORE}
    | {name.lower() for name in local_options._IGNORED_DATA_FILENAMES}
)


# suffixes no loader can turn into rows; a script also needs trust_remote_code, which no load path
# here passes and datasets>=4 dropped.
_DATASET_NON_PAYLOAD_SUFFIXES = frozenset({".cff", ".md", ".py", ".pyc"})


def _is_payload_dir(name: str) -> bool:
    # the only rule datasets applies to a directory, which also covers a Mac zip's __MACOSX. the
    # metadata FILE names must not be applied here: license/train.parquet loads fine, and pruning that
    # subtree hid the dataset from On Device.
    return not name.startswith(".") and not name.startswith("__")


def _is_payload_name(name: str) -> bool:
    # as above, plus the metadata names: AppleDouble sidecars, every dotfile the list does not
    # enumerate, and the cards a cancelled download leaves behind.
    if not _is_payload_dir(name):
        return False
    return name.lower() not in _DATASET_NON_PAYLOAD_FILENAMES


def _is_payload_file(name: str) -> bool:
    if not _is_payload_name(name):
        return False
    # the resolver's suffix rule drops a trailing compression suffix: train.parquet.backup is still
    # parquet, data.py.gz is still a script.
    suffix = local_options._data_suffix(name)
    return suffix is None or suffix.lower() not in _DATASET_NON_PAYLOAD_SUFFIXES


def _is_present_payload_file(path: Path) -> bool:
    # existence is not enough: a zero-byte file, and a blobs/ link whose blob was pruned, both look like
    # payload and are not.
    if not _is_payload_file(path.name):
        return False
    if local_options._empty_payload(path):
        return False
    # bytes are not rows: a header-only csv or a [] json drops the file, not the snapshot, as datasets does.
    module = local_options._file_module(path.name)
    return module is None or not local_options._rowless(path, path.name, module)


def _snapshot_holds_payload(snapshot: Path) -> Optional[bool]:
    """True/False for this snapshot, or None when a subtree could not be read.

    None is not False: `os.walk` swallows `scandir` errors unless `onerror` is given, so a cache
    on an unavailable mount would read as empty and hide a dataset that was merely uninspectable.
    """
    unreadable = False

    def _note(_exc: OSError) -> None:
        nonlocal unreadable
        unreadable = True

    # a junction pointing at its own ancestor resolves back inside the snapshot, so containment alone
    # leaves data/loop/loop/... descending until the path length gives out.
    seen: set[Path] = set()
    pending: list[Path] = []

    def _book(entry: Path) -> Optional[Path]:
        """The resolved path, booked as visited, or None when already seen or unresolvable."""
        nonlocal unreadable
        try:
            resolved = entry.resolve(strict = True)
        except (OSError, RuntimeError, ValueError):
            unreadable = True
            return None
        if resolved in seen:
            return None
        seen.add(resolved)
        return resolved

    start = _book(snapshot)
    if start is None:
        return None
    pending.append(start)

    try:
        while pending:
            root = pending.pop()
            for directory, dirnames, filenames in os.walk(root, followlinks = False, onerror = _note):
                base = Path(directory)
                kept = []
                for name in dirnames:
                    # nothing under a hidden or __-prefixed dir can supply rows, so .hidden/notes.txt must not clear
                    # partial.
                    if not _is_payload_dir(name):
                        continue
                    entry = base / name
                    # containment, not a link-type test: is_symlink() is false for a Windows junction and is_junction()
                    # postdates 3.12, so only comparing resolved paths catches every redirect.
                    try:
                        redirected = not entry.resolve(strict = True).is_relative_to(root)
                        linked = entry.is_symlink()
                    except (OSError, RuntimeError, ValueError):
                        unreadable = True
                        continue
                    # walked as a root of its own, not taken as proof: migrated caches keep their data behind a
                    # redirect, and a stale one holds nothing.
                    if redirected:
                        target = _book(entry)
                        if target is not None:
                            pending.append(target)
                        continue
                    # a symlink back inside this root is skipped, since booking its target would prune the real
                    # directory whenever alias is listed before data; a junction reports False here.
                    if linked:
                        continue
                    if _book(entry) is None:
                        continue
                    kept.append(name)
                dirnames[:] = kept
                if any(_is_present_payload_file(base / name) for name in filenames):
                    return True
    except OSError:
        return None
    return None if unreadable else False


def _raw_dataset_cache_has_data(repo_id: str, cache_path: Path) -> bool:
    """Whether the snapshot a load would actually open holds anything beyond metadata.

    A cancelled download can leave just the card, which every structural check reads as complete,
    so the repo was offered On Device and then failed in load_dataset().

    Only the revision `dataset_snapshot_from_cache_path` selects counts, since that is what
    `training_dataset_cache_pin` pins: a payload-bearing sibling revision would still not be the
    one the run opens.
    """
    snapshot = dataset_snapshot_from_cache_path(str(cache_path), repo_id)
    if snapshot is None:
        return False
    # True, or unreadable -- and an uninspectable cache is not an empty one.
    return _snapshot_holds_payload(snapshot) is not False


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
            size_bytes, payload_mtime = _directory_stats(entry / "blobs")
            if size_bytes <= 0:
                size_bytes, payload_mtime = _directory_stats(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            snapshot_partial = (
                _hub_dataset_snapshot_count(entry) == 0
                or hf_cache_scan.is_snapshot_partial("dataset", repo_id, entry)
                or not _raw_dataset_cache_has_data(repo_id, entry)
            )
            last_modified = _dataset_last_modified(payload_mtime, entry / "snapshots", entry)
            row = {
                "repo_id": repo_id,
                "size_bytes": size_bytes,
                "cache_path": str(entry.resolve()),
                # blobs-but-no-snapshot, then row state, then a card-only snapshot.
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
                "partial_resumable": (
                    hf_cache_scan.partial_resume_available(
                        "dataset",
                        repo_id,
                        repo_cache_dir = entry,
                    )
                    if snapshot_partial
                    else False
                ),
            }
            if last_modified > 0:
                row["last_modified"] = last_modified
            if _prefer_dataset_cache_row(row, existing):
                _adopt_newer_last_modified(row, existing)
                seen_lower[key] = row
            elif existing is not None:
                _merge_last_modified(existing, row)
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


def _processed_dataset_cache_stats(path: Path) -> tuple[int, float]:
    total = 0
    last_modified = 0.0
    try:
        for directory, dirnames, filenames in os.walk(path, followlinks = False):
            base = Path(directory)
            dirnames[:] = [name for name in dirnames if not (base / name).is_symlink()]
            for filename in filenames:
                entry = base / filename
                try:
                    if entry.is_file() and not entry.is_symlink():
                        stat = entry.stat()
                        total += stat.st_size
                        last_modified = max(
                            last_modified,
                            _usable_mtime(stat.st_mtime),
                        )
                except OSError:
                    continue
    except OSError:
        return 0, 0.0
    return total, last_modified


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
            size_bytes, payload_mtime = _processed_dataset_cache_stats(entry)
            if size_bytes <= 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            processed_mtime = _dataset_last_modified(payload_mtime, entry)
            if existing is None or size_bytes > existing["size_bytes"]:
                processed_row = {
                    "repo_id": repo_id,
                    "size_bytes": size_bytes,
                    "cache_path": str(entry.resolve()),
                    "processed_cache": True,
                    "partial": False,
                }
                if processed_mtime > 0:
                    processed_row["last_modified"] = processed_mtime
                _adopt_newer_last_modified(processed_row, existing)
                seen_lower[key] = processed_row
            elif processed_mtime > 0:
                _merge_last_modified(existing, {"last_modified": processed_mtime})
    return sorted(seen_lower.values(), key = lambda c: c["repo_id"])


def _scan_app_processed_dataset_caches() -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for entry in iter_app_processed_dataset_caches():
        size_bytes, payload_mtime = _processed_dataset_cache_stats(entry.path)
        app_mtime = _dataset_last_modified(payload_mtime, entry.path)
        key = (
            entry.repo_id.casefold(),
            os.path.normcase(str(entry.hub_cache)),
        )
        existing = grouped.get(key)
        if existing is None:
            app_row = {
                "repo_id": entry.repo_id,
                "size_bytes": size_bytes,
                "cache_path": str(entry.path),
                "processed_cache": True,
                "app_processed_cache": True,
                "app_processed_hub_cache": str(entry.hub_cache),
                "partial": True,
            }
            if app_mtime > 0:
                app_row["last_modified"] = app_mtime
            grouped[key] = app_row
        else:
            existing["size_bytes"] += size_bytes
            _merge_last_modified(existing, {"last_modified": app_mtime})
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
                ) or not _raw_dataset_cache_has_data(repo_info.repo_id, cache_dir)
                repo_last_modified = _dataset_last_modified(
                    getattr(repo_info, "last_modified", None),
                    cache_dir / "snapshots",
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
                    "partial_resumable": (
                        hf_cache_scan.partial_resume_available(
                            "dataset",
                            repo_info.repo_id,
                            repo_cache_dir = cache_dir,
                        )
                        if snapshot_partial
                        else False
                    ),
                }
                if repo_last_modified > 0:
                    row["last_modified"] = repo_last_modified
                if _prefer_dataset_cache_row(row, existing):
                    _adopt_newer_last_modified(row, existing)
                    seen_lower[key] = row
                elif existing is not None:
                    _merge_last_modified(existing, row)
            except Exception as exc:
                label = getattr(repo_info, "repo_id", "<unknown>")
                logger.warning("Skipping cached dataset repo %s: %s", label, exc)
    for row in _scan_hub_dataset_cache_dirs():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if _prefer_dataset_cache_row(row, existing):
            _adopt_newer_last_modified(row, existing)
            seen_lower[key] = row
        elif existing is not None and bool(existing.get("partial")) == bool(row.get("partial")):
            existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
            _merge_last_modified(existing, row)
            existing["cache_path"] = existing.get("cache_path") or row.get("cache_path")
            if (
                existing.get("partial")
                and not existing.get("partial_transport")
                and row.get("partial_transport")
            ):
                existing["partial_transport"] = row["partial_transport"]
                # The resume verdict belongs to the transport it was measured against.
                existing["partial_resumable"] = bool(row.get("partial_resumable"))
    for row in _scan_processed_dataset_caches():
        key = row["repo_id"].lower()
        existing = seen_lower.get(key)
        if existing is None:
            seen_lower[key] = row
            continue
        existing["size_bytes"] = max(existing["size_bytes"], row["size_bytes"])
        _merge_last_modified(existing, row)
        # Preserve the raw path for scoped deletion and expose the processed Arrow path separately.
        if row.get("processed_cache"):
            existing["processed_cache"] = True
            existing["load_cache_path"] = row.get("cache_path")
        # annotating rather than replacing keeps the hub cache_path that scoped deletion needs.
        if existing.get("partial") and not row.get("partial"):
            existing["partial"] = False
            existing["partial_transport"] = None
            existing["partial_resumable"] = False
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
            _merge_last_modified(existing, row)
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
    # resolve_delete_target_root returns None.
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

    # A processed cache_path scopes to its own root, a Hub target to the datasets root sharing its cache
    # home, and an unspecified one stays global.
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

    # scan_cache_dir() skips blob-only or corrupt repos the revision delete cannot touch, yet the
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

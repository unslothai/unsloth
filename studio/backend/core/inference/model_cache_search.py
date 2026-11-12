# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Finds curated side models outside the active HF cache: the other known caches, then the account's custom folders,
each listed and probed under the Model Hub's per-folder budget (they can sit on slow or dead mounts)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

from loggers import get_logger

logger = get_logger(__name__)


# One status poll runs a dozen lookups; listings are reused briefly, per account (custom folders are per account).
_LOCATION_TTL_SECONDS = 2.0
_location_lock = threading.Lock()
# (row key, path, recursive, kind): kind "cache" is a Hub cache root, "folder" a custom folder.
_Row = tuple[str, Path, bool, str]
_location_rows_memo: dict[str, tuple[float, tuple[_Row, ...]]] = {}
# Keyed by kind too: a known cache that is also a custom folder is listed both ways (repos only, and loose files).
_location_listings_memo: dict[tuple[str, str, str], tuple[float, "LocationListing"]] = {}
_LOCATION_MEMO_MAX = 256
_T = TypeVar("_T")
_C = TypeVar("_C")


@dataclass(frozen = True)
class LocationListing:
    """Lowercased ``models--*`` dir -> Hub caches holding it, and file name -> directories holding it. A Hub cache root
    lists only its repos; a custom folder lists what the Model Hub scans there."""

    hub_caches: dict[str, tuple[Path, ...]]
    file_dirs: dict[str, tuple[Path, ...]]

    def caches_holding(self, repo: str) -> tuple[Path, ...]:
        return self.hub_caches.get(f"models--{repo.replace('/', '--')}".lower(), ())

    def dirs_holding(self, *names: str) -> list[Path]:
        first, *rest = (self.file_dirs.get(name.lower(), ()) for name in names)
        return [directory for directory in first if all(directory in other for other in rest)]


def account_scope() -> str:
    """Memo key: custom folders live in the acting account's studio.db."""
    try:
        from utils.paths.storage_roots import workspace_root
        return str(workspace_root())
    except Exception:  # noqa: BLE001 - no account context reads as the owner's
        return ""


def _memo_get(memo: dict, key):
    with _location_lock:
        entry = memo.get(key)
        if entry is not None and time.monotonic() - entry[0] < _LOCATION_TTL_SECONDS:
            return entry[1]
    return None


def _memo_put(memo: dict, key, value) -> None:
    with _location_lock:
        if len(memo) >= _LOCATION_MEMO_MAX:
            memo.clear()
        memo[key] = (time.monotonic(), value)


def _list_entries(directory: Path, limit: Optional[int]) -> list[os.DirEntry]:
    entries: list[os.DirEntry] = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if limit is not None and len(entries) >= limit:
                    break
                entries.append(entry)
    except OSError:
        pass
    return entries


def _entry_is(entry: os.DirEntry, kind: str) -> bool:
    try:
        return entry.is_dir() if kind == "dir" else entry.is_file()
    except OSError:
        return False


def _build_custom_folder_listing(path: Path, recursive: bool) -> LocationListing:
    """Uses the Model Hub's folder-scan rules so a model is found exactly where the Model Hub lists it."""
    from hub.services.models.local_inventory import _MAX_CUSTOM_FOLDER_ENTRIES, nested_scan_roots
    from hub.utils.inventory_scan import scan_folder_hf_caches

    hub_caches: dict[str, list[Path]] = {}
    file_dirs: dict[str, list[Path]] = {}
    if not path.is_dir():
        return LocationListing({}, {})
    walked = [path, *(nested_scan_roots(path) if recursive else ())]
    noted = set(walked)

    def note_files(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if _entry_is(entry, "file"):
                file_dirs.setdefault(entry.name.lower(), []).append(directory)

    def note_caches(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if entry.name.lower().startswith("models--") and _entry_is(entry, "dir"):
                hub_caches.setdefault(entry.name.lower(), []).append(directory)

    def subdirs(entries: list[os.DirEntry]) -> list[Path]:
        return [
            Path(entry.path)
            for entry in entries
            if not entry.name.startswith(".")
            and not entry.name.lower().startswith("models--")
            and _entry_is(entry, "dir")
        ]

    def note_subdir_files(directories: list[Path]) -> None:
        for child in directories:
            if child not in noted:
                noted.add(child)
                note_files(child, _list_entries(child, _MAX_CUSTOM_FOLDER_ENTRIES))

    root_entries: list[os.DirEntry] = []
    for directory in walked:
        entries = _list_entries(directory, _MAX_CUSTOM_FOLDER_ENTRIES)
        if directory == path:
            root_entries = entries
        note_files(directory, entries)
        for hub_cache in scan_folder_hf_caches(directory):
            note_caches(
                hub_cache,
                entries
                if hub_cache == directory
                else _list_entries(hub_cache, _MAX_CUSTOM_FOLDER_ENTRIES),
            )
        # Loose files one level down: the Model Hub lists a sub-folder holding a GGUF as a model.
        note_subdir_files(subdirs(entries))
    # <publisher>/<model>/ under the folder itself, which the Model Hub's LM Studio scan lists even when not recursive;
    # bounded like that scan, by one entry budget over the folder and its publisher folders.
    budget = _MAX_CUSTOM_FOLDER_ENTRIES - len(root_entries)
    for publisher in subdirs(root_entries):
        if budget <= 0:
            break
        entries = _list_entries(publisher, budget)
        budget -= len(entries)
        note_subdir_files(subdirs(entries))
    return LocationListing(
        {name: tuple(dirs) for name, dirs in hub_caches.items()},
        {name: tuple(dirs) for name, dirs in file_dirs.items()},
    )


def _build_cache_listing(root: Path) -> LocationListing:
    # Every repo: a cache with more than the custom-folder entry cap must not lose the one asked for.
    repos = [
        entry.name.lower()
        for entry in _list_entries(root, None)
        if entry.name.lower().startswith("models--") and _entry_is(entry, "dir")
    ]
    return LocationListing({name: (root,) for name in repos}, {})


def _bounded(key: str, work: Callable[[], _T]) -> Optional[_T]:
    """``work()`` under the Model Hub's per-folder budget; None on timeout, failure, or an already slow location."""
    from hub.services.models.local_inventory import run_custom_folder_bounded_sync
    from utils.paths.scan_folder_health import STATUS_SLOW, scan_folder_status

    if scan_folder_status(key) == STATUS_SLOW:
        return None
    try:
        return run_custom_folder_bounded_sync(key, work)
    except Exception as exc:  # noqa: BLE001 - one unreadable location must not hide the others
        logger.debug("Could not read a model location: %s", exc)
        return None


def _norm_key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _known_cache_rows() -> list[_Row]:
    """Non-active caches chat loads search. Paths are not resolved here: a dead mount would hang before the budget."""
    from core.inference.stt_sidecar import _active_hf_hub_cache
    from hub.utils.paths import hf_default_cache_dir, legacy_hf_cache_dir
    from utils.hf_cache_settings import known_hf_hub_cache_candidates

    try:
        known = known_hf_hub_cache_candidates()
    except Exception as exc:  # noqa: BLE001 - the legacy and default caches still count
        logger.debug("Could not list known Hub caches: %s", exc)
        known = []
    seen = {_norm_key(_active_hf_hub_cache())}
    rows: list[_Row] = []
    for root in (*known, legacy_hf_cache_dir(), hf_default_cache_dir()):
        key = _norm_key(root)
        if key not in seen:
            seen.add(key)
            rows.append((str(root), Path(root), False, "cache"))
    return rows


def _custom_folder_rows() -> list[_Row]:
    try:
        from storage.studio_db import list_scan_folders
        folders = list_scan_folders()
    except Exception as exc:  # noqa: BLE001 - an unreadable folder list must not fail a lookup
        logger.debug("Could not list custom folders: %s", exc)
        return []
    from hub.utils.paths import normalize_path
    return [
        (raw, Path(normalize_path(raw)).expanduser(), bool(folder.get("recursive")), "folder")
        for folder in folders
        if (raw := str(folder.get("path") or "")).strip()
    ]


def search_other_locations(
    candidates: Callable[[LocationListing], Iterable[_C]], probe: Callable[[_C], Optional[_T]]
) -> Optional[_T]:
    """First non-None ``probe(candidate)`` over each non-active location's ``candidates(listing)``. A location whose
    memoised listing names no candidate is skipped without touching the disk."""
    scope = account_scope()
    rows = _memo_get(_location_rows_memo, scope)
    if rows is None:
        rows = (*_known_cache_rows(), *_custom_folder_rows())
        _memo_put(_location_rows_memo, scope, rows)
    for raw, path, recursive, kind in rows:
        listing = _memo_get(_location_listings_memo, (scope, kind, raw))
        if listing is not None and not any(True for _ in candidates(listing)):
            continue

        def search(
            raw = raw,
            path = path,
            recursive = recursive,
            kind = kind,
            listing = listing,
        ) -> Optional[_T]:
            if listing is None:
                if kind == "cache":
                    listing = (
                        _build_cache_listing(path) if path.is_dir() else LocationListing({}, {})
                    )
                else:
                    listing = _build_custom_folder_listing(path, recursive)
                _memo_put(_location_listings_memo, (scope, kind, raw), listing)
            for candidate in candidates(listing):
                found = probe(candidate)
                if found is not None:
                    return found
            return None

        found = _bounded(raw, search)
        if found is not None:
            return found
    return None


def find_in_hub_caches(repo: str, lookup: Callable[[Path], Optional[_T]]) -> Optional[_T]:
    """``lookup(hub_cache)`` on the active cache, then on every other Hub cache holding *repo*."""
    from core.inference.stt_sidecar import _active_hf_hub_cache

    found = lookup(Path(_active_hf_hub_cache()))
    if found is not None:
        return found
    return search_other_locations(lambda listing: listing.caches_holding(repo), lookup)

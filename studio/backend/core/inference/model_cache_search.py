# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a curated side model (dictation, the FLUX.3 prompt upsampler, the mask segmenter) may sit on disk besides
the active Hugging Face cache: the other caches chat model loads search, then the user's custom folders. Those can sit
on slow or dead mounts, so each is listed and probed under the Model Hub's per-folder budget."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

from loggers import get_logger

logger = get_logger(__name__)


# One status poll runs a dozen lookups, so the locations are listed once and the listing is reused for a few seconds,
# per account: each account registers its own custom folders.
_LOCATION_TTL_SECONDS = 2.0
_location_lock = threading.Lock()
# (row key, path, recursive, kind): kind "cache" is a Hub cache root, "folder" a custom folder.
_Row = tuple[str, Path, bool, str]
_location_rows_memo: dict[str, tuple[float, tuple[_Row, ...]]] = {}
# Keyed by kind too: a known cache that is also a custom folder is listed both ways (repos only, and loose files).
_location_listings_memo: dict[tuple[str, str, str], tuple[float, "LocationListing"]] = {}
_LOCATION_MEMO_MAX = 256
# Entries read from one custom-folder directory, as the Model Hub's custom-folder scan (_MAX_CUSTOM_FOLDER_ENTRIES).
_CUSTOM_FOLDER_ENTRY_LIMIT = 2000
_T = TypeVar("_T")
_C = TypeVar("_C")


@dataclass(frozen = True)
class LocationListing:
    """What one location holds, by lowercased name: Hub caches per ``models--*`` repo directory, and the directories
    holding each file name. For a custom folder it covers the directories the Model Hub scans (the folder, and its
    nested folders when it is recursive), the ``hub`` cache under each, and their immediate sub-folders for loose files;
    for a Hub cache root, only the repos in it."""

    hub_caches: dict[str, tuple[Path, ...]]
    file_dirs: dict[str, tuple[Path, ...]]

    def caches_holding(self, repo: str) -> tuple[Path, ...]:
        return self.hub_caches.get(f"models--{repo.replace('/', '--')}".lower(), ())

    def dirs_holding(self, *names: str) -> list[Path]:
        first, *rest = (self.file_dirs.get(name.lower(), ()) for name in names)
        return [directory for directory in first if all(directory in other for other in rest)]


def account_scope() -> str:
    """The acting account's storage root. Custom folders live in its studio.db, so whatever is found through them
    is cached per account."""
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


def _list_entries(
    directory: Path, limit: Optional[int] = _CUSTOM_FOLDER_ENTRY_LIMIT
) -> list[os.DirEntry]:
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
    hub_caches: dict[str, list[Path]] = {}
    file_dirs: dict[str, list[Path]] = {}
    if not path.is_dir():
        return LocationListing({}, {})
    walked = [path]
    if recursive:
        from hub.services.models.local_inventory import nested_scan_roots
        walked.extend(nested_scan_roots(path))
    walked_set = set(walked)

    def note_files(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if _entry_is(entry, "file"):
                file_dirs.setdefault(entry.name.lower(), []).append(directory)

    def note_caches(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if entry.name.lower().startswith("models--") and _entry_is(entry, "dir"):
                hub_caches.setdefault(entry.name.lower(), []).append(directory)

    for directory in walked:
        entries = _list_entries(directory)
        note_caches(directory, entries)
        note_files(directory, entries)
        for entry in entries:
            name = entry.name
            if (
                name.startswith(".")
                or name.lower().startswith("models--")
                or not _entry_is(entry, "dir")
            ):
                continue
            child = Path(entry.path)
            child_entries = _list_entries(child)
            # A registered HF_HOME keeps its cache one level down, in hub/ (scan_folder_hf_caches).
            if name == "hub":
                note_caches(child, child_entries)
            if child not in walked_set:
                note_files(child, child_entries)
    return LocationListing(
        {name: tuple(dirs) for name, dirs in hub_caches.items()},
        {name: tuple(dirs) for name, dirs in file_dirs.items()},
    )


def _build_cache_listing(root: Path) -> LocationListing:
    # Every repo: a cache with more than the custom-folder entry cap must not lose the one asked for.
    repos = [
        entry.name.lower()
        for entry in _list_entries(root, limit = None)
        if entry.name.lower().startswith("models--") and _entry_is(entry, "dir")
    ]
    return LocationListing({name: (root,) for name in repos}, {})


def _bounded(key: str, work: Callable[[], _T]) -> Optional[_T]:
    """``work()`` under the Model Hub's per-folder budget and stuck-thread record, or None when it timed out, failed,
    or an earlier run under *key* is still stuck."""
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
    """The Hub caches chat model loads also search, besides the active one: previous Models folders, download
    folders, then the legacy and default caches. Stored paths are taken as written, not resolved: each is touched only
    inside its own bounded listing, so one dead mount cannot hide the others."""
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
    """The custom folders the acting account registered. Database only."""
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
    """The first answer of ``probe(candidate)`` over the candidates ``candidates(listing)`` names in each location
    besides the active cache: the other known caches, then the acting account's custom folders. Listing a location
    and probing its candidates run under the Model Hub's per-folder budget and stuck-thread record: a location that does
    not answer in time is skipped and marked slow, and one already marked slow is not touched. A location whose recent
    listing names no candidate is passed over without touching the disk."""
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
    """``lookup(hub_cache)`` on the active cache (downloads write only there), then on every other Hub cache holding
    *repo*: the known caches, then the caches inside custom folders."""
    from core.inference.stt_sidecar import _active_hf_hub_cache

    found = lookup(Path(_active_hf_hub_cache()))
    if found is not None:
        return found
    return search_other_locations(lambda listing: listing.caches_holding(repo), lookup)

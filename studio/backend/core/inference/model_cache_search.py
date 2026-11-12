# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a curated side model (dictation, the FLUX.3 prompt upsampler, the mask segmenter) may sit on disk besides
the active Hugging Face cache: the other caches chat model loads search, then the user's custom folders."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from loggers import get_logger

logger = get_logger(__name__)


# One status poll runs a dozen lookups, so a custom folder is listed once and the listing is reused for a few seconds,
# per account: each account registers its own custom folders.
_CUSTOM_FOLDER_TTL_SECONDS = 2.0
_custom_folder_lock = threading.Lock()
_custom_folder_rows_memo: dict[str, tuple[float, tuple[tuple[str, Path, bool], ...]]] = {}
_custom_folder_listings_memo: dict[tuple[str, str], tuple[float, "CustomFolderListing"]] = {}
_CUSTOM_FOLDER_MEMO_MAX = 256
# Entries read from one directory, as the Model Hub's custom-folder scan (_MAX_CUSTOM_FOLDER_ENTRIES).
_CUSTOM_FOLDER_ENTRY_LIMIT = 2000
_T = TypeVar("_T")
_C = TypeVar("_C")


@dataclass(frozen = True)
class CustomFolderListing:
    """What one custom folder holds, by lowercased name: Hub caches per ``models--*`` repo directory, and the
    directories holding each file name. Covers the directories the Model Hub scans (the folder, and its nested folders
    when it is recursive), the ``hub`` cache under each, and their immediate sub-folders for loose files."""

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
    with _custom_folder_lock:
        entry = memo.get(key)
        if entry is not None and time.monotonic() - entry[0] < _CUSTOM_FOLDER_TTL_SECONDS:
            return entry[1]
    return None


def _memo_put(memo: dict, key, value) -> None:
    with _custom_folder_lock:
        if len(memo) >= _CUSTOM_FOLDER_MEMO_MAX:
            memo.clear()
        memo[key] = (time.monotonic(), value)


def _custom_folder_rows(scope: str) -> tuple[tuple[str, Path, bool], ...]:
    """(row path, folder, recursive) for each custom folder the acting account registered. Database only."""
    rows = _memo_get(_custom_folder_rows_memo, scope)
    if rows is not None:
        return rows
    try:
        from storage.studio_db import list_scan_folders
        folders = list_scan_folders()
    except Exception as exc:  # noqa: BLE001 - an unreadable folder list must not fail a lookup
        logger.debug("STT: could not list custom folders: %s", exc)
        folders = []
    from hub.utils.paths import normalize_path

    rows = tuple(
        (raw, Path(normalize_path(raw)).expanduser(), bool(folder.get("recursive")))
        for folder in folders
        if (raw := str(folder.get("path") or "")).strip()
    )
    _memo_put(_custom_folder_rows_memo, scope, rows)
    return rows


def _list_entries(directory: Path) -> list[os.DirEntry]:
    entries: list[os.DirEntry] = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if len(entries) >= _CUSTOM_FOLDER_ENTRY_LIMIT:
                    break
                entries.append(entry)
    except OSError:
        pass
    return entries


def _entry_is_dir(entry: os.DirEntry) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _entry_is_file(entry: os.DirEntry) -> bool:
    try:
        return entry.is_file()
    except OSError:
        return False


def _build_custom_folder_listing(path: Path, recursive: bool) -> CustomFolderListing:
    hub_caches: dict[str, list[Path]] = {}
    file_dirs: dict[str, list[Path]] = {}
    if not path.is_dir():
        return CustomFolderListing({}, {})
    walked = [path]
    if recursive:
        from hub.services.models.local_inventory import nested_scan_roots
        walked.extend(nested_scan_roots(path))
    walked_set = set(walked)

    def note_files(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if _entry_is_file(entry):
                file_dirs.setdefault(entry.name.lower(), []).append(directory)

    def note_caches(directory: Path, entries: list[os.DirEntry]) -> None:
        for entry in entries:
            if entry.name.lower().startswith("models--") and _entry_is_dir(entry):
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
                or not _entry_is_dir(entry)
            ):
                continue
            child = Path(entry.path)
            child_entries = _list_entries(child)
            # A registered HF_HOME keeps its cache one level down, in hub/ (scan_folder_hf_caches).
            if name == "hub":
                note_caches(child, child_entries)
            if child not in walked_set:
                note_files(child, child_entries)
    return CustomFolderListing(
        {name: tuple(dirs) for name, dirs in hub_caches.items()},
        {name: tuple(dirs) for name, dirs in file_dirs.items()},
    )


def search_custom_folders(
    candidates: Callable[[CustomFolderListing], Iterable[_C]], probe: Callable[[_C], Optional[_T]]
) -> Optional[_T]:
    """The first answer of ``probe(candidate)`` over the candidates ``candidates(listing)`` names in each of the acting
    account's custom folders. Listing a folder and probing its candidates run under the Model Hub's per-folder budget
    and stuck-thread record: a folder that does not answer in time is skipped and marked slow, and one already marked
    slow is not touched. A folder whose recent listing names no candidate is passed over without touching the disk."""
    scope = account_scope()
    rows = _custom_folder_rows(scope)
    if not rows:
        return None
    from hub.services.models.local_inventory import run_custom_folder_bounded_sync
    from utils.paths.scan_folder_health import STATUS_SLOW, scan_folder_status

    for raw, path, recursive in rows:
        if scan_folder_status(raw) == STATUS_SLOW:
            continue
        listing = _memo_get(_custom_folder_listings_memo, (scope, raw))
        if listing is not None and not any(True for _ in candidates(listing)):
            continue

        def search(
            raw = raw,
            path = path,
            recursive = recursive,
            listing = listing,
        ) -> Optional[_T]:
            if listing is None:
                listing = _build_custom_folder_listing(path, recursive)
                _memo_put(_custom_folder_listings_memo, (scope, raw), listing)
            for candidate in candidates(listing):
                found = probe(candidate)
                if found is not None:
                    return found
            return None

        try:
            found = run_custom_folder_bounded_sync(raw, search)
        except Exception as exc:  # noqa: BLE001 - one unreadable folder must not hide the others
            logger.debug("STT: could not search a custom folder: %s", exc)
            continue
        if found is not None:
            return found
    return None


def search_custom_folder_hub_caches(
    repo: str, lookup: Callable[[Path], Optional[_T]]
) -> Optional[_T]:
    """``lookup(hub_cache)`` over the Hub caches inside custom folders that hold *repo*."""
    return search_custom_folders(lambda listing: listing.caches_holding(repo), lookup)


def known_hub_caches() -> Iterator[Path]:
    """The Hub caches Studio knows, active one first (downloads write only there): the roots chat model loads search
    too. Lazy, so a model found in the active cache reads nothing else. Custom folders are searched after these,
    through search_custom_folders."""
    seen: set[str] = set()

    def first_time(root: Path) -> bool:
        key = os.path.normcase(os.path.normpath(str(root)))
        if key in seen:
            return False
        seen.add(key)
        return True

    from core.inference.stt_sidecar import _active_hf_hub_cache

    active = Path(_active_hf_hub_cache())
    first_time(active)
    yield active
    try:
        from hub.utils.hf_cache_state import hf_cache_roots

        # Every known cache (previous Models folders, download folders), then the legacy and default ones.
        known = hf_cache_roots()
    except Exception as exc:  # noqa: BLE001 - the active cache still answered
        logger.debug("STT: could not list known Hub caches: %s", exc)
        known = []
    for root in known:
        if first_time(Path(root)):
            yield Path(root)


def find_in_hub_caches(repo: str, lookup: Callable[[Path], Optional[_T]]) -> Optional[_T]:
    """The first answer of ``lookup(hub_cache)`` over known_hub_caches(), then the custom-folder caches holding *repo*.
    Downloads write only to the active cache, which is searched first."""
    for root in known_hub_caches():
        found = lookup(root)
        if found is not None:
            return found
    return search_custom_folder_hub_caches(repo, lookup)

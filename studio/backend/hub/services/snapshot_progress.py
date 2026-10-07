# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared snapshot download-progress computation for models and datasets. Both scan the cache's ``blobs/`` dir, split finalized vs ``.incomplete`` bytes, filter to the target revision's expected hashes, and divide by its total size; only the ``metadata_resolver`` differs. One copy keeps the two from drifting (a prior hash-filter fix once landed only on the model copy, leaving datasets summing stale blobs against the wrong total)."""

from __future__ import annotations

import asyncio
import os
import stat as stat_module
import threading
from pathlib import Path
from typing import Callable, Generic, Optional, Sequence, TypeVar

from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.state_dir import RepoType
from hub.utils.hf_cache_state import (
    blob_bytes_present,
    incomplete_blob_hash,
    preferred_repo_cache_dirs,
    snapshot_selection_key,
)
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id
from utils.paths.path_utils import is_appledouble_metadata

logger = get_logger(__name__)

# (repo_id, hf_token) -> (expected_total_bytes, expected_blob_hashes)
SnapshotMetadataResolver = Callable[[str, Optional[str]], "tuple[int, frozenset[str]]"]
SnapshotExpectedFilesResolver = Callable[
    [str, Optional[str]], Sequence["download_manifest.ExpectedFile"]
]
VariantFileMatcher = Callable[[str], bool]

_progress_step_lock = threading.Lock()
_last_progress_step: dict[str, int] = {}


def _log_progress_step(job_key: str, repo_id: str, variant: Optional[str], progress: float) -> None:
    step = int(progress * 10)
    with _progress_step_lock:
        last = _last_progress_step.get(job_key, -1)
        if step == last:
            return
        _last_progress_step[job_key] = step
        if step < last:
            return
    logger.info(
        "hub_download_progress",
        repo_id = repo_id,
        variant = variant or "",
        percent = step * 10,
    )


def _empty_progress(expected_bytes: int, *, measured: bool = True) -> dict:
    """An all-zero reading. ``measured`` is the difference between "there is no cache dir for this repo" and "the scan itself failed": hydration retires a persisted job on the first and must not on the second, since a transient failure is not evidence that a partial cache was wiped. Carried as its own flag, not by omitting ``cache_path``: these dicts are serialized through DownloadProgressResponse, whose ``cache_path`` defaults to None, so the omission was reinstated as an explicit null before the frontend ever saw it and the distinction was lost on every route. An older backend sends neither field, which the frontend reads as unknown, so the rule still covers it."""
    reading = {
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": max(expected_bytes, 0),
        "progress": 0,
        "cache_measured": measured,
    }
    if measured:
        reading["cache_path"] = None
    return reading


_T = TypeVar("_T")


class _Lazy(Generic[_T]):
    """A value computed at most once, and only once something asks for it. The entry's manifest, its newest snapshot dir and the metadata file list are each wanted by two callers (the unknown-file-set byte reading and the completion check) and the completion check only wants them once its cheap byte guards have passed; taking them up front would put a state-dir lookup, a ``snapshots/`` listing and a metadata call on every poll of every repo that is still mid-download."""

    __slots__ = ("_compute", "_value", "_loaded")

    def __init__(self, compute: Callable[[], _T]) -> None:
        self._compute = compute
        self._value: Optional[_T] = None
        self._loaded = False

    def get(self) -> _T:
        if not self._loaded:
            self._value = self._compute()
            self._loaded = True
        return self._value  # type: ignore[return-value]


def _variant_bytes_on_disk(
    manifest: Optional[download_manifest.Manifest],
    snapshot_dir: Optional[Path],
    variant_file_matcher: Optional["VariantFileMatcher"],
    active_partial_hashes: "frozenset[str]" = frozenset(),
) -> int:
    """Bytes a variant owns, read from the snapshot dir instead of ``blobs/``. The snapshot dir is the one variant-scoped view of the cache: its entries are named per file, so a sibling quant is excluded by name, whereas in the shared ``blobs/`` dir a sibling's bytes are indistinguishable from this variant's and counting them wholesale is the "instant ~900 MB" bug. ``stat`` follows HF's symlink layout and reads the Windows copy layout directly, which matters even with resolved hashes since recent Hub clients can materialize completed files without retaining a finalized blob entry."""
    if snapshot_dir is None:
        return 0
    total = 0
    if manifest is not None:
        for expected in manifest.expected_files:
            if not download_manifest.expected_path_is_safe(expected.path):
                continue
            if expected.sha256 and expected.sha256 in active_partial_hashes:
                # A force/retry can leave the previous file beside its replacement; count one.
                continue
            try:
                total += (snapshot_dir / expected.path).stat().st_size
            except OSError:
                continue
        return total
    if variant_file_matcher is None:
        return 0
    return _materialized_bytes(snapshot_dir, variant_file_matcher)


def _walk_files(root: Path) -> "tuple[list[Path], bool]":
    """Every file under ``root``, and whether the traversal saw all of it. Not ``rglob``: it suppresses every OSError raised while scanning (documented behaviour since 3.13), so an unreadable subtree comes back as a SHORT list indistinguishable from an empty one, and a caller asking "is the variant here?" then answers a confident no about a directory it could not read. ``os.scandir`` reports the failure, and a subtree that is genuinely missing is not one."""
    files: list[Path] = []
    complete = True
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = list(scan)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            complete = False
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks = False):
                    stack.append(Path(entry.path))
                elif entry.is_file():
                    files.append(Path(entry.path))
            except OSError:
                complete = False
    return files, complete


def _variant_main_shard_present(
    snapshot_dir: Optional[Path], variant_file_matcher: Optional["VariantFileMatcher"]
) -> Optional[bool]:
    """Whether the variant's OWN files are in the snapshot dir. None when unanswerable. The narrower question ``companions = False`` asks: shared companions belong to every quant in the repo, so their presence says nothing about this one. Used on the path where the blob hashes could not be resolved, since the snapshot dir is still named per file and can settle absence even when the hash filter cannot; an unreadable or absent dir stays unknown."""
    if snapshot_dir is None or variant_file_matcher is None:
        return None
    # An unreadable entry may be the main shard, so a failure is not evidence of absence.
    entries, complete = _walk_files(snapshot_dir)
    for path in entries:
        relative = path.relative_to(snapshot_dir).as_posix()
        if is_appledouble_metadata(path):
            continue
        if variant_file_matcher(relative, companions = False):
            return True
    return None if not complete else False


def _retained_snapshot_dirs(entry: Path) -> list[Path]:
    """Every snapshot the repo cache dir keeps, newest first. A cache can hold several revisions, and the requested variant is not always in the newest: reading only that one reported a complete cached quant as 0 bytes and never verified its manifest, so it stayed at 99% and adoptable."""
    try:
        snapshots = [child for child in (entry / "snapshots").iterdir() if child.is_dir()]
    except OSError:
        return []
    return sorted(snapshots, key = snapshot_selection_key, reverse = True)


def _variant_present_in_any_snapshot(
    entry: Path, variant_file_matcher: Optional["VariantFileMatcher"]
) -> Optional[bool]:
    """``_variant_main_shard_present`` over every snapshot the repo dir retains. True as soon as one holds the variant's own file; False only when every snapshot was read and none did; None when there was nothing to read or a read failed, which is unknown."""
    snapshots = _retained_snapshot_dirs(entry)
    if not snapshots:
        return None
    verdicts = [
        _variant_main_shard_present(snapshot, variant_file_matcher) for snapshot in snapshots
    ]
    if any(verdict is True for verdict in verdicts):
        return True
    if any(verdict is None for verdict in verdicts):
        return None
    return False


def _materialized_bytes(snapshot_dir: Path, variant_file_matcher: "VariantFileMatcher") -> int:
    """Bytes the variant's files present in the snapshot dir. A predicate, not a file list, because this is the path where the file list is precisely what could not be determined. That makes it a lower bound on the wrong side for shared companions (the matcher accepts every mmproj and drafter in the repo, while a plan fetches one of each) so it is fit for a byte reading the caller clamps and displays, and not for deciding whether a download finished. ``stat`` follows the link, so a blob that was written but never linked contributes nothing, and the Windows copy layout is read as is."""
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return 0
    entries = [path for path in entries if not is_appledouble_metadata(path)]

    def _accepts(relative: str, *, companions: bool) -> bool:
        try:
            return bool(variant_file_matcher(relative, companions = companions))
        except TypeError:
            return bool(variant_file_matcher(relative))

    # Shared companions alone do not prove THIS quant is here.
    owns_a_main = False
    for path in entries:
        try:
            relative = path.relative_to(snapshot_dir).as_posix()
        except ValueError:
            continue
        if _accepts(relative, companions = False):
            try:
                if path.is_file():
                    owns_a_main = True
                    break
            except OSError:
                continue
    if not owns_a_main:
        return 0

    total = 0
    for path in entries:
        try:
            relative = path.relative_to(snapshot_dir).as_posix()
            if not _accepts(relative, companions = True) or not path.is_file():
                continue
            total += path.stat().st_size
        except (OSError, ValueError):
            continue
    return total


def _snapshot_complete_on_disk(
    *,
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    entry: Path,
    snapshot_dirs: "_Lazy[list[Path]]",
    entry_manifest: "_Lazy[Optional[download_manifest.Manifest]]",
    metadata_files: "_Lazy[tuple[download_manifest.ExpectedFile, ...]]",
    expected_total: int,
    completed_bytes: int,
    in_progress_bytes: int,
    expected_hashes: "frozenset[str]" = frozenset(),
) -> bool:
    if expected_total <= 0 or completed_bytes < expected_total or in_progress_bytes > 0:
        return False
    snapshots = snapshot_dirs.get()
    if not snapshots:
        return False
    if variant is None and hf_cache_scan.repo_cache_dir_has_incomplete_blobs(entry):
        return False
    if download_manifest.has_cancel_marker(
        repo_type,
        repo_id,
        variant,
        hub_cache = entry.parent,
    ):
        return False
    manifest = entry_manifest.get()
    if manifest is None:
        # Verify against HF metadata when no manifest exists; expected_bytes is only a hint.
        metadata_expected = metadata_files.get()
        if not metadata_expected:
            return False
        manifest = download_manifest.Manifest(
            repo_type = repo_type,
            repo_id = repo_id,
            variant = variant,
            started_at = "",
            expected_files = metadata_expected,
        )
    # Check ANY retained snapshot; require entries to resolve to known hashes (no sha256 read).
    for snap in snapshots:
        if not download_manifest.verify_against_disk(manifest, snap).ok:
            continue
        if not expected_hashes or _snapshot_resolves_to(manifest, snap, expected_hashes):
            return True
    return False


def _referenced_commits(entry: Path) -> "frozenset[str]":
    """Commits this repo cache dir still points at. HF records the commit a branch or tag resolved to in ``refs/<revision>`` on every snapshot_download whose revision was not already a raw sha, so for the default ``main`` the file is always there. It is the one revision marker that survives without a manifest."""
    commits: set[str] = set()
    try:
        refs = list((entry / "refs").rglob("*"))
    except OSError:
        return frozenset()
    for ref in refs:
        try:
            if not ref.is_file():
                continue
            commit = download_manifest.normalized_commit_hash(
                ref.read_text(encoding = "utf-8").strip()
            )
        except (OSError, ValueError):
            continue
        if commit:
            commits.add(commit)
    return frozenset(commits)


def _snapshot_is_stale_copy(
    snapshot: Path, manifest: "Optional[download_manifest.Manifest]"
) -> bool:
    """Whether ``snapshot`` names a revision this cache dir has moved off. Only asked where there is no symlink to read. HF names a snapshot dir after its commit, so a dir named by neither the manifest's recorded commit nor any live ref is an older revision, and its same-named files are not this download's bytes. Neither marker present leaves the question unanswerable, and unanswerable is not a mismatch."""
    commit_hash = download_manifest.normalized_commit_hash(getattr(manifest, "commit_hash", None))
    if commit_hash:
        return snapshot.name != commit_hash
    referenced = _referenced_commits(snapshot.parent.parent)
    return bool(referenced) and snapshot.name not in referenced


def _snapshot_resolves_to(
    manifest: "Optional[download_manifest.Manifest]",
    snapshot: Path,
    expected_hashes: "frozenset[str]",
) -> bool:
    """Whether every expected file in ``snapshot`` points at one of ``expected_hashes``. HF names a blob by its hash and the snapshot entry links to it, so the link target settles which revision is materialized here. A copy-layout cache (Windows without symlinks) has no target to read, and neither does a reading with no manifest to name the files, so both fall back to dating the snapshot by revision."""
    if manifest is None:
        return not _snapshot_is_stale_copy(snapshot, None)
    for expected in getattr(manifest, "expected_files", ()) or ():
        if not download_manifest.expected_path_is_safe(expected.path):
            continue
        entry = snapshot / expected.path
        try:
            if not entry.is_symlink():
                if _snapshot_is_stale_copy(snapshot, manifest):
                    return False
                continue
            target = os.path.basename(os.readlink(entry))
        except OSError:
            continue
        if target and target not in expected_hashes:
            return False
    return True


def manifest_matches_download(
    manifest: Optional[download_manifest.Manifest],
    metadata: Optional[download_registry.DownloadMetadata],
) -> bool:
    """A reused scope must not read a previous file set or revision's manifest."""
    files = frozenset(getattr(metadata, "scoped_files", ()) or ())
    if manifest is None or not files:
        return True
    if frozenset(file.path for file in manifest.expected_files) != files:
        return False
    hashes = frozenset(getattr(metadata, "progress_blob_hashes", ()) or ())
    return not hashes or hashes == frozenset(
        file.sha256 for file in manifest.expected_files if file.sha256
    )


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
    variant_file_matcher: Optional[VariantFileMatcher] = None,
    expected_files_resolver: Optional[SnapshotExpectedFilesResolver] = None,
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
    metadata_hub_cache = getattr(metadata, "hub_cache", None)
    active_root = Path(metadata_hub_cache) if metadata_hub_cache else None

    expected_total = max(expected_bytes, 0)
    meta_total, expected_hashes = metadata_resolver(repo_id, hf_token)
    meta_total = max(0, meta_total)
    expected_total = (
        meta_total if variant is not None and meta_total > 0 else max(expected_total, meta_total)
    )

    scoped_files = frozenset(getattr(metadata, "scoped_files", ()) or ())
    if variant is not None and scoped_files:
        variant_file_matcher = lambda path, **_kwargs: path in scoped_files

    # Without hashes, a variant must not count unscoped blobs shared by sibling quants.
    count_unscoped = variant is None
    # Empty hashes mean unknown (e.g. negatively cached model_info failure); use snapshot files.
    variant_file_set_unknown = variant is not None and not expected_hashes
    metadata_files: "_Lazy[tuple[download_manifest.ExpectedFile, ...]]" = _Lazy(
        lambda: (
            tuple(expected_files_resolver(repo_id, hf_token))
            if expected_files_resolver is not None
            else ()
        )
    )

    readings: list[tuple[int, int, Optional[str], bool, Optional[bool]]] = []
    # Collect per-root OSErrors so an unreadable root reads unknown, not absent.
    scan_errors: list = []
    cache_dirs = (
        preferred_repo_cache_dirs(
            repo_type,
            repo_id,
            force_active = force_active,
            active_root = active_root,
            scan_errors = scan_errors,
        )
        if active_root is not None
        else preferred_repo_cache_dirs(
            repo_type, repo_id, force_active = force_active, scan_errors = scan_errors
        )
    )
    for entry in cache_dirs:
        completed_bytes = 0
        # Keyed by logical blob: racing writers on one etag would overshoot when summed.
        partial_bytes: dict[str, int] = {}
        completed_hashes: set[str] = set()
        unattributable_partial = False
        cache_path = hf_cache_scan.resolve_hf_cache_realpath(entry)
        blobs_dir = entry / "blobs"
        try:
            # os.stat, not is_dir(): is_dir() swallows OSError into a false measured absence.
            blobs_present = stat_module.S_ISDIR(os.stat(blobs_dir).st_mode)
        except FileNotFoundError:
            blobs_present = False
        except OSError as exc:
            scan_errors.append(exc)
            blobs_present = False
        if blobs_present:
            try:
                blob_entries = list(blobs_dir.iterdir())
            except OSError as exc:
                scan_errors.append(exc)
                blob_entries = []
            for f in blob_entries:
                try:
                    if not f.is_file():
                        continue
                    partial_hash = incomplete_blob_hash(f.name)
                    if partial_hash is not None:
                        if expected_hashes:
                            if partial_hash not in expected_hashes:
                                continue
                        elif not count_unscoped:
                            unattributable_partial = True
                            continue
                        partial_bytes[partial_hash] = max(
                            partial_bytes.get(partial_hash, 0), blob_bytes_present(f)
                        )
                    else:
                        if expected_hashes:
                            if f.name not in expected_hashes:
                                continue
                        elif not count_unscoped:
                            continue
                        completed_hashes.add(f.name)
                        completed_bytes += f.stat().st_size
                except OSError as exc:
                    scan_errors.append(exc)
                    continue
        # A finalized blob supersedes its partials; counting both pins progress at 0.99.
        for blob_hash in completed_hashes:
            partial_bytes.pop(blob_hash, None)
        # Largest wins: freshest-mtime oscillates between live writers under a broken lock.
        in_progress_bytes = sum(partial_bytes.values())
        snapshot_dirs: "_Lazy[list[Path]]" = _Lazy(
            lambda entry = entry: _retained_snapshot_dirs(entry)
        )
        raw_manifest: "_Lazy[Optional[download_manifest.Manifest]]" = _Lazy(
            lambda entry = entry: download_manifest.read_manifest(
                repo_type, repo_id, variant, hub_cache = entry.parent
            )
        )
        entry_manifest: "_Lazy[Optional[download_manifest.Manifest]]" = _Lazy(
            lambda raw_manifest = raw_manifest: raw_manifest.get()
            if manifest_matches_download(raw_manifest.get(), metadata)
            else None
        )
        if variant is not None:
            # Best reading across retained snapshots (hf_hub 1.18 Windows copies leave blobs at zero).
            manifest = entry_manifest.get()
            on_disk = max(
                (
                    _variant_bytes_on_disk(
                        manifest,
                        snap,
                        variant_file_matcher,
                        frozenset(partial_bytes),
                    )
                    for snap in snapshot_dirs.get()
                    if manifest_matches_download(raw_manifest.get(), metadata)
                    and (
                        not expected_hashes
                        or _snapshot_resolves_to(manifest, snap, expected_hashes)
                    )
                ),
                default = 0,
            )
            if expected_total > 0:
                on_disk = min(on_disk, expected_total)
            completed_bytes = max(completed_bytes, on_disk)
        # Sibling quants keep the dir alive; False only on positive evidence of absence.
        target_present: Optional[bool] = None
        if variant is not None and not variant_file_set_unknown:
            # Scan materialized files, not the blob tally: leftover blobs and companions stay positive.
            scanned = _variant_present_in_any_snapshot(entry, variant_file_matcher)
            if scanned is not None:
                target_present = scanned or bool(in_progress_bytes)
            else:
                target_present = bool(completed_bytes or in_progress_bytes)
        elif variant is not None:
            scanned = _variant_present_in_any_snapshot(entry, variant_file_matcher)
            if scanned is not None:
                # An unattributable .incomplete blob may be this download's, so presence is unknown.
                target_present = None if (not scanned and unattributable_partial) else scanned
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
                    snapshot_dirs = snapshot_dirs,
                    entry_manifest = entry_manifest,
                    metadata_files = metadata_files,
                    expected_total = expected_total,
                    completed_bytes = completed_bytes,
                    in_progress_bytes = in_progress_bytes,
                    expected_hashes = expected_hashes,
                ),
                target_present,
            )
        )

    selected = max(
        readings,
        # complete_on_disk breaks byte-total ties between caches, else root order caps at 99%.
        key = lambda item: (item[0] + item[1], bool(item[3]), item[0]),
        default = None,
    )
    if selected is None:
        # Nothing measured and a root unlistable: unknown, not gone.
        if scan_errors:
            return _empty_progress(expected_bytes, measured = False)
        return empty

    completed_bytes, in_progress_bytes, cache_path, complete_on_disk, target_present = selected
    presence = [reading[4] for reading in readings]
    if any(verdict is True for verdict in presence):
        target_present = True
    elif any(verdict is None for verdict in presence):
        target_present = None
    downloaded_bytes = completed_bytes + in_progress_bytes
    # An incomplete scan is a lower bound; downgrade absence claims to unknown.
    scan_incomplete = bool(scan_errors)
    if scan_incomplete:
        if not target_present:
            target_present = None
    # Never subtract a baseline covering the whole total: '0 B of 0 B' evicts the job.
    effective_baseline_bytes = (
        completed_baseline_bytes
        if (
            not complete_on_disk
            and completed_baseline_bytes <= completed_bytes
            and completed_baseline_bytes < expected_total
        )
        else 0
    )
    display_completed_bytes = max(0, completed_bytes - effective_baseline_bytes)
    display_downloaded_bytes = max(0, downloaded_bytes - effective_baseline_bytes)

    if expected_total <= 0:
        return {
            "downloaded_bytes": display_downloaded_bytes,
            "completed_bytes": display_completed_bytes,
            "complete_on_disk": False,
            "expected_bytes": 0,
            "progress": 0,
            "cache_path": cache_path,
            "target_present": target_present,
            "cache_measured": not scan_incomplete,
        }

    display_expected_total = max(0, expected_total - effective_baseline_bytes)
    if downloaded_bytes == 0:
        return {
            **empty,
            "expected_bytes": display_expected_total,
            "cache_path": cache_path,
            "target_present": target_present,
            "cache_measured": not scan_incomplete,
        }

    # Cap at 0.99 until the manifest-backed disk check verifies completion.
    progress = (
        1.0
        if complete_on_disk
        else (
            min(display_downloaded_bytes / display_expected_total, 0.99)
            if display_expected_total > 0
            else 0
        )
    )
    if force_active:
        _log_progress_step(job_key, repo_id, variant, progress)
    return {
        "downloaded_bytes": display_downloaded_bytes,
        "completed_bytes": display_completed_bytes,
        "complete_on_disk": complete_on_disk,
        "expected_bytes": display_expected_total,
        "progress": round(progress, 3),
        "cache_path": cache_path,
        "target_present": target_present,
        "cache_measured": not scan_incomplete,
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
    variant_file_matcher: Optional[VariantFileMatcher] = None,
    expected_files_resolver: Optional[SnapshotExpectedFilesResolver] = None,
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
            variant_file_matcher = variant_file_matcher,
            expected_files_resolver = expected_files_resolver,
        )
    except Exception as e:
        logger.warning(
            "Error checking %s download progress for %s: %s: %s",
            repo_type,
            repo_id,
            type(e).__name__,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        return _empty_progress(expected_bytes, measured = False)

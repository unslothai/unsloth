# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared snapshot download-progress computation for models and datasets.

Both scan the cache's ``blobs/`` dir, split finalized vs ``.incomplete`` bytes,
filter to the target revision's expected hashes, and divide by its total size;
only the ``metadata_resolver`` differs. One copy keeps the two from drifting (a
prior hash-filter fix once landed only on the model copy, leaving datasets
summing stale blobs against the wrong total)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Callable, Generic, Optional, Sequence, TypeVar

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
# (repo_id, hf_token) -> the files HF says this target should contain. Optional,
# and the only thing that lets a materialized snapshot with no manifest settle.
SnapshotExpectedFilesResolver = Callable[
    [str, Optional[str]], Sequence["download_manifest.ExpectedFile"]
]
# repo-relative snapshot path -> whether it belongs to the variant being polled.
# Supplied per repo kind, so this module keeps knowing nothing about quant labels.
VariantFileMatcher = Callable[[str], bool]

# One progress log per 10% step per job, so an active download reports progress
# without emitting a line on every poll.
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
            return  # download restarted; resync without logging
    logger.info(
        "hub_download_progress",
        repo_id = repo_id,
        variant = variant or "",
        percent = step * 10,
    )


def _empty_progress(expected_bytes: int, *, measured: bool = True) -> dict:
    """An all-zero reading.

    ``measured`` is the difference between "there is no cache dir for this repo" and "the scan
    itself failed". Hydration retires a persisted job on the first and must not on the second: a
    transient failure is not evidence that a partial cache was wiped. A null ``cache_path`` says
    absent; omitting the key entirely says unknown, which is also what an older backend that
    never sent the field looks like, so the frontend rule is the same for both."""
    reading = {
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": max(expected_bytes, 0),
        "progress": 0,
    }
    if measured:
        reading["cache_path"] = None
    return reading


_T = TypeVar("_T")


class _Lazy(Generic[_T]):
    """A value computed at most once, and only once something asks for it.

    The entry's manifest, its newest snapshot dir and the metadata file list are
    each wanted by two callers -- the unknown-file-set byte reading and the
    completion check -- and the completion check only wants them once its cheap
    byte guards have passed. Taking them up front would put a state-dir lookup,
    a ``snapshots/`` listing and a metadata call on every poll of every repo
    that is still mid-download.
    """

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
) -> int:
    """Bytes a variant owns, read from the snapshot dir instead of ``blobs/``.

    Only used when the expected blob hashes could not be resolved. The snapshot
    dir is the one variant-scoped view of the cache: its entries are named per
    file, so a sibling quant is excluded by name, whereas in the shared
    ``blobs/`` dir a sibling's bytes are indistinguishable from this variant's
    and counting them wholesale is the "instant ~900 MB" bug. ``stat`` follows
    HF's symlink layout and reads the Windows copy layout directly.
    """
    if snapshot_dir is None:
        return 0
    total = 0
    if manifest is not None:
        for expected in manifest.expected_files:
            if not download_manifest.expected_path_is_safe(expected.path):
                continue
            try:
                total += (snapshot_dir / expected.path).stat().st_size
            except OSError:
                continue
        return total
    if variant_file_matcher is None:
        return 0
    return _materialized_bytes(snapshot_dir, variant_file_matcher)


def _materialized_bytes(snapshot_dir: Path, variant_file_matcher: "VariantFileMatcher") -> int:
    """Bytes the variant's files present in the snapshot dir.

    A predicate, not a file list, because this is the path where the file list
    is precisely what could not be determined. That makes it a lower bound on
    the wrong side for shared companions -- the matcher accepts every mmproj and
    drafter in the repo, while a plan fetches one of each -- so it is fit for a
    byte reading the caller clamps and displays, and not for deciding whether a
    download finished. ``stat`` follows the link, so a blob that was written but
    never linked contributes nothing, and the Windows copy layout is read as is.
    """
    total = 0
    try:
        entries = snapshot_dir.rglob("*")  # unordered: the result is a sum
    except OSError:
        return 0
    for path in entries:
        try:
            relative = path.relative_to(snapshot_dir).as_posix()
            if not variant_file_matcher(relative) or not path.is_file():
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
    snapshot_dir: "_Lazy[Optional[Path]]",
    entry_manifest: "_Lazy[Optional[download_manifest.Manifest]]",
    metadata_files: "_Lazy[tuple[download_manifest.ExpectedFile, ...]]",
    expected_total: int,
    completed_bytes: int,
    in_progress_bytes: int,
) -> bool:
    if expected_total <= 0 or completed_bytes < expected_total or in_progress_bytes > 0:
        return False
    snapshot = snapshot_dir.get()
    if snapshot is None:
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
        # No manifest, but HF metadata describes the same thing a manifest does:
        # the exact files this download was supposed to fetch, at their declared
        # sizes. Verify against that rather than refusing outright, because a
        # manifest that was never written (metadata was unreachable when the run
        # ended), was deleted, or sits under a cache scope this reader cannot
        # name is not evidence of an unfinished download -- and refusing left a
        # materialized snapshot partial forever.
        #
        # Nothing weaker will do here. The caller's expected_bytes is a catalog
        # hint, and a byte total assembled from the shared blobs/ dir cannot
        # tell this variant's bytes from a sibling's or a stray companion's, so
        # neither is grounds for calling a download complete. Checking the named
        # files against the snapshot dir also catches the case a blob-level
        # tally structurally cannot: HF writes a blob and then links it, so a
        # run killed in between leaves bytes that nothing points at.
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
    return download_manifest.verify_against_disk(manifest, snapshot).ok


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
    # Always resolve the revision's blob hashes so stale blobs from a superseded
    # revision can't inflate the count; hashes degrade to empty (count-all) only
    # when metadata is unavailable (e.g. offline). Take the larger total so a low
    # caller hint can't cap the bar below the revision's real size.
    meta_total, expected_hashes = metadata_resolver(repo_id, hf_token)
    meta_total = max(0, meta_total)
    expected_total = max(expected_total, meta_total)

    # Without resolved hashes, a variant must not count unscoped blobs: sibling
    # quants share one blobs/ dir, so a sibling's bytes (or .incomplete) would be
    # misattributed and make the bar jump backward. A no-variant snapshot owns
    # the whole dir, so it counts unscoped.
    count_unscoped = variant is None
    # An empty hash set is "the expected file set could not be determined", not
    # "this variant has no bytes" -- model_info failing (offline, or a 401 on a
    # gated repo) is negatively cached, so one failed lookup keeps the set empty
    # for the whole TTL. Filtering every blob out then reports a finished 33 GB
    # variant as "0 B of 33 GB" and, because completion is never observed, leaves
    # Retry/Resume on the card. Fall back to the variant's own files in the
    # snapshot dir, which stay attributable when the blob hashes do not.
    variant_file_set_unknown = variant is not None and not expected_hashes
    # Resolved at most once, and only if a reading gets far enough to need it.
    metadata_files: "_Lazy[tuple[download_manifest.ExpectedFile, ...]]" = _Lazy(
        lambda: tuple(expected_files_resolver(repo_id, hf_token))
        if expected_files_resolver is not None
        else ()
    )

    readings: list[tuple[int, int, Optional[str], bool]] = []
    # The enumeration suppresses OSError per root, so an unreadable cache root came back as
    # "no dirs" -- indistinguishable from a wiped cache, and hydration retires the job on
    # that. Collected so the empty answer below can say unknown instead of absent.
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
        in_progress_bytes = 0
        cache_path = hf_cache_scan.resolve_hf_cache_realpath(entry)
        blobs_dir = entry / "blobs"
        if blobs_dir.is_dir():
            try:
                blob_entries = list(blobs_dir.iterdir())
            except OSError:
                blob_entries = []
            for f in blob_entries:
                # Skip a blob that vanished mid-poll rather than zeroing the reading.
                try:
                    if not f.is_file():
                        continue
                    if f.name.endswith(INCOMPLETE_SUFFIX):
                        blob_hash = f.name[: -len(INCOMPLETE_SUFFIX)]
                        if expected_hashes:
                            if blob_hash not in expected_hashes:
                                continue
                        elif not count_unscoped:
                            continue
                        in_progress_bytes += blob_bytes_present(f)
                    else:
                        if expected_hashes:
                            if f.name not in expected_hashes:
                                continue
                        elif not count_unscoped:
                            continue
                        completed_bytes += f.stat().st_size
                except OSError:
                    continue
        snapshot_dir: "_Lazy[Optional[Path]]" = _Lazy(
            lambda entry = entry: latest_snapshot_dir(entry)
        )
        entry_manifest: "_Lazy[Optional[download_manifest.Manifest]]" = _Lazy(
            lambda entry = entry: download_manifest.read_manifest(
                repo_type,
                repo_id,
                variant,
                hub_cache = entry.parent,
            )
        )
        if variant_file_set_unknown:
            on_disk = _variant_bytes_on_disk(
                entry_manifest.get(),
                snapshot_dir.get(),
                variant_file_matcher,
            )
            # Clamped, because the matcher behind the no-manifest half of that
            # reading accepts every companion in the repo and so can overshoot.
            # A bar reading "44 GB of 33 GB" is a worse answer than a pinned one.
            if expected_total > 0:
                on_disk = min(on_disk, expected_total)
            completed_bytes = max(completed_bytes, on_disk)
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
                    snapshot_dir = snapshot_dir,
                    entry_manifest = entry_manifest,
                    metadata_files = metadata_files,
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
        # Nothing measured AND a root that could not be listed: the cache may be entirely
        # intact behind that error, so this is unknown, not gone.
        if scan_errors:
            return _empty_progress(expected_bytes, measured = False)
        return empty

    completed_bytes, in_progress_bytes, cache_path, complete_on_disk = selected
    downloaded_bytes = completed_bytes + in_progress_bytes
    # Subtract the companion baseline only while still counted in completed_bytes
    # and the variant is not yet verified complete, else genuine progress reads as
    # 0-byte. A baseline that covers the whole expected total is never subtracted
    # either: a variant that was already on disk when the job was claimed carries
    # exactly that, and netting it out leaves "0 B of 0 B" -- nothing for the bar
    # to draw and, to the frontend, a job to evict. complete_on_disk covered that
    # only while completion could be verified, which is the one thing an
    # unresolvable file set takes away.
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

    # Cap at 0.99 until the manifest-backed disk check verifies completion: on
    # resume, completed bytes can sit above the threshold while files still download.
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

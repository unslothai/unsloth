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
# The files HF says this target should contain: optional, and the only thing that lets a
# materialized snapshot with no manifest settle.
SnapshotExpectedFilesResolver = Callable[
    [str, Optional[str]], Sequence["download_manifest.ExpectedFile"]
]
# Supplied per repo kind, so this module keeps knowing nothing about quant labels.
VariantFileMatcher = Callable[[str], bool]

# One progress log per 10% step per job, so an active download reports progress
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
    transient failure is not evidence that a partial cache was wiped.

    Carried as its own flag, not by omitting ``cache_path``: these dicts are serialized through
    DownloadProgressResponse, whose ``cache_path`` defaults to None, so the omission was
    reinstated as an explicit null before the frontend ever saw it and the distinction was lost
    on every route. An older backend sends neither field, which the frontend reads as unknown,
    so the rule still covers it."""
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
    active_partial_hashes: "frozenset[str]" = frozenset(),
) -> int:
    """Bytes a variant owns, read from the snapshot dir instead of ``blobs/``.

    The snapshot dir is the one variant-scoped view of the cache: its entries
    are named per file, so a sibling quant is excluded by name, whereas in the
    shared ``blobs/`` dir a sibling's bytes are indistinguishable from this
    variant's and counting them wholesale is the "instant ~900 MB" bug.
    ``stat`` follows HF's symlink layout and reads the Windows copy layout
    directly. The latter matters even with resolved hashes: recent Hub clients
    can materialize completed files without retaining a finalized blob entry.
    """
    if snapshot_dir is None:
        return 0
    total = 0
    if manifest is not None:
        for expected in manifest.expected_files:
            if not download_manifest.expected_path_is_safe(expected.path):
                continue
            if expected.sha256 and expected.sha256 in active_partial_hashes:
                # A force or retry can leave the previous materialized file beside a replacement for the same
                # logical blob; count the current partial, not both generations.
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
    """Every file under ``root``, and whether the traversal saw all of it.

    Not ``rglob``: it suppresses every OSError raised while scanning (documented behaviour
    since 3.13), so an unreadable subtree comes back as a SHORT list that is indistinguishable
    from an empty one -- and a caller asking "is the variant here?" then answers a confident
    no about a directory it could not read. ``os.scandir`` reports the failure, and a subtree
    that is genuinely missing is not one.
    """
    files: list[Path] = []
    complete = True
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = list(scan)
        except (FileNotFoundError, NotADirectoryError):
            continue  # nothing there IS the answer, not a gap in the reading
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
                # DirEntry.is_dir/is_file raise rather than suppress, so this one is visible.
                complete = False
    return files, complete


def _variant_main_shard_present(
    snapshot_dir: Optional[Path], variant_file_matcher: Optional["VariantFileMatcher"]
) -> Optional[bool]:
    """Whether the variant's OWN files are in the snapshot dir. None when unanswerable.

    The narrower question ``companions = False`` asks: shared companions belong to every quant
    in the repo, so their presence says nothing about this one. Used on the path where the blob
    hashes could not be resolved -- the snapshot dir is still named per file, so it can settle
    absence even when the hash filter cannot, and an unreadable or absent dir stays unknown.
    """
    if snapshot_dir is None or variant_file_matcher is None:
        return None
    # An entry we could not read may BE the main shard, so a transient failure is not evidence the
    # variant is gone: a positive match elsewhere still settles it, otherwise the reading is unknown.
    entries, complete = _walk_files(snapshot_dir)
    for path in entries:
        relative = path.relative_to(snapshot_dir).as_posix()
        # A sidecar left by a deleted quant answers the quant matcher, so the job is re-adopted.
        if is_appledouble_metadata(path):
            continue
        if variant_file_matcher(relative, companions = False):
            return True
    return None if not complete else False


def _retained_snapshot_dirs(entry: Path) -> list[Path]:
    """Every snapshot the repo cache dir keeps, newest first.

    A cache can hold several revisions, and the requested variant is not always in the newest:
    reading only that one reported a complete cached quant as 0 bytes and never verified its
    manifest, so it stayed at 99% and adoptable.
    """
    try:
        snapshots = [child for child in (entry / "snapshots").iterdir() if child.is_dir()]
    except OSError:
        return []
    return sorted(snapshots, key = snapshot_selection_key, reverse = True)


def _variant_present_in_any_snapshot(
    entry: Path, variant_file_matcher: Optional["VariantFileMatcher"]
) -> Optional[bool]:
    """``_variant_main_shard_present`` over every snapshot the repo dir retains.

    True as soon as one holds the variant's own file; False only when every snapshot was read
    and none did; None when there was nothing to read or a read failed, which is unknown.
    """
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
    """Bytes the variant's files present in the snapshot dir.

    A predicate, not a file list, because this is the path where the file list
    is precisely what could not be determined. That makes it a lower bound on
    the wrong side for shared companions -- the matcher accepts every mmproj and
    drafter in the repo, while a plan fetches one of each -- so it is fit for a
    byte reading the caller clamps and displays, and not for deciding whether a
    download finished. ``stat`` follows the link, so a blob that was written but
    never linked contributes nothing, and the Windows copy layout is read as is.
    """
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return 0
    # Same false "still active" as the companion clause below, reached by a stranded sidecar.
    entries = [path for path in entries if not is_appledouble_metadata(path)]

    def _accepts(relative: str, *, companions: bool) -> bool:
        # A matcher that understands the distinction gets asked for it; only the GGUF matcher takes the keyword today.
        try:
            return bool(variant_file_matcher(relative, companions = companions))
        except TypeError:
            return bool(variant_file_matcher(relative))

    # Companions are shared by every quant in the repo, so alone they are not evidence THIS one is
    # here: a stranded companion left a positive reading that hydration re-adopts.
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
        # HF metadata names the same files a manifest does, so verify against it: a manifest never
        # written, deleted, or under an unnameable cache scope is not evidence of an unfinished download,
        # and refusing left a materialized snapshot partial forever.
        # Nothing weaker will do: expected_bytes is a catalog hint, and a blobs/ tally cannot tell this
        # variant's bytes from a sibling's.
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
    # ANY retained snapshot: the variant can be complete in an older revision while the newest holds
    # only a sibling, and checking the newest alone left that download at 99% forever.
    # An older revision can carry the same FILENAMES at the same sizes with different content and
    # verify_against_disk does not read sha256, so require the entries to resolve to known hashes;
    # with none resolved the filename check stands alone.
    for snap in snapshots:
        if not download_manifest.verify_against_disk(manifest, snap).ok:
            continue
        if not expected_hashes or _snapshot_resolves_to(manifest, snap, expected_hashes):
            return True
    return False


def _referenced_commits(entry: Path) -> "frozenset[str]":
    """Commits this repo cache dir still points at.

    HF records the commit a branch or tag resolved to in ``refs/<revision>`` on every
    snapshot_download whose revision was not already a raw sha, so for the default ``main``
    the file is always there. It is the one revision marker that survives without a manifest.
    """
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
    """Whether ``snapshot`` names a revision this cache dir has moved off.

    Only asked where there is no symlink to read. HF names a snapshot dir after its commit, so
    a dir named by neither the manifest's recorded commit nor any live ref is an older
    revision, and its same-named files are not this download's bytes. Neither marker present
    leaves the question unanswerable, and unanswerable is not a mismatch.
    """
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
    """Whether every expected file in ``snapshot`` points at one of ``expected_hashes``.

    HF names a blob by its hash and the snapshot entry links to it, so the link target settles
    which revision is materialized here. A copy-layout cache (Windows without symlinks) has no
    target to read, and neither does a reading with no manifest to name the files, so both fall
    back to dating the snapshot by revision.
    """
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
    # Always resolve the revision's blob hashes so stale blobs from a superseded revision cannot
    # inflate the count; they degrade to empty (count-all) only when metadata is unavailable (e.g.
    # offline). Take the larger total so a low caller hint cannot cap the bar.
    meta_total, expected_hashes = metadata_resolver(repo_id, hf_token)
    meta_total = max(0, meta_total)
    expected_total = max(expected_total, meta_total)

    # Without resolved hashes a variant must not count unscoped blobs, since sibling quants share one
    # blobs/ dir; a no-variant snapshot owns the whole dir and counts unscoped.
    count_unscoped = variant is None
    # An empty hash set means the expected file set could not be determined, not that the variant has
    # no bytes: model_info failing (offline, or a 401 on a gated repo) is negatively cached, so one
    # failed lookup would report a finished 33 GB variant as "0 B of 33 GB" for the whole TTL. Fall
    # back to the snapshot dir's own files.
    variant_file_set_unknown = variant is not None and not expected_hashes
    # Resolved at most once, and only if a reading gets far enough to need it.
    metadata_files: "_Lazy[tuple[download_manifest.ExpectedFile, ...]]" = _Lazy(
        lambda: (
            tuple(expected_files_resolver(repo_id, hf_token))
            if expected_files_resolver is not None
            else ()
        )
    )

    readings: list[tuple[int, int, Optional[str], bool, Optional[bool]]] = []
    # The enumeration suppresses OSError per root, so an unreadable cache root came back as "no dirs",
    # indistinguishable from a wiped cache, which hydration retires the job on.
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
        # Keyed by logical blob: a broken advisory lock leaves several writers racing on one etag, each
        # downloading the WHOLE file, so summing them overshoots.
        partial_bytes: dict[str, int] = {}
        completed_hashes: set[str] = set()
        # A partial attributable to no target is not evidence for this variant, nor against it while the
        # hashes are unresolved, and the by-name scan cannot see it.
        unattributable_partial = False
        cache_path = hf_cache_scan.resolve_hf_cache_realpath(entry)
        blobs_dir = entry / "blobs"
        # Skip a blob that vanished mid-poll rather than zeroing the reading.
        try:
            # os.stat, not Path.is_dir(): is_dir() swallows the OSError and answers False, turning a Windows ACL
            # or network-filesystem failure into a MEASURED absence hydration retires on.
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
                # An unreadable blobs dir is not an empty one: swallowing it produced a measured zero
                # (target_present false, cache_measured true) and retired a job whose cache was never read.
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
                    # A blob we could not inspect is not a blob that is not there: swallowing it produced a MEASURED
                    # absence, which hydration reads as gone.
                    scan_errors.append(exc)
                    continue
        # A finalized blobs/<hash> supersedes every partial for the same logical blob, so counting both
        # overshot the expected total and pinned a downloaded variant at 0.99 until the orphan was swept.
        for blob_hash in completed_hashes:
            partial_bytes.pop(blob_hash, None)
        # Largest wins deliberately: preferring the freshest mtime reads better against a corpse but
        # oscillates between two genuinely live writers, which is what a broken advisory lock produces.
        # A corpse should not outlive the job that made it (a terminal job sweeps its own blobs, and a
        # backend that died first is caught at boot); if one survives both, over-reading until the next
        # sweep is a smaller wrong than a reading that will not sit still.
        in_progress_bytes = sum(partial_bytes.values())
        snapshot_dirs: "_Lazy[list[Path]]" = _Lazy(
            lambda entry = entry: _retained_snapshot_dirs(entry)
        )
        entry_manifest: "_Lazy[Optional[download_manifest.Manifest]]" = _Lazy(
            lambda entry = entry: download_manifest.read_manifest(
                repo_type,
                repo_id,
                variant,
                hub_cache = entry.parent,
            )
        )
        if variant is not None:
            # The best reading across every retained snapshot, since the variant can live in an older
            # revision, and because huggingface_hub 1.18's Windows copy layout can move a completed file
            # straight into the snapshot and leave a blob-only tally at zero.
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
                    if not expected_hashes or _snapshot_resolves_to(manifest, snap, expected_hashes)
                ),
                default = 0,
            )
            # Clamped, because the matcher behind the no-manifest half accepts every companion in the repo and
            # so can overshoot.
            if expected_total > 0:
                on_disk = min(on_disk, expected_total)
            completed_bytes = max(completed_bytes, on_disk)
        # Sibling quants share one repo cache dir, so deleting a variant's files leaves the dir standing
        # and the reading came back "zero bytes, cache_path names a directory", which hydration adopts as
        # a phantom. False only on positive evidence of absence; anything less certain stays None.
        target_present: Optional[bool] = None
        if variant is not None and not variant_file_set_unknown:
            # The MATERIALIZED file, not the blob tally: deleting a snapshot symlink leaves the finalized blob
            # behind and a shared companion keeps the count positive, so a quant that is gone read as present.
            # Bytes only stand in when there is nothing readable to scan.
            scanned = _variant_present_in_any_snapshot(entry, variant_file_matcher)
            if scanned is not None:
                target_present = scanned or bool(in_progress_bytes)
            else:
                target_present = bool(completed_bytes or in_progress_bytes)
        elif variant is not None:
            # The byte reading already walked the snapshot dir, whose entries are named per file, so it can
            # answer whether a main shard of THIS quant is here; without it a repo dir kept alive by a sibling
            # read as "zero bytes, cache_path names a directory" and was adopted as a resumable phantom.
            # Across EVERY snapshot the entry retains, not only the newest: a quant living in an older
            # revision read as absent and hydration retired a job whose target is still usable.
            scanned = _variant_present_in_any_snapshot(entry, variant_file_matcher)
            if scanned is not None:
                # Unless the shared blobs/ dir holds an unattributable partial: an idle or restarted download
                # whose hashes were refused has its bytes in an .incomplete blob no snapshot links.
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
        # complete_on_disk last-but-one: two remembered caches can clamp to the SAME byte total while only
        # one has a manifest that verifies against disk, and root order then capped the response at 99%.
        key = lambda item: (item[0] + item[1], bool(item[3]), item[0]),
        default = None,
    )
    if selected is None:
        # Nothing measured AND a root that could not be listed: the cache may be entirely intact behind that
        # error, so this is unknown, not gone.
        if scan_errors:
            return _empty_progress(expected_bytes, measured = False)
        return empty

    completed_bytes, in_progress_bytes, cache_path, complete_on_disk, target_present = selected
    # Presence is a property of the SET of caches: a sibling-only repo dir and a cache still holding
    # this variant's manifest both read as zero bytes, so a positive reading anywhere wins.
    presence = [reading[4] for reading in readings]
    if any(verdict is True for verdict in presence):
        target_present = True
    elif any(verdict is None for verdict in presence):
        # Absence needs EVERY scanned cache to say so: one unknown reading is not evidence the target is gone.
        target_present = None
    downloaded_bytes = completed_bytes + in_progress_bytes
    # A reading taken while some root could not be listed is only ever a LOWER bound: the active root
    # raising EACCES/EIO while a remembered cache holds the repo dir gives target_present False and
    # zero bytes, which hydration reads as "deleted". Downgrade absence claims to unknown; a positive
    # reading is unaffected.
    scan_incomplete = bool(scan_errors)
    if scan_incomplete:
        if not target_present:
            target_present = None
    # Subtract the companion baseline only while it is still counted in completed_bytes and the
    # variant is unverified, and never when it covers the whole expected total: that leaves
    # "0 B of 0 B", which the frontend evicts as a dead job.
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
            # Zero bytes read out of an incomplete scan is not evidence of zero bytes on disk.
            "cache_measured": not scan_incomplete,
        }

    # Cap at 0.99 until the manifest-backed disk check verifies completion: on resume, completed bytes
    # can sit above the threshold while files still download.
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

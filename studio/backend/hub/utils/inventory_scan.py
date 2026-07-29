# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF cache inventory scanner.

Read-only walks of the HuggingFace hub cache plus legacy/default
cache locations. Builds the foundation that Hub inventory endpoints
and the DownloadRegistry both consume.

The worker spawn / transport-marker preparation / DownloadRegistry
layers built on top of these primitives live in download_registry.py.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

from hub.utils.gguf import (
    extract_quant_label,
    is_gguf_filename,
    is_mmproj_filename,
    is_mtp_drafter_path,
)
from hub.utils.state_dir import RepoType

from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    has_incomplete_blobs,
    hf_cache_roots,
    iter_repo_cache_dirs,
    latest_snapshot_dir,
    repo_cache_dir_has_incomplete_blobs,
)

# Inventory is invalidated explicitly on every app-driven cache mutation, so
# this TTL only bounds staleness from out-of-band edits while skipping re-walks
# on rapid UI navigation.
_HF_CACHE_SCANS_TTL_SECONDS = 15.0
_GGUF_SPLIT_RE = re.compile(r"-(\d{3,})-of-(\d{3,})(?=\.gguf$)", re.IGNORECASE)
# The transformers shard naming; each shard names the total the set needs.
_WEIGHT_SHARD_RE = re.compile(r"-(\d{3,})-of-(\d{3,})(?=\.(?:safetensors|bin)$)", re.IGNORECASE)
_hf_cache_scans_lock = threading.Lock()


@dataclass
class _HfCacheScanFlight:
    event: threading.Event
    epoch: int
    result: Optional[list] = None
    error: Optional[BaseException] = None


_hf_cache_scans_flight: Optional[_HfCacheScanFlight] = None
_hf_cache_scans_result: Optional[list] = None
_hf_cache_scans_cached_at: float = 0.0
# Bumped on every invalidation. A scan tags itself with the epoch it began
# under; an invalidation mid-scan changes the epoch so the in-flight result is
# neither cached nor served to callers that arrived after the mutation.
_hf_cache_scans_epoch: int = 0


def invalidate_hf_cache_scans() -> None:
    global _hf_cache_scans_result, _hf_cache_scans_cached_at, _hf_cache_scans_epoch
    with _hf_cache_scans_lock:
        _hf_cache_scans_result = None
        _hf_cache_scans_cached_at = 0.0
        _hf_cache_scans_epoch += 1


def all_hf_cache_scans() -> list:
    global _hf_cache_scans_flight, _hf_cache_scans_result, _hf_cache_scans_cached_at

    now = time.monotonic()
    with _hf_cache_scans_lock:
        if (
            _hf_cache_scans_result is not None
            and (now - _hf_cache_scans_cached_at) < _HF_CACHE_SCANS_TTL_SECONDS
        ):
            return list(_hf_cache_scans_result)
        start_epoch = _hf_cache_scans_epoch
        flight = _hf_cache_scans_flight
        # Only coalesce onto an in-flight scan from the current epoch; one that
        # began before an intervening invalidation is superseded so
        # post-mutation callers never receive pre-mutation data.
        if flight is None or flight.epoch != start_epoch:
            flight = _HfCacheScanFlight(event = threading.Event(), epoch = start_epoch)
            _hf_cache_scans_flight = flight
            owner = True
        else:
            owner = False

    if not owner:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return list(flight.result or [])

    try:
        scans = _compute_all_hf_cache_scans()
        with _hf_cache_scans_lock:
            flight.result = scans
            if _hf_cache_scans_epoch == flight.epoch:
                _hf_cache_scans_result = scans
                _hf_cache_scans_cached_at = time.monotonic()
        return scans
    except Exception as exc:
        with _hf_cache_scans_lock:
            if _hf_cache_scans_epoch == flight.epoch:
                _hf_cache_scans_result = None
                _hf_cache_scans_cached_at = 0.0
            flight.error = exc
        raise
    finally:
        with _hf_cache_scans_lock:
            if _hf_cache_scans_flight is flight:
                _hf_cache_scans_flight = None
            flight.event.set()


# Mirrors huggingface_hub's refs/ and snapshots/ walk skips, so a stray OS
# helper file is not mistaken for cache corruption.
_CACHE_ENTRIES_TO_IGNORE = frozenset({".DS_Store"})
_HF_REPO_TYPES = frozenset({"model", "dataset", "space"})


# Mirrors huggingface_hub's CachedFileInfo / CachedRevisionInfo / CachedRepoInfo
# field-for-field but is built here, not imported, so a field added or removed
# upstream cannot break the call; test_hf_cache_dangling_refs is the drift
# tripwire. Frozen (hashable) because HFCacheInfo.delete_revisions() keys a dict
# by repo and set-diffs ``revisions``.
@dataclass(frozen = True)
class _RecoveredFileInfo:
    file_name: str
    file_path: Path
    size_on_disk: int
    blob_path: Path
    blob_last_accessed: float
    blob_last_modified: float


@dataclass(frozen = True)
class _RecoveredRevisionInfo:
    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: frozenset
    refs: frozenset
    last_modified: float


@dataclass(frozen = True)
class _RecoveredRepoInfo:
    repo_id: str
    repo_type: str
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: frozenset
    last_accessed: float
    last_modified: float

    @property
    def refs(self) -> dict:
        return {ref: rev for rev in self.revisions for ref in rev.refs}


def _hf_repo_identity(repo_dir_name: str) -> Optional[tuple[str, str]]:
    """``models--Org--Model`` -> ``("model", "Org/Model")``, as huggingface_hub parses it."""
    if "--" not in repo_dir_name:
        return None
    repo_type, _, repo_id = repo_dir_name.partition("--")
    repo_type = repo_type[:-1]
    if repo_type not in _HF_REPO_TYPES or not repo_id:
        return None
    return repo_type, repo_id.replace("--", "/")


def _read_refs_by_commit(refs_dir: Path) -> Optional[dict[str, set[str]]]:
    """Map commit hash -> ref names under ``refs/``. None if unreadable."""
    refs_by_commit: dict[str, set[str]] = {}
    if not refs_dir.exists():
        return refs_by_commit
    if refs_dir.is_file():
        return None
    try:
        entries = sorted(refs_dir.rglob("*"))
    except OSError:
        return None
    for ref_path in entries:
        try:
            if ref_path.is_dir() or ref_path.name in _CACHE_ENTRIES_TO_IGNORE:
                continue
            commit = ref_path.read_text(encoding = "utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        # Ref names keep the platform-native separator, as huggingface_hub
        # stores them, so no posix normalisation here.
        refs_by_commit.setdefault(commit, set()).add(str(ref_path.relative_to(refs_dir)))
    return refs_by_commit


def _recover_repo_hidden_by_dangling_refs(repo_dir: Path) -> Optional[_RecoveredRepoInfo]:
    """Rebuild the scan entry for a repo dropped *solely* over leftover refs.

    ``_scan_cached_repo`` assembles every revision, then raises
    ``CorruptedCacheException`` because a ``refs/<branch>`` names a commit with
    no ``snapshots/<commit>/`` dir, so ``scan_cache_dir`` omits an intact repo.
    Studio creates that state itself: ``snapshot_download`` writes ``refs/main``
    at the live upstream sha *before* fetching the first file, never creates
    ``snapshots/<sha>/``, and no Studio caller pins ``revision`` -- no race
    needed, a patterns-match-nothing download returns having written only the
    ref. Any repo re-uploaded since download then vanishes from every inventory
    endpoint while the model picker's directory walk still lists it.

    Read-only (see test_hf_cache_dangling_refs): the offending ref is left
    exactly as it is. Returns None whenever anything *other* than leftover refs
    would have failed the upstream scan, so a corrupt repo stays omitted.
    """
    identity = _hf_repo_identity(repo_dir.name)
    if identity is None:
        return None
    repo_type, repo_id = identity
    snapshots_dir = repo_dir / "snapshots"
    refs_by_commit = _read_refs_by_commit(repo_dir / "refs")
    if refs_by_commit is None:
        return None
    try:
        if not snapshots_dir.is_dir():
            return None
        snapshot_entries = sorted(snapshots_dir.iterdir())
    except OSError:
        return None

    blob_stats: dict[Path, object] = {}
    revisions: set[_RecoveredRevisionInfo] = set()
    dangling = dict(refs_by_commit)
    for snapshot in snapshot_entries:
        if snapshot.name in _CACHE_ENTRIES_TO_IGNORE:
            continue
        try:
            if not snapshot.is_dir():
                # Upstream treats a file here as corruption; defer to it.
                return None
            entries = sorted(snapshot.rglob("*"))
        except OSError:
            return None
        files: set[_RecoveredFileInfo] = set()
        for entry in entries:
            try:
                if entry.is_dir():
                    continue
                blob_path = entry.resolve()
                stat = blob_stats.get(blob_path) or blob_path.stat()
            except OSError:
                # Broken symlink / unreadable blob: upstream raises here too.
                return None
            blob_stats[blob_path] = stat
            files.add(
                _RecoveredFileInfo(
                    file_name = entry.name,
                    file_path = entry,
                    size_on_disk = stat.st_size,
                    blob_path = blob_path,
                    blob_last_accessed = stat.st_atime,
                    blob_last_modified = stat.st_mtime,
                )
            )
        try:
            last_modified = (
                max(f.blob_last_modified for f in files) if files else snapshot.stat().st_mtime
            )
        except OSError:
            return None
        revisions.add(
            _RecoveredRevisionInfo(
                commit_hash = snapshot.name,
                snapshot_path = snapshot,
                size_on_disk = sum(blob_stats[blob].st_size for blob in {f.blob_path for f in files}),
                files = frozenset(files),
                refs = frozenset(dangling.pop(snapshot.name, set())),
                last_modified = last_modified,
            )
        )
    # No snapshot means nothing was fetched yet, so the repo is not on disk and
    # must not be reported as if it were.
    if not revisions:
        return None
    # Every ref resolved, so upstream dropped this repo for some other reason.
    if not dangling:
        return None
    try:
        repo_stats = repo_dir.stat()
    except OSError:
        return None
    return _RecoveredRepoInfo(
        repo_id = repo_id,
        repo_type = repo_type,
        repo_path = repo_dir,
        size_on_disk = sum(stat.st_size for stat in blob_stats.values()),
        nb_files = len(blob_stats),
        revisions = frozenset(revisions),
        last_accessed = (
            max((stat.st_atime for stat in blob_stats.values()), default = repo_stats.st_atime)
        ),
        last_modified = (
            max((stat.st_mtime for stat in blob_stats.values()), default = repo_stats.st_mtime)
        ),
    )


def _with_repos_hidden_by_dangling_refs(scan, cache_root: Path):
    """Add back the repos ``scan_cache_dir`` dropped over a dangling ref."""
    try:
        repo_dirs = sorted(entry for entry in cache_root.iterdir() if "--" in entry.name)
    except OSError:
        return scan
    known = getattr(scan, "repos", ())
    scanned: set[str] = set()
    for repo in known:
        try:
            scanned.add(str(Path(repo.repo_path).resolve(strict = False)))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
    recovered: list[_RecoveredRepoInfo] = []
    for repo_dir in repo_dirs:
        try:
            if str(repo_dir.resolve(strict = False)) in scanned:
                continue
            entry = _recover_repo_hidden_by_dangling_refs(repo_dir)
        except (OSError, RuntimeError, ValueError):
            continue
        if entry is None:
            continue
        logger.info(
            "Recovered HF cache repo %s hidden by a dangling ref (%d revision(s) on disk)",
            entry.repo_id,
            len(entry.revisions),
        )
        recovered.append(entry)
    if not recovered:
        return scan
    try:
        return replace(
            scan,
            repos = frozenset(known) | frozenset(recovered),
            size_on_disk = getattr(scan, "size_on_disk", 0)
            + sum(entry.size_on_disk for entry in recovered),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        # A scan shape we cannot rebuild is left untouched rather than dropped.
        logger.debug("Could not attach recovered HF cache repos: %s", exc)
        return scan


def _compute_all_hf_cache_scans() -> list:
    from huggingface_hub import scan_cache_dir

    scans: list = []
    for cache_root in hf_cache_roots():
        try:
            scan = scan_cache_dir(cache_dir = str(cache_root))
            # Only a warned-about scan can hide a repo, so a healthy cache is
            # never walked twice; getattr keeps a scan without .warnings from
            # taking the whole cache root down with it.
            if getattr(scan, "warnings", None):
                scan = _with_repos_hidden_by_dangling_refs(scan, cache_root)
            scans.append(scan)
        except Exception as exc:
            logger.warning("Could not scan HF cache %s: %s", cache_root, exc)
    return scans


def default_ref_snapshot(repo_dir: Path) -> Optional[Path]:
    """Snapshot dir that ``refs/main`` names in *repo_dir*, or ``None``.

    This is where ``from_pretrained(repo_id)`` lands, so callers compare it
    against the snapshot the row advertises: naming the repo id is only safe
    when the two hold the same payload.
    """
    ref_path = repo_dir / "refs" / "main"
    try:
        # No strip: huggingface_hub matches raw ref contents against the
        # snapshot dir name, so stray whitespace resolves nowhere.
        commit = ref_path.read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not commit:
        return None
    snapshot = repo_dir / "snapshots" / commit
    try:
        if not snapshot.is_dir():
            return None
        return snapshot.resolve()
    except (OSError, ValueError):
        return None


def default_ref_resolves_on_disk(repo_dir: Path) -> bool:
    """Whether ``refs/main`` names a snapshot that exists in *repo_dir*.

    When it does not, ``from_pretrained(repo_id)`` resolves nothing: offline
    fails outright, online silently fetches upstream HEAD instead of the
    snapshot on disk. Callers then hand out the snapshot path as the load
    identity instead of the repo id.
    """
    return default_ref_snapshot(repo_dir) is not None


def token_fingerprint(hf_token: Optional[str]) -> str:
    """16-char SHA256 prefix used as a cache-key qualifier for gated repos.

    Lets per-token size/snapshot caches refuse to serve a previously
    fetched value back to a different token (a private/gated repo's
    metadata is only valid for the credential that fetched it).
    """
    if not hf_token:
        return ""
    return hashlib.sha256(hf_token.encode()).hexdigest()[:16]


def resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Pick the most useful on-disk path for a HF cache repo dir.

    Prefers the most-recent snapshot dir (what ``from_pretrained``
    actually points at). Falls back to the cache repo root. Returns the
    resolved realpath so symlinks under ``snapshots/`` are followed back
    to ``blobs/``.
    """
    try:
        latest = latest_snapshot_dir(repo_dir)
        if latest is not None:
            return str(latest.resolve())
        return str(repo_dir.resolve())
    except Exception:
        return None


def resolve_snapshot_dir_for_scan(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Latest snapshot dir for a cache row, or the first populated HF cache root.

    Scanner-side counterpart to snapshot_download()'s return value (which the
    scanner cannot access). With a *repo_cache_dir*, returns its newest
    snapshot. Otherwise scans roots in priority order (active, legacy, default)
    and returns the newest snapshot in the first root that holds one; active is
    where snapshot_download writes, so it is authoritative. Within a root,
    picks by mtime (what from_pretrained resolves to) rather than refs/main,
    since the user may have downloaded a non-main commit.
    """
    if repo_cache_dir is not None:
        latest = latest_snapshot_dir(repo_cache_dir)
        if latest is None:
            return None
        try:
            return latest.resolve()
        except OSError:
            return None
    for repo_dir in iter_repo_cache_dirs(repo_type, repo_id):
        latest = latest_snapshot_dir(repo_dir)
        if latest is None:
            continue
        try:
            return latest.resolve()
        except OSError:
            continue
    return None


def _compose_partial(*signals: Callable[[], bool]) -> bool:
    return any(signal() for signal in signals)


def _hub_cache_for_repo_dir(repo_cache_dir: Optional[Path]) -> Optional[Path]:
    return repo_cache_dir.parent if repo_cache_dir is not None else None


def _legacy_partial(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    if repo_cache_dir is not None:
        return repo_cache_dir_has_incomplete_blobs(repo_cache_dir)
    return has_incomplete_blobs(repo_type, repo_id)


def _repo_cache_dir_incomplete_hashes(repo_cache_dir: Path) -> set[str]:
    blobs_dir = repo_cache_dir / "blobs"
    if not blobs_dir.is_dir():
        return set()
    hashes: set[str] = set()
    try:
        entries = list(blobs_dir.iterdir())
    except OSError:
        return hashes
    for blob in entries:
        try:
            if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                hashes.add(blob.name[: -len(INCOMPLETE_SUFFIX)])
        except OSError:
            continue
    return hashes


def _repo_cache_dir_has_non_gguf_broken_snapshot_symlinks(
    repo_cache_dir: Path, snapshot_dir: Optional[Path] = None
) -> bool:
    target = snapshot_dir if snapshot_dir is not None else latest_snapshot_dir(repo_cache_dir)
    if target is None:
        return False
    try:
        entries = list(target.rglob("*"))
    except OSError:
        return False
    for entry in entries:
        try:
            if not entry.is_symlink() or entry.exists():
                continue
            rel = entry.relative_to(target).as_posix()
            if is_gguf_filename(rel):
                continue
            return True
        except OSError:
            continue
    return False


def _is_latest_snapshot(repo_cache_dir: Path, snapshot_dir: Path) -> bool:
    latest = latest_snapshot_dir(repo_cache_dir)
    if latest is None:
        return False
    try:
        return latest.resolve() == snapshot_dir.resolve()
    except OSError:
        return latest == snapshot_dir


def _default_ref_names_an_absent_snapshot(repo_cache_dir: Path) -> bool:
    """Whether ``refs/main`` is present and names a commit with no snapshot dir.

    ``snapshot_download`` rewrites the ref with the resolved commit *before*
    fetching a single file and the snapshot dir appears only once the first file
    lands, so this is the window between the two. A missing ``refs/main`` is not
    the same: a repo fetched by commit hash never gets one.
    """
    ref_path = repo_cache_dir / "refs" / "main"
    try:
        commit = ref_path.read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not commit:
        return False
    try:
        return not (repo_cache_dir / "snapshots" / commit).is_dir()
    except (OSError, ValueError):
        return False


def _repo_signal_applies_to_snapshot(
    repo_cache_dir: Optional[Path], snapshot_dir: Optional[Path]
) -> bool:
    """Whether a repo-wide partial signal describes *snapshot_dir*.

    Cancel markers and ``.incomplete`` blobs carry no revision and are rewritten
    by each attempt, so both belong to the newest snapshot; a row advertising an
    older, already complete one must not inherit them and lose ``can_chat``.
    With nothing to attribute against, the signal is kept.

    A ``refs/main`` naming a commit with no directory pins that attempt to a
    revision absent from disk, so no snapshot may inherit it: leaving it on the
    newest one charged an interrupted update to the previous complete payload
    -- the very state the dangling-ref recovery restores rows from. It excuses
    only a snapshot that can actually serve the row: a pinned snapshot holding
    nothing but half a payload has no other evidence it is unfinished, and
    dropping the signal advertised files that are not on disk.
    """
    if repo_cache_dir is None or snapshot_dir is None:
        return True
    if _default_ref_names_an_absent_snapshot(repo_cache_dir):
        return _snapshot_cannot_serve_its_payload(snapshot_dir)
    return _is_latest_snapshot(repo_cache_dir, snapshot_dir)


def _gguf_variant_manifest_blob_hashes(
    repo_id: str, repo_cache_dir: Optional[Path] = None
) -> frozenset[str]:
    from hub.utils import download_manifest

    hashes: set[str] = set()
    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir)
    for variant, _path in download_manifest.iter_variant_manifests(
        "model",
        repo_id,
        hub_cache = hub_cache,
    ):
        manifest = download_manifest.read_manifest(
            "model",
            repo_id,
            variant,
            hub_cache = hub_cache,
        )
        if manifest is None:
            continue
        for expected in manifest.expected_files:
            if expected.sha256 and is_gguf_filename(expected.path):
                hashes.add(expected.sha256)
    return frozenset(hashes)


def _repo_cache_dir_has_snapshot_legacy_partial(
    repo_cache_dir: Path,
    *,
    ignored_blob_hashes: frozenset[str],
    snapshot_dir: Optional[Path] = None,
) -> bool:
    if _repo_cache_dir_has_non_gguf_broken_snapshot_symlinks(repo_cache_dir, snapshot_dir):
        return True
    # ``.incomplete`` blobs sit in ``blobs/`` with no revision, so they charge to
    # the newest snapshot only; an older complete row must not go partial (and
    # lose ``can_chat``) over a separate fetch.
    if snapshot_dir is not None and not _repo_signal_applies_to_snapshot(
        repo_cache_dir, snapshot_dir
    ):
        return False
    incomplete_hashes = _repo_cache_dir_incomplete_hashes(repo_cache_dir)
    return any(blob_hash not in ignored_blob_hashes for blob_hash in incomplete_hashes)


def _snapshot_legacy_partial(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> bool:
    if repo_type != "model":
        return _legacy_partial(repo_type, repo_id, repo_cache_dir)
    ignored_hashes = _gguf_variant_manifest_blob_hashes(repo_id, repo_cache_dir)
    if repo_cache_dir is not None:
        return _repo_cache_dir_has_snapshot_legacy_partial(
            repo_cache_dir,
            ignored_blob_hashes = ignored_hashes,
            snapshot_dir = snapshot_dir,
        )
    # No repo dir to attribute against, so the repo-wide signal is kept rather
    # than applied to the wrong root below.
    return any(
        _repo_cache_dir_has_snapshot_legacy_partial(
            entry,
            ignored_blob_hashes = ignored_hashes,
        )
        for entry in iter_repo_cache_dirs(repo_type, repo_id)
    )


def _completed_gguf_variants(snapshot_dir: Optional[Path]) -> set[str]:
    if snapshot_dir is None:
        return set()
    complete: set[str] = set()
    split_groups: dict[str, dict[int, set[int]]] = {}
    try:
        paths = list(snapshot_dir.rglob("*"))
    except OSError:
        return set()
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        rel = path.relative_to(snapshot_dir).as_posix()
        if not is_gguf_filename(rel) or is_mmproj_filename(rel) or is_mtp_drafter_path(rel):
            continue
        quant = extract_quant_label(rel)
        split = _GGUF_SPLIT_RE.search(path.name)
        if split is None:
            complete.add(quant)
            continue
        index = int(split.group(1))
        total = int(split.group(2))
        if index <= 0 or total <= 0 or index > total:
            continue
        split_groups.setdefault(quant, {}).setdefault(total, set()).add(index)
    for quant, groups in split_groups.items():
        for total, indices in groups.items():
            if indices == set(range(1, total + 1)):
                complete.add(quant)
                break
    return complete


def _offered_gguf_quants(snapshot_dir: Path) -> set[str]:
    from hub.utils.gguf import list_local_gguf_variants
    try:
        variants, _ = list_local_gguf_variants(str(snapshot_dir))
        return {v.quant for v in variants if getattr(v, "quant", None)}
    except Exception:
        return set()


def _snapshot_missing_a_weight_shard(snapshot_dir: Path) -> bool:
    """Whether a sharded non-GGUF weight set in *snapshot_dir* is missing a shard.

    ``from_pretrained`` reads every shard the set names, so one absent shard
    means the pinned snapshot cannot load. Grouped per directory and prefix
    because a snapshot may ship more than one set.
    """
    groups: dict[tuple[str, str, int], set[int]] = {}
    try:
        paths = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        match = _WEIGHT_SHARD_RE.search(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        total = int(match.group(2))
        if index <= 0 or total <= 0 or index > total:
            continue
        groups.setdefault((str(path.parent), path.name[: match.start()], total), set()).add(index)
    return any(
        indices != set(range(1, total + 1)) for (_dir, _prefix, total), indices in groups.items()
    )


def _snapshot_cannot_serve_its_payload(snapshot_dir: Optional[Path]) -> bool:
    """Whether *snapshot_dir*'s own contents prove it cannot serve a row.

    Judged on the pinned snapshot alone, with no ref, marker or manifest, since
    the dangling ref is exactly what stops being evidence. A snapshot offering
    quants is judged on those: one whole quant beside a broken one still loads
    -- the rule ``is_gguf_repo_partial`` keeps -- so the GGUF answer is
    unchanged by the shard walk below. One offering none is judged on its
    sharded safetensors/checkpoint sets. Either way a file naming its own total
    is the only proof accepted; holding nothing that names one is not proof of
    breakage, so the dangling ref still excuses that snapshot.
    """
    if snapshot_dir is None:
        return False
    offered = _offered_gguf_quants(snapshot_dir)
    if offered:
        return not (offered & _completed_gguf_variants(snapshot_dir))
    return _snapshot_missing_a_weight_shard(snapshot_dir)


def snapshot_variants_all_complete(snapshot: str) -> bool:
    """True when every quant the lister would advertise from *snapshot* is on disk.

    One complete quant is not enough: the picker enumerates the whole directory,
    so a half-downloaded split quant beside a good one still gets offered and the
    generated command asks llama-server for absent shards. Both sides label via
    ``extract_quant_label`` over snapshot-relative paths, so the sets compare.
    """
    from hub.utils.gguf import list_local_gguf_variants
    try:
        variants, _ = list_local_gguf_variants(snapshot)
        offered = {v.quant for v in variants if getattr(v, "quant", None)}
        if not offered:
            return False
        return offered <= _completed_gguf_variants(Path(snapshot))
    except Exception:
        return False


def snapshot_has_complete_variants(snapshot: str) -> bool:
    """True when at least one quant the lister would advertise from *snapshot* is
    on disk.

    Deliberately weaker than ``snapshot_variants_all_complete``: a snapshot
    mixing a whole quant with an interrupted split one still loads the whole one,
    and the lister trims the offer to that subset. Skipping it outright hides a
    complete quant behind an older revision's larger one that auto-load may not
    have the memory for. Selection and offered variants must agree on one
    directory, so every caller that pins a load id uses this predicate.
    """
    from hub.utils.gguf import list_local_gguf_variants
    try:
        variants, _ = list_local_gguf_variants(snapshot)
        offered = {v.quant for v in variants if getattr(v, "quant", None)}
        if not offered:
            return False
        return bool(offered & _completed_gguf_variants(Path(snapshot)))
    except Exception:
        return False


def snapshot_has_gguf_projector(snapshot: Optional[Path]) -> bool:
    """Whether *snapshot* itself holds a GGUF vision projector.

    Read through the same walk the variant lister reports ``has_vision`` from, so
    the row capability and the picker's flag cannot name different revisions. The
    loader's companion search does not leave the snapshot it resolved the quant
    in, so a projector fetched on its own into another revision is unreachable
    however many revisions the repo has.
    """
    if snapshot is None:
        return False
    from hub.utils.gguf import list_local_gguf_variants

    try:
        return bool(list_local_gguf_variants(str(snapshot))[1])
    except Exception:
        return False


def complete_snapshot_variants(snapshot: str) -> set[str]:
    """Quant labels in *snapshot* whose files are all on disk: the subset behind
    ``snapshot_variants_all_complete``'s all-or-nothing answer."""
    try:
        return _completed_gguf_variants(Path(snapshot))
    except (OSError, RuntimeError, ValueError):
        return set()


def _manifest_partial(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    snapshot_dir: Optional[Path] = None,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    from hub.utils import download_manifest

    manifest = download_manifest.read_manifest(
        repo_type,
        repo_id,
        variant,
        hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
    )
    if manifest is None:
        return False
    resolved = (
        snapshot_dir
        if snapshot_dir is not None
        else resolve_snapshot_dir_for_scan(repo_type, repo_id, repo_cache_dir)
    )
    if resolved is None:
        return True
    if repo_type == "model" and variant is not None:
        if download_manifest.verify_against_disk(manifest, resolved).ok:
            return False
        for candidate in _manifest_snapshot_dirs(repo_type, repo_id, repo_cache_dir):
            if candidate == resolved:
                continue
            if download_manifest.verify_against_disk(manifest, candidate).ok:
                return False
        return True
    return not download_manifest.verify_against_disk(manifest, resolved).ok


def _manifest_snapshot_dirs(
    repo_type: RepoType,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> list[Path]:
    repo_dirs = (
        [repo_cache_dir]
        if repo_cache_dir is not None
        else list(iter_repo_cache_dirs(repo_type, repo_id))
    )
    snapshots: list[Path] = []
    seen: set[str] = set()
    for repo_dir in repo_dirs:
        if repo_dir is None:
            continue
        snapshots_dir = repo_dir / "snapshots"
        try:
            if not snapshots_dir.is_dir():
                continue
            entries = list(snapshots_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                resolved = entry.resolve()
            except OSError:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            snapshots.append(resolved)
    return snapshots


def is_snapshot_partial(
    repo_type: RepoType,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> bool:
    """Repo-row partial flag for snapshot-style downloads (full-snapshot
    models — safetensors/adapter/checkpoint — and all datasets).

    Composes three signals, cheapest first:
      1. Cancel marker (single stat), charged to the newest snapshot.
      2. Snapshot-attributed legacy .incomplete blob / broken-symlink check.
      3. Manifest walk (stat per expected file under the latest snapshot).

    A manifest without a resolvable snapshot is partial: the worker got
    far enough to record expectations but did not leave a usable snapshot.

    *snapshot_dir* pins the legacy and manifest walks to the snapshot the row
    hands out as its load identity. Without it they use the newest snapshot, so a
    weightless metadata-only revision beside a complete download flags the row
    partial and ``can_chat`` goes false for a model that loads fine.

    The repo-wide manifest carries no revision either, so it is attributed like
    the marker and the ``.incomplete`` blobs: to the newest snapshot. Verifying
    it against an older pinned one compares two revisions, and any rename or size
    change between them flagged a complete, loadable row partial."""
    from hub.utils import download_manifest

    repo_signal_applies = _repo_signal_applies_to_snapshot(repo_cache_dir, snapshot_dir)
    return _compose_partial(
        lambda: repo_signal_applies
        and download_manifest.has_cancel_marker(
            repo_type,
            repo_id,
            None,
            hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
        ),
        lambda: _snapshot_legacy_partial(repo_type, repo_id, repo_cache_dir, snapshot_dir),
        lambda: repo_signal_applies
        and _manifest_partial(
            repo_type,
            repo_id,
            None,
            snapshot_dir,
            repo_cache_dir,
        ),
    )


def is_variant_partial(
    repo_id: str,
    variant: str,
    snapshot_dir: Optional[Path] = None,
    *,
    incomplete_blob_hashes: Optional[set[str]] = None,
    variant_blob_hashes: Optional[frozenset[str]] = None,
    repo_cache_dir: Optional[Path] = None,
    repo_signal_applies: bool = True,
) -> bool:
    """Per-variant partial detection. Owns its manifest, owns its marker.
    Used by the GGUF variants endpoint to flag a specific quant as broken
    without contaminating other quants in the same repo.

    snapshot_dir is an optional hint to avoid re-walking the cache when a
    caller is checking many variants of the same repo (see
    is_gguf_repo_partial for that usage).

    ``repo_signal_applies`` is the same attribution the repo-wide signals get:
    marker and manifest are keyed by (repo, variant) with no revision and are
    overwritten by the next attempt, so a caller pinning *snapshot_dir* to an
    older revision passes False rather than judge that quant by another
    revision's file list. Defaults True so the per-variant endpoint keeps
    reporting a cancelled or unfinished quant as broken."""
    from hub.utils import download_manifest
    return _compose_partial(
        lambda: repo_signal_applies
        and download_manifest.has_cancel_marker(
            "model",
            repo_id,
            variant,
            hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
        ),
        lambda: bool(
            incomplete_blob_hashes
            and variant_blob_hashes
            and incomplete_blob_hashes.intersection(variant_blob_hashes)
        ),
        lambda: repo_signal_applies
        and _manifest_partial(
            "model",
            repo_id,
            variant,
            snapshot_dir,
            repo_cache_dir,
        ),
    )


def is_gguf_repo_partial(
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    *,
    snapshot_dir: Optional[Path] = None,
) -> bool:
    """Repo-row partial flag for a GGUF repo. The inventory shows ONE row per
    GGUF repo (requires_variant=True); per-variant detail lives in
    GET /api/models/gguf-variants and uses is_variant_partial.

    *** DO NOT simplify this to "any variant partial -> repo partial" ***

    Tripwire scenario: user downloads Q8_0 fully, then starts Q4_K_M and
    cancels. Both variants share ONE inventory row. If row.partial flips True,
    _capabilities_for_format flips can_chat=False, so the user can no longer
    chat with the perfectly-good Q8_0 because of an unrelated cancelled Q4_K_M.

    Correct semantics: partial=True only when at least one variant is broken
    AND no other variant is clean. "Simplifying" to the obvious "any broken"
    form re-introduces this Q8+Q4 mixed-state regression.

    Composes signals:
      1. Cheap legacy fast-path (.incomplete blobs / broken symlinks).
      2. Per-variant manifest + marker enumeration, gated on "all broken".

    *snapshot_dir* pins all three to the snapshot the row hands out as its load
    id. Without it the completed quants come from the newest snapshot while the
    legacy walk and per-variant markers stay repo-wide, so an interrupted
    re-download flips can_chat off for the older complete quant the row
    advertises.
    """
    from hub.utils import download_manifest

    if snapshot_dir is None:
        snapshot_dir = resolve_snapshot_dir_for_scan(
            "model",
            repo_id,
            repo_cache_dir,
        )
    # Same attribution as is_snapshot_partial: .incomplete blobs, broken symlinks
    # and variant cancel markers all carry no revision, so they charge to the
    # newest snapshot.
    repo_signal_applies = _repo_signal_applies_to_snapshot(repo_cache_dir, snapshot_dir)
    has_legacy_partial = repo_signal_applies and _legacy_partial("model", repo_id, repo_cache_dir)
    complete_here = _completed_gguf_variants(snapshot_dir)
    variants: set[str] = set(complete_here)
    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir)
    for variant, _path in download_manifest.iter_variant_manifests(
        "model",
        repo_id,
        hub_cache = hub_cache,
    ):
        if (
            download_manifest.read_manifest(
                "model",
                repo_id,
                variant,
                hub_cache = hub_cache,
            )
            is not None
        ):
            variants.add(variant)
    for variant, _path in download_manifest.iter_variant_markers(
        "model",
        repo_id,
        hub_cache = hub_cache,
    ):
        if download_manifest.has_cancel_marker(
            "model",
            repo_id,
            variant,
            hub_cache = hub_cache,
        ):
            variants.add(variant)
    if not variants:
        return has_legacy_partial
    has_clean = False
    has_broken = has_legacy_partial
    for variant in variants:
        if is_variant_partial(
            repo_id,
            variant,
            snapshot_dir,
            repo_cache_dir = repo_cache_dir,
            # A quant fully present in the pinned snapshot loads from it whatever
            # a newer attempt's marker or manifest (another revision) says.
            repo_signal_applies = repo_signal_applies or variant not in complete_here,
        ):
            has_broken = True
        else:
            has_clean = True
    return has_broken and not has_clean


def partial_transport_for(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    repo_cache_dir: Optional[Path] = None,
) -> Optional[str]:
    """Transport to surface on a partial row's resume affordance.

    Prefers the cancel marker's transport, then the manifest's. The fallback
    matters for rows partial without a marker (an errored/interrupted download
    leaves the manifest but no marker) so the UI can still show HTTP-resume vs
    XET-redownload instead of the neutral retry label. ``None`` when neither is
    available."""
    from hub.utils import download_manifest

    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir)
    marker_transport = download_manifest.read_cancel_marker_transport(
        repo_type,
        repo_id,
        variant,
        hub_cache = hub_cache,
    )
    if marker_transport is not None:
        return marker_transport
    manifest = download_manifest.read_manifest(
        repo_type,
        repo_id,
        variant,
        hub_cache = hub_cache,
    )
    return manifest.transport if manifest is not None else None

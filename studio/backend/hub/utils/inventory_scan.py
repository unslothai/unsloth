# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF cache inventory scanner: read-only walks of the HuggingFace hub cache plus legacy/default cache locations, building the foundation the Hub inventory endpoints and the DownloadRegistry consume. The worker spawn / transport-marker preparation / DownloadRegistry layers built on these primitives live in download_registry.py."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import stat
import threading
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Awaitable, Callable, Hashable, NamedTuple, Optional, TypeVar

from loggers import get_logger

logger = get_logger(__name__)

from hub.utils.gguf import (
    gguf_variant_key,
    is_big_endian_gguf_path,
    is_gguf_filename,
    is_imatrix_filename,
    is_mmproj_filename,
    is_mtp_drafter_path,
)
from hub.utils.hf_tokens import (
    ANONYMOUS_CACHE_IDENTITY,
    HfTokenArg,
    is_anonymous,
    qualify_cache_identity,
)
from hub.utils.host_paths import scrub_paths, short_path_for_log
from hub.utils.state_dir import RepoType

from hub.utils.hf_cache_state import (
    has_incomplete_blobs,
    hf_cache_roots,
    incomplete_blob_hash,
    iter_repo_cache_dirs,
    latest_snapshot_dir,
    repo_cache_dir_has_incomplete_blobs,
)
from utils.paths.path_utils import drop_appledouble_metadata, is_appledouble_metadata

# Inventory is invalidated explicitly on every app-driven cache mutation, so this TTL only bounds staleness from out-of-band edits.
_HF_CACHE_SCANS_TTL_SECONDS = 15.0
_GGUF_SPLIT_RE = re.compile(r"-(\d{3,})-of-(\d{3,})(?=\.gguf$)", re.IGNORECASE)
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
# Scans carry their start epoch so a mid-scan invalidation is neither cached nor served.
_hf_cache_scans_epoch: int = 0

_T = TypeVar("_T")


async def shared_scan(
    flights: dict[Hashable, asyncio.Task[_T]], key: Hashable, factory: Callable[[], Awaitable[_T]]
) -> _T:
    """Shield same-loop callers behind one task for the same inventory key."""
    flight_key = (asyncio.get_running_loop(), key)
    flight = flights.get(flight_key)
    if flight is None or flight.done():
        flight = asyncio.create_task(factory())
        flights[flight_key] = flight

        def clear(task: asyncio.Task[_T]) -> None:
            if flights.get(flight_key) is task:
                flights.pop(flight_key, None)
            if not task.cancelled():
                task.exception()

        flight.add_done_callback(clear)
    return await asyncio.shield(flight)


def invalidate_hf_cache_scans() -> None:
    global _hf_cache_scans_result, _hf_cache_scans_cached_at, _hf_cache_scans_epoch
    with _hf_cache_scans_lock:
        _hf_cache_scans_result = None
        _hf_cache_scans_cached_at = 0.0
        _hf_cache_scans_epoch += 1


def hf_cache_scans_epoch() -> int:
    with _hf_cache_scans_lock:
        return _hf_cache_scans_epoch


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


def _cache_entries_to_ignore() -> frozenset:
    """huggingface_hub's ignore list: a stray OS file is not corruption. Read from upstream, not frozen, since newer hub versions skip more names (``Thumbs.db``, ``desktop.ini``) and hardcoding the old set made an Explorer file read as corruption. The literal is the fallback."""
    try:
        from huggingface_hub.utils import _cache_manager
        names = getattr(_cache_manager, "FILES_TO_IGNORE", None)
        if names:
            return frozenset(names)
    except Exception:
        pass
    return frozenset({".DS_Store"})


_CACHE_ENTRIES_TO_IGNORE = _cache_entries_to_ignore()
_HF_REPO_TYPES = frozenset({"model", "dataset", "space"})


# Mirrors huggingface_hub's Cached{File,Revision,Repo}Info field-for-field; frozen because
# HFCacheInfo.delete_revisions() set-diffs revisions.
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
        refs_by_commit.setdefault(commit, set()).add(str(ref_path.relative_to(refs_dir)))
    return refs_by_commit


def _recover_repo_dropped_by_scan(
    repo_dir: Path, *, scan_failed: bool = False
) -> Optional[_RecoveredRepoInfo]:
    """Recover readable revisions omitted by ``scan_cache_dir``, skipping bad entries. When the entire scan failed, include intact repos because upstream returned no inventory."""
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
    skipped = 0
    for snapshot in snapshot_entries:
        if snapshot.name in _CACHE_ENTRIES_TO_IGNORE:
            continue
        try:
            if not snapshot.is_dir():
                skipped += 1
                continue
            entries = drop_appledouble_metadata(sorted(snapshot.rglob("*")))
        except OSError:
            skipped += 1
            continue
        files: set[_RecoveredFileInfo] = set()
        for entry in entries:
            try:
                if entry.is_dir():
                    continue
                blob_path = entry.resolve()
                stat = blob_stats.get(blob_path) or blob_path.stat()
            except OSError:
                skipped += 1
                continue
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
            skipped += 1
            continue
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
    if not revisions:
        return None
    if not scan_failed and not dangling and not skipped:
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


def _with_repos_dropped_by_scan(
    scan,
    cache_root: Path,
    *,
    scan_failed: bool = False,
):
    """Add back the repos ``scan_cache_dir`` dropped over one bad entry."""
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
            entry = _recover_repo_dropped_by_scan(repo_dir, scan_failed = scan_failed)
        except (OSError, RuntimeError, ValueError):
            continue
        if entry is None:
            continue
        logger.info(
            "Recovered HF cache repo %s hidden by %s (%d revision(s) on disk)",
            entry.repo_id,
            "a failed cache scan"
            if scan_failed
            else (
                "a dangling ref" if _repo_has_a_dangling_ref(repo_dir) else "an unreadable entry"
            ),
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
        logger.debug("Could not attach recovered HF cache repos: %s", scrub_paths(exc))
        return scan


def _compute_all_hf_cache_scans() -> list:
    from huggingface_hub import HFCacheInfo, scan_cache_dir

    scans: list = []
    for cache_root in hf_cache_roots():
        try:
            scan = scan_cache_dir(cache_dir = str(cache_root))
            if getattr(scan, "warnings", None):
                scan = _with_repos_dropped_by_scan(scan, cache_root)
            scans.append(scan)
        except OSError as exc:
            # Filesystem errors can abort a whole root (e.g. Windows reparse points); recover read-only.
            logger.warning("Could not scan HF cache %s: %s", cache_root, exc)
            empty_fields = dict(size_on_disk = 0, repos = frozenset(), warnings = [])
            # huggingface_hub 1.x added this required field; older versions lack it.
            if any(field.name == "incomplete_files" for field in fields(HFCacheInfo)):
                empty_fields["incomplete_files"] = frozenset()
            recovered = _with_repos_dropped_by_scan(
                HFCacheInfo(**empty_fields),
                cache_root,
                scan_failed = True,
            )
            if recovered.repos:
                scans.append(recovered)
        except Exception as exc:
            logger.warning(
                "Could not scan HF cache %s: %s", short_path_for_log(cache_root), scrub_paths(exc)
            )
    return scans


def scan_folder_hf_caches(folder: Path) -> list[Path]:
    hub = folder / "hub"
    try:
        return [folder, hub] if hub.is_dir() else [folder]
    except OSError:
        return [folder]


def default_ref_snapshot(repo_dir: Path) -> Optional[Path]:
    """Snapshot dir that ``refs/main`` names in *repo_dir*, or ``None``. Where ``from_pretrained(repo_id)`` lands, so the repo id is safe as a load id only when this matches the snapshot the row advertises."""
    ref_path = repo_dir / "refs" / "main"
    try:
        # No strip: huggingface_hub matches raw ref contents to the dir name.
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


def token_fingerprint(hf_token: HfTokenArg) -> str:
    """16-char SHA256 prefix used as a cache-key qualifier for gated repos, so per-token size/snapshot caches refuse to serve a previously fetched value back to a different token. A forced-anonymous caller is a different credential from one that may still fall back to the ambient token, so it takes its own identity."""
    if is_anonymous(hf_token):
        return ANONYMOUS_CACHE_IDENTITY
    if not hf_token:
        return ""
    # Qualified: the same token value under a different cache authorization must not share verdicts.
    return qualify_cache_identity(hf_token, hashlib.sha256(hf_token.encode()).hexdigest()[:16])


def resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Pick the most useful on-disk path for a HF cache repo dir: the most-recent snapshot dir (what ``from_pretrained`` points at), else the cache repo root. Returns the resolved realpath so symlinks under ``snapshots/`` are followed back to ``blobs/``."""
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
    """Latest snapshot dir for a cache row, or the first populated HF cache root. Scanner-side counterpart to snapshot_download()'s return value, which the scanner cannot access. With a *repo_cache_dir*, returns its newest snapshot; otherwise scans roots in priority order (active, legacy, default) and returns the newest snapshot in the first root that holds one, active being where snapshot_download writes. Within a root, picks by mtime (what from_pretrained resolves to) rather than refs/main, since the user may have downloaded a non-main commit."""
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
            if not blob.is_file():
                continue
            blob_hash = incomplete_blob_hash(blob.name)
            if blob_hash is not None:
                hashes.add(blob_hash)
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
    """Whether ``refs/main`` is present and names a commit with no snapshot dir: the window between ``snapshot_download`` rewriting the ref and the first file landing. A *missing* ref is different, since a commit-pinned fetch never writes one."""
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


def repo_id_will_not_resolve(repo_cache_dir: Path) -> bool:
    """Whether loading *repo_cache_dir* by repo id lands on nothing. The active cache normally makes the id the right target, but not while ``refs/main`` names a commit with no directory: the load finds nothing even though a snapshot sits beside it."""
    return _default_ref_names_an_absent_snapshot(repo_cache_dir)


def default_ref_offers_no_whole_quant(repo_cache_dir: Path) -> bool:
    """Whether ``refs/main`` resolves to a snapshot whose every quant is short a shard. Loading by repo id follows that ref, so such a row needs a snapshot pinned even though the id resolves. A ref naming no snapshot belongs to ``repo_id_will_not_resolve``."""
    snapshot = default_ref_snapshot(repo_cache_dir)
    if snapshot is None:
        return False
    offered = _offered_gguf_quants(snapshot)
    if not offered:
        return False
    return not (offered & _completed_gguf_variants(snapshot))


def _repo_has_a_dangling_ref(repo_cache_dir: Path) -> bool:
    """Whether ANY ref under ``refs/`` names a commit with no snapshot dir. Every ref, because recovery is not limited to ``refs/main``; repos recovered only for unreadable entries correctly return False."""
    refs_by_commit = _read_refs_by_commit(repo_cache_dir / "refs")
    if refs_by_commit is None:
        return False
    snapshots_dir = repo_cache_dir / "snapshots"
    for commit in refs_by_commit:
        try:
            if not (snapshots_dir / commit).is_dir():
                return True
        except (OSError, ValueError):
            return False
    return False


def repo_signal_applies_to_snapshot(
    repo_cache_dir: Optional[Path], snapshot_dir: Optional[Path]
) -> bool:
    """Public form of the attribution the inventory row uses, for callers pinning a snapshot."""
    return _repo_signal_applies_to_snapshot(repo_cache_dir, snapshot_dir)


def _repo_signal_applies_to_snapshot(
    repo_cache_dir: Optional[Path],
    snapshot_dir: Optional[Path],
    *,
    quants: Optional[bool] = None,
) -> bool:
    """Whether a repo-wide partial signal describes *snapshot_dir*. Cancel markers, ``.incomplete`` blobs and the repo-wide manifest carry no revision and are rewritten by each attempt, so they belong to the newest snapshot and an older complete one must not inherit them and lose ``can_chat``; with nothing to attribute against, keep them. A ``refs/main`` naming a commit with no directory pins that attempt to a revision absent from disk, but that excuses only a snapshot that can serve the row. Keyed on ``refs/main`` alone, unlike the recovery guard: where it resolves the row loads by id and the manifest describes what that load reads, so a stale tag must not suppress it."""
    if repo_cache_dir is None or snapshot_dir is None:
        return True
    if _default_ref_names_an_absent_snapshot(repo_cache_dir):
        return _snapshot_cannot_serve_its_payload(snapshot_dir, quants = quants)
    return _is_latest_snapshot(repo_cache_dir, snapshot_dir) or (
        _snapshot_cannot_serve_its_payload(snapshot_dir, quants = quants)
    )


def _gguf_variant_manifest_blob_hashes(
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    variant_state = None,
) -> frozenset[str]:
    from hub.utils import download_manifest

    hashes: set[str] = set()
    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir)
    if variant_state is not None:
        manifests = variant_state.manifests()
    else:
        manifests = (
            (
                variant,
                download_manifest.read_manifest(
                    "model",
                    repo_id,
                    variant,
                    hub_cache = hub_cache,
                ),
            )
            for variant, _path in download_manifest.iter_variant_manifests(
                "model",
                repo_id,
                hub_cache = hub_cache,
            )
        )
    for _variant, manifest in manifests:
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
    if snapshot_dir is not None and not _repo_signal_applies_to_snapshot(
        repo_cache_dir, snapshot_dir, quants = False
    ):
        return False
    incomplete_hashes = _repo_cache_dir_incomplete_hashes(repo_cache_dir)
    return any(blob_hash not in ignored_blob_hashes for blob_hash in incomplete_hashes)


def _snapshot_legacy_partial(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
    variant_state = None,
) -> bool:
    if repo_type != "model":
        return _legacy_partial(repo_type, repo_id, repo_cache_dir)
    ignored_hashes = _gguf_variant_manifest_blob_hashes(
        repo_id,
        repo_cache_dir,
        variant_state,
    )
    if repo_cache_dir is not None:
        return _repo_cache_dir_has_snapshot_legacy_partial(
            repo_cache_dir,
            ignored_blob_hashes = ignored_hashes,
            snapshot_dir = snapshot_dir,
        )
    return any(
        _repo_cache_dir_has_snapshot_legacy_partial(
            entry,
            ignored_blob_hashes = ignored_hashes,
        )
        for entry in iter_repo_cache_dirs(repo_type, repo_id)
    )


_UNJUDGEABLE_FAMILY = object()


def _completed_gguf_variants(snapshot_dir: Optional[Path]) -> set[str]:
    if snapshot_dir is None:
        return set()
    from hub.utils.gguf import _resolve_gguf_dir

    snapshot_dir = _resolve_gguf_dir(snapshot_dir) or snapshot_dir
    split_groups: dict[str, dict[tuple[str, str, int], set[int]]] = {}
    # Lister and loader both take the lexicographically first file, hence the sort.
    selected: dict[str, object] = {}
    try:
        paths = sorted(snapshot_dir.rglob("*"))
    except OSError:
        return set()
    for path in paths:
        try:
            if not path.is_file():
                continue
            empty = path.stat().st_size <= 0
        except OSError:
            continue
        rel = path.relative_to(snapshot_dir).as_posix()
        if (
            not is_gguf_filename(rel)
            or is_mmproj_filename(rel)
            or is_mtp_drafter_path(rel)
            or is_imatrix_filename(rel)
        ):
            continue
        if is_appledouble_metadata(path):
            continue
        quant = gguf_variant_key(rel)
        # A big-endian build cannot vouch for a quant; loader's label, as the extractors disagree.
        from utils.models.model_config import _extract_quant_label as _loader_quant

        if is_big_endian_gguf_path(rel, _loader_quant(rel)):
            continue
        if empty:
            selected.setdefault(quant, _UNJUDGEABLE_FAMILY)
            continue
        split = _GGUF_SPLIT_RE.search(path.name)
        if split is None:
            selected.setdefault(quant, None)
            continue
        index = int(split.group(1))
        total = int(split.group(2))
        if index <= 0 or total <= 0 or index > total:
            selected.setdefault(quant, _UNJUDGEABLE_FAMILY)
            continue
        family = (
            path.parent.relative_to(snapshot_dir).as_posix(),
            path.name[: split.start()],
            total,
        )
        split_groups.setdefault(quant, {}).setdefault(family, set()).add(index)
        selected.setdefault(quant, family)
    complete: set[str] = set()
    for quant, family in selected.items():
        if family is None:
            complete.add(quant)
        elif isinstance(family, tuple):
            if _shard_family_is_whole(family, split_groups.get(quant, {}).get(family) or set()):
                complete.add(quant)
    return complete


def _offered_gguf_quants(snapshot_dir: Path) -> set[str]:
    from hub.utils.gguf import list_local_gguf_variants
    try:
        variants, _ = list_local_gguf_variants(str(snapshot_dir))
        return {v.quant for v in variants if getattr(v, "quant", None)}
    except Exception:
        return set()


_LOADER_WEIGHT_NAMES = {
    "base": {".safetensors": "model.safetensors", ".bin": "pytorch_model.bin"},
    "adapter": {".safetensors": "adapter_model.safetensors", ".bin": "adapter_model.bin"},
}


def _weight_family_kind(name: str) -> Optional[str]:
    """``"base"``, ``"adapter"``, or ``None`` for a training artefact such as ``optimizer.bin`` that no row loads, so an auxiliary set is never counted as a runnable family."""
    # Local import: hub.services imports hub.utils.
    from hub.services.models.common import (
        _is_adapter_weight_name,
        _is_transformers_bin_weight_name,
        _is_transformers_safetensors_weight_name,
    )

    if _is_adapter_weight_name(name):
        return "adapter"
    if _is_transformers_safetensors_weight_name(name) or _is_transformers_bin_weight_name(name):
        return "base"
    return None


class _SnapshotPayload(NamedTuple):
    """What one snapshot directory can load on its own."""

    model_format: Optional[str]
    groups: dict
    whole: dict
    unreadable_config_formats: frozenset
    unloadable_families: frozenset
    invisible_families: frozenset
    ungrouped: frozenset
    empty_ungrouped: frozenset
    empty_whole: dict
    nested: frozenset
    root_indexes: frozenset
    unusable_root_indexes: frozenset
    unreachable_root: frozenset


def _root_file_is_empty(snapshot_dir: Path, name: str) -> Optional[bool]:
    """None when *name* is not a file at the root, else whether it is zero bytes. The loaders open exact paths, so the filesystem answers the case question: a Config.json serves config.json on a case-insensitive volume and not on a case-sensitive one, and lowercasing every basename got that wrong in one direction or the other."""
    path = snapshot_dir / name
    try:
        if not path.is_file():
            return None
        return path.stat().st_size <= 0
    except OSError:
        return None


def _required_config_is_unreadable(path: Path, empty: bool) -> bool:
    """Whether a config the loader requires is one it cannot use. Empty is the obvious case, but from_pretrained parses this file before it looks at a single weight, so a truncated or non-object one fails the load just as surely."""
    if empty:
        return True
    try:
        with path.open("rb") as handle:
            return not isinstance(json.load(handle), dict)
    except (OSError, ValueError):
        return True


def _shard_family_is_whole(family: tuple, indices: set) -> bool:
    """Whether *indices* is exactly 1..total for *family*, compared without building the range: the total comes out of a filename, so it can name a set far larger than anything on disk."""
    total = family[2]
    return bool(indices) and len(indices) == total and min(indices) == 1 and max(indices) == total


def _weight_shard_family(snapshot_dir: Path, path: Path, match) -> tuple:
    """The key grouping ``path`` with its siblings. The suffix is part of the key: a .safetensors shard and a .bin shard sharing a prefix and a total are different sets, and merging them made two half sets look like one whole one. Numbering that cannot be satisfied (index 0, or one past the total) still names a family, since dropping it left a snapshot that classifies by filename with no family to be short of."""
    return (
        path.parent.relative_to(snapshot_dir).as_posix(),
        path.name[: match.start()],
        int(match.group(2)),
        path.suffix.lower(),
    )


def _snapshot_payload(snapshot_dir: Path) -> Optional[_SnapshotPayload]:
    """Classify *snapshot_dir* from its own contents, or None if it cannot be read. Untrusted, like ``_repo_non_gguf_model_payload``'s per-revision pass: a directory serves a load only if it holds the config its format needs. Presence is by filename, so an empty config still classifies and is reported separately."""
    from hub.services.models.common import (
        _classify_non_gguf_model_format,
        _is_adapter_weight_name,
        _is_checkpoint_weight_name,
        _is_discoverable_ungrouped_weight_name,
        _is_training_artefact_name,
        _is_transformers_safetensors_weight_name,
    )

    flags = dict.fromkeys(
        (
            "has_config",
            "has_adapter_config",
            "has_adapter_weights",
            "has_safetensors",
            "has_transformers_safetensors",
            "has_checkpoint_weights",
        ),
        False,
    )
    groups: dict[str, dict[tuple[str, str, int, str], set[int]]] = {"base": {}, "adapter": {}}
    whole: dict[str, set[str]] = {"base": set(), "adapter": set()}
    shard_names: dict[tuple[str, str, int, str], set[str]] = {}
    ungrouped: set[str] = set()
    empty_whole: dict[str, set[str]] = {"base": set(), "adapter": set()}
    empty_ungrouped: set[str] = set()
    nested: set[str] = set()
    unreachable_root: set[str] = set()
    unreadable: set[str] = set()
    try:
        paths = drop_appledouble_metadata(list(snapshot_dir.rglob("*")))
    except OSError:
        return None
    for path in paths:
        try:
            if not path.is_file():
                continue
            empty = path.stat().st_size <= 0
        except OSError:
            continue
        name = path.name.lower()
        if is_gguf_filename(name):
            continue
        at_root = path.parent == snapshot_dir
        if name in ("config.json", "adapter_config.json"):
            continue
        if empty:
            empty_kind = _weight_family_kind(path.name)
            empty_match = _WEIGHT_SHARD_RE.search(path.name)
            if empty_kind is None:
                if (
                    not _is_adapter_weight_name(name)
                    and not _is_training_artefact_name(name)
                    and (name.endswith(".safetensors") or _is_checkpoint_weight_name(name))
                ):
                    if at_root:
                        empty_ungrouped.add("base")
                    else:
                        nested.add("base")
                continue
            if empty_match is None:
                if at_root:
                    empty_whole[empty_kind].add(name)
                else:
                    nested.add(empty_kind)
                continue
            empty_family = _weight_shard_family(snapshot_dir, path, empty_match)
            groups[empty_kind].setdefault(empty_family, set())
            shard_names.setdefault(empty_family, set()).add(path.name)
            continue
        is_adapter = _is_adapter_weight_name(name)
        base_evidence = False
        if is_adapter:
            flags["has_adapter_weights"] = True
        elif name.endswith(".safetensors") and not _is_training_artefact_name(name):
            flags["has_safetensors"] = True
            base_evidence = True
            if _is_transformers_safetensors_weight_name(name):
                flags["has_transformers_safetensors"] = True
        if _is_checkpoint_weight_name(name) and not _is_training_artefact_name(name):
            flags["has_checkpoint_weights"] = True
            base_evidence = not is_adapter
        kind = _weight_family_kind(path.name)
        if kind is None:
            if base_evidence:
                if not at_root:
                    nested.add("base")
                elif _is_discoverable_ungrouped_weight_name(name):
                    ungrouped.add("base")
                else:
                    unreachable_root.add("base")
            continue
        match = _WEIGHT_SHARD_RE.search(path.name)
        if match is None:
            if at_root:
                whole[kind].add(name)
            else:
                nested.add(kind)
            continue
        family = _weight_shard_family(snapshot_dir, path, match)
        if not at_root:
            nested.add(kind)
        groups[kind].setdefault(family, set()).add(int(match.group(1)))
        shard_names.setdefault(family, set()).add(path.name)
    for config_name, formats in (
        ("config.json", ("safetensors", "checkpoint")),
        ("adapter_config.json", ("adapter",)),
    ):
        config_empty = _root_file_is_empty(snapshot_dir, config_name)
        if config_empty is None:
            continue
        flags["has_config" if config_name == "config.json" else "has_adapter_config"] = True
        if _required_config_is_unreadable(snapshot_dir / config_name, config_empty):
            unreadable.update(formats)
    model_format = _classify_non_gguf_model_format(**flags, trusted_hf_cache_repo = False)
    unloadable: set = set()
    invisible: set = set()
    root_indexes: set[str] = set()
    unusable_root_indexes: set[str] = set()
    parsed: dict[Path, object] = {}

    def index_shards(index_path: Path):
        if index_path not in parsed:
            parsed[index_path] = _read_index_shards(index_path)
        return parsed[index_path]

    for suffix, canonical in _LOADER_WEIGHT_NAMES["base"].items():
        index_path = snapshot_dir / f"{canonical}.index.json"
        try:
            if not index_path.is_file():
                continue
        except OSError:
            continue
        root_indexes.add(suffix)
        if _index_cannot_serve_its_shards(index_path, set(), index_shards(index_path)):
            unusable_root_indexes.add(suffix)
    for family in groups["base"]:
        index_path = (
            snapshot_dir / family[0] / f"{_LOADER_WEIGHT_NAMES['base'][family[3]]}.index.json"
        )
        try:
            present = index_path.is_file()
        except OSError:
            present = False
        if not present:
            invisible.add(family)
        elif _index_cannot_serve_its_shards(
            index_path, shard_names.get(family, set()), index_shards(index_path)
        ):
            unloadable.add(family)
    invisible |= set(groups["adapter"])
    return _SnapshotPayload(
        model_format,
        groups,
        whole,
        frozenset(unreadable),
        frozenset(unloadable),
        frozenset(invisible),
        frozenset(ungrouped),
        frozenset(empty_ungrouped),
        empty_whole,
        frozenset(nested),
        frozenset(root_indexes),
        frozenset(unusable_root_indexes),
        frozenset(unreachable_root),
    )


def _index_cannot_serve_its_shards(
    index_path: Path,
    family_files: set[str],
    shards: object = None,
) -> bool:
    """Whether *index_path* would fail to hand ``from_pretrained`` the family in *family_files*. Existing is not enough: the loader parses it and opens every ``weight_map`` name, so a truncated index or one naming a shard never written is as unloadable as no index at all, and the map is also the only list of files read, so one covering part of the numbered family silently drops the rest."""
    if shards is None:
        shards = _read_index_shards(index_path)
    if shards is False:
        return True
    named = {shard.name for shard in shards}
    # Coverage matters only for the family this index describes: the loader opens weight_map and nothing else.
    return not family_files <= named and not named.isdisjoint(family_files)


def _read_index_shards(index_path: Path):
    """The shards *index_path* names if each is a non-empty file and every numbered set is whole, else False."""
    try:
        if not index_path.is_file() or index_path.stat().st_size <= 0:
            return False
        with index_path.open(encoding = "utf-8") as handle:
            index = json.load(handle)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        # RecursionError escapes every caller's fail-open guard, and the loader parses this index with the same json module, so one too deep to parse there cannot serve its shards here either.
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    # Per unique shard, not per tensor: a MoE index maps tens of thousands of tensors onto a few dozen files.
    try:
        names = set(weight_map.values())
    except TypeError:
        return False
    shards: set[PurePosixPath] = set()
    for shard in names:
        if not isinstance(shard, str) or not shard:
            return False
        # Names are relative to the index: anything reaching outside is not a shard of this family.
        parts = PurePosixPath(shard.replace("\\", "/"))
        # PurePosixPath sees 'C:/x' as relative, but on Windows the join replaces the base dir.
        windows = PureWindowsPath(shard)
        if parts.is_absolute() or ".." in parts.parts or windows.is_absolute() or windows.drive:
            return False
        shards.add(parts)
    for shard in shards:
        try:
            shard_stat = (index_path.parent / shard).stat()
            if not stat.S_ISREG(shard_stat.st_mode) or shard_stat.st_size <= 0:
                return False
        except (OSError, ValueError):
            return False
    # A shard names its own total, so an index listing one of a set has to list the whole set: the loader silently drops whatever the map leaves out.
    declared: dict[tuple[str, str, int], set[int]] = {}
    for shard in shards:
        match = _WEIGHT_SHARD_RE.search(shard.name)
        if match is None:
            continue
        key = (str(shard.parent), shard.name[: match.start()], int(match.group(2)))
        declared.setdefault(key, set()).add(int(match.group(1)))
    if any(total <= 0 or len(seen) < total for (_d, _p, total), seen in declared.items()):
        return False
    return shards


def _snapshot_lacks_a_complete_weight_family(snapshot_dir: Path) -> bool:
    """Whether the payload *snapshot_dir* carries is short a shard. ``from_pretrained`` loads one family, so a whole safetensors set beside an interrupted ``.bin`` one still serves the row. Only the family the classified format names is judged: a snapshot can hold both a config.json and an adapter_config.json, so the config alone does not say which."""
    payload = _snapshot_payload(snapshot_dir)
    if payload is None:
        return False
    if payload.model_format in payload.unreadable_config_formats:
        return True
    wanted = "adapter" if payload.model_format == "adapter" else "base"
    other = "base" if wanted == "adapter" else "adapter"
    order = (wanted, other)
    for kind in order:
        # Both loaders try safetensors before pickle and never fall back, so judge in that order.
        unreachable = False
        for suffix in (".safetensors", ".bin"):
            selected = _LOADER_WEIGHT_NAMES[kind][suffix]
            canonical_empty = _root_file_is_empty(snapshot_dir, selected)
            if canonical_empty is True:
                return kind == wanted or wanted not in payload.ungrouped
            if canonical_empty is False:
                return kind != wanted and wanted not in payload.ungrouped
            if kind == "base" and suffix in payload.root_indexes:
                if suffix in payload.unusable_root_indexes:
                    return kind == wanted or wanted not in payload.ungrouped
                return kind != wanted and wanted not in payload.ungrouped
            families = {
                family: indices
                for family, indices in payload.groups[kind].items()
                if family[3] == suffix and family[0] in ("", ".")
            }
            if not families:
                if any(
                    root.endswith(suffix)
                    for root in payload.whole[kind] | payload.empty_whole[kind]
                ):
                    unreachable = True
                continue
            if all(family in payload.invisible_families for family in families):
                unreachable = True
                continue
            return all(
                not _shard_family_is_whole(family, indices) or family in payload.unloadable_families
                for family, indices in families.items()
            )
        if unreachable or (
            kind == wanted and kind in payload.nested and kind not in payload.ungrouped
        ):
            return True
    return (
        wanted in payload.empty_ungrouped or wanted in payload.unreachable_root
    ) and wanted not in payload.ungrouped


def _snapshot_cannot_serve_its_payload(
    snapshot_dir: Optional[Path], *, quants: Optional[bool] = None
) -> bool:
    """Whether *snapshot_dir*'s own contents prove it cannot serve a row. Judged on the pinned snapshot alone, since the dangling ref is exactly what stopped being evidence. Quants are judged on quants, otherwise on safetensors/checkpoint families, under ``is_gguf_repo_partial``'s rule: one complete quant or family is enough, and only a file naming its own total counts as proof. *quants* picks the row's format in a hybrid repo (True quants, False families, None either); otherwise a torn quant vetoes a whole weights row, and a whole one vouches for a torn family."""
    if snapshot_dir is None:
        return False
    if quants is not False:
        offered = _offered_gguf_quants(snapshot_dir)
        if offered:
            return not (offered & _completed_gguf_variants(snapshot_dir))
        if quants:
            return False
    return _snapshot_lacks_a_complete_weight_family(snapshot_dir)


def _recovered_snapshot_cannot_serve(
    repo_cache_dir: Optional[Path],
    snapshot_dir: Optional[Path],
    *,
    quants: Optional[bool] = None,
) -> bool:
    """Partial signal for a snapshot the dangling-ref recovery put back on a row. The interrupted attempt that wrote the ref may have left nothing else behind, so marker, manifest and ``.incomplete``/broken-symlink all read false and a snapshot short a shard looks runnable; its contents are the only evidence left. Scoped to the dangling case, since where the ref resolves upstream already publishes the row. Other recovered cases retain their own evidence: broken links mark the repo partial, while a stray snapshot file never represented a revision."""
    if repo_cache_dir is None or snapshot_dir is None:
        return False
    if not _repo_has_a_dangling_ref(repo_cache_dir):
        return False
    return _snapshot_cannot_serve_its_payload(snapshot_dir, quants = quants)


def snapshot_holds_a_complete_payload(
    snapshot_dir: Optional[Path], *, quants: Optional[bool] = None
) -> bool:
    """Whether *snapshot_dir* can serve a load from its own contents alone. The selection-side counterpart to the partial check: a row picks the newest snapshot that classifies, and filename-only classification cannot tell a whole payload from one short a shard, so without this a broken newer revision hides a complete older one."""
    if snapshot_dir is None:
        return False
    return not _snapshot_cannot_serve_its_payload(snapshot_dir, quants = quants)


def recovered_repo_is_unusable_by_repo_id(repo_info) -> bool:
    """Whether a recovered repo is one a caller that can only say ``repo_id`` must skip. The Hub inventory carries ``partial`` and a ``load_id``, so it describes these rows honestly; the compatibility ``/api/models/cached-models`` schema has neither, so an unusable recovery reads there as a plain cached model. False for every repo upstream already returns, so this only withholds rows this recovery added."""
    if not isinstance(repo_info, _RecoveredRepoInfo):
        return False
    repo_path = getattr(repo_info, "repo_path", None)
    if repo_path is None:
        return True
    landing = default_ref_snapshot(repo_path)
    if landing is None:
        return True
    if _snapshot_cannot_serve_its_payload(landing):
        return True
    if _offered_gguf_quants(landing):
        return False
    payload = _snapshot_payload(landing)
    return payload is None or payload.model_format is None


def snapshot_variants_all_complete(snapshot: str) -> bool:
    """True when every quant the lister would advertise from *snapshot* is on disk. One complete quant is not enough: the picker enumerates the whole directory, so a half-downloaded split quant beside a good one still gets offered."""
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
    """True when at least one quant the lister advertises from *snapshot* is on disk. Weaker than ``snapshot_variants_all_complete`` on purpose: a snapshot mixing a whole quant with an interrupted split one still loads the whole one, and the lister trims the offer to that subset. Every load-id pin uses this, so selection and offered variants name one directory."""
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
    """Whether *snapshot* itself holds a GGUF vision projector. Same walk the variant lister reports ``has_vision`` from, so row capability and picker flag cannot name different revisions."""
    if snapshot is None:
        return False
    from hub.utils.gguf import list_local_gguf_variants

    try:
        return bool(list_local_gguf_variants(str(snapshot))[1])
    except Exception:
        return False


def complete_snapshot_variants(snapshot: str) -> set[str]:
    """Quant labels in *snapshot* whose files are all on disk."""
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
    variant_state = None,
) -> bool:
    from hub.utils import download_manifest

    manifest = (
        variant_state.manifest_for(variant)
        if variant_state is not None and variant is not None
        else download_manifest.read_manifest(
            repo_type,
            repo_id,
            variant,
            hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
        )
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
    variant_state = None,
) -> bool:
    """Repo-row partial flag for snapshot-style downloads (full-snapshot models, i.e. safetensors/adapter/checkpoint, and all datasets).

    Composes four signals, cheapest first: the cancel marker (single stat, charged to the newest snapshot); the snapshot-attributed legacy .incomplete blob / broken-symlink check; the manifest walk (stat per expected file under the latest snapshot); and a recovered snapshot short a shard, for the dangling-ref rows the first three cannot judge.

    A manifest without a resolvable snapshot is partial: the worker recorded expectations but left no usable snapshot. *snapshot_dir* pins the legacy and manifest walks to the row's load identity; without it a metadata-only revision beside a complete download flags the row partial. Repo-wide signals are attributed by ``_repo_signal_applies_to_snapshot``.
    """
    from hub.utils import download_manifest

    repo_signal_applies = _repo_signal_applies_to_snapshot(
        repo_cache_dir, snapshot_dir, quants = False
    )
    return _compose_partial(
        lambda: (
            repo_signal_applies
            and download_manifest.has_cancel_marker(
                repo_type,
                repo_id,
                None,
                hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
            )
        ),
        lambda: _snapshot_legacy_partial(
            repo_type,
            repo_id,
            repo_cache_dir,
            snapshot_dir,
            variant_state,
        ),
        lambda: (
            repo_signal_applies
            and _manifest_partial(
                repo_type,
                repo_id,
                None,
                snapshot_dir,
                repo_cache_dir,
            )
        ),
        lambda: _recovered_snapshot_cannot_serve(repo_cache_dir, snapshot_dir, quants = False),
    )


def _current_revisions(repo_info):
    """The revisions to judge a cached repo by: just the one the loader will actually open. ``from_pretrained`` resolves the newest snapshot by mtime (:func:`latest_snapshot_dir`), so a repo cached twice (an older complete snapshot plus a newer companion-only scoped one) must be judged on the newer one. Scanning every revision let the old snapshot's denoiser satisfy the completeness check, so the row read as complete while the snapshot the loader picks has no transformer/unet, and offline loads then failed while online loads silently pulled the multi-GB weight. Falls back to the newest revision by ``last_modified``, then to every revision."""
    revisions = list(getattr(repo_info, "revisions", ()) or ())
    if len(revisions) <= 1:
        return revisions
    repo_path = getattr(repo_info, "repo_path", None)
    if repo_path is not None:
        latest = latest_snapshot_dir(Path(repo_path))
        if latest is not None:
            scoped = [
                rev
                for rev in revisions
                if getattr(rev, "snapshot_path", None) is not None
                and Path(rev.snapshot_path) == latest
            ]
            if scoped:
                return scoped
    dated = [rev for rev in revisions if getattr(rev, "last_modified", None) is not None]
    if dated:
        return [max(dated, key = lambda rev: rev.last_modified)]
    return revisions


_DENOISER_DIRS = ("transformer", "unet")
_DENOISER_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_DEFAULT_DENOISER_WEIGHTS = frozenset(
    f"diffusion_pytorch_model{suffix}" for suffix in _DENOISER_WEIGHT_SUFFIXES
)
# The only sharded index a default load resolves is SAFE_WEIGHTS_INDEX_NAME with no variant.
_SELECTED_DENOISER_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def snapshot_has_pipeline_index(snapshot: Optional[Path]) -> bool:
    """Whether *snapshot* carries a conventional or modular root pipeline index. The snapshot-scoped twin of :func:`repo_has_pipeline_index`, for callers that already know the ONE directory their row loads from: ``from_pretrained`` reads the manifest at the root of the revision it resolves, so a sibling revision's manifest says nothing about this row."""
    if snapshot is None:
        return False
    try:
        root = Path(snapshot)
        return (root / "model_index.json").is_file() or (
            root / "modular_model_index.json"
        ).is_file()
    except OSError:
        return False


def _manifest_denoiser_components(snapshot: Path) -> Optional[tuple[str, ...]]:
    """The denoiser subdirs this pipeline's root manifest declares, or None. Read off the manifest rather than the fixed ``_DENOISER_DIRS`` pair because multi-DiT pipelines carry more than one (Ideogram 4 adds ``unconditional_transformer/``, Wan 2.2's A14B experts ``transformer_2/``) and would otherwise pass on whichever the loop reached first. None means the manifest could not be READ and the caller keeps the fixed-pair rule; an EMPTY tuple means it read fine and names no denoiser under either spelling (Stable Cascade and Wuerstchen call theirs ``decoder``/``prior``), so there is nothing to prove absent and the caller must not hunt for directories that layout never had."""
    try:
        manifest_path = snapshot / "model_index.json"
        if not manifest_path.is_file():
            manifest_path = snapshot / "modular_model_index.json"
        with manifest_path.open("r", encoding = "utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(manifest, dict):
        return None
    found = []
    for key, value in manifest.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        name = key.lower()
        if name != "unet" and "transformer" not in name:
            continue
        if not isinstance(value, (list, tuple)) or not any(v for v in value):
            continue
        found.append(key)
    return tuple(found)


def _denoiser_index_shards(index: Path) -> Optional[set[str]]:
    """The shard names *index* maps, or None when it is absent, unparseable or maps nothing. None is "no evidence" rather than "incomplete": an index we cannot read proves nothing either way, so the caller keeps looking."""
    try:
        with index.open("r", encoding = "utf-8") as fh:
            weight_map = json.load(fh).get("weight_map")
    except (OSError, ValueError, AttributeError, RecursionError):
        return None
    if not isinstance(weight_map, dict):
        return None
    return {str(v) for v in weight_map.values() if v} or None


def _component_weights_complete(component: Path) -> bool:
    """Whether *component* holds a denoiser the loader could actually read.

    Presence of ONE weight file is not enough: a sharded denoiser is described by an ``*.index.json`` naming every shard, and a fetch that landed shard 1 of 2 alone satisfies a first-match test while failing at load.

    ``_SELECTED_DENOISER_INDEX`` settles the question whenever it EXISTS: it is the only sharded name diffusers resolves here, its presence alone makes the component sharded, and what follows is unconditional, since ``_get_checkpoint_shard_files`` opens exactly what it maps while the ``except IOError`` branch and the pickle fallback are both gated on ``not is_sharded``. So a short set fails, and so does an index too corrupt to parse.

    Every OTHER index vouches for nothing, because a default load never opens it: ``_fetch_index_file`` builds the safetensors name only, and the non-sharded fallback asks for the UNSHARDED ``diffusion_pytorch_model.bin``, never a ``.bin.index.json`` set beside it. Repos ship exactly that leftover (``stablediffusionapi/sdrealdream``), and our own download plan manufactures it. Nor does a dtype variant, whole or not: ``_add_variant`` inserts the variant before the last part, and a load passing no ``variant`` asks for the plain name with no fallback the other way.

    So with no selected index there are exactly two names left, the pair ``_get_model_file`` is handed.
    """
    # iterdir(), not glob(): glob would swallow the OSError and read as 'no weights'.
    next(component.iterdir(), None)
    selected = component / _SELECTED_DENOISER_INDEX
    if selected.is_file():
        if _index_cannot_serve_its_shards(selected, set()):
            return False
        shards = _denoiser_index_shards(selected)
        return bool(shards) and all(
            name.lower().endswith(_DENOISER_WEIGHT_SUFFIXES) for name in shards
        )
    return any((component / name).is_file() for name in _DEFAULT_DENOISER_WEIGHTS)


def snapshot_pipeline_missing_denoiser(snapshot: Optional[Path]) -> bool:
    """The companion-only-prefetch check scoped to ONE snapshot dir. Same signal as :func:`repo_pipeline_missing_denoiser` (a root ``model_index.json`` with no usable weights under the pipeline's denoiser components) but judged on the directory the caller's row actually loads: two snapshot selectors disagree, and a row pinned to a complete revision would otherwise be marked partial by a newer companion-only one beside it. Stricter than the repo-wide twin, since this decides whether a row is advertised as runnable: EVERY denoiser the manifest declares must be present, each with every shard its index names. Best-effort: a read error reports not-missing."""
    if not snapshot_has_pipeline_index(snapshot):
        return False
    try:
        root = Path(snapshot)
        declared = _manifest_denoiser_components(root)
        if declared is not None:
            # all(()) is True: a manifest declaring no denoiser has none to prove absent, so it reads complete rather than being hunted for one it never had.
            return not all(
                (root / name).is_dir() and _component_weights_complete(root / name)
                for name in declared
            )
        return not any(
            (root / name).is_dir() and _component_weights_complete(root / name)
            for name in _DENOISER_DIRS
        )
    except OSError:
        return False


def repo_has_pipeline_index(repo_info) -> bool:
    """Whether the cached snapshot carries a ROOT model_index.json, i.e. is loadable as a full diffusers pipeline (from_pretrained reads only the repo root). A nested subdir/model_index.json does not count: loading the repo root still fails, so the row must keep its single_file flag. CachedFileInfo.file_name is the basename, so a name match alone would also claim nested copies; scope by file_path when the scan provides it."""
    try:
        for rev in _current_revisions(repo_info):
            snapshot = getattr(rev, "snapshot_path", None)
            for f in rev.files:
                name = str(getattr(f, "file_name", "") or "")
                path = getattr(f, "file_path", None)
                if path is not None and snapshot is not None:
                    p = Path(path)
                    if p.name == "model_index.json" and p.parent == Path(snapshot):
                        return True
                elif name == "model_index.json":
                    return True
    except Exception:
        pass
    return False


def repo_pipeline_missing_denoiser(repo_info) -> bool:
    """True for a diffusers-pipeline snapshot (root ``model_index.json``) whose denoiser component (``transformer/`` or ``unet/``) carries NO weight file: the shape of a companion-only prefetch where a GGUF image load pulled the base repo's VAE / text-encoder / manifest but skipped the multi-GB transformer. :func:`is_snapshot_partial` misses this, since every file the manifest expected did arrive, so callers OR the two signals together. Best-effort: any scan error reports not-missing. Judged by :func:`snapshot_pipeline_missing_denoiser` on the revision the loader opens, so the compatibility ``/api/models/cached-models`` listing and the Hub inventory agree on a row; the walk below it is only for a scan that records no ``snapshot_path``."""
    if not repo_has_pipeline_index(repo_info):
        return False
    try:
        revisions = list(_current_revisions(repo_info))
        scoped = [getattr(rev, "snapshot_path", None) for rev in revisions]
        scoped = [snapshot for snapshot in scoped if snapshot is not None]
        if scoped:
            return all(snapshot_pipeline_missing_denoiser(Path(snapshot)) for snapshot in scoped)
        for rev in revisions:
            snapshot = getattr(rev, "snapshot_path", None)
            for f in rev.files:
                name = str(getattr(f, "file_name", "") or "")
                path = getattr(f, "file_path", None)
                parts: tuple[str, ...] = ()
                if path is not None and snapshot is not None:
                    try:
                        parts = Path(path).relative_to(Path(snapshot)).parts
                    except ValueError:
                        parts = ()
                if not parts:
                    parts = Path(name).parts
                if (
                    len(parts) >= 2
                    and parts[0].lower() in _DENOISER_DIRS
                    and parts[-1].lower().endswith(_DENOISER_WEIGHT_SUFFIXES)
                ):
                    return False
        return True
    except Exception:
        return False


def is_variant_partial(
    repo_id: str,
    variant: str,
    snapshot_dir: Optional[Path] = None,
    *,
    incomplete_blob_hashes: Optional[set[str]] = None,
    variant_blob_hashes: Optional[frozenset[str]] = None,
    repo_cache_dir: Optional[Path] = None,
    repo_signal_applies: bool = True,
    variant_state = None,
) -> bool:
    """Per-variant partial detection. Owns its manifest, owns its marker. Used by the GGUF variants endpoint to flag one quant as broken without contaminating the others in the same repo. *snapshot_dir* is an optional hint to avoid re-walking the cache when checking many variants of one repo. ``repo_signal_applies`` is ``_repo_signal_applies_to_snapshot``'s verdict: a caller pinning an older revision passes False rather than judge that quant by another revision's marker, manifest or blobs. Defaults True so the per-variant endpoint still reports a cancelled quant as broken."""
    from hub.utils import download_manifest
    return _compose_partial(
        lambda: (
            repo_signal_applies
            and (
                variant_state.has_marker(variant)
                if variant_state is not None
                else download_manifest.has_cancel_marker(
                    "model",
                    repo_id,
                    variant,
                    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir),
                )
            )
        ),
        lambda: (
            repo_signal_applies
            and bool(
                incomplete_blob_hashes
                and variant_blob_hashes
                and incomplete_blob_hashes.intersection(variant_blob_hashes)
            )
        ),
        lambda: (
            repo_signal_applies
            and _manifest_partial(
                "model",
                repo_id,
                variant,
                snapshot_dir,
                repo_cache_dir,
                variant_state,
            )
        ),
    )


def is_gguf_repo_partial(
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    *,
    snapshot_dir: Optional[Path] = None,
    variant_state = None,
) -> bool:
    """Repo-row partial flag for a GGUF repo. The inventory shows ONE row per GGUF repo (requires_variant=True); per-variant detail lives in GET /api/models/gguf-variants and uses is_variant_partial.

    *** DO NOT simplify this to "any variant partial -> repo partial" ***

    Tripwire: a user downloads Q8_0 fully, then starts Q4_K_M and cancels. Both variants share ONE inventory row, so flipping row.partial True flips can_chat=False and the perfectly-good Q8_0 becomes unchattable over an unrelated cancelled Q4_K_M.

    Correct semantics: partial=True only when at least one variant is broken AND no other variant is clean. Composes a cheap legacy fast-path (.incomplete blobs / broken symlinks) with a per-variant manifest + marker enumeration gated on "all broken". *snapshot_dir* pins both to the row's load id; without it an interrupted re-download flips can_chat off for the older complete quant.
    """
    from hub.utils import download_manifest

    if snapshot_dir is None:
        snapshot_dir = resolve_snapshot_dir_for_scan(
            "model",
            repo_id,
            repo_cache_dir,
        )
    repo_signal_applies = _repo_signal_applies_to_snapshot(
        repo_cache_dir, snapshot_dir, quants = True
    )
    has_legacy_partial = repo_signal_applies and _legacy_partial("model", repo_id, repo_cache_dir)
    complete_here = _completed_gguf_variants(snapshot_dir)
    variants: set[str] = set(complete_here)
    hub_cache = _hub_cache_for_repo_dir(repo_cache_dir)
    if variant_state is not None:
        manifests = variant_state.manifests()
    else:
        manifests = (
            (
                variant,
                download_manifest.read_manifest(
                    "model",
                    repo_id,
                    variant,
                    hub_cache = hub_cache,
                ),
            )
            for variant, _path in download_manifest.iter_variant_manifests(
                "model",
                repo_id,
                hub_cache = hub_cache,
            )
        )
    for variant, manifest in manifests:
        if manifest is not None:
            variants.add(variant)
    if variant_state is not None:
        variants.update(variant_state.marker_variants())
    else:
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
        return has_legacy_partial or _recovered_snapshot_cannot_serve(
            repo_cache_dir, snapshot_dir, quants = True
        )
    has_clean = False
    has_broken = has_legacy_partial
    for variant in variants:
        if is_variant_partial(
            repo_id,
            variant,
            snapshot_dir,
            repo_cache_dir = repo_cache_dir,
            repo_signal_applies = repo_signal_applies or variant not in complete_here,
            variant_state = variant_state,
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
    """Transport to surface on a partial row's resume affordance. Prefers the cancel marker's transport, then the manifest's; the fallback matters for rows partial without a marker (an errored download leaves the manifest but no marker) so the UI can still show HTTP-resume vs XET-redownload instead of the neutral retry label. ``None`` when neither is available."""
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


def partial_resume_available(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    """Whether THIS partial can be picked up byte for byte, rather than whether some partial somewhere could be. Both verdicts have to agree: the transport this row reports, and the registry's per-file check, which rejects a 1.18+ nonce partial nothing will reopen. The installed huggingface_hub cannot answer it alone, since a cache shared with a newer environment holds partials this one can never continue."""
    from hub.utils import download_registry

    if partial_transport_for(repo_type, repo_id, variant, repo_cache_dir) != "http":
        return False
    return download_registry.is_resumable_partial(
        repo_type,
        repo_id,
        variant,
        root = _hub_cache_for_repo_dir(repo_cache_dir),
    )

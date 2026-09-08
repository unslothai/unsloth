# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import os
import re
import shutil
import stat as stat_module
import sys
from pathlib import Path, PureWindowsPath
from typing import Iterable, Iterator, Optional


EXIT_CANCELLED = 130

TRANSPORT_HTTP = "http"
TRANSPORT_XET = "xet"
VALID_TRANSPORTS = frozenset({TRANSPORT_HTTP, TRANSPORT_XET})
# A request preference, deliberately NOT in VALID_TRANSPORTS: the on-disk .transport marker must
# keep naming the writer that produced a partial, or a resume picks the wrong strategy.
TRANSPORT_AUTO = "auto"
VALID_TRANSPORT_MODES = frozenset({TRANSPORT_HTTP, TRANSPORT_XET, TRANSPORT_AUTO})
TRANSPORT_MARKER_NAME = ".transport"
INCOMPLETE_SUFFIX = ".incomplete"
_PROCESS_UNIQUE_PARTIAL_RE = re.compile(
    r"^(?P<blob_hash>(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}))\.[0-9a-fA-F]{8}$"
)


def incomplete_blob_hash(name: str) -> Optional[str]:
    """Return the logical HF blob hash represented by a partial filename.

    huggingface_hub historically wrote ``<etag>.incomplete``. Version 1.18
    changed the writer to a process-unique ``<etag>.<8 hex>.incomplete`` path
    before atomically moving it into place. Hub etags used as cache blob names
    are Git SHA-1 or LFS SHA-256 hex digests, which lets us remove the nonce
    without mis-parsing an arbitrary legacy filename containing a dot.
    """
    if not name.endswith(INCOMPLETE_SUFFIX):
        return None
    stem = name[: -len(INCOMPLETE_SUFFIX)]
    if not stem:
        return None
    process_unique = _PROCESS_UNIQUE_PARTIAL_RE.fullmatch(stem)
    return process_unique.group("blob_hash") if process_unique else stem


# The last huggingface_hub line whose partials a later attempt can append to.
_LAST_RESUMABLE_PARTIAL_VERSION = (1, 17)

# huggingface_hub writes to a partial continuously, so anything still advancing has a live writer,
# possibly a client in another process no registry here can see.
ABANDONED_PARTIAL_SECONDS = 120


def hf_partials_are_resumable(hub_cache: Optional[str] = None) -> bool:
    """Whether an interrupted download leaves bytes the next attempt can reuse.

    Up to 1.17 huggingface_hub appended to a shared ``<etag>.incomplete`` and restarted from
    its length over a Range request. 1.18 moved the writer to a process-unique
    ``<etag>.<nonce>.incomplete``, opened ``"wb"`` and unlinked in a ``finally``
    (huggingface/huggingface_hub#4228), so an interrupted file is refetched from zero and
    whatever partial survives a hard kill can never be read again.

    An unreadable version answers True: not knowing which writer is installed is not grounds
    for deleting bytes that may still be resumable.

    On 1.18+ this also asks whether the download worker will put the 1.17 writer back
    (:mod:`hub.utils.resumable_partials`), since a restored resumer makes partials reusable again.
    That half turns on the filesystem the partial is on, so *hub_cache* names the root being asked
    about. Unsloth remembers several and they need not lock alike: without it, a selected cache on a
    network mount would condemn a local cache's partials to the abandoned-partial sweep. Omitting it
    asks about the cache in force, which is where a new download lands.

    Deliberately not cached here. The verdict is a fact about a filesystem, not about a path, so a
    result keyed on the path alone survives a remount at the same name and outlives a momentary
    failure to probe. The expensive part, the lock probe, is cached in
    :mod:`hub.utils.resumable_partials` against the mounted device instead, and a probe that could
    not run raises rather than answering, so nothing remembers a bad moment.
    """
    try:
        from huggingface_hub import __version__ as hf_version
    except Exception:  # noqa: BLE001 - an unimportable hub is the caller's problem, not ours
        return True
    release = []
    for chunk in str(hf_version).split(".")[:2]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            return True
        release.append(int(digits))
    if tuple(release) <= _LAST_RESUMABLE_PARTIAL_VERSION:
        return True
    try:
        from hub.utils.resumable_partials import _ProbeUnavailable, can_restore_partials
        try:
            return can_restore_partials(hub_cache)
        except _ProbeUnavailable:
            # Nothing was shown, so nothing is promised -- and nothing is remembered either.
            return False
    except Exception:  # noqa: BLE001 - no restoration is just the stock answer
        from loggers import get_logger
        get_logger(__name__).debug(
            "Resumable-partial restoration unavailable; partials stay unresumable.",
            exc_info = True,
        )
        return False


def invalidate_partial_resumability() -> None:
    """Re-decide resumability, for when the cache moves to another filesystem.

    The verdict depends on whether ``flock`` excludes a second writer where the partial lands, so
    it cannot be carried over from the old root.
    """
    try:
        from hub.utils.resumable_partials import invalidate_probe_cache
        invalidate_probe_cache()
    except Exception:  # noqa: BLE001 - nothing to invalidate is not an error
        pass


def partial_is_process_unique(name: str) -> bool:
    """Whether a partial filename carries the 1.18+ per-process nonce."""
    if not name.endswith(INCOMPLETE_SUFFIX):
        return False
    return _PROCESS_UNIQUE_PARTIAL_RE.fullmatch(name[: -len(INCOMPLETE_SUFFIX)]) is not None


def blob_download_lock_held(entry: Path, blob_hash: str) -> bool:
    """Whether some process holds huggingface_hub's per-blob download lock right now.

    hf takes ``<hub cache>/.locks/<repo dir>/<etag>.lock`` for the whole of a file download, so
    a lock we cannot take means a live writer -- including a client in another process that no
    peer registry here can see. It answers False when the lock cannot be probed at all, since
    upstream calls the lock best-effort and some filesystems grant it to everyone; callers pair
    it with a staleness check rather than trusting it alone.
    """
    lock_path = entry.parent / ".locks" / entry.name / f"{blob_hash}.lock"
    if not lock_path.exists():
        # hf creates the lock file before taking the lock, so no file means no writer; it is also the answer
        # for a SoftFileLock, whose file IS the lock.
        return False
    try:
        from filelock import FileLock, Timeout
    except Exception:  # noqa: BLE001 - no filelock at all means no opinion
        return False
    try:
        with FileLock(str(lock_path), timeout = 0):
            return False
    except Timeout:
        return True
    except Exception:  # noqa: BLE001 - deliberately broad, see below
        # A filesystem without flock raises NotImplementedError, which upstream answers by retrying as a
        # SoftFileLock; retrying is pointless for a PROBE, since we only reach here because the file
        # exists. What matters is that the exception does not escape: a wrong "free" deletes a live
        # writer's file.
        return True


def partial_is_resumable(name: str, hub_cache: Optional[Path | str] = None) -> bool:
    """Whether any later attempt could append to this particular partial.

    Two conditions, and the layout half matters on its own: a nonce partial is private to the
    process that created it, so even a legacy writer will not reopen it. That combination is
    reachable whenever caches are shared across environments, which this repo's own pins
    produce (Python 3.10+ takes hub >= 1.23, older takes 0.36.2, one cache between them).

    *hub_cache* is the root the partial lives under. Callers walking more than one cache must pass
    it, since the answer is partly a property of that root's filesystem.
    """
    if partial_is_process_unique(name):
        return False
    return hf_partials_are_resumable(str(hub_cache) if hub_cache is not None else None)


def _safe_is_dir(path: Path, scan_errors: Optional[list] = None) -> bool:
    """``Path.is_dir()`` returning False instead of raising when the path or a
    parent is unreadable (e.g. a restricted ``~/.cache/huggingface/hub``), so
    cache enumeration skips that root rather than 500ing.

    ``scan_errors`` collects the swallowed error. A caller that only wants the dirs does not
    care, but "we could not even stat the root" and "the root is not there" are different
    answers to a hydrating job -- the first is not evidence the cache was deleted.

    os.stat, not Path.is_dir(): as of 3.14 is_dir() answers False for EVERY OSError instead of
    raising some and suppressing others, so the handler below could never see a permission or
    network-mount failure and the root was recorded as a measured absence. Stat says which it
    was.
    """
    try:
        return stat_module.S_ISDIR(os.stat(path).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False  # genuinely not a directory here; that IS the answer
    except (OSError, ValueError) as exc:
        if scan_errors is not None:
            scan_errors.append(exc)
        return False


def same_existing_path(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except (OSError, ValueError):
        return False


def hf_cache_root(
    *,
    create: bool = False,
    root: Optional[Path] = None,
    scan_errors: Optional[list] = None,
) -> Optional[Path]:
    from utils.hf_cache_settings import get_hf_cache_paths

    root = root or get_hf_cache_paths().hub_cache
    if create:
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError as exc:
            return None
        return root
    return root if _safe_is_dir(root, scan_errors) else None


def hf_cache_roots(scan_errors: Optional[list] = None) -> list[Path]:
    from hub.utils.paths import hf_default_cache_dir, legacy_hf_cache_dir
    from utils.hf_cache_settings import known_hf_hub_caches

    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not _safe_is_dir(path, scan_errors):
            return
        try:
            key = str(path.resolve())
        except OSError as exc:
            # A root that stats but will not resolve is a root we could not read, not one that is gone:
            # dropping it silently let the progress scan answer "measured, no cache", which retires a download
            # whose files may be intact.
            if scan_errors is not None:
                scan_errors.append(exc)
            return
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    for configured in known_hf_hub_caches():
        _add(configured)
    _add(legacy_hf_cache_dir())
    _add(hf_default_cache_dir())
    return roots


def target_dir_name(repo_type: str, repo_id: str) -> str:
    return repo_cache_dir_name(repo_type, repo_id).lower()


def repo_cache_dir_name(repo_type: str, repo_id: str) -> str:
    return f"{repo_type}s--{repo_id.replace('/', '--')}"


def resolve_destructive_case_matches(target: str, candidates: Iterable[str]) -> Optional[set[str]]:
    values = list(candidates)
    exact = {candidate for candidate in values if candidate == target}
    if exact:
        return exact
    folded = {candidate for candidate in values if candidate.lower() == target.lower()}
    if len(folded) <= 1:
        return folded
    return None


def _blob_dir_is_partial(blobs_dir: Path) -> bool:
    try:
        for blob in blobs_dir.iterdir():
            if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                return True
    except OSError:
        return False
    return False


def blob_bytes_present(path: Path) -> int:
    """Sparse-aware on-disk size: XET/``hf_transfer`` ``.incomplete`` partials
    report a full ``st_size`` while only some blocks are allocated, so prefer
    ``st_blocks``, falling back to ``st_size`` where it is unreported (Windows,
    some network filesystems)."""
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None and blocks > 0:
        return min(blocks * 512, st.st_size)
    # A present zero is not a missing field: a parallel writer that sets the partial to its final
    # length before its first chunk lands sits exactly here, and a mount that never populates
    # st_blocks looks the same, so confirm the emptiness directly before believing st_size.
    if blocks == 0 and _holds_no_data(path):
        return 0
    if sys.platform == "win32":
        allocated = _windows_allocated_size(path)
        if allocated is not None:
            return min(allocated, st.st_size)
    return st.st_size


def _holds_no_data(path: Path) -> bool:
    """Whether the file has no allocated extent anywhere, asked of the kernel rather than
    inferred. ``SEEK_DATA`` past the end of the last extent is ENXIO, so a file with nothing
    written raises on the very first seek. Every other answer -- an unsupported seek, an
    unreadable path -- leaves the caller's size fallback in charge.
    """
    seek_data = getattr(os, "SEEK_DATA", None)
    if seek_data is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.lseek(fd, 0, seek_data)
    except OSError as exc:
        return exc.errno == errno.ENXIO
    finally:
        os.close(fd)
    return False


def _windows_allocated_size(path: Path) -> Optional[int]:
    """Best-effort allocated-byte count for sparse files on Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        get_compressed_file_size = kernel32.GetCompressedFileSizeW
        get_compressed_file_size.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_compressed_file_size.restype = wintypes.DWORD

        high = wintypes.DWORD(0)
        ctypes.set_last_error(0)
        low = get_compressed_file_size(str(path), ctypes.byref(high))
        if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
            return None
        return (int(high.value) << 32) + int(low)
    except Exception:
        return None


def snapshot_selection_key(snapshot: Path) -> tuple[float, str]:
    """The one ordering every snapshot selector uses: mtime, then resolved path.

    mtime alone is not a total order, and each selector broke ties by its own
    iteration order (frozenset vs iterdir), so the inventory row and the variant
    picker could name different snapshots. The path breaks ties identically.
    """
    try:
        mtime = snapshot.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        return mtime, str(snapshot.resolve())
    except (OSError, RuntimeError, ValueError):
        return mtime, str(snapshot)


def latest_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    """Newest immediate child of ``repo_dir/snapshots``, or None.

    mtime is the signal huggingface_hub's from_pretrained resolves to; ties fall
    to ``snapshot_selection_key`` so every caller names the same directory.
    """
    snapshots_dir = repo_dir / "snapshots"
    try:
        if not snapshots_dir.is_dir():
            return None
        snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshots:
            return None
        return max(snapshots, key = snapshot_selection_key)
    except OSError:
        return None


def ref_snapshot_dir(repo_dir: Path, ref: str = "main") -> Optional[Path]:
    if not ref or ref in {".", ".."} or Path(ref).name != ref or PureWindowsPath(ref).name != ref:
        return None
    try:
        repo_root = repo_dir.resolve(strict = True)
        refs = (repo_root / "refs").resolve(strict = True)
        ref_path = (refs / ref).resolve(strict = True)
        if (
            not same_existing_path(refs.parent, repo_root)
            or not refs.is_dir()
            or not same_existing_path(ref_path.parent, refs)
            or not ref_path.is_file()
            or ref_path.stat().st_size > 256
        ):
            return None
        commit = ref_path.read_text(encoding = "utf-8").strip()
    except (OSError, RuntimeError, UnicodeError):
        return None
    if (
        not commit
        or len(commit) > 256
        or commit in {".", ".."}
        or Path(commit).name != commit
        or PureWindowsPath(commit).name != commit
    ):
        return None
    try:
        snapshots = (repo_root / "snapshots").resolve(strict = True)
        if not same_existing_path(snapshots.parent, repo_root) or not snapshots.is_dir():
            return None
        snapshot = (snapshots / commit).resolve(strict = True)
        snapshot.relative_to(snapshots)
    except (OSError, RuntimeError, ValueError):
        return None
    return snapshot if _safe_is_dir(snapshot) else None


def validated_repo_cache_path(
    local_path: Optional[str], repo_type: str, repo_id: str
) -> Optional[tuple[Path, Path]]:
    if not local_path or not repo_id:
        return None
    try:
        resolved = Path(local_path).expanduser().resolve(strict = True)
        expected = target_dir_name(repo_type, repo_id)
        repo_dir = next(
            (
                candidate
                for candidate in (resolved, *resolved.parents)
                if candidate.name.lower() == expected
            ),
            None,
        )
        if repo_dir is None:
            return None
        allowed_roots = [root.resolve(strict = True) for root in hf_cache_roots() if root.exists()]
        repo_dir = repo_dir.resolve(strict = True)
        if not any(same_existing_path(repo_dir.parent, root) for root in allowed_roots):
            return None
        resolved.relative_to(repo_dir)
        return repo_dir, resolved
    except (OSError, RuntimeError, ValueError):
        return None


def latest_snapshot_from_cache_path(
    local_path: Optional[str],
    repo_type: str,
    repo_id: str,
    metadata_filenames: tuple[str, ...] = (),
    required_groups: tuple[tuple[str, ...], ...] = (),
) -> Optional[str]:
    validated = validated_repo_cache_path(local_path, repo_type, repo_id)
    if validated is None:
        return None
    repo_dir, selected = validated
    try:

        def has_metadata(path: Path) -> bool:
            # required_groups is an AND of ORs: "loadable" means metadata alone or weights alone is not enough.
            for group in required_groups:
                if not any((path / name).is_file() for name in group):
                    return False
            if not metadata_filenames:
                return True
            return any((path / name).is_file() for name in metadata_filenames)

        snapshots = (repo_dir / "snapshots").resolve(strict = True)
        if not same_existing_path(snapshots.parent, repo_dir) or not snapshots.is_dir():
            return None
        if not same_existing_path(selected, repo_dir):
            if not same_existing_path(selected.parent, snapshots) or not selected.is_dir():
                return None
            return str(selected) if has_metadata(selected) else None

        candidates: list[Path] = []
        pinned = ref_snapshot_dir(repo_dir)
        if pinned is not None and has_metadata(pinned):
            return str(pinned)
        for path in snapshots.iterdir():
            try:
                candidate = path.resolve(strict = True)
            except (OSError, RuntimeError):
                continue
            if (
                same_existing_path(candidate.parent, snapshots)
                and candidate.is_dir()
                and has_metadata(candidate)
            ):
                candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(key = snapshot_selection_key, reverse = True)
        return str(candidates[0].resolve())
    except Exception:
        return None


def snapshot_has_broken_symlinks(snapshot: Path) -> bool:
    """Whether ``snapshot`` links to a blob that is not finalized.

    Scoped to one snapshot on purpose. A ``.incomplete`` blob under ``blobs/``
    belongs to whichever revision or scoped file set is fetching it, and the cache
    is shared, so its presence says nothing about the revision being validated.
    What does is a link this snapshot owns whose target is not there yet.

    Windows hydrates the cache with copies rather than links, so there is nothing
    to dangle: a half-fetched file is simply absent, which the payload inventory
    catches instead.
    """
    try:
        for entry in snapshot.rglob("*"):
            if entry.is_symlink() and not entry.exists():
                return True
    except OSError:
        return False
    return False


def _repo_dir_has_broken_snapshot_symlinks(repo_dir: Path) -> bool:
    latest = latest_snapshot_dir(repo_dir)
    if latest is None:
        return False
    return snapshot_has_broken_symlinks(latest)


def iter_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    scan_errors: Optional[list] = None,
) -> Iterator[Path]:
    """Cache dirs for this repo, skipping a root that cannot be listed.

    ``scan_errors`` collects those skips. Suppressing them is right for every caller that
    only wants the dirs, but hydration reads "no dirs" as "the cache was wiped and this
    persisted job can be retired" -- and a root that raised EACCES or EIO is not evidence
    of that. A caller that passes a list can tell the two apart.
    """
    target = target_dir_name(repo_type, repo_id)
    for root in hf_cache_roots(scan_errors = scan_errors):
        try:
            for entry in root.iterdir():
                if entry.name.lower() == target:
                    yield entry
        except OSError as exc:
            if scan_errors is not None:
                scan_errors.append(exc)
            continue


def iter_destructive_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
) -> Iterator[Path]:
    target = repo_cache_dir_name(repo_type, repo_id)
    folded_target = target.lower()
    if root is not None:
        scoped = hf_cache_root(root = root)
        bases = [scoped] if scoped is not None else []
    else:
        bases = hf_cache_roots()
    for base in bases:
        try:
            entries = [entry for entry in base.iterdir() if entry.name.lower() == folded_target]
        except OSError:
            continue
        matched_names = resolve_destructive_case_matches(
            target,
            (entry.name for entry in entries),
        )
        if not matched_names:
            continue
        for entry in entries:
            if entry.name in matched_names:
                yield entry


def iter_active_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
    scan_errors: Optional[list] = None,
) -> Iterator[Path]:
    # The stat itself can fail on a restricted root, and that is not evidence of absence.
    root = hf_cache_root(root = root, scan_errors = scan_errors)
    if root is None:
        return
    target = target_dir_name(repo_type, repo_id)
    try:
        for entry in root.iterdir():
            if entry.name.lower() == target:
                yield entry
    except OSError as exc:
        if scan_errors is not None:
            scan_errors.append(exc)
        return


def preferred_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    force_active: bool = False,
    active_root: Optional[Path] = None,
    scan_errors: Optional[list] = None,
) -> list[Path]:
    # iter_active_repo_cache_dirs already matches case-insensitively, so the canonical name below is
    # only ever a placeholder for a repo dir that is not there yet.
    active_entries = list(
        iter_active_repo_cache_dirs(repo_type, repo_id, root = active_root, scan_errors = scan_errors)
    )
    if active_entries:
        return active_entries
    if force_active:
        # A running or cancelling job writes into the active root and nowhere else, so its progress may
        # only be read from there: hf_cache_root returns None for a root not yet created, and falling
        # through would read a previous cache's completed copy as this run's progress.
        root = hf_cache_root(root = active_root) or active_root or _configured_hub_cache()
        if root is not None:
            canonical = repo_cache_dir_name(repo_type, repo_id)
            return [root / canonical]
    return list(iter_repo_cache_dirs(repo_type, repo_id, scan_errors = scan_errors))


def _configured_hub_cache() -> Optional[Path]:
    # Path()-wrapped: the setting is typed Path, but a caller handing back a str would turn `root /
    # name` into a TypeError that surfaces as an empty progress reading.
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        configured = get_hf_cache_paths().hub_cache
        return Path(configured) if configured else None
    except Exception:
        return None


def has_incomplete_blobs(repo_type: str, repo_id: str) -> bool:
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False


def has_active_incomplete_blobs(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False


def repo_cache_dir_has_incomplete_blobs(repo_dir: Path) -> bool:
    blobs_dir = repo_dir / "blobs"
    return (blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir)) or (
        _repo_dir_has_broken_snapshot_symlinks(repo_dir)
    )


def _prune_empty_dirs(root: Path) -> bool:
    removed = False
    try:
        dirs = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key = lambda path: len(path.parts),
            reverse = True,
        )
    except OSError:
        dirs = []
    for directory in [*dirs, root]:
        try:
            directory.rmdir()
            removed = True
        except FileNotFoundError:
            continue
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
    return removed


def purge_partial_repo(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    removed = False
    for entry in iter_destructive_repo_cache_dirs(repo_type, repo_id, root = root):
        blobs_dir = entry / "blobs"
        if blobs_dir.is_dir():
            for blob in blobs_dir.iterdir():
                if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                    try:
                        blob.unlink()
                        removed = True
                    except FileNotFoundError:
                        continue
        if _prune_empty_dirs(entry):
            removed = True
    return removed


def purge_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    removed = False
    for entry in iter_destructive_repo_cache_dirs(repo_type, repo_id, root = root):
        try:
            if entry.is_symlink() or not entry.is_dir():
                continue
            shutil.rmtree(entry)
            removed = True
        except FileNotFoundError:
            continue
    return removed


def scoped_delete_root(repo_type: str, repo_id: str, cache_path: Optional[str]) -> Optional[Path]:
    """Resolve the single cache root a delete of this repo may touch.

    Returns the active hub cache when *cache_path* is falsy, the owning cache
    root when *cache_path* points inside a known cache, or ``None`` when
    *cache_path* is set but not inside any known cache (caller should reject).
    This keeps a delete of one inventory row from removing copies in other,
    previously selected caches.
    """
    from utils.hf_cache_settings import get_hf_cache_paths

    if not cache_path:
        return Path(get_hf_cache_paths().hub_cache).resolve(strict = False)
    try:
        resolved = Path(cache_path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return None
    expected = repo_cache_dir_name(repo_type, repo_id).lower()
    repo_dir = next(
        (
            candidate
            for candidate in (resolved, *resolved.parents)
            if candidate.name.lower() == expected
        ),
        None,
    )
    if repo_dir is None:
        return None
    allowed = {r.resolve(strict = False) for r in hf_cache_roots()}
    root = repo_dir.parent.resolve(strict = False)
    return root if root in allowed else None


def resolve_delete_target_root(
    repo_type: str, repo_id: str, cache_path: Optional[str], owner_roots
) -> Optional[Path]:
    """Pick the single cache root a delete of this repo should target.

    An explicit *cache_path* wins (``None`` when it is not a known cache, so the
    caller can reject it). Otherwise prefer the active cache when it holds a
    copy, else the sole cache that does -- so a model that lives only in a
    previously selected cache stays deletable while other caches are untouched.
    """
    if cache_path:
        return scoped_delete_root(repo_type, repo_id, cache_path)
    from utils.hf_cache_settings import get_hf_cache_paths

    active = Path(get_hf_cache_paths().hub_cache).resolve(strict = False)
    roots = list(owner_roots)
    if active in roots:
        return active
    if len(roots) == 1:
        return roots[0]
    return active


def with_load_subdirs(model_name: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """Extend snapshot filenames with the subdirectories a load actually reads.

    Spark-TTS / BiCodec keep the trainable model under ``<snapshot>/LLM``, so such a
    snapshot carries no root-level ``config.json`` and no root-level weights. Every
    cache probe that decides "is this snapshot usable" has to agree on that, or the
    snapshot resolves in one place and is rejected in the next: the start preflight, the
    worker's revalidation and the provenance attester each get their own answer.

    Detection can raise offline or for a gated repo, so a failure degrades to root-only.

    Asked offline on purpose. Every caller here is deciding whether a cache already on
    disk is usable, which was pure filesystem work before this helper existed; letting
    it reach the hub would put a network round trip, with no timeout, in front of local
    snapshot resolution. The subdir layout is a property of the cached snapshot, so the
    local answer is the correct one here.
    """
    try:
        from utils.security import security_load_subdirs
        subdirs = security_load_subdirs(model_name, local_files_only = True)
    except Exception:
        # Degrading to root-only is fail-closed at every caller, but a real cache permission or corruption
        # fault then reaches the user as "your cached model isn't cached" with no clue why.
        from loggers import get_logger
        get_logger(__name__).debug(
            "Load-subdir detection failed for %s; using root only.",
            model_name,
            exc_info = True,
        )
        return names
    if not subdirs:
        return names
    return names + tuple(
        f"{subdir.strip('/')}/{name}" for subdir in subdirs if subdir for name in names
    )

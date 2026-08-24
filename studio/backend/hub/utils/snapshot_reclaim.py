# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safely promote a verified Hub snapshot to the active ``main`` ref."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable, Optional

from filelock import FileLock

from hub.utils.paths import is_redirect_stat
from utils.hf_cache_settings import get_hf_cache_paths
from utils.hf_repo_ids import is_valid_repo_id
from utils.paths.path_utils import is_appledouble_metadata


@dataclass(frozen = True)
class PreviousMainRef:
    repo_dir: Optional[Path]
    revision: Optional[str]
    promotion_safe: bool
    reason: Optional[str] = None
    allow_unpinned_download: bool = False


@dataclass(frozen = True)
class SnapshotPromotionResult:
    previous_revision: Optional[str]


class ConcurrentMainRefError(RuntimeError):
    pass


class RedirectedRepoError(ValueError):
    pass


class SnapshotRefsUnverifiable(RuntimeError):
    pass


class _MainRefCleanupError(RuntimeError):
    pass


_MAIN_REF_LOCK_TIMEOUT_SECONDS = 60
_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS = (0.05, 0.1)
_REFS_STAGING_DIRECTORY_NAME = ".unsloth-refs-tmp"
_REFS_STAGING_RETRY_DELAYS_SECONDS = (0.01, 0.05, 0.1)
_OS_METADATA_REF_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
_COMMIT_REVISION_LENGTH = 40
_COMMIT_REVISION_CHARACTERS = frozenset("0123456789abcdefABCDEF")
_IS_WINDOWS = os.name == "nt"
_IS_LINUX = sys.platform.startswith("linux")
_IS_DARWIN = sys.platform == "darwin"
_HARDLINK_FALLBACK_ERRNOS = frozenset(
    value
    for value in (
        errno.EPERM,
        errno.EINVAL,
        errno.EMLINK,
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)
_HARDLINK_FALLBACK_WINERRORS = frozenset({1, 50, 87, 1142})
_NATIVE_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


def _normalized_revision(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or PureWindowsPath(normalized).name != normalized
    ):
        return None
    return normalized


def _parse_main_ref(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if value.endswith("\r\n"):
        candidate = value[:-2]
    elif value.endswith("\n"):
        candidate = value[:-1]
    else:
        candidate = value
    if "\r" in candidate or "\n" in candidate:
        return None
    normalized = _normalized_revision(candidate)
    return normalized if normalized == candidate else None


def _read_main_ref_text(path: Path) -> str:
    with path.open("r", encoding = "utf-8", newline = "") as handle:
        return handle.read()


def _same_existing_path(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except (OSError, ValueError):
        return False


def _lstat_or_none(path: Path) -> Optional[os.stat_result]:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _require_real_directory(path: Path, expected: Path, label: str) -> Path:
    path_stat = path.lstat()
    if is_redirect_stat(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{label} is not a real directory: {path}")
    resolved = path.resolve(strict = True)
    if resolved != expected:
        raise ValueError(f"{label} is redirected outside its exact cache location: {path}")
    return resolved


def _ensure_real_directory(path: Path, expected: Path, label: str) -> Path:
    try:
        path.mkdir()
    except FileExistsError:
        pass
    return _require_real_directory(path, expected, label)


def _refs_staging_directory(repo_dir: Path) -> Path:
    """Promotion scratch directory, a sibling of ``refs`` in *repo_dir*.

    Scratch must not live in ``refs``: third-party readers glob it, dotfiles
    included, and read every entry as a ref. A sibling keeps the same filesystem,
    so the hardlink and no-replace rename paths are unaffected. A directory left
    by an interrupted promotion is reused; its contents are inert.
    """
    return _ensure_real_directory(
        repo_dir / _REFS_STAGING_DIRECTORY_NAME,
        repo_dir / _REFS_STAGING_DIRECTORY_NAME,
        "Hub cache refs staging path",
    )


def _open_staged_temporary(repo_dir: Path) -> tuple[Path, Path, int]:
    """Create the staging directory plus an exclusive scratch ref inside it.

    Cache maintenance rmdirs empty directories without the main-ref lock, so
    retry the mkdir-then-open window rather than fail a valid promotion. Once the
    scratch exists the directory stays non-empty.
    """
    for attempt in range(len(_REFS_STAGING_RETRY_DELAYS_SECONDS) + 1):
        try:
            staging = _refs_staging_directory(repo_dir)
            temporary = staging / f".unsloth-main-{uuid.uuid4().hex}"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileNotFoundError:
            if attempt == len(_REFS_STAGING_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_REFS_STAGING_RETRY_DELAYS_SECONDS[attempt])
            continue
        return staging, temporary, fd
    raise ConcurrentMainRefError("the refs staging directory kept vanishing during promotion")


def _discard_refs_staging_directory(staging: Path) -> None:
    """Drop the staging directory if empty; never fail a promotion for it.

    ``rmdir`` needs it empty, so a deliberately kept displaced ref survives.
    """
    try:
        staging.rmdir()
    except OSError:
        pass


def _main_ref_lock(repo_dir: Path) -> FileLock:
    lock_root = _ensure_real_directory(
        repo_dir.parent / ".locks",
        repo_dir.parent / ".locks",
        "Hub cache locks path",
    )
    repo_locks = _ensure_real_directory(
        lock_root / repo_dir.name,
        lock_root / repo_dir.name,
        "Hub cache repo locks path",
    )
    lock_path = repo_locks / "unsloth-main.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    lock_stat = lock_path.lstat()
    if is_redirect_stat(lock_stat) or not stat.S_ISREG(lock_stat.st_mode):
        raise ValueError(f"Hub cache main-ref lock is not a regular file: {lock_path}")
    return FileLock(
        str(lock_path),
        timeout = _MAIN_REF_LOCK_TIMEOUT_SECONDS,
    )


def _repo_cache_prefix(repo_type: str) -> str:
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"Unsupported Hugging Face repo type: {repo_type!r}")
    return f"{repo_type}s"


def _canonical_repo_dir(
    root: Path, repo_type: str, repo_id: str, *, require_existing: bool
) -> Path:
    if not is_valid_repo_id(repo_id):
        raise ValueError(f"Invalid Hugging Face repo id: {repo_id!r}")
    root_path = root.expanduser()
    root_real = root_path.resolve(strict = require_existing)
    repo_path = root_path / f"{_repo_cache_prefix(repo_type)}--{repo_id.replace('/', '--')}"
    repo_stat = _lstat_or_none(repo_path)
    if repo_stat is not None and is_redirect_stat(repo_stat):
        raise RedirectedRepoError(f"Hub cache repo is redirected: {repo_path}")
    repo_real = repo_path.resolve(strict = require_existing)
    if repo_real != root_real / repo_path.name:
        raise ValueError(f"Hub cache repo escapes its root: {repo_path}")
    if require_existing and not repo_real.is_dir():
        raise ValueError(f"Hub cache repo is not a directory: {repo_path}")
    return repo_real


def capture_previous_main_ref(repo_id: str, *, repo_type: str = "model") -> PreviousMainRef:
    """Capture the active root's ``refs/main`` before the pinned download."""
    repo_dir: Optional[Path] = None
    try:
        hub_cache = Path(get_hf_cache_paths().hub_cache)
        repo_dir = _canonical_repo_dir(
            hub_cache,
            repo_type,
            repo_id,
            require_existing = False,
        )
        if _lstat_or_none(repo_dir) is None:
            return PreviousMainRef(repo_dir, None, True)
        repo_dir = _canonical_repo_dir(
            hub_cache,
            repo_type,
            repo_id,
            require_existing = True,
        )
    except RedirectedRepoError as exc:
        return PreviousMainRef(
            repo_dir,
            None,
            False,
            str(exc),
            allow_unpinned_download = True,
        )
    except OSError as exc:
        return PreviousMainRef(
            repo_dir,
            None,
            False,
            str(exc),
            allow_unpinned_download = True,
        )
    except (RuntimeError, ValueError) as exc:
        return PreviousMainRef(repo_dir, None, False, str(exc))

    try:
        refs = repo_dir / "refs"
        refs_stat = _lstat_or_none(refs)
        if refs_stat is None:
            return PreviousMainRef(repo_dir, None, True)
        if is_redirect_stat(refs_stat) or not stat.S_ISDIR(refs_stat.st_mode):
            return PreviousMainRef(repo_dir, None, False, "refs is not a real directory")
        if refs.resolve(strict = True) != repo_dir / "refs":
            return PreviousMainRef(repo_dir, None, False, "refs is redirected")
        main = refs / "main"
        main_stat = _lstat_or_none(main)
        if main_stat is None:
            return PreviousMainRef(repo_dir, None, True)
        if is_redirect_stat(main_stat) or not stat.S_ISREG(main_stat.st_mode):
            return PreviousMainRef(repo_dir, None, False, "refs/main is not a regular file")
        if main_stat.st_size == 0:
            # Debris from a promotion that died between the exclusive create and the
            # write. Treating it as absent keeps the repo pinned: the next promotion
            # reclaims it, where reading it as invalid would let the download run unpinned.
            return PreviousMainRef(repo_dir, None, True)
        if main_stat.st_size > 256:
            return PreviousMainRef(
                repo_dir,
                None,
                False,
                "refs/main is too large",
                allow_unpinned_download = True,
            )
        try:
            raw_revision = _read_main_ref_text(main)
        except UnicodeError:
            return PreviousMainRef(
                repo_dir,
                None,
                False,
                "refs/main is invalid",
                allow_unpinned_download = True,
            )
        except OSError as exc:
            return PreviousMainRef(
                repo_dir,
                None,
                False,
                f"refs/main could not be read: {exc}",
                allow_unpinned_download = True,
            )
        revision = _parse_main_ref(raw_revision)
        if revision is None:
            return PreviousMainRef(
                repo_dir,
                None,
                False,
                "refs/main is invalid",
                allow_unpinned_download = True,
            )
        return PreviousMainRef(repo_dir, revision, True)
    except OSError as exc:
        return PreviousMainRef(
            repo_dir,
            None,
            False,
            str(exc),
            allow_unpinned_download = True,
        )
    except RuntimeError as exc:
        return PreviousMainRef(repo_dir, None, False, str(exc))


def _validated_snapshot_target(
    repo_type: str, repo_id: str, revision: str, snapshot_path: str | Path
) -> tuple[Path, Path]:
    repo_label = repo_type.capitalize()
    normalized_revision = _normalized_revision(revision)
    if normalized_revision is None:
        raise ValueError(f"{repo_label} metadata did not provide a safe commit revision")
    if not is_valid_repo_id(repo_id):
        raise ValueError(f"Invalid Hugging Face repo id: {repo_id!r}")

    raw_snapshot = Path(snapshot_path).expanduser()
    raw_snapshots = raw_snapshot.parent
    raw_repo = raw_snapshots.parent
    expected_name = f"{_repo_cache_prefix(repo_type)}--{repo_id.replace('/', '--')}"
    if (
        raw_snapshot.name != normalized_revision
        or raw_snapshots.name != "snapshots"
        or raw_repo.name.casefold() != expected_name.casefold()
    ):
        raise ValueError(f"Unexpected {repo_type} snapshot path: {snapshot_path}")
    hub_root = raw_repo.parent.resolve(strict = True)
    repo_dir = _require_real_directory(
        raw_repo,
        hub_root / raw_repo.name,
        "Hub cache repo",
    )
    snapshots = _require_real_directory(
        raw_snapshots,
        repo_dir / "snapshots",
        "Hub snapshots path",
    )
    snapshot = _require_real_directory(
        raw_snapshot,
        snapshots / normalized_revision,
        "Hub snapshot",
    )
    return repo_dir, snapshot


def _stat_fingerprint(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_mode,
    )


def _read_stable_ref_bytes(
    path: Path,
    before: os.stat_result,
    *,
    label: str = "refs/main",
) -> bytes:
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_size > 256
                or _stat_fingerprint(opened) != _stat_fingerprint(before)
            ):
                raise ConcurrentMainRefError(f"{label} changed while being opened")
            payload = handle.read(257)
            opened_after = os.fstat(handle.fileno())
    except ConcurrentMainRefError:
        raise
    except OSError as exc:
        raise ConcurrentMainRefError(f"{label} became unreadable: {exc}") from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise ConcurrentMainRefError(f"{label} changed while being read: {exc}") from exc
    if (
        len(payload) > 256
        or _stat_fingerprint(opened) != _stat_fingerprint(opened_after)
        or _stat_fingerprint(opened_after) != _stat_fingerprint(after)
    ):
        raise ConcurrentMainRefError(f"{label} changed while being read")
    return payload


def _is_os_metadata_dropping(path: Path) -> bool:
    """True for a file browser dropping in ``refs``, which is not a ref.

    A real ``.DS_Store`` is kilobytes, so it trips the size guard and fails the
    whole scan; ``Thumbs.db`` and ``desktop.ini`` are the Explorer equivalents.
    ``._`` companions are settled by magic bytes, not the prefix, since a ref
    could be named that way. ``.icloud`` is deliberately absent: it is an evicted
    ref, still pinning a revision, so skipping it would let reclaim delete a
    live snapshot. Unskipped it fails the commit-shape check and blocks reclaim.
    """
    if path.name.casefold() in _OS_METADATA_REF_NAMES:
        return True
    return is_appledouble_metadata(path)


def _commit_shaped_revision(revision: Optional[str]) -> Optional[str]:
    """*revision* if it has the shape of a Hub commit hash; ``None`` otherwise.

    A ref file holds the 40-character hex commit that names a snapshot
    directory. Requiring that shape stops a short binary dropping -- NUL bytes
    included -- from being counted as a live revision by the mere fact that it
    survived the path checks.
    """
    if revision is None or len(revision) != _COMMIT_REVISION_LENGTH:
        return None
    return revision if set(revision) <= _COMMIT_REVISION_CHARACTERS else None


def referenced_snapshot_revisions(repo_dir: str | Path) -> frozenset[str]:
    revisions: set[str] = set()
    try:
        raw_repo = Path(repo_dir).expanduser()
        repo_stat = raw_repo.lstat()
        if is_redirect_stat(repo_stat) or not stat.S_ISDIR(repo_stat.st_mode):
            raise SnapshotRefsUnverifiable("cache repository is redirected")
        repo = raw_repo.resolve(strict = True)
        raw_refs = raw_repo / "refs"
        refs_stat = _lstat_or_none(raw_refs)
        if refs_stat is None:
            return frozenset()
        refs = _require_real_directory(raw_refs, repo / "refs", "Hub cache refs")

        def raise_walk_error(error: OSError) -> None:
            raise error

        for current, directories, files in os.walk(
            refs,
            topdown = True,
            onerror = raise_walk_error,
            followlinks = False,
        ):
            current_path = Path(current)
            for name in directories:
                try:
                    directory_stat = (current_path / name).lstat()
                except FileNotFoundError as exc:
                    raise SnapshotRefsUnverifiable(
                        f"ref directory changed while being scanned: {name}"
                    ) from exc
                if is_redirect_stat(directory_stat) or not stat.S_ISDIR(directory_stat.st_mode):
                    raise SnapshotRefsUnverifiable(
                        f"ref directory is redirected or invalid: {name}"
                    )
            for name in files:
                ref = current_path / name
                if _is_os_metadata_dropping(ref):
                    continue
                try:
                    before = ref.lstat()
                except FileNotFoundError as exc:
                    raise SnapshotRefsUnverifiable(
                        f"ref changed while being scanned: {name}"
                    ) from exc
                if (
                    is_redirect_stat(before)
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_size > 256
                ):
                    raise SnapshotRefsUnverifiable(f"ref is redirected or invalid: {name}")
                payload = _read_stable_ref_bytes(
                    ref,
                    before,
                    label = f"ref {ref.relative_to(refs)}",
                )
                try:
                    revision = _parse_main_ref(payload.decode("utf-8"))
                except UnicodeError as exc:
                    raise SnapshotRefsUnverifiable(f"ref is not valid UTF-8: {name}") from exc
                commit = _commit_shaped_revision(revision)
                if commit is None:
                    raise SnapshotRefsUnverifiable(f"ref is invalid: {name}")
                revisions.add(commit)
    except SnapshotRefsUnverifiable:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SnapshotRefsUnverifiable(f"refs scan failed ({type(exc).__name__}: {exc})") from exc
    return frozenset(revisions)


def _read_ref_path(main: Path) -> tuple[Optional[str], Optional[int]]:
    main_stat = _lstat_or_none(main)
    if main_stat is None:
        return None, None
    if not stat.S_ISREG(main_stat.st_mode) or main_stat.st_size > 256:
        raise ConcurrentMainRefError("refs/main is not a safe regular ref")
    try:
        raw_revision = _read_stable_ref_bytes(main, main_stat).decode("utf-8")
    except UnicodeError as exc:
        raise ConcurrentMainRefError(f"refs/main became unreadable: {exc}") from exc
    normalized = _parse_main_ref(raw_revision)
    if normalized is None:
        raise ConcurrentMainRefError("refs/main is not a valid revision")
    return normalized, stat.S_IMODE(main_stat.st_mode)


def _read_main_ref(refs: Path) -> tuple[Optional[str], Optional[int]]:
    return _read_ref_path(refs / "main")


def _assert_main_unchanged(refs: Path, expected: Optional[str]) -> Optional[int]:
    current, mode = _read_main_ref(refs)
    if current != expected:
        expected_label = expected or "absent"
        current_label = current or "absent"
        raise ConcurrentMainRefError(
            f"refs/main changed during download ({expected_label} -> {current_label}); "
            "leaving the verified snapshot unpromoted"
        )
    return mode


def _retry_main_ref_change(
    change: Callable[[], None], refs: Path, expected_previous: Optional[str]
) -> None:
    for attempt in range(len(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS) + 1):
        try:
            change()
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == len(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS[attempt])
            _assert_main_unchanged(refs, expected_previous)


def _hardlink_fallback_allowed(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    if winerror is not None:
        return winerror in _HARDLINK_FALLBACK_WINERRORS
    return exc.errno in _HARDLINK_FALLBACK_ERRNOS


def _libc_rename_noreplace(
    function_name: str, argument_types: list, arguments: tuple, source: Path, destination: Path
) -> bool:
    try:
        operation = getattr(ctypes.CDLL(None, use_errno = True), function_name)
    except (AttributeError, OSError):
        return False
    operation.argtypes = argument_types
    operation.restype = ctypes.c_int
    ctypes.set_errno(0)
    if operation(*arguments) == 0:
        return True
    error_code = ctypes.get_errno() or errno.EIO
    if error_code == errno.EEXIST:
        raise FileExistsError(error_code, os.strerror(error_code), destination)
    if error_code in _NATIVE_NOREPLACE_UNSUPPORTED_ERRNOS:
        return False
    raise OSError(
        error_code,
        os.strerror(error_code),
        source,
        None,
        destination,
    )


def _rename_noreplace(source: Path, destination: Path) -> bool:
    if _IS_WINDOWS:
        _retry_main_ref_change(
            lambda: os.rename(source, destination),
            destination.parent,
            None,
        )
        return True
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if _IS_LINUX:
        return _libc_rename_noreplace(
            "renameat2",
            [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ],
            (-100, source_bytes, -100, destination_bytes, 1),
            source,
            destination,
        )
    if _IS_DARWIN:
        return _libc_rename_noreplace(
            "renamex_np",
            [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint],
            (source_bytes, destination_bytes, 0x00000004),
            source,
            destination,
        )
    return False


def _read_ref_bytes(path: Path) -> tuple[bytes, int]:
    before = path.lstat()
    if is_redirect_stat(before) or not stat.S_ISREG(before.st_mode) or before.st_size > 256:
        raise ConcurrentMainRefError("refs/main source is not a safe regular ref")
    payload = _read_stable_ref_bytes(path, before)
    try:
        raw_revision = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ConcurrentMainRefError(f"refs/main source became unreadable: {exc}") from exc
    if _parse_main_ref(raw_revision) is None:
        raise ConcurrentMainRefError("refs/main source is not a valid revision")
    return payload, stat.S_IMODE(before.st_mode)


def _unlink_created_ref(path: Path, created_stat: os.stat_result) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _MainRefCleanupError(f"Created refs/main could not be inspected: {path}") from exc
    if not os.path.samestat(created_stat, current):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _MainRefCleanupError(f"Created refs/main could not be removed: {path}") from exc


def _assert_created_ref(path: Path, created_stat: os.stat_result) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise ConcurrentMainRefError("refs/main changed during promotion") from exc
    if not os.path.samestat(created_stat, current):
        raise ConcurrentMainRefError("refs/main changed during promotion")


def _reclaim_empty_main_ref(main: Path) -> bool:
    """Drop a zero-length ``refs/main``, which only a crashed promotion leaves.

    No writer publishes an empty ref, so removing it lets the next promotion
    through instead of stranding the repo on a ref that can never parse.
    """
    try:
        current = main.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(current.st_mode) or current.st_size:
        return False
    try:
        os.unlink(main)
    except OSError:
        return False
    return True


def _copy_main_ref_exclusively(refs: Path, payload: bytes, mode: int) -> None:
    """Create ``refs/main`` exclusively and write *payload* into it.

    Last resort for a cache with neither usable hard links nor a no-replace
    rename (FAT/exFAT, SMB, some FUSE). The exclusive create is the only atomic
    claim left there, so it is what stops a foreign writer's ref being
    overwritten. A crash before the write leaves an empty ref, reclaimed below
    and read as absent by capture rather than silently unpinning the repo.
    """
    main = refs / "main"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for attempt in range(len(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS) + 1):
        try:
            fd = os.open(main, flags, mode)
            break
        except FileExistsError:
            # Only our own crash debris is reclaimable; a real ref still wins.
            if not _reclaim_empty_main_ref(main):
                raise
            fd = os.open(main, flags, mode)
            break
        except PermissionError:
            if not _IS_WINDOWS or attempt == len(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS[attempt])
            _assert_main_unchanged(refs, None)

    created_stat = os.fstat(fd)
    try:
        _assert_created_ref(main, created_stat)
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(errno.EIO, "Could not write the complete refs/main value")
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(fd, mode)
        os.fsync(fd)
        _assert_created_ref(main, created_stat)
    except BaseException as operation_error:
        close_error = None
        try:
            os.close(fd)
        except OSError as exc:
            close_error = exc
        try:
            _unlink_created_ref(main, created_stat)
        except _MainRefCleanupError as cleanup_error:
            raise cleanup_error from operation_error
        if close_error is not None:
            raise close_error from operation_error
        raise
    else:
        try:
            os.close(fd)
        except OSError as close_error:
            try:
                _unlink_created_ref(main, created_stat)
            except _MainRefCleanupError as cleanup_error:
                raise cleanup_error from close_error
            raise


def _create_main_ref(refs: Path, source: Path, *, try_hardlink: bool) -> None:
    payload, mode = _read_ref_bytes(source)
    if try_hardlink:
        try:
            _retry_main_ref_change(
                lambda: os.link(source, refs / "main"),
                refs,
                None,
            )
            return
        except FileExistsError:
            raise
        except OSError as exc:
            if not _hardlink_fallback_allowed(exc):
                raise
    if _rename_noreplace(source, refs / "main"):
        return
    _copy_main_ref_exclusively(refs, payload, mode)


def _publish_main_ref(
    refs: Path,
    staging: Path,
    temporary: Path,
    expected_previous: Optional[str],
    previous_mode: Optional[int],
) -> None:
    main = refs / "main"
    if expected_previous is None:
        try:
            _create_main_ref(refs, temporary, try_hardlink = True)
        except FileExistsError as exc:
            raise ConcurrentMainRefError(
                "refs/main was created during download; leaving the verified snapshot unpromoted"
            ) from exc
        return

    probe = staging / f".unsloth-main-link-{uuid.uuid4().hex}"
    hardlinks_available = True
    try:
        os.link(temporary, probe)
    except FileExistsError:
        raise
    except OSError as exc:
        if not _hardlink_fallback_allowed(exc):
            raise
        hardlinks_available = False
    else:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass

    displaced = staging / f".unsloth-main-previous-{uuid.uuid4().hex}"
    claimed = False
    try:
        _retry_main_ref_change(
            lambda: os.rename(main, displaced),
            refs,
            expected_previous,
        )
        claimed = True
        claimed_revision, claimed_mode = _read_ref_path(displaced)
        if claimed_revision != expected_previous:
            raise ConcurrentMainRefError(
                f"refs/main changed during download ({expected_previous} -> "
                f"{claimed_revision or 'absent'}); leaving the verified snapshot unpromoted"
            )
        if claimed_mode is not None and claimed_mode != previous_mode:
            os.chmod(temporary, claimed_mode)
        try:
            _create_main_ref(
                refs,
                temporary,
                try_hardlink = hardlinks_available,
            )
        except FileExistsError as exc:
            raise ConcurrentMainRefError(
                "refs/main was updated during promotion; leaving the external revision active"
            ) from exc
    except BaseException as promotion_error:
        if claimed:
            if isinstance(promotion_error, _MainRefCleanupError):
                claimed = False
                raise RuntimeError(
                    f"refs/main could not be cleaned; previous ref remains at {displaced}"
                ) from promotion_error
            try:
                _create_main_ref(
                    refs,
                    displaced,
                    try_hardlink = hardlinks_available,
                )
            except FileExistsError:
                pass
            except Exception as restore_error:
                claimed = False
                raise RuntimeError(
                    f"refs/main could not be restored; previous ref remains at {displaced}"
                ) from restore_error
            try:
                displaced.unlink()
            except OSError:
                pass
            claimed = False
        raise
    else:
        if claimed:
            try:
                displaced.unlink()
            except OSError:
                pass


def _atomic_write_main_ref(repo_dir: Path, revision: str, expected_previous: Optional[str]) -> None:
    refs = repo_dir / "refs"
    try:
        refs_stat = refs.lstat()
    except FileNotFoundError:
        try:
            refs.mkdir(parents = False)
        except FileExistsError:
            pass
        refs_stat = refs.lstat()
    if is_redirect_stat(refs_stat) or not stat.S_ISDIR(refs_stat.st_mode):
        raise ValueError(f"Hub cache refs is not a real directory: {refs}")
    refs_real = refs.resolve(strict = True)
    if refs_real != repo_dir / "refs":
        raise ValueError(f"Hub cache refs escapes its repo: {refs}")

    previous_mode = _assert_main_unchanged(refs_real, expected_previous)
    main = refs_real / "main"

    staging, temporary, fd = _open_staged_temporary(repo_dir)
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            handle.write(revision)
            handle.flush()
            os.fsync(handle.fileno())
        if previous_mode is not None:
            os.chmod(temporary, previous_mode)
        latest_mode = _assert_main_unchanged(refs_real, expected_previous)
        if latest_mode is not None and latest_mode != previous_mode:
            os.chmod(temporary, latest_mode)
        _assert_main_unchanged(refs_real, expected_previous)
        _publish_main_ref(
            refs_real,
            staging,
            temporary,
            expected_previous,
            latest_mode,
        )
        if os.name != "nt":
            try:
                directory_fd = os.open(refs_real, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        _discard_refs_staging_directory(staging)


def promote_verified_snapshot(
    repo_type: str,
    repo_id: str,
    revision: str,
    snapshot_path: str | Path,
    previous: Optional[PreviousMainRef],
    *,
    after_promotion: Optional[Callable[[], None]] = None,
) -> SnapshotPromotionResult:
    """Promote *revision* if ``refs/main`` matches; run the callback under the same lock."""
    normalized_revision = _normalized_revision(revision)
    if normalized_revision is None:
        raise ValueError("Hub metadata did not provide a safe commit revision")
    repo_dir, _snapshot = _validated_snapshot_target(
        repo_type,
        repo_id,
        normalized_revision,
        snapshot_path,
    )
    previous_revision = previous.revision if previous is not None else None
    if (
        previous is None
        or previous.repo_dir is None
        or not _same_existing_path(previous.repo_dir, repo_dir)
    ):
        raise ValueError("Verified snapshot is outside the captured active Hub cache")
    if not previous.promotion_safe:
        raise ConcurrentMainRefError(
            f"refs/main could not be safely captured: {previous.reason or 'unknown error'}"
        )

    with _main_ref_lock(repo_dir):
        try:
            _atomic_write_main_ref(repo_dir, normalized_revision, previous_revision)
        except ConcurrentMainRefError:
            current_revision, _mode = _read_main_ref(repo_dir / "refs")
            if current_revision != normalized_revision:
                raise
        if after_promotion is not None:
            after_promotion()
    return SnapshotPromotionResult(previous_revision)

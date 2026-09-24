# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reuse files a new Hub revision did not change instead of downloading them again.

huggingface_hub dedups across revisions only through ``blobs/<etag>``: a new commit's snapshot
entry becomes a symlink to the blob that is already there. Where symlinks cannot be created
(Windows without Developer Mode or admin, FAT/exFAT, ``HF_HUB_DISABLE_SYMLINKS``),
``file_download._create_symlink`` MOVES a freshly downloaded blob into the snapshot instead, so
``blobs/`` stays empty. The next commit, even one that only edits the README, then finds no blob,
and ``hf_hub_download`` fetches every file again into the new ``snapshots/<commit>/`` directory.

Before a download starts, this module looks for the same relative path in the repo's other
snapshots. A candidate must be a regular file (a symlink means the blob layout, which
huggingface_hub already reuses) with the declared size, and its content must match the digest
the Hub reports for the target commit (LFS sha256, or the git blob id for small files). The
download worker proves that by hashing the local file right before placing it.
A matching file is hard linked into the target snapshot, or copied when hard links are
unavailable. huggingface_hub then finds the pointer path present and skips the file
(``os.path.exists(pointer_path)``).

Plans cannot hash multi-GB files on the request path, so ``reusable_paths`` also accepts the Hub
reporting the same digest for the candidate's own commit, or a digest the worker persisted
earlier. That is an estimate only: the worker re-proves the bytes before placing anything, and
downloads when they differ.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_HASH_CHUNK = 8 * 1024 * 1024
_DIGEST_CACHE_NAME = "file_digests.json"
_DIGEST_CACHE_LIMIT = 2048
_PATHS_INFO_BATCH = 100
_digest_cache_lock = threading.Lock()

# (commit, paths) -> {path: digest}. Digest is the LFS sha256, else the git blob id, the same
# value ``gguf_plan.sibling_sha256`` returns for a sibling.
RemoteDigests = Callable[[str, Sequence[str]], Mapping[str, str]]


@dataclass(frozen = True)
class ReuseResult:
    reused: tuple[str, ...] = ()
    reused_bytes: int = 0
    linked: int = 0
    copied: int = 0
    hashed_bytes: int = 0


def digest_kind(digest: Optional[str]) -> Optional[str]:
    if not isinstance(digest, str):
        return None
    if _SHA256.match(digest):
        return "sha256"
    if _GIT_SHA1.match(digest):
        return "git-sha1"
    return None


def file_digest(path: Path, kind: str) -> str:
    """sha256 of the content, or the git blob id (sha1 over ``blob <size>\\0`` + content)."""
    if kind == "sha256":
        h = hashlib.sha256()
    elif kind == "git-sha1":
        h = hashlib.sha1()
        h.update(b"blob %d\x00" % os.stat(path).st_size)
    else:
        raise ValueError(f"unsupported digest kind: {kind}")
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _digest_cache_path() -> Optional[Path]:
    try:
        from hub.utils.state_dir import state_root
        root = state_root(create = True)
    except Exception:  # noqa: BLE001 - the cache is an optimisation
        return None
    return None if root is None else root / _DIGEST_CACHE_NAME


def _digest_cache_key(path: Path, kind: str, st: os.stat_result) -> str:
    # Keyed by identity AND content markers, so a rewritten file usually misses. On POSIX ctime
    # also moves when mtime is restored; Windows reports creation time there, so it adds nothing.
    return "|".join(
        (
            kind,
            os.path.normcase(os.path.abspath(str(path))),
            str(st.st_size),
            str(st.st_mtime_ns),
            str(getattr(st, "st_ino", 0)),
            "" if os.name == "nt" else str(st.st_ctime_ns),
        )
    )


def _read_digest_cache(cache_path: Path) -> dict:
    try:
        with open(cache_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _remember_digest(key: str, digest: str) -> None:
    cache_path = _digest_cache_path()
    if cache_path is None:
        return
    with _digest_cache_lock:
        data = _read_digest_cache(cache_path)
        data.pop(key, None)
        data[key] = digest
        while len(data) > _DIGEST_CACHE_LIMIT:
            data.pop(next(iter(data)))
        tmp = cache_path.with_name(f".{cache_path.name}.tmp-{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding = "utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, cache_path)
        except OSError as exc:
            logger.debug("could not persist file digest cache: %s", exc)
            try:
                tmp.unlink()
            except OSError:
                pass


def cached_file_digest(
    path: Path,
    kind: str,
    *,
    compute: bool = True,
) -> tuple[Optional[str], int]:
    """(digest, bytes hashed now). With ``compute`` the file is always hashed: stat metadata cannot
    prove the bytes are unchanged (mtime can be restored, FAT/exFAT has 2 s granularity, Windows has
    no change time), so the persisted cache only serves read-only estimates (``compute=False``)."""
    try:
        st = os.stat(path)
    except OSError:
        return None, 0
    key = _digest_cache_key(path, kind, st)
    if not compute:
        cache_path = _digest_cache_path()
        if cache_path is None:
            return None, 0
        cached = _read_digest_cache(cache_path).get(key)
        return (cached, 0) if isinstance(cached, str) and digest_kind(cached) == kind else (None, 0)
    try:
        digest = file_digest(path, kind)
        after = os.stat(path)
    except OSError:
        return None, 0
    if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        return None, st.st_size  # changed while we read it: prove nothing
    _remember_digest(key, digest)
    return digest, st.st_size


def repo_cache_dir(
    repo_type: str,
    repo_id: str,
    hub_cache: Optional[str | Path] = None,
) -> Path:
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    root = Path(hub_cache) if hub_cache else Path(constants.HF_HUB_CACHE)
    return root / repo_folder_name(repo_id = repo_id, repo_type = repo_type)


def _is_regular_file(path: Path) -> bool:
    try:
        return stat_module.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _other_snapshots(repo_dir: Path, commit_hash: str) -> list[Path]:
    try:
        snapshots = [
            p for p in (repo_dir / "snapshots").iterdir() if p.is_dir() and p.name != commit_hash
        ]
    except OSError:
        return []
    from hub.utils.hf_cache_state import snapshot_selection_key
    return sorted(snapshots, key = snapshot_selection_key, reverse = True)


def _candidates(snapshots: Iterable[Path], rel_path: str, size: int) -> list[Path]:
    found = []
    seen: set[tuple[int, int]] = set()
    for snap in snapshots:
        candidate = snap / rel_path
        # A symlink points into blobs/, which huggingface_hub already reuses on its own; only a
        # materialized copy (the no-symlink layout) is invisible to it.
        if not _is_regular_file(candidate):
            continue
        try:
            st = os.stat(candidate)
        except OSError:
            continue
        if st.st_size != size:
            continue
        # Each reuse hard links the same file into one more snapshot: check those bytes once.
        inode = (st.st_dev, st.st_ino)
        if st.st_ino and inode in seen:
            continue
        seen.add(inode)
        found.append(candidate)
    return found


def _needs_reuse(repo_dir: Path, target: Path, digest: str) -> bool:
    if os.path.lexists(target):
        return False  # present (or a link huggingface_hub will repair itself)
    if os.path.exists(repo_dir / "blobs" / digest):
        return False  # huggingface_hub links or copies this blob without downloading
    return True


def find_reusable_copies(
    repo_dir: Path,
    commit_hash: str,
    expected: Mapping[str, tuple[int, str]],
    *,
    remote_digests: Optional[RemoteDigests] = None,
    allow_hashing: bool = True,
) -> tuple[dict[str, Path], int]:
    """Map each path in ``expected`` ({path: (size, digest)}) to a verified identical copy in another
    snapshot of ``repo_dir``. Returns (matches, bytes hashed)."""
    snapshots = _other_snapshots(repo_dir, commit_hash)
    if not snapshots:
        return {}, 0
    candidates = {
        path: _candidates(snapshots, path, size)
        for path, (size, digest) in expected.items()
        if size > 0 and digest_kind(digest) is not None
    }

    matches: dict[str, Path] = {}
    hashed = 0
    # The Hub says an older commit served the same bytes and the local copy has the full size. Good
    # enough for a plan's estimate; the worker passes no remote_digests and hashes instead. Newest
    # snapshot first, one request per commit, and none once every path has a match.
    if remote_digests is not None:
        for snap in snapshots:
            here = {
                path: candidate
                for path, found in candidates.items()
                if path not in matches
                for candidate in found
                if candidate == snap / path
            }
            if not here:
                continue
            try:
                remote = remote_digests(snap.name, sorted(here)) or {}
            except Exception as exc:  # noqa: BLE001 - squashed history, offline, 404: hash instead
                logger.debug("remote digests unavailable for %s: %s", snap.name, exc)
                continue
            for path, candidate in here.items():
                if remote.get(path) == expected[path][1]:
                    matches[path] = candidate
    for path, found in candidates.items():
        if path in matches:
            continue
        kind = digest_kind(expected[path][1])
        for candidate in found:
            local, spent = cached_file_digest(candidate, kind, compute = allow_hashing)
            hashed += spent
            if local == expected[path][1]:
                matches[path] = candidate
                break
    return matches, hashed


def _place(src: Path, dst: Path, size: int) -> Optional[str]:
    """Hard link ``src`` at ``dst`` (copy when linking fails). Atomic: a temp name, then replace."""
    dst.parent.mkdir(parents = True, exist_ok = True)
    # A cancel kills the worker, so a copy cut short never reached the cleanup below.
    for stale in dst.parent.glob(f".{glob.escape(dst.name)}.reuse-*"):
        try:
            stale.unlink()
        except OSError:
            pass
    tmp = dst.with_name(f".{dst.name}.reuse-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    how = None
    try:
        try:
            os.link(src, tmp)
            how = "link"
        except OSError:
            # No hard links here (FAT/exFAT, some network shares). A copy costs the disk space the
            # download would have used anyway, never the bandwidth.
            if shutil.disk_usage(dst.parent).free < size:
                return None
            shutil.copyfile(src, tmp)
            how = "copy"
        if os.path.lexists(dst):
            return None  # a concurrent writer got there first
        os.replace(tmp, dst)
        return how
    except OSError as exc:
        logger.info("could not reuse %s for %s: %s", src, dst, exc)
        return None
    finally:
        try:
            if os.path.lexists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _drop_superseded_partial(repo_dir: Path, digest: str, protected: frozenset[str]) -> None:
    """A resumable ``blobs/<digest>.incomplete`` from an earlier attempt is dead weight once the file
    is in place: huggingface_hub returns the pointer and never reopens it."""
    if digest in protected:
        return
    partial = repo_dir / "blobs" / f"{digest}.incomplete"
    if not partial.is_file():
        return
    try:
        from hub.utils.hf_cache_state import blob_download_lock_held
        if blob_download_lock_held(repo_dir, digest):
            return
        partial.unlink()
    except Exception as exc:  # noqa: BLE001 - leftover partials are swept elsewhere
        logger.debug("could not remove superseded partial %s: %s", partial, exc)


def reuse_unchanged_snapshot_files(
    repo_type: str,
    repo_id: str,
    commit_hash: Optional[str],
    expected_files: Sequence,
    *,
    hub_cache: Optional[str | Path] = None,
    remote_digests: Optional[RemoteDigests] = None,
    allow_hashing: bool = True,
    protected_blob_hashes: frozenset[str] = frozenset(),
) -> ReuseResult:
    """Materialize every file of ``expected_files`` (``ExpectedFile``-like: path, size, sha256) that
    ``snapshots/<commit_hash>/`` lacks but another snapshot holds with identical content. Never
    raises: anything it cannot prove is left for the normal download."""
    try:
        from hub.utils.download_manifest import expected_path_is_safe, normalized_commit_hash

        commit = normalized_commit_hash(commit_hash)
        if not commit:
            return ReuseResult()
        repo_dir = repo_cache_dir(repo_type, repo_id, hub_cache)
        if not (repo_dir / "snapshots").is_dir():
            return ReuseResult()
        target_root = repo_dir / "snapshots" / commit
        pending: dict[str, tuple[int, str]] = {}
        for item in expected_files:
            path = getattr(item, "path", None)
            size = int(getattr(item, "size", 0) or 0)
            digest = getattr(item, "sha256", None)
            if not expected_path_is_safe(path) or size <= 0 or digest_kind(digest) is None:
                continue
            if _needs_reuse(repo_dir, target_root / path, digest):
                pending[path] = (size, digest)
        if not pending:
            return ReuseResult()
        matches, hashed = find_reusable_copies(
            repo_dir,
            commit,
            pending,
            remote_digests = remote_digests,
            allow_hashing = allow_hashing,
        )
        reused, linked, copied, reused_bytes = [], 0, 0, 0
        for path, src in matches.items():
            size, digest = pending[path]
            how = _place(src, target_root / path, size)
            if how is None:
                continue
            reused.append(path)
            reused_bytes += size
            linked += how == "link"
            copied += how == "copy"
            _drop_superseded_partial(repo_dir, digest, protected_blob_hashes)
        return ReuseResult(tuple(reused), reused_bytes, linked, copied, hashed)
    except Exception as exc:  # noqa: BLE001 - reuse is an optimisation, never a failure
        logger.warning("snapshot reuse skipped for %s: %s", repo_id, exc)
        return ReuseResult()


def paths_in_snapshot(
    repo_type: str,
    repo_id: str,
    commit_hash: Optional[str],
    paths: Iterable[str],
    *,
    hub_cache: Optional[str | Path] = None,
) -> set[str]:
    """Paths already present in ``snapshots/<commit_hash>/``, which ``hf_hub_download`` skips (a
    dangling link is not present)."""
    try:
        from hub.utils.download_manifest import expected_path_is_safe, normalized_commit_hash

        commit = normalized_commit_hash(commit_hash)
        if not commit:
            return set()
        root = repo_cache_dir(repo_type, repo_id, hub_cache) / "snapshots" / commit
        return {p for p in paths if expected_path_is_safe(p) and os.path.exists(root / p)}
    except Exception:  # noqa: BLE001 - counting a present file again only makes the preflight stricter
        return set()


def hub_remote_digests(repo_type: str, repo_id: str, token) -> RemoteDigests:
    """A ``RemoteDigests`` backed by ``HfApi.get_paths_info`` (one request per commit)."""

    def lookup(commit: str, paths: Sequence[str]) -> Mapping[str, str]:
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        paths = list(paths)
        infos = []
        for start in range(0, len(paths), _PATHS_INFO_BATCH):
            infos.extend(
                api.get_paths_info(
                    repo_id,
                    paths[start : start + _PATHS_INFO_BATCH],
                    revision = commit,
                    repo_type = repo_type,
                )
            )
        out: dict[str, str] = {}
        for info in infos:
            lfs = getattr(info, "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs is not None else None
            if isinstance(lfs, dict):
                digest = lfs.get("sha256")
            digest = digest or getattr(info, "blob_id", None)
            path = getattr(info, "path", None)
            if isinstance(path, str) and isinstance(digest, str):
                out[path] = digest
        return out

    return lookup


def reusable_paths(
    repo_type: str,
    repo_id: str,
    commit_hash: Optional[str],
    sizes: Mapping[str, int],
    *,
    hub_cache: Optional[str | Path] = None,
    remote_digests: Optional[RemoteDigests] = None,
    allow_hashing: bool = False,
) -> set[str]:
    """Paths of ``sizes`` ({path: declared size}) missing from ``snapshots/<commit_hash>/`` that
    the reuse step would supply from an older snapshot. Read-only, for plans: no network unless a
    same-size local copy exists, and no hashing unless asked (a cached digest still counts)."""
    try:
        from hub.utils.download_manifest import expected_path_is_safe, normalized_commit_hash

        commit = normalized_commit_hash(commit_hash)
        if not commit or not sizes:
            return set()
        repo_dir = repo_cache_dir(repo_type, repo_id, hub_cache)
        snapshots = _other_snapshots(repo_dir, commit)
        if not snapshots:
            return set()
        target_root = repo_dir / "snapshots" / commit
        local = [
            path
            for path, size in sizes.items()
            if expected_path_is_safe(path)
            and int(size or 0) > 0
            and not os.path.lexists(target_root / path)
            and _candidates(snapshots, path, int(size))
        ]
        if not local or remote_digests is None:
            return set()
        target = remote_digests(commit, sorted(local)) or {}
        expected = {
            path: (int(sizes[path]), target[path])
            for path in local
            if digest_kind(target.get(path)) is not None
            and not os.path.exists(repo_dir / "blobs" / target[path])
        }
        matches, _ = find_reusable_copies(
            repo_dir,
            commit,
            expected,
            remote_digests = remote_digests,
            allow_hashing = allow_hashing,
        )
        return set(matches)
    except Exception as exc:  # noqa: BLE001 - a plan must never fail over an optimisation
        logger.debug("reusable_paths skipped for %s: %s", repo_id, exc)
        return set()

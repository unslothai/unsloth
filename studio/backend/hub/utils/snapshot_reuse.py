# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reuse unchanged files from older snapshots on a no-symlink cache, where ``file_download._create_symlink`` moves each blob into its snapshot and leaves ``blobs/`` empty."""

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
# Old snapshots a plan asks the Hub about; the rest stay counted (the plan is on the request path).
_REMOTE_DIGEST_COMMITS = 3
_digest_cache_lock = threading.Lock()

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


def _remember_digests(entries: Mapping[str, str]) -> None:
    cache_path = _digest_cache_path()
    if cache_path is None or not entries:
        return
    with _digest_cache_lock:
        data = _read_digest_cache(cache_path)
        for key, digest in entries.items():
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
    learned: Optional[dict] = None,
) -> tuple[Optional[str], int]:
    """(digest, bytes hashed). Stat metadata cannot prove unchanged bytes, so the persisted cache only serves ``compute=False`` estimates."""
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
        return None, st.st_size
    if learned is None:
        _remember_digests({key: digest})
    else:
        learned[key] = digest  # the caller persists a whole pass in one write
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


def cached_ref_commit(
    repo_type: str,
    repo_id: str,
    hub_cache: Optional[str | Path] = None,
    ref: str = "main",
) -> Optional[str]:
    try:
        from hub.utils.download_manifest import normalized_commit_hash
        text = (repo_cache_dir(repo_type, repo_id, hub_cache) / "refs" / ref).read_text(
            encoding = "utf-8"
        )
        return normalized_commit_hash(text.strip())
    except Exception:  # noqa: BLE001 - no readable ref means nothing to compare against
        return None


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
        # A symlink means the blob layout, which huggingface_hub already reuses.
        if not _is_regular_file(candidate):
            continue
        try:
            st = os.stat(candidate)
        except OSError:
            continue
        if st.st_size != size:
            continue
        inode = (st.st_dev, st.st_ino)
        if st.st_ino and inode in seen:
            continue
        seen.add(inode)
        found.append(candidate)
    return found


def _needs_reuse(repo_dir: Path, target: Path, digest: str) -> bool:
    if os.path.lexists(target):
        return False
    if os.path.exists(repo_dir / "blobs" / digest):
        return False
    return True


def find_reusable_copies(
    repo_dir: Path,
    commit_hash: str,
    expected: Mapping[str, tuple[int, str]],
    *,
    remote_digests: Optional[RemoteDigests] = None,
    allow_hashing: bool = True,
) -> tuple[dict[str, Path], int]:
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
    # Plan estimate only: the worker passes no remote_digests and hashes instead.
    if remote_digests is not None:
        asked = 0
        for snap in snapshots:
            if asked >= _REMOTE_DIGEST_COMMITS:
                break
            here = {
                path: candidate
                for path, found in candidates.items()
                if path not in matches
                for candidate in found
                if candidate == snap / path
            }
            if not here:
                continue
            asked += 1
            try:
                remote = remote_digests(snap.name, sorted(here)) or {}
            except Exception as exc:  # noqa: BLE001 - squashed history, offline, 404: hash instead
                logger.debug("remote digests unavailable for %s: %s", snap.name, exc)
                continue
            for path, candidate in here.items():
                if remote.get(path) == expected[path][1]:
                    matches[path] = candidate
    learned: dict[str, str] = {}
    for path, found in candidates.items():
        if path in matches:
            continue
        kind = digest_kind(expected[path][1])
        for candidate in found:
            local, spent = cached_file_digest(
                candidate, kind, compute = allow_hashing, learned = learned
            )
            hashed += spent
            if local == expected[path][1]:
                matches[path] = candidate
                break
    _remember_digests(learned)
    return matches, hashed


def _place(src: Path, dst: Path, size: int) -> Optional[str]:
    dst.parent.mkdir(parents = True, exist_ok = True)
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
            if shutil.disk_usage(dst.parent).free < size:
                return None
            shutil.copyfile(src, tmp)
            how = "copy"
        if os.path.lexists(dst):
            return None
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
    if digest in protected:
        return
    partial = repo_dir / "blobs" / f"{digest}.incomplete"
    if not partial.is_file():
        return
    lock_path = repo_dir.parent / ".locks" / repo_dir.name / f"{digest}.lock"
    try:
        from filelock import FileLock, Timeout
        lock_path.parent.mkdir(parents = True, exist_ok = True)
        # Held, not probed: huggingface_hub writes <digest>.incomplete only inside this lock, so
        # while we hold it no partial is live, and a peer arriving now waits instead of losing it.
        with FileLock(str(lock_path), timeout = 0):
            partial.unlink(missing_ok = True)
    except Timeout:
        return
    except Exception as exc:  # noqa: BLE001 - no flock here (SoftFileLock territory): leave it
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
    """Never raises: anything it cannot prove is left for the normal download."""
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
            # A peer is fetching this blob: a pointer placed now makes huggingface_hub keep its
            # finished blob in blobs/ as a second copy (it links only when the pointer is absent).
            if digest in protected_blob_hashes:
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
        from hub.utils.hf_cache_state import blob_download_lock_held

        for path, src in matches.items():
            size, digest = pending[path]
            # Rechecked here, not only at launch: a peer may have started on this blob while we hashed.
            if blob_download_lock_held(repo_dir, digest):
                continue
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
    digest_revision: Optional[str] = None,
) -> set[str]:
    """Read-only plan estimate: no network unless a same-size local copy exists."""
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
        # digest_revision names what the worker will fetch when commit_hash is only the local ref.
        target = remote_digests(digest_revision or commit, sorted(local)) or {}
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

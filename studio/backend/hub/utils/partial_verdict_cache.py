# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-repo partial verdict of the local model listing, kept across requests.

The listing judges every cached repo partial or not on every request: Hub state candidates,
``.incomplete`` blobs, broken links and the quant / shard sets of the newest snapshot. All of it
reads the repo dir and the Studio Hub state only, and every writer of those (huggingface_hub, the
Studio download worker, a delete) creates, renames or unlinks a directory entry, which changes the
containing directory's mtime. So a verdict is kept with a signature of exactly those directories:

* the repo dir, ``snapshots/``, ``blobs/``, ``refs/`` and every directory inside each snapshot
  (a new snapshot, a repointed link, an ``.incomplete`` appearing, completing by rename or being
  removed, a blob deleted);
* every file under ``refs/`` (``refs/main`` is rewritten in place);
* every regular file inside a snapshot and every link target outside ``blobs/`` (both can be
  rewritten in place; blobs are content-addressed and only ever renamed into place);
* for a blob that links into huggingface_hub's shared store (``<hub>/blobs/<xx>/<xet hash>``),
  the store file and its directory, since another repo's delete can sweep it;
* one token per scan over the whole Hub state tree (each manifest and cancel marker, plus the
  state root and the configured cache), so any state write misses every repo.

A repeat request stats those directories instead of walking the files. Nothing is stored when a
stat fails with anything but "absent", when a timestamp is too recent to trust (a change in the
same clock tick, or within FAT's 2 s mtime, would not move it), when the signature moved while the
verdict was computed, or when an invalidation ran meanwhile. A snapshot holding a linked
directory is not cached. Only Linux local filesystems known to move a directory's mtime on every
entry change are cached (ext4, xfs, btrfs, zfs, tmpfs, ...): FAT / exFAT drivers, network and FUSE
mounts (object-store mounts keep no directory mtimes; NFS caches them), macOS and Windows keep
judging every repo on every request. Entries expire after a few minutes, so a verdict computed
through a transient error the checks swallowed is not kept for long.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Hashable, NamedTuple, Optional

from utils.scan_memo import memoized, scan_memo_active

_MAX_ENTRIES = 4096
# A change inside this window of the signature can share its timestamp (coarse clocks, FAT 2 s).
_RACY_NS = 3_000_000_000
# The checks swallow most OSErrors, so a transient one (EMFILE, EIO) can store a wrong verdict the signature
# cannot see; recompute after a minute so it does not outlive the error for long.
_MAX_AGE_NS = 60_000_000_000
_ABSENT_ERRNOS = frozenset({2, 20})  # ENOENT, ENOTDIR
# Local filesystems whose directory mtime moves on every create, rename and unlink in it.
_TRUSTED_FS_TYPES = frozenset(
    {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "f2fs",
        "bcachefs",
        "jfs",
        "nilfs2",
        "reiserfs",
        "tmpfs",
        "ramfs",
        "overlay",
    }
)

_lock = threading.Lock()
_entries: "OrderedDict[Hashable, _Entry]" = OrderedDict()
_generation = 0
_last_token: Optional[tuple] = None


class _Unstable(Exception):
    """A stat failed for a reason other than absence; nothing is cached or served."""


class _Entry(NamedTuple):
    verdict: bool
    token: tuple
    dirs: tuple
    files: tuple
    signature: tuple
    stored_ns: int


def enabled() -> bool:
    return sys.platform.startswith("linux")


def _mount_fs_types() -> dict[tuple[int, int], set[str]]:
    types: dict[tuple[int, int], set[str]] = {}
    with open("/proc/self/mountinfo", encoding = "utf-8", errors = "replace") as handle:
        for line in handle:
            fields = line.split()
            try:
                major, minor = fields[2].split(":")
                fs_type = fields[fields.index("-", 6) + 1]
            except (IndexError, ValueError):
                continue
            types.setdefault((int(major), int(minor)), set()).add(fs_type)
    return types


def _trusted_devices(devices: set[int]) -> bool:
    """Whether every device is a mount of a trusted filesystem, read once per scan."""
    try:
        types = memoized(("partial_verdict_mount_types",), _mount_fs_types)
    except (OSError, ValueError):
        return False
    for dev in devices:
        found = types.get((os.major(dev), os.minor(dev)))
        # A device mounted twice with different types, or not listed at all, is not vouched for.
        if not found or len(found) != 1 or not found <= _TRUSTED_FS_TYPES:
            return False
    return True


def _devices(stats) -> set[int]:
    return {s[0] for s in stats if s is not None}


def invalidate() -> None:
    global _generation
    with _lock:
        _generation += 1
        _entries.clear()


def _stat(path: str) -> Optional[tuple]:
    try:
        st = os.stat(path)
    except OSError as exc:
        if exc.errno in _ABSENT_ERRNOS:
            return None
        raise _Unstable from exc
    except ValueError as exc:
        raise _Unstable from exc
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _scandir(path: str) -> list:
    try:
        with os.scandir(path) as it:
            return list(it)
    except OSError as exc:
        if exc.errno in _ABSENT_ERRNOS:
            return []
        raise _Unstable from exc


def _newest(stats) -> int:
    return max((max(s[4], s[5]) for s in stats if s is not None), default = 0)


def _compute_token() -> Optional[tuple]:
    from hub.utils import state_dir

    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        configured = os.fspath(get_hf_cache_paths().hub_cache)
    except Exception:
        configured = None
    try:
        root = state_dir.state_root()
        if root is None:
            return None
        items = []
        pending = [os.path.join(os.fspath(root), name) for name in ("manifests", "cancelled")]
        while pending:
            directory = pending.pop()
            items.append((directory, _stat(directory)))
            for entry in _scandir(directory):
                if entry.is_dir(follow_symlinks = False):
                    pending.append(entry.path)
                else:
                    items.append((entry.path, _stat(entry.path)))
        hub_real = os.path.realpath(configured) if configured else None
    except (_Unstable, OSError):
        return None
    items.sort()
    return (os.fspath(root), configured, hub_real, tuple(items))


def scan_token() -> Optional[tuple]:
    """The Hub state token for this scan, taken once; call it before the state is first read."""
    if not enabled() or not scan_memo_active():
        return None
    return memoized(("partial_verdict_token",), _scan_token_uncached)


def _scan_token_uncached() -> Optional[tuple]:
    global _last_token
    start_ns = time.time_ns()
    token = _compute_token()
    if token is None:
        return None
    stats = [stat for _path, stat in token[3]]
    # Too fresh to vouch for, or on a filesystem whose directory mtimes cannot be trusted:
    # this scan neither stores nor serves.
    if _newest(stats) > start_ns - _RACY_NS or not _trusted_devices(_devices(stats)):
        return None
    with _lock:
        # Interned, so entries compare by identity.
        if token == _last_token:
            return _last_token
        _last_token = token
        return token


def _walk_repo(repo_dir: str) -> Optional[tuple[tuple, tuple]]:
    """Directories and files whose stats decide the verdict; None when the layout cannot be vouched for."""
    blobs = os.path.join(repo_dir, "blobs")
    snapshots = os.path.join(repo_dir, "snapshots")
    refs = os.path.join(repo_dir, "refs")
    # Snapshot links resolve from where snapshots/ really is, so a linked one points them at blobs this walk would not stat.
    if os.path.islink(snapshots):
        return None
    dirs = [repo_dir, snapshots, blobs, refs]
    files: list[str] = []
    pending = [refs]
    while pending:
        directory = pending.pop()
        for entry in _scandir(directory):
            if entry.is_symlink():
                return None
            if entry.is_dir(follow_symlinks = False):
                dirs.append(entry.path)
                pending.append(entry.path)
            else:
                files.append(entry.path)
    blob_entries: Optional[dict[str, os.DirEntry]] = None
    blobs_real: Optional[str] = None
    linked_blobs: set[str] = set()
    for snapshot in _scandir(snapshots):
        if snapshot.is_symlink():
            return None
        if not snapshot.is_dir(follow_symlinks = False):
            files.append(snapshot.path)
            continue
        dirs.append(snapshot.path)
        pending = [snapshot.path]
        while pending:
            directory = pending.pop()
            for entry in _scandir(directory):
                if entry.is_symlink():
                    try:
                        target = os.readlink(entry.path)
                    except OSError:
                        raise _Unstable from None
                    target = os.path.normpath(os.path.join(directory, target))
                    parent = os.path.dirname(target)
                    if parent == blobs:
                        if blob_entries is None:
                            # One listing answers every link; an absent name is a broken link.
                            blob_entries = {blob.name: blob for blob in _scandir(blobs)}
                        blob = blob_entries.get(os.path.basename(target))
                        if blob is None or blob.path in linked_blobs:
                            continue
                        if blob.is_symlink():
                            # huggingface_hub's shared store: blobs/<etag> links to <hub>/blobs/<xx>/<xet hash>, a file
                            # that can be swept or replaced from another repo, so it is tracked like an outside target.
                            # Resolved from where blobs/ really is, as the kernel does for a relative link.
                            linked_blobs.add(blob.path)
                            try:
                                shared = os.readlink(blob.path)
                            except OSError:
                                raise _Unstable from None
                            if blobs_real is None:
                                blobs_real = os.path.realpath(blobs)
                            shared = os.path.normpath(os.path.join(blobs_real, shared))
                            if os.path.islink(shared) or os.path.isdir(shared):
                                return None
                            dirs.append(os.path.dirname(shared))
                            files.append(shared)
                        elif blob.is_dir(follow_symlinks = False):
                            return None
                    else:
                        if os.path.islink(target) or os.path.isdir(target):
                            return None
                        dirs.append(parent)
                        files.append(target)
                elif entry.is_dir(follow_symlinks = False):
                    dirs.append(entry.path)
                    pending.append(entry.path)
                else:
                    files.append(entry.path)
    return tuple(dict.fromkeys(dirs)), tuple(dict.fromkeys(files))


def _signature(dirs: tuple, files: tuple) -> tuple:
    return tuple(_stat(p) for p in dirs) + tuple(_stat(p) for p in files)


def cached_verdict(key: Hashable, repo_dir: Path, compute: Callable[[], bool]) -> bool:
    """compute()'s verdict for *repo_dir*, reused while nothing it reads has changed."""
    token = scan_token()
    if token is None:
        return compute()
    repo = os.path.normpath(os.fspath(repo_dir))
    key = (key, repo)
    now = time.time_ns()
    with _lock:
        entry = _entries.get(key)
        generation = _generation
    if entry is not None and entry.token is token and now - entry.stored_ns < _MAX_AGE_NS:
        try:
            if _signature(entry.dirs, entry.files) == entry.signature:
                with _lock:
                    if key in _entries:
                        _entries.move_to_end(key)
                return entry.verdict
        except _Unstable:
            return compute()
    try:
        walked = _walk_repo(repo)
        before = _signature(*walked) if walked is not None else None
    except _Unstable:
        walked = before = None
    verdict = compute()
    if walked is None or before is None or _newest(before) > now - _RACY_NS:
        return verdict
    if not _trusted_devices(_devices(before)):
        return verdict
    try:
        if _signature(*walked) != before:
            return verdict
    except _Unstable:
        return verdict
    with _lock:
        if generation != _generation:
            return verdict
        _entries[key] = _Entry(verdict, token, walked[0], walked[1], before, now)
        _entries.move_to_end(key)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last = False)
    return verdict

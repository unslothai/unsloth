# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional


EXIT_CANCELLED = 130

TRANSPORT_HTTP = "http"
TRANSPORT_XET = "xet"
VALID_TRANSPORTS = frozenset({TRANSPORT_HTTP, TRANSPORT_XET})
TRANSPORT_MARKER_NAME = ".transport"
INCOMPLETE_SUFFIX = ".incomplete"


def _safe_is_dir(path: Path) -> bool:
    """``Path.is_dir()`` returning False instead of raising when the path or a
    parent is unreadable (e.g. a restricted ``~/.cache/huggingface/hub``), so
    cache enumeration skips that root rather than 500ing."""
    try:
        return path.is_dir()
    except OSError:
        return False


def hf_cache_root(*, create: bool = False, root: Optional[Path] = None) -> Optional[Path]:
    from utils.hf_cache_settings import get_hf_cache_paths

    root = root or get_hf_cache_paths().hub_cache
    if create:
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError:
            return None
        return root
    return root if _safe_is_dir(root) else None


def hf_cache_roots() -> list[Path]:
    from hub.utils.paths import hf_default_cache_dir, legacy_hf_cache_dir
    from utils.hf_cache_settings import known_hf_hub_caches

    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not _safe_is_dir(path):
            return
        try:
            key = str(path.resolve())
        except OSError:
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
    blocks = getattr(st, "st_blocks", 0)
    if blocks > 0:
        return min(blocks * 512, st.st_size)
    if sys.platform == "win32":
        allocated = _windows_allocated_size(path)
        if allocated is not None:
            return min(allocated, st.st_size)
    return st.st_size


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


def latest_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    """Newest immediate child of ``repo_dir/snapshots`` by mtime, or None.

    mtime is the signal huggingface_hub's from_pretrained resolves to, so this
    points at whatever snapshot most recently landed on disk.
    """
    snapshots_dir = repo_dir / "snapshots"
    try:
        if not snapshots_dir.is_dir():
            return None
        snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshots:
            return None
        return max(snapshots, key = lambda entry: entry.stat().st_mtime)
    except OSError:
        return None


def _repo_dir_has_broken_snapshot_symlinks(repo_dir: Path) -> bool:
    latest = latest_snapshot_dir(repo_dir)
    if latest is None:
        return False
    try:
        for entry in latest.rglob("*"):
            if entry.is_symlink() and not entry.exists():
                return True
    except OSError:
        return False
    return False


def iter_repo_cache_dirs(repo_type: str, repo_id: str) -> Iterator[Path]:
    target = target_dir_name(repo_type, repo_id)
    for root in hf_cache_roots():
        try:
            for entry in root.iterdir():
                if entry.name.lower() == target:
                    yield entry
        except OSError:
            continue


def iter_destructive_repo_cache_dirs(
    repo_type: str, repo_id: str, *, root: Optional[Path] = None
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
) -> Iterator[Path]:
    root = hf_cache_root(root = root)
    if root is None:
        return
    target = target_dir_name(repo_type, repo_id)
    try:
        for entry in root.iterdir():
            if entry.name.lower() == target:
                yield entry
    except OSError:
        return


def preferred_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    force_active: bool = False,
    active_root: Optional[Path] = None,
) -> list[Path]:
    active_entries = list(iter_active_repo_cache_dirs(repo_type, repo_id, root = active_root))
    if active_entries:
        return active_entries
    if force_active:
        root = hf_cache_root(root = active_root)
        if root is not None:
            canonical = repo_cache_dir_name(repo_type, repo_id)
            return [root / canonical]
    return list(iter_repo_cache_dirs(repo_type, repo_id))


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


def purge_partial_repo(repo_type: str, repo_id: str, *, root: Optional[Path] = None) -> bool:
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


def purge_repo_cache_dirs(repo_type: str, repo_id: str, *, root: Optional[Path] = None) -> bool:
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


def scoped_delete_root(
    repo_type: str, repo_id: str, cache_path: Optional[str]
) -> Optional[Path]:
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
        (candidate for candidate in (resolved, *resolved.parents)
         if candidate.name.lower() == expected),
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

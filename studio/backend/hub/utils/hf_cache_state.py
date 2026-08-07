# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import shutil
import sys
from pathlib import Path, PureWindowsPath
from typing import Iterable, Iterator, Optional


EXIT_CANCELLED = 130

TRANSPORT_HTTP = "http"
TRANSPORT_XET = "xet"
VALID_TRANSPORTS = frozenset({TRANSPORT_HTTP, TRANSPORT_XET})
# A *request* preference, deliberately NOT in VALID_TRANSPORTS: "auto" is resolved to a real
# transport before anything is spawned, and the on-disk .transport marker must keep naming the
# writer that produced a partial, or a resume picks the wrong strategy.
TRANSPORT_AUTO = "auto"
VALID_TRANSPORT_MODES = frozenset({TRANSPORT_HTTP, TRANSPORT_XET, TRANSPORT_AUTO})
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


def same_existing_path(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except (OSError, ValueError):
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
            # required_groups is an AND of ORs: the snapshot must carry at least one file from every
            # group. That is what "loadable" means: metadata alone or weights alone is not enough.
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
        # Degrading to root-only is fail-closed at every caller, so nothing is wrongly accepted. But a
        # real cache permission or corruption fault then reaches the user as "your cached model isn't
        # cached" with no clue why, and four sites now share this handler.
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, Optional

from hub.utils.hf_cache_state import validated_repo_cache_path


_CACHE_VERSION = 1
_CACHE_DIRNAME = "snapshot-loads"
_METADATA_FILENAME = "metadata.json"


class UnsafeDatasetCachePathError(OSError):
    """A processed cache entry is not provably inside the trusted cache root.

    Kept distinct from a plain ``OSError`` so a caller doing best-effort bookkeeping can
    swallow "the write did not land" (purged entry, read-only home, full disk) without
    also swallowing "this path is no longer trusted". Subclasses ``OSError`` so existing
    ``except OSError`` callers keep catching it.
    """


@dataclass(frozen = True)
class AppProcessedDatasetCache:
    repo_id: str
    hub_cache: Path
    commit_hash: str
    path: Path
    cache_dir: Path
    complete: bool


def app_processed_dataset_cache_root() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / "hf-datasets" / _CACHE_DIRNAME


def _canonical_path(path: str | Path) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _hub_cache_key(path: Path) -> str:
    return _cache_key(os.path.normcase(str(path)))


def _repo_cache_key(repo_id: str) -> str:
    return _cache_key(repo_id.casefold())


def normalized_commit_hash(value: Any) -> Optional[str]:
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


def _safe_create_child(parent: Path, name: str, root: Path) -> Path:
    candidate = parent / name
    if candidate.is_symlink():
        raise OSError(f"Dataset cache path is a symlink: {candidate}")
    candidate.mkdir(exist_ok = True)
    resolved = candidate.resolve(strict = True)
    resolved.relative_to(root)
    return resolved


def _resolved_app_processed_dataset_cache_root(*, create: bool) -> Optional[Path]:
    from utils.paths.storage_roots import cache_root
    try:
        configured_root = Path(cache_root()).expanduser().absolute()
        root_path = app_processed_dataset_cache_root().expanduser().absolute()
        relative = root_path.relative_to(configured_root)
        if not relative.parts:
            return None
        if create:
            configured_root.mkdir(parents = True, exist_ok = True)
        trusted_root = configured_root.resolve(strict = True)
        if create:
            resolved = trusted_root
            for part in relative.parts:
                if part in {"", ".", ".."}:
                    return None
                resolved = _safe_create_child(resolved, part, trusted_root)
            return resolved
        if root_path.is_symlink() or not root_path.is_dir():
            return None
        resolved = root_path.resolve(strict = True)
        resolved.relative_to(trusted_root)
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _cache_root_is_present() -> bool:
    try:
        root_path = app_processed_dataset_cache_root().expanduser()
        return root_path.is_symlink() or root_path.exists()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _atomic_write_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("x", encoding = "utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok = True)
        except OSError:
            pass
        raise


def _metadata_payload(
    repo_id: str, hub_cache: Path, commit_hash: str, *, complete: bool
) -> dict[str, Any]:
    return {
        "version": _CACHE_VERSION,
        "repo_id": repo_id,
        "hub_cache": str(hub_cache),
        "commit_hash": commit_hash,
        "complete": complete,
    }


def prepare_app_processed_dataset_cache(repo_id: str, snapshot: Path) -> AppProcessedDatasetCache:
    validated = validated_repo_cache_path(str(snapshot), "dataset", repo_id)
    if validated is None:
        raise FileNotFoundError(f"Cached dataset snapshot for {repo_id} is unavailable")
    repo_dir, selected = validated
    try:
        snapshots = (repo_dir / "snapshots").resolve(strict = True)
        selected = selected.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FileNotFoundError(f"Cached dataset snapshot for {repo_id} is unavailable") from error
    if selected.parent != snapshots or not selected.is_dir():
        raise FileNotFoundError(f"Cached dataset snapshot for {repo_id} is unavailable")
    commit_hash = normalized_commit_hash(selected.name)
    if commit_hash is None:
        raise FileNotFoundError(f"Cached dataset snapshot for {repo_id} is unavailable")
    hub_cache = repo_dir.parent.resolve(strict = True)
    root = _resolved_app_processed_dataset_cache_root(create = True)
    if root is None:
        raise OSError("Dataset cache root is unavailable")
    hub_dir = _safe_create_child(root, _hub_cache_key(hub_cache), root)
    repo_path = _safe_create_child(hub_dir, _repo_cache_key(repo_id), root)
    entry_path = _safe_create_child(repo_path, commit_hash, root)
    metadata_path = entry_path / _METADATA_FILENAME
    existing = _read_cache_entry(entry_path, root)
    if existing is None:
        _atomic_write_metadata(
            metadata_path,
            _metadata_payload(
                repo_id,
                hub_cache,
                commit_hash,
                complete = False,
            ),
        )
    cache_dir = _safe_create_child(entry_path, "data", root)
    return AppProcessedDatasetCache(
        repo_id = repo_id,
        hub_cache = hub_cache,
        commit_hash = commit_hash,
        path = entry_path,
        cache_dir = cache_dir,
        complete = bool(existing and existing.complete),
    )


def mark_app_processed_dataset_cache_complete(entry: AppProcessedDatasetCache) -> None:
    root = _resolved_app_processed_dataset_cache_root(create = False)
    if root is None:
        # A root that is simply gone was purged out from under us -- benign. A root that
        # is still there but would not resolve inside the trusted root is a symlink or an
        # escape, so it must not be reported as a mere missing directory.
        if _cache_root_is_present():
            raise UnsafeDatasetCachePathError("Dataset cache root is not inside the trusted root")
        raise FileNotFoundError("Dataset cache root is unavailable")
    entry_path = entry.path.resolve(strict = True)
    try:
        entry_path.relative_to(root)
    except ValueError as error:
        raise UnsafeDatasetCachePathError(
            f"Dataset cache entry escapes the cache root: {entry.path}"
        ) from error
    if entry.path.is_symlink() or entry.cache_dir.is_symlink():
        raise UnsafeDatasetCachePathError(f"Dataset cache path is a symlink: {entry.path}")
    _atomic_write_metadata(
        entry_path / _METADATA_FILENAME,
        _metadata_payload(
            entry.repo_id,
            entry.hub_cache,
            entry.commit_hash,
            complete = True,
        ),
    )


def _read_cache_entry(entry_path: Path, root: Path) -> Optional[AppProcessedDatasetCache]:
    try:
        if entry_path.is_symlink() or not entry_path.is_dir():
            return None
        resolved = entry_path.resolve(strict = True)
        resolved.relative_to(root)
        metadata_path = resolved / _METADATA_FILENAME
        if metadata_path.is_symlink() or metadata_path.stat().st_size > 65536:
            return None
        payload = json.loads(metadata_path.read_text(encoding = "utf-8"))
    except (OSError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        return None
    repo_id = payload.get("repo_id")
    hub_cache = _canonical_path(payload.get("hub_cache"))
    commit_hash = normalized_commit_hash(payload.get("commit_hash"))
    from hub.utils.paths import is_valid_repo_id

    if (
        not isinstance(repo_id, str)
        or not is_valid_repo_id(repo_id)
        or hub_cache is None
        or commit_hash is None
        or resolved.name != commit_hash
        or resolved.parent.name != _repo_cache_key(repo_id)
        or resolved.parent.parent.name != _hub_cache_key(hub_cache)
    ):
        return None
    cache_dir = resolved / "data"
    try:
        if cache_dir.is_symlink() or not cache_dir.is_dir():
            return None
        cache_dir.resolve(strict = True).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return AppProcessedDatasetCache(
        repo_id = repo_id,
        hub_cache = hub_cache,
        commit_hash = commit_hash,
        path = resolved,
        cache_dir = cache_dir,
        complete = payload.get("complete") is True,
    )


def iter_app_processed_dataset_caches() -> Iterator[AppProcessedDatasetCache]:
    try:
        root = _resolved_app_processed_dataset_cache_root(create = False)
        if root is None:
            return
        hub_dirs = list(root.iterdir())
    except (OSError, RuntimeError):
        return
    for hub_dir in hub_dirs:
        try:
            if hub_dir.is_symlink() or not hub_dir.is_dir():
                continue
            repo_dirs = list(hub_dir.iterdir())
        except OSError:
            continue
        for repo_dir in repo_dirs:
            try:
                if repo_dir.is_symlink() or not repo_dir.is_dir():
                    continue
                entries = list(repo_dir.iterdir())
            except OSError:
                continue
            for entry_path in entries:
                entry = _read_cache_entry(entry_path, root)
                if entry is not None:
                    yield entry


def app_processed_dataset_cache_from_path(
    repo_id: str, path_value: str
) -> Optional[AppProcessedDatasetCache]:
    requested = _canonical_path(path_value)
    if requested is None:
        return None
    for entry in iter_app_processed_dataset_caches():
        if entry.repo_id.casefold() == repo_id.casefold() and requested in {
            entry.path,
            entry.cache_dir,
        }:
            return entry
    return None


def delete_app_processed_dataset_caches(
    repo_id: str, *, hub_cache: Optional[Path] = None
) -> tuple[bool, list[str]]:
    from hub.utils.paths import is_valid_repo_id

    if not is_valid_repo_id(repo_id):
        return False, []
    try:
        root = _resolved_app_processed_dataset_cache_root(create = False)
        if root is None:
            return False, []
        if hub_cache is not None:
            canonical_hub = hub_cache.expanduser().resolve(strict = False)
            hub_dirs = [root / _hub_cache_key(canonical_hub)]
        else:
            hub_dirs = list(root.iterdir())
    except (OSError, RuntimeError, ValueError):
        return False, []

    deleted = False
    failures: list[str] = []
    repo_key = _repo_cache_key(repo_id)
    for hub_dir in hub_dirs:
        try:
            if hub_dir.is_symlink() or not hub_dir.is_dir():
                continue
            resolved_hub = hub_dir.resolve(strict = True)
            resolved_hub.relative_to(root)
            target = resolved_hub / repo_key
            if not target.exists():
                continue
            if target.is_symlink() or not target.is_dir():
                failures.append(f"Unsafe processed dataset cache path: {target}")
                continue
            resolved_target = target.resolve(strict = True)
            resolved_target.relative_to(root)
            if any(child.is_symlink() for child in resolved_target.iterdir()):
                failures.append(f"Unsafe processed dataset cache entry under: {resolved_target}")
                continue
            shutil.rmtree(resolved_target)
            deleted = True
            try:
                resolved_hub.rmdir()
            except OSError:
                pass
        except Exception as error:
            failures.append(str(error))
    return deleted, failures

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from hub.utils.hf_cache_state import (
    iter_repo_cache_dirs,
    ref_snapshot_dir,
    validated_repo_cache_path,
)


TRAINING_DATA_EXTS = (".parquet", ".json", ".jsonl", ".csv")

_RESERVED_SPLIT_TOKENS = frozenset({"train", "test", "validation", "valid", "val", "eval"})


def hf_datasets_cache_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            resolved = path.expanduser().resolve(strict = True)
        except (OSError, RuntimeError, ValueError):
            return
        if not resolved.is_dir() or resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        add(Path(env_cache))

    try:
        from datasets import config as datasets_config

        add(Path(datasets_config.HF_DATASETS_CACHE))
    except Exception:
        pass

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "datasets")

    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    add(xdg_cache / "huggingface" / "datasets")
    return roots


def _rel_lower(snapshot: Path, path: Path) -> str:
    return path.relative_to(snapshot).as_posix().lower()


_SPLIT_ALIASES = {
    "validation": frozenset({"validation", "valid", "val"}),
    "valid": frozenset({"validation", "valid", "val"}),
    "val": frozenset({"validation", "valid", "val"}),
    "eval": frozenset({"eval", "validation", "valid", "val"}),
}


def _label_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}


def split_label_matches(text: str, split: str) -> bool:
    """Match a split name against a file path's tokens, expanding split aliases
    (validation/valid/val, eval) so cached and remote selection agree."""
    normalized = split.strip().lower()
    if not normalized:
        return False
    labels = _SPLIT_ALIASES.get(normalized, frozenset({normalized}))
    return bool(labels.intersection(_label_tokens(text)))


def _matches_label(snapshot: Path, path: Path, label: str) -> bool:
    label = label.strip().lower()
    if not label:
        return False
    rel = _rel_lower(snapshot, path)
    tokens = [token for token in re.split(r"[^a-z0-9]+", rel) if token]
    if label in tokens:
        return True
    if label in _RESERVED_SPLIT_TOKENS:
        return False
    return label in rel


def dataset_snapshot_from_cache_path(local_path: Optional[str], repo_id: str) -> Optional[Path]:
    validated = validated_repo_cache_path(local_path, "dataset", repo_id)
    if validated is None:
        return None
    repo_dir, selected = validated
    try:
        snapshots = (repo_dir / "snapshots").resolve(strict = True)
        if snapshots.parent != repo_dir or not snapshots.is_dir():
            return None
        if selected != repo_dir:
            return selected if selected.parent == snapshots and selected.is_dir() else None
        pinned = ref_snapshot_dir(repo_dir)
        if pinned is not None:
            return pinned
        candidates: list[Path] = []
        for path in snapshots.iterdir():
            try:
                candidate = path.resolve(strict = True)
            except (OSError, RuntimeError):
                continue
            if candidate.parent == snapshots and candidate.is_dir():
                candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(
            key = lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse = True,
        )
        return candidates[0].resolve()
    except Exception:
        return None


def processed_dataset_cache_path(local_path: Optional[str], repo_id: str) -> Optional[Path]:
    if not local_path or not repo_id:
        return None
    try:
        resolved = Path(local_path).expanduser().resolve(strict = True)
        expected = repo_id.replace("/", "___").lower()
        if (
            resolved.name.lower() != expected
            or resolved.parent not in set(hf_datasets_cache_roots())
            or not resolved.is_dir()
        ):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def latest_processed_dataset_cache_path(repo_id: str) -> Optional[Path]:
    if not repo_id:
        return None
    expected = repo_id.replace("/", "___")
    for root in hf_datasets_cache_roots():
        direct = processed_dataset_cache_path(str(root / expected), repo_id)
        if direct is not None:
            return direct
        try:
            matches = [
                entry
                for entry in root.iterdir()
                if entry.name.lower() == expected.lower()
            ]
        except OSError:
            continue
        if len(matches) != 1:
            continue
        matched = processed_dataset_cache_path(str(matches[0]), repo_id)
        if matched is not None:
            return matched
    return None


def latest_cached_dataset_snapshot(
    repo_id: str, local_path: Optional[str] = None
) -> Optional[Path]:
    local_snapshot = dataset_snapshot_from_cache_path(local_path, repo_id)
    if local_snapshot is not None:
        return local_snapshot

    newest: Optional[Path] = None
    newest_mtime = -1.0
    for entry in iter_repo_cache_dirs("dataset", repo_id):
        validated = validated_repo_cache_path(str(entry), "dataset", repo_id)
        if validated is None:
            continue
        repo_dir, _ = validated
        pinned = ref_snapshot_dir(repo_dir)
        if pinned is not None:
            return pinned.resolve()
        candidate = dataset_snapshot_from_cache_path(str(repo_dir), repo_id)
        if candidate is None:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest = candidate
            newest_mtime = mtime
    return newest


def latest_cached_dataset_path(
    repo_id: str, local_path: Optional[str] = None
) -> Optional[Path]:
    selected = dataset_cache_path_from_cache_path(local_path, repo_id)
    if selected is not None:
        return selected
    processed = latest_processed_dataset_cache_path(repo_id)
    if processed is not None:
        return processed
    return latest_cached_dataset_snapshot(repo_id, local_path)


def dataset_cache_path_from_cache_path(
    local_path: Optional[str], repo_id: str
) -> Optional[Path]:
    processed = processed_dataset_cache_path(local_path, repo_id)
    return processed or dataset_snapshot_from_cache_path(local_path, repo_id)


def is_cache_artifact_error(error: BaseException | None) -> bool:
    retryable_errno = {
        errno.EACCES,
        errno.EIO,
        errno.EISDIR,
        errno.ENOENT,
        errno.ENOTDIR,
        errno.EPERM,
        *(
            value
            for value in (getattr(errno, "EBADMSG", None), getattr(errno, "ESTALE", None))
            if value is not None
        ),
    }
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                NotADirectoryError,
                EOFError,
                json.JSONDecodeError,
            ),
        ):
            return True
        if isinstance(current, OSError) and current.errno in retryable_errno:
            return True
        if type(current).__name__ in {
            "ArrowIOError",
            "DataFilesNotFoundError",
            "DatasetNotFoundError",
            "LocalEntryNotFoundError",
            "SafetensorError",
        }:
            return True
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "can't load the model for",
                "can't load tokenizer for",
                "cached path",
                "cached snapshot",
                "does not appear to have a file named",
                "failed finding central directory",
                "invalid header length",
                "invalid load key",
                "invalid parquet",
                "metadata incomplete buffer",
                "no such file or directory",
                "not found in the cached files",
                "offline mode is enabled",
                "outgoing traffic has been disabled",
                "parquet magic bytes",
                "pickle data was truncated",
                "pytorchstreamreader failed",
                "safetensor header",
                "safetensors header",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_cached_hf_dataset(
    repo_id: str,
    local_path: Optional[str],
    *,
    subset: Optional[str],
    split: str,
    token: Optional[str] = None,
) -> Any:
    processed = processed_dataset_cache_path(local_path, repo_id)
    snapshot = None if processed is not None else dataset_snapshot_from_cache_path(
        local_path, repo_id
    )
    if processed is None and snapshot is None:
        raise FileNotFoundError(f"Cached dataset path for {repo_id} is unavailable")

    from datasets import DownloadConfig
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    kwargs: dict[str, Any] = {
        "path": repo_id if processed is not None else str(snapshot),
        "split": split,
        "download_config": DownloadConfig(local_files_only = True),
    }
    if processed is not None:
        kwargs["cache_dir"] = str(processed.parent)
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def cached_dataset_candidates(
    snapshot: Path,
    *,
    subset: Optional[str],
    train_split: str,
    extensions: tuple[str, ...],
    preferred_extensions: tuple[str, ...] = TRAINING_DATA_EXTS,
) -> list[Path]:
    try:
        files = [
            p for p in snapshot.rglob("*") if p.is_file() and p.name.lower().endswith(extensions)
        ]
    except OSError:
        return []
    if not files:
        return []

    subset_lower = subset.lower() if subset else ""
    split_lower = train_split.lower()

    def score(path: Path) -> tuple[int, int, str]:
        rel = _rel_lower(snapshot, path)
        subset_match = bool(subset_lower and _matches_label(snapshot, path, subset_lower))
        split_match = bool(split_lower and split_label_matches(rel, split_lower))
        location_rank = 3
        if split_match and (not subset_lower or subset_match):
            location_rank = 0
        elif split_match:
            location_rank = 1
        elif subset_match:
            location_rank = 2
        return (
            0 if path.name.lower().endswith(preferred_extensions) else 1,
            location_rank,
            rel,
        )

    return sorted(files, key = score)

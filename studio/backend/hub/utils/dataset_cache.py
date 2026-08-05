# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from hub.utils.dataset_processed_cache import (
    mark_app_processed_dataset_cache_complete,
    normalized_commit_hash,
    prepare_app_processed_dataset_cache,
)
from hub.utils.hf_cache_state import (
    iter_repo_cache_dirs,
    ref_snapshot_dir,
    same_existing_path,
    validated_repo_cache_path,
)


TRAINING_DATA_EXTS = (".parquet", ".json", ".jsonl", ".csv")

_RESERVED_SPLIT_TOKENS = frozenset({"train", "test", "validation", "valid", "val", "eval"})
_BARE_SPLIT_RE = re.compile(r"^\w+(?:\.\w+)*$")
_UNKNOWN_SPLIT_ERROR_RE = re.compile(
    r'^Unknown split "[^"\r\n]+"\. Should be one of \[[^\]\r\n]*\]\.$'
)


def _canonical_path(path: Any) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


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
        if not same_existing_path(snapshots.parent, repo_dir) or not snapshots.is_dir():
            return None
        if not same_existing_path(selected, repo_dir):
            return (
                selected
                if same_existing_path(selected.parent, snapshots) and selected.is_dir()
                else None
            )
        pinned = ref_snapshot_dir(repo_dir)
        if pinned is not None:
            return pinned
        candidates: list[Path] = []
        for path in snapshots.iterdir():
            try:
                candidate = path.resolve(strict = True)
            except (OSError, RuntimeError):
                continue
            if same_existing_path(candidate.parent, snapshots) and candidate.is_dir():
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
            or not any(
                same_existing_path(resolved.parent, root) for root in hf_datasets_cache_roots()
            )
            or not resolved.is_dir()
        ):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def processed_dataset_cache_has_artifacts(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False

    for directory, dirnames, filenames in os.walk(path, followlinks = False):
        base = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.endswith(".incomplete") and not (base / name).is_symlink()
        ]
        if "dataset_info.json" not in filenames:
            continue
        info_path = base / "dataset_info.json"
        try:
            if info_path.is_symlink() or not info_path.is_file():
                continue
            with info_path.open("r", encoding = "utf-8") as stream:
                if not isinstance(json.load(stream), dict):
                    continue
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for filename in filenames:
            entry = base / filename
            if entry.suffix.lower() != ".arrow":
                continue
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                with entry.open("rb") as stream:
                    if stream.read(1):
                        return True
            except OSError:
                continue
    return False


def latest_processed_dataset_cache_path(repo_id: str) -> Optional[Path]:
    if not repo_id:
        return None
    expected = repo_id.replace("/", "___")
    for root in hf_datasets_cache_roots():
        direct = processed_dataset_cache_path(str(root / expected), repo_id)
        if direct is not None and processed_dataset_cache_has_artifacts(direct):
            return direct
        try:
            matches = [entry for entry in root.iterdir() if entry.name.lower() == expected.lower()]
        except OSError:
            continue
        if len(matches) != 1:
            continue
        matched = processed_dataset_cache_path(str(matches[0]), repo_id)
        if matched is not None and processed_dataset_cache_has_artifacts(matched):
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


def latest_cached_dataset_path(repo_id: str, local_path: Optional[str] = None) -> Optional[Path]:
    selected = dataset_cache_path_from_cache_path(local_path, repo_id)
    if selected is not None:
        return selected
    processed = latest_processed_dataset_cache_path(repo_id)
    if processed is not None:
        return processed
    return latest_cached_dataset_snapshot(repo_id, local_path)


def resolved_dataset_snapshot_file(snapshot: str | Path, source_path: str) -> Optional[Path]:
    from hub.utils.download_manifest import expected_path_is_safe

    if not expected_path_is_safe(source_path):
        return None
    try:
        snapshot_path = Path(snapshot).resolve(strict = True)
        repo_dir = snapshot_path.parent.parent.resolve(strict = True)
        if not same_existing_path(snapshot_path.parent, repo_dir / "snapshots"):
            return None
        resolved = snapshot_path.joinpath(*PurePosixPath(source_path).parts).resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file() or not (
        resolved.is_relative_to(snapshot_path) or resolved.is_relative_to(repo_dir / "blobs")
    ):
        return None
    try:
        with resolved.open("rb"):
            pass
    except OSError:
        return None
    return resolved


def dataset_snapshot_contains_file(snapshot: str | Path, source_path: str) -> bool:
    return resolved_dataset_snapshot_file(snapshot, source_path) is not None


def complete_dataset_snapshot_path(local_path: Optional[str], repo_id: str) -> Optional[Path]:
    snapshot = dataset_snapshot_from_cache_path(local_path, repo_id)
    if snapshot is None:
        return None
    validated = validated_repo_cache_path(str(snapshot), "dataset", repo_id)
    if validated is None:
        return None
    repo_dir, selected = validated
    try:
        snapshot = snapshot.resolve(strict = True)
        selected = selected.resolve(strict = True)
        repo_dir = repo_dir.resolve(strict = True)
        hub_cache = repo_dir.parent.resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not same_existing_path(snapshot, selected) or not same_existing_path(
        snapshot.parent, repo_dir / "snapshots"
    ):
        return None

    from hub.utils import download_manifest

    manifest = download_manifest.read_dataset_completion(
        repo_id,
        snapshot.name,
        hub_cache = hub_cache,
    )
    manifest_hub_cache = _canonical_path(manifest.hub_cache) if manifest is not None else None
    if (
        manifest is None
        or manifest.repo_type != "dataset"
        or manifest.repo_id.casefold() != repo_id.casefold()
        or manifest.version != 2
        or not manifest.metadata_derived
        or manifest.commit_hash != snapshot.name
        or manifest_hub_cache is None
        or not same_existing_path(manifest_hub_cache, hub_cache)
        or not manifest.expected_files
    ):
        return None
    for expected in manifest.expected_files:
        if not dataset_snapshot_contains_file(snapshot, expected.path):
            return None
    if not download_manifest.verify_against_disk(manifest, snapshot).ok:
        return None
    return snapshot


def training_dataset_cache_pin(
    repo_id: str, local_path: Optional[str] = None
) -> tuple[Optional[Path], Optional[str]]:
    if local_path:
        selected = dataset_cache_path_from_cache_path(local_path, repo_id)
    else:
        selected = latest_cached_dataset_path(repo_id)
    if selected is None:
        return None, None
    processed = processed_dataset_cache_path(str(selected), repo_id)
    if processed is not None:
        return processed, None
    snapshot = dataset_snapshot_from_cache_path(str(selected), repo_id)
    if snapshot is None:
        return None, None
    commit_hash = normalized_commit_hash(snapshot.name)
    if commit_hash is None:
        return None, None
    return snapshot, commit_hash


def dataset_cache_path_from_cache_path(local_path: Optional[str], repo_id: str) -> Optional[Path]:
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
                "either model_file or model_proto must be specified",
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


def _is_unknown_dataset_split_error(error: BaseException | None) -> bool:
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValueError) and _UNKNOWN_SPLIT_ERROR_RE.fullmatch(
            str(current).strip()
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def dataset_cache_fallback_allowed(
    error: BaseException | None, *, require_exact: bool, revision: Optional[str]
) -> bool:
    if require_exact:
        return False
    offline = any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )
    if revision and offline:
        return False
    return is_cache_artifact_error(error) or _is_unknown_dataset_split_error(error)


def load_cached_hf_dataset(
    repo_id: str,
    local_path: Optional[str],
    *,
    subset: Optional[str],
    split: str,
    token: Optional[str] = None,
    row_limit: Optional[int] = None,
) -> Any:
    if row_limit is not None and (
        isinstance(row_limit, bool) or not isinstance(row_limit, int) or row_limit <= 0
    ):
        raise ValueError("row_limit must be a positive integer")
    processed = processed_dataset_cache_path(local_path, repo_id)
    snapshot = (
        None if processed is not None else dataset_snapshot_from_cache_path(local_path, repo_id)
    )
    if processed is None and snapshot is None:
        raise FileNotFoundError(f"Cached dataset path for {repo_id} is unavailable")

    from datasets import DownloadConfig

    if snapshot is not None:
        from datasets import load_dataset
    else:
        from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    stream_limited_snapshot = (
        snapshot is not None
        and row_limit is not None
        and _BARE_SPLIT_RE.fullmatch(split) is not None
    )
    app_cache = (
        prepare_app_processed_dataset_cache(repo_id, snapshot)
        if snapshot is not None and not stream_limited_snapshot
        else None
    )
    kwargs: dict[str, Any] = {
        "path": repo_id if processed is not None else str(snapshot),
        "split": split,
        "download_config": DownloadConfig(local_files_only = True),
    }
    if processed is not None:
        kwargs["cache_dir"] = str(processed.parent)
    elif app_cache is not None:
        kwargs["cache_dir"] = str(app_cache.cache_dir)
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token
    if stream_limited_snapshot:
        kwargs["streaming"] = True
        with tempfile.TemporaryDirectory(prefix = "unsloth-dataset-slice-") as cache_dir:
            kwargs["cache_dir"] = cache_dir
            requested_split = kwargs.pop("split")
            streams = load_dataset(**kwargs)
            available_splits = list(streams)
            if requested_split not in streams:
                raise ValueError(
                    f'Unknown split "{requested_split}". Should be one of {available_splits}.'
                )
            stream = streams[requested_split]
            features = getattr(stream, "features", None)
            info = getattr(stream, "info", None)
            if info is not None:
                info = info.copy()
                if not info.splits:
                    from datasets import SplitDict, SplitInfo
                    info.splits = SplitDict(
                        {name: SplitInfo(name = name) for name in available_splits}
                    )
            split_identity = getattr(stream, "split", None)
            rows = list(stream.take(row_limit))
            del stream, streams
        from datasets import Dataset

        schema = features or getattr(info, "features", None)
        if not rows and schema is not None:
            return Dataset.from_dict(
                {name: [] for name in schema},
                features = features,
                info = info,
                split = split_identity,
            )
        return Dataset.from_list(
            rows,
            features = features,
            info = info,
            split = split_identity,
        )
    dataset = load_dataset(**kwargs)
    if app_cache is not None:
        mark_app_processed_dataset_cache_complete(app_cache)
    return dataset


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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import re
from pathlib import Path
from typing import Optional

from hub.utils.hf_cache_state import iter_repo_cache_dirs


TRAINING_DATA_EXTS = (".parquet", ".json", ".jsonl", ".csv")


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
    if label in {"train", "test", "validation", "valid", "val", "eval"}:
        return False
    return label in rel


def dataset_snapshot_from_cache_path(local_path: Optional[str], repo_id: str) -> Optional[Path]:
    if not local_path or not repo_id:
        return None
    try:
        root = Path(local_path).expanduser()
        if not root.exists():
            return None
        expected_repo_dir = f"datasets--{repo_id.replace('/', '--')}".lower()
        if expected_repo_dir not in {part.lower() for part in root.parts}:
            return None
        if root.is_dir() and root.parent.name == "snapshots":
            return root.resolve()
        snapshots = root / "snapshots" if root.is_dir() else None
        if snapshots is None or not snapshots.is_dir():
            return None
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
        if not candidates:
            return None
        candidates.sort(
            key = lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse = True,
        )
        return candidates[0].resolve()
    except Exception:
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
        snapshots = entry / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            candidates = [s for s in snapshots.iterdir() if s.is_dir()]
        except OSError:
            continue
        for snap in candidates:
            try:
                mtime = snap.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest = snap
                newest_mtime = mtime
    return newest


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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable


SNAPSHOT_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.gguf",
    "*.onnx",
    "onnx/*",
    "openvino/*",
    "mlx/*",
    "*.bin.index.json.bak",
)
CONSOLIDATED_PATTERN = "consolidated*"
SNAPSHOT_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".npz",
)


def _filename(sibling) -> str:
    value = getattr(sibling, "rfilename", "")
    return value if isinstance(value, str) else ""


def _size(sibling) -> int:
    value = getattr(sibling, "size", None)
    return int(value) if isinstance(value, int) and value > 0 else 0


def repo_ships_transformers_weights(filenames: Iterable[str]) -> bool:
    for name in filenames:
        base = name.rsplit("/", 1)[-1].lower()
        if base.startswith("consolidated"):
            continue
        if base.endswith(".safetensors"):
            return True
        if base.endswith(".bin") and (
            base.startswith("model") or base.startswith("pytorch_model")
        ):
            return True
    return False


def resolve_snapshot_ignore_patterns_for_files(filenames: Iterable[str]) -> list[str]:
    names = list(filenames)
    ignore = list(SNAPSHOT_IGNORE_PATTERNS)
    if repo_ships_transformers_weights(names):
        ignore.append(CONSOLIDATED_PATTERN)
    return ignore


def sibling_matches_ignore(filename: str, ignore_patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(filename, pattern) for pattern in ignore_patterns)


def snapshot_download_siblings(siblings: Iterable) -> list:
    items = list(siblings)
    ignore_patterns = resolve_snapshot_ignore_patterns_for_files(
        _filename(sibling) for sibling in items
    )
    return [
        sibling
        for sibling in items
        if not sibling_matches_ignore(_filename(sibling), ignore_patterns)
    ]


def snapshot_download_size(siblings: Iterable) -> int:
    return sum(_size(sibling) for sibling in snapshot_download_siblings(siblings))


def snapshot_weight_size(siblings: Iterable) -> int:
    total = 0
    for sibling in snapshot_download_siblings(siblings):
        filename = _filename(sibling).lower()
        if filename.endswith(SNAPSHOT_WEIGHT_EXTENSIONS):
            total += _size(sibling)
    return total


def snapshot_download_lfs_hashes(siblings: Iterable) -> frozenset[str]:
    hashes: set[str] = set()
    for sibling in snapshot_download_siblings(siblings):
        sha = getattr(getattr(sibling, "lfs", None), "sha256", None)
        if isinstance(sha, str) and sha:
            hashes.add(sha)
    return frozenset(hashes)

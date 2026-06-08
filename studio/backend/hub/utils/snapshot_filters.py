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
SNAPSHOT_NON_BIN_WEIGHT_EXTENSIONS = tuple(
    ext for ext in SNAPSHOT_WEIGHT_EXTENSIONS if ext != ".bin"
)
SNAPSHOT_BIN_WEIGHT_PREFIXES = ("model", "pytorch_model", "adapter_model")


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
        if base.endswith(SNAPSHOT_NON_BIN_WEIGHT_EXTENSIONS):
            return True
        if base.endswith(".bin") and base.startswith(SNAPSHOT_BIN_WEIGHT_PREFIXES):
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


def total_size_for_siblings(siblings: Iterable) -> int:
    """Sum of declared sizes across siblings verbatim (no ignore filter).

    Use for repo types that download every file (datasets); models go
    through ``snapshot_download_size`` so the ignore patterns apply."""
    return sum(_size(sibling) for sibling in siblings)


def blob_hashes_for_siblings(siblings: Iterable) -> frozenset[str]:
    # Blob filename == file etag (LFS sha256, else git blob id). Collecting both
    # lets progress count exactly this revision's files without summing stale
    # blobs from other revisions.
    hashes: set[str] = set()
    for sibling in siblings:
        sha = getattr(getattr(sibling, "lfs", None), "sha256", None)
        if isinstance(sha, str) and sha:
            hashes.add(sha)
            continue
        blob_id = getattr(sibling, "blob_id", None)
        if isinstance(blob_id, str) and blob_id:
            hashes.add(blob_id)
    return frozenset(hashes)


def snapshot_download_blob_hashes(siblings: Iterable) -> frozenset[str]:
    return blob_hashes_for_siblings(snapshot_download_siblings(siblings))

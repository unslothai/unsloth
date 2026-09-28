# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Iterable
from utils.paths.path_utils import drop_shadowed_appledouble_names


SNAPSHOT_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.gguf",
    "*.onnx",
    "onnx/*",
    "openvino/*",
    "mlx/*",
    "*.bin.index.json.bak",
)
CONSOLIDATED_PATTERN = "consolidated*"
DUPLICATE_WEIGHT_FORMAT_PATTERNS: tuple[str, ...] = (
    "original/*",
    "metal/*",
    "coreml/*",
    "tf_model*.h5",
    "tf_model.h5.index.json",
    "flax_model*.msgpack",
    "flax_model.msgpack.index.json",
    "rust_model.ot",
)
# Resolved per repo, not globbed: a dtype variant is redundant only when the SAME variant ships
# as safetensors (test_prefetch_snapshot_scope.py::test_variant_keeps_bin_when_only_default_safetensors),
# and a glob kept whisper-large-v3's pytorch_model.bin.index.fp32.json while dropping its shards.
# Both shard layouts, since transformers writes the counter-first one (unsloth/models/_utils.py).
_BIN_WEIGHT_RES = (
    re.compile(r"pytorch_model(?:\.(?P<variant>[A-Za-z0-9_]+))?(?:[-_][0-9]+-of-[0-9]+)?\.bin"),
    re.compile(r"pytorch_model[-_][0-9]+-of-[0-9]+\.(?P<variant>[A-Za-z0-9_]+)\.bin"),
)


def _variant_shard_re(variant: str) -> "re.Pattern[str]":
    v = re.escape(variant)
    return re.compile(
        rf"model\.{v}[-_][0-9]+-of-[0-9]+\.safetensors|model[-_][0-9]+-of-[0-9]+\.{v}\.safetensors"
    )


_BIN_INDEX_RE = re.compile(
    r"pytorch_model(?:\.(?P<pre>[A-Za-z0-9_]+))?\.bin\.index(?:\.(?P<post>[A-Za-z0-9_]+))?\.json"
)
# [0-9] rather than \d: Python's \d also matches non-ASCII digits while JavaScript's does not, and the
# frontend mirror in studio/frontend/src/features/hub/lib/dataset-size.ts has to answer this identically.
SHARDED_SAFETENSORS_RE = re.compile(r"model[-_][0-9]+-of-[0-9]+\.safetensors")
SAFETENSORS_INDEX = "model.safetensors.index.json"
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
    # A "._consolidated.safetensors" does not start with "consolidated", so it would answer yes here and
    # then have every consolidated* file stripped from the download.
    for name in drop_shadowed_appledouble_names(list(filenames)):
        base = name.rsplit("/", 1)[-1].lower()
        if base.startswith("consolidated"):
            continue
        if base.endswith(SNAPSHOT_NON_BIN_WEIGHT_EXTENSIONS):
            return True
        if base.endswith(".bin") and base.startswith(SNAPSHOT_BIN_WEIGHT_PREFIXES):
            return True
    return False


def repo_ships_root_safetensors(filenames: Iterable[str]) -> bool:
    """Whether a load would find a COMPLETE root safetensors checkpoint.

    Numbered shards are resolved through the index; with none, a load looks for a single
    file and raises. So one shard is not evidence a checkpoint is there.
    """
    names = list(filenames)
    if any(name == "model.safetensors" for name in names):
        return True
    return SAFETENSORS_INDEX in names and any(
        SHARDED_SAFETENSORS_RE.fullmatch(name) for name in names
    )


def _variant_ships_as_safetensors(names: list[str], variant: str | None) -> bool:
    """Whether `variant` is already covered by a COMPLETE safetensors checkpoint.

    ``None`` is the canonical one, established by the root-safetensors gate. A named variant
    needs its own safetensors, and its sharded form needs an index, as the canonical one does.
    """
    if variant is None:
        return True
    if f"model.{variant}.safetensors" in names:
        return True
    # _add_variant puts the variant second-to-last, so this is the only index spelling a
    # variant load can find; accepting another would drop bins over an index nothing reads.
    if f"model.safetensors.index.{variant}.json" not in names:
        return False
    return any(_variant_shard_re(variant).fullmatch(name) for name in names)


def redundant_torch_bin_files(filenames: Iterable[str]) -> list[str]:
    """The pytorch_model .bin files, and their indexes, a safetensors copy makes redundant."""
    names = list(filenames)
    redundant = []
    for name in names:
        match = next((m for m in (rx.fullmatch(name) for rx in _BIN_WEIGHT_RES) if m), None)
        variant = match.group("variant") if match else None
        if not match:
            match = _BIN_INDEX_RE.fullmatch(name)
            if not match:
                continue
            variant = match.group("pre") or match.group("post")
        if _variant_ships_as_safetensors(names, variant):
            redundant.append(name)
    return redundant


def resolve_snapshot_ignore_patterns_for_files(filenames: Iterable[str]) -> list[str]:
    names = list(filenames)
    ignore = list(SNAPSHOT_IGNORE_PATTERNS)
    if repo_ships_transformers_weights(names):
        ignore.append(CONSOLIDATED_PATTERN)
    if repo_ships_root_safetensors(names):
        ignore.extend(DUPLICATE_WEIGHT_FORMAT_PATTERNS)
        # Exact names, not globs: fnmatch treats "." literally, so each entry matches only itself.
        ignore.extend(redundant_torch_bin_files(names))
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
    # Blob filename == file etag (LFS sha256, else git blob id), so collecting both lets progress count
    # exactly this revision's files without summing stale blobs from other revisions.
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


# mlx-lm's download filter, plus the weights of an exported adapter.
_MLX_LOAD_PATTERNS = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
    "adapters.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def mlx_load_siblings(siblings):
    return [
        sibling
        for sibling in siblings
        if any(fnmatchcase(sibling.rfilename, pattern) for pattern in _MLX_LOAD_PATTERNS)
    ]

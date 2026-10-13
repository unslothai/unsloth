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
# No q<N>: GGUF quant names (Q4_K_M) must never match. [0-9], not \d: mirrored in dataset-size.ts.
PRECISION_FOLDER_SOURCE = (
    "(?:mlx[-_]?)?(?:[0-9]{1,2}[-_]?bits?(?:[-_][a-z0-9]+)*"
    "|bf16|fp16|f16|fp32|f32|fp8|mxfp4|nvfp4)"
)
PRECISION_FOLDER_RE = re.compile(PRECISION_FOLDER_SOURCE, re.IGNORECASE)
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
# Per repo, not globbed: a dtype variant is redundant only if the SAME variant ships as safetensors.
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
# [0-9], not \d: must match the JS mirror in features/hub/lib/dataset-size.ts.
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
    """One numbered shard proves nothing: a complete set must be named by the index."""
    names = list(filenames)
    if any(name == "model.safetensors" for name in names):
        return True
    return SAFETENSORS_INDEX in names and any(
        SHARDED_SAFETENSORS_RE.fullmatch(name) for name in names
    )


def _variant_ships_as_safetensors(names: list[str], variant: str | None) -> bool:
    """A named variant needs its own safetensors, plus an index when sharded; None is the root
    checkpoint."""
    if variant is None:
        return True
    if f"model.{variant}.safetensors" in names:
        return True
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


def precision_variant_subfolders(filenames: Iterable[str]) -> dict[str, list[str]]:
    """Top-level folders that each hold a whole checkpoint at one precision (``4-bit/``, ``8bit/``,
    ``mlx-4bit/``, ``bf16/``), mapped to their files. A folder needs a precision-like name, its own
    ``config.json`` and safetensors weights, so component folders (``vae/``, ``mtp/``) never count."""
    by_folder: dict[str, list[str]] = {}
    for name in filenames:
        folder, sep, rest = name.partition("/")
        if sep and rest and "/" not in rest and PRECISION_FOLDER_RE.fullmatch(folder):
            by_folder.setdefault(folder, []).append(name)
    return {
        folder: names
        for folder, names in by_folder.items()
        if f"{folder}/config.json" in names
        and any(name.lower().endswith(".safetensors") for name in names)
    }


def resolve_snapshot_ignore_patterns_for_files(filenames: Iterable[str]) -> list[str]:
    names = list(filenames)
    ignore = list(SNAPSHOT_IGNORE_PATTERNS)
    if repo_ships_transformers_weights(names):
        ignore.append(CONSOLIDATED_PATTERN)
    if repo_ships_root_safetensors(names):
        ignore.extend(DUPLICATE_WEIGHT_FORMAT_PATTERNS)
        ignore.extend(redundant_torch_bin_files(names))
        # Only when the root is itself a checkpoint, or the folders are the only runnable copies.
        if "config.json" in names:
            ignore.extend(f"{folder}/*" for folder in precision_variant_subfolders(names))
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
    """Sums sibling sizes verbatim, with no ignore filter; models use snapshot_download_size instead."""
    return sum(_size(sibling) for sibling in siblings)


def blob_hashes_for_siblings(siblings: Iterable) -> frozenset[str]:
    # Blob filename == etag, so progress counts only this revision's files.
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

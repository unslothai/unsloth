# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standalone Engram tables a GGUF repo publishes beside its quants, and the files llama.cpp
reads when one replaces the table a quant ships with.

unsloth/Qwen3.8-Flash-Next-GGUF and unsloth/DeepSeek-V4.1-Flash-1017-GGUF keep every quant's
``per_layer_token_embd.weight`` in its own split shard and publish the same table at other
precisions as ``engram/<model>-engram-<PRECISION>.gguf_file``. llama.cpp b12233+ loads one with
``--ple`` (ggml-org/llama.cpp#31185): it replaces the quant's table, and the quant's table shard
may then be absent, so the swap needs no renaming or ``split.no`` stamping.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENGRAM_FOLDER = "engram"

_TABLE_RE = re.compile(
    r"^(?:.*/)?engram/[^/]+-engram-(?P<precision>[A-Za-z0-9][A-Za-z0-9_]*)\.gguf(?:_file)?$",
    re.IGNORECASE,
)


@dataclass(frozen = True)
class EngramTable:
    precision: str
    filename: str
    size_bytes: int


def engram_table_precision(filename: str) -> Optional[str]:
    """The precision label of a standalone Engram table file, or None for any other file."""
    match = _TABLE_RE.match(str(filename).replace("\\", "/"))
    return match.group("precision") if match else None


def engram_tables(files: Iterable[tuple[str, int]]) -> list[EngramTable]:
    """Standalone tables among ``(repo path, size)`` pairs, largest (most precise) first."""
    tables: dict[str, EngramTable] = {}
    for filename, size in files:
        precision = engram_table_precision(filename)
        if precision is None or not size:
            continue
        # One table per precision; a duplicate label keeps the first listed.
        tables.setdefault(precision.upper(), EngramTable(precision, filename, int(size)))
    return sorted(tables.values(), key = lambda t: (-t.size_bytes, t.precision))


def cached_engram_tables(snapshot: Path) -> list[Path]:
    """Standalone Engram tables in a local Hub snapshot."""
    folder = snapshot / ENGRAM_FOLDER
    try:
        entries = list(folder.iterdir()) if folder.is_dir() else []
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if engram_table_precision(f"{ENGRAM_FOLDER}/{entry.name}") and entry.is_file()
    ]


def find_engram_table(
    tables: Sequence[EngramTable], precision: Optional[str]
) -> Optional[EngramTable]:
    if not precision:
        return None
    wanted = precision.strip().upper()
    return next((t for t in tables if t.precision.upper() == wanted), None)


def bundled_table(
    shards: Iterable[tuple[str, int]], tables: Sequence[EngramTable]
) -> Optional[tuple[str, EngramTable]]:
    """``(shard, table)`` for the split shard that holds a quant's own table.

    The shard differs from the standalone file only in two ``split.*`` integers, so their byte
    sizes are equal; one match is required. A wrong guess cannot load the wrong weights: llama.cpp
    accepts one absent shard only when the set is a single tensor short of ``split.tensors.count``.
    """
    shards = list(shards)
    if len(shards) < 2:
        return None
    by_size = {t.size_bytes: t for t in tables}
    matches = [(name, by_size[int(size)]) for name, size in shards if int(size or 0) in by_size]
    return matches[0] if len(matches) == 1 else None


def files_llama_reads(shard_paths: Sequence[str], ple_path: Optional[str]) -> list[str]:
    """The GGUF files llama-server reads for ``shard_paths`` launched with ``--ple ple_path``.

    Mirrors llama-model-loader.cpp: a backbone shard that is absent, or that holds only the
    table, is skipped and the external table is appended.
    """
    if not ple_path:
        return list(shard_paths)
    from hub.utils.gguf_lookup_tables import ple_only_local

    kept = [
        path
        for index, path in enumerate(shard_paths)
        if index == 0 or (os.path.exists(path) and not ple_only_local(path))
    ]
    return kept + [ple_path]

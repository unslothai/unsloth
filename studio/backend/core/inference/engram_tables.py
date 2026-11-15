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
import struct
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


def _gguf_counts(path: Path, *, split_total: bool) -> Optional[int]:
    """The tensor count of a GGUF header, or with ``split_total`` its split.tensors.count."""
    from utils.models.gguf_metadata import _FIXED_VTYPE_SIZES, _skip_gguf_value

    ints = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24 or head[:4] != b"GGUF":
                return None
            _, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
            if not split_total:
                return n_tensors
            for _ in range(n_kv):
                (key_len,) = struct.unpack("<Q", f.read(8))
                key = f.read(key_len)
                (vtype,) = struct.unpack("<I", f.read(4))
                if key == b"split.tensors.count" and vtype in ints:
                    return struct.unpack(ints[vtype], f.read(_FIXED_VTYPE_SIZES[vtype]))[0]
                if not _skip_gguf_value(f, vtype):
                    return None
    except (OSError, struct.error):
        return None
    return None


def repo_snapshots(repo_id: str) -> list[Path]:
    """The cached snapshots of ``repo_id`` in the active and every known Hub cache root: the one
    place a repo's standalone tables are looked for. A plain listing, so the inventory scan can
    call it (gguf_cache_snapshots classifies, which reads the scan back)."""
    target = f"models--{repo_id.replace('/', '--')}".lower()
    try:
        from hub.utils.hf_cache_state import hf_cache_roots
        from utils.hf_cache_settings import get_hf_cache_paths
        roots = [get_hf_cache_paths().hub_cache, *hf_cache_roots()]
    except Exception:  # noqa: BLE001 -- no readable cache settings: nothing to search
        return []
    snapshots: list[Path] = []
    for root in roots:
        try:
            for repo_dir in Path(root).iterdir():
                if repo_dir.name.lower() == target and (repo_dir / "snapshots").is_dir():
                    snapshots += [
                        s for s in (repo_dir / "snapshots").iterdir() if s not in snapshots
                    ]
        except OSError:  # an unreadable root only narrows the search
            continue
    return snapshots


def cached_table_path(repo_id: str, table: EngramTable) -> Optional[str]:
    """A whole cached copy of ``table`` in any snapshot of the repo."""
    for snapshot in repo_snapshots(repo_id):
        path = snapshot.joinpath(*table.filename.split("/"))
        try:
            if path.stat().st_size == table.size_bytes:
                return str(path)
        except OSError:
            continue
    return None


def cached_sizes(repo_id: str, split: Sequence[str]) -> dict[str, int]:
    """Sizes of the cached ``split`` shards and standalone tables, for a load with no Hub listing."""
    sizes: dict[str, int] = {}
    for snapshot in repo_snapshots(repo_id):
        names = [*split, *(f"{ENGRAM_FOLDER}/{t.name}" for t in cached_engram_tables(snapshot))]
        for name in names:
            if name not in sizes:
                try:
                    sizes[name] = snapshot.joinpath(*name.split("/")).stat().st_size
                except OSError:
                    continue
    return sizes


def table_shard_skipped(present: Sequence[Path], total: int) -> bool:
    """Whether a Hub-cache split set with one shard absent loads with a cached standalone table:
    ``present`` (shard 1 first) is ``total - 1`` shards one tensor short of split.tensors.count,
    the gap llama.cpp fills from --ple. A shard cut off mid-download leaves more than one."""
    if total < 2 or len(present) != total - 1:
        return False
    root = next((p for p in Path(present[0]).parents if p.parent.name == "snapshots"), None)
    if root is None:
        return False
    repo_id = root.parent.parent.name[len("models--") :].replace("--", "/")
    if not any(map(cached_engram_tables, [root, *repo_snapshots(repo_id)])):
        return False
    expected = _gguf_counts(present[0], split_total = True)
    counts = [_gguf_counts(path, split_total = False) for path in present]
    return expected is not None and None not in counts and sum(counts) == expected - 1


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

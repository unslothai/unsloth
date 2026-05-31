# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils.paths import ensure_dir, unstructured_seed_cache_root

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 20000
_MIN_BREAK_RATIO = 0.6
_CACHE_DIR = unstructured_seed_cache_root()


def resolve_chunking(
    chunk_size: Any,
    chunk_overlap: Any,
) -> tuple[int, int]:
    size = _to_int(chunk_size, DEFAULT_CHUNK_SIZE)
    size = max(1, min(size, MAX_CHUNK_SIZE))
    overlap = _to_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP)
    overlap = max(0, min(overlap, max(0, size - 1)))
    return size, overlap


def build_unstructured_preview_rows(
    *,
    source_path: Path,
    preview_size: int,
    chunk_size: Any,
    chunk_overlap: Any,
) -> list[dict[str, str]]:
    parquet_path, rows = materialize_unstructured_seed_dataset(
        source_path = source_path,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    count = max(0, int(preview_size))
    if rows:
        return rows[:count]

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"pandas is required for unstructured seed processing: {exc}"
        ) from exc

    dataframe = pd.read_parquet(parquet_path).head(count)
    return [
        {"chunk_text": str(value.get("chunk_text", "")).strip()}
        for value in dataframe.to_dict(orient = "records")
        if str(value.get("chunk_text", "")).strip()
    ]


def build_multi_file_preview_rows(
    *,
    file_entries: list[tuple[Path, str]],
    preview_size: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[dict[str, str]]:
    cs = _to_int(chunk_size, DEFAULT_CHUNK_SIZE)
    co = _to_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP)
    _, rows = materialize_multi_file_unstructured_seed(
        file_entries = file_entries,
        chunk_size = cs,
        chunk_overlap = co,
    )
    return _round_robin_preview(rows, preview_size)


def _round_robin_preview(
    rows: list[dict[str, str]],
    preview_size: int,
) -> list[dict[str, str]]:
    """Pick preview rows round-robin across source files so every file is represented."""
    if not rows or preview_size <= 0:
        return []

    # Group rows by source_file, preserving order of first appearance
    from collections import OrderedDict

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        key = row.get("source_file", "")
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    result: list[dict[str, str]] = []
    iterators = [iter(chunks) for chunks in grouped.values()]
    while len(result) < preview_size and iterators:
        exhausted: list[int] = []
        for i, it in enumerate(iterators):
            if len(result) >= preview_size:
                break
            val = next(it, None)
            if val is not None:
                result.append(val)
            else:
                exhausted.append(i)
        for i in reversed(exhausted):
            iterators.pop(i)

    return result


def materialize_unstructured_seed_dataset(
    *,
    source_path: Path,
    chunk_size: Any,
    chunk_overlap: Any,
) -> tuple[Path, list[dict[str, str]]]:
    resolved = source_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"unstructured seed file not found: {resolved}")

    size, overlap = resolve_chunking(chunk_size, chunk_overlap)
    key = _compute_cache_key(
        source_path = resolved,
        chunk_size = size,
        chunk_overlap = overlap,
    )
    parquet_path = _CACHE_DIR / f"{key}.parquet"
    if parquet_path.exists():
        return parquet_path, []

    text = load_unstructured_text_file(resolved)
    chunks = split_text_into_chunks(
        text = text,
        chunk_size = size,
        chunk_overlap = overlap,
    )
    if not chunks:
        raise ValueError("No text found in unstructured seed source.")

    rows = [{"chunk_text": chunk} for chunk in chunks]
    ensure_dir(_CACHE_DIR)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"pandas is required for unstructured seed processing: {exc}"
        ) from exc

    tmp_path = _CACHE_DIR / f"{key}.tmp.parquet"
    pd.DataFrame(rows).to_parquet(tmp_path, index = False)
    tmp_path.replace(parquet_path)
    return parquet_path, rows


def materialize_multi_file_unstructured_seed(
    *,
    file_entries: list[tuple[Path, str]],  # (extracted_txt_path, original_filename)
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[Path, list[dict[str, str]]]:
    """Chunk multiple files and combine into one parquet dataset with source_file column."""
    chunk_size, chunk_overlap = resolve_chunking(chunk_size, chunk_overlap)
    cache_key = _compute_multi_file_cache_key(file_entries, chunk_size, chunk_overlap)
    cached = _CACHE_DIR / f"{cache_key}.parquet"
    if cached.exists():
        df = pd.read_parquet(cached)
        rows = df.to_dict(orient = "records")
        return cached, rows

    all_rows: list[dict[str, str]] = []
    for txt_path, orig_name in file_entries:
        text = load_unstructured_text_file(txt_path)
        chunks = split_text_into_chunks(
            text = text,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        for chunk in chunks:
            all_rows.append({"chunk_text": chunk, "source_file": orig_name})

    if not all_rows:
        raise ValueError("No text found in any uploaded files.")

    df = pd.DataFrame(all_rows)
    ensure_dir(_CACHE_DIR)
    tmp = _CACHE_DIR / f"{cache_key}.tmp.parquet"
    df.to_parquet(tmp, index = False)
    tmp.replace(cached)
    return cached, all_rows


def load_unstructured_text_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in {".txt", ".md"}:
        raise ValueError(f"Unsupported unstructured seed file type: {ext}")

    raw = path.read_text(encoding = "utf-8", errors = "ignore")
    return normalize_unstructured_text(raw)


def normalize_unstructured_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def split_text_into_chunks(
    *,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks: list[str] = []
    start = 0
    min_break_index = int(chunk_size * _MIN_BREAK_RATIO)
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            window = text[start:end]
            cut = _find_break_index(window, min_break_index)
            if cut is not None and cut > 0:
                end = start + cut

        if end <= start:
            end = min(text_len, start + chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = max(0, next_start)

    return chunks


def _find_break_index(window: str, min_index: int) -> int | None:
    breakpoints = ["\n\n", "\n", " "]
    for token in breakpoints:
        idx = window.rfind(token)
        if idx >= min_index:
            return idx + len(token)
    return None


def _to_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return fallback
    return parsed


def _compute_cache_key(
    *,
    source_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    stat = source_path.stat()
    payload = "|".join(
        [
            str(source_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(chunk_size),
            str(chunk_overlap),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compute_multi_file_cache_key(
    file_entries: list[tuple[Path, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    parts: list[str] = []
    for path, name in sorted(file_entries, key = lambda e: e[1]):
        st = path.stat()
        parts.append(f"{path}|{st.st_size}|{st.st_mtime_ns}|{name}")
    parts.append(f"cs={chunk_size}|co={chunk_overlap}")
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()

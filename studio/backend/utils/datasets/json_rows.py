# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""datasets types a JSON file from its first block, so a later row adding a key fails the whole load."""

import json
import os

_CHUNK_ROWS = 10_000
_CHUNK_BYTES = 16 << 20


def _row_chunks(path, size = _CHUNK_ROWS):
    with open(path, encoding = "utf-8-sig") as file:
        head = file.read(1)
        while head.isspace():
            head = file.read(1)
        file.seek(0)
        if head == "[":
            rows = json.load(file)
            if not isinstance(rows, list):
                raise ValueError("not a JSON array")
            for start in range(0, len(rows), size):
                yield rows[start : start + size]
            return
        rows, chunk_bytes = [], 0
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                chunk_bytes += len(line)
                if len(rows) == size or chunk_bytes >= _CHUNK_BYTES:
                    yield rows
                    rows, chunk_bytes = [], 0
        if rows:
            yield rows


def json_rows_features(files):
    """One schema every row of ``files`` fits (missing keys become nulls), or None if none does."""
    import pyarrow as pa
    from datasets import Features

    schema = None
    try:
        for path in files:
            for rows in _row_chunks(path):
                if not all(isinstance(row, dict) for row in rows):
                    return None
                # pa.array merges the keys of every object, at any depth.
                chunk = pa.Table.from_struct_array(pa.array(rows)).schema
                schema = (
                    chunk
                    if schema is None
                    else pa.unify_schemas([schema, chunk], promote_options = "permissive")
                )
    except (OSError, ValueError, TypeError, OverflowError, pa.ArrowException):
        return None
    return None if schema is None else Features.from_arrow_schema(schema)


def has_timestamp(data_type):
    """The Arrow JSON reader turns ISO date strings into timestamps, at any depth."""
    import pyarrow as pa

    if isinstance(data_type, pa.Schema) or pa.types.is_struct(data_type):
        return any(has_timestamp(field.type) for field in data_type)
    if hasattr(data_type, "value_type"):
        return has_timestamp(data_type.value_type)
    return pa.types.is_timestamp(data_type)


def head_rows(path, count):
    return next(_row_chunks(path, count), []) if count > 0 else []


def timestamps_as_strings(data_type):
    """``data_type`` with every timestamp, at any depth, read as the string written."""
    import pyarrow as pa

    if isinstance(data_type, pa.Schema):
        return pa.schema([f.with_type(timestamps_as_strings(f.type)) for f in data_type])
    if pa.types.is_timestamp(data_type):
        return pa.string()
    if pa.types.is_struct(data_type):
        return pa.struct([f.with_type(timestamps_as_strings(f.type)) for f in data_type])
    if pa.types.is_large_list(data_type):
        return pa.large_list(timestamps_as_strings(data_type.value_type))
    if pa.types.is_list(data_type):
        return pa.list_(timestamps_as_strings(data_type.value_type))
    return data_type


def _rows(files):
    for path, *_ in files:
        for rows in _row_chunks(path):
            yield from rows


def json_rows_dataset(files, cache_dir = None):
    """Every row of ``files`` as parsed by ``json``, so strings stay strings; None if no schema fits."""
    from datasets import Dataset

    features = json_rows_features(files)
    if features is None:
        return None
    # Size and mtime key the cache, so an edited file is reread.
    stats = [(str(path), os.stat(path).st_size, os.stat(path).st_mtime_ns) for path in files]
    return Dataset.from_generator(
        _rows,
        features = features,
        cache_dir = cache_dir,
        gen_kwargs = {"files": stats},
        # Long-context rows: a 1000-row writer batch held ~0.5 GB where the json loader held none.
        writer_batch_size = 100,
    )

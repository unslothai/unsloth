# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""datasets types a JSON file from its first block, so a later row adding a key fails the whole load."""

import json

_CHUNK_ROWS = 10_000


def _row_chunks(path):
    with open(path, encoding = "utf-8") as file:
        head = file.read(1)
        while head.isspace():
            head = file.read(1)
        file.seek(0)
        if head == "[":
            rows = json.load(file)
            if not isinstance(rows, list):
                raise ValueError("not a JSON array")
            for start in range(0, len(rows), _CHUNK_ROWS):
                yield rows[start : start + _CHUNK_ROWS]
            return
        rows = []
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == _CHUNK_ROWS:
                    yield rows
                    rows = []
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
    except (OSError, ValueError, TypeError, pa.ArrowException):
        return None
    return None if schema is None else Features.from_arrow_schema(schema)

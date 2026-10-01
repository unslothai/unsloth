# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning raw dataset cells into the text we are willing to train on."""

from pathlib import Path

# Feature ids tagging each column of a CSV read by `csv_as_text_kwargs`.
_CSV_TEXT = "csv_text"
_CSV_TYPED = "csv_typed"
# The csv loader types each column from the first chunk of the first file.
_CSV_CHUNK_ROWS = 10_000
# pandas' default missing-value markers.
_NA_CELLS = frozenset(
    ("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN")
    + ("<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null")
)


def cell_text(value):
    """The text to train on for a single Alpaca-style dataset cell.

    A blank cell is a missing value to every loader we use, so it arrives as
    None, and a column holding numbers arrives typed, so a cell typed as 1
    arrives as 1.0. Untreated, the first trains the word None and the second
    raises on .strip(). NaN counts as blank too: Arrow normalises a blank
    numeric cell to null, but a dataset built straight from pandas need not have
    passed through Arrow.
    """
    if value is None:
        return ""
    if isinstance(value, dict) and {"text", "answer_start"} <= value.keys():
        # SQuAD-style `answers` span: train the first answer, as _extract_column_value does.
        answer = value["text"]
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        return cell_text(answer)
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _column_ids(dataset) -> dict:
    features = getattr(dataset, "features", None) or {}
    return {column: getattr(feature, "id", None) for column, feature in features.items()}


def typed_csv_columns(dataset) -> frozenset:
    """CSV columns the csv loader would have typed: numbers, booleans or only missing cells."""
    return frozenset(column for column, tag in _column_ids(dataset).items() if tag == _CSV_TYPED)


def text_cell_check(dataset):
    """`(column, cell) -> bool`: False for CSV cells the csv loader would not read as strings."""
    ids = _column_ids(dataset)

    def is_text(column, cell):
        tag = ids.get(column)
        return tag not in (_CSV_TEXT, _CSV_TYPED) or (tag == _CSV_TEXT and cell not in _NA_CELLS)

    return is_text


def csv_as_text_kwargs(files):
    if Path(files[0]).suffix.lower() != ".csv":
        return {}
    import pandas as pd
    import pyarrow as pa
    from datasets import Features, Value

    with pd.read_csv(files[0], chunksize = _CSV_CHUNK_ROWS) as chunks:
        first = next(chunks)
    headers = [first.columns] + [pd.read_csv(path, nrows = 0).columns for path in files[1:]]
    # A string schema needs the same columns in every file; else keep the loader's defaults.
    if any(set(columns) != set(headers[0]) for columns in headers):
        return {}
    strings = {
        field.name
        for field in pa.Table.from_pandas(first).schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }
    tags = {name: _CSV_TEXT if name in strings else _CSV_TYPED for name in first.columns}
    return {
        "features": Features({name: Value("string", id = tag) for name, tag in tags.items()}),
        "keep_default_na": False,
        "na_values": [""],
    }

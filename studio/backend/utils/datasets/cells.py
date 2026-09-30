# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning one raw dataset cell into the text we are willing to train on."""

from pathlib import Path


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


def csv_as_text_kwargs(files, raw_text = False):
    # Raw text trains the first string column, so a numeric id column must keep its type.
    if raw_text or Path(files[0]).suffix.lower() != ".csv":
        return {}
    import pandas as pd
    from datasets import Features, Value

    columns = pd.read_csv(files[0], nrows = 0).columns
    return {
        "features": Features({name: Value("string") for name in columns}),
        "keep_default_na": False,
        "na_values": [""],
    }

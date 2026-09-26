# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning one raw dataset cell into the text we are willing to train on."""


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
        # A SQuAD-style `answers` cell arrives as {"text": [...], "answer_start": [...]}.
        # Train the answer, as AI Assist's mapping already does, not the dict's repr.
        answer = value["text"]
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        return cell_text(answer)
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:
        return ""
    return str(value)

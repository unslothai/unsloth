# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning one raw dataset cell into the text we are willing to train on."""


def cell_text(value):
    """The text to train on for a single Alpaca-style dataset cell.

    Two things arrive here that are not text. A blank cell is a missing value to
    every loader we use, so it arrives as None, and a column holding numbers is
    typed by the reader, so a cell someone typed as 1 arrives as the float 1.0.
    Handing either to a prompt template trains the word None into the row, and
    handing a non-string to the ChatML conversion raises on .strip().

    NaN is folded in with None: Arrow normalises a blank numeric cell to null
    before we see it, but a dataset built straight from pandas need not have
    passed through Arrow, and a NaN reaching a template writes the word nan.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:
        return ""
    return str(value)

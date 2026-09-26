# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation helpers for final text-SFT datasets."""

from __future__ import annotations

from typing import Any

from .iterable import is_streaming_dataset


_MISSING = object()


def _row_text_issue(row: Any, field_name: str) -> str | None:
    if not isinstance(row, dict):
        return f"is not a dictionary row (got {type(row).__name__})"

    value = row.get(field_name, _MISSING)
    if value is _MISSING:
        return f"is missing the `{field_name}` field"
    if not isinstance(value, str):
        return f"has a non-string `{field_name}` field (got {type(value).__name__})"
    if not value.strip():
        return f"has an empty `{field_name}` field"
    return None


def _validation_error(*, split_name: str, row_index: int, issue: str) -> ValueError:
    return ValueError(
        f"Dataset validation failed: {split_name} row {row_index} {issue}. "
        "Unsloth expects every row used for text SFT to contain non-empty text. "
        "Remove blank rows or fix the dataset formatting/template so it produces "
        "training text."
    )


def validate_text_sft_dataset(
    dataset: Any,
    *,
    field_name: str = "text",
    split_name: str = "train",
    is_vlm: bool = False,
) -> Any:
    """Check materialized text splits for an empty split or invalid first row.

    Leave streaming datasets untouched: probing a filtered stream can consume
    an unbounded invalid prefix before finding its first retained row.
    """
    if is_vlm or is_streaming_dataset(dataset) or not hasattr(dataset, "__len__"):
        return dataset

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset validation failed: the {split_name} split is empty. "
            "Add at least one training example before starting text SFT."
        )

    issue = _row_text_issue(dataset[0], field_name)
    if issue is not None:
        raise _validation_error(
            split_name = split_name,
            row_index = 0,
            issue = issue,
        )
    return dataset

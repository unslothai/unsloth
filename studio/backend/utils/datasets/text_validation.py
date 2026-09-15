# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation helpers for final text-SFT datasets."""

from __future__ import annotations

from typing import Any

from .iterable import is_streaming_dataset


_MISSING = object()


def _row_text_issue(row: Any, field_name: str, *, allow_non_string: bool) -> str | None:
    if not isinstance(row, dict):
        if allow_non_string:
            return None
        return f"is not a dictionary row (got {type(row).__name__})"

    value = row.get(field_name, _MISSING)
    if value is _MISSING:
        if allow_non_string:
            return None
        return f"is missing the `{field_name}` field"
    if not isinstance(value, str):
        if allow_non_string:
            return None
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


def _validate_streaming_row(
    row: Any, row_index: int, *, field_name: str, split_name: str, allow_non_string: bool
) -> Any:
    issue = _row_text_issue(
        row,
        field_name,
        allow_non_string = allow_non_string,
    )
    if issue is not None:
        raise _validation_error(
            split_name = split_name,
            row_index = row_index,
            issue = issue,
        )
    return row


def validate_text_sft_dataset(
    dataset: Any,
    *,
    field_name: str = "text",
    split_name: str = "train",
    is_vlm: bool = False,
    allow_non_string: bool = False,
) -> Any:
    """Return a dataset whose consumed text-SFT rows are guaranteed non-empty.

    Materialized datasets are validated eagerly and completely. Streaming
    datasets receive a lazy ``map`` validator so preparation never scans an
    unbounded source merely to observe a target number of kept rows.

    ``allow_non_string`` is used only by raw/CPT streaming preparation before
    its existing lazy filter. It preserves the established behavior of dropping
    null/non-string raw rows while still rejecting raw blank strings before EOS
    is appended.
    """

    if is_vlm:
        return dataset

    if is_streaming_dataset(dataset) or not hasattr(dataset, "__len__"):
        map_dataset = getattr(dataset, "map", None)
        if not callable(map_dataset):
            raise TypeError(
                "Streaming text dataset validation requires a lazy `map` implementation."
            )
        return map_dataset(
            _validate_streaming_row,
            with_indices = True,
            fn_kwargs = {
                "field_name": field_name,
                "split_name": split_name,
                "allow_non_string": allow_non_string,
            },
        )

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset validation failed: the {split_name} split is empty. "
            "Add at least one training example before starting text SFT."
        )

    for row_index, row in enumerate(dataset):
        issue = _row_text_issue(
            row,
            field_name,
            allow_non_string = allow_non_string,
        )
        if issue is not None:
            raise _validation_error(
                split_name = split_name,
                row_index = row_index,
                issue = issue,
            )

    return dataset

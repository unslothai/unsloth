# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation helpers for text SFT datasets."""

from __future__ import annotations

from typing import Any


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


def _dataset_length(dataset: Any) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def validate_non_empty_text_field(
    dataset: Any,
    *,
    field_name: str = "text",
    split_name: str = "train",
    max_scan_rows: int = 1000,
) -> None:
    """Raise a user-facing error if a text SFT split has invalid text rows.

    Unsloth's SFTTrainer probes the text field while preparing the dataset. Empty
    strings otherwise surface as low-level ``string index out of range`` errors,
    so Studio validates the formatted dataset first and reports the row/field.
    """

    row_count = _dataset_length(dataset)
    if row_count == 0:
        raise ValueError(
            f"Dataset validation failed: the {split_name} split is empty. "
            "Add at least one training example before starting text SFT."
        )

    if row_count is None:
        try:
            first_row = next(iter(dataset))
        except StopIteration as exc:
            raise ValueError(
                f"Dataset validation failed: the {split_name} split is empty. "
                "Add at least one training example before starting text SFT."
            ) from exc

        issue = _row_text_issue(first_row, field_name)
        if issue is not None:
            raise ValueError(
                f"Dataset validation failed: {split_name} row 0 {issue}. "
                "Unsloth expects every row used for text SFT to contain non-empty text. "
                "Remove blank rows or fix the dataset formatting/template so it "
                "produces training text."
            )
        return

    scan_rows = min(row_count, max_scan_rows)
    first_issue: tuple[int, str] | None = None
    issue_count = 0

    for row_index, row in enumerate(dataset):
        if row_index >= scan_rows:
            break

        issue = _row_text_issue(row, field_name)
        if issue is None:
            continue
        issue_count += 1
        if first_issue is None:
            first_issue = (row_index, issue)

    if first_issue is None:
        return

    first_row_index, first_issue_text = first_issue
    scanned_label = (
        f"{issue_count} invalid row(s) in the first {scan_rows} rows. " if issue_count > 1 else ""
    )
    raise ValueError(
        f"Dataset validation failed: {split_name} row {first_row_index} "
        f"{first_issue_text}. {scanned_label}"
        "Unsloth expects every row used for text SFT to contain non-empty text. "
        "Remove blank rows or fix the dataset formatting/template so it produces "
        "training text."
    )

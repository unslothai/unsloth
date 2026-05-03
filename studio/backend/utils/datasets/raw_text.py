# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Shared helpers for raw-text dataset preparation.
"""

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset


@dataclass(frozen = True)
class RawTextNotice:
    message: str
    level: Literal["info", "warning"]
    update_status: bool = False


@dataclass(frozen = True)
class RawTextPreparationResult:
    dataset: Dataset
    notices: list[RawTextNotice]


def _string_columns(dataset: Dataset) -> list[str]:
    feature_map = getattr(dataset, "features", {}) or {}
    string_cols: list[str] = []
    for col in dataset.column_names:
        feature = feature_map.get(col)
        dtype = str(getattr(feature, "dtype", ""))
        if dtype in {"string", "large_string"}:
            string_cols.append(col)
    return string_cols


def _split_scope(split_name: str | None) -> str:
    return f"the {split_name} split" if split_name else "this dataset"


def _drop_invalid_text_rows(
    dataset: Dataset,
    *,
    mode_title: str,
    split_scope: str,
) -> tuple[Dataset, list[RawTextNotice]]:
    filtered_dataset = dataset.filter(lambda ex: isinstance(ex["text"], str))
    dropped_rows = len(dataset) - len(filtered_dataset)
    if not dropped_rows:
        return filtered_dataset, []

    if len(filtered_dataset) == 0:
        raise ValueError(
            f"{mode_title} training requires at least one string 'text' value "
            f"in {split_scope}; all {dropped_rows} rows were null or non-string."
        )

    return filtered_dataset, [
        RawTextNotice(
            message = (
                f"{mode_title}: dropped {dropped_rows:,} row(s) with null or "
                f"non-string 'text' values from {split_scope}"
            ),
            level = "warning",
            update_status = True,
        )
    ]


def prepare_raw_text_dataset(
    dataset: Dataset,
    *,
    mode_label: str = "raw text",
    split_name: str | None = None,
    eos_token: str | None = None,
    append_eos: bool = False,
) -> RawTextPreparationResult:
    notices: list[RawTextNotice] = []
    mode_title = mode_label.capitalize()
    split_scope = _split_scope(split_name)

    if "text" not in dataset.column_names:
        string_cols = _string_columns(dataset)
        if not string_cols:
            raise ValueError(
                f"{mode_title} training requires a string 'text' column but none "
                f"was found in {split_scope} (columns: {dataset.column_names})."
            )

        renamed_col = string_cols[0]
        if len(string_cols) > 1:
            notices.append(
                RawTextNotice(
                    message = (
                        f"{mode_title}: dataset has {len(string_cols)} string "
                        f"columns ({string_cols}); auto-selecting '{renamed_col}' "
                        "as the training text. Rename the intended column to "
                        "'text' to override."
                    ),
                    level = "warning",
                    update_status = True,
                )
            )
        notices.append(
            RawTextNotice(
                message = (
                    f"{mode_title}: renaming column '{renamed_col}' -> 'text' "
                    f"for {split_scope}"
                ),
                level = "info",
            )
        )
        dataset = dataset.rename_column(renamed_col, "text")

    dataset, invalid_row_notices = _drop_invalid_text_rows(
        dataset,
        mode_title = mode_title,
        split_scope = split_scope,
    )
    notices.extend(invalid_row_notices)

    if append_eos:
        if not eos_token:
            notices.append(
                RawTextNotice(
                    message = (
                        f"{mode_title}: tokenizer has no eos_token; skipping EOS "
                        "append. Model will not learn document boundaries."
                    ),
                    level = "warning",
                )
            )
        else:

            def _append_eos(ex, _eos = eos_token):
                text = ex["text"]
                return {"text": text if text.endswith(_eos) else text + _eos}

            dataset = dataset.map(_append_eos)

    return RawTextPreparationResult(dataset = dataset, notices = notices)

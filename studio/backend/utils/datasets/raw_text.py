# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared helpers for raw-text dataset preparation."""

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset

from .iterable import is_streaming_dataset
from .text_validation import validate_text_sft_dataset


@dataclass(frozen = True)
class RawTextNotice:
    message: str
    level: Literal["info", "warning"]
    update_status: bool = False


@dataclass(frozen = True)
class RawTextPreparationResult:
    dataset: Dataset
    notices: list[RawTextNotice]


def resolve_column_names(dataset) -> list[str]:
    """Return the column names for *dataset*, guarding against None.

    IterableDataset.column_names is None until HF datasets>=X materialises
    it from the first batch; .map() also keeps it None.  Resolution order:
      1. dataset.column_names if truthy (regular Dataset or HF>=4.4)
      2. keys of dataset.features if available
      3. bounded first-row probe, consumes one element, safe on IterableDataset
         because HF re-iterates from the generator on the next pass
      4. [] as a last resort so callers never see None
    """
    col_names = getattr(dataset, "column_names", None)
    if col_names:
        return list(col_names)

    features = getattr(dataset, "features", None)
    if features:
        return list(features.keys())

    try:
        first_row = next(iter(dataset))
        return list(first_row.keys())
    except Exception:
        return []


def _string_columns(dataset: Dataset) -> list[str]:
    feature_map = getattr(dataset, "features", {}) or {}
    string_cols: list[str] = []
    for col in resolve_column_names(dataset):
        feature = feature_map.get(col)
        dtype = str(getattr(feature, "dtype", ""))
        if dtype in {"string", "large_string"}:
            string_cols.append(col)
    return string_cols


def _split_scope(split_name: str | None) -> str:
    return f"the {split_name} split" if split_name else "this dataset"


def _drop_invalid_text_rows(
    dataset: Dataset, *, mode_title: str, split_scope: str
) -> tuple[Dataset, list[RawTextNotice]]:
    # Lazy filter — drops rows whose 'text' is null/non-string before they reach
    # the tokenizer. Works on both Dataset and streaming IterableDataset.
    filtered_dataset = dataset.filter(lambda ex: isinstance(ex["text"], str))

    # Streaming datasets (IterableDataset) have no __len__, so we can't count the
    # dropped rows or verify the result is non-empty without consuming the whole
    # stream. Keep the filter, skip only the len()-based diagnostics.
    if not hasattr(dataset, "__len__"):
        return filtered_dataset, [
            RawTextNotice(
                message = (
                    f"{mode_title}: streaming dataset — rows with null or "
                    f"non-string 'text' in {split_scope} are dropped on the fly."
                ),
                level = "info",
            )
        ]

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
    validation_split_name = split_name or "raw text"

    col_names = resolve_column_names(dataset)
    if "text" not in col_names:
        string_cols = _string_columns(dataset)
        if not string_cols:
            raise ValueError(
                f"{mode_title} training requires a string 'text' column but none "
                f"was found in {split_scope} (columns: {col_names})."
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
                    f"{mode_title}: renaming column '{renamed_col}' -> 'text' " f"for {split_scope}"
                ),
                level = "info",
            )
        )
        dataset = dataset.rename_column(renamed_col, "text")

    streaming = is_streaming_dataset(dataset) or not hasattr(dataset, "__len__")
    if streaming:
        # Validate raw strings lazily before the existing filter/EOS maps.
        # Null/non-string rows remain eligible for the established lazy drop.
        dataset = validate_text_sft_dataset(
            dataset,
            split_name = validation_split_name,
            allow_non_string = True,
        )

    dataset, invalid_row_notices = _drop_invalid_text_rows(
        dataset,
        mode_title = mode_title,
        split_scope = split_scope,
    )
    notices.extend(invalid_row_notices)

    if not streaming:
        # Materialized raw datasets can be checked completely before EOS
        # mutation. Null/non-string rows have already been dropped above.
        dataset = validate_text_sft_dataset(
            dataset,
            split_name = validation_split_name,
        )

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

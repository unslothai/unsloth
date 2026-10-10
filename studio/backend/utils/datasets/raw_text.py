# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared helpers for raw-text dataset preparation."""

# `Dataset` is annotation-only: a module-scope `datasets` import drags torch in via
# datasets.formatting.torch_formatter.
from __future__ import annotations

from dataclasses import dataclass
import re
from itertools import islice
from typing import Literal, TYPE_CHECKING

from .cells import typed_csv_columns

if TYPE_CHECKING:
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
    source_column: str = "text"


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


def _text_columns(dataset: Dataset, string_cols: list[str]) -> list[str]:
    # Every column of an uploaded CSV is text; skip one pandas would have typed, like an id.
    typed = typed_csv_columns(dataset)
    return [col for col in string_cols if col not in typed] or string_cols


# One unit per CJK / kana / Thai character (scripts written without spaces), else per word, with
# unspaced runs cut every 16 characters: minified code counts by length, an id or hash stays short.
_UNSPACED = "\u0e00-\u0e7f\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff"
_TEXT_UNIT = re.compile(f"[{_UNSPACED}]|[^\\s{_UNSPACED}]{{1,16}}")


def _pick_text_column(dataset: Dataset, text_cols: list[str]) -> str:
    if len(text_cols) == 1:
        return text_cols[0]
    rows = list(islice(dataset.select_columns(text_cols), 100))
    return max(
        text_cols,
        key = lambda col: sum(
            len(_TEXT_UNIT.findall(row[col])) for row in rows if isinstance(row[col], str)
        ),
    )


def _split_scope(split_name: str | None) -> str:
    return f"the {split_name} split" if split_name else "this dataset"


def _drop_invalid_text_rows(
    dataset: Dataset,
    *,
    mode_title: str,
    split_scope: str,
    allow_empty: bool = False,
) -> tuple[Dataset, list[RawTextNotice]]:
    # Lazy filter — drops rows whose 'text' is null/non-string/blank before they reach
    # the tokenizer. Works on both Dataset and streaming IterableDataset.
    filtered_dataset = dataset.filter(
        lambda ex: isinstance(ex["text"], str) and bool(ex["text"].strip())
    )

    # Streaming datasets (IterableDataset) have no __len__, so we can't count the
    # dropped rows or verify the result is non-empty without consuming the whole
    # stream. Keep the filter, skip only the len()-based diagnostics.
    if not hasattr(dataset, "__len__"):
        return filtered_dataset, [
            RawTextNotice(
                message = (
                    f"{mode_title}: streaming dataset — rows with null, non-string "
                    f"or blank 'text' in {split_scope} are dropped on the fly."
                ),
                level = "info",
            )
        ]

    dropped_rows = len(dataset) - len(filtered_dataset)
    if not dropped_rows:
        return filtered_dataset, []

    # An empty eval split falls through to the trainer, which warns and skips evaluation.
    if len(filtered_dataset) == 0 and not allow_empty:
        raise ValueError(
            f"{mode_title} training requires at least one non-blank string 'text' value "
            f"in {split_scope}; all {dropped_rows} rows were null, non-string or blank."
        )

    return filtered_dataset, [
        RawTextNotice(
            message = (
                f"{mode_title}: dropped {dropped_rows:,} row(s) with null, non-string "
                f"or blank 'text' values from {split_scope}"
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
    text_column: str | None = None,
) -> RawTextPreparationResult:
    notices: list[RawTextNotice] = []
    mode_title = mode_label[:1].upper() + mode_label[1:]
    split_scope = _split_scope(split_name)
    renamed_col = "text"

    col_names = resolve_column_names(dataset)
    if "text" not in col_names:
        string_cols = _string_columns(dataset)
        if not string_cols:
            raise ValueError(
                f"{mode_title} training requires a string 'text' column but none "
                f"was found in {split_scope} (columns: {col_names})."
            )

        text_cols = _text_columns(dataset, string_cols)
        # An eval split reuses the train split's column: per-split word counts can disagree.
        if text_column in string_cols:
            renamed_col = text_column
        else:
            renamed_col = _pick_text_column(dataset, text_cols)
        if len(text_cols) > 1 and renamed_col != text_column:
            notices.append(
                RawTextNotice(
                    message = (
                        f"{mode_title}: dataset has {len(text_cols)} string "
                        f"columns ({text_cols}); auto-selecting '{renamed_col}' "
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

    dataset, invalid_row_notices = _drop_invalid_text_rows(
        dataset,
        mode_title = mode_title,
        split_scope = split_scope,
        allow_empty = split_name == "eval",
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

    return RawTextPreparationResult(dataset = dataset, notices = notices, source_column = renamed_col)

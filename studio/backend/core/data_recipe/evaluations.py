# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apply evaluations (column comparisons that produce a score column).

`data_designer` doesn't know what an evaluation is — its `ColumnConfig` and
`ProcessorType` schemas are closed enums — so we run them ourselves on the
DataFrame after `designer.preview()` / `designer.create()` returns.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.json_score.api import _extract_json, score_from_text


def score_dataframe(df: "pd.DataFrame", evaluations: list[dict[str, Any]]) -> "pd.DataFrame":
    """Apply every evaluation to `df` in place, returning the same df."""
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        if evaluation.get("processor_type") == "json_document_score":
            _apply_json_document_score(
                df,
                prediction_column = str(evaluation.get("prediction_column", "")),
                reference_column = str(evaluation.get("reference_column", "")),
                schema = evaluation.get("schema"),
                default_comparator = str(evaluation.get("default_comparator", "string")),
                score_column = str(evaluation.get("score_column", "doc_score")),
                breakdown_column = evaluation.get("breakdown_column") or None,
            )
    return df


def score_parquet_dir(parquet_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    """Apply every evaluation to every parquet batch in `parquet_dir`.

    No-op when evaluations is empty. Each parquet is read, scored, and
    rewritten atomically via tmp-file rename.
    """
    if not evaluations:
        return
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files under {parquet_dir}")
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        score_dataframe(df, evaluations)
        tmp_file = parquet_file.with_suffix(parquet_file.suffix + ".tmp")
        df.to_parquet(tmp_file, index = False)
        os.replace(tmp_file, parquet_file)


def _coerce_value(value: Any) -> Any:
    """Parquet list columns come back as numpy.ndarray — convert to list so
    json_score's isinstance(value, list) check works. Other types pass through."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _coerce_reference(value: Any) -> Any:
    """Reference columns may also be stringified dicts when produced by a
    Jinja-templated Formula. `score_from_text` only parses its prediction
    arg, so parse the reference here too if it's a parseable string."""
    coerced = _coerce_value(value)
    if isinstance(coerced, str):
        parsed = _extract_json(coerced)
        if parsed is not None:
            return parsed
    return coerced


def _apply_json_document_score(
    df: "pd.DataFrame",
    *,
    prediction_column: str,
    reference_column: str,
    schema: Any,
    default_comparator: str,
    score_column: str,
    breakdown_column: str | None,
) -> None:
    if prediction_column not in df.columns:
        raise ValueError(
            f"prediction_column {prediction_column!r} not in dataset (have: {list(df.columns)})"
        )
    if reference_column not in df.columns:
        raise ValueError(
            f"reference_column {reference_column!r} not in dataset (have: {list(df.columns)})"
        )

    want_breakdown = bool(breakdown_column)
    predictions = df[prediction_column].to_numpy()
    references = df[reference_column].to_numpy()

    scores: list[float] = []
    breakdowns: list[str] = []
    for prediction_value, reference_value in zip(predictions, references):
        score, node = score_from_text(
            _coerce_reference(reference_value),
            _coerce_value(prediction_value),
            schema,
            default_comparator = default_comparator,
            return_key_scores = True,
        )
        scores.append(float(score))
        if want_breakdown:
            breakdown = dataclasses.asdict(node) if dataclasses.is_dataclass(node) else node
            breakdowns.append(json.dumps(breakdown))

    df[score_column] = scores
    if want_breakdown and breakdown_column:
        df[breakdown_column] = breakdowns

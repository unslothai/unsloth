# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Score a JSON prediction column against a reference column, in place.

Reads every `*.parquet` under `parquet_dir`, computes a per-row score using
`eval.json_score.score_from_text`, and writes a new column back into the
same parquet file. Optionally also writes a per-field breakdown column (the
ScoreNode serialized as JSON).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.json_score.api import score_from_text


def _coerce_value(value: Any) -> Any:
    """Parquet list columns come back as numpy.ndarray — convert to list so
    json_score's isinstance(value, list) check works. Other types pass through."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _score_row(
    *,
    prediction: Any,
    reference: Any,
    schema: Any,
    default_comparator: str,
    want_breakdown: bool,
) -> tuple[float, Any]:
    score, node = score_from_text(
        _coerce_value(reference),
        _coerce_value(prediction),
        schema,
        default_comparator=default_comparator,
        return_key_scores=True,
    )
    breakdown = (
        dataclasses.asdict(node)
        if want_breakdown and dataclasses.is_dataclass(node)
        else (node if want_breakdown else None)
    )
    return float(score), breakdown


def _score_dataframe(
    df: "pd.DataFrame",
    *,
    prediction_column: str,
    reference_column: str,
    schema: Any,
    default_comparator: str,
    score_column: str,
    breakdown_column: str | None,
) -> "pd.DataFrame":
    """Add score (and optionally breakdown) columns to `df` in place and return it."""
    if prediction_column not in df.columns:
        raise ValueError(
            f"prediction_column {prediction_column!r} not in dataset "
            f"(have: {list(df.columns)})"
        )
    if reference_column not in df.columns:
        raise ValueError(
            f"reference_column {reference_column!r} not in dataset "
            f"(have: {list(df.columns)})"
        )

    want_breakdown = bool(breakdown_column)
    predictions = df[prediction_column].to_numpy()
    references = df[reference_column].to_numpy()

    scores: list[float] = []
    breakdowns: list[str] = []
    for prediction_value, reference_value in zip(predictions, references):
        score, breakdown = _score_row(
            prediction=prediction_value,
            reference=reference_value,
            schema=schema,
            default_comparator=default_comparator,
            want_breakdown=want_breakdown,
        )
        scores.append(score)
        if want_breakdown:
            breakdowns.append(json.dumps(breakdown))

    df[score_column] = scores
    if want_breakdown and breakdown_column:
        df[breakdown_column] = breakdowns
    return df


def run_json_document_score(
    parquet_dir: Path,
    *,
    prediction_column: str,
    reference_column: str,
    schema: Any,
    default_comparator: str,
    score_column: str,
    breakdown_column: str | None,
) -> None:
    """Add `score_column` (and optionally `breakdown_column`) to every
    parquet file in `parquet_dir`, scoring `prediction_column` against
    `reference_column` using json_score."""
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files under {parquet_dir}")

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        _score_dataframe(
            df,
            prediction_column=prediction_column,
            reference_column=reference_column,
            schema=schema,
            default_comparator=default_comparator,
            score_column=score_column,
            breakdown_column=breakdown_column,
        )
        tmp_file = parquet_file.with_suffix(parquet_file.suffix + ".tmp")
        df.to_parquet(tmp_file, index=False)
        os.replace(tmp_file, parquet_file)


def run_json_document_score_on_dataframe(
    df: "pd.DataFrame",
    *,
    prediction_column: str,
    reference_column: str,
    schema: Any,
    default_comparator: str,
    score_column: str,
    breakdown_column: str | None,
) -> "pd.DataFrame":
    """In-memory entrypoint for the preview path. Returns the (mutated) df."""
    return _score_dataframe(
        df,
        prediction_column=prediction_column,
        reference_column=reference_column,
        schema=schema,
        default_comparator=default_comparator,
        score_column=score_column,
        breakdown_column=breakdown_column,
    )

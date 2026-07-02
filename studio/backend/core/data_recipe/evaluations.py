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
from eval.json_score.schema import normalize_schema


def apply_document_score(df: "pd.DataFrame", evaluation: dict[str, Any]) -> None:
    """Apply one json_document_score evaluation to `df` in place."""
    prediction_column = str(evaluation.get("prediction_column", ""))
    reference_column = str(evaluation.get("reference_column", ""))
    schema = evaluation.get("schema")
    default_comparator = str(evaluation.get("default_comparator", "string"))
    score_column = str(evaluation.get("score_column", "doc_score"))
    breakdown_column = evaluation.get("breakdown_column")

    if prediction_column not in df.columns:
        raise ValueError(
            f"prediction_column {prediction_column!r} not in dataset (have: {list(df.columns)})"
        )
    if reference_column not in df.columns:
        raise ValueError(
            f"reference_column {reference_column!r} not in dataset (have: {list(df.columns)})"
        )

    # Normalize the schema once up front — score_from_text passes an already-Node
    # through unchanged, so this avoids re-walking + revalidating on every row.
    node = normalize_schema(schema) if schema is not None else None
    want_breakdown = bool(breakdown_column)
    predictions = df[prediction_column].to_numpy()
    references = df[reference_column].to_numpy()

    scores: list[float] = []
    breakdowns: list[str] = []
    for prediction_value, reference_value in zip(predictions, references):
        score, node_result = score_from_text(
            _coerce_reference(reference_value),
            _coerce_value(prediction_value),
            node,
            default_comparator = default_comparator,
            return_key_scores = True,
        )
        scores.append(float(score))
        if want_breakdown:
            breakdown = (
                dataclasses.asdict(node_result)
                if dataclasses.is_dataclass(node_result)
                else node_result
            )
            breakdowns.append(json.dumps(breakdown))

    df[score_column] = scores
    if want_breakdown and breakdown_column:
        df[breakdown_column] = breakdowns


def score_parquet_dir(parquet_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    """Apply every evaluation to every parquet batch in `parquet_dir`.

    No-op when evaluations is empty. Each parquet is read, scored, and
    rewritten atomically via tmp-file rename. Files whose dataframes weren't
    mutated (no matching processor_type) are left untouched — a bare read/write
    can change compression, row-group layout, or pandas metadata vs. what
    `data_designer` produced.
    """
    if not evaluations:
        return
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files under {parquet_dir}")
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        changed = False
        for evaluation in evaluations:
            if evaluation.get("processor_type") == "json_document_score":
                apply_document_score(df, evaluation)
                changed = True
        if not changed:
            continue
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
    parsed = _extract_json(coerced)
    return parsed if parsed is not None else coerced

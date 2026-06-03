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
from pathlib import Path
from typing import Any

import pandas as pd

from eval.json_score.api import score_from_text


def _score_node_to_dict(node: Any) -> Any:
    """ScoreNode is a dataclass; recurse into its `children`."""
    if dataclasses.is_dataclass(node):
        d = dataclasses.asdict(node)
        return d
    return node


def _score_row(
    *,
    prediction: Any,
    reference: Any,
    schema: Any,
    default_comparator: str,
    want_breakdown: bool,
) -> tuple[float, Any]:
    score, node = score_from_text(
        reference,
        prediction,
        schema,
        default_comparator=default_comparator,
        return_key_scores=True,
    )
    breakdown = _score_node_to_dict(node) if want_breakdown else None
    return float(score), breakdown


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

    want_breakdown = bool(breakdown_column)

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
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

        scores: list[float] = []
        breakdowns: list[str | None] = []
        for _, row in df.iterrows():
            score, breakdown = _score_row(
                prediction=row[prediction_column],
                reference=row[reference_column],
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

        df.to_parquet(parquet_file, index=False)

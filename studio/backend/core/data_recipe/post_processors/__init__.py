# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned post-processors for the recipe `evaluations` field.

`data_designer` doesn't know about evaluations, so we run them out-of-band:
after `designer.create()` (or `preview()`) returns, the worker calls
`apply_studio_post_processors` against the resulting parquet (or DataFrame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_document_score import (
    run_json_document_score,
    run_json_document_score_on_dataframe,
)


def apply_studio_post_processors(
    *, base_dataset_path: Path, evaluations: list[dict[str, Any]]
) -> None:
    """Run every evaluation against the recipe's parquet output.

    Evaluations run in declaration order; later evaluations see earlier
    evaluations' output.
    """
    parquet_dir = base_dataset_path / "parquet-files"
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        evaluation_type = evaluation.get("evaluation_type")
        if evaluation_type == "json_document_score":
            run_json_document_score(
                parquet_dir,
                prediction_column = str(evaluation.get("prediction_column", "")),
                reference_column = str(evaluation.get("reference_column", "")),
                schema = evaluation.get("schema"),
                default_comparator = str(evaluation.get("default_comparator", "string")),
                score_column = str(evaluation.get("score_column", "doc_score")),
                breakdown_column = (
                    evaluation.get("breakdown_column") if evaluation.get("breakdown_column") else None
                ),
            )


def apply_studio_post_processors_to_dataframe(*, df, evaluations: list[dict[str, Any]]):
    """In-memory variant of apply_studio_post_processors. Runs evaluations
    against an in-memory DataFrame (used by the preview path). Returns the
    (mutated) df."""
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        evaluation_type = evaluation.get("evaluation_type")
        if evaluation_type == "json_document_score":
            df = run_json_document_score_on_dataframe(
                df,
                prediction_column = str(evaluation.get("prediction_column", "")),
                reference_column = str(evaluation.get("reference_column", "")),
                schema = evaluation.get("schema"),
                default_comparator = str(evaluation.get("default_comparator", "string")),
                score_column = str(evaluation.get("score_column", "doc_score")),
                breakdown_column = (
                    evaluation.get("breakdown_column") if evaluation.get("breakdown_column") else None
                ),
            )
    return df

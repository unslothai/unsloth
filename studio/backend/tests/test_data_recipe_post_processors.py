# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.data_recipe.post_processors import (
    STUDIO_PROCESSOR_TYPES,
    apply_studio_post_processors,
)
from core.data_recipe.post_processors.json_document_score import (
    run_json_document_score,
)


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "prediction": [
                json.dumps({"name": "Alice", "age": 30}),
                json.dumps({"name": "Bob", "age": 25}),
                "this is not json at all",
            ],
            "reference": [
                {"name": "Alice", "age": 30},
                {"name": "Bobby", "age": 25},
                {"name": "Cara", "age": 40},
            ],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index=False)
    return parquet_dir


def test_run_json_document_score_adds_score_column(parquet_dir: Path) -> None:
    # Use the exact-match "categorical" comparator so the row-1 expectation
    # (name mismatch -> 0, age match -> 1, avg -> 0.5) is unambiguous; the
    # default "string" comparator is fuzzy (Levenshtein) and would give 0.8
    # for "Bob" vs "Bobby".
    run_json_document_score(
        parquet_dir,
        prediction_column="prediction",
        reference_column="reference",
        schema=None,
        default_comparator="categorical",
        score_column="doc_score",
        breakdown_column=None,
    )

    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "doc_score" in df.columns
    scores = df["doc_score"].tolist()
    # Row 0: perfect match -> 1.0
    assert scores[0] == pytest.approx(1.0)
    # Row 1: name mismatch, age match -> 0.5
    assert scores[1] == pytest.approx(0.5)
    # Row 2: unparseable -> 0.0
    assert scores[2] == pytest.approx(0.0)


def test_run_json_document_score_adds_breakdown_when_requested(parquet_dir: Path) -> None:
    run_json_document_score(
        parquet_dir,
        prediction_column="prediction",
        reference_column="reference",
        schema=None,
        default_comparator="string",
        score_column="doc_score",
        breakdown_column="doc_score_breakdown",
    )

    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "doc_score_breakdown" in df.columns
    breakdown_0 = json.loads(df["doc_score_breakdown"].iloc[0])
    assert breakdown_0["score"] == pytest.approx(1.0)
    assert "children" in breakdown_0


def test_run_json_document_score_missing_column_raises(parquet_dir: Path) -> None:
    with pytest.raises(ValueError, match="prediction_column 'missing' not in dataset"):
        run_json_document_score(
            parquet_dir,
            prediction_column="missing",
            reference_column="reference",
            schema=None,
            default_comparator="string",
            score_column="doc_score",
            breakdown_column=None,
        )


def test_studio_processor_types_includes_json_document_score() -> None:
    assert "json_document_score" in STUDIO_PROCESSOR_TYPES


def test_apply_studio_post_processors_dispatches_by_type(parquet_dir: Path) -> None:
    base_dataset_path = parquet_dir.parent
    apply_studio_post_processors(
        base_dataset_path=base_dataset_path,
        processors=[
            {
                "processor_type": "json_document_score",
                "name": "score",
                "prediction_column": "prediction",
                "reference_column": "reference",
                "schema": None,
                "default_comparator": "string",
                "score_column": "doc_score",
                "breakdown_column": None,
            }
        ],
    )
    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "doc_score" in df.columns


def test_run_json_document_score_handles_list_column(tmp_path: Path) -> None:
    """A list-typed parquet column round-trips through pyarrow as numpy.ndarray.
    Make sure that's treated as a list, not as 'unparseable'."""
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "prediction": [[1, 2, 3], [1, 2, 4]],
            "reference": [[1, 2, 3], [1, 2, 3]],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index=False)

    run_json_document_score(
        parquet_dir,
        prediction_column="prediction",
        reference_column="reference",
        schema=None,
        default_comparator="categorical",
        score_column="doc_score",
        breakdown_column=None,
    )
    out = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    # Row 0: exact list match -> 1.0; Row 1: last element differs -> 2/3
    assert out["doc_score"].iloc[0] == pytest.approx(1.0)
    assert out["doc_score"].iloc[1] == pytest.approx(2 / 3)


def test_run_json_document_score_uses_schema_field_comparators(tmp_path: Path) -> None:
    """Verify the schema arg is threaded through to normalize_schema and
    drives per-field comparator selection."""
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "prediction": [json.dumps({"name": "ALICE", "id": "42"})],
            "reference": [{"name": "alice", "id": "42"}],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index=False)

    # `categorical` with `case_insensitive=True` on `name` → "ALICE" == "alice";
    # plain `categorical` on `id` → "42" == "42". There is no `case_insensitive`
    # comparator in the registry — it's a parameter on `categorical_comparator`.
    schema = {
        "name": {"type": "categorical", "case_insensitive": True},
        "id": "categorical",
    }
    run_json_document_score(
        parquet_dir,
        prediction_column="prediction",
        reference_column="reference",
        schema=schema,
        default_comparator="categorical",
        score_column="doc_score",
        breakdown_column=None,
    )
    out = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert out["doc_score"].iloc[0] == pytest.approx(1.0)


def test_apply_studio_post_processors_ignores_non_studio_types(parquet_dir: Path) -> None:
    # schema_transform is owned by data_designer — we should not touch it.
    apply_studio_post_processors(
        base_dataset_path=parquet_dir.parent,
        processors=[
            {"processor_type": "schema_transform", "name": "x", "template": {}},
        ],
    )
    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert list(df.columns) == ["prediction", "reference"]

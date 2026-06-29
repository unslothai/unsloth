# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.data_recipe.evaluations import score_dataframe, score_parquet_dir


def _doc_score_eval(**overrides) -> dict:
    base = {
        "processor_type": "json_document_score",
        "name": "score",
        "prediction_column": "prediction",
        "reference_column": "reference",
        "schema": None,
        "default_comparator": "string",
        "score_column": "doc_score",
        "breakdown_column": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
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
    df.to_parquet(parquet_dir / "batch_00000.parquet", index = False)
    return parquet_dir


def test_score_parquet_dir_adds_score_column(parquet_dir: Path) -> None:
    # Exact-match "categorical" comparator so row-1 (name mismatch, age match)
    # scores cleanly at 0.5; the default "string" comparator is fuzzy and would
    # give 0.8 for "Bob" vs "Bobby".
    score_parquet_dir(parquet_dir, [_doc_score_eval(default_comparator = "categorical")])

    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "doc_score" in df.columns
    scores = df["doc_score"].tolist()
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.0)


def test_score_parquet_dir_adds_breakdown_when_requested(parquet_dir: Path) -> None:
    score_parquet_dir(parquet_dir, [_doc_score_eval(breakdown_column = "doc_score_breakdown")])

    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "doc_score_breakdown" in df.columns
    breakdown_0 = json.loads(df["doc_score_breakdown"].iloc[0])
    assert breakdown_0["score"] == pytest.approx(1.0)
    assert "children" in breakdown_0


def test_score_parquet_dir_missing_column_raises(parquet_dir: Path) -> None:
    with pytest.raises(ValueError, match = "prediction_column 'missing' not in dataset"):
        score_parquet_dir(parquet_dir, [_doc_score_eval(prediction_column = "missing")])


def test_score_parquet_dir_handles_list_column(tmp_path: Path) -> None:
    """A list-typed parquet column round-trips through pyarrow as numpy.ndarray.
    Make sure that's treated as a list, not as 'unparseable'."""
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    df = pd.DataFrame(
        {
            "prediction": [[1, 2, 3], [1, 2, 4]],
            "reference": [[1, 2, 3], [1, 2, 3]],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index = False)

    score_parquet_dir(parquet_dir, [_doc_score_eval(default_comparator = "categorical")])
    out = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert out["doc_score"].iloc[0] == pytest.approx(1.0)
    assert out["doc_score"].iloc[1] == pytest.approx(2 / 3)


def test_score_parquet_dir_uses_schema_field_comparators(tmp_path: Path) -> None:
    """Verify the schema arg drives per-field comparator selection."""
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    df = pd.DataFrame(
        {
            "prediction": [json.dumps({"name": "ALICE", "id": "42"})],
            "reference": [{"name": "alice", "id": "42"}],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index = False)

    schema = {
        "name": {"type": "categorical", "case_insensitive": True},
        "id": "categorical",
    }
    score_parquet_dir(
        parquet_dir,
        [_doc_score_eval(default_comparator = "categorical", schema = schema)],
    )
    out = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert out["doc_score"].iloc[0] == pytest.approx(1.0)


def test_score_dataframe_ignores_unknown_processor_type(parquet_dir: Path) -> None:
    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    score_dataframe(df, [{"processor_type": "unknown_score", "name": "x"}])
    assert list(df.columns) == ["prediction", "reference"]


def test_score_parquet_dir_is_noop_when_no_evaluations(parquet_dir: Path) -> None:
    score_parquet_dir(parquet_dir, [])
    df = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert list(df.columns) == ["prediction", "reference"]


def test_score_parquet_dir_runs_in_declaration_order(tmp_path: Path) -> None:
    """Two evaluations on the same parquet — both columns should be present."""
    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    df = pd.DataFrame(
        {
            "prediction": [json.dumps({"a": 1})],
            "reference": [{"a": 1}],
        }
    )
    df.to_parquet(parquet_dir / "batch_00000.parquet", index = False)

    score_parquet_dir(
        parquet_dir,
        [
            _doc_score_eval(name = "first", score_column = "score_a"),
            _doc_score_eval(name = "second", score_column = "score_b"),
        ],
    )
    out = pd.read_parquet(parquet_dir / "batch_00000.parquet")
    assert "score_a" in out.columns
    assert "score_b" in out.columns


def test_score_dataframe_adds_score_column() -> None:
    df = pd.DataFrame(
        {
            "prediction": [json.dumps({"a": 1})],
            "reference": [{"a": 1}],
        }
    )
    score_dataframe(df, [_doc_score_eval(default_comparator = "categorical")])
    assert "doc_score" in df.columns
    assert df["doc_score"].iloc[0] == pytest.approx(1.0)


def test_build_config_builder_strips_studio_processor_types(monkeypatch):
    """Studio-owned processor types (json_document_score) share the
    recipe.processors[] array with data_designer's native types but
    data_designer's ProcessorConfigT union doesn't know about them — they
    must be stripped before reaching DataDesignerConfigBuilder.from_config."""
    from core.data_recipe import service

    captured_config: dict = {}

    class _StubBuilder:
        def add_processor(self, *, processor_type, **kwargs):
            pass

    class _StubBuilderFactory:
        @staticmethod
        def from_config(cfg):
            captured_config.update(cfg)
            return _StubBuilder()

    class _StubProcessorType:
        SCHEMA_TRANSFORM = type("E", (), {"value": "schema_transform"})()

        def __new__(cls, value):
            if value == "schema_transform":
                return cls.SCHEMA_TRANSFORM
            raise ValueError(f"{value!r} is not a valid ProcessorType")

    monkeypatch.setattr(service, "_apply_data_designer_image_context_patch", lambda: None)
    monkeypatch.setattr(
        service,
        "split_oxc_local_callable_validators",
        lambda recipe_core: (recipe_core, []),
    )
    monkeypatch.setattr(
        service,
        "register_oxc_local_callable_validators",
        lambda **_: None,
    )

    import sys
    import types

    fake_config = types.ModuleType("data_designer.config")
    fake_config.DataDesignerConfigBuilder = _StubBuilderFactory
    fake_processors = types.ModuleType("data_designer.config.processors")
    fake_processors.ProcessorType = _StubProcessorType
    monkeypatch.setitem(sys.modules, "data_designer.config", fake_config)
    monkeypatch.setitem(sys.modules, "data_designer.config.processors", fake_processors)

    service.build_config_builder(
        {
            "processors": [
                {"processor_type": "schema_transform", "name": "a", "template": {}},
                {"processor_type": "json_document_score", "name": "b"},
            ],
        }
    )
    assert captured_config["data_designer"]["processors"] == [
        {"processor_type": "schema_transform", "name": "a", "template": {}},
    ]

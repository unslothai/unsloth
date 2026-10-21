# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest


def _preview(tmp_path: Path, seed: Path):
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        "seed_config": {"source": {"seed_type": "local", "path": str(seed)}},
        "columns": [{"column_type": "expression", "name": "copy", "expr": "{{ text }}"}],
    }
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    return designer.preview(build_config_builder(recipe), num_records = 1).dataset


def test_seed_file_name_cannot_close_the_duckdb_literal(tmp_path: Path) -> None:
    # The engine reads a seed as FROM '<path>': this name joined any host file onto a.csv.
    pytest.importorskip("data_designer")
    (tmp_path / "secret.txt").write_text("secret", encoding = "utf-8")
    (tmp_path / "a.csv").write_text("text\nhello\n", encoding = "utf-8")
    target = str(tmp_path / "secret.txt").replace("/", "@")
    seed = tmp_path / (
        f"a.csv', (SELECT content AS stolen FROM read_text(replace('{target}', '@', chr(47)))) t -- .csv"
    )
    seed.write_text("text\nnamed\n", encoding = "utf-8")

    dataset = _preview(tmp_path, seed)

    assert sorted(dataset.columns) == ["copy", "text"]
    assert list(dataset["copy"]) == ["named"]


def test_seed_file_name_with_an_apostrophe_is_readable(tmp_path: Path) -> None:
    pytest.importorskip("data_designer")
    seed = tmp_path / "bob's rows.csv"
    seed.write_text("text\nhello\n", encoding = "utf-8")

    assert list(_preview(tmp_path, seed)["copy"]) == ["hello"]


def test_hub_seed_path_cannot_close_the_duckdb_literal() -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import _apply_data_designer_seed_uri_quote_patch
    from data_designer.config.seed_source import HuggingFaceSeedSource
    from data_designer.engine.resources.seed_reader import HuggingFaceSeedReader

    _apply_data_designer_seed_uri_quote_patch()
    reader = HuggingFaceSeedReader()
    reader.attach(HuggingFaceSeedSource(path = "datasets/a/b.parquet'; SELECT 1 -- "), None)

    assert reader.get_dataset_uri() == "hf://datasets/a/b.parquet''; SELECT 1 -- "

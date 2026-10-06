# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from types import SimpleNamespace

import pytest


def _recipe(seed_file: Path, *, keep_upper: bool) -> dict:
    return {
        "model_providers": [
            {"name": "p", "endpoint": "http://127.0.0.1:9", "provider_type": "openai"}
        ],
        "model_configs": [
            {"alias": "m", "model": "x", "provider": "p", "inference_parameters": {}}
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed_file)},
            "sampling_strategy": "ordered",
            "selection_strategy": {"start": 0, "end": 1},
        },
        "columns": [
            {
                "column_type": "llm-text",
                "name": "draft",
                "drop": True,
                "model_alias": "m",
                "prompt": "rewrite {{ output }}",
            },
            {
                "column_type": "expression",
                "name": "upper",
                "drop": not keep_upper,
                "expr": "{{ draft }}",
            },
        ],
    }


@pytest.fixture(name = "seed_file")
def fixture_seed_file(tmp_path: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"instruction": ["a"], "output": ["c"]}), path)
    return path


def _validate(recipe: dict):
    pytest.importorskip("fastapi")
    pytest.importorskip("data_designer")
    from routes.data_recipe import validate as validate_module

    return validate_module.validate(SimpleNamespace(recipe = recipe), via_api_key = False)


def test_all_blocks_dropped_names_the_ui_toggle_and_blocks(seed_file: Path) -> None:
    response = _validate(_recipe(seed_file, keep_upper = False))

    assert response.valid is False
    [error] = [e for e in response.errors if e.code == "all_columns_dropped"]
    assert '"Keep out of final dataset" (draft, upper)' in error.message
    assert "drop=False" not in error.message


def test_one_block_kept_is_not_all_dropped(seed_file: Path) -> None:
    response = _validate(_recipe(seed_file, keep_upper = True))

    assert all(e.code != "all_columns_dropped" for e in response.errors)


def test_message_skips_internal_row_id() -> None:
    pytest.importorskip("fastapi")
    from routes.data_recipe.validate import _all_blocks_dropped_message

    columns = [
        SimpleNamespace(name = "_internal_row_id", drop = True),
        SimpleNamespace(name = "draft", drop = True),
    ]
    assert "(draft)." in _all_blocks_dropped_message(columns)

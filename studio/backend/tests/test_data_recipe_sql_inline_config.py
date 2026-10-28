# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest


def _check(tmp_path: Path, sql: str) -> dict:
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        "columns": [
            {"column_type": "expression", "name": "query", "expr": sql},
            {
                "column_type": "validation",
                "name": "check",
                "validator_type": "code",
                "validator_params": {"code_lang": "sql:ansi"},
                "target_columns": ["query"],
            },
        ],
    }
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    return designer.preview(build_config_builder(recipe), num_records = 1).dataset["check"][0]


@pytest.mark.parametrize("comment", ["-- sqlfluff", "--sqlfluff"])
def test_sql_cell_cannot_name_a_folder_to_import(tmp_path: Path, comment: str) -> None:
    # sqlfluff reads "-- sqlfluff:<key>:<value>" lines of the linted text as its own config, and
    # templater:jinja:library_path imports every module under that folder in the recipe worker.
    pytest.importorskip("data_designer")
    library = tmp_path / "library"
    library.mkdir()
    imported = tmp_path / "imported.txt"
    (library / "helper_macros.py").write_text(
        f"open({str(imported)!r}, 'w').close()\n", encoding = "utf-8"
    )

    result = _check(tmp_path, f"{comment}:templater:jinja:library_path:{library}\nSELECT 1")

    assert result["is_valid"]
    assert not imported.exists()


def test_sql_cell_cannot_pick_its_own_dialect(tmp_path: Path) -> None:
    pytest.importorskip("data_designer")
    tsql_only = "SELECT [a] FROM [t]"

    assert not _check(tmp_path, tsql_only)["is_valid"]
    assert not _check(tmp_path, f"-- sqlfluff:dialect:tsql\n{tsql_only}")["is_valid"]

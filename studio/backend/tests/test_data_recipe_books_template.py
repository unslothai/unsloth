# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/features/data-recipes/learning-recipes/books-and-documents.json"
)


def test_books_template_turns_extracted_text_into_text_rows_without_a_model(tmp_path):
    pytest.importorskip("data_designer")
    pytest.importorskip("data_designer_unstructured_seed")
    from core.data_recipe.service import build_config_builder, create_data_designer

    paragraph = "It was a bright cold day in April, and the clocks were striking thirteen. "
    extracted = tmp_path / "book.extracted.txt"
    extracted.write_text("\n\n".join(paragraph * 20 for _ in range(10)), encoding = "utf-8")
    recipe = json.loads(_TEMPLATE.read_text(encoding = "utf-8"))["recipe"]
    assert recipe["model_providers"] == [] and recipe["model_configs"] == []
    source = recipe["seed_config"]["source"]
    source.pop("path")
    source["paths"] = [str(extracted)]

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    rows = designer.preview(build_config_builder(recipe), num_records = 2).dataset

    assert list(rows.columns) == ["source_file", "text"]
    assert len(rows) == 2
    assert rows["text"][0].startswith("It was a bright cold day")
    assert all(0 < len(text) <= source["chunk_size"] for text in rows["text"])

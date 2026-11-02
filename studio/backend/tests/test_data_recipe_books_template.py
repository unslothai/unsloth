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


def test_books_template_rows_start_on_a_word_boundary(tmp_path):
    pytest.importorskip("data_designer")
    pytest.importorskip("data_designer_unstructured_seed")
    from core.data_recipe.service import build_config_builder, create_data_designer

    paragraphs = (" ".join(f"word{p}x{w}" + "o" * (w % 5) for w in range(150)) for p in range(30))
    book = "\n\n".join(paragraphs)
    extracted = tmp_path / "book.extracted.txt"
    extracted.write_text(book, encoding = "utf-8")
    recipe = json.loads(_TEMPLATE.read_text(encoding = "utf-8"))["recipe"]
    source = recipe["seed_config"]["source"]
    assert source["chunk_overlap"] > 0
    source.pop("path")
    source["paths"] = [str(extracted)]

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    rows = designer.preview(build_config_builder(recipe), num_records = 4).dataset

    assert len(rows) == 4
    for text in rows["text"]:
        start = book.index(text)
        assert start == 0 or book[start - 1].isspace(), text[:30]


def test_overlap_is_kept_when_text_has_no_spaces():
    pytest.importorskip("data_designer")
    chunking = pytest.importorskip("data_designer_unstructured_seed.chunking")

    text = "".join(chr(0x4E00 + (i * 7919) % 20000) for i in range(20000))
    chunks = chunking.split_text_into_chunks(text = text, chunk_size = 6000, chunk_overlap = 600)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.endswith(current[:600])


def test_seed_cache_is_rebuilt_when_the_chunker_changes(tmp_path, monkeypatch):
    pytest.importorskip("data_designer")
    chunking = pytest.importorskip("data_designer_unstructured_seed.chunking")

    monkeypatch.setattr(chunking, "_CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "book.txt"
    source.write_text("alpha beta gamma " * 50, encoding = "utf-8")
    kwargs = dict(file_entries = [(source, "book.txt")], chunk_size = 300, chunk_overlap = 50)

    monkeypatch.setattr(chunking, "_CHUNKER_VERSION", chunking._CHUNKER_VERSION - 1)
    old_path, _ = chunking.materialize_multi_file_unstructured_seed(**kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(chunking, "_CACHE_DIR", tmp_path / "cache")
    new_path, _ = chunking.materialize_multi_file_unstructured_seed(**kwargs)

    assert new_path != old_path

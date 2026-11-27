# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A seed that names one uploaded file many times reads and chunks it once."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))
_PKG = (
    _BACKEND
    / "plugins"
    / "data-designer-unstructured-seed"
    / "src"
    / "data_designer_unstructured_seed"
)


def _load_chunking():
    spec = importlib.util.spec_from_file_location("_uss_chunking_repeat", _PKG / "chunking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chunking = _load_chunking()


def test_a_file_named_many_times_is_chunked_once(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "_CACHE_DIR", tmp_path / "cache")
    doc = tmp_path / "f.extracted.txt"
    doc.write_text(
        "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(40)), encoding = "utf-8"
    )
    other = tmp_path / "g.extracted.txt"
    other.write_text("Another document. " * 40, encoding = "utf-8")

    once, total_once = chunking.build_multi_file_preview(
        file_entries = [(doc, "doc.txt"), (other, "other.txt")],
        preview_size = 5,
        chunk_size = 400,
        chunk_overlap = 0,
    )
    reads = []
    real = chunking.load_unstructured_text_file
    monkeypatch.setattr(
        chunking, "load_unstructured_text_file", lambda path: reads.append(path) or real(path)
    )
    monkeypatch.setattr(chunking, "_CACHE_DIR", tmp_path / "cache2")
    repeated, total_repeated = chunking.build_multi_file_preview(
        file_entries = [(doc, f"doc{i}.txt") for i in range(200)] + [(other, "other.txt")],
        preview_size = 5,
        chunk_size = 400,
        chunk_overlap = 0,
    )
    assert total_repeated == total_once
    assert len(reads) == 2, f"read {len(reads)} times for 2 distinct files"
    assert [row["chunk_text"] for row in repeated] == [row["chunk_text"] for row in once]

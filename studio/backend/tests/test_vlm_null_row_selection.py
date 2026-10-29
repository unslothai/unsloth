# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VLM conversion finds rows without an image or text from Arrow validity, with the same rows as walking the column."""

import io
import sys
from pathlib import Path

import pytest
from datasets import Dataset, concatenate_datasets
from datasets import Image as ImageFeature
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets import format_conversion  # noqa: E402
from utils.datasets.format_conversion import convert_to_vlm_format  # noqa: E402


def _png(width):
    out = io.BytesIO()
    Image.new("RGB", (width, 1)).save(out, format = "PNG")
    return out.getvalue()


def _reference(dataset, column):
    return [i for i, value in enumerate(dataset[column]) if value is not None]


def _cases():
    text = ["a", None, "", "b", None, " ", "c"]
    base = Dataset.from_dict({"text": text, "n": list(range(len(text)))})
    yield "strings", base, "text"
    yield "no nulls", Dataset.from_dict({"text": ["x", "", "y"]}), "text"
    yield "all null", Dataset.from_dict({"text": [None, None]}), "text"
    yield "empty", base.select([]), "text"
    yield "lists", Dataset.from_dict({"text": [["a"], None, [], [None], ["b", "c"]]}), "text"
    yield "structs", Dataset.from_dict({"text": [{"a": None}, None, {"a": "x"}]}), "text"
    yield "selected", base.select([6, 4, 1, 0, 2]), "text"
    yield "shuffled", base.shuffle(seed = 3), "text"
    yield "filtered", base.filter(lambda r: r["n"] % 2 == 0), "text"
    yield "chunks", concatenate_datasets([base, base.select([1, 0]), base]), "text"
    images = Dataset.from_dict(
        {"image": [{"bytes": _png(1), "path": None}, None, {"bytes": _png(3), "path": None}]}
    )
    yield "image feature", images.cast_column("image", ImageFeature()), "image"


@pytest.mark.parametrize("name,dataset,column", list(_cases()), ids = [c[0] for c in _cases()])
def test_non_null_rows_match_the_column_walk(name, dataset, column):
    assert list(format_conversion._non_null_rows(dataset, column)) == _reference(dataset, column)


def test_image_cells_are_not_decoded(monkeypatch):
    ds = Dataset.from_dict({"image": [{"bytes": _png(2), "path": None}, None]}).cast_column(
        "image", ImageFeature()
    )
    decoded = []
    real = ImageFeature.decode_example
    monkeypatch.setattr(
        ImageFeature,
        "decode_example",
        lambda self, *a, **k: decoded.append(1) or real(self, *a, **k),
    )

    assert list(format_conversion._non_null_rows(ds, "image")) == [0]
    assert decoded == []


def test_conversion_does_not_walk_the_columns(monkeypatch):
    ds = Dataset.from_dict(
        {
            "image": [f"https://example.com/{i}.png" if i != 2 else None for i in range(6)],
            "caption": [f"c{i}" if i != 4 else None for i in range(6)],
        }
    )
    walked = []
    real_getitem = Dataset.__getitem__

    def getitem(self, key):
        if isinstance(key, str) and self.format["type"] != "arrow":
            walked.append(key)
        return real_getitem(self, key)

    monkeypatch.setattr(Dataset, "__getitem__", getitem)
    monkeypatch.setattr(
        format_conversion, "_fetch_row_image", lambda url: Image.new("RGB", (int(url[-5]) + 1, 1))
    )
    logs = []
    monkeypatch.setattr(
        format_conversion.logger, "info", lambda msg, *a, **k: logs.append(str(msg))
    )

    out = convert_to_vlm_format(ds, instruction = "Describe.", text_column = "caption")

    assert walked == []
    assert [m["messages"][1]["content"][0]["text"] for m in out] == ["c0", "c1", "c3", "c5"]
    assert "Skipping 1/6 rows without an image" in logs
    assert "Skipping 1/5 rows without text" in logs


def test_objects_without_an_arrow_view_walk_the_column():
    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            return [r[key] for r in self.rows]

    assert format_conversion._non_null_rows(Rows([{"t": "a"}, {"t": None}, {"t": ""}]), "t") == [
        0,
        2,
    ]


@pytest.mark.parametrize("fmt", ["numpy", "pandas"])
def test_formatted_datasets_keep_the_column_walk(fmt):
    dataset = Dataset.from_dict({"text": [1.0, None, float("nan"), 2.0]}).with_format(fmt)

    assert list(format_conversion._non_null_rows(dataset, "text")) == _reference(dataset, "text")

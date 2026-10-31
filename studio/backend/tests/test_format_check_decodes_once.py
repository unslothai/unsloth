# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A format check decodes each preview row's media once, and answers exactly as before."""

from __future__ import annotations

import io

import pytest
from datasets import Dataset, Features, Image, Sequence, Value

from hub.schemas.datasets import CheckFormatRequest
from hub.services.datasets import formatting

ROWS = 10


def _jpeg(seed: int) -> dict:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 48), (seed * 20 % 256, 40, 90)).save(buf, format = "JPEG")
    return {"bytes": buf.getvalue(), "path": None}


def _datasets():
    image = Features({"image": Image(), "caption": Value("string")})
    return {
        "image": Dataset.from_dict(
            {
                "image": [_jpeg(i) for i in range(ROWS)],
                "caption": [f"a photo {i}" for i in range(ROWS)],
            },
            features = image,
        ),
        "image_messages": Dataset.from_dict(
            {
                "images": [[_jpeg(i)] for i in range(ROWS)],
                "messages": [
                    [
                        {"role": "user", "content": "<image>what"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                    for i in range(ROWS)
                ],
            },
            features = Features(
                {
                    "images": Sequence(Image()),
                    "messages": [{"role": Value("string"), "content": Value("string")}],
                }
            ),
        ),
        "sharegpt": Dataset.from_dict(
            {
                "conversations": [
                    [{"from": "human", "value": f"h{i}"}, {"from": "gpt", "value": f"g{i}"}]
                    for i in range(ROWS)
                ]
            }
        ),
        "alpaca": Dataset.from_dict(
            {
                "instruction": [f"q{i}" for i in range(ROWS)],
                "output": [f"a{i}" for i in range(ROWS)],
            }
        ),
        "unmapped": Dataset.from_dict(
            {"foo": [f"x{i}" for i in range(ROWS)], "bar": list(range(ROWS))}
        ),
        "url_image": Dataset.from_dict(
            {
                "image_url": [f"https://example.com/{i}.jpg" for i in range(ROWS)],
                "caption": [f"a photo {i}" for i in range(ROWS)],
            }
        ),
        "empty": Dataset.from_dict({"text": []}, features = Features({"text": Value("string")})),
    }


@pytest.fixture
def decodes(monkeypatch):
    monkeypatch.setattr(formatting, "account_hf_token", lambda token: token)
    monkeypatch.setattr(formatting, "account_path", lambda *a, **k: None)
    monkeypatch.setattr(formatting, "visible_cached_path", lambda *a, **k: None)
    count = {"image": 0}
    decode = Image.decode_example

    def counted(self, *args, **kwargs):
        count["image"] += 1
        return decode(self, *args, **kwargs)

    monkeypatch.setattr(Image, "decode_example", counted)
    return count


def _check(
    monkeypatch,
    dataset,
    *,
    is_vlm = False,
):
    monkeypatch.setattr(formatting, "load_preview_slice", lambda *a, **k: (dataset, len(dataset)))
    return formatting.check_format_response(CheckFormatRequest(dataset_name = "o/d", is_vlm = is_vlm))


@pytest.mark.parametrize("is_vlm", [False, True])
def test_image_preview_decodes_each_row_once(monkeypatch, decodes, is_vlm):
    response = _check(monkeypatch, _datasets()["image"], is_vlm = is_vlm)
    assert decodes["image"] == ROWS
    assert len(response.preview_samples) == ROWS
    assert response.preview_samples[0]["image"]["type"] == "image"


def _response_json(monkeypatch, dataset, is_vlm):
    try:
        return _check(monkeypatch, dataset, is_vlm = is_vlm).model_dump_json()
    except Exception as exc:  # the error is part of the answer
        return (
            f"{type(exc).__name__}: {getattr(exc, 'status_code', '')} {getattr(exc, 'detail', exc)}"
        )


@pytest.mark.parametrize("name", sorted(_datasets()))
@pytest.mark.parametrize("is_vlm", [False, True])
def test_response_matches_reading_the_dataset_directly(monkeypatch, decodes, name, is_vlm):
    dataset = _datasets()[name]
    cached = _response_json(monkeypatch, dataset, is_vlm)
    monkeypatch.setattr(formatting, "_decode_preview_rows_once", lambda preview: preview)
    direct = _response_json(monkeypatch, dataset, is_vlm)
    assert cached == direct


def test_rows_are_fresh_dicts_and_the_rest_is_the_dataset():
    dataset = _datasets()["alpaca"]
    view = formatting._decode_preview_rows_once(dataset)
    first = view[0]
    first["instruction"] = "changed"
    assert view[0] == dataset[0]
    assert next(iter(view)) == dataset[0]
    assert list(view) == [dataset[i] for i in range(ROWS)]
    assert view[-1] == dataset[-1]
    assert view[2:4] == dataset[2:4]
    assert view["instruction"] == dataset["instruction"]
    assert view.column_names == dataset.column_names
    assert view.features == dataset.features
    assert len(view) == len(dataset)
    mapped = view.map(lambda row: {"output": row["output"].upper()})
    assert isinstance(mapped, Dataset) and mapped[0]["output"] == "A0"
    with pytest.raises(IndexError):
        view[ROWS]


def test_a_failed_decode_is_not_cached():
    calls = {"n": 0}

    class Flaky:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("first read fails")
            return {"a": 1}

    view = formatting._DecodedPreviewRows(Flaky())
    with pytest.raises(OSError):
        view[0]
    assert view[0] == {"a": 1} and view[0] == {"a": 1}
    assert calls["n"] == 2

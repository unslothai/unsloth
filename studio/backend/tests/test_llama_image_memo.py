# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""History images sent again on a later turn reuse the earlier normalization instead of decoding the image again."""

import base64
import copy
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from routes import inference as inf


def _b64(img, fmt):
    buf = io.BytesIO()
    img.save(buf, format = fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _payloads():
    opaque_png = _b64(Image.new("RGB", (32, 24), (10, 200, 30)), "PNG")
    jpeg = _b64(Image.new("RGB", (32, 24), (200, 10, 30)), "JPEG")
    transparent = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
    transparent.putpixel((0, 0), (0, 0, 0, 0))
    gif = _b64(Image.new("RGB", (8, 8), (1, 2, 3)), "GIF")
    return [opaque_png, jpeg, _b64(transparent, "PNG"), gif]


@pytest.fixture(autouse = True)
def _fresh_memo(monkeypatch):
    monkeypatch.setattr(inf, "_LLAMA_IMAGE_MEMO", inf.collections.OrderedDict())
    monkeypatch.setattr(inf, "_llama_image_memo_chars", 0)


def _history(payloads):
    return [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + p}}],
        }
        for p in payloads
    ]


def test_a_repeated_history_is_normalized_once_with_the_same_result(monkeypatch):
    payloads = _payloads()
    expected = [inf._llama_image_data_url(base64.b64decode(p)) for p in payloads]
    calls = []
    original = inf._llama_image_data_url

    def spy(raw):
        calls.append(len(raw))
        return original(raw)

    monkeypatch.setattr(inf, "_llama_image_data_url", spy)
    for turn in range(3):
        messages = copy.deepcopy(_history(payloads))
        assert inf._normalize_openai_image_parts_for_llama(messages) is True
        assert [m["content"][0]["image_url"]["url"] for m in messages] == expected
    assert len(calls) == len(payloads)


def test_a_payload_that_fails_is_tried_again_and_still_rejected(monkeypatch):
    calls = []
    original = inf._llama_image_data_url

    def spy(raw):
        calls.append(1)
        return original(raw)

    monkeypatch.setattr(inf, "_llama_image_data_url", spy)
    bad = base64.b64encode(b"not an image").decode("ascii")
    for _ in range(2):
        with pytest.raises(HTTPException) as err:
            inf._normalize_openai_image_parts_for_llama(_history([bad]))
        assert err.value.status_code == 400
    assert len(calls) == 2


def test_the_memo_stays_within_its_budget(monkeypatch):
    payloads = _payloads()
    monkeypatch.setattr(inf, "_LLAMA_IMAGE_MEMO_MAX_CHARS", 200)
    for p in payloads:
        inf._llama_image_data_url_from_b64(p)
        assert inf._llama_image_memo_chars == sum(len(v) for v in inf._LLAMA_IMAGE_MEMO.values())
        assert inf._llama_image_memo_chars <= 200
    for p in payloads:
        assert inf._llama_image_data_url_from_b64(p) == inf._llama_image_data_url(
            base64.b64decode(p)
        )


def test_the_memo_keeps_a_bounded_number_of_entries(monkeypatch):
    monkeypatch.setattr(inf, "_LLAMA_IMAGE_MEMO_MAX_ENTRIES", 3)
    payloads = [_b64(Image.new("RGB", (4, 4), (i, 0, 0)), "PNG") for i in range(10)]
    for p in payloads:
        inf._llama_image_data_url_from_b64(p)
    assert len(inf._LLAMA_IMAGE_MEMO) == 3
    assert inf._llama_image_memo_chars == sum(len(v) for v in inf._LLAMA_IMAGE_MEMO.values())

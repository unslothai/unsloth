# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/rag/extract-text ``boxes``: Tesseract's text blocks of an image, opt-in."""

import io
import os
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from routes import rag as rag_routes

_BLOCKS = [{"text": "Hello world", "lines": 1, "box": [0.1, 0.2, 0.5, 0.3]}]


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file = io.BytesIO(data), filename = name)


def _png(size = (40, 30)) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def tesseract(monkeypatch):
    from core.rag import captioner, pdf_ocr

    calls = []
    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: {})
    monkeypatch.setattr(
        pdf_ocr, "ocr_pages", lambda path, pages: calls.append("text") or {1: "Hello world"}
    )
    monkeypatch.setattr(
        pdf_ocr,
        "ocr_image_blocks",
        lambda path: calls.append("blocks") or ("Hello world", list(_BLOCKS)),
    )
    return calls


def test_boxes_return_tesseract_blocks(rag_home, tesseract):
    result = rag_routes.extract_text(_upload("sign.png", _png()), boxes = True, subject = "owner")
    assert result == {
        "filename": "sign.png",
        "text": "Hello world",
        "truncated": False,
        "blocks": _BLOCKS,
    }
    assert tesseract == ["blocks"]


def test_default_response_is_unchanged(rag_home, tesseract):
    result = rag_routes.extract_text(_upload("sign.png", _png()), subject = "owner")
    assert result == {"filename": "sign.png", "text": "Hello world", "truncated": False}
    assert tesseract == ["text"]


def test_a_vision_model_read_has_no_blocks(rag_home, tesseract, monkeypatch):
    from core.rag import captioner

    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: {1: "Hola"})
    result = rag_routes.extract_text(_upload("sign.png", _png()), boxes = True, subject = "owner")
    assert result == {"filename": "sign.png", "text": "Hola", "truncated": False}
    assert tesseract == []


class _FakePage:
    rect = SimpleNamespace(width = 200.0, height = 100.0)

    def get_textpage_ocr(self, **kwargs):
        return "textpage"

    def get_text(
        self,
        kind,
        textpage = None,
    ):
        assert textpage == "textpage"
        if kind == "text":
            return "Hello  world\nsecond line\n"
        span = lambda text: {"text": text}  # noqa: E731
        return {
            "blocks": [
                {
                    "bbox": (20, 10, 120, 40),
                    "lines": [
                        {"spans": [span("Hello "), span(" world")]},
                        {"spans": [span("second"), span(" line")]},
                    ],
                },
                {"bbox": (0, 0, 10, 10), "lines": [{"spans": [span("  ")]}]},
                {"bbox": (150, 90, 230, 110), "lines": [{"spans": [span("edge")]}]},
            ]
        }


class _FakeDoc:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, index):
        return _FakePage()


def test_blocks_are_lines_joined_with_boxes_as_page_fractions(monkeypatch):
    import pymupdf

    from core.rag import pdf_ocr

    monkeypatch.setattr(pymupdf, "open", lambda path: _FakeDoc())
    text, blocks = pdf_ocr.ocr_image_blocks("image.png")
    assert text == "Hello  world\nsecond line"
    assert blocks == [
        {"text": "Hello world second line", "lines": 2, "box": [0.1, 0.1, 0.6, 0.4]},
        {"text": "edge", "lines": 1, "box": [0.75, 0.9, 1.0, 1.0]},
    ]


def test_an_ocr_failure_gives_no_text_and_no_blocks(monkeypatch):
    import pymupdf

    from core.rag import pdf_ocr

    def broken(path):
        raise RuntimeError("no tessdata")

    monkeypatch.setattr(pymupdf, "open", broken)
    assert pdf_ocr.ocr_image_blocks("image.png") == ("", [])


def _tessdata() -> str | None:
    prefix = os.environ.get("TESSDATA_PREFIX")
    return prefix if prefix and os.path.isfile(os.path.join(prefix, "eng.traineddata")) else None


@pytest.mark.skipif(_tessdata() is None, reason = "needs Tesseract eng.traineddata")
def test_real_tesseract_finds_the_text_blocks(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    from core.rag import pdf_ocr

    image = Image.new("RGB", (800, 400), (30, 90, 160))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size = 36)
    draw.rectangle([40, 40, 500, 160], fill = (250, 240, 200))
    draw.text((60, 60), "Hello world", fill = "black", font = font)
    draw.text((420, 280), "Exit here", fill = "white", font = font)
    path = tmp_path / "sign.png"
    image.save(path)
    text, blocks = pdf_ocr.ocr_image_blocks(str(path))
    assert text == pdf_ocr.ocr_pages(str(path), [1])[1]
    assert [block["text"] for block in blocks] == ["Hello world", "Exit here"]
    first, second = (block["box"] for block in blocks)
    assert 0.05 < first[0] < 0.1 and 0.12 < first[1] < 0.2
    assert 0.5 < second[0] < 0.55 and 0.68 < second[1] < 0.75

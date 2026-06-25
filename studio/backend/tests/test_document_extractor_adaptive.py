# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adaptive fast-path extraction tests.

Covers the behavior that makes document extraction fast-by-default:
born-digital PDFs render no page images (and issue no VLM calls), while
scanned/image-only pages are detected and rendered for VLM OCR. Also checks
that rendered pages use the transcription prompt rather than the figure-caption
prompt.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import pymupdf  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from core.chat import document_extractor as dx  # noqa: E402
from core.chat.vlm_capability import VlmCapability  # noqa: E402


def _born_digital_pdf(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    try:
        for index in range(pages):
            page = doc.new_page(width = 612, height = 792)
            rect = pymupdf.Rect(72, 72, 540, 720)
            text = f"Page {index + 1} Heading\n\n" + (
                "Born digital paragraph text that wraps across the page. " * 8
            )
            page.insert_textbox(rect, text, fontsize = 11)
        return doc.tobytes()
    finally:
        doc.close()


def _scanned_pdf(pages: int = 1) -> bytes:
    """Pages that are a single full-page raster with no text layer."""
    raster = PILImage.new("RGB", (850, 1100), (210, 210, 210))
    buf = io.BytesIO()
    raster.save(buf, format = "PNG")
    png = buf.getvalue()
    doc = pymupdf.open()
    try:
        for _ in range(pages):
            page = doc.new_page(width = 612, height = 792)
            page.insert_image(page.rect, stream = png)
        return doc.tobytes()
    finally:
        doc.close()


def _page_kinds(figures) -> list[str]:
    return [fig.kind for fig in figures]


def test_page_is_scanned_distinguishes_text_from_image():
    born = pymupdf.open(stream = _born_digital_pdf(), filetype = "pdf")
    scanned = pymupdf.open(stream = _scanned_pdf(), filetype = "pdf")
    try:
        assert dx._page_is_scanned(born[0]) is False
        assert dx._page_is_scanned(scanned[0]) is True
    finally:
        born.close()
        scanned.close()


def test_born_digital_renders_no_page_images():
    markdown, figures, page_count, _truncated, _seen = dx._extract_pdf(
        _born_digital_pdf(pages = 3),
        max_figures = 10,
        use_vlm_ocr = False,
        max_visual_payloads = 3,
    )
    assert page_count == 3
    assert markdown.strip()  # text layer extracted
    # No scanned pages -> no full-page renders at all.
    assert "page" not in _page_kinds(figures)


def test_scanned_page_is_rendered_in_default_mode():
    _markdown, figures, page_count, _truncated, _seen = dx._extract_pdf(
        _scanned_pdf(pages = 2),
        max_figures = 10,
        use_vlm_ocr = False,
        max_visual_payloads = 3,
    )
    assert page_count == 2
    page_figures = [fig for fig in figures if fig.kind == "page"]
    figure_figures = [fig for fig in figures if fig.kind == "figure"]
    assert len(page_figures) == 2
    assert page_figures[0].image_base64  # rendered + encoded for the VLM
    # A scanned page's full-page raster must not be re-extracted as a duplicate
    # kind="figure" by the embedded-image loop.
    assert figure_figures == []


def test_use_vlm_ocr_renders_every_page():
    _markdown, figures, page_count, _truncated, _seen = dx._extract_pdf(
        _born_digital_pdf(pages = 3),
        max_figures = 10,
        use_vlm_ocr = True,
        max_visual_payloads = 3,
    )
    page_figures = [fig for fig in figures if fig.kind == "page"]
    assert page_count == 3
    assert len(page_figures) == 3  # forced full-page render despite text layer


def test_rendered_pages_use_transcription_prompt(monkeypatch):
    """A kind='page' figure must be captioned with the OCR transcription prompt
    and a larger token budget, not the short figure-description prompt."""
    captured: dict[str, object] = {}

    async def _fake_describe(*, prompt, max_tokens, **_kwargs):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return "transcribed text", None

    def _fake_extract_sync(
        file_bytes,
        filename,
        options,
        content_type = "",
    ):
        figure = dx.ExtractedFigure(
            id = "page-1",
            page = 1,
            caption = None,
            kind = "page",
            image_mime = "image/jpeg",
            image_base64 = "QUJD",  # opaque to the stubbed describe call
        )
        return "", [figure], 1, 0, 0

    monkeypatch.setattr(dx, "_describe_image_via_vlm", _fake_describe)
    monkeypatch.setattr(dx, "_run_extract_sync", _fake_extract_sync)

    cap = VlmCapability(
        is_vlm = True,
        endpoint_url = "http://127.0.0.1:9/",
        model_name = "vision-model",
        source = "gguf",
    )

    result = asyncio.run(
        dx.extract_document(
            b"%PDF-1.4 fake",
            "scan.pdf",
            describe_images = True,
            use_vlm_ocr = True,
            max_figures = 5,
            max_visual_payloads = 3,
            capability = cap,
        )
    )

    assert captured["prompt"] == dx._OCR_PAGE_PROMPT
    assert captured["max_tokens"] == 1024
    assert result.figures[0].caption == "transcribed text"


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/rag/extract-text max_chars: scanned pages past it are not OCR'd; without it every page is."""

import io

import pytest
from fastapi import UploadFile

from routes import rag as rag_routes


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file = io.BytesIO(data), filename = name)


def _scan(doc):
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pix.clear_with(200)
    page = doc.new_page()
    page.insert_image(page.rect, pixmap = pix)


def _pdf(
    typed: int,
    scanned: int,
    line: str = "Typed words on page {n}.",
) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for n in range(1, typed + 1):
        doc.new_page().insert_text((72, 100), line.format(n = n))
    for _ in range(scanned):
        _scan(doc)
    return doc.tobytes()


def _vision(monkeypatch, seen: list):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))

    def ocr(pngs):
        seen.extend(sorted(pngs))
        return {n: f"Scanned words of page {n}." for n in pngs}

    monkeypatch.setattr(captioner, "ocr_pages", ocr)
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda *a, **k: pytest.fail("Tesseract ran"))


def _tesseract(monkeypatch, seen: list):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)

    def ocr(
        path,
        pages,
        *,
        until = None,
    ):
        out = {}
        for n in pages:
            if until is not None and until(n, out):
                break
            seen.append(n)
            out[n] = f"Tesseract words of page {n}."
        return out

    monkeypatch.setattr(pdf_ocr, "ocr_pages", ocr)


def _extract(data: bytes, **kwargs) -> dict:
    return rag_routes.extract_text(_upload("long.pdf", data), subject = "owner", **kwargs)


def test_scanned_pages_past_the_kept_text_are_not_ocrd(rag_home, monkeypatch):
    data = _pdf(typed = 6, scanned = 4)
    seen: list = []
    _vision(monkeypatch, seen)
    full = _extract(data)
    assert seen == [7, 8, 9, 10]

    seen.clear()
    capped = _extract(data, max_chars = 60)
    assert seen == []
    assert len(capped["text"]) > 60
    assert capped["text"][:60] == full["text"][:60]
    assert capped["truncated"] is True and "warning" not in capped


@pytest.mark.parametrize("engine", [_vision, _tesseract])
def test_ocr_stops_in_page_order_once_the_text_passes_the_cap(rag_home, monkeypatch, engine):
    data = _pdf(typed = 1, scanned = 6)
    seen: list = []
    engine(monkeypatch, seen)
    full = _extract(data)
    assert seen == [2, 3, 4, 5, 6, 7]

    seen.clear()
    # Page 1 is 22 characters and each scan 24 or 26 more after its separator: past 70 after page 3.
    capped = _extract(data, max_chars = 70)
    assert seen == [2, 3]
    assert len(capped["text"]) > 70
    assert capped["text"] == full["text"][: len(capped["text"])]


def test_a_cap_the_text_never_reaches_changes_nothing(rag_home, monkeypatch):
    data = _pdf(typed = 2, scanned = 2)
    seen: list = []
    _vision(monkeypatch, seen)
    full = _extract(data)
    assert _extract(data, max_chars = 50_000) == full
    assert seen == [3, 4, 3, 4]


def test_unread_scans_inside_the_kept_text_still_warn(rag_home, monkeypatch):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda path, pages, *, until = None: {})
    data = _pdf(typed = 1, scanned = 1)
    result = _extract(data, max_chars = 1000)
    assert "page 2 of long.pdf" in result["warning"]


def test_tesseract_until_stops_before_the_page(tmp_path, monkeypatch):
    import pymupdf

    from core.rag import pdf_ocr

    path = tmp_path / "scan.pdf"
    path.write_bytes(_pdf(typed = 0, scanned = 4))
    asked = []

    def fake_ocr(page, **kwargs):
        asked.append(page.number + 1)
        return page.get_textpage()

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", fake_ocr)
    pdf_ocr.ocr_pages(str(path), [1, 2, 3, 4], until = lambda number, out: number >= 3)
    assert asked == [1, 2]


def test_the_form_field_reaches_the_route_and_is_optional(rag_home, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)
    data = _pdf(typed = 6, scanned = 4)
    seen: list = []
    _vision(monkeypatch, seen)

    def post(**form):
        return client.post(
            "/api/rag/extract-text",
            files = {"file": ("long.pdf", data, "application/pdf")},
            data = form,
        )

    assert post().status_code == 200 and seen == [7, 8, 9, 10]
    seen.clear()
    assert post(max_chars = "60").status_code == 200 and seen == []
    assert post(max_chars = "0").status_code == 422

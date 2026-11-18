# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/rag/extract-text: the composer's reader for documents the browser cannot open."""

import io
import os

import pytest
from fastapi import HTTPException, UploadFile

from routes import rag as rag_routes
from utils.paths import rag_uploads_root

MHT = (
    "MIME-Version: 1.0\r\n"
    'Content-Type: multipart/related; boundary="b"\r\n\r\n'
    "--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
    "<html><body><h1>Quarterly</h1><p>Revenue rose 12%.</p></body></html>\r\n"
    "--b--\r\n"
)


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file = io.BytesIO(data), filename = name)


def _uploads_left() -> list[str]:
    root = rag_uploads_root()
    return os.listdir(root) if os.path.isdir(root) else []


def test_extracts_text_and_keeps_no_copy(rag_home):
    result = rag_routes.extract_text(_upload("report.mht", MHT.encode()), subject = "owner")
    assert result["filename"] == "report.mht"
    assert "Quarterly" in result["text"] and "Revenue rose 12%." in result["text"]
    assert result["truncated"] is False
    assert _uploads_left() == []


def test_works_without_sqlite_vec(rag_home, monkeypatch):
    from storage import rag_db

    monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    result = rag_routes.extract_text(
        _upload("page.xhtml", b"<html><body><p>hello</p></body></html>"), subject = "owner"
    )
    assert result["text"] == "hello"


def test_unsupported_type_is_refused(rag_home):
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload("model.gguf", b"GGUF"), subject = "owner")
    assert exc.value.status_code == 400
    assert _uploads_left() == []


@pytest.mark.parametrize("name", ["broken.doc", "broken.docm", "broken.epub"])
def test_damaged_document_is_422_and_cleaned_up(rag_home, name):
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload(name, b"not a document"), subject = "owner")
    assert exc.value.status_code == 422
    assert name in exc.value.detail
    assert _uploads_left() == []


def test_text_is_capped_on_a_character_boundary(rag_home, monkeypatch):
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 7)
    html = "<html><body><p>ééééé</p></body></html>".encode()
    result = rag_routes.extract_text(_upload("page.xhtml", html), subject = "owner")
    assert result == {"filename": "page.xhtml", "text": "ééé", "truncated": True}


def test_cap_stops_at_the_page_that_crosses_it(rag_home, monkeypatch):
    from core.rag import parsers

    pages = [parsers._page(text, n) for n, text in enumerate(["first", "  ", "second", "third"], 1)]
    monkeypatch.setattr(parsers, "parse", lambda path: pages)
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 12)
    result = rag_routes.extract_text(_upload("deck.ppt", b"x"), subject = "owner")
    assert (result["text"], result["truncated"]) == ("first\n\nsecon", True)
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 1024)
    result = rag_routes.extract_text(_upload("deck.ppt", b"x"), subject = "owner")
    assert (result["text"], result["truncated"]) == ("first\n\nsecond\n\nthird", False)


def _png() -> bytes:
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.clear_with(255)
    return pix.tobytes("png")


def test_an_image_is_read_by_the_vision_model(rag_home, monkeypatch):
    from core.rag import captioner

    seen = []
    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: seen.append(pngs) or {1: "Hola mundo"})
    result = rag_routes.extract_text(_upload("sign.png", _png()), subject = "owner")
    assert result == {"filename": "sign.png", "text": "Hola mundo", "truncated": False}
    assert list(seen[0]) == [1] and seen[0][1].startswith(b"\x89PNG")
    assert _uploads_left() == []


def test_an_image_nothing_can_read_is_422_and_cleaned_up(rag_home, monkeypatch):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: {})
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda path, pages: {})
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload("photo.jpg", _png()), subject = "owner")
    assert exc.value.status_code == 422
    assert "vision model" in exc.value.detail
    assert _uploads_left() == []


def _image(fmt: str, size: tuple[int, int], **save) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, fmt, **save)
    return out.getvalue()


@pytest.mark.parametrize(
    "name, data, expected",
    [
        ("shot.webp", _image("WEBP", (40, 30)), (40, 30)),
        ("photo.jpg", _image("JPEG", (40, 30), dpi = (72, 72)), (40, 30)),
        ("scan.tiff", _image("TIFF", (40, 30), dpi = (600, 600)), (40, 30)),
        ("huge.png", _image("PNG", (4000, 3000)), None),
    ],
    ids = ["webp", "72dpi", "600dpi", "over-budget"],
)
def test_an_image_reaches_the_model_at_its_own_pixels(rag_home, monkeypatch, name, data, expected):
    from PIL import Image

    from core.rag import captioner

    seen = []
    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: seen.append(pngs[1]) or {1: "ok"})
    assert rag_routes.extract_text(_upload(name, data), subject = "owner")["text"] == "ok"
    size = Image.open(io.BytesIO(seen[0])).size
    if expected:
        assert size == expected
    else:
        assert size[0] * size[1] <= rag_routes._IMAGE_MAX_PIXELS and size[0] / size[
            1
        ] == pytest.approx(4 / 3, 0.01)


def test_a_transparent_image_is_read_on_white(rag_home, monkeypatch):
    from PIL import Image

    from core.rag import captioner

    out = io.BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(out, "PNG")
    seen = []
    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: seen.append(pngs[1]) or {1: "ok"})
    rag_routes.extract_text(_upload("logo.png", out.getvalue()), subject = "owner")
    assert Image.open(io.BytesIO(seen[0])).getpixel((0, 0)) == (255, 255, 255)


def test_tesseract_reads_the_image_at_its_own_pixels(rag_home, monkeypatch):
    import pymupdf

    from core.rag import captioner, pdf_ocr

    rendered = []

    def tesseract(path, pages):
        with pymupdf.open(path) as doc:
            pix = doc[0].get_pixmap(dpi = rag_routes.config.OCR_DPI)
        rendered.append((pix.width, pix.height))
        return {1: "Hallo"}

    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: {})
    monkeypatch.setattr(pdf_ocr, "ocr_pages", tesseract)
    data = _image("JPEG", (40, 30), dpi = (72, 72))
    assert rag_routes.extract_text(_upload("sign.jpg", data), subject = "owner")["text"] == "Hallo"
    assert rendered == [(40, 30)]
    assert _uploads_left() == []


def test_an_oversized_image_is_refused_before_it_is_decoded(rag_home, monkeypatch):
    from PIL import Image

    from core.rag import captioner

    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: {1: "never"})
    data = _image("PNG", (40, 30))
    loads = []
    real_load = Image.Image.load
    monkeypatch.setattr(
        Image.Image, "load", lambda self: loads.append(self.size) or real_load(self)
    )
    monkeypatch.setattr(rag_routes, "_IMAGE_DECODE_MAX_PIXELS", 40 * 30 - 1)
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload("poster.png", data), subject = "owner")
    assert exc.value.status_code == 422 and "too large" in exc.value.detail
    assert loads == []
    assert _uploads_left() == []


def test_an_upright_photo_stays_upright_after_shrinking(rag_home, monkeypatch):
    from PIL import Image

    from core.rag import captioner

    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees: shown portrait
    out = io.BytesIO()
    Image.new("RGB", (4000, 3000), "white").save(out, "JPEG", exif = exif)
    seen = []
    monkeypatch.setattr(captioner, "ocr_pages", lambda pngs: seen.append(pngs[1]) or {1: "ok"})
    rag_routes.extract_text(_upload("phone.jpg", out.getvalue()), subject = "owner")
    width, height = Image.open(io.BytesIO(seen[0])).size
    assert height > width and width * height <= rag_routes._IMAGE_MAX_PIXELS


def _typed_and_scanned_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Typed first page.")
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pix.clear_with(200)
    scan = doc.new_page()
    scan.insert_image(scan.rect, pixmap = pix)
    return doc.tobytes()


def test_scanned_pdf_pages_are_read(rag_home, monkeypatch):
    from core.rag import captioner, pdf_ocr

    seen = []
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(
        captioner, "ocr_pages", lambda pngs: seen.append(sorted(pngs)) or {2: "scanned words"}
    )
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda path, pages: {})
    result = rag_routes.extract_text(
        _upload("contract.pdf", _typed_and_scanned_pdf()), subject = "owner"
    )
    assert result["text"] == "Typed first page.\n\nscanned words"
    assert seen == [[2]]
    assert _uploads_left() == []


def test_scanned_pdf_pages_fall_back_to_tesseract(rag_home, monkeypatch):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    monkeypatch.setattr(
        pdf_ocr, "ocr_pages", lambda path, pages: {n: "tesseract words" for n in pages}
    )
    result = rag_routes.extract_text(
        _upload("contract.pdf", _typed_and_scanned_pdf()), subject = "owner"
    )
    assert result["text"] == "Typed first page.\n\ntesseract words"


def test_unreadable_scanned_pdf_pages_are_not_dropped(rag_home, monkeypatch):
    from core.rag import captioner, pdf_ocr

    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda path, pages: {})
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload("contract.pdf", _typed_and_scanned_pdf()), subject = "owner")
    assert exc.value.status_code == 422
    assert "page 2" in exc.value.detail and "contract.pdf" in exc.value.detail
    assert _uploads_left() == []

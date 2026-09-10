# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image-only PDFs must yield searchable text or an actionable upload error."""

import pytest
import pymupdf

from core.rag import captioner, config, ingestion, parsers, pdf_ocr, store, tool
from .test_data_recipe_seed import _load_seed_route, _run_upload, _block_files
from .test_rag_ocr_fallback import _ingest


def scanned_pdf(
    path,
    *,
    mixed = False,
    header = False,
    scans = 1,
):
    with pymupdf.open() as original:
        page = original.new_page()
        page.insert_text((72, 100), "Invoice zebra-42 total 123 dollars", fontsize = 20)
        png = page.get_pixmap(dpi = 120).tobytes("png")
    with pymupdf.open() as doc:
        if mixed:
            page = doc.new_page()
            page.insert_text((72, 100), "Digital content quokka-17 is preserved.")
        for _ in range(scans):
            page = doc.new_page()
            if header:
                page.insert_text((40, 30), "A digitally added header above the scanned body")
            page.insert_image(pymupdf.Rect(0, 60, 595, 842), stream = png)
        doc.save(str(path))


@pytest.fixture
def local_ocr(monkeypatch):
    """Only native recognition is stubbed: exercise page selection and extraction."""
    calls = []

    def recognize(page, **kwargs):
        calls.append(page.number + 1)
        assert kwargs["full"] is True
        page.insert_text((72, 130), "Invoice zebra-42 total 123 dollars")
        return page.get_textpage()

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", recognize)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    monkeypatch.setattr(config, "OCR_SCANNED", True)
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 20)
    monkeypatch.setattr(config, "CAPTION_IMAGES", False)
    return calls


@pytest.mark.parametrize("mixed", [False, True])
def test_recipe_scanned_pdf_extracts_local_ocr(monkeypatch, tmp_path, local_ocr, mixed):
    route = _load_seed_route(monkeypatch, tmp_path)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf, mixed = mixed)
    result = _run_upload(route, pdf.name, pdf.read_bytes())
    assert result.status == "ok", result.error
    extracted = next(route.UNSTRUCTURED_UPLOAD_ROOT.rglob("*.extracted.txt")).read_text()
    assert "zebra-42" in extracted
    assert local_ocr == [2 if mixed else 1]
    if mixed:
        assert extracted.index("quokka-17") < extracted.index("zebra-42")


@pytest.mark.parametrize("mixed", [False, True])
def test_chat_scanned_pdf_searchable_without_vision(
    rag_conn, stub_embeddings, tmp_path, local_ocr, mixed
):
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf, mixed = mixed)
    doc = _ingest(rag_conn, "t1", pdf.name, pdf)
    assert doc["status"] == "completed", doc["error"]
    text, sources = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "zebra-42" in text
    assert local_ocr == [2 if mixed else 1]
    if mixed:
        assert "quokka-17" in text


def test_scan_with_digital_header_still_gets_ocr(tmp_path, local_ocr, monkeypatch):
    pdf = tmp_path / "header.pdf"
    scanned_pdf(pdf, header = True)
    pages = parsers.parse(str(pdf))
    assert pages[0].needs_ocr
    route = _load_seed_route(monkeypatch, tmp_path)
    text = route._extract_text_from_file(pdf, ".pdf")
    assert text.count("digitally added header") == 1
    assert "zebra-42" in text


@pytest.mark.parametrize("mixed", [False, True])
def test_recipe_missing_ocr_rejects_incomplete_pdf(monkeypatch, tmp_path, mixed):
    route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda *a: {})
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf, mixed = mixed)
    result = _run_upload(route, pdf.name, pdf.read_bytes())
    assert result.status == "error"
    assert f"scanned PDF pages: {2 if mixed else 1}" in result.error
    assert "TESSDATA_PREFIX" in result.error
    assert _block_files(route) == []


@pytest.mark.parametrize("vision_failure", [False, True])
def test_chat_unreadable_pdf_emits_error(
    rag_conn, stub_embeddings, monkeypatch, tmp_path, vision_failure
):
    monkeypatch.setattr(config, "OCR_SCANNED", True)
    monkeypatch.setattr(config, "CAPTION_IMAGES", False)
    monkeypatch.setattr(
        captioner, "vision_endpoint", lambda: ("http://unused", "local") if vision_failure else None
    )
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: None)
    monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda *a: {})
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf, mixed = True)
    doc_id, job_id = ingestion.start_ingestion(
        store.thread_scope("t1"), None, "t1", pdf.name, str(pdf)
    )
    events = list(ingestion.job_events(job_id))
    doc = store.get_document(rag_conn, doc_id)
    assert doc["status"] == "failed"
    assert "scanned PDF pages: 2" in doc["error"]
    assert not doc["num_chunks"]
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "complete" for e in events)


def test_chat_vision_failure_falls_back_locally(
    rag_conn, stub_embeddings, monkeypatch, tmp_path, local_ocr
):
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://unused", "local"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: None)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf)
    doc = _ingest(rag_conn, "t1", pdf.name, pdf)
    assert doc["status"] == "completed"
    assert local_ocr == [1]


@pytest.mark.parametrize("recipe", [False, True])
def test_ocr_page_cap_does_not_silently_drop_pages(
    rag_conn, stub_embeddings, monkeypatch, tmp_path, local_ocr, recipe
):
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 1)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf, scans = 2)
    if recipe:
        route = _load_seed_route(monkeypatch, tmp_path)
        result = _run_upload(route, pdf.name, pdf.read_bytes())
        assert result.status == "error"
        error = result.error
    else:
        doc = _ingest(rag_conn, "t1", pdf.name, pdf)
        assert doc["status"] == "failed"
        error = doc["error"]
    assert "scanned PDF pages: 2" in error
    assert local_ocr == [1]


def test_blank_separator_needs_no_ocr(tmp_path):
    pdf = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(str(pdf))
    assert not parsers.parse(str(pdf))[0].needs_ocr


def test_local_ocr_engine_failure_returns_no_text(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Missing language data")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", unavailable)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf)
    assert pdf_ocr.ocr_pages(str(pdf), [1]) == {}


def test_recipe_respects_disabled_ocr(monkeypatch, tmp_path, local_ocr):
    monkeypatch.setattr(config, "OCR_SCANNED", False)
    route = _load_seed_route(monkeypatch, tmp_path)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf)
    result = _run_upload(route, pdf.name, pdf.read_bytes())
    assert result.status == "error"
    assert "scanned PDF pages: 1" in result.error
    assert local_ocr == []


def test_failed_scan_replacement_preserves_searchable_original(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    old_path = tmp_path / "old.txt"
    old_path.write_text("Existing searchable content quokka-17")
    old = _ingest(rag_conn, "t1", old_path.name, old_path)
    assert old["status"] == "completed"
    monkeypatch.setattr(config, "OCR_SCANNED", False)
    monkeypatch.setattr(config, "CAPTION_IMAGES", False)
    pdf = tmp_path / "scan.pdf"
    scanned_pdf(pdf)
    scope = store.thread_scope("t1")
    doc_id = store.create_document(
        rag_conn,
        scope = scope,
        filename = pdf.name,
        sha256 = "new",
        thread_id = "t1",
        status = "pending",
        stored_path = str(pdf),
    )
    job_id = ingestion._new_job(rag_conn, doc_id, scope)
    ingestion._run(job_id, doc_id, scope, str(pdf), None, replaces = (old["id"], str(old_path)))
    assert store.get_document(rag_conn, doc_id)["status"] == "failed"
    assert store.get_document(rag_conn, old["id"])["status"] == "completed"
    text, _ = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "quokka-17" in text

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scanned-PDF OCR fallback: a PDF page with no text layer is rendered and transcribed
by the vision model during ingestion, so image-only PDFs become searchable. The vision
call is stubbed, so no model is needed."""

import pymupdf

from core.rag import captioner, ingestion, parsers, store, tool


def _image_only_pdf(path, *, pages = 1):
    """A PDF whose pages carry only a raster image, so get_text returns ''."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 120))
    pix.clear_with(220)
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, pixmap = pix)
    doc.save(str(path))
    doc.close()


def _text_pdf(path, body):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), body, fontsize = 11)
    doc.save(str(path))
    doc.close()


def _ingest(rag_conn, thread_id, filename, path):
    """Drive the real ingestion worker synchronously and return the document row."""
    scope = store.thread_scope(thread_id)
    document_id = store.create_document(
        rag_conn,
        scope = scope,
        filename = filename,
        sha256 = filename,
        thread_id = thread_id,
        status = "pending",
        stored_path = str(path),
    )
    job_id = ingestion._new_job(rag_conn, document_id, scope)
    ingestion._run(job_id, document_id, scope, str(path), None)
    return store.get_document(rag_conn, document_id)


# ── parsers.render_pdf_pages ─────────────────────────────────────────


def test_render_pdf_pages_returns_png_per_page(tmp_path):
    pdf = tmp_path / "two.pdf"
    _image_only_pdf(pdf, pages = 2)
    out = parsers.render_pdf_pages(str(pdf), [1, 2], dpi = 72)
    assert set(out) == {1, 2}
    assert all(b.startswith(b"\x89PNG") for b in out.values())


def test_render_pdf_pages_excludes_unwanted(tmp_path):
    pdf = tmp_path / "three.pdf"
    _image_only_pdf(pdf, pages = 3)
    out = parsers.render_pdf_pages(str(pdf), [2], dpi = 72)
    assert set(out) == {2}


def test_render_pdf_pages_empty_request(tmp_path):
    pdf = tmp_path / "one.pdf"
    _image_only_pdf(pdf, pages = 1)
    assert parsers.render_pdf_pages(str(pdf), [], dpi = 72) == {}


# ── captioner.ocr_pages gating ───────────────────────────────────────


def test_ocr_pages_no_endpoint(monkeypatch):
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    assert captioner.ocr_pages({1: b"x"}) == {}


def test_collapse_runaway_caps_repeated_lines():
    # A looping model repeats a line hundreds of times; the guard caps it, keeps repeats.
    text = "\n".join(["TITLE"] * 200 + ["body"] + ["Add & Norm"] * 3)
    out = captioner._collapse_runaway(text)
    lines = out.splitlines()
    assert lines.count("TITLE") == 3  # 200 -> 3
    assert lines.count("Add & Norm") == 3  # legitimate triple survives
    assert "body" in lines


def test_collapse_runaway_caps_interleaved_repeats():
    # Models also loop non-consecutively; the global per-line cap bounds those too.
    text = "\n".join(["Llion Vaswani Google", "Niki Parmar Google"] * 40)
    out = captioner._collapse_runaway(text)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines.count("Llion Vaswani Google") <= 8
    assert lines.count("Niki Parmar Google") <= 8


def test_collapse_runaway_noop_on_normal_text():
    text = "Heading\n\nFirst paragraph.\nSecond paragraph.\n\nFooter"
    assert captioner._collapse_runaway(text) == text


def test_ocr_pages_applies_runaway_guard(monkeypatch):
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: "\n".join(["X"] * 50))
    out = captioner.ocr_pages({1: b"img"}, endpoint = ("http://x", "local"))
    assert out[1].splitlines().count("X") == 3  # guard applied to stored text


def test_ocr_pages_transcribes_and_caps(monkeypatch):
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner.config, "OCR_MAX_PAGES", 1)
    calls = []
    monkeypatch.setattr(
        captioner,
        "_ocr_one",
        lambda base, model, b, t: (calls.append(1) or "transcribed text"),
    )
    out = captioner.ocr_pages({1: b"a", 2: b"b"}, endpoint = ("http://x", "local"))
    assert out == {1: "transcribed text"}  # page 2 dropped by the cap
    assert len(calls) == 1


def test_ocr_scanned_pages_merges_short_text_layer(rag_conn, monkeypatch):
    # Near-empty pages can still have meaningful extractable text; OCR augments it
    # rather than replacing it with a fallible vision transcription.
    scope = store.thread_scope("t1")
    document_id = store.create_document(
        rag_conn, scope = scope, filename = "scan.pdf", sha256 = "h"
    )
    job_id = ingestion._new_job(rag_conn, document_id, scope)
    pages = [parsers.Page("ID-42", 1, 5)]

    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner.config, "OCR_MIN_CHARS", 16)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(parsers, "render_pdf_pages", lambda *a, **k: {1: b"png"})
    monkeypatch.setattr(captioner, "ocr_pages", lambda page_pngs: {1: "OCR body text"})

    out, ocred = ingestion._ocr_scanned_pages(pages, "scan.pdf", rag_conn, job_id)
    assert ocred == {1}
    assert out[0].text == "ID-42\n\nOCR body text"


# ── end-to-end ingestion ─────────────────────────────────────────────


def test_scanned_pdf_is_ocred_into_chunks(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(
        captioner,
        "_ocr_one",
        lambda base, model, b, t: "Invoice total is zebra-42 due Friday",
    )

    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf, pages = 1)
    doc = _ingest(rag_conn, "t1", "scan.pdf", pdf)

    assert doc["status"] == "completed"
    assert doc["num_chunks"] >= 1
    # The OCR'd text is now indexed and reaches whole-document injection.
    text, _sources = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "zebra-42" in text


def test_scanned_page_past_ocr_cap_is_still_captioned(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # OCR is capped to one page, so page 2 is scanned but never transcribed. Figure
    # captioning must still cover it (we exclude only the pages OCR actually handled),
    # so a chart on an un-OCR'd scanned page is not silently dropped.
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner.config, "OCR_MAX_PAGES", 1)
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: "scanned page alpha")
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: "figure caption bravo")

    pdf = tmp_path / "scan2.pdf"
    _image_only_pdf(pdf, pages = 2)
    doc = _ingest(rag_conn, "t1", "scan2.pdf", pdf)

    assert doc["status"] == "completed"
    text, _ = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "scanned page alpha" in text  # page 1 OCR'd, within the cap
    assert (
        "figure caption bravo" in text
    )  # page 2 past the cap -> captioned, not dropped


def test_born_digital_pdf_skips_ocr(rag_conn, stub_embeddings, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(
        captioner, "_ocr_one", lambda *a: called.append(1) or "should not run"
    )

    pdf = tmp_path / "digital.pdf"
    _text_pdf(pdf, "Real born digital body text. " * 30 + "marker-quokka")
    doc = _ingest(rag_conn, "t1", "digital.pdf", pdf)

    assert doc["status"] == "completed"
    assert called == []  # page had real text -> never considered scanned
    text, _sources = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "marker-quokka" in text


def _ingest_with_ocr(rag_conn, thread_id, path, ocr):
    scope = store.thread_scope(thread_id)
    document_id = store.create_document(
        rag_conn,
        scope = scope,
        filename = "scan.pdf",
        sha256 = str(path) + str(ocr),
        thread_id = thread_id,
        status = "pending",
        stored_path = str(path),
    )
    job_id = ingestion._new_job(rag_conn, document_id, scope)
    ingestion._run(job_id, document_id, scope, str(path), None, ocr = ocr)
    return store.get_document(rag_conn, document_id)


def test_ocr_override_false_skips_ocr_when_config_on(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # Config default ON, but the per-upload toggle (ocr=False) skips OCR.
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: "should not run")
    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf, pages = 1)
    doc = _ingest_with_ocr(rag_conn, "t1", pdf, ocr = False)
    assert doc["num_chunks"] == 0  # scanned page left empty


def test_ocr_override_true_runs_ocr_when_config_off(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # Config default OFF, but the per-upload toggle (ocr=True) forces OCR on.
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", False)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: "forced ocr text quokka")
    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf, pages = 1)
    doc = _ingest_with_ocr(rag_conn, "t1", pdf, ocr = True)
    assert doc["num_chunks"] >= 1
    text, _ = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "quokka" in text


def test_ocr_disabled_leaves_scanned_pdf_empty(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    monkeypatch.setattr(captioner.config, "OCR_SCANNED", False)

    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf, pages = 1)
    doc = _ingest(rag_conn, "t1", "scan.pdf", pdf)

    # With OCR off, a text-less scanned page yields no chunks (prior behavior).
    assert doc["status"] == "completed"
    assert doc["num_chunks"] == 0
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000) is None

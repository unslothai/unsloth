# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scan with more pages than OCR can reach fails before any OCR work, with the same error."""

from core.rag import captioner, config

from .test_pdf_local_ocr import local_ocr, scanned_pdf  # noqa: F401
from .test_rag_ocr_fallback import _ingest


def test_a_scan_past_the_ocr_cap_fails_without_ocr(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, local_ocr
):
    path = tmp_path / "book.pdf"
    scanned_pdf(path, scans = 5)
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 2)
    result = _ingest(rag_conn, "book", path.name, path)
    assert result["status"] == "failed"
    assert "Could not read scanned PDF pages: 3, 4, 5." in result["error"]
    assert result["num_chunks"] == 0
    assert local_ocr == []


def test_a_scan_within_the_ocr_cap_is_still_read(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, local_ocr
):
    path = tmp_path / "short.pdf"
    scanned_pdf(path, scans = 2)
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 2)
    result = _ingest(rag_conn, "short", path.name, path)
    assert result["status"] == "completed", result["error"]
    assert local_ocr == [1, 2]


def test_captioning_that_could_cover_the_rest_still_runs_ocr(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, local_ocr
):
    path = tmp_path / "captioned.pdf"
    scanned_pdf(path, scans = 3)
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 2)
    monkeypatch.setattr(config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(config, "CAPTION_MAX_PAGES", 4)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://unused", "vision"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *a: None)
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: None)
    _ingest(rag_conn, "captioned", path.name, path)
    assert local_ocr == [1, 2]

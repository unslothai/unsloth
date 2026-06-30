# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF text extraction: layout-aware Markdown (pymupdf4llm) with plain-text fallback."""

from __future__ import annotations

import pytest

pytest.importorskip("pymupdf")


def _table_pdf(path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 70), "Quarterly Results", fontsize = 16)
    rows = [("Quarter", "Revenue", "Growth"), ("Q1", "$1.2M", "12%"), ("Q2", "$1.5M", "25%")]
    y = 90
    for r in rows:
        page.insert_textbox(pymupdf.Rect(40, y, 250, y + 20), r[0], fontsize = 11)
        page.insert_textbox(pymupdf.Rect(250, y, 400, y + 20), r[1], fontsize = 11)
        page.insert_textbox(pymupdf.Rect(400, y, 540, y + 20), r[2], fontsize = 11)
        y += 24
    doc.save(str(path))
    doc.close()


def test_pdf_extracts_markdown_table(tmp_path, monkeypatch):
    # With Markdown on, the layout is emitted as Markdown markup (heading, and a pipe table
    # where the extractor detects one) that flat get_text never produces.
    pytest.importorskip("pymupdf4llm")
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "Q2" in text and "$1.5M" in text  # cell values preserved
    assert "#" in text or "|" in text  # Markdown markup (heading or table pipes)


def test_pdf_markdown_off_uses_plain_text(tmp_path, monkeypatch):
    # The toggle (RAG_PDF_MARKDOWN=0) falls back to flat PyMuPDF text: content is still
    # there, but with no Markdown markup.
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", False)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "Q2" in text and "$1.5M" in text
    assert "#" not in text and "|" not in text  # plain text path emits no Markdown markup


def test_pdf_markdown_disables_pymupdf4llm_ocr(monkeypatch):
    # Studio owns the OCR policy; Markdown extraction must not invoke PyMuPDF4LLM's
    # automatic OCR path behind the user's per-upload OCR toggle.
    from core.rag import parsers

    captured = {}

    class _FakePymupdf4llm:
        @staticmethod
        def to_markdown(doc, **kwargs):
            captured.update(kwargs)
            return [{"text": "plain markdown"}]

    class _Doc:
        page_count = 1

    monkeypatch.setitem(__import__("sys").modules, "pymupdf4llm", _FakePymupdf4llm)
    assert parsers._pdf_markdown(_Doc()) == ["plain markdown"]
    assert captured["use_ocr"] is False


def test_pdf_markdown_falls_back_when_lib_missing(tmp_path, monkeypatch):
    # If pymupdf4llm extraction returns None (missing/failed), parsing still yields the
    # plain-text pages rather than raising.
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    monkeypatch.setattr(parsers, "_pdf_markdown", lambda doc: None)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    pages = parsers.parse(str(pdf))
    assert pages and "Quarter" in pages[0].text

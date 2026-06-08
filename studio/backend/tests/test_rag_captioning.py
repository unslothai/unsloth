# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multimodal captioning tests: gating, grouping, splice, retrieval."""

from __future__ import annotations

from core.rag import captioner
from core.rag.parsers import Page, ParsedImage


def _img(page):
    return ParsedImage(image_bytes = b"\x89PNG fake", page_number = page, xref = page)


def test_caption_images_disabled_by_default(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", False)
    assert captioner.caption_images([_img(1)], endpoint = ("http://x", "local")) == {}


def test_caption_images_groups_by_page(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner.config, "CAPTION_MAX_IMAGES", 8)
    monkeypatch.setattr(captioner, "_caption_one", lambda base, model, b, t: "a chart of results")
    out = captioner.caption_images([_img(1), _img(1), _img(3)], endpoint = ("http://x", "local"))
    assert out == {1: ["a chart of results", "a chart of results"], 3: ["a chart of results"]}


def test_caption_images_respects_cap(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner.config, "CAPTION_MAX_IMAGES", 2)
    calls = []
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: (calls.append(1) or "cap"))
    captioner.caption_images([_img(i) for i in range(5)], endpoint = ("http://x", "local"))
    assert len(calls) == 2


def test_caption_images_no_endpoint(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    assert captioner.caption_images([_img(1)]) == {}


def test_splice_captions_appends_to_right_page():
    pages = [Page("body one", 1, 8), Page("body two", 2, 8)]
    out = captioner.splice_captions(pages, {2: ["a diagram of X"]})
    assert out[0].text == "body one"
    assert "a diagram of X" in out[1].text
    assert out[1].text.startswith("body two")
    assert out[1].char_count == len(out[1].text)


def test_splice_captions_noop_when_empty():
    pages = [Page("body", 1, 4)]
    assert captioner.splice_captions(pages, {}) is pages


def test_render_pdf_figures_detects_drawing(tmp_path):
    import pymupdf

    from core.rag.parsers import render_pdf_figures

    pdf = tmp_path / "fig.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(60, 60, 540, 460))
    for i in range(8):
        shape.draw_line((80, 80 + i * 40), (520, 80 + i * 40))
    shape.finish(color = (0, 0, 0), fill = (0.8, 0.8, 0.9))
    shape.commit()
    doc.save(str(pdf))
    doc.close()

    figs = render_pdf_figures(str(pdf))
    assert figs, "expected at least one rendered figure region"
    assert figs[0].image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert figs[0].page_number == 1


def test_captioned_text_is_searchable(rag_home, stub_embeddings, monkeypatch):
    from core.rag import retrieval, store
    from storage import rag_db

    pages = [Page("Section 1 intro text about models.", 1, 33)]
    pages = captioner.splice_captions(
        pages, {1: ["bar chart comparing throughput across quantizations"]}
    )
    from core.rag import chunking, embeddings

    chunks = chunking.chunk_pages(
        pages, max_tokens = 128, overlap = 16, count = embeddings.token_counter(None)
    )
    vecs = embeddings.encode([c.text for c in chunks], normalize = True)

    conn = rag_db.get_connection()
    try:
        kb_id = store.create_kb(conn, name = "kb")
        scope = store.kb_scope(kb_id)
        doc_id = store.create_document(conn, scope = scope, filename = "d.pdf", sha256 = "h")
        store.add_chunks(conn, scope, doc_id, chunks, vecs)
        hits = retrieval.retrieve_lexical(conn, scope, "throughput quantizations", k = 5)
    finally:
        conn.close()
    assert hits, "spliced caption text should be retrievable via lexical search"

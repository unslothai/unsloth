# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF region locator + citation preview route tests."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("sqlite_vec")


def _make_pdf(path) -> None:
    import pymupdf

    doc = pymupdf.open()
    body = (
        "BERT is designed to pre-train deep bidirectional representations.\n"
        "The two pre-training objectives are masked language modeling and next "
        "sentence prediction.\n"
        "The Transformer base model uses eight attention heads in each layer.\n"
    )
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize = 11)
    doc.save(str(path))
    doc.close()


def _ingest(home, pdf_path):
    from core.rag import ingestion, store
    from storage import rag_db

    conn = rag_db.get_connection()
    kb_id = store.create_kb(conn, name = "kb")
    conn.close()
    doc_id, job_id = ingestion.start_ingestion(
        store.kb_scope(kb_id), kb_id, None, "doc.pdf", str(pdf_path)
    )
    t0 = time.time()
    while time.time() - t0 < 30:
        s = ingestion.get_job_status(job_id)
        if s and s["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert s and s["status"] == "completed", s
    return kb_id, doc_id


def test_chunks_carry_pdf_regions(rag_home, stub_embeddings):
    from utils.paths import ensure_dir, rag_uploads_root

    pdf = ensure_dir(rag_uploads_root()) / "doc.pdf"
    _make_pdf(pdf)
    kb_id, doc_id = _ingest(rag_home, pdf)

    from storage import rag_db

    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT pdf_regions_json FROM chunks WHERE document_id=?", (doc_id,)
        ).fetchall()
        stored_path = conn.execute(
            "SELECT stored_path FROM documents WHERE id=?", (doc_id,)
        ).fetchone()["stored_path"]
    finally:
        conn.close()

    assert rows, "no chunks were stored"
    assert stored_path and stored_path.endswith(".pdf")
    with_regions = [r for r in rows if r["pdf_regions_json"]]
    assert with_regions, "expected at least one chunk with PDF highlight regions"
    import json

    region = json.loads(with_regions[0]["pdf_regions_json"])[0]
    for key in ("pageIndex", "x", "y", "width", "height"):
        assert key in region
        if key in ("x", "y", "width", "height"):
            assert 0.0 <= region[key] <= 1.0


def test_preview_routes_and_signed_file(rag_home, stub_embeddings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.rag import router

    from utils.paths import ensure_dir, rag_uploads_root

    pdf = ensure_dir(rag_uploads_root()) / "doc.pdf"
    _make_pdf(pdf)
    kb_id, doc_id = _ingest(rag_home, pdf)

    app = FastAPI()
    app.include_router(router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    c = TestClient(app)

    res = c.post(
        "/api/rag/search",
        json = {
            "query": "masked language modeling next sentence",
            "kb_id": kb_id,
            "mode": "lexical",
        },
    ).json()["results"]
    assert res
    chunk_id = res[0]["chunkId"]

    pt = c.get(
        f"/api/rag/documents/{doc_id}/preview-target", params = {"chunk_id": chunk_id}
    ).json()
    assert pt["mediaKind"] == "pdf"
    assert pt["text"]

    url = c.get(f"/api/rag/documents/{doc_id}/file-url").json()["url"]
    full = c.get(url)
    assert full.status_code == 200 and full.content[:4] == b"%PDF"
    rng = c.get(url, headers = {"Range": "bytes=0-99"})
    assert rng.status_code in (200, 206)
    assert (
        c.get(
            f"/api/rag/documents/{doc_id}/file-signed",
            params = {"token": "bad.token.sig"},
        ).status_code
        == 401
    )


def test_norm_token_decomposes_ligatures():
    # NFKC folds ligature glyphs to ASCII so anchors match (search_for misses these).
    from core.rag.locators import _norm_token

    assert _norm_token("signiﬁcant") == "significant"  # ﬁ
    assert _norm_token("eﬀort.") == "effort"  # ﬀ + trailing punct
    assert _norm_token("**Bold**") == "bold"
    assert _norm_token("...") == ""


def test_locator_handles_midword_anchor_and_locates_line():
    # A span beginning mid-word still locates: first/last tokens dropped.
    import pymupdf

    from core.rag.locators import LocatorMatch, _regions_for_match

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 200), "alpha beta gamma delta epsilon zeta eta theta", fontsize = 12
    )
    page_text = doc[0].get_text("text")  # mirrors what the parser stores
    start = page_text.index("lpha")
    end = page_text.index("theta") + 3
    match = LocatorMatch(page_index = 0, page_number = 1, start = start, end = end)
    rects = _regions_for_match(doc, page_text, match)
    doc.close()

    assert rects, "expected a located region for the interior phrase"
    r = rects[0]
    for k in ("pageIndex", "pageNumber", "x", "y", "width", "height"):
        assert k in r
    # Drawn near y=200 on a ~842pt page -> normalized y in the top half.
    assert 0.0 < r["y"] < 0.5
    assert r["width"] > 0 and r["height"] > 0


def test_locator_anchors_through_markdown_table_pipes():
    # Markdown table cells are pipe-joined with no spaces; the locator splits on pipes
    # so a table-row chunk still anchors to the raw PDF word stream.
    import pymupdf

    from core.rag.locators import LocatorMatch, _regions_for_match

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 200), "Quarter Revenue Growth Q1 sales strong here", fontsize = 12
    )
    # What the Markdown parser stores for the row (cells joined by pipes, no spaces).
    page_text = "|Quarter|Revenue|Growth|Q1|sales|strong|here|"
    match = LocatorMatch(page_index = 0, page_number = 1, start = 0, end = len(page_text))
    rects = _regions_for_match(doc, page_text, match)
    doc.close()

    assert rects, "a Markdown table row should still anchor to the page words"


def test_sign_verify_roundtrip(rag_home):
    from routes import rag as rag_routes

    tok = rag_routes._sign_document("doc-123")
    assert rag_routes._verify_document_token(tok) == "doc-123"
    assert (
        rag_routes._verify_document_token("doc-123.0.deadbeef") is None
    )  # expired/bad
    assert rag_routes._verify_document_token("garbage") is None

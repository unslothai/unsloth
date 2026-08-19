# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multimodal captioning tests: gating, grouping, splice, retrieval."""

from __future__ import annotations

from core.rag import captioner
from core.rag.parsers import Page, ParsedImage


def _img(page):
    return ParsedImage(image_bytes = b"\x89PNG fake", page_number = page, xref = page)


def test_caption_images_runs_when_images_present(monkeypatch):
    # Policy lives in ingestion (_run); caption_images captions given images + endpoint.
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", False)
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: "a chart")
    out = captioner.caption_images([_img(1)], endpoint = ("http://x", "local"))
    assert out == {1: ["a chart"]}


def test_caption_images_groups_by_page(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_MAX_IMAGES", 8)
    monkeypatch.setattr(captioner, "_caption_one", lambda base, model, b, t: "a chart of results")
    out = captioner.caption_images([_img(1), _img(1), _img(3)], endpoint = ("http://x", "local"))
    assert out == {1: ["a chart of results", "a chart of results"], 3: ["a chart of results"]}


def test_caption_images_respects_cap(monkeypatch):
    monkeypatch.setattr(captioner.config, "CAPTION_MAX_IMAGES", 2)
    calls = []
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: (calls.append(1) or "cap"))
    captioner.caption_images([_img(i) for i in range(5)], endpoint = ("http://x", "local"))
    assert len(calls) == 2


def test_caption_images_no_endpoint(monkeypatch):
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    assert captioner.caption_images([_img(1)]) == {}


def test_caption_runaway_guard_applied(monkeypatch):
    # A looping vision model must not flood the index; captions pass _collapse_runaway.
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: "\n".join(["LOOP"] * 40))
    out = captioner.caption_images([_img(1)], endpoint = ("http://x", "local"))
    assert out[1][0].splitlines().count("LOOP") == 3  # 40 -> 3


def test_caption_prompt_and_token_budget(monkeypatch):
    # Caption and OCR keep separate prompts + token caps over the shared _vision_complete.
    captured: dict = {}

    def fake_vision_complete(base_url, model, image_bytes, *, prompt, timeout, max_tokens):
        captured.update(prompt = prompt, timeout = timeout, max_tokens = max_tokens)
        return "ok"

    monkeypatch.setattr(captioner, "_vision_complete", fake_vision_complete)
    monkeypatch.setattr(captioner.config, "CAPTION_MAX_TOKENS", 277)

    captioner._caption_one("http://x", "local", b"img", 12.0)
    prompt = captured["prompt"].lower()
    # Unified prompt: transcribe every label (recall) + axis/legend coverage + describe.
    assert "transcribe" in prompt
    assert ("axis" in prompt or "axes" in prompt) and "legend" in prompt
    assert "do not invent" in prompt
    assert captured["max_tokens"] == 277
    assert captured["timeout"] == 12.0

    captured.clear()
    monkeypatch.setattr(captioner.config, "OCR_MAX_TOKENS", 999)
    captioner._ocr_one("http://x", "local", b"img", 5.0)
    assert captured["max_tokens"] == 999
    assert "transcribe" in captured["prompt"].lower()


def test_pages_with_figures_and_tiles(tmp_path):
    from core.rag import parsers

    pdf = tmp_path / "fig.pdf"
    _figure_pdf(pdf)
    pgs = parsers.pages_with_figures(str(pdf), max_pages = 4)
    assert pgs == [1]
    tiles = parsers.render_pdf_figure_tiles(str(pdf), pgs, rows = 2, cols = 2, fullpage = True)
    assert len(tiles) == 5  # full page + 2x2 grid
    assert all(t.image_bytes[:8] == b"\x89PNG\r\n\x1a\n" and t.page_number == 1 for t in tiles)
    capped = parsers.render_pdf_figure_tiles(
        str(pdf), pgs, rows = 2, cols = 2, fullpage = True, max_tiles = 3
    )
    assert len(capped) == 3  # max_tiles budget honored


def test_render_pdf_figure_tiles_zero_grid_no_crash(tmp_path):
    # A misconfigured rows/cols=0 must clamp to 1, not raise ZeroDivisionError.
    import pymupdf

    from core.rag import parsers

    pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()

    out = parsers.render_pdf_figure_tiles(str(pdf), [1], rows = 0, cols = 0, fullpage = True)
    assert len(out) == 2  # full page + a single 1x1 tile, no crash


def test_pages_with_figures_excludes_given_pages(tmp_path):
    # Pages OCR already transcribed (passed as exclude_pages) are skipped; every other
    # figure page is still returned for tiling.
    import pymupdf

    from core.rag import parsers

    def _draw_chart(page):
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(60, 140, 540, 520))
        for i in range(8):
            shape.draw_line((80, 160 + i * 40), (520, 160 + i * 40))
        shape.finish(color = (0, 0, 0), fill = (0.8, 0.8, 0.9))
        shape.commit()

    pdf = tmp_path / "charts.pdf"
    doc = pymupdf.open()
    _draw_chart(doc.new_page())
    _draw_chart(doc.new_page())
    doc.save(str(pdf))
    doc.close()

    assert parsers.pages_with_figures(str(pdf), max_pages = 4) == [1, 2]
    assert parsers.pages_with_figures(str(pdf), max_pages = 4, exclude_pages = {1}) == [2]
    assert parsers.pages_with_figures(str(pdf), max_pages = 4, exclude_pages = {2}) == [1]


def test_run_skips_figure_work_without_vision_model(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # No vision model -> the whole figure pass (detection + rasterization) is skipped.
    from core.rag import parsers

    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)
    touched: list[str] = []
    monkeypatch.setattr(
        parsers, "pages_with_figures", lambda *a, **k: touched.append("detect") or []
    )
    monkeypatch.setattr(
        parsers, "render_pdf_figure_tiles", lambda *a, **k: touched.append("render") or []
    )

    pdf = tmp_path / "fig.pdf"
    _figure_pdf(pdf)
    _ingest_with_caption(rag_conn, "t1", pdf, None)  # follow config (ON), but no model
    assert touched == []  # neither figure detection nor tiling ran


def test_vision_complete_sends_auth_header(monkeypatch):
    # Direct-stream serves llama-server with --api-key; vision calls must send the bearer.
    import httpx

    monkeypatch.setattr(
        captioner, "_vision_auth_headers", lambda: {"Authorization": "Bearer secret"}
    )
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, *, json, timeout, headers, trust_env):
        captured.update(url = url, headers = headers, trust_env = trust_env)
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    out = captioner._vision_complete(
        "http://x", "local", b"img", prompt = "p", timeout = 5.0, max_tokens = 8
    )
    assert out == "ok"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["trust_env"] is False


def test_vision_complete_omits_header_when_unauthenticated(monkeypatch):
    # No api-key configured -> no spurious Authorization header on plain llama-server.
    import httpx

    monkeypatch.setattr(captioner, "_vision_auth_headers", lambda: None)
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, *, json, timeout, headers, trust_env):
        captured["headers"] = headers
        captured["trust_env"] = trust_env
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    captioner._vision_complete("http://x", "local", b"i", prompt = "p", timeout = 5.0, max_tokens = 8)
    assert captured["headers"] is None
    assert captured["trust_env"] is False


def test_merge_page_captions_dedups():
    out = captioner.merge_page_captions({1: ["MatMul\nScale", "Scale\nSoftMax"]})
    text = out[1][0]
    assert text.lower().count("scale") == 1  # repeated label from overlapping tiles dropped
    assert "MatMul" in text and "SoftMax" in text


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


# ── per-upload caption override (parallels test_rag_ocr_fallback.py) ──


def _figure_pdf(path):
    """A born-digital PDF: a page with real text (so it is not treated as scanned)
    plus a vector drawing region that figure detection picks up as a figure."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(40, 40, 550, 120),
        "Quarterly revenue report. The chart below shows the trend.",
        fontsize = 11,
    )
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(60, 140, 540, 520))
    for i in range(8):
        shape.draw_line((80, 160 + i * 40), (520, 160 + i * 40))
    shape.finish(color = (0, 0, 0), fill = (0.8, 0.8, 0.9))
    shape.commit()
    doc.save(str(path))
    doc.close()


def _ingest_with_caption(rag_conn, thread_id, path, caption):
    from core.rag import ingestion, store

    scope = store.thread_scope(thread_id)
    document_id = store.create_document(
        rag_conn,
        scope = scope,
        filename = "fig.pdf",
        sha256 = str(path) + str(caption),
        thread_id = thread_id,
        status = "pending",
        stored_path = str(path),
    )
    job_id = ingestion._new_job(rag_conn, document_id, scope)
    # _run(job_id, document_id, scope, stored_path, model_name, ocr, caption)
    ingestion._run(job_id, document_id, scope, str(path), None, None, caption)
    return store.get_document(rag_conn, document_id)


def test_caption_override_true_runs_when_config_off(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # Config default OFF, but the per-upload toggle (caption=True) forces captioning.
    from core.rag import tool

    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", False)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: "bar chart of revenue wombat-7")

    pdf = tmp_path / "fig.pdf"
    _figure_pdf(pdf)
    _ingest_with_caption(rag_conn, "t1", pdf, True)

    text, _ = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "wombat-7" in text  # the spliced figure caption reached the index


def test_caption_override_false_skips_when_config_on(
    rag_conn, stub_embeddings, monkeypatch, tmp_path
):
    # Config default ON, but the per-upload toggle (caption=False) skips captioning.
    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    called = []
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: called.append(1) or "should not run")

    pdf = tmp_path / "fig.pdf"
    _figure_pdf(pdf)
    _ingest_with_caption(rag_conn, "t1", pdf, False)

    assert called == []  # no vision caption calls despite config ON


def test_caption_none_follows_config(rag_conn, stub_embeddings, monkeypatch, tmp_path):
    # Omitted override (None) falls back to config.CAPTION_IMAGES.
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://x", "local"))
    seen = []
    monkeypatch.setattr(captioner, "_caption_one", lambda *a: seen.append(1) or "chart caption")

    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", False)
    pdf_off = tmp_path / "off.pdf"
    _figure_pdf(pdf_off)
    _ingest_with_caption(rag_conn, "t1", pdf_off, None)
    assert seen == []  # config OFF + no override -> no captioning

    monkeypatch.setattr(captioner.config, "CAPTION_IMAGES", True)
    pdf_on = tmp_path / "on.pdf"
    _figure_pdf(pdf_on)
    _ingest_with_caption(rag_conn, "t2", pdf_on, None)
    assert seen  # config ON + no override -> captioning runs

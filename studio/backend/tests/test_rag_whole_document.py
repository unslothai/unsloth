# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whole-document context mode: a thread-attached file small enough to fit is
injected in full (every chunk, in order) instead of top-K retrieval. Covers the
new store query, the tool-level renderer, and the auto-inject wiring + fallback.
No embedder is needed - the whole-doc path does no query embedding."""

import json

from core.rag import store, tool
from core.rag.chunking import Chunk
from core.inference import tools as inf_tools

# A vector per chunk just to satisfy add_chunks (the whole-doc path never reads
# vectors); dimension is arbitrary but must be consistent within a connection.
_VEC = [0.1, 0.2, 0.3, 0.4]


def _chunk(
    text,
    index = 0,
    page = None,
    tokens = None,
):
    return Chunk(
        text = text,
        token_count = tokens if tokens is not None else len(text.split()),
        page_number = page,
        source_page_index = 0,
        chunk_index = index,
        page_char_start = 0,
        page_char_end = len(text),
    )


def _add_doc(
    conn,
    scope,
    doc_id,
    filename,
    sha,
    texts,
    *,
    status = "completed",
    tokens = None,
    pages = None,
):
    chunks = [
        _chunk(
            t,
            i,
            page = (pages[i] if pages else None),
            tokens = (tokens[i] if tokens else None),
        )
        for i, t in enumerate(texts)
    ]
    vectors = [list(_VEC) for _ in texts]
    store.create_document(conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id)
    store.add_chunks(conn, scope, doc_id, chunks, vectors)
    store.set_document_status(conn, doc_id, status, num_chunks = len(texts))


def _injected_text(result) -> str:
    """The text spliced into the conversation as the synthetic tool result."""
    tool_msg = next(m for m in result["messages"] if m.get("role") == "tool")
    return tool_msg["content"]


# ── store.all_chunks_for_scope ───────────────────────────────────────


def test_all_chunks_for_scope_orders_by_document_then_index(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "first.pdf", "h1", ["a", "b", "c"])
    _add_doc(rag_conn, scope, "d2", "second.pdf", "h2", ["x", "y"])
    rows = store.all_chunks_for_scope(rag_conn, scope)
    assert [r["id"] for r in rows] == ["d1:0", "d1:1", "d1:2", "d2:0", "d2:1"]
    assert rows[0]["filename"] == "first.pdf"
    assert rows[-1]["filename"] == "second.pdf"
    assert rows[0]["text"] == "a"


def test_all_chunks_for_scope_excludes_non_completed(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "done", "done.pdf", "h1", ["ready"])
    _add_doc(rag_conn, scope, "pend", "pend.pdf", "h2", ["indexing"], status = "pending")
    rows = store.all_chunks_for_scope(rag_conn, scope)
    assert [r["id"] for r in rows] == ["done:0"]


def test_all_chunks_for_scope_empty_scope(rag_conn):
    assert store.all_chunks_for_scope(rag_conn, store.thread_scope("nope")) == []


def test_all_chunks_for_scope_isolates_scopes(rag_conn):
    _add_doc(rag_conn, store.thread_scope("t1"), "d1", "f", "h1", ["mine"])
    _add_doc(rag_conn, store.thread_scope("t2"), "d2", "f", "h2", ["theirs"])
    rows = store.all_chunks_for_scope(rag_conn, store.thread_scope("t1"))
    assert [r["text"] for r in rows] == ["mine"]


# ── store.scope_token_estimate (cheap whole-doc budget pre-check) ─────


def test_scope_token_estimate_sums_without_hydrating(rag_conn):
    # Stored counts sum directly; zero/missing falls back to length/4; non-completed out.
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "a.pdf", "h1", ["alpha", "bravo"], tokens = [10, 20])
    # token_count 0 -> length/4 fallback: a 40-char chunk estimates to 10 tokens.
    _add_doc(rag_conn, scope, "d2", "b.pdf", "h2", ["x" * 40], tokens = [0])
    _add_doc(rag_conn, scope, "d3", "c.pdf", "h3", ["pending"], status = "pending", tokens = [99])
    assert store.scope_token_estimate(rag_conn, scope) == 10 + 20 + 10
    assert store.scope_token_estimate(rag_conn, store.thread_scope("none")) == 0


def test_scope_token_estimate_matches_row_sum(rag_conn):
    # Must agree with the exact per-row sum it short-circuits (one stored count, one
    # length/4 fallback), so the pre-check never disagrees with the full path.
    from core.rag.tool import _row_token_count

    scope = store.thread_scope("t1")
    _add_doc(
        rag_conn, scope, "d1", "a.pdf", "h1", ["a long-ish chunk body here", "tail"], tokens = [0, 5]
    )
    rows = store.all_chunks_for_scope(rag_conn, scope)
    assert store.scope_token_estimate(rag_conn, scope) == sum(_row_token_count(r) for r in rows)


# ── tool.whole_document_context ──────────────────────────────────────


def test_whole_document_context_returns_full_text_and_sources(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(
        rag_conn,
        scope,
        "d1",
        "report.pdf",
        "h1",
        ["chapter one body", "chapter two body"],
        pages = [1, 2],
    )
    result = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert result is not None
    text, sources = result
    # Every chunk is present, in order, as <chunk> blocks.
    assert "chapter one body" in text
    assert "chapter two body" in text
    assert '<chunk id="1"' in text
    assert '<chunk id="2"' in text
    assert text.index("chapter one") < text.index("chapter two")
    # Source-map mirrors retrieval's shape, with no score on the whole-doc path.
    assert [s["citationId"] for s in sources] == [1, 2]
    assert all(s["filename"] == "report.pdf" for s in sources)
    assert all(s["score"] is None for s in sources)
    assert [s["page"] for s in sources] == [1, 2]
    assert [s["chunkId"] for s in sources] == ["d1:0", "d1:1"]


def test_whole_document_context_none_over_budget(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "big.pdf", "h1", ["huge"], tokens = [50_000])
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000) is None
    # Same doc fits under a larger budget.
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 100_000) is not None


def test_whole_document_context_none_when_empty(rag_conn):
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000) is None


def test_whole_document_context_non_positive_budget_returns_none(rag_conn):
    # A non-positive budget disables whole-doc (RAG_WHOLE_DOC_MAX_TOKENS=0 footgun)
    # rather than injecting the whole corpus unbounded.
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "a.pdf", "h1", ["tiny body"])
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 0) is None
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = -5) is None


def test_whole_document_context_none_without_scope(rag_conn):
    # No thread scope -> None (whole-doc is thread-attachment only).
    assert tool.whole_document_context(max_tokens = 6000) is None


def test_whole_document_context_null_token_count_enforces_budget(rag_conn):
    # A missing token_count must not bypass the budget; fall back to a length estimate.
    big = "word " * 20_000  # ~20k tokens by length estimate
    _add_doc(rag_conn, store.thread_scope("t1"), "d1", "big.pdf", "h1", [big], tokens = [None])
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000) is None
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 1_000_000) is not None


def test_whole_document_context_spans_multiple_docs(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "a.pdf", "h1", ["alpha text"])
    _add_doc(rag_conn, scope, "d2", "b.pdf", "h2", ["bravo text"])
    text, sources = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "alpha text" in text and "bravo text" in text
    assert {s["filename"] for s in sources} == {"a.pdf", "b.pdf"}


# ── build_rag_autoinject wiring ──────────────────────────────────────


def _convo(text = "summarize the whole document"):
    return [{"role": "user", "content": text}]


def test_build_rag_autoinject_uses_whole_doc(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "doc.pdf", "h1", ["whole alpha part", "whole bravo part"])
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1"})
    assert result is not None
    injected = _injected_text(result)
    # Both chunks present -> the model receives the entire file, not top-K.
    assert "whole alpha part" in injected
    assert "whole bravo part" in injected
    # Tool-message content is chunk text only; the citation JSON tail is internal.
    assert inf_tools.RAG_SOURCES_SENTINEL not in injected


def test_build_rag_autoinject_whole_doc_runs_when_autoinject_false(rag_conn, monkeypatch):
    # Large-model Auto sets autoinject=False, but whole-doc is a separate thread-doc
    # context mode and should still inject a fitting attachment.
    _add_doc(rag_conn, store.thread_scope("t1"), "d1", "doc.pdf", "h1", ["entire file body"])
    monkeypatch.setattr(
        tool,
        "search_for_autoinject",
        lambda **kw: (_ for _ in ()).throw(AssertionError("retrieval should not run")),
    )
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "autoinject": False})
    assert result is not None
    assert "entire file body" in _injected_text(result)


def test_build_rag_autoinject_explicit_off_disables_whole_doc(rag_conn, monkeypatch):
    # The UI Off switch sends both autoinject=False and whole_doc=False.
    _add_doc(rag_conn, store.thread_scope("t1"), "d1", "doc.pdf", "h1", ["small body"])
    monkeypatch.setattr(
        tool,
        "search_for_autoinject",
        lambda **kw: (_ for _ in ()).throw(AssertionError("retrieval should not run")),
    )
    assert (
        inf_tools.build_rag_autoinject(
            _convo(), {"thread_id": "t1", "autoinject": False, "whole_doc": False}
        )
        is None
    )


def test_build_rag_autoinject_falls_back_over_budget(rag_conn, monkeypatch):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "big.pdf", "h1", ["overflow"], tokens = [50_000])

    sentinel = ("TOPK_FALLBACK_TEXT", [{"citationId": 1, "filename": "big.pdf", "text": "x"}])
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: sentinel)

    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1"})
    assert result is not None
    assert _injected_text(result) == "TOPK_FALLBACK_TEXT"


def test_build_rag_autoinject_context_budget_falls_back(rag_conn, monkeypatch):
    # Runtime context can be smaller than RAG_WHOLE_DOC_MAX_TOKENS; cap whole-doc to
    # the active context and fall back to retrieval when it would overflow.
    _add_doc(
        rag_conn, store.thread_scope("t1"), "d1", "small.pdf", "h1", ["fits global"], tokens = [900]
    )
    sentinel = ("TOPK_CONTEXT_FALLBACK", [{"citationId": 1, "filename": "small.pdf", "text": "x"}])
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: sentinel)
    result = inf_tools.build_rag_autoinject(
        _convo(), {"thread_id": "t1", "context_length": 1200, "whole_doc": True}
    )
    assert result is not None
    assert _injected_text(result) == "TOPK_CONTEXT_FALLBACK"


def test_whole_doc_budget_reserves_image_parts(monkeypatch):
    from core.rag import config

    monkeypatch.setattr(config, "WHOLE_DOC_MAX_TOKENS", 10_000)
    scope = {"context_length": 7000, "response_headroom": 1000}
    text_only = [{"role": "user", "content": [{"type": "text", "text": "summarize"}]}]
    with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarize"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]

    assert (
        inf_tools._whole_doc_budget(scope, text_only)
        - inf_tools._whole_doc_budget(scope, with_image)
        == inf_tools._IMAGE_PART_TOKEN_ESTIMATE
    )


def test_build_rag_autoinject_server_kill_switch_blocks_whole_doc(rag_conn, monkeypatch):
    # RAG_THREAD_WHOLE_DOC=0 stays authoritative; browser requests should not
    # turn it back on by default.
    from core.rag import config

    monkeypatch.setattr(config, "THREAD_WHOLE_DOC", False)
    _add_doc(rag_conn, store.thread_scope("t1"), "d1", "doc.pdf", "h1", ["small body"])
    monkeypatch.setattr(
        tool,
        "search_for_autoinject",
        lambda **kw: (_ for _ in ()).throw(AssertionError("retrieval should not run")),
    )
    assert (
        inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "autoinject": False}) is None
    )


def test_whole_document_context_budgets_rendered_wrappers(rag_conn):
    # Many tiny chunks add wrapper overhead beyond raw chunk token counts; budget
    # the rendered prompt, not just stored text.
    texts = ["x" for _ in range(120)]
    _add_doc(
        rag_conn,
        store.thread_scope("t1"),
        "d1",
        "many-pages.pdf",
        "h1",
        texts,
        tokens = [1 for _ in texts],
    )
    assert tool.whole_document_context(scope_thread_id = "t1", max_tokens = 500) is None


def test_build_rag_autoinject_whole_doc_disabled_via_override(rag_conn, monkeypatch):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "doc.pdf", "h1", ["small body"])

    sentinel = ("TOPK_TEXT", [{"citationId": 1, "filename": "doc.pdf", "text": "x"}])
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: sentinel)

    # whole_doc=False forces retrieval even though the doc fits.
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "whole_doc": False})
    assert result is not None
    assert _injected_text(result) == "TOPK_TEXT"


def test_build_rag_autoinject_kb_scope_never_whole_doc(rag_conn, monkeypatch):
    # A KB-only scope (no thread) goes through retrieval, never whole-doc.
    kb_scope = store.kb_scope("K1")
    _add_doc(rag_conn, kb_scope, "d1", "kb.pdf", "h1", ["kb body one", "kb body two"])

    sentinel = ("KB_RETRIEVAL_TEXT", [{"citationId": 1, "filename": "kb.pdf", "text": "x"}])
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: sentinel)

    result = inf_tools.build_rag_autoinject(_convo(), {"kb_id": "K1"})
    assert result is not None
    assert _injected_text(result) == "KB_RETRIEVAL_TEXT"


def test_whole_document_context_thread_scope_only(rag_conn):
    # A project corpus chunk is never whole-doc injected, even with a thread attachment.
    _add_doc(rag_conn, store.thread_scope("t1"), "td", "thread.txt", "h1", ["thread attachment"])
    _add_doc(rag_conn, store.project_scope("p1"), "pd", "project.txt", "h2", ["project corpus"])
    text, sources = tool.whole_document_context(scope_thread_id = "t1", max_tokens = 6000)
    assert "thread attachment" in text
    assert "project corpus" not in text
    assert {s["filename"] for s in sources} == {"thread.txt"}


def test_build_rag_autoinject_appends_project_retrieval(rag_conn, monkeypatch):
    # Project chat: thread attachment whole-doc'd AND project sources retrieved, merged.
    _add_doc(
        rag_conn,
        store.thread_scope("t1"),
        "td",
        "thread.txt",
        "h1",
        ["thread chunk one", "thread chunk two"],
    )
    proj = (
        "PROJ",
        [
            {
                "citationId": 1,
                "chunkId": "pj:0",
                "documentId": "pj",
                "filename": "project.txt",
                "page": None,
                "text": "project passage zeta",
                "score": 0.91,
            }
        ],
    )
    captured = {}

    def fake_search(**kw):
        captured.update(kw)
        return proj

    monkeypatch.setattr(tool, "search_for_autoinject", fake_search)
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "project_id": "p1"})
    injected = _injected_text(result)
    # Whole thread attachment AND the project passage are both injected.
    assert "thread chunk one" in injected
    assert "thread chunk two" in injected
    assert "project passage zeta" in injected
    # The companion retrieval was scoped to the project only (not thread or KB).
    assert captured.get("scope_project_id") == "p1"
    assert captured.get("scope_thread_id") is None
    assert captured.get("scope_kb_id") is None
    # Citation ids are sequential across the merged set: thread 1,2 then project 3.
    assert '<chunk id="1"' in injected
    assert '<chunk id="2"' in injected
    assert '<chunk id="3"' in injected


def test_build_rag_autoinject_skips_project_companion_over_budget(rag_conn, monkeypatch):
    _add_doc(rag_conn, store.thread_scope("t1"), "td", "thread.txt", "h1", ["thread body"])
    project_text = "project overflow " * 2000
    proj = (
        "PROJ",
        [
            {
                "citationId": 1,
                "chunkId": "pj:0",
                "documentId": "pj",
                "filename": "project.txt",
                "page": None,
                "text": project_text,
                "score": 0.91,
            }
        ],
    )

    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: proj)
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "project_id": "p1"})
    injected = _injected_text(result)
    assert "thread body" in injected
    assert "project overflow" not in injected


def test_build_rag_autoinject_thread_whole_doc_ignores_project_size(rag_conn, monkeypatch):
    # A large project corpus must not push a small thread attachment over budget;
    # whole-doc resolves the thread scope alone (companion retrieval stubbed out).
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: None)
    _add_doc(rag_conn, store.thread_scope("t1"), "td", "thread.txt", "h1", ["small thread file"])
    _add_doc(
        rag_conn, store.project_scope("p1"), "pd", "project.txt", "h2", ["big"], tokens = [50_000]
    )
    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1", "project_id": "p1"})
    assert "small thread file" in _injected_text(result)


def test_build_rag_autoinject_kb_defers_to_retrieval(rag_conn, monkeypatch):
    # A KB selection is exclusive: a thread attachment can't preempt it; KB uses retrieval.
    _add_doc(rag_conn, store.thread_scope("t1"), "td", "thread.txt", "h1", ["thread attachment"])
    sentinel = ("KB_RETRIEVAL", [{"citationId": 1, "filename": "kb.pdf", "text": "x"}])
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: sentinel)
    result = inf_tools.build_rag_autoinject(_convo(), {"kb_id": "K1", "thread_id": "t1"})
    assert _injected_text(result) == "KB_RETRIEVAL"


def test_build_rag_autoinject_no_scope_returns_none(rag_conn):
    assert inf_tools.build_rag_autoinject(_convo(), None) is None
    assert inf_tools.build_rag_autoinject(_convo(), {}) is None


def test_build_rag_autoinject_args_carry_user_query(rag_conn):
    scope = store.thread_scope("t1")
    _add_doc(rag_conn, scope, "d1", "doc.pdf", "h1", ["small body"])
    result = inf_tools.build_rag_autoinject(_convo("what is in here"), {"thread_id": "t1"})
    assistant_msg = next(m for m in result["messages"] if m.get("role") == "assistant")
    args = json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"])
    assert args["query"] == "what is in here"


# ── end-to-end: real ingestion pipeline -> whole-doc injection ────────


def test_real_ingestion_feeds_whole_document(rag_conn, stub_embeddings, tmp_path):
    """Drive the real ingestion worker on a multi-paragraph file, then confirm whole-doc
    injection splices the entire document, not just retrieved chunks."""
    from core.rag import ingestion

    scope = store.thread_scope("t1")
    body = (
        "# Quarterly Report\n\n"
        + ("Revenue rose across every region this period. " * 40)
        + "\n\nThe unique closing marker is xyzzy-sentinel for the final page. " * 40
    )
    src = tmp_path / "report.md"
    src.write_text(body, encoding = "utf-8")

    document_id = store.create_document(
        rag_conn,
        scope = scope,
        filename = "report.md",
        sha256 = "sha-e2e",
        thread_id = "t1",
        status = "pending",
        stored_path = str(src),
    )
    job_id = ingestion._new_job(rag_conn, document_id, scope)
    ingestion._run(job_id, document_id, scope, str(src), None)

    doc = store.get_document(rag_conn, document_id)
    assert doc["status"] == "completed"
    assert doc["num_chunks"] >= 2  # the doc chunked into multiple pieces

    result = inf_tools.build_rag_autoinject(_convo(), {"thread_id": "t1"})
    assert result is not None
    injected = _injected_text(result)
    # Opening and ending both present -> the whole file reached the model.
    assert "Revenue rose" in injected
    assert "xyzzy-sentinel" in injected
    # Every stored chunk is represented as a numbered block.
    assert injected.count("<chunk id=") == doc["num_chunks"]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retrieval + tool tests: RRF fusion, min-score floor, scope, source-map."""

import math

import pytest

from core.rag import config, retrieval, store, tool
from core.rag.chunking import Chunk

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@pytest.fixture
def bow_embeddings(monkeypatch):
    """Bag-of-words embedder matching the vectors stored in the db."""
    from core.rag import embeddings

    monkeypatch.setattr(
        embeddings,
        "encode",
        lambda texts, *, model_name = None, normalize = True: [_embed(t) for t in texts],
    )
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: len(VOCAB))


def _chunk(
    text,
    index = 0,
    page = None,
):
    return Chunk(
        text = text,
        token_count = len(text.split()),
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
    text,
    page = None,
):
    store.create_document(
        conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id
    )
    store.add_chunks(conn, scope, doc_id, [_chunk(text, 0, page)], [_embed(text)])


def test_rrf_ranks_doc_in_both_lists_first():
    # A chunk near the top of both rankings beats one in a single list.
    lexical = [
        retrieval.Hit("a", 1.0, lexical_score = 1.0),
        retrieval.Hit("b", 0.5, lexical_score = 0.5),
    ]
    dense = [
        retrieval.Hit("a", 0.9, dense_score = 0.9),
        retrieval.Hit("c", 0.8, dense_score = 0.8),
    ]
    fused = retrieval._rrf([lexical, dense], rrf_k = 60, top_k = 10)
    assert fused[0].chunk_id == "a"
    assert fused[0].lexical_score == 1.0 and fused[0].dense_score == 0.9


def test_retrieve_hybrid_returns_relevant_chunk(rag_conn, bow_embeddings):
    _add_doc(rag_conn, "kb_a", "d1", "f1", "h1", "alpha bravo charlie")
    _add_doc(rag_conn, "kb_a", "d2", "f2", "h2", "golf hotel delta")
    hits = retrieval.retrieve_hybrid(rag_conn, "kb_a", "alpha bravo", k = 5)
    assert hits[0].chunk_id == "d1:0"


def test_retrieve_dense_round_trips(rag_conn, bow_embeddings):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", "alpha alpha")
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", "hotel golf")
    hits = retrieval.retrieve_dense(rag_conn, "kb_a", "alpha", 5)
    assert hits[0].chunk_id == "d1:0"
    assert hits[0].dense_score is not None and hits[0].dense_score > 0.99


def test_filter_min_score_gates_dense_hits():
    hits = [
        retrieval.Hit("a", 1.0, dense_score = 0.9),
        retrieval.Hit("b", 0.5, dense_score = 0.2),
        retrieval.Hit("c", 0.4, lexical_score = 0.4),  # no dense_score -> kept
    ]
    out = retrieval.filter_min_score(hits, 0.5)
    ids = {h.chunk_id for h in out}
    assert ids == {"a", "c"}  # b below floor, c lexical-only passes
    assert retrieval.filter_min_score(hits, 0.0) == hits  # floor off = identity


def test_tool_kb_scope_wins_over_thread(rag_conn, bow_embeddings, monkeypatch):
    seen = {}

    def fake(conn, scope, q, **k):
        seen["scope"] = scope
        return []

    monkeypatch.setattr(retrieval, "retrieve_hybrid", fake)
    tool.search_knowledge_base(query = "q", scope_kb_id = "K", scope_thread_id = "T")
    assert seen["scope"] == "kb_K"


def test_tool_empty_query_errors(rag_home):
    assert tool.search_knowledge_base(query = "   ").startswith("Error")


def test_tool_missing_scope_message(rag_home):
    out = tool.search_knowledge_base(query = "hello")
    assert "No documents" in out


def test_tool_formats_chunks_and_sources(rag_conn, bow_embeddings, monkeypatch):
    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", "body text here", page = 3)
    monkeypatch.setattr(
        retrieval,
        "retrieve_hybrid",
        lambda conn, scope, q, **k: [retrieval.Hit("d1:0", 1.0)],
    )
    text, sources = tool.search_knowledge_base_with_sources(query = "q", scope_kb_id = "a")
    assert '<chunk id="1" source="paper.pdf" page="3">' in text
    assert "body text here" in text
    assert sources == [
        {
            "citationId": 1,
            "chunkId": "d1:0",
            "documentId": "d1",
            "filename": "paper.pdf",
            "page": 3,
            "text": "body text here",
            "score": 1.0,
        }
    ]


def test_tool_kb_scope_retrieves_from_db(rag_conn, bow_embeddings):
    # End-to-end (no retrieve stub): doc found via its scope_kb_id (#8).
    _add_doc(rag_conn, "kb_K", "d1", "kb.pdf", "h1", "alpha bravo charlie", page = 1)
    text, sources = tool.search_knowledge_base_with_sources(
        query = "alpha bravo", scope_kb_id = "K"
    )
    assert "No matching chunks" not in text
    assert sources and sources[0]["chunkId"] == "d1:0"
    assert sources[0]["filename"] == "kb.pdf"
    # A different KB id sees nothing (scope isolation).
    other, other_sources = tool.search_knowledge_base_with_sources(
        query = "alpha bravo", scope_kb_id = "OTHER"
    )
    assert other_sources == [] and "No matching chunks" in other


def test_dispatcher_appends_sources_sentinel(rag_conn, bow_embeddings, monkeypatch):
    # JSON source-map appended after the sentinel; text before it stays clean.
    import json

    from core.inference import tools

    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", "body text here", page = 3)
    monkeypatch.setattr(
        retrieval,
        "retrieve_hybrid",
        lambda conn, scope, q, **k: [retrieval.Hit("d1:0", 1.0)],
    )
    out = tools._search_knowledge_base({"query": "q"}, {"kb_id": "a"})
    assert tools.RAG_SOURCES_SENTINEL in out
    model_text, _, payload = out.partition(tools.RAG_SOURCES_SENTINEL)
    assert "__RAG_SOURCES__" not in model_text  # model never sees the JSON
    assert '<chunk id="1"' in model_text
    sources = json.loads(payload)
    assert sources[0]["documentId"] == "d1"
    assert sources[0]["chunkId"] == "d1:0"
    assert sources[0]["page"] == 3


def test_dispatcher_no_sentinel_when_no_hits(rag_home, monkeypatch):
    from core.inference import tools

    monkeypatch.setattr(retrieval, "retrieve_hybrid", lambda conn, scope, q, **k: [])
    out = tools._search_knowledge_base({"query": "hello"}, {"kb_id": "missing"})
    assert tools.RAG_SOURCES_SENTINEL not in out


def test_search_for_autoinject_gates_on_dense_score(
    rag_conn, bow_embeddings, monkeypatch
):
    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", "body text here", page = 3)

    def _hits(score, **kw):
        return lambda conn, scope, q, **k: [
            retrieval.Hit("d1:0", 1.0, **{kw["key"]: score})
        ]

    # Strong dense hit -> injected.
    monkeypatch.setattr(retrieval, "retrieve_hybrid", _hits(0.8, key = "dense_score"))
    found = tool.search_for_autoinject(query = "q", scope_kb_id = "a", min_dense_score = 0.55)
    assert found is not None
    text, sources = found
    assert '<chunk id="1"' in text and sources[0]["chunkId"] == "d1:0"

    # Dense below floor -> nothing injected.
    monkeypatch.setattr(retrieval, "retrieve_hybrid", _hits(0.30, key = "dense_score"))
    assert (
        tool.search_for_autoinject(query = "q", scope_kb_id = "a", min_dense_score = 0.55)
        is None
    )

    # Lexical-only hit (no dense score) does not auto-inject.
    monkeypatch.setattr(retrieval, "retrieve_hybrid", _hits(1.0, key = "lexical_score"))
    assert (
        tool.search_for_autoinject(query = "q", scope_kb_id = "a", min_dense_score = 0.55)
        is None
    )


def test_search_for_autoinject_bm25_gates_on_dense_probe(
    rag_conn, bow_embeddings, monkeypatch
):
    # BM25 hits carry no cosine, so the gate uses a dense 1-NN probe (#5).
    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", "body text here", page = 3)
    monkeypatch.setattr(
        retrieval,
        "retrieve_hybrid",
        lambda conn, scope, q, **k: [retrieval.Hit("d1:0", 1.0, lexical_score = 2.5)],
    )

    monkeypatch.setattr(
        retrieval,
        "retrieve_dense",
        lambda conn, scope, q, k = None, **kw: [
            retrieval.Hit("d1:0", 0.82, dense_score = 0.82)
        ],
    )
    found = tool.search_for_autoinject(
        query = "q", scope_kb_id = "a", mode = "lexical", min_dense_score = 0.70
    )
    assert found is not None and found[1][0]["chunkId"] == "d1:0"

    monkeypatch.setattr(
        retrieval,
        "retrieve_dense",
        lambda conn, scope, q, k = None, **kw: [
            retrieval.Hit("d1:0", 0.40, dense_score = 0.40)
        ],
    )
    assert (
        tool.search_for_autoinject(
            query = "q", scope_kb_id = "a", mode = "lexical", min_dense_score = 0.70
        )
        is None
    )


def test_search_for_autoinject_empty_query_or_scope(rag_home):
    assert tool.search_for_autoinject(query = "  ", scope_kb_id = "a") is None
    assert tool.search_for_autoinject(query = "hello") is None  # no scope


def test_build_rag_autoinject_emits_pipeline(monkeypatch):
    # Auto-inject yields the same tool card + source-map a real call would.
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    monkeypatch.setattr(
        tool,
        "search_for_autoinject",
        lambda **k: (
            '<chunk id="1" source="d.pdf">hi</chunk>',
            [{"citationId": 1, "filename": "d.pdf"}],
        ),
    )
    conv = [{"role": "user", "content": "When was DeepSeek V4 released?"}]
    out = tools.build_rag_autoinject(conv, {"thread_id": "t1"})
    assert out is not None
    kinds = [e["type"] for e in out["events"]]
    assert "tool_start" in kinds and "tool_end" in kinds
    te = next(e for e in out["events"] if e["type"] == "tool_end")
    assert te["tool_name"] == "search_knowledge_base"
    assert tools.RAG_SOURCES_SENTINEL in te["result"]
    assert (
        out["messages"][0]["tool_calls"][0]["function"]["name"]
        == "search_knowledge_base"
    )
    assert "__RAG_SOURCES__" not in out["messages"][1]["content"]


def test_build_rag_autoinject_skips_without_hit(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    monkeypatch.setattr(tool, "search_for_autoinject", lambda **k: None)
    assert (
        tools.build_rag_autoinject(
            [{"role": "user", "content": "hi"}], {"thread_id": "t1"}
        )
        is None
    )


def test_build_rag_autoinject_enabled_by_default(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.delenv("RAG_AUTOINJECT", raising = False)
    monkeypatch.delenv("RAG_AUTOINJECT_MIN_SCORE", raising = False)
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    seen: dict = {}

    def fake(**k):
        seen.update(k)
        return ("x", [{"citationId": 1}])

    monkeypatch.setattr(tool, "search_for_autoinject", fake)
    out = tools.build_rag_autoinject(
        [{"role": "user", "content": "hi"}], {"thread_id": "t1"}
    )
    assert out is not None
    assert seen["min_dense_score"] == 0.70  # high-precision floor by default


def test_build_rag_autoinject_caps_top_k(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setenv("RAG_AUTOINJECT", "1")
    monkeypatch.setenv("RAG_AUTOINJECT_TOP_K", "4")
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    seen: dict = {}

    def fake(**k):
        seen.update(k)
        return ("x", [{"citationId": 1}])

    monkeypatch.setattr(tool, "search_for_autoinject", fake)
    conv = [{"role": "user", "content": "q"}]
    tools.build_rag_autoinject(conv, {"thread_id": "t1"})
    assert seen["top_k"] == 4  # lean default
    tools.build_rag_autoinject(conv, {"thread_id": "t1", "default_top_k": 2})
    assert seen["top_k"] == 2  # lower user setting wins


def test_build_rag_autoinject_disabled_by_env(monkeypatch):
    from core.inference import tools

    monkeypatch.setenv("RAG_AUTOINJECT", "0")
    assert (
        tools.build_rag_autoinject(
            [{"role": "user", "content": "hi"}], {"thread_id": "t1"}
        )
        is None
    )
    # No scope -> also a no-op.
    monkeypatch.delenv("RAG_AUTOINJECT", raising = False)
    assert tools.build_rag_autoinject([{"role": "user", "content": "hi"}], None) is None


def test_retrieve_hybrid_mode_selects_backend(monkeypatch):
    # ``mode`` runs only the chosen backend; hybrid uses config counts + rrf_k.
    calls: list = []
    monkeypatch.setattr(
        retrieval,
        "retrieve_lexical",
        lambda c, s, q, k = None: calls.append(("lex", k)) or [],
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_dense",
        lambda c, s, q, k = None, *, model_name = None: calls.append(("dense", k)) or [],
    )
    monkeypatch.setattr(
        retrieval,
        "_rrf",
        lambda rankings, rrf_k, top_k: calls.append(("rrf", rrf_k, top_k)) or [],
    )

    calls.clear()
    retrieval.retrieve_hybrid(None, "kb_a", "q", k = 5, mode = "lexical")
    assert [c[0] for c in calls] == ["lex"]  # dense + rrf skipped

    calls.clear()
    retrieval.retrieve_hybrid(None, "kb_a", "q", k = 5, mode = "dense")
    assert [c[0] for c in calls] == ["dense"]

    calls.clear()
    retrieval.retrieve_hybrid(None, "kb_a", "q", k = 5, mode = "hybrid")
    # Candidate pools + rrf_k come from config (no per-request override).
    assert ("lex", config.TOP_K_LEXICAL) in calls
    assert ("dense", config.TOP_K_DENSE) in calls
    rrf = next(c for c in calls if c[0] == "rrf")
    assert rrf[1] == config.RRF_K and rrf[2] == 5  # config rrf_k + final top_k


def test_scope_overrides_reach_retrieval(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    seen: dict = {}

    def fake_search(**kw):
        seen.update(kw)
        return ("text", [])

    monkeypatch.setattr(tool, "search_knowledge_base_with_sources", fake_search)
    tools._search_knowledge_base(
        {"query": "q"},
        {"kb_id": "a", "mode": "dense", "default_top_k": 11},
    )
    assert seen["mode"] == "dense"
    assert seen["top_k"] == 11
    # Unknown mode falls back to hybrid.
    seen.clear()
    tools._search_knowledge_base({"query": "q"}, {"kb_id": "a", "mode": "bogus"})
    assert seen["mode"] == "hybrid"


def test_build_rag_autoinject_scope_overrides_env(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    seen: dict = {}

    def fake_autoinject(**k):
        seen.update(k)
        return ('<chunk id="1" source="d.pdf">hi</chunk>', [{"citationId": 1}])

    monkeypatch.setattr(tool, "search_for_autoinject", fake_autoinject)
    conv = [{"role": "user", "content": "q"}]

    # Scope enables + overrides the floor though env says off.
    monkeypatch.setenv("RAG_AUTOINJECT", "0")
    out = tools.build_rag_autoinject(
        conv,
        {
            "thread_id": "t1",
            "autoinject": True,
            "autoinject_min_score": 0.8,
            "mode": "dense",
        },
    )
    assert out is not None
    assert seen["min_dense_score"] == 0.8
    assert seen["mode"] == "dense"

    # Explicit False disables even with the env default on.
    monkeypatch.setenv("RAG_AUTOINJECT", "1")
    assert (
        tools.build_rag_autoinject(conv, {"thread_id": "t1", "autoinject": False})
        is None
    )

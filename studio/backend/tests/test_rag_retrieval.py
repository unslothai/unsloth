# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retrieval + tool tests: RRF fusion, min-score floor, scope precedence,
chunk formatting and citation source-map. Uses a deterministic bag-of-words
embedder so no model download is needed."""

import math

import pytest

from core.rag import retrieval, store, tool
from core.rag.chunking import Chunk

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@pytest.fixture
def bow_embeddings(monkeypatch):
    """Bag-of-words embedder consistent with the vectors stored in the db."""
    from core.rag import embeddings

    monkeypatch.setattr(
        embeddings,
        "encode",
        lambda texts, *, model_name = None, normalize = True: [_embed(t) for t in texts],
    )
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: len(VOCAB))


def _chunk(text, index = 0, page = None):
    return Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = page,
        source_page_index = 0,
        chunk_index = index,
        page_char_start = 0,
        page_char_end = len(text),
    )


def _add_doc(conn, scope, doc_id, filename, sha, text, page = None):
    store.create_document(
        conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id
    )
    store.add_chunks(conn, scope, doc_id, [_chunk(text, 0, page)], [_embed(text)])


# --------------------------------------------------------------------------
# RRF / hybrid
# --------------------------------------------------------------------------
def test_rrf_ranks_doc_in_both_lists_first():
    # A chunk near the top of both rankings must beat one in only a single list.
    lexical = [
        retrieval.Hit("a", 1.0, lexical_score = 1.0),
        retrieval.Hit("b", 0.5, lexical_score = 0.5),
    ]
    dense = [
        retrieval.Hit("a", 0.9, dense_score = 0.9),
        retrieval.Hit("c", 0.8, dense_score = 0.8),
    ]
    fused = retrieval._rrf([lexical, dense], rrf_k = 60, top_k = 10)
    assert fused[0].chunk_id == "a"  # present in both rankings
    # carried scores survive fusion
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
    assert ids == {"a", "c"}  # b dropped (below floor), c lexical-only passes
    assert retrieval.filter_min_score(hits, 0.0) == hits  # floor off = identity


# --------------------------------------------------------------------------
# Tool: scope precedence, messages, formatting, sources
# --------------------------------------------------------------------------
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


def test_dispatcher_appends_sources_sentinel(rag_conn, bow_embeddings, monkeypatch):
    """tools._search_knowledge_base appends the JSON source-map after the
    sentinel, and the model-facing text before the sentinel is clean."""
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
    # Model never sees the JSON.
    assert "__RAG_SOURCES__" not in model_text
    assert '<chunk id="1"' in model_text
    sources = json.loads(payload)
    assert sources[0]["documentId"] == "d1"
    assert sources[0]["chunkId"] == "d1:0"
    assert sources[0]["page"] == 3


def test_dispatcher_no_sentinel_when_no_hits(rag_home, monkeypatch):
    """No sentinel is appended when the search returns no sources."""
    from core.inference import tools

    # Stub retrieval so the test never reaches the real embedder (keeps it
    # runnable in environments without sentence-transformers installed).
    monkeypatch.setattr(retrieval, "retrieve_hybrid", lambda conn, scope, q, **k: [])
    out = tools._search_knowledge_base({"query": "hello"}, {"kb_id": "missing"})
    assert tools.RAG_SOURCES_SENTINEL not in out

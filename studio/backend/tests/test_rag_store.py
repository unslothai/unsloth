# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Store tests: incremental writes, dedupe, delete, scope, dense + lexical."""

import math

from core.rag import store
from core.rag.chunking import Chunk

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


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


def _add_doc(conn, scope, doc_id, filename, sha, texts):
    chunks = [_chunk(t, i) for i, t in enumerate(texts)]
    vectors = [embed(t) for t in texts]
    store.create_document(
        conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id
    )
    store.add_chunks(conn, scope, doc_id, chunks, vectors)


def test_lexical_returns_only_matching_docs(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    _add_doc(rag_conn, "kb_a", "d2", "d2.txt", "h2", ["golf hotel india"])
    hits = store.search_lexical(rag_conn, "kb_a", "alpha", 10)
    assert [cid for cid, _ in hits] == ["d1:0"]  # d2 not returned (score 0)


def test_scope_isolation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    _add_doc(rag_conn, "kb_b", "d2", "f", "h2", ["alpha bravo"])
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_b", "alpha", 10)] == [
        "d2:0"
    ]


def test_match_query_sanitizes_special_chars():
    assert store._match_query('AND OR "quote" (paren) -dash') != ""


def test_lexical_does_not_crash_on_punctuation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    # Must not raise on FTS operators in the query.
    store.search_lexical(rag_conn, "kb_a", 'NEAR("x" AND', 5)


def test_dense_ranks_by_cosine(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha alpha"])
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", ["hotel golf"])
    ranked = store.search_dense(rag_conn, "kb_a", embed("alpha"), 10)
    assert ranked[0][0] == "d1:0" and ranked[0][1] > 0.99


def test_dense_empty_before_any_ingest(rag_conn):
    # No chunks_vec table yet -> [], no crash.
    assert store.search_dense(rag_conn, "kb_a", embed("alpha"), 10) == []


def test_dedupe_by_hash(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "SHA", ["alpha"])
    assert store.document_by_hash(rag_conn, "kb_a", "SHA") == "d1"
    assert store.document_by_hash(rag_conn, "kb_a", "OTHER") is None


def test_delete_document_purges_all_tables(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    store.delete_document(rag_conn, "d1")
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []
    assert store.search_dense(rag_conn, "kb_a", embed("alpha"), 10) == []
    assert store.chunks_by_id(rag_conn, ["d1:0"]) == {}
    assert store.get_document(rag_conn, "d1") is None


def test_incremental_add_is_flat(rag_conn):
    # Adding doc2 must not touch doc1's fts rowids (append, not rebuild).
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo charlie"])
    before = rag_conn.execute(
        "SELECT rowid, chunk_id FROM chunks_fts WHERE scope='kb_a'"
    ).fetchall()
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", ["delta echo foxtrot"])
    after = rag_conn.execute(
        "SELECT rowid, chunk_id FROM chunks_fts WHERE scope='kb_a' AND chunk_id LIKE 'd1:%'"
    ).fetchall()
    before_d1 = [
        (r["rowid"], r["chunk_id"]) for r in before if r["chunk_id"].startswith("d1:")
    ]
    after_d1 = [(r["rowid"], r["chunk_id"]) for r in after]
    assert before_d1 == after_d1


def test_chunks_by_id_joins_filename(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", ["body text here"])
    rows = store.chunks_by_id(rag_conn, ["d1:0"])
    assert rows["d1:0"]["filename"] == "paper.pdf"
    assert rows["d1:0"]["text"] == "body text here"


def test_kb_crud_and_delete_cascades(rag_conn):
    kb_id = store.create_kb(rag_conn, name = "My KB", description = "d", kb_id = "K1")
    assert store.get_kb(rag_conn, kb_id)["name"] == "My KB"
    assert [k["id"] for k in store.list_kbs(rag_conn)] == ["K1"]

    scope = store.kb_scope("K1")
    _add_doc(rag_conn, scope, "doc1", "f", "h1", ["alpha bravo"])
    store.delete_kb(rag_conn, "K1")
    assert store.get_kb(rag_conn, "K1") is None
    assert store.list_documents(rag_conn, scope) == []
    assert store.search_lexical(rag_conn, scope, "alpha", 10) == []

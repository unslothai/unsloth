# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunking asks a sentence-transformers counter for token counts in batches. The stored rows must be
exactly those of one-at-a-time counting, with far fewer tokenizer calls."""

import os
import random
import threading
import time

import pytest

from core.rag import chunking, embeddings, ingestion, store
from core.rag.chunking import chunk_pages
from core.rag.parsers import Page
from storage import rag_db

import sqlite3

# Same gate as the rag_conn fixture: an interpreter whose sqlite3 cannot load extensions (the
# macos-15 runner's) never has RAG, so the end-to-end ingestion tests skip there.
needs_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE or not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "sqlite-vec cannot be loaded by this Python's sqlite3",
)

_REAL_TOKEN_COUNTER = embeddings.token_counter


class _Tok:
    """Fast-tokenizer stand-in whose counts are not additive over concatenation, like real BPE."""

    is_fast = True

    def __init__(self):
        self.encodes = 0
        self.batches = 0

    @staticmethod
    def _ids(text):
        return [0] * (len(text.split()) + len(text) % 3 + text.count("##"))

    def encode(
        self,
        text,
        add_special_tokens = False,
    ):
        self.encodes += 1
        return self._ids(text)

    def __call__(
        self,
        texts,
        add_special_tokens = False,
        **kwargs,
    ):
        self.batches += 1
        return {"input_ids": [self._ids(t) for t in texts]}


class _Model:
    def __init__(self, tok):
        self.tokenizer = tok


def _one_at_a_time(count):
    """The same counter without its batch form, so chunking takes the per-text path."""
    return lambda text: count(text)


def _document(seed = 7, pages = 12):
    rng = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]

    def sentence():
        return " ".join(rng.choice(words) for _ in range(rng.randint(3, 30))) + ". "

    out = []
    for n in range(pages):
        parts = [f"# Section {n}\n"]
        for _ in range(rng.randint(1, 6)):
            parts.append("## Sub\n" if rng.random() < 0.3 else "")
            parts.append("".join(sentence() for _ in range(rng.randint(1, 12))) + "\n\n")
        # Unbroken runs reach the character and fixed-slice levels.
        parts.append("x" * rng.randint(0, 400) + "\n")
        parts.append(" \n\r\n" if rng.random() < 0.3 else "")
        out.append(Page(text = "".join(parts), page_number = n + 1, char_count = 0))
    return out


@pytest.mark.parametrize("max_tokens,overlap", [(8, 0), (24, 6), (64, 16), (500, 64)])
def test_batched_chunks_equal_one_at_a_time_chunks(monkeypatch, max_tokens, overlap):
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(_Tok()))
    pages = _document()
    batched = chunk_pages(
        pages,
        max_tokens = max_tokens,
        overlap = overlap,
        count = embeddings._st_token_counter("org/embedder"),
    )
    serial = chunk_pages(
        pages,
        max_tokens = max_tokens,
        overlap = overlap,
        count = _one_at_a_time(embeddings._st_token_counter("org/embedder")),
    )
    assert batched == serial
    assert len(batched) > len(pages)


def test_batched_count_matches_single_counts_and_memo(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    count = embeddings._st_token_counter("org/embedder")
    texts = ["a b c", "d e", "a b c", "", "f g h i j"] * 200
    assert count.many(texts) == [len(_Tok._ids(t)) for t in texts]
    encodes = tok.encodes
    batches = tok.batches
    # Memoized: a repeat costs no tokenizer call, batched or single.
    assert count.many(["a b c", "d e"]) == [5, 2]
    assert count("f g h i j") == 5
    assert (tok.encodes, tok.batches) == (encodes, batches)
    assert count.many([]) == []


def test_batched_count_leaves_the_same_environment(monkeypatch):
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(_Tok()))
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    count = embeddings._st_token_counter("org/embedder")
    count.many(["x y", "z"])
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    count.many(["x y", "z"])
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_a_backend_without_a_batch_form_is_counted_one_text_at_a_time(monkeypatch):
    seen = []

    class _Backend:
        def token_counter(self, *, model_name = None):
            return lambda text: seen.append(text) or len(text.split())

    monkeypatch.setattr(embeddings, "_get_backend", lambda model_name = None: _Backend())
    count = _REAL_TOKEN_COUNTER("org/embedder")
    assert not hasattr(count, "many")
    pages = _document(pages = 3)
    chunk_pages(pages, max_tokens = 16, overlap = 4, count = count)
    serial_order = list(seen)
    seen.clear()
    chunk_pages(pages, max_tokens = 16, overlap = 4, count = _one_at_a_time(count))
    assert seen == serial_order


def _ingest_with(monkeypatch, tmp_path, counter_factory, name):
    path = tmp_path / name
    path.write_text("\n\n".join(p.text for p in _document(seed = 11, pages = 40)), encoding = "utf-8")
    monkeypatch.setattr(embeddings, "token_counter", counter_factory)
    monkeypatch.setattr(embeddings, "chunk_tokens", lambda model_name = None: 32)
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, name, str(path), background = False)
    assert ingestion.get_job_status(job_id)["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        chunks = [
            tuple(r)[1:]
            for r in conn.execute(
                "SELECT id, chunk_index, text, page_number, source_page_index, page_char_start, "
                "page_char_end, token_count, whole_document_token_count, kind, pdf_regions_json "
                "FROM chunks WHERE document_id=? ORDER BY chunk_index",
                (doc_id,),
            )
        ]
        fts = [
            (r["text"], r["chunk_id"].replace(doc_id, "D"))
            for r in conn.execute(
                "SELECT text, chunk_id FROM chunks_fts WHERE chunk_id LIKE ? ORDER BY rowid",
                (f"{doc_id}:%",),
            )
        ]
        vectors = [
            (r["chunk_id"].replace(doc_id, "D"), bytes(r["embedding"]))
            for r in conn.execute(
                "SELECT chunk_id, embedding FROM chunks_vec WHERE chunk_id LIKE ? ORDER BY rowid",
                (f"{doc_id}:%",),
            )
        ]
        store.delete_document(conn, doc_id)
    finally:
        conn.close()
    return chunks, fts, vectors


@needs_sqlite_vec
def test_ingestion_stores_the_same_rows_batched_or_not(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    batched = _ingest_with(
        monkeypatch,
        tmp_path,
        lambda model_name = None: embeddings._st_token_counter(model_name),
        "batched.md",
    )
    batched_calls = tok.encodes + tok.batches
    tok.encodes = tok.batches = 0
    serial = _ingest_with(
        monkeypatch,
        tmp_path,
        lambda model_name = None: _one_at_a_time(embeddings._st_token_counter(model_name)),
        "serial.md",
    )
    serial_calls = tok.encodes + tok.batches
    assert batched == serial
    assert len(batched[0]) > 100 and len(batched[1]) == len(batched[2]) == len(batched[0])
    # Work count: one batched tokenize per split level and stage, not one per piece.
    assert batched_calls * 10 < serial_calls
    assert batched_calls < 40


@needs_sqlite_vec
def test_ingestion_tokenizes_a_document_in_few_calls(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """Fails on a per-piece counter: through the real token_counter, a 40-page document costs a handful
    of tokenizer calls (each with one environment swap) instead of one per piece."""
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    monkeypatch.setattr(
        embeddings,
        "_get_backend",
        lambda model_name = None: embeddings._SentenceTransformersBackend(),
    )
    chunks, _, _ = _ingest_with(monkeypatch, tmp_path, _REAL_TOKEN_COUNTER, "doc.md")
    assert len(chunks) > 100
    assert tok.encodes + tok.batches < 40


@needs_sqlite_vec
def test_delete_during_batched_chunking_cancels_and_stores_nothing(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    monkeypatch.setattr(
        embeddings, "token_counter", lambda model_name = None: embeddings._st_token_counter(None)
    )
    path = tmp_path / "doc.md"
    path.write_text("\n\n".join(p.text for p in _document(pages = 20)), encoding = "utf-8")
    scope = store.kb_scope("K1")
    known = {}
    doc_id_known = threading.Event()
    real_chunk_pages = chunking.chunk_pages

    def delete_while_chunking(*args, **kwargs):
        assert hasattr(kwargs["count"], "many")
        out = real_chunk_pages(*args, **kwargs)
        doc_id_known.wait(30)
        conn = rag_db.get_connection()
        try:
            store.delete_document(conn, known["id"])
        finally:
            conn.close()
        return out

    monkeypatch.setattr(chunking, "chunk_pages", delete_while_chunking)
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.md", str(path))
    known["id"] = doc_id
    doc_id_known.set()
    deadline = time.time() + 30
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert status["status"] == "cancelled"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id) is None
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    finally:
        conn.close()


def test_add_chunks_refuses_unpaired_vectors(rag_conn):
    from core.rag.chunking import Chunk

    chunks = [
        Chunk(
            text = f"alpha {i}",
            token_count = 2,
            page_number = None,
            source_page_index = 0,
            chunk_index = i,
            page_char_start = 0,
            page_char_end = 7,
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match = "3 chunks but 2 vectors"):
        store.add_chunks(rag_conn, "kb_x", "doc", chunks, [[0.1, 0.2], [0.3, 0.4]])
    rag_conn.rollback()
    assert rag_conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert rag_conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0

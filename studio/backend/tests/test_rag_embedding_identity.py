# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``documents.embedding_model`` names the embedder, not just the model.

llama-server ignores the configured model name and embeds through its GGUF companion
with its own pooling, so the same name can mean two vector spaces on one machine. An
index written by one backend must not silently answer the other's queries.
"""

from core.rag import config, embeddings, ingestion, retrieval, store
from storage import rag_db

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding = "utf-8")
    return str(path)


def _ingest(tmp_path, scope, name, text):
    document_id, _ = ingestion.start_ingestion(
        scope = scope,
        kb_id = None,
        thread_id = None,
        filename = name,
        stored_path = _write(tmp_path, name, text),
        model_name = MODEL,
        background = False,
    )
    return document_id


def test_index_written_by_llama_is_stale_for_a_sentence_transformers_query(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """The scenario a CPU upgrade produces: documents embedded through the GGUF while
    sentence-transformers was failing, then queried by sentence-transformers once it
    works. Same model name, different pooling."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
    scope = store.kb_scope("K1")
    _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    conn = rag_db.get_connection()
    try:
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL) == []
    finally:
        conn.close()


def test_index_written_by_sentence_transformers_is_stale_for_a_llama_query(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """And the other direction, which is what a runtime fallback produces."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    scope = store.kb_scope("K2")
    _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
    conn = rag_db.get_connection()
    try:
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL) == []
    finally:
        conn.close()


def test_the_same_backend_still_answers_its_own_index(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """The filter must only drop the other backend's rows."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
    scope = store.kb_scope("K3")
    _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")
    conn = rag_db.get_connection()
    try:
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL)
    finally:
        conn.close()


def test_re_uploading_after_a_backend_change_reindexes_instead_of_deduping(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """Identical bytes used to dedupe against a row from the other backend, so there
    was no way to repair the index short of renaming the file."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
    scope = store.kb_scope("K4")
    first = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    second = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")
    assert second != first
    conn = rag_db.get_connection()
    try:
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL)
    finally:
        conn.close()


def test_ingestion_records_the_backend_that_took_over_mid_job(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """The row is created before the encode, and an ST encode failure swaps the
    process to llama-server, so the identity is only correct once vectors exist."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    swapped = {"done": False}
    real_encode = embeddings.encode

    def encode_then_swap(texts, **kwargs):
        swapped["done"] = True
        monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode", encode_then_swap)
    scope = store.kb_scope("K5")
    document_id = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")
    assert swapped["done"]
    conn = rag_db.get_connection()
    try:
        stored = store.get_document(conn, document_id)["embedding_model"]
    finally:
        conn.close()
    assert stored.startswith("llama-server:")


def test_legacy_rows_keep_answering_and_are_reported(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """Rows written before the tag existed could be either backend's. Dropping them
    would empty dense search over every corpus indexed so far, and re-embedding one
    unasked is not kinder, so they are still served and counted instead."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    scope = store.kb_scope("K6")
    document_id = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")
    conn = rag_db.get_connection()
    try:
        store.set_document_embedding_model(conn, document_id, MODEL)  # pre-tag spelling
        assert store.count_untagged_documents(conn) == 1
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL)
        monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL)
    finally:
        conn.close()


def test_null_rows_are_still_assumed_current(rag_home, stub_embeddings, monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    scope = store.kb_scope("K7")
    document_id = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie")
    conn = rag_db.get_connection()
    try:
        conn.execute("UPDATE documents SET embedding_model=NULL WHERE id=?", (document_id,))
        conn.commit()
        assert store.count_untagged_documents(conn) == 0
        assert retrieval.retrieve_dense(conn, scope, "alpha bravo", k = 5, model_name = MODEL)
    finally:
        conn.close()


def test_identity_distinguishes_the_backends_and_the_gguf_repo(monkeypatch):
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    as_st = embeddings.embedding_identity(MODEL)
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
    as_llama = embeddings.embedding_identity(MODEL)
    assert as_st != as_llama
    assert config.embedding_identity_model(as_st) == MODEL
    assert config.embedding_identity_model(as_llama) == MODEL
    monkeypatch.setattr(config, "EMBED_GGUF_REPO", "LLukas22/all-MiniLM-L6-v2-GGUF")
    monkeypatch.setenv("RAG_EMBED_GGUF_REPO", "LLukas22/all-MiniLM-L6-v2-GGUF")
    assert embeddings.embedding_identity(MODEL) != as_llama


def test_untagged_values_match_on_the_model_name_alone():
    tagged = config.embedding_identity("sentence-transformers", MODEL)
    assert config.embedding_identity_matches(None, tagged) is True
    assert config.embedding_identity_matches(MODEL, tagged) is True
    assert config.embedding_identity_matches("other/model", tagged) is False
    assert config.embedding_identity_matches(tagged, tagged) is True
    assert (
        config.embedding_identity_matches(
            config.embedding_identity("llama-server", MODEL, gguf_repo = "r"), tagged
        )
        is False
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``documents.embedding_model`` names the embedder, not just the model.

llama-server ignores the configured model name and embeds through its GGUF companion
with its own pooling, so the same name can mean two vector spaces on one machine. An
index written by one backend must not silently answer the other's queries.
"""

import math

from core.rag import config, embeddings, ingestion, retrieval, store
from core.rag.chunking import Chunk
from storage import rag_db

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# A model may also be a local path, and on Windows that carries the separator.
WINDOWS_MODEL = r"C:\models\bge-small-en-v1.5"

_VOCAB = ["alpha", "bravo", "charlie", "delta"]


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


def _vector(text):
    v = [float(text.lower().count(w)) for w in _VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _put(conn, scope, document_id, texts, embedding_model):
    """Index one document's chunks directly, under a chosen identity."""
    chunks = [
        Chunk(
            text = t,
            token_count = len(t.split()),
            page_number = None,
            source_page_index = 0,
            chunk_index = i,
            page_char_start = 0,
            page_char_end = len(t),
        )
        for i, t in enumerate(texts)
    ]
    store.create_document(
        conn,
        scope = scope,
        filename = f"{document_id}.txt",
        sha256 = document_id,
        status = "completed",
        document_id = document_id,
        embedding_model = embedding_model,
    )
    store.add_chunks(conn, scope, document_id, chunks, [_vector(t) for t in texts])


def test_a_local_model_path_survives_the_identity_round_trip():
    """Colons separate the identity's segments and a Windows path carries one, so the
    model used to read back as ``C``."""
    st = config.embedding_identity("sentence-transformers", WINDOWS_MODEL)
    llama = config.embedding_identity(
        "llama-server", WINDOWS_MODEL, gguf_repo = WINDOWS_MODEL + "-GGUF"
    )
    assert st != llama
    assert config.embedding_identity_model(st) == WINDOWS_MODEL
    assert config.embedding_identity_model(llama) == WINDOWS_MODEL
    # The pre-tag spelling of such a row is the bare path, and it still has to match.
    assert config.embedding_identity_matches(WINDOWS_MODEL, st) is True
    assert config.embedding_identity_matches(r"C:\models\other", st) is False


def test_a_local_path_model_keeps_answering_its_legacy_rows(rag_conn):
    """The upgrade must not empty dense search for a corpus indexed under a path."""
    _put(rag_conn, "kb_w", "d1", ["alpha bravo"], WINDOWS_MODEL)
    hits = store.search_dense(
        rag_conn,
        "kb_w",
        _vector("alpha bravo"),
        5,
        embedding_model = config.embedding_identity("sentence-transformers", WINDOWS_MODEL),
    )
    assert [cid for cid, _ in hits] == ["d1:0"]


def test_stale_backend_chunks_do_not_starve_the_current_backend(rag_conn):
    """What a partial reindex leaves: both backends in one scope. The other one's
    distances come from another space, so they can fill every fetched candidate slot
    while the compatible chunks sit further down the KNN list."""
    stale = config.embedding_identity("llama-server", MODEL, gguf_repo = "r")
    current = config.embedding_identity("sentence-transformers", MODEL)
    for i in range(40):
        _put(rag_conn, "kb_s", f"old{i}", ["alpha"], stale)
    _put(rag_conn, "kb_s", "new", ["alpha bravo"], current)
    hits = store.search_dense(rag_conn, "kb_s", _vector("alpha"), 5, embedding_model = current)
    assert [cid for cid, _ in hits] == ["new:0"]


def test_the_identity_comes_from_the_encode_not_from_the_process_after_it(rag_conn, monkeypatch):
    """A concurrent ST failure swaps the process embedder for the rest of its life. A
    query sentence-transformers had already encoded must still be answered by the
    sentence-transformers half of the index, not by the backend that took over."""
    _put(
        rag_conn,
        "kb_r",
        "d1",
        ["alpha bravo"],
        config.embedding_identity("sentence-transformers", MODEL),
    )

    class _SwapsMidEncode:
        def encode(
            self,
            texts,
            *,
            model_name = None,
            normalize = True,
        ):
            vectors = [_vector(t) for t in texts]
            monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
            return vectors

    monkeypatch.setattr(embeddings, "_backend", _SwapsMidEncode())
    monkeypatch.setattr(
        embeddings, "_backend_key", (config.EMBED_BACKEND or "auto").strip().lower()
    )
    hits = retrieval.retrieve_dense(rag_conn, "kb_r", "alpha bravo", k = 5, model_name = MODEL)
    assert [h.chunk_id for h in hits] == ["d1:0"]


def test_a_swap_between_batches_re_embeds_the_document(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """A document whose batches straddle the swap would hold vectors from two spaces
    under one identity, so it restarts under the backend that took over."""
    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    monkeypatch.setattr(ingestion, "_EMBED_BATCH", 1)
    monkeypatch.setattr(config, "CHUNK_TOKENS", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    passes = {"n": 0}
    real_pass = ingestion._embed_pass

    def swap_during_the_first_pass(texts, model_name):
        passes["n"] += 1
        if passes["n"] == 1:
            calls = {"n": 0}
            real_encode = embeddings.encode_with_identity

            def swap_after_one_batch(batch, **kwargs):
                calls["n"] += 1
                out = real_encode(batch, **kwargs)
                if calls["n"] == 1:
                    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
                return out

            monkeypatch.setattr(embeddings, "encode_with_identity", swap_after_one_batch)
        return real_pass(texts, model_name)

    monkeypatch.setattr(ingestion, "_embed_pass", swap_during_the_first_pass)
    scope = store.kb_scope("K8")
    document_id = _ingest(tmp_path, scope, "doc.txt", "alpha bravo charlie delta echo foxtrot")
    assert passes["n"] == 2
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, document_id)["embedding_model"].startswith("llama-server:")
    finally:
        conn.close()


def test_widening_survives_a_scope_larger_than_one_parameter_batch(rag_conn):
    """The widened candidate set outgrows a single bound-parameter batch."""
    stale = config.embedding_identity("llama-server", MODEL, gguf_repo = "r")
    current = config.embedding_identity("sentence-transformers", MODEL)
    _put(rag_conn, "kb_b", "old", ["alpha"] * 2000, stale)
    _put(rag_conn, "kb_b", "new", ["alpha bravo"], current)
    hits = store.search_dense(rag_conn, "kb_b", _vector("alpha"), 5, embedding_model = current)
    assert [cid for cid, _ in hits] == ["new:0"]


def test_a_saturated_scope_keeps_widening_after_another_scope_is_full(rag_conn):
    """A project-and-thread search, which is the shape retrieval actually asks for.

    vec0 constrains its partition key by equality, so each scope is its own KNN list
    with its own stale prefix. A thread scope that hands over k compatible but weak
    hits must not stop the project scope widening past the other embedder's vectors
    burying a stronger chunk, or the merge ranks a top-k it never fetched."""
    stale = config.embedding_identity("llama-server", MODEL, gguf_repo = "r")
    current = config.embedding_identity("sentence-transformers", MODEL)
    # The thread scope answers on the first fetch, with the weakest hits in the corpus.
    _put(rag_conn, "thread_t", "weak", ["alpha bravo charlie delta"] * 5, current)
    # The project scope's whole first fetch is another embedder's, and the compatible
    # chunk that outranks every thread hit sits just behind it.
    _put(rag_conn, "kb_p", "old", ["alpha"] * 20, stale)
    _put(rag_conn, "kb_p", "new", ["alpha bravo"], current)
    hits = store.search_dense(
        rag_conn, ["kb_p", "thread_t"], _vector("alpha"), 5, embedding_model = current
    )
    assert hits[0][0] == "new:0"


def test_the_web_ranker_labels_a_page_with_the_backend_that_encoded_it(rag_home, monkeypatch):
    """A concurrent ST failure swaps the process embedder for the rest of its life.

    A page sentence-transformers had already encoded must not be stored as
    llama-server: the hybrid query right below it then searches those mislabeled
    vectors instead of filtering them out."""
    from core.rag import web_rank

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: False)
    monkeypatch.setattr(
        embeddings, "token_counter", lambda model_name = None: lambda t: max(1, len(t.split()))
    )

    class _SwapsMidEncode:
        def encode(
            self,
            texts,
            *,
            model_name = None,
            normalize = True,
        ):
            vectors = [_vector(t) for t in texts]
            monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda: True)
            return vectors

    monkeypatch.setattr(embeddings, "_backend", _SwapsMidEncode())
    monkeypatch.setattr(
        embeddings, "_backend_key", (config.EMBED_BACKEND or "auto").strip().lower()
    )

    labels: list[str | None] = []
    real_create = store.create_document

    def record(conn, **kwargs):
        labels.append(kwargs.get("embedding_model"))
        return real_create(conn, **kwargs)

    monkeypatch.setattr(web_rank.store, "create_document", record)
    web_rank.retrieve_web_chunks(
        [{"text": "alpha bravo charlie", "title": "page", "url": "https://a"}],
        "alpha",
        top_n = 3,
        min_score = 0.0,
        model_name = MODEL,
    )
    assert labels == [config.embedding_identity("sentence-transformers", MODEL)]


def test_resolving_the_embedder_does_not_hold_the_write_lock(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    """Naming the embedder can take seconds on a fresh process: it searches for the
    llama-server binary, runs nvidia-smi with a ten second timeout, and on a host
    without it imports torch. Inside the admission transaction that is a RESERVED
    lock held for all of it, and rag.db opens every connection with a five second
    busy_timeout, so an unrelated ingest or a job heartbeat fails outright with
    "database is locked" rather than waiting."""
    import sqlite3

    from utils.paths import rag_db_path

    probes: list[tuple[bool, str]] = []
    real_identity = embeddings.embedding_identity

    def probing_identity(model_name = None):
        other = sqlite3.connect(str(rag_db_path()))
        try:
            other.execute("PRAGMA busy_timeout = 100")
            other.execute("BEGIN IMMEDIATE")
            other.rollback()
            probes.append((True, ""))
        except sqlite3.OperationalError as exc:
            probes.append((False, str(exc)))
        finally:
            other.close()
        return real_identity(model_name)

    monkeypatch.setattr(embeddings, "embedding_identity", probing_identity)
    _ingest(tmp_path, store.kb_scope("K1"), "doc.txt", "alpha bravo charlie")

    assert probes, "embedding_identity was never called"
    assert all(ok for ok, _ in probes), [err for ok, err in probes if not ok]

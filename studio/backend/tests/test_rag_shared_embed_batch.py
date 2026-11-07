# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Documents ingesting at the same time embed in shared batches (#16715), each keeping its own
embedding identity, progress and failures.

The stub embedder returns instantly, so a document would normally finish before the next one is
parsed. These tests hold the queue's "a document is on its way" count, which is what a real sync
has while other files parse, and raise the linger so the wait is not a race.
"""

import struct
import threading
import time
from contextlib import contextmanager

import pytest

from core.rag import config, embeddings, ingestion, store
from storage import rag_db

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


_real_gate = embeddings.batches_across_documents


@pytest.fixture(autouse = True)
def _embedder_on_an_accelerator(monkeypatch):
    # The stub embedder resolves to sentence-transformers on CPU, which keeps per-document calls.
    monkeypatch.setattr(embeddings, "batches_across_documents", lambda model_name = None: True)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding = "utf-8")
    return str(path)


def _wait_finished(job_id, timeout = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("ingestion did not finish in time")


@contextmanager
def _documents_still_parsing(monkeypatch):
    """Stands in for files a sync is still parsing, so batches wait for them."""
    monkeypatch.setattr(ingestion, "_EMBED_LINGER_SECONDS", 30.0)
    model = config.effective_embedding_model()
    ingestion._embed_queue.expect(model)
    try:
        yield
    finally:
        ingestion._embed_queue.unexpect(model)


def _record_batches(monkeypatch):
    """Returns the list of text batches the embedder is actually called with."""
    batches: list[list[str]] = []
    real = embeddings.encode_with_identity
    lock = threading.Lock()

    def recording(texts, **kwargs):
        with lock:
            batches.append(list(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode_with_identity", recording)
    return batches


def _record_stages(monkeypatch):
    """Returns (job_id, stage, progress) as the job rows are written; SSE queues are reaped."""
    stages: list[tuple[str, str, float]] = []
    real = ingestion._progress
    lock = threading.Lock()

    def recording(conn, job_id, stage, progress):
        with lock:
            stages.append((job_id, stage, progress))
        return real(conn, job_id, stage, progress)

    monkeypatch.setattr(ingestion, "_progress", recording)
    return stages


def _start(
    scope,
    tmp_path,
    name,
    text,
    slots = None,
    model_name = None,
):
    return ingestion.start_ingestion(
        scope,
        "K1",
        None,
        name,
        _write(tmp_path, name, text),
        model_name = model_name,
        stage_slots = slots,
    )


def test_two_documents_embed_in_one_call(rag_home, stub_embeddings, tmp_path, monkeypatch):
    """The batch-of-1 path: before this, a one-chunk file always paid a whole encode call."""
    batches = _record_batches(monkeypatch)
    scope = store.kb_scope("K1")
    # One slot, like a linked-folder sync with a single parse worker.
    slots = threading.Semaphore(1)
    with _documents_still_parsing(monkeypatch):
        jobs = [
            _start(scope, tmp_path, "a.txt", "alpha document words", slots),
            _start(scope, tmp_path, "b.txt", "bravo document words", slots),
        ]
    for _, job_id in jobs:
        assert _wait_finished(job_id)["status"] == "completed"

    assert len(batches) == 1, batches
    assert sorted(text.split()[0] for text in batches[0]) == ["alpha", "bravo"]
    conn = rag_db.get_connection()
    try:
        for document_id, _ in jobs:
            assert store.get_document(conn, document_id)["num_chunks"] == 1
        assert store.search_lexical(conn, scope, "alpha", 5)
        assert store.search_lexical(conn, scope, "bravo", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_a_shared_batch_keeps_each_document_s_own_vectors(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """Vectors must come back on the chunk they were made from, not in the batch's order."""
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(1)
    with _documents_still_parsing(monkeypatch):
        jobs = {
            name: _start(scope, tmp_path, f"{name}.txt", " ".join([name] * 40), slots)
            for name in ("alpha", "bravo", "charlie")
        }
    for _, job_id in jobs.values():
        assert _wait_finished(job_id)["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        for name, (document_id, _) in jobs.items():
            rows = conn.execute(
                "SELECT id, text FROM chunks WHERE document_id=? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
            assert rows and all(name in row["text"] for row in rows)
            for row in rows:
                stored = conn.execute(
                    "SELECT embedding FROM chunks_vec WHERE chunk_id=?", (row["id"],)
                ).fetchone()
                assert stored is not None
                expected = embeddings.encode([row["text"]])[0]
                written = struct.unpack(f"<{len(expected)}f", stored["embedding"])
                assert written == pytest.approx(tuple(expected), abs = 1e-6)
    finally:
        conn.close()


def test_one_failing_document_does_not_fail_the_others(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """A failed shared batch retries per document, so a text the embedder rejects fails alone."""
    real = embeddings.encode_with_identity

    def refuse_bravo(texts, **kwargs):
        if any("bravo" in text for text in texts):
            raise RuntimeError("synthetic encode failure")
        return real(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode_with_identity", refuse_bravo)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(1)
    with _documents_still_parsing(monkeypatch):
        good = _start(scope, tmp_path, "a.txt", "alpha document words", slots)
        bad = _start(scope, tmp_path, "b.txt", "bravo document words", slots)

    assert _wait_finished(good[1])["status"] == "completed"
    assert _wait_finished(bad[1])["status"] == "failed"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, good[0])["status"] == "completed"
        assert store.get_document(conn, bad[0])["status"] == "failed"
        assert store.search_lexical(conn, scope, "alpha", 5)
        assert not store.search_lexical(conn, scope, "bravo", 5)
    finally:
        conn.close()


def test_documents_under_different_models_are_not_batched_together(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """One batch is one encode call, so it can only hold texts for one model."""
    batches = _record_batches(monkeypatch)
    models = []
    real_encode = embeddings.encode

    def note_model(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        models.append(model_name)
        return real_encode(texts, model_name = model_name, normalize = normalize)

    monkeypatch.setattr(embeddings, "encode", note_model)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(1)
    with _documents_still_parsing(monkeypatch):
        jobs = [
            _start(scope, tmp_path, name, f"{name} document words", slots, model_name = model)
            for name, model in (("a.txt", "model/one"), ("b.txt", "model/two"))
        ]
    for _, job_id in jobs:
        assert _wait_finished(job_id)["status"] == "completed"

    assert len(batches) == 2, batches
    assert all(len(batch) == 1 for batch in batches)
    assert sorted(models) == ["model/one", "model/two"]


def test_a_document_with_a_full_batch_of_chunks_embeds_without_waiting(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """It gains nothing from waiting, and holding its chunks for others would cost memory."""
    from core.rag import config

    monkeypatch.setattr(ingestion, "_EMBED_BATCH", 4)
    monkeypatch.setattr(config, "CHUNK_TOKENS", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    batches = _record_batches(monkeypatch)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(1)
    started = time.monotonic()
    with _documents_still_parsing(monkeypatch):
        document_id, job_id = _start(
            scope,
            tmp_path,
            "big.txt",
            " ".join(f"word{i}" for i in range(40)),
            slots,
        )
        assert _wait_finished(job_id, timeout = 20.0)["status"] == "completed"
    # It never waited out the 30 s linger for the documents still parsing.
    assert time.monotonic() - started < 20.0

    assert batches and all(len(batch) <= 4 for batch in batches)
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, document_id)["num_chunks"] > 4
    finally:
        conn.close()


def test_a_waiting_document_reports_its_embedding_progress(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """Progress must move when its slice lands in another document's batch."""
    stages = _record_stages(monkeypatch)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(1)
    with _documents_still_parsing(monkeypatch):
        jobs = [
            _start(scope, tmp_path, f"doc{i}.txt", f"document {i} words", slots) for i in range(3)
        ]
    for _, job_id in jobs:
        assert _wait_finished(job_id)["status"] == "completed"

    for _, job_id in jobs:
        written = [(stage, progress) for job, stage, progress in stages if job == job_id]
        assert "embedding" in [stage for stage, _ in written]
        assert [progress for _, progress in written] == sorted(progress for _, progress in written)


def test_a_backend_swap_inside_a_shared_batch_re_embeds_that_document(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """Vectors from two spaces under one identity are unsearchable, so the document re-embeds
    whole under the backend that took over."""
    from core.rag import config

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda *_a, **_k: False)
    monkeypatch.setattr(ingestion, "_EMBED_BATCH", 2)
    monkeypatch.setattr(config, "CHUNK_TOKENS", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    real = embeddings.encode_with_identity
    swapped = {"done": False}

    def swap_after_the_first_batch(texts, **kwargs):
        out = real(texts, **kwargs)
        if not swapped["done"]:
            swapped["done"] = True
            monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda *_a, **_k: True)
        return out

    monkeypatch.setattr(embeddings, "encode_with_identity", swap_after_the_first_batch)
    scope = store.kb_scope("K1")
    document_id, job_id = _start(
        scope,
        tmp_path,
        "doc.txt",
        " ".join(f"word{i}" for i in range(12)),
    )
    assert _wait_finished(job_id)["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        identity = store.get_document(conn, document_id)["embedding_model"]
    finally:
        conn.close()
    assert identity.startswith("llama-server:"), identity


def test_the_queue_keeps_nothing_after_a_failed_document(rag_home, stub_embeddings, tmp_path):
    """A document that stops early must not leave a slice others would wait on."""
    scope = store.kb_scope("K1")
    empty = _start(scope, tmp_path, "empty.txt", "   ")
    assert _wait_finished(empty[1])["status"] == "failed"
    good = _start(scope, tmp_path, "good.txt", "alpha document words")
    assert _wait_finished(good[1])["status"] == "completed"

    assert not ingestion._embed_queue._pending
    assert not ingestion._embed_queue._expected


@pytest.mark.parametrize("documents", (2, 9))
def test_many_small_documents_need_far_fewer_encode_calls(
    rag_home, stub_embeddings, tmp_path, monkeypatch, documents
):
    batches = _record_batches(monkeypatch)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(2)
    with _documents_still_parsing(monkeypatch):
        jobs = [
            _start(scope, tmp_path, f"note{i}.txt", f"note {i} searchable words", slots)
            for i in range(documents)
        ]
    for _, job_id in jobs:
        assert _wait_finished(job_id)["status"] == "completed"

    assert sum(len(batch) for batch in batches) == documents
    assert len(batches) == 1, batches


def test_cpu_sentence_transformers_keeps_one_call_per_document(
    rag_home, stub_embeddings, tmp_path, monkeypatch
):
    """CPU encode pads every text to the batch's longest, which costs more than the calls it saves."""
    monkeypatch.setattr(embeddings, "batches_across_documents", _real_gate)
    assert not embeddings.batches_across_documents(None)
    batches = _record_batches(monkeypatch)
    scope = store.kb_scope("K1")
    slots = threading.Semaphore(2)
    with _documents_still_parsing(monkeypatch):
        jobs = [
            _start(scope, tmp_path, f"note{i}.txt", f"note {i} searchable words", slots)
            for i in range(4)
        ]
    for _, job_id in jobs:
        assert _wait_finished(job_id)["status"] == "completed"

    assert sorted(len(batch) for batch in batches) == [1, 1, 1, 1], batches
    assert not ingestion._embed_queue._pending


def test_two_accounts_never_share_a_batch(monkeypatch):
    """The encode runs as whichever caller took the batch, so it may only hold that account's
    texts: its token, model path and device choice are the account's."""
    from utils.account_context import AccountContext, run_as

    queue = ingestion._EmbedQueue()
    monkeypatch.setattr(ingestion, "_EMBED_LINGER_SECONDS", 30.0)
    seen: list[list[str]] = []
    lock = threading.Lock()

    def fake_encode(texts, **kwargs):
        with lock:
            seen.append(list(texts))
        return [[float(len(text))] for text in texts], "identity"

    monkeypatch.setattr(embeddings, "encode_with_identity", fake_encode)
    results: dict[str, object] = {}
    arrived = threading.Barrier(3)
    queue.expect()

    def embed_as(account):
        def go():
            arrived.wait(30)
            results[account.account_id] = queue.embed([f"{account.account_id} text"], None)

        run_as(account, go)

    threads = [
        threading.Thread(target = embed_as, args = (AccountContext(f"acct-{i}", f"user{i}"),))
        for i in (1, 2)
    ]
    for thread in threads:
        thread.start()
    arrived.wait(30)
    # Both are in the queue now; releasing the expected document lets each batch leave.
    queue.unexpect()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()

    assert sorted(len(batch) for batch in seen) == [1, 1], seen
    assert sorted(batch[0] for batch in seen) == ["acct-1 text", "acct-2 text"]
    assert set(results) == {"acct-1", "acct-2"}


def test_a_large_document_does_not_hold_back_a_small_one(monkeypatch):
    """The small one's progress callback, and with it the unlink check, must run within a batch."""
    monkeypatch.setattr(ingestion, "_EMBED_BATCH", 4)
    queue = ingestion._EmbedQueue()
    big = ingestion._EmbedRequest([f"big {i}" for i in range(40)], None)
    small = ingestion._EmbedRequest(["small"], None)
    queue._pending += [big, small]

    pieces = queue._take(big.key)

    assert (small, 0, 1) in pieces
    assert sum(end - start for _, start, end in pieces) == 4
    assert queue._pending == [big]


def test_a_parse_under_another_model_does_not_hold_a_batch(monkeypatch):
    monkeypatch.setattr(ingestion, "_EMBED_LINGER_SECONDS", 30.0)
    monkeypatch.setattr(
        embeddings, "encode_with_identity", lambda texts, **kwargs: ([[1.0]] * len(texts), "id")
    )
    queue = ingestion._EmbedQueue()
    queue.expect("other-model")
    started = time.monotonic()
    vectors, _, _ = queue.embed(["a note"], None)
    assert vectors == [[1.0]]
    assert time.monotonic() - started < 5
    queue.unexpect("other-model")
    assert not queue._expected


def test_a_gpu_choice_that_loaded_on_cpu_keeps_per_document_calls(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(embeddings, "active_backend_is_llama", lambda model_name = None: False)
    monkeypatch.setattr(embeddings, "_device", lambda: "cuda")
    monkeypatch.setattr(embeddings, "_model", None)
    assert _real_gate(None)
    monkeypatch.setattr(embeddings, "_model", SimpleNamespace(device = "cpu"))
    assert not _real_gate(None)
    monkeypatch.setattr(embeddings, "_model", SimpleNamespace(device = "cuda:0"))
    assert _real_gate(None)

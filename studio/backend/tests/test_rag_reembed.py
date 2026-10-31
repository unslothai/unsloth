# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import hashlib
import threading

import pytest

from core.rag import chunking, config, embeddings, ingestion, reembed, retrieval, store
from storage import rag_db

OLD, NEW, NEWER = "org/old-embedder", "org/new-embedder", "org/newer-embedder"
WIDTHS = {OLD: 8, NEW: 12, NEWER: 12}  # same width: NEWER's rewrite must remove NEW's rows


@pytest.fixture
def models(rag_home, stub_embeddings, monkeypatch):
    state = {"model": OLD, "encoded": [], "fail": lambda texts: False}

    def encode(texts, model_name, **_):
        if model_name != OLD and (failure := state["fail"](texts)):
            raise failure if isinstance(failure, Exception) else RuntimeError("embedder down")
        state["encoded"].append((model_name, list(texts)))
        digests = (hashlib.sha256(f"{model_name}{t}".encode()).digest() for t in texts)
        return [[b / 255.0 + 1e-3 for b in d[: WIDTHS[model_name]]] for d in digests]

    monkeypatch.setattr(embeddings, "encode", encode)
    monkeypatch.setattr(embeddings, "max_tokens", lambda model_name = None: None)
    monkeypatch.setattr(reembed, "_jobs", {})
    monkeypatch.setattr(reembed, "_checked", set(), raising = False)
    monkeypatch.setattr(reembed, "_RETIRE_JOIN_SECONDS", 0.2)
    current = lambda fresh = False: state["model"] if fresh else state.get("cached", state["model"])
    monkeypatch.setattr(config, "effective_embedding_model", current)
    return state


def _connect():
    return contextlib.closing(rag_db.get_connection())


def _ingest(tmp_path, scope, name, text):
    (tmp_path / name).write_text(text, encoding = "utf-8")
    return ingestion.start_ingestion(
        scope, None, None, name, str(tmp_path / name), model_name = OLD, background = False
    )[0]


def _row(document_id):
    sql = "SELECT id, text FROM chunks WHERE document_id=? ORDER BY chunk_index"
    with _connect() as c:
        chunks = [tuple(r) for r in c.execute(sql, (document_id,))]
        return store.get_document(c, document_id)["embedding_model"], chunks


def _run_job():
    reembed.start()
    reembed._jobs["reembed"].thread.join(timeout = 10)
    return reembed.status()


def test_stale_documents_regain_dense_search_under_the_same_ids(models, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TOKENS", 3)
    scope = store.project_scope("P1")
    first = _ingest(tmp_path, scope, "a.txt", "alpha bravo charlie")
    second = _ingest(tmp_path, scope, "b.txt", "kilo lima mike\n\ndelta echo foxtrot")
    turn = _ingest(tmp_path, store.conversation_archive_scope("T1"), "turn.txt", "golf hotel")
    chunks = [_row(d)[1] for d in (first, second)]
    with _connect() as c:  # as archive_turns writes it
        store.set_document_status(c, turn, "completed", num_chunks = 0)
    models["model"] = NEW
    assert reembed.stale_counts() == {"documents": 2, "archived": 1}
    assert ((job := _run_job())["status"], job["done"], job["failed"]) == ("completed", 3, 0)
    assert reembed.stale_counts() == {"documents": 0, "archived": 0}
    assert [_row(first), _row(second)] == [(embeddings.embedding_identity(NEW), c) for c in chunks]
    with _connect() as c:
        hits = retrieval.retrieve_dense(c, scope, "delta echo foxtrot", k = 5, model_name = NEW)
    assert hits[0].chunk_id == chunks[1][1][0]


def test_documents_the_same_model_wrote_through_another_backend_re_embed_on_search(
    models, tmp_path, monkeypatch
):
    scope = store.kb_scope("K10")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    doc = _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    llama = _row(doc)[0]
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    assert reembed.stale_counts() == {"documents": 1, "archived": 0}
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha bravo", k = 1, model_name = OLD)
    assert reembed.status() is not None
    reembed._jobs["reembed"].thread.join(timeout = 10)
    assert reembed.status()["status"] == "completed"
    assert _row(doc)[0] == embeddings.embedding_identity(OLD) != llama
    with _connect() as c:
        hits = retrieval.retrieve_dense(c, scope, "alpha bravo", k = 1, model_name = OLD)
    assert hits[0].chunk_id == f"{doc}:0"


def test_a_model_switch_leaves_re_embedding_to_the_user(models, tmp_path):
    scope = store.kb_scope("K11")
    _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    models["model"] = NEW
    with _connect() as c:
        assert retrieval.retrieve_dense(c, scope, "alpha bravo", k = 1, model_name = NEW) == []
    assert reembed.status() is None


def test_documents_left_from_a_model_switch_keep_re_embedding_with_the_user(
    models, tmp_path, monkeypatch
):
    other = "org/other-embedder"
    monkeypatch.setitem(WIDTHS, other, WIDTHS[OLD])
    scope = store.kb_scope("K12")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    (tmp_path / "b.txt").write_text("charlie delta", encoding = "utf-8")
    ingestion.start_ingestion(
        scope, None, None, "b.txt", str(tmp_path / "b.txt"), model_name = other, background = False
    )
    assert reembed.stale_counts() == {"documents": 2, "archived": 0}
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha bravo", k = 1, model_name = OLD)
    assert reembed.status() is None


def test_a_runtime_fallback_to_llama_server_leaves_the_st_vectors_alone(
    models, tmp_path, monkeypatch
):
    scope = store.kb_scope("K13")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    doc = _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    st = _row(doc)[0]
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha bravo", k = 1, model_name = OLD)
    assert reembed.status() is None and _row(doc)[0] == st


def test_a_model_switch_stops_an_automatic_re_embed(models, tmp_path, monkeypatch):
    scope = store.kb_scope("K14")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    healthy = _ingest(tmp_path, scope, "b.txt", "charlie delta")
    encode = embeddings.encode

    def switch_while_a_encodes(texts, model_name, **kw):
        if texts == ["alpha bravo"]:
            models["model"] = NEW
        return encode(texts, model_name, **kw)

    monkeypatch.setattr(embeddings, "encode", switch_while_a_encodes)
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha", k = 1, model_name = OLD)
    reembed._jobs["reembed"].thread.join(timeout = 10)
    assert reembed.status()["status"] == "cancelled"
    assert _row(healthy)[0] == embeddings.embedding_identity(OLD)
    assert all(m != NEW for m, _ in models["encoded"])


def test_a_fallback_during_an_automatic_re_embed_stops_it(models, tmp_path, monkeypatch):
    scope = store.kb_scope("K15")
    llama = {"on": True}
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: llama["on"])
    _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    llama["on"] = False
    healthy = _ingest(tmp_path, scope, "b.txt", "charlie delta")
    st = _row(healthy)[0]
    encode = embeddings.encode

    def fall_back_while_a_encodes(texts, model_name, **kw):
        if texts == ["alpha bravo"]:
            llama["on"] = True
        return encode(texts, model_name, **kw)

    monkeypatch.setattr(embeddings, "encode", fall_back_while_a_encodes)
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha", k = 1, model_name = OLD)
    reembed._jobs["reembed"].thread.join(timeout = 10)
    assert reembed.status()["status"] == "cancelled"
    assert _row(healthy)[0] == st


def test_a_re_embed_click_during_an_automatic_job_follows_a_model_switch(
    models, tmp_path, monkeypatch
):
    scope = store.kb_scope("K16")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    doc = _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    encode = embeddings.encode

    def switch_and_click_while_a_encodes(texts, model_name, **kw):
        if texts == ["alpha bravo"] and model_name == OLD:
            models["model"] = NEW
            reembed.start()
        return encode(texts, model_name, **kw)

    monkeypatch.setattr(embeddings, "encode", switch_and_click_while_a_encodes)
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha", k = 1, model_name = OLD)
    reembed._jobs["reembed"].thread.join(timeout = 10)
    assert reembed.status()["status"] == "completed"
    assert _row(doc)[0] == embeddings.embedding_identity(NEW)


def test_an_automatic_job_leaves_a_document_another_model_finished_meanwhile(
    models, tmp_path, monkeypatch
):
    other = "org/other-embedder"
    monkeypatch.setitem(WIDTHS, other, WIDTHS[OLD])
    scope = store.kb_scope("K17")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: True)
    _ingest(tmp_path, scope, "a.txt", "alpha bravo")
    monkeypatch.setattr(embeddings, "_identity_backend_is_llama", lambda name: False)
    (tmp_path / "b.txt").write_text("charlie delta", encoding = "utf-8")
    encode = embeddings.encode
    late = []

    def finish_an_upload_while_a_encodes(texts, model_name, **kw):
        if texts == ["alpha bravo"] and not late:
            late.append(
                ingestion.start_ingestion(
                    scope, None, None, "b.txt", str(tmp_path / "b.txt"),
                    model_name = other, background = False,
                )[0]
            )  # fmt: skip
        return encode(texts, model_name, **kw)

    monkeypatch.setattr(embeddings, "encode", finish_an_upload_while_a_encodes)
    with _connect() as c:
        retrieval.retrieve_dense(c, scope, "alpha", k = 1, model_name = OLD)
    reembed._jobs["reembed"].thread.join(timeout = 10)
    assert reembed.status()["status"] == "cancelled"
    assert _row(late[0])[0] == embeddings.embedding_identity(other)


def test_failures_keep_the_old_index_and_only_a_streak_ends_the_job(models, tmp_path, monkeypatch):
    monkeypatch.setattr(reembed, "_MAX_CONSECUTIVE_FAILURES", 2)
    scope = store.kb_scope("K1")
    broken = _ingest(tmp_path, scope, "broken.txt", "poison alpha")
    healthy = _ingest(tmp_path, scope, "ok.txt", "bravo charlie")
    _ingest(tmp_path, scope, "broken2.txt", "poison bravo")
    original = (rag_db.VECTORS_DROPPED + embeddings.embedding_identity(OLD), _row(broken)[1])
    models.update(model = NEW, fail = lambda texts: any("poison" in t for t in texts))
    assert ((job := _run_job())["status"], job["done"], job["failed"]) == ("completed", 1, 2)
    assert (_row(broken), _row(healthy)[0]) == (original, embeddings.embedding_identity(NEW))
    models["fail"] = lambda texts: True
    job = _run_job()
    assert (job["status"], job["remaining"], job["error"]) == ("failed", 0, "embedder down")
    models["fail"] = lambda texts: embeddings.EmbeddingModelDownloadRequiredError("gone")
    assert ((job := _run_job())["status"], job["failed"], job["error"]) == ("failed", 0, "gone")


def test_a_model_switch_mid_job_redoes_what_the_previous_model_wrote(models, tmp_path):
    scope = store.kb_scope("K3")
    ids = [_ingest(tmp_path, scope, f"{n}.txt", f"text {n}") for n in "ab"]
    switch_while_a_encodes = lambda texts: texts == ["text a"] and models.update(model = NEWER)
    models.update(model = NEW, fail = switch_while_a_encodes)
    assert _run_job()["status"] == "completed"
    assert {_row(d)[0] for d in ids} == {embeddings.embedding_identity(NEWER)}
    with _connect() as c:
        assert c.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2
        hit = retrieval.retrieve_dense(c, scope, "text b", k = 1, model_name = NEWER)[0]
    assert (hit.chunk_id, round(hit.score, 4)) == (f"{ids[1]}:0", 1.0)


def test_vectors_superseded_mid_encode_are_not_written_over_the_new_index(models, tmp_path):
    scope = store.kb_scope("K8")
    [_ingest(tmp_path, scope, f"{n}.txt", f"text {n}") for n in "ab"]
    switch_in_another_process = lambda texts: "cached" not in models and (
        _ingest(tmp_path, scope, "c.txt", "text c") and models.update(model = OLD, cached = NEW)
    )
    models.update(model = NEW, fail = switch_in_another_process)
    assert ((job := _run_job())["status"], job["done"]) == ("completed", 0)
    with _connect() as c:
        assert c.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 3


def test_documents_a_width_change_stripped_stay_stale_after_switching_back(models, tmp_path):
    ids = [_ingest(tmp_path, store.kb_scope("K9"), f"{n}.txt", f"text {n}") for n in "ab"]
    models.update(model = NEW, fail = lambda t: t == ["text a"] and reembed.cancel() and False)
    assert _run_job()["status"] == "cancelled"  # b re-embedded, which dropped a's vectors
    models["model"] = OLD
    assert (reembed.stale_counts()["documents"], _run_job()["done"]) == (2, 2)
    old, chunk = embeddings.embedding_identity(OLD), chunking.Chunk("text b", 2, None, 0, 0, 0, 6)
    with _connect() as c:
        store.set_document_status(c, ids[1], "pending")  # still being ingested
        rag_db.ensure_vec(c, 4, ids[0]) or rag_db.ensure_vec(c, 6, ids[0])  # ids[0] is writing
        assert store.get_document(c, ids[1])["embedding_model"] == rag_db.VECTORS_DROPPED + old
        assert store.count_untagged_documents(c) == 0
        store.add_chunks(c, store.kb_scope("K9"), ids[1], [chunk], [[0.5] * 6])  # its writer lands
    assert [_row(d)[0] for d in ids] == [old, old]


def test_cancel_and_retirement_stop_before_the_document_in_flight_is_written(models, tmp_path):
    reembed.retire_account_reembed()  # no job yet
    document_id = _ingest(tmp_path, store.kb_scope("K6"), "a.txt", "alpha bravo")
    original = _row(document_id)
    models.update(model = NEW, fail = lambda texts: reembed.cancel() and False)
    assert (_run_job()["status"], _row(document_id)) == ("cancelled", original)
    entered, release = threading.Event(), threading.Event()
    models["fail"] = lambda texts: entered.set() or not release.wait(5)
    assert reembed.start()["remaining"] == 1 and entered.wait(5)
    with pytest.raises(RuntimeError, match = "not stopped"):
        reembed.retire_account_reembed()
    job = reembed._jobs["reembed"]  # still registered, so the retry below waits for it
    release.set()
    reembed.retire_account_reembed()
    assert (job.thread.is_alive(), job.status, reembed.status()) == (False, "cancelled", None)
    assert _row(document_id) == original


def test_chunks_past_the_limit_embed_a_prefix(models, tmp_path, monkeypatch):
    for i, text in enumerate(("six seven eight", "ten eleven twelve", "one two three four five")):
        _ingest(tmp_path, store.kb_scope("K4"), f"{i}.txt", text)
    limits = iter([3, 3, 2])  # read per document, newest first
    monkeypatch.setattr(embeddings, "max_tokens", lambda model_name = None: next(limits))
    models["model"] = NEW
    assert _run_job()["clipped_chunks"] == 2
    sent = [t for m, t in models["encoded"] if m == NEW]
    assert sent == [["one two three "], ["ten eleven twelve"], ["six seven "]]


def test_a_document_changed_while_it_encodes_is_not_overwritten_but_redone(models, tmp_path):
    [_ingest(tmp_path, store.kb_scope("K5"), f"{n}.txt", f"text {n}") for n in "ab"]
    widen = lambda: rag_db.ensure_vec(c := rag_db.get_connection(), 12) or c.commit() or c.close()
    models.update(model = NEW, widen = widen, fail = lambda t: models.pop("widen", bool)())
    assert ((job := _run_job())["status"], job["done"]) == ("completed", 2)
    sent = [t for m, [t] in models["encoded"] if m == NEW]
    assert sent == ["text b", "text b", "text a"]  # widened while b encoded: b re-read


def test_retirement_racing_a_start_waits_for_its_worker(models, tmp_path, monkeypatch):
    _ingest(tmp_path, store.kb_scope("K7"), "a.txt", "alpha bravo")
    models["model"] = NEW
    errors, racers = [], []

    def retire():
        try:
            reembed.retire_account_reembed()
        except Exception as exc:
            errors.append(exc)

    class Racing(threading.Thread):
        def start(self):
            racers.append(racer := threading.Thread(target = retire))
            racer.start()
            racer.join(0.3)  # retirement lands between publishing the job and starting its worker
            super().start()

    monkeypatch.setattr(reembed, "account_thread", lambda **kw: Racing(**kw))
    reembed.start()
    racers[0].join(5)
    assert errors == [] and reembed.status() is None


def test_a_swap_committed_mid_search_cannot_pass_an_old_space_score(models, tmp_path, monkeypatch):
    scope = store.kb_scope("K9")
    doc = _ingest(tmp_path, scope, "a.txt", "text a")
    models["model"] = NEW
    assert _run_job()["status"] == "completed"
    models["model"] = NEWER  # same width: NEW's vectors stay in the table until swapped
    original = store._drop_incompatible

    def swap_between_reads(conn, *args):
        # The swap's commit, from another connection, after the KNN read scored NEW's vector.
        with _connect() as other:
            other.execute(
                "UPDATE documents SET embedding_model=? WHERE id=?",
                (embeddings.embedding_identity(NEWER), doc),
            )
            other.commit()
        return original(conn, *args)

    monkeypatch.setattr(store, "_drop_incompatible", swap_between_reads)
    with _connect() as c:
        hits = retrieval.retrieve_dense(c, scope, "text a", k = 5, model_name = NEWER)
    assert hits == []

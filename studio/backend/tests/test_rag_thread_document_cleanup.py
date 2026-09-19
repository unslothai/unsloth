# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.rag import ingestion, store
from routes import chat_history, rag as rag_routes
from storage import rag_db


@pytest.fixture
def client(rag_home, stub_embeddings):
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _create_thread(client, thread_id):
    response = client.post(
        "/api/chat/threads",
        json = {"id": thread_id, "title": "t", "modelType": "base", "createdAt": 1},
    )
    assert response.status_code == 200, response.text


def _upload(client, thread_id, name, text):
    response = client.post(
        f"/api/rag/threads/{thread_id}/documents",
        files = {"file": (name, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    deadline = time.time() + 30
    while time.time() < deadline:
        status = ingestion.get_job_status(body["jobId"])
        if status and status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    return body["documentId"]


def _stored_path(document_id):
    conn = rag_db.get_connection()
    try:
        return store.get_document(conn, document_id)["stored_path"]
    finally:
        conn.close()


def _document_ids(client):
    response = client.get("/api/rag/documents")
    assert response.status_code == 200, response.text
    return {doc["id"] for doc in response.json()["documents"]}


def _chunk_count(thread_id):
    conn = rag_db.get_metadata_connection()
    try:
        scope = store.thread_scope(thread_id)
        chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE scope=?", (scope,)).fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE scope=?", (scope,)).fetchone()[0]
        return chunks + fts
    finally:
        conn.close()


def test_deleting_a_thread_removes_its_uploaded_documents(client):
    _create_thread(client, "doomed")
    _create_thread(client, "kept")
    doomed = [
        _upload(client, "doomed", f"doomed{i}.txt", f"alpha bravo charlie {i} " * 50)
        for i in range(2)
    ]
    kept = _upload(client, "kept", "kept.txt", "delta echo foxtrot " * 50)
    doomed_paths, kept_path = [_stored_path(d) for d in doomed], _stored_path(kept)
    assert all(os.path.isfile(path) for path in doomed_paths)

    response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["doomed"]})

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {kept}
    assert _chunk_count("doomed") == 0
    assert not any(os.path.exists(path) for path in doomed_paths)
    assert os.path.isfile(kept_path)
    assert _chunk_count("kept") > 0


def test_clearing_history_removes_every_threads_uploaded_documents(client):
    paths = []
    for thread_id in ("first", "second"):
        _create_thread(client, thread_id)
        paths.append(
            _stored_path(_upload(client, thread_id, f"{thread_id}.txt", f"{thread_id} words " * 50))
        )

    response = client.request("DELETE", "/api/chat")

    assert response.status_code == 200, response.text
    assert _document_ids(client) == set()
    assert not any(os.path.exists(path) for path in paths)


def test_deleting_a_project_removes_its_member_threads_documents(client):
    response = client.post(
        "/api/chat/projects", json = {"id": "proj", "name": "p", "createdAt": 1, "updatedAt": 1}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/chat/threads",
        json = {
            "id": "member",
            "title": "t",
            "modelType": "base",
            "createdAt": 1,
            "projectId": "proj",
        },
    )
    assert response.status_code == 200, response.text
    path = _stored_path(_upload(client, "member", "member.txt", "golf hotel india " * 50))
    _create_thread(client, "outsider")
    outsider = _upload(client, "outsider", "outsider.txt", "sierra tango uniform " * 50)

    response = client.delete("/api/chat/projects/proj")

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {outsider}
    assert not os.path.exists(path)


def test_a_recreated_thread_keeps_documents_uploaded_after_the_cutoff(client):
    _create_thread(client, "recreated")
    old = _upload(client, "recreated", "old.txt", "juliet kilo lima " * 50)
    old_path = _stored_path(old)
    cutoff = datetime.now(timezone.utc).isoformat()
    fresh = _upload(client, "recreated", "fresh.txt", "mike november oscar " * 50)

    chat_history._remove_thread_rag_data(["recreated"])
    assert _document_ids(client) == {old, fresh}

    chat_history._remove_thread_rag_data(["recreated"], cutoff = cutoff)

    assert _document_ids(client) == {fresh}
    assert not os.path.exists(old_path)
    assert os.path.isfile(_stored_path(fresh))


def test_thread_documents_go_without_sqlite_vec(client, monkeypatch):
    from storage import studio_db

    _create_thread(client, "vecless")
    document_id = _upload(client, "vecless", "vecless.txt", "papa quebec romeo " * 50)
    path = _stored_path(document_id)
    studio_db.delete_chat_threads(["vecless"])

    def no_vec():
        raise rag_db.RagExtensionUnavailable("vec0 will not load")

    with monkeypatch.context() as patch:
        patch.setattr(rag_db, "get_connection", no_vec)
        chat_history._remove_thread_rag_data(["vecless"])

    assert _document_ids(client) == set()
    assert _chunk_count("vecless") == 0
    assert not os.path.exists(path)


def _add_message(client, thread_id, message_id):
    response = client.put(
        f"/api/chat/threads/{thread_id}/messages/{message_id}",
        json = {"id": message_id, "threadId": thread_id, "role": "user", "createdAt": 1},
    )
    assert response.status_code == 200, response.text


def _fork(client, thread_id, message_id, new_thread_id):
    response = client.post(
        f"/api/chat/threads/{thread_id}/fork",
        json = {"messageId": message_id, "newThreadId": new_thread_id, "createdAt": 2},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _thread_documents(client, thread_id):
    response = client.get(f"/api/rag/threads/{thread_id}/documents")
    assert response.status_code == 200, response.text
    return response.json()["documents"]


def _search(client, thread_id, query, mode):
    response = client.post(
        "/api/rag/search", json = {"query": query, "thread_id": thread_id, "mode": mode}
    )
    assert response.status_code == 200, response.text
    return response.json()["results"]


def test_forking_a_thread_copies_its_uploaded_documents(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    text = "victor whiskey xray " * 50
    source = _upload(client, "source", "source.txt", text)

    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None

    documents = _thread_documents(client, "fork")
    assert [(d["filename"], d["status"]) for d in documents] == [("source.txt", "completed")]
    copy = documents[0]["id"]
    assert copy != source
    assert documents[0]["numChunks"] == _thread_documents(client, "source")[0]["numChunks"]
    for mode in ("lexical", "dense"):
        hits = _search(client, "fork", "victor whiskey xray", mode)
        assert hits and {hit["documentId"] for hit in hits} == {copy}
    assert _stored_path(copy) != _stored_path(source)
    with open(_stored_path(copy), encoding = "utf-8") as copied:
        assert copied.read() == text


def test_a_forks_documents_are_independent_of_the_source_thread(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    _upload(client, "source", "source.txt", "yankee zulu alpha " * 50)
    _fork(client, "source", "m1", "fork")
    copy = _thread_documents(client, "fork")[0]["id"]
    copy_path = _stored_path(copy)

    _upload(client, "source", "later.txt", "bravo charlie delta " * 50)
    assert [d["id"] for d in _thread_documents(client, "fork")] == [copy]

    response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["source"]})

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {copy}
    assert os.path.isfile(copy_path)
    assert _search(client, "fork", "yankee zulu alpha", "dense")


def test_a_fork_survives_documents_that_cannot_be_copied(client, monkeypatch):
    from core.rag import conversation_archive

    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)

    def broken(conn, *args, **kwargs):
        conn.execute("INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('x', 'x', 'x')")
        raise RuntimeError("disk full")

    monkeypatch.setattr(conversation_archive.store, "copy_document", broken, raising = False)
    before = set(os.listdir(os.path.dirname(_stored_path(source))))

    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]

    assert "not copied" in (warning or "")

    assert _thread_documents(client, "fork") == []
    assert _document_ids(client) == {source}
    assert set(os.listdir(os.path.dirname(_stored_path(source)))) == before
    conn = rag_db.get_metadata_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE scope='x'").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_fork_warns_when_existing_documents_cannot_load_without_vec(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    with monkeypatch.context() as unavailable:
        unavailable.setattr(rag_db, "rag_available", lambda: False)
        warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")
    assert _thread_documents(client, "fork") == []
    assert _document_ids(client) == {source}


def test_a_fork_without_documents_needs_no_warning_without_vec(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None


def test_a_fork_warns_when_an_upload_is_still_pending(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    conn = rag_db.get_connection()
    try:
        store.create_document(
            conn,
            scope = store.thread_scope("source"),
            thread_id = "source",
            filename = "pending.txt",
            sha256 = "pending-upload",
            status = "pending",
        )
    finally:
        conn.close()
    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")


def test_a_fork_copies_files_before_taking_the_rag_write_lock(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    for index in range(2):
        _upload(client, "source", f"source{index}.txt", f"echo foxtrot golf {index} " * 50)
    copy_upload = ingestion._copy_upload
    writable = []

    def copy_with_concurrent_writer(path):
        conn = rag_db.get_metadata_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 20")
            conn.execute("BEGIN IMMEDIATE")
            writable.append(True)
        except sqlite3.OperationalError:
            writable.append(False)
        finally:
            conn.rollback()
            conn.close()
        return copy_upload(path)

    monkeypatch.setattr(ingestion, "_copy_upload", copy_with_concurrent_writer)
    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None
    assert writable == [True, True]


def test_a_fork_warns_if_a_source_document_is_deleted_during_file_copy(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    copy_upload = ingestion._copy_upload

    def copy_then_delete(path):
        copied = copy_upload(path)
        response = client.delete(f"/api/rag/documents/{source}")
        assert response.status_code == 200, response.text
        return copied

    monkeypatch.setattr(ingestion, "_copy_upload", copy_then_delete)
    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")
    assert _thread_documents(client, "fork") == []


def test_deleting_a_fork_during_file_copy_does_not_leave_orphan_documents(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    upload_dir = os.path.dirname(_stored_path(source))
    before = set(os.listdir(upload_dir))
    copy_upload = ingestion._copy_upload

    def copy_then_delete_fork(path):
        copied = copy_upload(path)
        response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["fork"]})
        assert response.status_code == 200, response.text
        return copied

    monkeypatch.setattr(ingestion, "_copy_upload", copy_then_delete_fork)
    _fork(client, "source", "m1", "fork")
    assert _document_ids(client) == {source}
    assert set(os.listdir(upload_dir)) == before


def _cite(client, thread_id, message_id, document_id):
    sources = [
        {
            "citationId": 1,
            "chunkId": f"{document_id}:0",
            "documentId": document_id,
            "filename": "source.txt",
        }
    ]
    part = {
        "type": "tool-call",
        "toolCallId": "call-1",
        "toolName": "search_documents",
        "args": {"query": "hotel"},
        "result": "hotel india\n__RAG_SOURCES__:" + json.dumps(sources),
    }
    response = client.put(
        f"/api/chat/threads/{thread_id}/messages/{message_id}",
        json = {
            "id": message_id,
            "threadId": thread_id,
            "parentId": "m1",
            "role": "assistant",
            "content": [part],
            "createdAt": 2,
        },
    )
    assert response.status_code == 200, response.text


def _cited(message):
    result = message["content"][0]["result"]
    return json.loads(result.split("__RAG_SOURCES__:", 1)[1])[0]


def test_a_forks_copied_messages_cite_the_forks_documents(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "hotel india juliet " * 50)
    _cite(client, "source", "m2", source)

    forked = _fork(client, "source", "m2", "fork")

    copy = _thread_documents(client, "fork")[0]["id"]
    stored = client.get("/api/chat/threads/fork/messages").json()["messages"]
    for messages in (forked["messages"], stored):
        cited = _cited([m for m in messages if m["role"] == "assistant"][0])
        assert (cited["documentId"], cited["chunkId"]) == (copy, f"{copy}:0")
    parent = client.get("/api/chat/threads/source/messages/m2").json()
    assert (_cited(parent)["documentId"], _cited(parent)["chunkId"]) == (source, f"{source}:0")

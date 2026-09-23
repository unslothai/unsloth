# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
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
    app.include_router(chat_history.router, prefix="/api/chat")
    app.include_router(rag_routes.router, prefix="/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _create_thread(client, thread_id):
    response = client.post(
        "/api/chat/threads",
        json={"id": thread_id, "title": "t", "modelType": "base", "createdAt": 1},
    )
    assert response.status_code == 200, response.text


def _upload(client, thread_id, name, text):
    response = client.post(
        f"/api/rag/threads/{thread_id}/documents",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
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

    response = client.request("DELETE", "/api/chat/threads", json={"ids": ["doomed"]})

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
        "/api/chat/projects", json={"id": "proj", "name": "p", "createdAt": 1, "updatedAt": 1}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/chat/threads",
        json={
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

    chat_history._remove_thread_rag_data(["recreated"], cutoff=cutoff)

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

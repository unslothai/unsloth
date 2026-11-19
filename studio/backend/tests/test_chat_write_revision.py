# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat writes report the revisions their own transaction moved, so the tab that wrote can tell
its write from another device's, and an unchanged re-save moves nothing other devices poll."""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth.authentication import get_current_subject
from routes import chat_history
from storage import studio_db

HEADER = chat_history.CHAT_REVISION_HEADER


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _thread(thread_id = "t1"):
    return {
        "id": thread_id,
        "title": "T",
        "modelType": "base",
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _message(
    message_id,
    created_at,
    text = "hi",
    parent = None,
    thread_id = "t1",
):
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent,
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "createdAt": created_at,
    }


def _revisions(client, thread_id = "t1"):
    body = client.get(f"/api/chat/revision?threadId={thread_id}").json()
    return body["threads"][thread_id], body["revision"]


def _span(response):
    assert response.status_code == 200, response.text
    thread_before, thread_after, history_before, history_after = map(
        int, response.headers[HEADER].split(",")
    )
    return (thread_before, history_before), (thread_after, history_after)


def test_each_chat_write_reports_the_revisions_around_it(client):
    writes = [
        lambda: client.post("/api/chat/threads", json = _thread()),
        lambda: client.put(
            "/api/chat/threads/t1/messages/m1", json = _message("m1", 1_700_000_000_001)
        ),
        lambda: client.put(
            "/api/chat/threads/t1/messages",
            json = {
                "messages": [
                    _message("m1", 1_700_000_000_001),
                    _message("m2", 1_700_000_000_002, parent = "m1"),
                ]
            },
        ),
        lambda: client.patch("/api/chat/threads/t1", json = {"title": "Renamed"}),
    ]
    for write in writes:
        before = _revisions(client)
        span_before, span_after = _span(write())
        assert span_before == before
        assert span_after == _revisions(client)
        assert span_after[0] > span_before[0]


def test_a_write_from_elsewhere_shows_as_a_gap_between_spans(client):
    _, mine = _span(client.post("/api/chat/threads", json = _thread()))
    # Another device writes the same chat between this tab's two writes.
    studio_db.upsert_chat_message(_message("other", 1_700_000_000_005))
    before, _ = _span(
        client.put("/api/chat/threads/t1/messages/m1", json = _message("m1", 1_700_000_000_006))
    )
    assert before[0] > mine[0] and before[1] > mine[1]


def test_a_rejected_write_reports_nothing(client):
    _span(client.post("/api/chat/threads", json = _thread()))
    response = client.patch(
        "/api/chat/threads/t1", json = {"title": "X", "expectedTitle": "not the title"}
    )
    assert response.status_code == 409
    assert HEADER not in response.headers


def test_an_unchanged_resave_moves_no_revision(client):
    client.post("/api/chat/threads", json = _thread())
    messages = [
        _message("m1", 1_700_000_000_001),
        _message("m2", 1_700_000_000_002, parent = "m1"),
    ]
    client.put("/api/chat/threads/t1/messages", json = {"messages": messages})
    before = _revisions(client)
    response = client.put("/api/chat/threads/t1/messages", json = {"messages": messages})
    assert response.status_code == 200
    assert _revisions(client) == before
    # A real change still moves it.
    messages[1] = _message("m2", 1_700_000_000_002, text = "edited", parent = "m1")
    client.put("/api/chat/threads/t1/messages", json = {"messages": messages})
    assert _revisions(client)[0] > before[0]


def test_updated_at_still_ratchets_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message("m1", 1_700_000_000_500))
    assert studio_db.get_chat_thread("t1")["updatedAt"] == 1_700_000_000_500
    studio_db.upsert_chat_message(_message("m0", 1_700_000_000_100))
    assert studio_db.get_chat_thread("t1")["updatedAt"] == 1_700_000_000_500
    studio_db.upsert_chat_message(_message("m2", 1_700_000_000_900))
    assert studio_db.get_chat_thread("t1")["updatedAt"] == 1_700_000_000_900


def test_cross_origin_clients_can_read_the_header():
    import main
    exposed = [
        header
        for middleware in main.app.user_middleware
        for header in middleware.kwargs.get("expose_headers", ())
    ]
    assert HEADER in exposed

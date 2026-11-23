# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The autosave's echoMessages=false sync stores the same rows without re-reading the thread back."""

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


def _messages(
    thread_id,
    turns,
    text = "hi",
):
    out = []
    parent = None
    for i in range(turns):
        for role in ("user", "assistant"):
            message_id = f"{thread_id}-{role}-{i}"
            out.append(
                {
                    "id": message_id,
                    "threadId": thread_id,
                    "parentId": parent,
                    "role": role,
                    "content": [{"type": "text", "text": f"{text} {i}"}],
                    "metadata": {"turn": i},
                    "createdAt": 1_700_000_000_000 + len(out),
                }
            )
            parent = message_id
    return out


def _create(client, thread_id):
    response = client.post(
        "/api/chat/threads",
        json = {
            "id": thread_id,
            "title": "T",
            "modelType": "base",
            "archived": False,
            "createdAt": 1_700_000_000_000,
        },
    )
    assert response.status_code == 200, response.text


def _stored(thread_id):
    return [
        {k: v for k, v in m.items() if k != "threadId"}
        for m in studio_db.list_chat_messages(thread_id)
    ]


def _span_shape(response):
    values = response.headers[HEADER].split(",")
    assert len(values) == 4 and all(v.lstrip("-").isdigit() for v in values)


def test_no_echo_stores_identical_rows_with_a_small_body_and_the_revision_header(client):
    for thread_id in ("echo", "quiet"):
        _create(client, thread_id)
    for turns, text in ((3, "hi"), (5, "hi"), (5, "edited")):
        echoed = client.put(
            "/api/chat/threads/echo/messages", json = {"messages": _messages("echo", turns, text)}
        )
        quiet = client.put(
            "/api/chat/threads/quiet/messages?echoMessages=false",
            json = {"messages": _messages("quiet", turns, text)},
        )
        assert echoed.status_code == quiet.status_code == 200
        assert len(echoed.json()["messages"]) == turns * 2
        assert quiet.json() == {"messages": []}
        _span_shape(echoed)
        _span_shape(quiet)
        rename = lambda rows: [
            {k: (v.replace("echo", "quiet") if isinstance(v, str) else v) for k, v in row.items()}
            for row in rows
        ]
        assert rename(_stored("echo")) == _stored("quiet")


def test_default_and_explicit_true_still_echo_the_stored_thread(client):
    _create(client, "t1")
    messages = _messages("t1", 2)
    for url in ("/api/chat/threads/t1/messages", "/api/chat/threads/t1/messages?echoMessages=true"):
        response = client.put(url, json = {"messages": messages})
        assert response.status_code == 200
        assert [m["id"] for m in response.json()["messages"]] == [m["id"] for m in messages]


def test_no_echo_prune_still_deletes(client):
    _create(client, "t1")
    messages = _messages("t1", 2)
    client.put("/api/chat/threads/t1/messages", json = {"messages": messages})
    response = client.put(
        "/api/chat/threads/t1/messages?echoMessages=false",
        json = {"messages": messages[:2], "pruneMissing": True},
    )
    assert response.status_code == 200
    assert response.json() == {"messages": []}
    assert [m["id"] for m in _stored("t1")] == [m["id"] for m in messages[:2]]


def test_no_echo_keeps_the_error_statuses(client):
    response = client.put(
        "/api/chat/threads/missing/messages?echoMessages=false",
        json = {"messages": _messages("missing", 1)},
    )
    assert response.status_code == 404
    _create(client, "t1")
    response = client.put(
        "/api/chat/threads/t1/messages?echoMessages=false",
        json = {"messages": _messages("other", 1)},
    )
    assert response.status_code == 400


def test_storage_skips_the_reread_only_when_asked(client, monkeypatch):
    _create(client, "t1")
    reads = []
    real = studio_db.list_chat_messages
    monkeypatch.setattr(
        studio_db, "list_chat_messages", lambda *a, **k: reads.append(a) or real(*a, **k)
    )
    assert studio_db.sync_chat_messages("t1", _messages("t1", 1), return_rows = False) is None
    assert reads == []
    assert len(studio_db.sync_chat_messages("t1", _messages("t1", 2))) == 4
    assert len(reads) == 1

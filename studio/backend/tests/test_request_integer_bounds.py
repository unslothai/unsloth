# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request integer sqlite cannot store is a refused request, not an OverflowError (a 500)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from routes import chat_history, prompts, scheduled_tasks

TOO_BIG = 2**63
TOO_SMALL = -(2**63) - 1
LARGEST = 2**63 - 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    # A regressed project create makes its folder before it fails: keep that out of ~/Documents.
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.include_router(prompts.router, prefix = "/api/prompts")
    app.include_router(scheduled_tasks.router, prefix = "/api/scheduled-tasks")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    # A handler that raises must show as a 500 here, not as an exception out of the client.
    client = TestClient(app, raise_server_exceptions = False)
    # A fork reads its source thread and tip message before it binds anything from the request.
    assert client.post("/api/chat/threads", json = _thread()).status_code == 200
    assert client.put("/api/chat/threads/t1/messages/m1", json = _message()).status_code == 200
    return client


def _thread(**fields):
    return {"id": "t1", "modelType": "base", "createdAt": 1, **fields}


def _message(**fields):
    return {"id": "m1", "threadId": "t1", "role": "user", "createdAt": 1, **fields}


def _project(**fields):
    return {"id": "p1", "name": "p", "createdAt": 1, "updatedAt": 1, **fields}


def _prompt(**fields):
    return {"id": "e1", "name": "n", "text": "t", "createdAt": 1, "updatedAt": 1, **fields}


def _prompt_list(**fields):
    return {"id": "l1", "name": "n", "items": [], "createdAt": 1, "updatedAt": 1, **fields}


CASES = [
    ("post", "/api/chat/threads", _thread, "createdAt"),
    ("post", "/api/chat/threads", _thread, "updatedAt"),
    ("patch", "/api/chat/threads/t1", lambda **f: f, "createdAt"),
    ("patch", "/api/chat/threads/t1", lambda **f: f, "updatedAt"),
    ("put", "/api/chat/threads/t1/messages/m1", _message, "createdAt"),
    ("post", "/api/chat/threads/t1/fork", lambda **f: {"newThreadId": "f1", **f}, "createdAt"),
    ("post", "/api/chat/projects", _project, "createdAt"),
    ("post", "/api/chat/projects", _project, "updatedAt"),
    ("patch", "/api/chat/projects/p1", lambda **f: f, "createdAt"),
    ("patch", "/api/chat/projects/p1", lambda **f: f, "updatedAt"),
    ("put", "/api/prompts/entries/e1", _prompt, "createdAt"),
    ("put", "/api/prompts/entries/e1", _prompt, "updatedAt"),
    ("put", "/api/prompts/lists/l1", _prompt_list, "createdAt"),
    ("put", "/api/prompts/lists/l1", _prompt_list, "updatedAt"),
]


@pytest.mark.parametrize("value", [TOO_BIG, TOO_SMALL])
@pytest.mark.parametrize("method, path, body, field", CASES)
def test_a_timestamp_sqlite_cannot_hold_is_refused(client, method, path, body, field, value):
    response = getattr(client, method)(path, json = body(**{field: value}))

    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "path, param",
    [
        ("/api/chat/attachments", "offset"),
        ("/api/scheduled-tasks/runs", "since"),
    ],
)
def test_a_page_cursor_sqlite_cannot_hold_is_refused(client, path, param):
    assert client.get(path, params = {param: TOO_BIG}).status_code == 422
    assert client.get(path, params = {param: LARGEST}).status_code == 200


def test_the_largest_storable_timestamp_is_still_saved(client):
    response = client.put("/api/prompts/entries/e1", json = _prompt(createdAt = LARGEST))

    assert response.status_code == 200, response.text

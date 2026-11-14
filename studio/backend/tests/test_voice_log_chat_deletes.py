# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every chat delete path takes the deleted chats' voice transcripts out of the voice decision log."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import voice_log
from routes import chat_history
from storage import studio_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setenv("UNSLOTH_VOICE_LOG_DIR", str(tmp_path / "voice"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _thread(tid, project_id = None):
    t = {"id": tid, "title": tid, "modelType": "gguf", "createdAt": 1}
    if project_id:
        t["projectId"] = project_id
    studio_db.upsert_chat_thread(t)
    voice_log.append(
        "judge", {"text": f"secret said in {tid}"}, "hold", 0.2, 1.0, thread_id = tid, account = "owner"
    )


def _logged():
    return sorted(r["thread_id"] for r in voice_log.read_rows())


def test_delete_threads_removes_voice_log_rows(client):
    _thread("t-del")
    r = client.request("DELETE", "/api/chat/threads", json = {"ids": ["t-del"]})
    assert _logged() == []


def test_clear_all_chats_removes_voice_log_rows(client):
    _thread("t-clear")
    r = client.request("DELETE", "/api/chat")
    assert r.status_code == 200
    assert _logged() == []


def test_delete_project_removes_its_chats_voice_log_rows(client):
    studio_db.upsert_chat_project({"id": "p1", "name": "p1", "createdAt": 1, "updatedAt": 1})
    _thread("t-proj", project_id = "p1")
    r = client.request("DELETE", "/api/chat/projects/p1")
    assert r.status_code == 200
    assert _logged() == [], "project delete left its chats' voice transcripts in the log"

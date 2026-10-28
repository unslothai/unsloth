# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The thread lists the sidebar loads advertise the chat-title repair guards, so the frontend no longer fetches
the ~1 MB /openapi.json to learn one flag. The flag is derived from the same model the schema probe reads, so the
two answers cannot drift, and /openapi.json still declares the fields for older frontends."""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from auth.authentication import get_current_subject
from routes import chat_history
from storage import studio_db

GUARD_FIELDS = ("expectedTitle", "expectedOpeningMessageId")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def _schema_declares_guards(document: dict) -> bool:
    """Python twin of the frontend's schemaDeclaresRepairGuards (openapi-support.ts)."""
    properties = (
        document.get("components", {})
        .get("schemas", {})
        .get("ChatThreadPatch", {})
        .get("properties", {})
    )
    return all(field in properties for field in GUARD_FIELDS)


def test_thread_list_advertises_the_guards(app):
    body = TestClient(app).get("/api/chat/threads").json()
    assert body["threadPatchGuards"] is True
    assert body["threads"] == []


def test_sidebar_rows_advertise_the_guards(app):
    body = TestClient(app).post("/api/chat/sidebar/rows", json = {}).json()
    assert body["threadPatchGuards"] is True
    assert body["rows"] == [] and body["hasMore"] is False


def test_flag_agrees_with_the_schema_older_frontends_probe(app):
    client = TestClient(app)
    document = client.get("/openapi.json").json()
    # Older frontends still read this; it must keep declaring the fields.
    assert _schema_declares_guards(document) is True
    assert client.get("/api/chat/threads").json()["threadPatchGuards"] == _schema_declares_guards(
        document
    )


def test_flag_follows_the_patch_model():
    assert chat_history.THREAD_PATCH_GUARDS == all(
        name in chat_history.ChatThreadPatch.model_fields for name in GUARD_FIELDS
    )
    assert chat_history.THREAD_PATCH_GUARDS is True

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage as auth_storage
from auth.authentication import get_current_subject
from routes import auth as auth_routes
from routes import chat_history as chat_history_routes
from storage import studio_db
from utils.paths import (
    assets_root,
    project_workspaces_root,
    studio_db_path,
    studio_root,
)
from utils.paths.storage_roots import cache_root
from utils.workspace_context import reset_workspace_subject, set_workspace_subject


def _bind(subject: str):
    return set_workspace_subject(subject)


def _thread(title: str) -> dict:
    return {
        "id": "same-client-id",
        "title": title,
        "modelType": "base",
        "modelId": "",
        "createdAt": 1,
    }


def test_workspace_roots_keep_legacy_layout_and_isolate_other_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))

    legacy = _bind("unsloth")
    try:
        legacy_db = studio_db_path()
        legacy_assets = assets_root()
        legacy_projects = project_workspaces_root()
        shared_cache = cache_root()
        assert legacy_db == studio_root() / "studio.db"
        assert legacy_assets == studio_root() / "assets"
        assert legacy_projects == tmp_path / "projects"
    finally:
        reset_workspace_subject(legacy)

    alice = _bind("alice")
    try:
        alice_db = studio_db_path()
        alice_assets = assets_root()
        alice_projects = project_workspaces_root()
        assert alice_db.parent.parent == studio_root() / "workspaces"
        assert alice_assets.parent == alice_db.parent
        assert alice_projects.is_relative_to(tmp_path / "projects" / "Users")
        assert cache_root() == shared_cache
    finally:
        reset_workspace_subject(alice)

    bob = _bind("bob")
    try:
        assert studio_db_path() != alice_db
        assert assets_root() != alice_assets
        assert project_workspaces_root() != alice_projects
        assert cache_root() == shared_cache
    finally:
        reset_workspace_subject(bob)


def test_same_thread_id_and_settings_are_private_per_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    alice = _bind("alice")
    try:
        assert studio_db.upsert_chat_thread(_thread("Alice only"))["title"] == "Alice only"
        studio_db.upsert_chat_settings({"systemPrompt": "Alice secret"})
        alice_db = studio_db_path()
    finally:
        reset_workspace_subject(alice)

    bob = _bind("bob")
    try:
        assert studio_db.count_chat_threads() == 0
        assert studio_db.list_chat_settings() == {}
        assert studio_db.upsert_chat_thread(_thread("Bob only"))["title"] == "Bob only"
        studio_db.upsert_chat_settings({"systemPrompt": "Bob secret"})
        assert studio_db_path() != alice_db
    finally:
        reset_workspace_subject(bob)

    alice_again = _bind("alice")
    try:
        assert studio_db.get_chat_thread("same-client-id")["title"] == "Alice only"
        assert studio_db.list_chat_settings()["systemPrompt"] == "Alice secret"
    finally:
        reset_workspace_subject(alice_again)


def test_legacy_unsloth_account_is_promoted_once_during_role_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "auth" / "auth.db"
    db_path.parent.mkdir(parents = True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE auth_user (
            username TEXT PRIMARY KEY,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            jwt_secret TEXT NOT NULL,
            must_change_password INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO auth_user VALUES (?, ?, ?, ?, ?)",
        ("unsloth", "salt", "hash", "secret", 0),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(auth_storage, "DB_PATH", db_path)

    migrated = auth_storage.get_connection()
    try:
        row = migrated.execute(
            "SELECT is_admin FROM auth_user WHERE username = 'unsloth'"
        ).fetchone()
        assert row["is_admin"] == 1
    finally:
        migrated.close()


@pytest.fixture
def account_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password")
    auth_storage.create_initial_user(
        "unsloth",
        "owner-password",
        secrets.token_urlsafe(64),
        is_admin = True,
    )
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    with TestClient(app) as client:
        yield client, app


def test_owner_can_create_list_and_delete_standard_accounts(account_client):
    client, _app = account_client
    created = client.post(
        "/api/auth/users",
        json = {"username": "alice", "password": "temporary-password"},
    )
    assert created.status_code == 201
    assert created.json() == {
        "username": "alice",
        "is_admin": False,
        "must_change_password": True,
    }

    listed = client.get("/api/auth/users")
    assert listed.status_code == 200
    assert [user["username"] for user in listed.json()["users"]] == ["unsloth", "alice"]

    assert client.delete("/api/auth/users/alice").status_code == 204
    assert auth_storage.get_user_and_secret("alice") is None


def test_standard_account_cannot_manage_users(account_client):
    client, app = account_client
    auth_storage.create_managed_user("alice", "temporary-password")
    app.dependency_overrides[get_current_subject] = lambda: "alice"

    assert client.get("/api/auth/users").status_code == 403
    assert client.post(
        "/api/auth/users",
        json = {"username": "bob", "password": "temporary-password"},
    ).status_code == 403


def test_authenticated_chat_routes_select_the_token_subject_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    for username in ("alice", "bob"):
        auth_storage.create_initial_user(
            username,
            f"{username}-password",
            secrets.token_urlsafe(64),
        )

    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    app.include_router(chat_history_routes.router, prefix = "/api/chat")

    with TestClient(app) as client:
        def headers(username: str) -> dict[str, str]:
            response = client.post(
                "/api/auth/login",
                json = {"username": username, "password": f"{username}-password"},
            )
            assert response.status_code == 200
            return {"Authorization": f"Bearer {response.json()['access_token']}"}

        alice_headers = headers("alice")
        bob_headers = headers("bob")
        assert client.post(
            "/api/chat/threads",
            headers = alice_headers,
            json = _thread("Alice route"),
        ).status_code == 200

        bob_threads = client.get("/api/chat/threads", headers = bob_headers)
        assert bob_threads.status_code == 200
        assert bob_threads.json() == {"threads": []}

        assert client.post(
            "/api/chat/threads",
            headers = bob_headers,
            json = _thread("Bob route"),
        ).status_code == 200
        assert client.get(
            "/api/chat/threads/same-client-id",
            headers = alice_headers,
        ).json()["title"] == "Alice route"
        assert client.get(
            "/api/chat/threads/same-client-id",
            headers = bob_headers,
        ).json()["title"] == "Bob route"

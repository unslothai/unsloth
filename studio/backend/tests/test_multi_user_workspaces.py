# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import secrets
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage as auth_storage
from auth.authentication import get_current_subject
from routes import auth as auth_routes
from routes import chat_history as chat_history_routes
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, ApiUsageWriter
from utils.paths import (
    assets_root,
    project_workspaces_root,
    studio_db_path,
    studio_root,
    workspace_root,
)
from utils.paths.storage_roots import cache_root
from utils.workspace_context import (
    current_workspace_subject,
    reset_workspace_subject,
    set_workspace_subject,
    workspace_thread,
)


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


def test_runtime_artifacts_and_oauth_tokens_follow_the_authenticated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.inference import (
        audio_gallery,
        image_gallery,
        mcp_client,
        search_images,
        tools,
        video_gallery,
    )

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    locations: dict[str, set[Path]] = {}

    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            root = workspace_root().resolve()
            store = mcp_client._oauth_store()
            paths = {
                image_gallery.gallery_dir().resolve(),
                audio_gallery.gallery_dir().resolve(),
                video_gallery.gallery_dir().resolve(),
                search_images._cache_dir().resolve(),
                Path(tools.sandbox_root()).resolve(),
                Path(tools._orphan_records_dir()).resolve(),
                Path(tools._spill_records_dir()).resolve(),
                Path(store._data_directory).resolve(),
            }
            assert all(path.is_relative_to(root) for path in paths if "sandboxes" not in path.parts)
            assert Path(tools.sandbox_root()).is_relative_to(
                (tmp_path / "sandboxes" / "workspaces").resolve()
            )
            locations[subject] = paths
        finally:
            reset_workspace_subject(token)

    assert locations["alice"].isdisjoint(locations["bob"])

    owner = _bind("unsloth")
    try:
        assert image_gallery.gallery_dir() == studio_root() / "images"
        assert Path(tools.sandbox_root()) == tmp_path / "sandboxes"
    finally:
        reset_workspace_subject(owner)


def test_managed_accounts_cannot_browse_or_register_host_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from hub.services.models import folder_browser as hub_folder_browser
    from hub.storage import scan_folders as hub_scan_folders
    from routes import models as model_routes

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    external = tmp_path / "external-models"
    external.mkdir()
    owner_private_model = studio_root() / "owner-private-model.gguf"
    owner_private_model.parent.mkdir(parents = True)
    owner_private_model.write_bytes(b"owner")

    alice = _bind("alice")
    try:
        private_root = workspace_root()
        private_models = private_root / "models"
        private_models.mkdir(parents = True)
        private_model = private_models / "alice.gguf"
        private_model.write_bytes(b"alice")
        roots = {
            path.resolve()
            for path in model_routes._build_browse_allowlist(
                media_roots = [external],
                drive_roots = [],
            )
        }
        assert private_root.resolve() in roots
        assert studio_root().resolve() not in roots
        assert external.resolve() not in roots
        assert Path.home().resolve() not in roots
        assert Path(model_routes._normalize_browse_request_path(None)) == private_root
        with pytest.raises(HTTPException) as exc_info:
            model_routes._resolve_browse_target(str(studio_root()), list(roots))
        assert exc_info.value.status_code == 403
        assert model_routes._is_sizable_local_path(str(private_model))
        assert not model_routes._is_sizable_local_path(str(owner_private_model))

        hub_roots = {
            path.resolve()
            for path in hub_folder_browser._build_browse_allowlist(
                media_roots = [external],
                drive_roots = [],
            )
        }
        assert private_root.resolve() in hub_roots
        assert studio_root().resolve() not in hub_roots
        assert external.resolve() not in hub_roots

        with pytest.raises(ValueError, match = "inside their workspace"):
            studio_db.add_scan_folder_with_status(str(external))
        with pytest.raises(ValueError, match = "inside their workspace"):
            hub_scan_folders.add_scan_folder_with_status(str(external))
        assert studio_db.add_scan_folder_with_status(str(private_models))[0]["path"] == str(
            private_models.resolve()
        )

        # A row left by an older build is ignored rather than becoming an
        # allowlist escape after upgrade.
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)",
                (str(external.resolve()), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        assert [folder["path"] for folder in studio_db.list_scan_folders()] == [
            str(private_models.resolve())
        ]
    finally:
        reset_workspace_subject(alice)

    owner = _bind("unsloth")
    try:
        roots = {
            path.resolve()
            for path in model_routes._build_browse_allowlist(
                media_roots = [external],
                drive_roots = [],
            )
        }
        assert studio_root().resolve() in roots
        assert external.resolve() in roots
        assert Path.home().resolve() in roots
        assert Path(model_routes._normalize_browse_request_path(None)) == Path.home()
        assert model_routes._is_sizable_local_path(str(owner_private_model))
    finally:
        reset_workspace_subject(owner)


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


def test_background_threads_keep_the_workspace_that_started_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    barrier = threading.Barrier(3)

    def write(subject: str, title: str) -> None:
        token = _bind(subject)
        try:
            worker = workspace_thread(
                target = lambda: (
                    barrier.wait(),
                    studio_db.upsert_chat_thread(_thread(title)),
                )
            )
        finally:
            reset_workspace_subject(token)
        worker.start()
        workers.append(worker)

    workers: list[threading.Thread] = []
    write("alice", "Alice background")
    write("bob", "Bob background")
    barrier.wait()
    for worker in workers:
        worker.join(timeout = 5)
        assert not worker.is_alive()

    assert current_workspace_subject() == "unsloth"
    for subject, expected in (("alice", "Alice background"), ("bob", "Bob background")):
        token = _bind(subject)
        try:
            assert studio_db.get_chat_thread("same-client-id")["title"] == expected
        finally:
            reset_workspace_subject(token)
    assert studio_db.get_chat_thread("same-client-id") is None


def test_api_usage_writer_routes_each_receipt_to_its_account_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    writer = ApiUsageWriter()

    for subject in ("alice", "bob"):
        assert writer.submit(
            ApiUsageReceipt(
                id = f"{subject}-usage",
                subject = subject,
                endpoint = "/v1/chat/completions",
                model = "test-model",
                status = "completed",
                prompt_tokens = 2,
                completion_tokens = 3,
                total_tokens = 5,
                created_at = now,
            )
        )
    assert writer.stop()

    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            conn = studio_db.get_connection()
            try:
                rows = conn.execute(
                    "SELECT id, subject FROM api_usage_events ORDER BY id"
                ).fetchall()
            finally:
                conn.close()
            assert [(row["id"], row["subject"]) for row in rows] == [(f"{subject}-usage", subject)]
        finally:
            reset_workspace_subject(token)

    owner_conn = studio_db.get_connection()
    try:
        assert owner_conn.execute("SELECT COUNT(*) FROM api_usage_events").fetchone()[0] == 0
    finally:
        owner_conn.close()


def test_training_spawn_metadata_selects_managed_user_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.training.worker import _bind_worker_workspace
    from utils.paths import outputs_root

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    token = _bind_worker_workspace({"subject": "alice"})
    assert token is not None
    try:
        assert outputs_root().is_relative_to(studio_root() / "workspaces")
        assert outputs_root() != studio_root() / "outputs"
    finally:
        reset_workspace_subject(token)


def test_data_recipe_job_state_is_hidden_from_other_accounts():
    from core.data_recipe.jobs.manager import JobManager

    manager = JobManager()
    manager._job = SimpleNamespace(job_id = "alice-job")
    manager._workspace_subject = "alice"

    alice = _bind("alice")
    try:
        assert manager.owns_workspace()
        assert manager.get_current_job_id() == "alice-job"
    finally:
        reset_workspace_subject(alice)

    bob = _bind("bob")
    try:
        assert not manager.owns_workspace()
        assert manager.get_current_job_id() is None
        assert manager.get_status("alice-job") is None
        assert manager.get_analysis("alice-job") is None
        assert manager.get_dataset("alice-job", limit = 20) is None
        assert manager.subscribe("alice-job") is None
        assert manager.cancel("alice-job") is False
    finally:
        reset_workspace_subject(bob)


def test_data_recipe_publish_path_cannot_cross_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from routes.data_recipe.jobs import _workspace_artifact_path
    from utils.paths import recipe_datasets_root

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    alice = _bind("alice")
    try:
        alice_artifact = recipe_datasets_root() / "alice-job"
        alice_artifact.mkdir(parents = True)
        assert _workspace_artifact_path(str(alice_artifact)) == str(alice_artifact.resolve())
    finally:
        reset_workspace_subject(alice)

    bob = _bind("bob")
    try:
        with pytest.raises(HTTPException) as exc_info:
            _workspace_artifact_path(str(alice_artifact))
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "dataset not found"
    finally:
        reset_workspace_subject(bob)


def test_private_local_model_index_entries_are_hidden_from_other_accounts():
    from core.inference.local_model_resolver import _LocalGgufEntry, _resolve_from_index

    entry = _LocalGgufEntry(
        loader_id = "private/model",
        load_path = "/private/alice/model.gguf",
        variants = (),
        workspace_subject = "alice",
    )
    index = {"private/model": entry}

    alice = _bind("alice")
    try:
        assert _resolve_from_index("private/model", index) == (
            "/private/alice/model.gguf",
            None,
            "private/model",
        )
    finally:
        reset_workspace_subject(alice)

    bob = _bind("bob")
    try:
        assert _resolve_from_index("private/model", index) is None
    finally:
        reset_workspace_subject(bob)


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
    monkeypatch.setattr(
        auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password"
    )
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
        json = {"username": "alice"},
    )
    assert created.status_code == 201
    created_body = created.json()
    assert created_body["username"] == "alice"
    assert created_body["is_admin"] is False
    assert created_body["must_change_password"] is True
    assert created_body["setup_code_expired"] is False
    assert created_body["setup_code_expires_at"]
    assert re.fullmatch(r"[A-Z2-9]{4}(?:-[A-Z2-9]{4}){3}", created_body["setup_code"])

    listed = client.get("/api/auth/users")
    assert listed.status_code == 200
    assert [user["username"] for user in listed.json()["users"]] == ["unsloth", "alice"]
    assert all("setup_code" not in user for user in listed.json()["users"])

    assert client.delete("/api/auth/users/alice").status_code == 204
    assert auth_storage.get_user_and_secret("alice") is None


def test_setup_code_is_hashed_expires_and_is_not_listed(account_client):
    client, _app = account_client
    created = client.post("/api/auth/users", json = {"username": "alice"})
    assert created.status_code == 201
    setup_code = created.json()["setup_code"]

    conn = auth_storage.get_connection()
    try:
        row = conn.execute(
            "SELECT password_hash, setup_code_expires_at FROM auth_user WHERE username = 'alice'"
        ).fetchone()
        assert setup_code not in row["password_hash"]
        assert row["setup_code_expires_at"]
        conn.execute(
            "UPDATE auth_user SET setup_code_expires_at = ? WHERE username = 'alice'",
            (datetime(2000, 1, 1, tzinfo = timezone.utc).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()

    expired = client.post(
        "/api/auth/login",
        json = {"username": "alice", "password": setup_code},
    )
    wrong = client.post(
        "/api/auth/login",
        json = {"username": "alice", "password": "definitely-wrong"},
    )
    assert expired.status_code == wrong.status_code == 401
    assert expired.json() == wrong.json()
    listed_user = next(
        user
        for user in client.get("/api/auth/users").json()["users"]
        if user["username"] == "alice"
    )
    assert listed_user["setup_code_expired"] is True
    assert "setup_code" not in listed_user


def test_regenerating_pending_setup_code_revokes_old_code_and_refresh_session(account_client):
    client, _app = account_client
    created = client.post("/api/auth/users", json = {"username": "alice"}).json()
    first_code = created["setup_code"]
    first_login = client.post(
        "/api/auth/login",
        json = {"username": "alice", "password": first_code},
    )
    assert first_login.status_code == 200

    regenerated = client.post("/api/auth/users/alice/setup-code")
    assert regenerated.status_code == 200
    second_code = regenerated.json()["setup_code"]
    assert second_code != first_code
    assert (
        client.post(
            "/api/auth/login",
            json = {"username": "alice", "password": first_code},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/refresh",
            json = {"refresh_token": first_login.json()["refresh_token"]},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login",
            json = {"username": "alice", "password": second_code},
        ).status_code
        == 200
    )


def test_setup_code_becomes_permanent_password_then_cannot_be_regenerated(account_client):
    client, _app = account_client
    setup_code = client.post("/api/auth/users", json = {"username": "alice"}).json()["setup_code"]
    first_login = client.post(
        "/api/auth/login",
        json = {"username": "alice", "password": setup_code},
    )
    changed = client.post(
        "/api/auth/change-password",
        headers = {"Authorization": f"Bearer {first_login.json()['access_token']}"},
        json = {"current_password": setup_code, "new_password": "alice-permanent-password"},
    )
    assert changed.status_code == 200

    conn = auth_storage.get_connection()
    try:
        row = conn.execute(
            "SELECT must_change_password, setup_code_expires_at FROM auth_user WHERE username = 'alice'"
        ).fetchone()
        assert row["must_change_password"] == 0
        assert row["setup_code_expires_at"] is None
    finally:
        conn.close()
    assert (
        client.post(
            "/api/auth/login",
            json = {"username": "alice", "password": setup_code},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login",
            json = {"username": "alice", "password": "alice-permanent-password"},
        ).status_code
        == 200
    )
    assert client.post("/api/auth/users/alice/setup-code").status_code == 409


def test_standard_account_cannot_manage_users(account_client):
    client, app = account_client
    auth_storage.create_managed_user("alice")
    app.dependency_overrides[get_current_subject] = lambda: "alice"

    assert client.get("/api/auth/users").status_code == 403
    assert client.post("/api/auth/users", json = {"username": "bob"}).status_code == 403


def test_only_owner_can_change_installation_wide_server_access(account_client):
    from fastapi import HTTPException
    from routes.settings import _require_install_admin

    _client, _app = account_client
    auth_storage.create_managed_user("alice")

    assert _require_install_admin("unsloth") == "unsloth"
    with pytest.raises(HTTPException) as exc_info:
        _require_install_admin("alice")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Only the installation owner can change server access."


def test_real_tokens_enforce_roles_and_deletion_revokes_sessions_but_keeps_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(
        auth_storage,
        "_BOOTSTRAP_PW_PATH",
        tmp_path / "auth" / ".bootstrap_password",
    )
    auth_storage.create_initial_user(
        "unsloth",
        "owner-password",
        secrets.token_urlsafe(64),
        is_admin = True,
    )
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")

    with TestClient(app) as client:
        owner_login = client.post(
            "/api/auth/login",
            json = {"username": "unsloth", "password": "owner-password"},
        )
        assert owner_login.status_code == 200
        owner_headers = {"Authorization": f"Bearer {owner_login.json()['access_token']}"}
        created_alice = client.post(
            "/api/auth/users",
            headers = owner_headers,
            json = {"username": "alice"},
        )
        assert created_alice.status_code == 201
        alice_setup_code = created_alice.json()["setup_code"]

        first_login = client.post(
            "/api/auth/login",
            json = {"username": "alice", "password": alice_setup_code},
        )
        assert first_login.status_code == 200
        first_headers = {"Authorization": f"Bearer {first_login.json()['access_token']}"}
        changed = client.post(
            "/api/auth/change-password",
            headers = first_headers,
            json = {
                "current_password": alice_setup_code,
                "new_password": "alice-permanent-password",
            },
        )
        assert changed.status_code == 200
        alice_tokens = changed.json()
        alice_headers = {"Authorization": f"Bearer {alice_tokens['access_token']}"}
        assert client.get("/api/auth/me", headers = alice_headers).status_code == 200
        assert client.get("/api/auth/users", headers = alice_headers).status_code == 403

        token = _bind("alice")
        try:
            original_workspace = workspace_root()
            original_workspace.mkdir(parents = True, exist_ok = True)
            marker = original_workspace / "retained-after-account-delete.txt"
            marker.write_text("private data", encoding = "utf-8")
        finally:
            reset_workspace_subject(token)

        assert client.delete("/api/auth/users/alice", headers = owner_headers).status_code == 204
        assert client.get("/api/auth/me", headers = alice_headers).status_code == 401
        assert (
            client.post(
                "/api/auth/refresh",
                json = {"refresh_token": alice_tokens["refresh_token"]},
            ).status_code
            == 401
        )

        recreated = client.post(
            "/api/auth/users",
            headers = owner_headers,
            json = {"username": "alice"},
        )
        assert recreated.status_code == 201
        token = _bind("alice")
        try:
            assert workspace_root() == original_workspace
            assert marker.read_text(encoding = "utf-8") == "private data"
        finally:
            reset_workspace_subject(token)


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
        assert (
            client.post(
                "/api/chat/threads",
                headers = alice_headers,
                json = _thread("Alice route"),
            ).status_code
            == 200
        )

        bob_threads = client.get("/api/chat/threads", headers = bob_headers)
        assert bob_threads.status_code == 200
        assert bob_threads.json() == {"threads": []}

        assert (
            client.post(
                "/api/chat/threads",
                headers = bob_headers,
                json = _thread("Bob route"),
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/chat/threads/same-client-id",
                headers = alice_headers,
            ).json()["title"]
            == "Alice route"
        )
        assert (
            client.get(
                "/api/chat/threads/same-client-id",
                headers = bob_headers,
            ).json()["title"]
            == "Bob route"
        )

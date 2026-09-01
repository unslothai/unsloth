# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import inspect
import secrets
import re
import sqlite3
import threading
from datetime import datetime, timezone
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage as auth_storage
from auth.authentication import get_current_subject, require_install_admin
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
    workspace_key,
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
    # Not a keyless caller: account management is refused to one.
    app.dependency_overrides[auth_routes.authenticated_without_credential] = lambda: False
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


def test_recreating_a_name_whose_files_are_locked_says_so(account_client, monkeypatch):
    client, _app = account_client
    monkeypatch.setattr(
        auth_storage,
        "create_managed_user",
        lambda username: (_ for _ in ()).throw(ValueError("Close anything using them")),
    )
    # The expected recreate-after-delete case, not a bug: it must reach the owner
    # as the instruction storage wrote rather than as an opaque 500.
    refused = client.post("/api/auth/users", json = {"username": "alice"})
    assert refused.status_code == 409
    assert "Close anything using them" in refused.json()["detail"]


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


def test_real_tokens_enforce_roles_and_deletion_revokes_sessions_and_retires_workspace(
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

        # The files are kept for recovery, but moved aside: the key is derived
        # from the username, so leaving them in place would hand them to whoever
        # registers the name next.
        assert not marker.exists()
        retired = sorted(original_workspace.parent.glob(f"{original_workspace.name}-deleted-*"))
        assert len(retired) == 1
        assert (retired[0] / marker.name).read_text(encoding = "utf-8") == "private data"

        recreated = client.post(
            "/api/auth/users",
            headers = owner_headers,
            json = {"username": "alice"},
        )
        assert recreated.status_code == 201
        token = _bind("alice")
        try:
            assert workspace_root() == original_workspace
            assert not marker.exists()
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


def test_an_unclaimed_training_backend_is_visible_to_every_account():
    """The singleton is built lazily, in whichever request context touches it first.

    Snapshotting the subject in __init__ pinned the *idle* backend to that first
    caller, so every other account got 404 on progress/metrics, a silent no-op
    reset, and a permanent "idle" status before it had ever started a run. State
    nobody has claimed belongs to nobody, so it is visible to all; the moment a
    run claims it the check is strict again.
    """
    from core.training.training import TrainingBackend
    from utils.workspace_context import reset_workspace_subject, set_workspace_subject

    token = set_workspace_subject("alice")
    try:
        backend = TrainingBackend()
    finally:
        reset_workspace_subject(token)

    assert backend.owns_workspace("alice")
    assert backend.owns_workspace("bob")
    assert backend.owns_workspace("unsloth")


def test_a_claimed_training_backend_is_private_to_the_account_that_started_it():
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    with backend._lock:
        backend._active_workspace_subject = "alice"

    assert backend.owns_workspace("alice")
    assert not backend.owns_workspace("bob")
    assert not backend.owns_workspace("unsloth")


def test_keyless_callers_cannot_manage_accounts(account_client):
    """Keyless admission resolves to the owner, so require_admin alone lets an
    unauthenticated caller mint setup codes. Account management is an effect that
    outlives the setting, so it needs a credential of its own."""
    client, app = account_client
    app.dependency_overrides[auth_routes.authenticated_without_credential] = lambda: True

    assert client.get("/api/auth/users").status_code == 403
    assert client.post("/api/auth/users", json = {"username": "mallory"}).status_code == 403
    assert client.post("/api/auth/users/alice/setup-code").status_code == 403
    assert client.delete("/api/auth/users/alice").status_code == 403


def test_managed_accounts_cannot_export_to_an_absolute_path(tmp_path):
    """The absolute-path escape hatch (gh 6082) is an owner convenience. For a
    managed account it is a write primitive into any reachable directory."""
    from utils.paths.storage_roots import exports_root, resolve_export_write_dir

    outside = str(tmp_path / "outside")

    token = set_workspace_subject("alice")
    try:
        with pytest.raises(ValueError):
            resolve_export_write_dir(outside)
        assert resolve_export_write_dir("nested/run").is_relative_to(exports_root())
    finally:
        reset_workspace_subject(token)

    # The owner keeps it.
    assert resolve_export_write_dir(outside) == Path(outside)


def test_settings_memos_do_not_serve_one_workspace_value_to_another():
    import utils.vram_budget_settings as vram
    from storage.studio_db import upsert_app_settings

    key = "vram_budget_fraction"
    for subject, value in (("alice", 0.11), ("bob", 0.99)):
        token = set_workspace_subject(subject)
        try:
            upsert_app_settings({key: value})
            vram._invalidate(key)
        finally:
            reset_workspace_subject(token)

    reads = {}
    for subject in ("alice", "bob"):
        token = set_workspace_subject(subject)
        try:
            reads[subject] = float(vram._cached_setting(key))
        finally:
            reset_workspace_subject(token)
    assert reads == {"alice": 0.11, "bob": 0.99}


def test_mcp_session_keys_are_private_to_a_workspace():
    """scope carries client-chosen thread ids, so two accounts can present the
    same one and would otherwise share a live stdio child."""
    from core.inference.mcp_client import _session_key

    keys = {}
    for subject in ("alice", "bob"):
        token = set_workspace_subject(subject)
        try:
            keys[subject] = _session_key("stdio://tool", None, "s=same:t=same")
        finally:
            reset_workspace_subject(token)
    assert keys["alice"] != keys["bob"]


def test_the_openai_model_catalog_cache_is_per_workspace():
    from routes.inference import _CATALOG_CACHE, _catalog_is_fresh

    _CATALOG_CACHE["subject"] = "alice"
    _CATALOG_CACHE["at"] = time.monotonic()
    try:
        assert _catalog_is_fresh("alice", time.monotonic())
        assert not _catalog_is_fresh("bob", time.monotonic())
    finally:
        _CATALOG_CACHE["subject"] = None
        _CATALOG_CACHE["at"] = 0.0


def test_active_generations_are_named_and_cancellable_only_by_their_own_account():
    from state import active_generations

    active_generations.reset_for_tests()
    alice_event, bob_event = threading.Event(), threading.Event()
    token = _bind("alice")
    try:
        alice = active_generations.ActiveGeneration(alice_event, thread_id = "shared-thread")
        alice.__enter__()
    finally:
        reset_workspace_subject(token)
    token = _bind("bob")
    try:
        bob = active_generations.ActiveGeneration(bob_event, thread_id = "shared-thread")
        bob.__enter__()
        assert [e["thread_id"] for e in active_generations.snapshot("bob")] == ["shared-thread"]
        assert active_generations.active_thread_ids("alice") == ["shared-thread"]
        # Bob presenting Alice's conversation id must not stop her generation.
        assert active_generations.cancel_thread("shared-thread", "bob") == 1
        assert bob_event.is_set() and not alice_event.is_set()
    finally:
        bob.__exit__()
        token2 = _bind("alice")
        try:
            alice.__exit__()
        finally:
            reset_workspace_subject(token2)
        reset_workspace_subject(token)
        active_generations.reset_for_tests()


def test_cancel_registry_keys_do_not_collide_across_accounts():
    from routes.inference import _scoped_cancel_key

    keys = {}
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            keys[subject] = _scoped_cancel_key("session-1")
        finally:
            reset_workspace_subject(token)
    assert keys["alice"] != keys["bob"]


def test_a_managed_password_change_keeps_the_owner_desktop_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(
        auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password"
    )
    auth_storage.create_initial_user(
        "unsloth", "owner-password", secrets.token_urlsafe(64), is_admin = True
    )
    setup_code = auth_storage.create_managed_user("casey")["setup_code"]
    raw_secret = auth_storage.create_desktop_secret()

    # A managed account completing setup from a browser: is_desktop is false there.
    assert auth_storage.update_password("casey", "casey-permanent-pw") is not None
    assert auth_storage.validate_desktop_secret(raw_secret) == "unsloth"

    # The owner's own browser change still revokes it.
    assert auth_storage.update_password("unsloth", "owner-new-pw") is not None
    assert auth_storage.validate_desktop_secret(raw_secret) is None


def test_seed_upload_roots_follow_the_authenticated_workspace():
    import routes.data_recipe.seed as seed_routes

    roots = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            roots[subject] = seed_routes._unstructured_upload_root()
        finally:
            reset_workspace_subject(token)
    assert roots["unsloth"] != roots["alice"]
    assert workspace_key("alice") in str(roots["alice"])


def test_deleting_an_account_retires_its_projects_and_sandbox_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(
        auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password"
    )
    auth_storage.create_initial_user(
        "unsloth", "owner-password", secrets.token_urlsafe(64), is_admin = True
    )
    auth_storage.create_managed_user("casey")

    roots = auth_storage._subject_owned_roots("casey")
    assert len(roots) == 3
    for root in roots:
        root.mkdir(parents = True, exist_ok = True)
        (root / "private.txt").write_text("casey", encoding = "utf-8")

    auth_storage.delete_managed_user("casey")
    auth_storage.create_managed_user("casey")
    for root in auth_storage._subject_owned_roots("casey"):
        assert not (root / "private.txt").exists()


def test_sandbox_lifecycle_keys_are_private_to_a_workspace():
    from core.inference import tools

    keys = {}
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            keys[subject] = tools._session_key("shared-session")
        finally:
            reset_workspace_subject(token)
    assert keys["alice"] != keys["bob"]
    assert tools._subject_of_session_key(keys["alice"]) == "alice"


def test_the_media_model_index_is_not_shared_between_accounts():
    from core.inference import media_model_index as mmi

    mmi.invalidate_index()
    token = _bind("alice")
    try:
        mmi._index[(current_workspace_subject(), "image")] = (time.monotonic(), {"m": object()})
        assert mmi._cached_index("image") == {"m": mmi._index[("alice", "image")][1]["m"]}
    finally:
        reset_workspace_subject(token)
    token = _bind("bob")
    try:
        assert ("bob", "image") not in mmi._index
    finally:
        reset_workspace_subject(token)
        mmi.invalidate_index()


def test_a_training_start_request_id_cannot_replay_another_accounts_outcome():
    from core.training.training import TrainingBackend

    backend = TrainingBackend.__new__(TrainingBackend)
    backend._lock = threading.RLock()
    backend._start_requests = {}
    backend._start_cancel_tombstones = {}
    backend._start_cancel_tombstone_reservations = {}
    backend._pending_start_key = None
    backend._status_start_key = None
    backend._current_start_key = None

    token = _bind("alice")
    try:
        outcome, record = backend.reserve_start_request("same-id", "job-alice")
        assert outcome == "reserved" and record.subject == "alice"
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Bob must not be handed Alice's record, and must not be told it exists.
        assert backend.peek_start_request("same-id") is None
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        assert backend.peek_start_request("same-id").job_id == "job-alice"
        # The registry is keyed by workspace, so Bob's rejection cannot land on
        # top of Alice's pending record and be replayed to her as her own outcome.
        assert set(backend._start_requests) == {("alice", "same-id")}
    finally:
        reset_workspace_subject(token)


def _auth_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(
        auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password"
    )
    auth_storage.create_initial_user(
        "unsloth", "owner-password", secrets.token_urlsafe(64), is_admin = True
    )


def test_a_recreated_username_gets_a_schema_not_a_missing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    token = _bind("casey")
    try:
        studio_db.upsert_chat_thread(_thread("first casey"))
        assert studio_db.list_chat_threads()
    finally:
        reset_workspace_subject(token)

    auth_storage.delete_managed_user("casey")
    auth_storage.create_managed_user("casey")
    token = _bind("casey")
    try:
        # The path is the same, so a cached "schema ready" would raise no such table.
        assert studio_db.list_chat_threads() == []
    finally:
        reset_workspace_subject(token)


def test_a_username_whose_files_could_not_be_released_cannot_be_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    for root in auth_storage._subject_owned_roots("casey"):
        root.mkdir(parents = True, exist_ok = True)
        (root / "private.txt").write_text("casey", encoding = "utf-8")

    real_rename = Path.rename
    monkeypatch.setattr(
        Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("locked"))
    )
    auth_storage.delete_managed_user("casey")
    assert auth_storage.username_is_retired("casey")
    with pytest.raises(ValueError, match = "could not be released"):
        auth_storage.create_managed_user("casey")

    # Once the handle goes, the retry retires the files and the name frees up.
    monkeypatch.setattr(Path, "rename", real_rename)
    assert not auth_storage.username_is_retired("casey")
    auth_storage.create_managed_user("casey")
    for root in auth_storage._subject_owned_roots("casey"):
        assert not (root / "private.txt").exists()


def test_streamed_tool_workers_keep_the_callers_workspace():
    from core.inference import tool_stream_exec
    src = inspect.getsource(tool_stream_exec.stream_tool_execution)
    assert "run_in_workspace(bound_subject" in src


def test_signed_media_links_name_the_workspace_that_minted_them():
    from utils import signed_media_links

    secret = b"x" * 32
    tokens = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            tokens[subject] = signed_media_links.sign(secret, "img_1", 3600)
        finally:
            reset_workspace_subject(token)
    assert tokens["unsloth"] != tokens["alice"]
    assert signed_media_links.verify(secret, tokens["alice"]) == ("img_1", "alice")

    # A token minted before the subject was carried still reads as the owner.
    import hashlib
    import hmac
    import time as _t

    exp = int(_t.time()) + 3600
    payload = f"img_1.{exp}"
    legacy = f"{payload}.{hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()}"
    assert signed_media_links.verify(secret, legacy) == ("img_1", "unsloth")
    assert signed_media_links.verify(secret, "img_1.1.deadbeef") == (None, "unsloth")


def test_openai_video_jobs_are_not_listed_or_deletable_by_another_account():
    from routes import video as video_routes
    video_routes._jobs.clear()
    try:
        token = _bind("alice")
        try:
            video_routes._jobs["vid_alice"] = video_routes._VideoJob(
                id = "vid_alice",
                created_at = 1,
                prompt = "alice private prompt",
                model = "m",
                size = "auto",
                seconds = "auto",
                subject = "alice",
            )
            assert "vid_alice" in video_routes._my_jobs_locked()
        finally:
            reset_workspace_subject(token)

        token = _bind("bob")
        try:
            assert video_routes._my_jobs_locked() == {}
            assert not video_routes._job_is_mine(video_routes._jobs["vid_alice"])
        finally:
            reset_workspace_subject(token)
    finally:
        video_routes._jobs.clear()


def test_deleting_a_thread_cancels_only_this_accounts_generation():
    from state import active_generations

    active_generations.reset_for_tests()
    alice_event, bob_event = threading.Event(), threading.Event()
    handles = []
    for subject, event in (("alice", alice_event), ("bob", bob_event)):
        token = _bind(subject)
        try:
            handle = active_generations.ActiveGeneration(event, thread_id = "same-thread")
            handle.__enter__()
            handles.append((subject, handle))
        finally:
            reset_workspace_subject(token)
    try:
        token = _bind("bob")
        try:
            assert active_generations.cancel_thread("same-thread", current_workspace_subject()) == 1
        finally:
            reset_workspace_subject(token)
        assert bob_event.is_set() and not alice_event.is_set()
    finally:
        for subject, handle in handles:
            token = _bind(subject)
            try:
                handle.__exit__()
            finally:
                reset_workspace_subject(token)
        active_generations.reset_for_tests()


def test_closing_one_accounts_mcp_row_leaves_the_others_session_alive():
    from core.inference import mcp_client

    keys = {}
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            keys[subject] = mcp_client._session_key("stdio:same-cmd", None, "")
        finally:
            reset_workspace_subject(token)
    saved = dict(mcp_client._mcp_sessions)
    mcp_client._mcp_sessions.clear()
    try:
        for subject, key in keys.items():
            mcp_client._mcp_sessions[key] = SimpleNamespace(close = lambda: None)
        token = _bind("alice")
        try:
            mcp_client.close_stdio_sessions("stdio:same-cmd")
        finally:
            reset_workspace_subject(token)
        assert keys["alice"] not in mcp_client._mcp_sessions
        assert keys["bob"] in mcp_client._mcp_sessions
    finally:
        mcp_client._mcp_sessions.clear()
        mcp_client._mcp_sessions.update(saved)


def test_process_exit_closes_every_accounts_mcp_sessions():
    from core.inference import mcp_client

    keys = {}
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            keys[subject] = mcp_client._session_key("stdio:same-cmd", None, "")
        finally:
            reset_workspace_subject(token)
    saved = dict(mcp_client._mcp_sessions)
    mcp_client._mcp_sessions.clear()
    try:
        for key in keys.values():
            mcp_client._mcp_sessions[key] = SimpleNamespace(close = lambda: None)
        # atexit runs on the main thread, which holds the default workspace, so a
        # workspace-confined close would strand every managed account's child.
        mcp_client._close_sessions_at_exit()
        assert mcp_client._mcp_sessions == {}
    finally:
        mcp_client._mcp_sessions.clear()
        mcp_client._mcp_sessions.update(saved)


def test_a_delete_that_cannot_retire_leaves_the_name_reserved_from_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    for root in auth_storage._subject_owned_roots("casey"):
        root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(
        Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("locked"))
    )
    auth_storage.delete_managed_user("casey")
    # Written in the delete transaction, so the row and the tombstone are never
    # both absent for a create racing the delete.
    conn = auth_storage.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT 1 FROM retired_usernames WHERE username = ?", ("casey",)
            ).fetchone()
            is not None
        )
        assert (
            conn.execute("SELECT 1 FROM auth_user WHERE username = ?", ("casey",)).fetchone()
            is None
        )
    finally:
        conn.close()


def test_a_create_cannot_slip_past_a_tombstone_it_did_not_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    conn = auth_storage.get_connection()
    try:
        conn.execute(
            "INSERT INTO retired_usernames (username, created_at) VALUES (?, ?)",
            ("casey", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    # Stands in for the racing create whose pre-commit read saw a free name: the
    # insert itself must refuse, or it binds to a workspace a delete is renaming.
    with pytest.raises(ValueError):
        auth_storage.create_initial_user(
            "casey",
            "code",
            secrets.token_urlsafe(64),
            reject_if_retired = True,
        )
    conn = auth_storage.get_connection()
    try:
        assert (
            conn.execute("SELECT 1 FROM auth_user WHERE username = ?", ("casey",)).fetchone()
            is None
        )
    finally:
        conn.close()


def test_roots_that_cannot_be_resolved_keep_the_name_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    # A reduced install where the sandbox root will not import. The workspace
    # tree still renames cleanly, so without the completeness flag retirement
    # reports success and a namesake reopens the untouched project directory.
    monkeypatch.setattr(
        auth_storage,
        "_resolve_subject_owned_roots",
        lambda username: ([], False),
    )
    auth_storage.delete_managed_user("casey")
    assert auth_storage._retire_workspace_directory("casey") is False
    conn = auth_storage.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT 1 FROM retired_usernames WHERE username = ?", ("casey",)
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def test_a_managed_account_cannot_load_a_model_by_absolute_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from routes.inference import _reject_uncontained_local_path

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    outside = tmp_path / "owner-private.gguf"
    outside.write_bytes(b"weights")

    token = _bind("alice")
    try:
        mine = workspace_root() / "models"
        mine.mkdir(parents = True, exist_ok = True)
        private = mine / "alice.gguf"
        private.write_bytes(b"weights")
        with pytest.raises(HTTPException) as excinfo:
            _reject_uncontained_local_path(str(outside), "load")
        assert excinfo.value.status_code == 403
        # Its own file, and a hub repo id, both still pass.
        _reject_uncontained_local_path(str(private), "load")
        _reject_uncontained_local_path("unsloth/gemma-3-270m", "load")
    finally:
        reset_workspace_subject(token)

    # The owner keeps absolute paths: that is the single-user behaviour.
    token = _bind("unsloth")
    try:
        _reject_uncontained_local_path(str(outside), "load")
    finally:
        reset_workspace_subject(token)


def test_the_diffusion_lora_catalog_is_per_account():
    from core.inference import diffusion_lora

    dirs = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            dirs[subject] = diffusion_lora.loras_dir()
        finally:
            reset_workspace_subject(token)
    assert dirs["unsloth"] != dirs["alice"]
    assert dirs["unsloth"] == studio_root() / "loras" / "diffusion"
    assert workspace_key("alice") in str(dirs["alice"])


def test_a_new_recipe_job_drops_the_previous_accounts_event_subscribers():
    from core.data_recipe.jobs import manager

    src = inspect.getsource(manager)
    start = src.index("self._events.clear()")
    assert "self._subs.clear()" in src[start : start + 400]


def test_install_wide_settings_are_read_from_the_owner_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from storage.studio_db import get_install_setting, upsert_install_settings

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    token = _bind("unsloth")
    try:
        upsert_install_settings({"hugging_face_cache_home": "/shared/hf"})
    finally:
        reset_workspace_subject(token)

    # A managed account must see the owner's value, not an empty per-account one.
    token = _bind("alice")
    try:
        assert get_install_setting("hugging_face_cache_home") == "/shared/hf"
        assert studio_db.get_app_setting("hugging_face_cache_home") is None
    finally:
        reset_workspace_subject(token)


def test_a_preview_capability_names_the_workspace_that_minted_it():
    from utils.preview_token import (
        preview_token_subject,
        sign_preview_ref,
        verify_preview_ref,
    )

    tokens = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            tokens[subject] = sign_preview_ref("run-1/checkpoint-100")
        finally:
            reset_workspace_subject(token)
    assert tokens["unsloth"] != tokens["alice"]
    assert preview_token_subject(tokens["alice"]) == "alice"
    assert verify_preview_ref("run-1/checkpoint-100", tokens["alice"])
    # Alice's token must not be reshaped into one for the owner's identical ref.
    forged = tokens["unsloth"].split(".", 1)[0] + "." + tokens["alice"].split(".", 1)[1]
    assert not verify_preview_ref("run-1/checkpoint-100", forged)


def test_the_export_log_stream_stops_when_another_account_takes_the_buffer():
    import routes.export as export_routes

    src = inspect.getsource(export_routes)
    loop = src.index("entries, new_cursor = backend.get_logs_since(cursor)")
    # The re-check must sit inside the loop, not only at the route entry.
    assert "owns_workspace(current_subject)" in src[loop - 700 : loop]


def test_training_refuses_to_start_over_another_accounts_export():
    import routes.training as training_routes

    src = inspect.getsource(training_routes.start_training)
    assert "export_owns(current_subject)" in src
    assert "An export is running in another account" in src


def test_the_diffusion_dataset_interlock_only_blocks_the_running_account():
    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService.__new__(DiffusionTrainingService)
    service._lock = threading.RLock()
    service._reserved = True
    service._proc = None
    service._dataset_mutations = 0
    service._active_workspace_subject = "alice"

    token = _bind("bob")
    try:
        with service.dataset_mutation():
            pass  # Bob's own dataset tree is untouched by Alice's run.
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        with pytest.raises(Exception, match = "cannot be changed"):
            with service.dataset_mutation():
                pass
    finally:
        reset_workspace_subject(token)


def test_an_image_generation_is_invisible_and_uncancellable_to_other_accounts():
    from core.inference.diffusion import DiffusionBackend, _GenState

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._lock = threading.RLock()
    backend._generation_cancel_lock = threading.RLock()
    backend._gen = _GenState(total_steps = 20, step = 7, subject = "alice")
    cancel = threading.Event()
    backend._active_generate_cancel = cancel

    assert backend.generate_progress("alice")["active"] is True
    assert backend.generate_progress("bob")["active"] is False
    # Answered as idle rather than refused: Bob's page settles its button exactly
    # as it would against an engine with nothing running, which is what he sees.
    assert backend.cancel_generate("bob") is False
    assert not cancel.is_set()
    assert backend.cancel_generate("alice") is True
    assert cancel.is_set()

    # The teardown path passes no subject and must still stop whatever is running.
    cancel.clear()
    assert backend.cancel_generate() is True
    assert cancel.is_set()


def test_a_video_generation_is_invisible_and_uncancellable_to_other_accounts():
    from core.inference.video import VideoBackend

    backend = VideoBackend.__new__(VideoBackend)
    backend._lock = threading.RLock()
    backend._gen = {"active": True, "phase": "denoising", "step": 3, "total": 10}
    backend._generate_job_active = True
    backend._gen_video_id = "vid-1"
    backend._gen_subject = "alice"
    cancel = threading.Event()
    backend._active_generate_cancel = cancel

    assert backend.generate_progress("alice")["active"] is True
    assert backend.generate_progress("bob")["active"] is False
    assert backend.cancel_generate(None, "bob") is False
    assert not cancel.is_set()
    assert backend.cancel_generate(None, "alice") is True
    assert cancel.is_set()


def test_training_refuses_weights_outside_the_callers_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from routes.training import _reject_uncontained_training_path

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    outside = tmp_path / "someone-elses.safetensors"
    outside.write_bytes(b"weights")

    token = _bind("alice")
    try:
        mine = workspace_root() / "models"
        mine.mkdir(parents = True, exist_ok = True)
        private = mine / "alice.safetensors"
        private.write_bytes(b"weights")
        # Containing the outputs scopes what training WRITES; the base weights it
        # reads are still whatever path the request named.
        with pytest.raises(HTTPException) as excinfo:
            _reject_uncontained_training_path(str(outside))
        assert excinfo.value.status_code == 403
        _reject_uncontained_training_path(str(private))
        # A Hub repo id is not a path and must stay loadable.
        _reject_uncontained_training_path("unsloth/Llama-3.2-1B")
        _reject_uncontained_training_path(None)
    finally:
        reset_workspace_subject(token)

    owner = _bind("unsloth")
    try:
        _reject_uncontained_training_path(str(outside))
    finally:
        reset_workspace_subject(owner)


def test_only_the_owner_can_load_a_model_that_runs_its_own_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from routes.inference import _reject_remote_code_from_a_managed_account

    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(
        auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password"
    )
    auth_storage.create_initial_user(
        "unsloth", "owner-password", secrets.token_urlsafe(64), is_admin = True
    )
    auth_storage.create_managed_user("alice")

    token = _bind("alice")
    try:
        # A repo id is not a path, so containment never sees this one: the repo's
        # own Python would run as the backend user with every workspace readable.
        with pytest.raises(HTTPException) as excinfo:
            _reject_remote_code_from_a_managed_account(True)
        assert excinfo.value.status_code == 403
        _reject_remote_code_from_a_managed_account(False)
    finally:
        reset_workspace_subject(token)

    owner = _bind("unsloth")
    try:
        _reject_remote_code_from_a_managed_account(True)
    finally:
        reset_workspace_subject(owner)


def test_a_second_accounts_start_request_id_cannot_settle_the_first_ones():
    from core.training.training import TrainingBackend

    backend = TrainingBackend.__new__(TrainingBackend)
    backend._lock = threading.RLock()
    backend._start_requests = {}
    backend._start_cancel_tombstones = {}
    backend._start_cancel_tombstone_reservations = {}
    backend._pending_start_key = None
    backend._status_start_key = None
    backend._current_start_key = None

    token = _bind("alice")
    try:
        assert backend.reserve_start_request("same-id", "job-alice")[0] == "reserved"
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The pending interlock is install-wide (one GPU), so Bob is refused. His
        # rejection must land under his own key, not on top of Alice's pending
        # record, or Alice's resolve reads his outcome as hers.
        outcome, record = backend.reserve_start_request("same-id", "job-bob")
        assert outcome == "conflict" and record.state == "rejected"
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        settled = backend.resolve_start_request(
            "same-id", state = "accepted", message = "Training is starting"
        )
        assert settled.state == "accepted" and settled.job_id == "job-alice"
    finally:
        reset_workspace_subject(token)

    assert set(backend._start_requests) == {("alice", "same-id"), ("bob", "same-id")}


def test_the_research_supervisor_reads_the_account_list_off_the_event_loop():
    import asyncio as _asyncio

    from core.research_runs import ResearchSupervisor

    supervisor = ResearchSupervisor.__new__(ResearchSupervisor)
    supervisor._workspaces_cache = None
    supervisor._workspaces_cache_expires = 0.0
    calls = []

    def _workspaces():
        # Stands in for the auth.db open, which applies a five second busy
        # timeout: on the loop it would stall every request and inference stream.
        calls.append(threading.current_thread())
        return ["unsloth", "alice"]

    supervisor._workspaces = _workspaces

    async def _run():
        loop_thread = threading.current_thread()
        first = await supervisor._workspaces_async()
        second = await supervisor._workspaces_async()
        return first, second, loop_thread

    first, second, loop_thread = _asyncio.run(_run())
    assert first == second == ["unsloth", "alice"]
    # Once, not twice: an idle supervisor stops touching the database at all.
    assert len(calls) == 1
    assert calls[0] is not loop_thread


def test_the_upload_cap_is_installation_wide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from utils import upload_limits

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    owner = _bind("unsloth")
    try:
        upload_limits.set_upload_limit_mb(2048)
    finally:
        reset_workspace_subject(owner)

    token = _bind("alice")
    try:
        # MaxBodyMiddleware resolves this before anything has authenticated, so a
        # per-account value could be saved and then never honoured. Both sides now
        # read the same place.
        assert upload_limits.get_upload_limit_mb() == 2048
    finally:
        reset_workspace_subject(token)


def test_deleting_an_account_stops_the_jobs_it_owns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    stopped = []

    class _Backend:
        def is_training_active(self):
            return True

        def owns_workspace(self, subject):
            return subject == "casey"

        def stop_training(self, save = True):
            stopped.append(("training", current_workspace_subject(), save))

    class _Service:
        def is_active(self):
            return True

        def owns_workspace(self, subject = None):
            return subject == "casey"

        def stop(self, save = True):
            stopped.append(("diffusion", current_workspace_subject(), save))

    class _Export:
        def is_export_active(self):
            return True

        def owns_workspace(self, subject = None):
            return subject == "casey"

        def cancel_export(self):
            stopped.append(("export", current_workspace_subject(), False))

    import core.training.training as training_module
    import core.training.diffusion_training_service as diffusion_module
    import core.export as export_module

    monkeypatch.setattr(training_module, "get_training_backend", lambda: _Backend())
    monkeypatch.setattr(diffusion_module, "get_diffusion_training_service", lambda: _Service())
    monkeypatch.setattr(export_module, "get_export_backend", lambda: _Export())

    auth_storage.delete_managed_user("casey")
    # Otherwise the worker outlives the row that authorised it: the owner's
    # ownership guards hide it, and the deleted account cannot sign in to stop it.
    assert [entry[0] for entry in stopped] == ["training", "diffusion", "export"]
    # Stopped in the deleted account's own workspace, so each subsystem's
    # per-workspace state is the one being torn down.
    assert {entry[1] for entry in stopped} == {"casey"}


def test_a_managed_account_cannot_migrate_the_owners_legacy_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sandboxes"))
    legacy = tmp_path / "home" / "studio_sandbox"
    session = legacy / "chat-1"
    session.mkdir(parents = True)
    (session / "owner-notes.txt").write_text("owner", encoding = "utf-8")
    monkeypatch.setattr(tools, "_legacy_sandbox_root", lambda: str(legacy))
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", False)

    token = _bind("alice")
    try:
        mine = Path(tools.sandbox_root())
        # Session ids reach this path from the caller, so naming an owner session
        # is not a secret; the move would both expose and destroy the original.
        tools._migrate_one_legacy_session(str(mine), "chat-1")
    finally:
        reset_workspace_subject(token)

    assert (session / "owner-notes.txt").exists()


def test_destroying_the_shared_cache_is_owner_only():
    import inspect

    from hub.routes import datasets as hub_datasets
    from hub.routes import inventory as hub_inventory
    from routes import models as model_routes

    for func in (
        model_routes.delete_cached_model,
        hub_inventory.delete_cached_model,
        hub_datasets.delete_cached_dataset,
    ):
        # The model and dataset caches stayed installation-wide by design, so a
        # delete here discards whatever any account downloaded, possibly from a
        # gated repo only they can fetch again. Reading them stays open.
        default = inspect.signature(func).parameters["current_subject"].default
        assert getattr(default, "dependency", None) is require_install_admin


def test_a_managed_accounts_preview_prompts_are_filed_under_their_own_name():
    import inspect

    from routes import preview as preview_routes

    src = inspect.getsource(preview_routes._serve_chat)
    # The context manager binds the filesystem, but these two take the subject as
    # an explicit argument, and the chat one records the prompt in the
    # process-global API monitor.
    assert "subject = current_workspace_subject()" in src
    assert "DEFAULT_ADMIN_USERNAME" not in src


def test_unloading_the_image_engine_cannot_end_another_accounts_generation():
    from core.inference.diffusion import DiffusionBackend, _GenState
    from utils.workspace_context import ForeignWorkspaceActiveError

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._lock = threading.RLock()
    backend._generation_cancel_lock = threading.RLock()
    backend._gen = _GenState(total_steps = 20, step = 7, subject = "alice")
    cancel = threading.Event()
    backend._active_generate_cancel = cancel

    # Scoping cancel_generate alone left this open: unload signals the same event,
    # so the authenticated unload route was still a way to end somebody else's run.
    with pytest.raises(ForeignWorkspaceActiveError):
        backend._refuse_foreign_teardown("bob")
    assert not cancel.is_set()
    backend._refuse_foreign_teardown("alice")
    # The engine's own teardown path passes nothing and must never be refused.
    backend._refuse_foreign_teardown(None)


def test_unloading_the_video_backend_cannot_end_another_accounts_generation():
    from core.inference.video import VideoBackend
    from utils.workspace_context import ForeignWorkspaceActiveError

    backend = VideoBackend.__new__(VideoBackend)
    backend._lock = threading.RLock()
    backend._generate_job_active = True
    backend._gen_subject = "alice"

    with pytest.raises(ForeignWorkspaceActiveError):
        backend.unload("bob")
    # Nothing running means nothing to protect, whoever asks.
    backend._generate_job_active = False
    assert backend._gen_subject == "alice"


def test_a_name_stays_reserved_while_its_jobs_are_still_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _auth_db(tmp_path, monkeypatch)
    auth_storage.create_managed_user("casey")
    for root in auth_storage._subject_owned_roots("casey"):
        root.mkdir(parents = True, exist_ok = True)
    # Quiescing signals; it does not wait. A worker still unwinding stays bound to
    # the subject, so releasing the name lets a namesake share its workspace.
    monkeypatch.setattr(auth_storage, "_quiesce_workspace_jobs", lambda username: None)
    monkeypatch.setattr(auth_storage, "_workspace_jobs_active", lambda username: True)
    auth_storage.delete_managed_user("casey")
    assert auth_storage.username_is_retired("casey") is True

    monkeypatch.setattr(auth_storage, "_workspace_jobs_active", lambda username: False)
    # Once the worker is gone the existing retry on the create path releases it.
    assert auth_storage.username_is_retired("casey") is False


def test_workspace_jobs_active_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _auth_db(tmp_path, monkeypatch)
    import core.training.training as training_module

    def _boom():
        raise RuntimeError("subsystem unavailable")

    monkeypatch.setattr(training_module, "get_training_backend", _boom)
    # One name reserved a while longer beats a live worker writing into the files
    # of whoever registers that name next.
    assert auth_storage._workspace_jobs_active("casey") is True


def test_remote_code_training_and_export_are_owner_only():
    import inspect

    from routes import export as export_routes
    from routes import training as training_routes

    start = inspect.getsource(training_routes.start_training)
    assert "_reject_remote_code_from_a_managed_account(request.trust_remote_code)" in start
    load = inspect.getsource(export_routes.load_checkpoint)
    assert "_reject_remote_code_from_a_managed_account(request.trust_remote_code)" in load


def test_controlnets_are_per_account_like_the_diffusion_loras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.inference import diffusion_controlnet, diffusion_lora

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    seen = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            seen[subject] = diffusion_controlnet.controlnets_dir().resolve()
            # The sibling catalog moved to the workspace root in this branch; this
            # one was left behind, so the owner's local weights were in every
            # account's picker.
            assert seen[subject].parent.parent == diffusion_lora.loras_dir().parent.parent
        finally:
            reset_workspace_subject(token)
    assert seen["unsloth"] != seen["alice"]
    assert seen["unsloth"] == (studio_root() / "controlnets" / "diffusion").resolve()


def test_media_and_export_loads_refuse_paths_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import inspect

    from routes import export as export_routes
    from routes import inference as inference_routes
    from routes import video as video_routes

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    # The text load path got containment through _resolve_model_identifier_for_request;
    # these three never went near it and their validators accept any local path.
    assert '_reject_uncontained_local_path(request.model_path, "load")' in inspect.getsource(
        inference_routes.load_diffusion_model_gated
    )
    assert '_reject_uncontained_local_path(request.model_path, "load")' in inspect.getsource(
        video_routes.load_video_model_gated
    )
    assert '_reject_uncontained_local_path(request.checkpoint_path, "export")' in inspect.getsource(
        export_routes.load_checkpoint
    )


def test_a_recipe_cannot_seed_from_another_accounts_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException
    from core.data_recipe.jobs.manager import _reject_uncontained_recipe_paths

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    outside = tmp_path / "owner-private.jsonl"
    outside.write_text("secret", encoding = "utf-8")

    token = _bind("alice")
    try:
        mine = workspace_root() / "datasets"
        mine.mkdir(parents = True, exist_ok = True)
        ours = mine / "seed.jsonl"
        ours.write_text("mine", encoding = "utf-8")

        # Nested, because the artifact root only confines what the worker writes
        # and the recipe itself is forwarded verbatim.
        with pytest.raises(HTTPException) as excinfo:
            _reject_uncontained_recipe_paths(
                {"columns": [{"seed": {"source": {"type": "local", "path": str(outside)}}}]}
            )
        assert excinfo.value.status_code == 403
        with pytest.raises(HTTPException):
            _reject_uncontained_recipe_paths({"seed": {"paths": [str(ours), str(outside)]}})
        _reject_uncontained_recipe_paths({"seed": {"paths": [str(ours)]}})
        # A string field named "path" that is not a file on disk is untouched.
        _reject_uncontained_recipe_paths({"path": "some/relative/thing"})
    finally:
        reset_workspace_subject(token)


def test_the_embedding_model_setting_cannot_name_another_workspace():
    import inspect

    from routes import settings as settings_routes

    src = inspect.getsource(settings_routes.update_embedding_model)
    # Caching the choice per workspace stopped it reaching another account's RAG;
    # it did not stop the choice itself naming a path the loader then opens.
    assert '_reject_uncontained_local_path(model, "use embedding models from")' in src


def test_the_embedding_resolution_memo_is_per_account():
    from utils import embedding_model_settings as ems

    saved = dict(ems._resolved_gguf_memo)
    ems._resolved_gguf_memo.clear()
    try:
        stored = (None, None, "alice/repo", "llama", True, {"files": ["a.gguf"]})
        token = _bind("alice")
        try:
            ems._remember_resolution("shared-model", stored)
        finally:
            reset_workspace_subject(token)
        token = _bind("bob")
        try:
            # Keyed by model alone, whichever account resolved last decided which
            # weights the other's ingestion loaded mid-index.
            assert ems._remembered("shared-model") is None
        finally:
            reset_workspace_subject(token)
        token = _bind("alice")
        try:
            assert ems._remembered("shared-model")[0] == "alice/repo"
        finally:
            reset_workspace_subject(token)
    finally:
        ems._resolved_gguf_memo.clear()
        ems._resolved_gguf_memo.update(saved)


def test_the_model_catalog_cache_is_not_read_across_accounts():
    from routes import inference as inference_routes

    saved = dict(inference_routes._CATALOG_CACHE)
    saved_adv = dict(inference_routes._ADVERTISED_CACHE)
    try:
        inference_routes._CATALOG_CACHE.update(
            subject = "alice",
            at = 1.0,
            models = [SimpleNamespace(model_id = "shared-alias", id = None, path = "/alice/m.gguf")],
        )
        inference_routes._ADVERTISED_CACHE.update(at = None, subject = None, paths = {})
        token = _bind("alice")
        try:
            assert inference_routes._advertised_local_path("shared-alias") == "/alice/m.gguf"
        finally:
            reset_workspace_subject(token)
        token = _bind("bob")
        try:
            # Otherwise Bob's completion probes Alice's private scan-folder path,
            # and the rejection it produces tells him that model is there.
            assert inference_routes._advertised_local_path("shared-alias") is None
            assert inference_routes._innermost_indexed_owner("/alice/m.gguf") is None
        finally:
            reset_workspace_subject(token)
    finally:
        inference_routes._CATALOG_CACHE.clear()
        inference_routes._CATALOG_CACHE.update(saved)
        inference_routes._ADVERTISED_CACHE.clear()
        inference_routes._ADVERTISED_CACHE.update(saved_adv)


def test_one_accounts_clear_all_does_not_fence_anothers_search_images():
    from core.inference import search_images
    search_images.reset_registry_for_tests()
    try:
        token = _bind("bob")
        try:
            # Bob samples the generation his in-flight registration will check.
            bob_generation = search_images.cache_generation()
        finally:
            reset_workspace_subject(token)

        token = _bind("alice")
        try:
            search_images.state_for_tests().registry["aaaaaaaaaaaa"] = {
                "thumbnail": "https://example.test/a.jpg",
                "source": "https://example.test/a",
                "created": time.monotonic(),
                "policy": None,
            }
            # Alice clears all her chats. The snapshot used to pick up every
            # account's ids and the fence refused every account's in-flight work.
            ids = search_images.snapshot_and_fence_registrations()
            assert ids is None or "aaaaaaaaaaaa" in ids
        finally:
            reset_workspace_subject(token)

        token = _bind("bob")
        try:
            assert search_images.cache_generation() == bob_generation
            with search_images._registry_lock:
                assert search_images._reaped_since_locked("ffffffffffff", bob_generation) is False
        finally:
            reset_workspace_subject(token)
    finally:
        search_images.reset_registry_for_tests()


def test_the_unstructured_chunk_cache_follows_the_calling_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import importlib.util

    # Loaded by path: the plugin package's __init__ pulls in data_designer, which
    # is not installed here, and this module has no relative imports of its own.
    source = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "data-designer-unstructured-seed"
        / "src"
        / "data_designer_unstructured_seed"
        / "chunking.py"
    )
    spec = importlib.util.spec_from_file_location("_unstructured_chunking", source)
    chunking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunking)

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    seen = {}
    for subject in ("unsloth", "alice"):
        token = _bind(subject)
        try:
            # Pinned at import, whichever account opened the first preview owned
            # the directory for the process and everyone else wrote under it.
            seen[subject] = chunking._cache_dir().resolve()
        finally:
            reset_workspace_subject(token)
    assert seen["unsloth"] != seen["alice"]


def test_two_accounts_can_use_the_same_chat_run_id():
    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    supervisor = ChatGenerationSupervisor.__new__(ChatGenerationSupervisor)
    supervisor._tasks = {}
    supervisor._cancel_events = {}
    supervisor._active_registrations = {}
    supervisor._subjects = {}
    supervisor._activities = {}
    supervisor._shutdown_runs = set()
    supervisor._stopping = False

    alice_event = threading.Event()
    bob_event = threading.Event()
    for subject, event in (("alice", alice_event), ("bob", bob_event)):
        token = _bind(subject)
        try:
            key = supervisor._key("same-run")
            supervisor._cancel_events[key] = event
            supervisor._subjects[key] = subject
        finally:
            reset_workspace_subject(token)

    # Keyed by the id alone, Bob's entry replaced Alice's and his cancel signalled
    # her producer while his own database run stayed queued forever.
    assert supervisor.owns_run("same-run", "alice") is True
    assert supervisor.owns_run("same-run", "bob") is True

    token = _bind("bob")
    try:
        supervisor._cancel_locally("same-run")
    finally:
        reset_workspace_subject(token)
    assert bob_event.is_set() and not alice_event.is_set()


def test_a_run_id_registered_to_one_account_is_not_owned_by_another():
    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    supervisor = ChatGenerationSupervisor.__new__(ChatGenerationSupervisor)
    supervisor._subjects = {("alice", "run-1"): "alice"}
    assert supervisor.owns_run("run-1", "bob") is False
    # An id this supervisor has never seen, a run from before a restart say, stays
    # cancellable by the owner, which is what the previous default did.
    assert supervisor.owns_run("unknown-run", "unsloth") is True
    assert supervisor.owns_run("unknown-run", "bob") is False


def test_the_diffusion_load_worker_runs_in_the_requesting_workspace(
    monkeypatch: pytest.MonkeyPatch,
):
    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._lock = threading.RLock()
    backend._loading = None
    backend._load_token = 0
    backend._cancel_event = threading.Event()

    class _Fam:
        name = "flux"
        base_repo = "base/repo"

    monkeypatch.setattr(DiffusionBackend, "validate_load_request", lambda self, *a, **k: _Fam())
    monkeypatch.setattr(DiffusionBackend, "assert_precision_available", lambda self, *a, **k: None)
    monkeypatch.setattr(DiffusionBackend, "status", lambda self: {})

    seen: dict[str, object] = {}
    done = threading.Event()

    def _record(self, **kwargs):
        # loras_dir() is workspace-dependent, so the subject the worker sees decides
        # which account's adapters the load-time bake resolves against.
        seen["subject"] = current_workspace_subject()
        seen["loras"] = kwargs.get("loras")
        done.set()

    monkeypatch.setattr(DiffusionBackend, "_run_load", _record)

    token = _bind("alice")
    try:
        backend.begin_load("some/repo", loras = [("alice-only", 1.0)])
    finally:
        reset_workspace_subject(token)

    assert done.wait(timeout = 5)
    assert seen["subject"] == "alice"
    assert seen["loras"] == [("alice-only", 1.0)]


def test_me_reports_this_accounts_own_password_requirement(
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
        owner_headers = {
            "Authorization": "Bearer "
            + client.post(
                "/api/auth/login",
                json = {"username": "unsloth", "password": "owner-password"},
            ).json()["access_token"]
        }
        setup_code = client.post(
            "/api/auth/users", headers = owner_headers, json = {"username": "alice"}
        ).json()["setup_code"]
        alice_headers = {
            "Authorization": "Bearer "
            + client.post(
                "/api/auth/login",
                json = {"username": "alice", "password": setup_code},
            ).json()["access_token"]
        }

        # Reachable DURING the forced change, and describing the CALLER: /auth/status
        # is unauthenticated and answers for the owner, so a signed-in managed account
        # has nowhere else to read its own requirement from, and a client that used
        # /status instead followed the owner's recovery into a redirect loop.
        me = client.get("/api/auth/me", headers = alice_headers)
        assert me.status_code == 200
        assert me.json() == {
            "username": "alice",
            "is_admin": False,
            "must_change_password": True,
        }

        owner_me = client.get("/api/auth/me", headers = owner_headers)
        assert owner_me.json()["must_change_password"] is False

        changed = client.post(
            "/api/auth/change-password",
            headers = alice_headers,
            json = {
                "current_password": setup_code,
                "new_password": "alice-permanent-password",
            },
        )
        assert changed.status_code == 200
        settled_headers = {"Authorization": f"Bearer {changed.json()['access_token']}"}
        assert (
            client.get("/api/auth/me", headers = settled_headers).json()["must_change_password"]
            is False
        )


def test_media_renders_and_recipe_jobs_are_quiesced_before_a_name_is_released(
    monkeypatch: pytest.MonkeyPatch,
):
    import sys
    import types

    cancelled: list[str] = []

    class _Engine:
        def __init__(self, name: str, active: bool):
            self.name = name
            self.active = active

        def generate_progress(self, subject = None):
            assert subject == "alice"
            return {"active": self.active}

        def cancel_generate(self, subject = None):
            assert subject == "alice"
            cancelled.append(self.name)
            self.active = False
            return True

    diffusion = _Engine("diffusion", True)
    video = _Engine("video", False)
    monkeypatch.setitem(
        sys.modules,
        "core.inference.diffusion",
        types.SimpleNamespace(get_diffusion_backend = lambda: diffusion),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.inference.video",
        types.SimpleNamespace(get_video_backend = lambda: video),
    )
    # Never imported in this process, so it cannot be holding a render and must
    # not be imported just to ask.
    monkeypatch.delitem(sys.modules, "core.inference.sd_cpp_backend", raising = False)

    class _Manager:
        def __init__(self):
            self.alive = True
            self.cancelled: list[str] = []

        def is_active(self):
            return self.alive

        def owns_workspace(self, subject = None):
            return subject == "alice"

        def get_current_job_id(self):
            return "recipe-1"

        def cancel(self, job_id):
            self.cancelled.append(job_id)
            self.alive = False
            return True

    manager = _Manager()
    monkeypatch.setitem(
        sys.modules,
        "core.data_recipe.jobs.manager",
        types.SimpleNamespace(get_job_manager = lambda: manager),
    )

    assert auth_storage._workspace_jobs_active("alice") is True
    auth_storage._quiesce_workspace_jobs("alice")
    assert cancelled == ["diffusion"]
    assert manager.cancelled == ["recipe-1"]
    # Only once both are stopped may the tombstone be released, or the render
    # persists into whoever takes the name next.
    assert auth_storage._workspace_jobs_active("alice") is False

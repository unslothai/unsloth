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
    LEGACY_WORKSPACE_SUBJECT,
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
    service._dataset_mutations = {}
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
    backend._active_generate_cancel_subject = "alice"

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
    backend._active_generate_cancel_subject = "alice"

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
    backend._loading = None
    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    backend._active_generate_cancel_subject = "alice"

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


def test_cancelling_a_run_does_not_signal_a_namesake_in_another_workspace():
    from state import active_generations

    active_generations.reset_for_tests()
    events = {}
    handles = []
    try:
        for subject in ("alice", "bob"):
            token = _bind(subject)
            try:
                events[subject] = threading.Event()
                handle = active_generations.ActiveGeneration(
                    events[subject],
                    thread_id = f"thread-{subject}",
                    run_id = "same-run",
                )
                handle.__enter__()
                handles.append(handle)
            finally:
                reset_workspace_subject(token)

        # Run ids are client-supplied, so an unscoped cancel_run reached every
        # workspace holding a registration under the same id.
        assert active_generations.cancel_run("same-run", subject = "bob") == 1
        assert events["bob"].is_set()
        assert not events["alice"].is_set()
    finally:
        for handle in handles:
            handle.__exit__(None, None, None)
        active_generations.reset_for_tests()


def test_sandbox_workdirs_are_cached_per_workspace(tmp_path, monkeypatch):
    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))
    tools._workdirs.clear()

    # A client-chosen session id, presented by both accounts.
    dirs = {}
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            dirs[subject] = tools.get_sandbox_workdir("shared-session")
        finally:
            reset_workspace_subject(token)

    assert dirs["alice"] != dirs["bob"]
    # Keyed by the id alone, the second call overwrote the first entry and the
    # first account's next tool call ran in the second account's sandbox.
    for subject, expected in dirs.items():
        token = _bind(subject)
        try:
            assert tools.get_sandbox_workdir("shared-session") == expected
        finally:
            reset_workspace_subject(token)
    tools._workdirs.clear()


def test_queued_image_cancels_only_reach_the_callers_own_requests():
    import threading as _threading

    from core.inference.diffusion import DiffusionBackend

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._generation_cancel_lock = _threading.RLock()
    backend._gen = None
    backend._active_generate_cancel = None
    backend._active_generate_cancel_subject = None
    backend._generation_owns_slot = False
    backend._teardown_waiters = 1
    backend._transition_owns_slot = False
    alice_queued = _threading.Event()
    bob_queued = _threading.Event()
    backend._queued_generate_cancels = {alice_queued: "alice", bob_queued: "bob"}

    # _gen is None while a request waits behind a transition, so the subject check
    # on the live generation has nothing to compare and every queued event was set.
    assert backend.cancel_generate(subject = "bob") is True
    assert bob_queued.is_set() and not alice_queued.is_set()

    # Admitted but not yet denoising: still not another account's to cancel.
    backend._queued_generate_cancels = {}
    admitted = _threading.Event()
    backend._active_generate_cancel = admitted
    backend._active_generate_cancel_subject = "alice"
    assert backend.cancel_generate(subject = "bob") is False
    assert not admitted.is_set()
    assert backend.cancel_generate(subject = "alice") is True
    assert admitted.is_set()


def test_the_image_persist_marker_is_read_per_account():
    from routes import inference as inference_routes

    inference_routes._diffusion_persist_active.clear()
    inference_routes._begin_image_persist("alice")
    assert inference_routes._diffusion_persist_active.get("bob", 0) == 0
    # Still process-wide for liveness: one account's persist keeps the box busy.
    assert inference_routes.generation_in_flight() is True
    inference_routes._begin_image_persist("alice")
    inference_routes._end_image_persist("alice")
    assert inference_routes._diffusion_persist_active.get("alice") == 1
    inference_routes._end_image_persist("alice")
    assert inference_routes._diffusion_persist_active == {}
    assert inference_routes.generation_in_flight() is False


def test_an_in_flight_model_load_is_not_another_accounts_to_tear_down():
    import threading as _threading

    from core.inference.diffusion import DiffusionBackend, _LoadingState
    from utils.workspace_context import ForeignWorkspaceActiveError

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._lock = _threading.RLock()
    backend._generation_cancel_lock = _threading.RLock()
    backend._gen = None
    backend._loading = _LoadingState(
        repo_id = "alice/private-model",
        base_repo = "alice/private-model",
        subject = "alice",
    )

    # _gen is None during the download, so the generation guard had nothing to
    # look at and the authenticated unload route ended the other account's pull.
    with pytest.raises(ForeignWorkspaceActiveError):
        backend._refuse_foreign_teardown("bob")
    backend._refuse_foreign_teardown("alice")
    # The engine's own teardown paths still pass, whoever started the load.
    backend._refuse_foreign_teardown(None)

    # A load that already failed is nobody's to protect.
    backend._loading.error = "boom"
    backend._refuse_foreign_teardown("bob")


def test_a_retirement_during_schema_creation_is_not_undone_by_the_add():
    from storage import schema_cache

    cache: set[str] = set()
    schema_cache.register(cache)

    # What a store does: read the generation, create the schema, then record the
    # path. A retirement landing in that window used to be erased by the add, so
    # the retired pathname stayed "ready" and the namesake's fresh empty database
    # failed its first query with no such table until the process restarted.
    generation = schema_cache.generation()
    schema_cache.forget_all()
    schema_cache.mark_ready(cache, "/workspaces/alice/studio.db", generation)
    assert cache == set()

    # The ordinary path still caches.
    generation = schema_cache.generation()
    schema_cache.mark_ready(cache, "/workspaces/alice/studio.db", generation)
    assert cache == {"/workspaces/alice/studio.db"}
    schema_cache.forget_all()
    assert cache == set()


def test_a_privately_loaded_media_model_is_not_generated_with_by_others(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))

    token = _bind("alice")
    try:
        private = workspace_root() / "models" / "secret-model"
        private.mkdir(parents = True, exist_ok = True)
        alice_status = {"loaded": True, "repo_id": str(private)}
        # Her own model: nothing to refuse.
        inference_routes._reject_foreign_private_resident_model(alice_status, "image")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Resident, and Bob names no model at all: the load-time containment check
        # is long past, so without this he generates with her weights and reads the
        # path back out of the recipe.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_foreign_private_resident_model(alice_status, "image")
        assert exc.value.status_code == 403
        # A Hub repo id is not a path and stays shared, which is the design.
        inference_routes._reject_foreign_private_resident_model(
            {"loaded": True, "repo_id": "black-forest-labs/FLUX.1-dev"}, "image"
        )
        inference_routes._reject_foreign_private_resident_model({"loaded": False}, "image")
    finally:
        reset_workspace_subject(token)


def test_the_shared_model_cache_stays_shared_between_accounts(tmp_path, monkeypatch):
    from routes import inference as inference_routes
    from utils.paths.storage_roots import cache_root

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))

    # The legacy owner's workspace_root() IS studio_root(), and cache_root() sits
    # inside it, so a plain "under the workspace" test would call every shared
    # model the owner's private one and stop every other account generating.
    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        cached = cache_root() / "models--org--model"
        cached.mkdir(parents = True, exist_ok = True)
        shared_status = {"loaded": True, "repo_id": str(cached)}
    finally:
        reset_workspace_subject(token)

    for subject in (LEGACY_WORKSPACE_SUBJECT, "alice", "bob"):
        token = _bind(subject)
        try:
            inference_routes._reject_foreign_private_resident_model(shared_status, "image")
        finally:
            reset_workspace_subject(token)


def test_load_progress_hides_another_accounts_download():
    import threading as _threading

    from core.inference.diffusion import DiffusionBackend, _LoadingState

    backend = DiffusionBackend.__new__(DiffusionBackend)
    backend._lock = _threading.RLock()
    backend._state = None
    backend._loading = _LoadingState(
        repo_id = "/workspaces/alice-abc/models/secret",
        base_repo = "/workspaces/alice-abc/models/secret",
        expected_bytes = 4_000_000_000,
        subject = "alice",
    )

    # The payload names the repo being pulled, which for a private path is the
    # other account's own directory.
    assert backend.load_progress("bob")["phase"] is None
    assert backend.load_progress("alice")["phase"] is not None
    # The engine's own probes keep the unfiltered view.
    assert backend.load_progress()["phase"] is not None


def test_only_the_owner_may_point_the_backend_at_a_private_address(monkeypatch):
    from urllib.parse import urlparse

    from fastapi import HTTPException

    from core.inference.providers import validate_provider_base_url
    from routes import mcp_servers as mcp_routes

    monkeypatch.delenv("UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS", raising = False)

    private = "http://127.0.0.1:11434/v1"
    # A public IP literal, not a hostname: _reject_non_public resolves names, and
    # this suite must not depend on DNS. Note the consequence for real installs,
    # which the opt-in env flag already had: an unresolvable host fails closed, so
    # a managed account on a box with no DNS cannot add a provider it could not
    # have reached either.
    public = "https://8.8.8.8/v1"

    # A local Ollama or llama.cpp endpoint is the ordinary reason to run Unsloth,
    # so the owner keeps it. Single-user installs only ever have the owner.
    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert validate_provider_base_url(private) == private
        mcp_routes._reject_private_target_for_managed_accounts(urlparse(private))
    finally:
        reset_workspace_subject(token)

    # A managed account cannot reach that address from its browser, so naming it
    # here would make the backend probe it on the account's behalf.
    token = _bind("alice")
    try:
        with pytest.raises(ValueError):
            validate_provider_base_url(private)
        with pytest.raises(HTTPException) as exc:
            mcp_routes._reject_private_target_for_managed_accounts(urlparse(private))
        assert exc.value.status_code == 403
        # Public targets are unaffected for everybody.
        assert validate_provider_base_url(public) == public
        mcp_routes._reject_private_target_for_managed_accounts(urlparse(public))
    finally:
        reset_workspace_subject(token)


def test_the_owner_check_does_not_depend_on_a_readable_auth_database(monkeypatch):
    from auth import storage as storage_mod

    def _explode(_username):
        raise RuntimeError("auth.db is unreadable")

    monkeypatch.setattr(storage_mod, "is_admin", _explode)
    # The seeded owner is decided without a lookup, so a single-user install keeps
    # working when auth.db cannot be read.
    assert storage_mod.is_installation_owner(LEGACY_WORKSPACE_SUBJECT) is True
    # Anything else fails closed: withhold the capability rather than grant it.
    assert storage_mod.is_installation_owner("alice") is False


def test_only_the_owner_may_configure_or_run_a_local_mcp_command():
    from fastapi import HTTPException

    from auth.authentication import require_ui_session_for_local_commands

    # stdio MCP runs an executable of the caller's choosing as the server's OS
    # user, which reaches every account's files. That is administration, not a
    # per-account setting, and path containment cannot help: the command is an
    # executable name, not a path into a workspace.
    require_ui_session_for_local_commands(False, LEGACY_WORKSPACE_SUBJECT)
    with pytest.raises(HTTPException) as exc:
        require_ui_session_for_local_commands(False, "alice")
    assert exc.value.status_code == 403
    # The older API-key rule still holds, for the owner too.
    with pytest.raises(HTTPException):
        require_ui_session_for_local_commands(True, LEGACY_WORKSPACE_SUBJECT)


def test_the_stdio_spawn_itself_refuses_a_managed_account(monkeypatch):
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)

    # Second line of defence, behind the route check: a row that predates the
    # route guard must not still become a process running as the server's user.
    token = _bind("alice")
    try:
        with pytest.raises(PermissionError):
            mcp_client._client("npx some-server", None)
    finally:
        reset_workspace_subject(token)


def test_a_managed_training_run_cannot_borrow_the_owners_hub_token():
    from core.training.training import _ambient_credentials_suppressed_for

    # The training child is spawned, so it copies the live parent environment.
    # An owner HF_TOKEN there, or a cached hub login, would let a managed account
    # train on a private repo it cannot read.
    assert _ambient_credentials_suppressed_for(LEGACY_WORKSPACE_SUBJECT) == {}
    suppressed = _ambient_credentials_suppressed_for("alice")
    assert suppressed["HF_TOKEN"] == ""
    assert suppressed["HUGGING_FACE_HUB_TOKEN"] == ""
    # Blanking the variables is not enough on its own: without this the hub still
    # reaches for the machine's cached login.
    assert suppressed["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_a_managed_training_run_cannot_borrow_the_owners_wandb_identity():
    from fastapi import HTTPException

    from core.training.training import _ambient_credentials_suppressed_for
    from routes.training import _reject_wandb_without_an_account_token

    # The worker only overwrites WANDB_API_KEY when the request carried a token,
    # so an inherited owner key uploaded the run under the owner's identity.
    assert _ambient_credentials_suppressed_for("alice")["WANDB_API_KEY"] == ""

    token = _bind("alice")
    try:
        with pytest.raises(HTTPException) as exc:
            _reject_wandb_without_an_account_token(True, None)
        assert exc.value.status_code == 403
        # Its own key is fine, and so is not using W&B at all.
        _reject_wandb_without_an_account_token(True, "alice-key")
        _reject_wandb_without_an_account_token(False, None)
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        # The owner's own key in the environment is the owner's to use.
        _reject_wandb_without_an_account_token(True, None)
    finally:
        reset_workspace_subject(token)


def test_a_retired_workspace_path_is_still_recognised_as_private(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))

    token = _bind("alice")
    try:
        private = workspace_root() / "models" / "secret-model"
        private.mkdir(parents = True, exist_ok = True)
        resident = {"loaded": True, "repo_id": str(private)}
    finally:
        reset_workspace_subject(token)

    # Deleting the account renames the workspace while an idle pipeline is still
    # resident. Classified by existence, the retired path read as a Hub id and the
    # departed account's weights were handed to whoever generated next.
    import shutil

    shutil.rmtree(private)
    assert not private.exists()

    token = _bind("bob")
    try:
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_foreign_private_resident_model(resident, "image")
        assert exc.value.status_code == 403
    finally:
        reset_workspace_subject(token)

    # Shape, not existence: a Hub id stays shared whether or not anything is there.
    assert inference_routes._looks_like_a_local_model_path("black-forest-labs/FLUX.1-dev") is False
    assert inference_routes._looks_like_a_local_model_path("/var/lib/x/model") is True
    assert inference_routes._looks_like_a_local_model_path("a/b/c") is True


def test_a_managed_recipe_cannot_read_provider_keys_from_the_server_environment():
    from fastapi import HTTPException

    from routes.data_recipe.jobs import _reject_env_credentials_from_a_managed_account

    # api_key_env is resolved with os.getenv in the spawned worker, and both the
    # variable name and the endpoint come from the request, so this hands the
    # secret to an endpoint of the caller's choosing rather than merely spending
    # it. No containment on the recipe's paths can address that.
    providers = [
        {"provider_type": "external", "api_key_env": "OPENAI_API_KEY"},
        {"provider_type": "external", "api_key": "sk-its-own"},
    ]
    token = _bind("alice")
    try:
        with pytest.raises(HTTPException) as exc:
            _reject_env_credentials_from_a_managed_account(providers)
        assert exc.value.status_code == 403
        # Supplying the key directly is fine.
        _reject_env_credentials_from_a_managed_account([providers[1]])
        _reject_env_credentials_from_a_managed_account(
            [{"provider_type": "external", "api_key_env": "  "}]
        )
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        # The owner's environment is the owner's, and naming a variable in it is
        # how the feature is meant to be used.
        _reject_env_credentials_from_a_managed_account(providers)
    finally:
        reset_workspace_subject(token)


def test_the_recipe_env_secret_guard_covers_every_provider_collection():
    from fastapi import HTTPException

    from routes.data_recipe.jobs import reject_env_credentials_in_recipe

    # The first version of this guard sat inside _inject_local_providers, which
    # returns early for a recipe with no local providers and never sees
    # mcp_providers, so the ordinary external-only recipe walked straight past it.
    external_only = {
        "model_providers": [
            {"provider_type": "external", "api_key_env": "OPENAI_API_KEY"},
        ]
    }
    mcp_only = {
        "mcp_providers": [{"api_key_env": "SOME_SECRET"}],
    }
    token = _bind("alice")
    try:
        for recipe in (external_only, mcp_only):
            with pytest.raises(HTTPException) as exc:
                reject_env_credentials_in_recipe(recipe)
            assert exc.value.status_code == 403
        # No provider collections at all, and a recipe that is not a dict, are
        # both reached by the validate route and must not raise.
        reject_env_credentials_in_recipe({"columns": []})
        reject_env_credentials_in_recipe(None)
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        reject_env_credentials_in_recipe(external_only)
        reject_env_credentials_in_recipe(mcp_only)
    finally:
        reset_workspace_subject(token)


def test_managed_recipe_workers_do_not_inherit_the_owners_github_token():
    from core.training.training import _ambient_credentials_suppressed_for

    # The GitHub seed plugin reads these deliberately when a recipe leaves its
    # token blank, so an inherited owner token reads the owner's private
    # repositories into the caller's dataset.
    suppressed = _ambient_credentials_suppressed_for("alice")
    assert suppressed["GH_TOKEN"] == ""
    assert suppressed["GITHUB_TOKEN"] == ""
    assert _ambient_credentials_suppressed_for(LEGACY_WORKSPACE_SUBJECT) == {}


def test_status_routes_redact_a_foreign_private_resident_model(tmp_path, monkeypatch):
    from routes import inference as inference_routes

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))

    token = _bind("alice")
    try:
        private = workspace_root() / "models" / "secret-model"
        private.mkdir(parents = True, exist_ok = True)
        resident = {
            "loaded": True,
            "repo_id": str(private),
            "base_repo": str(private),
            "resolved": str(private),
            "family": "flux",
        }
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Refusing the generation was not enough: the status payload still named
        # the model and spelled out its absolute workspace path.
        redacted = inference_routes._redact_foreign_private_resident_model(resident)
        assert redacted["loaded"] is False
        for field in ("repo_id", "base_repo", "resolved"):
            assert redacted[field] is None
        # Reported idle rather than refused, so the shape stays valid.
        assert redacted["family"] == "flux"
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        assert inference_routes._redact_foreign_private_resident_model(resident) == resident
    finally:
        reset_workspace_subject(token)

    # A shared Hub model is untouched for everyone.
    shared = {"loaded": True, "repo_id": "black-forest-labs/FLUX.1-dev"}
    token = _bind("bob")
    try:
        assert inference_routes._redact_foreign_private_resident_model(shared) == shared
    finally:
        reset_workspace_subject(token)


def test_image_generation_is_pinned_to_the_status_it_authorised(monkeypatch):
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import load_identity

    seen = {}
    authorised = {
        "loaded": True,
        "repo_id": "shared/model",
        "base_repo": "shared/base",
        "family": "flux",
    }

    def _status(self):
        return dict(authorised)

    def _generate(self, **kwargs):
        seen["expected_load"] = kwargs.get("expected_load")
        raise RuntimeError("stop here")

    monkeypatch.setattr(DiffusionBackend, "status", _status)
    monkeypatch.setattr(DiffusionBackend, "generate", _generate)

    backend = DiffusionBackend.__new__(DiffusionBackend)
    # What the route does: authorise a status, then carry that exact identity into
    # generate so the slot can refuse a load that committed in between rather than
    # letting this request, which named no model, adopt it.
    status = backend.status()
    pinned = load_identity(status.get("repo_id"), status.get("base_repo"), status.get("family"))
    try:
        backend.generate(expected_load = pinned)
    except RuntimeError:
        pass

    assert seen["expected_load"] == load_identity("shared/model", "shared/base", "flux")
    # A different resident model does not match what was authorised, which is what
    # the slot compares under its own lock.
    assert seen["expected_load"] != load_identity(
        "/workspaces/alice-abc/models/secret", "shared/base", "flux"
    )


def test_the_openai_image_route_rechecks_on_every_attempt():
    import inspect

    from routes import inference as inference_routes

    source = inspect.getsource(inference_routes._generate_openai_images)
    check = "_reject_foreign_private_resident_model(status,"
    body_after_loop = source.split("for attempt in range(2):", 1)[1]
    # Inside the retry loop, not once before it: the retry pins itself to a status
    # read after the replacement, so a private load landing in between would be
    # adopted if the check stayed outside.
    assert check in body_after_loop


def test_deleting_an_account_closes_its_cached_mcp_sessions(monkeypatch):
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_mcp_sessions", {}, raising = False)
    for subject in ("alice", "bob"):
        token = _bind(subject)
        try:
            mcp_client._mcp_sessions[mcp_client._session_key("https://x/mcp", None, "s1")] = (
                object()
            )
        finally:
            reset_workspace_subject(token)

    # The key holds the username, which is reusable, so a session left behind is
    # one a namesake could check out inside the idle TTL and inherit.
    assert mcp_client.workspace_has_cached_sessions("alice") is True
    assert mcp_client.workspace_has_cached_sessions("bob") is True
    assert mcp_client.workspace_has_cached_sessions("carol") is False


def test_a_managed_account_cannot_reach_a_private_mcp_address_at_connect_time():
    from core.inference import mcp_client

    # The route check happens when the row is written, which a hostname the
    # account controls can outlive by being rebound afterwards.
    token = _bind("alice")
    try:
        with pytest.raises(PermissionError):
            mcp_client._revalidate_http_destination("http://127.0.0.1:8080/mcp")
        # A public literal still connects.
        mcp_client._revalidate_http_destination("https://8.8.8.8/mcp")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        mcp_client._revalidate_http_destination("http://127.0.0.1:8080/mcp")
    finally:
        reset_workspace_subject(token)


def test_one_accounts_dataset_mutation_does_not_block_another_from_training():
    import threading as _threading

    from core.training.diffusion_training_service import (
        DatasetMutationInFlight,
        DiffusionTrainingService,
    )

    service = DiffusionTrainingService.__new__(DiffusionTrainingService)
    service._lock = _threading.RLock()
    service._reserved = False
    service._proc = None
    service._gpu_admissions = 0
    service._active_workspace_subject = None
    service._dataset_mutations = {}

    token = _bind("bob")
    try:
        with service.dataset_mutation():
            # Counted together, Bob's long import refused Alice's unrelated start
            # for its whole duration.
            token_alice = _bind("alice")
            try:
                assert service._dataset_mutations == {"bob": 1}
                assert service._dataset_mutations.get(current_workspace_subject()) is None
            finally:
                reset_workspace_subject(token_alice)

            # Bob's own start is still refused, which is the interlock's point.
            with pytest.raises(DatasetMutationInFlight):
                service.reserve()
    finally:
        reset_workspace_subject(token)

    # Popped, not left at zero, so the map cannot grow per account seen.
    assert service._dataset_mutations == {}


def test_only_the_account_that_started_a_download_may_cancel_it(monkeypatch):
    from fastapi import HTTPException

    from hub.services import download_lifecycle

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)

    token = _bind("alice")
    try:
        download_lifecycle.note_download_initiator("alice-job")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The registry is keyed by repository alone, so naming one was enough to
        # abort somebody else's large or gated pull.
        with pytest.raises(HTTPException) as exc:
            download_lifecycle.require_download_cancel_permission("alice-job")
        assert exc.value.status_code == 403
        # A job with no recorded initiator, from before a restart, stays
        # cancellable rather than stranded.
        download_lifecycle.require_download_cancel_permission("unknown-job")
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        download_lifecycle.require_download_cancel_permission("alice-job")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        download_lifecycle.require_download_cancel_permission("alice-job")
    finally:
        reset_workspace_subject(token)


def test_a_managed_export_worker_cannot_borrow_the_owners_hub_token(monkeypatch):
    from core.export import orchestrator as export_orchestrator

    captured: dict = {}

    class _FakeQueue:
        def put(self, *_a, **_k):
            pass

    class _FakeProcess:
        pid = 4242

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

        def is_alive(self):
            return False

    class _FakeContext:
        Queue = _FakeQueue
        Process = _FakeProcess

    monkeypatch.setattr(export_orchestrator, "_CTX", _FakeContext)
    monkeypatch.setattr("utils.process_lifetime.adopt_pid", lambda *_a, **_k: None, raising = False)

    backend = export_orchestrator.ExportOrchestrator.__new__(export_orchestrator.ExportOrchestrator)
    backend._export_active = False
    backend._workspace_subject = None
    backend._proc = None

    # A remote checkpoint passes the containment guard because it is not a local
    # path, so the only thing standing between a managed account and an
    # owner-private repo is what the child is allowed to authenticate as.
    backend._spawn_subprocess({"subject": "alice", "checkpoint_path": "owner/private"})
    child_env = captured["args"][2]
    assert child_env["HF_TOKEN"] == ""
    assert child_env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"

    captured.clear()
    backend._spawn_subprocess(
        {"subject": LEGACY_WORKSPACE_SUBJECT, "checkpoint_path": "owner/private"}
    )
    owner_env = captured["args"][2]
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in owner_env
    assert owner_env.get("HF_TOKEN", None) != ""


def test_a_managed_embedding_choice_resolves_without_the_owners_hub_token(monkeypatch):
    from routes import settings as settings_routes

    monkeypatch.setattr(settings_routes, "_ambient_hf_token", lambda: "owner-token", raising = False)

    # A token the caller supplied is theirs, whoever they are.
    assert settings_routes._hub_token_for_subject("mine") == "mine"

    token = _bind("alice")
    try:
        # Not None: huggingface_hub reads None as "find an implicit login", which is
        # exactly the owner's cached credential this must not spend on an
        # owner-private repo the caller merely named.
        assert settings_routes._hub_token_for_subject(None) is False
        assert settings_routes._hub_token_for_subject("   ") is False
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert settings_routes._hub_token_for_subject(None) == "owner-token"
    finally:
        reset_workspace_subject(token)


def test_the_ambient_hub_token_is_owner_only():
    from routes import settings as settings_routes
    token = _bind("alice")
    try:
        assert settings_routes._ambient_hf_token() is None
    finally:
        reset_workspace_subject(token)


def test_a_resident_controlnet_is_not_reused_across_workspaces():
    from core.inference.diffusion import DiffusionBackend

    class _Resolved:
        def __init__(self, id, path, is_local):
            self.id = id
            self.path = path
            self.is_local = is_local

    # controlnets_dir() is per account, so the same catalog id names different
    # weights for different accounts; keying the resident model by id alone handed
    # the second caller the first one's private ControlNet.
    alice = DiffusionBackend._controlnet_cache_key(
        _Resolved("my-cn", "/home/u/workspaces/alice/controlnets/diffusion/my-cn", True)
    )
    bob = DiffusionBackend._controlnet_cache_key(
        _Resolved("my-cn", "/home/u/workspaces/bob/controlnets/diffusion/my-cn", True)
    )
    assert alice != bob

    # A hub model is install-wide by design and must still be shared.
    hub_one = DiffusionBackend._controlnet_cache_key(
        _Resolved("flux-union-pro", "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", False)
    )
    hub_two = DiffusionBackend._controlnet_cache_key(
        _Resolved("flux-union-pro", "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", False)
    )
    assert hub_one == hub_two


def test_a_forced_model_swap_cannot_stop_another_accounts_chats(monkeypatch):
    import threading as _threading

    from fastapi import HTTPException

    from routes import inference as inference_routes
    from state import active_generations

    active_generations.reset_for_tests()

    def _register(subject, handle):
        with active_generations._LOCK:
            active_generations._ACTIVE[handle] = {
                "handle": handle,
                "thread_id": f"thread-{handle}",
                "run_id": f"run-{handle}",
                "model": "m",
                "kind": "chat",
                "started_at": 0.0,
                "subject": subject,
                "event": _threading.Event(),
            }

    try:
        _register("alice", "a1")
        _register("bob", "b1")

        token = _bind("bob")
        try:
            # Bob may not see Alice's chat, but force_cancel_active used to cancel
            # it anyway: the subject filter only hid the ids from the refusal.
            with pytest.raises(HTTPException) as exc:
                inference_routes._raise_or_cancel_active_generations(
                    force = True, action = "Unloading the model", caller_scoped = True
                )
            assert exc.value.status_code == 409
            assert exc.value.detail["error"] == "foreign_active_generations"
            assert exc.value.detail["running"] == 1
            assert exc.value.detail["thread_ids"] == []
        finally:
            reset_workspace_subject(token)

        # Alice's stream is untouched; Bob's own is still his to stop.
        assert active_generations._ACTIVE["a1"]["event"].is_set() is False
        assert active_generations.count("bob") == 1
        assert active_generations.cancel_all("bob") == 1
        assert active_generations._ACTIVE["a1"]["event"].is_set() is False

        # The sidecar swap stays install-wide: it replaces the runtime under every
        # stream, so leaving another account's running would be worse.
        assert active_generations.cancel_all() == 2
        assert active_generations._ACTIVE["a1"]["event"].is_set() is True
    finally:
        active_generations.reset_for_tests()


def test_a_lone_account_can_still_force_its_own_swap():
    import threading as _threading

    from routes import inference as inference_routes
    from state import active_generations

    active_generations.reset_for_tests()
    try:
        with active_generations._LOCK:
            active_generations._ACTIVE["only"] = {
                "handle": "only",
                "thread_id": "t",
                "run_id": "r",
                "model": "m",
                "kind": "chat",
                "started_at": 0.0,
                "subject": LEGACY_WORKSPACE_SUBJECT,
                "event": _threading.Event(),
            }

        token = _bind(LEGACY_WORKSPACE_SUBJECT)
        try:
            # The single-account install is the same install it was: one subject
            # owns everything, so nothing is foreign and the force still works.
            assert (
                inference_routes._raise_or_cancel_active_generations(
                    force = True, action = "Unloading the model", caller_scoped = True
                )
                == 1
            )
        finally:
            reset_workspace_subject(token)
        assert active_generations._ACTIVE["only"]["event"].is_set() is True
    finally:
        active_generations.reset_for_tests()


def test_only_the_owner_may_revoke_every_accounts_preview_links():
    import inspect

    from auth.authentication import require_install_admin
    from routes.settings import rotate_preview_links

    # One installation-wide signing secret, so rotating it revokes the owner's
    # links and every other account's, not just the caller's.
    dependency = inspect.signature(rotate_preview_links).parameters["current_subject"].default
    assert dependency.dependency is require_install_admin


def test_training_refuses_rather_than_cancelling_a_foreign_render(monkeypatch):
    from core.inference import diffusion_engine_router, video as video_module
    from routes.training import _foreign_media_render_active
    from utils.workspace_context import ForeignWorkspaceActiveError

    class _Busy:
        def _refuse_foreign_teardown(self, subject):
            raise ForeignWorkspaceActiveError("Another account is generating an image right now.")

    class _Idle:
        def _refuse_foreign_teardown(self, subject):
            return None

    class _Broken:
        def _refuse_foreign_teardown(self, subject):
            raise RuntimeError("probe unavailable")

    def _engines(diffusion, video):
        monkeypatch.setattr(
            diffusion_engine_router, "get_active_diffusion_engine", lambda: diffusion
        )
        monkeypatch.setattr(video_module, "get_video_backend", lambda: video)

    # _free_vram_for_training unloads both engines with no subject, which is their
    # own path and tears down whatever is running, so the refusal belongs here.
    _engines(_Busy(), _Idle())
    assert (
        _foreign_media_render_active("bob") == "Another account is generating an image right now."
    )

    _engines(_Idle(), _Idle())
    assert _foreign_media_render_active("bob") is None

    # A probe that cannot answer must not become a way to block training.
    _engines(_Broken(), _Idle())
    assert _foreign_media_render_active("bob") is None


def test_the_video_backend_can_be_asked_before_it_is_torn_down():
    import inspect

    from core.inference.video import VideoBackend

    # unload() must ask through the same helper the training guard asks, so the
    # two can never disagree about what counts as a foreign render.
    assert hasattr(VideoBackend, "_refuse_foreign_teardown")
    assert "_refuse_foreign_teardown(subject)" in inspect.getsource(VideoBackend.unload)


def test_the_github_env_token_status_matches_what_the_worker_inherits(monkeypatch):
    from routes.data_recipe.seed import get_github_env_token_status

    monkeypatch.setenv("GH_TOKEN", "owner-github-token")

    # The dialog uses this to tell the user the token field can be left blank.
    # A managed account's recipe worker is spawned with both variables blanked,
    # so a true answer there sends them into a Check that fails with no token.
    assert get_github_env_token_status(current_subject = "alice") == {"has_token": False}
    assert get_github_env_token_status(current_subject = LEGACY_WORKSPACE_SUBJECT) == {
        "has_token": True
    }

    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    assert get_github_env_token_status(current_subject = LEGACY_WORKSPACE_SUBJECT) == {
        "has_token": False
    }


def test_a_deleted_username_is_held_while_a_media_load_is_in_flight(monkeypatch):
    from auth import storage as auth_storage

    class _Backend:
        def __init__(
            self,
            loading_subject = None,
            generating = False,
        ):
            self.loading_subject = loading_subject
            self.generating = generating
            self.unloaded_for = []
            self.cancelled_for = []

        def generate_progress(self, subject = None):
            return {"active": self.generating and subject == "alice"}

        def load_progress(self, subject = None):
            if self.loading_subject is None or self.loading_subject != subject:
                return {"phase": "ready"}
            return {"phase": "downloading"}

        def cancel_generate(self, subject = None):
            self.cancelled_for.append(subject)

        def unload(self, subject = None):
            self.unloaded_for.append(subject)
            self.loading_subject = None

    # The probe looked only at generate_progress, so a load in flight read as idle
    # and the tombstone could clear while it was still running. The loading state
    # carries the subject, so a namesake matches it.
    loading = _Backend(loading_subject = "alice")
    assert auth_storage._media_load_active(loading, "alice") is True
    assert auth_storage._media_load_active(loading, "bob") is False
    assert auth_storage._media_load_active(_Backend(), "alice") is False

    monkeypatch.setattr(auth_storage, "_loaded_media_backends", lambda: [loading])
    assert auth_storage._workspace_jobs_active("alice") is True
    assert auth_storage._workspace_jobs_active("bob") is False

    # And the quiesce path has to reach it: cancel_generate does not touch a load.
    auth_storage._quiesce_workspace_jobs("alice")
    assert loading.unloaded_for == ["alice"]
    assert auth_storage._workspace_jobs_active("alice") is False


def test_a_managed_account_may_load_what_the_catalog_offered_it(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes
    from routes import models as models_routes

    shared_cache = tmp_path / "shared" / "hf"
    lm_studio = tmp_path / "shared" / "lmstudio"
    elsewhere = tmp_path / "someone-elses" / "private"
    for path in (shared_cache, lm_studio, elsewhere):
        path.mkdir(parents = True)

    monkeypatch.setattr(
        models_routes,
        "advertised_shared_model_roots",
        lambda: [str(shared_cache.resolve()), str(lm_studio.resolve())],
    )

    token = _bind("alice")
    try:
        # collect_local_models scans these for every account, so refusing the load
        # left the picker offering models that 403ed, and OpenAI auto-switch
        # resolving one and then failing at the same gate.
        inference_routes._reject_uncontained_local_path(
            str(shared_cache / "models--org--thing"), "load"
        )
        inference_routes._reject_uncontained_local_path(str(lm_studio / "org" / "thing"), "load")
        # Everything else is still refused: the catalog never offered it.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_uncontained_local_path(str(elsewhere), "load")
        assert exc.value.status_code == 403
    finally:
        reset_workspace_subject(token)


def test_a_managed_account_can_export_to_a_path_its_own_browser_returned(tmp_path, monkeypatch):
    from utils.paths.storage_roots import exports_root, resolve_export_write_dir

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))

    token = _bind("alice")
    try:
        root = exports_root()
        root.mkdir(parents = True, exist_ok = True)
        chosen = root / "my-export"
        chosen.mkdir()
        # The folder browser returns an absolute path even for a directory inside
        # the caller's own workspace, so refusing on shape alone refused the
        # ordinary case.
        assert resolve_export_write_dir(str(chosen)) == chosen.resolve()
        # An absolute path outside it is still refused.
        outside = tmp_path / "outside"
        outside.mkdir()
        with pytest.raises(ValueError):
            resolve_export_write_dir(str(outside))
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        # The owner keeps the arbitrary-path escape hatch (#6082).
        outside = tmp_path / "owner-drive"
        assert resolve_export_write_dir(str(outside)) == outside
    finally:
        reset_workspace_subject(token)


def test_the_startup_precache_setting_lives_in_the_owners_database(monkeypatch):
    import inspect

    from auth.authentication import require_install_admin
    from routes.settings import update_helper_precache
    from utils import helper_precache_settings

    seen = {}

    def _fake_upsert(values):
        seen["write_subject"] = current_workspace_subject()
        seen["values"] = values

    monkeypatch.setattr("storage.studio_db.upsert_app_settings", _fake_upsert)
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, default: seen.setdefault("read_subject", current_workspace_subject()) and None,
    )

    token = _bind("alice")
    try:
        # _start_helper_precache_if_enabled runs outside any request, so it reads
        # the owner's database whatever anyone else stored.
        helper_precache_settings.set_helper_precache_enabled(True)
        helper_precache_settings.get_helper_precache_enabled()
    finally:
        reset_workspace_subject(token)

    assert seen["write_subject"] == LEGACY_WORKSPACE_SUBJECT
    assert seen["read_subject"] == LEGACY_WORKSPACE_SUBJECT

    # And the write is owner-only, since it is one install-wide behaviour.
    dependency = inspect.signature(update_helper_precache).parameters["current_subject"].default
    assert dependency.dependency is require_install_admin


def test_a_managed_training_run_cannot_use_the_hosts_iam_role():
    from fastapi import HTTPException

    from models.training import S3Config
    from routes.training import _reject_ambient_s3_for_a_managed_account

    iam = S3Config(bucket = "someone-elses-bucket", use_iam_role = True)
    own_keys = S3Config(
        bucket = "my-bucket", access_key_id = "AKIA-mine", secret_access_key = "secret-mine"
    )

    token = _bind("alice")
    try:
        # _build_s3_client falls back to boto3's default chain, which on an EC2 or
        # container host is the installation's own role, so any bucket that
        # identity can read would be readable by naming it here.
        with pytest.raises(HTTPException) as exc:
            _reject_ambient_s3_for_a_managed_account(iam)
        assert exc.value.status_code == 403
        # Its own keys are fine, and so is not using S3 at all.
        _reject_ambient_s3_for_a_managed_account(own_keys)
        _reject_ambient_s3_for_a_managed_account(None)
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        # The host's identity is the owner's, so their run is unchanged.
        _reject_ambient_s3_for_a_managed_account(iam)
    finally:
        reset_workspace_subject(token)


def test_a_managed_worker_cannot_use_the_hosts_aws_identity():
    from core.training.training import _ambient_credentials_suppressed_for

    suppressed = _ambient_credentials_suppressed_for("alice")
    assert suppressed["AWS_ACCESS_KEY_ID"] == ""
    assert suppressed["AWS_SESSION_TOKEN"] == ""
    # The container-credential endpoints are a separate provider from the plain
    # variables, and so is IMDS, which is what hands out an EC2 instance role.
    assert suppressed["AWS_CONTAINER_CREDENTIALS_FULL_URI"] == ""
    assert suppressed["AWS_EC2_METADATA_DISABLED"] == "true"
    assert _ambient_credentials_suppressed_for(LEGACY_WORKSPACE_SUBJECT) == {}


def test_a_deleted_accounts_finished_diffusion_run_does_not_outlive_it():
    import threading as _threading

    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService.__new__(DiffusionTrainingService)
    service._lock = _threading.RLock()
    service._reserved = False
    service._proc = None
    service._active_workspace_subject = "alice"
    service._state = {"active": False, "status": "completed", "job_id": "alice-job"}

    # A terminal run is not active, so the delete path's stop had nothing to stop
    # and the singleton kept the subject beside the finished run.
    service.reset_retained_state("alice")
    assert service._active_workspace_subject is None
    assert service._state["job_id"] is None
    assert service._state["status"] == "idle"

    # Another account's state is not this account's to drop.
    service._active_workspace_subject = "bob"
    service._state = {"active": False, "status": "completed", "job_id": "bob-job"}
    service.reset_retained_state("alice")
    assert service._state["job_id"] == "bob-job"

    # Neither is a live run, whoever owns it: the delete path stops it first.
    service._active_workspace_subject = "alice"
    service._reserved = True
    service._state = {"active": True, "status": "running", "job_id": "alice-live"}
    service.reset_retained_state("alice")
    assert service._state["job_id"] == "alice-live"


def test_download_activity_is_not_an_install_wide_listing(monkeypatch):
    from hub.services import download_lifecycle

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)

    token = _bind("alice")
    try:
        download_lifecycle.note_download_initiator("org/private-model")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The registry is install-wide, so an unfiltered activity list handed Bob
        # the repository id, variant and scoped filenames of Alice's pull.
        assert download_lifecycle.download_is_visible_to_caller("org/private-model") is False
        # A job with no recorded initiator is hidden from a managed account, the
        # opposite of the cancel rule: publishing one cannot be undone.
        assert download_lifecycle.download_is_visible_to_caller("unknown-job") is False
        download_lifecycle.require_download_cancel_permission("unknown-job")

        # A shared slot: Bob adopting the same repo joins the initiators rather
        # than replacing Alice, so both keep their own view and their own cancel.
        download_lifecycle.note_download_initiator("org/private-model")
        assert download_lifecycle.download_is_visible_to_caller("org/private-model") is True
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        assert download_lifecycle.download_is_visible_to_caller("org/private-model") is True
        download_lifecycle.require_download_cancel_permission("org/private-model")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert download_lifecycle.download_is_visible_to_caller("unknown-job") is True
    finally:
        reset_workspace_subject(token)


def test_a_retired_workspace_does_not_keep_its_database_open(tmp_path, monkeypatch):
    from storage import studio_db

    managed = tmp_path / "workspaces" / "alice-0123456789ab" / "studio.db"
    managed.parent.mkdir(parents = True)
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: managed)
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    try:
        assert studio_db.open_wal_keeper() is True
        key = str(managed.resolve())
        assert key in studio_db._wal_keepers

        # Windows refuses to rename a directory holding an open file, so a keeper
        # left on a deleted account's database blocks the retirement for the life
        # of the process and the username stays tombstoned.
        studio_db.close_wal_keeper_for(managed)
        assert key not in studio_db._wal_keepers
        # Idempotent: the retire path runs before the rename, and may run again.
        studio_db.close_wal_keeper_for(managed)
    finally:
        studio_db.close_wal_keeper()


def test_a_managed_load_cannot_reach_a_private_repo_on_the_servers_login(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_ANONYMOUS_HUB_ACCESS", {}, raising = False)

    answers = {"org/public-model": True, "org/owner-private": False, "org/unknown": None}
    asked = []

    def _fake_probe(repo_id, repo_type):
        asked.append((repo_id, repo_type))
        return answers[repo_id]

    monkeypatch.setattr(inference_routes, "_hub_repo_is_anonymously_readable", _fake_probe)
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_shared_cache", lambda repo, kind: False)
    guard = inference_routes._reject_private_hub_repo_without_an_account_token

    token = _bind("alice")
    try:
        # A public repo needs no credential, so it loads exactly as it did: this
        # is not "managed accounts must paste a token to use the Hub".
        guard("org/public-model", None)
        # A repo the caller could not otherwise reach is the whole finding: the
        # load path hands the id down with no token and the Hub client falls back
        # to the owner's implicit login.
        with pytest.raises(HTTPException) as exc:
            guard("org/owner-private", None)
        assert exc.value.status_code == 403
        # Their own token answers it, whatever the repo is.
        guard("org/owner-private", "hf_alices_own_token")
        # A local path is containment's question, not this one.
        guard("/some/absolute/path", None)
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        # The credential is the owner's own, so their loads are untouched and the
        # probe is never even paid.
        before = len(asked)
        guard("org/owner-private", None)
        assert len(asked) == before
    finally:
        reset_workspace_subject(token)


def test_an_unanswerable_hub_probe_falls_back_to_the_shared_cache(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setattr(
        inference_routes, "_hub_repo_is_anonymously_readable", lambda repo, kind: None
    )
    guard = inference_routes._reject_private_hub_repo_without_an_account_token

    from hub.services import download_lifecycle

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)

    token = _bind("alice")
    try:
        monkeypatch.setattr(
            inference_routes, "_repo_is_in_the_shared_cache", lambda repo, kind: True
        )
        # Offline, or the Hub unreachable. Cache presence on its own is what put
        # another account's private weights within reach, so it is not the
        # answer; this account having asked for the repository is.
        with pytest.raises(HTTPException):
            guard("org/already-here", None)
        download_lifecycle.note_download_initiator("org/already-here", replaces_previous_job = True)
        guard("org/already-here", None)
        # And a repo nobody fetched here is refused rather than guessed at.
        monkeypatch.setattr(
            inference_routes, "_repo_is_in_the_shared_cache", lambda repo, kind: False
        )
        with pytest.raises(HTTPException):
            guard("org/not-here", None)
    finally:
        reset_workspace_subject(token)


def test_the_video_load_asks_the_same_credential_question():
    import inspect

    from routes import video as video_routes

    source = inspect.getsource(video_routes)
    # The video backend builds its own HfApi and calls from_pretrained with the
    # request's token, so the guard has to run on the route, beside containment.
    assert "_reject_private_hub_repo_without_an_account_token(request.model_path" in source
    # And on the base repo, which is fetched the same way under a separate id.
    assert "_reject_private_hub_repo_without_an_account_token(request.base_repo" in source


def test_embedding_resolution_is_contained_like_the_save(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes.settings import resolve_embedding_model

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    elsewhere = tmp_path / "someone-elses" / "workspace"
    elsewhere.mkdir(parents = True)

    token = _bind("alice")
    try:
        # The GET probes for checkpoints under whatever it is given, so an
        # absolute path was a recursive read of another account's workspace whose
        # answer distinguished a real model from anything else there.
        with pytest.raises(HTTPException) as exc:
            resolve_embedding_model(model = str(elsewhere), hf_token = None, current_subject = "alice")
        assert exc.value.status_code == 403
    finally:
        reset_workspace_subject(token)


def test_a_video_that_finishes_between_polls_keeps_its_account():
    from routes import video as video_routes

    mine = video_routes._VideoJob(
        id = "vid-1",
        created_at = 0,
        prompt = "p",
        model = "m",
        size = "s",
        seconds = "4",
        status = "in_progress",
        subject = "alice",
    )
    # A record written before this field, or by a path that does not round-trip
    # it, comes back with "", which _job_is_mine reads as the owner's own legacy
    # state; the managed caller could then still delete the clip through its
    # workspace-scoped record while the cleanup guard called it foreign.
    replacement = video_routes._VideoJob(
        id = "vid-1",
        created_at = 0,
        prompt = "p",
        model = "m",
        size = "s",
        seconds = "4",
        status = "completed",
    )
    replacement.subject = replacement.subject or mine.subject
    assert replacement.subject == "alice"

    token = _bind("alice")
    try:
        assert video_routes._job_is_mine(replacement) is True
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert video_routes._job_is_mine(replacement) is False
    finally:
        reset_workspace_subject(token)


def test_only_the_account_that_started_a_dictation_download_may_cancel_it(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_STT_DOWNLOAD_INITIATORS", {}, raising = False)

    token = _bind("alice")
    try:
        inference_routes._note_stt_download_initiator("transformers")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The cancel payload is optional and defaults to the shared Transformers
        # downloader, so a caller needed no job identifier at all.
        with pytest.raises(HTTPException) as exc:
            inference_routes._require_stt_download_cancel_permission("transformers")
        assert exc.value.status_code == 403
        # A different engine, and an unrecorded download, stay cancellable.
        inference_routes._require_stt_download_cancel_permission("gguf")
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        inference_routes._require_stt_download_cancel_permission("transformers")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        inference_routes._require_stt_download_cancel_permission("transformers")
    finally:
        reset_workspace_subject(token)


def test_a_finished_recipe_job_survives_another_accounts_start():
    import threading as _threading
    from collections import deque

    from core.data_recipe.jobs.manager import Job, JobManager

    manager = JobManager.__new__(JobManager)
    manager._lock = _threading.RLock()
    manager._job = Job(job_id = "alice-job", status = "completed", started_at = 0.0)
    manager._job.analysis = {"rows": 10}
    manager._proc = None
    manager._events = deque()
    manager._subs = []
    manager._seq = 0
    manager._workspace_subject = "alice"
    manager._finished_jobs = {}

    # Bob starting a run replaces the singleton's only _job.
    manager._retain_finished_job_locked()
    manager._workspace_subject = "bob"
    manager._job = Job(job_id = "bob-job", status = "pending", started_at = 1.0)

    token = _bind("alice")
    try:
        # Alice's artifact still exists and she never started a replacement, so
        # her status, analysis and dataset page must still resolve. Without this
        # any account could make somebody's finished run unpublishable by
        # starting their own.
        assert manager.get_current_job_id() == "alice-job"
        assert manager.get_analysis("alice-job") == {"rows": 10}
        # And she still cannot read Bob's live job.
        assert manager.get_analysis("bob-job") is None
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        assert manager.get_current_job_id() == "bob-job"
        assert manager.get_analysis("alice-job") is None
    finally:
        reset_workspace_subject(token)


def test_the_delete_path_also_clears_state_that_outlived_the_work(monkeypatch):
    import inspect

    from auth import storage as auth_storage

    source = inspect.getsource(auth_storage._quiesce_workspace_jobs)
    # Everything above the divider stops something that is running. These clear
    # what a finished job leaves behind, which nothing above reaches because by
    # then there is nothing left to stop.
    for step in (
        "retained diffusion training state",
        "retained training state",
        "idle export worker",
        "retained recipe job",
        "completed video record",
        "API monitor entries",
        "private resident media models",
    ):
        assert step in source, step


def test_a_finished_training_run_does_not_outlive_its_account():
    import threading as _threading

    from core.training.training import TrainingBackend

    backend = TrainingBackend.__new__(TrainingBackend)
    backend._lock = _threading.RLock()
    backend._proc = None
    backend._spawn_in_progress = False
    backend._active_workspace_subject = "alice"
    backend._current_start_key = ("alice", "req-1")
    reset_calls = []
    backend.reset_training_state = lambda *a, **k: reset_calls.append(a) or "ok"

    # Another account's state is not this account's to drop.
    backend.reset_retained_state("bob")
    assert backend._active_workspace_subject == "alice"
    assert reset_calls == []

    backend.reset_retained_state("alice")
    assert backend._active_workspace_subject is None
    assert backend._current_start_key is None
    assert len(reset_calls) == 1

    # A live run is refused, whoever asks: the delete path stops it first.
    backend._active_workspace_subject = "alice"
    backend._spawn_in_progress = True
    backend.reset_retained_state("alice")
    assert backend._active_workspace_subject == "alice"


def test_a_finished_recipe_job_does_not_outlive_its_account():
    import threading as _threading
    from collections import deque

    from core.data_recipe.jobs.manager import Job, JobManager

    manager = JobManager.__new__(JobManager)
    manager._lock = _threading.RLock()
    manager._proc = None
    manager._job = Job(job_id = "alice-job", status = "completed", started_at = 0.0)
    manager._events = deque([{"seq": 1}])
    manager._subs = [object()]
    manager._seq = 3
    manager._workspace_subject = "alice"
    manager._finished_jobs = {"alice": Job(job_id = "older", status = "completed", started_at = 0.0)}

    # cancel() on a terminal job succeeds without clearing any of this, so
    # /jobs/current handed a namesake the old id and its rows came back with it.
    manager.reset_retained_state("alice")
    assert manager._job is None
    assert manager._workspace_subject is None
    assert manager._finished_jobs == {}
    assert list(manager._events) == []
    assert manager._subs == []


def test_a_completed_video_record_does_not_outlive_its_account():
    import threading as _threading

    from core.inference.video import VideoBackend

    backend = VideoBackend.__new__(VideoBackend)
    backend._lock = _threading.RLock()
    backend._generate_job_active = False
    backend._gen_subject = "alice"
    backend._gen = {"phase": "completed", "video": {"id": "vid-1", "prompt": "secret"}}

    # The record carries the whole recipe, and generate_progress reports it as
    # inactive, so nothing that stops a render reaches it.
    assert backend.forget_terminal_video(subject = "bob") is False
    assert backend._gen["phase"] == "completed"
    assert backend.forget_terminal_video(subject = "alice") is True
    assert backend._gen == {"active": False}


def test_only_media_loaded_from_the_deleted_workspace_is_unloaded(tmp_path):
    from auth import storage as auth_storage

    private = tmp_path / "workspaces" / "alice-0123456789ab"
    private.mkdir(parents = True)
    root = str(private.resolve())

    # A local path names one account's private weights.
    assert (
        auth_storage._status_names_a_path_under({"repo_id": str(private / "models" / "mine")}, root)
        is True
    )
    # A hub repo id is install-wide by design and must not be torn down.
    assert (
        auth_storage._status_names_a_path_under({"repo_id": "Shakker-Labs/FLUX.1"}, root) is False
    )
    # Neither is another account's path, nor an empty status.
    assert (
        auth_storage._status_names_a_path_under(
            {"repo_id": str(tmp_path / "workspaces" / "bob-x" / "m")}, root
        )
        is False
    )
    assert auth_storage._status_names_a_path_under({"repo_id": None}, root) is False


def test_a_research_run_holds_the_username_until_it_is_cancelled(tmp_path, monkeypatch):
    from auth import storage as auth_storage
    from storage import research_runs_db

    unfinished = {"alice": ["run-1", "run-2"], "bob": []}
    cancelled = []

    monkeypatch.setattr(
        research_runs_db,
        "unfinished_run_ids",
        lambda: unfinished.get(current_workspace_subject(), []),
    )
    monkeypatch.setattr(
        research_runs_db, "request_cancel", lambda run_id: cancelled.append(run_id) or "cancelling"
    )
    # Isolate the probe from every other subsystem this function asks about.
    monkeypatch.setattr(auth_storage, "_loaded_media_backends", lambda: [])

    # A supervisor between model calls holds no lease this process can see, but
    # the run row is still non-terminal and the run reopens this account's
    # databases under its own pathnames.
    assert auth_storage._workspace_jobs_active("alice") is True

    auth_storage._quiesce_workspace_jobs("alice")
    assert cancelled == ["run-1", "run-2"]

    unfinished["alice"] = []
    assert auth_storage._workspace_jobs_active("alice") is False


def test_only_the_deleted_accounts_folder_sync_worker_is_stopped(monkeypatch):
    import threading as _threading

    from core.rag import folder_sync

    class _Worker:
        def __init__(self):
            self.joined = False
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout = None):
            self.joined = True
            self._alive = False

    alice = (_Worker(), _threading.Event(), _threading.Event())
    bob = (_Worker(), _threading.Event(), _threading.Event())
    monkeypatch.setattr(
        folder_sync, "_workspace_workers", {"alice": alice, "bob": bob}, raising = False
    )

    assert folder_sync.workspace_sync_worker_active("alice") is True
    assert folder_sync.workspace_sync_worker_active("carol") is False

    # stop_auto_sync() stops every workspace's worker, which is a process
    # shutdown; deleting one account must not stop everybody else's sync.
    folder_sync.stop_workspace_auto_sync("alice")
    assert alice[1].is_set() and alice[2].is_set()
    assert alice[0].joined is True
    assert "alice" not in folder_sync._workspace_workers
    assert bob[1].is_set() is False
    assert bob[0].joined is False
    assert folder_sync.workspace_sync_worker_active("bob") is True

    # Unknown accounts are a no-op rather than an error.
    folder_sync.stop_workspace_auto_sync("carol")


def test_a_failed_start_hands_ownership_back_to_the_previous_account():
    import threading as _threading

    from core.training.training import TrainingBackend

    backend = TrainingBackend.__new__(TrainingBackend)
    backend._lock = _threading.RLock()
    backend._proc = None
    backend._active_workspace_subject = "alice"
    backend._current_start_key = ("alice", "req-alice")

    def _fail(*_a, **_k):
        # Everything between the claim and the spawn can refuse: the pump join,
        # the config build, the sidecar check, the GPU selection.
        backend._active_workspace_subject = "bob"
        backend._current_start_key = ("bob", "req-bob")
        return False

    backend._start_training_with_lifecycle_reserved_impl = _fail
    assert backend._start_training_with_lifecycle_reserved("job-1") is False
    # Alice keeps her completed status and metrics; Bob, who started nothing,
    # cannot read them.
    assert backend._active_workspace_subject == "alice"
    assert backend._current_start_key == ("alice", "req-alice")

    def _raise(*_a, **_k):
        backend._active_workspace_subject = "bob"
        raise RuntimeError("spawn refused")

    backend._start_training_with_lifecycle_reserved_impl = _raise
    with pytest.raises(RuntimeError):
        backend._start_training_with_lifecycle_reserved("job-2")
    assert backend._active_workspace_subject == "alice"

    class _Live:
        def is_alive(self):
            return True

    def _succeed(*_a, **_k):
        backend._active_workspace_subject = "bob"
        backend._proc = _Live()
        return True

    backend._start_training_with_lifecycle_reserved_impl = _succeed
    assert backend._start_training_with_lifecycle_reserved("job-3") is True
    # A live worker means the claim is the truth and must not be rolled back.
    assert backend._active_workspace_subject == "bob"


def test_a_refused_recipe_start_leaves_the_previous_job_intact(monkeypatch):
    import threading as _threading
    from collections import deque

    from fastapi import HTTPException

    from core.data_recipe.jobs import manager as manager_module
    from core.data_recipe.jobs.manager import Job, JobManager

    service = JobManager.__new__(JobManager)
    service._lock = _threading.RLock()
    service._proc = None
    service._job = Job(job_id = "alice-job", status = "completed", started_at = 0.0)
    service._events = deque([{"seq": 1}])
    service._subs = []
    service._seq = 1
    service._workspace_subject = "alice"
    service._finished_jobs = {}

    def _refuse(_recipe):
        raise HTTPException(status_code = 403, detail = "no")

    monkeypatch.setattr(manager_module, "_reject_uncontained_recipe_paths", _refuse)

    token = _bind("bob")
    try:
        # The refusal is preflight on the recipe alone, so it now runs before
        # anything is replaced: a forbidden path used to destroy Alice's finished
        # job and leave Bob with a replacement that never started.
        with pytest.raises(HTTPException):
            service.start(recipe = {}, run = {})
    finally:
        reset_workspace_subject(token)

    assert service._workspace_subject == "alice"
    assert service._job is not None and service._job.job_id == "alice-job"
    assert list(service._events) == [{"seq": 1}]


def test_a_shared_cache_path_still_answers_for_the_repo_it_holds(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes
    from routes import models as models_routes

    shared_cache = tmp_path / "shared" / "hf"
    private_snapshot = shared_cache / "models--org--private" / "snapshots" / "abc"
    public_snapshot = shared_cache / "models--org--public" / "snapshots" / "def"
    plain_folder = shared_cache / "just-a-folder"
    for path in (private_snapshot, public_snapshot, plain_folder):
        path.mkdir(parents = True)

    monkeypatch.setattr(
        models_routes,
        "advertised_shared_model_roots",
        lambda: [str(shared_cache.resolve())],
    )
    monkeypatch.setattr(
        inference_routes,
        "_hub_repo_is_anonymously_readable",
        lambda repo_id, repo_type: repo_id != "org/private",
    )

    token = _bind("alice")
    try:
        # Alice's private pull lands in the install-wide cache, and its snapshot
        # path is then in the catalog every account browses. Accepting the path
        # because the catalog offered it also skipped the credential question,
        # since by then it is an existing local path and not a repository id.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_uncontained_local_path(str(private_snapshot), "load")
        assert exc.value.status_code == 403
        # A public repo in the same cache is unchanged, and a cache directory that
        # names no repository is an ordinary shared folder.
        inference_routes._reject_uncontained_local_path(str(public_snapshot), "load")
        inference_routes._reject_uncontained_local_path(str(plain_folder), "load")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        inference_routes._reject_uncontained_local_path(str(private_snapshot), "load")
    finally:
        reset_workspace_subject(token)


def test_a_dictation_download_keeps_its_owner_while_it_runs(monkeypatch):
    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_STT_DOWNLOAD_INITIATORS", {}, raising = False)

    class _Module:
        def __init__(self) -> None:
            self.downloading = False

        def download_status(self) -> dict:
            return {"downloading": self.downloading}

    module = _Module()

    token = _bind("alice")
    try:
        assert inference_routes._claim_stt_download("transformers", module) is True
        module.downloading = True
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Recording unconditionally handed Alice's running transfer to whoever
        # asked next, who could then cancel it.
        assert inference_routes._claim_stt_download("transformers", module) is False
        assert inference_routes._STT_DOWNLOAD_INITIATORS["transformers"] == "alice"
        # Once it settles the slot is free, and a start that never happens gives
        # it straight back rather than leaving a stale owner behind.
        module.downloading = False
        assert inference_routes._claim_stt_download("transformers", module) is True
        inference_routes._release_stt_download_claim("transformers", "bob")
        assert "transformers" not in inference_routes._STT_DOWNLOAD_INITIATORS
    finally:
        reset_workspace_subject(token)


def test_a_cached_private_dictation_model_stays_with_its_downloader(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_STT_MODEL_DOWNLOADERS", {}, raising = False)
    monkeypatch.setattr(
        inference_routes,
        "_hub_repo_is_anonymously_readable",
        lambda repo_id, repo_type: repo_id != "alice/private-whisper",
    )

    token = _bind("alice")
    try:
        inference_routes._note_stt_model_downloader("alice/private-whisper")
        inference_routes._reject_private_stt_model_from_another_account("alice/private-whisper")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The load route carries no Hub credential and the sidecar reuses any
        # complete snapshot it finds, so the repository name was the whole lock.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_private_stt_model_from_another_account("alice/private-whisper")
        assert exc.value.status_code == 403
        # Public checkpoints, which is every curated id, load as before.
        inference_routes._reject_private_stt_model_from_another_account("openai/whisper-large-v3")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        inference_routes._reject_private_stt_model_from_another_account("alice/private-whisper")
    finally:
        reset_workspace_subject(token)


def test_a_retired_username_leaves_no_embedding_memo_behind(monkeypatch):
    from utils import embedding_model_settings

    monkeypatch.setattr(embedding_model_settings, "_resolved_gguf_memo", {}, raising = False)
    monkeypatch.setattr(embedding_model_settings, "_cached", {}, raising = False)

    embedding_model_settings._resolved_gguf_memo[("alice", "bge-m3")] = (
        "alice/private-embeddings",
        "gguf",
        False,
        ["model.gguf"],
    )
    embedding_model_settings._resolved_gguf_memo[("bob", "bge-m3")] = (None, None, False, None)
    embedding_model_settings._cached["alice"] = (0.0, (None, None, None, None, False, None))

    # A username is reusable, so without this the namesake resolves the same
    # model to the previous holder's repository and indexes in its embedding
    # space, and a reset can persist that into the replacement's own database.
    embedding_model_settings.forget_workspace("alice")

    assert ("alice", "bge-m3") not in embedding_model_settings._resolved_gguf_memo
    assert ("bob", "bge-m3") in embedding_model_settings._resolved_gguf_memo
    assert "alice" not in embedding_model_settings._cached


def test_a_download_that_is_still_starting_is_not_cancellable_by_anyone(monkeypatch):
    from fastapi import HTTPException

    from hub.services import download_lifecycle

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)

    class _Registry:
        def __init__(self) -> None:
            self.live = True

        def adoptable(self, key: str) -> bool:
            return self.live

    registry = _Registry()

    token = _bind("bob")
    try:
        # claim() publishes the job before the caller records itself, so a cancel
        # aimed at that window found no initiator and was authorized to kill a
        # transfer that was not its own.
        with pytest.raises(HTTPException) as exc:
            download_lifecycle.require_download_cancel_permission("org/model", registry)
        assert exc.value.status_code == 409
        # A key with no live job behind it stays cancellable: a download from
        # before a restart has no initiator either, and refusing those strands it.
        registry.live = False
        download_lifecycle.require_download_cancel_permission("org/model", registry)
        download_lifecycle.require_download_cancel_permission("org/model")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        registry.live = True
        download_lifecycle.require_download_cancel_permission("org/model", registry)
    finally:
        reset_workspace_subject(token)


def test_model_inspection_routes_contain_the_paths_they_read(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes
    from routes import models as models_routes

    elsewhere = tmp_path / "someone-elses" / "workspace" / "adapter"
    elsewhere.mkdir(parents = True)
    monkeypatch.setattr(models_routes, "advertised_shared_model_roots", lambda: [])

    token = _bind("alice")
    try:
        # get_gguf_variants walks the directory and reports its GGUF filenames,
        # sizes and quantizations; get_lora_base_model reads adapter_config.json
        # and returns the private base model it names.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_uncontained_local_path(str(elsewhere), "inspect")
        assert exc.value.status_code == 403
        # A Hub repo id is not a path and is left alone, as is a path that does
        # not exist: neither reads anything.
        inference_routes._reject_uncontained_local_path("unsloth/gemma-3-4b-it-GGUF", "inspect")
        inference_routes._reject_uncontained_local_path(str(tmp_path / "absent"), "inspect")
        inference_routes._reject_uncontained_local_path(None, "inspect")
    finally:
        reset_workspace_subject(token)


def test_the_lan_port_is_owner_only_like_the_rest_of_lan_access():
    import inspect

    from routes import settings as settings_routes

    # The listener is installation-wide: save_lan_access_port clears the shared
    # bind failure and writes the choice where the listener starts from, so a
    # managed session reaching it erased state only the owner can act on.
    for route in ("update_lan_access_port", "update_lan_access_auto_start"):
        signature = inspect.signature(getattr(settings_routes, route))
        dependency = signature.parameters["current_subject"].default
        assert dependency.dependency is settings_routes._require_install_admin, route


class _LoadRequestDouble:
    """The two fields the identifier resolution reads off a request."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.native_path_lease = None
        self.trust_remote_code = False
        self.hf_token = None


def test_a_private_resident_text_model_is_not_served_to_another_account(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes
    from routes import models as models_routes

    private = tmp_path / "alice-workspace" / "model.gguf"
    private.parent.mkdir(parents = True)
    private.write_bytes(b"")
    monkeypatch.setattr(models_routes, "advertised_shared_model_roots", lambda: [])
    monkeypatch.setattr(inference_routes, "_RESIDENT_TEXT_OWNER", None, raising = False)
    monkeypatch.setattr(
        inference_routes,
        "_resident_text_model_identifiers",
        lambda: [str(private)],
    )

    token = _bind("alice")
    try:
        inference_routes._note_text_model_loader(str(private), "alice")
        inference_routes._reject_generation_from_a_foreign_private_model()
        assert inference_routes.resident_text_model_is_foreign() is False
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Both text backends are process-wide and neither recorded who filled
        # them, so a path Bob cannot browse or load still answered his
        # completions request with its weights.
        with pytest.raises(HTTPException) as exc:
            inference_routes._reject_generation_from_a_foreign_private_model()
        assert exc.value.status_code == 409
        # And the status route reports it as nothing loaded rather than handing
        # over the absolute workspace path and the model's configuration.
        assert inference_routes.resident_text_model_is_foreign() is True

        # The record is pinned, not a bounded history: a caller cannot name
        # enough identifiers to push the resident model's owner out of it.
        for index in range(300):
            inference_routes._resolve_model_identifier_for_request(
                _LoadRequestDouble(f"/tmp/does-not-exist-{index}"),
                operation = "validate-model",
            )
        assert inference_routes._text_model_loader(str(private)) == "alice"
        with pytest.raises(HTTPException):
            inference_routes._reject_generation_from_a_foreign_private_model()
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        inference_routes._reject_generation_from_a_foreign_private_model()
    finally:
        reset_workspace_subject(token)

    # Retirement drops the record, so a recreated namesake does not match it.
    inference_routes.forget_text_model_owner("alice")
    token = _bind("alice")
    try:
        with pytest.raises(HTTPException):
            inference_routes._reject_generation_from_a_foreign_private_model()
    finally:
        reset_workspace_subject(token)


def test_one_workspace_cannot_evict_anothers_mcp_sessions(monkeypatch):
    from core.inference import mcp_client

    class _Session:
        def __init__(self, last_used: float) -> None:
            self.last_used = last_used
            self.in_flight = 0

    sessions = {
        ("http://a", (), "", "alice"): _Session(1.0),
        ("http://b", (), "one", "bob"): _Session(2.0),
        ("http://b", (), "two", "bob"): _Session(3.0),
        ("http://b", (), "three", "bob"): _Session(4.0),
    }
    monkeypatch.setattr(mcp_client, "_mcp_sessions", sessions, raising = False)
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 4, raising = False)

    idle = [(s.last_used, k) for k, s in sessions.items()]
    # Alice's session is the least recently used, so plain LRU handed Bob a way
    # to close her idle browser or REPL and destroy its server-side state by
    # opening enough scopes of his own.
    candidates = mcp_client._eviction_candidates_locked(idle, "bob")
    assert all(key[3] == "bob" for _, key in candidates)

    # With nobody over their share the global order stands: a full cache is a
    # capacity question, not one account crowding out another.
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 16, raising = False)
    assert mcp_client._eviction_candidates_locked(idle, "alice") == idle


def test_only_the_scanning_account_discards_the_code_it_downloaded():
    from fastapi import HTTPException

    from routes import models as models_routes

    models_routes._SCAN_CREATED_REMOTE_CODE.clear()

    token = _bind("alice")
    try:
        models_routes._note_scan_created_remote_code("org/code-dep", "alice")
        models_routes._reject_discarding_another_accounts_remote_code("org/code-dep")
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The loaded-model checks compare the supplied id with the resident
        # model, so they never protected a separate auto_map code dependency
        # another account's model still needs to load offline.
        with pytest.raises(HTTPException) as exc:
            models_routes._reject_discarding_another_accounts_remote_code("org/code-dep")
        assert exc.value.status_code == 403
        with pytest.raises(HTTPException):
            models_routes._reject_discarding_another_accounts_remote_code("org/never-scanned")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        models_routes._reject_discarding_another_accounts_remote_code("org/code-dep")
    finally:
        reset_workspace_subject(token)


def test_a_private_cached_dataset_is_not_readable_by_another_account(monkeypatch):
    from hub.services.datasets import cache_access
    from routes import inference as inference_routes

    monkeypatch.setattr(cache_access, "_dataset_downloaders", {}, raising = False)
    monkeypatch.setattr(cache_access, "_pending_downloads", {}, raising = False)
    monkeypatch.setattr(
        inference_routes,
        "_hub_repo_is_anonymously_readable",
        lambda repo_id, repo_type: repo_id != "alice/private-set",
    )

    token = _bind("alice")
    try:
        # Starting a download proves nothing: the worker has not authenticated
        # yet, so asking for one is not the grant.
        cache_access.note_dataset_download_attempt("job-1", "alice/private-set")
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is False
        cache_access.confirm_dataset_download("job-1")
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is True
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The cache is installation-wide and check-format only rejected the
        # anonymous sentinel, so any nonempty token read Alice's rows.
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is False
        # Public datasets, which is nearly all of them, are unaffected.
        assert cache_access.caller_may_read_cached_dataset("org/public-set") is True
        # A doomed download does not buy the grant either.
        cache_access.note_dataset_download_attempt("job-2", "alice/private-set")
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is False
    finally:
        reset_workspace_subject(token)

    # An unanswerable Hub withholds rather than guesses: offline and rate limited
    # both read as None, and treating that as permission published the inventory
    # and the rows for as long as the Hub was unreachable.
    monkeypatch.setattr(
        inference_routes, "_hub_repo_is_anonymously_readable", lambda repo_id, repo_type: None
    )
    token = _bind("bob")
    try:
        assert cache_access.caller_may_read_cached_dataset("org/public-set") is False
    finally:
        reset_workspace_subject(token)

    # And the grant is retired with the account, so a namesake does not inherit it.
    cache_access.forget_workspace("alice")
    token = _bind("alice")
    try:
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is False
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert cache_access.caller_may_read_cached_dataset("alice/private-set") is True
    finally:
        reset_workspace_subject(token)


def test_a_preview_link_does_not_survive_its_account(monkeypatch, tmp_path):
    from auth import storage as auth_storage
    from utils import preview_token

    incarnations: dict[str, str] = {}
    minted: list[str] = []

    def _incarnation(subject: str, *, create: bool = True) -> str:
        if subject not in incarnations:
            if not create:
                return ""
            minted.append(subject)
            incarnations[subject] = f"gen-{len(minted)}"
        return incarnations[subject]

    monkeypatch.setattr(auth_storage, "preview_link_incarnation", _incarnation)
    monkeypatch.setattr(
        preview_token, "get_or_create_preview_link_secret", lambda: b"secret", raising = False
    )

    token = _bind("alice")
    try:
        link = preview_token.sign_preview_ref("run-1/checkpoint-40")
        assert preview_token.verify_preview_ref("run-1/checkpoint-40", link) is True
    finally:
        reset_workspace_subject(token)

    # Deleting the account drops its incarnation, so the shared link stops
    # verifying rather than waiting for a namesake to produce the same ref.
    incarnations.pop("alice")
    assert preview_token.verify_preview_ref("run-1/checkpoint-40", link) is False

    # And a recreated namesake mints a different one, so it never inherits the
    # old links even once its own run reaches that ref.
    token = _bind("alice")
    try:
        assert preview_token.sign_preview_ref("run-1/checkpoint-40") != link
        assert preview_token.verify_preview_ref("run-1/checkpoint-40", link) is False
    finally:
        reset_workspace_subject(token)

    # The owner is the installation and cannot be recreated, so its links keep
    # the original payload and stay valid across this change.
    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        owner_link = preview_token.sign_preview_ref("run-2")
        assert preview_token.verify_preview_ref("run-2", owner_link) is True
        assert LEGACY_WORKSPACE_SUBJECT not in incarnations
    finally:
        reset_workspace_subject(token)


def test_the_remote_code_scan_contains_the_directory_it_reads(tmp_path, monkeypatch):
    import inspect

    from routes import models as models_routes

    source = inspect.getsource(models_routes.scan_model_remote_code)
    # The findings carry source snippets from the files it reads, so the paths
    # pass containment before the scanner opens anything.
    guarded, _, rest = source.partition("_reject_uncontained_local_path")
    assert rest, "the scan no longer contains the paths it reads"
    assert "load_scan_target" not in guarded
    for field in ("model_name", "model_local_path", "model_snapshot_path"):
        assert field in rest.split("hf_token_arg")[0], field


def test_the_idle_unload_loop_asks_the_owning_workspace(monkeypatch):
    import inspect

    from core.inference import llama_keepwarm
    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_RESIDENT_TEXT_OWNER", ("m", "alice"), raising = False)
    monkeypatch.setattr(inference_routes, "_resident_text_model_identifiers", lambda: ["m"])
    assert inference_routes.resident_text_model_workspace() == "alice"

    monkeypatch.setattr(inference_routes, "_resident_text_model_identifiers", lambda: [])
    assert inference_routes.resident_text_model_workspace() is None

    # The loop is created at startup, outside any request, so every settings read
    # in it landed in the owner's workspace whoever had loaded the model.
    source = inspect.getsource(llama_keepwarm.idle_unload_loop)
    for setting in (
        "get_auto_unload_idle_seconds",
        "get_auto_unload_keep_kv",
        "get_auto_unload_api_only",
    ):
        for line in source.splitlines():
            stripped = line.strip()
            # The import block at the top of the loop names them too.
            if not stripped.startswith(setting) and setting in stripped:
                assert "_in_owning_workspace" in stripped, stripped


def test_a_replacement_download_does_not_inherit_the_last_jobs_initiators(monkeypatch):
    from hub.services import download_lifecycle

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)

    token = _bind("alice")
    try:
        download_lifecycle.note_download_initiator("org/model", replaces_previous_job = True)
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Claiming the key for a new job says the previous set belonged to the
        # last one. Asking the registry could not: claim() publishes the
        # replacement before this runs, so the job always looked live and Alice
        # kept both the view of Bob's transfer and the right to cancel it.
        download_lifecycle.note_download_initiator("org/model", replaces_previous_job = True)
        assert download_lifecycle._download_initiators["org/model"] == {"bob"}
        # An adopter joins the running job rather than replacing its initiators.
        download_lifecycle.note_download_initiator("org/model")
        assert download_lifecycle._download_initiators["org/model"] == {"bob"}
    finally:
        reset_workspace_subject(token)

    token = _bind("alice")
    try:
        assert download_lifecycle.download_is_visible_to_caller("org/model") is False
        download_lifecycle.note_download_initiator("org/model")
        assert download_lifecycle._download_initiators["org/model"] == {"alice", "bob"}
    finally:
        reset_workspace_subject(token)


def test_another_accounts_dictation_model_is_not_named_in_status(monkeypatch):
    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_STT_MODEL_DOWNLOADERS", {}, raising = False)
    monkeypatch.setattr(
        inference_routes,
        "_hub_repo_is_anonymously_readable",
        lambda repo_id, repo_type: repo_id != "alice/private-whisper",
    )

    token = _bind("alice")
    try:
        inference_routes._note_stt_model_downloader("alice/private-whisper")
        assert (
            inference_routes._redacted_stt_model("alice/private-whisper") == "alice/private-whisper"
        )
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # The sidecars are installation-wide, so the status route named whichever
        # private repository somebody else had loaded, and a model= query
        # answered whether it is in the shared cache.
        assert inference_routes._redacted_stt_model("alice/private-whisper") is None
        assert inference_routes._redacted_stt_model("openai/whisper-large-v3") is not None
        redacted = inference_routes._redacted_stt_download_status(
            {"downloading": True, "model": "alice/private-whisper", "bytes_done": 5}
        )
        assert redacted["model"] is None
        assert redacted["downloading"] is True and redacted["bytes_done"] == 5
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        assert (
            inference_routes._redacted_stt_model("alice/private-whisper") == "alice/private-whisper"
        )
    finally:
        reset_workspace_subject(token)


def test_a_failed_teardown_leaves_the_model_fenced_rather_than_unowned(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_RESIDENT_TEXT_OWNER", None, raising = False)
    monkeypatch.setattr(
        inference_routes, "_resident_text_model_identifiers", lambda: ["alice/private-model"]
    )
    monkeypatch.setattr(
        inference_routes, "_hub_repo_is_anonymously_readable", lambda repo_id, kind: False
    )

    token = _bind("alice")
    try:
        inference_routes._note_text_model_loader("alice/private-model", "alice")
    finally:
        reset_workspace_subject(token)

    # Retirement fences before it unloads, because both unloads are best effort
    # and their failure is swallowed. An unowned Hub repository would otherwise
    # pass the containment fallback, which for a repo id is no containment.
    inference_routes.retire_text_model_owner("alice")

    for who in ("bob", "alice"):
        token = _bind(who)
        try:
            with pytest.raises(HTTPException):
                inference_routes._reject_generation_from_a_foreign_private_model()
        finally:
            reset_workspace_subject(token)

    # Once nothing is resident the record goes, and a later load owns it again.
    monkeypatch.setattr(inference_routes, "_resident_text_model_identifiers", lambda: [])
    inference_routes.forget_text_model_owner()
    assert inference_routes._RESIDENT_TEXT_OWNER is None


def test_a_foreign_load_in_flight_is_not_named_in_status(monkeypatch):
    from routes import inference as inference_routes

    class _Attempt:
        def __init__(self, model_path: str, subject: str) -> None:
            self.model_path = model_path
            self.subject = subject

    alices = _Attempt("/home/alice/workspace/secret.gguf", "alice")
    monkeypatch.setattr(inference_routes, "_running_load_attempt", alices, raising = False)
    monkeypatch.setattr(inference_routes, "_pending_load_attempts", {"t": alices}, raising = False)

    # The residency check cannot see this: the model is not resident yet, and the
    # attempt carries the repository id or the local checkpoint's basename.
    import inspect

    source = inspect.getsource(inference_routes.get_status)
    assert "_mine(" in source and 'getattr(attempt, "subject"' in source


def test_remote_code_grants_do_not_outlive_the_account_that_earned_them():
    from fastapi import HTTPException

    from routes import models as models_routes

    models_routes._SCAN_CREATED_REMOTE_CODE.clear()
    token = _bind("alice")
    try:
        models_routes._note_scan_created_remote_code("org/code-dep", "alice")
        models_routes._reject_discarding_another_accounts_remote_code("org/code-dep")
    finally:
        reset_workspace_subject(token)

    # A namesake would otherwise inherit the right to delete a cached code
    # dependency another account's approved model still loads from.
    models_routes.forget_scan_created_remote_code("alice")
    token = _bind("alice")
    try:
        with pytest.raises(HTTPException):
            models_routes._reject_discarding_another_accounts_remote_code("org/code-dep")
    finally:
        reset_workspace_subject(token)


def test_a_cached_private_embedding_repo_needs_more_than_being_cached(monkeypatch):
    from fastapi import HTTPException

    from routes import inference as inference_routes
    from routes import settings as settings_routes

    monkeypatch.setattr(
        inference_routes,
        "_hub_repo_is_anonymously_readable",
        lambda repo_id, kind: repo_id != "alice/private-embeddings",
    )

    token = _bind("bob")
    try:
        # The plan answers from the shared cache before it asks about
        # credentials, and the PUT persists it, so the process-wide embedder
        # would load Alice's weights for Bob's RAG.
        with pytest.raises(HTTPException) as exc:
            settings_routes._reject_private_embedding_repo("alice/private-embeddings", None)
        assert exc.value.status_code == 403
        # Public repos are unaffected, a curated slashless alias is resolved
        # against the sentence-transformers namespace and is not a private
        # download, and this account's own token answers for itself.
        settings_routes._reject_private_embedding_repo("BAAI/bge-m3", None)
        settings_routes._reject_private_embedding_repo("bge-m3", None)
        settings_routes._reject_private_embedding_repo("alice/private-embeddings", "hf_bobs")
    finally:
        reset_workspace_subject(token)

    token = _bind(LEGACY_WORKSPACE_SUBJECT)
    try:
        settings_routes._reject_private_embedding_repo("alice/private-embeddings", None)
    finally:
        reset_workspace_subject(token)


def test_only_the_loading_account_may_cancel_its_load(monkeypatch):
    from routes import inference as inference_routes

    class _Attempt:
        def __init__(self, subject: str) -> None:
            self.model_path = "unsloth/gemma-3-4b-it-GGUF"
            self.subject = subject

    monkeypatch.setattr(inference_routes, "_running_load_attempt", _Attempt("alice"), raising = False)

    token = _bind("alice")
    try:
        assert inference_routes._running_load_attempt_is_mine() is True
    finally:
        reset_workspace_subject(token)

    token = _bind("bob")
    try:
        # Both stop-loading fast paths run ahead of the scoped generation check,
        # so naming the same public identifier was enough to kill somebody
        # else's multi-gigabyte load, repeatedly.
        assert inference_routes._running_load_attempt_is_mine() is False
    finally:
        reset_workspace_subject(token)

    # Nothing loading means there is no load to protect, and the unload falls
    # through to the paths that decide on the resident model instead.
    monkeypatch.setattr(inference_routes, "_running_load_attempt", None, raising = False)
    token = _bind("bob")
    try:
        assert inference_routes._running_load_attempt_is_mine() is True
    finally:
        reset_workspace_subject(token)


def test_the_metadata_inspection_routes_contain_their_paths():
    import inspect

    from routes import models as models_routes

    # A config read reports the adapter base model, modality and context length
    # of whatever directory it is pointed at, and a miss is itself a probe.
    for route, fields in (
        (models_routes.get_model_config, ("model_name", "local_path")),
        (models_routes.check_vision_model, ("model_name",)),
        (models_routes.check_embedding_model, ("model_name",)),
    ):
        source = inspect.getsource(route)
        guarded, _, rest = source.partition("_reject_uncontained_local_path")
        assert rest, route.__name__
        assert "hf_token_arg" not in guarded, route.__name__
        for field in fields:
            assert field in rest.split("hf_token_arg")[0], (route.__name__, field)


def test_a_public_resident_model_is_shared_the_way_it_always_was(tmp_path, monkeypatch):
    from routes import inference as inference_routes
    from routes import models as models_routes

    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(
        models_routes, "advertised_shared_model_roots", lambda: [str(shared.resolve())]
    )
    monkeypatch.setattr(inference_routes, "_RESIDENT_TEXT_OWNER", None, raising = False)
    monkeypatch.setattr(
        inference_routes, "_hub_repo_is_anonymously_readable", lambda repo, kind: True
    )
    monkeypatch.setattr(
        inference_routes, "_resident_text_model_identifiers", lambda: ["unsloth/gemma-3-4b-it"]
    )
    inference_routes._note_text_model_loader("unsloth/gemma-3-4b-it", "alice")

    token = _bind("bob")
    try:
        # Bob may load this identifier himself, so refusing it because Alice got
        # there first served a 409 for a model he could have loaded a moment
        # later anyway. Accessibility decides; the loader is only a fast path.
        assert inference_routes.resident_text_model_is_foreign() is False
    finally:
        reset_workspace_subject(token)

    monkeypatch.setattr(
        inference_routes, "_hub_repo_is_anonymously_readable", lambda repo, kind: False
    )
    token = _bind("bob")
    try:
        assert inference_routes.resident_text_model_is_foreign() is True
    finally:
        reset_workspace_subject(token)


def test_download_and_dictation_grants_do_not_outlive_their_account(monkeypatch):
    from hub.services import download_lifecycle
    from routes import inference as inference_routes

    monkeypatch.setattr(download_lifecycle, "_download_initiators", {}, raising = False)
    monkeypatch.setattr(inference_routes, "_STT_MODEL_DOWNLOADERS", {}, raising = False)
    monkeypatch.setattr(inference_routes, "_STT_DOWNLOAD_INITIATORS", {}, raising = False)

    token = _bind("alice")
    try:
        download_lifecycle.note_download_initiator("org/private", replaces_previous_job = True)
        inference_routes._note_stt_download_initiator("transformers")
        inference_routes._note_stt_model_downloader("alice/private-whisper")
    finally:
        reset_workspace_subject(token)

    # A download is not a workspace job, so retirement's quiescing never saw it.
    download_lifecycle.forget_workspace_initiators("alice")
    inference_routes.forget_stt_model_downloader("alice")

    token = _bind("alice")
    try:
        assert download_lifecycle.download_is_visible_to_caller("org/private") is False
        assert inference_routes._STT_MODEL_DOWNLOADERS == {}
        assert inference_routes._STT_DOWNLOAD_INITIATORS == {}
    finally:
        reset_workspace_subject(token)


def test_the_first_scan_to_pull_a_repository_keeps_it():
    from routes import models as models_routes

    models_routes._SCAN_CREATED_REMOTE_CODE.clear()
    # Two accounts scanning the same uncached repository both find it absent.
    # Taking the later claim let the second account's decline cleanup delete the
    # cached code while the first was approving or loading it.
    models_routes._note_scan_created_remote_code("org/code-dep", "alice")
    models_routes._note_scan_created_remote_code("org/code-dep", "bob")
    assert models_routes._SCAN_CREATED_REMOTE_CODE["org/code-dep"] == "alice"


def test_every_transcription_route_asks_the_ownership_question():
    import inspect

    from routes import inference as inference_routes

    # /audio/transcribe, /audio/transcribe/raw and /v1/audio/transcriptions all
    # reach this, and each can load the model it names.
    source = inspect.getsource(inference_routes._transcribe_audio_result)
    guarded, _, rest = source.partition("_reject_private_stt_model_from_another_account")
    assert rest, "implicit transcription loads are unguarded"
    assert "load_stt" not in guarded

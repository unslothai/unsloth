# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real credentials and project routes share the owner's saved workspace location."""

import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage
from auth.authentication import create_access_token
from routes import chat_history, settings as settings_routes
from storage import studio_db
from utils import project_workspace_settings as settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as
from utils.keyless_api_access import _reset_scope_cache
from utils.paths.storage_roots import project_workspaces_root


@pytest.fixture
def tmp_path(linkable_tmp_path):
    return linkable_tmp_path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()
    _reset_scope_cache()
    marker = bind_account(OWNER)
    try:
        for username in (storage.DEFAULT_ADMIN_USERNAME, "alice"):
            storage.create_initial_user(
                username,
                "test-password",
                secrets.token_urlsafe(32),
                must_change_password = False,
            )
        app = FastAPI()
        app.include_router(settings_routes.router, prefix = "/api/settings")
        app.include_router(chat_history.router, prefix = "/api/chat")
        with TestClient(app) as test_client:
            yield test_client
    finally:
        reset_account(marker)
        storage._reset_api_key_hash_cache()
        policy.invalidate_account_cache()
        _reset_scope_cache()


def _headers(username = storage.DEFAULT_ADMIN_USERNAME, *, desktop = False):
    return {"Authorization": f"Bearer {create_access_token(username, desktop = desktop)}"}


def _create_project(client, headers, project_id):
    return client.post(
        "/api/chat/projects",
        headers = headers,
        json = {"id": project_id, "name": project_id, "createdAt": 1, "updatedAt": 1},
    )


def _save_home(client, headers, selected):
    response = client.put(
        "/api/settings/projects-home",
        headers = headers,
        json = {"projects_home": str(selected)},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("desktop", [False, True], ids = ["browser", "desktop"])
def test_owner_session_settings_control_new_project_routes(client, tmp_path, desktop):
    headers = _headers(desktop = desktop)
    original = _create_project(client, headers, "original")
    assert original.status_code == 200, original.text
    original_root = Path(original.json()["rootPath"])
    original_file = original_root / "sandbox" / "keep.txt"
    original_file.write_text("original contents")
    selected = tmp_path / "Project files – 日本語"
    selected.mkdir()

    saved = _save_home(client, headers, selected)
    assert client.get("/api/settings/projects-home", headers = headers).json() == saved
    created = _create_project(client, headers, "new")
    assert created.status_code == 200, created.text
    new_root = Path(created.json()["rootPath"])
    assert new_root.parent == selected
    assert (new_root / "sandbox").is_dir()
    reset = client.put("/api/settings/projects-home", headers = headers, json = {"projects_home": None})
    assert reset.status_code == 200, reset.text
    assert reset.json()["source"] == "default"
    for project_id, expected in (("original", original_root), ("new", new_root)):
        project = client.get(f"/api/chat/projects/{project_id}", headers = headers)
        assert project.status_code == 200, project.text
        assert project.json()["rootPath"] == str(expected)
    assert original_file.read_text() == "original contents"


def test_unauthenticated_requests_cannot_read_or_change_the_default(client, tmp_path):
    selected = tmp_path / "chosen"
    selected.mkdir()
    assert client.get("/api/settings/projects-home").status_code == 401
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(selected)})
    assert response.status_code == 401
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_SETTING_KEY) is None


def test_managed_session_cannot_read_owner_setting_or_move_its_projects(client, tmp_path):
    selected = tmp_path / "owner-projects"
    selected.mkdir()
    _save_home(client, _headers(), selected)
    account = storage.get_user_record("alice")
    alice = AccountContext(account["account_id"], "alice")
    expected = run_as(alice, project_workspaces_root)
    headers = _headers("alice")

    assert client.get("/api/settings/projects-home", headers = headers).status_code == 403
    refused = client.put(
        "/api/settings/projects-home", headers = headers, json = {"projects_home": None}
    )
    assert refused.status_code == 403
    project = _create_project(client, headers, "member-project")
    assert project.status_code == 200, project.text
    assert Path(project.json()["rootPath"]).parent == expected
    assert project_workspaces_root() == selected
    assert run_as(alice, studio_db.get_app_setting, settings.PROJECTS_HOME_SETTING_KEY) is None


@pytest.mark.parametrize("reset", [False, True])
def test_real_owner_api_key_cannot_save_or_reset_the_default(client, tmp_path, reset):
    selected = tmp_path / "chosen"
    selected.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _save_home(client, _headers(), selected)
    key, _ = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "workspace-test")
    response = client.put(
        "/api/settings/projects-home",
        headers = {"Authorization": f"Bearer {key}"},
        json = {"projects_home": None if reset else str(replacement)},
    )
    assert response.status_code == 403, response.text
    assert project_workspaces_root() == selected


def _symlink(alias, target):
    try:
        alias.symlink_to(target, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")


def test_selected_directory_symlink_is_canonicalized_before_project_creation(client, tmp_path):
    selected = tmp_path / "chosen"
    selected.mkdir()
    alias = tmp_path / "chosen-link"
    _symlink(alias, selected)
    headers = _headers()

    assert _save_home(client, headers, alias)["projects_home"] == str(selected.resolve())
    alias.unlink()
    created = _create_project(client, headers, "canonical")
    assert created.status_code == 200, created.text
    assert Path(created.json()["rootPath"]).parent == selected


@pytest.mark.parametrize("target_kind", ["studio", "project"])
def test_alias_cannot_bypass_protected_project_location_checks(client, tmp_path, target_kind):
    headers = _headers()
    project = _create_project(client, headers, "existing")
    assert project.status_code == 200, project.text
    selected = tmp_path / "chosen"
    selected.mkdir()
    _save_home(client, headers, selected)
    keys = [settings.PROJECTS_HOME_SETTING_KEY, settings.PROJECTS_HOME_HISTORY_SETTING_KEY]
    before = studio_db.get_app_settings(keys)
    target = tmp_path / "studio" if target_kind == "studio" else Path(project.json()["rootPath"])
    alias = tmp_path / "protected-link"
    _symlink(alias, target)

    response = client.put(
        "/api/settings/projects-home", headers = headers, json = {"projects_home": str(alias)}
    )

    assert response.status_code == 400, response.text
    assert studio_db.get_app_settings(keys) == before


def test_unreadable_settings_never_reroute_project_creation(client, tmp_path, monkeypatch):
    headers = _headers()
    selected = tmp_path / "chosen"
    selected.mkdir()
    _save_home(client, headers, selected)

    def refuse_read(_keys):
        raise PermissionError("settings database is unreadable")

    monkeypatch.setattr(studio_db, "get_app_settings", refuse_read)
    with pytest.raises(PermissionError, match = "settings database is unreadable"):
        settings.get_projects_home_status()
    with pytest.raises(PermissionError, match = "settings database is unreadable"):
        settings.owner_project_workspaces_roots()
    with pytest.raises(PermissionError, match = "settings database is unreadable"):
        _create_project(client, headers, "unreadable")
    assert studio_db.get_chat_project("unreadable") is None
    assert list(selected.iterdir()) == []


def test_replaced_saved_directory_returns_an_actionable_project_error(client, tmp_path):
    headers = _headers()
    old = _create_project(client, headers, "existing")
    assert old.status_code == 200, old.text
    selected = tmp_path / "chosen"
    selected.mkdir()
    _save_home(client, headers, selected)
    selected.rmdir()
    selected.write_text("a file replaced the directory")

    response = _create_project(client, headers, "cannot-create")

    assert response.status_code == 500, response.text
    assert "Settings > System > Storage" in response.json()["detail"]
    assert selected.read_text() == "a file replaced the directory"
    assert studio_db.get_chat_project("cannot-create") is None
    existing = client.get("/api/chat/projects/existing", headers = headers)
    assert existing.status_code == 200, existing.text
    assert existing.json()["rootPath"] == old.json()["rootPath"]


@pytest.mark.parametrize("missing", ["workspace", "parent", "sandbox"])
@pytest.mark.parametrize("method", ["GET", "POST", "PATCH"])
def test_offline_existing_project_is_not_recreated_or_modified(client, tmp_path, missing, method):
    headers = _headers()
    home = tmp_path / "selected-projects"
    home.mkdir()
    _save_home(client, headers, home)
    created = _create_project(client, headers, "offline-project")
    assert created.status_code == 200, created.text
    before = studio_db.get_chat_project("offline-project")
    root = Path(before["rootPath"])
    marker = root / "sandbox" / "keep.txt"
    marker.write_text("original project contents")
    # This recorded location must remain authoritative after the default changes.
    reset = client.put("/api/settings/projects-home", headers = headers, json = {"projects_home": None})
    assert reset.status_code == 200, reset.text
    absent = {"workspace": root, "parent": home, "sandbox": root / "sandbox"}[missing]
    offline = tmp_path / "offline-storage"
    absent.rename(offline)

    payload = {"name": "Renamed while offline", "updatedAt": 2}
    url = "/api/chat/projects/offline-project"
    if method == "POST":
        payload.update({"id": "offline-project", "createdAt": 1})
        url = "/api/chat/projects"
    response = client.request(
        method, url, headers = headers, **({"json": payload} if method != "GET" else {})
    )

    assert response.status_code == 503, response.text
    unavailable = root / "sandbox" if missing == "sandbox" else root
    assert str(unavailable) in response.json()["detail"]
    detail = response.json()["detail"].lower()
    assert "unavailable" in detail
    assert "reconnect" in detail or "restore" in detail
    assert not absent.exists()
    assert root.exists() is (missing == "sandbox")
    assert studio_db.get_chat_project("offline-project") == before
    assert (offline / marker.relative_to(absent)).read_text() == "original project contents"

    offline.rename(absent)
    restored = client.get("/api/chat/projects/offline-project", headers = headers)
    assert restored.status_code == 200, restored.text
    assert restored.json()["rootPath"] == str(root)
    assert marker.read_text() == "original project contents"


@pytest.mark.parametrize("missing", ["workspace", "sandbox"])
def test_offline_project_metadata_remains_available_to_its_account(client, tmp_path, missing):
    headers = _headers()
    created = _create_project(client, headers, "offline-export")
    assert created.status_code == 200, created.text
    before = studio_db.update_chat_project(
        "offline-export", {"instructions": "Keep these instructions"}
    )
    root = Path(before["rootPath"])
    absent = root if missing == "workspace" else root / "sandbox"
    absent.rename(tmp_path / "offline-storage")
    url = "/api/chat/projects/offline-export?require_workspace=false"

    response = client.get(url, headers = headers)

    assert response.status_code == 200, response.text
    assert response.json()["instructions"] == "Keep these instructions"
    assert response.json()["rootPath"] == str(root)
    assert client.get(url).status_code == 401
    assert client.get(url, headers = _headers("alice")).status_code == 404
    assert client.get("/api/chat/projects/offline-export", headers = headers).status_code == 503
    assert not absent.exists()
    assert studio_db.get_chat_project("offline-export") == before


def test_metadata_read_does_not_allocate_a_legacy_project_workspace(client):
    headers = _headers()
    created = _create_project(client, headers, "legacy")
    assert created.status_code == 200, created.text
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_projects SET root_path = NULL WHERE id = 'legacy'")
        conn.commit()
    finally:
        conn.close()

    response = client.get("/api/chat/projects/legacy?require_workspace=false", headers = headers)

    assert response.status_code == 200, response.text
    assert response.json()["rootPath"] is None
    assert studio_db.get_chat_project("legacy")["rootPath"] is None

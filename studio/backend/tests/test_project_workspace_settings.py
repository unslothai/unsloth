# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Changing the owner's default project folder affects new workspaces only."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes import settings as settings_routes
from storage import studio_db
from utils import hf_cache_settings
from utils import project_workspace_settings as settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as
from utils.paths.storage_roots import project_workspaces_root, shared_project_workspaces_root

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def tmp_path(linkable_tmp_path):
    return linkable_tmp_path


@pytest.fixture(autouse = True)
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    marker = bind_account(OWNER)
    try:
        yield
    finally:
        reset_account(marker)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()

    async def subject(request: Request):
        account = ALICE if request.headers.get("x-test-account") == "alice" else OWNER
        marker = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(marker)

    async def via_api_key(request: Request):
        return request.headers.get("x-test-api-key") == "1"

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = via_api_key
    app.include_router(settings_routes.router, prefix = "/api/settings")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    with TestClient(app) as test_client:
        yield test_client


def _project(project_id: str, name: str = "Example") -> dict:
    return {"id": project_id, "name": name, "createdAt": 1, "updatedAt": 1}


def _folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    return folder.resolve()


def test_default_status_does_not_create_a_project_folder(tmp_path):
    expected = tmp_path / "Documents" / "Unsloth Studio" / "Projects"
    status = settings.get_projects_home_status()
    assert Path(status["projects_home"]) == expected
    assert status["source"] == "default"
    assert status["editable"] is True
    assert status["is_custom"] is False
    assert not expected.exists()


def test_saved_default_is_used_for_new_projects_and_survives_a_fresh_process(tmp_path):
    selected = _folder(tmp_path, "chosen")
    status = settings.set_projects_home(str(selected))
    assert status["source"] == "studio"
    assert status["editable"] is True
    assert status["is_custom"] is True
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_SETTING_KEY) == str(selected)
    created = studio_db.upsert_chat_project(_project("new"))
    assert Path(created["rootPath"]).parent == selected
    assert (Path(created["rootPath"]) / "sandbox").is_dir()

    backend = Path(__file__).resolve().parents[1]
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json; from utils.project_workspace_settings import get_projects_home_status; "
            "print(json.dumps(get_projects_home_status()))",
        ],
        cwd = backend,
        env = dict(os.environ),
        text = True,
    )
    restored = json.loads(result.strip().splitlines()[-1])
    assert restored["projects_home"] == str(selected)
    assert restored["source"] == "studio"


def test_switch_and_reset_keep_existing_workspaces_and_their_contents(tmp_path):
    old = studio_db.upsert_chat_project(_project("old"))
    old_root = Path(old["rootPath"])
    old_file = old_root / "sandbox" / "keep.txt"
    old_file.write_text("original project")
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    custom = studio_db.upsert_chat_project(_project("custom"))
    custom_root = Path(custom["rootPath"])
    custom_file = custom_root / "sandbox" / "keep.txt"
    custom_file.write_text("custom project")

    updated = studio_db.upsert_chat_project({**old, "name": "Renamed", "updatedAt": 2})
    assert updated["rootPath"] == str(old_root)
    assert studio_db.ensure_chat_project_workspace("old")["rootPath"] == str(old_root)

    reset = settings.set_projects_home(None)
    assert reset["source"] == "default"
    assert reset["is_custom"] is False
    assert studio_db.get_chat_project("custom")["rootPath"] == str(custom_root)
    assert studio_db.ensure_chat_project_workspace("custom")["rootPath"] == str(custom_root)
    assert old_file.read_text() == "original project"
    assert custom_file.read_text() == "custom project"
    after_reset = studio_db.upsert_chat_project(_project("reset"))
    assert Path(after_reset["rootPath"]).parent == old_root.parent
    assert not (selected / old_root.name).exists()


def test_deleting_an_old_project_after_a_switch_only_removes_its_own_files(tmp_path):
    old = studio_db.upsert_chat_project(_project("old"))
    old_root = Path(old["rootPath"])
    (old_root / "sandbox" / "remove.txt").write_text("old")
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    new = studio_db.upsert_chat_project(_project("new"))
    new_file = Path(new["rootPath"]) / "sandbox" / "keep.txt"
    new_file.write_text("new")

    studio_db.delete_chat_project("old", delete_files = True)
    assert studio_db.get_chat_project("old") is None
    assert not old_root.exists()
    assert new_file.read_text() == "new"


def test_previous_custom_locations_remain_recorded_after_switch_and_reset(tmp_path):
    first = _folder(tmp_path, "first")
    second = _folder(tmp_path, "second")
    settings.set_projects_home(str(first))
    settings.set_projects_home(str(second))
    assert str(first) in studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY)
    settings.set_projects_home(None)
    history = studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY)
    assert str(first) in history
    assert str(second) in history


def test_owner_default_does_not_move_managed_account_workspaces(tmp_path):
    shared = shared_project_workspaces_root()
    alice_root = run_as(ALICE, project_workspaces_root)
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))

    assert project_workspaces_root() == selected
    assert shared_project_workspaces_root() == shared
    assert run_as(ALICE, project_workspaces_root) == alice_root
    alice = run_as(ALICE, studio_db.upsert_chat_project, _project("alice"))
    assert Path(alice["rootPath"]).parent == alice_root
    assert run_as(ALICE, studio_db.get_app_setting, settings.PROJECTS_HOME_SETTING_KEY) is None


def test_project_root_paths_include_archived_and_keep_account_scope(tmp_path):
    owner_root = str(tmp_path / "owner-active")
    archived_root = str(tmp_path / "owner-archived")
    alice_root = str(tmp_path / "alice-private")

    def seed(rows):
        conn = studio_db.get_connection()
        try:
            conn.executemany(
                "INSERT INTO chat_projects "
                "(id, name, root_path, archived, created_at, updated_at) "
                "VALUES (?, 'Example', ?, ?, 1, 1)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    seed(
        [
            ("active", owner_root, 0),
            ("archived", archived_root, 1),
            ("rootless", None, 0),
            ("empty", "", 0),
        ]
    )
    run_as(ALICE, seed, [("active", alice_root, 0)])

    assert set(studio_db.list_chat_project_root_paths()) == {owner_root, archived_root}
    assert run_as(ALICE, studio_db.list_chat_project_root_paths) == [alice_root]


def test_projects_environment_override_wins_without_overwriting_saved_default(
    tmp_path, monkeypatch
):
    selected = _folder(tmp_path, "chosen")
    forced = _folder(tmp_path, "environment")
    settings.set_projects_home(str(selected))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(forced))

    status = settings.get_projects_home_status()
    assert status["projects_home"] == str(forced)
    assert status["source"] == "environment"
    assert status["editable"] is False
    assert status["environment_variable"] == "UNSLOTH_STUDIO_PROJECTS_HOME"
    assert project_workspaces_root() == forced
    with pytest.raises(RuntimeError):
        settings.set_projects_home(None)
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_SETTING_KEY) == str(selected)
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME")
    assert project_workspaces_root() == selected


def test_missing_configured_folder_fails_new_creation_without_recreating_it(tmp_path):
    old = studio_db.upsert_chat_project(_project("old"))
    selected = _folder(tmp_path, "detached-volume")
    settings.set_projects_home(str(selected))
    selected.rmdir()

    assert project_workspaces_root() == selected
    assert not selected.exists()
    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.upsert_chat_project(_project("new"))
    assert not selected.exists()
    assert studio_db.get_chat_project("new") is None
    updated = studio_db.upsert_chat_project({**old, "name": "Still usable", "updatedAt": 2})
    assert updated["rootPath"] == old["rootPath"]
    assert studio_db.ensure_chat_project_workspace("old")["rootPath"] == old["rootPath"]


@pytest.mark.parametrize("part", ["root", "sandbox"])
def test_existing_workspace_replaced_by_a_file_is_left_untouched(tmp_path, part):
    project = studio_db.upsert_chat_project(_project("existing"))
    root = Path(project["rootPath"])
    target = root if part == "root" else root / "sandbox"
    target.rename(tmp_path / "original-directory")
    target.write_text("replacement file")
    with pytest.raises(studio_db.ProjectWorkspaceUnavailableError):
        studio_db.ensure_chat_project_workspace(project["id"])
    assert target.read_text() == "replacement file"
    assert studio_db.get_chat_project(project["id"]) == project


@pytest.mark.parametrize(
    "path", ["/Volumes/Projects/Example/sandbox", r"C:\Users\Test\Projects\Example\sandbox"]
)
def test_unavailable_workspace_message_preserves_path_and_explains_recovery(path):
    cause = FileNotFoundError(2, "No such file or directory", path)
    error = studio_db.ProjectWorkspaceUnavailableError(path, cause)

    assert path in str(error)
    assert "Reconnect" in str(error)
    assert "restore" in str(error)


def test_legacy_project_without_a_recorded_root_still_allocates_a_workspace(tmp_path):
    project = studio_db.upsert_chat_project(_project("legacy"))
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_projects SET root_path = NULL WHERE id = ?", (project["id"],))
        conn.commit()
    finally:
        conn.close()
    allocated = studio_db.ensure_chat_project_workspace(project["id"])
    assert Path(allocated["rootPath"]).parent == selected
    assert (Path(allocated["rootPath"]) / "sandbox").is_dir()
    assert Path(project["rootPath"]).is_dir()


@pytest.mark.parametrize("reset", [False, True])
def test_concurrent_setting_change_cannot_recreate_a_missing_selected_folder(
    tmp_path, monkeypatch, reset
):
    selected = _folder(tmp_path, "detached-volume")
    replacement = _folder(tmp_path, "replacement")
    settings.set_projects_home(str(selected))
    read_status = settings.get_projects_home_status

    def change_after_read():
        snapshot = read_status()
        monkeypatch.setattr(settings, "get_projects_home_status", read_status)
        selected.rmdir()
        settings.set_projects_home(None if reset else str(replacement))
        return snapshot

    monkeypatch.setattr(settings, "get_projects_home_status", change_after_read)
    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.upsert_chat_project(_project("concurrent"))
    assert not selected.exists()
    assert studio_db.get_chat_project("concurrent") is None
    created = studio_db.upsert_chat_project(_project("after-change"))
    assert Path(created["rootPath"]).parent == Path(read_status()["projects_home"])


@pytest.mark.parametrize("rootless", [False, True])
@pytest.mark.parametrize("stage", ["project", "sandbox"])
def test_a_folder_disappearing_during_creation_is_not_recreated(
    tmp_path, monkeypatch, rootless, stage
):
    project = _project("disappears")
    if rootless:
        studio_db.upsert_chat_project(project)
        conn = studio_db.get_connection()
        try:
            conn.execute("UPDATE chat_projects SET root_path = NULL WHERE id = ?", (project["id"],))
            conn.commit()
        finally:
            conn.close()
    selected = _folder(tmp_path, "detached-volume")
    settings.set_projects_home(str(selected))
    mkdir = Path.mkdir
    removed = False

    def disappear_before_mkdir(path, *args, **kwargs):
        nonlocal removed
        is_project = path.parent == selected
        is_sandbox = path.name == "sandbox" and path.parent.parent == selected
        if not removed and (
            (stage == "project" and is_project) or (stage == "sandbox" and is_sandbox)
        ):
            if is_sandbox:
                path.parent.rmdir()
            selected.rmdir()
            removed = True
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", disappear_before_mkdir)
    with pytest.raises(studio_db.ProjectWorkspaceError):
        if rootless:
            studio_db.ensure_chat_project_workspace(project["id"])
        else:
            studio_db.upsert_chat_project(project)
    assert removed
    assert not selected.exists()
    saved = studio_db.get_chat_project(project["id"])
    assert saved["rootPath"] is None if rootless else saved is None


def test_settings_api_round_trips_and_resets_the_default(client, tmp_path):
    selected = _folder(tmp_path, "chosen")
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(selected)})
    assert response.status_code == 200, response.text
    assert response.json()["projects_home"] == str(selected)
    assert response.json()["source"] == "studio"
    assert client.get("/api/settings/projects-home").json() == response.json()
    reset = client.put("/api/settings/projects-home", json = {"projects_home": None})
    assert reset.status_code == 200, reset.text
    assert reset.json()["source"] == "default"


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_settings_api_refuses_managed_accounts(client, method, tmp_path):
    selected = _folder(tmp_path, "chosen")
    response = client.request(
        method,
        "/api/settings/projects-home",
        headers = {"x-test-account": "alice"},
        json = {"projects_home": str(selected)},
    )
    assert response.status_code == 403, response.text
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_SETTING_KEY) is None


def test_settings_api_refuses_changes_controlled_by_environment(client, tmp_path, monkeypatch):
    forced = _folder(tmp_path, "environment")
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(forced))
    response = client.put("/api/settings/projects-home", json = {"projects_home": None})
    assert response.status_code == 409, response.text
    assert project_workspaces_root() == forced


def test_api_key_cannot_change_the_owner_default(client, tmp_path):
    selected = _folder(tmp_path, "chosen")
    response = client.put(
        "/api/settings/projects-home",
        headers = {"x-test-api-key": "1"},
        json = {"projects_home": str(selected)},
    )
    assert response.status_code == 403, response.text
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_SETTING_KEY) is None


@pytest.mark.parametrize("body", [{}, {"projects_home": 42}, {"projects_home": []}])
def test_malformed_payload_does_not_reset_the_saved_default(client, tmp_path, body):
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    response = client.put("/api/settings/projects-home", json = body)
    assert response.status_code == 422, response.text
    assert project_workspaces_root() == selected


@pytest.mark.parametrize("invalid_kind", ["missing", "file", "relative", "sensitive", "root"])
def test_invalid_directory_does_not_replace_the_saved_default(client, tmp_path, invalid_kind):
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    history = studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY)
    invalid = tmp_path / invalid_kind
    if invalid_kind == "file":
        invalid.write_text("not a directory")
    elif invalid_kind == "relative":
        invalid = Path("relative-projects")
    elif invalid_kind == "sensitive":
        invalid = tmp_path / ".ssh" / "projects"
        invalid.mkdir(parents = True)
    elif invalid_kind == "root":
        invalid = Path(tmp_path.anchor)
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(invalid)})
    assert response.status_code == 400, response.text
    assert project_workspaces_root() == selected
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY) == history
    if invalid_kind == "missing":
        assert not invalid.exists()
    elif invalid_kind == "file":
        assert response.json()["detail"] == "The selected projects location is not a folder."


@pytest.mark.parametrize("reserved_kind", ["sandbox", "cache", "cache_history", "hub_cache"])
@pytest.mark.parametrize("overlap", ["same", "inside", "contains"])
def test_reserved_storage_cannot_be_selected_as_projects_home(
    client, tmp_path, monkeypatch, reserved_kind, overlap
):
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    history = studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY)
    reserved = tmp_path / "external-storage" / "reserved"
    reserved.mkdir(parents = True)
    if reserved_kind == "sandbox":
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(reserved))
    elif reserved_kind == "cache":
        studio_db.upsert_app_settings({hf_cache_settings.CACHE_HOME_SETTING_KEY: str(reserved)})
    elif reserved_kind == "cache_history":
        studio_db.upsert_app_settings(
            {hf_cache_settings.CACHE_HISTORY_SETTING_KEY: [str(reserved)]}
        )
    else:
        monkeypatch.setattr(
            hf_cache_settings, "_EXPLICIT_CACHE_ENV", {"HF_HUB_CACHE": str(reserved)}
        )
    candidate = reserved
    if overlap == "inside":
        candidate = _folder(reserved, "projects")
    elif overlap == "contains":
        candidate = reserved.parent

    response = client.put("/api/settings/projects-home", json = {"projects_home": str(candidate)})

    assert response.status_code == 400, response.text
    assert project_workspaces_root() == selected
    assert studio_db.get_app_setting(settings.PROJECTS_HOME_HISTORY_SETTING_KEY) == history


@pytest.mark.parametrize("reset", [False, True])
def test_database_write_failure_preserves_the_saved_default_and_history(
    tmp_path, monkeypatch, reset
):
    first = _folder(tmp_path, "first")
    selected = _folder(tmp_path, "chosen")
    candidate = _folder(tmp_path, "candidate")
    settings.set_projects_home(str(first))
    settings.set_projects_home(str(selected))
    keys = [settings.PROJECTS_HOME_SETTING_KEY, settings.PROJECTS_HOME_HISTORY_SETTING_KEY]
    before = studio_db.get_app_settings(keys)

    def fail_write(_values):
        raise sqlite3.OperationalError("database write refused")

    monkeypatch.setattr(studio_db, "upsert_app_settings", fail_write)
    with pytest.raises(sqlite3.OperationalError, match = "database write refused"):
        settings.set_projects_home(None if reset else str(candidate))

    assert studio_db.get_app_settings(keys) == before
    assert project_workspaces_root() == selected
    assert list(candidate.iterdir()) == []


def test_unwritable_folder_does_not_replace_the_saved_default(client, tmp_path, monkeypatch):
    selected = _folder(tmp_path, "chosen")
    settings.set_projects_home(str(selected))
    unwritable = _folder(tmp_path, "unwritable")
    real_write_test = settings.tempfile.NamedTemporaryFile

    def reject_write(*args, **kwargs):
        if Path(kwargs.get("dir", "")) == unwritable:
            raise PermissionError("write refused")
        return real_write_test(*args, **kwargs)

    monkeypatch.setattr(settings.tempfile, "NamedTemporaryFile", reject_write)
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(unwritable)})
    assert response.status_code == 400, response.text
    assert project_workspaces_root() == selected
    assert list(unwritable.iterdir()) == []


@pytest.mark.parametrize(
    ("relative", "archived"), [(".", False), ("sandbox", False), ("sandbox/nested", True)]
)
def test_existing_project_cannot_contain_the_new_projects_parent(client, relative, archived):
    parent = studio_db.upsert_chat_project(_project("parent-project"))
    if archived:
        studio_db.update_chat_project(parent["id"], {"archived": True})
    root = Path(parent["rootPath"])
    selected = root / relative
    selected.mkdir(parents = True, exist_ok = True)
    keep = root / "sandbox" / "keep.txt"
    keep.write_text("Existing project content")
    before = settings.get_projects_home_status()

    response = client.put("/api/settings/projects-home", json = {"projects_home": str(selected)})

    assert response.status_code == 400, response.text
    assert (
        response.json()["detail"] == "Choose a projects folder outside existing project workspaces."
    )
    assert settings.get_projects_home_status() == before
    assert studio_db.get_chat_project(parent["id"])["rootPath"] == str(root)
    assert keep.read_text() == "Existing project content"


def test_existing_project_container_remains_a_valid_projects_parent(client):
    first = studio_db.upsert_chat_project(_project("first-project"))
    container = Path(first["rootPath"]).parent
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(container)})
    assert response.status_code == 200, response.text
    second = studio_db.upsert_chat_project(_project("second-project"))
    assert Path(second["rootPath"]).parent == container
    assert Path(first["rootPath"]).is_dir()


@pytest.mark.parametrize(
    "resolved_path",
    [
        r"\\server\share\projects",
        r"\\?\UNC\server\share\projects",
        r"\\.\C:\projects",
        r"\??\C:\projects",
    ],
    ids = ["unc", "extended-unc", "device", "nt-device"],
)
def test_resolved_network_or_device_alias_is_rejected_before_stat(
    client, tmp_path, monkeypatch, resolved_path
):
    selected = _folder(tmp_path, "chosen")
    keys = [settings.PROJECTS_HOME_SETTING_KEY, settings.PROJECTS_HOME_HISTORY_SETTING_KEY]
    studio_db.upsert_app_settings({keys[0]: str(selected), keys[1]: []})
    before = studio_db.get_app_settings(keys)
    alias = _folder(tmp_path, "local-alias")

    class UncheckedTarget(PureWindowsPath):
        def stat(self, *args, **kwargs):
            raise AssertionError("Resolved network or device target reached stat")

    # Model only the alias resolution; never contact a share or device on any host.
    target = UncheckedTarget(resolved_path)
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == alias:
            return target
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    response = client.put("/api/settings/projects-home", json = {"projects_home": str(alias)})

    assert response.status_code == 400, response.text
    assert "Choose a local projects folder" in response.text
    assert studio_db.get_app_settings(keys) == before

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from auth import policy
from routes import chat_history, models as models_route
from storage import studio_db
from utils import host_policy, project_workspace_settings
from utils.account_context import OWNER, AccountContext, as_api_caller, run_as


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(host_policy, "_local_bind", True)
    monkeypatch.setattr(host_policy, "_remote_connector_active", False)
    monkeypatch.setattr(host_policy, "_lan_connector_active", False)


def test_permitted_only_for_the_owner_on_an_unpublished_loopback_server(local, monkeypatch):
    assert policy.server_folder_pick_permitted() is True
    assert run_as(as_api_caller(OWNER), policy.server_folder_pick_permitted) is False
    assert run_as(AccountContext("a" * 32, "alice"), policy.server_folder_pick_permitted) is False

    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    assert policy.server_folder_pick_permitted() is False
    monkeypatch.setattr(host_policy, "_remote_connector_active", False)
    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    assert policy.server_folder_pick_permitted() is False
    monkeypatch.setattr(host_policy, "_lan_connector_active", False)
    monkeypatch.setattr(host_policy, "_auto_enabled", False)
    assert policy.server_folder_pick_permitted() is True
    monkeypatch.setattr(host_policy, "_local_bind", False)
    assert policy.server_folder_pick_permitted() is False


def test_the_body_takes_exactly_one_of_grant_or_path():
    assert chat_history.ChatProjectWorkingFolder(serverPath = "/x").server_path == "/x"
    assert chat_history.ChatProjectWorkingFolder(nativePathLease = "l").native_path_lease == "l"
    with pytest.raises(ValidationError):
        chat_history.ChatProjectWorkingFolder()
    with pytest.raises(ValidationError):
        chat_history.ChatProjectWorkingFolder(serverPath = "/x", nativePathLease = "l")


def test_a_path_is_refused_once_the_server_is_published(local, monkeypatch, tmp_path):
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    with pytest.raises(HTTPException) as exc_info:
        chat_history._resolve_server_working_folder(str(tmp_path))
    assert exc_info.value.status_code == 403


def test_a_path_goes_through_the_browser_allowlist_and_the_folder_checks(
    local, monkeypatch, tmp_path
):
    home = tmp_path / "home"
    repo = home / "code" / "app"
    repo.mkdir(parents = True)
    outside = tmp_path / "outside"
    outside.mkdir()
    studio = home / "studio"
    studio.mkdir()
    monkeypatch.setattr(models_route, "_build_browse_allowlist", lambda *_a, **_k: [home.resolve()])
    monkeypatch.setattr(studio_db, "is_denied_system_path", lambda _p: False)
    from hub.storage import scan_folders
    from utils.paths import storage_roots

    monkeypatch.setattr(scan_folders, "is_denied_system_path", lambda _p: False)
    for name in ("studio_root", "shared_project_workspaces_root", "shared_tmp_root"):
        monkeypatch.setattr(storage_roots, name, lambda: studio)
    from core.inference import tools

    monkeypatch.setattr(tools, "shared_sandbox_root", lambda: studio)

    path, identity = chat_history._resolve_server_working_folder(str(repo))
    assert path == str(repo.resolve())
    assert identity == (repo.stat().st_dev, repo.stat().st_ino)

    with pytest.raises(HTTPException) as exc_info:
        chat_history._resolve_server_working_folder(str(outside))
    assert exc_info.value.status_code == 403
    for reserved in (studio, home):
        with pytest.raises(HTTPException) as exc_info:
            chat_history._resolve_server_working_folder(str(reserved))
        assert exc_info.value.status_code == 400
    (home / ".ssh").mkdir()
    with pytest.raises(HTTPException) as exc_info:
        chat_history._resolve_server_working_folder(str(home / ".ssh"))
    assert exc_info.value.status_code == 403


@pytest.fixture
def saved_project_home(local, monkeypatch, linkable_tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(linkable_tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(linkable_tmp_path / "Documents"))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(
        models_route, "_build_browse_allowlist", lambda *_a, **_k: [linkable_tmp_path.resolve()]
    )
    studio_db.upsert_chat_project({"id": "p1", "name": "Linked", "createdAt": 1, "updatedAt": 1})
    home = linkable_tmp_path / "code" / "projects"
    home.mkdir(parents = True)
    project_workspace_settings.set_projects_home(str(home))
    return home


@pytest.mark.parametrize("reset", [False, True], ids = ["current", "former"])
def test_server_path_accepts_an_unrelated_sibling_in_a_saved_project_home(
    saved_project_home, reset
):
    candidate = saved_project_home / "repo"
    candidate.mkdir()
    if reset:
        project_workspace_settings.set_projects_home(None)

    saved = chat_history.set_project_working_folder(
        "p1", chat_history.ChatProjectWorkingFolder(serverPath = str(candidate)), current_subject = "u"
    )

    assert saved.workingFolder == str(candidate)
    assert saved.workingFolderStatus == "ok"


@pytest.mark.parametrize("reset", [False, True], ids = ["current", "former"])
def test_server_path_protects_saved_project_homes_and_live_or_orphaned_roots(
    saved_project_home, reset
):
    from core.inference import tools

    roots = []
    for state in ("live", "pending-delete", "kept"):
        managed = studio_db.upsert_chat_project(
            {"id": state, "name": state, "createdAt": 1, "updatedAt": 1}
        )
        root = Path(managed["rootPath"])
        roots.append(root)
        if state != "live":
            studio_db.delete_chat_project(state)
            tools.record_orphaned_project(
                state, str(root / "sandbox"), state == "pending-delete", str(root)
            )
    if reset:
        project_workspace_settings.set_projects_home(None)

    for candidate in (*roots, saved_project_home, saved_project_home.parent):
        with pytest.raises(HTTPException) as caught:
            chat_history.set_project_working_folder(
                "p1",
                chat_history.ChatProjectWorkingFolder(serverPath = str(candidate)),
                current_subject = "u",
            )
        assert caught.value.status_code == 400
        assert "Unsloth's own" in caught.value.detail
    assert studio_db.get_chat_project("p1")["workingFolder"] is None

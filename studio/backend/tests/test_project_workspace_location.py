# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a project's workspace folder lands, and what happens when it cannot.

Project workspaces are the only thing Unsloth writes into the user's Documents,
so a Documents folder it guesses wrong about breaks project creation and
nothing else. On Windows that guess is wrong by default whenever OneDrive's
Known Folder Move has repointed Documents at the synced copy.
"""

import os
from pathlib import Path

import pytest

from utils.paths.storage_roots import (
    _documents_from_registry_value,
    _windows_documents_dir,
    documents_root,
    project_workspaces_root,
)


def test_a_redirected_documents_folder_is_read_as_written():
    """REG_SZ is already absolute, including the OneDrive case this is for."""
    assert _documents_from_registry_value(r"C:\Users\t\OneDrive\Documents", False) == Path(
        r"C:\Users\t\OneDrive\Documents"
    )
    # A variable with nothing to expand into stays put rather than vanishing.
    assert _documents_from_registry_value(r"%NOT_A_REAL_VAR%\Documents", True) == Path(
        r"%NOT_A_REAL_VAR%\Documents"
    )


def test_expansion_uses_windows_syntax(monkeypatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\tombino")
    assert _documents_from_registry_value(r"%USERPROFILE%\Documents", True) == Path(
        r"C:\Users\tombino\Documents"
    )
    # Without the flag the value is taken literally, variables and all.
    assert _documents_from_registry_value(r"%USERPROFILE%\Documents", False) == Path(
        r"%USERPROFILE%\Documents"
    )


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_an_unusable_registry_value_falls_through(value):
    """Anything but a real string has to fall back, not become Path('.')."""
    assert _documents_from_registry_value(value, True) is None


@pytest.mark.skipif(os.name == "nt", reason = "the registry read is the point on Windows")
def test_the_registry_is_only_read_on_windows():
    assert _windows_documents_dir() is None


def test_the_override_still_wins(tmp_path, monkeypatch):
    """Whatever Documents resolves to, this is the documented way out."""
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "elsewhere"))
    assert documents_root() == tmp_path / "elsewhere"
    assert project_workspaces_root() == (tmp_path / "elsewhere" / "Unsloth Studio" / "Projects")


def test_the_projects_override_wins_outright(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "documents"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    assert project_workspaces_root() == tmp_path / "projects"


def _probe_payload():
    from routes import chat_history
    return chat_history.ChatProjectCreate(
        id = "11111111-2222-3333-4444-555555555555",
        name = "Probe",
        instructions = "",
        archived = False,
        createdAt = 1,
        updatedAt = 1,
    )


def test_the_workspace_error_carries_the_folder_it_could_not_make(tmp_path, monkeypatch):
    """The failing path, not the root it was derived from.

    An existing project keeps a recorded rootPath that can sit anywhere, so the
    configured projects root is not always the folder that failed.
    """
    from storage import studio_db
    from storage.studio_db import ProjectWorkspaceError, _ensure_project_workspace

    blocked = tmp_path / "read-only" / "child"

    # The refusal is stubbed rather than staged with chmod: root ignores a
    # read-only directory, and Windows does not enforce one this way at all.
    def refuse(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(studio_db, "ensure_dir", refuse)
    with pytest.raises(ProjectWorkspaceError) as caught:
        _ensure_project_workspace(str(blocked))
    assert caught.value.path == str(blocked)


def test_creating_a_project_says_which_folder_failed(tmp_path, monkeypatch):
    """A folder Unsloth cannot create is the one failure this route has.

    It used to surface as a bare 500, which says nothing about which folder or
    what to do, and the folder is one the user can move.
    """
    from fastapi import HTTPException

    from routes import chat_history
    from storage.studio_db import ProjectWorkspaceError

    blocked = tmp_path / "no-entry"
    monkeypatch.setattr(
        chat_history,
        "upsert_chat_project",
        lambda payload, external_workspace_path = None, external_workspace_identity = None: (
            _ for _ in ()
        ).throw(ProjectWorkspaceError(str(blocked), PermissionError(13, "Permission denied"))),
    )

    with pytest.raises(HTTPException) as caught:
        chat_history.save_project(_probe_payload(), current_subject = "tester")

    assert caught.value.status_code == 500
    detail = str(caught.value.detail)
    assert str(blocked) in detail
    assert "UNSLOTH_STUDIO_PROJECTS_HOME" in detail
    # The raw OSError text stays in the log, not in the response.
    assert "Permission denied" not in detail


def test_a_database_folder_failure_is_not_blamed_on_the_projects_folder(monkeypatch):
    """The same upsert opens studio.db before it picks a workspace.

    That folder is UNSLOTH_STUDIO_HOME's, so answering it with "set
    UNSLOTH_STUDIO_PROJECTS_HOME" sends the user to fix the wrong path.
    """
    from routes import chat_history

    monkeypatch.setattr(
        chat_history,
        "upsert_chat_project",
        lambda payload, external_workspace_path = None, external_workspace_identity = None: (
            _ for _ in ()
        ).throw(PermissionError(13, "studio.db")),
    )

    with pytest.raises(PermissionError):
        chat_history.save_project(_probe_payload(), current_subject = "tester")


def test_external_project_creation_requires_a_native_folder_grant():
    from fastapi import HTTPException

    payload = _probe_payload().model_copy(update = {"workspaceKind": "external"})

    with pytest.raises(HTTPException) as caught:
        from routes import chat_history
        chat_history.save_project(payload, current_subject = "tester")

    assert caught.value.status_code == 400
    assert "selected workspace folder" in str(caught.value.detail)


def test_external_project_creation_uses_only_the_verified_folder(tmp_path, monkeypatch):
    from routes import chat_history

    selected = tmp_path / "project"
    selected.mkdir()
    payload = _probe_payload().model_copy(
        update = {"workspaceKind": "external", "nativePathLease": "signed"}
    )
    captured = {}

    monkeypatch.setattr(
        chat_history,
        "_resolve_project_workspace_path",
        lambda lease: (str(selected.resolve()), ("1", "2")),
    )

    def save(
        project,
        external_workspace_path = None,
        external_workspace_identity = None,
    ):
        captured["path"] = external_workspace_path
        captured["identity"] = external_workspace_identity
        return {
            **project,
            "rootPath": str(tmp_path / "managed"),
            "workspacePath": external_workspace_path,
            "workspaceKind": "external",
            "workspaceAvailable": True,
            "sandboxPath": external_workspace_path,
        }

    monkeypatch.setattr(chat_history, "upsert_chat_project", save)

    created = chat_history.save_project(payload, current_subject = "tester")

    assert captured["path"] == str(selected.resolve())
    assert captured["identity"] == ("1", "2")
    assert created.workspacePath == str(selected.resolve())


def test_external_project_creation_rejects_an_existing_project_before_using_the_grant(monkeypatch):
    from fastapi import HTTPException

    from routes import chat_history

    payload = _probe_payload().model_copy(
        update = {"workspaceKind": "external", "nativePathLease": "signed"}
    )
    monkeypatch.setattr(chat_history, "get_chat_project", lambda project_id: {"id": project_id})
    monkeypatch.setattr(
        chat_history,
        "_resolve_project_workspace_path",
        lambda lease: (_ for _ in ()).throw(AssertionError("grant was used")),
    )

    with pytest.raises(HTTPException) as caught:
        chat_history.save_project(payload, current_subject = "tester")

    assert caught.value.status_code == 409


def test_busy_project_does_not_consume_the_workspace_grant(monkeypatch):
    from fastapi import HTTPException

    from core.inference import tools
    from routes import chat_history

    resolved = []
    monkeypatch.setattr(
        chat_history,
        "_resolve_project_workspace_path",
        lambda lease: resolved.append(lease),
    )
    monkeypatch.setattr(
        tools,
        "update_project_workspace_when_idle",
        lambda project_id, update: (False, None),
    )
    payload = chat_history.ChatProjectPatch(
        workspaceKind = "external",
        nativePathLease = "signed",
    )

    with pytest.raises(HTTPException) as caught:
        chat_history.patch_project("project-1", payload, current_subject = "tester")

    assert caught.value.status_code == 409
    assert resolved == []


def test_busy_orphan_workspace_returns_conflict(monkeypatch):
    from fastapi import HTTPException

    from core.inference import tools
    from routes import chat_history
    from storage.studio_db import ProjectWorkspaceConflictError

    monkeypatch.setattr(
        chat_history,
        "_resolve_project_workspace_path",
        lambda lease: ("/selected/project", ("1", "2")),
    )
    monkeypatch.setattr(
        tools,
        "update_project_workspace_when_idle",
        lambda project_id, update: (True, update()),
    )

    def reject_busy_workspace(project_id, path, identity):
        raise ProjectWorkspaceConflictError(
            "Wait for active tool calls in the selected folder to finish"
        )

    monkeypatch.setattr(chat_history, "set_chat_project_workspace", reject_busy_workspace)
    payload = chat_history.ChatProjectPatch(
        workspaceKind = "external",
        nativePathLease = "signed",
    )

    with pytest.raises(HTTPException) as caught:
        chat_history.patch_project("project-1", payload, current_subject = "tester")

    assert caught.value.status_code == 409
    assert "active tool calls" in str(caught.value.detail)


def test_unavailable_external_project_does_not_break_the_project_list(monkeypatch):
    from routes import chat_history

    project = {
        "id": "project-1",
        "name": "Project",
        "instructions": "",
        "rootPath": "/managed/project-1",
        "workspacePath": "/moved/project",
        "workspaceKind": "external",
        "workspaceAvailable": False,
        "sandboxPath": "/moved/project",
        "archived": False,
        "createdAt": 1,
        "updatedAt": 1,
    }
    monkeypatch.setattr(chat_history, "list_chat_projects", lambda include_archived: [project])
    monkeypatch.setattr(
        chat_history,
        "ensure_chat_project_workspace",
        lambda project_id: (_ for _ in ()).throw(AssertionError("external folder was recreated")),
    )

    response = chat_history.list_projects(current_subject = "tester")

    assert response.projects[0].workspaceAvailable is False
    assert response.projects[0].workspacePath == "/moved/project"

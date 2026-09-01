# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Folder-backed project ownership, identity, and persistence boundaries."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
import os
import shlex
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core.inference import tools
from core.agent_workspace.common import (
    AgentWorkspaceError,
    project_workspace,
    workspace_fingerprint,
)
from core.agent_workspace.execution import execution_boundary_status
from core.agent_workspace.state import (
    begin_verification_run,
    finish_verification_run,
    set_verification_config,
)
from core.agent_workspace.verification import GOAL_COMPLETION_VERIFICATION_DETAIL
from routes import chat_history
from storage import studio_db
from utils import native_path_leases as leases


_LEASE_SECRET = b"project-folder-test-secret-value"


@pytest.fixture(autouse = True)
def _isolated_project_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "managed-projects"))
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(_LEASE_SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None)
    leases._USED_NONCES.clear()
    with tools._sessions_free:
        tools._deleting_project_sessions.clear()
    yield
    with tools._sessions_free:
        tools._deleting_project_sessions.clear()
        tools._sessions_free.notify_all()
    leases._USED_NONCES.clear()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign_folder(
    path: Path,
    *,
    operation: str = "open-project",
    path_kind: str = "document-folder",
    path_type: str = "directory",
) -> str:
    """Mint the folder grant emitted by the native picker."""
    stat = path.stat()
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 2,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": path_kind,
        "path_type": path_type,
        "source_kind": "dialog",
        "token_id_hash": hashlib.sha256(b"project_path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": os.urandom(16).hex(),
        "display_label": path.name,
        "size_bytes": None,
        "modified_ms": None,
        "device_id": format(stat.st_dev, "x"),
        "file_id": format(stat.st_ino, "x"),
    }
    envelope_nonce = os.urandom(leases._LEASE_V2_NONCE_BYTES)
    plaintext = json.dumps(payload, separators = (",", ":")).encode("utf-8")
    ciphertext = leases._xor_lease_stream(_LEASE_SECRET, envelope_nonce, plaintext)
    envelope = envelope_nonce + ciphertext
    signature = hmac.new(
        _LEASE_SECRET,
        leases._LEASE_V2_AUTH_DOMAIN + envelope,
        hashlib.sha256,
    ).digest()
    return f"2.{_b64(envelope)}.{_b64(signature)}"


def _folder_project(project_id: str, root: Path, **overrides) -> dict:
    stat = root.stat()
    project = {
        "id": project_id,
        "name": root.name,
        "instructions": "",
        "rootPath": str(root),
        "workspaceKind": "folder",
        "workspaceDeviceId": str(stat.st_dev),
        "workspaceFileId": str(stat.st_ino),
        "goal": None,
        "goalStatus": None,
        "goalUpdatedAt": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
    }
    project.update(overrides)
    return project


def _managed_project(project_id: str) -> dict:
    return {
        "id": project_id,
        "name": "Managed",
        "instructions": "",
        "workspaceKind": "managed",
        "archived": False,
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
    }


def _open_folder(path: Path, name: str = "Repository") -> chat_history.ChatProject:
    return chat_history.open_project_folder(
        chat_history.OpenProjectFolderRequest(
            nativePathLease = _sign_folder(path),
            name = name,
        ),
        current_subject = "tester",
    )


def _project_api_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[chat_history.get_current_subject] = lambda: "tester"
    return TestClient(app)


def _assert_renderer_project_has_no_paths(payload, *paths: Path) -> None:
    encoded = json.dumps(payload)
    assert "rootPath" not in encoded
    assert "sandboxPath" not in encoded
    assert "workspaceDeviceId" not in encoded
    assert "workspaceFileId" not in encoded
    for path in paths:
        assert str(path.resolve()) not in encoded


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation": "link-documents"},
        {"path_kind": "attachment"},
        {"path_type": "file"},
    ],
)
def test_project_folder_lease_requires_exact_operation_kind_and_type(tmp_path, overrides):
    folder = tmp_path / "repository"
    folder.mkdir()

    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_path(_sign_folder(folder, **overrides))

    assert caught.value.status_code == 400


@pytest.mark.parametrize("field", ["rootPath", "sandboxPath"])
def test_managed_project_api_rejects_renderer_filesystem_paths(tmp_path, field):
    payload = {
        "id": "renderer-path-probe",
        "name": "Path probe",
        "instructions": "",
        "archived": False,
        "createdAt": 1,
        "updatedAt": 1,
        field: str(tmp_path / "renderer-controlled"),
    }

    with _project_api_client() as client:
        response = client.post("/api/chat/projects", json = payload)

    assert response.status_code == 422
    assert studio_db.get_chat_project(payload["id"]) is None


def test_project_api_never_serializes_canonical_workspace_paths(tmp_path, monkeypatch):
    repository = tmp_path / "private-repository"
    repository.mkdir()
    managed_payload = {
        "id": "managed-public-view",
        "name": "Managed",
        "instructions": "",
        "archived": False,
        "createdAt": 1,
        "updatedAt": 1,
    }
    client = _project_api_client()

    with client:
        created_managed = client.post("/api/chat/projects", json = managed_payload)
        native_path_lease = _sign_folder(repository)
        assert native_path_lease.startswith("2.")
        opened_folder = client.post(
            "/api/chat/projects/open-folder",
            json = {
                "nativePathLease": native_path_lease,
                "name": "Repository",
            },
        )
        assert created_managed.status_code == 200
        assert opened_folder.status_code == 200
        folder_id = opened_folder.json()["id"]

        stored = studio_db.get_chat_project(folder_id)
        assert stored is not None
        assert stored["rootPath"] == str(repository.resolve())
        assert stored["sandboxPath"] == str(repository.resolve())

        responses = [
            created_managed.json(),
            opened_folder.json(),
            client.get("/api/chat/projects").json(),
            client.get(f"/api/chat/projects/{folder_id}").json(),
            client.get("/api/chat/export").json(),
        ]

        raw_patch = client.patch(
            f"/api/chat/projects/{folder_id}",
            json = {"rootPath": str(tmp_path / "swapped")},
        )
        assert raw_patch.status_code == 422

        monkeypatch.setattr(
            chat_history.agent_background_manager,
            "begin_project_deletion",
            lambda _project_id: None,
        )
        monkeypatch.setattr(
            chat_history.agent_background_manager,
            "finish_project_deletion",
            lambda _project_id: None,
        )
        monkeypatch.setattr(
            chat_history.agent_background_manager,
            "cancel_project_tasks_and_wait",
            lambda _project_id: None,
        )
        monkeypatch.setattr(
            chat_history, "begin_verification_project_deletion", lambda _project_id: None
        )
        monkeypatch.setattr(
            chat_history, "finish_verification_project_deletion", lambda _project_id: None
        )
        monkeypatch.setattr(
            chat_history,
            "cancel_project_verifications_and_wait",
            lambda _project_id: None,
        )
        monkeypatch.setattr(chat_history, "list_active_worktrees", lambda _project_id: [])
        monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
        monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_args: None)
        monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda *_args: None)
        monkeypatch.setattr(
            chat_history, "_remove_conversation_archives", lambda *_args, **_kwargs: None
        )

        async def remove_sandboxes(_ids, _delete_files):
            return 0, []

        monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
        deleted = client.delete(f"/api/chat/projects/{folder_id}")
        assert deleted.status_code == 200
        responses.append(deleted.json())

    for response in responses:
        _assert_renderer_project_has_no_paths(response, repository)


def test_valid_project_folder_lease_is_signed_identity_bound_and_single_use(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)

    root, label, device_id, file_id = chat_history._resolve_project_folder_path(lease)

    assert root == str(folder.resolve())
    assert label == "repository"
    assert device_id == str(folder.stat().st_dev)
    assert file_id == str(folder.stat().st_ino)
    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_path(lease)
    assert caught.value.status_code == 400


def test_project_folder_lease_rejects_a_replaced_directory(tmp_path):
    folder = tmp_path / "repository"
    replaced = tmp_path / "replaced-repository"
    folder.mkdir()
    lease = _sign_folder(folder)
    folder.rename(replaced)
    folder.mkdir()

    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_path(lease)

    assert caught.value.status_code == 400
    assert "changed" in str(caught.value.detail)


def test_opening_the_exact_folder_reuses_and_unarchives_its_project(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    first = _open_folder(folder, "First name")
    studio_db.update_chat_project(first.id, {"archived": True})

    reopened = _open_folder(folder, "Ignored replacement name")

    assert reopened.id == first.id
    assert reopened.name == "First name"
    assert reopened.archived is False
    assert [project["id"] for project in studio_db.list_chat_projects(True)] == [first.id]


@pytest.mark.parametrize("existing_is_parent", [True, False])
def test_overlapping_folder_roots_are_rejected(tmp_path, existing_is_parent):
    parent = tmp_path / "repository"
    child = parent / "package"
    child.mkdir(parents = True)
    existing, candidate = (parent, child) if existing_is_parent else (child, parent)
    _open_folder(existing)

    with pytest.raises(HTTPException) as caught:
        _open_folder(candidate)

    assert caught.value.status_code == 409
    assert "overlaps" in str(caught.value.detail)


def test_folder_project_cannot_claim_the_managed_workspace_root(tmp_path):
    managed_root = tmp_path / "managed-projects"
    managed_root.mkdir()

    with pytest.raises(HTTPException) as caught:
        _open_folder(managed_root)

    assert caught.value.status_code == 409
    assert "overlaps" in str(caught.value.detail)
    assert studio_db.list_chat_projects(include_archived = True) == []


def test_concurrent_parent_and_child_folder_claims_are_serialized(tmp_path):
    parent = tmp_path / "repository"
    child = parent / "package"
    child.mkdir(parents = True)
    studio_db.list_chat_projects(include_archived = True)

    def claim(path: Path):
        try:
            return _open_folder(path)
        except HTTPException as exc:
            return exc

    with ThreadPoolExecutor(max_workers = 2) as pool:
        first = pool.submit(claim, parent)
        second = pool.submit(claim, child)
        results = [first.result(timeout = 10), second.result(timeout = 10)]

    successes = [result for result in results if isinstance(result, chat_history.ChatProject)]
    conflicts = [result for result in results if isinstance(result, HTTPException)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].status_code == 409
    assert "overlaps" in str(conflicts[0].detail)
    stored = studio_db.list_chat_projects(include_archived = True)
    assert [project["id"] for project in stored] == [successes[0].id]


def test_concurrent_same_folder_claims_reuse_one_project(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    studio_db.list_chat_projects(include_archived = True)

    with ThreadPoolExecutor(max_workers = 2) as pool:
        first = pool.submit(_open_folder, folder, "First")
        second = pool.submit(_open_folder, folder, "Second")
        opened = [first.result(timeout = 10), second.result(timeout = 10)]

    assert opened[0].id == opened[1].id
    stored = studio_db.list_chat_projects(include_archived = True)
    assert len(stored) == 1
    assert stored[0]["id"] == opened[0].id


def test_reopen_moved_folder_reuses_persisted_identity(tmp_path):
    original = tmp_path / "original-repository"
    moved = tmp_path / "moved-repository"
    original.mkdir()
    created = _open_folder(original, "Original name")

    original.rename(moved)
    reopened = _open_folder(moved, "Replacement name")

    assert reopened.id == created.id
    assert reopened.name == "Original name"
    stored = studio_db.get_chat_project(created.id)
    assert stored is not None
    assert stored["rootPath"] == str(moved.resolve())
    assert stored["workspaceAvailable"] is True
    assert len(studio_db.list_chat_projects(True)) == 1


def test_folder_workspace_rejects_missing_and_read_only_roots_without_a_row(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.upsert_chat_project(_folder_project("missing", tmp_path, rootPath = str(missing)))
    assert studio_db.get_chat_project("missing") is None

    read_only = tmp_path / "read-only"
    read_only.mkdir()
    real_access = os.access
    monkeypatch.setattr(
        studio_db.os,
        "access",
        lambda path, mode: False if Path(path) == read_only else real_access(path, mode),
    )
    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.upsert_chat_project(_folder_project("read-only", read_only))
    assert studio_db.get_chat_project("read-only") is None


def test_folder_workspace_rejects_persistence_without_native_identity(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = _folder_project("missing-identity", folder)
    project.pop("workspaceDeviceId")
    project.pop("workspaceFileId")

    with pytest.raises(studio_db.ProjectWorkspaceError, match = "identity"):
        studio_db.upsert_chat_project(project)

    assert studio_db.get_chat_project("missing-identity") is None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "symlinks are unavailable")
def test_folder_workspace_rejects_symlink_selected_roots(tmp_path):
    target = tmp_path / "target"
    link = tmp_path / "link"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_path(_sign_folder(link))

    assert caught.value.status_code == 400
    assert studio_db.list_chat_projects(True) == []


def test_folder_workspace_persists_as_the_project_tool_cwd(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    stored = studio_db.upsert_chat_project(_folder_project("folder-cwd", folder))

    assert stored["rootPath"] == str(folder.resolve())
    assert stored["sandboxPath"] == str(folder.resolve())
    assert tools._get_project_workdir(tools.project_session_id(stored["id"])) == str(
        folder.resolve()
    )

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    reopened = studio_db.get_chat_project(stored["id"])
    assert reopened is not None
    assert reopened["workspaceKind"] == "folder"
    assert reopened["rootPath"] == str(folder.resolve())
    assert reopened["sandboxPath"] == str(folder.resolve())


def test_project_session_cache_rebinds_to_the_current_workspace_identity(tmp_path):
    first_root = tmp_path / "first-repository"
    second_root = tmp_path / "second-repository"
    first_root.mkdir()
    second_root.mkdir()
    project_id = "reused-project-id"
    session_id = tools.project_session_id(project_id)
    studio_db.upsert_chat_project(_folder_project(project_id, first_root))
    assert tools._get_workdir(session_id) == str(first_root.resolve())

    studio_db.delete_chat_project(project_id)
    studio_db.upsert_chat_project(_folder_project(project_id, second_root))

    assert tools._get_workdir(session_id) == str(second_root.resolve())
    result = tools.execute_tool(
        "edit_file",
        {"path": "current.txt", "old_string": "", "new_string": "current"},
        session_id = session_id,
        disable_sandbox = True,
    )
    assert not result.startswith("Error:")
    assert not (first_root / "current.txt").exists()
    assert (second_root / "current.txt").read_text(encoding = "utf-8") == "current"


def test_folder_project_terminal_cannot_escape_even_in_full_access(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-terminal", folder))
    relative_escape = tmp_path / "relative-escape.txt"
    absolute_escape = tmp_path / "absolute-escape.txt"
    command = (
        "(printf escaped > ../relative-escape.txt) 2>/dev/null || true; "
        f"(printf escaped > {shlex.quote(str(absolute_escape))}) 2>/dev/null || true; "
        "printf confined > inside.txt"
    )

    result = tools._bash_exec(
        command,
        session_id = tools.project_session_id(project["id"]),
        disable_sandbox = True,
    )

    assert not relative_escape.exists()
    assert not absolute_escape.exists()
    status = execution_boundary_status()
    if not status.available:
        assert result.startswith("Execution error:")
        assert not (folder / "inside.txt").exists()
    else:
        assert (folder / "inside.txt").read_text(encoding = "utf-8") == "confined"
        assert "Execution error" not in result


def test_folder_project_python_cannot_escape_even_in_full_access(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-python", folder))
    outside = tmp_path / "python-escape.txt"
    code = (
        "from pathlib import Path\n"
        "try:\n"
        "    Path('../python-escape.txt').write_text('escaped')\n"
        "except OSError:\n"
        "    pass\n"
        "Path('inside-python.txt').write_text('confined')\n"
    )

    result = tools._python_exec(
        code,
        session_id = tools.project_session_id(project["id"]),
        disable_sandbox = True,
    )

    assert not outside.exists()
    status = execution_boundary_status()
    if not status.available:
        assert result.startswith("Execution error:")
        assert not (folder / "inside-python.txt").exists()
    else:
        assert (folder / "inside-python.txt").read_text(encoding = "utf-8") == "confined"
        assert "Execution error" not in result


@pytest.mark.parametrize("disable_sandbox", [False, True], ids = ["sandboxed", "bypass"])
@pytest.mark.parametrize("path_kind", ["absolute", "traversal"])
def test_folder_project_edit_file_never_escapes_workspace(tmp_path, disable_sandbox, path_kind):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-edit", folder))
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding = "utf-8")
    path = str(outside) if path_kind == "absolute" else "../outside.txt"

    result = tools.execute_tool(
        "edit_file",
        {"path": path, "old_string": "keep", "new_string": "escaped"},
        session_id = tools.project_session_id(project["id"]),
        disable_sandbox = disable_sandbox,
    )

    assert result.startswith("Error:")
    assert outside.read_text(encoding = "utf-8") == "keep"


@pytest.mark.parametrize("disable_sandbox", [False, True], ids = ["sandboxed", "bypass"])
def test_folder_project_edit_file_writes_inside_workspace(tmp_path, disable_sandbox):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-edit-inside", folder))
    session_id = tools.project_session_id(project["id"])

    created = tools.execute_tool(
        "edit_file",
        {"path": "src/app.py", "old_string": "", "new_string": "value = 1\n"},
        session_id = session_id,
        disable_sandbox = disable_sandbox,
    )
    edited = tools.execute_tool(
        "edit_file",
        {"path": "src/app.py", "old_string": "value = 1", "new_string": "value = 2"},
        session_id = session_id,
        disable_sandbox = disable_sandbox,
    )

    assert created.startswith("Created")
    assert edited.startswith("Edited")
    assert (folder / "src" / "app.py").read_text(encoding = "utf-8") == "value = 2\n"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "symlinks are unavailable")
@pytest.mark.parametrize("disable_sandbox", [False, True], ids = ["sandboxed", "bypass"])
def test_folder_project_edit_file_rejects_parent_symlink_swap(
    tmp_path, monkeypatch, disable_sandbox
):
    folder = tmp_path / "repository"
    nested = folder / "nested"
    displaced = folder / "displaced"
    outside = tmp_path / "outside"
    nested.mkdir(parents = True)
    outside.mkdir()
    (nested / "target.txt").write_text("inside", encoding = "utf-8")
    outside_target = outside / "target.txt"
    outside_target.write_text("outside", encoding = "utf-8")
    project = studio_db.upsert_chat_project(_folder_project("folder-edit-race", folder))
    real_open = tools.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "nested" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            nested.rename(displaced)
            nested.symlink_to(outside, target_is_directory = True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tools.os, "open", racing_open)
    result = tools.execute_tool(
        "edit_file",
        {"path": "nested/target.txt", "old_string": "inside", "new_string": "changed"},
        session_id = tools.project_session_id(project["id"]),
        disable_sandbox = disable_sandbox,
    )

    assert swapped is True, result
    assert result.startswith("Error:")
    assert outside_target.read_text(encoding = "utf-8") == "outside"
    assert (displaced / "target.txt").read_text(encoding = "utf-8") == "inside"


@pytest.mark.parametrize("replace", [False, True], ids = ["missing", "replaced"])
def test_unavailable_folder_project_tool_fails_without_managed_fallback(tmp_path, replace):
    folder = tmp_path / "repository"
    original = tmp_path / "original-repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-unavailable", folder))
    folder.rename(original)
    if replace:
        folder.mkdir()

    result = tools._bash_exec(
        "printf should-not-run > tool-ran.txt",
        session_id = tools.project_session_id(project["id"]),
        disable_sandbox = True,
    )

    assert "unavailable" in result.lower()
    assert not (original / "tool-ran.txt").exists()
    assert not (folder / "tool-ran.txt").exists()
    assert not any(
        candidate.name == "tool-ran.txt"
        for candidate in (tmp_path / "studio").rglob("tool-ran.txt")
    )


def test_persisted_folder_project_rejects_path_identity_replacement(tmp_path):
    folder = tmp_path / "repository"
    original = tmp_path / "original-repository"
    folder.mkdir()
    project = studio_db.upsert_chat_project(_folder_project("folder-identity", folder))

    folder.rename(original)
    folder.mkdir()

    listed = studio_db.get_chat_project(project["id"])
    assert listed is not None
    assert listed["workspaceAvailable"] is False
    with pytest.raises(studio_db.ProjectWorkspaceError, match = "identity"):
        studio_db.ensure_chat_project_workspace(project["id"])


@pytest.mark.parametrize("delete_files", [False, True])
def test_deleting_folder_project_never_deletes_or_orphan_registers_repository(
    tmp_path, monkeypatch, delete_files
):
    folder = tmp_path / "repository"
    folder.mkdir()
    sentinel = folder / "user-work.txt"
    sentinel.write_text("preserve me", encoding = "utf-8")
    project = studio_db.upsert_chat_project(_folder_project("folder-delete", folder))
    orphan_calls = []
    workspace_delete_calls = []

    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(
        tools,
        "record_orphaned_project",
        lambda *args, **kwargs: orphan_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        studio_db,
        "delete_project_workspace",
        lambda payload: workspace_delete_calls.append(payload),
    )

    deleted = asyncio.run(
        chat_history.delete_project(
            project["id"],
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
            delete_files = delete_files,
            current_subject = "tester",
        )
    )

    assert deleted.id == project["id"]
    assert studio_db.get_chat_project(project["id"]) is None
    assert sentinel.read_text(encoding = "utf-8") == "preserve me"
    assert orphan_calls == []
    assert workspace_delete_calls == []


def test_storage_delete_files_flag_never_removes_a_folder_workspace(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    sentinel = folder / "user-work.txt"
    sentinel.write_text("preserve me", encoding = "utf-8")
    project = studio_db.upsert_chat_project(_folder_project("folder-storage-delete", folder))

    deleted = studio_db.delete_chat_project(project["id"], delete_files = True)

    assert deleted is not None
    assert deleted["id"] == project["id"]
    assert sentinel.read_text(encoding = "utf-8") == "preserve me"


def test_managed_workspace_delete_requires_and_uses_persisted_root_identity(monkeypatch):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    project = studio_db.upsert_chat_project(_managed_project("managed-delete"))
    root = Path(project["rootPath"])
    (root / "owned.txt").write_text("owned", encoding = "utf-8")

    assert project["managedRootDeviceId"] == str(root.stat().st_dev)
    assert project["managedRootFileId"] == str(root.stat().st_ino)
    sandbox = Path(project["sandboxPath"])
    assert project["workspaceDeviceId"] == str(sandbox.stat().st_dev)
    assert project["workspaceFileId"] == str(sandbox.stat().st_ino)
    is_project, edit_identity = tools._project_edit_scope(
        f"{tools._PROJECT_SESSION_PREFIX}{project['id']}"
    )
    assert is_project is True
    assert edit_identity == (sandbox.stat().st_dev, sandbox.stat().st_ino)

    without_identity = dict(project)
    without_identity.pop("managedRootDeviceId")
    without_identity.pop("managedRootFileId")
    studio_db.delete_project_workspace(without_identity)
    assert root.is_dir()

    studio_db.delete_project_workspace(project)
    assert not root.exists()


def test_managed_project_ids_with_the_same_legacy_prefix_get_distinct_roots():
    first = studio_db.upsert_chat_project(_managed_project("abcdefgh-first"), create_only = True)
    second = studio_db.upsert_chat_project(_managed_project("abcdefgh-second"), create_only = True)

    assert first["rootPath"] != second["rootPath"]
    assert Path(first["rootPath"]).is_dir()
    assert Path(second["rootPath"]).is_dir()


def test_managed_create_atomically_rejects_an_overlapping_root(monkeypatch):
    shared = Path(os.environ["UNSLOTH_STUDIO_PROJECTS_HOME"]) / "Managed-abcdefgh"
    monkeypatch.setattr(studio_db, "_default_project_root", lambda _project: str(shared))

    first = studio_db.upsert_chat_project(_managed_project("abcdefgh-first"), create_only = True)
    with pytest.raises(studio_db.ProjectWorkspaceOverlapError):
        studio_db.upsert_chat_project(_managed_project("abcdefgh-second"), create_only = True)

    assert studio_db.get_chat_project(first["id"]) is not None
    assert studio_db.get_chat_project("abcdefgh-second") is None


def test_managed_delete_refuses_a_root_referenced_by_another_live_project(monkeypatch):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    first = studio_db.upsert_chat_project(_managed_project("root-owner-one"))
    second = studio_db.upsert_chat_project(_managed_project("root-owner-two"))
    root = Path(first["rootPath"])
    sentinel = root / "preserve.txt"
    sentinel.write_text("preserve", encoding = "utf-8")
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = ? WHERE id = ?",
            (str(root), second["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    studio_db.delete_project_workspace(first)

    assert root.is_dir()
    assert sentinel.read_text(encoding = "utf-8") == "preserve"


@pytest.mark.parametrize(
    ("old_id", "new_id", "legacy_collision"),
    [
        ("managed-recreate-race", "managed-recreate-race", False),
        ("abcdefgh-first", "abcdefgh-second", True),
    ],
)
def test_managed_delete_serializes_recreation_after_the_final_check(
    tmp_path, monkeypatch, old_id, new_id, legacy_collision
):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    if legacy_collision:
        shared = Path(os.environ["UNSLOTH_STUDIO_PROJECTS_HOME"]) / "Managed-abcdefgh"
        monkeypatch.setattr(
            studio_db,
            "_default_project_root",
            lambda _project: str(shared),
        )
    project = studio_db.upsert_chat_project(_managed_project(old_id), create_only = True)
    old_root = Path(project["rootPath"])
    old_marker = old_root / "old.txt"
    old_marker.write_text("old", encoding = "utf-8")

    after_final_check = threading.Event()
    allow_workspace_delete = threading.Event()
    creator_started = threading.Event()
    creator_completed = threading.Event()
    delete_results = []
    delete_errors = []
    create_errors = []
    real_delete_workspace = studio_db.delete_project_workspace

    def blocked_workspace_delete(payload):
        # delete_project_workspace is entered only after the route's last live
        # row check. The recreating POST begins inside this barrier.
        after_final_check.set()
        assert allow_workspace_delete.wait(timeout = 5)
        return real_delete_workspace(payload)

    monkeypatch.setattr(studio_db, "delete_project_workspace", blocked_workspace_delete)
    monkeypatch.setattr(studio_db, "sandbox_is_referenced_elsewhere", lambda *_a: False)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_a: None)
    monkeypatch.setattr(
        chat_history,
        "_remove_conversation_archives",
        lambda *_a, **_k: None,
    )

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)

    def run_delete():
        try:
            delete_results.append(
                asyncio.run(
                    chat_history.delete_project(
                        old_id,
                        SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
                        delete_files = True,
                        current_subject = "tester",
                    )
                )
            )
        except BaseException as exc:
            delete_errors.append(exc)

    def run_create():
        creator_started.set()
        try:
            chat_history.save_project(
                chat_history.ChatProjectCreate(
                    id = new_id,
                    name = "Managed",
                    archived = False,
                    createdAt = 1_700_000_000_100,
                    updatedAt = 1_700_000_000_100,
                ),
                current_subject = "tester",
            )
            recreated = studio_db.get_chat_project(new_id)
            assert recreated is not None
            (Path(recreated["rootPath"]) / "new.txt").write_text("new", encoding = "utf-8")
        except BaseException as exc:
            create_errors.append(exc)
        finally:
            creator_completed.set()

    delete_thread = threading.Thread(target = run_delete)
    create_thread = threading.Thread(target = run_create)
    delete_thread.start()
    assert after_final_check.wait(timeout = 5)
    create_thread.start()
    assert creator_started.wait(timeout = 2)
    assert not creator_completed.wait(timeout = 0.2)
    assert studio_db.get_chat_project(new_id) is None

    allow_workspace_delete.set()
    delete_thread.join(timeout = 5)
    create_thread.join(timeout = 5)

    assert not delete_thread.is_alive()
    assert not create_thread.is_alive()
    assert delete_errors == []
    assert create_errors == []
    assert delete_results[0].id == old_id
    recreated = studio_db.get_chat_project(new_id)
    assert recreated is not None
    recreated_root = Path(recreated["rootPath"])
    assert recreated_root == old_root
    assert not old_marker.exists()
    assert (recreated_root / "new.txt").read_text(encoding = "utf-8") == "new"


def test_managed_workspace_rejects_a_replaced_sandbox(tmp_path):
    project = studio_db.upsert_chat_project(_managed_project("managed-sandbox-swap"))
    sandbox = Path(project["sandboxPath"])
    displaced = tmp_path / "original-sandbox"
    sandbox.rename(displaced)
    sandbox.mkdir()

    with pytest.raises(AgentWorkspaceError, match = "unavailable"):
        project_workspace(project["id"])

    assert displaced.is_dir()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "symlinks are unavailable")
def test_replaced_managed_root_symlink_never_deletes_its_target(tmp_path):
    project = studio_db.upsert_chat_project(_managed_project("managed-symlink"))
    root = Path(project["rootPath"])
    displaced = tmp_path / "original-managed-root"
    suffix = "managed"
    target = tmp_path / f"victim-{suffix}"
    target.mkdir()
    sentinel = target / "user-data.txt"
    sentinel.write_text("preserve", encoding = "utf-8")
    root.rename(displaced)
    try:
        root.symlink_to(target, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    studio_db.delete_project_workspace(project)

    assert root.is_symlink()
    assert displaced.is_dir()
    assert sentinel.read_text(encoding = "utf-8") == "preserve"


def test_managed_delete_rechecks_the_entry_moved_after_a_forced_symlink_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    project = studio_db.upsert_chat_project(_managed_project("managed-race"))
    root = Path(project["rootPath"])
    displaced = tmp_path / "displaced-owned-root"
    victim = tmp_path / "victim-managed"
    victim.mkdir()
    sentinel = victim / "user-data.txt"
    sentinel.write_text("preserve", encoding = "utf-8")
    real_rename = Path.rename
    swapped = False

    def racing_rename(source: Path, target: Path):
        nonlocal swapped
        if source == root and not swapped:
            swapped = True
            real_rename(source, displaced)
            source.symlink_to(victim, target_is_directory = True)
        return real_rename(source, target)

    monkeypatch.setattr(Path, "rename", racing_rename)

    studio_db.delete_project_workspace(project)

    assert swapped is True
    assert displaced.is_dir()
    assert root.is_symlink()
    assert sentinel.read_text(encoding = "utf-8") == "preserve"


def test_deleted_managed_project_does_not_invent_a_missing_persisted_identity(tmp_path):
    project_id = "legacy-managed"
    root = tmp_path / "Legacy-legacy-m"
    sandbox = root / "sandbox"
    sandbox.mkdir(parents = True)

    tools.record_orphaned_project(
        project_id,
        str(sandbox),
        True,
        str(root),
        None,
        None,
    )

    record = tools._read_orphan_record(tools._ORPHAN_PROJECT, project_id)
    assert record is not None
    assert record["managedRootDeviceId"] is None
    assert record["managedRootFileId"] is None


def test_project_delete_fences_tool_entry_before_idle_wait_and_through_workspace_delete(
    monkeypatch,
):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    project = studio_db.upsert_chat_project(_managed_project("delete-tool-race"))
    session_id = tools.project_session_id(project["id"])
    delete_entered = threading.Event()
    allow_delete = threading.Event()
    tool_paused = threading.Event()
    allow_tool = threading.Event()
    route_result = []
    route_errors = []
    tool_result = []
    real_delete_workspace = studio_db.delete_project_workspace

    def blocked_workspace_delete(payload):
        delete_entered.set()
        assert allow_delete.wait(timeout = 5)
        return real_delete_workspace(payload)

    monkeypatch.setattr(studio_db, "delete_project_workspace", blocked_workspace_delete)
    monkeypatch.setattr(studio_db, "sandbox_is_referenced_elsewhere", lambda *_a: False)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_a: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)

    def delayed_tool_entry():
        tool_paused.set()
        assert allow_tool.wait(timeout = 5)
        try:
            with tools._session_in_flight(session_id):
                tool_result.append("entered")
        except tools.ProjectSessionDeleting:
            tool_result.append("fenced")
        try:
            tools._get_project_workdir(session_id)
        except tools.ProjectSessionDeleting:
            tool_result.append("resolver-fenced")

    def run_delete():
        try:
            route_result.append(
                asyncio.run(
                    chat_history.delete_project(
                        project["id"],
                        SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
                        delete_files = True,
                        current_subject = "tester",
                    )
                )
            )
        except BaseException as exc:
            route_errors.append(exc)

    tool_thread = threading.Thread(target = delayed_tool_entry)
    delete_thread = threading.Thread(target = run_delete)
    tool_thread.start()
    assert tool_paused.wait(timeout = 2)
    delete_thread.start()
    assert delete_entered.wait(timeout = 5)
    allow_tool.set()
    tool_thread.join(timeout = 2)
    assert not tool_thread.is_alive()
    assert tool_result == ["fenced", "resolver-fenced"]
    assert tools._project_session_deletion_fenced(session_id) is True

    allow_delete.set()
    delete_thread.join(timeout = 5)

    assert not delete_thread.is_alive()
    assert route_errors == []
    assert route_result[0].id == project["id"]
    assert tools._project_session_deletion_fenced(session_id) is False
    assert not Path(project["rootPath"]).exists()


def test_project_delete_releases_tool_fence_on_not_found():
    project_id = "missing-project-fence"
    session_id = tools.project_session_id(project_id)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            chat_history.delete_project(
                project_id,
                SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
                current_subject = "tester",
            )
        )

    assert caught.value.status_code == 404
    assert tools._project_session_deletion_fenced(session_id) is False
    with tools._session_in_flight(session_id):
        pass


def test_project_delete_releases_tool_fence_when_the_route_task_is_cancelled():
    project_id = "cancelled-project-delete"
    session_id = tools.project_session_id(project_id)

    async def scenario():
        entered = asyncio.Event()

        async def held_delete(_project_id: str):
            entered.set()
            await asyncio.Event().wait()

        fenced_delete = chat_history._fence_project_tool_session(held_delete)
        task = asyncio.create_task(fenced_delete(project_id))
        await entered.wait()
        assert tools._project_session_deletion_fenced(session_id) is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert tools._project_session_deletion_fenced(session_id) is False


def test_folder_identity_rejects_a_second_project_with_a_forged_identity(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = studio_db.upsert_chat_project(_folder_project("identity-one", first_root))
    duplicate = _folder_project(
        "identity-two",
        second_root,
        workspaceDeviceId = first["workspaceDeviceId"],
        workspaceFileId = first["workspaceFileId"],
    )

    with pytest.raises(studio_db.ProjectWorkspaceError, match = "identity"):
        studio_db.upsert_chat_project(duplicate)

    assert [project["id"] for project in studio_db.list_chat_projects(True)] == ["identity-one"]


def test_goal_state_persists_and_remains_project_scoped(tmp_path, monkeypatch):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = studio_db.upsert_chat_project(_folder_project("goal-one", first_root))
    second = studio_db.upsert_chat_project(_folder_project("goal-two", second_root))

    updated = chat_history.patch_project(
        first["id"],
        chat_history.ChatProjectPatch(
            goal = "  Ship folder projects  ",
            goalUpdatedAt = 1_700_000_000_500,
        ),
        current_subject = "tester",
    )

    assert updated.goal == "Ship folder projects"
    assert updated.goalStatus == "active"
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    assert studio_db.get_chat_project(first["id"])["goal"] == "Ship folder projects"
    assert studio_db.get_chat_project(first["id"])["goalStatus"] == "active"
    assert studio_db.get_chat_project(second["id"])["goal"] is None
    assert studio_db.get_chat_project(second["id"])["goalStatus"] is None


def test_project_patch_authoritatively_gates_goal_completion(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    studio_db.upsert_chat_project(
        _folder_project(
            "goal-gated",
            root,
            goal = "Ship the workspace",
            goalStatus = "active",
            goalUpdatedAt = 7,
        )
    )
    check = {
        "name": "test",
        "kind": "test",
        "command": "test-command",
        "required": True,
        "timeoutSeconds": 10,
        "logLimitBytes": 1024,
    }
    config = set_verification_config("goal-gated", [check], require_for_goal_completion = True)
    client = _project_api_client()

    blocked = client.patch(
        "/api/chat/projects/goal-gated",
        json = {"goalStatus": "completed", "goalUpdatedAt": 8},
    )

    assert blocked.status_code == 409
    assert blocked.json() == {"detail": GOAL_COMPLETION_VERIFICATION_DETAIL}
    assert str(root) not in blocked.text
    assert studio_db.get_chat_project("goal-gated")["goalStatus"] == "active"

    fingerprint = workspace_fingerprint(root)
    run = begin_verification_run(
        "goal-gated",
        fingerprint,
        config_revision = config["revision"],
    )
    finish_verification_run(
        run["id"],
        "passed",
        fingerprint,
        [{"name": "test", "required": True, "status": "passed"}],
    )

    completed = client.patch(
        "/api/chat/projects/goal-gated",
        json = {"goalStatus": "completed", "goalUpdatedAt": 9},
    )

    assert completed.status_code == 200
    assert completed.json()["goalStatus"] == "completed"


def test_artifact_diffing_is_disabled_only_for_folder_project_workspaces(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    folder_project = studio_db.upsert_chat_project(_folder_project("folder-artifacts", folder))
    managed_project = studio_db.upsert_chat_project(_managed_project("managed-artifacts"))

    assert (
        tools._tracks_workspace_artifacts(tools.project_session_id(folder_project["id"])) is False
    )
    assert (
        tools._tracks_workspace_artifacts(tools.project_session_id(managed_project["id"])) is True
    )
    assert tools._tracks_workspace_artifacts("ordinary-chat") is True

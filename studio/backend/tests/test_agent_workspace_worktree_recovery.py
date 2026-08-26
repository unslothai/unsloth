# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import sqlite3
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace import state as workspace_state
from core.agent_workspace import worktrees as worktree_service
from routes import agent_workspace as agent_workspace_routes
from routes import chat_history as chat_history_routes
from storage import studio_db


def _git(
    root: Path,
    *args: str,
    check: bool = True,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd = root,
        check = check,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "base")


def _folder_project(root: Path, project_id: str = "project") -> None:
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _setup(tmp_path: Path, monkeypatch) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    return repository


def _record(project_id: str = "project") -> dict:
    records = workspace_state.list_worktrees(project_id)
    assert len(records) == 1
    return records[0]


def test_create_db_reservation_failure_never_mutates_git(tmp_path, monkeypatch):
    repository = _setup(tmp_path, monkeypatch)
    hidden = tmp_path / "private" / "studio.db"
    with monkeypatch.context() as patch:
        patch.setattr(
            worktree_service,
            "save_worktree",
            lambda _record: (_ for _ in ()).throw(sqlite3.OperationalError(str(hidden))),
        )
        with pytest.raises(AgentWorkspaceError, match = "reserve durable state") as exc_info:
            worktree_service.create_worktree("project")

    assert str(hidden) not in str(exc_info.value)
    assert _git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert _git(repository, "branch", "--list", "unsloth-studio/*") == ""
    assert workspace_state.list_worktrees("project") == []


def test_create_ambiguous_db_commit_is_settled_without_git_mutation(tmp_path, monkeypatch):
    repository = _setup(tmp_path, monkeypatch)
    real_save = worktree_service.save_worktree

    def commit_then_fail(record):
        real_save(record)
        raise sqlite3.OperationalError("ambiguous commit acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "save_worktree", commit_then_fail)
        with pytest.raises(AgentWorkspaceError, match = "reserve durable state"):
            worktree_service.create_worktree("project")

    assert _record()["status"] == "creating"
    assert _git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1
    result = worktree_service.reconcile_worktrees_on_startup()
    assert result["removed"] == 1
    assert _record()["status"] == "removed"


def test_git_add_failure_settles_empty_creating_record_without_deleting_unknown_files(
    tmp_path, monkeypatch
):
    repository = _setup(tmp_path, monkeypatch)
    real_git = worktree_service._git

    def fail_add(root, args, **kwargs):
        if args[:2] == ["worktree", "add"]:
            raise AgentWorkspaceError("injected worktree add failure")
        return real_git(root, args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "_git", fail_add)
        with pytest.raises(AgentWorkspaceError, match = "injected"):
            worktree_service.create_worktree("project")

    record = _record()
    assert record["status"] == "removed"
    assert not Path(record["path"]).exists()
    assert not Path(record["markerPath"]).exists()
    assert _git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_startup_settles_crash_after_db_reservation_before_git_add(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    real_git = worktree_service._git

    def fail_add(root, args, **kwargs):
        if args[:2] == ["worktree", "add"]:
            raise AgentWorkspaceError("simulated process loss")
        return real_git(root, args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "_git", fail_add)
        patch.setattr(worktree_service, "_settle_failed_creation", lambda *_args: None)
        with pytest.raises(AgentWorkspaceError, match = "simulated"):
            worktree_service.create_worktree("project")

    record = _record()
    assert record["status"] == "creating"
    assert Path(record["path"]).parent.is_dir()

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["removed"] == 1
    assert _record()["status"] == "removed"
    assert not Path(record["path"]).parent.exists()


def test_marker_publish_failure_preserves_unproven_checkout_for_manual_recovery(
    tmp_path, monkeypatch
):
    repository = _setup(tmp_path, monkeypatch)
    hidden = tmp_path / "private" / "owner.json"
    with monkeypatch.context() as patch:
        patch.setattr(
            worktree_service,
            "_write_marker",
            lambda _path, _payload: (_ for _ in ()).throw(OSError(str(hidden))),
        )
        with pytest.raises(AgentWorkspaceError, match = "preserved any checkout") as exc_info:
            worktree_service.create_worktree("project")

    record = _record()
    target = Path(record["path"])
    assert str(hidden) not in str(exc_info.value)
    assert record["status"] == "needs_attention"
    assert target.is_dir()
    assert not Path(record["markerPath"]).exists()
    assert str(target) in _git(repository, "worktree", "list", "--porcelain")
    worktree_service.reconcile_worktrees_on_startup()
    assert _record()["status"] == "needs_attention"
    assert target.is_dir()
    with pytest.raises(AgentWorkspaceError, match = "cannot prove"):
        worktree_service.cleanup_worktree("project", record["id"])


def test_create_final_db_failure_is_activated_from_durable_marker_on_startup(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    real_transition = worktree_service.transition_worktree_status
    hidden = tmp_path / "private" / "studio.db"

    def fail_activation(worktree_id, expected, status):
        if status == "active":
            raise sqlite3.OperationalError(str(hidden))
        return real_transition(worktree_id, expected, status)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "transition_worktree_status", fail_activation)
        with pytest.raises(AgentWorkspaceError, match = "startup recovery") as exc_info:
            worktree_service.create_worktree("project")

    record = _record()
    assert str(hidden) not in str(exc_info.value)
    assert record["status"] == "creating"
    assert Path(record["path"]).is_dir()
    assert Path(record["markerPath"]).is_file()

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["activated"] == 1
    assert _record()["status"] == "active"


def test_startup_retains_orphan_marker_when_database_ownership_was_lost(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    conn = workspace_state.connection()
    try:
        conn.execute("DELETE FROM agent_worktrees WHERE id = ?", (created["id"],))
        conn.commit()
    finally:
        conn.close()
    assert workspace_state.get_worktree(created["id"]) is None

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["attention"] == 1
    assert result["imported"] == 0
    assert workspace_state.get_worktree(created["id"]) is None
    assert Path(created["path"]).is_dir()
    assert Path(created["markerPath"]).is_file()


def test_cleanup_db_reservation_failure_leaves_active_worktree_untouched(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    real_transition = worktree_service.transition_worktree_status

    def fail_removing(worktree_id, expected, status):
        if status == "removing":
            raise sqlite3.OperationalError("/private/hidden/studio.db")
        return real_transition(worktree_id, expected, status)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "transition_worktree_status", fail_removing)
        with pytest.raises(AgentWorkspaceError, match = "reserve durable state"):
            worktree_service.cleanup_worktree("project", created["id"])

    assert _record()["status"] == "active"
    assert Path(created["path"]).is_dir()
    assert Path(created["markerPath"]).is_file()


def test_cleanup_ambiguous_removing_commit_recovers_active_without_running_git(
    tmp_path, monkeypatch
):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    real_transition = worktree_service.transition_worktree_status

    def commit_then_fail(worktree_id, expected, status):
        result = real_transition(worktree_id, expected, status)
        if status == "removing":
            raise sqlite3.OperationalError("ambiguous commit acknowledgement")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "transition_worktree_status", commit_then_fail)
        with pytest.raises(AgentWorkspaceError, match = "reserve durable state"):
            worktree_service.cleanup_worktree("project", created["id"])

    assert _record()["status"] == "removing"
    assert Path(created["path"]).is_dir()
    result = worktree_service.reconcile_worktrees_on_startup()
    assert result["activated"] == 1
    assert _record()["status"] == "active"


def test_dirty_cleanup_fails_closed_and_restores_active_state(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    target = Path(created["path"])
    (target / "dirty.txt").write_text("keep me\n", encoding = "utf-8")

    with pytest.raises(AgentWorkspaceError):
        worktree_service.cleanup_worktree("project", created["id"])

    assert _record()["status"] == "active"
    assert (target / "dirty.txt").read_text(encoding = "utf-8") == "keep me\n"
    assert Path(created["markerPath"]).is_file()


def test_cleanup_final_db_failure_is_completed_from_marker_on_startup(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    real_transition = worktree_service.transition_worktree_status

    def fail_removed(worktree_id, expected, status):
        if status == "removed":
            raise sqlite3.OperationalError("/private/hidden/studio.db")
        return real_transition(worktree_id, expected, status)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "transition_worktree_status", fail_removed)
        with pytest.raises(AgentWorkspaceError, match = "pending startup recovery"):
            worktree_service.cleanup_worktree("project", created["id"])

    assert _record()["status"] == "removing"
    assert not Path(created["path"]).exists()
    assert Path(created["markerPath"]).is_file()

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["removed"] == 1
    assert _record()["status"] == "removed"
    assert not Path(created["markerPath"]).exists()


def test_marker_unlink_failure_occurs_after_removed_state_and_retries_on_startup(
    tmp_path, monkeypatch
):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    marker = Path(created["markerPath"])
    real_unlink = Path.unlink

    def fail_marker_unlink(path, *args, **kwargs):
        if path == marker:
            raise OSError("injected marker cleanup failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_marker_unlink)
        removed = worktree_service.cleanup_worktree("project", created["id"])

    assert removed["status"] == "removed"
    assert marker.is_file()
    result = worktree_service.reconcile_worktrees_on_startup()
    assert result["removed"] == 1
    assert not marker.exists()


def test_tampered_marker_and_branch_registration_are_never_removed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    marker_path = Path(created["markerPath"])
    marker = json.loads(marker_path.read_text(encoding = "utf-8"))
    marker["token"] = "x" * 43
    marker_path.write_text(json.dumps(marker), encoding = "utf-8")

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["attention"] == 1
    assert _record()["status"] == "needs_attention"
    assert Path(created["path"]).is_dir()
    with pytest.raises(AgentWorkspaceError, match = "cannot prove"):
        worktree_service.cleanup_worktree("project", created["id"])


def test_foreign_branch_in_owned_path_fails_closed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    target = Path(created["path"])
    _git(target, "checkout", "-qb", "user-owned-branch")

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["attention"] == 1
    assert _record()["status"] == "needs_attention"
    assert target.is_dir()
    with pytest.raises(AgentWorkspaceError, match = "cannot prove"):
        worktree_service.cleanup_worktree("project", created["id"])


def test_removing_state_with_registered_checkout_recovers_active_without_retrying_delete(
    tmp_path, monkeypatch
):
    _setup(tmp_path, monkeypatch)
    created = worktree_service.create_worktree("project")
    target = Path(created["path"])
    (target / "user-work.txt").write_text("preserve\n", encoding = "utf-8")
    workspace_state.transition_worktree_status(created["id"], {"active"}, "removing")

    result = worktree_service.reconcile_worktrees_on_startup()

    assert result["activated"] == 1
    assert _record()["status"] == "active"
    assert (target / "user-work.txt").read_text(encoding = "utf-8") == "preserve\n"


def test_project_deletion_fence_rejects_new_worktree_operations(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    worktree_service.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "being deleted"):
            worktree_service.create_worktree("project")
    finally:
        worktree_service.finish_project_deletion("project")

    created = worktree_service.create_worktree("project")
    worktree_service.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "being deleted"):
            worktree_service.cleanup_worktree("project", created["id"])
    finally:
        worktree_service.finish_project_deletion("project")
    assert Path(created["path"]).is_dir()


def test_preplanted_container_symlink_is_preserved_and_never_traversed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    fixed_id = "12345678-1234-4234-8234-123456789abc"
    storage = tmp_path / "studio-projects" / ".agent-worktrees"
    bucket = storage / worktree_service._project_storage_key("project")
    bucket.mkdir(parents = True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "sentinel.txt"
    sentinel.write_text("user data\n", encoding = "utf-8")
    (bucket / fixed_id).symlink_to(foreign, target_is_directory = True)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service.uuid, "uuid4", lambda: __import__("uuid").UUID(fixed_id))
        with pytest.raises(AgentWorkspaceError, match = "destination already exists"):
            worktree_service.create_worktree("project")

    assert sentinel.read_text(encoding = "utf-8") == "user data\n"
    assert (bucket / fixed_id).is_symlink()
    assert _record()["status"] in {"removed", "needs_attention"}


@pytest.mark.skipif(os.name == "nt", reason = "POSIX hook fixture")
def test_studio_worktree_creation_disables_repository_hooks_without_changing_user_git(
    tmp_path, monkeypatch
):
    repository = _setup(tmp_path, monkeypatch)
    hook_result = tmp_path / "post-checkout-ran"
    hooks = repository / ".git" / "hooks"
    hooks.mkdir(exist_ok = True)
    post_checkout = hooks / "post-checkout"
    post_checkout.write_text(
        "#!/bin/sh\nprintf hook > " + str(hook_result) + "\n",
        encoding = "utf-8",
    )
    post_checkout.chmod(0o700)

    created = worktree_service.create_worktree("project")

    assert not hook_result.exists()
    worktree_service.cleanup_worktree("project", created["id"])
    assert not hook_result.exists()

    _git(repository, "checkout", "-qb", "ordinary-user-checkout")
    assert hook_result.read_text(encoding = "utf-8") == "hook"


def test_worktree_api_hides_paths_from_unexpected_storage_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    hidden = tmp_path / "private" / "studio.db"
    monkeypatch.setattr(
        agent_workspace_routes,
        "create_worktree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError(str(hidden))),
    )
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"

    response = TestClient(app).post("/api/agent-workspace/projects/project/worktrees", json = {})

    assert response.status_code == 500
    assert response.json()["detail"] == "Worktree creation could not be completed."
    assert str(hidden) not in response.text


def test_project_delete_unwinds_prior_fences_when_worktree_fence_fails(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        chat_history_routes.agent_background_manager,
        "begin_project_deletion",
        lambda _project: events.append("begin-background"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "begin_verification_project_deletion",
        lambda _project: events.append("begin-verification"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "begin_checkpoint_project_deletion",
        lambda _project: events.append("begin-checkpoint"),
    )

    def fail_worktree_fence(_project):
        events.append("begin-worktree")
        raise AgentWorkspaceError("injected worktree fence failure")

    monkeypatch.setattr(chat_history_routes, "begin_worktree_project_deletion", fail_worktree_fence)
    monkeypatch.setattr(
        chat_history_routes,
        "finish_checkpoint_project_deletion",
        lambda _project: events.append("finish-checkpoint"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "finish_verification_project_deletion",
        lambda _project: events.append("finish-verification"),
    )
    monkeypatch.setattr(
        chat_history_routes.agent_background_manager,
        "finish_project_deletion",
        lambda _project: events.append("finish-background"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            chat_history_routes.delete_project(
                "project", SimpleNamespace(), current_subject = "test-subject"
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        "begin-background",
        "begin-verification",
        "begin-checkpoint",
        "begin-worktree",
        "finish-checkpoint",
        "finish-verification",
        "finish-background",
    ]


def test_project_delete_releases_all_fences_in_reverse_order(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        chat_history_routes.agent_background_manager,
        "begin_project_deletion",
        lambda _project: events.append("begin-background"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "begin_verification_project_deletion",
        lambda _project: events.append("begin-verification"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "begin_checkpoint_project_deletion",
        lambda _project: events.append("begin-checkpoint"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "begin_worktree_project_deletion",
        lambda _project: events.append("begin-worktree"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "list_active_worktrees",
        lambda _project: [{"status": "creating"}],
    )
    monkeypatch.setattr(
        chat_history_routes,
        "finish_worktree_project_deletion",
        lambda _project: events.append("finish-worktree"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "finish_checkpoint_project_deletion",
        lambda _project: events.append("finish-checkpoint"),
    )
    monkeypatch.setattr(
        chat_history_routes,
        "finish_verification_project_deletion",
        lambda _project: events.append("finish-verification"),
    )
    monkeypatch.setattr(
        chat_history_routes.agent_background_manager,
        "finish_project_deletion",
        lambda _project: events.append("finish-background"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            chat_history_routes.delete_project(
                "project", SimpleNamespace(), current_subject = "test-subject"
            )
        )

    assert exc_info.value.status_code == 409
    assert events == [
        "begin-background",
        "begin-verification",
        "begin-checkpoint",
        "begin-worktree",
        "finish-worktree",
        "finish-checkpoint",
        "finish-verification",
        "finish-background",
    ]


def test_startup_recovery_uses_one_repository_probe_and_listing_for_many_rows(
    tmp_path, monkeypatch
):
    repository = _setup(tmp_path, monkeypatch)
    worktree_service._worktree_root()
    for position in range(64):
        worktree_id = str(uuid.uuid4())
        container, target, marker = worktree_service._expected_paths(
            "project", worktree_id, create_root = False
        )
        workspace_state.save_worktree(
            {
                "id": worktree_id,
                "projectId": "project",
                "gitRoot": str(repository),
                "path": str(target),
                "branch": f"unsloth-studio/recovery-{position}",
                "baseRef": "HEAD",
                "markerPath": str(marker),
                "markerTokenHash": "0" * 64,
                "backgroundTaskId": None,
                "status": "creating",
                "createdAt": position + 1,
                "updatedAt": position + 1,
            }
        )
        assert not container.exists()

    calls = {"repository": 0, "listing": 0}
    real_resolve = worktree_service._resolve_record_repository
    real_listing = worktree_service._worktree_entries

    def counted_resolve(record):
        calls["repository"] += 1
        return real_resolve(record)

    def counted_listing(root, **kwargs):
        calls["listing"] += 1
        return real_listing(root, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(worktree_service, "_resolve_record_repository", counted_resolve)
        patch.setattr(worktree_service, "_worktree_entries", counted_listing)
        result = worktree_service.reconcile_worktrees_on_startup()

    assert result["removed"] == 64
    assert calls == {"repository": 1, "listing": 1}
    assert all(
        record["status"] == "removed" for record in workspace_state.list_worktrees("project")
    )

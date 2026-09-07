# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.agent_workspace import worktrees as worktree_service
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.execution import (
    acquire_workspace_execution_slot,
    release_workspace_execution_slot,
)
from core.agent_workspace.state import (
    claim_background_task,
    create_agent_background_task,
    create_background_task,
    get_background_task,
    get_worktree,
    retry_background_task,
    update_background_task,
)
from storage import studio_db


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    ).stdout.strip()


def _setup(tmp_path: Path, monkeypatch) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "tracked.txt").write_text("base\n", encoding = "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-qm", "base")
    metadata = repository.stat()
    studio_db.upsert_chat_project(
        {
            "id": "project",
            "name": "Project",
            "instructions": "",
            "rootPath": str(repository),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Scoped goal",
            "goalStatus": "active",
            "goalUpdatedAt": 1,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    return repository


def _queued_agent() -> dict:
    return create_agent_background_task("project", "Work in the owned checkout")


def test_create_worktree_binds_supplied_task_in_database_and_marker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    task = _queued_agent()

    worktree = worktree_service.create_worktree("project", background_task_id = task["id"])

    linked_task = get_background_task(task["id"])
    linked_worktree = get_worktree(worktree["id"])
    marker = json.loads(Path(worktree["markerPath"]).read_text(encoding = "utf-8"))
    assert linked_task["worktreeId"] == worktree["id"]
    assert linked_worktree["backgroundTaskId"] == task["id"]
    assert marker["backgroundTaskId"] == task["id"]
    assert worktree_service.owned_worktree_path(
        "project", worktree["id"], background_task_id = task["id"]
    ) == Path(worktree["path"])

    update_background_task(task["id"], "cancelled")
    worktree_service.cleanup_worktree("project", worktree["id"])


def test_startup_reconciliation_completes_reserved_queued_task_activation(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    task = _queued_agent()
    real_transition = worktree_service.transition_worktree_status

    def fail_activation(worktree_id, expected, status):
        if status == "active":
            raise AgentWorkspaceError("simulated crash boundary")
        return real_transition(worktree_id, expected, status)

    with monkeypatch.context() as patch:
        patch.setattr(
            worktree_service,
            "transition_worktree_status",
            fail_activation,
        )
        with pytest.raises(AgentWorkspaceError, match = "durable state"):
            worktree_service.create_worktree("project", background_task_id = task["id"])

    record = get_worktree(worktree_service.list_worktrees("project")[0]["id"])
    assert record["status"] == "creating"
    assert get_background_task(task["id"])["worktreeId"] == record["id"]

    recovered = worktree_service.reconcile_worktrees_on_startup()

    assert recovered["activated"] == 1
    assert get_background_task(task["id"])["worktreeId"] == record["id"]
    assert get_worktree(record["id"])["backgroundTaskId"] == task["id"]
    update_background_task(task["id"], "cancelled")
    worktree_service.cleanup_worktree("project", record["id"])


def test_reserved_task_cannot_start_until_worktree_is_active(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    task = _queued_agent()
    add_started = threading.Event()
    release_add = threading.Event()
    original_git = worktree_service._git

    def blocked_git(root, args, **kwargs):
        if args[:2] == ["worktree", "add"]:
            add_started.set()
            if not release_add.wait(timeout = 5):
                raise AssertionError("worktree creation test did not release Git")
        return original_git(root, args, **kwargs)

    monkeypatch.setattr(worktree_service, "_git", blocked_git)
    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(
            worktree_service.create_worktree,
            "project",
            background_task_id = task["id"],
        )
        assert add_started.wait(timeout = 2)
        reserved = get_background_task(task["id"])
        assert reserved["worktreeId"] is not None
        assert get_worktree(reserved["worktreeId"])["status"] == "creating"
        with pytest.raises(AgentWorkspaceError, match = "not ready"):
            claim_background_task(task["id"])
        assert get_background_task(task["id"])["status"] == "queued"
        release_add.set()
        worktree = future.result(timeout = 10)

    assert worktree["status"] == "active"
    update_background_task(task["id"], "cancelled")
    worktree_service.cleanup_worktree("project", worktree["id"])


def test_failed_worktree_checkout_releases_queued_task_reservation(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    task = _queued_agent()
    original_git = worktree_service._git

    def fail_checkout(root, args, **kwargs):
        if args[:2] == ["worktree", "add"]:
            raise AgentWorkspaceError("injected checkout failure")
        return original_git(root, args, **kwargs)

    monkeypatch.setattr(worktree_service, "_git", fail_checkout)
    with pytest.raises(AgentWorkspaceError, match = "injected checkout failure"):
        worktree_service.create_worktree("project", background_task_id = task["id"])

    current = get_background_task(task["id"])
    record = worktree_service.list_worktrees("project")[0]
    assert current["status"] == "queued"
    assert current["worktreeId"] is None
    assert record["status"] == "removed"
    assert record["backgroundTaskId"] is None


def test_retry_transfers_worktree_ownership_in_one_state_transaction(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    original = _queued_agent()
    worktree = worktree_service.create_worktree("project", background_task_id = original["id"])
    update_background_task(original["id"], "cancelled")

    with monkeypatch.context() as patch:
        patch.setattr(
            worktree_service,
            "bind_background_task_worktree",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("retry must not use a second binding transaction")
            ),
        )
        retried = retry_background_task(original["id"])

    assert retried["worktreeId"] == worktree["id"]
    assert get_worktree(worktree["id"])["backgroundTaskId"] == retried["id"]
    worktree_service.sync_worktree_background_task_marker("project", worktree["id"], retried["id"])
    update_background_task(retried["id"], "cancelled")
    worktree_service.cleanup_worktree("project", worktree["id"])


def test_merge_and_cleanup_reject_every_live_agent_task_state(tmp_path, monkeypatch):
    repository = _setup(tmp_path, monkeypatch)
    task = _queued_agent()
    worktree = worktree_service.create_worktree("project", background_task_id = task["id"])
    expected_head = _git(repository, "rev-parse", "HEAD")

    for status in ("queued", "running", "cancelling"):
        if get_background_task(task["id"])["status"] != status:
            update_background_task(task["id"], status)
        with pytest.raises(AgentWorkspaceError, match = "Stop the linked"):
            worktree_service.merge_worktree("project", worktree["id"], expected_head)
        with pytest.raises(AgentWorkspaceError, match = "Stop the linked"):
            worktree_service.cleanup_worktree("project", worktree["id"])

    update_background_task(task["id"], "cancelled")
    assert worktree_service.cleanup_worktree("project", worktree["id"])["status"] == "removed"


def test_task_side_verification_link_also_blocks_merge_and_cleanup(tmp_path, monkeypatch):
    repository = _setup(tmp_path, monkeypatch)
    worktree = worktree_service.create_worktree("project")
    task = create_background_task(
        "project",
        "verification",
        {"selectedNames": None, "worktreeId": worktree["id"]},
        worktree_id = worktree["id"],
    )
    assert get_worktree(worktree["id"])["backgroundTaskId"] is None
    expected_head = _git(repository, "rev-parse", "HEAD")

    with pytest.raises(AgentWorkspaceError, match = "Stop the linked"):
        worktree_service.merge_worktree("project", worktree["id"], expected_head)
    with pytest.raises(AgentWorkspaceError, match = "Stop the linked"):
        worktree_service.cleanup_worktree("project", worktree["id"])

    update_background_task(task["id"], "cancelled")
    assert worktree_service.cleanup_worktree("project", worktree["id"])["status"] == "removed"


def test_unexpected_real_merge_conflict_is_retained_for_explicit_resolution(tmp_path, monkeypatch):
    repository = _setup(tmp_path, monkeypatch)
    worktree = worktree_service.create_worktree("project")
    worktree_path = Path(worktree["path"])
    (worktree_path / "tracked.txt").write_text("agent\n", encoding = "utf-8")
    _git(worktree_path, "add", "tracked.txt")
    _git(worktree_path, "commit", "-qm", "agent conflict")
    (repository / "tracked.txt").write_text("primary\n", encoding = "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-qm", "primary conflict")
    expected_head = _git(repository, "rev-parse", "HEAD")
    real_run_git = worktree_service._run_git

    def preflight_reports_clean(root, args, **kwargs):
        if args[:1] == ["merge-tree"]:
            return 0, "", False
        return real_run_git(root, args, **kwargs)

    monkeypatch.setattr(worktree_service, "_run_git", preflight_reports_clean)

    conflict = worktree_service.merge_worktree("project", worktree["id"], expected_head)

    assert conflict["merge"]["status"] == "conflict"
    assert conflict["merge"]["primaryWorkspaceChanged"] is True
    assert _git(repository, "rev-parse", "HEAD") == expected_head
    assert "UU tracked.txt" in _git(repository, "status", "--porcelain")
    assert (repository / ".git" / "MERGE_HEAD").is_file()
    contents = (repository / "tracked.txt").read_text(encoding = "utf-8")
    assert "<<<<<<< HEAD" in contents
    assert ">>>>>>>" in contents


@pytest.mark.parametrize("held_root", ["primary", "worktree"])
def test_merge_waits_for_managed_writer_slot(tmp_path, monkeypatch, held_root):
    repository = _setup(tmp_path, monkeypatch)
    worktree = worktree_service.create_worktree("project")
    worktree_path = Path(worktree["path"])
    (worktree_path / "agent.txt").write_text("agent\n", encoding = "utf-8")
    _git(worktree_path, "add", "agent.txt")
    _git(worktree_path, "commit", "-qm", "agent change")
    expected_head = _git(repository, "rev-parse", "HEAD")
    held_path = repository if held_root == "primary" else worktree_path
    held_metadata = held_path.stat()
    held_identity = (int(held_metadata.st_dev), int(held_metadata.st_ino))
    reached_held_slot = threading.Event()
    real_acquire = worktree_service.acquire_workspace_execution_slot

    assert acquire_workspace_execution_slot(held_identity)

    def observed_acquire(identity, cancel_event = None):
        if identity == held_identity:
            reached_held_slot.set()
        return real_acquire(identity, cancel_event)

    monkeypatch.setattr(worktree_service, "acquire_workspace_execution_slot", observed_acquire)
    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(
            worktree_service.merge_worktree,
            "project",
            worktree["id"],
            expected_head,
        )
        try:
            assert reached_held_slot.wait(timeout = 2)
            assert not future.done()
        finally:
            release_workspace_execution_slot(held_identity)
        merged = future.result(timeout = 10)

    assert merged["merge"]["status"] == "merged"
    worktree_service.cleanup_worktree("project", worktree["id"])


def test_cleanup_waits_for_foreground_worktree_writer_slot(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    worktree = worktree_service.create_worktree("project")
    worktree_path = Path(worktree["path"])
    metadata = worktree_path.stat()
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    reached_slot = threading.Event()
    real_acquire = worktree_service.acquire_workspace_execution_slot

    assert acquire_workspace_execution_slot(identity)

    def observed_acquire(requested, cancel_event = None):
        if requested == identity:
            reached_slot.set()
        return real_acquire(requested, cancel_event)

    monkeypatch.setattr(worktree_service, "acquire_workspace_execution_slot", observed_acquire)
    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(worktree_service.cleanup_worktree, "project", worktree["id"])
        try:
            assert reached_slot.wait(timeout = 2)
            assert not future.done()
            assert worktree_path.is_dir()
        finally:
            release_workspace_execution_slot(identity)
        removed = future.result(timeout = 10)

    assert removed["status"] == "removed"
    assert not worktree_path.exists()

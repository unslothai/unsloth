# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.inference_executor import _agent_messages, _agent_tools
from core.agent_workspace.state import (
    get_background_task,
    get_worktree,
    list_background_task_tree,
    update_background_task,
)
from core.agent_workspace.worktrees import cleanup_worktree, create_worktree
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


def _project(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "base.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "base.txt")
    _git(root, "commit", "-qm", "base")
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": "project",
            "name": "Project",
            "instructions": "Keep changes focused.",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Ship the feature",
            "goalStatus": "active",
            "goalUpdatedAt": 7,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _runtime() -> dict:
    return {
        "kind": "local",
        "model": "test-model",
        "providerId": None,
        "permissionMode": "off",
        "reasoningEffort": None,
        "maxOutputTokens": 8192,
    }


def _policy(**updates) -> dict:
    return {
        "enabled": True,
        "maxChildren": 4,
        "maxParallelChildren": 2,
        "maxDepth": 1,
        "totalChildOutputTokens": 16_384,
        "totalChildToolCalls": 50,
        "totalChildWallSeconds": 600,
        **updates,
    }


def _budget(**updates) -> dict:
    return {
        "maxOutputTokens": 4096,
        "maxToolCalls": 10,
        "wallSeconds": 120,
        **updates,
    }


def test_child_agents_have_real_lineage_inherited_runtime_and_completion_fence(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo"
    repository.mkdir()
    _project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "worktrees"))
    parent_worktree = create_worktree("project")
    child_worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 2)
    try:
        parent = manager.enqueue_agent(
            "project",
            "Coordinate independent work",
            runtime_selection = _runtime(),
            worktree_id = parent_worktree["id"],
            delegation_policy = _policy(),
            start = False,
        )
        child = manager.enqueue_child_agent(
            "project",
            parent["id"],
            "Inspect the parser",
            role = "explorer",
            budget = _budget(),
            worktree_id = child_worktree["id"],
            start = False,
        )

        assert child["parentTaskId"] == parent["id"]
        assert child["retryOfTaskId"] is None
        assert child["rootTaskId"] == parent["id"]
        assert child["delegationDepth"] == 1
        assert child["delegationRole"] == "explorer"
        assert child["delegationBudget"] == _budget()
        assert child["payload"]["runtime"] == {
            **parent["payload"]["runtime"],
            "maxOutputTokens": 4096,
        }
        assert child["goalSnapshot"] == parent["goalSnapshot"]
        assert get_worktree(child_worktree["id"])["backgroundTaskId"] == child["id"]

        update_background_task(parent["id"], "running")
        with pytest.raises(AgentWorkspaceError, match = "active child"):
            update_background_task(parent["id"], "completed", result = {})

        update_background_task(child["id"], "cancelled")
        retried = manager.retry(child["id"], start = False)
        assert retried["parentTaskId"] == parent["id"]
        assert retried["retryOfTaskId"] == child["id"]
        assert retried["rootTaskId"] == parent["id"]
        assert retried["delegationBudget"] == child["delegationBudget"]
        update_background_task(retried["id"], "cancelled")
        update_background_task(parent["id"], "completed", result = {"output": "done"})
        with pytest.raises(AgentWorkspaceError, match = "parent agent is no longer active"):
            manager.retry(retried["id"], start = False)

        tree = list_background_task_tree("project", retried["id"])
        assert tree["rootTaskId"] == parent["id"]
        assert {task["id"] for task in tree["tasks"]} == {
            parent["id"],
            child["id"],
            retried["id"],
        }
    finally:
        manager._executor.shutdown(wait = True)

    cleanup_worktree("project", child_worktree["id"])
    cleanup_worktree("project", parent_worktree["id"])


def test_delegation_budgets_are_reserved_atomically(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "worktrees"))
    parent_worktree = create_worktree("project")
    first_worktree = create_worktree("project")
    blocked_worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 2)
    try:
        parent = manager.enqueue_agent(
            "project",
            "Coordinate",
            runtime_selection = _runtime(),
            worktree_id = parent_worktree["id"],
            delegation_policy = _policy(maxChildren = 2, maxParallelChildren = 1),
            start = False,
        )
        first = manager.enqueue_child_agent(
            "project",
            parent["id"],
            "First",
            role = "verifier",
            budget = _budget(),
            worktree_id = first_worktree["id"],
            start = False,
        )
        with pytest.raises(AgentWorkspaceError, match = "parallel budget"):
            manager.enqueue_child_agent(
                "project",
                parent["id"],
                "Second",
                role = "reviewer",
                budget = _budget(),
                worktree_id = blocked_worktree["id"],
                start = False,
            )
        assert get_worktree(blocked_worktree["id"])["backgroundTaskId"] is None
        update_background_task(first["id"], "cancelled")
        update_background_task(parent["id"], "cancelled")
    finally:
        manager._executor.shutdown(wait = True)

    cleanup_worktree("project", blocked_worktree["id"])
    cleanup_worktree("project", first_worktree["id"])
    cleanup_worktree("project", parent_worktree["id"])


def test_cancelling_parent_cannot_admit_a_new_child(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "worktrees"))
    parent_worktree = create_worktree("project")
    child_worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        parent = manager.enqueue_agent(
            "project",
            "Coordinate",
            runtime_selection = _runtime(),
            worktree_id = parent_worktree["id"],
            delegation_policy = _policy(),
            start = False,
        )
        update_background_task(parent["id"], "running")
        update_background_task(parent["id"], "cancelling")
        with pytest.raises(AgentWorkspaceError, match = "active agent"):
            manager.enqueue_child_agent(
                "project",
                parent["id"],
                "Late child",
                role = "explorer",
                budget = _budget(),
                worktree_id = child_worktree["id"],
                start = False,
            )
        assert get_worktree(child_worktree["id"])["backgroundTaskId"] is None
        update_background_task(parent["id"], "cancelled")
    finally:
        manager._executor.shutdown(wait = True)
    cleanup_worktree("project", child_worktree["id"])
    cleanup_worktree("project", parent_worktree["id"])


def test_parent_failure_cancels_active_children(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "worktrees"))
    parent_worktree = create_worktree("project")
    child_worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        parent = manager.enqueue_agent(
            "project",
            "Coordinate",
            runtime_selection = _runtime(),
            worktree_id = parent_worktree["id"],
            delegation_policy = _policy(),
            start = False,
        )
        child = manager.enqueue_child_agent(
            "project",
            parent["id"],
            "Inspect",
            role = "explorer",
            budget = _budget(),
            worktree_id = child_worktree["id"],
            start = False,
        )
        update_background_task(parent["id"], "running")

        manager._run_agent(
            parent["id"],
            threading.Event(),
            lambda _context, _event: {},
        )

        assert get_background_task(parent["id"])["status"] == "failed"
        assert get_background_task(child["id"])["status"] == "cancelled"
    finally:
        manager._executor.shutdown(wait = True)
    cleanup_worktree("project", child_worktree["id"])
    cleanup_worktree("project", parent_worktree["id"])


def test_delegation_tools_are_exposed_only_with_remaining_server_policy(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "worktrees"))
    parent_worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        parent = manager.enqueue_agent(
            "project",
            "Coordinate",
            runtime_selection = _runtime(),
            worktree_id = parent_worktree["id"],
            delegation_policy = _policy(),
            start = False,
        )
        context = SimpleNamespace(
            task_id = parent["id"],
            project_id = "project",
            delegation_depth = 0,
            delegation_role = None,
            goal_snapshot = "Ship the feature",
            goal_status_snapshot = "active",
            plan_snapshot = None,
            cwd = Path(parent_worktree["path"]),
            expected_root_identity = None,
            instruction = "Coordinate",
        )
        names = {tool["function"]["name"] for tool in _agent_tools(context, full_access = False)}
        assert {"delegate_agent", "child_agent_status", "cancel_child_agent"} <= names
        assert "<child_agents>" in _agent_messages(context)[0]["content"]

        context.delegation_depth = 1
        names = {tool["function"]["name"] for tool in _agent_tools(context, full_access = False)}
        assert "delegate_agent" not in names
        update_background_task(parent["id"], "cancelled")
    finally:
        manager._executor.shutdown(wait = True)
    cleanup_worktree("project", parent_worktree["id"])

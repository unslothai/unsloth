# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace.common import AgentWorkspaceError, workspace_fingerprint
from core.agent_workspace.state import (
    begin_verification_run,
    create_plan,
    finish_verification_run,
    get_background_task,
    get_plan,
    get_worktree,
    set_verification_config,
    update_plan_task,
)
from core.agent_workspace.worktrees import (
    cleanup_worktree,
    create_worktree,
    merge_worktree,
)
from storage import studio_db
from routes import agent_workspace as agent_workspace_routes


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    ).stdout.strip()


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "shared.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "shared.txt")
    _git(root, "commit", "-qm", "base")


def _folder_project(root: Path, project_id: str = "project") -> dict:
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Original goal",
            "goalStatus": "active",
            "goalUpdatedAt": 7,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _wait_task(task_id: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = get_background_task(task_id)
        if task and task["status"] in {
            "cancelled",
            "completed",
            "failed",
            "interrupted",
        }:
            return task
        time.sleep(0.01)
    raise AssertionError("background task did not stop")


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def test_plan_task_completion_requires_fresh_named_verification(tmp_path):
    _folder_project(tmp_path)
    (tmp_path / "source.txt").write_text("source", encoding = "utf-8")
    config = set_verification_config(
        "project",
        [
            {
                "name": "unit",
                "kind": "test",
                "command": "true",
                "required": True,
                "timeoutSeconds": 10,
                "logLimitBytes": 1024,
            }
        ],
    )
    plan = create_plan(
        "project",
        "Plan",
        "Original goal",
        [{"title": "Verified task", "verification": [{"name": "unit"}]}],
    )
    task = plan["tasks"][0]

    with pytest.raises(AgentWorkspaceError, match = "fresh passing"):
        update_plan_task(plan["id"], task["id"], status = "completed")

    fingerprint = workspace_fingerprint(tmp_path)
    run = begin_verification_run("project", fingerprint, config_revision = config["revision"])
    finish_verification_run(
        run["id"],
        "passed",
        fingerprint,
        [{"name": "unit", "required": True, "status": "passed"}],
    )
    completed = update_plan_task(plan["id"], task["id"], status = "completed")

    assert completed["tasks"][0]["status"] == "completed"
    assert completed["completionSummary"]["remaining"] == 0


def test_agent_task_uses_immutable_goal_plan_and_primary_cwd(tmp_path):
    _folder_project(tmp_path)
    plan = create_plan("project", "Plan v1", "Original goal", [{"title": "Scoped task"}])
    observed = {}

    def executor(context, cancel_event):
        observed.update(
            goal = context.goal_snapshot,
            goal_status = context.goal_status_snapshot,
            plan = context.plan_snapshot,
            plan_revision = context.plan_revision,
            plan_task_id = context.plan_task_id,
            cwd = context.cwd,
            cancelled = cancel_event.is_set(),
        )
        return {"output": "done"}

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(executor)
    try:
        queued = manager.enqueue_agent(
            "project",
            "Implement the scoped task",
            plan_id = plan["id"],
            plan_task_id = plan["tasks"][0]["id"],
            start = False,
        )
        changed_project = studio_db.get_chat_project("project")
        studio_db.upsert_chat_project(
            {
                **changed_project,
                "goal": "Changed goal",
                "goalStatus": "paused",
                "goalUpdatedAt": 9,
            }
        )
        update_plan_task(plan["id"], plan["tasks"][0]["id"], blocker = "changed after queue")

        assert queued["status"] == "queued"
        assert queued["goalSnapshot"] == "Original goal"
        assert queued["planRevision"] == 0
        manager.start(queued["id"])
        finished = _wait_task(queued["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert finished["status"] == "completed"
    assert finished["startedAt"] is not None
    assert finished["completedAt"] is not None
    assert finished["appExitPolicy"] == "interrupt"
    assert observed["goal"] == "Original goal"
    assert observed["goal_status"] == "active"
    assert observed["plan"]["tasks"][0]["blocker"] is None
    assert observed["plan_revision"] == 0
    assert observed["plan_task_id"] == plan["tasks"][0]["id"]
    assert observed["cwd"] == tmp_path.resolve()
    assert observed["cancelled"] is False


def test_agent_task_failure_cancel_retry_and_output_bounds(tmp_path):
    _folder_project(tmp_path)
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        manager.register_agent_executor(
            lambda _context, _cancel: (_ for _ in ()).throw(RuntimeError("failed"))
        )
        failed = manager.enqueue_agent("project", "fail", start = True)
        failed = _wait_task(failed["id"])
        assert failed["status"] == "failed"

        retried = manager.retry(failed["id"], start = False)
        assert retried["status"] == "queued"
        assert retried["parentTaskId"] == failed["id"]
        assert retried["attempt"] == 2
        assert retried["goalSnapshot"] == failed["goalSnapshot"]

        entered = threading.Event()

        def cancellable(_context, cancel_event):
            entered.set()
            cancel_event.wait(timeout = 5)
            return {"output": "x" * (2 * 1024 * 1024)}

        manager.register_agent_executor(cancellable)
        manager.start(retried["id"])
        assert entered.wait(timeout = 2)
        manager.cancel(retried["id"])
        stopped = _wait_task(retried["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert stopped["status"] == "cancelled"
    assert stopped["cancelRequested"] is True
    assert stopped["result"]["outputTruncated"] is True
    assert len(stopped["result"]["output"].encode("utf-8")) <= 900 * 1024


def test_app_exit_marks_uncooperative_active_agent_interrupted(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def executor(_context, _cancel_event):
        entered.set()
        release.wait(timeout = 5)
        return {"output": "late"}

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(executor)
    try:
        task = manager.enqueue_agent("project", "long run", start = True)
        assert entered.wait(timeout = 2)
        stopped = manager.prepare_for_app_exit(timeout_seconds = 0.05)
        assert stopped[0]["status"] == "interrupted"
        assert get_background_task(task["id"])["status"] == "interrupted"
        release.set()
    finally:
        release.set()
        manager._executor.shutdown(wait = True)

    assert get_background_task(task["id"])["status"] == "interrupted"
    assert get_background_task(task["id"])["appExitContract"] == {
        "activeTaskState": "interrupted",
        "managedCommandsSurvive": False,
        "adapterMustHonorCancellation": True,
    }


def test_agent_start_requires_registered_adapter_without_claiming(tmp_path):
    _folder_project(tmp_path)
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        queued = manager.enqueue_agent("project", "wait", start = False)
        with pytest.raises(AgentWorkspaceError, match = "executor"):
            manager.start(queued["id"])
        assert get_background_task(queued["id"])["status"] == "queued"
    finally:
        manager._executor.shutdown(wait = True)


def test_agent_route_queues_provider_neutral_task_with_explicit_runtime(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(agent_workspace_routes, "_require_execution_boundary", lambda: None)
    agent_workspace_routes.background_manager.register_agent_executor(None)

    response = _client().post(
        "/api/agent-workspace/projects/project/background/agent",
        json = {
            "instruction": "Queue this",
            "runtime": {
                "kind": "local",
                "model": "local/model.gguf",
                "permissionMode": "off",
            },
            "start": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "agent"
    assert body["status"] == "queued"
    assert body["goalSnapshot"] == "Original goal"
    assert body["planSnapshot"] is None
    assert body["appExitPolicy"] == "interrupt"
    assert body["payload"]["runtime"]["model"] == "local/model.gguf"


def test_parallel_agent_tasks_receive_isolated_owned_worktree_cwds(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    first = create_worktree("project")
    second = create_worktree("project")
    barrier = threading.Barrier(2)
    observed = []
    observed_lock = threading.Lock()

    def executor(context, _cancel_event):
        barrier.wait(timeout = 5)
        with observed_lock:
            observed.append((context.worktree_id, context.cwd))
        return {"output": context.worktree_id}

    manager = BackgroundTaskManager(max_workers = 2)
    manager.register_agent_executor(executor)
    try:
        one = manager.enqueue_agent("project", "one", worktree_id = first["id"], start = True)
        two = manager.enqueue_agent("project", "two", worktree_id = second["id"], start = True)
        assert _wait_task(one["id"])["status"] == "completed"
        assert _wait_task(two["id"])["status"] == "completed"
    finally:
        manager._executor.shutdown(wait = True)

    assert set(observed) == {
        (first["id"], Path(first["path"])),
        (second["id"], Path(second["path"])),
    }
    cleanup_worktree("project", first["id"])
    cleanup_worktree("project", second["id"])


def test_cancelled_agent_cleans_only_clean_owned_worktree(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    entered = threading.Event()

    def executor(_context, cancel_event):
        entered.set()
        cancel_event.wait(timeout = 5)
        return {}

    manager = BackgroundTaskManager(max_workers = 1)
    manager.register_agent_executor(executor)
    try:
        task = manager.enqueue_agent(
            "project",
            "cancel",
            worktree_id = worktree["id"],
            cleanup_worktree_on_cancel = True,
            start = True,
        )
        assert entered.wait(timeout = 2)
        manager.cancel(task["id"])
        stopped = _wait_task(task["id"])
        deadline = time.monotonic() + 2
        while "worktreeCleanup" not in (stopped.get("result") or {}):
            if time.monotonic() >= deadline:
                raise AssertionError("worktree cleanup result was not persisted")
            time.sleep(0.01)
            stopped = get_background_task(task["id"])
    finally:
        manager._executor.shutdown(wait = True)

    assert stopped["result"]["worktreeCleanup"] == "removed"
    assert get_worktree(worktree["id"])["status"] == "removed"
    assert not Path(worktree["path"]).exists()


def test_retry_preserves_durable_worktree_link_after_interruption(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    manager = BackgroundTaskManager(max_workers = 1)
    try:
        task = manager.enqueue_agent("project", "recover", worktree_id = worktree["id"], start = False)
        from core.agent_workspace.state import update_background_task

        update_background_task(task["id"], "running")
        update_background_task(task["id"], "interrupted")
        retried = manager.retry(task["id"], start = False)
    finally:
        manager._executor.shutdown(wait = True)

    assert retried["parentTaskId"] == task["id"]
    assert retried["worktreeId"] == worktree["id"]
    assert get_worktree(worktree["id"])["backgroundTaskId"] == retried["id"]
    update_background_task(retried["id"], "cancelled")
    cleanup_worktree("project", worktree["id"])


def test_owned_worktree_merge_records_success(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    worktree_path = Path(worktree["path"])
    (worktree_path / "agent.txt").write_text("agent\n", encoding = "utf-8")
    _git(worktree_path, "add", "agent.txt")
    _git(worktree_path, "commit", "-qm", "agent change")
    expected = _git(repository, "rev-parse", "HEAD")

    merged = merge_worktree("project", worktree["id"], expected)

    assert merged["merge"]["status"] == "merged"
    assert merged["merge"]["expectedTargetHead"] == expected
    assert merged["merge"]["resultHead"] == _git(repository, "rev-parse", "HEAD")
    assert (repository / "agent.txt").read_text(encoding = "utf-8") == "agent\n"
    cleanup_worktree("project", worktree["id"])


def test_owned_worktree_merge_conflict_does_not_modify_primary(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    worktree_path = Path(worktree["path"])
    (worktree_path / "shared.txt").write_text("agent\n", encoding = "utf-8")
    _git(worktree_path, "add", "shared.txt")
    _git(worktree_path, "commit", "-qm", "agent conflict")
    (repository / "shared.txt").write_text("primary\n", encoding = "utf-8")
    _git(repository, "add", "shared.txt")
    _git(repository, "commit", "-qm", "primary conflict")
    expected = _git(repository, "rev-parse", "HEAD")

    conflict = merge_worktree("project", worktree["id"], expected)

    assert conflict["merge"]["status"] == "conflict"
    assert conflict["merge"]["primaryWorkspaceChanged"] is False
    assert _git(repository, "rev-parse", "HEAD") == expected
    assert _git(repository, "status", "--porcelain") == ""
    assert (repository / "shared.txt").read_text(encoding = "utf-8") == "primary\n"
    cleanup_worktree("project", worktree["id"])

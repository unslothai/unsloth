# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from storage import studio_db
from routes import project_tasks


@pytest.fixture
def engine():
    return pytest.importorskip("core.agent_workspace.task_state")


@pytest.fixture
def integration(engine):
    pytest.importorskip("core.agent_workspace.worktrees")
    return pytest.importorskip("core.agent_workspace.task_service")


@pytest.fixture
def repository(tmp_path, monkeypatch, integration):
    if os.name != "posix":
        pytest.skip("Owned worktree execution requires POSIX; Windows refusal is tested separately")
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    project = studio_db.upsert_chat_project(
        {"id": "project", "name": "Project", "createdAt": 1, "updatedAt": 1}
    )
    root = Path(project["sandboxPath"])
    root.mkdir(parents = True, exist_ok = True)
    for args in [
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "test@example.invalid"),
    ]:
        subprocess.run(["git", *args], cwd = root, check = True, capture_output = True)
    (root / "example.txt").write_text("before\n")
    subprocess.run(["git", "add", "example.txt"], cwd = root, check = True, capture_output = True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd = root, check = True, capture_output = True)
    from core.agent_workspace import task_executor, task_runtime

    monkeypatch.setattr(task_runtime, "validate_runtime", lambda _snapshot: None)
    monkeypatch.setattr(task_executor, "validate_runtime", lambda _snapshot: None)
    monkeypatch.setattr(
        integration,
        "capture_runtime",
        lambda *_args: {"kind": "provider", "model": "test", "credential": "private"},
    )
    monkeypatch.setattr(integration, "_runner", None)
    monkeypatch.setattr(integration, "_closing", False)
    yield root
    if integration._runner:
        assert integration._runner.shutdown(timeout = 10)


def _frame(delta, finish = None):
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        + "\n\n"
    )


class SimulatedModel:
    heals_text_tool_calls = False
    sanitizes_provider_frames = True
    reserved = 0

    def __init__(self, context):
        self.context = context
        self.turn = 0

    async def stream(self, *, messages, tools, tool_choice, cancel_event):
        from core.agent_workspace.task_runtime import _admission

        # Exercise the real one-slot admission queue with two parent workers.
        backend = SimpleNamespace(
            base_url = "test-single-slot", effective_parallel_slots = 1, context_length = 16384
        )
        async with _admission(backend, self.context):
            self.turn += 1
            self.reserved += 1024
            if self.context.task["role"] == "root":
                if self.turn == 1:
                    name, arguments = (
                        "task_delegate",
                        {
                            "instruction": "Change example.txt",
                            "role": "implementer",
                            "max_output_tokens": 4096,
                        },
                    )
                elif self.turn == 2:
                    child = next(
                        json.loads(m["content"])["id"]
                        for m in messages
                        if m.get("role") == "tool" and m.get("name") == "task_delegate"
                    )
                    name, arguments = "task_wait", {"task_id": child}
                else:
                    assert any(
                        '"status": "completed"' in m.get("content", "")
                        for m in messages
                        if m.get("role") == "tool"
                    )
                    yield _frame({"content": "Child changes are ready for review."}, "stop")
                    return
            elif self.turn == 1:
                name, arguments = (
                    "task_edit_file",
                    {"path": "example.txt", "expected": "before\n", "content": "after\n"},
                )
            else:
                yield _frame({"content": "Updated example.txt."}, "stop")
                return
            yield _frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-{self.turn}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                }
            )
            yield _frame({}, "tool_calls")


def test_two_parents_delegate_edit_and_wait_with_one_model_slot(
    repository, integration, monkeypatch
):
    from core.agent_workspace import task_executor, task_workspaces, worktrees
    from core.inference import llama_admission

    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_CONTROL", "1")
    llama_admission.reset_llama_admission_queues()
    monkeypatch.setattr(task_executor, "TaskTransport", SimulatedModel)
    created = _client().post(
        "/api/agent/projects/project/tasks",
        json = {
            "instruction": "Implement the change",
            "kind": "provider",
            "model": "test",
        },
    )
    assert created.status_code == 202, created.text
    roots = [
        created.json(),
        integration.submit("project", "Implement the change", kind = "provider", model = "test"),
    ]
    for task in roots:
        finished = integration.runner().wait("project", task["id"], timeout = 60)
        assert (
            finished is not None and finished["status"] == "completed"
        ), integration.state.list_tasks("project")
        assert "ready for review" in finished["result"]["output"]
        children = integration.state.list_children("project", task["id"])
        assert len(children) == 1 and children[0]["status"] == "completed"
        child_binding = task_workspaces.binding("project", children[0]["id"])
        path = worktrees.owned_worktree_path("project", child_binding["worktree_id"])
        assert (path / "example.txt").read_text() == "after\n"
        review = _client().get(f"/api/agent/projects/project/tasks/{children[0]['id']}/review")
        assert review.status_code == 200, review.text
        assert "+after" in review.json()["diff"]
        public = integration.public_task(children[0])
        assert public["worktreeId"] == child_binding["worktree_id"]
        assert "private" not in json.dumps(public) and "snapshot" not in public
    assert (repository / "example.txt").read_text() == "before\n"
    assert len(worktrees.list_project_worktrees("project")) == 4


def _owned_context(integration, *, role = "root"):
    from core.agent_workspace.task_workspaces import capture_workspace

    snapshot = {
        "runtime": {"kind": "provider", "model": "test"},
        "workspace": capture_workspace("project"),
    }
    task = integration.state.create_task("project", "Inspect", snapshot)
    task = integration.state.claim_task("project", task["id"], "owner")
    task["role"] = role
    return SimpleNamespace(
        task = task,
        cancel_event = threading.Event(),
        deadline = __import__("time").monotonic() + 120,
        check = lambda: integration.state.validate_owner(task["id"], "owner"),
    )


def test_native_checkout_fence_outlives_revoked_task_lease(repository, integration):
    from core.agent_workspace import task_workspaces, worktrees
    from core.agent_workspace.git_context import AgentWorkspaceError

    context = _owned_context(integration)
    with task_workspaces.task_workspace(context) as (_, worktree_id):
        integration.state.finish_task(context.task["id"], "owner", "failed")
        with pytest.raises(AgentWorkspaceError, match = "still using"):
            worktrees.cleanup_worktree("project", worktree_id)
        with pytest.raises(AgentWorkspaceError, match = "still using"):
            worktrees.merge_worktree(
                "project", worktree_id, context.task["snapshot"]["workspace"]["head"]
            )
    assert worktrees.cleanup_worktree("project", worktree_id)["status"] == "removed"


@pytest.mark.parametrize("role", ["root", "reviewer"])
def test_only_implementers_receive_file_mutation_authority(repository, integration, role):
    from core.agent_workspace.task_executor import TaskTools
    from core.agent_workspace.task_workspaces import task_workspace

    context = _owned_context(integration, role = role)
    with task_workspace(context) as (workspace, _):
        tools = TaskTools(context, workspace)
        with pytest.raises(integration.state.TaskStateError, match = "capability"):
            tools(
                "task_edit_file",
                {"path": "example.txt", "expected": "before\n", "content": "changed"},
            )
        assert tools("task_read_file", {"path": "example.txt"}) == "before\n"


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/tmp/outside",
        ".git/config",
        ".GIT/config",
        "nested/../../outside",
        "C:\\outside",
    ],
)
def test_task_paths_reject_root_and_git_metadata_escape(engine, path):
    from core.agent_workspace.task_executor import _path
    with pytest.raises(engine.TaskStateError):
        _path(path)


def test_symlink_edit_does_not_touch_external_file(repository, integration, tmp_path):
    from core.agent_workspace.task_executor import TaskTools
    from core.agent_workspace.task_workspaces import task_workspace

    outside = tmp_path / "outside"
    outside.write_text("keep")
    context = _owned_context(integration, role = "implementer")
    with task_workspace(context) as (workspace, _):
        (workspace.root / "link").symlink_to(outside)
        with pytest.raises(Exception):
            TaskTools(context, workspace)(
                "task_edit_file", {"path": "link", "expected": "keep", "content": "changed"}
            )
    assert outside.read_text() == "keep"


def test_live_tool_must_drain_before_checkout_can_be_released(repository, integration):
    from core.agent_workspace.task_executor import TaskTools

    context = _owned_context(integration)
    tools = TaskTools(context, None)
    entered, release, drained = threading.Event(), threading.Event(), threading.Event()
    tools._execute = lambda *_: (entered.set(), release.wait(5))
    with ThreadPoolExecutor(2) as pool:
        call = pool.submit(tools, "task_read_file", {})
        assert entered.wait(2)
        drain = pool.submit(lambda: (tools.close_and_drain(), drained.set()))
        assert not drained.wait(0.1)
        release.set()
        call.result(timeout = 2)
        drain.result(timeout = 2)
    with pytest.raises(integration.state.TaskStateError, match = "closed"):
        tools("task_read_file", {})


def test_runtime_drift_is_checked_before_any_tool(repository, integration, monkeypatch):
    from core.agent_workspace import task_executor

    context = _owned_context(integration)

    def changed(_):
        raise integration.state.TaskStateError("runtime changed")

    monkeypatch.setattr(task_executor, "validate_runtime", changed)
    with pytest.raises(integration.state.TaskStateError, match = "runtime changed"):
        task_executor.TaskTools(context, None)("task_read_file", {"path": "example.txt"})


def test_task_retirement_precedes_git_fence_and_unwinds_in_reverse(integration, monkeypatch):
    from core import project_retirement

    events = []
    tasks = SimpleNamespace(
        begin_task_retirement = lambda _: events.append("tasks") or "token",
        finish_task_retirement = lambda *_: events.append("tasks released"),
    )
    git = SimpleNamespace(
        begin_git_retirement = lambda *_, **__: events.append("git") or "token",
        finish_git_retirement = lambda *_: events.append("git released"),
    )
    monkeypatch.setattr(
        project_retirement, "_feature", lambda name: tasks if name == "task_service" else git
    )
    token = project_retirement.begin_project_retirement("project")
    project_retirement.finish_project_retirement("project", token)
    assert events == ["tasks", "git", "git released", "tasks released"]


def _client(auth = True):
    app = FastAPI()
    app.include_router(project_tasks.router)
    if auth:
        app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app)


def test_task_routes_require_authentication():
    assert _client(False).get("/api/agent/projects/project/tasks").status_code in {401, 403}


@pytest.mark.parametrize(
    "extra",
    [
        {"snapshot": {}},
        {"workspace": "/tmp"},
        {"owner": "x"},
        {"permissionMode": "full"},
        {"maxOutputTokens": True},
        {"childLimit": 9},
    ],
)
def test_task_routes_reject_renderer_authority_and_invalid_budgets(extra):
    response = _client().post(
        "/api/agent/projects/project/tasks",
        json = {"instruction": "Fix", "kind": "local", "model": "model", **extra},
    )
    assert response.status_code == 422


def test_missing_task_prerequisite_returns_503(monkeypatch):
    monkeypatch.setattr(project_tasks, "get_chat_project", lambda _: {"id": "project"})

    def unavailable(_name):
        raise ModuleNotFoundError("missing prerequisite")

    monkeypatch.setattr(project_tasks.importlib, "import_module", unavailable)
    assert _client().get("/api/agent/projects/project/tasks").status_code == 503


def test_task_detail_is_scoped_to_project(repository, integration):
    context = _owned_context(integration)
    studio_db.upsert_chat_project({"id": "other", "name": "Other", "createdAt": 1, "updatedAt": 1})
    response = _client().get(f"/api/agent/projects/other/tasks/{context.task['id']}")
    assert response.status_code == 409 and response.json()["detail"] == "Task not found."


def test_windows_refuses_before_creating_checkout(integration, monkeypatch):
    from core.agent_workspace import task_workspaces
    monkeypatch.setattr(task_workspaces, "os", SimpleNamespace(name = "nt"))
    with pytest.raises(integration.state.TaskStateError, match = "platform"):
        task_workspaces.capture_workspace("project")


def test_retirement_os_fence_waits_for_physical_executors_after_db_revocation(
    repository, integration
):
    from core.agent_workspace import task_workspaces
    import time

    held = task_workspaces.acquire_task_project_fence(
        "project", exclusive = False, deadline = time.monotonic() + 1
    )
    try:
        with pytest.raises(integration.state.TaskStateError, match = "still running"):
            task_workspaces.acquire_task_project_fence(
                "project", exclusive = True, deadline = time.monotonic() + 0.1
            )
    finally:
        task_workspaces.release_task_project_fence(held)
    exclusive = task_workspaces.acquire_task_project_fence(
        "project", exclusive = True, deadline = time.monotonic() + 1
    )
    task_workspaces.release_task_project_fence(exclusive)


def test_new_file_review_includes_content_and_refuses_symlinks(repository, integration, tmp_path):
    from core.agent_workspace import task_workspaces

    context = _owned_context(integration)
    outside = tmp_path / "outside"
    outside.write_text("private external content")
    with task_workspaces.task_workspace(context) as (workspace, _):
        (workspace.root / "new.txt").write_text("new contents\n")
        (workspace.root / "link").symlink_to(outside)
    review = task_workspaces.review_task_workspace("project", context.task["id"])
    files = {item["path"]: item for item in review["newFiles"]}
    assert "+new contents" in files["new.txt"]["diff"]
    assert files["link"]["unavailable"]
    assert "private external content" not in json.dumps(review)


def test_list_summaries_bound_results_without_leaking_runtime_fingerprints(repository, integration):
    context = _owned_context(integration)
    task = context.task
    task["result"] = {"output": "a" * 10000}
    task["snapshot"]["runtime"]["credential"] = "private-fingerprint"
    summary = integration.public_task(task, summary = True)
    assert len(summary["result"]["output"]) == 4096 and summary["resultTruncated"]
    assert "private-fingerprint" not in json.dumps(summary)
    assert len(integration.public_task(task)["result"]["output"]) == 10000

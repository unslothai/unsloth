# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent_workspace import task_commands as commands


def test_missing_prerequisite_is_reported_without_starting_a_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "core.agent_workspace.supervisor", None)
    assert commands.availability()["available"] is False


@pytest.fixture
def modules():
    pytest.importorskip("core.agent_workspace.task_executor")
    from core.agent_workspace import (
        supervisor,
        task_command_state,
        task_state,
        task_workspaces,
        task_executor,
        task_runtime,
        task_service,
    )
    return SimpleNamespace(
        supervisor = supervisor,
        evidence = task_command_state,
        state = task_state,
        workspaces = task_workspaces,
        executor = task_executor,
        runtime = task_runtime,
        service = task_service,
    )


@pytest.fixture
def project(tmp_path, monkeypatch, modules):
    if os.name != "posix":
        pytest.skip(
            "Owned task worktrees require POSIX; portable refusals are tested independently"
        )
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    item = studio_db.upsert_chat_project(
        {"id": "commands", "name": "Commands", "createdAt": 1, "updatedAt": 1}
    )
    root = Path(item["sandboxPath"])
    root.mkdir(parents = True, exist_ok = True)
    for argv in [
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.invalid"],
    ]:
        subprocess.run(["git", *argv], cwd = root, check = True, capture_output = True)
    (root / "example.txt").write_text("original\n")
    subprocess.run(["git", "add", "example.txt"], cwd = root, check = True, capture_output = True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd = root, check = True, capture_output = True)
    monkeypatch.setattr(modules.runtime, "validate_runtime", lambda _: None)
    monkeypatch.setattr(modules.executor, "validate_runtime", lambda _: None)
    return root


def context(
    modules,
    *,
    enabled = True,
    role = "implementer",
):
    from core.agent_workspace.task_runner import TaskContext, _Work

    snapshot = {
        "workspace": modules.workspaces.capture_workspace("commands"),
        "runtime": {"kind": "provider", "model": "test"},
        "commandsEnabled": enabled,
    }
    root = modules.state.create_task(
        "commands", "Check changes", snapshot, child_limit = 1, child_budget = 8192
    )
    root = modules.state.claim_task("commands", root["id"], "root-owner")
    if role == "root":
        task, owner = root, "root-owner"
    else:
        task = modules.state.create_child(
            root["id"], "root-owner", "Run tests", snapshot, role = role
        )
        task = modules.state.claim_task("commands", task["id"], "child-owner")
        owner = "child-owner"
    return TaskContext(None, _Work(task = task, owner = owner, deadline = time.monotonic() + 120))


@pytest.mark.parametrize(
    "enabled,role,allowed",
    [
        (False, "implementer", False),
        (True, "root", False),
        (True, "reviewer", False),
        (True, "implementer", True),
    ],
)
def test_commands_require_opted_in_implementer(project, modules, enabled, role, allowed):
    ctx = context(modules, enabled = enabled, role = role)
    advertised = modules.executor.task_tools(role, 1, enabled)
    assert any(tool["function"]["name"] == "task_run_command" for tool in advertised) == allowed
    if allowed:
        assert modules.evidence.require_context(ctx)["id"] == ctx.task["id"]
    else:
        with pytest.raises(modules.state.TaskStateError, match = "not permitted"):
            commands.execute_command(ctx, {"argv": ["true"], "timeout": 1})


@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": ["true"], "timeout": 1, "root": "/tmp"},
        {"argv": ["true"], "timeout": 1, "env": {}},
        {"argv": "true", "timeout": 1},
        {"argv": [], "timeout": 1},
        {"argv": ["true"], "timeout": True},
        {"argv": ["true"], "timeout": 121},
        {"argv": ["true"] * 65, "timeout": 1},
        {"argv": ["x" * 8193], "timeout": 1},
    ],
)
def test_command_arguments_cannot_inject_authority_or_exceed_limits(modules, arguments):
    with pytest.raises((modules.state.TaskStateError, modules.supervisor.AgentWorkspaceError)):
        commands._arguments(arguments)


def test_only_private_worker_capabilities_can_request_a_command(modules):
    with pytest.raises(modules.state.TaskStateError, match = "live task worker"):
        commands.execute_command(
            {"taskId": "one", "owner": "forged"}, {"argv": ["true"], "timeout": 1}
        )


def test_command_budget_is_atomic_and_never_refunded(project, modules):
    ctx = context(modules)

    def reserve(_):
        try:
            return modules.evidence.begin(ctx, ["python", "-m", "pytest"], 1)
        except modules.state.TaskStateError:
            return None

    with ThreadPoolExecutor(8) as pool:
        reservations = [value for value in pool.map(reserve, range(8)) if value]
    assert len(reservations) == 6
    for command_id, receipt in reservations:
        modules.evidence.finish(command_id, receipt, "failed", output = "failed")
    with pytest.raises(modules.state.TaskStateError, match = "six command"):
        modules.evidence.begin(ctx, ["true"], 1)
    assert len(modules.evidence.read("commands", ctx.task["id"])) == 6


def test_evidence_survives_cancellation_and_redacts_without_exposing_receipts(project, modules):
    ctx = context(modules)
    command_id, receipt = modules.evidence.begin(ctx, ["echo", "api_key=secret-example"], 1)
    modules.state.cancel_task("commands", ctx.task["id"])
    modules.evidence.finish(
        command_id,
        receipt,
        "cancelled",
        output = "api_key=secret-example\n" + "x" * 70000,
        output_bytes = 70024,
    )
    public = modules.evidence.read("commands", ctx.task["id"], command_id)
    assert public["status"] == "cancelled" and public["outputTruncated"]
    assert len(public["output"].encode()) <= 64 * 1024
    assert "secret-example" not in json.dumps(public) and receipt not in json.dumps(public)
    assert modules.evidence.read("commands", ctx.task["id"])[0]["previewTruncated"]
    with pytest.raises(modules.state.TaskStateError, match = "not found"):
        modules.evidence.read("other", ctx.task["id"], command_id)


def test_orphaned_command_is_never_reported_as_passed(project, modules):
    ctx = context(modules)
    command_id, _ = modules.evidence.begin(ctx, ["true"], 1)
    modules.state.finish_task(ctx.task["id"], "child-owner", "failed")
    assert modules.evidence.read("commands", ctx.task["id"], command_id)["status"] == "interrupted"


def test_workspace_is_resolved_from_the_task_binding_and_rechecked(project, modules, monkeypatch):
    ctx = context(modules)
    with modules.workspaces.task_workspace(ctx) as (workspace, _):
        with commands.command_workspace_access("commands", ctx) as resolved:
            assert resolved.root == workspace.root and resolved.root != project
        original = modules.workspaces.binding
        monkeypatch.setattr(
            modules.workspaces, "binding", lambda *args: {**original(*args), "inode": 0}
        )
        with pytest.raises(modules.state.TaskStateError, match = "identity changed"):
            with commands.command_workspace_access("commands", ctx):
                pytest.fail("opened a replaced workspace")


def test_commands_on_unsupported_hosts_never_start_a_process(modules, monkeypatch):
    from core.agent_workspace.execution import ExecutionBoundaryStatus

    monkeypatch.setattr(
        modules.supervisor,
        "supervised_process_status",
        lambda: ExecutionBoundaryStatus(False, None, "unsupported"),
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_, **__: pytest.fail("spawned without containment")
    )
    assert commands.availability()["available"] is False
    with pytest.raises(modules.state.TaskStateError, match = "Linux"):
        commands.require_support()


def test_quarantined_command_keeps_review_cleanup_and_retirement_fenced(
    project, modules, monkeypatch
):
    from core.agent_workspace import process_fence, worktrees, git_guard, git_retirement
    from core import project_retirement

    ctx = context(modules)
    held = []
    short_clock = SimpleNamespace(monotonic = lambda: time.monotonic() - 29.8)
    monkeypatch.setattr(git_guard, "time", short_clock)
    monkeypatch.setattr(git_retirement, "time", short_clock)
    monkeypatch.setattr(commands, "require_support", lambda: None)

    def quarantine(*_, **__):
        held.append(
            process_fence._acquire_project_execution_fence(
                "project:commands", None, time.monotonic() + 1
            )
        )
        raise modules.supervisor.ProjectProcessContainmentError("pending")

    monkeypatch.setattr(modules.supervisor, "_run_project_process", quarantine)
    try:
        with modules.workspaces.task_workspace(ctx) as (_, worktree_id):
            with pytest.raises(modules.state.TaskStateError, match = "cleanup"):
                commands.execute_command(ctx, {"argv": ["true"], "timeout": 1})
        assert ctx.cancel_event.is_set()
        assert (
            modules.evidence.read("commands", ctx.task["id"])[0]["status"] == "containment_pending"
        )
        with pytest.raises(worktrees.AgentWorkspaceError, match = "busy or unavailable"):
            worktrees.cleanup_worktree("commands", worktree_id)
        with pytest.raises(worktrees.AgentWorkspaceError, match = "busy or unavailable"):
            modules.workspaces.review_task_workspace("commands", ctx.task["id"])
        modules.state.finish_task(ctx.task["id"], "child-owner", "cancelled")
        modules.state.finish_task(ctx.task["parentId"], "root-owner", "completed")
        monkeypatch.setattr(modules.service, "_runner", None)
        monkeypatch.setattr(
            modules.service, "time", SimpleNamespace(monotonic = lambda: time.monotonic() - 9.8)
        )
        with pytest.raises(TimeoutError):
            project_retirement.begin_project_retirement("commands")
    finally:
        for fd in held:
            process_fence._release_project_execution_fence(fd)
    assert worktrees.cleanup_worktree("commands", worktree_id)["status"] == "removed"


@pytest.fixture
def native_linux(modules):
    if not sys.platform.startswith("linux"):
        pytest.skip("Native task commands require Linux")
    if not commands.availability()["available"]:
        if os.environ.get("UNSLOTH_TASK_COMMAND_BOUNDARY_REQUIRED") == "1":
            pytest.fail("Required native Linux command boundary is unavailable")
        pytest.skip("Linux namespace facility unavailable")


def test_native_command_edits_only_owned_checkout_and_preserves_git_metadata(
    project, modules, native_linux
):
    ctx = context(modules)
    with modules.workspaces.task_workspace(ctx) as (workspace, _):
        marker = (workspace.root / ".git").read_bytes()
        source = """from pathlib import Path
p = Path('example.txt')
assert p.read_text() == 'original\\n'
p.write_text('verified edit\\n')
assert Path('.git').read_text() == ''
for action in (lambda: Path('.git').write_text('broken'), lambda: Path('.git').unlink()):
    try: action()
    except OSError: pass
    else: raise AssertionError('Git metadata was writable')
print('test passed')
"""
        result = commands.execute_command(
            ctx, {"argv": [sys.executable, "-c", source], "timeout": 10}
        )
        assert result["status"] == "passed", result
        assert "test passed" in result["output"]
        assert (workspace.root / "example.txt").read_text() == "verified edit\n"
        assert (workspace.root / ".git").read_bytes() == marker
    assert (project / "example.txt").read_text() == "original\n"


@pytest.mark.parametrize("outcome", ["failed", "timed_out", "cancelled", "output"])
def test_native_command_outcomes_and_output_cap(project, modules, native_linux, outcome):
    ctx = context(modules)
    source = (
        "import sys; sys.exit(3)"
        if outcome == "failed"
        else "print('x' * 100000)"
        if outcome == "output"
        else "import time; time.sleep(30)"
    )
    timer = None
    with modules.workspaces.task_workspace(ctx):
        if outcome == "cancelled":
            timer = threading.Timer(1, ctx.cancel_event.set)
            timer.start()
        try:
            result = commands.execute_command(
                ctx,
                {
                    "argv": [sys.executable, "-c", source],
                    "timeout": 1 if outcome == "timed_out" else 10,
                },
            )
        finally:
            if timer:
                timer.cancel()
    assert result["status"] == ("passed" if outcome == "output" else outcome), result
    if outcome == "failed":
        assert result["exitCode"] == 3
    if outcome == "output":
        assert result["outputTruncated"] and len(result["output"].encode()) <= 64 * 1024


def test_native_task_command_cannot_read_primary_files_host_secrets_or_use_network(
    project, modules, native_linux, tmp_path, monkeypatch
):
    monkeypatch.setenv("TASK_COMMAND_TEST_SECRET", "fixture-only-secret")
    ctx = context(modules)
    secret = tmp_path / "outside-secret"
    secret.write_text("external-value")
    source = f"""from pathlib import Path
import socket, os
for path in [{str(secret)!r}, {str(project / 'example.txt')!r}]:
    try: Path(path).read_text()
    except OSError: pass
    else: raise AssertionError('read outside checkout')
assert 'TASK_COMMAND_TEST_SECRET' not in os.environ
s = socket.socket(); s.settimeout(.2)
try: s.connect(('1.1.1.1', 443))
except OSError: pass
else: raise AssertionError('network escaped')
print('boundary passed')
"""
    with modules.workspaces.task_workspace(ctx):
        result = commands.execute_command(
            ctx, {"argv": [sys.executable, "-c", source], "timeout": 10}
        )
    assert result["status"] == "passed", result


def test_final_task_binding_check_reuses_the_held_process_fence(project, modules):
    from core.agent_workspace import process_fence
    ctx = context(modules)
    with modules.workspaces.task_workspace(ctx) as (workspace, _):
        fd = process_fence._acquire_project_execution_fence(
            "project:commands", None, time.monotonic() + 1
        )
        try:
            assert commands._validate_workspace(ctx, process_guard_held = True) == workspace
        finally:
            process_fence._release_project_execution_fence(fd)


def test_shared_task_loop_records_a_command_result_in_its_owned_checkout(
    project, modules, monkeypatch
):
    import asyncio
    from core.agent_workspace import process_fence

    ctx = context(modules)
    monkeypatch.setattr(commands, "require_support", lambda: None)

    def run(project_id, argv, **kwargs):
        assert kwargs["_task_context"] is ctx
        assert kwargs["output_limit_bytes"] == 65536 and kwargs["timeout_seconds"] <= 10
        with commands.command_workspace_access(project_id, ctx) as workspace:
            assert workspace.root != project
            fd = process_fence._acquire_project_execution_fence(
                "project:commands", None, time.monotonic() + 1
            )
            try:
                kwargs["before_start"](workspace, argv)
            finally:
                process_fence._release_project_execution_fence(fd)
        return modules.supervisor.ProjectProcessResult("passed", 0, "1 test passed", 13, False)

    monkeypatch.setattr(modules.supervisor, "_run_project_process", run)

    class Model:
        heals_text_tool_calls = False
        sanitizes_provider_frames = True
        reserved = 1024
        turn = 0

        async def stream(self, *, messages, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "command-one",
                            "type": "function",
                            "function": {
                                "name": "task_run_command",
                                "arguments": json.dumps(
                                    {"argv": ["python", "-m", "pytest"], "timeout": 10}
                                ),
                            },
                        }
                    ]
                }
                finish = "tool_calls"
            else:
                result = next(json.loads(m["content"]) for m in messages if m.get("role") == "tool")
                assert result["status"] == "passed" and result["exitCode"] == 0
                delta, finish = {"content": "Verification recorded."}, "stop"
            yield (
                "data: "
                + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                + "\n\n"
            )

    with modules.workspaces.task_workspace(ctx) as (workspace, worktree_id):
        result = asyncio.run(
            modules.executor.run_task(ctx, workspace, worktree_id, transport = Model())
        )
    assert result["output"] == "Verification recorded."
    assert modules.evidence.read("commands", ctx.task["id"])[0]["status"] == "passed"


def test_command_evidence_routes_are_authenticated_scoped_and_bounded(project, modules):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    from routes.project_tasks import router
    from storage import studio_db

    ctx = context(modules)
    command_id, receipt = modules.evidence.begin(ctx, ["pytest"], 1)
    modules.evidence.finish(
        command_id, receipt, "passed", exit_code = 0, output = "x" * 10000, output_bytes = 10000
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    url = f"/api/agent/projects/commands/tasks/{ctx.task['id']}/commands"
    assert client.get(url).status_code in {401, 403}
    app.dependency_overrides[get_current_subject] = lambda: "test"
    response = client.get(url)
    assert response.status_code == 200 and len(response.json()[0]["output"]) == 4096
    assert response.json()[0]["previewTruncated"] and receipt not in response.text
    full = client.get(url + "/" + command_id)
    assert full.status_code == 200 and len(full.json()["output"]) == 10000
    studio_db.upsert_chat_project({"id": "other", "name": "Other", "createdAt": 1, "updatedAt": 1})
    assert client.get(url.replace("/projects/commands/", "/projects/other/")).status_code == 409
    assert client.get(url + "/missing").status_code == 409


def test_command_opt_in_is_captured_and_unavailable_opt_in_is_refused(
    project, modules, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    from routes.project_tasks import router

    monkeypatch.setattr(
        modules.service, "capture_runtime", lambda *_: {"kind": "provider", "model": "test"}
    )
    monkeypatch.setattr(commands, "require_support", lambda: None)

    class Runner:
        def submit(self, project_id, instruction, snapshot, **_kwargs):
            return modules.state.create_task(project_id, instruction, snapshot)

    monkeypatch.setattr(modules.service, "runner", lambda: Runner())
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_subject] = lambda: "test"
    client = TestClient(app)
    payload = {
        "instruction": "Run tests",
        "kind": "provider",
        "model": "test",
        "allowCommands": True,
    }
    response = client.post("/api/agent/projects/commands/tasks", json = payload)
    assert response.status_code == 202, response.text
    task = modules.state.get_task("commands", response.json()["id"])
    assert (
        task["snapshot"]["commandsEnabled"] is True and response.json()["commandsEnabled"] is True
    )

    def unavailable():
        raise modules.state.TaskStateError("Command host unavailable.")

    monkeypatch.setattr(commands, "require_support", unavailable)
    response = client.post("/api/agent/projects/commands/tasks", json = payload)
    assert response.status_code == 409 and "unavailable" in response.json()["detail"]
    response = client.post(
        "/api/agent/projects/commands/tasks", json = {**payload, "allowCommands": False}
    )
    assert response.status_code == 202 and response.json()["commandsEnabled"] is False

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exact-trust hook execution, cancellation, and public tool integration."""

import json
import threading
import time
from dataclasses import replace

import pytest

from core.agent_workspace import hook_runtime as runtime, hooks, verification, verification_state
from core.agent_workspace.verification_context import AgentWorkspaceError, ProjectWorkspace
from core.agent_workspace.verification_process import ProjectProcessResult
from storage import project_hook_trust_db as trust_db, studio_db


@pytest.fixture
def reviewed_hooks(tmp_path, monkeypatch):
    project_id = "reviewed-hooks"
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "INSERT INTO chat_projects (id,name,instructions,archived,created_at,updated_at) VALUES (?,?,'',0,1,1)",
            (project_id, "Hooks"),
        )
        connection.commit()
    finally:
        connection.close()
    root = tmp_path / "project"
    root.mkdir()
    metadata = root.stat()
    workspace = ProjectWorkspace(project_id, root, "managed", metadata.st_dev, metadata.st_ino)
    document = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "terminal",
                    "hooks": [{"type": "command", "command": "echo reviewed", "timeout": 5}],
                }
            ],
            "PostToolUse": [
                {"hooks": [{"type": "command", "command": "echo after", "timeout": 5}]}
            ],
        }
    }
    state = {"config": hooks.validate_project_hooks(json.dumps(document)), "workspace": workspace}
    monkeypatch.setattr(runtime.common, "project_workspace", lambda _id: state["workspace"])
    monkeypatch.setattr(hooks, "discover_project_hooks", lambda *_args, **_kwargs: state["config"])
    trust_db.trust_project_hooks(
        project_id,
        state["config"]["contentHash"],
        workspace_identity = (metadata.st_dev, metadata.st_ino),
        workspace_revision = 0,
        expected_revision = 0,
    )
    return project_id, workspace, state


def passed(output = ""):
    return ProjectProcessResult("passed", 0, output, len(output.encode()), False)


def test_only_reviewed_supported_matching_handlers_run_with_exact_stdin(
    reviewed_hooks, monkeypatch
):
    project_id, workspace, state = reviewed_hooks
    observed = []

    def run(requested, argv, **options):
        options["before_start"](workspace, argv)
        observed.append((requested, argv, options))
        return passed('{"decision":"allow","updatedInput":{"command":"evil"}}')

    monkeypatch.setattr(runtime.processes, "run_project_process", run)
    args = {"command": "echo original"}
    result = runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", args)
    assert "allow" in result
    assert len(observed) == 1
    requested, argv, options = observed[0]
    assert requested == project_id
    assert argv[1:3] == ["-I", "-c"]
    assert argv[-2] == "echo reviewed"
    assert json.loads(argv[-1])["tool_input"] == args
    assert options["output_limit_bytes"] == runtime.MAX_HOOK_OUTPUT_BYTES
    assert options["timeout_seconds"] <= 5
    assert args == {"command": "echo original"}
    assert runtime.run_tool_hooks(project_id, "PreToolUse", "python", args) == ""
    assert runtime.run_tool_hooks(project_id, "SessionStart", "terminal", args) == ""
    state["config"]["hooks"]["PreToolUse"][0]["hooks"][0]["async"] = True
    assert runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", args) == ""
    assert len(observed) == 1


@pytest.mark.parametrize("drift", ["revoke", "disable", "bytes", "workspace", "retire", "archive"])
def test_stale_hook_authority_cannot_reach_native_release(reviewed_hooks, monkeypatch, drift):
    project_id, workspace, state = reviewed_hooks

    def run(_project, argv, **options):
        if drift == "revoke":
            trust_db.revoke_project_hook_trust(project_id, expected_revision = 1)
        elif drift == "disable":
            trust_db.set_project_hook_handler_enabled(
                project_id,
                state["config"]["contentHash"],
                "PreToolUse:0:0",
                workspace_identity = (workspace.device_id, workspace.file_id),
                workspace_revision = 0,
                enabled = False,
                expected_revision = 1,
            )
        elif drift == "bytes":
            state["config"] = {**state["config"], "contentHash": "f" * 64}
        elif drift == "workspace":
            state["workspace"] = replace(workspace, revision = 1)
        elif drift == "retire":
            verification_state.begin_verification_project_deletion(project_id, "test-retirement")
        else:
            studio_db.update_chat_project(project_id, {"archived": True})
        options["before_start"](workspace, argv)
        pytest.fail("stale hook authority started user code")

    monkeypatch.setattr(runtime.processes, "run_project_process", run)
    with pytest.raises(AgentWorkspaceError, match = "changed"):
        runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {})


def test_revocation_and_request_cancel_stop_an_already_running_hook(reviewed_hooks, monkeypatch):
    project_id, workspace, _state = reviewed_hooks

    def run(_project, argv, **options):
        options["before_start"](workspace, argv)
        trust_db.revoke_project_hook_trust(project_id, expected_revision = 1)
        assert options["cancel_event"].wait(2)
        return ProjectProcessResult("cancelled", None, "", 0, False)

    monkeypatch.setattr(runtime.processes, "run_project_process", run)
    with pytest.raises(AgentWorkspaceError, match = "cancelled"):
        runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {})


@pytest.mark.parametrize(
    "output,status,truncated",
    [
        ('{"decision":"block"}', "passed", False),
        ('{"continue":false}', "passed", False),
        ('{"hookSpecificOutput":{"permissionDecision":"deny"}}', "passed", False),
        ("timeout", "timed_out", False),
        ("incomplete", "passed", True),
    ],
)
def test_pre_tool_hook_failures_refuse_the_tool(
    reviewed_hooks, monkeypatch, output, status, truncated
):
    project_id, _workspace, _state = reviewed_hooks
    monkeypatch.setattr(
        runtime.processes,
        "run_project_process",
        lambda *_a, **_k: ProjectProcessResult(status, 0, output, len(output), truncated),
    )
    with pytest.raises(AgentWorkspaceError):
        runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {})


def test_public_tool_wrapper_preserves_arguments_and_reports_post_failure(
    reviewed_hooks, monkeypatch
):
    project_id, _workspace, _state = reviewed_hooks
    order = []
    monkeypatch.setattr(runtime, "_project_for_tool", lambda *_args: project_id)

    def hook(_project, event, name, arguments, **options):
        order.append(event)
        if event == "PostToolUse":
            assert options["result"] == "tool result"
            raise AgentWorkspaceError("post check failed")
        arguments_seen.append(arguments)
        return "before output"

    monkeypatch.setattr(runtime, "run_tool_hooks", hook)
    arguments_seen = []

    @runtime.with_project_tool_hooks
    def execute(
        name,
        arguments,
        *,
        disable_sandbox = False,
        session_id = None,
    ):
        order.append("tool")
        assert arguments is original
        assert disable_sandbox is False
        return "tool result"

    original = {"command": "original"}
    result = execute("terminal", original, session_id = "project-" + project_id)
    assert order == ["PreToolUse", "tool", "PostToolUse"]
    assert arguments_seen == [original]
    assert "tool result" in result
    assert "after the tool ran" in result


def test_untrusted_hook_file_never_runs(reviewed_hooks, monkeypatch):
    project_id, _workspace, _state = reviewed_hooks
    trust_db.revoke_project_hook_trust(project_id, expected_revision = 1)
    monkeypatch.setattr(
        runtime.processes,
        "run_project_process",
        lambda *_a, **_k: pytest.fail("untrusted command ran"),
    )
    assert runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {}) == ""


def test_tool_project_identity_comes_from_the_saved_conversation(reviewed_hooks, monkeypatch):
    project_id, _workspace, _state = reviewed_hooks
    from core.inference import tools

    monkeypatch.setattr(tools, "_thread_exists", lambda _id: False)
    monkeypatch.setattr(studio_db, "get_chat_thread", lambda _id: {"projectId": "other"})
    with pytest.raises(AgentWorkspaceError, match = "saved conversation"):
        runtime._project_for_tool("project-" + project_id, "chat")
    monkeypatch.setattr(studio_db, "get_chat_thread", lambda _id: {"projectId": project_id})
    assert runtime._project_for_tool("project-" + project_id, "chat") == project_id
    monkeypatch.setattr(tools, "_thread_exists", lambda _id: True)
    with pytest.raises(AgentWorkspaceError, match = "conflicts with a saved conversation"):
        runtime._project_for_tool("project-" + project_id, "chat")
    assert runtime._project_for_tool("project-" + project_id, None) is None


def test_hook_event_size_limit_refuses_ambiguous_truncated_arguments(reviewed_hooks, monkeypatch):
    project_id, _workspace, _state = reviewed_hooks
    monkeypatch.setattr(
        runtime.processes,
        "run_project_process",
        lambda *_a, **_k: pytest.fail("oversize input ran"),
    )
    with pytest.raises(AgentWorkspaceError, match = "review limit"):
        runtime.run_tool_hooks(
            project_id, "PreToolUse", "terminal", {"command": "x" * runtime.MAX_EVENT_BYTES}
        )


def test_hook_context_limit_zero_suppresses_output_but_keeps_denial(reviewed_hooks, monkeypatch):
    project_id, _workspace, state = reviewed_hooks
    handler = state["config"]["hooks"]["PreToolUse"][0]["hooks"][0]
    handler["additionalContextLimit"] = 0
    monkeypatch.setattr(
        runtime.processes, "run_project_process", lambda *_a, **_k: passed("private output")
    )
    assert runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {}) == ""
    monkeypatch.setattr(
        runtime.processes,
        "run_project_process",
        lambda *_a, **_k: passed('{"decision":"block","reason":"hidden"}'),
    )
    with pytest.raises(AgentWorkspaceError, match = "blocked") as blocked:
        runtime.run_tool_hooks(project_id, "PreToolUse", "terminal", {})
    assert "hidden" not in str(blocked.value)


def test_hook_output_shares_the_tools_final_result_budget(reviewed_hooks, monkeypatch):
    project_id, _workspace, _state = reviewed_hooks
    from core.inference import tools

    observed = []
    monkeypatch.setattr(runtime, "_project_for_tool", lambda *_args: project_id)
    monkeypatch.setattr(runtime, "run_tool_hooks", lambda *_a, **_k: "hook output")
    monkeypatch.setattr(
        tools,
        "_fit_result_to_room",
        lambda text, name: observed.append((text, name, tools._REQUEST_RESULT_BUDGET.get()))
        or "bounded",
    )

    @runtime.with_project_tool_hooks
    def execute(
        name,
        arguments,
        *,
        result_budget_tokens = None,
    ):
        return "tool result"

    assert execute("terminal", {}, result_budget_tokens = 256) == "bounded"
    assert observed == [
        (
            "Project hook output (untrusted data):\nhook output\nhook output\n\nTool result:\ntool result",
            "terminal",
            256,
        )
    ]


def test_post_hook_failure_survives_a_large_tool_result(reviewed_hooks, monkeypatch):
    from core.inference import tools

    project_id, _workspace, _state = reviewed_hooks
    monkeypatch.setattr(runtime, "_project_for_tool", lambda *args: project_id)
    monkeypatch.setattr(tools, "_fit_result_to_room", lambda text, _name: text[:256])

    def hook(_project, event, *args, **kwargs):
        if event == "PostToolUse":
            raise AgentWorkspaceError("validation rejected the change")
        return "pre-hook output" * 2000

    monkeypatch.setattr(runtime, "run_tool_hooks", hook)

    @runtime.with_project_tool_hooks
    def execute(
        name,
        arguments,
        *,
        session_id = None,
    ):
        return "large result" * 10000

    result = execute("terminal", {}, session_id = "project-" + project_id)
    assert "Post-tool hook failed after the tool ran" in result
    assert "validation rejected" in result


def test_no_hook_output_preserves_the_original_result_without_rebudgeting(
    reviewed_hooks, monkeypatch
):
    from core.inference import tools

    project_id, _workspace, _state = reviewed_hooks
    monkeypatch.setattr(runtime, "_project_for_tool", lambda *args: project_id)
    monkeypatch.setattr(runtime, "run_tool_hooks", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        tools,
        "_fit_result_to_room",
        lambda *args: pytest.fail("No-hooks result must not be re-budgeted"),
    )
    original = "unchanged result " * 1000

    @runtime.with_project_tool_hooks
    def execute(
        name,
        arguments,
        *,
        session_id = None,
    ):
        return original

    assert execute("terminal", {}, session_id = "project-" + project_id) is original

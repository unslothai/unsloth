# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from contextlib import nullcontext

import pytest

from core.agent_workspace import hook_runtime as runtime


@pytest.mark.parametrize("text", ["界", "😀", '"', "\\", "\n", "a"])
@pytest.mark.parametrize("argument_size", [0, 15_000])
def test_post_hook_response_fits_serialized_budget_without_changing_arguments(text, argument_size):
    arguments = {"command": "a" * argument_size}
    result = text * 4096
    encoded = runtime._event_payload("PostToolUse", "p", "terminal", arguments, result)
    assert len(encoded.encode("utf-8")) <= runtime.MAX_EVENT_BYTES
    payload = json.loads(encoded)
    assert payload["tool_input"] == arguments
    assert result.startswith(payload["tool_response"])
    assert payload["tool_response"]
    assert payload["tool_response_truncated"] == (payload["tool_response"] != result)


def test_post_hook_trust_failure_retains_completed_tool_context(monkeypatch):
    monkeypatch.setattr(runtime, "_project_for_tool", lambda *_: "p")
    monkeypatch.setattr(
        runtime.project_hook_trust_db,
        "get_project_hook_trust_record",
        lambda _: {"hasStoredTrust": True},
    )
    monkeypatch.setattr(runtime.common, "project_workspace_access", lambda _: nullcontext())
    calls = []

    def hooks(_project, event, *_args, **_kwargs):
        if event == "PostToolUse":
            raise runtime.project_hook_trust_db.ProjectHookTrustStateError("trust row corrupt")
        return ""

    monkeypatch.setattr(runtime, "run_tool_hooks", hooks)

    @runtime.with_project_tool_hooks
    def execute(name, arguments, session_id):
        calls.append(arguments)
        return "Edited file.py"

    result = execute("edit_file", {"path": "file.py"}, "project-p")
    assert len(calls) == 1
    assert "Post-tool hook failed after the tool ran" in result
    assert "Edited file.py" in result
    assert not result.startswith("Error:")


@pytest.mark.parametrize("name", ["edit_file", "python", "terminal", "web_search"])
@pytest.mark.parametrize("session_id", [None, "chat-session", "project-unconfigured"])
def test_unconfigured_tools_keep_original_arguments_results_and_routing(
    monkeypatch, name, session_id
):
    lookups = []

    def trust(project_id):
        lookups.append(project_id)
        return {"hasStoredTrust": False}

    monkeypatch.setattr(runtime.project_hook_trust_db, "get_project_hook_trust_record", trust)
    monkeypatch.setattr(
        runtime, "_project_for_tool", lambda *_args: pytest.fail("No extra project/thread routing")
    )
    monkeypatch.setattr(
        runtime.common,
        "project_workspace_access",
        lambda *_args: pytest.fail("No extra workspace lease"),
    )
    monkeypatch.setattr(
        runtime, "run_tool_hooks", lambda *_args, **_kwargs: pytest.fail("No hook execution")
    )
    arguments = {"code": "original"}
    original = "  original result\r\n" * 1000

    def raw(name, arguments, *, session_id):
        assert arguments is expected_arguments
        return original

    expected_arguments = arguments
    wrapped = runtime.with_project_tool_hooks(raw)
    assert wrapped(name, arguments, session_id = session_id) is original
    assert lookups == (
        ["unconfigured"]
        if name in runtime.HOOKED_TOOLS and session_id == "project-unconfigured"
        else []
    )

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

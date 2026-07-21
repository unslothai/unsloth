# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

import unsloth_cli.claude_subagent_mcp as bridge


def test_protocol_lists_and_calls_local_agent():
    initialized = bridge._response(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert initialized["result"]["serverInfo"]["name"] == "unsloth-local-agent"

    listed = bridge._response({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tool = listed["result"]["tools"][0]
    assert tool["name"] == "unsloth_agent"
    assert "spawn an Unsloth or local agent" in tool["description"]
    assert tool["inputSchema"]["required"] == ["task"]
    assert tool["_meta"]["anthropic/maxResultSizeChars"] == 100_000

    called = bridge._response(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "unsloth_agent", "arguments": {"task": " inspect this "}},
        },
        run_agent = lambda task: f"completed: {task}",
    )
    assert called["result"] == {
        "content": [{"type": "text", "text": "completed: inspect this"}],
        "isError": False,
    }


def test_protocol_returns_tool_errors_to_parent():
    response = bridge._response(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "unsloth_agent", "arguments": {"task": "test"}},
        },
        run_agent = lambda task: (_ for _ in ()).throw(RuntimeError("local failure")),
    )
    assert response["result"]["isError"] is True
    assert response["result"]["content"][0]["text"] == "local failure"


def test_stdio_server_ignores_notifications_and_answers_requests():
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}),
        ]
    )
    output = io.StringIO()
    bridge.serve(io.StringIO(requests), output)
    assert json.loads(output.getvalue()) == {"jsonrpc": "2.0", "id": 4, "result": {}}


@pytest.mark.parametrize(
    ("bypass", "permission"),
    [("0", "acceptEdits"), ("1", "bypassPermissions")],
)
def test_local_child_uses_unsloth_without_overwriting_parent_auth(
    monkeypatch, tmp_path, bypass, permission
):
    captured = {}
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_BASE_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_API_KEY", "sk-unsloth-test")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_MODEL", "unsloth/model-GGUF:Q4_K_M")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS", bypass)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cloud-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "cloud-oauth")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model: ["--settings", "{}"])

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode = 0,
            stdout = json.dumps({"is_error": False, "result": "LOCAL_OK"}),
            stderr = "",
        )

    monkeypatch.setattr(bridge.subprocess, "run", run)
    assert bridge.run_local_agent("reply exactly LOCAL_OK") == "LOCAL_OK"
    command = captured["command"]
    assert command[:3] == [
        "/usr/local/bin/claude",
        "--model",
        "unsloth/model-GGUF:Q4_K_M",
    ]
    assert command[command.index("--permission-mode") + 1] == permission
    assert "--no-session-persistence" in command
    assert captured["cwd"] == str(tmp_path)
    child_env = captured["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8888"
    assert child_env["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-test"
    assert child_env["ANTHROPIC_MODEL"] == "unsloth/model-GGUF:Q4_K_M"
    assert child_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "32768"
    assert child_env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child_env


def test_result_parser_accepts_diagnostics_before_json():
    output = "connector warning\n" + json.dumps({"is_error": False, "result": "OK"})
    assert bridge._result_text(output) == "OK"

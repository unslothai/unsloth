# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
import os
import subprocess

import pytest

import unsloth_cli.codex_subagent_mcp as bridge


def _write_config(tmp_path, *, bypass_permissions = False):
    path = tmp_path / "subagent.json"
    path.write_text(
        json.dumps(
            {
                "api_key": "sk-unsloth-test",
                "codex_home": str(tmp_path / "child"),
                "bypass_permissions": bypass_permissions,
            }
        )
    )
    return path


def test_protocol_uses_codex_specific_tool_name():
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": bridge._CODEX_SUBAGENT_MCP_TOOL,
                        "arguments": {"task": " inspect this "},
                    },
                }
            ),
        ]
    )
    output = io.StringIO()
    bridge.serve(
        io.StringIO(requests),
        output,
        run_agent = lambda task, cancel_event: f"completed: {task}",
        tool_name = bridge._CODEX_SUBAGENT_MCP_TOOL,
    )
    responses = {
        response["id"]: response for response in map(json.loads, output.getvalue().splitlines())
    }
    assert responses[1]["result"]["tools"][0]["name"] == "spawn_local_agent"
    assert responses[2]["result"] == {
        "content": [{"type": "text", "text": "completed: inspect this"}],
        "isError": False,
    }


@pytest.mark.parametrize("bypass_permissions", [False, True])
def test_local_child_uses_explicit_unsloth_profile(monkeypatch, tmp_path, bypass_permissions):
    config = _write_config(tmp_path, bypass_permissions = bypass_permissions)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-key")
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = {}

    class Process:
        pid = 1234
        returncode = 0

        def communicate(self, timeout):
            captured["timeout"] = timeout
            return (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "LOCAL_OK"},
                    }
                ),
                "",
            )

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("reply exactly LOCAL_OK") == "LOCAL_OK"
    command = captured["command"]
    assert command[:4] == ["/usr/local/bin/codex", "--oss", "--profile", "unsloth_api"]
    if bypass_permissions:
        assert "--dangerously-bypass-approvals-and-sandbox" in command
    else:
        assert command[4:8] == ["--sandbox", "workspace-write", "--ask-for-approval", "never"]
    assert command[command.index("exec") + 1 : command.index("exec") + 4] == [
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
    ]
    assert command[-1].endswith("Task: reply exactly LOCAL_OK")
    assert captured["cwd"] == os.getcwd()
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE
    if os.name == "nt":
        assert captured["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured["start_new_session"] is True
    assert captured["env"]["CODEX_HOME"] == str(tmp_path / "child")
    assert captured["env"][bridge._CODEX_ENV_KEY] == "sk-unsloth-test"


def test_local_child_returns_last_agent_message():
    output = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "intermediate"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "final"},
                }
            ),
        ]
    )
    assert bridge._result_text(output) == "final"


def test_local_child_reports_failed_turn():
    output = json.dumps({"type": "turn.failed", "error": {"message": "local failure"}})
    with pytest.raises(RuntimeError, match = "local failure"):
        bridge._result_text(output)


def test_local_child_process_is_stopped_on_cancellation(monkeypatch, tmp_path):
    config = _write_config(tmp_path)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    cancel_event = bridge.threading.Event()
    stopped = []

    class Process:
        pid = 1234
        returncode = None

        def communicate(self, timeout):
            cancel_event.set()
            raise subprocess.TimeoutExpired("codex", timeout)

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)

    def stop(child):
        stopped.append(child)
        child.returncode = -15

    monkeypatch.setattr(bridge, "_stop_child", stop)
    with pytest.raises(RuntimeError, match = "cancelled"):
        bridge.run_local_agent("wait", cancel_event)
    assert stopped == [process]

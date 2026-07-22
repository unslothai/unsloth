# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
import os

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


def test_stdio_cancellation_reaches_the_running_local_agent():
    requests = "\n".join(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "call-1",
                    "method": "tools/call",
                    "params": {"name": "unsloth_agent", "arguments": {"task": "wait"}},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": "call-1", "reason": "user cancelled"},
                }
            ),
        ]
    )
    output = io.StringIO()
    cancelled = []

    def run_agent(task, cancel_event):
        assert task == "wait"
        assert cancel_event.wait(timeout = 1)
        cancelled.append(task)
        raise RuntimeError("The local Claude agent was cancelled.")

    bridge.serve(io.StringIO(requests), output, run_agent = run_agent)
    assert cancelled == ["wait"]
    assert output.getvalue() == ""


def test_stdio_sigint_stops_the_running_local_agent(monkeypatch):
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {"name": "unsloth_agent", "arguments": {"task": "wait"}},
        }
    )
    handlers = {}
    started = bridge.threading.Event()
    cancelled = []

    def set_handler(signum, handler):
        previous = handlers.get(signum, bridge.signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(bridge.signal, "signal", set_handler)

    class InterruptingInput:
        def __init__(self):
            self.sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self.sent:
                self.sent = True
                return request + "\n"
            assert started.wait(timeout = 1)
            handlers[bridge.signal.SIGINT](bridge.signal.SIGINT, None)
            raise AssertionError("SIGINT handler must unwind the stdin loop")

    def run_agent(task, cancel_event):
        assert task == "wait"
        started.set()
        assert cancel_event.wait(timeout = 1)
        # Real Claude Code sends SIGINT twice. The second one must not abort cleanup.
        handlers[bridge.signal.SIGINT](bridge.signal.SIGINT, None)
        cancelled.append(task)
        raise RuntimeError("The local Claude agent was cancelled.")

    output = io.StringIO()
    bridge.serve(InterruptingInput(), output, run_agent = run_agent)
    assert cancelled == ["wait"]
    assert output.getvalue() == ""


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

    class Process:
        pid = 1234
        returncode = 0

        def communicate(self, timeout):
            captured["timeout"] = timeout
            return json.dumps({"is_error": False, "result": "LOCAL_OK"}), ""

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("reply exactly LOCAL_OK") == "LOCAL_OK"
    command = captured["command"]
    assert command[:3] == ["/usr/local/bin/claude", "--model", "unsloth/model-GGUF:Q4_K_M"]
    assert command[command.index("--permission-mode") + 1] == permission
    assert "--no-session-persistence" in command
    assert captured["cwd"] == str(tmp_path)
    assert captured["stdout"] is bridge.subprocess.PIPE
    assert captured["stderr"] is bridge.subprocess.PIPE
    if os.name == "nt":
        assert captured["creationflags"] == bridge.subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured["start_new_session"] is True
    child_env = captured["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8888"
    assert child_env["ANTHROPIC_AUTH_TOKEN"] == "sk-unsloth-test"
    assert child_env["ANTHROPIC_MODEL"] == "unsloth/model-GGUF:Q4_K_M"
    assert child_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "32768"
    assert child_env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in child_env


def test_local_child_process_is_stopped_on_cancellation(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_BASE_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_API_KEY", "sk-unsloth-test")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_MODEL", "unsloth/model-GGUF:Q4_K_M")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model: [])
    cancel_event = bridge.threading.Event()
    stopped = []

    class Process:
        pid = 1234
        returncode = None

        def communicate(self, timeout):
            cancel_event.set()
            raise bridge.subprocess.TimeoutExpired("claude", timeout)

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


def test_windows_cancellation_stops_the_child_process_tree(monkeypatch):
    monkeypatch.setattr(bridge.os, "name", "nt")
    captured = {}

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout = None):
            captured["wait_timeout"] = timeout
            self.returncode = 1

        def terminate(self):
            raise AssertionError("taskkill should handle the process tree")

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(bridge.subprocess, "run", run)
    bridge._stop_child(Process())

    assert captured["command"] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert captured["capture_output"] is True
    assert captured["check"] is False
    assert captured["wait_timeout"] == bridge._CANCEL_GRACE_SECONDS


def test_result_parser_accepts_diagnostics_before_json():
    output = "connector warning\n" + json.dumps({"is_error": False, "result": "OK"})
    assert bridge._result_text(output) == "OK"

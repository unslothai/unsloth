# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

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


def test_protocol_exposes_read_only_agent_for_claude_plan_mode():
    listed = bridge._response(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        run_read_only_agent = lambda task: task,
        read_only_tool_name = "unsloth_plan_agent",
    )
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert tools["unsloth_agent"]["annotations"]["readOnlyHint"] is False
    assert tools["unsloth_plan_agent"]["annotations"] == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }

    called = bridge._response(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "unsloth_plan_agent",
                "arguments": {"task": " inspect this "},
            },
        },
        run_agent = lambda task: f"write: {task}",
        run_read_only_agent = lambda task: f"plan: {task}",
        read_only_tool_name = "unsloth_plan_agent",
    )
    assert called["result"] == {
        "content": [{"type": "text", "text": "plan: inspect this"}],
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
    assert captured["stdin"] is bridge.subprocess.DEVNULL
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


def test_read_only_local_child_uses_plan_mode(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_BASE_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_API_KEY", "sk-unsloth-test")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_MODEL", "unsloth/model-GGUF:Q4_K_M")
    monkeypatch.setenv("UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(bridge, "_claude_flags", lambda model: [])

    class Process:
        pid = 1234
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"is_error": False, "result": "PLAN_OK"}), ""

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        captured["command"] = command
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("plan this", read_only = True) == "PLAN_OK"
    command = captured["command"]
    assert command[command.index("--permission-mode") + 1] == "plan"
    prompt = command[command.index("--append-system-prompt") + 1]
    assert "read-only local coding subagent" in prompt


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
        return bridge.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bridge.subprocess, "run", run)
    bridge._stop_child(Process())

    assert captured["command"] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert captured["capture_output"] is True
    assert captured["check"] is False
    assert captured["wait_timeout"] == bridge._CANCEL_GRACE_SECONDS


def test_windows_failed_taskkill_still_terminates_the_child(monkeypatch):
    monkeypatch.setattr(bridge.os, "name", "nt")
    captured = {}

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout = None):
            self.returncode = 1

        def terminate(self):
            captured["terminated"] = True
            self.returncode = 1

    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda command, **kwargs: bridge.subprocess.CompletedProcess(command, 1),
    )
    bridge._stop_child(Process())

    assert captured.get("terminated") is True


@pytest.mark.skipif(os.name == "nt", reason = "POSIX process groups")
def test_stop_child_kills_survivors_after_leader_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "_CANCEL_GRACE_SECONDS", 0.2)
    marker = tmp_path / "grandchild-survived"
    grandchild = (
        "import pathlib, sys, time; time.sleep(1.0); "
        "pathlib.Path(sys.argv[1]).write_text('alive')"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])",
            grandchild,
            str(marker),
        ],
        start_new_session = True,
    )
    process.wait()

    bridge._stop_child(process)

    time.sleep(1.2)
    assert not marker.exists()


def test_result_parser_accepts_diagnostics_before_json():
    output = "connector warning\n" + json.dumps({"is_error": False, "result": "OK"})
    assert bridge._result_text(output) == "OK"

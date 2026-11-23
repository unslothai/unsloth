# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import io
import json
import os
import subprocess
import threading

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


@pytest.mark.parametrize("auto_route", [None, True])
def test_main_describes_the_tool_for_auto_route(tmp_path, monkeypatch, auto_route):
    path = _write_config(tmp_path)
    if auto_route:
        path.write_text(json.dumps({**json.loads(path.read_text()), "auto_route": True}))
    captured = {}
    monkeypatch.setattr(bridge, "serve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(bridge.sys, "argv", ["bridge", str(path)])
    monkeypatch.delenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, raising = False)
    bridge.main()
    expected = (
        bridge._CODEX_AUTO_ROUTE_TOOL_DESCRIPTION
        if auto_route
        else bridge._CODEX_SUBAGENT_TOOL_DESCRIPTION
    )
    assert captured["tool_description"] == expected


@pytest.mark.parametrize("auto_route", [None, True])
def test_auto_route_failure_says_the_step_was_not_done(tmp_path, monkeypatch, auto_route):
    path = _write_config(tmp_path)
    if auto_route:
        path.write_text(json.dumps({**json.loads(path.read_text()), "auto_route": True}))
    captured = {}
    monkeypatch.setattr(bridge, "serve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(bridge.sys, "argv", ["bridge", str(path)])

    def fail(
        task,
        cancel_event = None,
        **kwargs,
    ):
        raise RuntimeError("Connection refused")

    monkeypatch.setattr(bridge, "run_local_agent", fail)
    bridge.main()
    with pytest.raises(RuntimeError) as failure:
        captured["run_agent"]("task", threading.Event())
    assert "Connection refused" in str(failure.value)
    assert ("was not done" in str(failure.value)) is bool(auto_route)


def test_protocol_uses_codex_specific_tool_name():
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize"}),
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
        tool_description = bridge._CODEX_SUBAGENT_TOOL_DESCRIPTION,
        instructions = bridge._SERVER_INSTRUCTIONS,
    )
    responses = {
        response["id"]: response for response in map(json.loads, output.getvalue().splitlines())
    }
    assert responses[0]["result"]["instructions"] == bridge._SERVER_INSTRUCTIONS
    assert len(bridge._SERVER_INSTRUCTIONS) <= 512
    assert responses[1]["result"]["tools"][0]["name"] == "spawn_local_agent"
    assert (
        "Use this tool instead of the built-in spawn_agent tool"
        in responses[1]["result"]["tools"][0]["description"]
    )
    assert responses[1]["result"]["tools"][0]["annotations"]["destructiveHint"] is True
    assert responses[2]["result"] == {
        "content": [{"type": "text", "text": "completed: inspect this"}],
        "isError": False,
    }


@pytest.mark.parametrize("bypass_permissions", [False, True])
@pytest.mark.parametrize("wsl_bridge", [False, True])
def test_local_child_uses_explicit_unsloth_profile(
    monkeypatch, tmp_path, bypass_permissions, wsl_bridge
):
    config = _write_config(tmp_path, bypass_permissions = bypass_permissions)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    credential_names = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")
    for name in credential_names:
        monkeypatch.setenv(name, "cloud-key")
    monkeypatch.setenv("CODEX_SQLITE_HOME", str(tmp_path / "parent-sqlite"))
    if wsl_bridge:
        monkeypatch.setattr(
            bridge,
            "_wsl_shim_env",
            lambda command, env, unset: (
                env,
                (
                    bridge._CODEX_ENV_KEY,
                    "CODEX_HOME/p",
                    "CODEX_SQLITE_HOME/p",
                    *unset,
                    "PWD/p",
                ),
            ),
        )
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    captured = {}

    class Process:
        pid = 1234
        returncode = 0

        def communicate(
            self,
            input = None,
            timeout = None,
        ):
            captured["timeout"] = timeout
            captured["input"] = input
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
    # A refused connection to the local server must fail the run, not wait for the network forever.
    assert command[4:6] == ["-c", "features.unbounded_connection_retries=false"]
    if bypass_permissions:
        assert "--dangerously-bypass-approvals-and-sandbox" in command
    else:
        assert command[6:10] == ["--sandbox", "workspace-write", "--ask-for-approval", "never"]
    assert command[command.index("exec") + 1 : command.index("exec") + 4] == [
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
    ]
    assert command[-1] == "-"
    assert captured["input"].endswith("Task: reply exactly LOCAL_OK")
    assert captured["cwd"] == os.getcwd()
    assert captured["stdin"] is subprocess.PIPE
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE
    if os.name == "nt":
        assert captured["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert captured["start_new_session"] is True
    assert captured["env"]["CODEX_HOME"] == str(tmp_path / "child")
    assert captured["env"]["CODEX_SQLITE_HOME"] == str(tmp_path / "child")
    assert captured["env"][bridge._CODEX_ENV_KEY] == "sk-unsloth-test"
    if wsl_bridge:
        assert all(captured["env"][name] == "" for name in credential_names)
        wslenv = captured["env"]["WSLENV"].split(":")
        assert all(
            name in {entry.split("/", 1)[0] for entry in wslenv} for name in bridge._CODEX_ENV_UNSET
        )
        assert "CODEX_SQLITE_HOME/p" in wslenv
        assert "PWD/p" in wslenv
    else:
        assert all(name not in captured["env"] for name in credential_names)


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


def test_local_child_prioritizes_failed_turn_over_progress():
    output = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "still working"},
                }
            ),
            json.dumps({"type": "turn.failed", "error": {"message": "local failure"}}),
        ]
    )
    with pytest.raises(RuntimeError, match = "local failure"):
        bridge._result_text(output)


def test_auto_route_result_shows_the_last_command_the_child_ran():
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "sed -i 's/\\d{3}/555/' about.txt && cat about.txt",
                "aggregated_output": "Office phone: 555-0147\n",
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Updated to 555-0199."},
        },
    ]
    output = "\n".join(json.dumps(event) for event in events)
    assert bridge._result_text(output) == "Updated to 555-0199."
    shown = bridge._result_text(output, show_last_command = True)
    assert shown.startswith("Updated to 555-0199.")
    assert "Office phone: 555-0147" in shown
    assert "cat about.txt" in shown


def test_failed_child_reports_its_last_error_not_its_banner(monkeypatch, tmp_path):
    config = _write_config(tmp_path)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    reason = "stream disconnected before completion: error sending request"

    class Process:
        pid = 1234
        returncode = 1

        def communicate(
            self,
            input = None,
            timeout = None,
        ):
            events = [
                {"type": "turn.started"},
                {"type": "turn.failed", "error": {"message": reason}},
            ]
            return "\n".join(
                json.dumps(event) for event in events
            ), "Reading additional input from stdin...\n"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError) as failure:
        bridge.run_local_agent("hello")
    assert str(failure.value) == reason


def test_local_child_process_is_stopped_on_cancellation(monkeypatch, tmp_path):
    config = _write_config(tmp_path)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    cancel_event = bridge.threading.Event()
    stopped = []

    class Process:
        pid = 1234
        returncode = None

        def communicate(
            self,
            input = None,
            timeout = None,
        ):
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


def test_local_child_is_spawned_through_the_shim_resolver(monkeypatch, tmp_path):
    # Pins the wiring (#9167); the parser itself is covered in test_start.py.
    config = _write_config(tmp_path)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: r"C:\\nodejs\\codex.cmd")
    captured = {}

    def resolver(
        executable,
        arguments,
        environment = None,
    ):
        captured["resolver"] = (executable, arguments, environment)
        return ["C:\\nodejs\\node.exe", "index.js", *arguments]

    class Process:
        pid = 1234
        returncode = 0

        def communicate(
            self,
            input = None,
            timeout = None,
        ):
            captured["input"] = input
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
        return Process()

    monkeypatch.setattr(bridge, "_resolved_launch_command", resolver)
    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent("multi\nline\ntask") == "LOCAL_OK"

    executable, arguments, environment = captured["resolver"]
    assert executable == r"C:\\nodejs\\codex.cmd"
    assert arguments[0] == "--oss"
    assert arguments[-1] == "-"
    assert environment is not None and bridge._CODEX_ENV_KEY in environment
    assert captured["command"][0] == "C:\\nodejs\\node.exe"
    assert captured["input"].endswith("Task: multi\nline\ntask")


def test_prompt_reaches_the_child_over_stdin_once_even_after_a_poll_timeout(monkeypatch, tmp_path):
    # A non-npm codex.cmd runs argv through cmd.exe, so the task must not be in it.
    config = _write_config(tmp_path)
    monkeypatch.setenv(bridge._CODEX_SUBAGENT_CONFIG_ENV, str(config))
    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/codex")
    task = 'summarize the README" & echo INJECTED-BY-TASK & "'
    captured = {"inputs": []}

    class Process:
        pid = 1234
        returncode = 0

        def communicate(
            self,
            input = None,
            timeout = None,
        ):
            captured["inputs"].append(input)
            if len(captured["inputs"]) == 1:
                raise subprocess.TimeoutExpired("codex", timeout)
            event = {"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}
            return json.dumps(event), ""

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        captured["command"] = command
        captured["stdin"] = kwargs.get("stdin")
        return Process()

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run_local_agent(task) == "OK"
    assert not any(task in part for part in captured["command"])
    assert captured["command"][-1] == "-"
    assert captured["stdin"] is subprocess.PIPE
    first, retry = captured["inputs"]
    assert first.endswith(f"Task: {task}") and first.startswith(bridge._SUBAGENT_INSTRUCTIONS)
    assert retry is None

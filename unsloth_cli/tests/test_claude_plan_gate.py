# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic plan-mode routing for the local Claude subagent.

SKILL.md asks the parent model to pick the read-only tool in plan mode, which a
small local model can forget. The generated plugin also ships a PreToolUse hook
that reads permission_mode directly, so the editing agent is denied by rule.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from unsloth_cli.commands import start


def _plugin(tmp_path):
    return start.write_claude_subagent_plugin(tmp_path, {"UNSLOTH_CLAUDE_SUBAGENT_MODEL": "m"})


def _run_gate(script, payload):
    return subprocess.run(
        [sys.executable, str(script)],
        input = payload,
        capture_output = True,
        text = True,
        timeout = 30,
    )


def test_plugin_registers_a_pretooluse_hook_on_the_editing_tool(tmp_path):
    plugin = _plugin(tmp_path)

    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text())["hooks"]["PreToolUse"]

    [entry] = hooks
    # Only the destructive tool is gated; the read-only agent stays reachable.
    assert entry["matcher"] == start._CLAUDE_SUBAGENT_TOOL
    assert start._CLAUDE_SUBAGENT_PLAN_TOOL not in json.dumps(hooks)
    [hook] = entry["hooks"]
    assert hook["type"] == "command"
    assert sys.executable in hook["command"]
    # The interpreter is quoted: unquoted, any space in the path splits the command.
    assert f'"{sys.executable}"' in hook["command"]
    # The gate path rides as base64, never as a literal the shell can expand.
    encoded = start._b64_path(plugin / "hooks" / "plan_gate.py")
    assert encoded in hook["command"]
    assert str(plugin / "hooks" / "plan_gate.py") not in hook["command"]
    # A hook with no timeout stalls the parent for as long as it hangs.
    assert 0 < hook["timeout"] <= 30


def test_gate_script_is_written_and_compiles(tmp_path):
    plugin = _plugin(tmp_path)
    gate = plugin / "hooks" / "plan_gate.py"

    compile(gate.read_text(), str(gate), "exec")  # syntax-valid as shipped


def test_gate_denies_the_editing_tool_in_plan_mode(tmp_path):
    gate = _plugin(tmp_path) / "hooks" / "plan_gate.py"

    result = _run_gate(gate, json.dumps({"permission_mode": "plan"}))

    assert result.returncode == 0
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    # The reason is shown to the model, so it must name the tool to call instead.
    assert "unsloth_plan_agent" in output["permissionDecisionReason"]


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "bypassPermissions", "dontAsk", "auto"])
def test_gate_allows_every_non_plan_mode(tmp_path, mode):
    gate = _plugin(tmp_path) / "hooks" / "plan_gate.py"

    result = _run_gate(gate, json.dumps({"permission_mode": mode}))

    assert result.returncode == 0
    assert result.stdout.strip() == ""  # no decision -> normal permission flow


@pytest.mark.parametrize("payload", ["", "not json", "[]", "null", "{}"])
def test_gate_fails_open_on_unusable_input(tmp_path, payload):
    # A hook crash would block the parent session, so anything unparsable allows.
    gate = _plugin(tmp_path) / "hooks" / "plan_gate.py"

    result = _run_gate(gate, payload)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_plugin_still_writes_the_mcp_server_and_skill(tmp_path):
    # The hook is additive; the existing wiring must be untouched.
    plugin = _plugin(tmp_path)

    assert (plugin / ".mcp.json").exists()
    assert (plugin / "skills" / "local-agent" / "SKILL.md").exists()
    assert (plugin / ".claude-plugin" / "plugin.json").exists()


def test_wsl_run_clears_a_gate_left_by_an_earlier_windows_run(tmp_path, monkeypatch):
    # The plugin dir survives across runs when persisted, so a gate written by a
    # Windows run would otherwise be shipped into the distro with an interpreter
    # path it cannot execute.
    plugin = _plugin(tmp_path)
    gate = plugin / "hooks" / "plan_gate.py"
    hooks = plugin / "hooks" / "hooks.json"
    assert gate.exists() and hooks.exists()

    monkeypatch.setattr(start, "_wsl_windows_executable", lambda _argv: True)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    _plugin(tmp_path)

    assert not gate.exists()
    assert not hooks.exists()


def test_hook_command_survives_a_missing_gate_and_a_path_with_spaces(tmp_path):
    # Handing the path straight to the interpreter makes a missing gate exit 2,
    # which Claude treats as a blocking error: the editing tool would then be
    # denied in every mode, not just plan. Going through runpy makes it exit 1.
    plugin = _plugin(tmp_path / "dir with space")
    hook = json.loads((plugin / "hooks" / "hooks.json").read_text())
    command = hook["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    # Works normally through the real shell path Claude uses.
    denied = subprocess.run(
        command,
        input = json.dumps({"permission_mode": "plan"}),
        shell = True,
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert denied.returncode == 0
    assert json.loads(denied.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"

    (plugin / "hooks" / "plan_gate.py").unlink()
    gone = subprocess.run(
        command,
        input = json.dumps({"permission_mode": "default"}),
        shell = True,
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert gone.returncode != 2, "exit 2 blocks the tool in every mode"
    assert gone.stdout.strip() == ""


@pytest.mark.parametrize("hostile", ["sub$(echo X)", "tick`echo X`", "var$HOME", "pct%TEMP%pct"])
def test_gate_survives_shell_metacharacters_in_its_path(tmp_path, hostile):
    # The hook command is run by a shell. A temp root holding these expands under
    # sh (or cmd, for %VAR%) before Python sees the path, so the gate is not found
    # and exits 1, which fails open and silently drops the routing message.
    plugin = _plugin(tmp_path / hostile)
    command = json.loads((plugin / "hooks" / "hooks.json").read_text())["hooks"]["PreToolUse"][0][
        "hooks"
    ][0]["command"]

    denied = subprocess.run(
        command,
        input = json.dumps({"permission_mode": "plan"}),
        shell = True,
        capture_output = True,
        text = True,
        timeout = 60,
    )

    assert denied.returncode == 0, denied.stderr
    decision = json.loads(denied.stdout)["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"

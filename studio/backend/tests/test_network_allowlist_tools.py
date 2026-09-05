# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The tool layer's side of the network allowlist: plan policy, trailer, descriptions."""

from __future__ import annotations

import sys

import pytest

from core.inference import os_sandbox
from core.inference import tools as tools_module
from core.inference.network_proxy import NetworkAudit


@pytest.fixture
def capture_plan(monkeypatch):
    """Stub prepare_tool_launch: record the plan, run the argv unsandboxed."""
    plans: list[os_sandbox.ToolLaunchPlan] = []
    audit_holder: dict[str, NetworkAudit | None] = {"audit": None}

    def fake_prepare(spec):
        plans.append(spec)
        return os_sandbox.PreparedSandboxLaunch(
            argv = spec.argv,
            workdir = spec.workdir,
            env = spec.env,
            preexec_fn = spec.preexec_fn,
            backend = "test-passthrough",
            network_audit = audit_holder["audit"],
        )

    monkeypatch.setattr(tools_module, "prepare_tool_launch", fake_prepare)
    return plans, audit_holder


@pytest.mark.skipif(sys.platform == "win32", reason = "posix launch path")
def test_python_tool_forwards_the_network_policy_into_the_plan(capture_plan, tmp_path, monkeypatch):
    plans, _ = capture_plan
    result = tools_module.execute_tool(
        "python",
        {"code": "print('net ok')"},
        session_id = None,
        tool_execution_mode = "os_isolation_required",
        network_policy = "allowlist",
    )
    assert "net ok" in result
    assert plans[-1].network_policy == "allowlist"
    assert plans[-1].requested_mode == "os_isolation_required"

    tools_module.execute_tool(
        "python", {"code": "print(1)"}, tool_execution_mode = "os_isolation_required"
    )
    assert plans[-1].network_policy == "deny", "no choice means the sandbox stays offline"


@pytest.mark.skipif(sys.platform == "win32", reason = "posix launch path")
def test_terminal_tool_forwards_the_network_policy_into_the_plan(capture_plan):
    plans, _ = capture_plan
    result = tools_module.execute_tool(
        "terminal",
        {"command": "echo shell-net"},
        tool_execution_mode = "os_isolation_required",
        network_policy = "allowlist",
    )
    assert "shell-net" in result
    assert plans[-1].network_policy == "allowlist"


@pytest.mark.skipif(sys.platform == "win32", reason = "posix launch path")
def test_full_access_never_carries_an_allowlist(capture_plan):
    plans, _ = capture_plan
    tools_module.execute_tool(
        "python",
        {"code": "print(2)"},
        disable_sandbox = True,
        tool_execution_mode = "full",
        network_policy = "allowlist",
    )
    assert plans[-1].requested_mode == "full"
    assert plans[-1].network_policy == "deny"


def test_unknown_policy_is_passed_through_for_the_sandbox_layer_to_refuse():
    assert tools_module._requested_network_policy("bogus", False) == "bogus"
    assert tools_module._requested_network_policy(None, False) == "deny"
    assert tools_module._requested_network_policy("", False) == "deny"
    assert tools_module._requested_network_policy("allowlist", True) == "deny"


@pytest.mark.skipif(sys.platform == "win32", reason = "posix launch path")
def test_refused_hosts_are_appended_to_the_tool_result(capture_plan):
    plans, holder = capture_plan
    audit = NetworkAudit()
    audit.record_denied("evil.example", "host is not on the network allowlist")
    audit.record_denied("evil.example", "host is not on the network allowlist")
    audit.record_allowed("pypi.org")
    holder["audit"] = audit
    result = tools_module.execute_tool(
        "python",
        {"code": "print('body')"},
        tool_execution_mode = "os_isolation_required",
        network_policy = "allowlist",
    )
    assert result.startswith("body")
    assert "[network] Connections refused by the sandbox network allowlist:" in result
    assert "evil.example (2 attempts): host is not on the network allowlist" in result
    assert "pypi.org" not in result

    holder["audit"] = NetworkAudit()
    clean = tools_module.execute_tool(
        "terminal",
        {"command": "echo body"},
        tool_execution_mode = "os_isolation_required",
        network_policy = "allowlist",
    )
    assert "[network]" not in clean


@pytest.mark.skipif(sys.platform == "win32", reason = "posix launch path")
def test_refused_hosts_survive_a_timeout(capture_plan):
    plans, holder = capture_plan
    audit = NetworkAudit()
    audit.record_denied("slow.example", "host is not on the network allowlist")
    holder["audit"] = audit
    result = tools_module.execute_tool(
        "python",
        {"code": "import time; time.sleep(30)"},
        timeout = 1,
        tool_execution_mode = "os_isolation_required",
        network_policy = "allowlist",
    )
    assert "timed out" in result
    assert "slow.example" in result


def test_descriptions_gain_the_allowlisted_hosts_only_when_asked(monkeypatch):
    monkeypatch.setattr(tools_module.sys, "platform", "linux")
    specs = [dict(tools_module.PYTHON_TOOL), dict(tools_module.TERMINAL_TOOL), {"type": "function", "function": {"name": "web_search", "description": "search"}}]
    assert tools_module.apply_os_isolated_tool_descriptions(specs) is specs
    assert tools_module.apply_os_isolated_tool_descriptions(specs, network_allowlist = ()) is specs

    noted = tools_module.apply_os_isolated_tool_descriptions(
        specs, network_allowlist = ("pypi.org", "*.hf.co")
    )
    assert noted is not specs
    for index in (0, 1):
        description = noted[index]["function"]["description"]
        assert "admits only these hosts: pypi.org, *.hf.co" in description
        assert "HTTPS (port 443)" in description
        assert "refused hosts are listed at the end of the tool output" in description
    assert noted[2] is specs[2]
    # Module constants are untouched.
    assert "admits only these hosts" not in tools_module.PYTHON_TOOL["function"]["description"]
    assert "admits only these hosts" not in tools_module.TERMINAL_TOOL["function"]["description"]


def test_windows_cmd_note_and_allowlist_note_compose(monkeypatch):
    monkeypatch.setattr(tools_module.sys, "platform", "win32")
    monkeypatch.setattr(tools_module, "_windows_bash", lambda: r"C:\\Git\\bin\\bash.exe")
    bash_terminal = {
        **tools_module.TERMINAL_TOOL,
        "function": {
            **tools_module.TERMINAL_TOOL["function"],
            "description": "Run a command." + tools_module._TERMINAL_BASH_NOTE,
        },
    }
    out = tools_module.apply_os_isolated_tool_descriptions(
        [bash_terminal, tools_module.PYTHON_TOOL], network_allowlist = ["pypi.org"]
    )
    terminal = out[0]["function"]["description"]
    assert "The shell is cmd, not bash" in terminal
    assert "admits only these hosts: pypi.org" in terminal
    assert "admits only these hosts: pypi.org" in out[1]["function"]["description"]
    # Without an allowlist, the python tool is returned by identity as before.
    plain = tools_module.apply_os_isolated_tool_descriptions([bash_terminal, tools_module.PYTHON_TOOL])
    assert plain[1] is tools_module.PYTHON_TOOL

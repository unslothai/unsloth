# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Broker routing contracts; synthetic outcomes do not qualify native isolation."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference import os_sandbox as sandbox, tool_isolation

_REAL_TERMINAL_SELECTOR = sandbox.selected_windows_terminal


class FreshBackend:
    identity = "windows-lpac"
    requires_fresh_qualification = True
    supports_network_allowlist = False

    def __init__(self):
        self.calls, self.prepared = [], []
        self.available = {"python": False, "terminal": False}
        self.revisions = {"python": "one", "terminal": "one"}

    def probe_for_kind(
        self,
        kind,
        selected = None,
    ):
        self.calls.append((kind, selected))
        available = self.available[kind]
        return sandbox.SandboxCapability(
            self.identity,
            available,
            "fixed observation",
            available = available,
            profile_id = kind + "-profile",
            limitations = (kind + "-limitations",),
            probe_generation = kind + ":" + str(selected) + ":" + self.revisions[kind],
            qualification_generation = kind + ":" + str(selected) + ":" + self.revisions[kind],
        )

    def prepare_for_profile(self, spec, profile, **kwargs):
        self.prepared.append((spec, profile, kwargs))
        return sandbox.PreparedSandboxLaunch(spec.argv, spec.workdir, spec.env, None, self.identity)


@pytest.fixture
def routing(monkeypatch):
    backend = FreshBackend()
    monkeypatch.setattr(sandbox.sys, "platform", "win32")
    monkeypatch.setattr(sandbox, "_platform_backend", lambda: backend)
    monkeypatch.setattr(sandbox.sys, "executable", "python.exe")
    monkeypatch.setattr(
        sandbox, "selected_windows_terminal", lambda: sandbox.os.path.realpath("terminal.exe")
    )
    monkeypatch.setattr(sandbox, "_limited_isolation_backend", lambda: None)
    monkeypatch.setattr(sandbox, "_environment_class", lambda **kwargs: "windows")
    monkeypatch.setattr(sandbox, "_environment_fingerprint", lambda *args, **kwargs: "fixed-os")
    monkeypatch.setattr(sandbox, "descendant_sweep_supported", lambda: True)
    monkeypatch.setattr(tool_isolation, "_LIMITED_GRANTS", tool_isolation.LimitedGrantStore())
    return backend


def _plan(
    tmp_path,
    kind,
    mode = "os_isolation_required",
    grant = None,
):
    return sandbox.ToolLaunchPlan(
        (kind + ".exe", "fixed-script"),
        str(tmp_path),
        {},
        execution_kind = kind,
        requested_mode = mode,
        current_subject = "test:user",
        tool_ui_session_id = "test-session",
        limited_grant = grant,
    )


def _grant(capability):
    return tool_isolation.issue_limited_grant(
        current_subject = "test:user",
        tool_ui_session_id = "test-session",
        probe_generation = capability.probe_generation,
    ).token


def test_fresh_capabilities_never_share_os_cache_between_kinds(routing):
    first = sandbox.capability_snapshot(execution_kind = "python", selected_executable = "python.exe")
    terminal = sandbox.capability_snapshot(
        execution_kind = "terminal", selected_executable = "terminal.exe"
    )
    again = sandbox.capability_snapshot(execution_kind = "python", selected_executable = "python.exe")
    assert routing.calls == [
        ("python", "python.exe"),
        ("terminal", "terminal.exe"),
        ("python", "python.exe"),
    ]
    assert first.probe_generation == again.probe_generation
    assert first.probe_generation != terminal.probe_generation
    assert not first.available and not terminal.available


def test_selected_runtime_change_refreshes_generation_without_force(routing):
    first = sandbox.capability_snapshot(execution_kind = "python", selected_executable = "one.exe")
    second = sandbox.capability_snapshot(execution_kind = "python", selected_executable = "two.exe")
    assert first.probe_generation != second.probe_generation
    routing.revisions["python"] = "two"
    third = sandbox.capability_snapshot(execution_kind = "python", selected_executable = "two.exe")
    assert third.probe_generation != second.probe_generation


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_required_unqualified_kind_never_prepares_or_falls_back(routing, tmp_path, kind):
    routing.available["terminal" if kind == "python" else "python"] = True
    with pytest.raises(sandbox.SandboxUnavailableError, match = "OS_ISOLATION_UNAVAILABLE"):
        sandbox.prepare_tool_launch(_plan(tmp_path, kind))
    assert routing.calls == [(kind, kind + ".exe")]
    assert not routing.prepared


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_required_routes_exact_selected_kind_profile_and_record(routing, tmp_path, kind):
    routing.available[kind] = True
    prepared = sandbox.prepare_tool_launch(_plan(tmp_path, kind))
    assert routing.calls == [(kind, kind + ".exe")]
    assert routing.prepared[0][0].execution_kind == kind
    assert routing.prepared[0][1] == kind + "-profile"
    assert routing.prepared[0][2] == {"expected_generation": kind + ":" + kind + ".exe:one"}
    assert prepared.execution_record.os_isolation
    assert prepared.execution_record.effective_mode == "os_isolation_required"
    assert prepared.execution_record.profile_id == kind + "-profile"


def test_limited_consent_is_independent_for_terminal_when_python_qualifies(routing, tmp_path):
    routing.available["python"] = True
    capability = sandbox.capability_snapshot()
    prepared = sandbox.prepare_tool_launch(
        _plan(tmp_path, "terminal", "limited", _grant(capability))
    )
    assert not prepared.execution_record.os_isolation
    assert prepared.execution_record.effective_mode == "limited"
    assert prepared.execution_record.network_policy == "unrestricted"
    assert not routing.prepared


def test_python_limited_consent_never_authorizes_terminal(routing, tmp_path):
    capability = sandbox.capability_snapshot(
        execution_kind = "python", selected_executable = "python.exe"
    )
    with pytest.raises(sandbox.SandboxUnavailableError, match = "capability changed"):
        sandbox.prepare_tool_launch(_plan(tmp_path, "terminal", "limited", _grant(capability)))
    assert not routing.prepared


def test_limited_consent_cannot_survive_new_runtime_generation(routing, tmp_path):
    capability = sandbox.capability_snapshot()
    grant = _grant(capability)
    routing.revisions["terminal"] = "two"
    with pytest.raises(sandbox.SandboxUnavailableError, match = "capability changed"):
        sandbox.prepare_tool_launch(_plan(tmp_path, "terminal", "limited", grant))
    assert not routing.prepared


def test_recovered_kind_refuses_old_limited_consent(routing, tmp_path):
    capability = sandbox.capability_snapshot()
    grant = _grant(capability)
    routing.available["terminal"] = True
    routing.available["python"] = True
    with pytest.raises(sandbox.SandboxUnavailableError, match = "OS isolation is available"):
        sandbox.prepare_tool_launch(_plan(tmp_path, "terminal", "limited", grant))
    assert not routing.prepared


def test_full_access_does_not_probe_or_consume_limited_consent(routing, tmp_path):
    prepared = sandbox.prepare_tool_launch(_plan(tmp_path, "terminal", "full"))
    assert prepared.execution_record.effective_mode == "full"
    assert not routing.calls and not routing.prepared


def test_limited_allowlist_is_refused_before_launch(routing, tmp_path):
    capability = sandbox.capability_snapshot(
        execution_kind = "terminal", selected_executable = "terminal.exe"
    )
    with pytest.raises(sandbox.SandboxUnavailableError, match = "allowlist requires OS isolation"):
        sandbox.prepare_tool_launch(
            replace(
                _plan(tmp_path, "terminal", "limited", _grant(capability)),
                network_policy = "allowlist",
            )
        )
    assert not routing.prepared


def test_aggregate_never_claims_both_tools_available_from_python_alone(routing):
    routing.available["python"] = True
    result = sandbox.capability_snapshot()
    assert not result.available and not result.qualified
    assert result.protection_state == "unavailable"
    assert "Python:" in result.reason and "Terminal:" in result.reason
    assert [kind for kind, selected in routing.calls] == ["python", "terminal"]


def test_aggregate_generation_binds_both_tools_and_selected_shell(routing, monkeypatch):
    first = sandbox.capability_snapshot()
    routing.revisions["python"] = "two"
    second = sandbox.capability_snapshot()
    routing.revisions["terminal"] = "two"
    third = sandbox.capability_snapshot()
    monkeypatch.setattr(sandbox, "selected_windows_terminal", lambda: "different-shell.exe")
    fourth = sandbox.capability_snapshot()
    assert len({item.probe_generation for item in (first, second, third, fourth)}) == 4


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_explicit_aggregate_limited_consent_authorizes_both_selected_tools(routing, tmp_path, kind):
    capability = sandbox.capability_snapshot()
    prepared = sandbox.prepare_tool_launch(_plan(tmp_path, kind, "limited", _grant(capability)))
    assert prepared.execution_record.effective_mode == "limited"
    assert not prepared.execution_record.os_isolation
    assert not routing.prepared


def test_aggregate_consent_refuses_another_executable(routing, tmp_path):
    capability = sandbox.capability_snapshot()
    spec = replace(
        _plan(tmp_path, "terminal", "limited", _grant(capability)), argv = ("another.exe", "fixed")
    )
    with pytest.raises(sandbox.SandboxUnavailableError, match = "differs from the selected"):
        sandbox.prepare_tool_launch(spec)
    assert not routing.prepared


def test_api_facing_capability_receives_the_aggregate(routing):
    routing.available["python"] = True
    result = tool_isolation.capability_snapshot(force = True)
    assert not result.available
    assert [kind for kind, selected in routing.calls] == ["python", "terminal"]


@pytest.fixture(scope = "module")
def shell_tools():
    # Import before the per-test routing fixture replaces sys.executable.
    from core.inference import tools
    return tools


@pytest.mark.parametrize("system_root", [None, "relative-system-root"])
def test_unselectable_terminal_preserves_python_capability_and_aggregate_consent(
    shell_tools, routing, monkeypatch, tmp_path, system_root
):
    monkeypatch.setattr(shell_tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(sandbox, "selected_windows_terminal", _REAL_TERMINAL_SELECTOR)
    if system_root is None:
        monkeypatch.delenv("SystemRoot", raising = False)
    else:
        monkeypatch.setenv("SystemRoot", system_root)
    routing.available["python"] = True
    first = sandbox.capability_snapshot()
    assert routing.calls == [("python", sandbox.os.path.realpath("python.exe"))]
    assert not first.available and not first.qualified
    assert "Terminal: The Windows system shell location is unavailable" in first.reason
    assert "terminal:terminal_runtime_unselectable" in first.limitations
    assert sandbox.capability_snapshot().probe_generation == first.probe_generation
    prepared = sandbox.prepare_tool_launch(_plan(tmp_path, "python", "limited", _grant(first)))
    assert prepared.execution_record.effective_mode == "limited"
    assert not prepared.execution_record.os_isolation
    assert not routing.prepared
    monkeypatch.setattr(
        sandbox, "selected_windows_terminal", lambda: sandbox.os.path.realpath("terminal.exe")
    )
    assert sandbox.capability_snapshot().probe_generation != first.probe_generation


@pytest.mark.parametrize("system_root", [None, "relative-system-root"])
def test_unselectable_terminal_never_uses_python_consent_or_substitutes_an_executable(
    shell_tools, routing, monkeypatch, tmp_path, system_root
):
    monkeypatch.setattr(shell_tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(sandbox, "selected_windows_terminal", _REAL_TERMINAL_SELECTOR)
    if system_root is None:
        monkeypatch.delenv("SystemRoot", raising = False)
    else:
        monkeypatch.setenv("SystemRoot", system_root)
    routing.available["python"] = True
    grant = _grant(sandbox.capability_snapshot())
    for executable in ("terminal.exe", str(tmp_path / "arbitrary.exe")):
        plan = replace(_plan(tmp_path, "terminal", "limited", grant), argv = (executable, "fixed"))
        with pytest.raises(sandbox.SandboxUnavailableError, match = "differs from the selected"):
            sandbox.prepare_tool_launch(plan)
        with pytest.raises(sandbox.SandboxUnavailableError, match = "OS_ISOLATION_UNAVAILABLE"):
            sandbox.prepare_tool_launch(replace(plan, requested_mode = "os_isolation_required"))
    assert not routing.prepared

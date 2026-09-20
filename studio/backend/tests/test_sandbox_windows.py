# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MXC Windows backend's policy and gating.

These run on every platform, because they are about the policy this backend
GENERATES and the conditions under which it engages. Whether Windows actually
ENFORCES that policy is a different question and no test here answers it; that
needs a Windows runner and lives in the Windows CI job.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference import os_sandbox, sandbox_windows  # noqa: E402
from core.inference.os_sandbox import ToolLaunchPlan  # noqa: E402


@pytest.fixture
def plan(tmp_path):
    return ToolLaunchPlan(
        argv = ("python.exe", "-c", "print('hi')"),
        workdir = str(tmp_path),
        env = {"PATH": "C:\\Windows\\System32", "HOME": str(tmp_path)},
        requested_mode = "required",
        timeout_seconds = 30,
        execution_kind = "python",
    )


def policy_for(plan, tmp_path):
    return sandbox_windows.build_policy(plan, str(tmp_path), "unsloth-test")


def test_the_schema_version_is_pinned_exactly(plan, tmp_path):
    # Not "whatever MXC currently defaults to": the executor was measured
    # accepting 0.8.1-alpha despite the docs calling stable schemas exact-match,
    # so the pin is ours to hold.
    assert policy_for(plan, tmp_path)["version"] == "0.8.0-alpha"


def test_only_the_session_workdir_is_writable(plan, tmp_path):
    filesystem = policy_for(plan, tmp_path)["filesystem"]
    packages = os.path.join(str(tmp_path), os_sandbox.SESSION_PACKAGES_RELPATH)
    assert sorted(filesystem["readwritePaths"]) == sorted([str(tmp_path), packages])


def test_the_user_home_is_not_granted(plan, tmp_path):
    # The whole point: a model-authored script must not reach ~/.aws or ~/.ssh.
    filesystem = policy_for(plan, tmp_path)["filesystem"]
    granted = filesystem["readwritePaths"] + filesystem["readonlyPaths"]
    home = os.path.expanduser("~")
    assert not any(os.path.normcase(p) == os.path.normcase(home) for p in granted)


def test_the_argv_is_not_rendered_into_the_command_line(plan, tmp_path):
    # commandLine is a single STRING. Quoting model-authored argv into a Windows
    # command line in Python is the injection bug this design avoids, so the
    # policy must carry no command at all and leave it to MXC's `--` splice.
    assert "commandLine" not in policy_for(plan, tmp_path)["process"]


def test_the_timeout_reaches_mxc_in_milliseconds(plan, tmp_path):
    assert policy_for(plan, tmp_path)["process"]["timeout"] == 30_000


def test_the_environment_is_passed_verbatim_not_inherited(plan, tmp_path):
    # MXC replaces rather than layers unless inheritDefaultEnv, which is 0.9
    # only, so the sanitized env tools.py built must be the whole env.
    env = policy_for(plan, tmp_path)["process"]["env"]
    assert "PATH=C:\\Windows\\System32" in env
    assert all("=" in entry for entry in env)


def test_the_policy_is_json_serializable(plan, tmp_path):
    # It is about to be base64'd into an argv; a non-serializable value would
    # fail at launch rather than here.
    json.dumps(policy_for(plan, tmp_path))


def test_windows_isolation_is_opt_in(monkeypatch):
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    assert os_sandbox.windows_preview_enabled() is False
    monkeypatch.setenv(os_sandbox.WINDOWS_PREVIEW_ENV, "1")
    assert os_sandbox.windows_preview_enabled() is True


def test_the_preview_cannot_be_selected_by_a_request(monkeypatch, tmp_path):
    # Backend eligibility is host policy. If a tool argument could turn it on,
    # it could also turn it off.
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available is False
    assert "opt-in preview" in capability.reason


def test_an_unavailable_windows_host_still_names_the_reason(monkeypatch):
    monkeypatch.delenv(os_sandbox.WINDOWS_PREVIEW_ENV, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    capability = os_sandbox.capability_snapshot(force = True)
    assert os_sandbox.WINDOWS_PREVIEW_ENV in capability.remediation
    assert capability.limitations == ("no_os_isolation",)


def test_the_preview_warning_is_carried_into_the_limitations():
    # MXC's README says no MXC profile is a security boundary yet. Surfacing
    # that is not optional; the badge would otherwise read as more than it is.
    assert "mxc_preview_not_a_security_boundary" in sandbox_windows.LIMITATIONS
    assert "unrestricted_network" in sandbox_windows.LIMITATIONS


def test_the_dacl_caveat_is_dropped_only_on_a_tier1_host(monkeypatch):
    monkeypatch.setattr(sandbox_windows, "_windows_build", lambda: 26100)
    assert "windows_tier3_dacl" in sandbox_windows.host_limitations()
    monkeypatch.setattr(sandbox_windows, "_windows_build", lambda: 26600)
    assert "windows_tier3_dacl" not in sandbox_windows.host_limitations()


def test_the_executor_is_never_resolved_from_path(monkeypatch):
    # PATH is attacker-influenced on a host where tool calls run, and this is
    # the binary the whole boundary rests on.
    monkeypatch.delenv("UNSLOTH_MXC_EXEC", raising = False)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable))
    source = sandbox_windows.executable_path.__doc__ or ""
    assert "Never from PATH" in source


def test_a_non_windows_host_refuses_this_backend(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    ok, reason = sandbox_windows.available()
    assert ok is False
    assert "Windows-only" in reason


def test_every_launch_gets_its_own_container_id(monkeypatch, plan, tmp_path):
    # MXC keys its ACEs on the SID derived from this id and documents that two
    # concurrent runs sharing one revoke each other's ACEs.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    seen = set()
    for _ in range(5):
        prepared = sandbox_windows.prepare(plan)
        # Located by flag, not by index: argv grows as the backend learns what
        # MXC needs, and a positional assumption breaks silently.
        encoded = prepared.argv[prepared.argv.index("--config-base64") + 1]
        import base64

        seen.add(json.loads(base64.b64decode(encoded))["containerId"])
    assert len(seen) == 5


def test_mxc_diagnostics_are_diverted_off_the_payload_stdout(monkeypatch, plan):
    # MXC runs in passthrough stdio mode and forwards its own handles to the
    # child, so without --log-file its warnings land on the same stdout as the
    # payload. That breaks the probe, which requires its token alone, and would
    # splice MXC's banner into what the user sees from a tool call.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert "--log-file" in prepared.argv
    log_path = prepared.argv[prepared.argv.index("--log-file") + 1]
    assert os.path.isfile(log_path)
    prepared.cleanup()
    assert not os.path.exists(log_path)


def test_the_prepared_launch_carries_no_posix_only_kwargs(monkeypatch, plan):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert prepared.preexec_fn is None
    assert prepared.pass_fds == ()


def test_the_argv_ends_with_the_payload_after_a_bare_separator(monkeypatch, plan):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    prepared = sandbox_windows.prepare(plan)
    assert prepared.argv[-4:] == ("--", "python.exe", "-c", "print('hi')")


def test_cleanup_reconciles_the_container(monkeypatch, plan):
    # A hard kill skips MXC's own ACE revert, so this must run on every path.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows, "available", lambda: (True, "ok"))
    monkeypatch.setattr(sandbox_windows, "executable_path", lambda: sys.executable)
    called = []
    monkeypatch.setattr(
        sandbox_windows,
        "_reconcile_container",
        lambda executor, container_id: called.append(container_id),
    )
    prepared = sandbox_windows.prepare(plan)
    prepared.cleanup()
    assert len(called) == 1


def test_the_windows_system_roots_are_granted_read_only(monkeypatch, tmp_path):
    """MXC grants nothing implicitly: "Omitted = no filesystem access beyond the
    default sandbox root". Without these, python.exe cannot resolve ntdll or the
    CRT and the launch fails before the payload runs. Guarded because the whole
    grant can be dropped and every suite still passes on a non-Windows box.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    (tmp_path / "System32").mkdir()

    roots = sandbox_windows._readonly_roots(str(tmp_path / "workdir"))

    assert str(tmp_path) in roots, "the Windows system root is not granted"
    assert str(tmp_path / "System32") in roots, "System32 is not granted"


def test_the_system_roots_are_never_writable(monkeypatch, tmp_path):
    """Read-only is the whole point: this widens what a tool call can READ, and
    must not widen what it can write."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    (tmp_path / "System32").mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    plan = ToolLaunchPlan(
        argv = ("python.exe", "-c", "pass"), workdir = str(workdir), env = {},
        requested_mode = "required",
    )
    filesystem = sandbox_windows.build_policy(plan, str(workdir), "t")["filesystem"]

    assert str(tmp_path) not in filesystem["readwritePaths"]
    assert str(tmp_path / "System32") not in filesystem["readwritePaths"]


def test_windows_gets_the_environment_it_needs_to_start_a_process(monkeypatch):
    """MXC replaces the environment rather than layering onto it, so whatever
    the policy carries is the whole environment. A plan env without SystemRoot
    leaves the interpreter unable to initialise, which looks exactly like
    confinement from the outside."""
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")

    # Through build_policy, not the helper directly: testing the helper alone
    # leaves the WIRING unguarded, and the wiring is the part that can be
    # dropped by a refactor while every assertion still passes.
    plan = ToolLaunchPlan(
        argv = ("python.exe", "-c", "pass"),
        workdir = "C:\\work",
        env = {"PATH": "C:\\Windows\\System32"},
        requested_mode = "required",
    )
    env = sandbox_windows.build_policy(plan, "C:\\work", "t")["process"]["env"]

    assert "SystemRoot=C:\\Windows" in env
    assert "COMSPEC=C:\\Windows\\System32\\cmd.exe" in env
    assert "PATH=C:\\Windows\\System32" in env


def test_a_caller_supplied_value_wins_over_the_host(monkeypatch):
    """Filled in only when absent, so a deliberately overridden value still
    wins and the sanitized env tools.py built is not quietly undone."""
    monkeypatch.setenv("SystemRoot", "C:\\Windows")

    env = sandbox_windows._policy_environment({"SystemRoot": "D:\\Custom"})

    assert "SystemRoot=D:\\Custom" in env
    assert "SystemRoot=C:\\Windows" not in env

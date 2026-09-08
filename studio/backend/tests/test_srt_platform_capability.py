# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference import os_sandbox, srt_probe


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("available", [False, True])
def test_supported_platform_uses_live_probe_not_strict_dns_gate(monkeypatch, platform, available):
    monkeypatch.setattr(os_sandbox.sys, "platform", platform)
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: "identity")
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return available, "runtime check result"

    monkeypatch.setattr(srt_probe, "probe", probe)
    capability = os_sandbox.capability_snapshot(
        force = True, execution_kind = "terminal", selected_executable = "selected-bash"
    )
    assert calls == [
        {"force": True, "execution_kind": "terminal", "selected_executable": "selected-bash"}
    ]
    assert capability.available is available
    assert capability.protection_state == ("preview" if available else "unavailable")
    assert capability.profile_id == "srt-0.0.75-native-v1"
    assert capability.limitations
    assert capability.network_policies == ("deny",)
    if platform == "darwin":
        assert "host_files_readable" in capability.limitations


def test_missing_linux_prerequisites_are_named_in_the_remediation(monkeypatch):
    # SRT_TOOL_ISOLATION.md tells the user to install prerequisites "if the capability
    # message requests them", so the message has to actually request them.
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda name, **kwargs: None if name == "socat" else "/usr/bin/" + name)
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: False)
    remediation = os_sandbox._linux_unavailable_remediation()
    assert "socat" in remediation
    assert "bwrap" not in remediation
    assert os_sandbox._BLOCKED_REMEDIATION in remediation


def test_apparmor_userns_denial_is_named_when_every_binary_is_present(monkeypatch):
    # The Ubuntu 23.10+ default. bwrap is installed and the probe still fails, so
    # "install the prerequisites" is the one instruction that cannot help here.
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda name, **kwargs: "/usr/bin/" + name)
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: True)
    remediation = os_sandbox._linux_unavailable_remediation()
    assert "apparmor_restrict_unprivileged_userns" in remediation
    assert "does not change host security policy" in remediation


def test_unexplained_linux_failure_keeps_the_plain_message(monkeypatch):
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda name, **kwargs: "/usr/bin/" + name)
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: False)
    assert os_sandbox._linux_unavailable_remediation() == os_sandbox._BLOCKED_REMEDIATION


def test_apparmor_probe_is_read_only_and_false_without_the_sysctl(monkeypatch):
    # No sysctl file (non-Ubuntu, or a kernel without AppArmor) must not be reported
    # as a denial, and the check must never run anything that changes host policy.
    def refuse(*args, **kwargs):
        raise AssertionError("the diagnosis must not spawn anything when the sysctl is absent")

    monkeypatch.setattr(os_sandbox.subprocess, "run", refuse)
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("absent"))
    )
    assert os_sandbox._linux_userns_blocked_by_apparmor() is False


def test_macos_read_disclosure_reaches_execution_record(monkeypatch, tmp_path):
    monkeypatch.setattr(os_sandbox.sys, "platform", "darwin")
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: "identity")
    monkeypatch.setattr(srt_probe, "probe", lambda **kwargs: (True, "write controls passed"))
    monkeypatch.setattr(os_sandbox.srt_adapter, "request_for", lambda *args, **kwargs: {})
    prepared = os_sandbox.prepare_tool_launch(
        os_sandbox.ToolLaunchPlan(
            argv = (os_sandbox.sys.executable, "-c", "print(1)"),
            workdir = str(tmp_path),
            env = {},
        )
    )
    assert "host_files_readable" in prepared.execution_record.as_dict()["limitations"]
    assert prepared.execution_record.effective_mode == "os_isolation_required"

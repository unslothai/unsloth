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


def test_macos_read_disclosure_reaches_execution_record(monkeypatch, tmp_path):
    monkeypatch.setattr(os_sandbox.sys, "platform", "darwin")
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: "identity")
    monkeypatch.setattr(srt_probe, "probe", lambda **kwargs: (True, "write controls passed"))
    monkeypatch.setattr(os_sandbox.srt_adapter, "request_for", lambda *args, **kwargs: {})
    prepared = os_sandbox.prepare_tool_launch(os_sandbox.ToolLaunchPlan(
        argv = (os_sandbox.sys.executable, "-c", "print(1)"),
        workdir = str(tmp_path), env = {},
    ))
    assert "host_files_readable" in prepared.execution_record.as_dict()["limitations"]
    assert prepared.execution_record.effective_mode == "os_isolation_required"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Diagnostic and admission seams; these tests do not claim native confinement."""

from dataclasses import asdict
from unittest.mock import Mock

import pytest

from core.inference import os_sandbox, srt_probe, srt_adapter, tool_isolation
from core.inference.srt_diagnostics import ProbeReason, REASONS, environment_context


def test_ripgrep_dependency_survives_capability_mapping(monkeypatch):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(
        srt_probe,
        "probe",
        lambda **kwargs: (False, ProbeReason("dependency_missing", "dependency", "ripgrep")),
    )
    public = asdict(tool_isolation._shape_capability(os_sandbox.capability_snapshot()))
    assert public["diagnostic"]["dependency"] == "ripgrep"
    assert not public["available"]
    assert not public["nested_eligible"]


@pytest.mark.parametrize("code", [*REASONS, "unknown-secret-value"])
def test_structured_failure_blocks_before_tool_spawn(monkeypatch, tmp_path, code):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(srt_probe, "probe", lambda **kw: (False, ProbeReason(code, "policy")))
    spawn = Mock(side_effect = AssertionError("tool must not start"))
    monkeypatch.setattr(srt_adapter, "spawn", spawn)
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", spawn)
    snapshot = os_sandbox.capability_snapshot()
    public = asdict(tool_isolation._shape_capability(snapshot))
    expected = code if code in REASONS else "probe_failed"
    assert public["reason_code"] == expected
    assert public["diagnostic"]["stage"] == "policy"
    assert "secret" not in str(public)
    assert not public["available"]
    with pytest.raises(os_sandbox.SandboxUnavailableError):
        os_sandbox.prepare_tool_launch(
            os_sandbox.ToolLaunchPlan(
                argv = (os_sandbox.sys.executable, "-c", "print(1)"),
                workdir = str(tmp_path),
                env = {},
            )
        )
    spawn.assert_not_called()


def test_exception_diagnostics_never_expose_output(monkeypatch):
    monkeypatch.setattr(srt_probe, "_cache", {})

    def fail(**kw):
        raise RuntimeError("token=secret /home/private/credential https://user:password@host")

    monkeypatch.setattr(srt_probe, "_native_probe", fail)
    available, reason = srt_probe.probe(force = True)
    assert not available
    assert reason.code == "probe_failed"
    assert "secret" not in reason and "private" not in reason


@pytest.mark.parametrize(
    "returncode,code", [(0, "enforcement_failed"), (1, "operation_unsupported")]
)
@pytest.mark.parametrize("variant", ["standard", "nested"])
def test_namespace_diagnostic_uses_only_fixed_control(monkeypatch, returncode, code, variant):
    from types import SimpleNamespace

    run = Mock(return_value = SimpleNamespace(returncode = returncode))
    monkeypatch.setattr(srt_probe.subprocess, "run", run)
    reason = srt_probe._failed_linux_launch_reason(isolation_variant = variant)
    assert reason.code == code
    args, kwargs = run.call_args
    assert args[0][-1] == "/bin/true"
    assert "--cap-drop" in args[0]
    assert ("--proc" in args[0]) == (variant == "standard")
    assert ("--bind" in args[0]) == (variant == "nested")
    assert kwargs["timeout"] == 3


def test_namespace_diagnostic_timeout_stays_blocked(monkeypatch):
    run = Mock(side_effect = srt_probe.subprocess.TimeoutExpired("control", 3))
    monkeypatch.setattr(srt_probe.subprocess, "run", run)
    assert srt_probe._failed_linux_launch_reason().code == "probe_timeout"


def test_timeout_receipt_is_diagnostic_not_success():
    import json
    import os
    from types import SimpleNamespace

    read, write = os.pipe()
    proc = SimpleNamespace(_srt_control_fd = read, poll = lambda: 124)
    try:
        os.write(
            write,
            json.dumps(
                {"v": 1, "event": "exit", "code": None, "signal": "SIGTERM", "reason": "timeout"}
            ).encode()
            + b"\n",
        )
        os.close(write)
        with pytest.raises(srt_adapter.SrtError):
            srt_adapter.verify_success(proc)
        assert srt_adapter.completion_receipt(proc)["reason"] == "timeout"
    finally:
        srt_adapter.release_control(proc)


def test_only_bounded_policy_counts_cross_public_boundary():
    reason = ProbeReason(
        "policy_oversized",
        "policy",
        details = {"field": "readRoots", "count": 1025, "limit": 1024, "secret": "DO_NOT_COPY"},
    )
    detail = reason.fields()["diagnostic"]
    assert detail["field"] == "readRoots" and detail["count"] == 1025
    assert "secret" not in detail
    assert not ProbeReason(
        "policy_invalid", details = {"field": "/private/secret", "count": 1025, "limit": 1024}
    ).details


@pytest.mark.parametrize(
    "marker,release,expected",
    [
        (False, False, "linux"),
        (False, True, "linux"),
        (True, False, "linux"),
        (True, True, "colab"),
    ],
)
def test_colab_requires_server_facts(monkeypatch, marker, release, expected):
    from core.inference import srt_diagnostics

    monkeypatch.setattr(srt_diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(
        srt_diagnostics.os.path, "isdir", lambda path: marker and path == "/var/colab"
    )
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test" if release else "")
    assert environment_context() == expected


def test_environment_recycling_invalidates_probe_cache(monkeypatch):
    monkeypatch.setattr(srt_probe, "_cache", {})
    identity = ["container-1"]
    monkeypatch.setattr(srt_probe, "boundary_identity", lambda: identity[0])
    native = Mock(
        return_value = (False, ProbeReason("dependency_missing", "dependency", "bubblewrap"))
    )
    monkeypatch.setattr(srt_probe, "_native_probe", native)
    srt_probe.probe()
    srt_probe.probe()
    assert native.call_count == 1
    identity[0] = "container-2"
    srt_probe.probe()
    assert native.call_count == 2
    native.return_value = (True, "controlled success")
    assert srt_probe.probe(force = True)[0]


@pytest.mark.parametrize("environment", ["colab", "linux"])
def test_limited_authority_reaches_record(monkeypatch, tmp_path, environment):
    monkeypatch.setattr(os_sandbox, "environment_context", lambda: environment)
    monkeypatch.setattr(srt_probe, "probe", lambda **kw: (False, ProbeReason("dependency_missing")))
    monkeypatch.setattr(tool_isolation, "validate_limited_grant", lambda *a, **kw: None)
    snapshot = os_sandbox.capability_snapshot()
    prepared = os_sandbox.prepare_tool_launch(
        os_sandbox.ToolLaunchPlan(
            argv = (os_sandbox.sys.executable,),
            workdir = str(tmp_path),
            env = {},
            requested_mode = "limited",
            current_subject = "test",
            tool_ui_session_id = "test",
            limited_grant = "mocked-consent",
        )
    )
    record = prepared.execution_record.as_dict()
    assert record["authority_disclosure"] == snapshot.limited_disclosure
    assert ("Colab" in record["authority_disclosure"]) is (environment == "colab")
    assert "does not add an OS sandbox" in record["authority_disclosure"]
    assert record["os_isolation"] is False

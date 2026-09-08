# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Final admission seams; these are not native enforcement evidence."""

from dataclasses import replace
import sys
import threading
from unittest.mock import Mock
import pytest
from core.inference import os_sandbox, srt_adapter, srt_nested, srt_probe


def test_consent_identity_changes_when_selected_shell_or_venv_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(srt_adapter, "installation_identity", lambda: "installation")
    monkeypatch.setattr(os_sandbox, "boundary_identity", lambda: "container")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    initial = os_sandbox._runtime_identity()
    shell = tmp_path / "venv" / "bin" / "bash"
    shell.parent.mkdir(parents = True)
    shell.write_text("benign selected-shell fixture")
    installed = os_sandbox._runtime_identity()
    assert installed != initial
    shell.write_text("replaced selected-shell fixture with different contents")
    assert os_sandbox._runtime_identity() != installed
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "different-venv"))
    assert os_sandbox._runtime_identity() != initial


@pytest.fixture
def admission(monkeypatch, tmp_path):
    capability = os_sandbox.SandboxCapability(
        backend = "srt",
        qualified = False,
        reason = "container restriction",
        available = False,
        environment = "linux",
        probe_generation = "generation",
        nested_eligible = True,
        nested_profile_id = srt_nested.NESTED_PROFILE,
        nested_disclosure = srt_nested.NESTED_DISCLOSURE,
    )
    store = srt_nested.NestedGrantStore()
    monkeypatch.setattr(srt_nested, "NESTED_GRANTS", store)
    token = store.issue(
        current_subject = "actor", tool_ui_session_id = "page", probe_generation = "generation"
    )
    spec = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "print('tool')"),
        workdir = str(tmp_path),
        env = {},
        requested_mode = "container_isolation",
        current_subject = "actor",
        tool_ui_session_id = "page",
        nested_grant = token.token,
        execution_kind = "python",
        cancel_event = threading.Event(),
    )
    snapshot = Mock(return_value = capability)
    probe = Mock(return_value = (True, "controlled seam"))
    request = Mock(return_value = {"isolationVariant": "nested"})
    spawn = Mock(return_value = object())
    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    monkeypatch.setattr(srt_probe, "probe", probe)
    monkeypatch.setattr(srt_adapter, "request_for", request)
    monkeypatch.setattr(srt_adapter, "spawn", spawn)
    host_spawn = Mock(side_effect = AssertionError("unsandboxed fallback"))
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", host_spawn)
    return spec, capability, snapshot, probe, request, spawn, host_spawn


@pytest.mark.parametrize(
    "failure",
    [
        "grant",
        "subject",
        "session",
        "ineligible",
        "standard_recovered",
        "probe",
        "generation",
        "cancel",
        "allowlist",
    ],
)
def test_nested_refusals_start_zero_tool_processes(admission, failure):
    spec, cap, snapshot, probe, request, spawn, host_spawn = admission
    if failure == "grant":
        spec = replace(spec, nested_grant = "not-issued")
    if failure == "subject":
        spec = replace(spec, current_subject = "other")
    if failure == "session":
        spec = replace(spec, tool_ui_session_id = "other")
    if failure == "ineligible":
        snapshot.return_value = replace(cap, nested_eligible = False)
    if failure == "standard_recovered":
        snapshot.return_value = replace(cap, available = True)
    if failure == "probe":
        probe.return_value = (False, "bounded timeout")
    if failure == "generation":
        snapshot.side_effect = [cap, cap, replace(cap, probe_generation = "new")]
    if failure == "cancel":
        spec.cancel_event.set()
    if failure == "allowlist":
        spec = replace(spec, network_policy = "allowlist")
    with pytest.raises(os_sandbox.SandboxUnavailableError):
        os_sandbox.prepare_tool_launch(spec)
    spawn.assert_not_called()
    host_spawn.assert_not_called()
    request.assert_not_called()


def test_nested_record_and_final_revalidation(admission):
    spec, cap, snapshot, probe, request, spawn, host_spawn = admission
    prepared = os_sandbox.prepare_tool_launch(spec)
    record = prepared.execution_record
    assert record.effective_mode == record.requested_mode == "container_isolation"
    assert record.profile_id == srt_nested.NESTED_PROFILE
    assert record.authority_disclosure == srt_nested.NESTED_DISCLOSURE
    assert record.network_policy == "deny"
    assert request.call_args.kwargs["isolation_variant"] == "nested"
    assert probe.call_args.kwargs == dict(
        force = True,
        execution_kind = "python",
        selected_executable = sys.executable,
        isolation_variant = "nested",
    )
    snapshot.return_value = replace(cap, probe_generation = "changed-after-preparation")
    with pytest.raises(os_sandbox.SandboxUnavailableError):
        os_sandbox.spawn_prepared_launch(prepared)
    spawn.assert_not_called()
    host_spawn.assert_not_called()


def test_valid_nested_admission_uses_only_srt_once(admission):
    spec, cap, snapshot, probe, request, spawn, host_spawn = admission
    prepared = os_sandbox.prepare_tool_launch(spec)
    assert os_sandbox.spawn_prepared_launch(prepared) is spawn.return_value
    assert spawn.call_count == 1
    assert probe.call_count == 2
    host_spawn.assert_not_called()

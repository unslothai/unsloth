# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Exercise authorization at the actual Python/Terminal execution boundary."""

import pytest
import threading

from core.inference import os_sandbox, tool_isolation, tools


@pytest.fixture
def tool_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id: str(tmp_path))
    from core.inference import srt_probe

    monkeypatch.setattr(
        srt_probe, "probe", lambda **kwargs: (False, "controlled unavailable backend")
    )
    return tmp_path


def _grant():
    capability = tool_isolation.capability_snapshot()
    return tool_isolation.issue_limited_grant(
        current_subject = "srt-contract-user",
        tool_ui_session_id = "srt-contract-session",
        probe_generation = capability.probe_generation,
    )


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_required_refuses_before_payload_and_has_no_record(tool_session, kind):
    records = []
    execute = tools._python_exec if kind == "python" else tools._bash_exec
    payload = "print('PAYLOAD_STARTED')" if kind == "python" else "echo PAYLOAD_STARTED"
    result = execute(payload, None, 5, "call", launch_record_callback = records.append)
    assert "OS_ISOLATION_UNAVAILABLE" in result
    assert "PAYLOAD_STARTED" not in result
    assert records == []


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_limited_requires_consent_at_payload_boundary(tool_session, kind):
    execute = tools._python_exec if kind == "python" else tools._bash_exec
    payload = "print('PAYLOAD_STARTED')" if kind == "python" else "echo PAYLOAD_STARTED"
    records = []
    result = execute(
        payload,
        None,
        5,
        "call",
        tool_execution_mode = "limited",
        current_subject = "srt-contract-user",
        tool_ui_session_id = "srt-contract-session",
        limited_grant = "invalid",
        launch_record_callback = records.append,
    )
    assert "authorization failed" in result
    assert "PAYLOAD_STARTED" not in result
    assert records == []


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_consented_limited_executes_selected_tool_and_records_actual_mode(tool_session, kind):
    grant = _grant()
    records = []
    execute = tools._python_exec if kind == "python" else tools._bash_exec
    payload = "print('SRT_LIMITED_EXECUTED')" if kind == "python" else "echo SRT_LIMITED_EXECUTED"
    result = execute(
        payload,
        None,
        10,
        "call",
        tool_execution_mode = "limited",
        current_subject = "srt-contract-user",
        tool_ui_session_id = "srt-contract-session",
        limited_grant = grant.token,
        launch_record_callback = records.append,
    )
    assert "SRT_LIMITED_EXECUTED" in result, result
    assert len(records) == 1
    assert records[0].backend == "process-guard"
    assert records[0].os_isolation is False
    assert records[0].network_policy == "unrestricted"


def test_full_does_not_depend_on_srt_probe(tool_session, monkeypatch):
    monkeypatch.setattr(
        os_sandbox, "capability_snapshot", lambda **kwargs: pytest.fail("Full probed SRT")
    )
    records = []
    result = tools._python_exec(
        "print('FULL_EXECUTED')",
        None,
        10,
        "call",
        disable_sandbox = True,
        launch_record_callback = records.append,
    )
    assert "FULL_EXECUTED" in result
    assert records[0].effective_mode == "full"
    assert records[0].os_isolation is False


def test_cancelled_request_never_spawns(tool_session, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(
        os_sandbox.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("cancelled request spawned"),
    )
    result = tools._python_exec(
        "print('SHOULD_NOT_RUN')",
        timeout = 10,
        session_id = "call",
        cancel_event = cancelled,
        disable_sandbox = True,
    )
    assert "cancelled" in result.lower()
    assert "SHOULD_NOT_RUN" not in result


def test_disabled_timeout_is_not_reported_as_enforced(tool_session):
    prepared = os_sandbox.prepare_tool_launch(
        os_sandbox.ToolLaunchPlan(
            argv = ("python", "-c", "print(1)"),
            workdir = str(tool_session),
            env = {},
            requested_mode = "full",
            timeout_seconds = None,
        )
    )
    assert "timeout" not in prepared.execution_record.retained_safeguards

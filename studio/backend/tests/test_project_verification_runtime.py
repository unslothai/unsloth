# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime contracts for revision-bound project verification."""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from core.agent_workspace import (
    verification_process as supervisor,
    verification,
    verification_state,
)
from core.agent_workspace import verification_process as execution
from core.agent_workspace.verification_context import AgentWorkspaceError, ProjectWorkspace
from core.inference import tools as inference_tools
from storage import studio_db


def _workspace(
    tmp_path: Path,
    project_id: str = "project",
    revision: int = 7,
):
    root = tmp_path / project_id
    root.mkdir(exist_ok = True)
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "folder",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
        revision = revision,
    )


def _check(
    name: str,
    command: str,
    *,
    required: bool = True,
    timeout: int = 19,
    log_limit: int = 8192,
):
    return {
        "name": name,
        "kind": "custom",
        "command": command,
        "required": required,
        "timeoutSeconds": timeout,
        "logLimitBytes": log_limit,
    }


def _profile(workspace: ProjectWorkspace, checks: list[dict]):
    return {
        "projectId": workspace.project_id,
        "checks": checks,
        "revision": 3,
        "configHash": "a" * 64,
        "workspaceDeviceId": int(workspace.device_id),
        "workspaceFileId": int(workspace.file_id),
        "workspaceRevision": int(workspace.revision),
        "updatedAt": 1,
    }


def _process_result(
    status: str = "passed",
    *,
    output: str = "ok",
    exit_code: int | None = 0,
    output_bytes: int | None = None,
    output_truncated: bool = False,
):
    return supervisor.ProjectProcessResult(
        status = status,
        exit_code = exit_code,
        output = output,
        output_bytes = (len(output.encode("utf-8")) if output_bytes is None else output_bytes),
        output_truncated = output_truncated,
    )


def _create_stored_project(project_id: str) -> None:
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            INSERT INTO chat_projects (
                id, name, instructions, archived, created_at, updated_at
            ) VALUES (?, 'Verification Project', '', 0, 1, 1)
            """,
            (project_id,),
        )
        connection.commit()
    finally:
        connection.close()


class _StateHarness:
    def __init__(self):
        self.progress: list[list[dict]] = []
        self.completions: list[dict] = []
        self.cancel_requested: set[tuple[str, str]] = set()
        self.cancel_observer = None
        self.owner_ids: list[str] = []

    def update(self, project_id: str, run_id: str, results: list[dict], *, owner_id: str):
        assert project_id
        assert run_id
        self.owner_ids.append(owner_id)
        self.progress.append([dict(result) for result in results])
        return {"projectId": project_id, "id": run_id, "status": "running"}

    def complete(
        self,
        project_id: str,
        run_id: str,
        results: list[dict],
        *,
        owner_id: str,
        terminal_status = None,
        error = None,
    ):
        self.owner_ids.append(owner_id)
        if (project_id, run_id) in self.cancel_requested:
            status = "cancelled"
        elif terminal_status is not None:
            status = terminal_status
        elif any(result.get("required") and result.get("status") != "passed" for result in results):
            status = "failed"
        else:
            status = "passed"
        record = {
            "projectId": project_id,
            "id": run_id,
            "status": status,
            "results": [dict(result) for result in results],
            "terminalStatus": terminal_status,
            "error": error,
        }
        self.completions.append(record)
        return record

    def request_cancel(self, project_id: str, run_id: str):
        if self.cancel_observer is not None:
            self.cancel_observer()
        self.cancel_requested.add((project_id, run_id))
        return (
            {"projectId": project_id, "id": run_id, "status": "running"},
            True,
        )


def _install_state_harness(monkeypatch, state: _StateHarness):
    monkeypatch.setattr(
        verification.verification_state,
        "update_verification_run_progress",
        state.update,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "complete_verification_run",
        state.complete,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "request_verification_cancel",
        state.request_cancel,
    )


@pytest.fixture(autouse = True)
def _clear_runtime_owners():
    with verification._ACTIVE_LOCK:
        stale = list(verification._ACTIVE_BY_PROJECT.values())
        stale_deletions = list(verification._DELETING_PROJECTS.values())
        verification._ACTIVE_BY_PROJECT.clear()
        verification._ACTIVE_BY_RUN.clear()
        verification._DELETING_PROJECTS.clear()
    for active in stale:
        active.cancel_event.set()
    for deletion in stale_deletions:
        deletion.stopped.set()
    yield
    with verification._ACTIVE_LOCK:
        stale = list(verification._ACTIVE_BY_PROJECT.values())
        stale_deletions = list(verification._DELETING_PROJECTS.values())
    for active in stale:
        active.cancel_event.set()
        active.completed.wait(timeout = 2)
    for deletion in stale_deletions:
        deletion.stopped.set()
    with verification._ACTIVE_LOCK:
        verification._ACTIVE_BY_PROJECT.clear()
        verification._ACTIVE_BY_RUN.clear()
        verification._DELETING_PROJECTS.clear()


def test_supervisor_entrypoint_preserves_sealed_authority_cancel_event_and_limits(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    check = _check("tests", "python -m pytest", timeout = 23, log_limit = 12345)
    profile = _profile(workspace, [check])
    capability = verification._capability(profile, workspace, check, "run-sealed")
    cancellation = threading.Event()
    callback = lambda _chunk: None
    observed = {}

    def may_spawn(project_id, run_id, owner_id, **proof):
        observed["proof"] = (project_id, run_id, owner_id, proof)
        return True

    def run(project_id, argv, **options):
        observed["run"] = (project_id, tuple(argv), options)
        options["before_start"](workspace, argv)
        return _process_result()

    monkeypatch.setattr(
        verification.verification_state,
        "verification_run_may_spawn",
        may_spawn,
    )
    monkeypatch.setattr(supervisor.common, "project_workspace", lambda _project_id: workspace)
    monkeypatch.setattr(supervisor, "run_project_process", run)

    result = supervisor._run_project_verification_process(
        capability,
        timeout_seconds = check["timeoutSeconds"],
        output_limit_bytes = check["logLimitBytes"],
        cancel_event = cancellation,
        output_callback = callback,
    )

    assert result.status == "passed"
    assert capability._seal is verification._CAPABILITY_SEAL
    assert capability.project_id == workspace.project_id
    assert capability.run_id == "run-sealed"
    assert capability.owner_id == verification.PROCESS_OWNER_ID
    assert capability.config_revision == profile["revision"]
    assert capability.config_hash == profile["configHash"]
    assert capability.workspace_identity == (int(workspace.device_id), int(workspace.file_id))
    assert capability.workspace_revision == workspace.revision
    assert capability.argv == verification._shell_argv(check["command"])
    assert capability.argv[0] == ("cmd.exe" if os.name == "nt" else "/bin/sh")
    project_id, argv, options = observed["run"]
    assert project_id == workspace.project_id
    assert argv == capability.argv
    assert options["timeout_seconds"] == check["timeoutSeconds"]
    assert options["output_limit_bytes"] == check["logLimitBytes"]
    assert options["cancel_event"] is cancellation
    assert options["output_callback"] is callback
    assert observed["proof"] == (
        workspace.project_id,
        "run-sealed",
        verification.PROCESS_OWNER_ID,
        {
            "revision": profile["revision"],
            "config_hash": profile["configHash"],
            "workspace_identity": capability.workspace_identity,
            "workspace_revision": workspace.revision,
        },
    )


@pytest.mark.parametrize("stale_part", ["seal", "config", "workspace"])
def test_capability_drift_is_rejected_immediately_before_spawn(stale_part, tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    check = _check("tests", "python -m pytest")
    profile = _profile(workspace, [check])
    capability = verification._capability(profile, workspace, check, "run-drift")
    before_spawn_finished = False

    if stale_part == "seal":
        capability = replace(capability, _seal = object())
    elif stale_part == "workspace":
        workspace = replace(workspace, revision = workspace.revision + 1)

    monkeypatch.setattr(
        verification.verification_state,
        "verification_run_may_spawn",
        lambda *_args, **_kwargs: stale_part != "config",
    )

    def run(_project_id, _argv, **options):
        nonlocal before_spawn_finished
        options["before_start"](workspace, _argv)
        before_spawn_finished = True
        return _process_result()

    monkeypatch.setattr(supervisor.common, "project_workspace", lambda _project_id: workspace)
    monkeypatch.setattr(supervisor, "run_project_process", run)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("stale verification authority reached Popen"),
    )

    with pytest.raises(AgentWorkspaceError, match = "changed"):
        supervisor._run_project_verification_process(
            capability,
            timeout_seconds = 10,
            output_limit_bytes = 4096,
            cancel_event = threading.Event(),
        )
    assert before_spawn_finished is False


@pytest.mark.parametrize(
    "results, expected_status",
    [
        ([("optional", False, "failed"), ("required", True, "passed")], "passed"),
        ([("required", True, "failed"), ("optional", False, "passed")], "failed"),
    ],
)
def test_live_progress_and_required_optional_aggregation_use_ordered_supervisor_results(
    results, expected_status, tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    checks = [_check(name, f"run-{name}", required = required) for name, required, _status in results]
    profile = _profile(workspace, checks)
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-1")
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    calls = []

    def run(capability, **options):
        index = len(calls)
        calls.append((capability, options))
        options["output_callback"](f"partial-{index}")
        status = results[index][2]
        return _process_result(
            status,
            output = f"final-{index}",
            exit_code = 0 if status == "passed" else 9,
        )

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)

    verification._execute_run(active, profile, workspace)

    assert active.completed.is_set()
    assert state.owner_ids
    assert set(state.owner_ids) == {verification.PROCESS_OWNER_ID}
    assert [call[0].argv[-1] for call in calls] == [check["command"] for check in checks]
    assert all(call[1]["cancel_event"] is active.cancel_event for call in calls)
    assert [call[1]["timeout_seconds"] for call in calls] == [
        check["timeoutSeconds"] for check in checks
    ]
    assert [call[1]["output_limit_bytes"] for call in calls] == [
        check["logLimitBytes"] for check in checks
    ]
    for snapshot in state.progress:
        assert [result["name"] for result in snapshot] == [
            check["name"] for check in checks[: len(snapshot)]
        ]
        running_indexes = [
            index for index, result in enumerate(snapshot) if result["status"] == "running"
        ]
        assert running_indexes in ([], [len(snapshot) - 1])
    terminal = state.completions[-1]
    assert terminal["status"] == expected_status
    assert terminal["terminalStatus"] is None
    assert [result["name"] for result in terminal["results"]] == [
        name for name, _required, _status in results
    ]
    assert [result["output"] for result in terminal["results"]] == [
        f"final-{index}" for index in range(len(results))
    ]


def test_live_output_sqlite_runs_off_reader_path_and_cannot_block_cancellation(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-publisher")
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    publisher_entered = threading.Event()
    release_publisher = threading.Event()
    callback_returned = threading.Event()
    cancellation_observed = threading.Event()
    publisher_threads: list[int] = []
    reader_threads: list[int] = []
    original_update = state.update

    def blocking_update(project_id, run_id, results, *, owner_id):
        if results and results[-1]["status"] == "running" and results[-1]["output"]:
            publisher_threads.append(threading.get_ident())
            publisher_entered.set()
            assert release_publisher.wait(timeout = 2)
        return original_update(
            project_id,
            run_id,
            results,
            owner_id = owner_id,
        )

    monkeypatch.setattr(
        verification.verification_state,
        "update_verification_run_progress",
        blocking_update,
    )

    def run(_capability, **options):
        reader_threads.append(threading.get_ident())
        options["output_callback"]("live output")
        callback_returned.set()
        assert options["cancel_event"].wait(timeout = 2)
        cancellation_observed.set()
        return _process_result("cancelled", output = "live output", exit_code = None)

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)
    worker = threading.Thread(
        target = verification._execute_run,
        args = (active, profile, workspace),
    )
    worker.start()
    try:
        assert publisher_entered.wait(timeout = 2)
        assert callback_returned.wait(timeout = 1)
        active.cancel_event.set()
        assert cancellation_observed.wait(timeout = 1)
    finally:
        active.cancel_event.set()
        release_publisher.set()
        worker.join(timeout = 2)

    assert not worker.is_alive()
    assert publisher_threads and reader_threads
    assert publisher_threads[0] != reader_threads[0]
    assert state.completions[-1]["status"] == "cancelled"
    assert state.completions[-1]["results"][0]["output"] == "live output"


def test_invalid_utf8_live_snapshots_remain_terminal_output_prefix(monkeypatch):
    native = pytest.importorskip("core.agent_workspace.supervisor")
    chunks = [
        bytes.fromhex("91"),
        bytes.fromhex("547819d54e94"),
        bytes.fromhex("289fdc18d883d5da"),
        bytes.fromhex("ae"),
        bytes.fromhex("d6b1f818"),
        bytes.fromhex("2da6dc810cf9"),
        b"",
    ]
    pending = iter(chunks)
    streamed: list[str] = []
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: next(pending))

    output = native._OutputBuffer(12, streamed.append)
    assert output.read_available(123, max_chunks = 64) is True
    terminal_output, output_bytes, output_truncated, truncation_notice = output.result()

    assert output_bytes == sum(len(chunk) for chunk in chunks)
    assert output_truncated is True
    assert streamed[-1] == truncation_notice
    check = _check("tests", "run-tests")
    started_at = int(time.time() * 1000)
    previous: list[dict] = []
    visible = ""
    for chunk in streamed:
        visible += chunk
        snapshot = verification._bounded_utf8(visible, 12)
        assert terminal_output.startswith(snapshot)
        current = [verification._running_result(check, started_at, snapshot)]
        verification_state._validate_progress_transition(previous, current)
        previous = current

    terminal = [
        verification._terminal_result(
            check,
            _process_result(
                output = terminal_output,
                output_bytes = output_bytes,
                output_truncated = output_truncated,
            ),
            started_at,
        )
    ]
    verification_state._validate_progress_transition(previous, terminal)


def test_terminal_and_error_timing_survive_a_backward_wall_clock(monkeypatch):
    check = _check("tests", "run-tests")
    started_at = 10_000
    monkeypatch.setattr(verification.time, "time", lambda: 9.0)
    monkeypatch.setattr(verification.time, "monotonic", lambda: 25.25)

    terminal = verification._terminal_result(
        check,
        _process_result(),
        started_at,
        started_monotonic = 25.0,
    )
    failed = verification._error_result(
        check,
        started_at,
        "failed",
        RuntimeError("failed"),
        started_monotonic = 25.0,
    )
    clamped_terminal = verification._terminal_result(
        check,
        _process_result(),
        started_at,
    )
    clamped_error = verification._error_result(
        check,
        started_at,
        "failed",
        RuntimeError("failed"),
    )

    for result in (terminal, failed):
        assert result["completedAt"] == 10_250
        assert result["durationMs"] == 250
        verification_state._validate_result(result, check, run_started_at = started_at)
    for result in (clamped_terminal, clamped_error):
        assert result["completedAt"] == started_at
        assert result["durationMs"] == 0
        verification_state._validate_result(result, check, run_started_at = started_at)


def test_execute_run_never_starts_a_check_before_its_durable_run(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    check = _check("tests", "run-tests")
    profile = _profile(workspace, [check])
    run_started_at = 10_000
    active = verification._ActiveVerification(
        workspace.project_id,
        run_id = "run-clock-anchor",
        run_started_at = run_started_at,
        run_started_monotonic = 50.0,
    )
    state = _StateHarness()

    def strict_update(project_id, run_id, results, *, owner_id):
        for result in results:
            verification_state._validate_result(
                result,
                check,
                run_started_at = run_started_at,
            )
        return state.update(project_id, run_id, results, owner_id = owner_id)

    def strict_complete(project_id, run_id, results, **options):
        for result in results:
            verification_state._validate_result(
                result,
                check,
                run_started_at = run_started_at,
            )
        return state.complete(project_id, run_id, results, **options)

    monkeypatch.setattr(
        verification.verification_state,
        "update_verification_run_progress",
        strict_update,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "complete_verification_run",
        strict_complete,
    )
    monkeypatch.setattr(verification.time, "time", lambda: 9.0)
    monkeypatch.setattr(verification.time, "monotonic", lambda: 50.25)
    monkeypatch.setattr(
        supervisor,
        "_run_project_verification_process",
        lambda *_args, **_options: _process_result(),
    )

    verification._execute_run(active, profile, workspace)

    assert state.completions[-1]["status"] == "passed"
    [result] = state.completions[-1]["results"]
    assert result["startedAt"] == 10_250
    assert result["completedAt"] == 10_250
    assert active.completed.is_set()


def test_progress_publisher_start_failure_terminalizes_and_releases(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-start-fail")
    with verification._ACTIVE_LOCK:
        verification._ACTIVE_BY_PROJECT[workspace.project_id] = active
        verification._ACTIVE_BY_RUN[active.run_id] = active
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    stopped = threading.Event()

    class FailingPublisher:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("publisher thread unavailable")

        def stop(self):
            stopped.set()

    monkeypatch.setattr(verification, "_ProgressPublisher", FailingPublisher)
    monkeypatch.setattr(
        supervisor,
        "_run_project_verification_process",
        lambda *_args, **_kwargs: pytest.fail("publisher failure reached command execution"),
    )

    verification._execute_run(active, profile, workspace)

    assert state.completions[-1]["status"] == "failed"
    assert state.completions[-1]["results"] == []
    assert state.completions[-1]["error"] == (
        "Verification failed before every check could complete."
    )
    assert stopped.is_set()
    assert active.completed.is_set()
    assert verification._ACTIVE_BY_PROJECT == {}
    assert verification._ACTIVE_BY_RUN == {}


def test_stale_first_check_progress_does_not_stop_second_check_publication(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    checks = [
        _check("first", "run-first"),
        _check("second", "run-second"),
    ]
    profile = _profile(workspace, checks)
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-stale")
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    first_publish_entered = threading.Event()
    release_first_publish = threading.Event()
    first_publish_rejected = threading.Event()
    second_publish_persisted = threading.Event()
    original_update = state.update

    def racing_update(project_id, run_id, results, *, owner_id):
        tail = results[-1] if results else None
        if tail and tail["name"] == "first" and tail["status"] == "running" and tail["output"]:
            first_publish_entered.set()
            assert release_first_publish.wait(timeout = 2)
            first_publish_rejected.set()
            raise verification_state.VerificationConflictError("Progress was superseded.")
        record = original_update(
            project_id,
            run_id,
            results,
            owner_id = owner_id,
        )
        if tail and tail["name"] == "first" and tail["status"] == "passed":
            release_first_publish.set()
        if tail and tail["name"] == "second" and tail["status"] == "running" and tail["output"]:
            second_publish_persisted.set()
        return record

    monkeypatch.setattr(
        verification.verification_state,
        "update_verification_run_progress",
        racing_update,
    )
    calls = []

    def run(_capability, **options):
        index = len(calls)
        calls.append(index)
        if index == 0:
            options["output_callback"]("first live")
            assert first_publish_entered.wait(timeout = 2)
            return _process_result(output = "first final")
        assert first_publish_rejected.wait(timeout = 2)
        options["output_callback"]("second live")
        assert second_publish_persisted.wait(timeout = 2)
        return _process_result(output = "second final")

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)

    verification._execute_run(active, profile, workspace)

    assert calls == [0, 1]
    assert second_publish_persisted.is_set()
    assert active.cancel_event.is_set() is False
    assert any(
        snapshot[-1]["name"] == "second"
        and snapshot[-1]["status"] == "running"
        and snapshot[-1]["output"] == "second live"
        for snapshot in state.progress
    )
    assert state.completions[-1]["status"] == "passed"
    assert [result["output"] for result in state.completions[-1]["results"]] == [
        "first final",
        "second final",
    ]


def test_aggregate_output_budget_reserves_every_truncation_decoration(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    checks = [
        _check("first", "run-first", log_limit = 30),
        _check("second", "run-second", log_limit = 30),
    ]
    profile = _profile(workspace, checks)
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-budget")
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    monkeypatch.setattr(verification_state, "MAX_RUN_RESULT_BYTES", 32)
    monkeypatch.setattr(verification_state, "MAX_TRUNCATION_DECORATION_BYTES", 4)
    observed_limits = []

    def run(_capability, **options):
        limit = options["output_limit_bytes"]
        observed_limits.append(limit)
        output = "x" * (limit + verification_state.MAX_TRUNCATION_DECORATION_BYTES)
        return _process_result(
            output = output,
            output_bytes = limit + 100,
            output_truncated = True,
        )

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)

    verification._execute_run(active, profile, workspace)

    assert observed_limits == [23, 1]
    final_results = state.completions[-1]["results"]
    assert sum(len(result["output"].encode("utf-8")) for result in final_results) == 32
    assert state.completions[-1]["status"] == "passed"


def test_control_heavy_output_fits_the_persisted_json_budget():
    first_size = verification_state.MAX_LOG_LIMIT_BYTES
    second_size = verification_state.MAX_RUN_RESULT_BYTES - first_size
    payload = [
        {"output": "\x01" * first_size},
        {"output": "\x02" * second_size},
    ]

    encoded = verification_state._canonical_json(
        payload,
        limit = verification_state.MAX_RUN_RESULTS_JSON_BYTES,
        label = "Verification results",
    )

    assert sum(len(item["output"].encode("utf-8")) for item in payload) == (
        verification_state.MAX_RUN_RESULT_BYTES
    )
    assert len(encoded.encode("utf-8")) <= verification_state.MAX_RUN_RESULTS_JSON_BYTES


def test_cancel_is_persisted_before_signal_and_wins_completion_race(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-race")
    with verification._ACTIVE_LOCK:
        verification._ACTIVE_BY_PROJECT[workspace.project_id] = active
        verification._ACTIVE_BY_RUN[active.run_id] = active
    state = _StateHarness()
    state.cancel_observer = lambda: (
        pytest.fail("cancel signal preceded durable request")
        if active.cancel_event.is_set()
        else None
    )
    _install_state_harness(monkeypatch, state)
    entered = threading.Event()
    release = threading.Event()

    def run(_capability, **_options):
        entered.set()
        assert release.wait(timeout = 2)
        return _process_result("passed", output = "completed")

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)
    worker = threading.Thread(target = verification._execute_run, args = (active, profile, workspace))
    worker.start()
    assert entered.wait(timeout = 2)

    run_record, accepted = verification.cancel_verification(workspace.project_id, active.run_id)
    assert accepted is True
    assert run_record["id"] == active.run_id
    assert active.cancel_event.is_set()
    release.set()
    worker.join(timeout = 2)

    assert not worker.is_alive()
    assert state.completions[-1]["status"] == "cancelled"
    assert state.completions[-1]["results"][0]["status"] == "passed"
    assert active.completed.is_set()


def test_async_start_returns_running_while_the_supervisor_owns_execution(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    monkeypatch.setattr(
        verification.verification_state,
        "get_verification_config",
        lambda project_id: profile if project_id == workspace.project_id else None,
    )
    begin_proofs = []

    def begin(project_id, **proof):
        begin_proofs.append((project_id, proof))
        return {
            "projectId": project_id,
            "id": "run-async",
            "status": "running",
            "results": [],
            "startedAt": 1,
        }

    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_run",
        begin,
    )

    @contextlib.contextmanager
    def access(project_id, **_options):
        assert project_id == workspace.project_id
        yield workspace

    monkeypatch.setattr(verification.common, "project_workspace_access", access)
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_run",
        lambda *_args, **_kwargs: False,
    )
    entered = threading.Event()
    release = threading.Event()

    def run(_capability, **options):
        entered.set()
        assert not options["cancel_event"].is_set()
        assert release.wait(timeout = 2)
        return _process_result()

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)

    active = None
    record = verification.start_project_verification(
        workspace.project_id,
        config_revision = profile["revision"],
        workspace_revision = workspace.revision,
    )
    try:
        assert record == {
            "projectId": workspace.project_id,
            "id": "run-async",
            "status": "running",
            "results": [],
            "startedAt": 1,
        }
        assert begin_proofs == [
            (
                workspace.project_id,
                {
                    "config_revision": profile["revision"],
                    "config_hash": profile["configHash"],
                    "checks": profile["checks"],
                    "workspace_identity": (
                        int(workspace.device_id),
                        int(workspace.file_id),
                    ),
                    "workspace_revision": workspace.revision,
                    "owner_id": verification.PROCESS_OWNER_ID,
                },
            )
        ]
        assert entered.wait(timeout = 2)
        with verification._ACTIVE_LOCK:
            active = verification._ACTIVE_BY_RUN["run-async"]
        assert active.completed.is_set() is False
    finally:
        release.set()
        if active is not None:
            active.completed.wait(timeout = 2)
    assert state.completions[-1]["status"] == "passed"


def test_background_run_holds_managed_workspace_lease_until_release(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "workspace-lease-project")
    profile = _profile(workspace, [_check("tests", "run-tests")])
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    monkeypatch.setattr(
        verification.verification_state,
        "get_verification_config",
        lambda project_id: profile if project_id == workspace.project_id else None,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_run",
        lambda project_id, **_proof: {
            "projectId": project_id,
            "id": "run-workspace-lease",
            "status": "running",
            "results": [],
            "startedAt": 1,
        },
    )
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_run",
        lambda *_args, **_kwargs: False,
    )

    @contextlib.contextmanager
    def access(
        project_id,
        *,
        cancel_event = None,
        deadline = None,
    ):
        assert project_id == workspace.project_id
        with inference_tools._session_in_flight(
            inference_tools.project_session_id(project_id),
            cancel_event = cancel_event,
            deadline = deadline,
        ):
            yield workspace

    monkeypatch.setattr(verification.common, "project_workspace_access", access)
    entered = threading.Event()
    release = threading.Event()

    def run(_capability, **_options):
        entered.set()
        assert release.wait(timeout = 2)
        return _process_result()

    monkeypatch.setattr(supervisor, "_run_project_verification_process", run)

    active = None
    verification.start_project_verification(
        workspace.project_id,
        config_revision = profile["revision"],
        workspace_revision = workspace.revision,
    )
    try:
        assert entered.wait(timeout = 2)
        key = inference_tools._session_key(inference_tools.project_session_id(workspace.project_id))
        with inference_tools._active_sessions_lock:
            assert inference_tools._active_sessions.get(key, 0) > 0
        with verification._ACTIVE_LOCK:
            active = verification._ACTIVE_BY_RUN["run-workspace-lease"]
    finally:
        release.set()
        if active is not None:
            assert active.completed.wait(timeout = 2)

    with inference_tools._active_sessions_lock:
        assert inference_tools._active_sessions.get(key, 0) == 0
    assert state.completions[-1]["status"] == "passed"


@pytest.mark.parametrize("drift", ["workspace_revision", "config_revision", "workspace_binding"])
def test_start_rejects_workspace_or_config_drift_before_supervisor(drift, tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    requested_workspace_revision = workspace.revision
    requested_config_revision = profile["revision"]
    if drift == "workspace_revision":
        requested_workspace_revision += 1
    elif drift == "config_revision":
        requested_config_revision += 1
    else:
        profile["workspaceFileId"] = int(workspace.file_id) + 1

    @contextlib.contextmanager
    def access(project_id, **_options):
        assert project_id == workspace.project_id
        yield workspace

    monkeypatch.setattr(verification.common, "project_workspace_access", access)
    monkeypatch.setattr(
        verification.verification_state,
        "get_verification_config",
        lambda _project_id: profile,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_run",
        lambda *_args, **_kwargs: pytest.fail("drift created a durable run"),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_project_verification_process",
        lambda *_args, **_kwargs: pytest.fail("drift reached the supervisor"),
    )

    with pytest.raises(AgentWorkspaceError, match = "changed|different"):
        verification.start_project_verification(
            workspace.project_id,
            config_revision = requested_config_revision,
            workspace_revision = requested_workspace_revision,
        )
    assert verification._ACTIVE_BY_PROJECT == {}
    assert verification._ACTIVE_BY_RUN == {}


@pytest.mark.parametrize(
    "reason",
    [
        "Project command execution is disabled until a Windows filesystem sandbox is available.",
        "macOS sandbox-exec cannot prove detached descendants are gone.",
    ],
)
def test_unavailable_native_supervision_is_blocked_without_popen(reason, tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    profile = _profile(workspace, [_check("tests", "run-tests")])
    active = verification._ActiveVerification(workspace.project_id, run_id = "run-blocked")
    state = _StateHarness()
    _install_state_harness(monkeypatch, state)
    monkeypatch.setattr(
        verification.verification_state,
        "verification_run_may_spawn",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        supervisor,
        "run_project_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            execution.ProjectExecutionUnavailable(reason)
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("unavailable supervision reached Popen"),
    )

    verification._execute_run(active, profile, workspace)

    terminal = state.completions[-1]
    assert terminal["status"] == "blocked"
    assert terminal["terminalStatus"] == "blocked"
    assert terminal["results"][0]["status"] == "blocked"
    assert reason in terminal["results"][0]["error"]


def test_capacity_same_project_and_cross_project_cancel_are_scoped(monkeypatch):
    first = verification._reserve_project("project-0")
    with pytest.raises(AgentWorkspaceError, match = "already running"):
        verification._reserve_project("project-0")
    for index in range(1, verification.MAX_ACTIVE_VERIFICATIONS):
        verification._reserve_project(f"project-{index}")
    with pytest.raises(AgentWorkspaceError, match = "capacity"):
        verification._reserve_project("overflow")

    first.run_id = "run-0"
    with verification._ACTIVE_LOCK:
        verification._ACTIVE_BY_RUN[first.run_id] = first

    def refuse_cross_project(project_id, run_id):
        assert (project_id, run_id) == ("other-project", "run-0")
        raise AgentWorkspaceError("Verification run not found.")

    monkeypatch.setattr(
        verification.verification_state,
        "request_verification_cancel",
        refuse_cross_project,
    )
    with pytest.raises(AgentWorkspaceError, match = "not found"):
        verification.cancel_verification("other-project", "run-0")
    assert first.cancel_event.is_set() is False


def test_non_domain_heartbeat_failure_cancels_the_owned_run(monkeypatch):
    active = verification._ActiveVerification("heartbeat-project", run_id = "run-heartbeat")
    monkeypatch.setattr(
        verification.verification_state,
        "HEARTBEAT_INTERVAL_SECONDS",
        0,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    verification._heartbeat_run(active)

    assert active.cancel_event.is_set()


def test_project_deletion_fences_new_runs_and_cancel_wait_is_bounded(monkeypatch):
    project_id = "delete-project"
    active = verification._reserve_project(project_id)
    active.run_id = "run-delete"
    with verification._ACTIVE_LOCK:
        verification._ACTIVE_BY_RUN[active.run_id] = active
    persisted = []
    fences = []

    def begin_fence(requested_project, fence_id):
        fences.append(("begin", requested_project, fence_id))
        return {"revision": 7}

    def finish_fence(requested_project, fence_id, revision):
        fences.append(("finish", requested_project, fence_id, revision))

    def request_cancel(requested_project):
        persisted.append((requested_project, active.cancel_event.is_set()))
        return (
            {
                "projectId": requested_project,
                "id": active.run_id,
                "status": "running",
            },
            True,
        )

    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_project_deletion",
        begin_fence,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_project_deletion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "finish_verification_project_deletion",
        finish_fence,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "request_project_verification_cancel",
        request_cancel,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "active_verification_run_lifecycle",
        lambda requested_project: (
            {"projectId": requested_project, "id": active.run_id, "status": "running"}
            if not active.completed.is_set()
            else None
        ),
    )

    verification.begin_project_deletion(project_id)
    deletion = verification._DELETING_PROJECTS[project_id]
    try:
        assert fences == [("begin", project_id, deletion.fence_id)]
        with pytest.raises(AgentWorkspaceError, match = "deletion"):
            verification._reserve_project(project_id)

        started = time.monotonic()
        with pytest.raises(AgentWorkspaceError, match = "Timed out"):
            verification.cancel_project_verifications_and_wait(
                project_id,
                timeout_seconds = 0.01,
            )
        assert time.monotonic() - started < 1
        assert persisted == [(project_id, False)]
        assert active.cancel_event.is_set()

        verification._release_active(active)
        verification.cancel_project_verifications_and_wait(project_id, timeout_seconds = 1)
    finally:
        verification.finish_project_deletion(project_id)
    assert fences == [
        ("begin", project_id, deletion.fence_id),
        ("finish", project_id, deletion.fence_id, 7),
    ]
    assert persisted == [(project_id, False), (project_id, True)]
    replacement = verification._reserve_project(project_id)
    assert replacement.project_id == project_id


def test_deletion_heartbeat_uses_stable_owner_generation(monkeypatch):
    active = verification._ActiveDeletion(
        project_id = "delete-generation-project",
        fence_id = "delete-generation-fence",
        revision = 17,
    )
    calls = []

    def heartbeat(project_id, fence_id, revision):
        calls.append((project_id, fence_id, revision))
        active.stopped.set()

    monkeypatch.setattr(
        verification.verification_state,
        "DELETION_FENCE_HEARTBEAT_INTERVAL_SECONDS",
        0,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_project_deletion",
        heartbeat,
    )

    verification._heartbeat_deletion(active)

    assert calls == [(active.project_id, active.fence_id, 17)]
    assert not active.fence_lost.is_set()


@pytest.mark.parametrize("failure_point", ["construct", "start"])
def test_deletion_heartbeat_start_failure_releases_durable_fence(failure_point, monkeypatch):
    project_id = "delete-start-failure"
    calls = []

    def begin(requested_project, fence_id):
        calls.append(("begin", requested_project, fence_id))
        return {"revision": 11}

    def finish(requested_project, fence_id, revision):
        calls.append(("finish", requested_project, fence_id, revision))

    class FailingThread:
        def start(self):
            raise RuntimeError("heartbeat start failed")

    def thread_factory(**_kwargs):
        if failure_point == "construct":
            raise RuntimeError("heartbeat construction failed")
        return FailingThread()

    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_project_deletion",
        begin,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "finish_verification_project_deletion",
        finish,
    )
    monkeypatch.setattr(verification.threading, "Thread", thread_factory)

    with pytest.raises(RuntimeError, match = "heartbeat"):
        verification.begin_project_deletion(project_id)

    assert [call[0] for call in calls] == ["begin", "finish"]
    assert calls[0][1:] == calls[1][1:3]
    assert calls[1][3] == 11
    assert verification._DELETING_PROJECTS == {}


def test_retirement_poll_never_decodes_large_corrupt_evidence(tmp_path, monkeypatch):
    clock = [1_000]
    monkeypatch.setattr(verification_state, "_now_ms", lambda: clock[0])
    project_id = "corrupt-retirement-project"
    workspace = _workspace(tmp_path, project_id)
    _create_stored_project(project_id)
    profile = verification_state.set_verification_config(
        project_id,
        [_check("tests", "run-tests", log_limit = 2 * 1024 * 1024)],
        workspace_identity = (int(workspace.device_id), int(workspace.file_id)),
        workspace_revision = workspace.revision,
        expected_revision = 0,
    )
    run = verification_state.begin_verification_run(
        project_id,
        owner_id = verification.PROCESS_OWNER_ID,
        config_revision = profile["revision"],
        config_hash = profile["configHash"],
        checks = profile["checks"],
        workspace_identity = (int(workspace.device_id), int(workspace.file_id)),
        workspace_revision = workspace.revision,
    )
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE agent_verification_runs SET results_json = ? WHERE id = ?",
            ("x" * (4 * 1024 * 1024), run["id"]),
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(
        verification_state,
        "_run_from_row",
        lambda _row: pytest.fail("retirement decoded verification evidence"),
    )
    cancelled, accepted = verification_state.request_project_verification_cancel(project_id)
    assert accepted is True
    assert cancelled["id"] == run["id"]
    assert cancelled["cancelRequested"] is True
    clock[0] = run["leaseExpiresAt"] + 1
    monkeypatch.setattr(
        verification,
        "_acquire_deletion_execution_fence",
        lambda _active, _deadline: None,
    )

    verification.begin_project_deletion(project_id)
    try:
        verification.cancel_project_verifications_and_wait(project_id, timeout_seconds = 1)
    finally:
        verification.finish_project_deletion(project_id)

    connection = studio_db.get_connection()
    try:
        row = connection.execute(
            "SELECT status, history_sequence FROM agent_verification_runs WHERE id = ?",
            (run["id"],),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row["status"] == "cancelled"
    assert row["history_sequence"] is not None


@pytest.mark.skipif(os.name != "posix", reason = "POSIX execution fence required")
def test_retirement_wait_uses_the_supervisor_fence_until_lifecycle_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    project_id = "retirement-fence-project"
    fence_id = verification._execution_fence_id(project_id)
    prior_tree_fence = supervisor._acquire_project_execution_fence(
        fence_id,
        threading.Event(),
        time.monotonic() + 2,
    )
    wait_entered = threading.Event()
    wait_finished = threading.Event()
    mutation_started = threading.Event()
    failures = []
    original_acquire = supervisor._acquire_project_execution_fence
    observed_fence_ids = []

    def acquire(requested_fence_id, cancel_event, deadline):
        observed_fence_ids.append(requested_fence_id)
        wait_entered.set()
        return original_acquire(requested_fence_id, cancel_event, deadline)

    monkeypatch.setattr(supervisor, "_acquire_project_execution_fence", acquire)
    monkeypatch.setattr(
        verification.verification_state,
        "begin_verification_project_deletion",
        lambda *_args, **_kwargs: {"revision": 13},
    )
    monkeypatch.setattr(
        verification.verification_state,
        "heartbeat_verification_project_deletion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "finish_verification_project_deletion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verification.verification_state,
        "request_project_verification_cancel",
        lambda *_args, **_kwargs: (None, False),
    )
    monkeypatch.setattr(
        verification.verification_state,
        "active_verification_run_lifecycle",
        lambda *_args, **_kwargs: None,
    )

    def retire():
        try:
            verification.cancel_project_verifications_and_wait(
                project_id,
                timeout_seconds = 2,
            )
            wait_finished.set()
            mutation_started.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            failures.append(exc)

    verification.begin_project_deletion(project_id)
    worker = threading.Thread(target = retire)
    worker.start()
    try:
        assert wait_entered.wait(timeout = 1)
        assert not wait_finished.wait(timeout = 0.05)
        assert not mutation_started.is_set()
        supervisor._release_project_execution_fence(prior_tree_fence)
        prior_tree_fence = None
        assert wait_finished.wait(timeout = 1)
        deletion = verification._DELETING_PROJECTS[project_id]
        assert deletion.execution_fence_fd is not None
        assert observed_fence_ids == [fence_id]
    finally:
        if prior_tree_fence is not None:
            supervisor._release_project_execution_fence(prior_tree_fence)
        verification.finish_project_deletion(project_id)
        worker.join(timeout = 2)

    assert not worker.is_alive()
    assert failures == []
    assert verification._DELETING_PROJECTS == {}


def test_archived_project_is_rejected_at_begin_and_pre_spawn_transactions(tmp_path):
    begin_project_id = "archived-begin-project"
    begin_workspace = _workspace(tmp_path, begin_project_id)
    _create_stored_project(begin_project_id)
    begin_profile = verification_state.set_verification_config(
        begin_project_id,
        [_check("tests", "run-tests")],
        workspace_identity = (
            int(begin_workspace.device_id),
            int(begin_workspace.file_id),
        ),
        workspace_revision = begin_workspace.revision,
        expected_revision = 0,
    )
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE chat_projects SET archived = 1 WHERE id = ?",
            (begin_project_id,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(verification_state.VerificationConflictError, match = "Archived"):
        verification_state.begin_verification_run(
            begin_project_id,
            owner_id = verification.PROCESS_OWNER_ID,
            config_revision = begin_profile["revision"],
            config_hash = begin_profile["configHash"],
            checks = begin_profile["checks"],
            workspace_identity = (
                int(begin_workspace.device_id),
                int(begin_workspace.file_id),
            ),
            workspace_revision = begin_workspace.revision,
        )

    spawn_project_id = "archived-spawn-project"
    spawn_workspace = _workspace(tmp_path, spawn_project_id)
    _create_stored_project(spawn_project_id)
    spawn_profile = verification_state.set_verification_config(
        spawn_project_id,
        [_check("tests", "run-tests")],
        workspace_identity = (
            int(spawn_workspace.device_id),
            int(spawn_workspace.file_id),
        ),
        workspace_revision = spawn_workspace.revision,
        expected_revision = 0,
    )
    run = verification_state.begin_verification_run(
        spawn_project_id,
        owner_id = verification.PROCESS_OWNER_ID,
        config_revision = spawn_profile["revision"],
        config_hash = spawn_profile["configHash"],
        checks = spawn_profile["checks"],
        workspace_identity = (
            int(spawn_workspace.device_id),
            int(spawn_workspace.file_id),
        ),
        workspace_revision = spawn_workspace.revision,
    )
    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE chat_projects SET archived = 1 WHERE id = ?",
            (spawn_project_id,),
        )
        connection.commit()
    finally:
        connection.close()

    assert not verification_state.verification_run_may_spawn(
        spawn_project_id,
        run["id"],
        verification.PROCESS_OWNER_ID,
        revision = spawn_profile["revision"],
        config_hash = spawn_profile["configHash"],
        workspace_identity = (
            int(spawn_workspace.device_id),
            int(spawn_workspace.file_id),
        ),
        workspace_revision = spawn_workspace.revision,
    )


def test_heartbeat_storage_loss_is_persisted_as_failure_not_user_cancellation(monkeypatch):
    active = verification._ActiveVerification("heartbeat-project", run_id = "run-heartbeat")
    monkeypatch.setattr(verification_state, "HEARTBEAT_INTERVAL_SECONDS", 0)

    def fail_heartbeat(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(verification_state, "heartbeat_verification_run", fail_heartbeat)
    verification._heartbeat_run(active)
    saved = []
    monkeypatch.setattr(
        verification_state,
        "complete_verification_run",
        lambda *args, **kwargs: saved.append(kwargs),
    )
    verification._execute_run(active, {"checks": []}, None, release_active = False)
    assert saved[0]["terminal_status"] == "failed"
    assert "heartbeat" in saved[0]["error"]

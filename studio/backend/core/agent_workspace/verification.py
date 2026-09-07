# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Project verification with bounded logs, cancellation and freshness evidence."""

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    initialize_parent_lifetime,
    spawn_on_lifetime_thread,
)

from .common import (
    AgentWorkspaceError,
    _terminate_bounded_process,
    agent_child_env,
    now_ms,
    project_workspace,
    workspace_fingerprint,
    workspace_fingerprint_complete,
)
from .execution import ProjectExecutionBoundary, ProjectExecutionUnavailable
from .state import (
    begin_verification_run,
    finish_verification_run,
    get_verification_config,
    get_verification_run,
    latest_primary_verification_run,
    list_verification_runs,
)
from .worktrees import owned_worktree_path


DEFAULT_LOG_LIMIT = 256 * 1024
MAX_LOG_LIMIT = 2 * 1024 * 1024
MAX_RUN_LOG_BYTES = 4 * 1024 * 1024
MAX_CHECKS = 32
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_CANCEL: dict[str, threading.Event] = {}
_ACTIVE_PROCESSES: dict[str, tuple[subprocess.Popen, Optional[int]]] = {}
_ACTIVE_PROJECT_RUNS: dict[str, dict[str, tuple[threading.Event, threading.Event]]] = {}
_DELETING_PROJECTS: set[str] = set()
GOAL_COMPLETION_VERIFICATION_DETAIL = (
    "Goal completion requires a fresh passing verification run from the main "
    "project workspace. Run every required check after the latest source or "
    "verification-setting change, then try again."
)


def begin_project_deletion(project_id: str) -> None:
    """Fence new foreground and background verification runs for one project."""
    with _ACTIVE_LOCK:
        if project_id in _DELETING_PROJECTS:
            raise AgentWorkspaceError("Project verification deletion is already in progress.")
        _DELETING_PROJECTS.add(project_id)


def finish_project_deletion(project_id: str) -> None:
    with _ACTIVE_LOCK:
        _DELETING_PROJECTS.discard(project_id)


def _register_project_run(
    project_id: str, cancel_event: threading.Event
) -> tuple[str, threading.Event]:
    invocation_id = str(uuid.uuid4())
    completed = threading.Event()
    with _ACTIVE_LOCK:
        if project_id in _DELETING_PROJECTS:
            raise AgentWorkspaceError("Project deletion is in progress.")
        _ACTIVE_PROJECT_RUNS.setdefault(project_id, {})[invocation_id] = (
            cancel_event,
            completed,
        )
    return invocation_id, completed


def _finish_project_run(project_id: str, invocation_id: str, completed: threading.Event) -> None:
    with _ACTIVE_LOCK:
        project_runs = _ACTIVE_PROJECT_RUNS.get(project_id)
        if project_runs is not None:
            project_runs.pop(invocation_id, None)
            if not project_runs:
                _ACTIVE_PROJECT_RUNS.pop(project_id, None)
        completed.set()


def cancel_project_verifications_and_wait(project_id: str, *, timeout_seconds: float = 30) -> None:
    """Cancel and join every registered verification before project deletion."""
    if timeout_seconds <= 0:
        raise AgentWorkspaceError("Project verification cancellation timed out.")
    deadline = time.monotonic() + timeout_seconds
    with _ACTIVE_LOCK:
        runs = list(_ACTIVE_PROJECT_RUNS.get(project_id, {}).values())
        for cancel_event, _completed in runs:
            cancel_event.set()
        cancel_events = {cancel_event for cancel_event, _completed in runs}
        active_processes = [
            active
            for run_id, active in _ACTIVE_PROCESSES.items()
            if _ACTIVE_CANCEL.get(run_id) in cancel_events
        ]
    for process, group_id in active_processes:
        _terminate_bounded_process(process, group_id)
    for _cancel_event, completed in runs:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not completed.wait(timeout = remaining):
            raise AgentWorkspaceError("Timed out while stopping project verification runs.")
    with _ACTIVE_LOCK:
        if _ACTIVE_PROJECT_RUNS.get(project_id):
            raise AgentWorkspaceError(
                "Project still has active verification runs. Wait for them to stop and retry."
            )


def _shell_argv(command: str) -> list[str]:
    if os.name == "nt":
        return ["cmd.exe", "/d", "/s", "/c", command]
    return ["/bin/sh", "-c", command]


def _capture_output(pipe, output: bytearray, counters: dict, output_limit: int) -> None:
    try:
        while True:
            chunk = pipe.read(64 * 1024)
            if not chunk:
                break
            counters["total"] += len(chunk)
            remaining = output_limit - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def execute_check(
    check: dict,
    *,
    root: Path,
    cancel_event: threading.Event,
    run_id: str,
    expected_root_identity: Optional[tuple[int, int]] = None,
) -> dict:
    command = str(check.get("command") or "").strip()
    if not command:
        raise AgentWorkspaceError("Verification commands cannot be empty.")
    timeout_seconds = max(1, min(int(check.get("timeoutSeconds") or 300), 3600))
    output_limit = max(
        1024, min(int(check.get("logLimitBytes") or DEFAULT_LOG_LIMIT), MAX_LOG_LIMIT)
    )
    started = now_ms()
    try:
        boundary = ProjectExecutionBoundary.open(root, expected_root_identity)
    except ProjectExecutionUnavailable as exc:
        raise AgentWorkspaceError(str(exc)) from exc
    try:
        if not boundary.acquire_execution_slot(cancel_event):
            completed = now_ms()
            boundary.close()
            return {
                "name": str(check.get("name") or check.get("kind") or "check")[:120],
                "kind": str(check.get("kind") or "custom"),
                "command": command,
                "required": bool(check.get("required", True)),
                "status": "cancelled",
                "exitCode": None,
                "output": "",
                "outputBytes": 0,
                "outputTruncated": False,
                "timeoutSeconds": timeout_seconds,
                "startedAt": started,
                "completedAt": completed,
                "durationMs": completed - started,
            }
    except ProjectExecutionUnavailable as exc:
        boundary.close()
        raise AgentWorkspaceError(str(exc)) from exc
    popen_options = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "env": boundary.apply_environment(
            agent_child_env(
                root,
                {"UNSLOTH_STUDIO_PROJECT_ID": str(check.get("projectId") or "")},
            )
        ),
    }
    popen_options["start_new_session"] = True
    lifetime_options = child_popen_kwargs()
    popen_options.update(boundary.popen_kwargs(lifetime_options.pop("preexec_fn", None)))
    popen_options.update(lifetime_options)

    try:
        argv = boundary.wrap_argv(_shell_argv(command))
        initialize_parent_lifetime()
        process = spawn_on_lifetime_thread(lambda: subprocess.Popen(argv, **popen_options))
    except ProjectExecutionUnavailable as exc:
        boundary.close()
        raise AgentWorkspaceError(str(exc)) from exc
    except Exception:
        boundary.close()
        raise
    adopt_pid(process.pid)
    group_id = process.pid if os.name != "nt" else None
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES[run_id] = (process, group_id)
    output = bytearray()
    counters = {"total": 0}
    reader = threading.Thread(
        target = _capture_output,
        args = (process.stdout, output, counters, output_limit),
        daemon = True,
    )
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    result_status = "failed"
    try:
        while process.poll() is None:
            if cancel_event.wait(timeout = 0.05):
                result_status = "cancelled"
                _terminate_bounded_process(process, group_id)
                break
            if time.monotonic() >= deadline:
                result_status = "timed_out"
                _terminate_bounded_process(process, group_id)
                break
        try:
            exit_code = process.wait(timeout = 3)
        except subprocess.TimeoutExpired:
            _terminate_bounded_process(process, group_id)
            exit_code = process.poll()
        if result_status not in {"cancelled", "timed_out"}:
            result_status = "passed" if exit_code == 0 else "failed"
    finally:
        reader.join(timeout = 0.2)
        if reader.is_alive():
            _terminate_bounded_process(process, group_id)
            reader.join(timeout = 3)
        forget_pid(process.pid)
        with _ACTIVE_LOCK:
            active = _ACTIVE_PROCESSES.get(run_id)
            if active is not None and active[0] is process:
                _ACTIVE_PROCESSES.pop(run_id, None)
        boundary.close()

    completed = now_ms()
    return {
        "name": str(check.get("name") or check.get("kind") or "check")[:120],
        "kind": str(check.get("kind") or "custom"),
        "command": command,
        "required": bool(check.get("required", True)),
        "status": result_status,
        "exitCode": exit_code,
        "output": bytes(output).decode("utf-8", errors = "replace"),
        "outputBytes": counters["total"],
        "outputTruncated": counters["total"] > len(output),
        "timeoutSeconds": timeout_seconds,
        "startedAt": started,
        "completedAt": completed,
        "durationMs": completed - started,
    }


def run_project_verification(
    project_id: str,
    selected_names: Optional[list[str]] = None,
    *,
    cancel_event: Optional[threading.Event] = None,
    on_run_started = None,
    worktree_id: Optional[str] = None,
    config_revision: Optional[int] = None,
) -> dict:
    cancel_event = cancel_event or threading.Event()
    invocation_id, completed = _register_project_run(project_id, cancel_event)
    run_id: Optional[str] = None
    try:
        workspace = project_workspace(project_id)
        execution_root = (
            owned_worktree_path(project_id, worktree_id) if worktree_id else workspace.root
        )
        expected_root_identity = (
            (workspace.device_id, workspace.file_id)
            if not worktree_id and workspace.device_id is not None and workspace.file_id is not None
            else None
        )
        config = get_verification_config(project_id)
        if config_revision is not None and config["revision"] != config_revision:
            raise AgentWorkspaceError(
                "Verification settings changed after this run was prepared. Refresh and retry."
            )
        checks = list(config["checks"])
        if selected_names is not None:
            normalized_names = [str(name).strip() for name in selected_names]
            if any(not name for name in normalized_names):
                raise AgentWorkspaceError("Verification check names cannot be blank.")
            selected = set(normalized_names)
            if len(selected) != len(normalized_names):
                raise AgentWorkspaceError("Selected verification check names must be unique.")
            configured = {str(check.get("name") or "") for check in checks}
            missing = selected - configured
            if missing:
                raise AgentWorkspaceError(
                    "One or more selected verification checks are not configured. Refresh and retry."
                )
            checks = [check for check in checks if check.get("name") in selected]
        if not checks:
            raise AgentWorkspaceError("No verification checks are configured.")
        if len(checks) > MAX_CHECKS:
            raise AgentWorkspaceError(f"At most {MAX_CHECKS} verification checks can run at once.")

        source = workspace_fingerprint(execution_root)
        record = begin_verification_run(
            project_id,
            source,
            worktree_id,
            config_revision = config["revision"],
        )
        run_id = record["id"]
        with _ACTIVE_LOCK:
            _ACTIVE_CANCEL[run_id] = cancel_event
        if on_run_started is not None:
            on_run_started(run_id)
        results = []
        remaining_log_bytes = MAX_RUN_LOG_BYTES
        try:
            for check in checks:
                if cancel_event.is_set():
                    break
                result = execute_check(
                    {**check, "projectId": project_id},
                    root = execution_root,
                    cancel_event = cancel_event,
                    run_id = run_id,
                    expected_root_identity = expected_root_identity,
                )
                encoded = result["output"].encode("utf-8", errors = "replace")
                if len(encoded) > remaining_log_bytes:
                    result["output"] = encoded[:remaining_log_bytes].decode(
                        "utf-8", errors = "replace"
                    )
                    result["outputTruncated"] = True
                remaining_log_bytes = max(0, remaining_log_bytes - len(encoded))
                results.append(result)
                if results[-1]["status"] == "cancelled":
                    break
            if cancel_event.is_set() or any(item["status"] == "cancelled" for item in results):
                status = "cancelled"
            elif any(item["required"] and item["status"] != "passed" for item in results):
                status = "failed"
            else:
                status = "passed"
            final = workspace_fingerprint(execution_root)
            record = finish_verification_run(run_id, status, final, results)
            record["changedDuringRun"] = source != final
            record["evidenceComplete"] = workspace_fingerprint_complete(
                source
            ) and workspace_fingerprint_complete(final)
            record["unverifiable"] = not record["evidenceComplete"]
            record["stale"] = record["changedDuringRun"] or record["unverifiable"]
            record["worktreeId"] = worktree_id
            return record
        except Exception:
            final = workspace_fingerprint(execution_root)
            finish_verification_run(run_id, "failed", final, results)
            raise
    finally:
        with _ACTIVE_LOCK:
            if run_id is not None:
                _ACTIVE_CANCEL.pop(run_id, None)
                _ACTIVE_PROCESSES.pop(run_id, None)
        _finish_project_run(project_id, invocation_id, completed)


def cancel_verification(run_id: str) -> bool:
    with _ACTIVE_LOCK:
        event = _ACTIVE_CANCEL.get(run_id)
        active = _ACTIVE_PROCESSES.get(run_id)
        if event is None:
            return False
        event.set()
    if active is not None:
        _terminate_bounded_process(active[0], active[1])
    return True


def verification_run_with_freshness(run_id: str) -> Optional[dict]:
    run = get_verification_run(run_id)
    if run is None:
        return None
    try:
        current_root = (
            owned_worktree_path(run["projectId"], run["worktreeId"])
            if run.get("worktreeId")
            else project_workspace(run["projectId"]).root
        )
        current = workspace_fingerprint(current_root)
        final = run["finalFingerprint"] or run["sourceFingerprint"]
        run["changedDuringRun"] = run["sourceFingerprint"] != final
        run["evidenceComplete"] = all(
            workspace_fingerprint_complete(value)
            for value in (run["sourceFingerprint"], final, current)
        )
        run["unverifiable"] = not run["evidenceComplete"]
        run["stale"] = run["changedDuringRun"] or current != final or run["unverifiable"]
        run["currentFingerprint"] = current
    except AgentWorkspaceError:
        run["changedDuringRun"] = (
            run["finalFingerprint"] is not None
            and run["sourceFingerprint"] != run["finalFingerprint"]
        )
        run["stale"] = True
        run["evidenceComplete"] = False
        run["unverifiable"] = True
        run["currentFingerprint"] = None
    return run


def verification_runs_with_freshness(project_id: str, limit: int = 20) -> list[dict]:
    return [
        fresh
        for run in list_verification_runs(project_id, limit)
        if (fresh := verification_run_with_freshness(run["id"])) is not None
    ]


def require_goal_completion_verification(project_id: str) -> int:
    """Enforce the optional fresh-evidence policy for project goal completion."""
    config = get_verification_config(project_id)
    if not config["requireForGoalCompletion"]:
        return int(config["revision"])

    checks = list(config["checks"])
    latest = latest_primary_verification_run(project_id)
    run = verification_run_with_freshness(latest["id"]) if latest is not None else None
    required_names = [
        str(check.get("name") or "").strip()
        for check in checks
        if bool(check.get("required", True))
    ]
    passed_names = {
        str(result.get("name") or "").strip()
        for result in (run.get("results", []) if run else [])
        if result.get("status") == "passed"
    }
    evidence_is_current = bool(
        checks
        and run
        and run.get("worktreeId") is None
        and run.get("status") == "passed"
        and run.get("completedAt") is not None
        and run.get("finalFingerprint") is not None
        and run.get("evidenceComplete") is True
        and run.get("stale") is False
        and run.get("configRevision") == config["revision"]
        and all(name and name in passed_names for name in required_names)
    )
    if not evidence_is_current:
        raise AgentWorkspaceError(GOAL_COMPLETION_VERIFICATION_DETAIL)
    return int(config["revision"])

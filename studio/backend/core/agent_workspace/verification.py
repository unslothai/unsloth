# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable project verification over the supervised command boundary."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from . import verification_context as common, verification_process as supervisor, verification_state
from .verification_context import AgentWorkspaceError, ProjectWorkspace


MAX_ACTIVE_VERIFICATIONS = 4
PREPARE_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 30.0
PROGRESS_FLUSH_SECONDS = 0.2
PROCESS_OWNER_ID = secrets.token_hex(16)

_CAPABILITY_SEAL = object()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_BY_PROJECT: dict[str, "_ActiveVerification"] = {}
_ACTIVE_BY_RUN: dict[str, "_ActiveVerification"] = {}
_DELETING_PROJECTS: dict[str, "_ActiveDeletion"] = {}


@dataclass(eq = False)
class _ActiveVerification:
    project_id: str
    cancel_event: threading.Event = field(default_factory = threading.Event)
    completed: threading.Event = field(default_factory = threading.Event)
    run_id: Optional[str] = None
    run_started_at: Optional[int] = None
    run_started_monotonic: Optional[float] = None
    thread: Optional[threading.Thread] = None
    heartbeat_thread: Optional[threading.Thread] = None
    persistence_failure: Optional[str] = None


@dataclass(eq = False)
class _ActiveDeletion:
    project_id: str
    fence_id: str = field(default_factory = lambda: secrets.token_hex(16))
    revision: Optional[int] = None
    stopped: threading.Event = field(default_factory = threading.Event)
    fence_lost: threading.Event = field(default_factory = threading.Event)
    heartbeat_thread: Optional[threading.Thread] = None
    execution_fence_fd: Optional[int] = None


class _ProgressPublisher:
    """Publish coalesced progress without blocking the process supervisor."""

    def __init__(self, active: _ActiveVerification, run_id: str) -> None:
        self._active = active
        self._run_id = run_id
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._latest: Optional[list[dict]] = None
        self._thread = threading.Thread(
            target = self._run,
            name = f"project-verification-progress-{run_id[:8]}",
            daemon = True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, results: list[dict]) -> None:
        if self._stopped.is_set():
            return
        with self._lock:
            self._latest = results
        self._wake.set()

    def stop(self) -> None:
        self._stopped.set()
        with self._lock:
            self._latest = None
        self._wake.set()

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stopped.is_set():
                return
            with self._lock:
                latest = self._latest
                self._latest = None
            if latest is None:
                continue
            try:
                verification_state.update_verification_run_progress(
                    self._active.project_id,
                    self._run_id,
                    latest,
                    owner_id = PROCESS_OWNER_ID,
                )
            except verification_state.VerificationConflictError:
                continue
            except BaseException:  # noqa: BLE001 - persistence loss cancels execution
                self._active.persistence_failure = (
                    "Verification progress could not be saved; execution was stopped."
                )
                self._active.cancel_event.set()
                return


@dataclass(frozen = True)
class _VerificationProcessCapability:
    project_id: str
    run_id: str
    owner_id: str
    config_revision: int
    config_hash: str
    workspace_identity: tuple[int, int]
    workspace_revision: int
    argv: tuple[str, ...]
    _seal: object = field(repr = False, compare = False)


def _workspace_identity(workspace: ProjectWorkspace) -> tuple[int, int]:
    if workspace.device_id is None or workspace.file_id is None:
        raise AgentWorkspaceError("Project workspace identity is unavailable.")
    return int(workspace.device_id), int(workspace.file_id)


def _shell_argv(command: str) -> tuple[str, ...]:
    if os.name == "nt":
        return ("cmd.exe", "/d", "/s", "/c", command)
    return ("/bin/sh", "-c", command)


def _safe_error(error: BaseException) -> str:
    if isinstance(error, AgentWorkspaceError):
        return str(error)
    return "Verification failed before every check could complete."


def _bounded_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors = "ignore")


def _check_output_limit(check: dict, results: list[dict], future_checks: int) -> int:
    retained = sum(len(result["output"].encode("utf-8")) for result in results)
    decoration = verification_state.MAX_TRUNCATION_DECORATION_BYTES
    future_reserve = future_checks * (decoration + 1)
    available = verification_state.MAX_RUN_RESULT_BYTES - retained - future_reserve
    capture_limit = available - decoration
    if capture_limit < 1:
        raise AgentWorkspaceError("Verification output evidence budget is exhausted.")
    return min(int(check["logLimitBytes"]), capture_limit)


def _reserve_project(project_id: str) -> _ActiveVerification:
    active = _ActiveVerification(project_id = project_id)
    with _ACTIVE_LOCK:
        if project_id in _DELETING_PROJECTS:
            raise AgentWorkspaceError("Project deletion is in progress.")
        if project_id in _ACTIVE_BY_PROJECT:
            raise AgentWorkspaceError("Project verification is already running.")
        if len(_ACTIVE_BY_PROJECT) >= MAX_ACTIVE_VERIFICATIONS:
            raise AgentWorkspaceError("Project verification capacity is full. Retry shortly.")
        _ACTIVE_BY_PROJECT[project_id] = active
    return active


def _publish_run(active: _ActiveVerification, run: dict) -> None:
    run_id = str(run["id"])
    run_started_at = int(run["startedAt"])
    run_started_monotonic = time.monotonic()
    with _ACTIVE_LOCK:
        if _ACTIVE_BY_PROJECT.get(active.project_id) is not active:
            raise AgentWorkspaceError("Project verification admission was lost.")
        active.run_id = run_id
        active.run_started_at = run_started_at
        active.run_started_monotonic = run_started_monotonic
        _ACTIVE_BY_RUN[run_id] = active


def _release_active(active: _ActiveVerification) -> None:
    with _ACTIVE_LOCK:
        if _ACTIVE_BY_PROJECT.get(active.project_id) is active:
            _ACTIVE_BY_PROJECT.pop(active.project_id, None)
        if active.run_id is not None and _ACTIVE_BY_RUN.get(active.run_id) is active:
            _ACTIVE_BY_RUN.pop(active.run_id, None)
    active.completed.set()


def _heartbeat_run(active: _ActiveVerification) -> None:
    interval = verification_state.HEARTBEAT_INTERVAL_SECONDS
    while not active.completed.wait(interval):
        run_id = active.run_id
        if run_id is None:
            continue
        try:
            cancel_requested = verification_state.heartbeat_verification_run(
                active.project_id,
                run_id,
                PROCESS_OWNER_ID,
            )
        except BaseException:  # noqa: BLE001 - any durable heartbeat loss cancels execution
            active.persistence_failure = (
                "Verification heartbeat could not be saved; execution was stopped."
            )
            active.cancel_event.set()
            return
        if cancel_requested:
            active.cancel_event.set()


def _heartbeat_deletion(active: _ActiveDeletion) -> None:
    interval = verification_state.DELETION_FENCE_HEARTBEAT_INTERVAL_SECONDS
    while not active.stopped.wait(interval):
        revision = active.revision
        if revision is None:
            active.fence_lost.set()
            return
        try:
            verification_state.heartbeat_verification_project_deletion(
                active.project_id,
                active.fence_id,
                revision,
            )
        except BaseException:  # noqa: BLE001 - any heartbeat loss invalidates retirement
            active.fence_lost.set()
            return


def _execution_fence_id(project_id: str) -> str:
    return "project:" + project_id


def _acquire_deletion_execution_fence(active: _ActiveDeletion, deadline: float) -> None:
    if active.execution_fence_fd is not None or os.name != "posix":
        return
    try:
        descriptor = supervisor._acquire_project_execution_fence(
            _execution_fence_id(active.project_id),
            active.stopped,
            deadline,
        )
    except InterruptedError as exc:
        raise AgentWorkspaceError("Project verification retirement was cancelled.") from exc
    except TimeoutError as exc:
        raise AgentWorkspaceError(
            "Timed out while proving the prior project verification process tree stopped."
        ) from exc
    except AgentWorkspaceError:
        raise
    except Exception as exc:
        raise AgentWorkspaceError(
            "Project verification process-tree fencing is unavailable."
        ) from exc
    with _ACTIVE_LOCK:
        if (
            _DELETING_PROJECTS.get(active.project_id) is not active
            or active.stopped.is_set()
            or active.fence_lost.is_set()
        ):
            supervisor._release_project_execution_fence(descriptor)
            raise AgentWorkspaceError("Project verification retirement authority was lost.")
        active.execution_fence_fd = descriptor


def _profile_matches_workspace(profile: dict, workspace: ProjectWorkspace) -> bool:
    return (
        profile.get("workspaceDeviceId") == _workspace_identity(workspace)[0]
        and profile.get("workspaceFileId") == _workspace_identity(workspace)[1]
        and profile.get("workspaceRevision") == int(workspace.revision)
    )


def _capability(profile: dict, workspace: ProjectWorkspace, check: dict, run_id: str):
    return _VerificationProcessCapability(
        project_id = workspace.project_id,
        run_id = run_id,
        owner_id = PROCESS_OWNER_ID,
        config_revision = int(profile["revision"]),
        config_hash = str(profile["configHash"]),
        workspace_identity = _workspace_identity(workspace),
        workspace_revision = int(workspace.revision),
        argv = _shell_argv(check["command"]),
        _seal = _CAPABILITY_SEAL,
    )


def _revalidate_verification_capability(
    capability: _VerificationProcessCapability, workspace: ProjectWorkspace
) -> None:
    """Revalidate persisted verification authority immediately before Popen."""
    if (
        not isinstance(capability, _VerificationProcessCapability)
        or capability._seal is not _CAPABILITY_SEAL
        or workspace.project_id != capability.project_id
        or _workspace_identity(workspace) != capability.workspace_identity
        or int(workspace.revision) != capability.workspace_revision
    ):
        raise AgentWorkspaceError("Project verification workspace changed before execution.")
    if not verification_state.verification_run_may_spawn(
        capability.project_id,
        capability.run_id,
        capability.owner_id,
        revision = capability.config_revision,
        config_hash = capability.config_hash,
        workspace_identity = capability.workspace_identity,
        workspace_revision = capability.workspace_revision,
    ):
        raise AgentWorkspaceError("Project verification settings changed before execution.")


def _running_result(
    check: dict,
    started_at: int,
    output: str = "",
) -> dict:
    return {
        "name": check["name"],
        "kind": check["kind"],
        "command": check["command"],
        "required": check["required"],
        "status": "running",
        "exitCode": None,
        "output": output,
        "outputBytes": len(output.encode("utf-8")),
        "outputTruncated": False,
        "timeoutSeconds": check["timeoutSeconds"],
        "startedAt": started_at,
        "completedAt": None,
        "durationMs": None,
    }


def _completion_timing(
    started_at: int, *, started_monotonic: float | None = None
) -> tuple[int, int]:
    if started_monotonic is None:
        completed_at = max(started_at, int(time.time() * 1000))
        return completed_at, completed_at - started_at
    duration_ms = max(0, int((time.monotonic() - started_monotonic) * 1000))
    return started_at + duration_ms, duration_ms


def _terminal_result(
    check: dict,
    process_result,
    started_at: int,
    *,
    started_monotonic: float | None = None,
) -> dict:
    completed_at, duration_ms = _completion_timing(
        started_at,
        started_monotonic = started_monotonic,
    )
    return {
        "name": check["name"],
        "kind": check["kind"],
        "command": check["command"],
        "required": check["required"],
        "status": process_result.status,
        "exitCode": process_result.exit_code,
        "output": process_result.output,
        "outputBytes": process_result.output_bytes,
        "outputTruncated": process_result.output_truncated,
        "timeoutSeconds": check["timeoutSeconds"],
        "startedAt": started_at,
        "completedAt": completed_at,
        "durationMs": duration_ms,
    }


def _error_result(
    check: dict,
    started_at: int,
    status: str,
    error: BaseException,
    *,
    started_monotonic: float | None = None,
) -> dict:
    completed_at, duration_ms = _completion_timing(
        started_at,
        started_monotonic = started_monotonic,
    )
    return {
        "name": check["name"],
        "kind": check["kind"],
        "command": check["command"],
        "required": check["required"],
        "status": status,
        "exitCode": None,
        "output": "",
        "outputBytes": 0,
        "outputTruncated": False,
        "timeoutSeconds": check["timeoutSeconds"],
        "startedAt": started_at,
        "completedAt": completed_at,
        "durationMs": duration_ms,
        "error": _safe_error(error),
    }


def _check_start_timing(
    active: _ActiveVerification, completed_results: list[dict]
) -> tuple[int, float]:
    monotonic_now = time.monotonic()
    wall_now = int(time.time() * 1000)
    evidence_floor = active.run_started_at or 0
    if completed_results:
        previous = completed_results[-1]
        evidence_floor = max(
            evidence_floor,
            int(previous["completedAt"] or previous["startedAt"]),
        )
    projected = evidence_floor
    if active.run_started_at is not None and active.run_started_monotonic is not None:
        elapsed_ms = max(
            0,
            int((monotonic_now - active.run_started_monotonic) * 1000),
        )
        projected = active.run_started_at + elapsed_ms
    return max(wall_now, evidence_floor, projected), monotonic_now


def _execute_run(
    active: _ActiveVerification,
    profile: dict,
    workspace: ProjectWorkspace,
    *,
    release_active: bool = True,
) -> None:
    run_id = active.run_id
    if run_id is None:
        if release_active:
            _release_active(active)
        return
    results: list[dict] = []
    terminal_status = None
    terminal_error = None
    publisher: Optional[_ProgressPublisher] = None
    try:
        publisher = _ProgressPublisher(active, run_id)
        publisher.start()
        for check_index, check in enumerate(profile["checks"]):
            if active.cancel_event.is_set():
                terminal_status = "cancelled"
                break
            output_limit = _check_output_limit(
                check,
                results,
                len(profile["checks"]) - check_index - 1,
            )
            started_at, started_monotonic = _check_start_timing(active, results)
            streamed: list[str] = []
            last_flush = [0.0]

            def publish_output(chunk: str) -> None:
                streamed.append(chunk)
                now = time.monotonic()
                if now - last_flush[0] < PROGRESS_FLUSH_SECONDS:
                    return
                last_flush[0] = now
                visible_output = _bounded_utf8("".join(streamed), output_limit)
                publisher.submit([*results, _running_result(check, started_at, visible_output)])

            verification_state.update_verification_run_progress(
                active.project_id,
                run_id,
                [*results, _running_result(check, started_at)],
                owner_id = PROCESS_OWNER_ID,
            )
            try:
                process_result = supervisor._run_project_verification_process(
                    _capability(profile, workspace, check, run_id),
                    timeout_seconds = check["timeoutSeconds"],
                    output_limit_bytes = output_limit,
                    cancel_event = active.cancel_event,
                    output_callback = publish_output,
                )
                result = _terminal_result(
                    check,
                    process_result,
                    started_at,
                    started_monotonic = started_monotonic,
                )
            except AgentWorkspaceError as exc:
                result = _error_result(
                    check,
                    started_at,
                    "blocked",
                    exc,
                    started_monotonic = started_monotonic,
                )
                terminal_status = "blocked"
                terminal_error = _safe_error(exc)
            except BaseException as exc:  # noqa: BLE001 - persist safe terminal evidence
                result = _error_result(
                    check,
                    started_at,
                    "failed",
                    exc,
                    started_monotonic = started_monotonic,
                )
                terminal_status = "failed"
                terminal_error = _safe_error(exc)
            results.append(result)
            verification_state.update_verification_run_progress(
                active.project_id,
                run_id,
                results,
                owner_id = PROCESS_OWNER_ID,
            )
            if result["status"] in {"cancelled", "blocked"}:
                terminal_status = result["status"]
                break
            if terminal_status is not None:
                break
        if active.persistence_failure is not None:
            terminal_status = "failed"
            terminal_error = active.persistence_failure
        verification_state.complete_verification_run(
            active.project_id,
            run_id,
            results,
            owner_id = PROCESS_OWNER_ID,
            terminal_status = terminal_status,
            error = terminal_error,
        )
    except BaseException as exc:  # noqa: BLE001 - startup recovery owns a failed DB write
        try:
            verification_state.complete_verification_run(
                active.project_id,
                run_id,
                results,
                owner_id = PROCESS_OWNER_ID,
                terminal_status = "failed",
                error = _safe_error(exc),
            )
        except BaseException:
            pass
    finally:
        if publisher is not None:
            publisher.stop()
        if release_active:
            _release_active(active)


def _workspace_matches(left: ProjectWorkspace, right: ProjectWorkspace) -> bool:
    return (
        left.project_id == right.project_id
        and left.root == right.root
        and left.kind == right.kind
        and _workspace_identity(left) == _workspace_identity(right)
        and int(left.revision) == int(right.revision)
    )


def _execute_run_with_workspace_lease(
    active: _ActiveVerification,
    profile: dict,
    prepared_workspace: ProjectWorkspace,
    lease_ready: threading.Event,
    startup_errors: list[BaseException],
    deadline: float,
) -> None:
    try:
        with common.project_workspace_access(
            active.project_id,
            cancel_event = active.cancel_event,
            deadline = deadline,
        ) as current_workspace:
            if not _workspace_matches(current_workspace, prepared_workspace):
                raise AgentWorkspaceError(
                    "Project workspace changed before verification could hold its lease."
                )
            if not _profile_matches_workspace(profile, current_workspace):
                raise AgentWorkspaceError("Project verification settings changed before execution.")
            lease_ready.set()
            _execute_run(
                active,
                profile,
                current_workspace,
                release_active = False,
            )
    except BaseException as exc:  # noqa: BLE001 - worker owns durable startup failure
        if not lease_ready.is_set():
            startup_errors.append(exc)
        run_id = active.run_id
        if run_id is not None:
            try:
                verification_state.complete_verification_run(
                    active.project_id,
                    run_id,
                    [],
                    owner_id = PROCESS_OWNER_ID,
                    terminal_status = (
                        "blocked" if isinstance(exc, AgentWorkspaceError) else "failed"
                    ),
                    error = _safe_error(exc),
                )
            except BaseException:
                pass
    finally:
        lease_ready.set()
        _release_active(active)


def start_project_verification(
    project_id: str, *, config_revision: int, workspace_revision: int
) -> dict:
    """Pin and start one saved profile, returning its durable running record."""
    if (
        isinstance(config_revision, bool)
        or not isinstance(config_revision, int)
        or config_revision < 1
        or isinstance(workspace_revision, bool)
        or not isinstance(workspace_revision, int)
        or workspace_revision < 0
    ):
        raise AgentWorkspaceError("Project verification revision is invalid.")
    active = _reserve_project(project_id)
    deadline = time.monotonic() + PREPARE_TIMEOUT_SECONDS
    run = None
    try:
        with common.project_workspace_access(
            project_id,
            cancel_event = active.cancel_event,
            deadline = deadline,
        ) as workspace:
            if int(workspace.revision) != workspace_revision:
                raise AgentWorkspaceError(
                    "Project workspace changed. Refresh verification settings and retry."
                )
            profile = verification_state.get_verification_config(project_id)
            if int(profile["revision"]) != config_revision:
                raise AgentWorkspaceError(
                    "Project verification settings changed. Refresh and retry."
                )
            if not _profile_matches_workspace(profile, workspace):
                raise AgentWorkspaceError(
                    "Saved verification settings belong to a different project workspace."
                )
            if not profile["checks"]:
                raise AgentWorkspaceError("No verification checks are configured.")
            run = verification_state.begin_verification_run(
                project_id,
                config_revision = profile["revision"],
                config_hash = profile["configHash"],
                checks = profile["checks"],
                workspace_identity = _workspace_identity(workspace),
                workspace_revision = int(workspace.revision),
                owner_id = PROCESS_OWNER_ID,
            )
            _publish_run(active, run)
            if active.cancel_event.is_set():
                verification_state.request_verification_cancel(project_id, run["id"])
            heartbeat_thread = threading.Thread(
                target = _heartbeat_run,
                args = (active,),
                name = f"project-verification-heartbeat-{run['id'][:8]}",
                daemon = True,
            )
            lease_ready = threading.Event()
            startup_errors: list[BaseException] = []
            thread = threading.Thread(
                target = _execute_run_with_workspace_lease,
                args = (
                    active,
                    profile,
                    workspace,
                    lease_ready,
                    startup_errors,
                    deadline,
                ),
                name = f"project-verification-{run['id'][:8]}",
                daemon = True,
            )
            active.heartbeat_thread = heartbeat_thread
            active.thread = thread
            heartbeat_thread.start()
            thread.start()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not lease_ready.wait(remaining):
                active.cancel_event.set()
                raise AgentWorkspaceError(
                    "Timed out while acquiring the project verification workspace lease."
                )
            if startup_errors:
                startup_error = startup_errors[0]
                if isinstance(startup_error, AgentWorkspaceError):
                    raise startup_error
                raise AgentWorkspaceError(
                    "Project verification could not acquire its workspace lease."
                ) from startup_error
            return run
    except BaseException:
        if run is not None:
            try:
                verification_state.complete_verification_run(
                    project_id,
                    run["id"],
                    [],
                    owner_id = PROCESS_OWNER_ID,
                    terminal_status = "blocked",
                    error = "Verification could not start.",
                )
            except BaseException:
                pass
        _release_active(active)
        raise


def cancel_verification(project_id: str, run_id: str) -> tuple[dict, bool]:
    """Durably request cancellation before signalling the owned process runner."""
    run, accepted = verification_state.request_verification_cancel(project_id, run_id)
    if accepted:
        with _ACTIVE_LOCK:
            active = _ACTIVE_BY_RUN.get(run_id)
        if active is not None and active.project_id == project_id:
            active.cancel_event.set()
    return run, accepted


def begin_project_deletion(project_id: str) -> None:
    active = _ActiveDeletion(project_id = project_id)
    durable_fence_started = False
    with _ACTIVE_LOCK:
        if project_id in _DELETING_PROJECTS:
            raise AgentWorkspaceError("Project deletion is already in progress.")
        _DELETING_PROJECTS[project_id] = active
    try:
        fence = verification_state.begin_verification_project_deletion(
            project_id,
            active.fence_id,
        )
        revision = fence.get("revision") if isinstance(fence, dict) else None
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise AgentWorkspaceError("Project deletion authority is unavailable.")
        active.revision = revision
        durable_fence_started = True
        heartbeat_thread = threading.Thread(
            target = _heartbeat_deletion,
            args = (active,),
            name = f"project-verification-delete-{active.fence_id[:8]}",
            daemon = True,
        )
        active.heartbeat_thread = heartbeat_thread
        heartbeat_thread.start()
    except BaseException:
        active.stopped.set()
        if durable_fence_started and active.revision is not None:
            try:
                verification_state.finish_verification_project_deletion(
                    project_id,
                    active.fence_id,
                    active.revision,
                )
            except BaseException:
                pass
        with _ACTIVE_LOCK:
            if _DELETING_PROJECTS.get(project_id) is active:
                _DELETING_PROJECTS.pop(project_id, None)
        raise


def finish_project_deletion(project_id: str) -> None:
    with _ACTIVE_LOCK:
        active = _DELETING_PROJECTS.get(project_id)
    if active is None:
        return
    active.stopped.set()
    try:
        if active.revision is None:
            raise AgentWorkspaceError("Project deletion authority is unavailable.")
        verification_state.finish_verification_project_deletion(
            project_id,
            active.fence_id,
            active.revision,
        )
    finally:
        if active.execution_fence_fd is not None:
            supervisor._release_project_execution_fence(active.execution_fence_fd)
            active.execution_fence_fd = None
        with _ACTIVE_LOCK:
            if _DELETING_PROJECTS.get(project_id) is active:
                _DELETING_PROJECTS.pop(project_id, None)


def cancel_project_verifications_and_wait(
    project_id: str, *, timeout_seconds: float = SHUTDOWN_TIMEOUT_SECONDS
) -> None:
    deadline = time.monotonic() + timeout_seconds
    try:
        verification_state.request_project_verification_cancel(project_id)
    except AgentWorkspaceError:
        pass
    with _ACTIVE_LOCK:
        active = _ACTIVE_BY_PROJECT.get(project_id)
    if active is not None:
        active.cancel_event.set()
    while True:
        with _ACTIVE_LOCK:
            deletion = _DELETING_PROJECTS.get(project_id)
        if deletion is None:
            raise AgentWorkspaceError("Project verification retirement is not fenced.")
        if deletion.fence_lost.is_set():
            raise AgentWorkspaceError("Project verification retirement authority was lost.")
        with _ACTIVE_LOCK:
            local = _ACTIVE_BY_PROJECT.get(project_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AgentWorkspaceError(
                "Timed out while stopping project verification. Project deletion was cancelled."
            )
        if local is not None:
            local.completed.wait(min(0.1, remaining))
            continue
        running = verification_state.active_verification_run_lifecycle(project_id)
        if running is None:
            _acquire_deletion_execution_fence(deletion, deadline)
            return
        time.sleep(min(0.1, remaining))


def shutdown_project_verifications(*, timeout_seconds: float = SHUTDOWN_TIMEOUT_SECONDS) -> int:
    """Cancel and join every owned run, returning the unfinished count."""
    deadline = time.monotonic() + timeout_seconds
    with _ACTIVE_LOCK:
        active_runs = list(_ACTIVE_BY_PROJECT.values())
    for active in active_runs:
        if active.run_id is not None:
            try:
                verification_state.request_verification_cancel(
                    active.project_id,
                    active.run_id,
                )
            except AgentWorkspaceError:
                pass
        active.cancel_event.set()
    unfinished = 0
    for active in active_runs:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not active.completed.wait(remaining):
            unfinished += 1
    return unfinished


def execution_status() -> dict:
    status = supervisor.supervised_process_status()
    return {
        "available": status.available,
        "backend": status.backend,
        "reason": status.reason,
    }


__all__ = [
    "MAX_ACTIVE_VERIFICATIONS",
    "begin_project_deletion",
    "cancel_project_verifications_and_wait",
    "cancel_verification",
    "execution_status",
    "finish_project_deletion",
    "shutdown_project_verifications",
    "start_project_verification",
]

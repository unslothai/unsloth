# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated API for project-scoped agent workspace services."""

import asyncio
import threading
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.background import manager as background_manager
from core.agent_workspace.common import AgentWorkspaceError, git_root, project_workspace
from core.agent_workspace.discovery import (
    build_repository_map,
    secure_repository_traversal_supported,
)
from core.agent_workspace.execution import execution_boundary_status
from core.agent_workspace.git_service import (
    confirm_prepared_commit,
    create_checkpoint,
    git_diff,
    git_status,
    prepare_commit,
    rollback_checkpoint,
)
from core.agent_workspace.github_handoff import (
    consume_pull_request_handoff,
    prepare_pull_request_handoff,
    pull_request_review_binding_current,
)
from core.agent_workspace.instructions import (
    resolve_agents_instructions,
    secure_instruction_traversal_supported,
)
from core.agent_workspace.project_context import (
    create_project_context_snapshot,
    project_context_snapshot_response,
)
from core.agent_workspace.review import (
    build_pull_request_draft,
    build_review_summary,
    redact_review_text,
)
from core.agent_workspace.state import (
    create_plan,
    get_background_task,
    get_plan,
    get_verification_config,
    list_background_tasks,
    list_plans,
    list_worktrees,
    set_verification_config,
    update_plan_status,
    update_plan_task,
)
from core.agent_workspace.verification import (
    cancel_verification,
    run_project_verification,
    verification_run_with_freshness,
    verification_runs_with_freshness,
)
from core.agent_workspace.worktrees import (
    cleanup_worktree,
    create_worktree,
    merge_worktree,
)
from core.inference.mcp_client import (
    call_tool_sync,
    is_stdio,
    list_tools_async,
    parse_server_headers,
    probe_timeout,
    stdio_mcp_enabled,
)
from storage import mcp_servers_db
from storage.studio_db import get_chat_project
from routes.provider_credentials import require_ui_session


router = APIRouter()
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]


def _public_git_status(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "head",
            "branch",
            "detached",
            "clean",
            "counts",
            "files",
            "truncated",
        )
    }


def _public_checkpoint(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "id",
            "projectId",
            "commitSha",
            "ownedPaths",
            "sourceFingerprint",
            "createdAt",
        )
    }


def _public_prepared_commit(record: dict) -> dict:
    allowed = (
        "id",
        "projectId",
        "status",
        "branch",
        "baseHead",
        "message",
        "ownedPaths",
        "sourceFingerprint",
        "createdAt",
        "expiresAt",
        "confirmationToken",
        "files",
        "diff",
        "diffTruncated",
        "commitSha",
        "refName",
        "confirmedAt",
    )
    return {key: record[key] for key in allowed if key in record}


def _public_worktree(record: dict) -> dict:
    public = {
        key: record[key]
        for key in (
            "id",
            "projectId",
            "branch",
            "baseRef",
            "backgroundTaskId",
            "status",
            "createdAt",
            "updatedAt",
        )
    }
    if "merge" in record:
        public["merge"] = record["merge"]
    return public


def _redact_background_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_review_text(value, "")
    if isinstance(value, list):
        return [_redact_background_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_background_value(item) for key, item in value.items()}
    return value


def _public_background_task(record: dict) -> dict:
    public = dict(record)
    public["error"] = redact_review_text(str(record["error"]), "") if record.get("error") else None
    public_result = _redact_background_value(record.get("result"))
    if isinstance(public_result, dict):
        # The reserved session is a server-owned execution binding. It is not a
        # public cancellation or workspace capability.
        public_result.pop("sessionId", None)
    public["result"] = public_result
    return public


def _workspace_error(exc: AgentWorkspaceError) -> HTTPException:
    detail = redact_review_text(str(exc), "")
    status = 404 if "not found" in detail.lower() else 409
    return HTTPException(status_code = status, detail = detail)


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None:
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _require_execution_boundary() -> None:
    status = execution_boundary_status()
    if not status.available:
        raise AgentWorkspaceError(
            status.reason or "Project command execution is unavailable on this host."
        )


async def _github_connector_tools(server_id: str) -> tuple[dict, list[dict]]:
    server = await asyncio.to_thread(mcp_servers_db.get_server, server_id)
    if server is None or not server.get("is_enabled"):
        raise AgentWorkspaceError("The selected GitHub connector is unavailable.")
    if is_stdio(server["url"]) and not stdio_mcp_enabled():
        raise AgentWorkspaceError("Local MCP connectors are disabled on this host.")
    try:
        tools = await list_tools_async(
            url = server["url"],
            headers = parse_server_headers(server),
            timeout = probe_timeout(server["url"], bool(server.get("use_oauth"))),
            use_oauth = bool(server.get("use_oauth")),
        )
    except Exception as exc:
        raise AgentWorkspaceError("The selected GitHub connector could not be reached.") from exc
    return server, tools


def _bounded_connector_result(value: str, limit: int = 32_000) -> tuple[str, bool]:
    redacted = redact_review_text(value, "")
    encoded = redacted.encode("utf-8", errors = "replace")
    truncated = len(encoded) > limit
    return encoded[:limit].decode("utf-8", errors = "replace"), truncated


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str = Field(min_length = 1, max_length = 120)
    kind: Literal["test", "lint", "build", "custom"] = "custom"
    command: str = Field(min_length = 1, max_length = 16_384)
    required: bool = True
    timeoutSeconds: int = Field(default = 300, ge = 1, le = 3600)
    logLimitBytes: int = Field(default = 256 * 1024, ge = 1024, le = 2 * 1024 * 1024)


class ProjectContextSnapshotRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    query: str = Field(default = "", max_length = 16_384)


class VerificationConfigRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    checks: list[VerificationCheck] = Field(default_factory = list, max_length = 32)
    requireForGoalCompletion: bool = False
    expectedRevision: int = Field(ge = 0)


class VerificationRunRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    selectedNames: Optional[list[str]] = Field(default = None, max_length = 32)
    worktreeId: Optional[str] = Field(default = None, max_length = 128)
    configRevision: int = Field(ge = 0)


class PlanTaskRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: str = Field(min_length = 1, max_length = 500)
    status: Literal["pending", "running", "blocked", "completed", "cancelled"] = "pending"
    blocker: Optional[str] = Field(default = None, max_length = 4000)
    verification: list[dict[str, Any]] = Field(default_factory = list, max_length = 32)


class PlanCreateRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: str = Field(min_length = 1, max_length = 500)
    tasks: list[PlanTaskRequest] = Field(default_factory = list, max_length = 500)


class PlanPatchRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    status: Literal["active", "blocked", "completed", "cancelled"]
    expectedRevision: int = Field(ge = 0)


class PlanTaskPatchRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    status: Optional[Literal["pending", "running", "blocked", "completed", "cancelled"]] = None
    blocker: Optional[str] = Field(default = None, max_length = 4000)
    verification: Optional[list[dict[str, Any]]] = Field(default = None, max_length = 32)
    expectedRevision: int = Field(ge = 0)


class CheckpointRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    ownedPaths: list[str] = Field(min_length = 1, max_length = 5000)


class PrepareCommitRequest(CheckpointRequest):
    message: str = Field(min_length = 1, max_length = 32_000)


class ConfirmPreparedCommitRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    confirmationToken: str = Field(min_length = 32, max_length = 256)


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    expectedCurrentFingerprint: str = Field(min_length = 64, max_length = 64)


class WorktreeRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    branch: Optional[str] = Field(default = None, max_length = 136)
    baseRef: str = Field(default = "HEAD", min_length = 1, max_length = 256)
    backgroundTaskId: Optional[str] = Field(default = None, max_length = 128)


class BackgroundVerificationRequest(VerificationRunRequest):
    start: bool = True


class BackgroundAgentRuntimeRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    kind: Literal["local", "provider"]
    model: str = Field(min_length = 1, max_length = 512)
    providerId: Optional[str] = Field(default = None, max_length = 256)
    permissionMode: Literal["off", "full"]
    reasoningEffort: Optional[str] = Field(default = None, max_length = 64)
    maxOutputTokens: int = Field(default = 8192, ge = 1, le = 32_768)


class BackgroundAgentRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    instruction: str = Field(min_length = 1, max_length = 32_768)
    runtime: BackgroundAgentRuntimeRequest
    planId: Optional[str] = Field(default = None, max_length = 128)
    planTaskId: Optional[str] = Field(default = None, max_length = 128)
    worktreeId: Optional[str] = Field(default = None, max_length = 128)
    cleanupWorktreeOnCancel: bool = False
    start: bool = True


class WorktreeMergeRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    expectedTargetHead: str = Field(min_length = 40, max_length = 64)


class PullRequestDraftRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: str = Field(default = "", max_length = 120)
    bodyNote: str = Field(default = "", max_length = 8000)


class PullRequestHandoffRequest(PullRequestDraftRequest):
    model_config = ConfigDict(extra = "forbid")

    serverId: str = Field(
        min_length = 1,
        max_length = 128,
        pattern = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    owner: str = Field(min_length = 1, max_length = 100)
    repository: str = Field(min_length = 1, max_length = 100)
    base: str = Field(min_length = 1, max_length = 255)
    head: str = Field(min_length = 1, max_length = 255)
    draft: bool = True


class PullRequestHandoffConfirmRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    serverId: str = Field(
        min_length = 1,
        max_length = 128,
        pattern = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    confirmationToken: str = Field(min_length = 32, max_length = 256)
    expectedRequestDigest: str = Field(pattern = r"^[0-9a-f]{64}$")


@router.get("/projects/{project_id}/workspace")
def workspace_capabilities(
    project_id: str, current_subject: str = Depends(get_current_subject)
) -> dict:
    project = _project(project_id)
    unavailable_capabilities = {
        "instructions": False,
        "repositoryMap": False,
        "verification": False,
        "plans": False,
        "background": False,
        "git": False,
        "worktrees": False,
        "review": False,
    }
    try:
        workspace = project_workspace(project_id)
    except AgentWorkspaceError as exc:
        return {
            "projectId": project_id,
            "workspaceKind": project.get("workspaceKind") or "managed",
            "available": False,
            "error": redact_review_text(str(exc), ""),
            "isGitRepository": False,
            "capabilities": unavailable_capabilities,
        }
    try:
        git_root(workspace.root)
        is_git_repository = True
    except AgentWorkspaceError:
        is_git_repository = False
    execution = execution_boundary_status()
    instruction_traversal = secure_instruction_traversal_supported()
    repository_traversal = secure_repository_traversal_supported()
    return {
        "projectId": project_id,
        "workspaceKind": workspace.kind,
        "available": True,
        "error": None,
        "isGitRepository": is_git_repository,
        "executionBoundary": {
            "available": execution.available,
            "backend": execution.backend,
            "reason": execution.reason,
        },
        "capabilities": {
            "instructions": instruction_traversal,
            "repositoryMap": repository_traversal,
            "verification": execution.available,
            "plans": True,
            "background": execution.available,
            "git": is_git_repository,
            "worktrees": is_git_repository and workspace.kind == "folder",
            "review": True,
        },
    }


@router.post("/projects/{project_id}/context-snapshots")
def project_context_snapshot(
    project_id: str,
    payload: Optional[ProjectContextSnapshotRequest] = None,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return project_context_snapshot_response(
            create_project_context_snapshot(project_id, (payload.query if payload else ""))
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/instructions")
def project_instructions(
    project_id: str,
    target: Optional[str] = Query(default = None, max_length = 4096),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        workspace = project_workspace(project_id)
        expected_identity = (
            (workspace.device_id, workspace.file_id)
            if workspace.device_id is not None and workspace.file_id is not None
            else None
        )
        return resolve_agents_instructions(
            workspace.root,
            target,
            expected_identity = expected_identity,
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/repository-map")
async def repository_map(
    project_id: str,
    request: Request,
    max_paths: int = Query(default = 20_000, ge = 1, le = 100_000),
    max_total_bytes: int = Query(default = 2 * 1024 * 1024, ge = 1024, le = 16 * 1024 * 1024),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    cancelled = threading.Event()
    scan_task = None
    try:
        workspace = project_workspace(project_id)
        expected_identity = (
            (workspace.device_id, workspace.file_id)
            if workspace.device_id is not None and workspace.file_id is not None
            else None
        )
        scan_task = asyncio.create_task(
            asyncio.to_thread(
                build_repository_map,
                workspace.root,
                max_paths = max_paths,
                max_total_bytes = max_total_bytes,
                cancelled = cancelled.is_set,
                expected_identity = expected_identity,
            )
        )
        disconnected = False
        while not scan_task.done():
            done, _pending = await asyncio.wait({scan_task}, timeout = 0.025)
            if done:
                break
            if await request.is_disconnected():
                disconnected = True
                cancelled.set()
                break
        result = await scan_task
        if disconnected:
            raise HTTPException(status_code = 499, detail = "Request cancelled.")
        return result
    except asyncio.CancelledError:
        cancelled.set()
        raise
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc
    finally:
        if scan_task is not None and not scan_task.done():
            cancelled.set()


@router.get("/projects/{project_id}/verification")
def verification_config(
    project_id: str, current_subject: str = Depends(get_current_subject)
) -> dict:
    _project(project_id)
    return get_verification_config(project_id)


@router.put("/projects/{project_id}/verification")
def save_verification_config(
    project_id: str,
    payload: VerificationConfigRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        config = set_verification_config(
            project_id,
            [check.model_dump() for check in payload.checks],
            require_for_goal_completion = payload.requireForGoalCompletion,
            expected_revision = payload.expectedRevision,
        )
        config["shellContract"] = (
            "Commands are trusted user configuration and run through the platform shell "
            "inside the host project-execution boundary. Writes are limited to the "
            "project and isolated scratch space, private home data outside the project "
            "is unreadable, and network access is disabled."
        )
        return config
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/verify")
def run_verification(
    project_id: str,
    payload: VerificationRunRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        _require_execution_boundary()
        return run_project_verification(
            project_id,
            payload.selectedNames,
            worktree_id = payload.worktreeId,
            config_revision = payload.configRevision,
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/verifications")
def verification_runs(
    project_id: str,
    limit: int = Query(default = 20, ge = 1, le = 100),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    return {"runs": verification_runs_with_freshness(project_id, limit)}


@router.get("/projects/{project_id}/verifications/{run_id}")
def verification_run(
    project_id: str,
    run_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    run = verification_run_with_freshness(run_id)
    if run is None or run["projectId"] != project_id:
        raise HTTPException(status_code = 404, detail = "Verification run not found.")
    return run


@router.post("/projects/{project_id}/verifications/{run_id}/cancel")
def cancel_verification_run(
    project_id: str,
    run_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    run = verification_run_with_freshness(run_id)
    if run is None or run["projectId"] != project_id:
        raise HTTPException(status_code = 404, detail = "Verification run not found.")
    return {"cancelRequested": cancel_verification(run_id)}


@router.get("/projects/{project_id}/git/status")
def project_git_status(
    project_id: str, current_subject: str = Depends(get_current_subject)
) -> dict:
    try:
        return _public_git_status(git_status(project_id))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/git/diff")
def project_git_diff(
    project_id: str,
    staged: bool = Query(False),
    max_bytes: int = Query(default = 512_000, ge = 4096, le = 2_000_000),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return git_diff(project_id, staged = staged, max_bytes = max_bytes)
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/git/checkpoints")
def save_git_checkpoint(
    project_id: str,
    payload: CheckpointRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_checkpoint(create_checkpoint(project_id, payload.ownedPaths))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/git/commits/prepare")
def prepare_git_commit(
    project_id: str,
    payload: PrepareCommitRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_prepared_commit(
            prepare_commit(project_id, payload.ownedPaths, payload.message)
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/git/commits/preparations/{preparation_id}/confirm")
def confirm_git_prepared_commit(
    project_id: str,
    preparation_id: str,
    payload: ConfirmPreparedCommitRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_prepared_commit(
            confirm_prepared_commit(project_id, preparation_id, payload.confirmationToken)
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/git/checkpoints/{checkpoint_id}/rollback")
def rollback_git_checkpoint(
    project_id: str,
    checkpoint_id: str,
    payload: RollbackRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return rollback_checkpoint(project_id, checkpoint_id, payload.expectedCurrentFingerprint)
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/plans")
def save_plan(
    project_id: str,
    payload: PlanCreateRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    project = _project(project_id)
    try:
        return create_plan(
            project_id,
            payload.title,
            project.get("goal"),
            [task.model_dump() for task in payload.tasks],
            goal_updated_at = project.get("goalUpdatedAt"),
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/plans")
def project_plans(project_id: str, current_subject: str = Depends(get_current_subject)) -> dict:
    _project(project_id)
    return {"plans": list_plans(project_id)}


@router.patch("/projects/{project_id}/plans/{plan_id}")
def patch_plan(
    project_id: str,
    plan_id: str,
    payload: PlanPatchRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    plan = get_plan(plan_id)
    if plan is None or plan["projectId"] != project_id:
        raise HTTPException(status_code = 404, detail = "Plan not found.")
    try:
        return (
            update_plan_status(plan_id, payload.status, expected_revision = payload.expectedRevision)
            or plan
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.patch("/projects/{project_id}/plans/{plan_id}/tasks/{task_id}")
def patch_plan_task(
    project_id: str,
    plan_id: str,
    task_id: str,
    payload: PlanTaskPatchRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    plan = get_plan(plan_id)
    if plan is None or plan["projectId"] != project_id:
        raise HTTPException(status_code = 404, detail = "Plan not found.")
    try:
        task_updates: dict[str, Any] = {
            "status": payload.status,
            "verification": payload.verification,
            "expected_revision": payload.expectedRevision,
        }
        if "blocker" in payload.model_fields_set:
            task_updates["blocker"] = payload.blocker
        updated = update_plan_task(plan_id, task_id, **task_updates)
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc
    if updated is None:
        raise HTTPException(status_code = 404, detail = "Plan task not found.")
    return updated


@router.post("/projects/{project_id}/background/verification")
def queue_background_verification(
    project_id: str,
    payload: BackgroundVerificationRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        _require_execution_boundary()
        return _public_background_task(
            background_manager.enqueue_verification(
                project_id,
                payload.selectedNames,
                worktree_id = payload.worktreeId,
                config_revision = payload.configRevision,
                start = payload.start,
            )
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/background/agent")
def queue_background_agent(
    project_id: str,
    payload: BackgroundAgentRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    _project(project_id)
    if payload.runtime.kind == "provider":
        require_ui_session(via_api_key)
    try:
        _require_execution_boundary()
        return _public_background_task(
            background_manager.enqueue_agent(
                project_id,
                payload.instruction,
                runtime_selection = payload.runtime.model_dump(exclude_none = True),
                plan_id = payload.planId,
                plan_task_id = payload.planTaskId,
                worktree_id = payload.worktreeId,
                cleanup_worktree_on_cancel = payload.cleanupWorktreeOnCancel,
                start = payload.start,
            )
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/background")
def background_tasks(
    project_id: str,
    limit: int = Query(default = 100, ge = 1, le = 500),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    return {
        "tasks": [
            _public_background_task(task) for task in list_background_tasks(project_id, limit)
        ]
    }


def _background_for_project(project_id: str, task_id: str) -> dict:
    task = get_background_task(task_id)
    if task is None or task["projectId"] != project_id:
        raise HTTPException(status_code = 404, detail = "Background task not found.")
    return task


def _require_ui_for_provider_task(task: dict, via_api_key: bool) -> None:
    runtime = (task.get("payload") or {}).get("runtime") or {}
    if task.get("kind") == "agent" and runtime.get("kind") == "provider":
        require_ui_session(via_api_key)


@router.post("/projects/{project_id}/background/{task_id}/start")
def start_background_task(
    project_id: str,
    task_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    task = _background_for_project(project_id, task_id)
    _require_ui_for_provider_task(task, via_api_key)
    try:
        _require_execution_boundary()
        return _public_background_task(background_manager.start(task_id))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/background/{task_id}/cancel")
def cancel_background_task(
    project_id: str,
    task_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _background_for_project(project_id, task_id)
    try:
        return _public_background_task(background_manager.cancel(task_id))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/background/{task_id}/retry")
def retry_background(
    project_id: str,
    task_id: str,
    start: bool = Query(True),
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    task = _background_for_project(project_id, task_id)
    _require_ui_for_provider_task(task, via_api_key)
    try:
        _require_execution_boundary()
        return _public_background_task(background_manager.retry(task_id, start = start))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/worktrees")
def save_worktree(
    project_id: str,
    payload: WorktreeRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_worktree(
            create_worktree(
                project_id,
                branch = payload.branch,
                base_ref = payload.baseRef,
                background_task_id = payload.backgroundTaskId,
            )
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code = 500, detail = "Worktree creation could not be completed."
        ) from exc


@router.get("/projects/{project_id}/worktrees")
def project_worktrees(project_id: str, current_subject: str = Depends(get_current_subject)) -> dict:
    _project(project_id)
    try:
        return {"worktrees": [_public_worktree(record) for record in list_worktrees(project_id)]}
    except Exception as exc:
        raise HTTPException(status_code = 500, detail = "Worktree state is unavailable.") from exc


@router.delete("/projects/{project_id}/worktrees/{worktree_id}")
def remove_worktree(
    project_id: str,
    worktree_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_worktree(cleanup_worktree(project_id, worktree_id))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code = 500, detail = "Worktree cleanup could not be completed."
        ) from exc


@router.post("/projects/{project_id}/worktrees/{worktree_id}/merge")
def merge_owned_worktree(
    project_id: str,
    worktree_id: str,
    payload: WorktreeMergeRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return _public_worktree(merge_worktree(project_id, worktree_id, payload.expectedTargetHead))
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code = 500, detail = "Worktree merge could not be completed."
        ) from exc


@router.get("/projects/{project_id}/review")
def project_review(project_id: str, current_subject: str = Depends(get_current_subject)) -> dict:
    try:
        return build_review_summary(project_id)
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-draft")
def pull_request_draft(
    project_id: str,
    payload: PullRequestDraftRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        return build_pull_request_draft(project_id, title = payload.title, body_note = payload.bodyNote)
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-handoff/prepare")
async def prepare_connected_pull_request(
    project_id: str,
    payload: PullRequestHandoffRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    require_ui_session(via_api_key)
    _project(project_id)
    try:
        _server, tools = await _github_connector_tools(payload.serverId)
        return await asyncio.to_thread(
            prepare_pull_request_handoff,
            project_id,
            server_id = payload.serverId,
            owner = payload.owner,
            repository = payload.repository,
            base = payload.base,
            head = payload.head,
            title = payload.title,
            body_note = payload.bodyNote,
            draft = payload.draft,
            tools = tools,
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-handoff/{handoff_id}/confirm")
async def confirm_connected_pull_request(
    project_id: str,
    handoff_id: str,
    payload: PullRequestHandoffConfirmRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    require_ui_session(via_api_key)
    _project(project_id)
    try:
        _probed_server, tools = await _github_connector_tools(payload.serverId)
        server, arguments, review_binding = await asyncio.to_thread(
            consume_pull_request_handoff,
            project_id,
            handoff_id,
            server_id = payload.serverId,
            confirmation_token = payload.confirmationToken,
            expected_request_digest = payload.expectedRequestDigest,
            tools = tools,
            include_review_binding = True,
        )
    except AgentWorkspaceError as exc:
        raise _workspace_error(exc) from exc

    headers = parse_server_headers(server)
    expected_config = (
        server.get("url"),
        server.get("headers_json"),
        bool(server.get("is_enabled")),
        bool(server.get("use_oauth")),
        server.get("updated_at"),
    )

    def _connector_config_current() -> bool:
        current = mcp_servers_db.get_server(payload.serverId)
        if current is None:
            return False
        connector_current = (
            current.get("url"),
            current.get("headers_json"),
            bool(current.get("is_enabled")),
            bool(current.get("use_oauth")),
            current.get("updated_at"),
        ) == expected_config
        return connector_current and pull_request_review_binding_current(project_id, review_binding)

    result = await asyncio.to_thread(
        call_tool_sync,
        url = server["url"],
        headers = headers,
        name = "create_pull_request",
        args = arguments,
        timeout = 90,
        use_oauth = bool(server.get("use_oauth")),
        scope = f"agent-workspace:pull-request:{handoff_id}",
        config_check = _connector_config_current,
    )
    safe_result, truncated = _bounded_connector_result(result)
    if result.startswith("Error:"):
        raise HTTPException(
            status_code = 502,
            detail = (
                "The connector did not confirm submission. Check GitHub before "
                "creating another handoff."
            ),
        )
    return {
        "id": handoff_id,
        "requestDigest": payload.expectedRequestDigest,
        "connector": {
            "id": payload.serverId,
            "displayName": str(server.get("display_name") or "GitHub"),
        },
        "submitted": True,
        "connectorResult": safe_result,
        "connectorResultTruncated": truncated,
    }

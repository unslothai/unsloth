# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated project worktree endpoints.

Only the lifecycle metadata is returned. Filesystem paths and ownership tokens
stay server-side, where the worktree service can revalidate them on every use.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.git_context import AgentWorkspaceError
from core.agent_workspace.checkpoints import (
    create_checkpoint,
    list_checkpoints,
    rollback_checkpoint,
)
from core.agent_workspace.git_service import workspace_fingerprint
from core.agent_workspace.github_handoff import (
    consume_pull_request_handoff,
    prepare_pull_request_handoff,
    pull_request_review_binding_current,
    remote_head_probe,
    require_remote_head,
)
from core.agent_workspace.review import (
    build_pull_request_draft,
    build_review_summary,
    redact_review_text,
)
from core.inference.mcp_client import (
    call_tool_sync,
    is_stdio,
    list_tools_async,
    parse_server_headers,
    probe_timeout,
    stdio_mcp_enabled,
)
from routes.provider_credentials import require_ui_session
from storage import mcp_servers_db
from core.agent_workspace.worktrees import (
    cleanup_worktree,
    create_worktree,
    list_project_worktrees,
    merge_owned_worktree,
)
from storage.studio_db import get_chat_project


ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]


async def require_git_request(
    project_id: str,
    request: Request,
    workspace_revision: Annotated[int, Query(alias = "workspaceRevision", ge = 0)],
    via_api_key: ViaApiKey = False,
    _subject: str = Depends(get_current_subject),
):
    project = await asyncio.to_thread(_project, project_id)
    if int(project.get("workspaceRevision") or 0) != workspace_revision:
        raise HTTPException(
            status_code = 409, detail = "Project workspace changed. Refresh Git review."
        )
    if request.method != "GET":
        require_ui_session(via_api_key)
    from core.agent_workspace.git_context import git_request_revision

    token = git_request_revision.set(workspace_revision)
    try:
        yield
    finally:
        git_request_revision.reset(token)


router = APIRouter(dependencies = [Depends(require_git_request)])


class WorktreeCreateRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    branch: Optional[str] = Field(default = None, min_length = 1, max_length = 136)
    baseRef: str = Field(default = "HEAD", min_length = 1, max_length = 256)


class WorktreeMergeRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    expectedTargetHead: str = Field(min_length = 40, max_length = 64)


class CheckpointRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    ownedPaths: list[str] = Field(min_length = 1, max_length = 5000)


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    expectedCurrentFingerprint: str = Field(min_length = 64, max_length = 64)


class PullRequestDraftRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: str = Field(default = "", max_length = 120)
    bodyNote: str = Field(default = "", max_length = 8000)


class PullRequestHandoffRequest(PullRequestDraftRequest):
    serverId: str = Field(min_length = 1, max_length = 128)
    owner: str = Field(min_length = 1, max_length = 100)
    repository: str = Field(min_length = 1, max_length = 100)
    base: str = Field(min_length = 1, max_length = 255)
    head: str = Field(min_length = 1, max_length = 255)
    draft: bool = True


class PullRequestHandoffConfirmRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    serverId: str = Field(min_length = 1, max_length = 128)
    confirmationToken: str = Field(min_length = 32, max_length = 256)
    expectedRequestDigest: str = Field(pattern = r"^[0-9a-f]{64}$")


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _error(exc: AgentWorkspaceError) -> HTTPException:
    status = 404 if "not found" in str(exc).lower() else 409
    return HTTPException(status_code = status, detail = str(exc))


def _public(record: dict) -> dict:
    return {
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
        if key in record
    } | ({"merge": record["merge"]} if record.get("merge") is not None else {})


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


async def _connector_tools(server_id: str) -> tuple[dict, list[dict]]:
    server = await asyncio.to_thread(mcp_servers_db.get_server, server_id)
    if server is None or not server.get("is_enabled"):
        raise AgentWorkspaceError("The selected GitHub connector is unavailable.")
    if server.get("use_oauth"):
        from core.inference import mcp_client
        if getattr(mcp_client, "MCP_ONE_SHOT_CONFIG_CHECK_VERSION", 0) < 1:
            raise AgentWorkspaceError(
                "OAuth GitHub handoff is unavailable until connector configuration checks are installed."
            )
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
    redacted = redact_review_text(str(value))
    encoded = redacted.encode("utf-8", errors = "replace")
    return encoded[:limit].decode("utf-8", errors = "ignore"), len(encoded) > limit


@router.get("/projects/{project_id}/git/checkpoints")
def project_checkpoints(
    project_id: str, _current_subject: str = Depends(get_current_subject)
) -> dict:
    _project(project_id)
    return {"checkpoints": [_public_checkpoint(record) for record in list_checkpoints(project_id)]}


@router.post("/projects/{project_id}/git/checkpoints")
def save_project_checkpoint(
    project_id: str,
    payload: CheckpointRequest,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return _public_checkpoint(create_checkpoint(project_id, payload.ownedPaths))
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.post("/projects/{project_id}/git/checkpoints/{checkpoint_id}/rollback")
def rollback_project_checkpoint(
    project_id: str,
    checkpoint_id: str,
    payload: RollbackRequest,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        result = rollback_checkpoint(
            project_id,
            checkpoint_id,
            payload.expectedCurrentFingerprint,
        )
        return {
            "checkpoint": _public_checkpoint(result["checkpoint"]),
            "currentFingerprint": result["currentFingerprint"],
        }
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.get("/projects/{project_id}/review")
def project_review(project_id: str, _current_subject: str = Depends(get_current_subject)) -> dict:
    _project(project_id)
    try:
        return build_review_summary(project_id)
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-draft")
def pull_request_draft(
    project_id: str,
    payload: PullRequestDraftRequest,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return build_pull_request_draft(
            project_id,
            title = payload.title,
            body_note = payload.bodyNote,
        )
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-handoff/prepare")
async def prepare_pull_request(
    project_id: str,
    payload: PullRequestHandoffRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    require_ui_session(via_api_key)
    _project(project_id)
    try:
        _server, tools = await _connector_tools(payload.serverId)
        remote_head_probe(
            tools, {"owner": payload.owner, "repo": payload.repository, "head": payload.head}
        )
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
        raise _error(exc) from exc


@router.post("/projects/{project_id}/review/pull-request-handoff/{handoff_id}/confirm")
async def confirm_pull_request(
    project_id: str,
    handoff_id: str,
    payload: PullRequestHandoffConfirmRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
) -> dict:
    require_ui_session(via_api_key)
    _project(project_id)
    try:
        _server, tools = await _connector_tools(payload.serverId)
        server, request, binding = await asyncio.to_thread(
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
        raise _error(exc) from exc

    expected_config = (
        server.get("url"),
        server.get("headers_json"),
        bool(server.get("is_enabled")),
        bool(server.get("use_oauth")),
        server.get("updated_at"),
    )

    def _connector_current() -> bool:
        current = mcp_servers_db.get_server(payload.serverId)
        if current is None:
            return False
        actual = (
            current.get("url"),
            current.get("headers_json"),
            bool(current.get("is_enabled")),
            bool(current.get("use_oauth")),
            current.get("updated_at"),
        )
        return actual == expected_config and pull_request_review_binding_current(
            project_id,
            binding,
        )

    try:
        probe = remote_head_probe(tools, request)
        remote_result = await asyncio.to_thread(
            call_tool_sync,
            url = server["url"],
            headers = parse_server_headers(server),
            name = "get_commit",
            args = probe,
            timeout = 30,
            use_oauth = bool(server.get("use_oauth")),
            scope = f"git-head-review:{handoff_id}",
            config_check = _connector_current,
        )
        require_remote_head(remote_result, binding["head"])
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc

    result = await asyncio.to_thread(
        call_tool_sync,
        url = server["url"],
        headers = parse_server_headers(server),
        name = "create_pull_request",
        args = request,
        timeout = 300,
        use_oauth = bool(server.get("use_oauth")),
        scope = f"agent-workspace:pull-request:{handoff_id}",
        config_check = _connector_current,
    )
    bounded, truncated = _bounded_connector_result(result)
    if bounded.startswith("Error:"):
        raise HTTPException(
            status_code = 502,
            detail = "GitHub handoff outcome is unknown. Check GitHub before retrying.",
        )
    return {
        "submitted": True,
        "result": bounded,
        "resultTruncated": truncated,
        "reviewBinding": binding,
    }


@router.get("/projects/{project_id}/worktrees")
def project_worktrees(
    project_id: str, _current_subject: str = Depends(get_current_subject)
) -> dict:
    _project(project_id)
    return {"worktrees": [_public(record) for record in list_project_worktrees(project_id)]}


@router.post("/projects/{project_id}/worktrees")
def create_project_worktree(
    project_id: str,
    payload: WorktreeCreateRequest,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return _public(create_worktree(project_id, branch = payload.branch, base_ref = payload.baseRef))
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.delete("/projects/{project_id}/worktrees/{worktree_id}")
def remove_project_worktree(
    project_id: str,
    worktree_id: str,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return _public(cleanup_worktree(project_id, worktree_id))
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.post("/projects/{project_id}/worktrees/{worktree_id}/merge")
def merge_project_worktree(
    project_id: str,
    worktree_id: str,
    payload: WorktreeMergeRequest,
    _current_subject: str = Depends(get_current_subject),
) -> dict:
    _project(project_id)
    try:
        return _public(merge_owned_worktree(project_id, worktree_id, payload.expectedTargetHead))
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


__all__ = ["router"]


class PrepareCommitRequest(CheckpointRequest):
    message: str = Field(min_length = 1, max_length = 32000)


class ConfirmCommitRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    confirmationToken: str = Field(min_length = 32, max_length = 256)


@router.get("/projects/{project_id}/git/manage")
def git_management_state(project_id: str):
    from core.agent_workspace.git_guard import project_retirement_available
    from core.agent_workspace.git_service import repository_head, project_git
    from core.agent_workspace.prepared_commit_state import list_ref_bearing_preparations

    project = _project(project_id)
    result = {
        "projectId": project_id,
        "workspaceRevision": int(project.get("workspaceRevision") or 0),
        "mutationsAvailable": os.name == "posix" and project_retirement_available(),
        "checkpoints": [_public_checkpoint(record) for record in list_checkpoints(project_id)],
        "worktrees": [_public(record) for record in list_project_worktrees(project_id)],
        "preparedCommits": [
            {key: record[key] for key in ("id", "status", "refName", "commitSha", "message")}
            for record in list_ref_bearing_preparations(project_id)
        ],
        "head": None,
        "fingerprint": None,
    }
    if not project_retirement_available():
        result["unavailableReason"] = (
            "Git changes are unavailable until project lifecycle support is installed."
        )
    if os.name == "posix":
        try:
            root, repository = project_git(project_id)
            result.update(head = repository_head(repository), fingerprint = workspace_fingerprint(root))
        except AgentWorkspaceError as exc:
            result["unavailableReason"] = str(exc)
    return result


@router.post("/projects/{project_id}/git/commits/prepare")
def prepare_git_commit(project_id: str, payload: PrepareCommitRequest):
    from core.agent_workspace.prepared_commits import prepare_commit
    try:
        return prepare_commit(project_id, payload.ownedPaths, payload.message)
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.post("/projects/{project_id}/git/commits/preparations/{preparation_id}/confirm")
def confirm_git_commit(project_id: str, preparation_id: str, payload: ConfirmCommitRequest):
    from core.agent_workspace.prepared_commits import confirm_prepared_commit
    try:
        return confirm_prepared_commit(project_id, preparation_id, payload.confirmationToken)
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.delete("/projects/{project_id}/git/checkpoints/{checkpoint_id}")
def remove_git_checkpoint(project_id: str, checkpoint_id: str):
    from core.agent_workspace.checkpoints import delete_checkpoint
    try:
        return {"removed": delete_checkpoint(project_id, checkpoint_id)}
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.delete("/projects/{project_id}/git/commits/preparations/{preparation_id}")
def remove_git_preparation(project_id: str, preparation_id: str):
    from core.agent_workspace.prepared_commits import remove_prepared_commit
    try:
        return remove_prepared_commit(project_id, preparation_id)
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc


@router.get("/projects/{project_id}/worktrees/{worktree_id}/location")
def owned_worktree_location(
    project_id: str,
    worktree_id: str,
    via_api_key: ViaApiKey = False,
):
    require_ui_session(via_api_key)
    from core.agent_workspace.worktrees import owned_worktree_path
    try:
        return {"path": str(owned_worktree_path(project_id, worktree_id))}
    except AgentWorkspaceError as exc:
        raise _error(exc) from exc

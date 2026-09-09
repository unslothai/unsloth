# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated review and trust controls for project-local lifecycle hooks."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.verification_context import AgentWorkspaceError, project_workspace_access
from core.agent_workspace.hook_runtime import handler_is_supported
from core.agent_workspace.verification import execution_status
from core.agent_workspace.hooks import HOOK_EVENTS, discover_project_hooks
from storage.project_hook_trust_db import (
    ProjectHookTrustConflictError,
    ProjectHookTrustStateError,
    get_project_hook_trust_record,
    get_project_hook_trust_state,
    revoke_project_hook_trust,
    set_project_hook_handler_enabled,
    trust_project_hooks,
)
from storage.studio_db import get_chat_project


router = APIRouter()
_HASH_PATTERN = r"^[0-9a-f]{64}$"


class TrustProjectHooksRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    contentHash: str = Field(pattern = _HASH_PATTERN)
    workspaceRevision: int = Field(ge = 0)
    revision: int = Field(ge = 0)


class RevokeProjectHooksRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    revision: int = Field(ge = 0)


class SetProjectHookHandlerRequest(TrustProjectHooksRequest):
    enabled: bool


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _workspace_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code = 409, detail = str(exc))


def _require_ui_session(via_api_key: bool) -> None:
    if via_api_key:
        raise HTTPException(
            status_code = 403,
            detail = "Project hook authority can only be changed from the Unsloth UI.",
        )


def _workspace_identity(workspace: Any) -> tuple[int, int]:
    return int(workspace.device_id), int(workspace.file_id)


def _discover_workspace(workspace: Any) -> dict[str, Any]:
    return discover_project_hooks(
        workspace.root,
        expected_identity = _workspace_identity(workspace),
    )


def _current_handler_ids(config: dict[str, Any]) -> set[str]:
    return {
        handler["id"]
        for groups in config["hooks"].values()
        for group in groups
        for handler in group["hooks"]
    }


def _trust_snapshot_state(
    project_id: str, config: dict[str, Any], workspace: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _attempt in range(3):
        state = get_project_hook_trust_state(
            project_id,
            config.get("contentHash"),
            workspace_identity = _workspace_identity(workspace),
            workspace_revision = int(workspace.revision),
        )
        record = get_project_hook_trust_record(project_id)
        if state["revision"] == record["revision"]:
            return state, record
    raise ProjectHookTrustStateError("Hook trust changed while its snapshot was being read.")


def _stored_trust(record: dict[str, Any]) -> dict[str, Any] | None:
    if not record["hasStoredTrust"]:
        return None
    return {
        "contentHash": record["storedContentHash"],
        "revision": record["revision"],
    }


def _snapshot(project_id: str, config: dict[str, Any], workspace: Any) -> dict[str, Any]:
    state, record = _trust_snapshot_state(project_id, config, workspace)
    disabled = set(state["disabledHandlerIds"])
    try:
        execution = execution_status()
    except Exception:
        execution = {
            "available": False,
            "reason": "Hook execution capability could not be checked.",
        }
    rendered_hooks = {
        event: [
            {
                "matcher": group["matcher"],
                "hooks": [
                    {
                        **handler,
                        "enabled": handler["id"] not in disabled,
                        "active": (
                            state["trusted"]
                            and handler["id"] not in disabled
                            and handler_is_supported(event, handler)
                            and execution["available"]
                        ),
                    }
                    for handler in group["hooks"]
                ],
            }
            for group in groups
        ]
        for event, groups in config["hooks"].items()
    }
    return {
        key: config[key]
        for key in (
            "sourcePath",
            "exists",
            "contentHash",
            "description",
            "groupCount",
            "handlerCount",
        )
    } | {
        "execution": execution,
        "workspaceAvailable": True,
        "workspaceRevision": int(workspace.revision),
        "trusted": state["trusted"],
        "revision": state["revision"],
        "storedTrust": _stored_trust(record),
        "hooks": rendered_hooks,
    }


def _unavailable_snapshot(record: dict[str, Any], *, workspace_revision: int) -> dict[str, Any]:
    """Return no current-source claims when a workspace cannot be inspected."""
    return {
        "workspaceAvailable": False,
        "workspaceRevision": workspace_revision,
        "sourcePath": ".codex/hooks.json",
        "exists": False,
        "contentHash": None,
        "description": None,
        "groupCount": 0,
        "handlerCount": 0,
        "trusted": False,
        "revision": record["revision"],
        "storedTrust": _stored_trust(record),
        "hooks": {event: [] for event in HOOK_EVENTS},
    }


def _revoked_record(project_id: str, revision: int) -> dict[str, Any]:
    return {
        "projectId": project_id,
        "storedContentHash": None,
        "hasStoredTrust": False,
        "revision": revision,
    }


def _require_current_hash(config: dict[str, Any], requested_hash: str) -> None:
    if not config.get("exists") or config.get("contentHash") != requested_hash:
        raise HTTPException(
            status_code = 409,
            detail = "Project hooks changed. Refresh and review the current file.",
        )


def _require_workspace_revision(workspace: Any, requested_revision: int) -> None:
    if int(workspace.revision) != requested_revision:
        raise HTTPException(
            status_code = 409,
            detail = "Project workspace changed. Refresh and review the current hooks file.",
        )


@router.get("/projects/{project_id}/hooks")
def project_hooks(project_id: str, _current_subject: str = Depends(get_current_subject)):
    project = _project(project_id)
    if project.get("workspaceAvailable") is False:
        try:
            return _unavailable_snapshot(
                get_project_hook_trust_record(project_id),
                workspace_revision = int(project.get("workspaceRevision") or 0),
            )
        except (ProjectHookTrustStateError, ValueError) as exc:
            raise _workspace_error(exc) from exc
    try:
        with project_workspace_access(project_id) as workspace:
            try:
                config = _discover_workspace(workspace)
                return _snapshot(project_id, config, workspace)
            except (AgentWorkspaceError, ProjectHookTrustStateError, ValueError) as exc:
                raise _workspace_error(exc) from exc
    except HTTPException:
        raise
    except AgentWorkspaceError:
        try:
            return _unavailable_snapshot(
                get_project_hook_trust_record(project_id),
                workspace_revision = int(project.get("workspaceRevision") or 0),
            )
        except (ProjectHookTrustStateError, ValueError) as exc:
            raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/hooks/trust")
def project_hooks_trust_state(
    project_id: str, _current_subject: str = Depends(get_current_subject)
):
    """Read persisted trust without opening or interpreting the project workspace."""
    _project(project_id)
    try:
        record = get_project_hook_trust_record(project_id)
        return {
            "storedTrust": _stored_trust(record),
            "revision": record["revision"],
        }
    except (ProjectHookTrustStateError, ValueError) as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/hooks/trust")
def project_hooks_trust(
    project_id: str,
    request: TrustProjectHooksRequest,
    via_api_key: Annotated[bool, Depends(authenticated_via_api_key)] = False,
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    _project(project_id)
    try:
        with project_workspace_access(project_id) as workspace:
            _require_workspace_revision(workspace, request.workspaceRevision)
            config = _discover_workspace(workspace)
            _require_current_hash(config, request.contentHash)
            trust_project_hooks(
                project_id,
                config["contentHash"],
                workspace_identity = _workspace_identity(workspace),
                workspace_revision = int(workspace.revision),
                expected_revision = request.revision,
            )
            return _snapshot(project_id, _discover_workspace(workspace), workspace)
    except HTTPException:
        raise
    except (
        AgentWorkspaceError,
        ProjectHookTrustConflictError,
        ProjectHookTrustStateError,
        ValueError,
    ) as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/hooks/revoke")
def project_hooks_revoke(
    project_id: str,
    request: RevokeProjectHooksRequest,
    via_api_key: Annotated[bool, Depends(authenticated_via_api_key)] = False,
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    project = _project(project_id)
    try:
        revoked = revoke_project_hook_trust(project_id, expected_revision = request.revision)
    except (
        ProjectHookTrustConflictError,
        ProjectHookTrustStateError,
        ValueError,
    ) as exc:
        raise _workspace_error(exc) from exc
    if project.get("workspaceAvailable") is False:
        return _unavailable_snapshot(
            _revoked_record(project_id, revoked["revision"]),
            workspace_revision = int(project.get("workspaceRevision") or 0),
        )
    try:
        with project_workspace_access(project_id) as workspace:
            return _snapshot(project_id, _discover_workspace(workspace), workspace)
    except (AgentWorkspaceError, ProjectHookTrustStateError, ValueError):
        return _unavailable_snapshot(
            _revoked_record(project_id, revoked["revision"]),
            workspace_revision = int(project.get("workspaceRevision") or 0),
        )


@router.post("/projects/{project_id}/hooks/handlers/{handler_id}")
def project_hook_handler(
    project_id: str,
    handler_id: str,
    request: SetProjectHookHandlerRequest,
    via_api_key: Annotated[bool, Depends(authenticated_via_api_key)] = False,
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    _project(project_id)
    try:
        with project_workspace_access(project_id) as workspace:
            _require_workspace_revision(workspace, request.workspaceRevision)
            config = _discover_workspace(workspace)
            _require_current_hash(config, request.contentHash)
            if handler_id not in _current_handler_ids(config):
                raise HTTPException(status_code = 404, detail = "Project hook handler not found.")
            set_project_hook_handler_enabled(
                project_id,
                config["contentHash"],
                handler_id,
                workspace_identity = _workspace_identity(workspace),
                workspace_revision = int(workspace.revision),
                enabled = request.enabled,
                expected_revision = request.revision,
            )
            return _snapshot(project_id, _discover_workspace(workspace), workspace)
    except HTTPException:
        raise
    except (
        AgentWorkspaceError,
        ProjectHookTrustConflictError,
        ProjectHookTrustStateError,
        ValueError,
    ) as exc:
        raise _workspace_error(exc) from exc


__all__ = ["router"]

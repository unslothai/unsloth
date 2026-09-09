# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated controls for project verification profiles and runs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace import verification_context as common, verification, verification_state
from core.agent_workspace.verification_context import AgentWorkspaceError
from storage.studio_db import get_chat_project


router = APIRouter()


class VerificationCheckRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid", strict = True)

    name: Annotated[str, Field(min_length = 1, max_length = 120)]
    kind: Annotated[str, Field(min_length = 1, max_length = 64)] = "custom"
    command: Annotated[str, Field(min_length = 1, max_length = 16 * 1024)]
    required: bool = True
    timeoutSeconds: Annotated[int, Field(ge = 1, le = 3600)] = 300
    logLimitBytes: Annotated[int, Field(ge = 1024, le = 2 * 1024 * 1024)] = 256 * 1024


class SaveVerificationRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid", strict = True)

    checks: Annotated[list[VerificationCheckRequest], Field(max_length = 32)]
    expectedRevision: Annotated[int, Field(ge = 0)]
    workspaceRevision: Annotated[int, Field(ge = 0)]


class StartVerificationRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid", strict = True)

    configRevision: Annotated[int, Field(ge = 1)]
    workspaceRevision: Annotated[int, Field(ge = 0)]


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _require_ui_session(via_api_key: bool) -> None:
    if via_api_key:
        raise HTTPException(
            status_code = 403,
            detail = "Project verification can only be changed or run from the Unsloth UI.",
        )


def _workspace_error(exc: BaseException) -> HTTPException:
    return HTTPException(status_code = 409, detail = str(exc))


def _public_run(run: dict | None) -> dict | None:
    if run is None:
        return None
    public_fields = (
        "id",
        "projectId",
        "status",
        "configRevision",
        "workspaceRevision",
        "evidenceRevision",
        "checks",
        "results",
        "cancelRequested",
        "error",
        "startedAt",
        "updatedAt",
        "completedAt",
        "historySequence",
    )
    return {key: run[key] for key in public_fields if key in run} | {
        "sourceFreshness": "unverified",
    }


def _public_run_summary(run: dict) -> dict:
    public_fields = (
        "id",
        "projectId",
        "status",
        "evidenceStatus",
        "configRevision",
        "workspaceRevision",
        "evidenceRevision",
        "cancelRequested",
        "error",
        "startedAt",
        "updatedAt",
        "completedAt",
        "historySequence",
    )
    return {key: run[key] for key in public_fields if key in run} | {
        "sourceFreshness": "unverified",
    }


def _execution_status() -> dict:
    try:
        return verification.execution_status()
    except Exception:  # noqa: BLE001 - capability reads fail closed
        return {
            "available": False,
            "backend": None,
            "reason": "Secure supervised project commands are unavailable.",
        }


def _config_snapshot(project_id: str, project: dict) -> dict[str, Any]:
    profile = verification_state.get_verification_config(project_id)
    workspace_available = project.get("workspaceAvailable") is not False
    workspace_revision = int(project.get("workspaceRevision") or 0)
    workspace_bound = False
    if workspace_available:
        try:
            with common.project_workspace_access(project_id) as workspace:
                workspace_revision = int(workspace.revision)
                workspace_bound = (
                    profile.get("workspaceDeviceId") == int(workspace.device_id)
                    and profile.get("workspaceFileId") == int(workspace.file_id)
                    and profile.get("workspaceRevision") == workspace_revision
                )
        except AgentWorkspaceError:
            workspace_available = False
    active_run = verification_state.active_verification_run(project_id)
    return {
        "projectId": project_id,
        "workspaceAvailable": workspace_available,
        "workspaceRevision": workspace_revision,
        "active": workspace_bound,
        "activeRun": _public_run(active_run),
        "checks": profile["checks"],
        "revision": profile["revision"],
        "updatedAt": profile["updatedAt"],
        "sourceFreshness": "unverified",
        "execution": _execution_status(),
    }


@router.get("/projects/{project_id}/verification")
def project_verification_config(
    project_id: str, _current_subject: str = Depends(get_current_subject)
):
    project = _project(project_id)
    try:
        return _config_snapshot(project_id, project)
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc


@router.put("/projects/{project_id}/verification")
def save_project_verification_config(
    project_id: str,
    payload: SaveVerificationRequest,
    via_api_key: bool = Depends(authenticated_via_api_key),
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    project = _project(project_id)
    try:
        checks = verification_state.normalize_verification_checks(
            [check.model_dump() for check in payload.checks]
        )
    except (AgentWorkspaceError, ValueError) as exc:
        raise HTTPException(status_code = 422, detail = str(exc)) from exc
    try:
        with common.project_workspace_access(project_id) as workspace:
            if int(workspace.revision) != payload.workspaceRevision:
                raise AgentWorkspaceError(
                    "Project workspace changed. Refresh verification settings and retry."
                )
            verification_state.set_verification_config(
                project_id,
                checks,
                workspace_identity = (int(workspace.device_id), int(workspace.file_id)),
                workspace_revision = int(workspace.revision),
                expected_revision = payload.expectedRevision,
            )
        return _config_snapshot(project_id, project)
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc


@router.post("/projects/{project_id}/verifications")
def start_project_verification_run(
    project_id: str,
    payload: StartVerificationRequest,
    via_api_key: bool = Depends(authenticated_via_api_key),
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    _project(project_id)
    try:
        return _public_run(
            verification.start_project_verification(
                project_id,
                config_revision = payload.configRevision,
                workspace_revision = payload.workspaceRevision,
            )
        )
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/verifications")
def project_verification_runs(
    project_id: str,
    limit: Annotated[int, Query(ge = 1, le = 100)] = 20,
    _current_subject: str = Depends(get_current_subject),
):
    _project(project_id)
    try:
        return {
            "runs": [
                _public_run_summary(run)
                for run in verification_state.list_verification_run_summaries(
                    project_id,
                    limit = limit,
                )
            ]
        }
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc


@router.get("/projects/{project_id}/verifications/{run_id}")
def project_verification_run(
    project_id: str,
    run_id: str,
    afterEvidenceRevision: Annotated[int | None, Query(ge = 1)] = None,
    _current_subject: str = Depends(get_current_subject),
):
    _project(project_id)
    try:
        if afterEvidenceRevision is not None:
            marker = verification_state.get_verification_run_evidence_revision(
                project_id,
                run_id,
            )
            if marker is None:
                raise HTTPException(status_code = 404, detail = "Verification run not found.")
            if marker == afterEvidenceRevision:
                return Response(status_code = 204)
            if marker < afterEvidenceRevision:
                raise HTTPException(
                    status_code = 409,
                    detail = "Verification evidence revision regressed. Refresh the run.",
                )
        run = verification_state.get_verification_run(project_id, run_id)
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc
    if run is None:
        raise HTTPException(status_code = 404, detail = "Verification run not found.")
    return _public_run(run)


@router.post("/projects/{project_id}/verifications/{run_id}/cancel")
def cancel_project_verification_run(
    project_id: str,
    run_id: str,
    via_api_key: bool = Depends(authenticated_via_api_key),
    _current_subject: str = Depends(get_current_subject),
):
    _require_ui_session(via_api_key)
    _project(project_id)
    try:
        run, accepted = verification.cancel_verification(project_id, run_id)
    except (AgentWorkspaceError, ValueError) as exc:
        raise _workspace_error(exc) from exc
    return {"cancelRequested": accepted, "run": _public_run(run)}


__all__ = ["router"]

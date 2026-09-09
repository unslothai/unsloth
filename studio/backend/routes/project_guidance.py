# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated project instructions, skills, and `/init` endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from auth.authentication import get_current_subject
from core.agent_workspace.common import AgentWorkspaceError, project_workspace_access
from core.agent_workspace.initialization import initialize_project_agents
from core.agent_workspace.instructions import resolve_agents_instructions
from core.agent_workspace.skills import discover_project_skills
from storage.studio_db import get_chat_project


router = APIRouter()


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _workspace_http_error(exc: AgentWorkspaceError) -> HTTPException:
    return HTTPException(status_code = 409, detail = str(exc))


@router.get("/projects/{project_id}/instructions")
def project_instructions(
    project_id: str,
    target: Optional[str] = Query(default = None, max_length = 4096),
    _current_subject: str = Depends(get_current_subject),
):
    _project(project_id)
    try:
        with project_workspace_access(project_id) as workspace:
            return resolve_agents_instructions(
                workspace.root,
                target,
                expected_identity = (workspace.device_id, workspace.file_id),
            )
    except AgentWorkspaceError as exc:
        raise _workspace_http_error(exc) from exc


@router.get("/projects/{project_id}/skills")
def project_skills(project_id: str, _current_subject: str = Depends(get_current_subject)):
    _project(project_id)
    try:
        with project_workspace_access(project_id) as workspace:
            result = discover_project_skills(
                workspace.root,
                expected_identity = (workspace.device_id, workspace.file_id),
            )
        return {
            **result,
            "skills": [
                {
                    key: skill[key]
                    for key in (
                        "name",
                        "description",
                        "path",
                        "size",
                        "modifiedNs",
                        "ambiguous",
                    )
                    if key in skill
                }
                for skill in result["skills"]
            ],
        }
    except AgentWorkspaceError as exc:
        raise _workspace_http_error(exc) from exc


@router.post("/projects/{project_id}/init")
def project_init(project_id: str, _current_subject: str = Depends(get_current_subject)):
    _project(project_id)
    try:
        return initialize_project_agents(project_id)
    except AgentWorkspaceError as exc:
        raise _workspace_http_error(exc) from exc


__all__ = ["router"]

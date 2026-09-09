# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated, read-only Git review routes for primary project workspaces."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from auth.authentication import get_current_subject
from core.agent_workspace.git_context import AgentWorkspaceError
from core.agent_workspace.git_review import (
    DEFAULT_MAX_BYTES,
    MAX_MAX_BYTES,
    build_diff_manifest,
    git_status,
)
from storage.studio_db import get_chat_project


router = APIRouter()


def _project(project_id: str) -> dict:
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise HTTPException(status_code = 404, detail = "Project not found.")
    return project


def _require_workspace_revision(project: dict, requested: int) -> None:
    if int(project.get("workspaceRevision") or 0) != requested:
        raise HTTPException(
            status_code = 409,
            detail = "Project workspace changed. Refresh Git review.",
        )


def _require_response_revision(response: dict, requested: int) -> dict:
    if int(response.get("workspaceRevision", -1)) != requested:
        raise HTTPException(
            status_code = 409,
            detail = "Project workspace changed during Git review. Refresh it.",
        )
    _require_workspace_revision(_project(response["projectId"]), requested)
    return response


@router.get("/projects/{project_id}/git/status")
def project_git_status(
    project_id: str,
    workspace_revision: Annotated[int, Query(alias = "workspaceRevision", ge = 0)],
    _current_subject: str = Depends(get_current_subject),
):
    project = _project(project_id)
    _require_workspace_revision(project, workspace_revision)
    try:
        return _require_response_revision(git_status(project_id), workspace_revision)
    except HTTPException:
        raise
    except (AgentWorkspaceError, OverflowError, ValueError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.get("/projects/{project_id}/git/diff")
def project_git_diff(
    project_id: str,
    workspace_revision: Annotated[int, Query(alias = "workspaceRevision", ge = 0)],
    mode: Literal["head", "staged", "unstaged"] = "head",
    max_bytes: Annotated[
        int,
        Query(alias = "maxBytes", ge = 4_096, le = MAX_MAX_BYTES),
    ] = DEFAULT_MAX_BYTES,
    _current_subject: str = Depends(get_current_subject),
):
    project = _project(project_id)
    _require_workspace_revision(project, workspace_revision)
    try:
        return _require_response_revision(
            build_diff_manifest(
                project_id,
                mode = mode,
                max_bytes = max_bytes,
            ),
            workspace_revision,
        )
    except HTTPException:
        raise
    except (AgentWorkspaceError, OverflowError, ValueError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


__all__ = ["router"]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated, bounded project task API; no caller-supplied execution authority."""

import asyncio
import importlib
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from auth.authentication import get_current_subject
from storage.studio_db import get_chat_project


def service():
    try:
        module = importlib.import_module("core.agent_workspace.task_service")
        try:
            module.require_prerequisites()
        except module.state.TaskStateError:
            raise HTTPException(
                503,
                "Project task prerequisite versions are incompatible. Update task support first.",
            ) from None
        return module
    except ImportError:
        raise HTTPException(
            503, "Install project task, Git, secure edit, and lifecycle support first."
        ) from None


@asynccontextmanager
async def lifespan(_app):
    yield
    try:
        module = service()
    except HTTPException:
        return
    if not await asyncio.to_thread(module.shutdown):
        logging.getLogger(__name__).error("Project tasks have not drained during shutdown")


router = APIRouter(
    prefix = "/api/agent/projects/{project_id}/tasks",
    dependencies = [Depends(get_current_subject)],
    lifespan = lifespan,
)


class SubmitTask(BaseModel):
    model_config = ConfigDict(extra = "forbid", strict = True)
    instruction: str = Field(min_length = 1, max_length = 16000)
    kind: Literal["local", "provider"]
    model: str = Field(min_length = 1, max_length = 512)
    providerId: str | None = Field(default = None, min_length = 1, max_length = 256)
    maxOutputTokens: int = Field(default = 8192, ge = 1024, le = 32768)
    childLimit: int = Field(default = 2, ge = 0, le = 8)
    childBudget: int = Field(default = 8192, ge = 0, le = 131072)
    timeout: int = Field(default = 900, ge = 1, le = 3600)
    allowCommands: bool = False


def project(project_id: str):
    if get_chat_project(project_id) is None:
        raise HTTPException(404, "Project not found.")


def invoke(call):
    module = service()
    try:
        return call(module)
    except HTTPException:
        raise
    except module.state.TaskStateError as exc:
        raise HTTPException(409, str(exc)) from None
    except Exception:
        # Runtime and Git errors may contain provider URLs, paths or secrets.
        raise HTTPException(
            409,
            "The project task operation could not complete. Refresh and check the selected runtime and project.",
        ) from None


@router.get("")
def list_tasks(project_id: str, limit: int = Query(default = 100, ge = 1, le = 100)):
    project(project_id)
    return invoke(lambda m: m.public_tasks(project_id, limit = limit))


@router.post("", status_code = 202)
def submit_task(project_id: str, request: SubmitTask):
    project(project_id)
    return invoke(
        lambda m: m.submit(
            project_id,
            request.instruction,
            kind = request.kind,
            model = request.model,
            provider_id = request.providerId,
            max_output_tokens = request.maxOutputTokens,
            child_limit = request.childLimit,
            child_budget = request.childBudget,
            timeout = request.timeout,
            allow_commands = request.allowCommands,
        )
    )


def command_module():
    try:
        return importlib.import_module("core.agent_workspace.task_commands")
    except ImportError:
        raise HTTPException(503, "Task command support is not installed.") from None


@router.get("/capabilities")
def task_capabilities(project_id: str):
    project(project_id)
    try:
        return {"commands": command_module().availability()}
    except HTTPException:
        return {
            "commands": {"available": False, "reason": "Task command support is not installed."}
        }


@router.get("/{task_id}/commands")
def task_commands(project_id: str, task_id: str):
    project(project_id)
    return invoke(
        lambda m: command_module().list_commands(
            project_id, m.state.get_task(project_id, task_id)["id"]
        )
    )


@router.get("/{task_id}/commands/{command_id}")
def task_command_detail(project_id: str, task_id: str, command_id: str):
    project(project_id)
    return invoke(
        lambda m: command_module().get_command(
            project_id, m.state.get_task(project_id, task_id)["id"], command_id
        )
    )


@router.get("/{task_id}")
def get_task(project_id: str, task_id: str):
    project(project_id)
    return invoke(lambda m: m.public_task(m.state.get_task(project_id, task_id)))


@router.get("/{task_id}/events")
def events(
    project_id: str,
    task_id: str,
    after: int = Query(default = 0, ge = 0),
):
    project(project_id)
    return invoke(lambda m: m.state.task_events(project_id, task_id, after = after))


@router.get("/{task_id}/review")
def review_task(project_id: str, task_id: str):
    project(project_id)

    def read(module):
        module.state.get_task(project_id, task_id)
        from core.agent_workspace.task_workspaces import review_task_workspace
        return review_task_workspace(project_id, task_id)

    return invoke(read)


@router.post("/{task_id}/cancel")
def cancel_task(project_id: str, task_id: str):
    project(project_id)
    return invoke(lambda m: m.cancel(project_id, task_id))


@router.post("/{task_id}/retry", status_code = 202)
def retry_task(project_id: str, task_id: str):
    project(project_id)
    return invoke(lambda m: m.retry(project_id, task_id))

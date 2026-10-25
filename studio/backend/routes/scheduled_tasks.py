# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled tasks: a prompt that runs on a schedule and posts the answer to a new chat."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.scheduled_tasks.schedule import InvalidSchedule, Schedule
from core.scheduled_tasks.scheduler import next_run_for
from storage import scheduled_tasks_db as db

router = APIRouter()

_NOT_FOUND = "Scheduled task not found"


def _not_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


Title = Annotated[str, Field(max_length = 120), AfterValidator(_not_blank)]
Prompt = Annotated[str, Field(max_length = 20_000), AfterValidator(_not_blank)]
TimeZoneName = Annotated[str, Field(max_length = 64)]
# Which finished runs get a toast and system notification; every run's chat still shows unread in Recents.
Notify = Literal["all", "failures", "none"]
TaskModel = Annotated[str, Field(max_length = 512), AfterValidator(_not_blank)]
# lowest/highest are relative and resolved per run (local levels are known only once loaded); null = model default.
ReasoningEffort = Literal[
    "off", "lowest", "highest", "minimal", "low", "medium", "high", "xhigh", "max"
]
ProjectId = Annotated[str, Field(max_length = 128), AfterValidator(_not_blank)]
# Fields an update may set back to null, unlike the rest, where null means unchanged.
_CLEARABLE = ("model", "reasoningEffort", "projectId")


class ScheduledTaskCreate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: Title
    prompt: Prompt
    schedule: Schedule
    timezone: TimeZoneName
    enabled: bool = True
    notify: Notify = "all"
    named: bool = True
    model: Optional[TaskModel] = None
    reasoningEffort: Optional[ReasoningEffort] = None
    allowModelLoad: bool = False
    archiveSuccessful: bool = False
    projectId: Optional[ProjectId] = None


class ScheduledTaskUpdate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    title: Optional[Title] = None
    prompt: Optional[Prompt] = None
    schedule: Optional[Schedule] = None
    timezone: Optional[TimeZoneName] = None
    enabled: Optional[bool] = None
    notify: Optional[Notify] = None
    named: Optional[bool] = None
    model: Optional[TaskModel] = None
    reasoningEffort: Optional[ReasoningEffort] = None
    allowModelLoad: Optional[bool] = None
    archiveSuccessful: Optional[bool] = None
    projectId: Optional[ProjectId] = None


class RunNowRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    loadModel: bool = False


class ScheduledTaskRun(BaseModel):
    id: str
    taskId: str
    source: Literal["schedule", "manual"]
    status: Literal["queued", "running", "succeeded", "failed"]
    scheduledFor: Optional[int] = None
    createdAt: int
    startedAt: Optional[int] = None
    finishedAt: Optional[int] = None
    threadId: Optional[str] = None
    model: Optional[str] = None
    # no_model, model_not_loaded, model_load_failed, connection_unavailable, timeout, cancelled, interrupted or empty.
    errorCode: Optional[str] = None
    error: Optional[str] = None


class FinishedScheduledTaskRun(ScheduledTaskRun):
    taskTitle: str
    taskNotify: Notify
    taskModel: Optional[str] = None


class ScheduledTask(BaseModel):
    id: str
    title: str
    prompt: str
    schedule: Schedule
    timezone: str
    enabled: bool
    notify: Notify
    named: bool
    model: Optional[str] = None
    reasoningEffort: Optional[ReasoningEffort] = None
    allowModelLoad: bool = False
    archiveSuccessful: bool = False
    projectId: Optional[str] = None
    status: Literal["active", "paused", "completed"]
    nextRunAt: Optional[int] = None
    createdAt: int
    updatedAt: int
    lastRun: Optional[ScheduledTaskRun] = None


class ScheduledTaskList(BaseModel):
    tasks: list[ScheduledTask]
    # Server clock (nextRunAt and run times use it); clients time refreshes from it.
    now: int


class ScheduledTaskActivity(BaseModel):
    now: int
    runs: list[FinishedScheduledTaskRun]
    more: bool = False
    activeRuns: int
    nextRunAt: Optional[int] = None


def _response(task: dict) -> dict:
    status = (
        "paused" if not task["enabled"] else "completed" if task["nextRunAt"] is None else "active"
    )
    return {**task, "status": status}


def _existing_project(project_id: Optional[str]) -> Optional[str]:
    from storage import studio_db
    if project_id is not None and studio_db.get_chat_project(project_id) is None:
        raise HTTPException(status_code = 400, detail = "Project not found")
    return project_id


def _next_run(task: dict, now: int) -> Optional[int]:
    try:
        following = next_run_for(task, now)
    except InvalidSchedule as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    if following is None and task["enabled"]:
        raise HTTPException(status_code = 400, detail = "Pick a time in the future.")
    return following if task["enabled"] else None


def _require_ui_session(via_api_key: bool) -> None:
    """Tasks run as the account and may load models or spend saved keys: signed-in users only, never API keys."""
    if via_api_key:
        raise HTTPException(
            status_code = 403,
            detail = "Scheduled tasks can only be changed from the Unsloth app, not with an API key.",
        )


def _wake_scheduler(request: Request) -> None:
    scheduler = getattr(request.app.state, "scheduled_task_scheduler", None)
    if scheduler is not None:
        scheduler.wake()


@router.get("", response_model = ScheduledTaskList)
def list_scheduled_tasks(current_subject: str = Depends(get_current_subject)):
    return {"tasks": [_response(task) for task in db.list_tasks()], "now": db.now_ms()}


@router.post("", status_code = 201, response_model = ScheduledTask)
def create_scheduled_task(
    payload: ScheduledTaskCreate,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_ui_session(via_api_key)
    now = db.now_ms()
    task = {
        "id": str(uuid.uuid4()),
        "ownerSubject": current_subject,
        "title": payload.title,
        "prompt": payload.prompt,
        "schedule": payload.schedule.model_dump(),
        "timezone": payload.timezone,
        "enabled": payload.enabled,
        "notify": payload.notify,
        "named": payload.named,
        "model": payload.model,
        "reasoningEffort": payload.reasoningEffort,
        "allowModelLoad": payload.allowModelLoad,
        "archiveSuccessful": payload.archiveSuccessful,
        "projectId": _existing_project(payload.projectId),
        "createdAt": now,
        "updatedAt": now,
    }
    task["nextRunAt"] = _next_run(task, now)
    try:
        created = db.create_task(task)
    except db.TaskLimitReached:
        raise HTTPException(
            status_code = 409,
            detail = f"You can have up to {db.MAX_TASKS} scheduled tasks. Delete one to add another.",
        ) from None
    _wake_scheduler(request)
    return _response(created)


# Declared ahead of /{task_id} so "runs" is never read as a task id.
@router.get("/runs", response_model = ScheduledTaskActivity)
def scheduled_task_activity(
    since: Optional[int] = Query(default = None, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    """Runs finished since ``since`` (ms), for a client that notifies when results arrive."""
    return {"now": db.now_ms(), **db.activity(since)}


@router.get("/{task_id}", response_model = ScheduledTask)
def get_scheduled_task(task_id: str, current_subject: str = Depends(get_current_subject)):
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return _response(task)


@router.patch("/{task_id}", response_model = ScheduledTask)
def update_scheduled_task(
    task_id: str,
    payload: ScheduledTaskUpdate,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_ui_session(via_api_key)
    changes = payload.model_dump(exclude_unset = True, exclude_none = True)
    changes.update(
        {key: getattr(payload, key) for key in _CLEARABLE if key in payload.model_fields_set}
    )
    if changes.get("projectId") is not None:
        _existing_project(changes["projectId"])
    now = db.now_ms()
    # Only on schedule, time zone or enabled changes: a rename or resent value keeps an overdue run due.
    updated = db.update_task(task_id, changes, lambda task: _next_run(task, now))
    if updated is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    _wake_scheduler(request)
    return _response(updated)


@router.delete("/{task_id}", status_code = 204, response_class = Response)
def delete_scheduled_task(
    task_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_ui_session(via_api_key)
    if not db.delete_task(task_id):
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    scheduler = getattr(request.app.state, "scheduled_task_scheduler", None)
    if scheduler is not None:
        scheduler.cancel_task_run(task_id)
    return Response(status_code = 204)


@router.post("/{task_id}/run", status_code = 202, response_model = ScheduledTaskRun)
def run_scheduled_task_now(
    task_id: str,
    request: Request,
    payload: Optional[RunNowRequest] = None,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Queue a run outside the schedule; the next scheduled run is unaffected."""
    _require_ui_session(via_api_key)
    try:
        run = db.request_run(task_id, allow_model_load = payload is not None and payload.loadModel)
    except db.TaskRunActive:
        raise HTTPException(status_code = 409, detail = "This task is already running.") from None
    if run is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    _wake_scheduler(request)
    return run

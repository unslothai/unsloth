# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled training starts: a saved training request that starts at a chosen local time."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training.schedules import InvalidStartTime, request_payload, resolve_start_time
from models.training import TrainingStartRequest
from storage import training_schedules_db as db

router = APIRouter()

_NOT_FOUND = "Scheduled training run not found"
_ALREADY_SCHEDULED = "A training run is already scheduled. Cancel it before scheduling another."

TrainingScheduleStatus = Literal["pending", "starting", "started", "cancelled", "failed", "missed"]


class TrainingScheduleCreate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    # Wall clock in ``timezone``, as the date and time inputs give it.
    start_at: str = Field(..., max_length = 16)
    timezone: str = Field(..., min_length = 1, max_length = 64)
    request: TrainingStartRequest


class TrainingScheduleResponse(BaseModel):
    id: str
    status: TrainingScheduleStatus
    startAt: int
    startAtLocal: str
    timezone: str
    modelName: str
    projectName: Optional[str] = None
    createdAt: int
    updatedAt: int
    jobId: Optional[str] = None
    error: Optional[str] = None


class TrainingScheduleList(BaseModel):
    schedules: list[TrainingScheduleResponse]
    # Server clock, so clients time the next start from the same clock the scheduler uses.
    now: int


def _require_ui_session(via_api_key: bool) -> None:
    """A start may spend saved keys and load models: signed-in users only, never API keys."""
    if via_api_key:
        raise HTTPException(
            status_code = 403,
            detail = "Training schedules can only be changed from the Unsloth app, not with an API key.",
        )


def _wake_scheduler(request: Request) -> None:
    scheduler = getattr(request.app.state, "training_schedule_scheduler", None)
    if scheduler is not None:
        scheduler.wake()


@router.get("", response_model = TrainingScheduleList)
def list_training_schedules(current_subject: str = Depends(get_current_subject)):
    return {"schedules": db.list_schedules(), "now": db.now_ms()}


@router.post("", status_code = 201, response_model = TrainingScheduleResponse)
async def create_training_schedule(
    payload: TrainingScheduleCreate,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_ui_session(via_api_key)
    now = db.now_ms()
    try:
        start_at = resolve_start_time(payload.start_at, payload.timezone, now)
    except InvalidStartTime as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    # Imported here: routes.training is the module that owns admission and it imports this router's package.
    from routes.training import _admit_training_request, _prepare_training_request

    start = payload.request.model_copy(deep = True)
    # A scheduled start is its own attempt; a stale client id would replay an earlier outcome.
    start.start_request_id = None
    try:
        # The preview may keep running: nothing starts now. Admission still runs, so an untrainable
        # configuration is refused when it is scheduled rather than a day later.
        await _prepare_training_request(start, via_api_key, stop_token_preview = False)
        await _admit_training_request(start, current_subject, via_api_key)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    schedule = {
        "id": str(uuid.uuid4()),
        "ownerSubject": current_subject,
        "status": db.PENDING,
        "startAt": start_at,
        "startAtLocal": payload.start_at,
        "timezone": payload.timezone,
        "modelName": start.model_name,
        "projectName": start.project_name,
        "createdAt": now,
        "updatedAt": now,
    }
    try:
        created = db.create_schedule(schedule, request_payload(start))
    except db.TrainingScheduleExists:
        raise HTTPException(status_code = 409, detail = _ALREADY_SCHEDULED) from None
    _wake_scheduler(request)
    return created


@router.delete("/{schedule_id}", status_code = 204)
def cancel_training_schedule(
    request: Request,
    schedule_id: str = ApiPath(..., min_length = 1, max_length = 128),
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_ui_session(via_api_key)
    schedule, cancelled = db.cancel_pending(schedule_id)
    if schedule is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    if not cancelled:
        raise HTTPException(
            status_code = 409,
            detail = "This scheduled training run has already started or ended.",
        )
    _wake_scheduler(request)
    return Response(status_code = 204)

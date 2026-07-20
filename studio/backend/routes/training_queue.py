# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training queue API routes."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training.queue import get_training_queue_manager
from loggers import get_logger
from models import (
    TrainingQueueItem,
    TrainingQueueMoveRequest,
    TrainingQueueStateResponse,
    TrainingStartRequest,
)
from utils.utils import log_and_http_error

router = APIRouter()
logger = get_logger(__name__)


# Only the display-safe project name leaves the stored request; the rest of
# the payload may carry credentials (hf_token) and must never leave the backend.
def _project_name(request_json) -> Optional[str]:
    try:
        data = json.loads(request_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("project_name")
    return value if isinstance(value, str) and value.strip() else None


# TrainingQueueItem has no request payload field: the stored request may carry
# credentials (hf_token) and must never leave the backend.
def _item_model(item: dict) -> TrainingQueueItem:
    return TrainingQueueItem(
        id = item["id"],
        position = item["position"],
        status = item["status"],
        model_name = item["model_name"],
        dataset_summary = item["dataset_summary"],
        project_name = _project_name(item.get("request_json")),
        job_id = item.get("job_id"),
        result_status = item.get("result_status"),
        error_message = item.get("error_message"),
        created_at = item["created_at"],
        started_at = item.get("started_at"),
        finished_at = item.get("finished_at"),
    )


def _state_response() -> TrainingQueueStateResponse:
    state = get_training_queue_manager().state()
    return TrainingQueueStateResponse(
        paused = state["paused"],
        paused_reason = state["paused_reason"],
        pending_count = state["pending_count"],
        max_pending = state["max_pending"],
        active_job_id = state["active_job_id"],
        items = [_item_model(item) for item in state["items"]],
    )


@router.get("/queue", response_model = TrainingQueueStateResponse)
async def get_queue_state(current_subject: str = Depends(get_current_subject)):
    try:
        return _state_response()
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to read training queue",
            event = "training_queue.state_failed",
            log = logger,
        )


@router.post("/queue/items", response_model = TrainingQueueItem, status_code = 201)
async def enqueue_item(
    request: TrainingStartRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    # Same guard as POST /start: an idle backend launches a queued job
    # immediately, and the training VRAM hook would unload the chat model out
    # from under an in-flight API inference stream.
    if via_api_key is True:
        from core.inference.llama_keepwarm import other_inference_request_count
        if other_inference_request_count(current_request_counted = False) > 0:
            raise HTTPException(
                status_code = 409,
                detail = (
                    "Cannot queue training over the API while an inference request is in "
                    "progress. Wait for it to finish, or queue the run from the Studio UI."
                ),
            )
    try:
        # Origin is persisted so the queue runner can defer API-queued launches
        # while an inference request is in flight (see _api_item_must_wait).
        item = get_training_queue_manager().enqueue(
            request, subject = current_subject, via_api_key = via_api_key is True
        )
        return _item_model(item)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to enqueue training job",
            event = "training_queue.enqueue_failed",
            log = logger,
        )


@router.delete("/queue/items/{item_id}")
async def remove_item(item_id: str, current_subject: str = Depends(get_current_subject)):
    from storage.studio_db import get_queue_item

    if get_queue_item(item_id) is None:
        raise HTTPException(status_code = 404, detail = "Queue item not found")
    if not get_training_queue_manager().remove(item_id):
        raise HTTPException(
            status_code = 409,
            detail = "Only pending items can be removed from the queue.",
        )
    return {"status": "ok"}


@router.post("/queue/items/{item_id}/move", response_model = TrainingQueueStateResponse)
async def move_item(
    item_id: str,
    body: TrainingQueueMoveRequest,
    current_subject: str = Depends(get_current_subject),
):
    from storage.studio_db import get_queue_item

    if get_queue_item(item_id) is None:
        raise HTTPException(status_code = 404, detail = "Queue item not found")
    if not get_training_queue_manager().move(item_id, body.direction):
        raise HTTPException(
            status_code = 409,
            detail = "Item can't move that way (not pending, or already at the edge).",
        )
    return _state_response()


@router.post("/queue/pause", response_model = TrainingQueueStateResponse)
async def pause_queue(current_subject: str = Depends(get_current_subject)):
    get_training_queue_manager().pause("user")
    return _state_response()


@router.post("/queue/resume", response_model = TrainingQueueStateResponse)
async def resume_queue(current_subject: str = Depends(get_current_subject)):
    get_training_queue_manager().resume()
    return _state_response()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``cancel``: stop Studio work by kind."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from studio_mcp import export_jobs
from studio_mcp.caller import current_caller
from studio_mcp.outputs import CancelResult
from studio_mcp.tools import DESTRUCTIVE, opt_text, route_json

CancelKind = Literal[
    "training",
    "training_start",
    "diffusion_training",
    "export",
    "recipe",
    "image",
    "video",
    "chat",
    "dataset_download",
]
START_CANCELLED = "training_start_cancelled"
EXPORT_STATUS = "/api/export/status"
NEEDS_ID = {
    "training_start": "the start request id",
    "recipe": "the recipe job id",
    "chat": "the chat's cancel_id",
    "dataset_download": "the dataset's repo_id",
}


def _message(answer: Any) -> Optional[str]:
    if isinstance(answer, dict):
        return opt_text(answer.get("message")) or opt_text(answer.get("status"))
    return None


def _stopped(kind: str, answer: Any) -> bool:
    """Whether the route says it stopped something. Each route answers an idle cancel normally, so success alone means nothing."""
    if not isinstance(answer, dict):
        return False
    if kind in ("image", "video", "chat"):
        return bool(answer.get("cancelled"))
    if kind == "training":
        return answer.get("status") == "stopped"
    if kind == "training_start":
        # Rejected for another reason (a bad model, say) is not a cancel.
        return answer.get("state") == "rejected" and answer.get("error_code") == START_CANCELLED
    if kind == "diffusion_training":
        return answer.get("status") == "stopping"
    if kind == "export":
        return answer.get("message") == "Export cancelled"
    # "cancelled" here is a job that had already ended; only "cancelling" means this call stopped it.
    if kind == "recipe":
        return answer.get("status") == "cancelling"
    return answer.get("state") == "cancelling"


def _export_is_this_job(job: export_jobs.ExportJob, status: Any) -> bool:
    """Studio has one export worker and no op ids: the running op is this job's only when its kind is the step the job is on."""
    if not isinstance(status, dict) or not status.get("is_export_active"):
        return False
    op = opt_text(status.get("active_op_kind")) or ""
    if job.phase == "loading":
        return op == "load_checkpoint"
    if job.phase == "exporting":
        return op == f"export_{job.format}"
    return False


async def cancel(
    kind: CancelKind,
    id: Optional[str] = None,
    save: bool = True,
) -> CancelResult:
    """Stop Studio work. "training" stops the LLM training run ``id`` (or the current one) at its next safe point, saving a checkpoint unless ``save`` is false. "training_start" withdraws a start request that has not begun. "diffusion_training" stops image LoRA training. "export" stops the running export and unloads its checkpoint; with ``id`` (an export_model job) only while that job's own step is running. "recipe" stops recipe job ``id``. "image" and "video" stop the generation in progress. "chat" stops the reply with cancel_id ``id``. "dataset_download" stops downloading the dataset ``id`` (its repo id). Audio runs cannot be cancelled. ``cancelled`` says whether this call stopped something."""
    caller = current_caller()
    if kind in NEEDS_ID and not id:
        raise ToolError(f"cancel(kind={kind!r}) needs id: {NEEDS_ID[kind]}.")
    if kind == "training":
        if not id:
            status = await route_json("GET", "/api/train/status", caller = caller)
            id = opt_text(status.get("job_id")) if isinstance(status, dict) else None
            if not id or not (isinstance(status, dict) and status.get("is_training_running")):
                return CancelResult(
                    kind = kind, cancelled = False, message = "No training job is running."
                )
        answer = await route_json(
            "POST",
            "/api/train/stop",
            caller = caller,
            json_body = {"save": save, "expected_job_id": id},
        )
    elif kind == "training_start":
        answer = await route_json(
            "POST", f"/api/train/start-requests/{quote(id, safe = '')}/cancel", caller = caller
        )
    elif kind == "diffusion_training":
        answer = await route_json("POST", "/api/train/diffusion/stop", caller = caller)
    elif kind == "export":
        job = export_jobs.lookup(caller.account_id, id) if id else None
        # Studio has one export worker, and cancelling it stops whatever it is doing, so only cancel
        # when the op it is running is the one meant. An idle worker still holds its checkpoint.
        if job is not None and job.finished:
            return CancelResult(
                kind = kind, id = id, cancelled = False, message = f"The export already {job.status}."
            )
        status = await route_json("GET", EXPORT_STATUS, caller = caller)
        if not (isinstance(status, dict) and status.get("is_export_active")):
            return CancelResult(kind = kind, id = id, cancelled = False, message = "No export is running.")
        if job is not None and not _export_is_this_job(job, status):
            return CancelResult(
                kind = kind,
                id = id,
                cancelled = False,
                message = "The export running now is not this job; it was left alone.",
            )
        answer = await route_json("POST", "/api/export/cancel", caller = caller)
        if not _stopped(kind, answer):
            return CancelResult(kind = kind, id = id, cancelled = False, message = _message(answer))
        if job is not None:
            export_jobs.mark_cancelled(job)
        return CancelResult(
            kind = kind,
            id = id,
            cancelled = True,
            message = "The export was stopped and its checkpoint unloaded.",
        )
    elif kind == "recipe":
        answer = await route_json(
            "POST", f"/api/data-recipe/jobs/{quote(id, safe = '')}/cancel", caller = caller
        )
    elif kind == "image":
        answer = await route_json("POST", "/api/inference/images/generate/cancel", caller = caller)
    elif kind == "video":
        answer = await route_json("POST", "/api/inference/video/generate/cancel", caller = caller)
    elif kind == "chat":
        answer = await route_json(
            "POST", "/api/inference/cancel", caller = caller, json_body = {"cancel_id": id}
        )
    else:
        answer = await route_json(
            "POST", "/api/hub/datasets/download/cancel", caller = caller, json_body = {"repo_id": id}
        )
    return CancelResult(
        kind = kind, id = id, cancelled = _stopped(kind, answer), message = _message(answer)
    )


def register_cancel(mcp: FastMCP) -> None:
    mcp.tool(cancel, annotations = DESTRUCTIVE)

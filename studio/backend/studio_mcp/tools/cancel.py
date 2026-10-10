# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``cancel``: stop Unsloth Studio work by kind."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError

from studio_mcp import export_jobs
from studio_mcp.caller import current_caller
from studio_mcp.outputs import CancelResult
from studio_mcp.tools import DESTRUCTIVE, EXPORT_STATUS, as_dict, opt_text, route_json

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
LEFT_ALONE = "The export running now is not this job; it was left alone."
NEEDS_ID = {
    "training_start": "the start request id",
    "recipe": "the recipe job id",
    "chat": "the chat's cancel_id",
    "dataset_download": "the dataset's repo_id",
}
# The route each other kind posts to, and the body field that carries its id, if any.
ROUTES = {
    "training_start": ("/api/train/start-requests/{id}/cancel", None),
    "diffusion_training": ("/api/train/diffusion/stop", None),
    "recipe": ("/api/data-recipe/jobs/{id}/cancel", None),
    "image": ("/api/inference/images/generate/cancel", None),
    "video": ("/api/inference/video/generate/cancel", None),
    "chat": ("/api/inference/cancel", "cancel_id"),
    "dataset_download": ("/api/hub/datasets/download/cancel", "repo_id"),
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


# How long a cancel waits for a job whose export just finished to record its own result.
SETTLE_S = 2.0


def _export_is_this_job(job: export_jobs.ExportJob, status: Any) -> bool:
    """Unsloth Studio has one export worker and no op ids: the running op is this job's only when its kind is the step the job is on."""
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
    """Stop Unsloth Studio work. "training" stops the LLM training run ``id`` (or the current one) at its next safe point, saving a checkpoint unless ``save`` is false. "training_start" withdraws a start request that has not begun. "diffusion_training" stops image LoRA training. "export" stops the running export and unloads its checkpoint; with ``id`` (an export_model job) only while that job's own step is running. "recipe" stops recipe job ``id``. "image" and "video" stop the generation in progress. "chat" stops the reply with cancel_id ``id``. "dataset_download" stops downloading the dataset ``id`` (its repo id). Audio runs cannot be cancelled. ``cancelled`` says whether this call stopped something."""
    caller = current_caller()

    def done(cancelled: bool, message: Optional[str]) -> CancelResult:
        return CancelResult(kind = kind, id = id, cancelled = cancelled, message = message)

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
    elif kind == "export":
        job = export_jobs.lookup(caller.account_id, id) if id else None
        # Studio has one export worker, and cancelling it stops whatever it is doing, so only cancel
        # when the op it is running is the one meant. An idle worker still holds its checkpoint.
        if job is not None and job.finished:
            return done(False, f"The export already {job.status}.")
        status = await route_json("GET", EXPORT_STATUS, caller = caller)
        if not as_dict(status).get("is_export_active"):
            if job is None:
                return done(False, "No export is running.")
            if job.phase == "exporting":
                # Either its export just ended and the answer is on its way, or it is about to
                # send the export. Give it a moment, then look again: a finished export is
                # reported as such, and one that has started is stopped on the worker below.
                await export_jobs.settle(job, SETTLE_S)
                if job.finished:
                    return done(False, f"The export already {job.status}.")
                status = await route_json("GET", EXPORT_STATUS, caller = caller)
                if as_dict(status).get("is_export_active") and not _export_is_this_job(job, status):
                    # Someone else's op took the worker first and this job's export waits behind it.
                    return done(False, LEFT_ALONE)
            if not _export_is_this_job(job, status):
                # Between steps: stop the job's own task and leave the worker alone.
                export_jobs.mark_cancelled(job)
                return done(True, "The export job was stopped before its next step.")
        elif job is not None and not _export_is_this_job(job, status):
            return done(False, LEFT_ALONE)
        answer = await route_json("POST", "/api/export/cancel", caller = caller)
        if not _stopped(kind, answer):
            return done(False, _message(answer))
        if job is not None:
            export_jobs.mark_cancelled(job)
        return done(True, "The export was stopped and its checkpoint unloaded.")
    else:
        path, id_field = ROUTES[kind]
        answer = await route_json(
            "POST",
            path.format(id = quote(id or "", safe = "")),
            caller = caller,
            **({"json_body": {id_field: id}} if id_field else {}),
        )
    return done(_stopped(kind, answer), _message(answer))


TOOLS = ((cancel, DESTRUCTIVE),)

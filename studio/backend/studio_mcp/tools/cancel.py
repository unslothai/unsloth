# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``cancel``: stop Unsloth Studio work by kind."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError

from studio_mcp import export_jobs, job_store
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
NEEDS_ID = {
    "training_start": "the start request id",
    "recipe": "the recipe job id",
    "chat": "the chat's cancel_id",
    "dataset_download": "the dataset's repo_id",
}
# The route each other kind posts to, and the body field that carries its id (or ``save``), if any.
ROUTES = {
    "training_start": ("/api/train/start-requests/{id}/cancel", None),
    "diffusion_training": ("/api/train/diffusion/stop", "save"),
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


async def _cancel_export_job(caller: Any, job: export_jobs.ExportJob) -> tuple[bool, str]:
    """Stop only this job's own step: the route stops the worker when that op runs and otherwise refuses it when it starts."""
    for _ in range(3):
        op_id = job.op_id
        if op_id is None:
            break
        answer = await route_json(
            "POST", "/api/export/cancel", caller = caller, json_body = {"expected_op_id": op_id}
        )
        if _stopped("export", answer):
            job_store.mark_cancelled(job)
            return True, "The export was stopped and its checkpoint unloaded."
        # Its step just ended, or is queued behind someone else's: let a finished job report itself.
        await job_store.settle(job, SETTLE_S)
        if job.finished:
            return False, f"The export already {job.status}."
        if job.op_id == op_id:
            break
    job_store.mark_cancelled(job)
    return True, "The export job was stopped before its next step."


async def cancel(
    kind: CancelKind,
    id: Optional[str] = None,
    save: bool = True,
) -> CancelResult:
    """Stop Unsloth Studio work. "training" stops the LLM training run ``id`` (or the current one) at its next safe point, saving a checkpoint unless ``save`` is false. "training_start" withdraws a start request that has not begun. "diffusion_training" stops image LoRA training, saving the partial adapter unless ``save`` is false. "export" stops the running export and unloads its checkpoint; with ``id`` (an export_model job) only that job's own step, never someone else's export. "recipe" stops recipe job ``id``. "image" and "video" stop the generation in progress. "chat" stops the reply with cancel_id ``id``. "dataset_download" stops downloading the dataset ``id`` (its repo id). Audio runs cannot be cancelled. ``cancelled`` says whether this call stopped something."""
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
        if id:
            job = export_jobs.lookup(caller.account_id, id)
            if job.finished:
                return done(False, f"The export already {job.status}.")
            return done(*await _cancel_export_job(caller, job))
        status = await route_json("GET", EXPORT_STATUS, caller = caller)
        if not as_dict(status).get("is_export_active"):
            return done(False, "No export is running.")
        answer = await route_json("POST", "/api/export/cancel", caller = caller)
        if not _stopped(kind, answer):
            return done(False, _message(answer))
        return done(True, "The export was stopped and its checkpoint unloaded.")
    else:
        path, id_field = ROUTES[kind]
        answer = await route_json(
            "POST",
            path.format(id = quote(id or "", safe = "")),
            caller = caller,
            **({"json_body": {id_field: save if id_field == "save" else id}} if id_field else {}),
        )
    return done(_stopped(kind, answer), _message(answer))


TOOLS = ((cancel, DESTRUCTIVE),)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The kinds of work get_job reads and cancel stops, each registered with studio_mcp.jobs: how to read its state and how to stop it."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from studio_mcp import audio_runs, export_jobs, job_store, loading, model_downloads, video_queue
from studio_mcp import checkpoints as named
from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.jobs import Cancel, Kind, Stop, register
from studio_mcp.media import (
    INLINE_CAP,
    THUMB_SIDE,
    ReturnMedia,
    image_content,
    media_result,
    public_url,
    resource_link,
    signed_url,
    video_frame_path,
)
from studio_mcp.outputs import (
    ExportInfo,
    ExportJobSummary,
    JobStatus,
    JobSummary,
    RecipeInfo,
    TrainingJob,
    VideoInfo,
)
from studio_mcp.tools import EXPORT_STATUS, as_dict, integer, leaf_name, number, opt_text
from studio_mcp.tools import route_json
from studio_mcp.tools.status import SLOT_ROUTES, _diffusion_training, _training


def _message(answer: Any) -> Optional[str]:
    if isinstance(answer, dict):
        return opt_text(answer.get("message")) or opt_text(answer.get("status"))
    return None


def _outcome(answer: Any, stopped: Callable[[dict], Any], job_id: Optional[str]) -> Stop:
    """Each route answers an idle cancel normally, so success alone means nothing: ``stopped`` reads whether it stopped something."""
    return Stop(isinstance(answer, dict) and bool(stopped(answer)), _message(answer), job_id)


def _route_cancel(path: str, id_field: Optional[str], stopped: Callable[[dict], Any]) -> Cancel:
    """A cancel that posts to ``path``, with the id (or ``save``) in body field ``id_field`` if any."""

    async def cancel(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
        answer = await route_json(
            "POST",
            path.format(id = quote(job_id or "", safe = "")),
            caller = caller,
            **(
                {"json_body": {id_field: save if id_field == "save" else job_id}}
                if id_field
                else {}
            ),
        )
        return _outcome(answer, stopped, job_id)

    return cancel


def _cancelled(answer: dict) -> Any:
    return answer.get("cancelled")


# "cancelled" here is a job that had already ended; only "cancelling" means this call stopped it.
def _cancelling(answer: dict) -> bool:
    return answer.get("state") == "cancelling"


def _recipe_cancelling(answer: dict) -> bool:
    return answer.get("status") == "cancelling"


# Video.


def _video_path(video_id: str, suffix: str = "") -> str:
    return f"/v1/videos/{quote(video_id, safe = '')}{suffix}"


async def _video_job(
    caller: Caller,
    video_id: str,
    return_media: ReturnMedia = "inline",
    preview_px: Optional[int] = None,
) -> ToolResult:
    job = as_dict(await route_json("GET", _video_path(video_id), caller = caller))
    error = as_dict(job.get("error"))
    contents: list[Any] = []
    video = None
    if job.get("status") == "completed":
        # A picture at most: the MP4 can be hundreds of MB and is shared as a link.
        video_url = public_url(caller, _video_path(video_id, "/content"))
        download_url = await signed_url(caller, "video", video_id)
        thumb, mime = None, "image/webp"
        if return_media == "inline":
            thumb = await forward(
                caller, "GET", _video_path(video_id, "/content"), params = {"variant": "thumbnail"}
            )
        elif return_media == "preview":
            params = {"at": "first", "max_px": preview_px or THUMB_SIDE}
            thumb = await forward(caller, "GET", video_frame_path(video_id), params = params)
            mime = "image/png"
        inline = thumb is not None and thumb.status_code == 200 and len(thumb.content) <= INLINE_CAP
        if inline:
            contents.append(image_content(thumb.content, mime))
        contents.append(resource_link(download_url or video_url, f"{video_id}.mp4", "video/mp4"))
        video = VideoInfo(url = video_url, download_url = download_url, thumbnail_inline = inline)
    status = JobStatus(
        kind = "video",
        id = opt_text(job.get("id")) or video_id,
        status = "cancelled" if error.get("code") == "cancelled" else opt_text(job.get("status")),
        progress_percent = number(job.get("progress")),
        error = opt_text(error.get("message")),
        error_code = opt_text(error.get("code")),
        video = video,
        seed = integer(job.get("seed")),
    )
    return media_result(contents, status)


async def _video_jobs(caller: Caller) -> ToolResult:
    listing = await route_json("GET", "/v1/videos", caller = caller)
    rows = as_dict(listing).get("data")
    jobs = [
        JobSummary.from_route(
            row,
            status = (
                "cancelled"
                if as_dict(row.get("error")).get("code") == "cancelled"
                else opt_text(row.get("status")) or "queued"
            ),
            progress_percent = number(row.get("progress")),
        )
        for row in rows or []
        if isinstance(row, dict) and opt_text(row.get("id"))
    ]
    return media_result([], JobStatus(kind = "video", jobs = jobs))


async def _read_video(
    caller: Caller,
    job_id: Optional[str],
    *,
    return_media: ReturnMedia = "inline",
    preview_px: Optional[int] = None,
    **_unused: Any,
) -> ToolResult:
    if job_id is None:
        return await _video_jobs(caller)
    if video_queue.is_queue_id(job_id):
        item = video_queue.lookup(caller.account_id, job_id)
        if not item.video_id:
            queued = await video_queue.job_status(item)
            # Unless it started while that was read: then it is the video it became.
            if not item.video_id:
                return media_result([], queued)
        return await _video_job(caller, item.video_id, return_media, preview_px)
    return await _video_job(caller, job_id, return_media, preview_px)


_cancel_rendering = _route_cancel("/api/inference/video/generate/cancel", "video_id", _cancelled)


async def _cancel_video(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
    if job_id and video_queue.is_queue_id(job_id):
        return Stop(*video_queue.cancel(caller.account_id, job_id), job_id)
    return await _cancel_rendering(caller, job_id, save)


# Data recipes.

RECIPE_ROUTES = "/api/data-recipe"


def _recipe_dataset(artifact_path: Any) -> Optional[str]:
    leaf = leaf_name(artifact_path) if opt_text(artifact_path) else None
    return f"recipes/{leaf}" if leaf else None


async def _read_recipe(
    caller: Caller,
    job_id: Optional[str],
    *,
    rows: Optional[int] = None,
    offset: int = 0,
    **_unused: Any,
) -> ToolResult:
    path = (
        f"{RECIPE_ROUTES}/jobs/{quote(job_id, safe = '')}/status"
        if job_id
        else f"{RECIPE_ROUTES}/jobs/current"
    )
    if job_id:
        job = await route_json("GET", path, caller = caller)
    else:
        response = await forward(caller, "GET", path)
        # No recipe has run since Unsloth Studio started: nothing to report, not an error.
        if response.status_code == 404:
            return media_result([], JobStatus(kind = "recipe", jobs = []))
        job = raise_for_route(response)
    job = as_dict(job)
    job_id = opt_text(job.get("job_id")) or job_id
    progress = as_dict(job.get("progress"))
    data_rows = total = None
    if rows and job_id:
        response = await forward(
            caller,
            "GET",
            f"{RECIPE_ROUTES}/jobs/{quote(job_id, safe = '')}/dataset",
            params = {"limit": rows, "offset": offset},
        )
        # 404 while no rows are written yet, 422 when the job ended without any (its detail names host
        # paths): the status still answers, without rows.
        page = None if response.status_code in (404, 422) else raise_for_route(response)
        if isinstance(page, dict):
            data_rows = [row for row in page.get("dataset") or [] if isinstance(row, dict)]
            total = integer(page.get("total"))
    status = JobStatus.from_route(
        job,
        kind = "recipe",
        id = job_id,
        progress_percent = number(progress.get("percent")),
        recipe = RecipeInfo.from_route(
            job,
            dataset = _recipe_dataset(job.get("artifact_path")),
            total_rows = total,
            data_rows = data_rows,
        ),
    )
    return media_result([], status)


# Exports.


def _export_status(job: export_jobs.ExportJob) -> JobStatus:
    return JobStatus(
        kind = "export",
        id = job.job_id,
        status = job.status,
        error = job.error,
        export = ExportInfo(format = job.format, output = job.output, phase = job.phase),
    )


async def _read_export(caller: Caller, job_id: Optional[str], **_unused: Any) -> ToolResult:
    if job_id is None:
        jobs = [
            ExportJobSummary(
                id = job.job_id,
                status = job.status,
                format = job.format,
                phase = job.phase,
                output = job.output,
                error = job.error,
            )
            for job in reversed(export_jobs.jobs_of(caller.account_id))
        ]
        return media_result([], JobStatus(kind = "export", jobs = jobs))
    # Settles from its own export call, not the export status (that is whichever op ran last).
    return media_result([], _export_status(export_jobs.lookup(caller.account_id, job_id)))


# How long a cancel waits for a job whose export just finished to record its own result.
SETTLE_S = 2.0
EXPORT_STOPPED = "The export was stopped and its checkpoint unloaded."


def _export_stopped(answer: dict) -> bool:
    return answer.get("message") == "Export cancelled"


async def _cancel_export_job(caller: Any, job: export_jobs.ExportJob) -> tuple[bool, str]:
    """Stop only this job's own step: the route stops the worker when that op runs and otherwise refuses it when it starts."""
    for _ in range(3):
        op_id = job.op_id
        if op_id is None:
            break
        answer = await route_json(
            "POST", "/api/export/cancel", caller = caller, json_body = {"expected_op_id": op_id}
        )
        if isinstance(answer, dict) and _export_stopped(answer):
            job_store.mark_cancelled(job)
            return True, EXPORT_STOPPED
        # Its step just ended, or is queued behind someone else's: let a finished job report itself.
        await job_store.settle(job, SETTLE_S)
        if job.finished:
            return False, f"The export already {job.status}."
        if job.op_id == op_id:
            break
    job_store.mark_cancelled(job)
    return True, "The export job was stopped before its next step."


async def _cancel_export(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
    if job_id:
        job = export_jobs.lookup(caller.account_id, job_id)
        if job.finished:
            return Stop(False, f"The export already {job.status}.", job_id)
        return Stop(*await _cancel_export_job(caller, job), job_id)
    status = await route_json("GET", EXPORT_STATUS, caller = caller)
    if not as_dict(status).get("is_export_active"):
        return Stop(False, "No export is running.", job_id)
    answer = await route_json("POST", "/api/export/cancel", caller = caller)
    if not (isinstance(answer, dict) and _export_stopped(answer)):
        return Stop(False, _message(answer), job_id)
    return Stop(True, EXPORT_STOPPED, job_id)


# Training runs.

# Training phases by job status; any other phase means the run is still going.
TRAINING_STATUS = {
    "completed": "completed",
    "error": "failed",
    "stopped": "cancelled",
    "idle": "idle",
}


async def _read_training(
    kind: str, caller: Caller, job_id: Optional[str], **_unused: Any
) -> ToolResult:
    payload = as_dict(await route_json("GET", SLOT_ROUTES[kind], caller = caller))
    slot = (_training if kind == "training" else _diffusion_training)(payload)
    # An id that is not the current job (or one with nothing running) is never echoed back as idle.
    if job_id and job_id != slot.job_id:
        # list_training_runs reads LLM runs only.
        where = (
            f'; list_training_runs(run_id="{job_id}") has its result' if kind == "training" else ""
        )
        raise ToolError(f"{job_id} is not the current training job{where}")
    phase = "running" if kind == "diffusion_training" and slot.phase == "busy" else slot.phase
    status = TRAINING_STATUS.get(phase, "running")
    training = TrainingJob(**slot.model_dump())
    if status == "completed":
        details = as_dict(payload.get("details")) if kind == "training" else payload
        run, found = await _run_folder(caller, opt_text(details.get("output_dir")))
        training = training.model_copy(
            update = {
                "run_folder": run,
                "checkpoints": [c.name for c in found if run and c.run == run],
                # list_training_runs reads LLM runs only.
                "loss_history": (
                    f'list_training_runs(run_id="{slot.job_id}")'
                    if kind == "training" and slot.job_id
                    else None
                ),
            }
        )
    return media_result(
        [],
        JobStatus(
            kind = kind,
            id = slot.job_id or job_id,
            status = status,
            progress_percent = slot.progress_percent,
            error = slot.error,
            training = training,
        ),
    )


async def _run_folder(caller: Caller, output: Optional[str]) -> tuple[Optional[str], list]:
    """The run folder name for a status ``output_dir``: an API key gets an opaque reference, matched like list_training_runs does."""
    from hub.utils.host_paths import cache_reference

    found, folders = await named.list_checkpoints(caller)
    if not output:
        return None, found
    for folder, path in folders.items():
        if output in (path, cache_reference(path)):
            return folder, found
    return (None if output.startswith("ref:") else leaf_name(output)), found


async def _cancel_training(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
    if not job_id:
        status = await route_json("GET", "/api/train/status", caller = caller)
        job_id = opt_text(status.get("job_id")) if isinstance(status, dict) else None
        if not job_id or not (isinstance(status, dict) and status.get("is_training_running")):
            return Stop(False, "No training job is running.", None)
    answer = await route_json(
        "POST",
        "/api/train/stop",
        caller = caller,
        json_body = {"save": save, "expected_job_id": job_id},
    )
    return _outcome(answer, lambda a: a.get("status") == "stopped", job_id)


def _diffusion_stopping(answer: dict) -> bool:
    return answer.get("status") == "stopping"


START_CANCELLED = "training_start_cancelled"


def _start_withdrawn(answer: dict) -> bool:
    # Rejected for another reason (a bad model, say) is not a cancel.
    return answer.get("state") == "rejected" and answer.get("error_code") == START_CANCELLED


# Model loads.


def _load_status(job: loading.LoadJob) -> JobStatus:
    return JobStatus(
        kind = "load",
        id = job.job_id,
        status = job.status,
        progress_percent = (
            100.0
            if job.status == "completed"
            else round(100.0 * job.fraction, 1)
            if job.fraction is not None
            else None
        ),
        error = job.error,
        load = job.result if job.status == "completed" else None,
    )


async def _read_load(caller: Caller, job_id: Optional[str], **_unused: Any) -> ToolResult:
    if job_id is None:
        jobs = [
            JobSummary(id = job.job_id, status = job.status)
            for job in loading.LOADS.jobs_of(caller.account_id)
        ]
        return media_result([], JobStatus(kind = "load", jobs = jobs))
    return media_result([], _load_status(loading.LOADS.lookup(caller.account_id, job_id)))


# Model downloads.


async def _read_download(caller: Caller, job_id: Optional[str], **_unused: Any) -> ToolResult:
    if not job_id:
        raise ToolError('get_job(kind="download") needs id: the id download_model returned.')
    return media_result([], await model_downloads.status(caller, job_id))


async def _cancel_download(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
    stopped, state = await model_downloads.cancel_parts(caller, job_id)
    if stopped:
        return Stop(True, "Unsloth Studio is stopping the download.", job_id)
    return Stop(False, f"Nothing to stop: the download is {state or 'not found'}.", job_id)


# generate_audio runs.

KEPT_TAKES = "finished takes stay in Audio history."
_cancel_take = _route_cancel("/api/inference/audio/cancel", "cancel_id", _cancelled)


async def _cancel_audio(caller: Caller, job_id: Optional[str], save: bool) -> Stop:
    between = audio_runs.stop_run(caller.account_id, job_id)
    if between:
        return Stop(True, f"The run stops before its next take; {KEPT_TAKES}", job_id)
    stop = await _cancel_take(caller, job_id, save)
    if between is False:
        # The route stopped the take in flight, or it just ended; either way no take follows.
        return Stop(True, f"The run was stopped; {KEPT_TAKES}", job_id)
    return stop


register(Kind("video", read = _read_video, cancel = _cancel_video, poll_after_s = 15))
register(
    Kind(
        "recipe",
        read = _read_recipe,
        cancel = _route_cancel(RECIPE_ROUTES + "/jobs/{id}/cancel", None, _recipe_cancelling),
        poll_after_s = 10,
        follows_current = True,
        needs_id = "the recipe job id",
    )
)
register(Kind("export", read = _read_export, cancel = _cancel_export, poll_after_s = 15))
register(
    Kind(
        "training",
        read = partial(_read_training, "training"),
        cancel = _cancel_training,
        poll_after_s = 30,
        follows_current = True,
    )
)
register(
    Kind(
        "diffusion_training",
        read = partial(_read_training, "diffusion_training"),
        cancel = _route_cancel("/api/train/diffusion/stop", "save", _diffusion_stopping),
        poll_after_s = 30,
        follows_current = True,
        takes_id = False,
    )
)
register(Kind("load", read = _read_load, poll_after_s = 10))
register(
    Kind(
        "download",
        read = _read_download,
        cancel = _cancel_download,
        poll_after_s = 15,
        needs_id = "the id download_model returned",
    ),
    "model_download",
)
register(
    Kind(
        "training_start",
        cancel = _route_cancel("/api/train/start-requests/{id}/cancel", None, _start_withdrawn),
        needs_id = "the start request id",
    )
)
register(
    Kind(
        "image",
        cancel = _route_cancel("/api/inference/images/generate/cancel", None, _cancelled),
        takes_id = False,
    )
)
register(
    Kind(
        "chat",
        cancel = _route_cancel("/api/inference/cancel", "cancel_id", _cancelled),
        needs_id = "the chat's cancel_id",
    )
)
register(
    Kind(
        "audio",
        cancel = _cancel_audio,
        needs_id = "the run's cancel_id (generate_audio returns it)",
    )
)
register(
    Kind(
        "dataset_download",
        cancel = _route_cancel("/api/hub/datasets/download/cancel", "repo_id", _cancelling),
        needs_id = "the dataset's repo_id",
    )
)

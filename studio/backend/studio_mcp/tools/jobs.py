# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Long-running Unsloth Studio jobs: ``run_recipe`` starts a data recipe and ``get_job`` reports on any job, waiting for it to move when asked."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import checkpoints as named
from studio_mcp import export_jobs, loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.media import INLINE_CAP, image_content, media_result, public_url, resource_link
from studio_mcp.outputs import (
    ExportInfo,
    JobStatus,
    JobSummary,
    RecipeError,
    RecipeInfo,
    RecipeResult,
    TrainingJob,
    VideoInfo,
)
from studio_mcp.tools import READ_ONLY, WRITES, as_dict, integer, leaf_name, number, opt_text
from studio_mcp.tools import route_json
from studio_mcp.tools.status import SLOT_ROUTES, _diffusion_training, _training


def _video_path(video_id: str, suffix: str = "") -> str:
    return f"/v1/videos/{quote(video_id, safe = '')}{suffix}"


async def _video_job(caller: Caller, video_id: str) -> ToolResult:
    job = as_dict(await route_json("GET", _video_path(video_id), caller = caller))
    error = as_dict(job.get("error"))
    contents: list[Any] = []
    video = None
    if job.get("status") == "completed":
        # The thumbnail only: the MP4 can be hundreds of MB and is shared as a link.
        video_url = public_url(caller, _video_path(video_id, "/content"))
        thumb = await forward(
            caller, "GET", _video_path(video_id, "/content"), params = {"variant": "thumbnail"}
        )
        inline = thumb.status_code == 200 and len(thumb.content) <= INLINE_CAP
        if inline:
            contents.append(image_content(thumb.content, "image/webp"))
        contents.append(resource_link(video_url, f"{video_id}.mp4", "video/mp4"))
        video = VideoInfo(url = video_url, thumbnail_inline = inline)
    status = JobStatus(
        kind = "video",
        id = opt_text(job.get("id")) or video_id,
        status = opt_text(job.get("status")),
        progress_percent = number(job.get("progress")),
        error = opt_text(error.get("message")),
        video = video,
    )
    return media_result(contents, status)


async def _video_jobs(caller: Caller) -> ToolResult:
    listing = await route_json("GET", "/v1/videos", caller = caller)
    rows = as_dict(listing).get("data")
    jobs = [
        JobSummary.from_route(
            row,
            status = opt_text(row.get("status")) or "queued",
            progress_percent = number(row.get("progress")),
        )
        for row in rows or []
        if isinstance(row, dict) and opt_text(row.get("id"))
    ]
    return media_result([], JobStatus(kind = "video", jobs = jobs))


RECIPE_ROUTES = "/api/data-recipe"


def _recipe_dataset(artifact_path: Any) -> Optional[str]:
    leaf = leaf_name(artifact_path) if opt_text(artifact_path) else None
    return f"recipes/{leaf}" if leaf else None


async def _recipe_job(
    caller: Caller, job_id: Optional[str], rows: Optional[int], offset: int
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


def _export_status(job: export_jobs.ExportJob) -> JobStatus:
    return JobStatus(
        kind = "export",
        id = job.job_id,
        status = job.status,
        error = job.error,
        export = ExportInfo(format = job.format, output = job.output, phase = job.phase),
    )


async def _export_job(caller: Caller, job_id: Optional[str]) -> ToolResult:
    if job_id is None:
        jobs = [
            JobSummary(id = job.job_id, status = job.status)
            for job in export_jobs.jobs_of(caller.account_id)
        ]
        return media_result([], JobStatus(kind = "export", jobs = jobs))
    # Settles from its own export call, not the export status (that is whichever op ran last).
    return media_result([], _export_status(export_jobs.lookup(caller.account_id, job_id)))


# Training phases by job status; any other phase means the run is still going.
TRAINING_STATUS = {
    "completed": "completed",
    "error": "failed",
    "stopped": "cancelled",
    "idle": "idle",
}


async def _training_job(caller: Caller, kind: str, job_id: Optional[str]) -> ToolResult:
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


async def _load_job(caller: Caller, job_id: Optional[str]) -> ToolResult:
    if job_id is None:
        jobs = [
            JobSummary(id = job.job_id, status = job.status)
            for job in loading.LOADS.jobs_of(caller.account_id)
        ]
        return media_result([], JobStatus(kind = "load", jobs = jobs))
    return media_result([], _load_status(loading.LOADS.lookup(caller.account_id, job_id)))


async def _read_once(
    caller: Caller, kind: str, job_id: Optional[str], rows: Optional[int], offset: int
) -> ToolResult:
    if kind == "recipe":
        return await _recipe_job(caller, job_id, rows, offset)
    if kind == "export":
        return await _export_job(caller, job_id)
    if kind == "load":
        return await _load_job(caller, job_id)
    if kind in ("training", "diffusion_training"):
        return await _training_job(caller, kind, job_id)
    if job_id is None:
        return await _video_jobs(caller)
    return await _video_job(caller, job_id)


# Job statuses that mean the job is still going; get_job(wait_s) waits only on these.
RUNNING = {"running", "queued", "in_progress", "created", "pending", "active", "cancelling"}


def _moment(structured: Any) -> tuple[Any, Any, Any]:
    # (status, phase, percent): a wait ends when the first two change. Export has no percent, only a phase.
    job = as_dict(structured)
    phase = as_dict(job.get("export")).get("phase") or as_dict(job.get("training")).get("phase")
    return job.get("status"), phase, job.get("progress_percent")


async def get_job(
    kind: Literal["video", "recipe", "export", "training", "diffusion_training", "load"],
    id: Optional[str] = None,
    wait_s: Annotated[
        int,
        Field(
            ge = 0,
            le = loading.MAX_WAIT_S,
            description = (
                "Seconds to wait for the job to finish or move on, 0 to 240. "
                "50 stays under Codex's 60 s default tool timeout; above that, raise the client's timeout."
            ),
        ),
    ] = 0,
    ctx: Optional[Context] = None,
    rows: Annotated[Optional[int], Field(ge = 1, le = 500)] = None,
    offset: Annotated[int, Field(ge = 0)] = 0,
) -> ToolResult:
    """The state of a long-running job.
    Follow every long job with ``wait_s`` instead of polling: the call checks every 2 s, reports progress, and returns once the job finishes, its status changes (for an export or a training run, also its phase), or ``wait_s`` runs out; a running job then gives ``poll_after_s``.
    kind "video": with ``id``, its status and progress, and once completed a thumbnail plus the video's URL; without ``id``, recent video jobs.
    kind "recipe": a data recipe job's status and progress (without ``id``, the current one), and with ``rows`` that many generated rows from ``offset``.
    Unsloth Studio runs one recipe at a time; a recipe job id is not tied to an account.
    kind "export": an export_model job by its id (without ``id``, this key's export jobs), with its output named relative to Unsloth Studio's exports folder.
    kind "training" or "diffusion_training": the current run started by start_training (``id`` is its job id), and once completed its run folder and checkpoint names; ``loss_history`` names the list_training_runs call with the full loss curve.
    kind "load": a load_model call that returned ``status`` "loading", by its ``job_id``; once completed, ``load`` is the load's result with what it unloaded."""
    caller = current_caller()
    deadline = time.monotonic() + min(wait_s, loading.MAX_WAIT_S)
    result = await _read_once(caller, kind, id, rows, offset)
    start = _moment(result.structured_content)
    # A listing has nothing to wait on.
    waits = id is not None or kind in ("recipe", "training", "diffusion_training")
    moved = polled = False
    while waits and start[0] in RUNNING:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        await asyncio.sleep(min(loading.POLL_INTERVAL_S, left))
        try:
            # Recipe rows are read once, for the answer, not on every poll; a read that stalls past
            # the deadline ends the wait with the last answer.
            latest = await asyncio.wait_for(
                _read_once(caller, kind, id, None, offset),
                timeout = max(deadline - time.monotonic(), 0.01),
            )
        except asyncio.TimeoutError:
            break
        except Exception:
            # Like a load's progress poll: a failed read during the wait is skipped.
            continue
        result, polled = latest, True
        now = _moment(result.structured_content)
        if now[:2] != start[:2]:
            moved = True
            break
        if ctx is not None and now[2] is not None:
            await ctx.report_progress(min(now[2] / 100.0, 1.0), 1.0, str(now[1] or now[0]))
    if polled and rows:
        result = await _read_once(caller, kind, id, rows, offset)
    # A wait that ended because the job moved on can be followed again at once.
    return _with_poll_after(kind, result, 0 if moved else POLL_AFTER_S[kind])


# Seconds a caller not using wait_s should leave before asking again, by kind.
POLL_AFTER_S = {
    "video": 15,
    "recipe": 10,
    "export": 15,
    "training": 30,
    "diffusion_training": 30,
    "load": 10,
}


def _with_poll_after(kind: str, result: ToolResult, after: float) -> ToolResult:
    structured = as_dict(result.structured_content)
    if structured.get("status") not in RUNNING or structured.get("jobs") is not None:
        return result
    status = JobStatus.model_validate({**structured, "poll_after_s": after})
    return media_result(list(result.content[1:]), status)


async def run_recipe(
    recipe: dict[str, Any], mode: Literal["validate", "preview", "full"] = "validate"
) -> RecipeResult:
    """Validate a Data Recipe, or run it. "validate" checks it and returns any errors; "preview" generates a few rows; "full" generates the whole dataset and saves it. A run returns a job id at once: follow it with get_job(kind="recipe"). Recipes that define a local (stdio) MCP server can only be run from the Unsloth Studio UI."""
    if mode == "validate":
        payload = await route_json(
            "POST", f"{RECIPE_ROUTES}/validate", json_body = {"recipe": recipe}
        )
        payload = as_dict(payload)
        errors = [
            RecipeError.from_route(e, message = opt_text(e.get("message")) or "invalid")
            for e in payload.get("errors") or []
            if isinstance(e, dict)
        ]
        return RecipeResult(mode = mode, valid = payload.get("valid") is True, errors = errors)
    created = await route_json(
        "POST",
        f"{RECIPE_ROUTES}/jobs",
        json_body = {"recipe": recipe, "run": {"execution_type": mode}},
    )
    job_id = opt_text(created.get("job_id")) if isinstance(created, dict) else None
    if job_id is None:
        raise ToolError("Unsloth Studio did not start the recipe")
    return RecipeResult(mode = mode, job_id = job_id)


TOOLS = ((run_recipe, WRITES), (get_job, READ_ONLY, JobStatus.model_json_schema()))

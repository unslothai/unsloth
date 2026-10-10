# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Long-running Unsloth Studio jobs: ``run_recipe`` starts a data recipe and ``get_job`` reports on any job."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.media import INLINE_CAP, image_content, media_result, public_url, resource_link
from studio_mcp import export_jobs
from studio_mcp.outputs import (
    ExportInfo,
    JobStatus,
    JobSummary,
    RecipeError,
    RecipeInfo,
    RecipeResult,
    VideoInfo,
)
from studio_mcp.tools import READ_ONLY, WRITES, as_dict, integer, leaf_name, number, opt_text
from studio_mcp.tools import route_json


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
        page = await route_json(
            "GET",
            f"{RECIPE_ROUTES}/jobs/{quote(job_id, safe = '')}/dataset",
            caller = caller,
            params = {"limit": rows, "offset": offset},
        )
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
    # The job settles from its own export call. The export status is not read here: it describes
    # whichever op ran last, which may be an export started from the Export page.
    return media_result([], _export_status(export_jobs.lookup(caller.account_id, job_id)))


async def get_job(
    kind: Literal["video", "recipe", "export"],
    id: Optional[str] = None,
    rows: Annotated[Optional[int], Field(ge = 1, le = 500)] = None,
    offset: Annotated[int, Field(ge = 0)] = 0,
) -> ToolResult:
    """The state of a long-running job. kind "video": with ``id``, its status and progress, and once completed a thumbnail plus the video's URL; without ``id``, recent video jobs. kind "recipe": a data recipe job's status and progress (without ``id``, the current one), and with ``rows`` that many generated rows from ``offset``. Unsloth Studio runs one recipe at a time; a recipe job id is not tied to an account. kind "export": an export_model job by its id (without ``id``, this key's export jobs), with its output named relative to Unsloth Studio's exports folder."""
    caller = current_caller()
    if kind == "recipe":
        return await _recipe_job(caller, id, rows, offset)
    if kind == "export":
        return await _export_job(caller, id)
    if id is None:
        return await _video_jobs(caller)
    return await _video_job(caller, id)


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

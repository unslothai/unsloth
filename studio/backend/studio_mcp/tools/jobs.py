# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``get_job``: the state of a long-running Studio job."""

from __future__ import annotations

from typing import Any, Literal, Optional
from urllib.parse import quote

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from studio_mcp.caller import Caller, current_caller
from studio_mcp.forward import forward
from studio_mcp.media import INLINE_CAP, image_content, media_result, public_url, resource_link
from studio_mcp.outputs import JobStatus, JobSummary, VideoInfo
from studio_mcp.tools import READ_ONLY, number, route_json, opt_text


def _video_path(video_id: str, suffix: str = "") -> str:
    return f"/v1/videos/{quote(video_id, safe = '')}{suffix}"


async def _video_job(caller: Caller, video_id: str) -> ToolResult:
    job = await route_json("GET", _video_path(video_id), caller = caller)
    job = job if isinstance(job, dict) else {}
    error = job.get("error") if isinstance(job.get("error"), dict) else {}
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
    rows = listing.get("data") if isinstance(listing, dict) else None
    jobs = [
        JobSummary(
            id = row["id"],
            status = opt_text(row.get("status")) or "queued",
            progress_percent = number(row.get("progress")),
        )
        for row in rows or []
        if isinstance(row, dict) and opt_text(row.get("id"))
    ]
    return media_result([], JobStatus(kind = "video", jobs = jobs))


async def get_job(kind: Literal["video"], id: Optional[str] = None) -> ToolResult:
    """The state of a long-running job. kind "video": with ``id``, its status and progress, and once completed a thumbnail plus the video's URL; without ``id``, recent video jobs."""
    caller = current_caller()
    if id is None:
        return await _video_jobs(caller)
    return await _video_job(caller, id)


def register_jobs(mcp: FastMCP) -> None:
    mcp.tool(get_job, annotations = READ_ONLY, output_schema = JobStatus.model_json_schema())

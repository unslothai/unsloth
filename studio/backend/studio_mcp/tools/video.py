# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``generate_video``: starts a video job; get_job follows it."""

from __future__ import annotations

from typing import Any, Optional

from fastmcp.exceptions import ToolError

from studio_mcp.caller import current_caller
from studio_mcp.inputs import ImageInput, data_url, resolve_image
from studio_mcp.outputs import VideoJobRef
from studio_mcp.tools import WRITES, opt_text, present, route_json

LOAD_VIDEO_HINT = "Load a video model with load_model(kind='video') first."


def job_ref(job: Any) -> VideoJobRef:
    if not isinstance(job, dict) or not opt_text(job.get("id")):
        raise ToolError("Unsloth Studio returned no video job")
    return VideoJobRef.from_route(job, status = opt_text(job.get("status")) or "queued")


async def generate_video(
    prompt: str,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    first_frame: Optional[ImageInput] = None,
    model: Optional[str] = None,
) -> VideoJobRef:
    """Start a video with the video model loaded in Unsloth Studio (load_model(kind="video") first) and return its job at once; poll get_job(kind="video", id=...) until it is completed, which returns a thumbnail and the video's URL. ``seconds`` and ``size`` ("WIDTHxHEIGHT") are strings, as in the OpenAI API; leave them out to use the model's defaults. Each video model takes only certain sizes, and an unsupported one is refused with the list it accepts. ``first_frame`` starts the video from an image."""
    caller = current_caller()
    body: dict[str, Any] = {"prompt": prompt, **present(seconds = seconds, size = size, model = model)}
    if first_frame is not None:
        data, mime = await resolve_image(caller, first_frame)
        body["input_reference"] = data_url(data, mime)
    job = await route_json(
        "POST",
        "/v1/videos",
        caller = caller,
        json_body = body,
        hub_header = True,
        hints = {503: LOAD_VIDEO_HINT},
    )
    return job_ref(job)


TOOLS = ((generate_video, WRITES),)

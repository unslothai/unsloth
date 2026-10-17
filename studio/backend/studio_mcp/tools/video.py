# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``generate_video``: starts a video job; get_job follows it."""

from __future__ import annotations

from typing import Annotated, Any, Optional, Union

from fastmcp.exceptions import ToolError
from pydantic import Field

from studio_mcp.caller import current_caller
from studio_mcp.inputs import (
    AudioInput,
    ImageInput,
    audio_ref,
    data_url,
    resolve_image,
    resolve_images,
)
from studio_mcp.outputs import VideoJobRef
from studio_mcp.tools import WRITES, opt_text, present, route_json

LOAD_VIDEO_HINT = "Load a video model with load_model(kind='video') first."
MAX_REFERENCE_IMAGE_BYTES = 96 * 1024 * 1024


def job_ref(job: Any) -> VideoJobRef:
    if not isinstance(job, dict) or not opt_text(job.get("id")):
        raise ToolError("Unsloth Studio returned no video job")
    return VideoJobRef.from_route(job, status = opt_text(job.get("status")) or "queued")


async def generate_video(
    prompt: str,
    seconds: Annotated[
        Optional[Union[int, str]],
        Field(
            description = "Clip length in seconds, a number or a string; left out, the model's default."
        ),
    ] = None,
    size: Optional[str] = None,
    first_frame: Optional[ImageInput] = None,
    model: Optional[str] = None,
    last_frame: Annotated[
        Optional[ImageInput],
        Field(description = "Image the clip ends on (studio_status video.supports_last_frame)."),
    ] = None,
    reference_images: Annotated[
        Optional[list[ImageInput]],
        Field(
            max_length = 9,
            description = "Up to 9 subject, style or scene references, called <Picture 1>, <Picture 2>, ... in the prompt (studio_status video.references).",
        ),
    ] = None,
    reference_audios: Annotated[
        Optional[list[AudioInput]],
        Field(
            max_length = 3,
            description = "Up to 3 voice or score references, called <Audio 1>, ... in the prompt; they need reference_images.",
        ),
    ] = None,
    negative_prompt: Annotated[
        Optional[str], Field(description = "What the video should avoid.")
    ] = None,
    seed: Annotated[
        Optional[int],
        Field(
            ge = 0,
            le = 2**53 - 1,
            description = "Reuse a take's seed to repeat it; left out, Unsloth Studio picks one and returns it.",
        ),
    ] = None,
    steps: Annotated[
        Optional[int],
        Field(ge = 1, le = 100, description = "Denoising steps; left out, the model's default."),
    ] = None,
    guidance: Annotated[
        Optional[float],
        Field(ge = 0, le = 20, description = "Guidance scale; left out, the model's default."),
    ] = None,
    num_frames: Annotated[
        Optional[int], Field(ge = 1, description = "Frame count instead of seconds; send one of them.")
    ] = None,
    fps: Annotated[
        Optional[int],
        Field(ge = 1, le = 120, description = "Playback frame rate; left out, the model's default."),
    ] = None,
) -> VideoJobRef:
    """Start a video with the video model loaded in Unsloth Studio (load_model(kind="video") first) and return its job at once.
    Poll get_job(kind="video", id=...) until it is completed, which returns a thumbnail and the video's URL.
    Unsloth Studio renders one video at a time: while one runs, another call is refused with HTTP 409, so wait for the first to finish.
    ``seconds`` is a number or a string and ``size`` is "WIDTHxHEIGHT" from studio_status video.size_presets; leave them out to use the model's defaults.
    ``model`` names a video model for Unsloth Studio to switch to when its automatic model switching is on; otherwise the loaded model runs.
    ``first_frame`` starts the video from an image and ``last_frame`` ends it on one, where the model takes it.
    ``reference_images`` and ``reference_audios`` condition a references model (studio_status video.references).
    The job carries the ``seed`` it used; pass it again with the same settings to repeat a take.
    studio_status video.prompt_hint says how the loaded model reads prompts."""
    caller = current_caller()
    body: dict[str, Any] = {
        "prompt": prompt,
        **present(
            seconds = seconds,
            size = size,
            model = model,
            negative_prompt = negative_prompt,
            seed = seed,
            steps = steps,
            guidance = guidance,
            num_frames = num_frames,
            fps = fps,
        ),
    }
    if reference_audios and not reference_images and first_frame is None:
        raise ToolError("reference_audios need a reference image to go with: add reference_images.")
    if last_frame is not None:
        data, mime = await resolve_image(caller, last_frame)
        body["last_frame"] = data_url(data, mime)
    if reference_images:
        body["reference_images"] = await resolve_images(
            caller,
            reference_images,
            total_cap = MAX_REFERENCE_IMAGE_BYTES,
            too_large = "The images together are larger than the 96 MiB one request takes",
        )
    if reference_audios:
        body["reference_audios"] = [await audio_ref(caller, audio) for audio in reference_audios]
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

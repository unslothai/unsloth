# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``generate_video`` chat tool: one clip on the video model already loaded in Unsloth Studio.

It goes through the same job path as ``POST /v1/videos`` (``_create_openai_video``) with no
model named, so it never loads, switches or unloads a model. The job runs on the video
backend's own thread; the tool polls it and copies the finished MP4 into the chat workdir.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import threading
import time
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

NOT_LOADED_MSG = (
    "Error: no video model is loaded in Unsloth Studio. "
    "Load one on the Video page, then try again."
)
BUSY_MSG = "Error: another video is rendering in Unsloth Studio; try again when it finishes."
STOPPED_MSG = "Error: video generation was stopped."
STILL_RENDERING_MSG = (
    "Error: the video is still rendering; find it on the Video page when it finishes."
)
OTHER_ACCOUNT_MSG = (
    "Error: another account is generating on the video model. Try again when it finishes."
)
NO_LOOP_MSG = "Error: Unsloth Studio could not reach its server loop to run this tool. Try again."
NO_PROMPT_MSG = "Error: generate_video needs a prompt."

_DEFAULT_TIMEOUT_S = 900.0
_CREATE_TIMEOUT_S = 60.0
_STOP_WAIT_S = 30.0
_POLL_S = 1.0

_DESCRIPTION = (
    "Generate one short video clip with the video model loaded in Unsloth Studio. Write the full "
    "prompt yourself from the user's request: subject, motion, setting, style and camera. Length "
    "and size default to the loaded model's own settings. It never loads or unloads a model."
)


def _seed_supported() -> bool:
    from models.inference import VideoJobCreateRequest
    return "seed" in VideoJobCreateRequest.model_fields


def _number_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def generate_video_schema(status: dict) -> dict:
    description = _DESCRIPTION
    defaults = (status or {}).get("defaults") or {}
    durations = [_number_text(d) for d in defaults.get("duration_presets") or []]
    sizes = [f"{p[0]}x{p[1]}" for p in defaults.get("resolution_presets") or [] if len(p) == 2]
    if durations:
        description += f" This model offers clips of {', '.join(durations)} seconds."
    if sizes:
        description += f" Its sizes include {', '.join(sizes[:4])}."
    if (status or {}).get("prompt_hint"):
        description += " " + str(status["prompt_hint"])
    properties: dict = {
        "prompt": {"type": "string", "description": "Full description of the clip to render"},
        "seconds": {"type": "number", "description": "Clip length in seconds (optional)"},
        "width": {"type": "integer", "description": "Width in pixels (optional, with height)"},
        "height": {"type": "integer", "description": "Height in pixels (optional, with width)"},
    }
    if _seed_supported():
        properties["seed"] = {"type": "integer", "description": "Seed to repeat a clip (optional)"}
    return {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": ["prompt"]},
        },
    }


def _http_message(exc) -> str:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        inner = detail.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if detail.get("message"):
            return str(detail["message"])
    return str(detail or exc)


def run_generate_video(
    arguments: dict,
    *,
    workdir: str,
    cancel_event = None,
    timeout: Optional[float] = None,
) -> str:
    """Render one clip; always returns the tool result text, never raises."""
    try:
        return _run(
            arguments if isinstance(arguments, dict) else {}, workdir, cancel_event, timeout
        )
    except Exception as exc:  # noqa: BLE001 - a tool result, never a crashed tool loop
        logger.warning("generate_video failed: %s", exc, exc_info = True)
        return f"Error: video generation failed: {exc}"


def _run(arguments: dict, workdir: str, cancel_event, timeout: Optional[float]) -> str:
    from fastapi import HTTPException

    from core.inference import media_keepwarm
    from core.inference.gpu_arbiter import VIDEO, GpuBusyForAnotherAccountError
    from core.inference.media_chat_tools import media_chat_status, server_loop
    from core.inference.video_families import VIDEO_GENERATION_BUSY_MSG
    from models.inference import VideoJobCreateRequest
    from utils.account_context import arun_as, current_account, run_as

    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt:
        return NO_PROMPT_MSG
    acct = current_account()
    loop = server_loop()
    if loop is None:
        return NO_LOOP_MSG
    if run_as(acct, media_chat_status, "text-to-video") is None:
        return NOT_LOADED_MSG

    fields: dict = {"prompt": prompt}
    if arguments.get("seconds") is not None:
        fields["seconds"] = _number_text(arguments["seconds"])
    width, height = arguments.get("width"), arguments.get("height")
    if width is not None and height is not None:
        fields["size"] = f"{_number_text(width)}x{_number_text(height)}"
    seed = arguments.get("seed")
    if seed is not None and _seed_supported():
        fields["seed"] = int(seed)
    else:
        seed = None
    body = VideoJobCreateRequest(**fields)

    cap = _DEFAULT_TIMEOUT_S if timeout is None or timeout >= 9999 else float(timeout)
    deadline = time.monotonic() + cap

    async def _create():
        from routes.video import _create_openai_video

        await media_keepwarm.begin_request(VIDEO)
        created = False
        try:
            # No model named: the auto-switch returns at once, so the subject is never consulted.
            job = await _create_openai_video(body, None, acct.username or "", None)
            created = True
            return job
        finally:
            media_keepwarm.end_request(VIDEO, counted = created)

    future = asyncio.run_coroutine_threadsafe(arun_as(acct, _create()), loop)
    # Never cancel a create in flight: past the backend's begin_generate the job already runs,
    # and a cancel there would leave it rendering with no record on the Video page.
    create_by = time.monotonic() + min(_CREATE_TIMEOUT_S, max(cap, 1.0))
    while not future.done():
        if cancel_event is not None and cancel_event.is_set():
            concurrent.futures.wait([future], timeout = _STOP_WAIT_S)
            if not future.done():
                # Stop the job as soon as the create hands one back.
                future.add_done_callback(lambda f: _cancel_created_job(acct, f))
                return STOPPED_MSG
            break
        if time.monotonic() >= create_by:
            return STILL_RENDERING_MSG
        concurrent.futures.wait([future], timeout = 0.25)
    try:
        job = future.result()
    except concurrent.futures.CancelledError:
        return STOPPED_MSG
    except GpuBusyForAnotherAccountError:
        return OTHER_ACCOUNT_MSG
    except HTTPException as exc:
        message = _http_message(exc)
        if exc.status_code == 409 and message == VIDEO_GENERATION_BUSY_MSG:
            return BUSY_MSG
        if exc.status_code == 503:
            return NOT_LOADED_MSG
        return f"Error: {message}"

    return _await_job(acct, job.id, workdir, prompt, seed, cancel_event, deadline)


def _cancel_created_job(acct, future) -> None:
    """Done-callback (server loop): stop the job a create finished after the user pressed Stop."""
    if future.cancelled() or future.exception() is not None:
        return

    def _cancel(video_id: str) -> None:
        from core.inference.video import get_video_backend
        from utils.account_context import run_as

        try:
            run_as(
                acct,
                get_video_backend().cancel_generate,
                expected_video_id = video_id,
                expected_account = acct.account_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate_video: stopping job %s failed: %s", video_id, exc)

    # Off the loop thread: the backend cancel takes its lock.
    threading.Thread(target = _cancel, args = (future.result().id,), daemon = True).start()


def _await_job(acct, video_id: str, workdir: str, prompt: str, seed, cancel_event, deadline) -> str:
    from core.inference import video_gallery
    from core.inference.video import get_video_backend
    from routes.video import _lookup_video
    from utils.account_context import run_as

    while True:
        view = run_as(acct, _lookup_video, video_id)
        if view is None:
            return "Error: the video was removed before it finished."
        if view.status == "completed":
            break
        if view.status == "failed":
            message = view.error.message if view.error else "video generation failed."
            return f"Error: {message}"
        if cancel_event is not None and cancel_event.is_set():
            run_as(
                acct,
                get_video_backend().cancel_generate,
                expected_video_id = video_id,
                expected_account = acct.account_id,
            )
            stop_by = time.monotonic() + _STOP_WAIT_S
            while time.monotonic() < stop_by:
                view = run_as(acct, _lookup_video, video_id)
                if view is None or view.status in ("completed", "failed"):
                    break
                time.sleep(min(_POLL_S, 0.25))
            # A clip that finished inside the stop window is still a saved clip: hand it over.
            if view is None or view.status != "completed":
                return STOPPED_MSG
            break
        if time.monotonic() >= deadline:
            return STILL_RENDERING_MSG
        if cancel_event is not None:
            cancel_event.wait(_POLL_S)
        else:
            time.sleep(_POLL_S)

    source = run_as(acct, video_gallery.owned_video_path, video_id)
    if source is None:
        return "Error: the finished video is missing from the Video page."
    root = os.path.realpath(workdir)
    os.makedirs(root, exist_ok = True)
    name = f"generated-video-{video_id.removeprefix('video_')[:8]}.mp4"
    target = os.path.join(root, name)
    shutil.copyfile(source, target)
    shown = " ".join(prompt.split())
    if len(shown) > 300:
        shown = shown[:299].rstrip() + "…"
    parts = [f"{view.seconds}s", view.size] + ([f"seed {seed}"] if seed is not None else [])
    text = f"Generated a video ({', '.join(parts)}) saved as {name}. Prompt: {shown}"
    return text + "\n__FILES__:" + json.dumps([{"name": name, "size": os.path.getsize(target)}])

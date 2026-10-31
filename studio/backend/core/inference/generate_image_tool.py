# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat tool: one image on the loaded image model for the caller's account; never loads/unloads models"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

GENERATE_IMAGE_TOOL_NAME = "generate_image"

ERR_NO_PROMPT = "Error: generate_image needs a prompt."
ERR_NOT_LOADED = "Error: no image model is loaded in Unsloth Studio. Load one on the Images page, then try again."
ERR_EDIT_ONLY = (
    "Error: the image model loaded in Unsloth Studio only edits images. "
    "Load a text-to-image model on the Images page."
)
ERR_BUSY = "Error: another image is generating in Unsloth Studio. Try again when it finishes."
ERR_STOPPED = "Error: image generation was stopped."
ERR_TIMEOUT = "Error: image generation took too long and was stopped."
ERR_OTHER_ACCOUNT = (
    "Error: another account is generating on the image model. Try again when it finishes."
)
ERR_NO_LOOP = "Error: Unsloth Studio could not reach its server loop to run this tool. Try again."

_DESCRIPTION = (
    "Generate one image with the image model loaded in Unsloth Studio. Write the full prompt "
    "yourself from the user's request: subject, setting, style, lighting and composition. Steps "
    "and guidance follow the loaded model's own defaults; the size defaults to 1024x1024. It "
    "never loads or unloads a model."
)

# A chat with no tool timeout ("no limit") still gets a ceiling, so a stuck render can't hold
# the turn forever.
_MAX_WAIT_S = 900.0
_STOP_WAIT_S = 30.0
_POLL_S = 0.25
_PROMPT_ECHO_CHARS = 300
_MIN_SIDE = 256
# DiffusionGenerateRequest's own ceiling.
_TRANSPORT_MAX_SIDE = 2752


def _size_limits(status: dict) -> tuple[int, int, int, int]:
    """(multiple, min side, max side, max pixels) of the loaded family, as the route enforces them."""
    cond = status.get("conditioning") or {}
    multiple = max(1, int(cond.get("dimension_multiple") or 16))
    max_side = min(_TRANSPORT_MAX_SIDE, int(cond.get("max_output_side") or 2048))
    max_side = max_side // multiple * multiple
    max_pixels = int(cond.get("max_output_pixels") or 2048 * 2048)
    min_side = -(-_MIN_SIDE // multiple) * multiple
    return multiple, min_side, max_side, max_pixels


def generate_image_schema(status: dict) -> dict:
    """The OpenAI function tool for the loaded image model; negative_prompt only when it has one."""
    multiple, min_side, max_side, max_pixels = _size_limits(status)
    bounds = f"{min_side} to {max_side}, a multiple of {multiple}"
    if max_pixels < max_side * max_side:
        bounds += f", at most {max_pixels:,} pixels in total"
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "description": "The full image prompt."},
        "width": {
            "type": "integer",
            "description": f"Width in pixels, {bounds}. Default 1024.",
        },
        "height": {
            "type": "integer",
            "description": f"Height in pixels, {bounds}. Default 1024.",
        },
        "seed": {
            "type": "integer",
            "description": "Seed for a repeatable image. Random when omitted.",
        },
    }
    if status.get("supports_negative_prompt"):
        properties["negative_prompt"] = {
            "type": "string",
            "description": "What the image should not show.",
        }
    description = _DESCRIPTION
    hint = status.get("prompt_hint")
    if hint:
        description += " " + str(hint)
    return {
        "type": "function",
        "function": {
            "name": GENERATE_IMAGE_TOOL_NAME,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": ["prompt"]},
        },
    }


def offers_txt2img(status: dict) -> bool:
    """False for an edit-only model, the same rule /v1/images/generations refuses on."""
    workflows = status.get("workflows")
    return not workflows or "txt2img" in workflows


def _snap_size(arguments: dict, status: dict) -> tuple[int, int]:
    """Width and height on the loaded family's grid and inside its bounds (the route refuses, never snaps)."""
    multiple, min_side, max_side, max_pixels = _size_limits(status)

    def snap(value: Any) -> int:
        return min(max_side, max(min_side, int(round(int(value) / multiple)) * multiple))

    w = snap(arguments.get("width") if arguments.get("width") is not None else 1024)
    h = snap(arguments.get("height") if arguments.get("height") is not None else 1024)
    while w * h > max_pixels and max(w, h) > min_side:
        if w >= h:
            w -= multiple
        else:
            h -= multiple
    return w, h


def _build_request(arguments: dict, prompt: str, status: dict):
    from models.inference import DiffusionGenerateRequest

    # Only what the model supplied: an omitted steps/guidance must stay unset so the core
    # resolves the loaded model's own defaults.
    fields: dict[str, Any] = {"prompt": prompt}
    if arguments.get("width") is not None or arguments.get("height") is not None:
        fields["width"], fields["height"] = _snap_size(arguments, status)
    if arguments.get("seed") is not None:
        fields["seed"] = int(arguments["seed"])
    negative = arguments.get("negative_prompt")
    if status.get("supports_negative_prompt") and isinstance(negative, str) and negative.strip():
        fields["negative_prompt"] = negative.strip()
    return DiffusionGenerateRequest(**fields)


async def _generate(request, ticket):
    from core.inference import media_keepwarm
    from core.inference.generation_ticket import generation_ticket
    from core.inference.gpu_arbiter import DIFFUSION
    from routes.inference import _generate_diffusion_image_core

    # The HTTP route is counted by the keep-warm middleware; this path never goes through it.
    await media_keepwarm.begin_request(DIFFUSION)
    try:
        with generation_ticket(ticket):
            return await _generate_diffusion_image_core(request)
    finally:
        media_keepwarm.end_request(DIFFUSION)


def _stop(future, ticket) -> None:
    """Stop this call's render on the GPU too, queued or running. Never the account-wide cancel:
    that reaches whichever render is active, possibly another chat's or the Images page's."""
    ticket.stop()
    concurrent.futures.wait([future], timeout = _STOP_WAIT_S)
    future.cancel()


def _http_error(exc) -> str:
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG, DIFFUSION_NOT_LOADED_MSG

    detail = exc.detail
    if isinstance(detail, dict):
        if detail.get("error") == "gpu_busy":
            return ERR_OTHER_ACCOUNT
        error = detail.get("error")
        text = detail.get("message") or (error.get("message") if isinstance(error, dict) else error)
    else:
        text = detail
    text = str(text or "image generation failed.")
    if text == DIFFUSION_CANCELLED_MSG:
        return ERR_STOPPED
    if text == DIFFUSION_NOT_LOADED_MSG:
        return ERR_NOT_LOADED
    return "Error: " + text


def _save_to_workdir(image, workdir: str, account) -> dict:
    from core.inference import image_gallery
    from utils.account_context import run_as

    source = run_as(account, image_gallery.image_path, image.id)
    if source is None:
        raise RuntimeError("the generated image is missing from the gallery.")
    name = f"generated-image-{image.id[:8]}.png"
    os.makedirs(workdir, exist_ok = True)
    target = os.path.join(workdir, name)
    shutil.copyfile(source, target)
    return {"name": name, "size": os.path.getsize(target)}


def _run(arguments: dict, workdir: str, cancel_event, timeout: Optional[float]) -> str:
    import asyncio

    from fastapi import HTTPException

    from core.inference.generation_ticket import GenerationTicket
    from core.inference.gpu_arbiter import DIFFUSION, GpuBusyForAnotherAccountError
    from core.inference.media_chat_tools import media_chat_status, server_loop
    from core.inference.media_keepwarm import engine_if_imported
    from utils.account_context import arun_as, current_account

    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt:
        return ERR_NO_PROMPT
    account = current_account()
    loop = server_loop()
    if loop is None:
        return ERR_NO_LOOP
    status = media_chat_status("text-to-image")
    if status is None:
        return ERR_NOT_LOADED
    if not offers_txt2img(status):
        return ERR_EDIT_ONLY
    engine = engine_if_imported(DIFFUSION)
    progress = getattr(engine, "generate_progress", None)
    # Never queue: the engine cancel only reaches the active run, so a queued one could not stop.
    if progress is not None and progress().get("active"):
        return ERR_BUSY
    try:
        request = _build_request(arguments, prompt, status)
    except (TypeError, ValueError) as exc:
        errors = getattr(exc, "errors", None)
        message = errors()[0].get("msg") if callable(errors) and errors() else str(exc)
        return f"Error: invalid generate_image arguments: {message}"

    ticket = GenerationTicket()
    future = asyncio.run_coroutine_threadsafe(arun_as(account, _generate(request, ticket)), loop)
    limit = _MAX_WAIT_S if timeout is None or timeout >= 9999 else max(1.0, float(timeout))
    deadline = time.monotonic() + limit
    stopped = None
    while not future.done():
        if cancel_event is not None and cancel_event.is_set():
            stopped = ERR_STOPPED
        elif time.monotonic() >= deadline:
            stopped = ERR_TIMEOUT
        if stopped:
            _stop(future, ticket)
            break
        concurrent.futures.wait([future], timeout = _POLL_S)
    # A render that finished inside the stop window is still a saved image: hand it over.
    if stopped and (not future.done() or future.cancelled() or future.exception() is not None):
        return stopped
    try:
        response = future.result()
    except concurrent.futures.CancelledError:
        return ERR_STOPPED
    except GpuBusyForAnotherAccountError:
        return ERR_OTHER_ACCOUNT
    except HTTPException as exc:
        return _http_error(exc)
    if not response.images:
        return "Error: the image model returned no image."
    image = response.images[0]
    saved = _save_to_workdir(image, workdir, account)
    shown = " ".join(prompt.split())[:_PROMPT_ECHO_CHARS]
    text = (
        f"Generated a {image.width}x{image.height} image (seed {image.seed}) and saved it as "
        f"{saved['name']}. Prompt used: {shown}"
    )
    return (
        text + "\n__FILES__:" + json.dumps([saved]) + "\n__IMAGES__:" + json.dumps([saved["name"]])
    )


def run_generate_image(
    arguments: Any,
    *,
    workdir: str,
    cancel_event = None,
    timeout: Optional[float] = None,
) -> str:
    """Generate one image into ``workdir``; always a string, an ``Error: ...`` one on failure."""
    try:
        return _run(
            arguments if isinstance(arguments, dict) else {}, workdir, cancel_event, timeout
        )
    except Exception as exc:  # noqa: BLE001 - a tool result, never a crashed turn
        logger.warning("generate_image failed: %s", exc, exc_info = True)
        return f"Error: image generation failed: {exc}"

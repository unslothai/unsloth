# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``generate_image`` chat tool: one image on the image model already loaded in Unsloth Studio.

The tool runs on a chat worker thread. It hands the Images route's core to the server loop,
bound to the calling account, so the image lands in that account's gallery and goes through
the same access checks. It never loads or unloads a model.
"""

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


def generate_image_schema(status: dict) -> dict:
    """The OpenAI function tool for the loaded image model; negative_prompt only when it has one."""
    properties: dict[str, Any] = {
        "prompt": {"type": "string", "description": "The full image prompt."},
        "width": {
            "type": "integer",
            "description": "Width in pixels, 256 to 2752, a multiple of 16. Default 1024.",
        },
        "height": {
            "type": "integer",
            "description": "Height in pixels, 256 to 2752, a multiple of 16. Default 1024.",
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


def _snap_size(value: Any) -> int:
    return min(2752, max(256, int(round(int(value) / 16)) * 16))


def _build_request(arguments: dict, prompt: str, status: dict):
    from models.inference import DiffusionGenerateRequest

    # Only what the model supplied: an omitted steps/guidance must stay unset so the core
    # resolves the loaded model's own defaults.
    fields: dict[str, Any] = {"prompt": prompt}
    for key in ("width", "height"):
        if arguments.get(key) is not None:
            fields[key] = _snap_size(arguments[key])
    if arguments.get("seed") is not None:
        fields["seed"] = int(arguments["seed"])
    negative = arguments.get("negative_prompt")
    if status.get("supports_negative_prompt") and isinstance(negative, str) and negative.strip():
        fields["negative_prompt"] = negative.strip()
    return DiffusionGenerateRequest(**fields)


async def _generate(request):
    from core.inference import media_keepwarm
    from core.inference.gpu_arbiter import DIFFUSION
    from routes.inference import _generate_diffusion_image_core

    # The HTTP route is counted by the keep-warm middleware; this path never goes through it.
    await media_keepwarm.begin_request(DIFFUSION)
    try:
        return await _generate_diffusion_image_core(request)
    finally:
        media_keepwarm.end_request(DIFFUSION)


def _stop(future, account_id: str) -> None:
    """Stop the render on the GPU too: cancelling the task alone leaves the engine thread running."""
    from core.inference.diffusion_engine_router import cancel_generation_for_account

    deadline = time.monotonic() + _STOP_WAIT_S
    while not future.done() and time.monotonic() < deadline:
        try:
            cancel_generation_for_account(account_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("generate_image: engine cancel failed: %s", exc)
        concurrent.futures.wait([future], timeout = _POLL_S)
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

    future = asyncio.run_coroutine_threadsafe(arun_as(account, _generate(request)), loop)
    limit = _MAX_WAIT_S if timeout is None or timeout >= 9999 else max(1.0, float(timeout))
    deadline = time.monotonic() + limit
    while not future.done():
        if cancel_event is not None and cancel_event.is_set():
            _stop(future, account.account_id)
            return ERR_STOPPED
        if time.monotonic() >= deadline:
            _stop(future, account.account_id)
            return ERR_TIMEOUT
        concurrent.futures.wait([future], timeout = _POLL_S)
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

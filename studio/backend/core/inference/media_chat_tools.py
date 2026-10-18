# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat tools that render on the image or video model already loaded in Unsloth Studio.

Every registry that needs to know these tools (sentinel stripping, always-safe, dedupe
exemptions, the Anthropic allow-list) reads ``MEDIA_CHAT_TOOLS`` so the lists can't drift.
The module has no heavy imports at load time; the status helpers import lazily.
"""

from __future__ import annotations

import asyncio
from typing import Optional

MEDIA_CHAT_TOOLS = frozenset({"generate_image", "generate_video"})

_TOOL_TASKS = (("generate_image", "text-to-image"), ("generate_video", "text-to-video"))

# The server's event loop. The tools run on a worker thread inside the chat tool loop and
# hand the generation coroutine to this loop, never to a loop of their own.
_server_loop: Optional[asyncio.AbstractEventLoop] = None


def remember_server_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _server_loop
    _server_loop = loop


def server_loop() -> Optional[asyncio.AbstractEventLoop]:
    loop = _server_loop
    if loop is None or loop.is_closed():
        return None
    return loop


def media_chat_status(task: str) -> Optional[dict]:
    """The resident image or video model's status, or None when there is none this account may use."""
    from hub.services.models import account_access
    from routes.inference import _resident_media_status

    status = _resident_media_status(task)
    if status is None:
        return None
    if account_access.managed_account():
        modality = "diffusion" if task == "text-to-image" else "video"
        try:
            account_access.require_media_generation_access(status, modality)
        except Exception:  # noqa: BLE001 - a refused account is simply not offered the tool
            return None
    return status


def offered_media_chat_tools(enabled_tools) -> list[dict]:
    """Schemas for the media tools the caller asked for whose model is loaded and visible."""
    from core.inference.generate_image_tool import generate_image_schema, offers_txt2img
    from core.inference.generate_video_tool import generate_video_schema

    tools: list[dict] = []
    for name, task in _TOOL_TASKS:
        if name not in enabled_tools:
            continue
        status = media_chat_status(task)
        if status is None:
            continue
        if name == "generate_image":
            if offers_txt2img(status):
                tools.append(generate_image_schema(status))
        else:
            tools.append(generate_video_schema(status))
    return tools

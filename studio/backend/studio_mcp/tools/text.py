# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text tools: ``chat``."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict

from studio_mcp.outputs import ChatResult, Usage
from studio_mcp.tools import WRITES, integer, route_json, text


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    role: Literal["system", "user", "assistant"]
    content: str


def _same_model(requested: str, answered: str) -> bool:
    # A GGUF id may come back with its quant pinned (repo:Q4_K_M).
    return answered.lower() == requested.lower() or answered.lower().startswith(
        requested.lower() + ":"
    )


async def chat(
    messages: Optional[list[ChatTurn]] = None,
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    images: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
) -> ChatResult:
    """Chat with the model loaded in Studio and return its reply. Send either ``prompt`` (one user turn) or ``messages`` (the whole conversation), plus an optional ``system`` prompt. ``model`` names which loaded model should answer; without it the active one does. Load a model with load_model first. Studio's server-side tools are not used."""
    if (messages is None) == (prompt is None):
        raise ToolError("Send either prompt or messages, not both and not neither.")
    if images:
        raise ToolError("chat does not take images yet. Describe the image in text instead.")
    turns = (
        [{"role": "user", "content": prompt}]
        if prompt is not None
        else [t.model_dump() for t in messages]
    )
    if not turns:
        raise ToolError("messages is empty.")
    if system:
        turns.insert(0, {"role": "system", "content": system})
    body = {
        "model": model or "default",
        "messages": turns,
        "stream": False,
        # Our own id, so this run never matches a cancel aimed at another chat.
        "cancel_id": f"mcp-{uuid.uuid4().hex}",
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    payload = await route_json("POST", "/v1/chat/completions", json_body = body)
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ToolError("Studio returned no reply")
    message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    answered = text(payload.get("model"))
    note = None
    if model and answered and not _same_model(model, answered):
        note = f"{answered} answered; {model} is not the model that served this request."
    return ChatResult(
        text = message.get("content") if isinstance(message.get("content"), str) else "",
        model = answered,
        finish_reason = text(choices[0].get("finish_reason")),
        usage = Usage(
            prompt_tokens = integer(usage.get("prompt_tokens")),
            completion_tokens = integer(usage.get("completion_tokens")),
            total_tokens = integer(usage.get("total_tokens")),
        )
        if usage
        else None,
        note = note,
    )


def register_text(mcp: FastMCP) -> None:
    mcp.tool(chat, annotations = WRITES)

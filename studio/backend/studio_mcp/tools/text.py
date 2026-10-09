# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text tools: ``chat``, ``embed`` and ``system_one``."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, Optional, Union

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from studio_mcp.caller import current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.outputs import ChatResult, DecisionAnswer, EmbedResult, SystemOneResult, Usage
from studio_mcp.tools import READ_ONLY, WRITES, integer, number, route_json, text

MAX_EMBED_INPUTS = 2048
EMBED_DOWNLOAD_HINT = (
    "Download the embedding model in Studio first (Settings), or load an embedding GGUF with "
    "load_model(kind='llm') and call embed again."
)


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


async def embed(
    texts: Annotated[list[str], Field(min_length = 1, max_length = MAX_EMBED_INPUTS)],
    model: Optional[str] = None,
) -> EmbedResult:
    """Embed up to 2048 texts and return one vector per text, in order. Uses the embedding GGUF loaded in Studio when there is one, else Studio's configured embedding model. ``model`` names a specific one."""
    body = {"input": texts}
    if model:
        body["model"] = model
    payload = await route_json(
        "POST", "/v1/embeddings", json_body = body, hints = {409: EMBED_DOWNLOAD_HINT}
    )
    rows = payload.get("data") if isinstance(payload, dict) else None
    rows = sorted(
        (
            row
            for row in rows or []
            if isinstance(row, dict) and isinstance(row.get("embedding"), list)
        ),
        key = lambda row: integer(row.get("index")) or 0,
    )
    if len(rows) != len(texts):
        raise ToolError(f"Studio returned {len(rows)} embeddings for {len(texts)} texts")
    embeddings = [[float(value) for value in row["embedding"]] for row in rows]
    return EmbedResult(
        model = text(payload.get("model")),
        dimensions = len(embeddings[0]),
        embeddings = embeddings,
    )


MAX_DECISION_QUESTIONS = 64
DECISION_API_OFF_HINT = "Turn on the Decision API in Settings > API."


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    type: Literal["noul", "choice", "score"]
    instructions: Optional[Union[str, dict[str, Any], list[Any]]] = None
    criteria: Optional[Union[str, dict[str, Any], list[Any]]] = None


def _floats(values: Any) -> Optional[dict[str, float]]:
    if not isinstance(values, dict):
        return None
    return {str(k): float(v) for k, v in values.items() if number(v) is not None}


def _answer(answer: dict) -> DecisionAnswer:
    legend = answer.get("legend")
    return DecisionAnswer(
        type = str(answer.get("type")),
        noul = number(answer.get("noul")),
        choice = text(answer.get("choice")),
        score = number(answer.get("score")),
        confidence = number(answer.get("confidence")),
        probabilities = _floats(answer.get("probabilities")),
        legend = {
            str(k): v if isinstance(v, str) else json.dumps(v, ensure_ascii = False)
            for k, v in legend.items()
        }
        if isinstance(legend, dict)
        else None,
    )


async def system_one(
    state: Union[str, dict[str, Any], list[Any]],
    questions: Annotated[
        dict[str, DecisionQuestion], Field(min_length = 1, max_length = MAX_DECISION_QUESTIONS)
    ],
    images: Annotated[Optional[list[dict]], Field(max_length = 4)] = None,
    model: str = "default",
) -> SystemOneResult:
    """Ask Studio's decision model (SystemOne) typed questions about a state, given as text or JSON. ``questions`` maps a name you pick to {"type", "instructions", "criteria"}, for example {"urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}}. "noul" is yes or no and answers a probability; "choice" needs criteria mapping each option name to a description; "score" needs criteria listing 1 to 10 levels, lowest first. ``model`` "default" uses the model picked in Settings. The Decision API must be on (Settings > API)."""
    if images:
        raise ToolError("system_one does not take images yet. Describe them in the state instead.")
    body = {
        "state": state,
        "model": model,
        "questions": {name: q.model_dump(exclude_none = True) for name, q in questions.items()},
    }
    response = await forward(current_caller(), "POST", "/v1/systemone", json_body = body)
    payload = raise_for_route(response, hints = {404: DECISION_API_OFF_HINT})
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise ToolError("Studio returned no answers")
    return SystemOneResult(
        model = text(payload.get("model")),
        answers = {str(k): _answer(v) for k, v in answers.items() if isinstance(v, dict)},
        request_id = text(response.headers.get("x-typesafe-request-id")),
    )


def register_text(mcp: FastMCP) -> None:
    mcp.tool(chat, annotations = WRITES)
    mcp.tool(embed, annotations = READ_ONLY)
    mcp.tool(system_one, annotations = READ_ONLY)

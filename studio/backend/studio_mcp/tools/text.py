# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text tools: ``chat``, ``embed`` and ``system_one``."""

from __future__ import annotations

import json
import math
import uuid
from typing import Annotated, Any, Literal, Optional, Union

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from state.tool_policy import tools_force_disabled
from studio_mcp import profile
from studio_mcp.caller import current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.inputs import ImageInput, resolve_images
from studio_mcp.outputs import (
    ChatResult,
    DecisionAnswer,
    EmbedMatch,
    EmbedResult,
    SystemOneResult,
    Usage,
)
from studio_mcp.progress import with_heartbeat
from studio_mcp.tools import READ_ONLY, WRITES, integer, number, opt_text, present, route_json

MAX_EMBED_INPUTS = 2048
DEFAULT_TOP_K = 10
# The chat route takes 128 MiB of base64 images per request.
MAX_CHAT_IMAGE_BYTES = 128 * 1024 * 1024 * 3 // 4
EMBED_DOWNLOAD_HINT = (
    "Download the embedding model in Unsloth Studio first (Settings), or load an embedding GGUF with "
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


async def _attach_images(turns: list[dict], images: list[ImageInput]) -> None:
    last_user = next((turn for turn in reversed(turns) if turn["role"] == "user"), None)
    if last_user is None:
        raise ToolError("images need a user turn to go with.")
    urls = await resolve_images(
        current_caller(),
        images,
        total_cap = MAX_CHAT_IMAGE_BYTES,
        too_large = "The images together are larger than the 96 MiB a chat request takes",
    )
    parts = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    last_user["content"] = [{"type": "text", "text": last_user["content"]}, *parts]


@with_heartbeat("Waiting for the reply")
async def chat(
    messages: Optional[list[ChatTurn]] = None,
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    images: Optional[list[ImageInput]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    enable_thinking: Annotated[
        Optional[bool],
        Field(
            description = "Turn a thinking model's reasoning on or off; unset keeps the model's default."
        ),
    ] = None,
    reasoning_effort: Annotated[
        Optional[Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]],
        Field(
            description = "Reasoning depth on models with effort levels; unset keeps the model's default."
        ),
    ] = None,
    response_format: Annotated[
        Optional[dict[str, Any]],
        Field(
            description = 'Constrain the reply, e.g. {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}; a backend that cannot constrain it refuses the request.'
        ),
    ] = None,
    model: Optional[str] = None,
    cancel_id: Annotated[
        Optional[str], Field(min_length = 8, max_length = 64, pattern = r"^[A-Za-z0-9_-]+$")
    ] = None,
    ctx: Optional[Context] = None,
) -> ChatResult:
    """Chat with the model loaded in Unsloth Studio and return its reply.
    Send either ``prompt`` (one user turn) or ``messages`` (the whole conversation), plus an optional ``system`` prompt.
    ``images`` go with the last user turn and need a vision model.
    ``model`` names which loaded model should answer; without it the active one does.
    Load a model with load_model first.
    A thinking model spends tokens on reasoning before the reply, so leave ``max_tokens`` unset or give it a few hundred or more.
    ``enable_thinking`` turns that reasoning on or off, and ``reasoning_effort`` sets its depth on models with effort levels.
    ``response_format`` asks for JSON ({"type": "json_object"}) or a JSON schema; a backend that cannot constrain the reply refuses it.
    Unsloth Studio's server-side tools are not used.
    While the reply is being written, the call reports progress every few seconds.
    To be able to stop a long reply from another call, pass your own ``cancel_id`` (8-64 letters, digits, - or _) and call cancel(kind="chat", id=cancel_id).
    """
    if (messages is None) == (prompt is None):
        raise ToolError("Send either prompt or messages, not both and not neither.")
    turns = (
        [{"role": "user", "content": prompt}]
        if prompt is not None
        else [t.model_dump() for t in messages]
    )
    if not turns:
        raise ToolError("messages is empty.")
    if images:
        await _attach_images(turns, images)
    cancel_id = cancel_id or f"mcp-{uuid.uuid4().hex}"
    if system:
        turns.insert(0, {"role": "system", "content": system})
    body = {
        "model": model or "default",
        "messages": turns,
        "stream": False,
        # `unsloth studio run` turns tools on for an omitted field; MCP chat never runs them.
        "enable_tools": False,
        # Per-run id (the route scopes it to the caller's account), so only a cancel aimed at this chat stops it.
        "cancel_id": cancel_id,
        **present(
            max_tokens = max_tokens,
            temperature = temperature,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            response_format = response_format,
        ),
    }
    # Beats a CLI --enable-tools too; forward's task copies this context into the route.
    with tools_force_disabled():
        payload = await route_json("POST", "/v1/chat/completions", json_body = body)
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ToolError("Unsloth Studio returned no reply")
    message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    answered = opt_text(payload.get("model"))
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    finish_reason = opt_text(choices[0].get("finish_reason"))
    notes = []
    if model and answered and not _same_model(model, answered):
        notes.append(f"{answered} answered; {model} is not the model that served this request.")
    if finish_reason == "length":
        notes.append(
            # Thinking models can spend the whole budget before the visible reply starts.
            "The reply ran out of tokens before any text, likely spent on the model's reasoning; raise max_tokens."
            if not text.strip()
            else "The reply was cut off at max_tokens; raise max_tokens for the rest."
        )
    return ChatResult(
        text = text,
        model = answered,
        finish_reason = finish_reason,
        usage = Usage.from_route(usage) if usage else None,
        note = " ".join(notes) or None,
        cancel_id = cancel_id,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


async def embed(
    texts: Annotated[list[str], Field(min_length = 1, max_length = MAX_EMBED_INPUTS)],
    model: Optional[str] = None,
    query: Annotated[
        Optional[str],
        Field(
            description = "Rank texts by cosine similarity to this text and return the best matches instead of vectors."
        ),
    ] = None,
    top_k: Annotated[
        Optional[int],
        Field(
            ge = 1,
            le = MAX_EMBED_INPUTS,
            description = "How many matches to return with query (default 10)",
        ),
    ] = None,
) -> EmbedResult:
    """Embed up to 2048 texts and return one vector per text, in order.
    Uses the embedding GGUF loaded in Unsloth Studio when there is one, else Unsloth Studio's configured embedding model.
    ``model`` names a specific one; the result names the model that answered.
    Compare vectors with cosine similarity.
    With ``query`` (and at most 2047 texts), the call ranks the texts by cosine similarity to it and returns the ``top_k`` best (default 10) as ``matches``, with no vectors.
    """
    if top_k is not None and query is None:
        raise ToolError("top_k needs query.")
    if query is not None and len(texts) > MAX_EMBED_INPUTS - 1:
        raise ToolError(f"With query, texts may hold at most {MAX_EMBED_INPUTS - 1}.")
    inputs = texts if query is None else [query, *texts]
    body: dict[str, Any] = {"input": inputs}
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
    if len(rows) != len(inputs):
        raise ToolError(f"Unsloth Studio returned {len(rows)} embeddings for {len(inputs)} texts")
    embeddings = [[float(value) for value in row["embedding"]] for row in rows]
    model_name = opt_text(payload.get("model"))
    if query is None:
        return EmbedResult(model = model_name, dimensions = len(embeddings[0]), embeddings = embeddings)
    scored = sorted(
        ((_cosine(embeddings[0], vector), index) for index, vector in enumerate(embeddings[1:])),
        key = lambda pair: (-pair[0], pair[1]),
    )
    return EmbedResult(
        model = model_name,
        dimensions = len(embeddings[0]),
        matches = [
            EmbedMatch(index = index, score = score, text = texts[index])
            for score, index in scored[: top_k or DEFAULT_TOP_K]
        ],
    )


MAX_DECISION_QUESTIONS = 64
# The Decision API's own image limits: PNG or JPEG, 4 MiB each, 8 MiB together.
MAX_DECISION_IMAGES = 4
MAX_DECISION_IMAGE_BYTES = 4 * 1024 * 1024
MAX_DECISION_IMAGES_BYTES = 8 * 1024 * 1024
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
    return DecisionAnswer.from_route(
        answer,
        type = str(answer.get("type")),
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
    images: Annotated[Optional[list[ImageInput]], Field(max_length = MAX_DECISION_IMAGES)] = None,
    model: str = "default",
) -> SystemOneResult:
    """Ask Unsloth Studio's decision model (SystemOne) typed questions about a state, given as text or JSON. ``questions`` maps a name you pick to {"type", "instructions", "criteria"}, for example {"urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}}. "noul" is yes or no and answers a probability; "choice" needs criteria mapping each option name to a description; "score" needs criteria listing 1 to 10 levels, lowest first. ``images`` (at most 4, PNG or JPEG) need a Clef model. ``model`` "default" uses the model picked in Settings. The Decision API must be on (Settings > API)."""
    body: dict[str, Any] = {
        "state": state,
        "model": model,
        "questions": {name: q.model_dump(exclude_none = True) for name, q in questions.items()},
    }
    if images:
        body["images"] = await resolve_images(
            current_caller(),
            images,
            total_cap = MAX_DECISION_IMAGES_BYTES,
            too_large = "The images together are larger than 8 MiB",
            max_bytes = MAX_DECISION_IMAGE_BYTES,
            mimes = ("image/png", "image/jpeg"),
        )
    response = await forward(current_caller(), "POST", "/v1/systemone", json_body = body)
    payload = raise_for_route(response, hints = {404: DECISION_API_OFF_HINT})
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise ToolError("Unsloth Studio returned no answers")
    return SystemOneResult(
        model = opt_text(payload.get("model")),
        answers = {str(k): _answer(v) for k, v in answers.items() if isinstance(v, dict)},
        request_id = opt_text(response.headers.get("x-typesafe-request-id")),
    )


FACTS = {"chat": {profile.REENTERS_CHAT}}

TOOLS = ((chat, WRITES), (embed, READ_ONLY), (system_one, READ_ONLY))

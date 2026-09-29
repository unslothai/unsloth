# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Jev-compatible System One API: typed decisions (noul / choice / score) read off a Laya checkpoint."""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject, security
from core.systemone import catalog, laya_runtime
from utils import systemone_settings

MAX_QUESTIONS = 64
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
MAX_STATE_CHARS = 200_000
MAX_QUESTION_CHARS = 20_000
_TYPES = ("noul", "choice", "score")
MCP_PATH = "/mcp/decisions"

router = APIRouter()

# typing.Union, not `|`: this alias is evaluated at import, and Studio still starts on Python 3.9.
JSONContent = Union[str, dict[str, Any], list[Any]]


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    type: str
    instructions: Optional[JSONContent] = None
    criteria: Optional[JSONContent] = None


class SystemOneRequest(BaseModel):
    # Unknown fields refused, not dropped: an ignored OpenJev extension (`images`) answers a different question.
    model_config = ConfigDict(extra = "allow")

    state: JSONContent
    model: str
    questions: dict[str, QuestionIn] = Field(min_length = 1)


def _error(
    status: int,
    error_type: str,
    message: str,
    retry_after: float | None = None,
) -> HTTPException:
    headers = {"Retry-After": str(max(1, math.ceil(retry_after)))} if retry_after else None
    return HTTPException(
        status_code = status,
        detail = {"error_type": error_type, "message": message},
        headers = headers,
    )


def _validate(name: str, question: QuestionIn) -> None:
    if len(name) + len(json.dumps(question.model_dump(), ensure_ascii = False)) > MAX_QUESTION_CHARS:
        raise _error(
            422,
            "invalid_request_error",
            f"Question is longer than {MAX_QUESTION_CHARS} characters",
        )
    if question.type not in _TYPES:
        raise _error(
            422, "api_usage_error", f'Question "{name}" has unknown type "{question.type}"'
        )
    criteria = question.criteria
    if question.type == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
        ):
            raise _error(
                422,
                "invalid_request_error",
                f'Noul "{name}" criteria may only have "true" and "false"',
            )
    elif question.type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise _error(
                422, "invalid_request_error", f'Choice "{name}" needs criteria naming its options'
            )
        if len(criteria) > MAX_CHOICES:
            raise _error(
                422, "invalid_request_error", f'Choice "{name}" has more than {MAX_CHOICES} options'
            )
    elif not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
        raise _error(
            422,
            "invalid_request_error",
            f'Score "{name}" needs 1 to {MAX_SCORE_LEVELS} criteria levels',
        )
    if criteria is not None:
        entries = criteria.items() if isinstance(criteria, dict) else enumerate(criteria)
        for key, value in entries:
            if value is not None and not isinstance(value, (str, dict, list)):
                raise _error(
                    422,
                    "invalid_request_error",
                    f'Question "{name}" criteria[{key!r}] must be text, an object, an array or null',
                )


@router.post("/systemone")
def system_one(
    payload: SystemOneRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_enabled()
    if payload.model_extra:
        raise _error(
            400,
            "api_usage_error",
            f"Unsupported field(s): {', '.join(sorted(payload.model_extra))}",
        )
    checkpoint = catalog.resolve(payload.model)
    if checkpoint is None:
        raise _error(400, "api_usage_error", f"Unknown model: {payload.model}")
    from auth.authentication import request_admitted_without_credential

    # Same rule as the OpenAI routes: a keyless caller never downloads or swaps in another model.
    if checkpoint != catalog.default_checkpoint() and request_admitted_without_credential(request):
        raise _error(
            403,
            "permission_error",
            "Keyless requests can only use the configured Decision API model; send an API key to pick another.",
        )
    result = _decide(checkpoint, payload.state, payload.questions)
    return JSONResponse(result, headers = {"x-typesafe-request-id": str(uuid4())})


def _require_enabled() -> None:
    if not systemone_settings.get_enabled():
        raise _error(
            404,
            "api_usage_error",
            "The Decision API is off. The Studio owner can turn it on in Settings > API.",
        )


def _decide(
    checkpoint: catalog.Checkpoint, state: JSONContent, questions: dict[str, QuestionIn]
) -> dict:
    if not questions:
        raise _error(422, "invalid_request_error", "At least one question is required")
    state_chars = (
        len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii = False))
    )
    if state_chars > MAX_STATE_CHARS:
        raise _error(
            422, "invalid_request_error", f"State is longer than {MAX_STATE_CHARS} characters"
        )
    if len(questions) > MAX_QUESTIONS:
        raise _error(422, "invalid_request_error", f"At most {MAX_QUESTIONS} questions per request")
    for name, question in questions.items():
        _validate(name, question)

    try:
        result = laya_runtime.decide(
            checkpoint, state, {name: q.model_dump() for name, q in questions.items()}
        )
    except laya_runtime.Unavailable as exc:
        raise _error(exc.status, exc.error_type, exc.message, exc.retry_after) from None
    if result.pop("truncated"):
        raise _error(
            422,
            "invalid_request_error",
            "State and questions exceed the Laya context window. "
            "Shorten the state or use fewer/shorter criteria.",
        )
    return result


class DecisionsAvailability(Middleware):
    async def on_list_tools(self, context, call_next):
        if not systemone_settings.get_enabled():
            return []
        return await call_next(context)


decisions_mcp = FastMCP("Unsloth Decisions")
decisions_mcp.add_middleware(DecisionsAvailability())


@decisions_mcp.tool
def decide(state: JSONContent, questions: dict[str, QuestionIn]) -> dict[str, Any]:
    """Ask Unsloth's local Laya decision model typed questions about a state (text or JSON).
    questions maps a name you pick to {"type", "instructions", "criteria"}, for example
    {"urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}}.
    "noul" is yes/no and answers a probability; "choice" needs criteria mapping each option name
    to a description; "score" needs criteria listing 1 to 10 levels, lowest first."""
    try:
        _require_enabled()
        return _decide(catalog.default_checkpoint(), state, questions)
    except HTTPException as exc:
        raise ToolError(exc.detail["message"]) from None


class RequireStudioAuth:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            auth_scope = {**scope, "path": MCP_PATH, "root_path": ""}
            try:
                await get_current_subject(await security(Request(auth_scope)))
            except HTTPException as exc:
                response = JSONResponse({"detail": exc.detail}, exc.status_code, exc.headers)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

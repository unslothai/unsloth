# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Jev-compatible System One API: typed decisions (noul / choice / score) read off a Laya checkpoint."""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.systemone import catalog, laya_runtime
from utils import systemone_settings

MAX_QUESTIONS = 64
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
# Laya reads about a thousand tokens of state; far past that only costs tokenizer time under the model lock.
MAX_STATE_CHARS = 200_000
_TYPES = ("noul", "choice", "score")

router = APIRouter()

JSONContent = str | dict[str, Any] | list[Any]


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    type: str
    instructions: JSONContent | None = None
    criteria: JSONContent | None = None


class SystemOneRequest(BaseModel):
    # Unknown top-level fields are refused below, not dropped: an OpenJev extension such as
    # `images` silently ignored would answer a different question than the caller asked.
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
    if question.type not in _TYPES:
        raise _error(
            400, "api_usage_error", f'Question "{name}" has unknown type "{question.type}"'
        )
    criteria = question.criteria
    if question.type == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
        ):
            raise _error(
                400,
                "invalid_request_error",
                f'Noul "{name}" criteria may only have "true" and "false"',
            )
    elif question.type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise _error(
                400, "invalid_request_error", f'Choice "{name}" needs criteria naming its options'
            )
        if len(criteria) > MAX_CHOICES:
            raise _error(
                400, "invalid_request_error", f'Choice "{name}" has more than {MAX_CHOICES} options'
            )
    elif not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
        raise _error(
            400,
            "invalid_request_error",
            f'Score "{name}" needs 1 to {MAX_SCORE_LEVELS} criteria levels',
        )


@router.post("/systemone")
def system_one(payload: SystemOneRequest, current_subject: str = Depends(get_current_subject)):
    if not systemone_settings.get_enabled():
        raise _error(
            404,
            "api_usage_error",
            "The Decision API is off. The Studio owner can turn it on in Settings > API.",
        )
    if payload.model_extra:
        raise _error(
            400,
            "api_usage_error",
            f"Unsupported field(s): {', '.join(sorted(payload.model_extra))}",
        )
    checkpoint = catalog.resolve(payload.model)
    if checkpoint is None:
        raise _error(400, "api_usage_error", f"Unknown model: {payload.model}")
    state_chars = (
        len(payload.state)
        if isinstance(payload.state, str)
        else len(json.dumps(payload.state, ensure_ascii = False))
    )
    if state_chars > MAX_STATE_CHARS:
        raise _error(
            400, "invalid_request_error", f"State is longer than {MAX_STATE_CHARS} characters"
        )
    if len(payload.questions) > MAX_QUESTIONS:
        raise _error(400, "invalid_request_error", f"At most {MAX_QUESTIONS} questions per request")
    for name, question in payload.questions.items():
        _validate(name, question)

    questions = {name: q.model_dump() for name, q in payload.questions.items()}
    try:
        result = laya_runtime.decide(checkpoint, payload.state, questions)
    except laya_runtime.Unavailable as exc:
        raise _error(exc.status, exc.error_type, exc.message, exc.retry_after) from None
    truncated = result.pop("truncated")
    headers = {"X-Unsloth-State-Truncated": "1"} if truncated else None
    return JSONResponse(result, headers = headers)

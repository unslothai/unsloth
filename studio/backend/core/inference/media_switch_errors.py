# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refusals the media model auto-switch answers a generation request with.

Every one of these is an ordinary outcome rather than a fault. The switch never downloads,
never cuts a running generation short, and never outlives the ~100 second window Unsloth's
secure-mode tunnel gives an origin response, so where it cannot serve the request it says which
of those it hit and asks the caller to retry.

Each refusal is rendered in the error shape of the route that raised it: the OpenAI-compatible
body for ``/v1``, a plain string for the native media routes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

RETRY_AFTER_S = 15
MAX_LISTED_MODELS = 8

# stands for "entries are missing but their size is unknown", so the refusal reports no figure
UNSIZED_MISSING = -1

BUSY_MSG = (
    "The {kind} model is busy with another request, so it could not be switched in time. "
    "Retry once the current generation finishes."
)
LOADING_MSG = (
    "Loading '{model}'. It was not resident when this request arrived and is still coming up; "
    "retry shortly."
)
SLOW_MSG = (
    "Selecting the {kind} model took too long to answer inside this request. It is still "
    "being prepared; retry shortly."
)
UNVERIFIED_MSG = (
    "Could not verify that '{model}' is fully downloaded, so it was not switched in. "
    "Auto-switch never downloads; load it once from the {kind} page and retry."
)
EDIT_ONLY_MSG = (
    "'{model}' is an edit-only model: it requires an input image, which this endpoint cannot "
    "supply. Name a text-to-image model instead."
)
UNSIZED_MSG = (
    "'{model}' is missing some of its weights. Auto-switch never downloads, so load it once "
    "from the {kind} page and retry."
)
INCOMPLETE_MSG = (
    "'{model}' is not fully downloaded: about {gb:.1f} GB of its companion weights are missing. "
    "Auto-switch never downloads, so load it once from the {kind} page and retry."
)


def refuse(
    message: str,
    *,
    status_code: int,
    openai_errors: bool,
    code: str,
    retry_after: int = 0,
):
    """The HTTPException to raise, in the error shape the calling route publishes."""
    from fastapi import HTTPException
    from utils.api_errors import openai_error_body

    detail: Any = message
    if openai_errors:
        detail = openai_error_body(message, status = status_code, code = code, param = "model")
    return HTTPException(
        status_code = status_code,
        detail = detail,
        headers = {"Retry-After": str(retry_after)} if retry_after else None,
    )


def slow_switch(kind: str, openai_errors: bool):
    """The refusal for a switch that ran out of budget before it could answer."""
    return refuse(
        SLOW_MSG.format(kind = kind),
        status_code = 503,
        openai_errors = openai_errors,
        code = "model_loading",
        retry_after = RETRY_AFTER_S,
    )


def busy(kind: str, openai_errors: bool):
    """The refusal for a backend that stayed busy for the whole drain."""
    return refuse(
        BUSY_MSG.format(kind = kind),
        status_code = 409,
        openai_errors = openai_errors,
        code = "model_busy",
        retry_after = RETRY_AFTER_S,
    )


def incomplete_message(model_id: str, missing: int, kind: str) -> str:
    """The refusal text, which only quotes a size when the plan could size what it is missing."""
    if missing == UNSIZED_MISSING:
        return UNSIZED_MSG.format(model = model_id, kind = kind)
    return INCOMPLETE_MSG.format(model = model_id, gb = missing / 1e9, kind = kind)


def format_available(ids: list[str]) -> str:
    if not ids:
        return ""
    shown = ", ".join(ids[:MAX_LISTED_MODELS])
    extra = len(ids) - MAX_LISTED_MODELS
    return f"{shown} and {extra} more" if extra > 0 else shown


async def bounded(coro, deadline: float, *, kind: str, openai_errors: bool):
    """Await *coro* within the switch budget, refusing rather than outliving the response window.

    The worker thread behind a ``to_thread`` keeps running after this returns; what matters is
    that the request stops waiting on it, since the caller's connection is the thing on a clock.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        # shield() yields a Future, which has no close(); a bare coroutine has no cancel().
        if hasattr(coro, "cancel"):
            coro.cancel()
        else:
            coro.close()
        raise slow_switch(kind, openai_errors)
    try:
        return await asyncio.wait_for(coro, timeout = remaining)
    except asyncio.TimeoutError:
        raise slow_switch(kind, openai_errors)


async def probe(fn, arg: Optional[Any], deadline: float, *, kind: str, openai_errors: bool) -> bool:
    """Run a blocking busy probe off the loop, refusing rather than guessing on an overrun.

    A spent budget is not a busy backend: reporting one sends the caller after a generation
    that does not exist, where the slow-switch 503 says what actually happened.
    """
    remaining = deadline - time.monotonic()
    call = asyncio.to_thread(fn, arg) if arg is not None else asyncio.to_thread(fn)
    if remaining <= 0:
        call.close()
        raise slow_switch(kind, openai_errors)
    try:
        return bool(await asyncio.wait_for(call, timeout = remaining))
    except asyncio.TimeoutError:
        raise slow_switch(kind, openai_errors)


__all__ = [
    "BUSY_MSG",
    "EDIT_ONLY_MSG",
    "INCOMPLETE_MSG",
    "LOADING_MSG",
    "MAX_LISTED_MODELS",
    "RETRY_AFTER_S",
    "SLOW_MSG",
    "UNSIZED_MISSING",
    "UNSIZED_MSG",
    "UNVERIFIED_MSG",
    "bounded",
    "busy",
    "format_available",
    "incomplete_message",
    "probe",
    "refuse",
    "slow_switch",
]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Jev-compatible System One API: typed decisions (noul / choice / score) from a Laya checkpoint or a saved connection."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import time
from typing import Any, Optional, Union
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject, security
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import answers_decisions_only, validate_provider_base_url
from core.systemone import catalog, laya_runtime
from routes.provider_credentials import provider_config_guard, resolve_provider_api_key_or_400
from storage import providers_db
from utils import systemone_settings
from utils.account_context import OWNER, arun_as, run_as

MAX_QUESTIONS = 64
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
MAX_STATE_CHARS = 200_000
MAX_QUESTION_CHARS = 20_000
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGES_BYTES = 8 * 1024 * 1024
# What llama.cpp's mtmd decodes in-process (stb_image); WebP needs an ffmpeg it may not have.
_IMAGE_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
_TYPES = ("noul", "choice", "score")
MCP_PATH = "/mcp/decisions"
LISTED_MODELS_TTL = 300.0

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
    # OpenJev's image extension: base64 data URLs, served by a Clef model through llama.cpp.
    images: Optional[list[str]] = None


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
async def system_one(
    payload: SystemOneRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    # Settings and key checks read SQLite: keep them off the event loop.
    await asyncio.to_thread(_require_enabled)
    if payload.model_extra:
        raise _error(
            400,
            "api_usage_error",
            f"Unsupported field(s): {', '.join(sorted(payload.model_extra))}",
        )
    checkpoint = await asyncio.to_thread(catalog.resolve, payload.model)
    from auth.authentication import request_admitted_without_credential

    # Same rule as the OpenAI routes: a keyless caller never downloads or swaps in another model.
    # Checked before the unknown-model answer, so it cannot tell which owner fine-tunes exist.
    if checkpoint != await asyncio.to_thread(
        catalog.default_checkpoint
    ) and await asyncio.to_thread(request_admitted_without_credential, request):
        raise _error(
            403,
            "permission_error",
            "Keyless requests can only use the configured Decision API model; send an API key to pick another.",
        )
    if checkpoint is None:
        raise _error(400, "api_usage_error", f"Unknown model: {payload.model}")
    result = await _decide(checkpoint, payload.state, payload.questions, payload.images)
    headers = {"x-typesafe-request-id": str(uuid4())}
    if isinstance(checkpoint, catalog.Checkpoint):
        headers["x-unsloth-decision-backend"] = result.pop("_backend", "pytorch")
    return JSONResponse(result, headers = headers)


def _require_enabled() -> None:
    if not systemone_settings.get_enabled():
        raise _error(
            404,
            "api_usage_error",
            "The Decision API is off. The Studio owner can turn it on in Settings > API.",
        )


def _validate_images(images: list[str]) -> None:
    if len(images) > MAX_IMAGES:
        raise _error(422, "invalid_request_error", f"At most {MAX_IMAGES} images per request")
    total = 0
    for index, url in enumerate(images):
        header, comma, data = url.partition(",")
        kind = header.removeprefix("data:").removesuffix(";base64")
        if not comma or kind not in _IMAGE_TYPES or header != f"data:{kind};base64":
            raise _error(
                422,
                "invalid_request_error",
                f"images[{index}] must be a PNG or JPEG base64 data URL; remote URLs are not fetched",
            )
        if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise _error(422, "invalid_request_error", f"images[{index}] is larger than 4 MiB")
        try:
            raw = base64.b64decode(data, validate = True)
        except (binascii.Error, ValueError):
            raise _error(
                422, "invalid_request_error", f"images[{index}] is not valid base64"
            ) from None
        total += len(raw)
        if not raw or len(raw) > MAX_IMAGE_BYTES or total > MAX_IMAGES_BYTES:
            raise _error(
                422,
                "invalid_request_error",
                "Each image must be at most 4 MiB, and all images together at most 8 MiB",
            )
        if not _decodes(raw):
            raise _error(
                422, "invalid_request_error", f"images[{index}] is not a readable PNG or JPEG image"
            )


def _decodes(raw: bytes) -> bool:
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(raw)) as image:
            # By content, as stb_image reads it: a mislabelled JPEG still decodes.
            if image.format not in _IMAGE_TYPES.values():
                return False
            image.verify()
    except Exception:
        return False
    return True


async def _decide(
    checkpoint: catalog.Checkpoint | catalog.Connection,
    state: JSONContent,
    questions: dict[str, QuestionIn],
    images: list[str] | None = None,
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
    if images:
        _validate_images(images)
        if isinstance(checkpoint, catalog.Connection):
            raise _error(
                400, "api_usage_error", "Images are not forwarded to Decision API connections"
            )

    if isinstance(checkpoint, catalog.Connection):
        return await _connection_decide(
            checkpoint,
            state,
            {name: q.model_dump(exclude_unset = True) for name, q in questions.items()},
        )
    try:
        result = await run_in_threadpool(
            laya_runtime.decide,
            checkpoint,
            state,
            {name: q.model_dump() for name, q in questions.items()},
            images or None,
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


async def _connection_decide(
    connection: catalog.Connection, state: JSONContent, questions: dict[str, dict[str, Any]]
) -> dict:
    # The connection and its key are the owner's; the URL check and the request run as the caller,
    # so a managed account keeps its egress policy.
    provider_id = connection.provider_id
    config = await asyncio.to_thread(run_as, OWNER, providers_db.get_provider, provider_id)
    if config is None or catalog.decision_models(config) is None:
        raise _error(
            503,
            "api_usage_error",
            "The Decision API connection was removed. Pick another model in Settings > API.",
        )
    # OpenRouter's list is a cache, empty until refreshed; a decision connection's saved models are authoritative.
    if (
        answers_decisions_only(config["provider_type"], config.get("api_type"))
        and connection.model not in config["models"]
    ):
        raise _error(
            503,
            "api_usage_error",
            f"'{connection.model}' is no longer enabled on '{config['display_name']}'. Pick another model in Settings > API.",
        )
    if not config["is_enabled"]:
        raise _error(503, "api_usage_error", f"Connection '{config['display_name']}' is disabled.")
    try:
        base_url = await asyncio.to_thread(validate_provider_base_url, config["base_url"])
    except ValueError as exc:
        raise _error(503, "api_usage_error", str(exc)) from None
    api_key = await arun_as(OWNER, _connection_key(provider_id, config))
    client = ExternalProviderClient(config["provider_type"], base_url, api_key)
    try:
        result = await client.create_decision(connection.model, state, questions)
    except httpx.HTTPStatusError as exc:
        raise _upstream_error(config["display_name"], exc.response) from None
    except (httpx.HTTPError, ValueError) as exc:
        raise _error(
            502, "api_error", f"Couldn't reach '{config['display_name']}': {type(exc).__name__}"
        ) from None
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise _error(
            502, "api_error", f"'{config['display_name']}' did not answer in the System One format."
        )
    return result


async def _connection_key(provider_id: str, config: dict) -> str:
    routing_fields = ("provider_type", "base_url", "api_type", "is_enabled")
    async with provider_config_guard(provider_id):
        current = await asyncio.to_thread(providers_db.get_provider, provider_id)
        if current is None or any(current.get(f) != config.get(f) for f in routing_fields):
            raise _error(409, "api_usage_error", "The connection changed while starting; retry.")
        try:
            api_key = await asyncio.to_thread(
                resolve_provider_api_key_or_400, provider_id, None, prefer_saved_key = True
            )
        except HTTPException as exc:
            raise _error(500, "api_error", exc.detail) from None
        latest = await asyncio.to_thread(providers_db.get_provider, provider_id)
        if latest is None or any(latest.get(f) != current.get(f) for f in routing_fields):
            raise _error(409, "api_usage_error", "The connection changed while starting; retry.")
    return api_key


async def refresh_listed_decision_models() -> None:
    rows = await asyncio.to_thread(run_as, OWNER, providers_db.list_providers)
    for row in rows:
        if row["provider_type"] != "openrouter" or not row["is_enabled"]:
            continue
        key = (row["id"], row["updated_at"])
        cached = catalog.LISTED_DECISION_MODELS.get(key)
        if cached and cached[1] and time.monotonic() - cached[0] < LISTED_MODELS_TTL:
            continue
        try:
            api_key = await asyncio.to_thread(
                run_as,
                OWNER,
                resolve_provider_api_key_or_400,
                row["id"],
                None,
                prefer_saved_key = True,
            )
            client = ExternalProviderClient(row["provider_type"], row["base_url"], api_key, 10.0)
            models = await arun_as(OWNER, client.list_decision_models())
        except Exception:
            models = []
        catalog.LISTED_DECISION_MODELS[key] = (time.monotonic(), models)


def decision_model_objects() -> list[dict[str, Any]]:
    if not systemone_settings.get_enabled():
        return []
    names = (
        "default",
        *(() if systemone_settings.runtime_unavailable_reason() else catalog.CHECKPOINTS),
    )
    return [
        {
            "id": name,
            "object": "model",
            "owned_by": "unsloth",
            "architecture": {
                "input_modalities": laya_runtime.input_modalities(catalog.resolve(name)),
                "output_modalities": ["decisions"],
            },
        }
        for name in names
    ]


def _upstream_error(name: str, response: httpx.Response) -> HTTPException:
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = (body.get("detail") or body.get("error")) if isinstance(body, dict) else None
    if isinstance(detail, str):
        detail = {"message": detail}
    elif not isinstance(detail, dict):
        detail = {}
    message = str(detail.get("message") or f"'{name}' answered HTTP {response.status_code}.")
    if response.status_code in (429, 503, 529):
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except ValueError:
            retry_after = None
        error_type = str(detail.get("error_type") or "overloaded")
        return _error(response.status_code, error_type, message, retry_after)
    return _error(502, "api_error", message)


class DecisionsAvailability(Middleware):
    async def on_list_tools(self, context, call_next):
        if not await asyncio.to_thread(systemone_settings.get_enabled):
            return []
        return await call_next(context)


decisions_mcp = FastMCP("Unsloth Decisions")
decisions_mcp.add_middleware(DecisionsAvailability())


@decisions_mcp.tool
async def decide(state: JSONContent, questions: dict[str, QuestionIn]) -> dict[str, Any]:
    """Ask Unsloth's decision model typed questions about a state (text or JSON).
    questions maps a name you pick to {"type", "instructions", "criteria"}, for example
    {"urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}}.
    "noul" is yes/no and answers a probability; "choice" needs criteria mapping each option name
    to a description; "score" needs criteria listing 1 to 10 levels, lowest first."""
    try:
        await asyncio.to_thread(_require_enabled)
        checkpoint = await asyncio.to_thread(catalog.default_checkpoint)
        result = await _decide(checkpoint, state, questions)
        result.pop("_backend", None)
        return result
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

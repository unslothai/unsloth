# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-checkpoint preview endpoints: /p/{run}[/{checkpoint}]/v1/..."""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from loggers import get_logger

from auth.authentication import get_current_subject
from auth.storage import DEFAULT_ADMIN_USERNAME
from models.inference import ChatCompletionRequest, LoadRequest
from routes.inference import load_model, openai_chat_completions
from utils.models.checkpoints import list_preview_targets, resolve_preview_checkpoint

logger = get_logger(__name__)

router = APIRouter()

# Public (no key) so a shared link opens in any browser; resolve_preview_checkpoint
# pins `run` under outputs_root, so this only ever serves your own checkpoints.


def _resolve_or_4xx(run: str, checkpoint: str | None):
    try:
        return resolve_preview_checkpoint(run, checkpoint)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))


async def _serve_chat(
    run: str,
    checkpoint: str | None,
    payload: ChatCompletionRequest,
    request: Request,
):
    path = _resolve_or_4xx(run, checkpoint)
    await load_model(LoadRequest(model_path = str(path)), request, DEFAULT_ADMIN_USERNAME)
    return await openai_chat_completions(payload, request, DEFAULT_ADMIN_USERNAME)


@router.get("")
async def list_previews(request: Request, current_subject: str = Depends(get_current_subject)):
    base = str(request.base_url)
    previews = []
    for target in list_preview_targets():
        previews.append({**target, "url": f"{base}p/{target['ref']}/v1"})
    return {"object": "list", "data": previews}


@router.post("/{run}/v1/chat/completions")
async def preview_chat_latest(
    run: str,
    payload: ChatCompletionRequest,
    request: Request,
):
    return await _serve_chat(run, None, payload, request)


@router.post("/{run}/{checkpoint}/v1/chat/completions")
async def preview_chat_checkpoint(
    run: str,
    checkpoint: str,
    payload: ChatCompletionRequest,
    request: Request,
):
    return await _serve_chat(run, checkpoint, payload, request)


def _models_response(run: str, checkpoint: str | None):
    path = _resolve_or_4xx(run, checkpoint)
    model_id = run if not checkpoint else f"{run}/{checkpoint}"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(path.stat().st_mtime),
                "owned_by": "unsloth-studio",
            }
        ],
    }


@router.get("/{run}/v1/models")
async def preview_models_latest(run: str):
    return _models_response(run, None)


@router.get("/{run}/{checkpoint}/v1/models")
async def preview_models_checkpoint(run: str, checkpoint: str):
    return _models_response(run, checkpoint)


# Self-contained public page (not the auth-gated SPA); only the title is interpolated.
_PREVIEW_PAGE_HTML = (
    Path(__file__).resolve().parent.parent / "assets" / "preview_page.html"
).read_text(encoding = "utf-8")

_PREVIEW_PAGE_CSP = (
    "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'"
)


def _preview_page(run: str, checkpoint: str | None) -> HTMLResponse:
    _resolve_or_4xx(run, checkpoint)
    title = run if not checkpoint else f"{run}/{checkpoint}"
    page = _PREVIEW_PAGE_HTML.replace("__TITLE__", html.escape(title))
    return HTMLResponse(page, headers = {"Content-Security-Policy": _PREVIEW_PAGE_CSP})


@router.get("/{run}", response_class = HTMLResponse)
async def preview_page_latest(run: str):
    return _preview_page(run, None)


@router.get("/{run}/{checkpoint}", response_class = HTMLResponse)
async def preview_page_checkpoint(run: str, checkpoint: str):
    return _preview_page(run, checkpoint)

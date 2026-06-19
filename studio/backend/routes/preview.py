# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-checkpoint preview endpoints: /p/{run}[/{checkpoint}]/v1/..."""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
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

# Serialize load+generate: the backend holds one model at a time, so concurrent
# previews for different checkpoints could otherwise run against the wrong one.
_preview_lock = asyncio.Lock()


def _resolve_or_4xx(run: str, checkpoint: str | None):
    try:
        return resolve_preview_checkpoint(run, checkpoint)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))


def _without_tools(payload: ChatCompletionRequest) -> ChatCompletionRequest:
    # Tools run shell/python on the host; this surface is public, so force them off.
    return payload.model_copy(update = {
        "tools": None,
        "enable_tools": False,
        "enabled_tools": None,
        "bypass_permissions": False,
        "openai_code_exec_container_id": None,
        "anthropic_code_exec_container_id": None,
    })


async def _serve_chat(
    run: str,
    checkpoint: str | None,
    payload: ChatCompletionRequest,
    request: Request,
):
    path = _resolve_or_4xx(run, checkpoint)
    payload = _without_tools(payload)
    async with _preview_lock:
        await load_model(LoadRequest(model_path = str(path)), request, DEFAULT_ADMIN_USERNAME)
        return await openai_chat_completions(payload, request, DEFAULT_ADMIN_USERNAME)


@router.get("")
async def list_previews(request: Request, current_subject: str = Depends(get_current_subject)):
    base = str(request.base_url)
    previews = []
    for target in list_preview_targets():
        ref = quote(target["ref"], safe = "/")
        previews.append({**target, "url": f"{base}p/{ref}/v1"})
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


# Serve the page's logo/fonts here too; the frontend static mount is absent in
# --api-only mode (Tauri), where they would otherwise 404.
_FRONTEND_DIST = (Path(__file__).resolve().parents[2] / "frontend" / "dist").resolve()
_PREVIEW_ASSET_MEDIA_TYPES = {
    ".png": "image/png",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


@router.get("/_assets/{asset_path:path}")
async def preview_asset(asset_path: str):
    target = (_FRONTEND_DIST / asset_path).resolve()
    media_type = _PREVIEW_ASSET_MEDIA_TYPES.get(target.suffix.lower())
    if media_type is None or _FRONTEND_DIST not in target.parents or not target.is_file():
        raise HTTPException(status_code = 404, detail = "Not found")
    return FileResponse(target, media_type = media_type)


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

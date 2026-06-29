# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-checkpoint preview endpoints: /p/{run}[/{checkpoint}]/v1/..."""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from loggers import get_logger

from auth.authentication import get_current_subject
from auth.storage import DEFAULT_ADMIN_USERNAME
from models.inference import ChatCompletionRequest, LoadRequest
from routes.inference import load_model, openai_chat_completions
from state.tool_policy import tools_force_disabled
from utils.models.checkpoints import list_preview_targets, resolve_preview_checkpoint

logger = get_logger(__name__)

router = APIRouter()

# Public (no key); resolve_preview_checkpoint pins `run` under outputs_root.
# One model loads at a time, so serialize load+generate across previews.
_preview_lock = asyncio.Lock()


def _resolve_or_4xx(run: str, checkpoint: str | None):
    try:
        return resolve_preview_checkpoint(run, checkpoint)
    except ValueError as exc:
        # Detail can carry the absolute install path on a symlink escape; log it,
        # return a generic message on this public route.
        logger.warning("preview path rejected: %s", exc)
        raise HTTPException(status_code = 400, detail = "Invalid run or checkpoint")
    except FileNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc))


def _sanitize_preview_payload(
    payload: ChatCompletionRequest, is_lora: bool
) -> ChatCompletionRequest:
    # Public surface: strip tools/MCP + provider routing (no host code / open proxy).
    # Normalize use_adapter (never trust the caller): pin True for LoRA, None for
    # merged. _apply_adapter_state mutates the shared model without restoring, so an
    # unpinned `false` would persist to later visitors who omit the field.
    return payload.model_copy(
        update = {
            "tools": None,
            "enable_tools": False,
            "enabled_tools": None,
            "mcp_enabled": False,
            "bypass_permissions": False,
            "confirm_tool_calls": False,
            "session_id": None,
            "rag_scope": None,
            "openai_code_exec_container_id": None,
            "anthropic_code_exec_container_id": None,
            "provider_id": None,
            "provider_type": None,
            "external_model": None,
            "encrypted_api_key": None,
            "provider_base_url": None,
            "use_adapter": True if is_lora else None,
        }
    )


async def _unlock_after(body_iterator):
    # Hold the lock until the stream drains so another checkpoint can't swap mid-stream.
    try:
        async for chunk in body_iterator:
            yield chunk
    finally:
        _preview_lock.release()


async def _serve_chat(
    run: str, checkpoint: str | None, payload: ChatCompletionRequest, request: Request
):
    path = _resolve_or_4xx(run, checkpoint)
    is_lora = (path / "adapter_config.json").exists()
    payload = _sanitize_preview_payload(payload, is_lora)
    await _preview_lock.acquire()
    keep_locked = False
    try:
        await load_model(LoadRequest(model_path = str(path)), request, DEFAULT_ADMIN_USERNAME)
        # Beats a process-wide `--enable-tools` (enable_tools=False alone wouldn't).
        with tools_force_disabled():
            response = await openai_chat_completions(payload, request, DEFAULT_ADMIN_USERNAME)
        if isinstance(response, StreamingResponse):
            response.body_iterator = _unlock_after(response.body_iterator)
            keep_locked = True
        return response
    finally:
        if not keep_locked:
            _preview_lock.release()


@router.get("")
async def list_previews(request: Request, current_subject: str = Depends(get_current_subject)):
    base = str(request.base_url)
    previews = []
    for target in list_preview_targets():
        ref = quote(target["ref"], safe = "/")
        previews.append({**target, "url": f"{base}p/{ref}/v1"})
    return {"object": "list", "data": previews}


@router.post("/{run}/v1/chat/completions")
async def preview_chat_latest(run: str, payload: ChatCompletionRequest, request: Request):
    return await _serve_chat(run, None, payload, request)


@router.post("/{run}/{checkpoint}/v1/chat/completions")
async def preview_chat_checkpoint(
    run: str, checkpoint: str, payload: ChatCompletionRequest, request: Request
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


# Serve logo/fonts here too: the SPA static mount is absent in --api-only (Tauri).
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
    if media_type is None or not target.is_relative_to(_FRONTEND_DIST) or not target.is_file():
        raise HTTPException(status_code = 404, detail = "Not found")
    return FileResponse(target, media_type = media_type)


# Self-contained public page; only the title is interpolated.
_PREVIEW_PAGE_HTML = (
    Path(__file__).resolve().parent.parent / "assets" / "preview_page.html"
).read_text(encoding = "utf-8")

_PREVIEW_PAGE_CSP = (
    "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self'; font-src 'self'; connect-src 'self'; base-uri 'none'"
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

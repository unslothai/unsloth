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
from routes.inference import (
    disable_openai_auto_switch_for_request,
    load_model,
    openai_chat_completions,
)
from state.tool_policy import tools_force_disabled
from utils.client_ip import client_ip
from utils.models.checkpoints import list_preview_targets, resolve_preview_checkpoint
from utils.preview_rate_limit import check_rate_limit
from utils.preview_sharing_settings import get_preview_sharing_enabled
from utils.preview_token import sign_preview_ref, verify_preview_ref

logger = get_logger(__name__)

router = APIRouter()

# A shared preview link is a public bearer capability; cap per-request generation
# so a single call can't tie up the (serialized) preview GPU indefinitely.
_PREVIEW_MAX_OUTPUT_TOKENS = 1024

# Capability-gated (signed ref required); resolve_preview_checkpoint pins `run`
# under outputs_root. One model loads at a time, so serialize load+generate.
_preview_lock = asyncio.Lock()


def _extract_token(request: Request) -> str | None:
    """Capability token from the ``?k=`` query (browser link + preview page) or an
    ``Authorization: Bearer`` header (OpenAI-compatible clients using it as api_key)."""
    token = request.query_params.get("k")
    if token:
        return token
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def _verify_or_404(run: str, checkpoint: str | None, request: Request) -> None:
    """Require a valid preview capability BEFORE any checkpoint resolve / model load.

    Missing or invalid tokens get a generic 404 -- identical to a non-existent ref --
    so the public surface never confirms whether a run/checkpoint exists. When an
    admin has switched public sharing off, every public request 404s regardless of
    token.

    Verify the (cheap, no-I/O) capability first: an unauthenticated caller with a
    bad/missing token is rejected without the kill-switch DB read, so spamming
    ``/p/...`` can't be used as an unbounded settings-DB sink, and the response is
    identical whether or not sharing is enabled (no on/off oracle).
    """
    ref = run if not checkpoint else f"{run}/{checkpoint}"
    if not verify_preview_ref(ref, _extract_token(request)):
        raise HTTPException(status_code = 404, detail = "Not found")
    if not get_preview_sharing_enabled():
        raise HTTPException(status_code = 404, detail = "Not found")


def _enforce_rate_limit(request: Request) -> None:
    """Throttle the GPU-backed preview chat per client IP (429 on exceed)."""
    retry_after = check_rate_limit(client_ip(request))
    if retry_after:
        raise HTTPException(
            status_code = 429,
            detail = "Too many preview requests. Please slow down.",
            headers = {"Retry-After": str(retry_after)},
        )


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
    #
    # Cap generation cost on this public, GPU-backed surface. Derive one effective
    # limit (mirroring _effective_max_tokens: max_completion_tokens wins, else the
    # legacy max_tokens) and pin BOTH fields to it, so a caller's lower limit is
    # honored and neither field can exceed the ceiling.
    requested = (
        payload.max_completion_tokens
        if payload.max_completion_tokens is not None
        else payload.max_tokens
    )
    capped_max_tokens = (
        min(requested, _PREVIEW_MAX_OUTPUT_TOKENS)
        if requested is not None
        else _PREVIEW_MAX_OUTPUT_TOKENS
    )
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
            "max_tokens": capped_max_tokens,
            "max_completion_tokens": capped_max_tokens,
            "n": 1,
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
    # Preview always serves the pinned checkpoint it loads below; a public caller's
    # `model` field must never trigger an OpenAI auto-switch to another GGUF.
    disable_openai_auto_switch_for_request(getattr(request, "scope", None))
    await _preview_lock.acquire()
    keep_locked = False
    try:
        await load_model(
            LoadRequest(model_path = str(path)), request, DEFAULT_ADMIN_USERNAME
        )
        # Beats a process-wide `--enable-tools` (enable_tools=False alone wouldn't).
        with tools_force_disabled():
            response = await openai_chat_completions(
                payload, request, DEFAULT_ADMIN_USERNAME
            )
        if isinstance(response, StreamingResponse):
            response.body_iterator = _unlock_after(response.body_iterator)
            keep_locked = True
        return response
    finally:
        if not keep_locked:
            _preview_lock.release()


@router.get("")
async def list_previews(
    request: Request, current_subject: str = Depends(get_current_subject)
):
    base = str(request.base_url)
    sharing_on = get_preview_sharing_enabled()
    previews = []
    for target in list_preview_targets():
        ref = quote(target["ref"], safe = "/")
        # Mint the capability for the authenticated owner: ``key`` for OpenAI
        # clients (Bearer / api_key), ``share_url`` for the browser link. When
        # public sharing is off, every public /p request 404s, so don't hand out
        # dead credentials -- omit the capability and signal the disabled state.
        token = sign_preview_ref(target["ref"]) if sharing_on else None
        previews.append(
            {
                **target,
                "url": f"{base}p/{ref}/v1",
                "key": token,
                "share_url": f"{base}p/{ref}?k={token}" if token else None,
            }
        )
    return {"object": "list", "data": previews, "sharing_enabled": sharing_on}


@router.post("/{run}/v1/chat/completions")
async def preview_chat_latest(
    run: str, payload: ChatCompletionRequest, request: Request
):
    _verify_or_404(run, None, request)
    _enforce_rate_limit(request)
    return await _serve_chat(run, None, payload, request)


@router.post("/{run}/{checkpoint}/v1/chat/completions")
async def preview_chat_checkpoint(
    run: str, checkpoint: str, payload: ChatCompletionRequest, request: Request
):
    _verify_or_404(run, checkpoint, request)
    _enforce_rate_limit(request)
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


# The models/page GET routes only stat the checkpoint dir (no GPU), so they are
# token-gated but not rate-limited; only the GPU-backed chat path is throttled.
@router.get("/{run}/v1/models")
async def preview_models_latest(run: str, request: Request):
    _verify_or_404(run, None, request)
    return _models_response(run, None)


@router.get("/{run}/{checkpoint}/v1/models")
async def preview_models_checkpoint(run: str, checkpoint: str, request: Request):
    _verify_or_404(run, checkpoint, request)
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
    if (
        media_type is None
        or not target.is_relative_to(_FRONTEND_DIST)
        or not target.is_file()
    ):
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
    # no-referrer: the capability token rides in the query string, so keep it out
    # of the Referer header on any outbound navigation.
    return HTMLResponse(
        page,
        headers = {
            "Content-Security-Policy": _PREVIEW_PAGE_CSP,
            "Referrer-Policy": "no-referrer",
        },
    )


@router.get("/{run}", response_class = HTMLResponse)
async def preview_page_latest(run: str, request: Request):
    _verify_or_404(run, None, request)
    return _preview_page(run, None)


@router.get("/{run}/{checkpoint}", response_class = HTMLResponse)
async def preview_page_checkpoint(run: str, checkpoint: str, request: Request):
    _verify_or_404(run, checkpoint, request)
    return _preview_page(run, checkpoint)

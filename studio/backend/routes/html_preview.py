# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTML preview route for assistant ```html fences.

srcdoc / data: / blob: iframes inherit the host CSP (script-src 'self')
per CSP3, so inline scripts and onclick handlers are silently blocked.
Serving the snippet from a same-origin URL with an overriding response
CSP is the only way to unlock interactive HTML.

Security:
- POST is auth-gated.
- GET is NOT auth-gated -- iframe subresource loads do not carry the
  Authorization bearer, so the URL token (192 bits, secrets.token_urlsafe(24))
  is the authorisation. Never persisted, wiped on TTL expiry.
- Response CSP: default-src 'none' + script-src 'unsafe-inline'.
  Scripts run; network egress (connect-src, frame-src, worker-src) blocked.
- iframe sandbox stays "allow-scripts allow-modals allow-popups" with NO
  allow-same-origin, so preview JS cannot reach window.parent.
- Size and live-entry caps bound per-process memory.
"""

from __future__ import annotations

import secrets
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Backend on sys.path for standalone load (matches routes/export.py).
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from auth.authentication import get_current_subject  # noqa: E402

router = APIRouter()


# Knobs (module-level so tests can monkeypatch).
MAX_HTML_PREVIEW_BYTES = 1_000_000  # 1 MiB; snippets, not full SPAs.
PREVIEW_TTL_SECONDS = 10 * 60       # 10 min interaction window.
MAX_LIVE_PREVIEWS = 256             # Per-process cap; oldest evicted first.

_PREVIEWS: dict[str, tuple[float, str]] = {}


def _sweep_expired(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    expired = [k for k, (t, _) in _PREVIEWS.items() if now - t > PREVIEW_TTL_SECONDS]
    for k in expired:
        _PREVIEWS.pop(k, None)


def _evict_overflow() -> None:
    if len(_PREVIEWS) <= MAX_LIVE_PREVIEWS:
        return
    sorted_keys = sorted(_PREVIEWS, key = lambda k: _PREVIEWS[k][0])
    for k in sorted_keys[: len(_PREVIEWS) - MAX_LIVE_PREVIEWS]:
        _PREVIEWS.pop(k, None)


def _build_html_doc(source: str) -> str:
    # <base target="_blank"> so <a> links open a new tab instead of
    # navigating the iframe away from the preview.
    return "<!doctype html>" '<base target="_blank">' + source


_PREVIEW_CSP = "; ".join(
    (
        "default-src 'none'",
        # The reason this route exists -- host CSP does not allow inline.
        "script-src 'unsafe-inline'",
        "style-src 'unsafe-inline'",
        # data: / blob: only -- no remote img beacon.
        "img-src data: blob:",
        "media-src data: blob:",
        "font-src data:",
        "connect-src 'none'",
        "worker-src 'none'",
        "frame-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        # Same-origin embedders only; also overrides X-Frame-Options.
        "frame-ancestors 'self'",
    )
)


class HtmlPreviewCreate(BaseModel):
    source: str = Field(..., max_length = MAX_HTML_PREVIEW_BYTES)


class HtmlPreviewCreateResponse(BaseModel):
    url: str
    expires_in_seconds: int


@router.post("", response_model = HtmlPreviewCreateResponse)
async def create_html_preview(
    payload: HtmlPreviewCreate,
    current_subject: str = Depends(get_current_subject),
) -> HtmlPreviewCreateResponse:
    """Stash HTML; return a same-origin token URL (the only auth for GET)."""
    _sweep_expired()
    source = payload.source
    if not isinstance(source, str):  # defensive; pydantic already enforces.
        raise HTTPException(status_code = 400, detail = "source must be a string")
    if len(source) > MAX_HTML_PREVIEW_BYTES:
        raise HTTPException(status_code = 413, detail = "HTML preview too large")
    token = secrets.token_urlsafe(24)
    _PREVIEWS[token] = (time.monotonic(), source)
    _evict_overflow()
    return HtmlPreviewCreateResponse(
        url = f"/api/preview/html/{token}",
        expires_in_seconds = PREVIEW_TTL_SECONDS,
    )


@router.get("/{preview_id}", response_class = HTMLResponse)
async def get_html_preview(preview_id: str) -> HTMLResponse:
    """Serve a stashed snippet with overriding CSP. Unauth: token IS the auth."""
    _sweep_expired()
    item = _PREVIEWS.get(preview_id)
    if item is None:
        raise HTTPException(status_code = 404, detail = "Preview expired or not found")
    _, source = item
    return HTMLResponse(
        content = _build_html_doc(source),
        headers = {
            "Content-Security-Policy": _PREVIEW_CSP,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            # Override the global DENY default so the host chat can iframe us.
            "X-Frame-Options": "SAMEORIGIN",
        },
    )

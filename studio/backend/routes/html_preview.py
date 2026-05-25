# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTML preview route for assistant ```html fences.

Reason this route exists: when the chat renderer embeds the assistant's
HTML via a ``srcdoc`` iframe, Chromium inherits the embedder CSP
(``script-src 'self'``), so inline scripts and ``onclick`` handlers are
silently blocked. Serving the same HTML from a same-origin URL with an
overriding response-header CSP is the only way to let assistant-generated
interactive HTML actually run while keeping the surrounding Studio CSP
strict.

Security shape:

* POST is auth-gated (``get_current_subject``). Only an authenticated
  caller can stash HTML into the in-memory store.
* GET is intentionally NOT auth-gated -- browsers do not attach the
  Authorization bearer to iframe subresource loads, so we instead make
  the URL itself the secret: ``secrets.token_urlsafe(24)`` (192 bits of
  entropy). The token leaves the server only in the POST response and
  is then placed into the iframe ``src`` by the requesting page. It is
  never persisted to disk, never logged, and is wiped on TTL expiry.
* The response CSP is ``default-src 'none'`` + ``script-src
  'unsafe-inline'`` so the preview is sandboxed from the network but
  inline scripts and event-handler attributes execute as intended.
* The iframe still has ``sandbox="allow-scripts allow-modals
  allow-popups"`` (no ``allow-same-origin``), so even though the URL
  is same-origin the document is treated as a unique opaque origin
  for SOP purposes -- script in the preview cannot reach
  ``window.parent`` storage, cookies, or DOM.
* A size cap and TTL cap bound the in-memory footprint per Studio
  process.
"""

from __future__ import annotations

import secrets
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Backend root on sys.path so ``auth`` imports resolve when this module
# is loaded standalone (matches the pattern used by routes/export.py).
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from auth.authentication import get_current_subject  # noqa: E402

router = APIRouter()


# ---------------------------------------------------------------------------
# Knobs (module-level so tests can monkeypatch).
# ---------------------------------------------------------------------------

# 1 MiB cap; assistant HTML previews are short snippets, not full SPAs.
MAX_HTML_PREVIEW_BYTES = 1_000_000

# 10 minutes. Long enough for a user to interact with the preview, short
# enough that a forgotten tab does not pin the entry.
PREVIEW_TTL_SECONDS = 10 * 60

# Defensive cap so a runaway producer cannot exhaust the worker. New POSTs
# evict the oldest entries past this watermark. Per-process, in-memory only.
MAX_LIVE_PREVIEWS = 256


_PREVIEWS: dict[str, tuple[float, str]] = {}


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _sweep_expired(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    expired = [k for k, (t, _) in _PREVIEWS.items() if now - t > PREVIEW_TTL_SECONDS]
    for k in expired:
        _PREVIEWS.pop(k, None)


def _evict_overflow() -> None:
    if len(_PREVIEWS) <= MAX_LIVE_PREVIEWS:
        return
    # Evict oldest first.
    sorted_keys = sorted(_PREVIEWS, key = lambda k: _PREVIEWS[k][0])
    for k in sorted_keys[: len(_PREVIEWS) - MAX_LIVE_PREVIEWS]:
        _PREVIEWS.pop(k, None)


def _build_html_doc(source: str) -> str:
    # ``<base target="_blank">`` mirrors the srcdoc fallback so any ``<a>``
    # without an explicit target opens in a new tab rather than navigating
    # the iframe (which would be UX-confusing).
    return "<!doctype html>" '<base target="_blank">' + source


_PREVIEW_CSP = "; ".join(
    (
        "default-src 'none'",
        # ``script-src 'unsafe-inline'`` enables BOTH ``<script>`` blocks and
        # ``onclick``-style attribute handlers. This is the entire reason the
        # route exists -- the host page's ``script-src 'self'`` does not.
        "script-src 'unsafe-inline'",
        "style-src 'unsafe-inline'",
        # ``data:`` / ``blob:`` only, NOT remote http(s). Inline JS cannot
        # exfiltrate by fetching a remote pixel since ``connect-src 'none'``
        # blocks fetch/XHR, but stripping remote ``img-src`` removes the
        # other classic beacon vector too.
        "img-src data: blob:",
        "media-src data: blob:",
        "font-src data:",
        "connect-src 'none'",
        "worker-src 'none'",
        "frame-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        # Restrict who can embed THIS preview. The Studio host page is
        # same-origin and is the only legitimate embedder. ``frame-ancestors
        # 'self'`` also overrides any global X-Frame-Options on modern
        # browsers, so a third-party site cannot iframe a leaked preview URL.
        "frame-ancestors 'self'",
    )
)


# ---------------------------------------------------------------------------
# Request / response models.
# ---------------------------------------------------------------------------


class HtmlPreviewCreate(BaseModel):
    source: str = Field(..., max_length = MAX_HTML_PREVIEW_BYTES)


class HtmlPreviewCreateResponse(BaseModel):
    url: str
    expires_in_seconds: int


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------


@router.post("", response_model = HtmlPreviewCreateResponse)
async def create_html_preview(
    payload: HtmlPreviewCreate,
    current_subject: str = Depends(get_current_subject),
) -> HtmlPreviewCreateResponse:
    """Stash an HTML snippet for same-origin iframe rendering.

    Returns a same-origin URL whose path includes a 192-bit random token.
    The token is the only authorisation for the subsequent GET.
    """
    _sweep_expired()
    source = payload.source
    if not isinstance(source, str):  # defensive; pydantic enforces str already
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
    """Serve a stashed HTML snippet with an overriding response CSP.

    Intentionally NOT auth-gated: the URL token IS the authorisation.
    The iframe in the chat page has no Authorization header to send,
    so making this require a bearer would break the only consumer.
    """
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
            # Override the global SecurityHeadersMiddleware default of
            # ``X-Frame-Options: DENY`` -- otherwise the preview page
            # refuses to be iframed by the host chat view at all.
            "X-Frame-Options": "SAMEORIGIN",
        },
    )

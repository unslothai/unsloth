# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API routes for the Codex SDK chat provider.

Two endpoints live here:

* ``GET /api/codex/status`` -- the availability probe. The frontend
  hits this at chat-page load time and uses ``installed`` to decide
  whether to surface the "codex" entry in the provider picker. When
  ``installed=True`` but ``logged_in=False``, the provider config
  dialog shows the "Sign in to Codex" affordance instead of the
  regular API-key field.

* ``POST /api/codex/login`` -- the device-auth helper. Spawns the
  ``codex auth login --device-auth`` CLI command, captures the
  verification URL from its output, and streams the rest of the auth
  exchange back as SSE so the UI can show progress. The URL appears
  in the first SSE event so the frontend can ``window.open`` it before
  the user wanders off.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from auth.authentication import get_current_subject
from core.inference.codex_availability import probe_codex_availability
from core.inference.codex_provider import stream_codex_device_login

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/status")
async def get_codex_status(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Return the Codex CLI / SDK availability snapshot.

    The frontend gates the provider entry on ``installed`` and gates
    the "Sign in to Codex" button on ``logged_in``. Both are
    best-effort and cheap to recompute; the route does not cache the
    probe because the user can install the CLI / SDK or run
    ``codex auth login`` between page loads and the picker should pick
    that up on the next refresh.
    """
    return await probe_codex_availability()


@router.post("/login")
async def codex_device_login(
    current_subject: str = Depends(get_current_subject),
) -> StreamingResponse:
    """Stream the ``codex auth login --device-auth`` exchange.

    Returns an SSE stream of events:

      ``data: {"type": "device_url", "url": "https://..."}``
      ``data: {"type": "log",  "line": "..."}`` (zero or more)
      ``data: {"type": "done", "ok": true}``

    The frontend opens the device URL in a new tab via
    ``window.open(url, "_blank", "noopener,noreferrer")`` as soon as
    the first event arrives, then renders the streamed log lines so
    the user can see the CLI making progress while they're at the
    verification page.
    """

    async def _to_sse() -> AsyncGenerator[str, None]:
        async for event in stream_codex_device_login():
            yield f"data: {json.dumps(event)}\n\n"
        # Frontend treats the trailing [DONE] the same way it does for
        # chat streams, so we emit it for parity.
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _to_sse(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Start gate for the torch warm: opens once the first HTTP response has been sent.

The warm's ``import torch`` (loading torch's extension module) and its CUDA probe each hold the GIL for a whole
C call, seconds on a loaded host. A request arriving meanwhile cannot run until the call returns, so the
desktop's first port probe (2 s timeout, then a doubling retry backoff) could miss and wait it out. Letting the
first answer go out before the warm's stages begin keeps that probe off the stall; the warm still starts in
the lifespan (so its in-progress marker reads as before) and runs every stage unchanged."""

from __future__ import annotations

import threading


class FirstResponseGateMiddleware:
    """Pure ASGI: sets ``event`` once the first HTTP response body is complete, then passes through."""

    def __init__(self, app, event: threading.Event) -> None:
        self.app = app
        self.event = event

    async def __call__(self, scope, receive, send) -> None:
        event = self.event
        if event.is_set() or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_and_mark(message) -> None:
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                event.set()

        try:
            await self.app(scope, receive, send_and_mark)
        finally:
            # A request that ends without a complete body (client gone, error) still counts as answered.
            event.set()

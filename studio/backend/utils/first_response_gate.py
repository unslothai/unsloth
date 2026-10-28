# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Start gate for the torch warm: its import torch / CUDA probe hold the GIL for whole C calls, so the desktop's
first 2 s port probe must be answered before they begin."""

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
            # Client gone or error: still answered.
            event.set()

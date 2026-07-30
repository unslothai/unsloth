# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Public base URL for /p share links: the gated preview listener plus a
# cloudflared quick tunnel. A --cloudflare launch already serves /p publicly,
# so its URL is reused instead.

from __future__ import annotations

import asyncio
import atexit
from typing import Optional

from loggers import get_logger

from cloudflare_tunnel import start_preview_tunnel, stop_studio_tunnel, stop_tunnel_if_url
from preview_public_server import listener
from utils.preview_sharing_settings import get_preview_sharing_enabled

logger = get_logger(__name__)


class PreviewSharingDisabled(RuntimeError):
    pass


class PreviewLinkUnavailable(RuntimeError):
    pass


class PreviewShareLink:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._url: Optional[str] = None

    def current(self, app) -> Optional[str]:
        # No base while sharing is off, even on a --cloudflare launch where the
        # studio URL is up: every /p request 404s, so advertising it misleads.
        if not get_preview_sharing_enabled():
            return None
        return getattr(app.state, "cloudflare_url", None) or self._url

    async def ensure(self, app) -> str:
        async with self._lock:
            # Under the lock so a create queued behind a disable fails instead
            # of resurrecting the tunnel that disable just tore down.
            if not get_preview_sharing_enabled():
                raise PreviewSharingDisabled("Public preview links are turned off in Settings.")
            studio_url = getattr(app.state, "cloudflare_url", None)
            if studio_url:
                return studio_url
            if self._url:
                return self._url

            port = await listener.start(app)
            # Blocking: downloads cloudflared on first use, then waits for the probe.
            url = None
            try:
                url = await asyncio.to_thread(start_preview_tunnel, port)
            finally:
                if not url:
                    await listener.stop()
            if not url:
                raise PreviewLinkUnavailable(
                    "Could not open a public preview tunnel. Check the network "
                    "connection and try again."
                )
            self._url = url
            # Backstop for an exit that bypasses _graceful_shutdown (run.py
            # registers the same for startup tunnels). Idempotent.
            atexit.register(stop_studio_tunnel)
            logger.info("preview_share_link.started")
            return url

    async def stop(self) -> None:
        # Stop only the exact tunnel we started: the shared slot may hold (or
        # have been replaced by) a studio-wide tunnel that is not ours to stop.
        async with self._lock:
            if self._url is None:
                return
            url, self._url = self._url, None
            await asyncio.to_thread(stop_tunnel_if_url, url)
            await listener.stop()


share_link = PreviewShareLink()

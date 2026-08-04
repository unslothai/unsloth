# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Public base URL for /p share links: the gated preview listener plus a
# cloudflared quick tunnel. A --cloudflare launch already serves /p publicly,
# so its URL is reused instead.

from __future__ import annotations

import asyncio
import atexit
import time
from typing import Optional

from loggers import get_logger

from cloudflare_tunnel import start_preview_tunnel, stop_studio_tunnel, stop_tunnel_if_url
from preview_public_server import listener
from utils.preview_sharing_settings import get_preview_sharing_enabled

logger = get_logger(__name__)

_DISABLED_MESSAGE = "Public preview links are turned off in Settings."
# Upper bound on waiting out a --cloudflare launch's startup tunnel; its
# start_studio_tunnel call finishes well within this.
_STUDIO_TUNNEL_WAIT_SECONDS = 120.0


class PreviewSharingDisabled(RuntimeError):
    pass


class PreviewLinkUnavailable(RuntimeError):
    pass


class PreviewShareLink:
    def __init__(self):
        # Created on first use: on Python 3.9 an import-time asyncio.Lock binds
        # the import loop, and contended waits on the serving loop then raise.
        self._lock: Optional[asyncio.Lock] = None
        self._url: Optional[str] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def current(self, app) -> Optional[str]:
        # No base while sharing is off, even on a --cloudflare launch where the
        # studio URL is up: every /p request 404s, so advertising it misleads.
        if not get_preview_sharing_enabled():
            return None
        return getattr(app.state, "cloudflare_url", None) or self._url

    async def _settled_studio_url(self, app) -> Optional[str]:
        # A --cloudflare launch may still be bringing its startup tunnel up
        # (the pending flag is set before the socket binds); wait for the
        # outcome instead of racing it for the shared tunnel slot.
        deadline = time.monotonic() + _STUDIO_TUNNEL_WAIT_SECONDS
        waited = False
        while getattr(app.state, "cloudflare_tunnel_pending", False):
            if time.monotonic() >= deadline:
                raise PreviewLinkUnavailable(
                    "The studio's public address is still starting. Try again shortly."
                )
            waited = True
            await asyncio.sleep(0.5)
        # The kill switch may have flipped while we waited; a disable persists
        # the setting before it can queue behind ensure's lock.
        if waited and not get_preview_sharing_enabled():
            raise PreviewSharingDisabled(_DISABLED_MESSAGE)
        return getattr(app.state, "cloudflare_url", None)

    async def ensure(self, app) -> str:
        async with self._get_lock():
            # Under the lock so a create queued behind a disable fails instead
            # of resurrecting the tunnel that disable just tore down.
            if not get_preview_sharing_enabled():
                raise PreviewSharingDisabled(_DISABLED_MESSAGE)
            studio_url = await self._settled_studio_url(app)
            if studio_url:
                return studio_url
            if self._url:
                return self._url

            port = await listener.start(app)
            # The startup tunnel may have begun while the listener bound;
            # recheck before racing it for the shared slot.
            try:
                studio_url = await self._settled_studio_url(app)
            except BaseException:
                await listener.stop()
                raise
            if studio_url:
                await listener.stop()
                return studio_url
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
            if not get_preview_sharing_enabled():
                # Disabled while the tunnel was coming up; the queued disable
                # cannot see this tunnel yet, so undo it here.
                await asyncio.to_thread(stop_tunnel_if_url, url)
                await listener.stop()
                raise PreviewSharingDisabled(_DISABLED_MESSAGE)
            self._url = url
            # Backstop for an exit that bypasses _graceful_shutdown (run.py
            # registers the same for startup tunnels). Idempotent.
            atexit.register(stop_studio_tunnel)
            logger.info("preview_share_link.started")
            return url

    async def stop(self) -> None:
        # Stop only the exact tunnel we started: the shared slot may hold (or
        # have been replaced by) a studio-wide tunnel that is not ours to stop.
        async with self._get_lock():
            if self._url is None:
                return
            url, self._url = self._url, None
            await asyncio.to_thread(stop_tunnel_if_url, url)
            await listener.stop()


share_link = PreviewShareLink()

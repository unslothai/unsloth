# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pooled httpx for non-streaming llama-server calls; streaming must bypass it to stop GPU decode."""

from __future__ import annotations

import asyncio
import weakref

import httpx

_LIMITS = httpx.Limits(max_connections = 64, max_keepalive_connections = 32)


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(limits = _LIMITS, trust_env = False)


# One client per loop: httpx binds its transport to the first loop. Weak keys free dead loops.
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)


def nonstreaming_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = _new_client()
        _clients[loop] = client
    return client


async def aclose() -> None:
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:
            pass

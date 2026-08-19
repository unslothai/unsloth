# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared pooled httpx.AsyncClient for NON-streaming calls to the local llama-server.

Streaming generation must NOT use this. It relies on ``Connection: close`` and
``max_keepalive_connections=0`` so a client disconnect tears down the upstream
socket and stops GPU decode (PR #5749). This pooled client is only for short
request/response proxy calls (non-streaming completions, embeddings) where
reusing a connection removes per-request setup cost. Per-request ``timeout`` is
still passed at each call site.
"""

from __future__ import annotations

import asyncio
import weakref

import httpx

_LIMITS = httpx.Limits(max_connections = 64, max_keepalive_connections = 32)


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(limits = _LIMITS, trust_env = False)


# One client per running event loop: an httpx client binds its transport to the
# loop it first runs on, so a single global instance breaks across a lifespan
# restart or a second test loop. Weak keys let a finished loop drop its client.
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

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

import httpx

_LIMITS = httpx.Limits(max_connections = 64, max_keepalive_connections = 32)


def _new_client() -> httpx.AsyncClient:
    try:
        return httpx.AsyncClient(limits = _LIMITS)
    except Exception:
        # Mirror external_provider: an unsupported env proxy scheme can raise.
        return httpx.AsyncClient(limits = _LIMITS, trust_env = False)


_client = _new_client()


def nonstreaming_client() -> httpx.AsyncClient:
    return _client


async def aclose() -> None:
    try:
        await _client.aclose()
    except Exception:
        pass

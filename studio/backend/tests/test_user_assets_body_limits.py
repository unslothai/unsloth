# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest

from main import MaxBodyMiddleware, _BODY_PROTECTED_PREFIXES


MUTATION_PATHS = [
    ("POST", "/api/user-assets/recipes"),
    ("PUT", "/api/user-assets/recipes/r1"),
    ("PUT", "/api/user-assets/recipes/r1/executions/e1"),
    ("POST", "/api/user-assets/legacy-import"),
]


async def _run_request(
    method: str,
    path: str,
    messages: list[dict],
    headers = (),
):
    app_called = False

    async def inner(scope, receive, send):
        nonlocal app_called
        app_called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = MaxBodyMiddleware(
        inner,
        max_bytes_getter = lambda: 10,
        protected_prefixes = _BODY_PROTECTED_PREFIXES,
    )
    sent: list[dict] = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": list(headers),
        },
        receive,
        send,
    )
    return app_called, sent


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path"), MUTATION_PATHS)
async def test_user_asset_mutations_reject_declared_oversized_bodies(method, path):
    called, sent = await _run_request(
        method,
        path,
        [{"type": "http.request", "body": b"", "more_body": False}],
        headers = [(b"content-length", b"11")],
    )
    assert not called
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path"), MUTATION_PATHS)
async def test_user_asset_mutations_reject_chunked_oversized_bodies(method, path):
    called, sent = await _run_request(
        method,
        path,
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"123456", "more_body": False},
        ],
    )
    assert not called
    assert sent[0]["status"] == 413

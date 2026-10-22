# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accept an sk-unsloth key in ``x-api-key``, the header Anthropic clients send (SDKs, the Microsoft 365
add-ins). Rewritten to a Bearer before anything runs, so every ``Authorization`` reader (auth dependency,
keyless classifier, keep-warm, MCP) sees one credential instead of learning a second scheme."""

from __future__ import annotations

from fastapi.security.utils import get_authorization_scheme_param

from auth.storage import API_KEY_PREFIX


def _authorization_is_empty(value: bytes) -> bool:
    scheme, token = get_authorization_scheme_param(value.decode("latin-1"))
    return not scheme or (scheme.lower() == "bearer" and not token)


def bearer_from_x_api_key(headers) -> list | None:
    """Headers with the key as ``Authorization``, or None to leave them alone. A Bearer always wins, and a
    non sk-unsloth value (a keyless client's dummy) keeps today's handling."""
    keys = []
    authorization = []
    for name, value in headers:
        lowered = bytes(name).lower()
        if lowered == b"x-api-key":
            keys.append(bytes(value))
        elif lowered == b"authorization":
            authorization.append(bytes(value))
    if len(keys) != 1 or len(authorization) > 1:
        return None
    if authorization and not _authorization_is_empty(authorization[0]):
        return None
    key = keys[0].strip()
    if not key.startswith(API_KEY_PREFIX.encode("ascii")):
        return None
    kept = [(name, value) for name, value in headers if bytes(name).lower() != b"authorization"]
    return [*kept, (b"authorization", b"Bearer " + key)]


class XApiKeyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            headers = bearer_from_x_api_key(scope.get("headers") or ())
            if headers is not None:
                scope["headers"] = headers
        await self.app(scope, receive, send)

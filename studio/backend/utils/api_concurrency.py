# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrency controls for API inference requests."""

from __future__ import annotations

import asyncio
import json
import os

from starlette.types import ASGIApp, Message, Receive, Scope, Send


_DEFAULT_MAX_CONCURRENCY = 1
_ALLOWED_QUEUE_POLICIES = {"wait", "reject"}

# inference_router is mounted at both /api/inference and /v1, so both prefixes
# must be listed. Management routes (load/unload/status/cancel) and embeddings
# are intentionally excluded.
_INFERENCE_ENDPOINT_PREFIXES = (
    "/api/inference/generate/stream",
    "/api/inference/audio/generate",
    "/api/inference/chat/completions",
    "/api/inference/completions",
    "/api/inference/messages",
    "/api/inference/responses",
    "/v1/generate/stream",
    "/v1/audio/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
)


def parse_api_max_concurrency(value: object | None = None) -> int:
    """Return a safe API max-concurrency value.

    Invalid, missing, or values below 1 fall back to the current default of 1
    to preserve single-active-request behavior unless users explicitly opt in.
    """
    if value is None:
        value = os.environ.get("UNSLOTH_API_MAX_CONCURRENCY")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CONCURRENCY
    return parsed if parsed >= 1 else _DEFAULT_MAX_CONCURRENCY


def parse_api_queue_policy(value: object | None = None) -> str:
    """Return the configured queue policy: ``wait`` or ``reject``."""
    if value is None:
        value = os.environ.get("UNSLOTH_API_QUEUE_POLICY")
    policy = str(value or "wait").strip().lower()
    return policy if policy in _ALLOWED_QUEUE_POLICIES else "wait"


def is_limited_inference_request(scope: Scope) -> bool:
    """True for HTTP generation endpoints covered by the concurrency gate."""
    if scope.get("type") != "http":
        return False
    if scope.get("method", "GET").upper() != "POST":
        return False
    path = scope.get("path") or ""
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in _INFERENCE_ENDPOINT_PREFIXES
    )


class AsyncConcurrencyLimiter:
    """Small fair-ish async limiter that supports wait and reject policies."""

    def __init__(self, max_concurrency: int):
        self.max_concurrency = parse_api_max_concurrency(max_concurrency)
        self._active = 0
        self._condition = asyncio.Condition()

    async def acquire(self, *, wait: bool) -> bool:
        async with self._condition:
            if not wait and self._active >= self.max_concurrency:
                return False
            while self._active >= self.max_concurrency:
                await self._condition.wait()
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify(1)

    @property
    def active(self) -> int:
        return self._active


class InferenceConcurrencyMiddleware:
    """ASGI middleware limiting concurrent inference generations.

    The slot is held until the final response body chunk is sent, which makes
    streaming responses count as active for their full lifetime.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_concurrency: int | None = None,
        queue_policy: str | None = None,
    ) -> None:
        self.app = app
        self.max_concurrency = parse_api_max_concurrency(max_concurrency)
        self.queue_policy = parse_api_queue_policy(queue_policy)
        self._limiter = AsyncConcurrencyLimiter(self.max_concurrency)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not is_limited_inference_request(scope):
            await self.app(scope, receive, send)
            return

        acquired = await self._limiter.acquire(wait = self.queue_policy == "wait")
        if not acquired:
            await self._send_rejected(send)
            return

        released = False

        async def release_once() -> None:
            nonlocal released
            if released:
                return
            released = True
            await self._limiter.release()

        async def send_with_release(message: Message) -> None:
            await send(message)
            if message.get("type") == "http.response.body" and not message.get(
                "more_body", False
            ):
                await release_once()

        try:
            await self.app(scope, receive, send_with_release)
        finally:
            await release_once()

    async def _send_rejected(self, send: Send) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "API concurrency limit reached. Please retry later.",
                    "type": "rate_limit_exceeded",
                    "code": "max_concurrency_exceeded",
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", b"1"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

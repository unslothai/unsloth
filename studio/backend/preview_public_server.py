# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Serves the same ASGI app behind a /p-only deny-by-default gate on a second
# loopback port, so a tunnel can expose preview pages without putting /api or
# the login page on the internet. Runs on the app's event loop so the preview
# model-load lock keeps serializing both surfaces.

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Optional
from urllib.parse import parse_qs

from loggers import get_logger

logger = get_logger(__name__)

# Bare /p (the JWT-authenticated listing route) is deliberately excluded.
_PREVIEW_PATH_PREFIX = "/p/"

# The preview surface only serves GET/HEAD pages and POST chat completions.
# Anything else 404s at the gate: FastAPI would answer 405 before the token
# check runs, which leaks which routes exist.
_ALLOWED_METHODS = {"GET", "HEAD", "POST"}

_CHAT_SUFFIX = "/v1/chat/completions"


def _bearer_token(scope) -> Optional[str]:
    for name, value in scope.get("headers") or ():
        if name == b"authorization":
            header = value.decode("latin-1")
            if header[:7].lower() == "bearer ":
                return header[7:].strip() or None
            return None
    return None


def _post_chat_authorized(scope, path: str) -> bool:
    # The only POST surface is chat completions. Verify the capability BEFORE
    # forwarding: FastAPI parses the request body ahead of the handler's own
    # token check, which would let unauthenticated callers burn CPU/memory on
    # large bodies with no rate limit. Mirrors routes.preview._extract_token
    # (?k= query, else Authorization: Bearer).
    if not path.endswith(_CHAT_SUFFIX):
        return False
    ref = path[len(_PREVIEW_PATH_PREFIX) : -len(_CHAT_SUFFIX)]
    if not ref:
        return False
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    token = (query.get("k") or [None])[0] or _bearer_token(scope)

    from utils.preview_token import verify_preview_ref  # lazy: keeps this module import-light

    return verify_preview_ref(ref, token)


# Tunnel-readiness probe, answered by the gate itself on a path outside /p so
# it can never shadow a run page (runs live under /p/{run}).
_PREVIEW_HEALTH_PATH = "/_preview_health"
_PREVIEW_HEALTH_SERVICE = "Unsloth Preview"
_PREVIEW_HEALTH_BODY = json.dumps({"service": _PREVIEW_HEALTH_SERVICE}).encode("utf-8")

_BIND_TIMEOUT = 10.0
_BIND_POLL_DELAY = 0.02
_SHUTDOWN_TIMEOUT = 5.0

_NOT_FOUND_BODY = json.dumps({"detail": "Not found"}).encode("utf-8")


def is_public_preview_path(path: str) -> bool:
    return path.startswith(_PREVIEW_PATH_PREFIX)


def _matches_preview_route(path: str, method: str) -> bool:
    # The prefix test alone is not enough: the wrapped app ends in a GET
    # catch-all that serves the SPA for any unmatched path, so /p/<anything>
    # (or /p/../<file>, which also defeats the prefix) would leak index.html
    # and the frontend build through the public port. Dot segments never
    # appear in a legitimate preview URL; everything else must match a route
    # the preview router itself can answer.
    if any(segment in (".", "..") for segment in path.split("/")):
        return False

    from starlette.routing import Match

    from routes.preview import router as preview_router  # lazy: keeps this module import-light

    probe = {
        "type": "http",
        "method": method,
        # The router is mounted under /p; its own routes are unprefixed.
        "path": path[len(_PREVIEW_PATH_PREFIX) - 1 :],
        "root_path": "",
    }
    # Match.FULL only. A Match.PARTIAL (right path, wrong verb) never reaches
    # FastAPI's 405: Starlette's router prefers ANY full match over a partial
    # one, and the wrapped app's GET catch-all fully matches every path -- so
    # e.g. GET /p/<run>/v1/chat/completions (a POST-only route) would be served
    # index.html instead, exactly the leak this gate exists to stop.
    for route in preview_router.routes:
        match, _ = route.matches(probe)
        if match is Match.FULL:
            return True
    return False


async def _send_json(send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _mask_405(send):
    # FastAPI answers an allowed verb on the wrong preview route with 405
    # before the token check runs; rewrite it to the generic 404 so verbs
    # cannot probe which routes exist.
    masked = False

    async def wrapped(message):
        nonlocal masked
        if masked:
            return
        if message["type"] == "http.response.start" and message.get("status") == 405:
            masked = True
            await _send_json(send, 404, _NOT_FOUND_BODY)
            return
        await send(message)

    return wrapped


async def _ack_lifespan(receive, send) -> None:
    # The wrapped app's lifespan already ran under the authenticated server.
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


class PreviewOnlyGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await _ack_lifespan(receive, send)
            return
        if scope_type == "websocket":
            # No preview route speaks websocket; refuse rather than proxy through.
            await send({"type": "websocket.close", "code": 1008})
            return
        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        if scope_type == "http" and method in ("GET", "HEAD") and path == _PREVIEW_HEALTH_PATH:
            await _send_json(send, 200, _PREVIEW_HEALTH_BODY)
            return
        if (
            scope_type != "http"
            or method not in _ALLOWED_METHODS
            or not is_public_preview_path(path)
            or not _matches_preview_route(path, method)
        ):
            await _send_json(send, 404, _NOT_FOUND_BODY)
            return
        if method == "POST" and not _post_chat_authorized(scope, path):
            await _send_json(send, 404, _NOT_FOUND_BODY)
            return
        await self.app(scope, receive, _mask_405(send))


def _bound_port(server) -> Optional[int]:
    for bound in getattr(server, "servers", None) or ():
        for sock in getattr(bound, "sockets", None) or ():
            try:
                return int(sock.getsockname()[1])
            except (OSError, IndexError, TypeError, ValueError):
                continue
    return None


async def _wait_for_bind(server, task) -> int:
    deadline = time.monotonic() + _BIND_TIMEOUT
    while True:
        if getattr(server, "started", False):
            port = _bound_port(server)
            if port:
                return port
        if task.done():
            task.result()  # re-raise the real startup failure, if any
            raise RuntimeError("Preview listener exited before it bound a port.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Preview listener did not bind in time.")
        await asyncio.sleep(_BIND_POLL_DELAY)


class PublicPreviewListener:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._server = None
        self._task: Optional[asyncio.Task] = None
        self._port: Optional[int] = None

    @property
    def port(self) -> Optional[int]:
        return self._port

    async def start(self, app) -> int:
        async with self._lock:
            if self._port is not None:
                return self._port
            import uvicorn

            class _EmbeddedServer(uvicorn.Server):
                # The studio process owns SIGINT/SIGTERM; an embedded server
                # must never swap the handlers the way stock serve() does.
                @contextlib.contextmanager
                def capture_signals(self):
                    yield

                def install_signal_handlers(self):  # older uvicorn
                    pass

            config = uvicorn.Config(
                PreviewOnlyGate(app),
                host = "127.0.0.1",
                port = 0,
                log_level = "warning",
                access_log = False,
                server_header = False,
                # The wrapped app's lifespan belongs to the authenticated server.
                lifespan = "off",
                # Config() would otherwise dictConfig() the process-wide uvicorn
                # loggers, downgrading the primary server's logging to WARNING.
                log_config = None,
                # uvicorn trusts loopback proxies by default, which would let a
                # visitor's X-Forwarded-For pick its rate-limit bucket. Keep the
                # real peer so utils.client_ip uses CF-Connecting-IP instead.
                proxy_headers = False,
            )
            server = _EmbeddedServer(config)
            task = asyncio.create_task(server.serve(), name = "unsloth-preview-public")
            try:
                port = await _wait_for_bind(server, task)
            except BaseException:
                server.should_exit = True
                task.cancel()
                raise
            self._server, self._task, self._port = server, task, port
            logger.info("preview_public_listener.started port=%s", port)
            return port

    async def stop(self) -> None:
        async with self._lock:
            server, task = self._server, self._task
            self._server, self._task, self._port = None, None, None
            if server is None:
                return
            server.should_exit = True
            if task is None:
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout = _SHUTDOWN_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
            except Exception as exc:
                logger.debug("preview_public_listener.stop error: %s", exc)


listener = PublicPreviewListener()

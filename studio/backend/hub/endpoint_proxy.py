# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Same-origin relays to a custom Hub endpoint and datasets server.

The page's CSP is fixed when it is served, and the desktop app's when it launches, so the
browser reaches an endpoint saved later through these. The Unsloth session goes in
``Authorization``; the browser's Hugging Face token travels in ``X-HF-Authorization`` and is
sent on to the configured endpoint only. The first path segment names the endpoint, so the
page's caches, keyed on its Hub URL, never mix two endpoints' answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import secrets
import threading
import weakref
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Callable, Optional
from urllib.parse import unquote, urlsplit

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask

from hub.browser_session import signed_in
from utils.client_ip import client_ip
from utils.hf_endpoint import endpoint_is_reachable_by, is_loopback_host

HUB_PREFIX = "/api/hub/proxy"
DATASETS_SERVER_PREFIX = "/api/hub/datasets-server-proxy"
TOKEN_HEADER = "X-HF-Authorization"
UPSTREAM_HEADER = "X-Hub-Upstream"

_PASSED_HEADERS = (
    "content-type",
    "etag",
    "last-modified",
    "cache-control",
    "x-repo-commit",
    "x-linked-etag",
    "x-linked-size",
    "x-error-code",
    "x-error-message",
    "x-total-count",
)
EXPOSED_HEADERS = ("Link", UPSTREAM_HEADER, *_PASSED_HEADERS[1:])

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_clients_lock = threading.Lock()
transport: Optional[httpx.AsyncBaseTransport] = None


def _client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    with _clients_lock:
        http = _clients.get(loop)
        if http is None:
            # httpx drops Authorization when a redirect leaves the endpoint's origin (LFS CDNs).
            http = _clients[loop] = httpx.AsyncClient(
                timeout = httpx.Timeout(30.0, connect = 10.0),
                follow_redirects = True,
                cookies = CookieJar(DefaultCookiePolicy(allowed_domains = [])),
                transport = transport,
                headers = {"User-Agent": "unsloth-studio"},
            )
        return http


# Keyed per process: /api/health is unauthenticated, and a bare hash of a private host:port is brute-forceable.
_TAG_KEY = secrets.token_bytes(32)


def _tag(upstream: str) -> str:
    return hmac.new(_TAG_KEY, upstream.encode(), hashlib.sha256).hexdigest()[:12]


def relay_path(prefix: str, upstream: str, default: str) -> Optional[str]:
    """The relay's path for ``upstream``, or None when the browser can use the default directly."""
    return None if upstream == default else f"{prefix}/{_tag(upstream)}"


def _refuse(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": message}, status_code = status, headers = {"X-Error-Message": message}
    )


def _rebase_link(link: str, upstream: str, base: str) -> str:
    """Point pagination links at the proxy; a mirror may name itself or huggingface.co."""
    prefix = urlsplit(upstream).path.rstrip("/")
    parts = []
    for part in link.split(","):
        target, sep, params = part.strip().partition(">")
        if not target.startswith("<"):
            parts.append(part.strip())
            continue
        url = urlsplit(target[1:])
        path = url.path[len(prefix) :] if prefix and url.path.startswith(prefix + "/") else url.path
        parts.append(f"<{base}{path}{'?' + url.query if url.query else ''}{sep}{params}")
    return ", ".join(parts)


ANONYMOUS_ASSET_LIMIT = 20 * 1024 * 1024
_ASSET_PATH = re.compile(r"^/(?:(?:datasets|spaces)/)?[^/]+/[^/]+/(?:resolve|raw)/[^/]+/.")
# Served under Studio's origin, so only inert raster images, never markup or scripts.
_ASSET_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


async def _capped(chunks, limit: int):
    """Bound what a signed-out README asset load can pull through the server."""
    sent = 0
    async for chunk in chunks:
        sent += len(chunk)
        if sent > limit:
            raise RuntimeError("anonymous relay limit exceeded")
        yield chunk


def _saved_only(endpoint: str) -> bool:
    """A redirect's Location would name an endpoint the owner-only settings route keeps private."""
    from utils.hub_settings import saved_only_endpoints
    return endpoint in saved_only_endpoints()


def build_router(prefix: str, upstream: Callable[[], str], *, anonymous_pages: bool) -> APIRouter:
    """``anonymous_pages`` redirects credential-less loads outside ``/api`` (README images,
    repository links) to the endpoint; a settings-saved endpoint seen from off the host is
    relayed without credentials instead, so its URL stays private."""
    router = APIRouter()

    @router.api_route("/{path:path}", methods = ["GET", "HEAD"])
    async def relay(path: str, request: Request) -> Response:
        raw = (request.scope.get("raw_path") or request.url.path.encode()).decode("latin-1")
        tag, _, rest = raw.partition(prefix)[2].removeprefix("/").partition("/")
        endpoint = upstream()
        if tag != _tag(endpoint):
            return _refuse(409, "The Hub endpoint changed. Reload to browse it.")
        rest = f"/{rest}"
        segments = [unquote(segment) for segment in rest.split("/")]
        # Re-split: an encoded slash can hide a dot segment from this check but not from the endpoint.
        if any(part in (".", "..") or "\\" in part for s in segments for part in s.split("/")):
            return _refuse(400, "Invalid path.")
        query = request.scope.get("query_string", b"").decode("latin-1")
        target = f"{endpoint}{rest}{'?' + query if query else ''}"

        anonymous = not await signed_in(request)
        if anonymous:
            if (
                "authorization" in request.headers
                or not anonymous_pages
                or segments[1:2] == ["api"]
                or not endpoint_is_reachable_by(endpoint, client_ip(request))
            ):
                return _refuse(401, "Sign in again to browse the Hub.")
            # A Location would name an endpoint only the owner may read: relay the asset instead.
            if not (_saved_only(endpoint) and not is_loopback_host(client_ip(request))):
                return RedirectResponse(target, status_code = 302)
            if not _ASSET_PATH.match(rest):
                return _refuse(401, "Sign in again to browse the Hub.")

        headers = {"Accept": request.headers.get("accept", "*/*")}
        if not anonymous and (token := request.headers.get(TOKEN_HEADER)):
            headers["Authorization"] = token
        http = _client()
        try:
            answer = await http.send(
                http.build_request(request.method, target, headers = headers), stream = True
            )
        except httpx.HTTPError as exc:
            return _refuse(502, f"The Hub endpoint could not be reached ({type(exc).__name__}).")
        body = answer.aiter_bytes()
        if anonymous:
            kind = answer.headers.get("content-type", "").split(";")[0].strip().lower()
            if answer.is_success and (not kind.startswith("image/") or kind == "image/svg+xml"):
                await answer.aclose()
                return _refuse(401, "Sign in again to browse the Hub.")
            if int(answer.headers.get("content-length") or 0) > ANONYMOUS_ASSET_LIMIT:
                await answer.aclose()
                return _refuse(413, "Sign in to download files this large.")
            body = _capped(body, ANONYMOUS_ASSET_LIMIT)
        passed = {name: answer.headers[name] for name in _PASSED_HEADERS if name in answer.headers}
        if link := answer.headers.get("link"):
            path = f"{request.scope.get('root_path', '')}{prefix}/{tag}"
            passed["Link"] = _rebase_link(
                link, endpoint, str(request.url.replace(path = path, query = ""))
            )
        passed[UPSTREAM_HEADER] = "1"
        if anonymous:
            passed.update(_ASSET_HEADERS)
        return StreamingResponse(
            body,
            status_code = answer.status_code,
            headers = passed,
            background = BackgroundTask(answer.aclose),
        )

    return router

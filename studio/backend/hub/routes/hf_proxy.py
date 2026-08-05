# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only passthrough to the Hugging Face API for filtered browsers.

Hub discovery normally queries huggingface.co straight from the browser.
Privacy tooling (DNS filtering, TLS-inspecting firewalls, tracker blockers)
can block those requests while the backend still has connectivity, and the UI
then reports "You're offline" even though downloads work. The frontend falls
back to this endpoint when its direct fetch fails, so discovery uses the same
network path as downloads.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token

router = APIRouter()

# read-only discovery hosts; anything else is refused so this endpoint cannot
# be used as an open proxy.
ALLOWED_PROXY_HOSTS = frozenset({"huggingface.co", "datasets-server.huggingface.co"})

# response headers the browser client consumes. link carries the pagination
# cursor for the @huggingface/hub listing iterators.
FORWARDED_RESPONSE_HEADERS = ("content-type", "link", "etag")

# Stamped on every response this route produces, so the frontend can tell a hub
# 404 (missing repo, pass it through) from an older backend that has no such
# route and answers the SPA catch-all's 404.
PROXY_MARKER_HEADER = "X-Unsloth-HF-Proxy"

PROXY_TIMEOUT_SECONDS = 30.0

# Redirects are followed by hand so every hop is re-validated; discovery
# endpoints do not redirect in normal operation.
MAX_PROXY_REDIRECTS = 5

# listing pages are ~100 kib and raw readmes a few mib; anything larger is not
# a discovery payload and gets refused instead of buffered.
MAX_PROXY_RESPONSE_BYTES = 20 * 1024 * 1024


def validate_proxy_url(raw_url: str) -> str:
    """Reject anything that is not a plain https url on an allowed hub host."""
    try:
        parts = urlsplit(raw_url)
        # Both parse lazily and raise ValueError on a malformed authority
        # ("https://huggingface.co:notaport/"), so read them inside the guard.
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        raise HTTPException(status_code = 400, detail = "Malformed proxy URL")
    if parts.scheme != "https":
        raise HTTPException(status_code = 403, detail = "Only https URLs can be proxied")
    if hostname not in ALLOWED_PROXY_HOSTS:
        raise HTTPException(
            status_code = 403,
            detail = "Only Hugging Face hub URLs can be proxied",
        )
    if port not in (None, 443):
        raise HTTPException(status_code = 403, detail = "Non-standard ports are not allowed")
    if parts.username or parts.password:
        raise HTTPException(status_code = 403, detail = "Credentials in URLs are not allowed")
    return raw_url


def forwarded_response_headers(upstream_headers: httpx.Headers) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in FORWARDED_RESPONSE_HEADERS:
        value = upstream_headers.get(name)
        if value:
            headers[name] = value
    headers[PROXY_MARKER_HEADER] = "1"
    return headers


@router.get("/hf-proxy")
async def proxy_hugging_face_get(
    url: str = Query(..., max_length = 4096, description = "Absolute Hugging Face URL to fetch"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    target = validate_proxy_url(url)
    request_headers = {"Accept": "application/json, text/plain, */*"}
    if hf_token:
        request_headers["Authorization"] = f"Bearer {hf_token}"
    try:
        # follow_redirects stays off: it would only ever validate the caller's
        # own URL, so a 30x off an allowed host could reach loopback or the
        # link-local metadata endpoint and return that body. Follow by hand and
        # re-validate each hop, as core/inference/tools.py does.
        async with httpx.AsyncClient(
            follow_redirects = False,
            timeout = PROXY_TIMEOUT_SECONDS,
        ) as client:
            for _hop in range(MAX_PROXY_REDIRECTS + 1):
                # stream so the size cap aborts an oversized body (e.g. a resolve/
                # weights url on an allowed host) instead of buffering it in full.
                async with client.stream(
                    "GET", target, headers = request_headers
                ) as upstream:
                    if upstream.status_code in (301, 302, 303, 307, 308):
                        location = upstream.headers.get("location")
                        if not location:
                            raise HTTPException(
                                status_code = 502,
                                detail = "Upstream redirect without a location",
                            )
                        # relative locations resolve against the current hop
                        target = validate_proxy_url(str(httpx.URL(target).join(location)))
                        continue
                    body = bytearray()
                    async for chunk in upstream.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_PROXY_RESPONSE_BYTES:
                            raise HTTPException(
                                status_code = 502,
                                detail = "Upstream response too large to proxy",
                            )
                    status_code = upstream.status_code
                    response_headers = forwarded_response_headers(upstream.headers)
                    break
            else:
                raise HTTPException(
                    status_code = 502, detail = "Too many upstream redirects"
                )
    except httpx.TimeoutException:
        raise HTTPException(status_code = 504, detail = "Hugging Face request timed out")
    except httpx.HTTPError:
        raise HTTPException(status_code = 502, detail = "Could not reach Hugging Face")
    return Response(
        content = bytes(body),
        status_code = status_code,
        headers = response_headers,
    )

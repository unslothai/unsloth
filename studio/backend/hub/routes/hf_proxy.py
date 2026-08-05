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

PROXY_TIMEOUT_SECONDS = 30.0

# listing pages are ~100 kib and raw readmes a few mib; anything larger is not
# a discovery payload and gets refused instead of buffered.
MAX_PROXY_RESPONSE_BYTES = 20 * 1024 * 1024


def validate_proxy_url(raw_url: str) -> str:
    """Reject anything that is not a plain https url on an allowed hub host."""
    try:
        parts = urlsplit(raw_url)
    except ValueError:
        raise HTTPException(status_code = 400, detail = "Malformed proxy URL")
    if parts.scheme != "https":
        raise HTTPException(status_code = 403, detail = "Only https URLs can be proxied")
    if parts.hostname not in ALLOWED_PROXY_HOSTS:
        raise HTTPException(
            status_code = 403,
            detail = "Only Hugging Face hub URLs can be proxied",
        )
    if parts.port not in (None, 443):
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
        async with httpx.AsyncClient(
            follow_redirects = True,
            timeout = PROXY_TIMEOUT_SECONDS,
        ) as client:
            upstream = await client.get(target, headers = request_headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code = 504, detail = "Hugging Face request timed out")
    except httpx.HTTPError:
        raise HTTPException(status_code = 502, detail = "Could not reach Hugging Face")
    if len(upstream.content) > MAX_PROXY_RESPONSE_BYTES:
        raise HTTPException(status_code = 502, detail = "Upstream response too large to proxy")
    return Response(
        content = upstream.content,
        status_code = upstream.status_code,
        headers = forwarded_response_headers(upstream.headers),
    )

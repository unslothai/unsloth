# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server-side Hugging Face discovery search.

The Model hub queries the Hub from the browser, which fails when the browser
cannot reach it but the server can: a client-side content blocker or DNS filter,
a CSP that omits the configured endpoint, or an ``HF_ENDPOINT`` mirror without
CORS headers. This route is the same-origin fallback.

Not a general proxy: the destination comes from ``hf_endpoint_url()`` and no
part of it can be influenced by the request, which is what keeps this from being
an SSRF primitive. The query is rebuilt from an allowlist, not forwarded.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from hub.utils.download_registry import scrub_secrets
from hub.utils.hf_errors import hf_error_status
from utils.utils import hf_endpoint_url


router = APIRouter()

# Discovery is interactive: a slow mirror should fail fast enough to fall back.
_REQUEST_TIMEOUT_SECONDS = 15.0
# `expand` can make a 500-item listing large, and it is buffered before parsing.
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_LIMIT = 500
_MAX_STRING_LENGTH = 256
_MAX_REPEATED_VALUES = 24

# Forwardable parameters. Anything else is rejected, not ignored, so `endpoint`,
# `url` or an absolute pagination target cannot slip through unnoticed.
_SCALAR_PARAMS = frozenset(
    {
        "search",
        "author",
        "sort",
        "direction",
        "limit",
        "cursor",
        "full",
        "config",
        # Sent by @huggingface/hub listModels / listDatasets.
        "pipeline_tag",
        "inference_provider",
        "apps",
        "p",
    }
)
_REPEATED_PARAMS = frozenset({"filter", "expand"})
_ALLOWED_PARAMS = _SCALAR_PARAMS | _REPEATED_PARAMS

_ALLOWED_SORTS = frozenset(
    {
        "downloads",
        "downloadsAllTime",
        "likes",
        "trendingScore",
        "lastModified",
        "createdAt",
        "author",
        "id",
    }
)
_ALLOWED_DIRECTIONS = frozenset({"-1", "1"})
_BOOLEAN_VALUES = frozenset({"true", "false"})

# Never echo an upstream auth failure as our own 401/403: authFetch reads a 401
# from this backend as an expired Studio session and clears the tokens, logging
# the user out. 424 says the dependency failed, not that the caller did.
_UPSTREAM_AUTH_STATUS = 424


class DiscoveryQueryError(ValueError):
    """A caller-supplied query that failed validation."""


def _reject(detail: str) -> None:
    raise DiscoveryQueryError(detail)


def _validated_scalar(name: str, raw: str) -> str:
    value = raw.strip()
    if len(value) > _MAX_STRING_LENGTH:
        _reject(f"Query parameter {name!r} is too long")
    if name == "sort":
        if value not in _ALLOWED_SORTS:
            _reject(f"Unsupported sort {value!r}")
    elif name == "direction":
        if value not in _ALLOWED_DIRECTIONS:
            _reject(f"Unsupported direction {value!r}")
    elif name == "limit":
        if not value.isdigit() or not 1 <= int(value) <= _MAX_LIMIT:
            _reject(f"Invalid limit {value!r}")
    elif name in ("full", "config"):
        if value.lower() not in _BOOLEAN_VALUES:
            _reject(f"Invalid boolean for {name!r}")
        value = value.lower()
    return value


def build_discovery_query(params: Any) -> List[tuple]:
    """Rebuild the upstream query from an allowlist.

    ``params`` is a Starlette QueryParams (multi-dict). Unknown or duplicated
    scalars are rejected, not dropped: ignoring them would let a caller believe
    a filter applied when it did not.
    """
    pairs: List[tuple] = []
    seen_scalars: set = set()
    repeated_counts: Dict[str, int] = {}

    items = params.multi_items() if hasattr(params, "multi_items") else list(params.items())
    for name, raw in items:
        if name not in _ALLOWED_PARAMS:
            _reject(f"Unsupported query parameter {name!r}")
        if name in _SCALAR_PARAMS:
            if name in seen_scalars:
                _reject(f"Duplicate query parameter {name!r}")
            seen_scalars.add(name)
            pairs.append((name, _validated_scalar(name, raw)))
            continue
        count = repeated_counts.get(name, 0) + 1
        if count > _MAX_REPEATED_VALUES:
            _reject(f"Too many values for {name!r}")
        repeated_counts[name] = count
        value = raw.strip()
        if len(value) > _MAX_STRING_LENGTH:
            _reject(f"Query parameter {name!r} is too long")
        pairs.append((name, value))
    return pairs


def build_upstream_url(resource: str, pairs: List[tuple]) -> str:
    """Target URL. The base comes only from HF_ENDPOINT, never from the caller."""
    base = hf_endpoint_url().rstrip("/")
    query = urlencode(pairs)
    return f"{base}/api/{resource}" + (f"?{query}" if query else "")


def parse_next_link(link_header: str) -> Optional[str]:
    """Extract the rel="next" target from an RFC 8288 Link header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        target = segments[0].strip()
        if not (target.startswith("<") and target.endswith(">")):
            continue
        for attr in segments[1:]:
            key, _, value = attr.strip().partition("=")
            if key.strip().lower() != "rel":
                continue
            if value.strip().strip('"').lower() == "next":
                return target[1:-1]
    return None


def rewrite_next_link(
    resource: str,
    next_url: Optional[str],
    base_url: str = "",
) -> Optional[str]:
    """Point the caller's next page back at this route.

    Handing the Hub's absolute next-page URL to the browser would bypass the
    proxy, and following a mirror's arbitrary URL would reintroduce SSRF. So the
    link is taken only if it stays on the configured endpoint, query revalidated.

    ``base_url`` makes the emitted link absolute. @huggingface/hub's
    parseLinkHeader only matches an ``<http(s)://...>`` target, so a relative one
    yields no next URL and pagination silently stops after the first page.
    """
    if not next_url:
        return None
    from urllib.parse import parse_qsl, urljoin, urlsplit

    endpoint = hf_endpoint_url()
    base = urlsplit(endpoint)
    # RFC 8288 targets may be relative; resolve first so a mirror emitting
    # `</api/models?cursor=...>` is not mistaken for an off-endpoint link.
    parsed = urlsplit(urljoin(endpoint, next_url))
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        return None
    try:
        pairs = build_discovery_query(_QueryPairs(parse_qsl(parsed.query)))
    except DiscoveryQueryError:
        return None
    query = urlencode(pairs)
    prefix = base_url.rstrip("/")
    return f"{prefix}/api/hub/discovery/{resource}" + (f"?{query}" if query else "")


class _QueryPairs:
    """Minimal multi-dict shim so parsed pairs reuse the same validator."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def multi_items(self):
        return list(self._pairs)


def _fetch_upstream(url: str, hf_token: Optional[str]) -> tuple:
    """Blocking upstream GET -> (status, body, link_header).

    Uses huggingface_hub's session so proxy settings and the user agent are
    inherited, matching utils/hf_token_validation.py.
    """
    from huggingface_hub.utils import get_session

    headers = {"Accept": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    response = get_session().get(
        url,
        headers = headers,
        timeout = _REQUEST_TIMEOUT_SECONDS,
        # A redirect could walk onto an internal address, so refuse the hop.
        allow_redirects = False,
        stream = True,
    )
    try:
        link = response.headers.get("Link", "")
        if response.status_code in (301, 302, 303, 307, 308):
            return response.status_code, b"", ""
        body = bytearray()
        for chunk in response.iter_content(chunk_size = 65536):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise HTTPException(
                    status_code = 502,
                    detail = "Hugging Face returned an oversized discovery response",
                )
        return response.status_code, bytes(body), link
    finally:
        response.close()


@router.get("/discovery/{resource}")
async def discovery_search(
    resource: Literal["models", "datasets"],
    request: Request,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    try:
        pairs = build_discovery_query(request.query_params)
    except DiscoveryQueryError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    url = build_upstream_url(resource, pairs)

    try:
        status, body, link = await asyncio.to_thread(_fetch_upstream, url, hf_token)
    except HTTPException:
        raise
    except Exception as e:
        scrubbed = scrub_secrets(str(e), hf_token = hf_token)
        mapped = hf_error_status(e)
        if mapped in (401, 403):
            raise HTTPException(status_code = _UPSTREAM_AUTH_STATUS, detail = scrubbed)
        if mapped is not None:
            raise HTTPException(status_code = mapped, detail = scrubbed)
        raise HTTPException(
            status_code = 502,
            detail = "Could not reach Hugging Face: " + scrubbed,
        )

    if status in (301, 302, 303, 307, 308):
        raise HTTPException(
            status_code = 502,
            detail = "Hugging Face redirected the discovery request; refusing to follow",
        )
    if status in (401, 403):
        # See _UPSTREAM_AUTH_STATUS: never surface this as our own 401/403.
        raise HTTPException(
            status_code = _UPSTREAM_AUTH_STATUS,
            detail = "Hugging Face rejected the credentials for this search",
        )
    if status >= 400:
        raise HTTPException(
            status_code = 502 if status >= 500 else status,
            detail = f"Hugging Face returned {status} for this search",
        )

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code = 502,
            detail = "Hugging Face returned a malformed discovery response",
        )

    headers = {}
    # Absolute, not relative: the Hub client's link parser only recognises an
    # http(s) target. The browser never fetches this URL -- the Hub transport
    # takes its query and re-issues it same-origin -- so only the path has to be
    # right, which is why deriving the host from the request is safe here.
    next_link = rewrite_next_link(
        resource,
        parse_next_link(link),
        str(request.base_url),
    )
    if next_link:
        # The Hub client paginates off this header; without it the proxy stops
        # after page one.
        headers["Link"] = f'<{next_link}>; rel="next"'
    return JSONResponse(content = payload, headers = headers)

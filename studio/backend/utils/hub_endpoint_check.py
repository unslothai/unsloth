# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check that a custom Hub endpoint really serves Studio, rather than handing it back to huggingface.co.

A mirror can answer every request and still not carry the bytes: hf-mirror.com outside
mainland China redirects the whole site to huggingface.co, a mirror may redirect file
downloads to the official CDN, and a mirror that passes the Xet headers through leaves
huggingface_hub fetching the chunks from Hugging Face's own CAS. huggingface_hub follows
all of these silently, so the owner never learns the mirror is not in use.

The check asks the endpoint the three questions a download asks, anonymously and without
downloading anything: the model metadata API, a HEAD of an LFS file, and the Xet token
route that names the storage the chunks come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlsplit

import httpx

from utils.hf_endpoint import validate_hub_endpoint

# A small, long-lived public repo whose weights are an LFS (and Xet) file on the official Hub.
PROBE_REPO = "openai-community/gpt2"
PROBE_FILE = "model.safetensors"

_OFFICIAL_DOMAINS = ("huggingface.co", "hf.co")
_MAX_REDIRECTS = 5
# Tests swap in an httpx.MockTransport.
transport: Optional[httpx.BaseTransport] = None

OK = "ok"
MIRROR = "mirror"
OFFICIAL = "official"
FAILED = "failed"
NONE = "none"


@dataclass(frozen = True)
class EndpointCheck:
    endpoint: str
    # "ok", "official" (redirected to huggingface.co) or "failed".
    api: str
    # "mirror" (served by the mirror or a CDN of its own), "official" or "failed".
    download: str
    # "none" (no Xet for this file), "mirror", "official" or "failed".
    xet: str
    detail: Optional[str] = None


def is_official_host(hostname: Optional[str]) -> bool:
    host = (hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in _OFFICIAL_DOMAINS)


def _follow(client: httpx.Client, method: str, url: str) -> tuple[str, httpx.Response]:
    """Follow redirects by hand until one leaves the endpoint's host.

    Returns ``(verdict, response)``: ``OFFICIAL`` when a redirect names huggingface.co,
    ``MIRROR`` when it names any other host (the mirror's CDN, never fetched itself), and
    ``OK`` for the final response on the endpoint's own host.
    """
    host = urlsplit(url).hostname
    for _ in range(_MAX_REDIRECTS):
        response = client.request(method, url)
        location = response.headers.get("location")
        if not response.is_redirect or not location:
            return OK, response
        url = urljoin(url, location)
        target = urlsplit(url).hostname
        if is_official_host(target):
            return OFFICIAL, response
        if target != host:
            return MIRROR, response
    raise httpx.TooManyRedirects(f"More than {_MAX_REDIRECTS} redirects", request = response.request)


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _check_api(client: httpx.Client, endpoint: str) -> tuple[str, Optional[str]]:
    try:
        verdict, response = _follow(client, "GET", f"{endpoint}/api/models/{PROBE_REPO}")
    except httpx.HTTPError as exc:
        return FAILED, f"The model API did not answer ({_describe(exc)})."
    if verdict == OFFICIAL:
        return OFFICIAL, None
    if verdict != OK:
        return FAILED, "The model API redirected away from the endpoint."
    if response.status_code != 200:
        return FAILED, f"The model API answered HTTP {response.status_code}."
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or body.get("id", body.get("modelId")) != PROBE_REPO:
        return FAILED, "The model API did not answer with Hugging Face model metadata."
    return OK, None


def _xet_refresh_route(response: httpx.Response, endpoint: str) -> Optional[str]:
    """The Xet token route huggingface_hub would call, rewritten onto the endpoint the same way."""
    if "x-xet-hash" not in response.headers:
        return None
    route = response.links.get("xet-auth", {}).get("url") or response.headers.get(
        "x-xet-refresh-route"
    )
    if not route:
        return None
    official_home = "https://huggingface.co"
    if route.startswith(official_home + "/"):
        route = endpoint + route[len(official_home) :]
    return route


def _check_xet(client: httpx.Client, route: str) -> tuple[str, Optional[str]]:
    if is_official_host(urlsplit(route).hostname):
        return OFFICIAL, None
    try:
        verdict, response = _follow(client, "GET", route)
    except httpx.HTTPError as exc:
        return FAILED, f"The Xet token route did not answer ({_describe(exc)})."
    if verdict == OFFICIAL:
        return OFFICIAL, None
    if verdict != OK or response.status_code != 200:
        return FAILED, f"The Xet token route answered HTTP {response.status_code}."
    cas_url = response.headers.get("x-xet-cas-url")
    if not cas_url:
        try:
            cas_url = response.json().get("casUrl")
        except (ValueError, AttributeError):
            cas_url = None
    if not isinstance(cas_url, str) or not cas_url:
        return FAILED, "The Xet token route named no storage server."
    return (OFFICIAL if is_official_host(urlsplit(cas_url).hostname) else MIRROR), None


def _check_download(client: httpx.Client, endpoint: str) -> tuple[str, str, Optional[str]]:
    url = f"{endpoint}/{PROBE_REPO}/resolve/main/{PROBE_FILE}"
    try:
        verdict, response = _follow(client, "HEAD", url)
    except httpx.HTTPError as exc:
        return FAILED, NONE, f"The file download did not answer ({_describe(exc)})."
    if verdict == OK and response.status_code != 200:
        return FAILED, NONE, f"The file download answered HTTP {response.status_code}."
    download = MIRROR if verdict == OK else verdict
    route = _xet_refresh_route(response, endpoint)
    if route is None:
        return download, NONE, None
    xet, detail = _check_xet(client, route)
    return download, xet, detail


def check_hub_endpoint(raw: str, *, timeout: float = 10.0) -> EndpointCheck:
    """Probe ``raw`` the way a Studio download would. Raises ValueError on an unusable endpoint."""
    endpoint = validate_hub_endpoint(raw)
    if not endpoint or is_official_host(urlsplit(endpoint).hostname):
        raise ValueError("This is the official Hugging Face Hub; there is no mirror to check.")
    with httpx.Client(
        timeout = httpx.Timeout(timeout, connect = min(timeout, 10.0)),
        follow_redirects = False,
        transport = transport,
        headers = {"User-Agent": "unsloth-studio"},
    ) as client:
        api, api_detail = _check_api(client, endpoint)
        download, xet, download_detail = _check_download(client, endpoint)
    return EndpointCheck(
        endpoint = endpoint,
        api = api,
        download = download,
        xet = xet,
        detail = api_detail or download_detail,
    )

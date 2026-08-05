# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /api/hub/hf-proxy passthrough exists for browsers whose privacy tooling
blocks direct fetches to huggingface.co while the backend still has
connectivity. These tests pin its two safety properties (host allowlist, header
filtering) and the token/error passthrough behaviour."""

import asyncio
import sys
import types

import httpx
import pytest
from fastapi import HTTPException

# Keep this test runnable without optional logging deps.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import hub.routes.hf_proxy as hf_proxy


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/api/models?limit=20",
        "https://datasets-server.huggingface.co/size?dataset=x",
    ],
)
def test_validate_proxy_url_allows_hub_hosts(url):
    assert hf_proxy.validate_proxy_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/api/models",
        "https://evil.example.com/api/models",
        "https://huggingface.co.evil.example.com/api/models",
        "https://huggingface.co:8443/api/models",
        "https://user:pass@huggingface.co/api/models",
        "file:///etc/passwd",
    ],
)
def test_validate_proxy_url_rejects_non_hub_urls(url):
    with pytest.raises(HTTPException) as exc_info:
        hf_proxy.validate_proxy_url(url)
    assert exc_info.value.status_code in (400, 403)


def test_forwarded_response_headers_keeps_only_client_facing_headers():
    upstream = httpx.Headers(
        {
            "content-type": "application/json",
            "link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"',
            "etag": '"deadbeef"',
            "set-cookie": "session=1",
            "x-powered-by": "hf",
        }
    )
    forwarded = hf_proxy.forwarded_response_headers(upstream)
    assert forwarded == {
        "content-type": "application/json",
        "link": '<https://huggingface.co/api/models?cursor=abc>; rel="next"',
        "etag": '"deadbeef"',
    }


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the upstream request."""

    last_request = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, headers = None):
        _FakeAsyncClient.last_request = (url, headers or {})
        outcome = _FakeAsyncClient.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _proxy(url, hf_token = None):
    return asyncio.run(
        hf_proxy.proxy_hugging_face_get(
            url = url,
            hf_token = hf_token,
            current_subject = "tester",
        )
    )


def test_proxy_forwards_hf_token_and_upstream_response(monkeypatch):
    upstream = types.SimpleNamespace(
        content = b'{"models": []}',
        status_code = 200,
        headers = httpx.Headers({"content-type": "application/json"}),
    )
    _FakeAsyncClient.outcome = upstream
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    response = _proxy("https://huggingface.co/api/models", hf_token = "hf_secret")

    url, headers = _FakeAsyncClient.last_request
    assert url == "https://huggingface.co/api/models"
    assert headers["Authorization"] == "Bearer hf_secret"
    assert response.status_code == 200
    assert response.body == b'{"models": []}'


def test_proxy_passes_through_upstream_error_status(monkeypatch):
    upstream = types.SimpleNamespace(
        content = b'{"error": "gated"}',
        status_code = 401,
        headers = httpx.Headers({"content-type": "application/json"}),
    )
    _FakeAsyncClient.outcome = upstream
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    response = _proxy("https://huggingface.co/api/models")

    _url, headers = _FakeAsyncClient.last_request
    assert "Authorization" not in headers
    assert response.status_code == 401


def test_proxy_maps_timeout_to_504(monkeypatch):
    _FakeAsyncClient.outcome = httpx.ConnectTimeout("timed out")
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 504


def test_proxy_maps_connection_failure_to_502(monkeypatch):
    _FakeAsyncClient.outcome = httpx.ConnectError("no route")
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 502

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
        # Stamped on every response this route produces so the frontend can
        # tell a proxied hub reply from an older backend's catch-all 404.
        hf_proxy.PROXY_MARKER_HEADER: "1",
    }


class _FakeStream:
    """Async context manager mimicking httpx's streaming response."""

    def __init__(self, upstream):
        self._upstream = upstream

    async def __aenter__(self):
        return self._upstream

    async def __aexit__(self, *exc_info):
        return False


class _FakeUpstream:
    def __init__(self, content, status_code, headers):
        self._content = content
        self.status_code = status_code
        self.headers = headers

    async def aiter_bytes(self):
        # two chunks so the size cap is exercised mid-stream, not only at the end
        half = max(1, len(self._content) // 2)
        for start in range(0, len(self._content), half):
            yield self._content[start : start + half]


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the upstream request."""

    last_request = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def stream(
        self,
        method,
        url,
        headers = None,
    ):
        _FakeAsyncClient.last_request = (url, headers or {})
        outcome = _FakeAsyncClient.outcome
        if isinstance(outcome, Exception):

            class _RaisingStream:
                async def __aenter__(self):
                    raise outcome

                async def __aexit__(self, *exc_info):
                    return False

            return _RaisingStream()
        return _FakeStream(outcome)


def _proxy(url, hf_token = None):
    return asyncio.run(
        hf_proxy.proxy_hugging_face_get(
            url = url,
            hf_token = hf_token,
            current_subject = "tester",
        )
    )


def test_proxy_forwards_hf_token_and_upstream_response(monkeypatch):
    upstream = _FakeUpstream(
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
    upstream = _FakeUpstream(
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


def test_proxy_rejects_oversized_body_mid_stream(monkeypatch):
    upstream = _FakeUpstream(
        content = b"x" * (hf_proxy.MAX_PROXY_RESPONSE_BYTES + 1),
        status_code = 200,
        headers = httpx.Headers({"content-type": "application/octet-stream"}),
    )
    _FakeAsyncClient.outcome = upstream
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 502


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


# ---------------------------------------------------------------------------
# Redirects are followed manually so the host allowlist applies to every hop.
# follow_redirects=True would only ever have validated the caller's own URL.
# ---------------------------------------------------------------------------
class _RedirectingClient:
    """Real redirect semantics against a scripted set of hops."""

    hops: list = []
    seen: list = []

    def __init__(self, **kwargs):
        _RedirectingClient.init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def stream(self, method, url, headers = None):
        _RedirectingClient.seen.append(url)
        outcome = _RedirectingClient.hops.pop(0)
        if isinstance(outcome, str):

            class _Redirect:
                status_code = 302
                headers = httpx.Headers({"location": outcome})

                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, *exc_info):
                    return False

                async def aiter_bytes(self_inner):
                    if False:
                        yield b""

            return _Redirect()
        return _FakeStream(outcome)


def _install_redirect_client(monkeypatch, hops):
    _RedirectingClient.hops = list(hops)
    _RedirectingClient.seen = []
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _RedirectingClient)


def test_redirects_are_not_followed_automatically(monkeypatch):
    """The client must be constructed with redirect following disabled."""
    _install_redirect_client(
        monkeypatch,
        [_FakeUpstream(b"{}", 200, httpx.Headers({}))],
    )
    _proxy("https://huggingface.co/api/models")
    assert _RedirectingClient.init_kwargs["follow_redirects"] is False


def test_redirect_to_disallowed_host_is_refused(monkeypatch):
    """A 302 off the allowlist must be rejected, not followed.

    Without per-hop validation this reached loopback and the link-local
    metadata address, and returned that body to the authenticated caller.
    """
    _install_redirect_client(monkeypatch, ["http://127.0.0.1:9/internal"])
    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 403
    assert _RedirectingClient.seen == ["https://huggingface.co/api/models"]


def test_redirect_to_link_local_metadata_is_refused(monkeypatch):
    _install_redirect_client(monkeypatch, ["http://169.254.169.254/latest/meta-data/"])
    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 403


def test_redirect_within_the_allowlist_is_followed(monkeypatch):
    """Legitimate same-allowlist redirects still work, including relative ones."""
    _install_redirect_client(
        monkeypatch,
        [
            "/api/models?limit=20",
            _FakeUpstream(
                b'{"ok":true}',
                200,
                httpx.Headers({"content-type": "application/json"}),
            ),
        ],
    )
    response = _proxy("https://huggingface.co/api/models")
    assert response.status_code == 200
    assert _RedirectingClient.seen == [
        "https://huggingface.co/api/models",
        "https://huggingface.co/api/models?limit=20",
    ]


def test_redirect_loop_is_bounded(monkeypatch):
    _install_redirect_client(
        monkeypatch,
        ["https://huggingface.co/api/models"] * (hf_proxy.MAX_PROXY_REDIRECTS + 1),
    )
    with pytest.raises(HTTPException) as exc_info:
        _proxy("https://huggingface.co/api/models")
    assert exc_info.value.status_code == 502
    assert "redirect" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# Malformed authorities must produce a client error, not an unhandled 500.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co:notaport/api/models",
        "https://huggingface.co:99999/api/models",
        "https://[oops/api/models",
    ],
)
def test_malformed_authority_is_a_client_error(url):
    with pytest.raises(HTTPException) as exc_info:
        hf_proxy.validate_proxy_url(url)
    assert exc_info.value.status_code in (400, 403)


def test_proxy_responses_carry_the_marker_header(monkeypatch):
    upstream = _FakeUpstream(
        b'{"ok":true}', 200, httpx.Headers({"content-type": "application/json"})
    )
    _FakeAsyncClient.outcome = upstream
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    response = _proxy("https://huggingface.co/api/models")
    assert response.headers.get(hf_proxy.PROXY_MARKER_HEADER) == "1"


def test_upstream_404_still_passes_through_with_the_marker(monkeypatch):
    """A genuine hub 404 (missing repo) must reach the caller unchanged.

    This is why the marker header exists rather than a status allowlist: the
    frontend has to distinguish this from an older backend's catch-all 404.
    """
    upstream = _FakeUpstream(b'{"error":"Repo not found"}', 404, httpx.Headers({}))
    _FakeAsyncClient.outcome = upstream
    monkeypatch.setattr(hf_proxy.httpx, "AsyncClient", _FakeAsyncClient)

    response = _proxy("https://huggingface.co/api/models/nope")
    assert response.status_code == 404
    assert response.headers.get(hf_proxy.PROXY_MARKER_HEADER) == "1"

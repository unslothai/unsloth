# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The endpoint check reports when a mirror hands Studio's traffic back to huggingface.co."""

from __future__ import annotations

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import utils.hub_endpoint_check as check

MIRROR = "https://mirror.example"
API = f"/api/models/{check.PROBE_REPO}"
RESOLVE = f"/{check.PROBE_REPO}/resolve/main/{check.PROBE_FILE}"
TOKEN_ROUTE = f"/api/models/{check.PROBE_REPO}/xet-read-token/abc123"
MODEL_INFO = {"id": check.PROBE_REPO, "sha": "abc123"}


def _xet_headers(auth_url: str) -> dict:
    return {
        "X-Xet-Hash": "f" * 64,
        "Link": f'<{auth_url}>; rel="xet-auth", '
        f'<https://cas-server.xethub.hf.co/v1/reconstructions/{"f" * 64}>; '
        'rel="xet-reconstruction-info"',
    }


@pytest.fixture
def mirror(monkeypatch):
    """Route table for a fake endpoint: path -> handler(request) -> httpx.Response."""
    routes: dict = {}
    seen: list = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        assert "authorization" not in request.headers
        handler = routes.get((request.url.host, request.url.path))
        if handler is None:
            return httpx.Response(404)
        return handler(request)

    monkeypatch.setattr(check, "transport", httpx.MockTransport(handle))
    routes["seen"] = seen
    return routes


def _serve(routes, url: str, response):
    parts = httpx.URL(url)
    routes[(parts.host, parts.path)] = response if callable(response) else lambda _: response


def test_a_mirror_that_serves_everything_itself_passes(mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    _serve(
        mirror,
        MIRROR + RESOLVE,
        httpx.Response(
            302,
            headers = {
                "Location": "https://cdn.mirror-cdn.example/blob?sig=1",
                **_xet_headers(f"https://huggingface.co{TOKEN_ROUTE}"),
            },
        ),
    )
    _serve(
        mirror,
        MIRROR + TOKEN_ROUTE,
        httpx.Response(200, headers = {"X-Xet-Cas-Url": "https://cas.mirror.example"}),
    )
    result = check.check_hub_endpoint("mirror.example/")
    assert result == check.EndpointCheck(MIRROR, "ok", "mirror", "mirror")
    # The mirror's CDN is a signed URL of its own: the check never fetches it.
    assert not any("mirror-cdn" in url for _, url in mirror["seen"])


def test_a_whole_site_redirect_to_huggingface_is_reported(mirror):
    # hf-mirror.com outside mainland China answers every path with a 308 to huggingface.co.
    def to_official(request):
        return httpx.Response(
            308, headers = {"Location": f"https://huggingface.co{request.url.path}"}
        )

    _serve(mirror, MIRROR + API, to_official)
    _serve(mirror, MIRROR + RESOLVE, to_official)
    result = check.check_hub_endpoint(MIRROR)
    assert (result.api, result.download, result.xet) == ("official", "official", "none")
    assert not any("huggingface.co" in url for _, url in mirror["seen"])


def test_downloads_redirected_to_the_official_cdn_are_reported(mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    _serve(
        mirror,
        MIRROR + RESOLVE,
        httpx.Response(302, headers = {"Location": "https://us.aws.cdn.hf.co/xet-bridge-us/x"}),
    )
    result = check.check_hub_endpoint(MIRROR)
    assert (result.api, result.download) == ("ok", "official")


def test_xet_chunks_from_huggingface_storage_are_reported(mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    # A pass-through mirror forwards the official Xet headers; huggingface_hub rewrites the
    # huggingface.co token route onto the endpoint, so the check asks the mirror too.
    _serve(
        mirror,
        MIRROR + RESOLVE,
        httpx.Response(200, headers = _xet_headers(f"https://huggingface.co{TOKEN_ROUTE}")),
    )
    _serve(
        mirror,
        MIRROR + TOKEN_ROUTE,
        httpx.Response(200, json = {"casUrl": "https://cas-server.xethub.hf.co"}),
    )
    result = check.check_hub_endpoint(MIRROR)
    assert (result.download, result.xet) == ("mirror", "official")
    assert ("GET", MIRROR + TOKEN_ROUTE) in mirror["seen"]


def test_relative_redirects_on_the_mirror_are_followed(mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    _serve(
        mirror,
        MIRROR + RESOLVE,
        httpx.Response(307, headers = {"Location": f"/{check.PROBE_REPO}/resolve/abc123/x"}),
    )
    _serve(mirror, f"{MIRROR}/{check.PROBE_REPO}/resolve/abc123/x", httpx.Response(200))
    result = check.check_hub_endpoint(MIRROR)
    assert (result.download, result.xet) == ("mirror", "none")


@pytest.mark.parametrize(
    "api_response, detail",
    [
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(200, text = "<html>proxy login</html>"), "model metadata"),
        (httpx.Response(200, json = {"id": "someone/else"}), "model metadata"),
    ],
)
def test_a_mirror_without_the_model_api_fails_with_a_reason(mirror, api_response, detail):
    _serve(mirror, MIRROR + API, api_response)
    _serve(mirror, MIRROR + RESOLVE, httpx.Response(200))
    result = check.check_hub_endpoint(MIRROR)
    assert result.api == "failed"
    assert detail in result.detail


def test_an_unreachable_mirror_fails_instead_of_raising(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request = request)

    monkeypatch.setattr(check, "transport", httpx.MockTransport(refuse))
    result = check.check_hub_endpoint(MIRROR)
    assert (result.api, result.download, result.xet) == ("failed", "failed", "none")
    assert "ConnectError" in result.detail


def test_a_redirect_loop_on_the_mirror_fails(mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    _serve(mirror, MIRROR + RESOLVE, httpx.Response(302, headers = {"Location": RESOLVE}))
    result = check.check_hub_endpoint(MIRROR)
    assert result.download == "failed"
    assert "TooManyRedirects" in result.detail


@pytest.mark.parametrize(
    "host, official",
    [
        ("huggingface.co", True),
        ("HuggingFace.co.", True),
        ("cas-server.xethub.hf.co", True),
        ("us.aws.cdn.hf.co", True),
        ("hf-mirror.com", False),
        ("huggingface.co.evil.example", False),
        ("nothf.co", False),
        (None, False),
    ],
)
def test_official_hosts(host, official):
    assert check.is_official_host(host) is official


@pytest.mark.parametrize("raw", ["", "https://huggingface.co", "hf.co"])
def test_the_official_hub_has_nothing_to_check(raw):
    with pytest.raises(ValueError):
        check.check_hub_endpoint(raw)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: False
    return TestClient(app, raise_server_exceptions = False)


def test_route_reports_the_check(client, mirror):
    _serve(mirror, MIRROR + API, httpx.Response(200, json = MODEL_INFO))
    _serve(
        mirror,
        MIRROR + RESOLVE,
        httpx.Response(302, headers = {"Location": "https://huggingface.co/x"}),
    )
    response = client.post("/hub/check", json = {"hf_endpoint": "mirror.example"})
    assert response.status_code == 200
    assert response.json() == {
        "hf_endpoint": MIRROR,
        "api": "ok",
        "download": "official",
        "xet": "none",
        "detail": None,
    }


@pytest.mark.parametrize("bad", ["http://example.com", "https://u:p@example.com", ""])
def test_route_rejects_what_cannot_be_checked(client, mirror, bad):
    response = client.post("/hub/check", json = {"hf_endpoint": bad})
    assert response.status_code == 400
    assert mirror["seen"] == []


def test_an_api_key_cannot_make_the_server_probe_a_url(client, mirror):
    client.app.dependency_overrides[settings.authenticated_via_api_key] = lambda: True
    response = client.post("/hub/check", json = {"hf_endpoint": MIRROR})
    assert response.status_code == 403
    assert mirror["seen"] == []


def test_only_the_owner_runs_the_check(client, mirror, monkeypatch):
    from fastapi import HTTPException

    async def refuse():
        raise HTTPException(status_code = 403)

    monkeypatch.setattr(settings.policy, "require_owner", refuse)
    assert client.post("/hub/check", json = {"hf_endpoint": MIRROR}).status_code == 403
    assert mirror["seen"] == []

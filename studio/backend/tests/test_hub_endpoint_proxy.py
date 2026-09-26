# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The same-origin relay to a custom Hub endpoint and datasets server."""

from __future__ import annotations

from pathlib import Path
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hub import endpoint_proxy

MIRROR = "https://mirror.test/hf"
HUB = endpoint_proxy.relay_path(endpoint_proxy.HUB_PREFIX, MIRROR, "https://huggingface.co")
DATASETS = endpoint_proxy.relay_path(
    endpoint_proxy.DATASETS_SERVER_PREFIX, MIRROR, "https://huggingface.co"
)


def _mirror(request: httpx.Request) -> httpx.Response:
    if request.url.host == "down.test":
        raise httpx.ConnectError("refused", request = request)
    headers = {"Set-Cookie": "s=1", "Link": f'<{MIRROR}/api/models?cursor=2>; rel="next"'}
    return httpx.Response(401 if "gated" in request.url.path else 200, json = [], headers = headers)


@pytest.fixture
def proxy(monkeypatch):
    seen, state = [], {"session": True, "upstream": MIRROR}
    monkeypatch.setattr(
        endpoint_proxy, "transport", httpx.MockTransport(lambda r: seen.append(r) or _mirror(r))
    )
    monkeypatch.setattr(endpoint_proxy, "_clients", endpoint_proxy.weakref.WeakKeyDictionary())

    async def signed_in(_request) -> bool:
        return state["session"]

    monkeypatch.setattr(endpoint_proxy, "signed_in", signed_in)
    app = FastAPI()
    for prefix, pages in (
        (endpoint_proxy.HUB_PREFIX, True),
        (endpoint_proxy.DATASETS_SERVER_PREFIX, False),
    ):
        router = endpoint_proxy.build_router(
            prefix, lambda: state["upstream"], anonymous_pages = pages
        )
        app.include_router(router, prefix = prefix)
    with TestClient(app, follow_redirects = False) as client:
        yield client, seen, state


def test_a_session_is_relayed_with_only_the_hub_token(proxy):
    client, seen, state = proxy
    headers = {"Authorization": "Bearer session", "X-HF-Authorization": "Bearer hf_x"}
    response = client.get(f"{HUB}/api/models?search=qwen", headers = headers)
    assert (response.status_code, response.headers["x-hub-upstream"]) == (200, "1")
    assert "set-cookie" not in response.headers
    assert response.headers["link"] == f'<http://testserver{HUB}/api/models?cursor=2>; rel="next"'
    assert (str(seen[0].url), seen[0].headers["authorization"]) == (
        f"{MIRROR}/api/models?search=qwen",
        "Bearer hf_x",
    )

    gated = client.get(f"{HUB}/api/models/org/gated/revision/refs%2Fpr%2F1")
    assert (gated.status_code, gated.headers["x-hub-upstream"]) == (401, "1")
    assert seen[1].url.raw_path == b"/hf/api/models/org/gated/revision/refs%2Fpr%2F1"
    assert "authorization" not in seen[1].headers and "cookie" not in seen[1].headers

    state["upstream"] = "https://down.test"
    assert client.get(f"{DATASETS}/splits").status_code == 409
    down = client.get(
        f"{endpoint_proxy.relay_path(endpoint_proxy.DATASETS_SERVER_PREFIX, 'https://down.test', '')}/splits"
    )
    assert (down.status_code, "x-hub-upstream" in down.headers) == (502, False)
    assert len(seen) == 3


def test_without_a_session_nothing_reaches_the_endpoint(proxy):
    client, seen, state = proxy
    state["session"] = False
    image = client.get(f"{HUB}/org/m/resolve/main/a.png?download=1")
    assert image.headers["location"] == f"{MIRROR}/org/m/resolve/main/a.png?download=1"
    for path, headers in (
        (f"{HUB}/api/models", {}),
        (f"{HUB}/org/m/resolve/main/README.md", {"Authorization": "Bearer stale"}),
        (f"{DATASETS}/splits?dataset=a/b", {}),
    ):
        refused = client.get(path, headers = headers)
        assert (refused.status_code, "x-hub-upstream" in refused.headers) == (401, False)
    bad = ("%2e%2e", "a%5Cb", "a%2F..%2Fb")
    assert {client.get(f"{HUB}/org/{b}/admin").status_code for b in bad} == {400}
    state["upstream"] = "http://10.0.0.5:8080"
    lan = endpoint_proxy.relay_path(endpoint_proxy.HUB_PREFIX, state["upstream"], "")
    hidden = client.get(f"{lan}/org/m/resolve/main/a.png")
    assert (hidden.status_code, "location" in hidden.headers) == (401, False)
    assert seen == []


def test_the_relay_tag_is_not_a_bare_hash_of_the_endpoint():
    import hashlib
    from hub import endpoint_proxy

    endpoint = "https://10.0.0.5:8443"
    tag = endpoint_proxy._tag(endpoint)
    assert tag == endpoint_proxy._tag(endpoint) and len(tag) == 12
    assert tag != hashlib.sha256(endpoint.encode()).hexdigest()[:12]


@pytest.mark.parametrize("peer, location", [("127.0.0.1", True), ("203.0.113.9", False)])
def test_an_anonymous_redirect_names_a_saved_endpoint_only_to_a_loopback_browser(
    proxy, monkeypatch, peer, location
):
    import utils.hub_settings as hub_settings

    client, seen, state = proxy
    state["session"] = False
    monkeypatch.setattr(endpoint_proxy, "client_ip", lambda _request: peer)
    monkeypatch.setattr(hub_settings, "_saved_only_endpoints", frozenset({MIRROR}), raising = False)
    page = client.get(f"{HUB}/org/m/resolve/main/a.png")
    assert (page.status_code, "location" in page.headers) == ((302, True) if location else (401, False))
    assert seen == []

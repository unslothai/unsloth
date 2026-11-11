# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Configured reverse-proxy UI access, using the real frontend and ASGI middleware."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from utils.local_proxy import (
    PROXY_ORIGIN_ENV,
    local_proxy_frontend_request,
    validate_local_proxy_origin,
)


HOST = "studio.example.ts.net"
ORIGIN = "https://" + HOST
PROXY_HEADERS = {
    "Host": HOST,
    "X-Forwarded-Host": HOST,
    "X-Forwarded-Proto": "https",
    "X-Forwarded-For": "100.64.0.10",
}


@pytest.fixture(autouse = True)
def proxy_disabled_by_default(monkeypatch):
    monkeypatch.delenv(PROXY_ORIGIN_ENV, raising = False)


def _scope(server = ("127.0.0.1", 8888), headers = None):
    return {
        "type": "http",
        "server": server,
        "client": ("100.64.0.10", 0),
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or PROXY_HEADERS).items()
        ],
    }


def _client(
    tmp_path,
    *,
    server = ("127.0.0.1", 8888),
    tunnel_only = True,
):
    import main
    from routes.auth import router

    (tmp_path / "index.html").write_text("<html><head></head><body>Studio</body></html>")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("export const ready = true;")
    app = FastAPI()
    app.include_router(router, prefix = "/api/auth")
    main.setup_frontend(app, tmp_path, tunnel_only = tunnel_only)

    async def transport(scope, receive, send):
        scope["server"] = server
        await ProxyHeadersMiddleware(app)(scope, receive, send)

    return TestClient(transport, client = ("127.0.0.1", 51000))


@pytest.mark.parametrize("path", ["/", "/chat", "/assets/app.js"])
def test_proxy_frontend_is_opt_in(tmp_path, path):
    assert _client(tmp_path).get(path, headers = PROXY_HEADERS).status_code == 404


@pytest.mark.parametrize("path", ["/", "/chat", "/index.html", "/assets/app.js"])
@pytest.mark.parametrize("backend_port", [8888, 43210])
def test_configured_proxy_serves_ui_and_assets(tmp_path, monkeypatch, path, backend_port):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    client = _client(tmp_path, server = ("127.0.0.1", backend_port))
    response = client.get(path, headers = {**PROXY_HEADERS, "Origin": ORIGIN})
    assert response.status_code == 200
    assert "__UNSLOTH_BOOTSTRAP__" not in response.text


@pytest.mark.parametrize("authority", [HOST.upper(), HOST + ":443", HOST + ":8443"])
def test_configured_external_port_and_host_case(monkeypatch, authority):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, "https://" + authority)
    headers = {**PROXY_HEADERS, "Host": authority, "X-Forwarded-Host": authority}
    assert local_proxy_frontend_request(_scope(headers = headers))


@pytest.mark.parametrize("configured_dot", [False, True])
@pytest.mark.parametrize("host_dot", [False, True])
@pytest.mark.parametrize("forwarded_dot", [False, True])
def test_dns_root_dot_in_routing_headers(monkeypatch, configured_dot, host_dot, forwarded_dot):
    origin = ORIGIN + ("." if configured_dot else "")
    monkeypatch.setenv(PROXY_ORIGIN_ENV, origin)
    headers = {
        **PROXY_HEADERS,
        "Host": HOST + ("." if host_dot else "") + ":443",
        "X-Forwarded-Host": HOST + ("." if forwarded_dot else ""),
        "Origin": origin,
    }
    assert local_proxy_frontend_request(_scope(headers = headers))


@pytest.mark.parametrize("configured_dot", [False, True])
def test_dns_root_dot_does_not_expand_browser_origin(monkeypatch, configured_dot):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN + ("." if configured_dot else ""))
    headers = {**PROXY_HEADERS, "Origin": ORIGIN + ("" if configured_dot else ".")}
    assert not local_proxy_frontend_request(_scope(headers = headers))


@pytest.mark.parametrize(
    "server", [None, ("0.0.0.0", 8888), ("192.0.2.1", 8888), ("::", 8888), ("::1%lo0", 8888)]
)
def test_forwarded_headers_do_not_replace_loopback_listener(monkeypatch, server):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    assert not local_proxy_frontend_request(_scope(server = server))


@pytest.mark.parametrize("server", [("::1", 8888), ("::ffff:127.0.0.1", 8888)])
def test_ipv6_loopback_listeners(monkeypatch, server):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    assert local_proxy_frontend_request(_scope(server = server))


@pytest.mark.parametrize(
    "header,value",
    [
        ("Host", "localhost:8888"),
        ("Host", HOST + ".evil.example"),
        ("Host", HOST + ":8443"),
        ("Host", "user@" + HOST),
        ("X-Forwarded-Host", "evil.example"),
        ("X-Forwarded-Host", ""),
        ("X-Forwarded-Proto", "http"),
        ("X-Forwarded-Proto", "https,http"),
        ("Origin", "https://evil.example"),
        ("Origin", "http://" + HOST),
        ("Origin", ORIGIN + "/path"),
        ("Origin", "null"),
        ("Origin", ""),
    ],
)
@pytest.mark.parametrize("path", ["/", "/assets/app.js"])
def test_wrong_authority_or_origin_cannot_get_frontend(tmp_path, monkeypatch, header, value, path):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    headers = {**PROXY_HEADERS, header: value}
    assert _client(tmp_path).get(path, headers = headers).status_code == 404


@pytest.mark.parametrize("name", [b"host", b"x-forwarded-host", b"x-forwarded-proto", b"origin"])
def test_duplicate_headers_are_rejected(monkeypatch, name):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    scope = _scope()
    scope["headers"].extend([(name, b"first"), (name, b"second")])
    assert not local_proxy_frontend_request(scope)


@pytest.mark.parametrize(
    "value",
    [
        "http://studio.example",
        "https://studio.example/path",
        "https://studio.example/",
        "https://user:password@studio.example",
        "https://studio.example?",
        "https://studio.example#",
        "https://*.example",
        "https://studio.example:0",
        "https://studio.example:65536",
        "https://studio.example:",
        "https://one.example,https://two.example",
        "https://[invalid",
        "https://studio.example\n.evil.example",
        "https://studio.example\\evil",
        "https://",
        "https://studio.example..",
        "https://studio..example",
        "https://.studio.example",
        "https://studio.-example",
        "https://studio.example-",
    ],
)
def test_malformed_configuration_fails_closed(monkeypatch, value):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, value)
    with pytest.raises(ValueError, match = PROXY_ORIGIN_ENV):
        validate_local_proxy_origin()
    assert not local_proxy_frontend_request(_scope())
    from utils.host_policy import tunnel_connector_active

    assert tunnel_connector_active(), "bad configuration must not re-enable local shortcuts"


@pytest.mark.parametrize("tunnel_only", [True, False])
def test_proxy_never_gets_bootstrap_even_if_it_strips_headers(tmp_path, monkeypatch, tunnel_only):
    import main

    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)

    def forbidden(*args):
        pytest.fail("configured proxy mode must never call bootstrap injection")

    monkeypatch.setattr(main, "_inject_bootstrap", forbidden)
    client = _client(tmp_path, tunnel_only = tunnel_only)
    assert client.get("/", headers = PROXY_HEADERS).status_code == 200
    # A badly configured proxy may appear to be a direct localhost browser.
    response = client.get("/", headers = {"Host": "127.0.0.1:8888"})
    assert response.status_code == 200
    assert "__UNSLOTH_BOOTSTRAP__" not in response.text


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("path", ["/", "/assets/app.js"])
def test_direct_loopback_frontend_is_preserved(tmp_path, monkeypatch, configured, path):
    import main

    # Bootstrap behavior has its own coverage; keep this admission check away
    # from the developer's account database when the setting is absent.
    monkeypatch.setattr(main, "_inject_bootstrap", lambda html, app: (html, None))
    if configured:
        monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    client = _client(tmp_path)
    assert client.get(path, headers = {"Host": "127.0.0.1:8888"}).status_code == 200


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("path", ["/", "/assets/app.js"])
def test_loopback_authority_with_proxy_headers_stays_rejected(
    tmp_path, monkeypatch, configured, path
):
    if configured:
        monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    headers = {**PROXY_HEADERS, "Host": "127.0.0.1:8888"}
    assert _client(tmp_path).get(path, headers = headers).status_code == 404


def test_invalid_setting_prevents_frontend_mount(tmp_path, monkeypatch):
    import main
    monkeypatch.setenv(PROXY_ORIGIN_ENV, "not-an-origin")
    with pytest.raises(ValueError, match = PROXY_ORIGIN_ENV):
        main.setup_frontend(FastAPI(), tmp_path)


def test_proxy_headers_do_not_authenticate_api_calls(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ORIGIN_ENV, ORIGIN)
    client = _client(tmp_path)
    assert client.get("/api/auth/api-keys", headers = PROXY_HEADERS).status_code == 401
    assert client.get("/api/not-a-route", headers = PROXY_HEADERS).status_code == 404
    assert client.get("/v1/not-a-route", headers = PROXY_HEADERS).status_code == 404

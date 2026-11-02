# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Forwarded headers rewrite the caller only where they are trusted.

uvicorn's defaults (proxy_headers=True, forwarded_allow_ips="127.0.0.1") let any loopback peer set
request.client from X-Forwarded-For. Behind a same-host proxy that relays the caller's header unchanged,
every per-IP limiter would key on text the caller chose. These tests drive uvicorn's own app pipeline,
built with the arguments run.py passes, so they exercise the rewriting uvicorn would really apply."""

import asyncio
import inspect
import types
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from routes import auth as auth_routes
from utils import keyless_api_access as kl
from utils.client_ip import client_ip
from utils.forwarded_headers import uvicorn_forwarded_kwargs, wrap_direct_server
from utils.host_policy import is_loopback_host

_TUNNEL_URL = "https://quiet-river.trycloudflare.com"
_TUNNEL_HOST = "quiet-river.trycloudflare.com"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising = False)

    def clear():
        for table in (auth_routes._LOGIN_BUCKETS, auth_routes._LOGIN_IP_BUCKETS):
            table.clear()
        for shard in auth_routes._LOGIN_IP_OVERFLOW:
            shard.clear()

    clear()
    yield
    clear()


async def _login(request):
    from models.auth import AuthLoginRequest
    try:
        await auth_routes.login(
            AuthLoginRequest(username = "nobody", password = "wrong-password"), request
        )
    except HTTPException as exc:
        status = exc.status_code
    else:
        status = 200
    return JSONResponse(
        {
            "status": status,
            "client": request.client.host,
            "scheme": request.url.scheme,
            "bucket": list(auth_routes._bucket_key(request, "nobody")),
            "client_ip": client_ip(request),
        }
    )


def _served(*, is_colab = False, cloudflare_url = None):
    """uvicorn's loaded app, built the way run.py builds it."""
    inner = Starlette(routes = [Route("/login", _login, methods = ["POST"])])
    inner.state.cloudflare_url = cloudflare_url
    kwargs = uvicorn_forwarded_kwargs(is_colab = is_colab)
    served = inner if kwargs["proxy_headers"] else wrap_direct_server(inner, inner.state)
    config = uvicorn.Config(served, log_level = "warning", **kwargs)
    config.load()
    return config.loaded_app


def _post_all(
    app,
    header_sets,
    peer = "127.0.0.1",
    host = "127.0.0.1:8888",
):
    async def run():
        transport = httpx.ASGITransport(app = app, client = (peer, 40000))
        async with httpx.AsyncClient(transport = transport, base_url = f"http://{host}") as c:
            return [(await c.post("/login", headers = h)).json() for h in header_sets]

    return asyncio.run(run())


@pytest.fixture
def _no_user(monkeypatch):
    monkeypatch.setattr(auth_routes.policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(auth_routes.storage, "get_user_and_secret", lambda username: None)
    monkeypatch.setattr(auth_routes.hashing, "equalize_login_work", lambda password: None)


def test_spoofed_forwarded_for_from_a_loopback_peer_does_not_pick_the_login_bucket(_no_user):
    """A same-host proxy relaying the caller's X-Forwarded-For: each guess names a fresh address."""
    attempts = auth_routes._LOGIN_MAX_FAILS + 3
    seen = _post_all(
        _served(),
        [
            {"x-forwarded-for": f"203.0.113.{i}", "x-forwarded-proto": "https"}
            for i in range(attempts)
        ],
    )
    assert {r["client"] for r in seen} == {"127.0.0.1"}
    assert {tuple(r["bucket"]) for r in seen} == {("127.0.0.1", "nobody")}
    assert {r["scheme"] for r in seen} == {"http"}
    statuses = [r["status"] for r in seen]
    assert statuses[: auth_routes._LOGIN_MAX_FAILS] == [401] * auth_routes._LOGIN_MAX_FAILS
    assert statuses[-1] == 429


def test_the_operator_opt_in_still_honours_forwarded_headers(_no_user, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")
    assert uvicorn_forwarded_kwargs(is_colab = False) == {"proxy_headers": True}
    seen = _post_all(
        _served(),
        [{"x-forwarded-for": "198.51.100.7", "x-forwarded-proto": "https"}],
    )
    assert seen[0]["client"] == "198.51.100.7"
    assert seen[0]["bucket"] == ["198.51.100.7", "nobody"]
    assert seen[0]["scheme"] == "https"


def test_colab_keeps_its_proxy_header_trust():
    assert uvicorn_forwarded_kwargs(is_colab = True) == {
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }


def test_run_server_and_the_lan_listener_use_the_shared_arguments():
    backend = Path(__file__).resolve().parents[1]
    run_src = (backend / "run.py").read_text(encoding = "utf-8")
    assert "uvicorn_forwarded_kwargs(is_colab = _IS_COLAB)" in run_src
    assert "wrap_direct_server(app, app.state)" in run_src
    import lan_access

    assert "uvicorn_forwarded_kwargs(is_colab = False)" in inspect.getsource(
        lan_access._listener_config
    )
    config = lan_access._listener_config(Starlette(), "192.168.1.5", 8888)
    assert config.proxy_headers is False


def test_the_managed_tunnel_keeps_its_visitor_address_and_https_scheme(_no_user):
    headers = {
        "host": _TUNNEL_HOST,
        "cf-connecting-ip": "198.51.100.7",
        "x-forwarded-for": "198.51.100.7",
        "x-forwarded-proto": "https",
    }
    seen = _post_all(_served(cloudflare_url = _TUNNEL_URL), [headers], host = _TUNNEL_HOST)
    assert seen[0]["client"] == "198.51.100.7"
    assert seen[0]["scheme"] == "https"
    assert seen[0]["client_ip"] == "198.51.100.7"


@pytest.mark.parametrize(
    "cloudflare_url, host, peer",
    [
        (None, _TUNNEL_HOST, "127.0.0.1"),  # no managed tunnel
        (_TUNNEL_URL, "other.trycloudflare.com", "127.0.0.1"),  # Host names another site
        (_TUNNEL_URL, _TUNNEL_HOST, "192.168.1.50"),  # not the local cloudflared
    ],
)
def test_cf_connecting_ip_is_ignored_outside_the_managed_tunnel(
    _no_user, cloudflare_url, host, peer
):
    headers = {"host": host, "cf-connecting-ip": "198.51.100.7", "x-forwarded-proto": "https"}
    seen = _post_all(_served(cloudflare_url = cloudflare_url), [headers], peer = peer, host = host)
    assert seen[0]["client"] == peer
    assert seen[0]["scheme"] == "http"
    assert seen[0]["client_ip"] != "198.51.100.7"


def _request(
    peer,
    headers = (),
    *,
    cloudflare_url = None,
    host = "127.0.0.1:8888",
    server = None,
):
    raw = [(b"host", host.encode())] + [(k.encode(), v.encode()) for k, v in headers]
    state = types.SimpleNamespace(
        bind_host = "127.0.0.1", secure = False, cloudflare_url = cloudflare_url
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "root_path": "",
            "headers": raw,
            "client": (peer, 50000),
            "server": server or ("127.0.0.1", 8888),
            "scheme": "http",
            "query_string": b"",
            "app": types.SimpleNamespace(state = state),
        }
    )


def test_client_ip_ignores_cf_connecting_ip_without_the_managed_tunnel():
    relayed = client_ip(_request("127.0.0.1", [("cf-connecting-ip", "198.51.100.7")]))
    assert relayed != "198.51.100.7"
    # Still not a local caller for the reachability checks that read client_ip().
    assert not is_loopback_host(relayed)
    assert client_ip(_request("127.0.0.1")) == "127.0.0.1"


def test_client_ip_honours_cf_connecting_ip_through_the_managed_tunnel():
    request = _request(
        "127.0.0.1",
        [("cf-connecting-ip", "198.51.100.7")],
        cloudflare_url = _TUNNEL_URL,
        host = _TUNNEL_HOST,
    )
    assert client_ip(request) == "198.51.100.7"
    forged_host = _request(
        "127.0.0.1",
        [("cf-connecting-ip", "198.51.100.7")],
        cloudflare_url = _TUNNEL_URL,
        host = "127.0.0.1:8888",
    )
    assert client_ip(forged_host) != "198.51.100.7"


_RELAY_HEADERS = [
    ("x-forwarded-for", "203.0.113.9"),
    ("x-real-ip", "203.0.113.9"),
    ("forwarded", "for=203.0.113.9"),
    ("cf-connecting-ip", "203.0.113.9"),
    ("x-forwarded-host", "studio.example.com"),
    ("via", "1.1 proxy"),
    ("x-forwarded-proto", "https"),
]


def _lan_request(headers = ()):
    return _request(
        "192.168.1.90",
        headers,
        host = "192.168.1.24:8888",
        server = ("192.168.1.24", 8888),
    )


@pytest.fixture
def _lan_listener(monkeypatch):
    import lan_access
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"]},
    )


def test_lan_keyless_inference_still_serves_a_direct_lan_client(_lan_listener):
    assert kl.keyless_transport_allowed(_lan_request(), "inference") is True


@pytest.mark.parametrize("header", _RELAY_HEADERS)
def test_lan_keyless_inference_refuses_a_relayed_request(_lan_listener, header):
    assert kl.keyless_transport_allowed(_lan_request([header]), "inference") is False

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account isolation at desktop, keyless and shared-tunnel entry points."""

import asyncio
import io
import secrets
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import StreamingResponse

from auth import authentication, policy, storage
from auth.bootstrap_timeout import enforce_bootstrap_password_deadline
from routes import auth as auth_routes
from utils import keyless_api_access as keyless
from utils.account_context import OWNER, arun_as, current_account, current_account_id, run_as


@pytest.fixture(autouse = True)
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    keyless._reset_scope_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield
    policy.invalidate_account_cache()
    keyless._reset_scope_cache()


def add_managed(*, must_change = False):
    storage.create_initial_user(
        "alice",
        "alice-password",
        secrets.token_urlsafe(32),
        must_change_password = must_change,
    )
    return storage.get_account("alice")


def client():
    app = FastAPI()
    app.state.secure = True
    app.state.bind_host = "127.0.0.1"
    app.state.cloudflare_url = "https://shared.trycloudflare.com"
    app.include_router(auth_routes.router, prefix = "/api/auth")
    app.add_middleware(keyless.KeylessToolPolicyMiddleware)

    @app.get("/account")
    async def account(subject = Depends(authentication.get_current_subject)):
        return {"subject": subject, "account_id": current_account_id()}

    @app.get(
        "/owner",
        dependencies = [
            Depends(authentication.get_current_subject),
            Depends(policy.require_owner),
        ],
    )
    async def owner():
        return {"account_id": current_account_id()}

    @app.api_route("/events", methods = ["GET", "POST"])
    async def events(subject = Depends(authentication.get_current_subject)):
        async def stream():
            await asyncio.sleep(0)
            yield f"data: {subject}:{current_account_id()}\n\n"

        return StreamingResponse(stream(), media_type = "text/event-stream")

    return TestClient(app, client = ("127.0.0.1", 55000))


def test_desktop_multi_validates_secret_and_never_mints_tokens(monkeypatch):
    raw = storage.create_desktop_secret()
    add_managed()

    def no_token(**kwargs):
        pytest.fail("Multi-account desktop auth must not mint a session")

    monkeypatch.setattr(auth_routes, "create_access_token", no_token)
    monkeypatch.setattr(auth_routes, "create_refresh_token", no_token)
    with client() as http:
        response = http.post("/api/auth/desktop-login", json = {"secret": raw})
        assert response.status_code == 200
        assert response.content == b'{"login_required":true,"login_mode":"multi"}'
        assert http.post("/api/auth/desktop-login", json = {"secret": raw + "x"}).status_code == 401


def test_desktop_single_response_bytes_and_return_to_single(monkeypatch):
    raw = storage.create_desktop_secret()
    minted = []

    def access(**kwargs):
        minted.append(kwargs)
        return "access"

    monkeypatch.setattr(auth_routes, "create_access_token", access)
    monkeypatch.setattr(auth_routes, "create_refresh_token", lambda **kwargs: "refresh")
    expected = b'{"access_token":"access","refresh_token":"refresh","token_type":"bearer","must_change_password":false}'
    with client() as http:
        single = http.post("/api/auth/desktop-login", json = {"secret": raw})
        assert single.content == expected
        add_managed()
        assert http.post("/api/auth/desktop-login", json = {"secret": raw}).json()["login_required"]
        storage.delete_user("alice")
        assert http.post("/api/auth/desktop-login", json = {"secret": raw}).content == expected
    assert len(minted) == 2
    assert all(
        call == {"subject": "unsloth", "desktop": True, "secret": storage.get_jwt_secret("unsloth")}
        for call in minted
    )


@pytest.mark.parametrize("desktop", [False, True])
def test_managed_account_cannot_set_desktop_initial_password(desktop):
    add_managed(must_change = True)
    before = storage.get_user_and_secret("alice")
    token = authentication.create_access_token(subject = "alice", desktop = desktop)
    with client() as http:
        response = http.post(
            "/api/auth/desktop-initial-password",
            headers = {"Authorization": f"Bearer {token}"},
            json = {"new_password": "changed-password"},
        )
    assert response.status_code == 403
    assert storage.get_user_and_secret("alice") == before


def request(
    *,
    scope = "full",
    lan = False,
    headers = None,
):
    address = "192.168.1.2" if lan else "127.0.0.1"
    app = SimpleNamespace(
        state = SimpleNamespace(
            bind_host = address,
            secure = False,
            cloudflare_url = None,
            lan_access_launch_managed = lan,
            lan_access_launch_addresses = [address] if lan else [],
            lan_access_port = 8888,
        )
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models" if scope == "inference" else "/account",
            "root_path": "",
            "query_string": b"",
            "scheme": "http",
            "server": (address, 8888),
            "client": ("192.168.1.3" if lan else "127.0.0.1", 55000),
            "headers": [(b"host", f"{address}:8888".encode()), *(headers or [])],
            "app": app,
        }
    )


@pytest.mark.parametrize("scope,lan", [("full", False), ("inference", False), ("inference", True)])
@pytest.mark.parametrize("bearer", [None, "not-needed", "lm-studio", "ollama"])
def test_keyless_refused_for_every_multi_account_entry(scope, lan, bearer, monkeypatch):
    monkeypatch.setattr(keyless, "get_keyless_api_access_scope", lambda: scope)
    headers = [] if bearer is None else [(b"authorization", f"Bearer {bearer}".encode())]
    req = request(scope = scope, lan = lan, headers = headers)
    assert keyless.keyless_request_allowed(req)
    alice = add_managed()
    assert not keyless.keyless_request_allowed(req)
    assert not keyless.asgi_request_is_keyless(req.scope, (scope, True))

    async def resolve():
        return await authentication.get_current_subject(await authentication.security(req))

    with pytest.raises(HTTPException) as error:
        asyncio.run(arun_as(alice, resolve()))
    assert error.value.status_code == 401
    storage.delete_user("alice")
    assert keyless.keyless_request_allowed(req)


@pytest.mark.parametrize("bearer", [None, "not-needed"])
def test_single_keyless_admission_explicitly_rebinds_owner(bearer, monkeypatch):
    # Start with a foreign context to prove that admission cannot inherit it.
    alice = add_managed()
    storage.delete_user("alice")
    monkeypatch.setattr(keyless, "get_keyless_api_access_scope", lambda: "full")
    headers = [] if bearer is None else [(b"authorization", f"Bearer {bearer}".encode())]
    req = request(headers = headers)

    async def resolve():
        credentials = await authentication.security(req)
        subject = await authentication.get_current_subject(credentials)
        return subject, current_account()

    assert asyncio.run(arun_as(alice, resolve())) == ("unsloth", OWNER)


def test_recorded_keyless_admission_is_refused_after_account_creation(monkeypatch):
    req = request()
    keyless.mark_keyless_admission(req, True)
    assert keyless.request_was_admitted_keyless(req) is True
    add_managed()
    assert keyless.request_was_admitted_keyless(req) is False
    with pytest.raises(HTTPException):
        asyncio.run(authentication.security(req))


@pytest.mark.parametrize("subject", ["unsloth", "alice"])
@pytest.mark.parametrize("credential", ["jwt", "api_key"])
@pytest.mark.parametrize("transport", ["tunnel", "lan"])
def test_network_request_uses_account_credentials(subject, credential, transport, monkeypatch):
    alice = add_managed()
    monkeypatch.setattr(keyless, "_read_settings", lambda: ("full", True))
    token = (
        authentication.create_access_token(subject = subject)
        if credential == "jwt"
        else storage.create_api_key(subject, "network")[0]
    )
    headers = {"Authorization": f"Bearer {token}", "Host": "192.168.1.2:8888"}
    if transport == "tunnel":
        headers.update({"Host": "shared.trycloudflare.com", "cf-connecting-ip": "203.0.113.7"})
    expected = OWNER.account_id if subject == "unsloth" else alice.account_id
    with client() as http:
        response = http.get("/account", headers = headers)
        assert response.status_code == 200
        assert response.json() == {"subject": subject, "account_id": expected}
        assert http.get("/owner", headers = headers).status_code == (
            200 if subject == "unsloth" else 403
        )
        for method in ("GET", "POST"):
            response = http.request(method, "/events", headers = headers)
            assert response.status_code == 200
            assert response.text == f"data: {subject}:{expected}\n\n"
    assert current_account() == OWNER


@pytest.mark.parametrize("bearer", [None, "not-needed", "invalid-token"])
def test_tunnel_never_admits_keyless_even_with_one_account(bearer, monkeypatch):
    monkeypatch.setattr(keyless, "_read_settings", lambda: ("full", True))
    headers = {"Host": "shared.trycloudflare.com", "cf-connecting-ip": "203.0.113.7"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    with client() as http:
        assert http.get("/account", headers = headers).status_code == 401


def test_bootstrap_html_single_bytes_and_multi_suppression(monkeypatch):
    import main
    import secrets as secrets_module

    monkeypatch.setattr(storage, "requires_password_change", lambda username: username == "unsloth")
    monkeypatch.setattr(secrets_module, "token_urlsafe", lambda size: "fixed-nonce")
    app = SimpleNamespace(state = SimpleNamespace(bootstrap_password = "owner-bootstrap"))
    html = b"<html><head></head><body>Studio</body></html>"
    expected = b'<html><head><script nonce="fixed-nonce">window.__UNSLOTH_BOOTSTRAP__={"username": "unsloth", "password": "owner-bootstrap"}</script></head><body>Studio</body></html>'
    assert main._inject_bootstrap(html, app) == (expected, "fixed-nonce")
    add_managed()
    assert main._inject_bootstrap(html, app) == (html, None)
    storage.delete_user("alice")
    assert main._inject_bootstrap(html, app) == (expected, "fixed-nonce")


def test_secure_banner_bytes_and_shared_url_survive_account_creation(monkeypatch, capsys):
    import run
    import startup_banner

    monkeypatch.setattr(run, "_cloudflare_url", "https://shared.trycloudflare.com")
    monkeypatch.setattr(run, "_public_reachable", None)
    monkeypatch.setattr(startup_banner, "stdout_supports_color", lambda: False)
    divider = "─" * 52
    expected = (
        "\n🦥 Unsloth Studio is running (secure)\n"
        f"{divider}\n"
        "  Secure link access via Cloudflare: https://shared.trycloudflare.com\n"
        "  On this machine only: http://127.0.0.1:8888/\n"
        f"{divider}\n"
        "Server-side tools are DISABLED (--disable-tools).\n"
        "\n  To stop Unsloth Studio: press Ctrl+C (Control+C, not Command+C, on macOS).\n"
        f"{divider}\n\n"
    ).encode()
    run._emit_secure_startup_output(8888, enable_tools = False)
    assert capsys.readouterr().out.encode() == expected
    add_managed()
    run._emit_secure_startup_output(8888, enable_tools = False)
    assert capsys.readouterr().out.encode() == expected


@pytest.mark.parametrize("owner_pending", [False, True])
def test_timeout_checks_only_owner_even_in_managed_context(owner_pending, monkeypatch, capsys):
    alice = add_managed(must_change = True)
    checked = []
    stopped = []

    def pending(username):
        checked.append(username)
        return owner_pending if username == "unsloth" else True

    monkeypatch.setattr(storage, "requires_password_change", pending)
    result = run_as(
        alice,
        enforce_bootstrap_password_deadline,
        storage,
        lambda: stopped.append(True),
        timeout_seconds = 1,
    )
    assert result is owner_pending
    assert checked == ["unsloth"]
    assert bool(stopped) is owner_pending
    assert "alice" not in capsys.readouterr().err


def test_terminal_gate_ignores_managed_setup_and_changes_only_owner(monkeypatch):
    import run
    from auth import terminal_prompt

    alice = add_managed(must_change = True)
    output = io.StringIO()
    monkeypatch.setattr(run.sys, "stderr", output)
    monkeypatch.setattr(run, "_stream_isatty", lambda stream: True)
    kwargs = dict(
        tunnel_will_start = True, host = "127.0.0.1", secure = True, api_only = False, frontend_served = True
    )
    assert run_as(alice, run._terminal_password_gate, **kwargs) == (True, False)
    assert output.getvalue() == ""
    before = storage.get_user_and_secret("alice")
    with storage.get_connection() as connection:
        connection.execute(
            "UPDATE auth_user SET must_change_password = 1 WHERE username = 'unsloth'"
        )
    keys = iter("new-owner-password\nnew-owner-password\n")
    monkeypatch.setattr(terminal_prompt, "_getch", lambda: next(keys))
    assert run_as(alice, run._terminal_password_gate, **kwargs) == (True, True)
    assert "Password updated for 'unsloth'." in output.getvalue()
    assert "alice" not in output.getvalue()
    assert storage.get_user_and_secret("alice") == before

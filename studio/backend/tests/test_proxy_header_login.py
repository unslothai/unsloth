# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sign-in from a trusted forward-auth proxy header (UNSLOTH_STUDIO_TRUSTED_USER_HEADER, #19842)."""

import importlib.util
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import authentication, policy, proxy_header, storage
from state import active_generations
from utils.account_context import OWNER, bind_account, reset_account
from utils.paths import storage_roots

_PROXY = {"host": "studio.example.com", "x-forwarded-proto": "https"}


def _load_route(name):
    path = Path(__file__).resolve().parents[1] / "routes" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_proxy_login_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(storage_roots.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "studio" / "auth" / "auth.db")
    monkeypatch.setattr(
        storage, "_BOOTSTRAP_PW_PATH", storage.DB_PATH.parent / ".bootstrap_password"
    )
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(64))
    auth = _load_route("auth")
    accounts = _load_route("accounts")
    app = FastAPI()
    app.include_router(auth.router, prefix = "/api/auth")
    app.include_router(accounts.router, prefix = "/api/accounts")
    return app


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROXY_HOSTS", "studio.example.com")
    monkeypatch.setenv(proxy_header.USER_HEADER_ENV, "Remote-User")
    monkeypatch.delenv(proxy_header.PROXY_SECRET_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    token = bind_account(OWNER)
    policy.invalidate_account_cache()
    active_generations.reset_for_tests()
    app = _app(tmp_path, monkeypatch)
    # A local proxy: the connection lands on Studio's loopback socket from a loopback peer.
    with TestClient(
        app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)
    ) as client:
        yield client
    active_generations.reset_for_tests()
    reset_account(token)
    policy.invalidate_account_cache()
    storage._reset_api_key_hash_cache()


def _owner_headers():
    return {"Authorization": "Bearer " + authentication.create_access_token("unsloth")}


def _set_up_account(client, username, password = "permanent-password"):
    body = client.post(
        "/api/accounts", headers = _owner_headers(), json = {"username": username}
    ).json()
    session = client.post(
        "/api/auth/login", json = {"username": username, "password": body["setup_code"]}
    ).json()
    response = client.post(
        "/api/auth/change-password",
        headers = {"Authorization": "Bearer " + session["access_token"]},
        json = {"current_password": body["setup_code"], "new_password": password},
    )
    assert response.status_code == 200
    return body["account"]["account_id"]


def _session(client, **headers):
    return client.get("/api/auth/proxy-session", headers = {**_PROXY, **headers})


def _proxy_login(client, **headers):
    return client.post("/api/auth/proxy-login", headers = {**_PROXY, **headers})


def test_the_proxy_user_signs_in_as_their_existing_account(env):
    account_id = _set_up_account(env, "alice")
    assert _session(env, **{"remote-user": "Alice"}).json() == {"username": "alice"}
    response = _proxy_login(env, **{"remote-user": "Alice"})
    assert response.status_code == 200
    token = response.json()
    assert token["account_id"] == account_id
    assert token["must_change_password"] is False
    # The issued pair is an ordinary session: it reaches the API and refreshes.
    me = env.get("/api/auth/api-keys", headers = {"Authorization": "Bearer " + token["access_token"]})
    assert me.status_code == 200
    refreshed = env.post("/api/auth/refresh", json = {"refresh_token": token["refresh_token"]})
    assert refreshed.status_code == 200
    assert refreshed.json()["account_id"] == account_id


def test_the_owner_signs_in_by_its_username(env):
    response = _proxy_login(env, **{"remote-user": "unsloth"})
    assert response.status_code == 200
    assert response.json()["account_id"] == "owner"


def test_nothing_changes_while_the_setting_is_off(env, monkeypatch):
    _set_up_account(env, "alice")
    monkeypatch.delenv(proxy_header.USER_HEADER_ENV)
    assert _session(env, **{"remote-user": "alice"}).json() == {"username": None}
    assert _proxy_login(env, **{"remote-user": "alice"}).status_code == 401


@pytest.mark.parametrize(
    "name", ["Authorization", "cookie", "Host", "x-forwarded-for", "x-api-key", "bad header", "x" * 80]
)
def test_headers_with_another_meaning_are_never_an_identity(env, monkeypatch, name):
    monkeypatch.setenv(proxy_header.USER_HEADER_ENV, name)
    assert proxy_header.trusted_user_header() is None


def test_a_header_from_outside_the_trusted_proxy_is_ignored(env, monkeypatch, tmp_path):
    _set_up_account(env, "alice")
    # Wrong Host: not one of UNSLOTH_STUDIO_PROXY_HOSTS (a DNS-rebinding page, or another site).
    assert _proxy_login(env, host = "evil.example", **{"remote-user": "alice"}).status_code == 401
    # No allowlist at all.
    monkeypatch.delenv("UNSLOTH_STUDIO_PROXY_HOSTS")
    assert _proxy_login(env, **{"remote-user": "alice"}).status_code == 401
    monkeypatch.setenv("UNSLOTH_STUDIO_PROXY_HOSTS", "studio.example.com")
    # A LAN client reaching Studio's network socket directly.
    with TestClient(
        env.app, base_url = "http://192.168.1.20:8888", client = ("192.168.1.50", 50000)
    ) as lan:
        assert _proxy_login(lan, **{"remote-user": "alice"}).status_code == 401
        assert _session(lan, **{"remote-user": "alice"}).json() == {"username": None}


def test_the_proxy_secret_is_required_when_set(env, monkeypatch):
    _set_up_account(env, "alice")
    monkeypatch.setenv(proxy_header.PROXY_SECRET_ENV, "s3cret-from-the-proxy")
    assert _proxy_login(env, **{"remote-user": "alice"}).status_code == 401
    assert (
        _proxy_login(
            env, **{"remote-user": "alice", "x-unsloth-proxy-secret": "wrong"}
        ).status_code
        == 401
    )
    response = _proxy_login(
        env, **{"remote-user": "alice", "x-unsloth-proxy-secret": "s3cret-from-the-proxy"}
    )
    assert response.status_code == 200


def test_no_account_is_created_or_guessed(env):
    _set_up_account(env, "alice")
    for value in ("mallory", "alice@example.com", "", " "):
        assert _proxy_login(env, **{"remote-user": value}).status_code == 401
    assert storage.get_user_record("mallory") is None


def test_a_repeated_header_is_refused(env):
    _set_up_account(env, "alice")
    _set_up_account(env, "bob")
    response = env.post(
        "/api/auth/proxy-login",
        headers = [*_PROXY.items(), ("remote-user", "alice"), ("remote-user", "bob")],
    )
    assert response.status_code == 401


def test_deactivated_and_unfinished_accounts_are_refused(env):
    account_id = _set_up_account(env, "alice")
    env.patch(
        f"/api/accounts/{account_id}", headers = _owner_headers(), json = {"is_active": False}
    )
    assert _proxy_login(env, **{"remote-user": "alice"}).status_code == 401
    # A pending setup code is finished at the password form, not skipped by the proxy.
    env.post("/api/accounts", headers = _owner_headers(), json = {"username": "carol"})
    assert _session(env, **{"remote-user": "carol"}).json() == {"username": None}
    assert _proxy_login(env, **{"remote-user": "carol"}).status_code == 401

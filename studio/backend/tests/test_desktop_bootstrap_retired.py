# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A desktop-owned backend never hands the seeded password to a page, and a verified desktop sign-in retires it."""

import importlib.util
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage

_HTML = b"<html><head></head><body>Studio</body></html>"
_DESKTOP = {"kind": "tauri", "token_sha256": "0" * 64}


@pytest.fixture(autouse = True)
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    policy.invalidate_account_cache()
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "seeded-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = True,
    )
    yield
    policy.invalidate_account_cache()


def _app_state():
    return SimpleNamespace(state = SimpleNamespace(bootstrap_password = "seeded-password-123"))


def test_desktop_owned_backend_never_injects_the_seeded_password(monkeypatch):
    import main

    monkeypatch.setattr(main, "_DESKTOP_OWNER", dict(_DESKTOP))
    html, nonce = main._inject_bootstrap(_HTML, _app_state())
    assert nonce is None
    assert b"__UNSLOTH_BOOTSTRAP__" not in html
    assert b"seeded-password-123" not in html


def test_web_backend_still_injects_the_seeded_password(monkeypatch):
    import main

    monkeypatch.setattr(main, "_DESKTOP_OWNER", None)
    html, nonce = main._inject_bootstrap(_HTML, _app_state())
    assert nonce
    assert b"window.__UNSLOTH_BOOTSTRAP__=" in html and b"seeded-password-123" in html


def test_desktop_banner_keeps_the_password_out_of_the_log(monkeypatch):
    import main

    monkeypatch.setattr(main, "_DESKTOP_OWNER", dict(_DESKTOP))
    state = SimpleNamespace(api_only = True)
    assert main.banner_autofill_available(state, {}) is True


def _auth_app(*, desktop_owned = True):
    route_path = Path(__file__).resolve().parents[1] / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_desktop_bootstrap_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth_route)
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    app.state.bootstrap_password = "seeded-password-123"
    app.state.desktop_owned = desktop_owned
    return app


def test_verified_desktop_login_retires_the_bootstrap_password():
    raw = storage.create_desktop_secret()
    app = _auth_app()
    with TestClient(app) as client:
        response = client.post("/api/auth/desktop-login", json = {"secret": raw})
    assert response.status_code == 200, response.text
    assert app.state.bootstrap_password is None
    # The web first-run flow is untouched: the account still has to change its password.
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is True


def test_desktop_login_to_an_ownerless_backend_keeps_the_web_autofill():
    raw = storage.create_desktop_secret()
    app = _auth_app(desktop_owned = False)
    with TestClient(app) as client:
        response = client.post("/api/auth/desktop-login", json = {"secret": raw})
    assert response.status_code == 200, response.text
    assert app.state.bootstrap_password == "seeded-password-123"


def test_failed_desktop_login_probe_keeps_the_bootstrap_password():
    storage.create_desktop_secret()
    app = _auth_app()
    with TestClient(app) as client:
        malformed = client.post("/api/auth/desktop-login", json = {"secret": "desktop-invalid"})
        wrong_secret = storage.DESKTOP_SECRET_PREFIX + secrets.token_urlsafe(48)
        assert storage.desktop_secret_is_well_formed(wrong_secret)
        wrong = client.post("/api/auth/desktop-login", json = {"secret": wrong_secret})
    assert malformed.status_code == 401
    assert wrong.status_code == 401
    assert app.state.bootstrap_password == "seeded-password-123"

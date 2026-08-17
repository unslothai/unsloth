# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keyless API access: what it admits, and what it still leaves alone."""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auth import storage
from auth.authentication import (
    KEYLESS_FALLBACK_SCHEME,
    KEYLESS_SCHEME,
    admitted_without_session,
    authenticated_via_api_key,
    create_access_token,
    get_current_credential,
    get_current_subject,
    security,
)
from utils.keyless_api_access import (
    _reset_scope_cache,
    access_exposure,
    asgi_request_is_keyless,
    get_keyless_api_access_scope,
    get_keyless_api_tools_enabled,
    keyless_request_allowed,
    scope_covers,
    set_keyless_api_access,
)


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


def seed_user():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )


def asgi_scope(
    *,
    headers = None,
    app_state = None,
    path = "/v1/chat/completions",
):
    """A real ASGI scope, so header casing behaves the way uvicorn delivers it."""
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
        ],
        "app": SimpleNamespace(
            state = SimpleNamespace(bind_host = "127.0.0.1") if app_state is None else app_state
        ),
    }


def request_for(**kwargs):
    return Request(asgi_scope(**kwargs))


def resolve(request):
    return asyncio.run(security(request))


def subject_of(request):
    return asyncio.run(get_current_subject(resolve(request)))


# --- the setting itself ------------------------------------------------------


def test_off_by_default():
    assert keyless_request_allowed(request_for()) is False


def test_full_scope_admits_everything():
    set_keyless_api_access("full")
    for path in ("/v1/chat/completions", "/api/train/start", "/api/settings/profile"):
        assert keyless_request_allowed(request_for(path = path)) is True, path


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1/chat/count_tokens",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/models",
        "/v1/models/unsloth/some-model",
        "/v1/responses",
    ],
)
def test_inference_scope_serves_the_openai_surface(path):
    set_keyless_api_access("inference")
    assert keyless_request_allowed(request_for(path = path)) is True


@pytest.mark.parametrize(
    "path",
    [
        # /v1 also aliases model loading, media and sandbox routes; none are in scope
        "/v1/load",
        "/v1/unload",
        "/v1/status",
        "/v1/load-progress",
        "/v1/audio/generate",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/images/generations",
        "/v1/generate/stream",
        "/v1/sandbox/abc",
        "/v1/sandbox/abc/reveal",
        "/v1/external/openai/containers/delete",
        "/v1/llama-flags",
        "/v1/validate",
        # near-misses that a prefix match would have handed away
        "/v1x/chat",
        "/v1abc",
        "/api/v1/train",
        "/api/train/start",
        "/api/settings/keyless-api-access",
        "",
        "/",
    ],
)
def test_inference_scope_leaves_everything_else_alone(path):
    assert scope_covers("inference", path) is False


def test_turning_it_off_takes_effect_at_once():
    """The cached scope must never outlive the write that closed it."""
    set_keyless_api_access("full")
    assert keyless_request_allowed(request_for()) is True
    set_keyless_api_access("off")
    assert keyless_request_allowed(request_for()) is False


def test_unreadable_settings_db_reads_as_off(monkeypatch):
    import storage.studio_db as studio_db

    set_keyless_api_access("full")
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    _reset_scope_cache()
    assert keyless_request_allowed(request_for()) is False


@pytest.mark.parametrize("value", ["maybe", object(), None, 2, True, "ON"])
def test_an_unknown_scope_is_refused(value):
    with pytest.raises(ValueError):
        set_keyless_api_access(value)


def test_an_unknown_stored_scope_reads_as_off():
    from storage.studio_db import upsert_app_settings
    from utils.keyless_api_access import KEYLESS_API_ACCESS_SETTING_KEY

    upsert_app_settings({KEYLESS_API_ACCESS_SETTING_KEY: "everything"})
    _reset_scope_cache()
    assert get_keyless_api_access_scope() == "off"


# --- the tool grant ----------------------------------------------------------


def test_tools_are_off_even_when_the_api_is_open():
    """Chat runs python and terminal on this host, so it is its own decision."""
    set_keyless_api_access("full")
    assert get_keyless_api_tools_enabled() is False


def test_tools_can_be_granted_alongside_a_scope():
    set_keyless_api_access("inference", tools = True)
    assert get_keyless_api_tools_enabled() is True


def test_tools_survive_a_scope_change_that_keeps_access_on():
    set_keyless_api_access("inference", tools = True)
    set_keyless_api_access("full")
    assert get_keyless_api_tools_enabled() is True


def test_turning_access_off_drops_the_tool_grant():
    """Otherwise turning keyless back on later would silently restore tools."""
    set_keyless_api_access("full", tools = True)
    set_keyless_api_access("off")
    assert get_keyless_api_tools_enabled() is False
    set_keyless_api_access("full")
    assert get_keyless_api_tools_enabled() is False


@pytest.mark.parametrize("value", ["maybe", object(), 2])
def test_an_unknown_tool_value_is_refused(value):
    with pytest.raises(ValueError):
        set_keyless_api_access("full", tools = value)


def test_a_damaged_settings_db_grants_no_tools(monkeypatch):
    import storage.studio_db as studio_db

    set_keyless_api_access("full", tools = True)
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    _reset_scope_cache()
    assert get_keyless_api_tools_enabled() is False


# --- the middleware view of a request ----------------------------------------


def test_the_middleware_sees_a_keyless_request():
    seed_user()
    set_keyless_api_access("inference")
    assert asgi_request_is_keyless(asgi_scope()) is True
    assert asgi_request_is_keyless(asgi_scope(headers = {"Authorization": "Bearer ollama"})) is True
    assert asgi_request_is_keyless(asgi_scope(path = "/api/train/start")) is False


def test_the_middleware_leaves_a_signed_in_session_alone():
    seed_user()
    set_keyless_api_access("full")
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    scope = asgi_scope(headers = {"Authorization": f"Bearer {token}"})
    assert asgi_request_is_keyless(scope) is False


def test_the_middleware_is_inert_when_access_is_off():
    seed_user()
    assert asgi_request_is_keyless(asgi_scope()) is False


# --- exposure is advisory, never a gate -------------------------------------


def test_loopback_bind_reports_no_exposure():
    assert access_exposure(SimpleNamespace(bind_host = "127.0.0.1")) is None


@pytest.mark.parametrize(
    "state, expected",
    [
        (SimpleNamespace(), "network"),
        (SimpleNamespace(bind_host = "0.0.0.0"), "network"),
        (SimpleNamespace(bind_host = "::"), "network"),
        (SimpleNamespace(bind_host = "192.168.1.10"), "network"),
        (SimpleNamespace(bind_host = "127.0.0.1", secure = True), "public_url"),
        (
            SimpleNamespace(bind_host = "127.0.0.1", cloudflare_url = "https://x.trycloudflare.com"),
            "public_url",
        ),
        (SimpleNamespace(bind_host = "127.0.0.1", remote_access_is_colab = True), "colab"),
    ],
)
def test_reachable_servers_are_reported(state, expected):
    assert access_exposure(state) == expected


def test_a_live_tunnel_is_reported_without_a_url_on_state(monkeypatch):
    from utils import host_policy
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    assert access_exposure(SimpleNamespace(bind_host = "127.0.0.1")) == "public_url"


def test_exposure_does_not_stop_a_request():
    """The admin's choice stands however the server is reached."""
    set_keyless_api_access("full")
    exposed = request_for(
        headers = {"x-forwarded-for": "203.0.113.7"},
        app_state = SimpleNamespace(bind_host = "0.0.0.0"),
    )
    assert keyless_request_allowed(exposed) is True


# --- what the auth dependency does with it ----------------------------------


def test_missing_header_authenticates_as_the_admin():
    seed_user()
    set_keyless_api_access("full")
    request = request_for()
    assert resolve(request).scheme == KEYLESS_SCHEME
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


def test_missing_header_still_fails_when_the_setting_is_off():
    seed_user()
    with pytest.raises(HTTPException) as excinfo:
        resolve(request_for())
    assert excinfo.value.status_code in (401, 403)


def test_a_bad_key_still_fails_when_the_setting_is_off():
    seed_user()
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for(headers = {"Authorization": "Bearer not-needed"}))
    assert excinfo.value.status_code == 401


@pytest.mark.parametrize("token", ["not-needed", "lm-studio", "ollama", "sk-unsloth-YOUR_KEY"])
def test_unusable_keys_are_ignored_rather_than_rejected(token):
    seed_user()
    set_keyless_api_access("full")
    request = request_for(headers = {"Authorization": f"Bearer {token}"})
    assert resolve(request).scheme == KEYLESS_FALLBACK_SCHEME
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


def test_a_jwt_shaped_dummy_key_is_ignored_too():
    """A token that merely parses as a JWT names no session here."""
    seed_user()
    set_keyless_api_access("full")
    stranger = jwt.encode({"sub": "someone-else"}, secrets.token_urlsafe(64), algorithm = "HS256")
    assert subject_of(request_for(headers = {"Authorization": f"Bearer {stranger}"})) == (
        storage.DEFAULT_ADMIN_USERNAME
    )


def test_a_malformed_header_is_ignored_too():
    seed_user()
    set_keyless_api_access("full")
    assert subject_of(request_for(headers = {"Authorization": "garbage"})) == (
        storage.DEFAULT_ADMIN_USERNAME
    )


def test_a_working_key_still_authenticates_as_itself():
    seed_user()
    set_keyless_api_access("full")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "test", expires_at = None
    )
    request = request_for(headers = {"Authorization": f"Bearer {raw}"})
    subject, generation = asyncio.run(get_current_credential(resolve(request)))
    assert subject == storage.DEFAULT_ADMIN_USERNAME
    assert generation is not None
    # stamped by the key path, so per-key attribution survives
    assert storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)[0]["last_used_at"] is not None


def test_an_expired_key_is_ignored_rather_than_rejected():
    """A stale key is a no-op once the server asks for none at all."""
    seed_user()
    set_keyless_api_access("full")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "expired",
        expires_at = (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat(),
    )
    assert subject_of(request_for(headers = {"Authorization": f"Bearer {raw}"})) == (
        storage.DEFAULT_ADMIN_USERNAME
    )


@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer ollama"}])
def test_keyless_callers_count_as_api_callers(headers):
    """Guards that refuse an API key must refuse a keyless caller too."""
    seed_user()
    set_keyless_api_access("full")
    credentials = resolve(request_for(headers = headers))
    assert asyncio.run(authenticated_via_api_key(credentials)) is True


def test_keyless_callers_are_not_mistaken_for_the_ui_by_the_route_checks():
    """Saved provider credentials stay withheld, as they are from an API key."""
    seed_user()
    from routes.inference import _request_has_api_key, _request_used_api_key

    set_keyless_api_access("full")
    assert _request_has_api_key(request_for()) is True
    assert _request_used_api_key(request_for()) is True
    set_keyless_api_access("off")
    assert _request_has_api_key(request_for()) is False


def test_keyless_callers_do_not_pass_as_the_ui():
    """Saved provider credentials are held back from a keyless caller, as from a key."""
    seed_user()
    set_keyless_api_access("full")
    assert admitted_without_session(request_for()) is True
    assert admitted_without_session(request_for(headers = {"Authorization": "Bearer x"})) is True


def test_a_signed_in_session_is_still_the_ui():
    seed_user()
    set_keyless_api_access("full")
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    request = request_for(headers = {"Authorization": f"Bearer {token}"})
    assert admitted_without_session(request) is False
    assert resolve(request).scheme == "Bearer"
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


def test_an_expired_session_token_still_fails():
    """Sign-in stays authoritative, so the app re-authenticates instead of
    silently running as the admin."""
    seed_user()
    set_keyless_api_access("full")
    expired = create_access_token(
        storage.DEFAULT_ADMIN_USERNAME, expires_delta = timedelta(seconds = -60)
    )
    request = request_for(headers = {"Authorization": f"Bearer {expired}"})
    assert resolve(request).scheme == "Bearer"
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request)
    assert excinfo.value.status_code == 401


def test_a_pending_password_change_is_still_enforced(monkeypatch):
    import auth.authentication as authentication

    storage.create_initial_user(
        username = "pending",
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = True,
    )
    monkeypatch.setattr(authentication, "DEFAULT_ADMIN_USERNAME", "pending")
    set_keyless_api_access("full")
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for())
    assert excinfo.value.status_code == 403


def test_the_openapi_security_scheme_name_is_unchanged():
    assert security.scheme_name == "HTTPBearer"

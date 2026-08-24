# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keyless API access: what it admits, and what it still leaves alone."""

from __future__ import annotations

import asyncio
import secrets
import threading
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
    admitted_without_credential,
    admitted_without_session,
    authenticated_via_api_key,
    create_access_token,
    get_current_credential,
    get_current_subject,
    security,
)
from utils.keyless_api_access import (
    KeylessToolPolicyMiddleware,
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
    method = None,
    root_path = "",
    server = ("127.0.0.1", 8000),
    client = ("127.0.0.1", 50000),
):
    """A real ASGI scope, so header casing behaves the way uvicorn delivers it."""
    return {
        "type": "http",
        "method": method or ("GET" if path.startswith("/v1/models") else "POST"),
        "path": path,
        "root_path": root_path,
        "query_string": b"",
        "scheme": "http",
        "server": server,
        "client": client,
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
        ],
        "app": SimpleNamespace(
            state = SimpleNamespace(
                bind_host = "127.0.0.1",
                secure = False,
                remote_access_is_colab = False,
                lan_access_is_colab = False,
                cloudflare_url = None,
            )
            if app_state is None
            else app_state
        ),
    }


def request_for(**kwargs):
    return Request(asgi_scope(**kwargs))


def app_state(**overrides):
    state = SimpleNamespace(
        bind_host = "127.0.0.1",
        secure = False,
        remote_access_is_colab = False,
        lan_access_is_colab = False,
        lan_access_secure_launch = False,
        cloudflare_url = None,
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


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
    "method,path",
    [
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/count_tokens"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/unsloth/model"),
        ("POST", "/v1/responses"),
    ],
)
def test_inference_scope_has_an_exact_method_and_path_matrix(method, path):
    assert scope_covers("inference", method, path) is True


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/chat/completions"),
        ("PUT", "/v1/chat/completions"),
        ("GET", "/v1/chat/count_tokens"),
        ("GET", "/v1/completions"),
        ("GET", "/v1/embeddings"),
        ("GET", "/v1/messages"),
        ("GET", "/v1/messages/count_tokens"),
        ("POST", "/v1/models"),
        ("POST", "/v1/models/unsloth/model"),
        ("GET", "/v1/responses"),
    ],
)
def test_inference_scope_rejects_wrong_methods(method, path):
    assert scope_covers("inference", method, path) is False


@pytest.mark.parametrize(
    "path,root_path",
    [
        ("/v1/models", "/studio"),
        ("/studio/v1/models", "/studio"),
        ("/studio/v1/models/", "/studio/"),
    ],
)
def test_inference_scope_normalizes_direct_and_root_path_prefixed_requests(path, root_path):
    assert scope_covers("inference", "GET", path, root_path) is True


def test_root_path_requires_a_path_segment_boundary():
    assert scope_covers("inference", "GET", "/studio-v2/v1/models", "/studio") is False


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
        "/v1/external/openai/containers/list",
        "/v1/external/openai/containers/create",
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
    assert scope_covers("inference", "POST", path) is False


def test_the_allowlist_matches_the_actual_v1_router_topology():
    from routes.inference import router

    registered = {
        (method, route.path)
        for route in router.routes
        for method in getattr(route, "methods", set())
    }
    intended = {
        ("POST", "/chat/completions"),
        ("POST", "/chat/count_tokens"),
        ("POST", "/completions"),
        ("POST", "/embeddings"),
        ("POST", "/messages"),
        ("POST", "/messages/count_tokens"),
        ("GET", "/models"),
        ("GET", "/models/{model_id:path}"),
        ("POST", "/responses"),
    }
    assert intended <= registered

    denied_side_effects = {
        ("POST", "/load"),
        ("POST", "/unload"),
        ("POST", "/validate"),
        ("POST", "/generate/stream"),
        ("POST", "/audio/generate"),
        ("POST", "/audio/speech"),
        ("POST", "/audio/transcriptions"),
        ("POST", "/images/generations"),
        ("GET", "/sandbox/{session_id}"),
        ("POST", "/sandbox/{session_id}/reveal"),
        ("POST", "/external/openai/containers/list"),
        ("POST", "/external/openai/containers/create"),
        ("POST", "/external/openai/containers/delete"),
    }
    assert denied_side_effects <= registered
    for method, path in denied_side_effects:
        concrete = path.replace("{session_id}", "abc")
        assert scope_covers("inference", method, f"/v1{concrete}") is False


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


def test_a_refresh_that_raced_the_closing_write_cannot_republish_the_open_scope():
    """A request already reading the DB when keyless access is switched off must not
    publish its stale answer over the value that write just cached."""
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    _reset_scope_cache()

    read_done = threading.Event()
    write_done = threading.Event()
    real_read = keyless._read_settings
    refreshed = []

    def read_then_wait_for_the_write():
        answer = real_read()  # the open scope, still the only thing in the db
        read_done.set()
        assert write_done.wait(timeout = 10)
        return answer

    keyless._read_settings = read_then_wait_for_the_write
    reader = threading.Thread(target = lambda: refreshed.append(keyless._settings()))
    reader.start()
    try:
        assert read_done.wait(timeout = 10)
        set_keyless_api_access("off", tools = False)
    finally:
        write_done.set()
        reader.join(timeout = 10)
        keyless._read_settings = real_read

    assert refreshed == [("off", False)]
    assert keyless._cached_settings[1:] == ("off", False)
    assert keyless_request_allowed(request_for()) is False


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


def _middleware_tool_policy(scope):
    from state.tool_policy import get_tool_policy, reset_tool_policy, set_tool_policy

    observed = []

    async def downstream(_scope, _receive, _send):
        observed.append(get_tool_policy())

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        return None

    set_tool_policy(True)
    try:
        asyncio.run(KeylessToolPolicyMiddleware(downstream)(scope, receive, send))
    finally:
        reset_tool_policy()
    return observed


def test_keyless_tools_are_forced_off_without_the_separate_grant():
    seed_user()
    set_keyless_api_access("inference", tools = False)
    assert _middleware_tool_policy(asgi_scope()) == [False]


def test_keyless_tools_remain_available_with_the_separate_grant():
    seed_user()
    set_keyless_api_access("inference", tools = True)
    assert _middleware_tool_policy(asgi_scope()) == [True]


def test_a_valid_api_key_keeps_its_existing_tool_policy():
    seed_user()
    set_keyless_api_access("inference", tools = False)
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "tool-client",
        expires_at = None,
    )
    scope = asgi_scope(headers = {"Authorization": f"Bearer {raw}"})
    assert _middleware_tool_policy(scope) == [True]


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


def test_the_middleware_leaves_a_working_api_key_alone():
    """An existing API client authenticates as itself, so it keeps the tools it had."""
    seed_user()
    set_keyless_api_access("inference")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    assert asgi_request_is_keyless(asgi_scope(headers = {"Authorization": f"Bearer {raw}"})) is False


@pytest.mark.parametrize("expired", [False, True])
def test_the_middleware_keeps_an_unusable_studio_key_authoritative(expired):
    seed_user()
    set_keyless_api_access("inference")
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "stale",
        expires_at = (
            (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat() if expired else None
        ),
    )
    if not expired:
        storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
    assert asgi_request_is_keyless(asgi_scope(headers = {"Authorization": f"Bearer {raw}"})) is False


def test_the_middleware_check_does_not_count_as_a_use():
    """It runs ahead of the real validation, so stamping there would double-count every
    request and take sqlite's global write lock for an advisory answer."""
    seed_user()
    set_keyless_api_access("inference")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    assert asgi_request_is_keyless(asgi_scope(headers = {"Authorization": f"Bearer {raw}"})) is False
    assert storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)[0]["last_used_at"] is None


def test_the_middleware_is_inert_when_access_is_off():
    seed_user()
    assert asgi_request_is_keyless(asgi_scope()) is False


# --- public and hosted transports fail closed -------------------------------


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


def test_a_lan_listener_is_network_reach_not_a_public_url(monkeypatch):
    """Both connectors answer remote_connector_active, so reading that alone told the
    operator anyone with a public URL was let in when only the local network was."""
    from utils import host_policy

    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    assert access_exposure(SimpleNamespace(bind_host = "127.0.0.1")) == "network"


def test_a_tunnel_still_wins_over_a_lan_listener(monkeypatch):
    from utils import host_policy

    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    assert access_exposure(SimpleNamespace(bind_host = "127.0.0.1")) == "public_url"


def test_a_public_or_wildcard_bind_never_receives_full_keyless_access():
    set_keyless_api_access("full")
    exposed = request_for(
        headers = {"x-forwarded-for": "203.0.113.7"},
        app_state = SimpleNamespace(bind_host = "0.0.0.0"),
    )
    assert keyless_request_allowed(exposed) is False


@pytest.mark.parametrize(
    "server,client,bind_host",
    [
        (("127.0.0.1", 8000), ("127.0.0.1", 50000), "127.0.0.1"),
        (("::1", 8000), ("::1", 50000), "::1"),
        (("::ffff:127.0.0.1", 8000), ("::ffff:127.0.0.1", 50000), "::1"),
        (("localhost", 8000), ("localhost", 50000), "localhost"),
    ],
)
def test_full_scope_accepts_only_authoritative_loopback_transports(server, client, bind_host):
    set_keyless_api_access("full")
    request = request_for(
        server = server,
        client = client,
        app_state = app_state(bind_host = bind_host),
    )
    assert keyless_request_allowed(request) is True


@pytest.mark.parametrize(
    "state_overrides",
    [
        {"remote_access_is_colab": True},
        {"lan_access_is_colab": True},
        {"secure": True},
        {"lan_access_secure_launch": True},
        {"cloudflare_url": "https://example.trycloudflare.com"},
    ],
)
def test_public_tunnel_and_colab_transports_fail_closed(state_overrides):
    set_keyless_api_access("inference")
    assert keyless_request_allowed(request_for(app_state = app_state(**state_overrides))) is False


def test_an_active_public_tunnel_fails_closed(monkeypatch):
    from utils import host_policy

    set_keyless_api_access("inference")
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    assert keyless_request_allowed(request_for()) is False


def test_an_exact_private_lan_socket_remains_distinct_from_an_active_tunnel(monkeypatch):
    import lan_access
    from utils import host_policy

    set_keyless_api_access("inference")
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"]},
    )
    request = request_for(
        server = ("192.168.1.24", 8888),
        client = ("192.168.1.90", 54321),
        app_state = app_state(),
    )
    assert keyless_request_allowed(request) is True


@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer not-needed"}])
def test_browser_origin_requests_never_receive_keyless_access(headers):
    set_keyless_api_access("inference")
    headers = dict(headers or {})
    headers["Origin"] = "https://attacker.example"
    request = request_for(headers = headers)
    assert keyless_request_allowed(request) is False
    assert asgi_request_is_keyless(request.scope) is False


def test_inference_scope_accepts_an_exact_live_private_lan_socket(monkeypatch):
    import lan_access

    set_keyless_api_access("inference")
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"]},
    )
    request = request_for(
        server = ("192.168.1.24", 8888),
        client = ("192.168.1.90", 54321),
        app_state = app_state(bind_host = "127.0.0.1"),
    )
    assert keyless_request_allowed(request) is True


def test_full_scope_never_expands_to_the_private_lan(monkeypatch):
    import lan_access

    set_keyless_api_access("full")
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"]},
    )
    request = request_for(
        server = ("192.168.1.24", 8888),
        client = ("192.168.1.90", 54321),
        app_state = app_state(bind_host = "127.0.0.1"),
    )
    assert keyless_request_allowed(request) is False


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


@pytest.mark.parametrize("token", ["not-needed", "lm-studio", "ollama"])
def test_approved_dummy_bearers_are_accepted(token):
    seed_user()
    set_keyless_api_access("full")
    request = request_for(headers = {"Authorization": f"Bearer {token}"})
    assert resolve(request).scheme == KEYLESS_FALLBACK_SCHEME
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


@pytest.mark.parametrize(
    "stranger",
    [
        "arbitrary-client-token",
        jwt.encode({"sub": "someone-else"}, secrets.token_urlsafe(64), algorithm = "HS256"),
    ],
)
def test_an_arbitrary_bearer_is_rejected(stranger):
    seed_user()
    set_keyless_api_access("full")
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for(headers = {"Authorization": f"Bearer {stranger}"}))
    assert excinfo.value.status_code == 401


@pytest.mark.parametrize("header", ["garbage", "Basic abc", "Bearer", "Bearer "])
def test_a_malformed_header_is_rejected(header):
    seed_user()
    set_keyless_api_access("full")
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for(headers = {"Authorization": header}))
    assert excinfo.value.status_code in (401, 403)


def test_duplicate_authorization_headers_are_rejected():
    seed_user()
    set_keyless_api_access("full")
    scope = asgi_scope()
    scope["headers"] = [
        (b"authorization", b"Bearer not-needed"),
        (b"authorization", b"Bearer not-needed"),
    ]
    request = Request(scope)
    assert asgi_request_is_keyless(scope) is False
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request)
    assert excinfo.value.status_code == 403


def test_approved_dummy_bearer_is_only_accepted_on_an_eligible_request():
    seed_user()
    set_keyless_api_access("inference")
    request = request_for(
        path = "/v1/load",
        headers = {"Authorization": "Bearer not-needed"},
    )
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request)
    assert excinfo.value.status_code == 401


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


def test_an_expired_key_is_rejected_without_downgrading():
    seed_user()
    set_keyless_api_access("full")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "expired",
        expires_at = (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat(),
    )
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for(headers = {"Authorization": f"Bearer {raw}"}))
    assert excinfo.value.status_code == 401


def test_a_revoked_key_is_rejected_without_downgrading():
    seed_user()
    set_keyless_api_access("full")
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "revoked",
        expires_at = None,
    )
    storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
    with pytest.raises(HTTPException) as excinfo:
        subject_of(request_for(headers = {"Authorization": f"Bearer {raw}"}))
    assert excinfo.value.status_code == 401


def test_root_path_prefixed_request_authenticates_keylessly():
    seed_user()
    set_keyless_api_access("inference")
    request = request_for(path = "/studio/v1/models", root_path = "/studio", method = "GET")
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


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


def test_lan_keyless_request_state_drives_external_workflow_and_monitoring(monkeypatch):
    import lan_access
    from core.inference.api_monitor import ApiMonitor
    from routes import inference

    seed_user()
    set_keyless_api_access("inference")
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"]},
    )
    request = request_for(
        path = "/v1/models",
        method = "GET",
        server = ("192.168.1.24", 8888),
        client = ("192.168.1.90", 54321),
        app_state = app_state(),
    )
    subject = subject_of(request)

    assert request.state.keyless_api_admitted is True
    assert inference._request_has_api_key(request) is True
    assert inference._request_used_api_key(request) is True
    assert inference._request_is_internal_workflow(request) is False
    assert inference._request_is_saved_credential_workflow(request) is False

    monitor = ApiMonitor(enabled = True)
    monitor.start(
        endpoint = "/v1/models",
        method = "GET",
        model = "",
        prompt = "",
        subject = subject,
        via_api_key = inference._request_used_api_key(request),
    )
    rows = monitor.snapshot(subject = storage.DEFAULT_ADMIN_USERNAME)
    assert len(rows) == 1
    assert rows[0]["via_api_key"] is True


def test_keyless_callers_do_not_pass_as_the_ui():
    """Saved provider credentials are held back from a keyless caller, as from a key."""
    seed_user()
    set_keyless_api_access("full")
    assert admitted_without_session(request_for()) is True
    assert admitted_without_session(request_for(headers = {"Authorization": "Bearer x"})) is False


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


def test_a_working_key_was_not_admitted_by_the_setting():
    """The narrower predicate the credential-minting guard reads."""
    seed_user()
    set_keyless_api_access("full")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    key_request = request_for(headers = {"Authorization": f"Bearer {raw}"})
    assert admitted_without_credential(resolve(key_request)) is False
    assert admitted_without_credential(resolve(request_for())) is True
    assert (
        admitted_without_credential(
            resolve(request_for(headers = {"Authorization": "Bearer ollama"}))
        )
        is True
    )


# --- credentials the setting must not be able to mint -------------------------


def api_key_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes import auth as auth_routes

    app = FastAPI()
    app.state.bind_host = "127.0.0.1"
    app.state.secure = False
    app.state.remote_access_is_colab = False
    app.state.lan_access_is_colab = False
    app.state.cloudflare_url = None
    app.include_router(auth_routes.router, prefix = "/api/auth")
    return TestClient(
        app,
        base_url = "http://127.0.0.1",
        client = ("127.0.0.1", 50000),
    )


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer ollama"}])
def test_a_keyless_caller_cannot_manage_api_keys(headers):
    """Both writes outlive the setting: switching keyless off neither withdraws a key it
    handed out nor restores one it destroyed. Listing goes with them because it is the
    step that names the key to revoke."""
    seed_user()
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "existing", expires_at = None
    )
    set_keyless_api_access("full")
    client = api_key_client()
    assert (
        client.post("/api/auth/api-keys", json = {"name": "minted"}, headers = headers).status_code
        == 403
    )
    assert client.get("/api/auth/api-keys", headers = headers).status_code == 403
    assert client.delete(f"/api/auth/api-keys/{row['id']}", headers = headers).status_code == 403
    assert [key["name"] for key in storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)] == [
        "existing"
    ]
    assert storage.validate_api_key(raw) is not None


def test_a_working_key_can_still_manage_them_while_keyless_is_on():
    seed_user()
    set_keyless_api_access("full")
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    client = api_key_client()
    headers = {"Authorization": f"Bearer {raw}"}
    created = client.post("/api/auth/api-keys", json = {"name": "second"}, headers = headers)
    assert created.status_code == 200
    assert created.json()["key"].startswith(storage.API_KEY_PREFIX)
    assert client.get("/api/auth/api-keys", headers = headers).status_code == 200
    assert client.delete(f"/api/auth/api-keys/{row['id']}", headers = headers).status_code == 200


def test_a_keyless_caller_cannot_sign_the_ui_out():
    """Logout revokes every refresh token for the subject, and those do not come back
    when keyless access is switched off."""
    seed_user()
    set_keyless_api_access("full")
    assert api_key_client().post("/api/auth/logout").status_code == 403


def test_a_signed_in_session_can_still_sign_out_while_keyless_is_on():
    seed_user()
    set_keyless_api_access("full")
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    response = api_key_client().post(
        "/api/auth/logout", headers = {"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 204


def test_a_signed_in_session_can_still_mint_one_while_keyless_is_on():
    seed_user()
    set_keyless_api_access("full")
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    response = api_key_client().post(
        "/api/auth/api-keys",
        json = {"name": "from-the-ui"},
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


# --- effects a stranger must not be able to start -----------------------------


def switch_attempt(
    monkeypatch,
    request,
    *,
    on_disk = True,
):
    """Drive _maybe_auto_switch_model with auto-switch and auto-download both on.

    Returns (loads, downloads): what the request would have made this server do.
    """
    import core.inference.local_model_resolver as resolver
    import core.inference.openai_auto_download as auto_download
    import routes.inference as inference
    import utils.openai_auto_switch_settings as switch_settings

    loads, downloads = [], []
    resolved = ("unsloth/Other-GGUF", None, "unsloth/Other-GGUF") if on_disk else None

    async def record_load(*args, **kwargs):
        loads.append(args[0] if args else kwargs)

    async def record_download(model, **kwargs):
        downloads.append(model)
        return None

    monkeypatch.setattr(switch_settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(switch_settings, "get_openai_auto_download_enabled", lambda: True)
    monkeypatch.setattr(switch_settings, "idle_unload_is_configured", lambda: False)
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda _m, **_k: resolved)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m, **_k: resolved)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(auto_download, "is_downloadable_ref", lambda _r: True)
    monkeypatch.setattr(auto_download, "maybe_auto_download", record_download)
    monkeypatch.setattr(inference, "_loaded_identity_satisfies", lambda _m: False)
    monkeypatch.setattr(inference, "_load_model_impl", record_load)
    monkeypatch.setattr(inference, "_auto_switch_waiters", {})
    try:
        asyncio.run(inference._maybe_auto_switch_model("unsloth/Other-GGUF", request, "tester"))
    except HTTPException as exc:
        # a named model this caller may not switch to must say so, never be answered by another
        assert exc.status_code in (404, 503), exc.status_code
    return loads, downloads


@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer ollama"}])
def test_a_keyless_caller_cannot_switch_the_loaded_model(monkeypatch, headers):
    """The dialog offers the loaded model, so a stranger must not be able to evict it
    for another one already on disk."""
    seed_user()
    set_keyless_api_access("inference")
    loads, downloads = switch_attempt(monkeypatch, request_for(headers = headers))
    assert (loads, downloads) == ([], [])


def test_a_keyless_caller_cannot_start_a_download(monkeypatch):
    """Same for a name that is not on disk: no gigabytes fetched on a stranger's word."""
    seed_user()
    set_keyless_api_access("inference")
    loads, downloads = switch_attempt(monkeypatch, request_for(), on_disk = False)
    assert (loads, downloads) == ([], [])


def test_a_working_key_can_still_switch_and_download(monkeypatch):
    """Both are the owner's own opt-ins, and a key is a credential they issued."""
    seed_user()
    set_keyless_api_access("inference")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    headers = {"Authorization": f"Bearer {raw}"}
    loads, _ = switch_attempt(monkeypatch, request_for(headers = headers))
    assert loads != []
    _, downloads = switch_attempt(monkeypatch, request_for(headers = headers), on_disk = False)
    assert downloads == ["unsloth/Other-GGUF"]


def test_a_switch_still_happens_when_keyless_is_off(monkeypatch):
    seed_user()
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    loads, _ = switch_attempt(
        monkeypatch, request_for(headers = {"Authorization": f"Bearer {token}"})
    )
    assert loads != []


def provider_attempt(monkeypatch, request):
    """POST a chat completion naming an external provider. Returns what got proxied."""
    import routes.inference as inference
    from models.inference import ChatCompletionRequest

    proxied = []

    async def record(payload, _request, _subject):
        proxied.append(payload.provider_base_url)
        return {}

    monkeypatch.setattr(inference, "_proxy_to_external_provider", record)
    payload = ChatCompletionRequest(
        model = "llama3",
        messages = [{"role": "user", "content": "hi"}],
        provider_type = "ollama",
        provider_base_url = "http://192.168.1.50:11434",
    )
    asyncio.run(inference.openai_chat_completions(payload, request, "tester"))
    return proxied


@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer ollama"}])
def test_a_keyless_caller_cannot_route_through_an_external_provider(monkeypatch, headers):
    """provider_base_url is egress from this host, and validate_provider_base_url allows
    loopback and LAN on purpose, so a stranger could reach the network behind it."""
    seed_user()
    set_keyless_api_access("inference")
    with pytest.raises(HTTPException) as excinfo:
        provider_attempt(monkeypatch, request_for(headers = headers))
    assert excinfo.value.status_code == 403


def test_a_working_key_can_still_route_through_one(monkeypatch):
    seed_user()
    set_keyless_api_access("inference")
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "client", expires_at = None
    )
    proxied = provider_attempt(monkeypatch, request_for(headers = {"Authorization": f"Bearer {raw}"}))
    assert proxied == ["http://192.168.1.50:11434"]


def test_a_session_can_still_route_through_one_with_keyless_off(monkeypatch):
    seed_user()
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    proxied = provider_attempt(
        monkeypatch, request_for(headers = {"Authorization": f"Bearer {token}"})
    )
    assert proxied == ["http://192.168.1.50:11434"]


# --- routes that read the bearer for themselves ------------------------------


def health_fields(headers = None):
    """/api/health resolves its own bearer, so it needs asking about keyless separately."""
    import main
    return asyncio.run(main.health_check(request_for(path = "/api/health", headers = headers)))


def test_health_answers_a_keyless_caller_in_full():
    seed_user()
    set_keyless_api_access("full")
    for headers in (None, {"Authorization": "Bearer not-needed"}):
        assert "version" in health_fields(headers), headers


def test_health_holds_the_authed_fields_back_when_access_is_off():
    """They fingerprint how the host is exposed, so nothing but a credential earns them."""
    seed_user()
    assert "version" not in health_fields()
    assert "version" not in health_fields({"Authorization": "Bearer unsloth-local"})


def test_health_is_outside_the_inference_scope():
    seed_user()
    set_keyless_api_access("inference")
    assert "version" not in health_fields()


SANDBOX_PATH = "/api/inference/sandbox/abc"


def authenticate_manually(request, token = None):
    from routes.inference import _authenticate_header_or_query
    return asyncio.run(_authenticate_header_or_query(request, token))


def test_the_sandbox_file_routes_follow_the_full_scope():
    """They resolve the bearer themselves, so `security` never sees them and they have
    to ask about keyless access separately."""
    seed_user()
    set_keyless_api_access("full")
    admin = storage.DEFAULT_ADMIN_USERNAME
    assert authenticate_manually(request_for(path = SANDBOX_PATH)) == admin
    assert (
        authenticate_manually(
            request_for(path = SANDBOX_PATH, headers = {"Authorization": "Bearer not-needed"})
        )
        == admin
    )
    # the ?token= form an <img src> has to use, since it cannot send a header
    assert authenticate_manually(request_for(path = SANDBOX_PATH), "not-needed") == admin


def test_the_sandbox_file_routes_are_outside_the_inference_scope():
    """Serving files is not the OpenAI surface, so opening that surface must not open them."""
    seed_user()
    set_keyless_api_access("inference")
    with pytest.raises(HTTPException) as excinfo:
        authenticate_manually(request_for(path = SANDBOX_PATH))
    assert excinfo.value.status_code == 401


def test_the_sandbox_file_routes_still_refuse_a_missing_token_when_access_is_off():
    seed_user()
    with pytest.raises(HTTPException) as excinfo:
        authenticate_manually(request_for(path = SANDBOX_PATH))
    assert excinfo.value.status_code == 401


def test_a_query_token_naming_an_expired_session_still_fails():
    """Sign-in stays authoritative on the ?token= path too, not only in the header."""
    seed_user()
    set_keyless_api_access("full")
    expired = create_access_token(
        storage.DEFAULT_ADMIN_USERNAME, expires_delta = timedelta(seconds = -60)
    )
    with pytest.raises(HTTPException) as excinfo:
        authenticate_manually(request_for(path = SANDBOX_PATH), expired)
    assert excinfo.value.status_code == 401


def test_the_openapi_security_scheme_name_is_unchanged():
    assert security.scheme_name == "HTTPBearer"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auth import storage
from auth.authentication import (
    KEYLESS_FALLBACK_SCHEME,
    KEYLESS_SCHEME,
    admitted_without_credential,
    admitted_without_session,
    create_access_token,
    get_current_subject,
    security,
)
from utils.keyless_api_access import (
    KeylessToolPolicyMiddleware,
    KEYLESS_ADMISSION_STATE_KEY,
    KEYLESS_API_ACCESS_SETTING_KEY,
    _reset_scope_cache,
    access_exposure,
    asgi_request_is_keyless,
    get_keyless_api_access_scope,
    get_keyless_api_access_settings,
    get_keyless_api_tools_enabled,
    keyless_request_allowed,
    scope_covers,
    set_keyless_api_access,
)


# Keep this security matrix compact enough for the PR test-line budget.
# fmt: off
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
    storage.create_initial_user(username = storage.DEFAULT_ADMIN_USERNAME,
                                password = "human-password-123", jwt_secret = secrets.token_urlsafe(64))


def app_state(**overrides):
    state = SimpleNamespace(bind_host = "127.0.0.1", secure = False,
                            remote_access_is_colab = False, lan_access_is_colab = False,
                            lan_access_secure_launch = False, cloudflare_url = None)
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def asgi_scope(*, path = "/v1/chat/completions", method = None, root_path = "",
               headers = None, state = None, server = ("127.0.0.1", 8000),
               client = ("127.0.0.1", 50000),):
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
            (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
        ],
        "app": SimpleNamespace(state = state or app_state()),
    }


def request_for(**kwargs):
    return Request(asgi_scope(**kwargs))


def bearer_request(token, **kwargs):
    return request_for(headers = {"Authorization": f"Bearer {token}"}, **kwargs)


def resolve(request):
    return asyncio.run(security(request))


def subject_of(request):
    return asyncio.run(get_current_subject(resolve(request)))


def test_exact_route_matrix_matches_registered_topology():
    from routes.inference import router

    allowed_posts = "/v1/chat/completions /v1/chat/count_tokens /v1/completions /v1/embeddings /v1/messages /v1/messages/count_tokens /v1/responses".split()
    denied_posts = "/v1/load /v1/unload /v1/validate /v1/generate/stream /v1/audio/speech /v1/images/generations /v1/external/openai/containers/create /v1x/chat".split()
    allowed = {("POST", path) for path in allowed_posts} | {
        ("GET", "/v1/models"), ("GET", "/v1/models/unsloth/model")}
    denied = {("POST", path) for path in denied_posts} | {
        ("POST", "/v1/models"), ("GET", "/v1/chat/completions"), ("GET", "/v1/sandbox/abc")}
    assert all(scope_covers("inference", method, path) for method, path in allowed)
    assert not any(scope_covers("inference", method, path) for method, path in denied)
    assert scope_covers("inference", "GET", "/studio/v1/models/", "/studio/")
    assert not scope_covers("inference", "GET", "/studio-v2/v1/models", "/studio")
    assert not scope_covers("off", "POST", "/v1/chat/completions")
    assert scope_covers("full", "POST", "/api/train/start")

    registered = {(method, route.path) for route in router.routes
                  for method in getattr(route, "methods", set())}
    intended = {("POST", path.removeprefix("/v1")) for path in allowed_posts} | {
        ("GET", "/models"), ("GET", "/models/{model_id:path}")}
    assert intended <= registered


def test_settings_are_immediate_and_fail_closed(monkeypatch):
    import storage.studio_db as studio_db

    set_keyless_api_access("full", tools = True)
    assert keyless_request_allowed(request_for()) is True
    assert get_keyless_api_tools_enabled() is True
    set_keyless_api_access("off")
    assert keyless_request_allowed(request_for()) is False
    assert get_keyless_api_tools_enabled() is False

    set_keyless_api_access("full", tools = True)
    monkeypatch.setattr(studio_db, "get_app_setting",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("db unavailable")))
    _reset_scope_cache()
    assert (get_keyless_api_access_scope(), get_keyless_api_tools_enabled()) == ("off", False)


def test_stale_refresh_cannot_reopen_a_closed_scope(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    _reset_scope_cache()
    read_done, write_done = threading.Event(), threading.Event()
    real_read = keyless._read_settings
    observed = []

    def delayed_read():
        value = real_read()
        read_done.set()
        assert write_done.wait(timeout = 10)
        return value

    monkeypatch.setattr(keyless, "_read_settings", delayed_read)
    reader = threading.Thread(target = lambda: observed.append(keyless._settings()))
    reader.start()
    try:
        assert read_done.wait(timeout = 10)
        set_keyless_api_access("off", tools = False)
    finally:
        write_done.set()
        reader.join(timeout = 10)
    assert observed == [("off", False)]


def test_overlapping_writes_publish_in_commit_order(monkeypatch):
    import storage.studio_db as studio_db

    first_committed = threading.Event()
    release_first = threading.Event()
    second_committed = threading.Event()
    real_upsert = studio_db.upsert_app_settings

    def delayed_upsert(settings):
        result = real_upsert(settings)
        if settings[KEYLESS_API_ACCESS_SETTING_KEY] == "full":
            first_committed.set()
            assert release_first.wait(timeout = 10)
        else:
            second_committed.set()
        return result

    monkeypatch.setattr(studio_db, "upsert_app_settings", delayed_upsert)
    first = threading.Thread(target = set_keyless_api_access,
                             args = ("full",), kwargs = {"tools": True})
    second = threading.Thread(target = set_keyless_api_access,
                              args = ("off",), kwargs = {"tools": False})
    first.start()
    assert first_committed.wait(timeout = 10)
    second.start()
    try:
        assert not second_committed.wait(timeout = 0.1)
    finally:
        release_first.set()
        first.join(timeout = 10)
        second.join(timeout = 10)

    assert not first.is_alive() and not second.is_alive()
    assert get_keyless_api_access_settings() == ("off", False)


def test_full_scope_is_loopback_only():
    set_keyless_api_access("full")
    for server, client, bind_host in (
        (("127.0.0.1", 8000), ("127.0.0.1", 50000), "127.0.0.1"),
        (("::1", 8000), ("::1", 50000), "::1"),
        (("::ffff:127.0.0.1", 8000), ("::ffff:127.0.0.1", 50000), "::1"),
        (("localhost", 8000), ("localhost", 50000), "localhost"),
    ):
        assert keyless_request_allowed(request_for(
            server = server, client = client, state = app_state(bind_host = bind_host)))


def test_public_browser_and_private_lan_boundaries(monkeypatch):
    import lan_access
    from utils import host_policy

    set_keyless_api_access("inference")
    for overrides in ({"remote_access_is_colab": True}, {"secure": True},
                      {"cloudflare_url": "https://x.trycloudflare.com"}):
        assert not keyless_request_allowed(request_for(state = app_state(**overrides)))
    assert not keyless_request_allowed(request_for(headers = {"Origin": "https://evil.example"}))
    assert access_exposure(app_state(
        bind_host = "studio.lan", lan_access_launch_managed = True,
        lan_access_launch_addresses = ("192.168.1.24",),
    )) == "private_lan"

    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"running": True, "port": 8888,
                                 "addresses": ["192.168.1.24"]})
    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    assert access_exposure(app_state()) == "private_lan"
    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"addresses": ["100.64.0.10"]})
    assert access_exposure(app_state()) == "network"
    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"running": True, "port": 8888,
                                 "addresses": ["192.168.1.24"]})
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    lan_request = request_for(method = "GET", path = "/v1/models",
                              server = ("192.168.1.24", 8888), client = ("192.168.1.90", 54321))
    assert keyless_request_allowed(lan_request)
    set_keyless_api_access("full")
    assert not keyless_request_allowed(lan_request)
    assert not keyless_request_allowed(request_for(state = app_state(bind_host = "0.0.0.0")))


def test_credentials_never_downgrade_to_keyless():
    seed_user()
    set_keyless_api_access("full")
    assert resolve(request_for()).scheme == KEYLESS_SCHEME
    for token in ("not-needed", "lm-studio", "ollama"):
        assert resolve(bearer_request(token)).scheme == KEYLESS_FALLBACK_SCHEME
    set_keyless_api_access("inference")
    with pytest.raises(HTTPException):
        subject_of(bearer_request("not-needed", path = "/v1/load"))
    set_keyless_api_access("full")

    for header in ("Bearer arbitrary", "garbage", "Basic abc", "Bearer"):
        with pytest.raises(HTTPException):
            subject_of(request_for(headers = {"Authorization": header}))
    duplicate = asgi_scope()
    duplicate["headers"] = [
        (b"authorization", b"Bearer not-needed"),
        (b"authorization", b"Bearer not-needed"),
    ]
    with pytest.raises(HTTPException):
        subject_of(Request(duplicate))

    session = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    assert subject_of(bearer_request(session)) == storage.DEFAULT_ADMIN_USERNAME
    expired_session = create_access_token(
        storage.DEFAULT_ADMIN_USERNAME, expires_delta = timedelta(seconds = -60))
    with pytest.raises(HTTPException):
        subject_of(bearer_request(expired_session))

    valid, _ = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "valid", expires_at = None)
    assert subject_of(bearer_request(valid)) == storage.DEFAULT_ADMIN_USERNAME
    for name, expires in (
        ("expired", (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat()),
        ("revoked", None),
    ):
        raw, row = storage.create_api_key(
            username = storage.DEFAULT_ADMIN_USERNAME, name = name, expires_at = expires
        )
        if name == "revoked":
            storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
        with pytest.raises(HTTPException):
            subject_of(bearer_request(raw))


def _middleware_policy(scope):
    from state.tool_policy import get_tool_policy, reset_tool_policy, set_tool_policy

    observed = []

    async def downstream(*_args):
        observed.append(get_tool_policy())

    set_tool_policy(True)
    try:
        asyncio.run(KeylessToolPolicyMiddleware(downstream)(
            scope, lambda: None, lambda _message: None))
    finally:
        reset_tool_policy()
    return observed


def test_tool_policy_and_api_identity(monkeypatch):
    from core.inference.llama_keepwarm import _carries_bearer_credentials
    from routes import inference

    seed_user()
    set_keyless_api_access("inference", tools = False)
    request = request_for()
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME
    assert request.state.keyless_api_admitted is True
    assert admitted_without_credential(resolve(request)) is True
    assert admitted_without_session(request) is True
    assert inference._request_has_api_key(request) is True
    assert inference._request_used_api_key(request) is True
    assert inference._request_is_saved_credential_workflow(request) is False
    scope = asgi_scope(state = app_state())
    scope["state"] = {KEYLESS_ADMISSION_STATE_KEY: True}
    assert _carries_bearer_credentials(scope, "/v1/chat/completions") is True
    assert _middleware_policy(asgi_scope()) == [False]

    set_keyless_api_access("inference", tools = True)
    assert _middleware_policy(asgi_scope()) == [True]
    key, _ = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "tools", expires_at = None)
    set_keyless_api_access("inference", tools = False)
    assert _middleware_policy(asgi_scope(headers = {"Authorization": f"Bearer {key}"})) == [True]

    set_keyless_api_access("off")

    async def enabled_between_layers(scope, *_args):
        set_keyless_api_access("inference")
        with pytest.raises(HTTPException):
            await security(Request(scope))

    asyncio.run(KeylessToolPolicyMiddleware(enabled_between_layers)(
        asgi_scope(), lambda: None, lambda _message: None))


def test_protected_side_effect_guards_remain_wired():
    import inspect
    from routes import auth, inference, mcp_servers, preview, rag, training_history, video

    auth_source = inspect.getsource(auth)
    assert auth_source.count("_require_a_credential_of_its_own(") >= 6
    assert "_require_a_credential_of_its_own" in inspect.getsource(auth.change_password)
    assert all(
        "request_admitted_without_credential" in inspect.getsource(handler)
        for handler in (inference._maybe_auto_switch_model, inference.openai_chat_completions)
    )
    assert all(
        "authenticated_without_credential" in inspect.getsource(handler)
        and "not no_credential" in inspect.getsource(handler)
        for handler in (
            preview.list_previews,
            training_history.list_training_runs,
            training_history.get_training_run_detail,
            training_history.update_training_run,
        )
    )
    assert all(
        "request_admitted_without_credential" in inspect.getsource(handler)
        for handler in (video.get_gallery_video_signed_url, rag.document_file_url)
    )
    assert "request_admitted_without_credential" in inspect.getsource(
        inference.openai_image_generations
    )
    assert all(
        "no_credential" in inspect.getsource(handler)
        for handler in (mcp_servers.list_mcp_servers, mcp_servers.update_mcp_server)
    )
    main_source = (Path(__file__).parents[1] / "main.py").read_text(encoding = "utf-8")
    assert 'app.state.secure = os.environ.get("UNSLOTH_SECURE") == "1"' in main_source
    assert security.scheme_name == "HTTPBearer"


def test_keyless_idle_restore_requires_the_requested_model(monkeypatch):
    from core.inference import llama_keepwarm as kw; import auth.authentication as authentication
    from studio.backend.tests import test_openai_auto_switch as auto
    backend = auto._FakeBackend(None); rec = auto._LoadRecorder(backend)
    auto._wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(auto.settings, "idle_unload_is_configured", lambda: True)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _r: True)
    auto._run_hook("org/B-GGUF"); assert rec.calls == []
    auto._run_hook("org/A-GGUF:Q4_K_M"); assert len(rec.calls) == 1
# fmt: on

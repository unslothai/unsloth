# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Official SDK transport and account-scoped OAuth device lifecycle, without a paid subscription."""

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace as Obj

import pytest
from fastapi import HTTPException

from core.inference import github_copilot as copilot
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import get_provider_info, list_available_providers
from routes import github_copilot_auth as auth


@pytest.fixture
def flow_store(monkeypatch, tmp_path):
    vault = {}
    monkeypatch.setenv("UNSLOTH_GITHUB_COPILOT_CLIENT_ID", "our-oauth-app")
    monkeypatch.setattr(
        auth.providers_db, "get_own_provider", lambda _: {"provider_type": "github_copilot"}
    )
    monkeypatch.setattr(auth, "current_credential_write", lambda _: nullcontext())
    monkeypatch.setattr(auth, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(
        auth.credential_secrets,
        "get_or_create_credential_encryption_key",
        lambda: None,
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "get_connection",
        lambda: sqlite3.connect(":memory:"),
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "upsert_secret",
        lambda kind, scope, value, **_: vault.__setitem__((kind, scope), value),
    )
    monkeypatch.setattr(
        auth.credential_secrets, "get_secret", lambda kind, scope: vault.get((kind, scope))
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "delete_secret",
        lambda kind, scope, **_: vault.pop((kind, scope), None),
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "provider_api_key_saved",
        lambda scope: ("token", scope) in vault,
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "delete_provider_api_key",
        lambda scope, **_: vault.pop(("token", scope), None),
    )
    now = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: now[0])
    return vault, now


async def begin(monkeypatch, result = None):
    async def post(path, data):
        assert path == "login/device/code"
        assert data == {"client_id": "our-oauth-app"}
        return result or {
            "device_code": "private-device-code",
            "user_code": "ABCD-EFGH",
            "expires_in": 900,
            "interval": 5,
        }

    monkeypatch.setattr(auth, "_github_post", post)
    return await auth.start("connection", ("user", None), False)


def test_device_token_stays_server_side_and_polling_is_throttled(monkeypatch, flow_store):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        assert "private-device-code" not in json.dumps(flow)
        assert flow["verification_uri"] == "https://github.com/login/device"
        # The initial poll must not hit GitHub.
        assert (await auth.poll("connection", flow["flow_id"], ("user", None), False))[
            "status"
        ] == "pending"
        now[0] += 5

        async def post(path, data):
            assert path == "login/oauth/access_token"
            assert data["device_code"] == "private-device-code"
            return {"access_token": "gho_private-token"}

        monkeypatch.setattr(auth, "_github_post", post)
        result = await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert result == {"status": "connected"}
        bundle = json.loads(vault[(auth.OAUTH_KIND, "connection")])
        assert bundle["access_token"] == "gho_private-token"
        assert bundle["client_id"] == "our-oauth-app"
        assert json.loads(vault[(auth.FLOW_KIND, "connection")]) == {
            "id": flow["flow_id"],
            "provider_id": "connection",
        }
        with pytest.raises(HTTPException) as exc:
            await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert exc.value.status_code == 404

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,status", [("access_denied", "denied"), ("expired_token", "expired")]
)
def test_denied_and_expired_never_save_a_token(monkeypatch, flow_store, error, status):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        now[0] += 5

        async def post(*_):
            return {"error": error}

        monkeypatch.setattr(auth, "_github_post", post)
        assert await auth.poll("connection", flow["flow_id"], ("user", None), False) == {
            "status": status
        }
        assert not vault

    asyncio.run(run())


def test_slow_down_and_local_expiry(monkeypatch, flow_store):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        now[0] += 5

        async def post(*_):
            return {"error": "slow_down"}

        monkeypatch.setattr(auth, "_github_post", post)
        result = await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert result == {"status": "pending", "interval": 10}
        assert json.loads(vault[(auth.FLOW_KIND, "connection")])["next_poll"] == 1015
        now[0] += 900
        assert await auth.poll("connection", flow["flow_id"], ("user", None), False) == {
            "status": "expired"
        }
        assert not vault

    asyncio.run(run())


def test_replacement_cancel_and_cross_connection_flows(monkeypatch, flow_store):
    async def run():
        first = await begin(monkeypatch)
        second = await begin(monkeypatch)
        for provider, flow_id in [
            ("connection", first["flow_id"]),
            ("different", second["flow_id"]),
        ]:
            with pytest.raises(HTTPException) as exc:
                await auth.poll(provider, flow_id, ("user", None), False)
            assert exc.value.status_code == 404
        await auth.disconnect("connection", ("user", None), False)
        with pytest.raises(HTTPException):
            await auth.poll("connection", second["flow_id"], ("user", None), False)

    asyncio.run(run())


def test_cancelling_a_reconnect_preserves_the_saved_authorization(monkeypatch, flow_store):
    vault, _ = flow_store
    bundle = {"access_token": "saved-token", "client_id": "our-oauth-app"}
    vault[(auth.OAUTH_KIND, "connection")] = json.dumps(bundle)

    async def run():
        first = await begin(monkeypatch)
        current = await begin(monkeypatch)
        await auth.cancel("connection", first["flow_id"], ("user", None), False)
        assert (auth.FLOW_KIND, "connection") in vault
        await auth.cancel("connection", current["flow_id"], ("user", None), False)
        assert (auth.FLOW_KIND, "connection") not in vault
        assert auth.auth_status("connection") == "connected"
        assert await auth.resolve_access("connection", None, allow_saved_key = True) == "saved-token"

    asyncio.run(run())


def test_api_keys_and_other_accounts_cannot_authorize(monkeypatch, flow_store):
    async def run():
        with pytest.raises(HTTPException) as exc:
            await auth.start("connection", ("user", None), True)
        assert exc.value.status_code == 403
        monkeypatch.setattr(auth.providers_db, "get_own_provider", lambda _: None)
        with pytest.raises(HTTPException) as exc:
            await auth.start("connection", ("user", None), False)
        assert exc.value.status_code == 404

    asyncio.run(run())


def test_missing_oauth_app_has_actionable_error(monkeypatch, flow_store):
    monkeypatch.delenv(auth.CLIENT_ID_ENV)
    monkeypatch.setattr(auth, "GITHUB_COPILOT_CLIENT_ID", None)
    with pytest.raises(HTTPException, match = "Studio administrator"):
        asyncio.run(auth.start("connection", ("user", None), False))


def test_local_client_id_overrides_default(monkeypatch, flow_store):
    monkeypatch.setattr(auth, "GITHUB_COPILOT_CLIENT_ID", "default-app")
    monkeypatch.setenv(auth.CLIENT_ID_ENV, " local-test-app ")
    assert auth._client_id() == "local-test-app"
    monkeypatch.setenv(auth.CLIENT_ID_ENV, " ")
    assert auth._client_id() == "default-app"


def test_unsupported_python_fails_before_authorization(monkeypatch, flow_store):
    monkeypatch.setattr(auth.sys, "version_info", (3, 10))
    with pytest.raises(HTTPException, match = "Python 3.11"):
        asyncio.run(auth.start("connection", ("user", None), False))


@pytest.mark.parametrize("new_connection", [False, True])
def test_fresh_studio_uses_default_client_id_without_locking_auth_db(
    monkeypatch, tmp_path, new_connection
):
    from auth import storage as auth_storage
    from routes import providers
    from models.providers import ProviderCreate

    monkeypatch.delenv(auth.CLIENT_ID_ENV, raising = False)
    monkeypatch.setattr(auth, "GITHUB_COPILOT_CLIENT_ID", "default-app")
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(auth, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(auth.providers_db, "studio_db_path", auth.studio_db_path)
    monkeypatch.setattr(auth.credential_secrets, "studio_db_path", auth.studio_db_path)
    monkeypatch.setattr(providers, "validate_provider_base_url", lambda base: base)
    auth_storage.create_initial_user("unsloth", "temporary-password", "test-jwt")
    credential = ("unsloth", auth_storage.credential_generation("test-jwt"))

    async def post(path, data):
        if path == "login/device/code":
            assert data["client_id"] == "default-app"
            return {"device_code": "private", "user_code": "ABCD-EFGH", "expires_in": 900}
        assert data["client_id"] == "default-app"
        return {"access_token": "live-test-token", "expires_in": 3600, "refresh_token": "refresh"}

    monkeypatch.setattr(auth, "_github_post", post)

    async def run():
        if new_connection:
            scope = "new"
        else:
            row = await providers.create_provider_config(
                ProviderCreate(provider_type = "github_copilot", display_name = "Copilot"),
                credential = credential,
                via_api_key = False,
            )
            scope = row.id
        assert auth_storage._credential_encryption_key_cache is None
        flow = await auth.start(scope, credential, False)
        assert flow["user_code"] == "ABCD-EFGH"
        assert providers.list_provider_configs("unsloth", False) == []
        if new_connection:
            assert auth.providers_db.list_providers() == []
            await auth.cancel(scope, flow["flow_id"], credential, False)
            assert auth.credential_secrets.get_secret(auth.FLOW_KIND, scope) is None
            assert auth.providers_db.list_providers() == []
            with pytest.raises(HTTPException) as exc:
                await auth.poll(scope, flow["flow_id"], credential, False)
            assert exc.value.status_code == 404
            flow = await auth.start(scope, credential, False)
        stored = json.loads(auth.credential_secrets.get_secret(auth.FLOW_KIND, scope))
        monkeypatch.setattr(auth, "time", Obj(time = lambda: stored["next_poll"]))
        waiting_for_github, release_approval = asyncio.Event(), asyncio.Event()

        async def hold_approval(path, data):
            waiting_for_github.set()
            await release_approval.wait()
            return await post(path, data)

        monkeypatch.setattr(auth, "_github_post", hold_approval)
        polling = asyncio.create_task(auth.poll(scope, flow["flow_id"], credential, False))
        await asyncio.wait_for(waiting_for_github.wait(), timeout = 5)
        try:
            assert (
                await asyncio.wait_for(
                    auth.cancel(scope, flow["flow_id"], credential, False), timeout = 5
                )
                is None
            )
        finally:
            release_approval.set()
        with pytest.raises(HTTPException) as exc:
            await polling
        assert exc.value.status_code == 404
        assert providers.list_provider_configs("unsloth", False) == []

        monkeypatch.setattr(auth, "_github_post", post)
        flow = await auth.start(scope, credential, False)
        stored = json.loads(auth.credential_secrets.get_secret(auth.FLOW_KIND, scope))
        result = await auth.poll(scope, flow["flow_id"], credential, False)
        assert result["status"] == "connected"
        provider_id = result["provider_id"] if new_connection else scope
        assert json.loads(auth.credential_secrets.get_secret(auth.FLOW_KIND, scope)) == {
            "id": flow["flow_id"],
            "provider_id": provider_id,
        }
        assert [provider.id for provider in providers.list_provider_configs("unsloth", False)] == [
            provider_id
        ]
        late_cancel = await auth.cancel(scope, flow["flow_id"], credential, False)
        assert late_cancel.status_code == 200
        assert json.loads(late_cancel.body) == {"provider_id": provider_id}
        assert (
            await auth.resolve_access(provider_id, None, allow_saved_key = True) == "live-test-token"
        )
        await providers.delete_provider_config(provider_id, credential, False)
        assert auth.providers_db.get_own_provider(provider_id) is None
        assert auth.credential_secrets.get_secret(auth.OAUTH_KIND, provider_id) is None

    asyncio.run(run())


def test_concurrent_requests_refresh_with_the_original_app_and_rotate_the_tokens(
    monkeypatch, flow_store
):
    vault, now = flow_store
    key = (auth.OAUTH_KIND, "connection")
    vault[key] = json.dumps(
        {
            "client_id": "original-app",
            "access_token": "old-access",
            "expires_at": now[0] - 1,
            "refresh_token": "old-refresh",
            "refresh_expires_at": now[0] + 50,
        }
    )
    calls = []

    async def post(path, data):
        calls.append(data)
        assert path == "login/oauth/access_token"
        await asyncio.sleep(0.05)
        return {
            "access_token": "new-access",
            "expires_in": 3600,
            "refresh_token": "new-refresh",
            "refresh_token_expires_in": 7200,
        }

    async def run():
        tokens = await asyncio.gather(
            *[auth.resolve_access("connection", None, allow_saved_key = True) for _ in range(2)]
        )
        assert tokens == ["new-access", "new-access"]

    monkeypatch.setattr(auth, "_github_post", post)
    asyncio.run(run())
    assert calls == [
        {
            "client_id": "original-app",
            "grant_type": "refresh_token",
            "refresh_token": "old-refresh",
        }
    ]
    assert json.loads(vault[key])["refresh_token"] == "new-refresh"


def test_expired_refresh_marks_connection_for_reauthorization(monkeypatch, flow_store):
    vault, now = flow_store
    key = (auth.OAUTH_KIND, "connection")
    vault[key] = json.dumps(
        {
            "client_id": "our-oauth-app",
            "access_token": "old-access",
            "expires_at": now[0] - 1,
            "refresh_token": "old-refresh",
        }
    )

    async def post(*_):
        return {"error": "bad_refresh_token"}

    monkeypatch.setattr(auth, "_github_post", post)
    with pytest.raises(HTTPException, match = "Reconnect") as error:
        asyncio.run(auth.resolve_access("connection", None, allow_saved_key = True))
    assert error.value.status_code == 401
    assert auth.auth_status("connection") == "reauthorization_required"


class Session:
    def __init__(self):
        self.closed = False
        self.aborted = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def on(self, callback):
        self.callback = callback

    async def send_and_wait(self, prompt, timeout):
        self.prompt = prompt
        self.callback(Obj(type = "assistant.message_delta", data = Obj(delta_content = "Hello")))
        self.callback(Obj(type = "assistant.message", data = Obj(content = "Hello")))
        self.callback(Obj(type = "assistant.usage", data = Obj(input_tokens = 10, output_tokens = 2)))

    async def abort(self):
        self.aborted = True


@pytest.fixture
def fake_sdk(monkeypatch):
    session = Session()
    state = {}

    class Client:
        async def create_session(self, **kwargs):
            state.update(kwargs)
            return session

        async def list_models(self):
            return [
                Obj(id = name, name = name, policy = policy)
                for name, policy in (
                    ("allowed", Obj(state = Obj(value = "enabled"))),
                    ("no-policy", None),
                    ("blocked", Obj(state = Obj(value = "disabled"))),
                )
            ]

    @asynccontextmanager
    async def client(token):
        assert token == "gho_user-token"
        try:
            yield Client()
        finally:
            state["client_closed"] = True

    monkeypatch.setattr(copilot, "sdk_client", client)
    return session, state


def test_transport_streams_once_and_preserves_editable_history(fake_sdk):
    session, state = fake_sdk

    async def run():
        client = ExternalProviderClient("github_copilot", copilot.BASE_URL, "gho_user-token")
        lines = [
            line
            async for line in client.stream_chat_completion(
                [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Remember 7"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "What number?"},
                ],
                "allowed",
            )
        ]
        chunks = [json.loads(line[6:]) for line in lines if "[DONE]" not in line]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello"
        assert chunks[-1]["usage"]["total_tokens"] == 12
        assert lines[-1] == "data: [DONE]\n\n"
        assert state["system_message"]["content"] == "Be concise"
        assert state["tools"] == {"available_tools": []}
        assert state["permissions"]["on_request"]().kind == "reject"
        assert "Remember 7" in session.prompt and "What number?" in session.prompt
        assert session.closed and state["client_closed"]

    asyncio.run(run())


def test_models_use_sdk_and_filter_disabled_models(fake_sdk):
    assert [m["id"] for m in asyncio.run(copilot.list_models("gho_user-token"))] == [
        "allowed",
        "no-policy",
    ]


@pytest.mark.parametrize("auto_available", [True, False])
def test_empty_model_catalog_uses_auto_only_when_the_sdk_confirms_it(monkeypatch, auto_available):
    async def empty_models():
        return []

    async def session_catalog():
        return Obj(auto = Obj() if auto_available else None)

    async def create_session(**kwargs):
        assert kwargs["model"] == "auto"
        assert kwargs["tools"] == {"available_tools": []}
        session = Session()
        session.rpc = Obj(model = Obj(list = session_catalog))
        return session

    @asynccontextmanager
    async def sdk_client(_):
        yield Obj(list_models = empty_models, create_session = create_session)

    monkeypatch.setattr(copilot, "sdk_client", sdk_client)
    available = asyncio.run(copilot.list_models("gho_user-token"))
    assert [model["id"] for model in available] == (["auto"] if auto_available else [])


def test_cancel_aborts_sdk_generation(fake_sdk, monkeypatch):
    session, state = fake_sdk

    async def waiting(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "send_and_wait", waiting)

    async def run():
        stream = copilot.stream_chat(
            "gho_user-token", [{"role": "user", "content": "Hi"}], "allowed"
        )
        await anext(stream)
        await stream.aclose()
        assert session.aborted and session.closed and state["client_closed"]

    asyncio.run(run())


def test_sdk_errors_never_return_success(fake_sdk, monkeypatch):
    session, _ = fake_sdk

    async def error(*args, **kwargs):
        session.callback(Obj(type = "session.error", data = Obj(message = "private-token")))

    monkeypatch.setattr(session, "send_and_wait", error)

    async def run():
        with pytest.raises(RuntimeError, match = "subscription") as exc:
            async for _ in copilot.stream_chat(
                "gho_user-token", [{"role": "user", "content": "Hi"}], "allowed"
            ):
                pass
        assert "private-token" not in str(exc.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "secret"}}]}],
        [{"role": "tool", "content": "result"}],
    ],
)
def test_unsupported_content_is_rejected(messages):
    with pytest.raises(ValueError):
        copilot._prompts(messages)


def test_fixed_routing_and_registry_contract():
    with pytest.raises(ValueError, match = "routing is fixed"):
        ExternalProviderClient("github_copilot", "https://elsewhere.example", "gho_secret")
    info = get_provider_info("github_copilot")
    assert info["auth_kind"] == "github_oauth" and not info["supports_tool_calling"]
    assert "github_copilot" not in {p["provider_type"] for p in list_available_providers()}
    assert "github_copilot" in {
        p["provider_type"] for p in list_available_providers(include_oauth = True)
    }


def test_subscription_rejects_key_edits_sharing_and_output_overrides():
    from routes import providers

    info = get_provider_info("github_copilot")
    for changes in (
        {"encrypted_api_key": "replacement"},
        {"clear_api_key": True},
        {"base_url": "https://elsewhere.example"},
    ):
        kwargs = {"encrypted_api_key": None, "base_url": None, "models": [], "updating": True}
        kwargs.update(changes)
        with pytest.raises(HTTPException) as error:
            providers._validate_provider_auth_contract(info, **kwargs)
        assert error.value.status_code == 400
    with pytest.raises(HTTPException):
        providers._validate_share_request(True, "github_copilot", copilot.BASE_URL)
    with pytest.raises(HTTPException):
        providers._validate_max_output_tokens_contract("github_copilot", True, 1024)

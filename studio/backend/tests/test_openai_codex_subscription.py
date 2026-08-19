# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64

from contextlib import asynccontextmanager, contextmanager

import importlib.util

import inspect
from pathlib import Path
import sys

import threading
from types import ModuleType, SimpleNamespace
import asyncio
import hashlib
import json
import time

import pytest

from core.inference import openai_codex_auth as codex_auth


from core.inference.openai_codex_auth import (
    OPENAI_CODEX_API_BASE,
    OPENAI_CODEX_DEVICE_REDIRECT_URI,
    OPENAI_CODEX_CLIENT_ID,
    OAuthFlow,
    _validate_token_payload,
    create_pkce,
    extract_chatgpt_account_id,
    safe_flow,
)
from core.inference.openai_codex_client import (
    CodexReauthorizationError,
    CodexTransportError,
    OpenAICodexClient,
    _responses_input,
    cached_subscription_models,
    forget_subscription_models,
    list_subscription_models,
    offered_subscription_model_ids,
)
from core.inference import openai_codex_client as codex_client

from core.inference.openai_responses_shared import normalize_function_schema
from core.inference.providers import get_provider_info, list_available_providers


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_protocol_constants_and_curated_provider_contract():
    assert OPENAI_CODEX_CLIENT_ID == "app_EMoamEEZ73f0CkXaXp7hrann"
    info = get_provider_info("openai_codex")
    assert info["base_url"] == OPENAI_CODEX_API_BASE
    assert info["auth_kind"] == "chatgpt_oauth"
    assert info["model_list_mode"] == "curated"
    assert info["base_url_editable"] is False
    assert info["model_ids_editable"] is False
    assert info["default_models"] == [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    ]
    assert OPENAI_CODEX_DEVICE_REDIRECT_URI == ("https://auth.openai.com/deviceauth/callback")
    row = next(
        item for item in list_available_providers() if item["provider_type"] == "openai_codex"
    )
    assert row["auth_kind"] == "chatgpt_oauth"


def test_provider_lock_can_release_from_another_executor_thread():
    lock = codex_auth._provider_file_lock("provider")
    assert lock.is_thread_local() is False


def test_pkce_uses_s256_and_high_entropy_verifier():
    verifier, challenge = create_pkce()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    assert len(verifier) >= 43
    assert challenge == expected
    assert create_pkce()[0] != verifier


def test_account_claim_and_token_response_are_validated_without_returning_raw_body():
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
    assert extract_chatgpt_account_id(token) == "acct-1"
    bundle = _validate_token_payload(
        {"access_token": token, "refresh_token": "refresh", "expires_in": 600}
    )
    assert bundle["account_id"] == "acct-1"
    assert bundle["expires_at"] > time.time()
    with pytest.raises(Exception, match = "invalid access token"):
        extract_chatgpt_account_id("not-a-jwt")


def test_safe_flow_never_exposes_pkce_state_or_device_identifier():
    flow = OAuthFlow(
        id = "opaque",
        provider_id = "provider",
        method = "browser",
        created_at = 1,
        expires_at = 2,
        state = "secret-state",
        verifier = "secret-verifier",
        device_auth_id = "secret-device",
        authorization_url = "https://auth.openai.com/oauth/authorize",
    )
    serialized = json.dumps(safe_flow(flow))
    assert "opaque" in serialized
    assert "secret-state" not in serialized
    assert "secret-verifier" not in serialized
    assert "secret-device" not in serialized


def test_responses_conversion_replays_only_opaque_reasoning_and_normalizes_tools():
    instructions, items = _responses_input(
        [
            {"role": "system", "content": "User system prompt"},
            {
                "role": "assistant",
                "content": "visible",
                "extra_content": {
                    "openai_codex_reasoning": [
                        {
                            "type": "reasoning",
                            "id": "r1",
                            "encrypted_content": "opaque",
                            "summary": [],
                        },
                        {"type": "reasoning", "id": "bad"},
                    ]
                },
            },
            {"role": "user", "content": "next"},
        ]
    )
    assert "User system prompt" in instructions
    assert any(item.get("encrypted_content") == "opaque" for item in items)
    assert not any(item.get("id") == "bad" for item in items)
    user_item = next(item for item in items if item.get("role") == "user")
    assert user_item["content"] == [{"type": "input_text", "text": "next"}]
    assistant_item = next(item for item in items if item.get("type") == "message")
    assert assistant_item["content"] == [
        {"type": "output_text", "text": "visible", "annotations": []}
    ]
    schema = normalize_function_schema(
        {"type": "object", "properties": {"nested": {"type": "object"}}}
    )
    assert schema["properties"]["nested"]["properties"] == {}

    combinators = normalize_function_schema(
        {
            "type": "object",
            "$defs": {"row": {"type": "object"}},
            "properties": {
                "choice": {
                    "anyOf": [
                        {"type": "object"},
                        {"type": "array", "items": {"type": "object"}},
                    ]
                }
            },
        }
    )
    assert combinators["$defs"]["row"]["properties"] == {}
    assert combinators["properties"]["choice"]["anyOf"][0]["properties"] == {}
    assert combinators["properties"]["choice"]["anyOf"][1]["items"]["properties"] == {}


def test_responses_conversion_stably_shortens_oversized_tool_call_ids():
    oversized = "call_" + "x" * 70
    _, items = _responses_input(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": oversized,
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": oversized, "content": "done"},
        ]
    )

    call = next(item for item in items if item.get("type") == "function_call")
    output = next(item for item in items if item.get("type") == "function_call_output")
    assert len(call["call_id"]) == 64
    assert call["call_id"] == output["call_id"]
    assert call["call_id"].startswith(oversized[:31] + "_")


def test_cancelled_oauth_exchange_cannot_persist_credentials(monkeypatch):
    persisted = []
    flow = OAuthFlow(
        id = "flow-cancelled",
        provider_id = "provider",
        method = "browser",
        created_at = time.time(),
        expires_at = time.time() + 60,
        persist_bundle = lambda _provider, bundle: persisted.append(bundle),
    )
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})

    async def token_request(_data):
        flow.status = "cancelled"
        return {"access_token": token, "refresh_token": "refresh", "expires_in": 600}

    monkeypatch.setattr(codex_auth, "_token_request", token_request)
    with pytest.raises(codex_auth.CodexAuthError, match = "cancelled"):
        asyncio.run(codex_auth._exchange_code(flow, "code"))
    assert persisted == []


def test_successful_callback_does_not_wait_for_its_own_connection(monkeypatch):
    class Server:
        closed = False

        def close(self):
            self.closed = True

        async def wait_closed(self):
            raise AssertionError("callback must respond before active connections close")

    server = Server()
    persisted = []
    flow = OAuthFlow(
        id = "flow-success",
        provider_id = "provider",
        method = "browser",
        created_at = time.time(),
        expires_at = time.time() + 60,
        persist_bundle = lambda _provider, bundle: persisted.append(bundle),
        server = server,
    )
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})

    async def token_request(_data):
        return {"access_token": token, "refresh_token": "refresh", "expires_in": 600}

    monkeypatch.setattr(codex_auth, "_token_request", token_request)
    asyncio.run(codex_auth._exchange_code(flow, "code"))

    assert flow.status == "connected"
    assert flow.server is None
    assert server.closed is True
    assert len(persisted) == 1


def test_fixed_host_transport_sends_subscription_headers_and_normalizes_sse():
    captured = {}

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"response.output_text.delta","delta":"hello"}'
            yield 'data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":1}}}'
            yield "data: [DONE]"

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, method, url, **kwargs):
            captured.update(method = method, url = url, **kwargs)
            return FakeStream()

        async def aclose(self):
            return None

    async def run():
        client = OpenAICodexClient("secret-token", "acct-1")
        await client._client.aclose()
        client._client = FakeClient()
        lines = [
            line
            async for line in client.stream(
                provider_id = "provider-1",
                thread_id = "thread-1",
                messages = [{"role": "user", "content": "hello"}],
                model = "gpt-5.4",
                max_tokens = 100,
                reasoning_effort = "none",
                tools = None,
                tool_choice = None,
            )
        ]
        await client.close()
        return lines

    lines = asyncio.run(run())
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["originator"] == "unsloth_studio"
    assert captured["headers"]["chatgpt-account-id"] == "acct-1"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["session-id"]
    assert captured["json"]["store"] is False
    assert "max_output_tokens" not in captured["json"]

    assert captured["json"]["reasoning"] == {"effort": "none", "summary": "auto"}
    assert captured["json"]["include"] == ["reasoning.encrypted_content"]
    assert any("hello" in line for line in lines)
    assert not any("secret-token" in line for line in lines)


def test_browser_state_mismatch_does_not_consume_flow(monkeypatch):
    persisted = []
    flow = OAuthFlow(
        id = "flow",
        provider_id = "provider",
        method = "browser",
        created_at = time.time(),
        expires_at = time.time() + 60,
        state = "expected-state",
        verifier = "secret-verifier",
        redirect_uri = "http://127.0.0.1:1455/auth/callback",
        persist_bundle = lambda _provider, bundle: persisted.append(bundle),
    )
    monkeypatch.setitem(codex_auth._flows, flow.id, flow)
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})

    async def token_request(data):
        assert data["code_verifier"] == "secret-verifier"
        return {"access_token": token, "refresh_token": "refresh", "expires_in": 600}

    monkeypatch.setattr(codex_auth, "_token_request", token_request)
    with pytest.raises(codex_auth.CodexAuthError, match = "state"):
        asyncio.run(
            codex_auth.complete_browser_flow(
                "provider",
                "flow",
                "http://127.0.0.1:1455/auth/callback?code=bad&state=wrong",
            )
        )
    assert flow.status == "pending"
    assert flow.consumed is False

    asyncio.run(
        codex_auth.complete_browser_flow(
            "provider",
            "flow",
            "http://127.0.0.1:1455/auth/callback?code=good&state=expected-state",
        )
    )
    assert flow.status == "connected"
    assert len(persisted) == 1
    with pytest.raises(codex_auth.CodexAuthError, match = "no longer active"):
        asyncio.run(
            codex_auth.complete_browser_flow(
                "provider",
                "flow",
                "http://127.0.0.1:1455/auth/callback?code=replay&state=expected-state",
            )
        )


def test_device_poll_shape_structured_pending_slow_down_and_exchange(monkeypatch):
    poll_bodies = []
    sleeps = []
    responses = [
        (400, {"error": {"code": "deviceauth_authorization_pending"}}),
        (400, {"error": "slow_down"}),
        (200, {"authorization_code": "auth-code", "code_verifier": "device-verifier"}),
    ]

    class FakeResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, *, json):
            poll_bodies.append(json)
            status, body = responses.pop(0)
            return FakeResponse(status, body)

    async def fake_sleep(delay):
        sleeps.append(delay)

    exchange_data = {}
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})

    async def token_request(data):
        exchange_data.update(data)
        return {"access_token": token, "refresh_token": "refresh", "expires_in": 600}

    persisted = []
    flow = OAuthFlow(
        id = "device-flow",
        provider_id = "provider",
        method = "device",
        created_at = time.time(),
        expires_at = time.time() + 60,
        device_auth_id = "device-id",
        user_code = "USER-CODE",
        interval = 1,
        redirect_uri = OPENAI_CODEX_DEVICE_REDIRECT_URI,
        persist_bundle = lambda _provider, bundle: persisted.append(bundle),
    )
    monkeypatch.setattr(codex_auth.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(codex_auth.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(codex_auth, "_token_request", token_request)

    asyncio.run(codex_auth._device_poll(flow))
    assert poll_bodies == [
        {"device_auth_id": "device-id", "user_code": "USER-CODE"},
        {"device_auth_id": "device-id", "user_code": "USER-CODE"},
        {"device_auth_id": "device-id", "user_code": "USER-CODE"},
    ]
    assert sleeps == [1, 1, 6]
    assert exchange_data["code"] == "auth-code"
    assert exchange_data["code_verifier"] == "device-verifier"
    assert exchange_data["redirect_uri"] == OPENAI_CODEX_DEVICE_REDIRECT_URI
    assert flow.status == "connected"
    assert len(persisted) == 1


def test_device_poll_persists_terminal_error_for_other_workers(monkeypatch):
    record = {
        "marker": "marker",
        "flow_id": "device-error",
        "method": "device",
        "created_at": time.time(),
        "expires_at": time.time() + 60,
        "state": "secret-state",
        "verifier": "secret-verifier",
        "device_auth_id": "device-id",
        "user_code": "USER-CODE",
        "status": "pending",
        "message": "",
    }

    class FakeResponse:
        status_code = 400

        def json(self):
            return {"error": {"code": "access_denied"}}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    async def fake_sleep(_delay):
        return None

    @asynccontextmanager
    async def oauth_guard(_provider_id):
        yield

    def upsert(_kind, _provider_id, value):
        record.clear()
        record.update(json.loads(value))

    flow = OAuthFlow(
        id = "device-error",
        provider_id = "provider",
        method = "device",
        created_at = record["created_at"],
        expires_at = record["expires_at"],
        device_auth_id = "device-id",
        user_code = "USER-CODE",
        interval = 1,
        marker = "marker",
    )
    monkeypatch.setattr(codex_auth.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(codex_auth.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(codex_auth, "provider_oauth_write_guard", oauth_guard)
    monkeypatch.setattr(codex_auth, "_oauth_flow_record", lambda _provider: record.copy())
    monkeypatch.setattr(codex_auth.credential_secrets, "upsert_secret", upsert)

    asyncio.run(codex_auth._device_poll(flow))

    assert flow.status == "error"
    assert record["status"] == "error"
    assert record["message"] == flow.message
    assert all(not record[key] for key in ("state", "verifier", "device_auth_id", "user_code"))
    persisted = codex_auth._load_persisted_oauth_flow("provider", flow.id)
    assert persisted is not None
    assert persisted.status == "error"
    assert persisted.message == flow.message


def test_expired_refreshable_bundle_stays_connected_and_reconnect_marker_is_sanitized(monkeypatch):
    bundle = {
        "access_token": "secret-access",
        "refresh_token": "secret-refresh",
        "expires_at": time.time() - 10,
        "account_id": "secret-account",
    }
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _provider: bundle)
    assert codex_auth.auth_status("provider") == "connected"
    bundle["reauthorization_required"] = True
    assert codex_auth.auth_status("provider") == "reauthorization_required"
    assert "secret" not in codex_auth.auth_status("provider")


def test_oauth_start_captures_generation_guarded_persistence(monkeypatch):
    routes_package = ModuleType("routes")
    routes_package.__path__ = []
    provider_credentials = ModuleType("routes.provider_credentials")
    provider_credentials.current_credential_write = lambda _credential: contextmanager(
        lambda: (yield)
    )()
    provider_credentials.require_ui_session = lambda _via_api_key: None
    monkeypatch.setitem(sys.modules, "routes", routes_package)
    monkeypatch.setitem(sys.modules, "routes.provider_credentials", provider_credentials)
    route_path = Path(__file__).parents[1] / "routes" / "openai_codex_auth.py"
    spec = importlib.util.spec_from_file_location("codex_auth_route_under_test", route_path)
    assert spec is not None and spec.loader is not None
    auth_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth_route)

    seen = []
    markers = {}
    monkeypatch.setattr(
        auth_route, "_provider", lambda _provider: {"provider_type": "openai_codex"}
    )
    monkeypatch.setattr(
        codex_auth.credential_secrets,
        "get_or_create_credential_encryption_key",
        lambda: seen.append("warmed"),
    )

    @contextmanager
    def guard(credential):
        seen.append(("guard", credential))
        yield

    @asynccontextmanager
    async def oauth_guard(_provider_id):
        yield

    monkeypatch.setattr(auth_route, "current_credential_write", guard)
    monkeypatch.setattr(codex_auth, "provider_oauth_write_guard", oauth_guard)
    monkeypatch.setattr(
        codex_auth,
        "save_oauth_flow_marker",
        lambda scope, marker, _flow = None: markers.__setitem__(scope, marker),
    )

    monkeypatch.setattr(
        codex_auth,
        "set_oauth_flow_marker_status",
        lambda scope, marker, _status: markers.pop(scope, None)
        if markers.get(scope) == marker
        else None,
    )
    monkeypatch.setattr(
        codex_auth,
        "oauth_flow_marker_matches",
        lambda scope, marker: markers.get(scope) == marker,
    )
    monkeypatch.setattr(
        codex_auth,
        "delete_oauth_flow_marker",
        lambda scope, marker = None: markers.pop(scope, None)
        if marker is None or markers.get(scope) == marker
        else None,
    )
    monkeypatch.setattr(
        codex_auth,
        "save_oauth_bundle",
        lambda scope, bundle: seen.append((scope, bundle)),
    )

    async def fake_start(provider_id, method, persist, marker):
        assert seen[0] == "warmed"
        assert inspect.iscoroutinefunction(persist)

        assert markers[provider_id] == marker
        await persist(provider_id, {"access_token": "not-returned"})
        return OAuthFlow(
            id = "opaque",
            provider_id = provider_id,
            method = method,
            created_at = 1,
            expires_at = 2,
            marker = marker,
        )

    monkeypatch.setattr(codex_auth, "start_flow", fake_start)
    result = asyncio.run(
        auth_route.start_oauth(
            "provider",
            auth_route.OAuthStartRequest(method = "browser"),
            credential = ("alice", "generation-1"),
            via_api_key = False,
        )
    )
    assert result["flow_id"] == "opaque"
    assert ("guard", ("alice", "generation-1")) in seen
    assert any(item[0] == "provider" for item in seen if isinstance(item, tuple))
    assert markers == {}
    assert "not-returned" not in json.dumps(result)


@pytest.mark.parametrize(
    "stream_lines",
    [
        ['data: {"type":"response.output_text.delta","delta":"partial"}'],
        ["data: {not-json}"],
        ['data: {"type":"response.output_text.delta","delta":42}'],
    ],
)
def test_malformed_or_truncated_stream_fails_without_token_leak(stream_lines):
    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for line in stream_lines:
                yield line

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

        async def aclose(self):
            return None

    async def run():
        client = OpenAICodexClient("super-secret-token", "acct-1")
        await client._client.aclose()
        client._client = FakeClient()
        try:
            return [
                line
                async for line in client.stream(
                    provider_id = "provider",
                    thread_id = "thread",
                    messages = [{"role": "user", "content": "hello"}],
                    model = "gpt-5.4",
                    max_tokens = None,
                    reasoning_effort = None,
                    tools = None,
                    tool_choice = None,
                )
            ]
        finally:
            await client.close()

    with pytest.raises(RuntimeError) as error:
        asyncio.run(run())
    assert "super-secret-token" not in str(error.value)


def test_structured_upstream_error_is_actionable_and_bounded():
    class FakeStream:
        async def __aenter__(self):
            return __import__("httpx").Response(
                400,
                json = {
                    "error": {"code": "invalid_request", "message": "Unsupported request field."}
                },
            )

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

        async def aclose(self):
            return None

    async def run():
        client = OpenAICodexClient("secret-token", "acct-1")
        await client._client.aclose()
        client._client = FakeClient()
        try:
            return [
                line
                async for line in client.stream(
                    provider_id = "provider",
                    thread_id = "thread",
                    messages = [{"role": "user", "content": "hello"}],
                    model = "gpt-5.4",
                    max_tokens = None,
                    reasoning_effort = None,
                    tools = None,
                    tool_choice = None,
                )
            ]
        finally:
            await client.close()

    with pytest.raises(CodexTransportError) as error:
        asyncio.run(run())
    assert error.value.status == 400
    assert "Unsupported request field" in str(error.value)
    assert "secret-token" not in str(error.value)


def test_bare_detail_upstream_error_reaches_the_user():
    """The subscription endpoint reports a rejected model as a bare "detail"."""

    class FakeStream:
        async def __aenter__(self):
            return __import__("httpx").Response(
                400,
                json = {
                    "detail": (
                        "The 'gpt-5.3-codex-spark' model is not supported when using "
                        "Codex with a ChatGPT account."
                    )
                },
            )

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

        async def aclose(self):
            return None

    async def run():
        client = OpenAICodexClient("secret-token", "acct-1")
        await client._client.aclose()
        client._client = FakeClient()
        try:
            return [
                line
                async for line in client.stream(
                    provider_id = "provider",
                    thread_id = "thread",
                    messages = [{"role": "user", "content": "hello"}],
                    model = "gpt-5.3-codex-spark",
                    max_tokens = None,
                    reasoning_effort = None,
                    tools = None,
                    tool_choice = None,
                )
            ]
        finally:
            await client.close()

    with pytest.raises(CodexTransportError) as error:
        asyncio.run(run())
    assert error.value.status == 400
    assert "is not supported" in str(error.value)
    assert "secret-token" not in str(error.value)


def _models_response(payload, status = 200):
    import httpx
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def get(
            self,
            url,
            headers = None,
            params = None,
        ):
            self.calls.append((url, params))
            return httpx.Response(status, json = payload)

        async def aclose(self):
            return None

    return FakeClient()


def test_subscription_model_list_keeps_only_listable_slugs(monkeypatch):
    fake = _models_response(
        {
            "models": [
                {
                    "slug": "gpt-5.4",
                    "visibility": "list",
                    "display_name": "GPT-5.4",
                    "context_window": 272000,
                    "input_modalities": ["text", "image"],
                    "supported_reasoning_levels": [
                        {"effort": "low"},
                        {"effort": "high"},
                    ],
                },
                # Internal review slug the picker must never offer.
                {"slug": "codex-auto-review", "visibility": "hide"},
                {"slug": "", "visibility": "list"},
                "not-a-model",
            ]
        }
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-1")

    models = asyncio.run(list_subscription_models("provider-1", "secret-token", "acct-1"))

    assert models == [
        {
            "id": "gpt-5.4",
            "display_name": "GPT-5.4",
            "context_length": 272000,
            "vision": True,
            "reasoning_efforts": ["low", "high"],
        }
    ]
    assert fake.calls[0][0] == f"{OPENAI_CODEX_API_BASE}/codex/models"
    assert fake.calls[0][1] == {"client_version": codex_auth.OPENAI_CODEX_CLIENT_VERSION}
    # A second call is served from cache rather than re-hitting upstream.
    assert asyncio.run(list_subscription_models("provider-1", "secret-token", "acct-1")) == models
    assert len(fake.calls) == 1
    # Outlives the cache so a slow save is still accepted by the provider routes.
    assert offered_subscription_model_ids("provider-1") == {"gpt-5.4"}
    forget_subscription_models("provider-1")
    assert cached_subscription_models("provider-1") is None
    assert offered_subscription_model_ids("provider-1") == set()


def test_subscription_catalog_is_dropped_when_the_account_changes(monkeypatch):
    """A reconnect can bind the same provider row to a different ChatGPT account.

    Nothing on the reauthorization path clears the catalog, so a lookup keyed only by
    provider would keep serving the previous plan's slugs for the whole TTL.
    """
    first = _models_response({"models": [{"slug": "gpt-5.4", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: first)
    forget_subscription_models("provider-4")
    asyncio.run(list_subscription_models("provider-4", "token-a", "acct-a"))
    assert offered_subscription_model_ids("provider-4") == {"gpt-5.4"}

    second = _models_response({"models": [{"slug": "gpt-5.5", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: second)
    # Same provider, new account: the stale catalog must not be served from cache.
    models = asyncio.run(list_subscription_models("provider-4", "token-b", "acct-b"))
    assert [model["id"] for model in models] == ["gpt-5.5"]
    assert offered_subscription_model_ids("provider-4") == {"gpt-5.5"}
    assert len(second.calls) == 1
    forget_subscription_models("provider-4")


def test_subscription_model_list_rejects_non_200(monkeypatch):
    monkeypatch.setattr(
        codex_client,
        "_create_http_client",
        lambda: _models_response({"detail": "Not Found"}, status = 404),
    )
    forget_subscription_models("provider-2")

    with pytest.raises(CodexTransportError) as error:
        asyncio.run(list_subscription_models("provider-2", "secret-token", "acct-1"))
    assert error.value.status == 404
    assert "secret-token" not in str(error.value)
    assert cached_subscription_models("provider-2") is None


def test_model_route_falls_back_to_curated_when_upstream_is_unusable(monkeypatch):
    from routes import openai_codex_auth as codex_routes

    curated = get_provider_info("openai_codex")["default_models"]
    monkeypatch.setattr(codex_routes, "_provider", lambda provider_id: {"id": provider_id})

    def call():
        return asyncio.run(
            codex_routes.list_subscription_models(
                "provider-3", _credential = ("user", "session"), via_api_key = False
            )
        )

    monkeypatch.setattr(codex_routes.codex_auth, "auth_status", lambda _id: "disconnected")
    disconnected = call()
    assert disconnected["source"] == "curated"
    assert [model["id"] for model in disconnected["models"]] == curated

    async def _resolve(_provider_id):
        return "secret-token", "acct-1"

    async def _boom(*_args, **_kwargs):
        raise CodexTransportError("upstream is down")

    monkeypatch.setattr(codex_routes.codex_auth, "auth_status", lambda _id: "connected")
    monkeypatch.setattr(codex_routes.codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_routes.codex_client, "list_subscription_models", _boom)
    assert call()["source"] == "curated"

    async def _models(*_args, **_kwargs):
        return [{"id": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "context_length": 272000}]

    monkeypatch.setattr(codex_routes.codex_client, "list_subscription_models", _models)
    live = call()
    assert live["source"] == "subscription"
    assert [model["id"] for model in live["models"]] == ["gpt-5.6-terra"]


def test_client_never_emits_done_marker_itself():
    # The route owns the one Chat-Completions [DONE] marker.
    assert not any("[DONE]" in line for line in asyncio.run(_successful_stream_lines()))


async def _successful_stream_lines():
    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"response.completed","response":{}}'
            yield "data: [DONE]"

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return False

    class Client:
        def stream(self, *_args, **_kwargs):
            return Stream()

        async def aclose(self):
            return None

    client = OpenAICodexClient("secret", "account")
    await client._client.aclose()
    client._client = Client()
    try:
        return [
            line
            async for line in client.stream(
                provider_id = "provider",
                thread_id = None,
                messages = [{"role": "user", "content": "hello"}],
                model = "gpt-5.4",
                max_tokens = None,
                reasoning_effort = None,
                tools = None,
                tool_choice = None,
            )
        ]
    finally:
        await client.close()


def test_browser_no_bind_fallback_keeps_registered_manual_redirect(monkeypatch):
    attempted = []

    async def no_bind(_handler, host, port):
        attempted.append((host, port))
        raise OSError("busy")

    monkeypatch.setattr(codex_auth.asyncio, "start_server", no_bind)
    flow = asyncio.run(codex_auth._start_browser_flow("provider", lambda *_args: None))
    assert attempted == [("127.0.0.1", 1455)]
    assert flow.server is None
    assert flow.redirect_uri == "http://localhost:1455/auth/callback"
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" in (
        flow.authorization_url
    )


def test_permanent_refresh_rejection_sets_only_sanitized_reconnect_marker(monkeypatch):
    bundle = {
        "access_token": "secret-access",
        "refresh_token": "secret-refresh",
        "expires_at": time.time() - 10,
        "account_id": "secret-account",
    }
    saved = []

    class Lock:
        def acquire(self):
            return None

        def release(self):
            return None

    async def rejected(_data):
        raise codex_auth.CodexReauthorizationRequired(
            "ChatGPT authorization is no longer valid. Please reconnect."
        )

    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _provider: bundle.copy())
    monkeypatch.setattr(
        codex_auth, "save_oauth_bundle", lambda _provider, value: saved.append(value)
    )
    monkeypatch.setattr(codex_auth, "FileLock", lambda *_args, **_kwargs: Lock())
    monkeypatch.setattr(codex_auth, "_token_request", rejected)
    codex_auth._refresh_locks.clear()

    with pytest.raises(codex_auth.CodexReauthorizationRequired) as error:
        asyncio.run(codex_auth.resolve_access("provider"))
    assert str(error.value) == "ChatGPT authorization is no longer valid. Please reconnect."
    assert saved and saved[-1]["reauthorization_required"] is True
    assert "secret-access" not in str(error.value)
    assert "secret-refresh" not in str(error.value)


def test_stale_unauthorized_response_does_not_poison_reconnected_bundle(monkeypatch):
    bundle = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_at": time.time() + 600,
        "account_id": "account",
    }
    saved = []
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _provider: bundle.copy())
    monkeypatch.setattr(
        codex_auth,
        "save_oauth_bundle",
        lambda _provider, value: saved.append(value),
    )

    codex_auth.mark_reauthorization_required("provider", "old-access")
    assert saved == []
    codex_auth.mark_reauthorization_required("provider", "new-access")
    assert saved[-1]["reauthorization_required"] is True


def test_expired_flow_is_removed_after_terminal_retention(monkeypatch):
    flow = OAuthFlow(
        id = "expired-flow",
        provider_id = "provider",
        method = "browser",
        created_at = time.time() - 10,
        expires_at = time.time() - 1,
    )
    codex_auth._flows[flow.id] = flow
    monkeypatch.setattr(codex_auth, "_FLOW_TERMINAL_RETENTION_SECONDS", 0)

    asyncio.run(codex_auth._expire_and_remove_flow(flow))

    assert flow.id not in codex_auth._flows
    assert flow.status == "cancelled"


def test_cancellation_interrupts_request_before_response_headers():
    cancel_event = threading.Event()
    entered = asyncio.Event()
    enter_cancelled = asyncio.Event()

    class SlowStream:
        async def __aenter__(self):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                enter_cancelled.set()
                raise

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return SlowStream()

        async def aclose(self):
            return None

    async def run():
        client = OpenAICodexClient("secret", "account")
        await client._client.aclose()
        client._client = FakeClient()
        task = asyncio.create_task(_collect_codex_lines(client, cancel_event))
        await entered.wait()
        cancel_event.set()
        lines = await asyncio.wait_for(task, timeout = 1)
        await client.close()
        return lines

    assert asyncio.run(run()) == []
    assert enter_cancelled.is_set()


async def _collect_codex_lines(client, cancel_event):
    return [
        line
        async for line in client.stream(
            provider_id = "provider",
            thread_id = "thread",
            messages = [{"role": "user", "content": "hello"}],
            model = "gpt-5.4",
            max_tokens = None,
            reasoning_effort = None,
            tools = None,
            tool_choice = None,
            cancel_event = cancel_event,
        )
    ]


def test_transient_refresh_failure_does_not_require_reauthorization(monkeypatch):
    class FakeResponse:
        status_code = 401
        headers = {}

        async def aread(self):
            return b""

        def json(self):
            return {"error": {"message": "expired"}}

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

        async def aclose(self):
            return None

    async def transient_refresh():
        raise codex_auth.CodexAuthError("Could not reach ChatGPT authentication.")

    async def run():
        client = OpenAICodexClient("secret", "account", refresh_access = transient_refresh)
        await client._client.aclose()
        client._client = FakeClient()
        try:
            await _collect_codex_lines(client, threading.Event())
        finally:
            await client.close()

    with pytest.raises(CodexTransportError) as error:
        asyncio.run(run())
    assert not isinstance(error.value, CodexReauthorizationError)


def test_codex_tool_loop_autoinjects_rag_before_first_model_call(monkeypatch):
    from core.inference import openai_codex_tool_loop as tool_loop
    from core.inference import studio_tool_loop as loop_core

    class FakeCodexClient:
        def __init__(self):
            self.messages = []

        async def stream(self, **kwargs):
            self.messages.append(kwargs["messages"])
            yield 'data: {"choices":[{"delta":{"content":"from docs"},"finish_reason":"stop"}]}'

    injected_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "rag_auto_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query":"docs"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "search_knowledge_base",
            "tool_call_id": "rag_auto_1",
            "content": "retrieved context",
        },
    ]
    monkeypatch.setattr(
        loop_core,
        "build_rag_autoinject",
        lambda *_args: {
            "events": [{"type": "status", "text": "Searching documents"}],
            "messages": injected_messages,
        },
        raising = False,
    )
    client = FakeCodexClient()

    async def run():
        return [
            line
            async for line in tool_loop.stream_codex_with_studio_tools(
                client,
                run = tool_loop.CodexRunContext(
                    provider_id = "provider",
                    thread_id = "thread",
                    session_id = "session",
                    messages = [{"role": "user", "content": "docs"}],
                    model = "gpt-5.6-sol",
                    reasoning_effort = "medium",
                ),
                policy = tool_loop.CodexToolPolicy(
                    tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
                    max_calls = 2,
                    timeout = 30,
                    permission_mode = "auto",
                    confirm_calls = True,
                    bypass_permissions = False,
                    rag_scope = {"thread_id": "thread"},
                ),
                cancel_event = threading.Event(),
            )
        ]

    lines = asyncio.run(run())
    assert lines[0] == 'data: {"type":"status","text":"Searching documents"}'
    assert client.messages[0][-2:] == injected_messages


def test_codex_studio_tool_loop_executes_and_continues(monkeypatch):
    from core.inference import openai_codex_tool_loop as tool_loop
    from core.inference import studio_tool_loop as loop_core

    class FakeCodexClient:
        def __init__(self):
            self.messages = []

        async def stream(self, **kwargs):
            self.messages.append(kwargs["messages"])
            if len(self.messages) == 1:
                yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"python","arguments":"{\\"code\\":\\"print(6 * 7)\\"}"}}]},"finish_reason":null}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
            else:
                yield 'data: {"choices":[{"delta":{"content":"The result is 42."},"finish_reason":null}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'

    executed = []
    monkeypatch.setattr(
        loop_core,
        "execute_tool",
        lambda name, arguments, **kwargs: executed.append((name, arguments, kwargs)) or "42",
    )
    client = FakeCodexClient()

    async def run():
        return [
            line
            async for line in tool_loop.stream_codex_with_studio_tools(
                client,
                run = tool_loop.CodexRunContext(
                    provider_id = "provider",
                    thread_id = "thread",
                    session_id = "sandbox",
                    messages = [{"role": "user", "content": "calculate"}],
                    model = "gpt-5.6-sol",
                    reasoning_effort = "medium",
                ),
                policy = tool_loop.CodexToolPolicy(
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "description": "Run Python",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    max_calls = 2,
                    timeout = 30,
                    permission_mode = "off",
                    confirm_calls = False,
                    bypass_permissions = False,
                    rag_scope = None,
                ),
                cancel_event = threading.Event(),
            )
        ]

    lines = asyncio.run(run())
    assert executed[0][0:2] == ("python", {"code": "print(6 * 7)"})
    assert any('"type":"tool_start"' in line for line in lines)
    assert any('"type":"tool_end"' in line and '"result":"42"' in line for line in lines)

    assert any(
        '"provenance":{"source":"local","round_id":1}' in line
        for line in lines
        if '"type":"tool_start"' in line or '"type":"tool_end"' in line
    )
    assert any("The result is 42." in line for line in lines)
    assert client.messages[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "python",
        "content": "42",
    }


def test_codex_tool_budget_resolves_parallel_overflow_without_executing_it(monkeypatch):
    from core.inference import openai_codex_tool_loop as tool_loop
    from core.inference import studio_tool_loop as loop_core

    class FakeCodexClient:
        def __init__(self):
            self.requests = []

        async def stream(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"python","arguments":"{}"}},{"index":1,"id":"call_2","type":"function","function":{"name":"terminal","arguments":"{}"}}]},"finish_reason":null}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
            else:
                yield 'data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'

    executed = []
    monkeypatch.setattr(
        loop_core,
        "execute_tool",
        lambda name, arguments, **kwargs: executed.append(name) or "ok",
    )
    client = FakeCodexClient()

    async def run():
        return [
            line
            async for line in tool_loop.stream_codex_with_studio_tools(
                client,
                run = tool_loop.CodexRunContext(
                    provider_id = "provider",
                    thread_id = "thread",
                    session_id = "session",
                    messages = [{"role": "user", "content": "go"}],
                    model = "gpt-5.6-sol",
                    reasoning_effort = "medium",
                ),
                policy = tool_loop.CodexToolPolicy(
                    tools = [{"type": "function", "function": {"name": "python"}}],
                    max_calls = 1,
                    timeout = 30,
                    permission_mode = "off",
                    confirm_calls = False,
                    bypass_permissions = False,
                    rag_scope = None,
                ),
                cancel_event = threading.Event(),
            )
        ]

    lines = asyncio.run(run())
    assert executed == ["python"]
    assert any("per-message tool-call limit" in line and "call_2" in line for line in lines)
    assert client.requests[1]["tools"] is None
    assert client.requests[1]["tool_choice"] == "none"
    replayed = client.requests[1]["messages"]
    assert [message["tool_call_id"] for message in replayed if message["role"] == "tool"] == [
        "call_1",
        "call_2",
    ]
    # The shared loop carries the local loops' closing nudge, so the tool results
    # are no longer the tail of the conversation.
    assert replayed[-1]["role"] == "user"
    assert "provide your final answer now" in replayed[-1]["content"]


def _codex_chat_gate(
    monkeypatch,
    model: str,
    resolve = None,
):
    """Drive the chat route far enough to answer "may this model be used?".

    The gate is one line inside ``_proxy_to_external_provider`` and is only
    reachable through the route, so the access resolver is stubbed to raise: a
    401 means the model was accepted and the request moved on, a 400 means it
    was refused.
    """
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest
    from routes import inference as inf

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": "openai_codex",
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT subscription",
            "is_enabled": True,
        },
    )

    async def _refuse(*_args, **_kwargs):
        raise codex_auth.CodexAuthError("stub: past the model gate")

    monkeypatch.setattr(codex_auth, "resolve_access", resolve or _refuse)

    async def _is_disconnected():
        return False

    request = SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = _is_disconnected,
    )
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}],
        provider_id = "codex-1",
        external_model = model,
        stream = True,
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf._proxy_to_external_provider(payload, request, current_subject = "t"))
    return excinfo.value


def test_chat_accepts_a_plan_listed_slug_the_seed_does_not_carry(monkeypatch):
    """A slug the picker offered and the provider routes saved must be chattable.

    ``/codex/models`` is the truth once connected, so a plan can list a model
    newer than the curated seed. The provider routes accept saving it; gating
    the chat route on the seed alone would reject the very model the user just
    picked, on every message.
    """
    listed = "gpt-5.7-nova"
    assert listed not in get_provider_info("openai_codex")["default_models"]

    forget_subscription_models("codex-1")
    refused = _codex_chat_gate(monkeypatch, listed)
    assert refused.status_code == 400
    assert "Choose a curated Codex model." in str(refused.detail)

    # Exactly what a picker fetch records for this connection.
    codex_client._offered_models["codex-1"] = {
        listed: {"id": listed, "display_name": listed, "vision": True}
    }
    try:
        accepted = _codex_chat_gate(monkeypatch, listed)
        assert accepted.status_code == 401, accepted.detail
        assert "Choose a curated Codex model." not in str(accepted.detail)
        # A slug no plan ever listed is still refused.
        never_listed = _codex_chat_gate(monkeypatch, "gpt-5.3-codex-spark")
        assert never_listed.status_code == 400
        assert "Choose a curated Codex model." in str(never_listed.detail)
    finally:
        forget_subscription_models("codex-1")


def test_chat_refetches_the_plan_catalog_after_a_restart(monkeypatch):
    """A restart empties the in-memory catalog; the saved slug must still work.

    Nothing refetches /codex/models on startup, so gating on the seed alone would
    reject a model the user legitimately saved until they reopened the connection
    editor.
    """
    listed = "gpt-5.7-nova"
    forget_subscription_models("codex-1")

    calls = []

    async def _resolve(_provider_id):
        calls.append(_provider_id)
        # The gate's refresh resolves first; the chat path's own call then stops
        # the request before it can reach upstream.
        if len(calls) > 1:
            raise codex_auth.CodexAuthError("stub: past the model gate")
        return "secret-token", "acct-1"

    fake = _models_response(
        {"models": [{"slug": listed, "visibility": "list", "input_modalities": ["text"]}]}
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    try:
        # Cold cache, exactly as after a restart: the gate fetches rather than refusing.
        accepted = _codex_chat_gate(monkeypatch, listed, resolve = _resolve)
        assert accepted.status_code == 401, accepted.detail
        assert offered_subscription_model_ids("codex-1") == {listed}
        assert len(calls) == 2
    finally:
        forget_subscription_models("codex-1")


def test_chat_refuses_when_the_catalog_cannot_be_refreshed(monkeypatch):
    """An unreachable upstream falls back to the seed instead of locking the user out."""
    forget_subscription_models("codex-1")

    async def _fail(_provider_id):
        raise codex_auth.CodexAuthError("disconnected")

    monkeypatch.setattr(codex_auth, "resolve_access", _fail)
    try:
        # A seed model still passes the gate.
        seeded = _codex_chat_gate(monkeypatch, "gpt-5.4")
        assert seeded.status_code == 401, seeded.detail
        # An unknown slug is refused rather than proxied blind.
        refused = _codex_chat_gate(monkeypatch, "gpt-5.7-nova")
        assert refused.status_code == 400
    finally:
        forget_subscription_models("codex-1")


def test_chat_reads_vision_support_from_the_plan_catalog(monkeypatch):
    """A dynamic slug's image support comes from /codex/models, not the static registry."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest
    from routes import inference as inf

    listed = "gpt-5.7-nova"
    assert listed not in get_provider_info("openai_codex")["model_capabilities"]

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": "openai_codex",
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT subscription",
            "is_enabled": True,
        },
    )

    async def _refuse(*_args, **_kwargs):
        raise codex_auth.CodexAuthError("stub: past the image gate")

    monkeypatch.setattr(codex_auth, "resolve_access", _refuse)

    async def _is_disconnected():
        return False

    def call():
        request = SimpleNamespace(
            headers = {},
            state = SimpleNamespace(skip_api_monitor = True),
            is_disconnected = _is_disconnected,
        )
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            provider_id = "codex-1",
            external_model = listed,
            stream = True,
        )
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inf._proxy_to_external_provider(payload, request, current_subject = "t"))
        return excinfo.value

    codex_client._offered_models["codex-1"] = {
        listed: {"id": listed, "display_name": listed, "vision": False}
    }
    try:
        refused = call()
        assert refused.status_code == 400
        assert "does not accept image input" in str(refused.detail)
        # The same slug listed as image-capable is carried through to the provider.
        codex_client._offered_models["codex-1"] = {
            listed: {"id": listed, "display_name": listed, "vision": True}
        }
        accepted = call()
        assert accepted.status_code == 401, accepted.detail
        assert "does not accept image input" not in str(accepted.detail)
    finally:
        forget_subscription_models("codex-1")

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

import httpx
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
            "listed": True,
        },
        # Kept and marked, not dropped: an account can still call a slug it saved while
        # the slug was listed, and only the picker needs to stop offering it.
        {
            "id": "codex-auto-review",
            "display_name": "codex-auto-review",
            "context_length": None,
            "vision": None,
            "reasoning_efforts": [],
            "listed": False,
        },
    ]
    assert fake.calls[0][0] == f"{OPENAI_CODEX_API_BASE}/codex/models"
    assert fake.calls[0][1] == {"client_version": codex_auth.OPENAI_CODEX_CLIENT_VERSION}
    # A second call is served from cache rather than re-hitting upstream.
    assert asyncio.run(list_subscription_models("provider-1", "secret-token", "acct-1")) == models
    assert len(fake.calls) == 1
    # Outlives the cache so a slow save is still accepted by the provider routes.
    # Cached for its metadata, but a hidden slug is not authorized by the fetch alone.
    assert offered_subscription_model_ids("provider-1") == {"gpt-5.4"}
    assert codex_client.offered_subscription_model("provider-1", "codex-auto-review") is not None
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


def test_persisting_a_new_account_drops_the_plan_catalog(monkeypatch):
    """Rebinding a connection to another ChatGPT account retires its catalog at once.

    The user can leave the form before the browser callback lands, so the post-connect
    picker refresh never runs, and the chat gate only refetches a catalog it does not
    already have. Nothing else would notice the account changed.
    """
    stored = {}

    monkeypatch.setattr(
        codex_auth.credential_secrets,
        "upsert_secret",
        lambda _kind, provider_id, raw: stored.__setitem__(provider_id, raw),
    )
    monkeypatch.setattr(
        codex_auth.credential_secrets,
        "get_secret",
        lambda _kind, provider_id: stored.get(provider_id),
    )

    def _bundle(account_id):
        return {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": 1,
            "account_id": account_id,
        }

    codex_auth.save_oauth_bundle("provider-5", _bundle("acct-a"))
    codex_client._offered_models["provider-5"] = {
        "gpt-5.7-nova": {"id": "gpt-5.7-nova", "listed": True}
    }
    try:
        # A refresh for the same account keeps it: only the account identity matters.
        codex_auth.save_oauth_bundle("provider-5", _bundle("acct-a"))
        assert offered_subscription_model_ids("provider-5") == {"gpt-5.7-nova"}

        codex_auth.save_oauth_bundle("provider-5", _bundle("acct-b"))
        assert offered_subscription_model_ids("provider-5") == set()
    finally:
        forget_subscription_models("provider-5")


def test_a_forced_reload_skips_the_cached_catalog(monkeypatch):
    """An explicit reload asks about plan changes, so the 600s cache must not answer it."""
    first = _models_response({"models": [{"slug": "gpt-5.4", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: first)
    forget_subscription_models("provider-6")
    asyncio.run(list_subscription_models("provider-6", "token", "acct-1"))
    assert len(first.calls) == 1

    second = _models_response({"models": [{"slug": "gpt-5.8-new", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: second)
    try:
        # Unforced, the fresh slug stays invisible for the rest of the TTL.
        cached = asyncio.run(list_subscription_models("provider-6", "token", "acct-1"))
        assert [model["id"] for model in cached] == ["gpt-5.4"]
        assert len(second.calls) == 0

        reloaded = asyncio.run(
            list_subscription_models("provider-6", "token", "acct-1", force = True)
        )
        assert [model["id"] for model in reloaded] == ["gpt-5.8-new"]
        assert len(second.calls) == 1
        # The refreshed catalog replaces what the picker and the chat gate read.
        assert offered_subscription_model_ids("provider-6") == {"gpt-5.8-new"}
    finally:
        forget_subscription_models("provider-6")


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
        return [
            {
                "id": "gpt-5.6-terra",
                "display_name": "GPT-5.6-Terra",
                "context_length": 272000,
                "listed": True,
            },
            # Present on the plan but not offered: reported as known, never in the picker.
            {"id": "codex-auto-review", "display_name": "codex-auto-review", "listed": False},
        ]

    monkeypatch.setattr(codex_routes.codex_client, "list_subscription_models", _models)
    live = call()
    assert live["source"] == "subscription"
    assert [model["id"] for model in live["models"]] == ["gpt-5.6-terra"]
    assert [model["id"] for model in live["known"]] == ["gpt-5.6-terra", "codex-auto-review"]
    # The hidden entry keeps its metadata so the picker can still describe it.
    assert live["known"][1]["display_name"] == "codex-auto-review"


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
    saved_models = None,
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
            "models": list(saved_models or []),
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


def test_codex_chat_receives_the_current_date(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes import inference as inf

    model = get_provider_info("openai_codex")["default_models"][0]
    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": "openai_codex",
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT subscription",
            "is_enabled": True,
            "models": [model],
        },
    )
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-1"})
    monkeypatch.setattr(codex_client, "subscription_catalog_matches_account", lambda *_args: True)
    monkeypatch.setattr(codex_client, "subscription_catalog_known", lambda _pid: False)
    monkeypatch.setattr(codex_client, "subscription_catalog_stale", lambda _pid: False)
    monkeypatch.setattr(codex_client, "saved_models_proven_for", lambda *_args: True)

    async def _resolve_access(_provider_id, **_kwargs):
        return "token", "acct-1"

    monkeypatch.setattr(codex_auth, "resolve_access", _resolve_access)
    captured = {}

    class FakeCodexClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def stream(self, **kwargs):
            captured.update(kwargs)

            async def _stream():
                yield 'data: {"type":"response.completed"}'

            return _stream()

        async def close(self):
            pass

    monkeypatch.setattr(codex_client, "OpenAICodexClient", FakeCodexClient)
    monkeypatch.setattr(
        inf,
        "current_date_prompt_line",
        lambda **_kwargs: "The current date is 2026-08-15.",
    )

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

    async def _run():
        response = await inf._proxy_to_external_provider(payload, request, current_subject = "t")
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(_run())

    assert captured["messages"][0] == {
        "role": "system",
        "content": "The current date is 2026-08-15.",
    }
    assert captured["messages"][1] == {"role": "user", "content": "hello"}


def test_chat_accepts_a_plan_listed_slug_the_seed_does_not_carry(monkeypatch):
    """A slug the picker offered and the provider routes saved must be chattable.

    ``/codex/models`` is the truth once connected, so a plan can list a model
    newer than the curated seed. The provider routes accept saving it; gating
    the chat route on the seed alone would reject the very model the user just
    picked, on every message.
    """
    listed = "gpt-5.7-nova"
    assert listed not in get_provider_info("openai_codex")["default_models"]

    # A catalog that simply does not list it: the refusal is about the model, not the
    # connection, so the gate never reaches for a refresh.
    forget_subscription_models("codex-1")
    codex_client._offered_models["codex-1"] = {"gpt-5.4": {"id": "gpt-5.4", "listed": True}}
    refused = _codex_chat_gate(monkeypatch, listed)
    assert refused.status_code == 400
    assert "Choose a curated Codex model." in str(refused.detail)

    # Exactly what a picker fetch records for this connection.
    codex_client._offered_models["codex-1"] = {
        listed: {"id": listed, "display_name": listed, "vision": True, "listed": True}
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
    """An unreachable catalog refuses the model; it does not declare the connection bad."""
    import httpx

    forget_subscription_models("codex-1")

    async def _resolve(_provider_id):
        return "secret-token", "acct-1"

    class Unreachable:
        async def get(self, *_args, **_kwargs):
            raise httpx.ConnectError("upstream down")

        async def aclose(self):
            return None

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Unreachable())
    try:
        refused = _codex_chat_gate(monkeypatch, "gpt-5.7-nova", resolve = _resolve)
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)
    finally:
        forget_subscription_models("codex-1")


def test_chat_asks_for_reconnection_rather_than_another_model(monkeypatch):
    """A dead connection is not a bad model choice, and the message has to say so.

    The catalog is unreadable because the credentials are, so refusing with "choose a
    curated model" would send the user to fix a selection that may be perfectly valid.
    """
    forget_subscription_models("codex-1")

    async def _needs_reauth(_provider_id):
        raise codex_auth.CodexAuthError("ChatGPT authorization expired. Reconnect.")

    try:
        refused = _codex_chat_gate(monkeypatch, "gpt-5.7-nova", resolve = _needs_reauth)
        assert refused.status_code == 401
        assert "Reconnect" in str(refused.detail)
        assert "Choose a curated Codex model." not in str(refused.detail)
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
        listed: {"id": listed, "display_name": listed, "vision": False, "listed": True}
    }
    try:
        refused = call()
        assert refused.status_code == 400
        assert "does not accept image input" in str(refused.detail)
        # The same slug listed as image-capable is carried through to the provider.
        codex_client._offered_models["codex-1"] = {
            listed: {"id": listed, "display_name": listed, "vision": True, "listed": True}
        }
        accepted = call()
        assert accepted.status_code == 401, accepted.detail
        assert "does not accept image input" not in str(accepted.detail)
    finally:
        forget_subscription_models("codex-1")


def test_chat_keeps_a_saved_slug_the_plan_stopped_listing(monkeypatch):
    """visibility is how a slug is presented, not whether the account may call it.

    An aged slug flips to "hide" and drops out of the normalized catalog. Rebuilding the
    allowlist from that catalog alone would refuse a model the user saved while it was
    listed and may still be using.
    """
    hidden = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    # The plan still returns it, marked hidden: that is what ageing out of the picker
    # looks like, as opposed to a slug the account cannot reach at all.
    codex_client._offered_models["codex-1"] = {
        "gpt-5.4": {"id": "gpt-5.4", "listed": True},
        hidden: {"id": hidden, "listed": False},
    }
    try:
        refused = _codex_chat_gate(monkeypatch, hidden)
        assert refused.status_code == 400

        accepted = _codex_chat_gate(monkeypatch, hidden, saved_models = [hidden])
        assert accepted.status_code == 401, accepted.detail
        assert "Choose a curated Codex model." not in str(accepted.detail)
    finally:
        forget_subscription_models("codex-1")


def test_chat_retires_a_saved_slug_the_new_account_does_not_carry(monkeypatch):
    """Reauthorizing to another account is a real revocation, unlike ageing out.

    The saved row still names the previous account's slugs, so trusting it forever would
    keep sending them upstream on an account that never had them.
    """
    stale = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    try:
        # No catalog read yet: the row is the only evidence, so it is still trusted.
        cold = _codex_chat_gate(monkeypatch, stale, saved_models = [stale])
        assert cold.status_code == 401, cold.detail

        # The new account's catalog does not carry it at all, hidden or otherwise.
        codex_client._offered_models["codex-1"] = {"gpt-5.5": {"id": "gpt-5.5", "listed": True}}
        refused = _codex_chat_gate(monkeypatch, stale, saved_models = [stale])
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)
    finally:
        forget_subscription_models("codex-1")


def test_chat_reports_reconnection_when_an_image_needs_the_catalog(monkeypatch):
    """The image gate must not report a text-only model when the connection is dead."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest
    from routes import inference as inf

    saved = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": "openai_codex",
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT subscription",
            "is_enabled": True,
            "models": [saved],
        },
    )

    async def _needs_reauth(*_args, **_kwargs):
        raise codex_auth.CodexAuthError("ChatGPT authorization expired. Reconnect.")

    monkeypatch.setattr(codex_auth, "resolve_access", _needs_reauth)

    async def _is_disconnected():
        return False

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
        external_model = saved,
        stream = True,
    )
    request = SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = _is_disconnected,
    )
    try:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inf._proxy_to_external_provider(payload, request, current_subject = "t"))
        assert excinfo.value.status_code == 401
        assert "does not accept image input" not in str(excinfo.value.detail)
    finally:
        forget_subscription_models("codex-1")


def test_a_hidden_slug_is_not_invocable_just_because_the_catalog_was_fetched(monkeypatch):
    """codex-auto-review and its kin are withheld from the picker on purpose.

    They are cached for their metadata and stay usable on a connection that already
    carries one, but a catalog fetch alone must not make an internal slug invocable.
    """
    hidden = "codex-auto-review"
    fake = _models_response(
        {
            "models": [
                {"slug": "gpt-5.4", "visibility": "list"},
                {"slug": hidden, "visibility": "hide"},
            ]
        }
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("codex-1")
    asyncio.run(list_subscription_models("codex-1", "token", "acct-1"))
    try:
        assert offered_subscription_model_ids("codex-1") == {"gpt-5.4"}
        refused = _codex_chat_gate(monkeypatch, hidden)
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)

        # Still reachable when the connection already carries it.
        accepted = _codex_chat_gate(monkeypatch, hidden, saved_models = [hidden])
        assert accepted.status_code == 401, accepted.detail
    finally:
        forget_subscription_models("codex-1")


def test_chat_stops_trusting_the_seed_once_the_plan_catalog_is_known(monkeypatch):
    """The registry seed bootstraps a connection; it says nothing about this account.

    gpt-5.6-sol is seeded but a Go plan does not carry it, and sending it upstream only
    to be refused is the failure this PR exists to remove.
    """
    seeded = get_provider_info("openai_codex")["default_models"][0]
    forget_subscription_models("codex-1")
    try:
        # Nothing read yet: the seed is all there is, so it is still accepted.
        cold = _codex_chat_gate(monkeypatch, seeded)
        assert cold.status_code == 401, cold.detail

        codex_client._offered_models["codex-1"] = {"gpt-5.5": {"id": "gpt-5.5", "listed": True}}
        refused = _codex_chat_gate(monkeypatch, seeded)
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)

        accepted = _codex_chat_gate(monkeypatch, "gpt-5.5")
        assert accepted.status_code == 401, accepted.detail
    finally:
        forget_subscription_models("codex-1")


def test_chat_does_not_trust_the_saved_row_after_a_rebind(monkeypatch):
    """Rebinding empties the catalog on purpose, so absence is not a cold start here."""
    stale_slug = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    try:
        # A plain cold start still trusts the row.
        cold = _codex_chat_gate(monkeypatch, stale_slug, saved_models = [stale_slug])
        assert cold.status_code == 401, cold.detail

        # Rebound: the row is not evidence, so the gate reads the new account's catalog
        # rather than trusting it, and that catalog does not carry the slug.
        codex_client.mark_subscription_catalog_stale("codex-1")

        async def _resolve(_provider_id):
            return "secret-token", "acct-b"

        fake = _models_response({"models": [{"slug": "gpt-5.5", "visibility": "list"}]})
        monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
        refused = _codex_chat_gate(
            monkeypatch, stale_slug, resolve = _resolve, saved_models = [stale_slug]
        )
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)
        assert len(fake.calls) == 1
    finally:
        forget_subscription_models("codex-1")


def test_disconnecting_leaves_the_saved_models_unproven(monkeypatch):
    """A disconnect erases the identity a later save would be compared against.

    The provider row keeps its models, so without a mark here the next account inherits
    them as cold-start evidence and sends them under its own credentials.
    """
    stored = {}
    monkeypatch.setattr(
        codex_auth.credential_secrets,
        "delete_secret",
        lambda _kind, provider_id: stored.pop(provider_id, None),
    )
    forget_subscription_models("provider-7")
    assert codex_client.subscription_catalog_stale("provider-7") is False

    codex_auth.delete_oauth_bundle("provider-7")
    assert codex_client.subscription_catalog_stale("provider-7") is True
    forget_subscription_models("provider-7")


def test_evicting_the_response_cache_keeps_authorization_evidence(monkeypatch):
    """The TTL cache is bounded; what a plan proved about a connection is not.

    Clearing both together made every other connection look cold, which is exactly the
    state that licenses a saved slug the account no longer carries.
    """
    codex_client._models_cache.clear()
    codex_client._offered_models.clear()
    codex_client._catalog_accounts.clear()
    codex_client._offered_models["other"] = {"gpt-5.4": {"id": "gpt-5.4", "listed": True}}
    codex_client._catalog_accounts["other"] = "acct-other"
    for filler in range(codex_client._MODELS_CACHE_MAX_ENTRIES):
        codex_client._models_cache[f"filler-{filler}"] = (time.time() + 600, [])

    fake = _models_response({"models": [{"slug": "gpt-5.5", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    try:
        asyncio.run(list_subscription_models("provider-8", "token", "acct-8"))
        assert codex_client.cached_subscription_models("filler-0") is None
        # The other connection's proof survives the eviction.
        assert offered_subscription_model_ids("other") == {"gpt-5.4"}
        assert codex_client._catalog_accounts["other"] == "acct-other"
    finally:
        forget_subscription_models("provider-8")
        forget_subscription_models("other")
        codex_client._models_cache.clear()


def test_a_superseded_catalog_read_does_not_commit(monkeypatch):
    """A read overtaken by a rebind must not reinstate the account it was reading for.

    Committing late would also clear the mark that says the saved models are unproven,
    turning a self-correcting state into a sticky one.
    """
    import httpx

    forget_subscription_models("provider-9")

    class Rebinding:
        async def get(self, *_args, **_kwargs):
            # The rebind lands while this read is out.
            forget_subscription_models("provider-9")
            mark = codex_client.mark_subscription_catalog_stale
            mark("provider-9")
            return httpx.Response(200, json = {"models": [{"slug": "gpt-5.4", "visibility": "list"}]})

        async def aclose(self):
            return None

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Rebinding())
    try:
        models = asyncio.run(list_subscription_models("provider-9", "token-a", "acct-a"))
        assert [model["id"] for model in models] == ["gpt-5.4"]
        # Returned to its caller, but never stored over what replaced it.
        assert codex_client.subscription_catalog_known("provider-9") is False
        assert codex_client.subscription_catalog_stale("provider-9") is True
    finally:
        forget_subscription_models("provider-9")


def test_chat_drops_a_catalog_another_worker_rebound(monkeypatch):
    """The OAuth bundle is shared through the DB; the catalog is per process.

    Unsloth serializes token refreshes across workers on purpose, so this process can hold
    a catalog for an account the connection has since moved off.
    """
    stale_slug = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    codex_client._offered_models["codex-1"] = {stale_slug: {"id": stale_slug, "listed": True}}
    codex_client._catalog_accounts["codex-1"] = "acct-a"

    # What the shared DB now says this connection is bound to.
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-b"})
    try:
        refused = _codex_chat_gate(monkeypatch, stale_slug, saved_models = [stale_slug])
        assert refused.status_code == 401, refused.detail
        # The catalog for the previous account is gone and the row is unproven again.
        assert codex_client.subscription_catalog_known("codex-1") is False
        assert codex_client.subscription_catalog_stale("codex-1") is True
    finally:
        forget_subscription_models("codex-1")


def test_the_model_route_reports_a_dead_connection(monkeypatch):
    """The editor route must say reconnect rather than answer with a healthy seed list."""
    from routes import openai_codex_auth as codex_routes

    curated = get_provider_info("openai_codex")["default_models"]
    monkeypatch.setattr(codex_routes, "_provider", lambda provider_id: {"id": provider_id})
    monkeypatch.setattr(codex_routes.codex_auth, "auth_status", lambda _id: "connected")

    async def _needs_reauth(_provider_id):
        raise codex_auth.CodexAuthError("ChatGPT authorization is no longer valid.")

    monkeypatch.setattr(codex_routes.codex_auth, "resolve_access", _needs_reauth)
    answered = asyncio.run(
        codex_routes.list_subscription_models(
            "provider-10", _credential = ("user", "session"), via_api_key = False
        )
    )
    # Not a 401: authFetch would read that as an expired Unsloth session, refresh it and
    # retry, and the retry would look like a healthy curated list.
    assert answered["source"] == "reauthorization_required"
    assert [model["id"] for model in answered["models"]] == curated


def test_a_catalog_401_spends_one_forced_refresh(monkeypatch):
    """Upstream can reject a token before its recorded expiry; the refresh may be fine.

    The responses transport already spends one forced refresh on that, so the editor
    should not be the only path that gives up and demands a reconnect.
    """
    import httpx

    forget_subscription_models("provider-11")
    calls = []

    class Rejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            calls.append(headers["Authorization"])
            if len(calls) == 1:
                return httpx.Response(401, json = {"detail": "expired"})
            return httpx.Response(200, json = {"models": [{"slug": "gpt-5.4", "visibility": "list"}]})

        async def aclose(self):
            return None

    refreshed = []

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        refreshed.append(force_refresh)
        return "fresh-token", "acct-1"

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Rejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    try:
        models = asyncio.run(list_subscription_models("provider-11", "stale-token", "acct-1"))
        assert [model["id"] for model in models] == ["gpt-5.4"]
        assert refreshed == [True]
        assert calls == ["Bearer stale-token", "Bearer fresh-token"]
    finally:
        forget_subscription_models("provider-11")


def test_a_second_catalog_401_asks_for_reconnection(monkeypatch):
    """A refresh that does not help is a real reauthorization, not an endless retry."""
    import httpx

    forget_subscription_models("provider-12")

    class AlwaysRejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(401, json = {"detail": "expired"})

        async def aclose(self):
            return None

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "fresh-token", "acct-1"

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: AlwaysRejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    try:
        with pytest.raises(CodexReauthorizationError):
            asyncio.run(list_subscription_models("provider-12", "stale-token", "acct-1"))
    finally:
        forget_subscription_models("provider-12")


def test_a_refresh_that_cannot_be_reached_stays_retryable(monkeypatch):
    """An unanswered refresh is not a rejected credential.

    resolve_access raises the permanent variant only when the refresh token itself was
    rejected, so anything else has to stay transient or the user is sent to reconnect a
    connection whose credentials are fine.
    """
    import httpx

    forget_subscription_models("provider-13")

    class Rejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(401, json = {"detail": "expired"})

        async def aclose(self):
            return None

    async def _unreachable(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        raise codex_auth.CodexAuthError("token endpoint unreachable")

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Rejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _unreachable)
    try:
        with pytest.raises(CodexTransportError) as excinfo:
            asyncio.run(list_subscription_models("provider-13", "stale", "acct-1"))
        assert not isinstance(excinfo.value, CodexReauthorizationError)
    finally:
        forget_subscription_models("provider-13")


def test_a_rejected_refresh_credential_is_a_reauthorization(monkeypatch):
    """The permanent variant still means reconnect."""
    import httpx

    forget_subscription_models("provider-14")

    class Rejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(401, json = {"detail": "expired"})

        async def aclose(self):
            return None

    async def _rejected(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        raise codex_auth.CodexReauthorizationRequired("refresh token rejected")

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Rejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _rejected)
    try:
        with pytest.raises(CodexReauthorizationError):
            asyncio.run(list_subscription_models("provider-14", "stale", "acct-1"))
    finally:
        forget_subscription_models("provider-14")


def test_a_catalog_is_not_committed_for_an_account_another_worker_replaced(monkeypatch):
    """The request counter is process-local; rebinding travels through the shared DB.

    A read this worker started is not retired by another worker's rebind, so the stored
    bundle is the only thing that can say the answer is for the wrong account.
    """
    import httpx

    forget_subscription_models("provider-15")

    class Slow:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(200, json = {"models": [{"slug": "gpt-5.4", "visibility": "list"}]})

        async def aclose(self):
            return None

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Slow())
    # While the read was out, another worker rebound the connection to acct-b.
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-b"})
    try:
        models = asyncio.run(list_subscription_models("provider-15", "token-a", "acct-a"))
        assert [model["id"] for model in models] == ["gpt-5.4"]
        assert codex_client.subscription_catalog_known("provider-15") is False

        # The same read for the account the DB actually names does commit.
        forget_subscription_models("provider-15")
        asyncio.run(list_subscription_models("provider-15", "token-b", "acct-b"))
        assert offered_subscription_model_ids("provider-15") == {"gpt-5.4"}
    finally:
        forget_subscription_models("provider-15")


def test_the_model_route_reports_an_already_marked_connection(monkeypatch):
    """A bundle marked by someone else still has to reach the editor as reconnect."""
    from routes import openai_codex_auth as codex_routes

    curated = get_provider_info("openai_codex")["default_models"]
    monkeypatch.setattr(codex_routes, "_provider", lambda provider_id: {"id": provider_id})
    monkeypatch.setattr(
        codex_routes.codex_auth, "auth_status", lambda _id: "reauthorization_required"
    )
    answered = asyncio.run(
        codex_routes.list_subscription_models(
            "provider-16", _credential = ("user", "session"), via_api_key = False
        )
    )
    assert answered["source"] == "reauthorization_required"
    assert [model["id"] for model in answered["models"]] == curated

    monkeypatch.setattr(codex_routes.codex_auth, "auth_status", lambda _id: "disconnected")
    plain = asyncio.run(
        codex_routes.list_subscription_models(
            "provider-16", _credential = ("user", "session"), via_api_key = False
        )
    )
    assert plain["source"] == "curated"


def test_an_overtaken_read_still_answers_its_own_caller(monkeypatch):
    """Its models came from upstream for this account even though a newer read owns the cache.

    Reporting nothing listed would let a manual reload overlapping a chat refuse a model
    the chat's own lookup had just seen.
    """
    import httpx

    forget_subscription_models("provider-17")

    class Overtaken:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            # A newer read for the same connection starts while this one is out.
            codex_client._begin_catalog_request("provider-17")
            return httpx.Response(200, json = {"models": [{"slug": "gpt-5.4", "visibility": "list"}]})

        async def aclose(self):
            return None

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "token", "acct-1"

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Overtaken())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-1"})
    try:
        listed = asyncio.run(codex_client.ensure_subscription_models("provider-17"))
        assert listed == {"gpt-5.4"}
        # Still not stored: the newer read owns the cache.
        assert codex_client.subscription_catalog_known("provider-17") is False
    finally:
        forget_subscription_models("provider-17")


def test_an_overtaken_read_is_dropped_when_the_account_moved(monkeypatch):
    """If a rebind is what overtook it, its models belong to the previous account."""
    import httpx

    forget_subscription_models("provider-18")

    class Overtaken:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            codex_client._begin_catalog_request("provider-18")
            return httpx.Response(200, json = {"models": [{"slug": "gpt-5.4", "visibility": "list"}]})

        async def aclose(self):
            return None

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "token", "acct-a"

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Overtaken())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-b"})
    try:
        assert asyncio.run(codex_client.ensure_subscription_models("provider-18")) == set()
    finally:
        forget_subscription_models("provider-18")


def test_a_cold_worker_does_not_trust_a_row_it_cannot_vouch_for(monkeypatch):
    """The stale mark dies with the process; the record next to the credentials does not."""
    saved = "gpt-5.7-nova"
    forget_subscription_models("codex-1")

    import httpx

    calls = []

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        calls.append(_provider_id)
        # The gate's own refresh resolves; the chat path's later call stops the request.
        if len(calls) > 1:
            raise codex_auth.CodexAuthError("stub: past the model gate")
        return "token", "acct-b"

    class Unreachable:
        async def get(self, *_args, **_kwargs):
            raise httpx.ConnectError("upstream down")

        async def aclose(self):
            return None

    # Cold: the refresh cannot reach upstream, so the catalog stays unknown and the
    # decision is the cold branch alone.
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: Unreachable())
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-b"})
    try:
        refused = _codex_chat_gate(monkeypatch, saved, resolve = _resolve, saved_models = [saved])
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)

        # The same cold process, with the row on record as proven for this account.
        monkeypatch.setattr(
            codex_auth,
            "load_oauth_bundle",
            lambda _pid: {"account_id": "acct-b", "catalog_account_id": "acct-b"},
        )
        proven_calls = []

        async def _always_refuse(
            _provider_id,
            force_refresh = False,
            expected_access_token = None,
        ):
            proven_calls.append(_provider_id)
            raise codex_auth.CodexAuthError("stub: past the model gate")

        accepted = _codex_chat_gate(
            monkeypatch, saved, resolve = _always_refuse, saved_models = [saved]
        )
        assert accepted.status_code == 401, accepted.detail
        # Allowed straight off the row, so the gate never reached for a catalog: the one
        # call is the chat path's own.
        assert len(proven_calls) == 1
    finally:
        forget_subscription_models("codex-1")


def test_reading_a_catalog_does_not_by_itself_prove_the_saved_row(monkeypatch):
    """Reading a catalog says which account answered, not that the row was judged by it.

    After a rebind the user can open the editor, which fetches, and then cancel without
    saving, leaving the previous account's slugs in the row. Recording proof on the read
    would authorize exactly those on the next cold start.
    """
    stored = {"provider-19": None}
    bundle = {
        "access_token": "token",
        "refresh_token": "refresh",
        "expires_at": 1,
        "account_id": "acct-1",
    }
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: dict(bundle))
    monkeypatch.setattr(
        codex_auth,
        "save_oauth_bundle",
        lambda provider_id, value: stored.__setitem__(provider_id, value),
    )
    fake = _models_response({"models": [{"slug": "gpt-5.4", "visibility": "list"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-19")
    try:
        asyncio.run(list_subscription_models("provider-19", "token", "acct-1"))
        assert stored["provider-19"] is None
    finally:
        forget_subscription_models("provider-19")


def test_a_token_refresh_keeps_the_catalog_proof(monkeypatch):
    """Rotating credentials is not rebinding, so the proof has to survive it.

    _validate_token_payload rebuilds the bundle from the token response alone, which
    knows nothing about which account the saved models were proven for.
    """
    stored = {}

    bundle = {
        "access_token": "old",
        "refresh_token": "refresh-1",
        "expires_at": 1,
        "account_id": "acct-1",
        "catalog_account_id": "acct-1",
    }

    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: dict(bundle))
    monkeypatch.setattr(
        codex_auth,
        "save_oauth_bundle",
        lambda provider_id, value: stored.__setitem__(provider_id, value),
    )
    monkeypatch.setattr(
        codex_auth,
        "_validate_token_payload",
        lambda _body, _previous: {
            "access_token": "new",
            "refresh_token": "refresh-1",
            "expires_at": 2,
            "account_id": "acct-1",
        },
    )

    async def _token_request(_data):
        return {}

    monkeypatch.setattr(codex_auth, "_token_request", _token_request)

    @asynccontextmanager
    async def _guard(_provider_id):
        yield

    monkeypatch.setattr(codex_auth, "provider_oauth_write_guard", _guard)

    token, account = asyncio.run(codex_auth.resolve_access("provider-20", force_refresh = True))
    assert (token, account) == ("new", "acct-1")
    assert stored["provider-20"]["catalog_account_id"] == "acct-1"


def test_recording_the_proof_never_overwrites_newer_credentials(monkeypatch):
    """The write is guarded and re-reads inside it, so a rotation cannot be undone."""
    stored = {
        "access_token": "rotated",
        "refresh_token": "refresh-2",
        "expires_at": 9,
        "account_id": "acct-1",
    }
    saved = []

    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: dict(stored))
    monkeypatch.setattr(codex_auth, "save_oauth_bundle", lambda _pid, value: saved.append(value))

    @asynccontextmanager
    async def _guard(_provider_id):
        yield

    monkeypatch.setattr(codex_auth, "provider_oauth_write_guard", _guard)

    asyncio.run(codex_auth.remember_catalog_account("provider-21", "acct-1"))
    assert len(saved) == 1
    # The tokens written back are the ones read inside the guard, not any older copy.
    assert saved[0]["access_token"] == "rotated"
    assert saved[0]["refresh_token"] == "refresh-2"
    assert saved[0]["catalog_account_id"] == "acct-1"


def test_a_second_catalog_401_is_recorded_on_the_connection(monkeypatch):
    """Raising alone leaves auth_status saying connected, so nothing offers Reconnect."""
    import httpx

    forget_subscription_models("provider-22")
    marked = []

    class AlwaysRejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(401, json = {"detail": "expired"})

        async def aclose(self):
            return None

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "fresh-token", "acct-1"

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: AlwaysRejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(
        codex_auth,
        "mark_reauthorization_required",
        lambda provider_id, expected_access_token = None: marked.append(
            (provider_id, expected_access_token)
        ),
    )
    try:
        with pytest.raises(CodexReauthorizationError):
            asyncio.run(list_subscription_models("provider-22", "stale", "acct-1"))
        # Recorded against the refreshed token, not the stale one it started with.
        assert marked == [("provider-22", "fresh-token")]
    finally:
        forget_subscription_models("provider-22")


def test_a_catalog_with_nothing_offerable_is_not_committed(monkeypatch):
    """The route answers such a catalog with the seed, so the two must not disagree.

    Committing it would leave the picker offering seeds while validation and chat read
    the same empty catalog and refuse them.
    """
    fake = _models_response({"models": [{"slug": "codex-auto-review", "visibility": "hide"}]})
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-23")
    try:
        models = asyncio.run(list_subscription_models("provider-23", "token", "acct-1"))
        assert [model["id"] for model in models] == ["codex-auto-review"]
        assert codex_client.subscription_catalog_known("provider-23") is False
    finally:
        forget_subscription_models("provider-23")


def test_a_cold_worker_rejects_credentials_for_another_account(monkeypatch):
    """With no catalog here, the catalog comparison has no opinion to offer.

    The row was authorized from the persisted proof against the account the gate read, so
    that is what the resolved credentials have to match.
    """
    saved = "gpt-5.7-nova"
    forget_subscription_models("codex-1")
    reads = []

    def _bundle(_pid):
        reads.append(1)
        # The gate reads acct-a and authorizes off the proof; a rebind lands before
        # resolve_access answers, which then returns acct-b.
        return {"account_id": "acct-a", "catalog_account_id": "acct-a"}

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "token-b", "acct-b"

    monkeypatch.setattr(codex_auth, "load_oauth_bundle", _bundle)
    try:
        refused = _codex_chat_gate(monkeypatch, saved, resolve = _resolve, saved_models = [saved])
        assert refused.status_code == 400
        assert "Choose a curated Codex model." in str(refused.detail)
        # The connection is left needing a fresh catalog before anything is trusted again.
        assert codex_client.subscription_catalog_stale("codex-1") is True
    finally:
        forget_subscription_models("codex-1")


def test_the_reauthorization_marker_is_written_under_the_guard(monkeypatch):
    """Read and write have to be one step, or a rotation in between is written back over.

    The streaming error path already takes this guard; the catalog path is the same kind
    of write and needs the same protection.
    """
    import httpx

    forget_subscription_models("provider-24")
    order = []

    class AlwaysRejecting:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            return httpx.Response(401, json = {"detail": "expired"})

        async def aclose(self):
            return None

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "fresh-token", "acct-1"

    @asynccontextmanager
    async def _guard(provider_id):
        order.append(("enter", provider_id))
        try:
            yield
        finally:
            order.append(("exit", provider_id))

    monkeypatch.setattr(codex_client, "_create_http_client", lambda: AlwaysRejecting())
    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_auth, "provider_oauth_write_guard", _guard)
    monkeypatch.setattr(
        codex_auth,
        "mark_reauthorization_required",
        lambda provider_id, expected_access_token = None: order.append(("mark", provider_id)),
    )
    try:
        with pytest.raises(CodexReauthorizationError):
            asyncio.run(list_subscription_models("provider-24", "stale", "acct-1"))
        assert order == [
            ("enter", "provider-24"),
            ("mark", "provider-24"),
            ("exit", "provider-24"),
        ]
    finally:
        forget_subscription_models("provider-24")


def test_one_malformed_entry_does_not_cost_the_whole_catalog(monkeypatch):
    """A scalar where a list belongs must drop that entry, not the account's plan.

    ``supported_reasoning_levels`` was the one field guarded with ``or []``, which only
    covers a falsy value. A scalar raised TypeError out of list_subscription_models, and
    every caller turns that into "no catalog": the picker falls back to the curated seed
    and the chat gate stops recognising slugs it had just been offering.
    """
    fake = _models_response(
        {
            "models": [
                {"slug": "gpt-5.4", "visibility": "list", "supported_reasoning_levels": 5},
                {"slug": "gpt-5.5", "visibility": "list", "supported_reasoning_levels": "low"},
                {
                    "slug": "gpt-5.6-sol",
                    "visibility": "list",
                    "supported_reasoning_levels": [{"effort": "high"}],
                },
            ]
        }
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-malformed")
    try:
        models = asyncio.run(list_subscription_models("provider-malformed", "token", "acct-1"))
        assert [model["id"] for model in models] == ["gpt-5.4", "gpt-5.5", "gpt-5.6-sol"]
        assert [model["reasoning_efforts"] for model in models] == [[], [], ["high"]]
        assert offered_subscription_model_ids("provider-malformed") == {
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.6-sol",
        }
    finally:
        forget_subscription_models("provider-malformed")


def test_a_repeated_slug_cannot_be_offered_and_refused_at_once(monkeypatch):
    """The offered list and the by-id map must describe a duplicate the same way.

    The list keeps every entry while the map keeps the last, so a slug listed once and
    hidden once was reported to the picker as offered and recorded for the chat gate as
    hidden. The picker offered it and every send refused it.
    """
    fake = _models_response(
        {
            "models": [
                {"slug": "gpt-5.4", "visibility": "list", "display_name": "First"},
                {"slug": "gpt-5.4", "visibility": "hide", "display_name": "Second"},
            ]
        }
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-duplicate")
    try:
        models = asyncio.run(list_subscription_models("provider-duplicate", "token", "acct-1"))
        assert [model["id"] for model in models] == ["gpt-5.4"]
        assert models[0]["display_name"] == "First"
        assert models[0]["listed"] is True
        offered = [model["id"] for model in models if model["listed"]]
        assert set(offered) == offered_subscription_model_ids("provider-duplicate")
    finally:
        forget_subscription_models("provider-duplicate")


def test_a_boolean_context_window_is_not_reported_as_a_length(monkeypatch):
    """bool is a subclass of int, so `true` would reach the picker as a context length."""
    fake = _models_response(
        {"models": [{"slug": "gpt-5.4", "visibility": "list", "context_window": True}]}
    )
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: fake)
    forget_subscription_models("provider-boolctx")
    try:
        models = asyncio.run(list_subscription_models("provider-boolctx", "token", "acct-1"))
        assert models[0]["context_length"] is None
    finally:
        forget_subscription_models("provider-boolctx")


def _gated_models_client(gate, slug):
    """A models endpoint whose response the test releases, not the network."""
    import httpx

    class Gated:
        async def get(
            self,
            _url,
            headers = None,
            params = None,
        ):
            await gate.wait()
            return httpx.Response(200, json = {"models": [{"slug": slug, "visibility": "list"}]})

        async def aclose(self):
            return None

    return Gated()


def test_a_disconnect_mid_read_retires_that_read_and_releases_its_ticket(monkeypatch):
    """Dropping the ticket must still retire the read that was holding it.

    forget_subscription_models releases the entry rather than writing a larger number
    over it, so an outstanding read finds nothing there instead of finding a mismatch.
    Either way it must answer its own caller without reinstating the catalog that was
    just dropped.
    """
    gate = asyncio.Event()
    monkeypatch.setattr(
        codex_client, "_create_http_client", lambda: _gated_models_client(gate, "gpt-5.4")
    )

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "token", "acct-1"

    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-1"})
    forget_subscription_models("provider-disconnect-race")

    async def run():
        task = asyncio.create_task(
            list_subscription_models("provider-disconnect-race", "token", "acct-1")
        )
        await asyncio.sleep(0)
        forget_subscription_models("provider-disconnect-race")
        assert "provider-disconnect-race" not in codex_client._catalog_requests
        gate.set()
        return await task

    try:
        models = asyncio.run(run())
        assert [model["id"] for model in models] == ["gpt-5.4"]
        assert codex_client.subscription_catalog_known("provider-disconnect-race") is False
        assert offered_subscription_model_ids("provider-disconnect-race") == set()
    finally:
        forget_subscription_models("provider-disconnect-race")


def test_a_read_started_after_a_release_cannot_be_matched_by_the_older_one(monkeypatch):
    """Why the ticket counter is shared by every connection rather than per connection.

    Releasing the entry and then counting up from it again would hand the next read the
    same number the outstanding one is holding, and that stale read would then commit
    its catalog over the newer one. Drawing from a counter that never reissues a value
    is what makes releasing the entry safe.
    """
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    clients = [
        _gated_models_client(first_gate, "old-slug"),
        _gated_models_client(second_gate, "new-slug"),
    ]
    monkeypatch.setattr(codex_client, "_create_http_client", lambda: clients.pop(0))

    async def _resolve(
        _provider_id,
        force_refresh = False,
        expected_access_token = None,
    ):
        return "token", "acct-1"

    monkeypatch.setattr(codex_auth, "resolve_access", _resolve)
    monkeypatch.setattr(codex_auth, "load_oauth_bundle", lambda _pid: {"account_id": "acct-1"})
    forget_subscription_models("provider-release-race")

    async def run():
        first = asyncio.create_task(
            list_subscription_models("provider-release-race", "token", "acct-1")
        )
        await asyncio.sleep(0)
        forget_subscription_models("provider-release-race")
        second = asyncio.create_task(
            list_subscription_models("provider-release-race", "token", "acct-1")
        )
        await asyncio.sleep(0)
        second_gate.set()
        await second
        first_gate.set()
        await first

    try:
        asyncio.run(run())
        assert offered_subscription_model_ids("provider-release-race") == {"new-slug"}
    finally:
        forget_subscription_models("provider-release-race")


def test_quota_metadata_marks_a_terminal_refusal():
    """A 429 is both "wait a moment" and "your plan is spent"; only the flag tells them apart."""
    from core.inference import openai_codex_client as codex_client

    response = httpx.Response(
        429,
        headers = {"retry-after": "30"},
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )
    throttled = codex_client._quota_metadata(response)
    assert throttled == {"retry_after": "30"}
    assert codex_client._quota_metadata(response, terminal = True) == {
        "retry_after": "30",
        "terminal": True,
    }


def test_quota_metadata_falls_back_to_retry_after_ms():
    """The client's own backoff already reads retry-after-ms; dropping it here left the delay
    honoured on this side of the proxy and guessed at on the other."""
    from core.inference import openai_codex_client as codex_client

    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    ms_only = httpx.Response(429, headers = {"retry-after-ms": "30000"}, request = request)
    assert codex_client._quota_metadata(ms_only) == {"retry_after": "30.0"}
    # Seconds win when both are present, and neither means no invented delay.
    both = httpx.Response(
        429,
        headers = {"retry-after": "45", "retry-after-ms": "30000"},
        request = request,
    )
    assert codex_client._quota_metadata(both) == {"retry_after": "45"}
    assert codex_client._quota_metadata(httpx.Response(429, request = request)) == {}


def _quota_error_for(monkeypatch, body):
    """Drive the real send/classify loop against one 429 and return the CodexQuotaError."""
    from core.inference import openai_codex_client as codex_client

    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")

    async def _no_pause(delay, cancel_event):
        return None

    monkeypatch.setattr(codex_client, "_retry_pause", _no_pause)

    class _Ctx:
        async def __aenter__(self):
            return httpx.Response(429, json = body, request = request)

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(codex_client, "_stream_response", lambda *a, **k: _Ctx())

    async def _run():
        async with codex_client._validated_stream_response(
            None,
            url = "https://chatgpt.com/backend-api/codex/responses",
            headers = {"Authorization": "Bearer t"},
            body = {},
            cancel_event = None,
            refresh_access = None,
        ):
            pass

    with pytest.raises(codex_client.CodexQuotaError) as raised:
        asyncio.run(_run())
    return raised.value


def test_a_terminal_code_is_recognised_behind_a_generic_message(monkeypatch):
    """_upstream_error_detail prefers the display message, so a terminal code arrived hidden
    behind a "slow down" sentence and the refusal was retried as if waiting could clear it."""
    hidden = _quota_error_for(
        monkeypatch,
        {"error": {"message": "Too many requests.", "code": "insufficient_quota"}},
    )
    assert hidden.metadata.get("terminal") is True


def test_a_transient_code_behind_a_generic_message_stays_retryable(monkeypatch):
    """The tightened classification must not start refusing throttles that a wait does clear."""
    throttled = _quota_error_for(
        monkeypatch,
        {"error": {"message": "Too many requests.", "code": "rate_limit_exceeded"}},
    )
    assert "terminal" not in throttled.metadata


def test_upstream_error_code_survives_a_body_that_cannot_be_read():
    """Parsing a body whose read failed raises StreamError, which is not an HTTPError. The
    classification runs after a failed read, so an uncaught one would replace the quota error
    the caller is supposed to see."""
    from core.inference import openai_codex_client as codex_client

    class _FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadTimeout("body read died mid-flight")
            yield b""

    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")

    async def _both():
        response = httpx.Response(429, stream = _FailingStream(), request = request)
        # The order the send loop uses: the message first, then the code off the same response.
        return (
            await codex_client._upstream_error_detail(response),
            await codex_client._upstream_error_code(response),
        )

    assert asyncio.run(_both()) == (None, None)


def test_upstream_error_code_reads_code_then_type():
    from core.inference import openai_codex_client as codex_client

    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")

    def _code_of(body):
        return asyncio.run(
            codex_client._upstream_error_code(httpx.Response(429, json = body, request = request))
        )

    assert _code_of({"error": {"message": "m", "code": "insufficient_quota"}}) == (
        "insufficient_quota"
    )
    assert _code_of({"error": {"message": "m", "type": "rate_limit_error"}}) == "rate_limit_error"
    assert _code_of({"detail": "nope"}) is None
    assert _code_of({"error": "flat string"}) is None


def test_terminal_quota_detail_is_recognised():
    from core.inference import openai_codex_client as codex_client
    assert codex_client._is_terminal_quota("You exceeded your current quota (insufficient_quota)")
    assert not codex_client._is_terminal_quota("Rate limit reached, try again in 30s")

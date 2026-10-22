# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Connections checks: rejected keys, edited base URLs, readable errors."""

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import HTTPException

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import validate_provider_base_url
from models.providers import ProviderModelsRequest, ProviderTestRequest
from routes import providers as providers_route
from storage import providers_db


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _Sent(list):
    state: dict


@pytest.fixture
def mock_http(monkeypatch):
    """Route the shared client through a handler; returns the list of sent requests."""
    sent = _Sent()
    state = {"handler": lambda request: httpx.Response(200, json = {"data": []})}

    def handle(request):
        sent.append(request)
        return state["handler"](request)

    monkeypatch.setattr(
        ep, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handle))
    )
    monkeypatch.setattr(providers_route, "validate_provider_base_url", lambda url: url.rstrip("/"))
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-bad"
    )
    sent.state = state
    return sent


def _test(payload: ProviderTestRequest):
    return _run(providers_route.test_provider(payload, "tester", False))


@pytest.mark.parametrize(
    "provider_type,check_url",
    [
        ("openrouter", "https://openrouter.ai/api/v1/key"),
        ("huggingface", "https://huggingface.co/api/whoami-v2"),
    ],
)
def test_rejected_key_fails_even_though_models_answers(mock_http, provider_type, check_url):
    def handle(request):
        if str(request.url) == check_url:
            return httpx.Response(401, json = {"error": "invalid"})
        return httpx.Response(200, json = {"data": [{"id": "m"}]})

    mock_http.state["handler"] = handle
    result = _test(ProviderTestRequest(provider_type = provider_type))
    assert not result.success
    assert "rejected the API key" in result.message
    assert any(str(r.url) == check_url for r in mock_http)
    key_request = next(r for r in mock_http if str(r.url) == check_url)
    assert key_request.headers["authorization"] == "Bearer sk-bad"


def test_key_check_network_error_keeps_the_models_result(mock_http):
    def handle(request):
        if request.url.path.endswith("/key"):
            raise httpx.ConnectError("down", request = request)
        return httpx.Response(200, json = {"data": [{"id": "m"}]})

    mock_http.state["handler"] = handle
    assert _test(ProviderTestRequest(provider_type = "openrouter")).success


def test_huggingface_key_check_stays_on_the_hub_host():
    client = ExternalProviderClient("huggingface", "https://my-proxy.example/v1", "hf_x")
    assert _run(client.api_key_rejected()) is False


def test_anthropic_lists_past_the_default_page(mock_http):
    client = ExternalProviderClient("anthropic", "https://api.anthropic.com/v1", "sk-ant")
    _run(client.list_models())
    assert mock_http[-1].url.params["limit"] == "1000"
    _run(ExternalProviderClient("openai", "https://api.openai.com/v1", "sk").list_models())
    assert "limit" not in mock_http[-1].url.params


def test_gemini_lists_past_the_default_page(mock_http):
    base = "https://generativelanguage.googleapis.com/v1beta"
    _run(ExternalProviderClient("gemini", base, "AIza").list_models())
    assert mock_http[-1].url.params["pageSize"] == "1000"


def test_gemini_through_a_proxy_gets_no_native_page_size(mock_http):
    _run(ExternalProviderClient("gemini", "https://litellm.example/v1", "sk").list_models())
    assert "pageSize" not in mock_http[-1].url.params


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "The server rejected the API key."),
        (403, "The server rejected the API key."),
        (404, "No /models endpoint at this URL (check that it ends in /v1)."),
        (405, "No /models endpoint at this URL (check that it ends in /v1)."),
        (500, "Failed to list models from custom."),
    ],
)
def test_list_models_errors_say_what_went_wrong(mock_http, status, expected):
    mock_http.state["handler"] = lambda request: httpx.Response(status)
    payload = ProviderModelsRequest(provider_type = "custom", base_url = "http://gw.example/v1")
    with pytest.raises(HTTPException) as caught:
        _run(providers_route.list_provider_models(payload, "tester", False))
    assert caught.value.detail == expected


def test_unreachable_server_is_named_with_a_docker_hint(mock_http):
    def refuse(request):
        raise httpx.ConnectError("All connection attempts failed", request = request)

    mock_http.state["handler"] = refuse
    payload = ProviderModelsRequest(provider_type = "custom", base_url = "http://localhost:11434/v1")
    with pytest.raises(HTTPException) as caught:
        _run(providers_route.list_provider_models(payload, "tester", False))
    assert caught.value.detail == (
        "Couldn't reach localhost:11434. Is the server running? "
        "If Studio runs in Docker, try host.docker.internal."
    )
    result = _test(ProviderTestRequest(provider_type = "custom", base_url = "http://10.0.0.9:8000/v1"))
    assert not result.success
    assert "Couldn't reach 10.0.0.9:8000. Is the server running?" in result.message
    assert "Docker" not in result.message
    assert "All connection attempts failed" not in result.message


def test_unlisted_model_still_connects_with_a_hint(mock_http):
    mock_http.state["handler"] = lambda request: httpx.Response(
        200, json = {"data": [{"id": "Qwen3-8B"}, {"id": "llama3"}]}
    )
    request = ProviderTestRequest(
        provider_type = "custom", base_url = "http://gw.example/v1", model_id = "qwen3-8b"
    )
    result = _test(request)
    assert result.success
    assert result.message == (
        "Connected, but model 'qwen3-8b' is not listed by this server. Did you mean 'Qwen3-8B'?"
    )
    result = _test(request.model_copy(update = {"model_id": "gpt-x"}))
    assert result.success
    assert result.message.endswith("Available: Qwen3-8B, llama3")
    result = _test(request.model_copy(update = {"model_id": "llama3"}))
    assert result.message == "Connected successfully. Found 2 model(s)."


def test_basic_auth_credentials_are_not_echoed(mock_http, monkeypatch):
    mock_http.state["handler"] = lambda request: httpx.Response(500)
    logged = []
    monkeypatch.setattr(providers_route.logger, "error", lambda *a, **k: logged.append((a, k)))
    result = _test(
        ProviderTestRequest(provider_type = "openai", base_url = "http://user:hunter2@gw.example/v1")
    )
    assert not result.success
    assert "hunter2" not in result.message and "gw.example" in result.message
    assert logged and "hunter2" not in repr(logged)


def test_responses_probe_stops_at_the_total_budget(monkeypatch):
    monkeypatch.setattr(providers_route, "_RESPONSES_PROBE_BUDGET_SECONDS", 0.3)
    monkeypatch.setattr(providers_route, "_PROVIDER_CONNECTIVITY_TIMEOUT_SECONDS", 0.2)
    tried = []

    class _SlowClient:
        async def list_models(self):
            return [{"id": f"m{i}"} for i in range(5)]

        def stream_chat_completion(self, *, model, **_kwargs):
            tried.append(model)

            async def stream():
                await asyncio.sleep(10)
                yield ""

            return stream()

    started = time.monotonic()
    result = _run(
        providers_route._test_custom_provider_connectivity(_SlowClient(), "", "responses")
    )
    assert not result.success and "timed out" in result.message
    # Coarse timers (Windows) can leave a sliver of budget for one more short probe.
    assert tried[:2] == ["m0", "m1"] and len(tried) <= 3
    assert time.monotonic() - started < 1.0


def _saved_row(monkeypatch, tmp_path, *, has_key: bool):
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    providers_db.create_provider(
        id = "row", provider_type = "custom", display_name = "Local", base_url = "http://a.example/v1"
    )
    monkeypatch.setattr(
        providers_route.credential_secrets, "provider_api_key_saved", lambda _id: has_key
    )


def test_keyless_row_lists_models_from_the_edited_base_url(monkeypatch, tmp_path):
    _saved_row(monkeypatch, tmp_path, has_key = False)
    payload = ProviderModelsRequest(
        provider_type = "custom", provider_id = "row", base_url = "http://b.example/v1"
    )
    assert providers_route._bind_saved_provider_target(payload).base_url == "http://b.example/v1"
    unset = payload.model_copy(update = {"base_url": None})
    assert providers_route._bind_saved_provider_target(unset).base_url == "http://a.example/v1"


def test_a_saved_key_stays_pinned_to_the_saved_base_url(monkeypatch, tmp_path):
    _saved_row(monkeypatch, tmp_path, has_key = True)
    payload = ProviderModelsRequest(
        provider_type = "custom", provider_id = "row", base_url = "http://b.example/v1"
    )
    assert providers_route._bind_saved_provider_target(payload).base_url == "http://a.example/v1"


def test_base_url_validation_runs_off_the_event_loop(mock_http, monkeypatch):
    threads = []

    def record(url):
        threads.append(threading.current_thread())
        return url

    monkeypatch.setattr(providers_route, "validate_provider_base_url", record)
    _test(ProviderTestRequest(provider_type = "custom", base_url = "http://gw.example/v1"))
    payload = ProviderModelsRequest(provider_type = "custom", base_url = "http://gw.example/v1")
    _run(providers_route.list_provider_models(payload, "tester", False))
    assert len(threads) == 2
    assert all(thread is not threading.main_thread() for thread in threads)


@pytest.mark.parametrize(
    "pasted,base",
    [
        ("http://127.0.0.1:11434/v1/chat/completions", "http://127.0.0.1:11434/v1"),
        ("http://127.0.0.1:11434/v1/completions/", "http://127.0.0.1:11434/v1"),
        ("http://127.0.0.1:8000/v1/responses", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1/Models", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1/messages?x=1", "http://127.0.0.1:8000/v1?x=1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
    ],
)
def test_pasted_endpoint_paths_normalise_to_the_base(pasted, base):
    assert validate_provider_base_url(pasted) == base
    assert validate_provider_base_url(base) == base

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check connection and Load models report what the provider actually said."""

import asyncio
import json
import logging

import httpx
import pytest

from core.inference import external_provider
from loggers.handlers import filter_sensitive_data
from models.providers import ProviderModelsRequest, ProviderTestRequest
from routes import providers as providers_route


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _mock(monkeypatch, handler):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport = httpx.MockTransport(wrapped))
    monkeypatch.setattr(external_provider, "_client", lambda: client)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "bogus-key"
    )
    return seen


@pytest.mark.parametrize("provider_type", ["openrouter", "huggingface"])
def test_check_connection_rejects_a_key_the_public_catalog_never_checks(monkeypatch, provider_type):
    def handler(request):
        # Both hosts serve /models without a key.
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json = {"data": [{"id": "m"}]})
        return httpx.Response(401, json = {"error": {"message": "Invalid credentials"}})

    _mock(monkeypatch, handler)
    result = _run(
        providers_route.test_provider(
            ProviderTestRequest(provider_type = provider_type), "tester", False
        )
    )
    assert result.success is False
    assert "refused the API key (HTTP 401)" in result.message


def test_check_connection_skips_the_key_check_for_another_base_url(monkeypatch):
    seen = _mock(monkeypatch, lambda request: httpx.Response(200, json = {"data": [{"id": "m"}]}))
    result = _run(
        providers_route.test_provider(
            ProviderTestRequest(provider_type = "huggingface", base_url = "http://proxy.test/v1"),
            "tester",
            False,
        )
    )
    assert result.success is True
    assert all(request.url.host == "proxy.test" for request in seen)


def test_gemini_and_anthropic_listings_ask_for_a_full_page(monkeypatch):
    seen = _mock(monkeypatch, lambda request: httpx.Response(200, json = {"data": []}))
    for provider_type, base_url in (
        ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
        ("anthropic", "https://api.anthropic.com/v1"),
    ):
        client = external_provider.ExternalProviderClient(
            provider_type = provider_type, base_url = base_url, api_key = "k"
        )
        _run(client.list_models())
    assert seen[0].url.params.get("pageSize") == "1000"
    assert seen[1].url.params.get("limit") == "1000"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401, json = {}), "refused the API key (HTTP 401)"),
        (httpx.Response(500, json = {}), "answered HTTP 500"),
        (httpx.Response(200, text = "<html>login</html>"), "usually ends in /v1"),
    ],
)
def test_load_models_says_why_it_failed(monkeypatch, response, expected):
    _mock(monkeypatch, lambda request: response)
    with pytest.raises(Exception) as info:
        _run(
            providers_route.list_provider_models(
                ProviderModelsRequest(
                    provider_type = "custom", base_url = "http://bob:pw@gateway.test/v1"
                ),
                "tester",
                False,
            )
        )
    assert expected in info.value.detail
    assert "gateway.test" not in info.value.detail


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("refused"), "Couldn't reach the server"),
        (httpx.ReadTimeout(""), "did not answer in time"),
    ],
)
def test_custom_connection_test_names_transport_failures(monkeypatch, error, expected):
    def handler(request):
        raise error

    _mock(monkeypatch, handler)
    result = _run(
        providers_route.test_provider(
            ProviderTestRequest(provider_type = "custom", base_url = "http://gateway.test/v1"),
            "tester",
            False,
        )
    )
    assert result.success is False
    assert expected in result.message


def test_logs_redact_a_password_in_a_base_url():
    error = httpx.HTTPStatusError(
        "Client error '401 Unauthorized' for url 'http://bob:hunter2@host.test/v1/models'",
        request = httpx.Request("GET", "http://host.test"),
        response = httpx.Response(401),
    )
    event = filter_sensitive_data(
        logging.getLogger(),
        "error",
        {
            "event": f"Failed to list models from custom: {error}",
            "positional_args": ("custom", error),
        },
    )
    text = json.dumps(event, default = str)
    assert "hunter2" not in text and "bob" not in text
    assert "http://<redacted>@host.test" in text

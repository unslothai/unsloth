# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""xAI, NVIDIA NIM, Perplexity, Cohere, Vercel AI Gateway, Azure OpenAI, Amazon Bedrock and Vertex AI."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import (
    PROVIDER_REGISTRY,
    VERTEX_EXPRESS_BASE_URL,
    is_google_gemini_host,
    list_available_providers,
    provider_probes_with_chat,
)
from models.providers import ProviderModelsRequest, ProviderTestRequest
from routes import providers as providers_route


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def sent(monkeypatch):
    """Requests the shared client sends; `sent.handler` answers them."""

    class Sent(list):
        handler = staticmethod(lambda request: httpx.Response(200, json = {"data": []}))

    box = Sent()

    def handle(request):
        box.append(request)
        return box.handler(request)

    monkeypatch.setattr(
        ep, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handle))
    )
    monkeypatch.setattr(providers_route, "validate_provider_base_url", lambda url: url.rstrip("/"))
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "key-123"
    )
    return box


def _sse(*events):
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})


_ANSWER = {"choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}]}


def _chat_body(
    sent,
    provider_type,
    base_url = None,
    **kwargs,
):
    sent.handler = lambda request: _sse(_ANSWER)
    client = ExternalProviderClient(
        provider_type, base_url or PROVIDER_REGISTRY[provider_type]["base_url"], "key-123"
    )

    async def go():
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}], model = "m", **kwargs
        ):
            pass

    _run(go())
    request = sent[-1]
    return request, json.loads(request.content)


def test_the_new_keyed_providers_are_offered_and_the_endpoint_presets_stay_hidden():
    shown = {row["provider_type"] for row in list_available_providers()}
    assert {"xai", "nvidia", "perplexity", "cohere", "vercel"} <= shown
    assert not {"azure", "bedrock"} & shown
    hidden = {row["provider_type"] for row in list_available_providers(include_hidden = True)}
    assert {"azure", "bedrock"} <= hidden


def test_xai_sends_the_token_field_it_wants_and_never_turns_reasoning_off(sent):
    request, body = _chat_body(
        sent, "xai", max_tokens = 64, enable_thinking = False, reasoning_effort = "none"
    )
    assert str(request.url) == "https://api.x.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer key-123"
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
    assert "presence_penalty" not in body
    assert "reasoning_effort" not in body
    assert body["stream_options"] == {"include_usage": True}
    _, body = _chat_body(sent, "xai", reasoning_effort = "xhigh")
    assert body["reasoning_effort"] == "xhigh"


def test_nvidia_caps_temperature_at_one_and_omits_the_penalty(sent):
    request, body = _chat_body(sent, "nvidia", temperature = 1.6, max_tokens = 32)
    assert str(request.url) == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert body["temperature"] == 1.0
    assert body["max_tokens"] == 32
    assert "presence_penalty" not in body and "stream_options" not in body


def test_cohere_maps_every_effort_onto_the_two_it_accepts(sent):
    _, body = _chat_body(sent, "cohere", reasoning_effort = "low")
    assert body["reasoning_effort"] == "high"
    _, body = _chat_body(sent, "cohere", enable_thinking = False)
    assert body["reasoning_effort"] == "none"
    _, body = _chat_body(sent, "cohere")
    assert "reasoning_effort" not in body


def test_cohere_lists_chat_models_from_its_native_endpoint(sent):
    def handle(request):
        assert request.url.host == "api.cohere.com" and request.url.path == "/v1/models"
        assert request.url.params["endpoint"] == "chat"
        return httpx.Response(
            200,
            json = {
                "models": [
                    {"name": "command-a-03-2025", "context_length": 256000},
                    {"name": "command-old", "is_deprecated": True},
                    {"name": "my-tune", "finetuned": True},
                ]
            },
        )

    sent.handler = handle
    models = _run(
        providers_route.list_provider_models(
            ProviderModelsRequest(provider_type = "cohere"), "tester", False
        )
    )
    assert [(m.id, m.context_length) for m in models] == [("command-a-03-2025", 256000)]


def test_vercel_keeps_language_models_and_checks_the_key_where_auth_is_required(sent):
    def handle(request):
        if request.url.path.endswith("/credits"):
            return httpx.Response(401, json = {"error": {"message": "bad"}})
        return httpx.Response(
            200,
            json = {
                "data": [
                    {"id": "openai/gpt-6", "type": "language", "context_window": 400000},
                    {"id": "openai/text-embedding-3-small", "type": "embedding"},
                    {"id": "x/judge", "type": "evaluation"},
                ]
            },
        )

    sent.handler = handle
    models = _run(
        providers_route.list_provider_models(
            ProviderModelsRequest(provider_type = "vercel"), "tester", False
        )
    )
    assert [(m.id, m.context_length) for m in models] == [("openai/gpt-6", 400000)]
    # The list is public, so only /credits can tell a bad key.
    result = _run(
        providers_route.test_provider(ProviderTestRequest(provider_type = "vercel"), "tester", False)
    )
    assert not result.success and result.error_kind == "auth"


def test_xai_and_nvidia_lists_drop_models_that_do_not_chat(sent):
    sent.handler = lambda request: httpx.Response(
        200,
        json = {
            "data": [
                {"id": "grok-4.7"},
                {"id": "grok-imagine-image-1"},
                {"id": "grok-voice-1"},
                {"id": "nvidia/nemotron-3-super-120b-a12b"},
                {"id": "nvidia/nv-embedqa-mistral-7b-v2"},
                {"id": "meta/llama-guard-4-12b"},
                {"id": "nvidia/nemotron-parse"},
            ]
        },
    )

    def ids(provider_type):
        return [
            m.id
            for m in _run(
                providers_route.list_provider_models(
                    ProviderModelsRequest(provider_type = provider_type), "tester", False
                )
            )
        ]

    assert "grok-4.7" in ids("xai")
    assert not [i for i in ids("xai") if "imagine" in i or "voice" in i]
    assert [i for i in ids("nvidia") if i.startswith(("nvidia/", "meta/"))] == [
        "nvidia/nemotron-3-super-120b-a12b"
    ]


@pytest.mark.parametrize(
    "provider_type,base_url",
    [
        ("azure", "https://res.openai.azure.com/openai/v1"),
        ("bedrock", "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"),
        ("perplexity", "https://api.perplexity.ai"),
    ],
)
def test_providers_without_a_model_list_are_checked_with_one_short_chat(
    sent, provider_type, base_url
):
    assert provider_probes_with_chat(provider_type, base_url)
    sent.handler = lambda request: _sse(_ANSWER)
    payload = ProviderTestRequest(
        provider_type = provider_type, base_url = base_url, model_id = "my-model"
    )
    result = _run(providers_route.test_provider(payload, "tester", False))
    assert result.success and "my-model answered" in result.message
    assert [r.url.path.rsplit("/", 2)[-2:] for r in sent] == [["chat", "completions"]]
    assert not any(r.url.path.endswith("/models") for r in sent)
    # Without a model there is nothing to send.
    result = _run(
        providers_route.test_provider(
            payload.model_copy(update = {"model_id": None}), "tester", False
        )
    )
    assert not result.success and "add a model ID" in result.message


@pytest.mark.parametrize(
    "status,kind",
    [(401, "auth"), (403, "auth"), (402, "billing"), (429, "rate_limit"), (404, "other")],
)
def test_a_chat_check_names_the_cause_from_the_upstream_status(sent, status, kind):
    sent.handler = lambda request: httpx.Response(status, json = {"error": {"message": "no"}})
    result = _run(
        providers_route.test_provider(
            ProviderTestRequest(
                provider_type = "azure",
                base_url = "https://res.openai.azure.com/openai/v1",
                model_id = "dep",
            ),
            "tester",
            False,
        )
    )
    assert not result.success and result.error_kind == kind


def test_azure_sends_max_completion_tokens_to_the_deployment(sent):
    request, body = _chat_body(
        sent,
        "azure",
        "https://res.openai.azure.com/openai/v1",
        max_tokens = 128,
        temperature = 0.7,
        top_p = 0.9,
        presence_penalty = 0.5,
    )
    assert str(request.url) == "https://res.openai.azure.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer key-123"
    assert body["max_completion_tokens"] == 128 and "max_tokens" not in body
    assert not {"temperature", "top_p", "presence_penalty"} & body.keys()


def test_the_azure_connection_check_sends_no_sampling_fields(sent):
    sent.handler = lambda request: _sse(_ANSWER)
    payload = ProviderTestRequest(
        provider_type = "azure", base_url = "https://res.openai.azure.com/openai/v1", model_id = "o3"
    )
    assert _run(providers_route.test_provider(payload, "tester", False)).success
    assert not {"temperature", "top_p", "presence_penalty"} & json.loads(sent[-1].content).keys()


def test_a_vertex_openapi_endpoint_stays_openai_compatible(sent):
    base = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
    client = ExternalProviderClient("gemini", base, "key-123")
    assert client._is_openai_compatible()
    assert client._auth_headers()["Authorization"] == "Bearer key-123"
    request, body = _chat_body(sent, "gemini", base)
    assert str(request.url) == base + "/chat/completions"
    assert body["model"] == "m"


def test_vertex_is_the_native_gemini_api_on_another_host(sent):
    assert is_google_gemini_host("aiplatform.googleapis.com")
    assert is_google_gemini_host("us-central1-aiplatform.googleapis.com")
    assert not is_google_gemini_host("aiplatform.googleapis.com.evil.example")
    assert provider_probes_with_chat("gemini", VERTEX_EXPRESS_BASE_URL)
    assert not provider_probes_with_chat("gemini", PROVIDER_REGISTRY["gemini"]["base_url"])
    client = ExternalProviderClient("gemini", VERTEX_EXPRESS_BASE_URL, "key-123")
    assert not client._is_openai_compatible()
    headers = client._auth_headers()
    assert headers["x-goog-api-key"] == "key-123" and "Authorization" not in headers
    sent.handler = lambda request: _sse(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "pong"}]}}]}
    )

    async def go():
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}], model = "gemini-2.5-flash"
        ):
            pass

    _run(go())
    assert str(sent[-1].url) == (
        f"{VERTEX_EXPRESS_BASE_URL}/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )


@pytest.mark.parametrize(
    "error,kind",
    [(httpx.ConnectError("refused"), "unreachable"), (httpx.ReadTimeout("slow"), "timeout")],
)
def test_a_chat_check_names_a_transport_failure(sent, error, kind):
    def fail(request):
        raise error

    sent.handler = fail
    result = _run(
        providers_route.test_provider(
            ProviderTestRequest(
                provider_type = "azure",
                base_url = "https://res.openai.azure.com/openai/v1",
                model_id = "dep",
            ),
            "tester",
            False,
        )
    )
    assert not result.success and result.error_kind == kind

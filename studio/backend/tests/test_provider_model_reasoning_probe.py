# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat-template reasoning probing + forwarding for connected llama.cpp servers.

The local GGUF/safetensors load paths classify reasoning controls from the
model's Jinja chat template; a connected llama.cpp server exposes that same
template (``GET /props``), so ``list_provider_models`` probes it and classifies
it with the same ``detect_reasoning_flags`` helper, and the chat-completions
proxy forwards the resolved ``enable_thinking`` / ``reasoning_effort`` back as
``chat_template_kwargs`` so the server's template renders them.

Stubbed httpx; no subprocess, GPU, or network. Cross-platform.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import httpx  # noqa: E402

import core.inference.external_provider as ep_mod  # noqa: E402
from core.inference.external_provider import ExternalProviderClient  # noqa: E402
from routes.providers import _probe_model_reasoning  # noqa: E402


# enable_thinking on/off gate PLUS a reasoning_effort ladder -- the Qwen3.8 shape.
QWEN38_TEMPLATE = (
    "{%- if enable_thinking is defined and enable_thinking %}"
    "{%- if reasoning_effort == 'low' %}{{- 'LOW' -}}"
    "{%- elif reasoning_effort == 'medium' %}{{- 'MEDIUM' -}}"
    "{%- elif reasoning_effort == 'xhigh' %}{{- 'XHIGH' -}}"
    "{%- endif %}{%- endif %}"
    "{%- for message in messages %}{{- message.content }}{%- endfor %}"
)


class _FakeResponse:
    def __init__(
        self,
        status_code = 200,
        body = None,
    ):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


class _FakeAsyncClient:
    def __init__(self):
        self.get_calls = []
        self.post_calls = []
        self.get_response: object = _FakeResponse(200, {})
        self.post_response: object = _FakeResponse(200, {})

    async def get(
        self,
        url,
        headers = None,
        timeout = None,
    ):
        self.get_calls.append((url, headers, timeout))
        if isinstance(self.get_response, Exception):
            raise self.get_response
        return self.get_response

    async def post(
        self,
        url,
        headers = None,
        json = None,
        timeout = None,
    ):
        self.post_calls.append((url, headers, json, timeout))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response


def _make_client(provider_type: str, base_url: str) -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = provider_type,
        base_url = base_url,
        api_key = "",
        timeout = 15.0,
    )


def _run(coro):
    return asyncio.run(coro)


# ── ExternalProviderClient.probe_chat_template ───────────────────────


def test_llama_cpp_probe_reads_chat_template_via_model_query(monkeypatch):
    fake = _FakeAsyncClient()
    fake.get_response = _FakeResponse(200, {"chat_template": QWEN38_TEMPLATE})
    monkeypatch.setattr(ep_mod, "_http_client", fake)

    template = _run(
        _make_client("llama_cpp", "http://127.0.0.1:8080/v1").probe_chat_template("any")
    )

    assert template == QWEN38_TEMPLATE
    # Router mode names each served model: /props is addressed per model, and it
    # lives at the server root, not under the OpenAI-compat /v1 prefix.
    assert fake.get_calls[0][0] == "http://127.0.0.1:8080/props?model=any"


def test_llama_cpp_probe_falls_back_to_bare_props(monkeypatch):
    # A single-model server has no per-model /props entry, so the ?model= probe
    # returns no template and the bare endpoint answers.
    fake = _FakeAsyncClient()

    def dispatch(
        url,
        headers = None,
        timeout = None,
    ):
        fake.get_calls.append((url, headers, timeout))
        if "model=" in url:
            return _FakeResponse(200, {"chat_template": ""})
        return _FakeResponse(200, {"chat_template": QWEN38_TEMPLATE})

    monkeypatch.setattr(ep_mod, "_http_client", fake)
    monkeypatch.setattr(fake, "get", dispatch)

    template = _run(
        _make_client("llama_cpp", "http://127.0.0.1:8080/v1").probe_chat_template("only-model")
    )

    assert template == QWEN38_TEMPLATE
    assert [url for url, _, _ in fake.get_calls] == [
        "http://127.0.0.1:8080/props?model=only-model",
        "http://127.0.0.1:8080/props",
    ]


def test_llama_cpp_probe_caches_per_model_id(monkeypatch):
    fake = _FakeAsyncClient()
    fake.get_response = _FakeResponse(200, {"chat_template": QWEN38_TEMPLATE})
    monkeypatch.setattr(ep_mod, "_http_client", fake)

    client = _make_client("llama_cpp", "http://127.0.0.1:8080/v1")
    assert _run(client.probe_chat_template("model-a")) == QWEN38_TEMPLATE
    assert _run(client.probe_chat_template("model-a")) == QWEN38_TEMPLATE
    assert _run(client.probe_chat_template("model-b")) == QWEN38_TEMPLATE
    # One /props read per distinct model id, cached per id.
    assert [url for url, _, _ in fake.get_calls] == [
        "http://127.0.0.1:8080/props?model=model-a",
        "http://127.0.0.1:8080/props?model=model-b",
    ]


def test_unsupported_provider_returns_none_without_http(monkeypatch):
    fake = _FakeAsyncClient()
    monkeypatch.setattr(ep_mod, "_http_client", fake)

    assert _run(_make_client("custom", "http://127.0.0.1:8080/v1").probe_chat_template("x")) is None
    assert (
        _run(_make_client("ollama", "http://127.0.0.1:11434/v1").probe_chat_template("x")) is None
    )
    assert fake.get_calls == []
    assert fake.post_calls == []


def test_probe_failure_returns_none(monkeypatch):
    fake = _FakeAsyncClient()
    fake.get_response = httpx.ConnectError("down")
    monkeypatch.setattr(ep_mod, "_http_client", fake)

    assert (
        _run(_make_client("llama_cpp", "http://127.0.0.1:8080/v1").probe_chat_template("x")) is None
    )


def test_probe_missing_template_key_returns_none(monkeypatch):
    fake = _FakeAsyncClient()
    fake.get_response = _FakeResponse(200, {"default_generation_settings": {}})
    monkeypatch.setattr(ep_mod, "_http_client", fake)

    assert (
        _run(_make_client("llama_cpp", "http://127.0.0.1:8080/v1").probe_chat_template("x")) is None
    )


# ── _probe_model_reasoning classification ────────────────────────────


class _ProbeClient:
    def __init__(self, template):
        self._template = template

    async def probe_chat_template(self, model_id):
        return self._template


def test_probe_model_reasoning_classifies_effort_levels():
    out = _run(
        _probe_model_reasoning(
            _ProbeClient(QWEN38_TEMPLATE),
            "llama_cpp",
            [{"id": "qwen3.8-14b"}],
        )
    )

    assert "qwen3.8-14b" in out
    reasoning = out["qwen3.8-14b"]
    assert reasoning.supports_reasoning is True
    assert reasoning.reasoning_style == "enable_thinking_effort"
    assert reasoning.reasoning_effort_levels == ["low", "medium", "xhigh"]
    assert reasoning.reasoning_always_on is False


def test_probe_model_reasoning_skips_unsupported_provider():
    out = _run(_probe_model_reasoning(_ProbeClient(QWEN38_TEMPLATE), "ollama", [{"id": "x"}]))
    assert out == {}


def test_probe_model_reasoning_skips_when_probe_returns_none():
    out = _run(_probe_model_reasoning(_ProbeClient(None), "llama_cpp", [{"id": "x"}]))
    assert out == {}


def test_probe_model_reasoning_skips_when_probe_raises():
    class _BoomClient:
        async def probe_chat_template(self, model_id):
            raise RuntimeError("server gone")

    out = _run(_probe_model_reasoning(_BoomClient(), "llama_cpp", [{"id": "x"}]))
    assert out == {}


def test_probe_model_reasoning_requires_probe_method():
    class _NoProbeClient:
        pass

    out = _run(_probe_model_reasoning(_NoProbeClient(), "llama_cpp", [{"id": "x"}]))
    assert out == {}


# ── routes: listing stays cheap, on-demand probe is explicit ─────────


def test_list_provider_models_does_not_probe_reasoning(monkeypatch):
    from routes import providers as providers_route
    from routes.providers import ProviderModelsRequest, list_provider_models

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def list_models(self):
            return [{"id": "qwen3.8-14b"}]

        async def close(self):
            return None

    # No `probe_chat_template` method: a catalog-time probe would AttributeError.
    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)

    result = _run(
        list_provider_models(
            ProviderModelsRequest(
                provider_type = "llama_cpp",
                base_url = "http://127.0.0.1:8080/v1",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )

    assert [m.id for m in result] == ["qwen3.8-14b"]


def test_get_provider_model_reasoning_probes_single_model(monkeypatch):
    from routes import providers as providers_route
    from routes.providers import (
        ProviderModelReasoningRequest,
        get_provider_model_reasoning,
    )

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def probe_chat_template(self, model_id):
            assert model_id == "qwen3.8-14b"
            return QWEN38_TEMPLATE

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)

    reasoning = _run(
        get_provider_model_reasoning(
            ProviderModelReasoningRequest(
                provider_type = "llama_cpp",
                base_url = "http://127.0.0.1:8080/v1",
                model_id = "qwen3.8-14b",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )

    assert reasoning is not None
    assert reasoning.supports_reasoning is True
    assert reasoning.reasoning_style == "enable_thinking_effort"
    assert reasoning.reasoning_effort_levels == ["low", "medium", "xhigh"]


def test_get_provider_model_reasoning_returns_none_for_other_providers(monkeypatch):
    from routes import providers as providers_route
    from routes.providers import (
        ProviderModelReasoningRequest,
        get_provider_model_reasoning,
    )

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)

    reasoning = _run(
        get_provider_model_reasoning(
            ProviderModelReasoningRequest(
                provider_type = "mistral",
                base_url = "https://api.mistral.ai/v1",
                model_id = "mistral-small-latest",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )

    assert reasoning is None


def test_get_provider_model_reasoning_requires_model_id(monkeypatch):
    from routes import providers as providers_route
    from routes.providers import (
        ProviderModelReasoningRequest,
        get_provider_model_reasoning,
    )

    with pytest.raises(Exception) as exc:
        _run(
            get_provider_model_reasoning(
                ProviderModelReasoningRequest(
                    provider_type = "llama_cpp",
                    base_url = "http://127.0.0.1:8080/v1",
                    model_id = "   ",
                ),
                _current_subject = "alice",
                via_api_key = False,
            )
        )
    assert "model_id" in str(exc.value)


# ── stream_chat_completion forwards reasoning as chat_template_kwargs ─


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_llama_cpp_client() -> ExternalProviderClient:
    return _make_client("llama_cpp", "http://127.0.0.1:8080/v1")


_SSE = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


def _stream_body(monkeypatch, provider_type, **kwargs):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content = _SSE, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client(provider_type, "http://127.0.0.1:8080/v1")
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "m",
                **kwargs,
            )
        )
        await client.close()
        return lines

    _drive(run())
    return captured.get("body") or {}


def test_llama_cpp_forwards_reasoning_effort(monkeypatch):
    body = _stream_body(monkeypatch, "llama_cpp", reasoning_effort = "xhigh")
    assert body["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}


def test_llama_cpp_forwards_enable_thinking_and_effort(monkeypatch):
    body = _stream_body(monkeypatch, "llama_cpp", enable_thinking = True, reasoning_effort = "medium")
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_llama_cpp_forwards_enable_thinking_alone(monkeypatch):
    body = _stream_body(monkeypatch, "llama_cpp", enable_thinking = False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_cpp_omits_chat_template_kwargs_without_reasoning_fields(monkeypatch):
    body = _stream_body(monkeypatch, "llama_cpp")
    assert "chat_template_kwargs" not in body


def test_vllm_still_forwards_enable_thinking(monkeypatch):
    body = _stream_body(monkeypatch, "vllm", enable_thinking = True)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_ollama_does_not_forward_reasoning_kwargs(monkeypatch):
    # Ollama's OpenAI-compat endpoint drops chat_template_kwargs (it has no
    # reasoning_effort concept in its Go templates), so we leave the body alone;
    # a native `think` path would be separate work.
    body = _stream_body(monkeypatch, "ollama", enable_thinking = True, reasoning_effort = "high")
    assert "chat_template_kwargs" not in body

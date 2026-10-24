# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The character estimate overcounts plain text. A prompt that fits the window but whose
estimate does not must be capped by its exact count, not floored to a one-token answer, on
every surface that sends a wire limit, as the OpenAI plain chat already is."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.inference as inf_mod
from core.inference.api_monitor import ApiMonitor
from core.inference.context_window import estimate_messages_tokens_dense
from core.inference.llama_admission import reset_llama_admission_queues
from fastapi.responses import JSONResponse

from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from routes.inference import (
    _build_openai_passthrough_body_async,
    _openai_llama_admission_enforced_max_tokens,
    anthropic_messages,
)

pytestmark = pytest.mark.usefixtures("share_pricing")

_CTX = 4096
_COUNTED = 3030
_TEXT = "word " * 3500


class _Request:
    def __init__(self, path = "/v1/messages"):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = path)
        self.method = "POST"
        self.app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4))

    async def is_disconnected(self):
        return False


def _backend(seen: dict, *, supports_tool_passthrough = False):
    def _gen_plain(**kwargs):
        seen["plain"] = kwargs
        yield "ok"

    def _gen_tools(**kwargs):
        seen["tools"] = kwargs
        yield {"type": "content", "text": "ok"}

    return SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        supports_tool_passthrough = supports_tool_passthrough,
        model_identifier = "test-model",
        context_length = _CTX,
        _kv_cache_context_total = _CTX,
        effective_parallel_slots = 4,
        markup_profile = None,
        count_chat_tokens = lambda *a, **k: _COUNTED,
        _request_reasoning_kwargs = lambda *_a, **_k: None,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
        base_url = "http://llama.exact-count.test:9999",
    )


def _counted_cap(
    payload,
    backend,
    request,
    *,
    conversation,
    pausable = True,
) -> int:
    return _openai_llama_admission_enforced_max_tokens(
        payload,
        request = request,
        llama_backend = backend,
        conversation = conversation,
        prompt_tokens = _COUNTED,
        prompt_counted = True,
        pausable = pausable,
    )


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    reset_llama_admission_queues()
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
    monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
    yield
    reset_llama_admission_queues()


def test_the_setup_overcounts_a_prompt_that_fits():
    estimate = estimate_messages_tokens_dense([{"role": "user", "content": _TEXT}])
    assert _COUNTED < _CTX < estimate, (estimate, _COUNTED)


def test_the_anthropic_plain_chat_is_capped_by_its_count(monkeypatch):
    seen: dict = {}
    backend = _backend(seen)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    request = _Request()
    payload = AnthropicMessagesRequest(
        max_tokens = _CTX, messages = [{"role": "user", "content": _TEXT}]
    )

    asyncio.run(anthropic_messages(payload, request = request, current_subject = "t"))

    allowance = seen["plain"]["admission_output_allowance"]
    expected = _counted_cap(payload, backend, request, conversation = seen["plain"]["messages"])
    assert expected > 1
    assert allowance == expected, (allowance, expected)


def test_the_anthropic_passthrough_is_capped_by_its_count(monkeypatch):
    seen: dict = {}
    backend = _backend(seen, supports_tool_passthrough = True)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    captured: dict = {}

    async def _fake_passthrough(
        llama_backend,
        openai_messages,
        openai_tools,
        temperature,
        top_p,
        top_k,
        max_tokens,
        *args,
        **kwargs,
    ):
        captured["max_tokens"] = max_tokens
        captured["messages"] = openai_messages
        captured["tools"] = openai_tools
        return JSONResponse(content = {"id": "msg_x"})

    monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _fake_passthrough)
    request = _Request()
    payload = AnthropicMessagesRequest.model_validate(
        {
            "max_tokens": _CTX,
            "messages": [{"role": "user", "content": _TEXT}],
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search the web.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
    )

    asyncio.run(anthropic_messages(payload, request = request, current_subject = "t"))

    expected = _counted_cap(
        payload, backend, request, conversation = captured["messages"], pausable = False
    )
    assert expected > 1
    assert captured["max_tokens"] == expected, (captured["max_tokens"], expected)


def test_the_openai_passthrough_body_is_capped_by_its_count():
    backend = _backend({})
    request = _Request("/v1/chat/completions")
    payload = ChatCompletionRequest.model_validate(
        {
            "model": "default",
            "messages": [{"role": "user", "content": _TEXT}],
            "max_tokens": _CTX,
            "stream": False,
        }
    )

    body = asyncio.run(
        _build_openai_passthrough_body_async(
            payload, backend_ctx = _CTX, llama_backend = backend, request = request
        )
    )

    expected = _counted_cap(
        payload, backend, request, conversation = body["messages"], pausable = False
    )
    assert expected > 1
    assert body["max_tokens"] == expected, (body["max_tokens"], expected)


def test_a_failed_count_keeps_the_estimate(monkeypatch):
    """A count that fails reads as the whole pool: the estimate's cap stands, never a larger one."""
    backend = _backend({})

    def _fail(*_a, **_k):
        raise RuntimeError("tokenizer down")

    backend.count_chat_tokens = _fail
    request = _Request("/v1/chat/completions")
    payload = ChatCompletionRequest.model_validate(
        {
            "model": "default",
            "messages": [{"role": "user", "content": _TEXT}],
            "max_tokens": _CTX,
            "stream": False,
        }
    )

    body = asyncio.run(
        _build_openai_passthrough_body_async(
            payload, backend_ctx = _CTX, llama_backend = backend, request = request
        )
    )

    assert body["max_tokens"] == _openai_llama_admission_enforced_max_tokens(
        payload, request = request, llama_backend = backend, pausable = False
    )

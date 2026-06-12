# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for llama-server route timeout policy."""

import asyncio
import os
import sys
from types import SimpleNamespace

import httpx

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import ChatCompletionRequest  # noqa: E402
import routes.inference as inf_mod  # noqa: E402


def _backend_ctx():
    return SimpleNamespace(
        base_url = "http://llama.test",
        context_length = 4096,
        is_loaded = True,
        _request_reasoning_kwargs = (
            lambda enable_thinking = None, reasoning_effort = None, preserve_thinking = None: None
        ),
    )


class _RecordingAsyncClient:
    seen: list[httpx.Timeout] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, timeout):
        self.seen.append(timeout)
        return httpx.Response(
            200,
            json = {
                "id": "chatcmpl-test",
                "model": "local",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


def _patch_recording_client(monkeypatch):
    _RecordingAsyncClient.seen = []
    monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _RecordingAsyncClient)
    return _RecordingAsyncClient.seen


def _assert_uncapped_generation_read(timeout: httpx.Timeout):
    data = timeout.as_dict()
    assert data["read"] is None
    assert data["connect"] == inf_mod._DEFAULT_FIRST_TOKEN_TIMEOUT_S
    assert data["write"] == inf_mod._DEFAULT_FIRST_TOKEN_TIMEOUT_S
    assert data["pool"] == inf_mod._DEFAULT_FIRST_TOKEN_TIMEOUT_S


def test_non_streaming_generation_timeout_has_no_body_read_cap():
    _assert_uncapped_generation_read(inf_mod._llama_non_streaming_generation_timeout())


def test_openai_passthrough_non_streaming_uses_uncapped_body_read(monkeypatch):
    seen = _patch_recording_client(monkeypatch)
    payload = ChatCompletionRequest(
        model = "local",
        messages = [{"role": "user", "content": "hi"}],
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    )

    asyncio.run(inf_mod._openai_passthrough_non_streaming(_backend_ctx(), payload, "local"))

    assert len(seen) == 1
    _assert_uncapped_generation_read(seen[0])


def test_anthropic_passthrough_non_streaming_uses_uncapped_body_read(monkeypatch):
    seen = _patch_recording_client(monkeypatch)

    asyncio.run(
        inf_mod._anthropic_passthrough_non_streaming(
            _backend_ctx(),
            [{"role": "user", "content": "hi"}],
            [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            temperature = 0.6,
            top_p = 0.95,
            top_k = 20,
            max_tokens = None,
            message_id = "msg_test",
            model_name = "local",
        )
    )

    assert len(seen) == 1
    _assert_uncapped_generation_read(seen[0])


def test_openai_completions_non_streaming_uses_uncapped_body_read(monkeypatch):
    seen = _patch_recording_client(monkeypatch)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", _backend_ctx)

    class _Request:
        async def json(self):
            return {"model": "local", "prompt": "hi", "stream": False}

    asyncio.run(inf_mod.openai_completions(_Request(), current_subject = "user"))

    assert len(seen) == 1
    _assert_uncapped_generation_read(seen[0])

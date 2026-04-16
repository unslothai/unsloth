import asyncio
import os, sys
from unittest.mock import patch

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

import pytest
from fastapi import HTTPException

from models.inference import ChatCompletionRequest


class _Llama:
    def __init__(self, is_loaded = True, supports_tools = True, is_vision = False):
        self.is_loaded = is_loaded
        self.supports_tools = supports_tools
        self.is_vision = is_vision
        self._is_audio = False
        self.model_identifier = "stub"
        self.base_url = "http://127.0.0.1:0"
        self._api_key = None


class _Inf:
    active_model_name = "hf"
    models = {"hf": {}}


class _Req:
    async def is_disconnected(self):
        return False


async def _marker_stream(*a, **kw):
    return ("passthrough_stream", None)


async def _marker_nonstream(*a, **kw):
    return ("passthrough_nonstream", None)


async def _call(payload, llama):
    from routes import inference as inf_mod

    with (
        patch.object(inf_mod, "get_llama_cpp_backend", return_value = llama),
        patch.object(inf_mod, "get_inference_backend", return_value = _Inf()),
        patch.object(inf_mod, "_openai_passthrough_stream", new = _marker_stream),
        patch.object(
            inf_mod, "_openai_passthrough_non_streaming", new = _marker_nonstream
        ),
    ):
        return await inf_mod.openai_chat_completions(
            payload, _Req(), current_subject = "u"
        )


def test_role_tool_on_non_gguf_rejected():
    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
    )
    llama = _Llama(is_loaded = False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call(payload, llama))
    assert ei.value.status_code == 400
    assert "role='tool'" in ei.value.detail or "tool" in ei.value.detail.lower()


def test_role_tool_on_gguf_without_tool_support_rejected():
    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
    )
    llama = _Llama(supports_tools = False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call(payload, llama))
    assert ei.value.status_code == 400


def test_role_tool_with_enable_tools_true_rejected():
    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
        enable_tools = True,
    )
    llama = _Llama()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call(payload, llama))
    assert ei.value.status_code == 400


def test_assistant_tool_only_still_rejected_on_non_passthrough():
    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
        ],
    )
    llama = _Llama(supports_tools = False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call(payload, llama))
    assert ei.value.status_code == 400


def test_no_tool_messages_passes_through():
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "plain"}],
    )
    llama = _Llama(supports_tools = False)
    # Should not raise the tool-shape guard; may raise later for non-GGUF/inf path,
    # but specifically this guard must not fire.
    try:
        asyncio.run(_call(payload, llama))
    except HTTPException as e:
        assert "role='tool'" not in (e.detail or "")
        assert "tool_calls-only" not in (e.detail or "")

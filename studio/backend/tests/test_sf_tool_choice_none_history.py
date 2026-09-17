# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe: tool_choice "none" on a named-template model must keep tool history."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import openai_chat_completions
from core.inference.api_monitor import ApiMonitor


LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}

# A named-template model: tool support lives ONLY in the tool_use branch.
DEFAULT_BODY = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
)
TOOL_USE_BODY = (
    "{% if tools %}tools available{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "<tool_call>{{ m.tool_calls }}</tool_call><|im_end|>\n{% endfor %}"
)
NAMED_TEMPLATE = {"default": DEFAULT_BODY, "tool_use": TOOL_USE_BODY}


class _Request:
    state = SimpleNamespace()
    url = SimpleNamespace(path = "/v1/chat/completions")
    method = "POST"
    scope: dict = {}
    headers = {"X-Unsloth-Events": "1"}

    async def is_disconnected(self):
        return False


class _ScriptedBackend:
    active_model_name = "sf-model"

    def __init__(self):
        self.models = {
            "sf-model": {
                "chat_template_info": {"template": NAMED_TEMPLATE},
                "context_length": 2048,
            }
        }
        self.calls: list = []
        self.reset_count = 0

    def generate_chat_response(
        self,
        *,
        messages,
        tools = None,
        stats_holder = None,
        **kwargs,
    ):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        yield "the weather is sunny"

    def generate_chat_completion_with_tools(
        self,
        *,
        messages,
        tools = None,
        **kwargs,
    ):
        self.calls.append({"loop": True, "messages": messages, "tools": tools, **kwargs})
        yield {"type": "content", "text": "the weather is sunny"}

    def reset_generation_state(self, caller_cancel_event = None):
        self.reset_count += 1

    def resize_image(self, image):
        return image


def _llama_stub():
    return SimpleNamespace(
        is_loaded = False, supports_tools = False, is_vision = False, context_length = None
    )


def _install(monkeypatch, backend):
    import routes.inference as inf
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    monkeypatch.setattr(inf, "api_monitor", ApiMonitor(max_entries = 8))
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _llama_stub())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: backend)


def _continued_exchange(**extra):
    base = dict(
        model = "default",
        messages = [
            ChatMessage(role = "user", content = "weather in SF?"),
            ChatMessage(
                role = "assistant",
                content = None,
                tool_calls = [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "sf weather"}'},
                    }
                ],
            ),
            ChatMessage(role = "tool", tool_call_id = "call_abc", name = "lookup", content = "sunny"),
        ],
        tools = [LOOKUP_TOOL],
        tool_choice = "none",
        enable_tools = True,
        stream = False,
    )
    base.update(extra)
    return ChatCompletionRequest(**base)


def _run(payload, monkeypatch, backend):
    _install(monkeypatch, backend)

    async def _go():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    return asyncio.run(_go())


def test_tool_history_survives_tool_choice_none(monkeypatch):
    backend = _ScriptedBackend()
    _run(_continued_exchange(), monkeypatch, backend)

    assert backend.calls, "generator never ran"
    msgs = backend.calls[0]["messages"]
    print("\nMESSAGES HANDED TO BACKEND:\n" + json.dumps(msgs, indent = 2, default = str))
    print("TOOLS HANDED TO BACKEND:", backend.calls[0]["tools"])

    assistant = [m for m in msgs if (m.get("role") if isinstance(m, dict) else None) == "assistant"]
    tool_msgs = [m for m in msgs if (m.get("role") if isinstance(m, dict) else None) == "tool"]
    assert assistant, "assistant tool-call turn dropped entirely"
    assert assistant[0].get("tool_calls"), "assistant tool_calls dropped"
    assert tool_msgs, "tool result dropped"
    assert tool_msgs[0].get("tool_call_id") == "call_abc", "tool_call_id dropped"
    assert backend.calls[0]["tools"] is None, "tool_choice none must advertise no tools"


def test_tool_history_survives_plain_openai_client(monkeypatch):
    """The common shape: an OpenAI client that never sets Unsloth's enable_tools."""
    backend = _ScriptedBackend()
    _run(_continued_exchange(enable_tools = None), monkeypatch, backend)

    msgs = backend.calls[0]["messages"]
    print("\nPLAIN CLIENT MESSAGES:\n" + json.dumps(msgs, indent = 2, default = str))
    assistant = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert assistant and assistant[0].get("tool_calls"), "assistant tool_calls dropped"
    assert tool_msgs and tool_msgs[0].get("tool_call_id") == "call_abc", "tool_call_id dropped"

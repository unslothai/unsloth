# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thinking-off chat replays each earlier answer with the empty think block it was generated
behind, so the next turn's prompt keeps llama-server's cached prefix."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend

QWEN_GEN = "<|im_start|>user\nStudio replay probe question.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _backend(supports = True) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._port = 48847
    backend._api_key = None
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = supports
    backend._chat_template = "tpl"
    return backend


def _history(reasoning = None) -> list[dict]:
    answer = {"role": "assistant", "content": "Pip paints."}
    if reasoning is not None:
        answer["reasoning_content"] = reasoning
    return [{"role": "user", "content": "story"}, answer, {"role": "user", "content": "more"}]


def test_no_reasoning_history_keeps_the_empty_think_block(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(LlamaCppBackend, "_empty_think_replays_with_preserve", lambda self: True)
    assert backend._replay_preserve_thinking(False, _history(), False) is True
    assert backend._replay_preserve_thinking(False, _history(""), False) is True


def test_the_users_choice_stands_wherever_there_is_reasoning_or_no_proof(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(LlamaCppBackend, "_empty_think_replays_with_preserve", lambda self: True)
    assert backend._replay_preserve_thinking(False, _history("thought about it"), False) is False
    # Omitted (every API client that never sends it) resolves the same way as an explicit off.
    assert backend._replay_preserve_thinking(None, _history(), False) is True
    assert backend._replay_preserve_thinking(None, _history("thought about it"), False) is None
    assert backend._replay_preserve_thinking(True, _history(), False) is True
    assert _backend(supports = False)._replay_preserve_thinking(False, _history(), False) is False
    # Thinking on: its reasoning is real, so the caller's choice to drop it stands.
    assert backend._replay_preserve_thinking(False, _history(), True) is False
    assert backend._replay_preserve_thinking(False, _history(), None) is False
    # Reasoning carried inline in the answer is still reasoning.
    inline = _history()
    inline[1]["content"] = "<think>\nplan\n</think>\n\nPip paints."
    assert backend._replay_preserve_thinking(False, inline, False) is False
    inline[1]["content"] = [{"type": "text", "text": "<think>plan</think>Pip paints."}]
    assert backend._replay_preserve_thinking(False, inline, False) is False
    # Nothing answered yet: nothing to replay, so no probe either.
    assert backend._replay_preserve_thinking(False, _history()[:1], False) is False
    monkeypatch.setattr(LlamaCppBackend, "_empty_think_replays_with_preserve", lambda self: False)
    assert backend._replay_preserve_thinking(False, _history(), False) is False


class _FakeClient:
    def __init__(self, render, calls):
        self._render = render
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        self._calls.append(json)
        prompt = self._render(json["messages"], json["chat_template_kwargs"]["preserve_thinking"])
        return type("R", (), {"status_code": 200, "json": lambda self: {"prompt": prompt}})()


def _qwen_render(messages, preserve):
    if len(messages) == 1:
        return QWEN_GEN
    think = "<think>\n\n</think>\n\n" if preserve else ""
    return (
        "<|im_start|>user\nStudio replay probe question.<|im_end|>\n<|im_start|>assistant\n"
        + think
        + "Studio replay probe answer.<|im_end|>\n<|im_start|>user\nStudio replay probe follow-up.<|im_end|>\n"
    )


def _plain_render(messages, preserve):
    # A template that renders the same either way: nothing to gain, so nothing changes.
    if len(messages) == 1:
        return "<|im_start|>user\nStudio replay probe question.<|im_end|>\n<|im_start|>assistant\n"
    return (
        "<|im_start|>user\nStudio replay probe question.<|im_end|>\n<|im_start|>assistant\n"
        "Studio replay probe answer.<|im_end|>\n<|im_start|>user\nStudio replay probe follow-up.<|im_end|>\n"
    )


def test_probe_reads_the_templates_own_render_once(monkeypatch):
    calls: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", lambda **kw: _FakeClient(_qwen_render, calls))
    backend = _backend()
    assert backend._empty_think_replays_with_preserve() is True
    assert backend._empty_think_replays_with_preserve() is True
    assert len(calls) == 3
    assert all(c["chat_template_kwargs"]["enable_thinking"] is False for c in calls)


def test_probe_refuses_a_template_that_renders_alike(monkeypatch):
    calls: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", lambda **kw: _FakeClient(_plain_render, calls))
    assert _backend()._empty_think_replays_with_preserve() is False


def test_probe_failure_keeps_the_setting_and_retries(monkeypatch):
    def boom(**kw):
        raise OSError("server busy")

    monkeypatch.setattr(llama_cpp.httpx, "Client", boom)
    backend = _backend()
    assert backend._empty_think_replays_with_preserve() is False
    assert getattr(backend, "_empty_think_preserve_probe", None) is None


class _RouteBackend:
    """What the passthrough body builder reads off the llama backend."""

    is_vision = False
    markup_profile = None

    def _request_reasoning_kwargs(self, enable_thinking, reasoning_effort, preserve_thinking):
        kwargs = {"enable_thinking": enable_thinking}
        if preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        return kwargs

    def _replay_preserve_thinking(
        self,
        preserve,
        messages,
        enable_thinking = None,
        effort = None,
    ):
        answered = any(m.get("role") == "assistant" for m in messages)
        return True if preserve is not True and enable_thinking is False and answered else preserve


def test_client_tool_passthrough_replays_the_empty_think_blocks_too():
    # Coding agents bring their own tools, so their turns ride the passthrough: it has to send the
    # preserve_thinking generation would, or every follow-up re-reads the previous turn.
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_openai_passthrough_body

    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "and again"},
        ],
        tools = [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
        ],
        enable_thinking = False,
    )
    body = _build_openai_passthrough_body(payload, backend_ctx = 4096, llama_backend = _RouteBackend())
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": True}


def test_gguf_history_carries_the_tool_call_ids_the_model_generated():
    from models.inference import ChatCompletionRequest
    from routes.inference import _openai_messages_for_gguf_chat

    minted = "call_0:2f1c0a9e-4b7d-4c1e-9a3b-5d6e7f809a1b"
    payload = ChatCompletionRequest(
        messages = [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": minted,
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": minted, "name": "python", "content": "ok"},
            {"role": "user", "content": "next"},
        ],
    )
    messages, _ = _openai_messages_for_gguf_chat(payload, is_vision = False)
    assert messages[1]["tool_calls"][0]["id"] == "call_0"
    assert messages[2]["tool_call_id"] == "call_0"

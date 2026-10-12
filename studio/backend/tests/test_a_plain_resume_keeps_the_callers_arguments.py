# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paused plain chat reopens with every argument its caller gave, as the respawn retry does.

The resume call dropped `thinking_budget_tokens`, so the resumed request ran without the
caller's reasoning limit.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import json
import textwrap
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Policy:
    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        pass

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        pass


def _backend() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48856
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    return backend


def test_the_resumed_request_keeps_the_thinking_budget(monkeypatch):
    signal = preemption.PreemptSignal()
    backend = _backend()
    opened: list[dict] = []

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
        preempt_event = None,
    ):
        opened.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
        preempt_event = None,
    ):
        delta = {"choices": [{"index": 0, "delta": {"content": "ab"}}]}
        yield "data: " + json.dumps(delta) + "\n"
        if len(opened) == 1:
            signal.request("kv_pressure")
            raise preemption.LlamaStreamPreempted
        done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield "data: " + json.dumps(done) + "\n"
        yield "data: [DONE]\n"

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "think, then answer"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = _Policy(),
            thinking_budget_tokens = 128,
        )
    )
    assert len(opened) == 2, len(opened)
    assert [p.get("thinking_budget_tokens") for p in opened] == [128, 128]


def test_the_resume_forwards_every_parameter():
    # `preempt_event` rides a `**` expansion, so it has no keyword of its own.
    source = textwrap.dedent(inspect.getsource(LlamaCppBackend.generate_chat_completion))
    function = ast.parse(source).body[0]
    params = {a.arg for a in function.args.args + function.args.kwonlyargs} - {"self"}
    resume = next(
        {k.arg for k in node.keywords if k.arg}
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate_chat_completion"
        and any(
            k.arg == "max_tokens" and ast.unparse(k.value) == "resume_max_tokens"
            for k in node.keywords
        )
    )
    assert params - resume - {"preempt_event"} == set()

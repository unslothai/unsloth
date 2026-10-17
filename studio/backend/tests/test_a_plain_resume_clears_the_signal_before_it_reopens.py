# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plain chat clears its pause signal before it reopens the stream, as the tool paths do."""

from __future__ import annotations

import contextlib
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


def _finish() -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + "\n"
    )


def test_the_resumed_attempt_opens_with_the_signal_clear(monkeypatch):
    signal = preemption.PreemptSignal()
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
    set_at_open: list[bool] = []

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
        set_at_open.append(signal.is_set())
        yield type("FakeResponse", (), {"status_code": 200})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
        preempt_event = None,
    ):
        if len(set_at_open) == 1:
            yield _delta("a")
            signal.request("kv_pressure")
            raise preemption.LlamaStreamPreempted
        yield _delta("b")
        yield _finish()
        yield "data: [DONE]\n"

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "write me a poem"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = preemption.NullPreemptionPolicy(),
        )
    )
    assert set_at_open == [False, False], set_at_open

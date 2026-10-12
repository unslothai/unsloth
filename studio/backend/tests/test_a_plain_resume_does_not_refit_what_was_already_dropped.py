# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paused plain chat resumes from the prompt it actually sent, as a respawn does.

The resume rebuilt its request from the caller's messages, so history the rolling fit had
evicted came back and was fitted out again, and the client was told about the same dropped
messages twice and added them up.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_cpp as llama_mod
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


def _run(monkeypatch):
    signal = preemption.PreemptSignal()
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48857
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    payloads: list[dict] = []
    fits: list[int] = []

    def fit(messages, **kwargs):
        # Evicts everything before the last turn, once per call.
        fits.append(len(messages))
        kept = messages[-1:]
        dropped = len(messages) - len(kept)
        return kept, ({"dropped_messages": dropped, "fits": True} if dropped else None)

    monkeypatch.setattr(llama_mod, "_fit_with_instruction_pins", fit)
    monkeypatch.setattr(
        llama_mod,
        "_archive_and_recall",
        lambda fitted, before, **k: {"conversation": fitted, "counts": {}},
    )
    monkeypatch.setattr(llama_mod, "_sticky_compaction_state", lambda *a, **k: (None, False))
    monkeypatch.setattr(llama_mod, "_records_boundary", lambda truncation: False)

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
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
        preempt_event = None,
    ):
        if len(payloads) == 1:
            yield _delta("Once")
            signal.request("kv_pressure")
            raise preemption.LlamaStreamPreempted
        yield _delta(" upon a time.")
        yield _finish()
        yield "data: [DONE]\n"

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    events = list(
        backend.generate_chat_completion(
            messages = [
                {"role": "user", "content": "an old question"},
                {"role": "assistant", "content": "an old answer"},
                {"role": "user", "content": "write me a poem"},
            ],
            cancel_event = threading.Event(),
            context_overflow = "truncate_oldest",
            preempt_event = signal,
            preempt_policy = preemption.NullPreemptionPolicy(),
        )
    )
    return events, payloads


def test_the_dropped_messages_are_reported_once(monkeypatch):
    events, payloads = _run(monkeypatch)
    reports = [e for e in events if isinstance(e, dict) and e.get("type") == "context_truncated"]
    assert [r["dropped_messages"] for r in reports] == [2], reports
    assert len(payloads) == 2


def test_the_resume_sends_the_fitted_prompt_and_its_partial(monkeypatch):
    _events, payloads = _run(monkeypatch)
    resumed = payloads[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"], resumed
    assert "an old question" not in json.dumps(resumed)
    assert resumed[-1]["content"] == "Once"

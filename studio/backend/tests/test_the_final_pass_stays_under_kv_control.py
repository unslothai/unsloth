# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The synthesised final answer is a generation like any other.

``generate_chat_completion_with_tools`` runs its rounds, then breaks into a separate
final pass that streams the answer with no tools attached. For a common tool run -- one
that exhausts its tool budget, or whose model stops asking -- that pass produces MOST of
what the user reads, and it was the one generation nothing bounded and nothing watched:

  * ``admission_output_allowance`` is the wire clamp that makes the reservation an
    enforced figure rather than a recorded one. Applied to every round's payload, and not
    to this one, so the pass that answers sent the whole window as its output cap while
    admission had reserved a share of it.
  * ``on_tokens`` is the ONLY thing that calls ``observe()``, and ``observe()`` is the
    only thing that plans an eviction. Without it the participant sat in the ledger at
    its last round-boundary figure while this pass decoded thousands more tokens into the
    shared cache, so the watermark could not fire on the growth that mattered most.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference.llama_cpp import LlamaCppBackend


_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _sse(delta: dict, finish = None) -> str:
    choice: dict = {"index": 0, "delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _tool_round() -> list[str]:
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "cats"}),
                        },
                    }
                ]
            },
            finish = "tool_calls",
        ),
        _done(),
    ]


def _final_answer(tokens: int) -> list[str]:
    return [*[_sse({"content": f"w{i} "}) for i in range(tokens)], _sse({}, finish = "stop"), _done()]


def _backend(monkeypatch, streams, payloads):
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
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
        preempt_event = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend


def _run(backend, **kwargs):
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search and answer"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            max_tool_iterations = 1,
            **kwargs,
        )
    )


class TestTheFinalPassIsClampedToWhatWasAdmitted:
    def test_the_allowance_bounds_the_final_payload(self, monkeypatch):
        payloads: list[dict] = []
        streams = [_tool_round(), _final_answer(3)]
        backend = _backend(monkeypatch, streams, payloads)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool", lambda *a, **k: "RESULT"
        )

        _run(backend, max_tokens = 3000, admission_output_allowance = 512)

        # The last request is the final pass; the ones before it are the rounds.
        assert payloads[-1]["max_tokens"] == 512, (
            "the pass that produces the answer sent an output cap larger than the room "
            f"admission reserved for it: {payloads[-1]['max_tokens']}"
        )
        assert payloads[0]["max_tokens"] == 512, "the rounds were already clamped"

    def test_no_allowance_leaves_the_callers_cap_alone(self, monkeypatch):
        payloads: list[dict] = []
        streams = [_tool_round(), _final_answer(3)]
        backend = _backend(monkeypatch, streams, payloads)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool", lambda *a, **k: "RESULT"
        )

        _run(backend, max_tokens = 300)

        assert payloads[-1]["max_tokens"] == 300


class TestTheFinalPassReportsItsGrowth:
    def test_the_watermark_sweep_hears_from_it(self, monkeypatch):
        """One report per `_TOKEN_REPORT_EVERY` chunks, as the in-loop stream does."""
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

        payloads: list[dict] = []
        # Enough content chunks for two reports on the final pass alone.
        streams = [_tool_round(), _final_answer(_TOKEN_REPORT_EVERY * 2 + 1)]
        backend = _backend(monkeypatch, streams, payloads)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool", lambda *a, **k: "RESULT"
        )
        seen: list[int] = []

        _run(backend, max_tokens = 3000, on_tokens = seen.append)

        assert seen, (
            "the final pass never told the preemptor it was decoding, so `observe()` "
            "never ran and the only thing that plans an eviction never saw the growth"
        )
        assert max(seen) >= _TOKEN_REPORT_EVERY

    def test_a_reporter_that_raises_never_fails_the_turn(self, monkeypatch):
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

        payloads: list[dict] = []
        streams = [_tool_round(), _final_answer(_TOKEN_REPORT_EVERY + 1)]
        backend = _backend(monkeypatch, streams, payloads)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool", lambda *a, **k: "RESULT"
        )

        def _boom(_n):
            raise RuntimeError("the sweep exploded")

        items = _run(backend, max_tokens = 3000, on_tokens = _boom)
        assert any(
            isinstance(item, dict) and item.get("type") == "content" for item in items
        ), "bookkeeping must never take the answer with it"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""For a reasoning-only model the promoted fallback IS the answer.

Qwen3 and its kind put the whole reply in ``reasoning_content``. The stream wraps that in
``<think>``, and at a clean stop `_finalize_reasoning_only_cumulative` appends the same
text again as visible content, because the frontend hides the thought block and shows the
fallback. That is the answer the user reads.

The fallback was built from the CURRENT attempt's ``reasoning_text`` alone. A pause
splits one reply into two attempts, so a reply of A then B promoted only B: the stitching
correctly restored A inside the thought, and the user was shown the second half of their
answer with the first half hidden in a block the UI does not render. Half an answer,
silently, on the surface a pause is supposed to be invisible on.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


def _reasoning(content: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": content}}]})
        + "\n"
    )


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


class _Recorder:
    def __init__(self, monkeypatch, streams, *, signal):
        self.payloads: list[dict] = []
        self.signal = signal
        self._streams = [list(stream) for stream in streams]
        self.backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend = self.backend
        backend._process = object()
        backend._healthy = True
        backend._port = 48853
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = True
        backend._reasoning_always_on = True
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

        recorder = self

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
            recorder.payloads.append(copy.deepcopy(payload))
            yield type(
                "FakeResponse", (), {"status_code": 200, "chunks": recorder._streams.pop(0)}
            )()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            for chunk in response.chunks:
                yield chunk
                if attempt == 0 and chunk.startswith("data: {"):
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _Policy:
    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint) -> None:
        pass

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        pass


class TestThePromotedFallbackCoversBothAttempts:
    def test_the_answer_is_not_cut_to_its_second_half(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_reasoning("The first half. "), _finish(), _done()],
                [_reasoning("The second half."), _finish(), _done()],
            ],
            signal = signal,
        )
        items = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "answer me"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = _Policy(),
                promote_reasoning_only = True,
            )
        )

        snapshots = [item for item in items if isinstance(item, str)]
        assert snapshots, "the chat produced no text at all"
        final = snapshots[-1]
        thought, _, fallback = final.partition("</think>")
        assert "The first half. " in thought and "The second half." in thought, (
            f"the thought lost an attempt: {final!r}"
        )
        assert "The first half. " in fallback, (
            "the promoted fallback IS the answer for a reasoning-only model, and it was "
            f"built from the resumed attempt alone: {final!r}"
        )
        assert "The second half." in fallback

    def test_an_uninterrupted_reasoning_only_chat_is_unchanged(self, monkeypatch):
        """No pause, so nothing is carried and the fallback is this attempt's own."""
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_reasoning("Just the one."), _finish(), _done()]],
            signal = signal,
        )
        # Never pauses: attempt 0 is the only one and the recorder pauses on it, so a
        # fresh recorder that never signals is what this needs.
        recorder.backend._iter_text_cancellable = (
            lambda response, _cancel_event, first_token_deadline = None, preempt_event = None: iter(
                response.chunks
            )
        )
        items = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "answer me"}],
                cancel_event = threading.Event(),
                promote_reasoning_only = True,
            )
        )
        final = [item for item in items if isinstance(item, str)][-1]
        assert final == "<think>Just the one.</think>Just the one."

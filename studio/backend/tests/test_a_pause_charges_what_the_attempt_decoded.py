# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an aborted attempt produced has to be charged, on both surfaces."""

from __future__ import annotations

import threading

from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _RecordingPolicy,
    delta as _delta,
    done as _done,
    finish as _finish,
    web_search_tool,
)

_TOOL = web_search_tool(required = True)


def _Recorder(monkeypatch, streams, *, signal, pause_after = 1):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = (0,),
        pause_after = pause_after,
    )


# Eight CJK tokens. chars // 4 calls this two.
_DENSE = "天地玄黄宇宙洪荒"


class TestTheToolLoopChargesItsPausedAttempt:
    """It charged zero, because it read only the usage a paused stream never sends."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "write me a poem"}],
                tools = [_TOOL],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_the_checkpoint_is_not_charged_zero(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time there was a cat"), _finish(), _done()],
                [_delta(" who slept."), _finish(), _done()],
            ],
            signal = signal,
        )
        self._run(recorder.backend, signal = signal, policy = policy)

        assert policy.checkpoints[0].visible_text == "Once upon a time there was a cat"
        assert policy.checkpoints[0].charged_tokens > 0, (
            "an attempt that decoded 32 characters was charged nothing, so nothing "
            "re-baselined the ledger and nothing spent the caller's cap"
        )

    def test_the_resumed_attempt_does_not_get_a_fresh_output_cap(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time there was a cat"), _finish(), _done()],
                [_delta(" who slept."), _finish(), _done()],
            ],
            signal = signal,
        )
        self._run(recorder.backend, signal = signal, policy = policy, max_tokens = 100)

        opened, resumed = recorder.payloads[0], recorder.payloads[1]
        assert opened["max_tokens"] == 100
        assert resumed["max_tokens"] < 100, (
            "the resumed attempt was handed the whole cap again, so the turn may emit "
            f"more than the caller allowed; got {resumed['max_tokens']}"
        )
        assert resumed["max_tokens"] >= 1, "a request for zero tokens returns nothing at all"


class TestThePlainPathChargesDenseTextByTokens:
    """chars // 4 is an approximation, and on CJK it is wrong by several times."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "写一首诗"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_eight_dense_tokens_are_not_charged_as_two(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        # One delta per token, which is what llama-server streams.
        recorder = _Recorder(
            monkeypatch,
            [
                [*[_delta(ch) for ch in _DENSE], _finish(), _done()],
                [_delta("。"), _finish(), _done()],
            ],
            signal = signal,
            pause_after = len(_DENSE),
        )
        self._run(recorder.backend, signal = signal, policy = policy)

        charged = policy.checkpoints[0].charged_tokens
        assert charged >= len(_DENSE), (
            f"{len(_DENSE)} tokens were streamed and {charged} were charged; the "
            "difference is cells the watermark cannot see and output cap the caller "
            "never agreed to"
        )


class TestAGiveUpStillReportsWhatItDecoded:
    """The turn ends on `length` with text on screen, so its usage cannot be zero."""

    def _run(self, backend, *, signal, policy, **kwargs):
        return list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "write me a poem"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
                **kwargs,
            )
        )

    def test_a_first_attempt_give_up_reports_its_completion_tokens(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [
                [
                    _delta("Once "),
                    _delta("upon "),
                    _delta("a "),
                    _delta("time"),
                    _finish(),
                    _done(),
                ],
            ],
            signal = signal,
            pause_after = 4,
        )
        items = self._run(recorder.backend, signal = signal, policy = policy)

        metadata = [
            item for item in items if isinstance(item, dict) and item.get("type") == "metadata"
        ]
        assert metadata, "a turn that gave up must still end on a terminal metadata event"
        assert metadata[-1]["finish_reason"] == "length"
        assert metadata[-1]["usage"].get("completion_tokens"), (
            "four deltas were streamed and shown; reporting zero completion tokens for "
            "them corrupts every usage-based client and monitor"
        )

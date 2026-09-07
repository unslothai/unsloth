# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an aborted attempt produced has to be charged, on both surfaces.

A pause aborts the upstream stream, so its terminal usage chunk never arrives, and
``timings_per_token`` is opt-in -- the route only asks for it when a monitor row is open.
For an ordinary chat both readings are therefore absent at exactly the moment the count
is needed, and the count is needed for three separate things:

  * ``note_replayed``, which tells the controller the resumed attempt carries the partial
    BACK as prompt. Skipped, the ledger undercounts the resumed chat by the whole partial,
    by more on every pause. That is the measured 2026-09-02 run: four pauses replaying
    564, 59, 1079 and 507 tokens, the ledger saw none of them, and the run went from zero
    context-exhaustion errors to four.
  * the caller's ``max_tokens``, which bounds NEW tokens. Not spent down, a chat paused n
    times may emit (n+1) times what it asked for.
  * the ``usage`` the response reports, which is the client's own accounting.

The plain path had the first two and estimated the third at four characters per token.
The tool loop had none of them, and the estimate is wrong for token-dense text: CJK and
emoji run nearer one character per token, so chars // 4 undercharges by a factor of
several on exactly the text a Chinese or Japanese session produces.
"""

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
    """Pauses the FIRST attempt after a set number of content deltas.

    No usage and no timings anywhere, which is the ordinary case: the final chunk that
    carries them is exactly the chunk a pause prevents.
    """
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
        """Zero skips `note_replayed` entirely -- it is gated on a non-zero charge --
        so the controller never learns the resumed attempt carries the partial as prompt.
        """
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
        """The next iteration rebuilds `max_tokens` from the caller's figure, so without
        an explicit continuation cap a request capped at 100 could emit 100 more after
        every pause.
        """
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

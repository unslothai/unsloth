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

import threading

from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _Policy,
    done as _done,
    finish as _finish,
    reasoning as _reasoning,
)


def _Recorder(monkeypatch, streams, *, signal):
    """A reasoning-capable backend whose first attempt pauses on its first chunk."""
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = (0,),
        port = 48853,
        supports_reasoning = True,
        reasoning_always_on = True,
    )


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
        assert (
            "The first half. " in thought and "The second half." in thought
        ), f"the thought lost an attempt: {final!r}"
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

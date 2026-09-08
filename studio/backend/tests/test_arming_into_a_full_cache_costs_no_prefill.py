# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat armed into a full cache becomes its own victim, and that costs it nothing."""

from __future__ import annotations

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    PreemptionController,
    reset_preemption_controllers,
)


@pytest.fixture(autouse = True)
def _clean_registry():
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


def _full_cache() -> PreemptionController:
    """Three chats decoding in an 8192-cell cache, as the live run had."""
    controller = PreemptionController("arming")
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    for index, tokens in enumerate((2032, 2032, 2032)):
        gen_id = f"decoding-{index}"
        controller.register(gen_id, tokens = tokens, signal = PreemptSignal())
        controller.note_tokens(gen_id, tokens)
    return controller


class TestTheArmingSweepPrefersTheNewcomer:
    def test_the_arriving_chat_is_the_only_victim(self):
        controller = _full_cache()
        controller.register("arriving", tokens = 2032, signal = PreemptSignal())

        victims = controller.plan_preemptions(needed = 0)

        assert [v.gen_id for v in victims] == ["arriving"], (
            "newest-first is the benchmarked policy and the newcomer is the newest; "
            "sparing it would evict a chat that is decoding to admit one that is not"
        )

    def test_no_chat_that_is_decoding_is_touched(self):
        """The three incumbents have real cells and real text on screen; the newcomer has neither."""
        controller = _full_cache()
        controller.register("arriving", tokens = 2032, signal = PreemptSignal())
        controller.plan_preemptions(needed = 0)

        for index in range(3):
            participant = controller.participant(f"decoding-{index}")
            assert participant.state == ParticipantState.DECODING
            assert not participant.preempt_event.is_set()

    def test_the_newcomer_holds_no_cells_to_free(self):
        """Which is why choosing it frees nothing and evicts nothing."""
        controller = _full_cache()
        controller.register("arriving", tokens = 2032, signal = PreemptSignal())
        assert controller.participant("arriving").measured is False

    def test_losing_repeatedly_still_promotes_it(self):
        """The one thing that must not follow: never being admitted at all."""
        controller = _full_cache()
        controller.register("arriving", tokens = 2032, signal = PreemptSignal())
        for _ in range(preemption.PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS):
            controller.plan_preemptions(needed = 0)
            controller.set_state("arriving", ParticipantState.DECODING)
        assert controller.participant("arriving").promoted is True


class TestNothingIsPrefilledBeforeThePause:
    def test_an_armed_signal_stops_the_request_before_it_is_sent(self):
        """The claim the whole trade-off rests on: there is no prefill to waste."""

        class _ExplodingClient:
            def stream(self, *args, **kwargs):
                raise AssertionError(
                    "llama-server was asked to prefill a prompt for a chat that had "
                    "already been told to stop"
                )

        signal = PreemptSignal()
        signal.request("kv_pressure")
        assert signal.is_set()

        with pytest.raises(preemption.LlamaStreamPreempted):
            with LlamaCppBackend._stream_with_retry(
                _ExplodingClient(),
                "http://127.0.0.1:1/v1/chat/completions",
                {"messages": []},
                None,
                preempt_event = signal,
            ):
                raise AssertionError("the stream opened despite a pending preemption")

    def test_a_clear_signal_does_open_the_request(self):
        """The negative above is only worth anything if the positive still works."""

        opened = []

        class _RecordingClient:
            def stream(self, *args, **kwargs):
                opened.append(kwargs.get("json"))
                raise RuntimeError("stop here; the POST is all this test needs to see")

        with pytest.raises(RuntimeError):
            with LlamaCppBackend._stream_with_retry(
                _RecordingClient(),
                "http://127.0.0.1:1/v1/chat/completions",
                {"messages": []},
                None,
                preempt_event = PreemptSignal(),
            ):
                pass
        assert opened == [{"messages": []}]

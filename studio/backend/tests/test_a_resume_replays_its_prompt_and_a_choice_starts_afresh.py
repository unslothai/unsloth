# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resume brings back its prompt and the partials folded into it, not the output room
admission charged above them; and the next of `n` choices starts from the original prompt
rather than inheriting the last choice's replay and resume count."""

from __future__ import annotations

import inspect

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import ParticipantState
import routes.inference as inference


def _controller(key: str) -> preemption.PreemptionController:
    controller = preemption.PreemptionController(key)
    # An 8192-cell cache with a 2048-token batch: the solo ceiling is 6142.
    controller.configure(budget = 8192, kv_unified = True, slots = 4, draft_tokens = 2, batch_tokens = 2048)
    return controller


class TestAResumeIsSizedByWhatItReplays:
    def test_before_the_first_token_the_replay_is_the_prompt(self):
        controller = _controller("http://replay-first")
        participant = controller.register("chat", tokens = 5600 + 1024, prompt_tokens = 5600)
        assert participant.replay_tokens() == 5600
        # The charge does not fit alone, the prompt does.
        assert controller.cannot_ever_fit(participant.tokens) is True
        assert controller.cannot_ever_fit(participant.replay_tokens()) is False

    def test_a_later_pause_replays_the_partial_too(self):
        controller = _controller("http://replay-later")
        participant = controller.register("chat", tokens = 5600 + 1024, prompt_tokens = 5600)
        controller.note_replayed("chat", 300)
        assert participant.replay_tokens() == 5900

    def test_with_no_split_the_charge_stands(self):
        controller = _controller("http://replay-nosplit")
        participant = controller.register("chat", tokens = 4000)
        assert participant.replay_tokens() == 4000

    def test_the_wait_asks_for_the_replay(self):
        source = " ".join(
            inspect.getsource(preemption.ControllerPreemptionPolicy.await_resume).split()
        )
        assert "want = max(0, int(participant.replay_tokens() or 0))" in source
        assert "participant.tokens or 0" not in source


class TestTheNextChoiceStartsAfresh:
    def test_restart_puts_the_ledger_back_to_the_charge(self):
        controller = _controller("http://choices")
        signal = preemption.PreemptSignal()
        participant = controller.register("gen", tokens = 6624, prompt_tokens = 5600, signal = signal)
        policy = preemption.ControllerPreemptionPolicy(controller, "gen", signal)
        controller.observe("gen", 200)
        controller.note_replayed("gen", 200)
        controller.set_state("gen", ParticipantState.PAUSED)
        signal.set()
        policy._resumes = 2
        assert participant.prompt_tokens == 5800

        policy.restart()

        assert policy._resumes == 0
        assert participant.prompt_tokens == 5600
        assert participant.base_tokens == 6624
        assert participant.tokens == 6624
        assert participant.generated_seen == 0
        assert participant.state == ParticipantState.DECODING
        assert signal.is_set() is False
        assert participant.measured is False

    def test_an_unbound_policy_restarts_nothing(self):
        preemption.DeferredPreemptionPolicy().restart()

    def test_every_choice_after_the_first_restarts_before_it_generates(self):
        source = " ".join(inspect.getsource(inference).split())
        site = source.index("for _idx in range(_n):")
        # To the loop's first generate rather than a fixed width: a guard added ahead of
        # the restart pushed `gguf_generate(_idx)` out of an 900-character window.
        window = source[site : source.index("gguf_generate(_idx)", site) + 40]
        assert "if _idx: " in window
        assert window.index("_plain_preempt_policy.restart()") < window.index("gguf_generate(_idx)")

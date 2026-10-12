# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A granted resume books the replay it sends, not the charge it was paused under.

A chat paused before its first token still carries admission's output allowance in its
charge, while its resume replays only the prompt and the lease commits only that. Keeping
the larger figure counted thousands of cells nobody holds through the whole resumed prefill,
so another chat was paused, or another resume refused, for room that was there.
"""

from __future__ import annotations

import pytest

from core.inference.llama_preemption import ParticipantState, PreemptionController

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _paused_before_its_first_token():
    controller = PreemptionController("http://127.0.0.1:65083")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    # 2000 of prompt under an 8000 charge: 6000 is output room.
    chat = controller.register("chat", tokens = 8000, prompt_tokens = 2000)
    controller.set_state("chat", ParticipantState.PAUSED)
    return controller, chat


def test_the_grant_books_the_replay():
    controller, chat = _paused_before_its_first_token()
    want = chat.replay_tokens()
    assert want == 2000
    assert controller.try_grant_resume("chat", want) is True
    assert chat.tokens == 2000, chat.tokens
    assert controller.snapshot().committed == 2000, (
        "the resumed prefill was counted with the output allowance the lease no longer holds"
    )


def test_a_restart_still_gets_the_original_charge_back():
    controller, chat = _paused_before_its_first_token()
    chat.charged_tokens, chat.charged_prompt_tokens = 8000, 2000
    controller.try_grant_resume("chat", chat.replay_tokens())
    controller.restart("chat")
    assert (chat.tokens, chat.prompt_tokens) == (8000, 2000)


def test_a_replayed_partial_is_still_booked():
    controller, chat = _paused_before_its_first_token()
    controller.note_replayed("chat", 500)
    want = chat.replay_tokens()
    assert want == 2500
    controller.try_grant_resume("chat", want)
    assert chat.tokens == 2500 and chat.prompt_tokens == 2500

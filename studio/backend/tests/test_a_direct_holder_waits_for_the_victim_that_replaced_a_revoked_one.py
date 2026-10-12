# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A direct holder waits on whoever replaced a victim whose choice was revoked.

A chat chosen as its stream finished a tool call has the choice undone, and the re-plan picks
another chat in its place. The holder only knew the first: it stopped waiting the moment that
one left PREEMPTING and went upstream while the replacement still decoded into its cells.
"""

import threading
import time

import pytest

from core.inference.llama_preemption import ParticipantState, PreemptionController

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _revoked_into_a_tool():
    controller = PreemptionController("http://127.0.0.1:65079")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    for gen_id, tokens in (("a", 8000), ("b", 7000)):
        controller.register(gen_id, tokens = tokens)
        controller.note_measured(gen_id)
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
    first = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
    assert len(first) == 1, first
    controller.note_state(first[0], ParticipantState.TOOLS_RUNNING)
    replacement = next(
        gen_id
        for gen_id, p in controller._participants.items()
        if p.state == ParticipantState.PREEMPTING
    )
    assert replacement != first[0]
    return controller, first[0], replacement


def test_the_wait_follows_the_replacement():
    controller, revoked, replacement = _revoked_into_a_tool()
    assert (
        controller.wait_released([revoked], 0.2) is False
    ), "the holder went upstream while the chat chosen in the revoked one's place decoded"


def test_it_ends_once_the_replacement_lets_go():
    controller, revoked, replacement = _revoked_into_a_tool()

    def pause():
        time.sleep(0.2)
        controller.set_state(replacement, ParticipantState.PAUSED)

    threading.Thread(target = pause, daemon = True).start()
    started = time.monotonic()
    assert controller.wait_released([revoked], 2.0) is True
    assert time.monotonic() - started >= 0.15



def test_a_victim_chosen_again_after_it_was_replaced_is_waited_on_again():
    # The first choice leaves its tool and decodes, then its replacement enters one and the
    # re-plan picks the first again: the wait has to see that choice too.
    controller, revoked, replacement = _revoked_into_a_tool()
    chosen_again = threading.Event()

    def churn():
        time.sleep(0.1)
        controller.note_state(revoked, ParticipantState.DECODING)
        controller.note_state(replacement, ParticipantState.TOOLS_RUNNING)
        if controller.participant(revoked).state == ParticipantState.PREEMPTING:
            chosen_again.set()
        time.sleep(0.4)
        controller.set_state(revoked, ParticipantState.PAUSED)

    threading.Thread(target = churn, daemon = True).start()
    started = time.monotonic()
    assert controller.wait_released([revoked], 2.0) is True
    assert chosen_again.is_set(), "the re-plan did not choose the first chat again"
    assert time.monotonic() - started >= 0.45, "the wait ended while it was chosen again"

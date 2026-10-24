# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A victim that declines its pause with nobody to stop in its place still holds its cells.
The unpausable holder must keep waiting rather than go upstream on room nobody freed."""

import threading
import time

import pytest

import routes.inference as inf_mod
from core.inference.llama_preemption import ParticipantState, PreemptionController

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _lone_victim_declined():
    controller = PreemptionController("http://127.0.0.1:65091")
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    controller.register("a", tokens = 6000)
    controller.note_measured("a")
    controller.note_resident(6000, started_at_seq = controller.residency_epoch())
    controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
    chosen = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
    assert chosen == ["a"], chosen
    controller.note_declined("a")
    assert controller.participant("a").state == ParticipantState.DECODING
    assert controller._replaced_by["a"] == []
    assert not controller.room_for("raw", 3000)
    return controller


def test_the_wait_does_not_end_while_the_declined_chat_holds_the_room():
    controller = _lone_victim_declined()
    assert (
        controller.wait_released(["a"], 0.2) is False
    ), "the holder went upstream on cells the declined chat still holds"


def test_the_wait_ends_once_the_declined_chat_finishes():
    controller = _lone_victim_declined()

    def finish():
        time.sleep(0.2)
        controller.unregister("a")

    threading.Thread(target = finish, daemon = True).start()
    started = time.monotonic()
    assert controller.wait_released(["a"], 2.0) is True
    assert time.monotonic() - started >= 0.15


def test_the_wait_ends_once_the_declined_chat_pauses_after_all():
    controller = _lone_victim_declined()

    def pause():
        time.sleep(0.1)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["a"]
        time.sleep(0.1)
        controller.set_state("a", ParticipantState.PAUSED)

    threading.Thread(target = pause, daemon = True).start()
    assert controller.wait_released(["a"], 2.0) is True


def test_the_route_wait_does_not_let_the_holder_through(monkeypatch):
    controller = _lone_victim_declined()
    monkeypatch.setattr(inf_mod, "_UNPAUSABLE_VICTIM_WAIT_S", 0.05)
    monkeypatch.setattr(inf_mod, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.3)
    controller.participant("raw").awaiting_victims = ["a"]
    assert inf_mod._openai_llama_wait_for_victims(controller, "raw") is False

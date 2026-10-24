# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio preemption pauses a chat only when the pause frees real room, and does not keep
pausing the chat it has just let back in. Live, four short chats asking for long answers at
ctx 4096 --parallel 4 paused one of them before its first token (granted again 4 ms later),
then again after 8 and 80 new tokens, each pause a whole re-prefill."""

from __future__ import annotations

import pytest

import routes.inference as inference
from core.inference.llama_preemption import (
    RESUME_GRACE_TOKENS,
    ParticipantState,
    PreemptionController,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

BASE = "http://127.0.0.1:65073"
PROMPT = 185
CHARGE = 1024


def _controller(key: str = BASE) -> PreemptionController:
    controller = PreemptionController(key)
    controller.configure(budget = 4096, kv_unified = True, slots = 4, batch_tokens = 0)
    return controller


def _ceiling(controller: PreemptionController) -> int:
    snapshot = controller.snapshot()
    return snapshot.budget - snapshot.buffer


def _chat(
    controller,
    gen_id,
    *,
    decoded = 0,
):
    participant = controller.register(gen_id, tokens = CHARGE, prompt_tokens = PROMPT)
    if decoded:
        controller.observe(gen_id, decoded, plan = False)
    return participant


def _pause_and_resume(controller, gen_id, *, decoded: int) -> int:
    """What `on_preempted` then a granted `await_resume` do to the ledger."""
    controller.note_replayed(gen_id, decoded)
    controller.set_state(gen_id, ParticipantState.PAUSED)
    want = controller.participant(gen_id).replay_tokens()
    assert controller.try_grant_resume(gen_id, want), "the test needs the resume granted"
    controller.note_resumed(gen_id)
    return want


class TestAChatWithNoTokenYetFreesNothing:
    def test_four_chats_armed_together_pause_nobody_before_their_first_token(self):
        controller = _controller()
        for gen_id in ("a", "b", "c", "d"):
            _chat(controller, gen_id)
        controller.note_resident(0, started_at_seq = controller.residency_epoch())
        assert controller.committed_tokens() > _ceiling(controller), "the ledger is over"
        victims = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
        assert victims == [], (
            "a chat paused before its first token holds no output and is granted its prompt "
            f"back at once; chose {victims}"
        )

    def test_once_they_decode_real_growth_still_pauses_the_newest(self):
        controller = _controller()
        for gen_id in ("a", "b", "c", "d"):
            _chat(controller, gen_id)
        for gen_id in ("a", "b", "c", "d"):
            controller.observe(gen_id, 600, plan = False)
        assert controller.plan_preemptions(needed = 0) == []
        for gen_id in ("a", "b", "c", "d"):
            controller.observe(gen_id, 700, plan = False)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["d"]

    def test_an_unstarted_chat_is_still_taken_when_nobody_else_can_be(self):
        controller = _controller()
        controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
        controller.register("chat", tokens = CHARGE, prompt_tokens = 900)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["chat"]

    def test_an_unstarted_chat_goes_after_one_that_has_decoded(self):
        controller = _controller()
        _chat(controller, "old", decoded = 1700)
        controller.register("new", tokens = 2500, prompt_tokens = 2000)
        controller.register("raw", tokens = 100, state = ParticipantState.STREAMING_RAW)
        victims = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
        assert victims and victims[0] == "old", victims


class TestAResumedChatIsNotPickedAgainAtOnce:
    def _resumed_beside_a_grower(self):
        controller = _controller()
        _chat(controller, "a", decoded = 1500)
        _chat(controller, "b", decoded = 800)
        _pause_and_resume(controller, "b", decoded = 800)
        controller.observe("b", 8, plan = False)
        controller.observe("a", 2200, plan = False)
        assert controller.committed_tokens() > _ceiling(controller), "real pressure"
        return controller

    def test_the_older_chat_is_paused_instead_while_the_resume_is_in_its_grace(self):
        controller = self._resumed_beside_a_grower()
        victims = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
        assert victims == [
            "a"
        ], f"newest-first re-picked the chat it had just let back in after 8 tokens: {victims}"

    def test_after_its_grace_it_is_newest_first_again(self):
        controller = self._resumed_beside_a_grower()
        controller.observe("b", 8 + RESUME_GRACE_TOKENS, plan = False)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["b"]

    def test_with_nobody_else_to_pause_the_resumed_chat_is_still_taken(self):
        controller = _controller()
        _chat(controller, "b", decoded = 800)
        _pause_and_resume(controller, "b", decoded = 800)
        controller.observe("b", 8, plan = False)
        controller.register("raw", tokens = 2385, state = ParticipantState.STREAMING_RAW)
        assert controller.committed_tokens() > _ceiling(controller)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["b"]

    def test_promotion_still_outranks_newest_first(self):
        controller = _controller()
        _chat(controller, "a", decoded = 1500)
        _chat(controller, "b", decoded = 800)
        controller.participant("b").consecutive_preemptions = 3
        controller.observe("a", 2200, plan = False)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["a"]

    def test_a_restart_carries_no_grace(self):
        controller = _controller()
        _chat(controller, "b", decoded = 800)
        _pause_and_resume(controller, "b", decoded = 800)
        assert controller.participant("b").resume_grace == RESUME_GRACE_TOKENS
        controller.restart("b")
        assert controller.participant("b").resume_grace == 0


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 4096


def test_a_resumed_chat_counted_twice_is_read_again_before_anyone_is_paused(monkeypatch):
    """The resumed chat's replay is in a reading taken before its first token, and counted on top
    of it until a later reading confirms it: a sweep in between paused somebody for room that
    exists. The token report has to take that reading first, inside the one-second throttle."""
    controller = _controller()
    _chat(controller, "a", decoded = 1500)
    controller.note_resident(PROMPT + 1500, started_at_seq = controller.residency_epoch())
    _chat(controller, "b", decoded = 884)
    want = _pause_and_resume(controller, "b", decoded = 884)
    resident = {"value": PROMPT + 1500 + want}
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
    monkeypatch.setattr(
        inference,
        "read_slot_occupancy",
        lambda scrape: {"idle": [], "resident": resident["value"], "idle_tokens": 0},
    )
    monkeypatch.setattr(inference, "reclaim_idle_slots", lambda *a, **k: 0)
    refresh, observe, _note = inference._openai_llama_residency_observer(
        llama_backend = _Backend(), completion_id = "b"
    )
    # The prefill landed and a reading saw it, before the first token arrived.
    refresh(controller)
    resident["value"] += 8
    observe(8)
    for gen_id in ("a", "b"):
        participant = controller.participant(gen_id)
        assert participant.state != ParticipantState.PREEMPTING, (
            f"{gen_id} was paused for room the cache had: committed "
            f"{controller.committed_tokens()} against {_ceiling(controller)}"
        )
        assert not participant.preempt_event.is_set()
    assert controller.participant("b").measured, "the forced reading confirms the replay"

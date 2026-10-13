# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A contended token report reads the cache and erases idle residue before it chooses a victim.

The cached reading can still count a finished request's idle slot. Planned off it, a live chat is
signalled, and the erase that follows frees the room it was paused for; nothing takes the signal
back, so the chat is aborted and replayed for nothing.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

BASE = "http://127.0.0.1:65072"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 8192


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _two_chats() -> PreemptionController:
    controller = PreemptionController(BASE)
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    for gen_id in ("live", "other"):
        controller.register(gen_id, lease = _Lease(), tokens = 2500)
        controller.note_measured(gen_id)
    controller.note_resident(9000, 3000, started_at_seq = controller.residency_epoch())
    return controller


def _observer(monkeypatch, controller, erased: list):
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
    monkeypatch.setattr(
        inference,
        "read_slot_occupancy",
        lambda scrape: {"idle": [(3, 3000)], "resident": 9000, "idle_tokens": 3000},
    )

    def reclaim(
        occupancy,
        erase,
        *,
        needed = 0,
        **_kwargs,
    ):
        erased.append(needed)
        return 3000

    monkeypatch.setattr(inference, "reclaim_idle_slots", reclaim)
    _refresh, observe, _note = inference._openai_llama_residency_observer(
        llama_backend = _Backend(), completion_id = "live"
    )
    return observe


def test_the_erase_lands_before_anybody_is_signalled(monkeypatch):
    controller = _two_chats()
    erased: list = []
    observe = _observer(monkeypatch, controller, erased)
    observe(100)
    assert erased, "the idle slot was never reclaimed"
    for gen_id in ("live", "other"):
        participant = controller.participant(gen_id)
        assert (
            participant.state != ParticipantState.PREEMPTING
        ), f"{gen_id} was signalled for room the idle erase gave back"
        assert not participant.preempt_event.is_set()


def test_a_report_with_nobody_else_still_plans_inline():
    controller = PreemptionController(BASE)
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    controller.register("solo", lease = _Lease(), tokens = 2500)
    assert controller.contended() is False
    assert controller.observe("solo", 10) == []
    assert controller.observe("solo", 20, plan = False) == []
    assert controller.participant("solo").generated_seen == 20

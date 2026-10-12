# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An idle-slot erase goes only to the slot the reading saw idle, on the task it saw.

llama-server defers an erase aimed at a processing slot and runs it once that slot goes idle
again, so a slot handed to another request after the `/slots` reading had the newer request's
cache erased when it parked or finished. That holder was not in the reading, so its next round
reserved only its growth while the whole conversation had to be prefilled again.
"""

from __future__ import annotations

from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
    reclaim_idle_slots,
    read_slot_occupancy,
)
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


BASE = "http://127.0.0.1:65083"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 8192


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _slot(slot_id, *, task, processing, tokens):
    return {
        "id": slot_id,
        "id_task": task,
        "is_processing": processing,
        "n_prompt_tokens": tokens,
    }


def _controller() -> PreemptionController:
    made = PreemptionController(BASE)
    made.configure(budget = 8192, kv_unified = True, slots = 4)
    return made


def _observe(monkeypatch, controller, readings, erased):
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: readings.pop(0))

    def erase(
        base,
        slot_id,
        timeout_s = 3.0,
        headers = None,
    ):
        erased.append(slot_id)
        return 3000

    monkeypatch.setattr(inference, "erase_llama_slot", erase)
    refresh, _observe_tokens, _note = inference._openai_llama_residency_observer(
        llama_backend = _Backend(), completion_id = "live"
    )
    refresh(controller, force = True)


def test_a_slot_handed_to_another_request_is_not_erased(monkeypatch):
    controller = _controller()
    controller.register("live", lease = _Lease(), tokens = 6000)
    reading = [
        _slot(1, task = 7, processing = False, tokens = 3000),
        _slot(0, task = 5, processing = True, tokens = 6000),
    ]
    # By the erase, slot 1 runs another request's task.
    taken = [
        _slot(1, task = 9, processing = True, tokens = 200),
        _slot(0, task = 5, processing = True, tokens = 6000),
    ]
    erased: list = []
    _observe(monkeypatch, controller, [reading, taken, taken, taken], erased)
    assert erased == [], "the erase was sent to a slot another request now holds"


def test_a_slot_still_idle_on_the_same_task_is_erased(monkeypatch):
    controller = _controller()
    controller.register("live", lease = _Lease(), tokens = 6000)
    reading = [
        _slot(1, task = 7, processing = False, tokens = 3000),
        _slot(0, task = 5, processing = True, tokens = 6000),
    ]
    erased: list = []
    _observe(monkeypatch, controller, [reading, reading, reading, reading], erased)
    assert erased == [1]


def test_a_newer_task_seen_after_the_erase_replays_the_holder_parked_since(monkeypatch):
    # Idle on the reading's task when checked, then handed over before the erase ran: the
    # chat that parked meanwhile may have lost its cells, so its next round replays them.
    controller = _controller()
    controller.register("live", lease = _Lease(), tokens = 6000)
    controller.register("later", lease = _Lease(), tokens = 1000)
    occupancy = read_slot_occupancy(lambda: [_slot(1, task = 7, processing = False, tokens = 3000)])
    readings = [
        [_slot(1, task = 7, processing = False, tokens = 3000)],
        [_slot(1, task = 9, processing = False, tokens = 1000)],
    ]
    ambiguous: list = []
    parked_before = controller.parked_holders()
    controller.note_state("later", ParticipantState.PARKED_ON_TOOL)
    freed = reclaim_idle_slots(
        occupancy,
        lambda slot_id: 1000,
        needed = 3000,
        recheck = lambda: readings.pop(0),
        ambiguous = ambiguous,
    )
    assert freed == 1000 and ambiguous == [1]
    controller.note_cells_possibly_erased(parked_before)
    assert controller.participant("later").replay_pending is True


def test_an_unreadable_recheck_erases_nothing():
    occupancy = read_slot_occupancy(lambda: [_slot(1, task = 7, processing = False, tokens = 3000)])
    sent: list = []
    freed = reclaim_idle_slots(
        occupancy, lambda slot_id: sent.append(slot_id) or 3000, needed = 3000, recheck = lambda: None
    )
    assert freed == 0 and sent == []

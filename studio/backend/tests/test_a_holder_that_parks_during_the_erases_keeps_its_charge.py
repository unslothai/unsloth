# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reclaim releases the holders its OWN reading saw parked, and nobody else.

`/slots` is scraped, the idle slots are erased over blocking HTTP that can take seconds, and
only then are the parked holders told their cells are gone. A chat that stops on a tool
inside that window has an idle slot the erase list never contained: its cells are still
resident, so handing its admission commitment back admits a waiter into KV that is occupied,
which is the shared-cache overflow this whole path exists to prevent.
"""

from __future__ import annotations

import inspect

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference


BASE = "http://127.0.0.1:65041"


class _Lease:
    def __init__(self):
        self.yielded = 0

    def yield_parked_commitment(self):
        self.yielded += 1
        return 1000


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 16384


def _controller() -> PreemptionController:
    made = PreemptionController(BASE)
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


class TestTheLedger:
    def test_a_holder_outside_the_reading_keeps_its_cells(self):
        c = _controller()
        early_lease, late_lease = _Lease(), _Lease()
        c.register("early", lease = early_lease, tokens = 2000)
        c.register("late", lease = late_lease, tokens = 2000)
        c.note_state("early", ParticipantState.PARKED_ON_TOOL)

        # What the sweep reads before it scrapes `/slots`.
        seen = c.parked_holders()
        assert seen == {"early": c.participant("early").park_seq}

        # And now, while the erases are in flight, the other chat stops on a tool.
        c.note_state("late", ParticipantState.TOOLS_RUNNING)

        assert c.note_cells_reclaimed(seen) == 1
        assert c.participant("early").cells_reclaimed is True
        assert early_lease.yielded == 1
        assert (
            c.participant("late").cells_reclaimed is False
        ), "no erase touched its slot; its cells are resident and its charge must stand"
        assert late_lease.yielded == 0
        assert c.participant("late").holds_kv is True

    def test_a_holder_that_parked_again_inside_the_window_keeps_its_charge(self):
        c = _controller()
        lease = _Lease()
        c.register("chat", lease = lease, tokens = 2000)
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        seen = c.parked_holders()
        # Its tool returned, it decoded (llama-server prefilled the prompt back in), and it
        # stopped on the next tool, all inside the erase window.
        c.note_state("chat", ParticipantState.DECODING)
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)

        assert c.note_cells_reclaimed(seen) == 0
        assert lease.yielded == 0
        assert c.participant("chat").holds_kv is True

    def test_the_reading_still_releases_what_it_covered(self):
        c = _controller()
        lease = _Lease()
        c.register("chat", lease = lease, tokens = 2000)
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        seen = c.parked_holders()

        assert c.note_cells_reclaimed(seen) == 1
        assert lease.yielded == 1
        assert c.participant("chat").holds_kv is False

    def test_no_reading_releases_everyone_as_before(self):
        c = _controller()
        c.register("chat", lease = _Lease(), tokens = 2000)
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        assert c.note_cells_reclaimed() == 1


class TestTheDisarmPath:
    """Its erases are the slowest of the three, by its own comment: seconds each."""

    def test_a_chat_that_parks_mid_erase_is_not_released(self, monkeypatch):
        controller = _controller()
        early_lease, late_lease = _Lease(), _Lease()
        controller.register("leaving", lease = _Lease(), tokens = 2000)
        controller.register("early", lease = early_lease, tokens = 2000)
        controller.register("late", lease = late_lease, tokens = 2000)
        controller.note_state("early", ParticipantState.PARKED_ON_TOOL)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
        readings = iter(
            [
                {"idle": [(0, 2000)], "resident": 6000, "idle_tokens": 2000},
                {"idle": [], "resident": 4000, "idle_tokens": 0},
            ]
        )
        monkeypatch.setattr(inference, "read_slot_occupancy", lambda scrape: next(readings))

        def _reclaim(
            occupancy,
            erase,
            *,
            needed = 0,
        ):
            # The window: this is the blocking HTTP the comment describes.
            controller.note_state("late", ParticipantState.PARKED_ON_TOOL)
            return 2000

        monkeypatch.setattr(inference, "reclaim_idle_slots", _reclaim)
        monkeypatch.setattr(inference, "erase_llama_slot", lambda base, slot_id, headers = None: 2000)

        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")

        assert controller.participant("early").cells_reclaimed is True
        assert early_lease.yielded == 1
        assert (
            controller.participant("late").cells_reclaimed is False
        ), "it parked after the scrape, so its cells were never in the erase list"
        assert late_lease.yielded == 0

    def test_the_token_path_reclaims_against_its_own_reading(self):
        """The sweep erases from a `/slots` snapshot up to a second old, so the holders it
        may release are the ones that snapshot was taken beside."""
        source = inspect.getsource(inference._openai_llama_residency_observer)
        for call in ("controller.note_cells_reclaimed(", "note_cells_reclaimed("):
            assert f"{call})" not in source, (
                "a release with no reading gives away every parked holder's commitment, "
                "including one that parked while the erases ran"
            )

    def test_the_reading_is_taken_before_the_scrape(self):
        """Read after it, a chat parking in between would be released for cells the
        scrape never counted, which is the same bug one line further along."""
        source = inspect.getsource(inference._openai_llama_preemption_disarm)
        assert source.index("parked_holders()") < source.index("read_slot_occupancy(")


class _ChargingLease:
    """A lease that re-charges: what `recost` does when a tool comes back mid-erase."""

    def __init__(self):
        self.charge_seq = 0
        self.asked_at = []

    def yield_parked_commitment(self, *, charged_at = None):
        self.asked_at.append(charged_at)
        if charged_at is not None and charged_at != self.charge_seq:
            return 0
        return 1000


class TestAHolderWhoseToolCameBackDuringTheErases:
    """The yields run after erases that take seconds. A holder that restated its prompt in
    between is prefilling those cells again, and its new charge is not for the erased ones."""

    def test_the_yield_carries_the_charge_it_was_decided_on(self):
        c = _controller()
        lease = _ChargingLease()
        c.register("chat", lease = lease, tokens = 6000)
        c.note_state("chat", ParticipantState.TOOLS_RUNNING)
        assert c.note_cells_reclaimed() == 1
        assert lease.asked_at == [0]

    def test_a_holder_restated_before_its_yield_is_skipped(self):
        # Two holders: the first's yield is where the interleaving lands, and it plays the
        # second's tool coming back (recost, then the round boundary) before its own yield runs.
        c = _controller()
        first, second = _ChargingLease(), _ChargingLease()
        c.register("first", lease = first, tokens = 2000)
        c.register("second", lease = second, tokens = 6000)
        c.note_state("first", ParticipantState.TOOLS_RUNNING)
        c.note_state("second", ParticipantState.TOOLS_RUNNING)

        def came_back(*, charged_at = None):
            first.asked_at.append(charged_at)
            second.charge_seq += 1
            c.note_tokens("second", 6200)
            return 1000

        first.yield_parked_commitment = came_back
        assert c.note_cells_reclaimed() == 2
        assert first.asked_at == [0]
        assert second.asked_at == [], "restated its prompt: nothing to hand back"
        assert c.participant("second").cells_reclaimed is False
        assert c.snapshot().prefilling == 6200

    def test_the_real_lease_refuses_a_stale_sequence(self):
        from core.inference.llama_admission import LlamaAdmissionLease

        class _Queue:
            def __init__(self):
                self.yielded = []

            def try_recost(self, old, new):
                return True

            def yield_commitment(self, tokens):
                self.yielded.append(tokens)

            def abandon_repark(self, restore = 0):
                pass

        queue = _Queue()
        lease = LlamaAdmissionLease(queue, slot = 0, tokens = 6000)
        decided_on = lease.charge_seq
        assert lease.recost(6200) is True
        assert lease.charge_seq == decided_on + 1
        assert lease.yield_parked_commitment(charged_at = decided_on) == 0
        assert lease.tokens == 6200
        assert queue.yielded == []
        assert lease.yield_parked_commitment(charged_at = lease.charge_seq) == 6200
        assert queue.yielded == [6200]
        assert lease.yield_parked_commitment() == 0

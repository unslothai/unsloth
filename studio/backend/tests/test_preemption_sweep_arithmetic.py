# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two places the sweep can free room it has not actually freed.

Both are arithmetic, and both end the same way: the planner reports it made space, the
prefill that triggered it goes ahead, and the cache is still full. That is the
``Context size has been exceeded`` this whole mechanism exists to remove, arrived at by
believing the ledger rather than the cache.
"""

from __future__ import annotations

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import ParticipantState, reclaim_idle_slots


def _controller(key: str) -> preemption.PreemptionController:
    controller = preemption.PreemptionController(key)
    controller.configure(
        budget = 16384,
        kv_unified = True,
        draft_tokens = 0,
        slots = 4,
        batch_tokens = 0,
    )
    return controller


class _Lease:
    def __init__(self, tokens: int):
        self.tokens = tokens
        self.is_released = False


class TestAReclaimedHolderIsNotAVictim:
    """It holds no cells, so pausing it frees none.

    ``note_cells_reclaimed`` flips ``cells_reclaimed``, which takes a participant out of
    ``holds_kv`` and therefore out of ``_committed_locked``. Its STATE is still
    PARKED_ON_TOOL, so ``preemptable`` stayed True and the planner picked it first --
    then subtracted its stale ``tokens`` from a total that had never included them, and
    stopped, satisfied, without choosing the live decoder that actually had the room.
    """

    def test_a_parked_holder_whose_cells_were_erased_is_not_chosen(self):
        """Ceiling 15616 of a 16384 cache. Three holders totalling 16000 are over it,
        but 9000 of that belongs to a parked holder whose cells were already erased and
        which is therefore already out of ``committed``.
        """
        controller = _controller("http://sweep-1")
        controller.register("parked", lease = _Lease(9000), tokens = 9000)
        controller.register("live-a", lease = _Lease(8000), tokens = 8000)
        controller.register("live-b", lease = _Lease(8000), tokens = 8000)
        controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        controller.note_state("live-a", ParticipantState.DECODING)
        controller.note_state("live-b", ParticipantState.DECODING)
        # Its slot was idle, so an idle reclaim took its cells.
        controller.note_cells_reclaimed()
        assert controller.committed_tokens() > 15616, "the sweep has to actually be under pressure"

        victims = controller.plan_preemptions(needed = 0)

        chosen = [v.gen_id for v in victims]
        assert "parked" not in chosen, (
            "a holder whose cells are already gone frees nothing; choosing it spends the "
            "sweep on a chat that cannot give anything back"
        )
        assert chosen, "somebody with cells still has to stop"
        assert not controller.participant("parked").preempt_event.is_set()

    def test_the_live_decoder_is_still_reachable_behind_it(self):
        """The consequence of the above, and the reason it matters.

        Parked holders sort FIRST -- they are the cheapest room -- so a reclaimed one
        absorbed the whole shortfall on paper and the loop broke before it ever looked
        at a chat that was decoding. Its 9000 came off a total that never contained
        them, so 16000 became 7000 and the sweep declared itself done having freed
        nothing at all.
        """
        controller = _controller("http://sweep-2")
        controller.register("parked", lease = _Lease(9000), tokens = 9000)
        controller.register("live-a", lease = _Lease(8000), tokens = 8000)
        controller.register("live-b", lease = _Lease(8000), tokens = 8000)
        controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        controller.note_state("live-a", ParticipantState.DECODING)
        controller.note_state("live-b", ParticipantState.DECODING)
        controller.note_cells_reclaimed()

        victims = controller.plan_preemptions(needed = 0)

        assert [v.gen_id for v in victims] == [
            "live-b"
        ], "newest-first among the holders that still have cells, with one left standing"


class TestAPartialReclaimDoesNotFreeEverybody:
    """``reclaim_idle_slots`` stops at ``needed``; ``note_cells_reclaimed`` does not.

    The erase loop breaks as soon as it has freed what was asked for, so a cache holding
    three idle slots can lose one. ``note_cells_reclaimed`` is global: it marks EVERY
    parked and tools-running holder as having lost its cells and makes each hand its
    admission commitment back. Applied after a partial erase, that gives away room that
    is still physically occupied.
    """

    def test_the_erase_loop_stops_once_it_has_what_it_needed(self):
        occupancy = {
            "resident": 9000,
            "idle_tokens": 9000,
            "idle": [(0, 3000), (1, 3000), (2, 3000)],
        }
        erased: list[int] = []

        def _erase(slot_id: int) -> int:
            erased.append(slot_id)
            return 3000

        freed = reclaim_idle_slots(occupancy, _erase, needed = 3000)

        assert freed == 3000
        assert erased == [0], "the other two slots are still holding their cells"
        assert freed < occupancy["idle_tokens"], (
            "which is exactly the condition the caller has to check before telling the "
            "ledger that every parked holder lost its cells"
        )

    def test_the_route_only_reports_a_reclaim_that_took_every_idle_slot(self):
        import routes.inference as inference
        from pathlib import Path

        source = Path(inference.__file__).read_text(encoding = "utf-8")
        assert (
            "if freed >= _idle_tokens:\n                            controller.note_cells_reclaimed()"
            in source
        ), (
            "note_cells_reclaimed is global, so it may only follow a reclaim that erased "
            "the whole idle residue"
        )

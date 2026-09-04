# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A finished chat keeps its prompt cache when nobody else wants the room.

WHAT WENT WRONG

`_openai_llama_preemption_disarm` erases llama-server's idle slots, because a ledger that
says there is room while the cache still holds the tokens is the crash rather than a stall
(`TestDroppingTheChargeWithoutTheCellsIsWorseThanNeither` has the measurements). While the
disarm ran on the tool path alone that erase was rare. Arming the plain, non-streaming and
Anthropic surfaces made it run after EVERY ordinary chat, and an idle slot's tokens are
exactly the prefix the next turn of the same conversation reuses.

CI caught it in one pass: `.github/scripts/assert-prompt-cache.sh api` sends two turns that
share a long system preamble and asserts turn two reports
`usage.prompt_tokens_details.cached_tokens > 0`. It came back zero. llama.cpp had a complete
prefix hit to offer and Studio had deleted it between the turns.

THE RULE THESE TESTS PIN

Unregister always, because the ledger has to be exact. Erase only when the cells are wanted.
With this generation already unregistered, `committed == 0` means it was the only chat in the
cache, so there is nobody to hand the room to and the cells are worth more as a prefix.
Should that change, the watermark sweep reclaims on its next pass; nothing depends on the
disarm being the only reclaimer.

The tests are behavioural on purpose. The condition is one `if` away from being deleted by a
refactor that still passes every source-level assertion in the wiring suite.
"""

import pytest

import routes.inference as inference
from core.inference.llama_preemption import ParticipantState, PreemptionController


BASE = "http://127.0.0.1:65001"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 16384


class _Lease:
    def __init__(self, tokens = 2000):
        self.tokens = tokens
        self.slot = 0
        self.finished = False


@pytest.fixture
def erasures(monkeypatch):
    """Record every slot llama-server would have been asked to forget."""
    seen = []

    monkeypatch.setattr(
        inference,
        "fetch_llama_slots",
        lambda base: [{"id": 0, "is_processing": False, "n_ctx_used": 2000}],
    )
    monkeypatch.setattr(
        inference,
        "read_slot_occupancy",
        lambda scrape: {"idle": [0], "resident": 2000, "idle_tokens": 2000},
    )

    def _reclaim(occupancy, erase, *, needed = 0):
        for slot_id in occupancy.get("idle", []):
            seen.append(slot_id)
            erase(slot_id)
        return 2000

    monkeypatch.setattr(inference, "reclaim_idle_slots", _reclaim)
    monkeypatch.setattr(inference, "erase_llama_slot", lambda base, slot_id: True)
    return seen


@pytest.fixture
def controller(monkeypatch):
    """A real controller on a key no other test uses, wired to the disarm's lookup."""
    made = PreemptionController(BASE)
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: made)
    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
    return made


class TestTheSingleUserCaseCIChecks:
    def test_a_lone_finished_chat_keeps_its_cells(self, controller, erasures):
        """Turn one of the two-turn probe, ending with nobody else in the cache."""
        controller.register("only-chat", lease = _Lease(), tokens = 2000)
        inference._openai_llama_preemption_disarm(
            llama_backend = _Backend(), gen_id = "only-chat"
        )
        assert controller.snapshot().committed == 0, "the charge must still be dropped"
        assert erasures == [], (
            "the prompt cache turn two would have reused was erased. This is the "
            "cached_tokens=0 failure, reproduced without a server."
        )

    def test_the_ledger_is_exact_either_way(self, controller, erasures):
        """Skipping the erase must never mean skipping the unregister.

        An over-counted ledger is the pessimistic failure, but it is still a failure: it
        serialises chats that would have fit.
        """
        controller.register("gone", lease = _Lease(), tokens = 2000)
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "gone")
        snapshot = controller.snapshot()
        assert snapshot.committed == 0
        assert snapshot.decoding == 0


class TestTheContendedCaseStillReclaims:
    """The behaviour the measurements above bought must survive unchanged."""

    def test_a_paused_chat_gets_the_room(self, controller, erasures):
        controller.register("leaving", lease = _Lease(), tokens = 2000)
        controller.register("waiting", lease = _Lease(), tokens = 3000)
        controller.set_state("waiting", ParticipantState.PAUSED)
        inference._openai_llama_preemption_disarm(
            llama_backend = _Backend(), gen_id = "leaving"
        )
        assert erasures == [0], (
            "somebody is paused for want of cells and the finished chat's were left in "
            "place, which is the livelock this reclaim exists to end"
        )

    def test_another_decoding_chat_counts_as_contention(self, controller, erasures):
        controller.register("leaving", lease = _Lease(), tokens = 2000)
        controller.register("busy", lease = _Lease(), tokens = 3000)
        controller.set_state("busy", ParticipantState.DECODING)
        inference._openai_llama_preemption_disarm(
            llama_backend = _Backend(), gen_id = "leaving"
        )
        assert erasures == [0]

    def test_a_raw_stream_counts_too(self, controller, erasures):
        """The counted-but-never-chosen surfaces hold KV without ever being preempted.

        They are invisible to `decoding`/`paused`/`parked`, which is why the condition is
        written on `committed`: it sums everything in `_HOLDS_KV`.
        """
        controller.register("leaving", lease = _Lease(), tokens = 2000)
        controller.register(
            "passthrough",
            lease = _Lease(),
            tokens = 3000,
            state = ParticipantState.STREAMING_RAW,
        )
        inference._openai_llama_preemption_disarm(
            llama_backend = _Backend(), gen_id = "leaving"
        )
        assert erasures == [0]


class TestItStillCannotFailAResponse:
    def test_a_broken_controller_is_swallowed(self, monkeypatch, erasures):
        def _boom(key):
            raise RuntimeError("no controller")

        monkeypatch.setattr(inference, "get_preemption_controller", _boom)
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "x")

    def test_a_backend_without_a_base_url_is_swallowed(self, controller, erasures):
        class _Headless:
            base_url = ""
            _kv_cache_unified = True

        inference._openai_llama_preemption_disarm(llama_backend = _Headless(), gen_id = "x")
        assert erasures == []

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A quarter of the cache was held back for a prefill that was not happening.

WHAT WENT WRONG

`preemption_buffer_tokens` reserved a whole `--batch-size` at all times. At the shape
Studio actually ships -- `-c 8192`, four slots, `--batch-size 2048`, MTP with two drafts
-- that is `max(192 * 4, 2048) + 2 * 4 = 2056`, so the shared ceiling sat at 6136 and
2056 cells were unusable even when every chat was decoding one token at a time and
nothing was prefilling at all. The user's requirement is the opposite: with a context of
N and P chats, every chat may use N minus a buffer, and the buffer must be as small as
possible.

The batch term itself was right, and the run that forced it is on
`TestTheBufferCanHoldOnePrefillChunk`: llama-server prefills in chunks and a chunk has to
find its cells at once, so a buffer smaller than a chunk cannot keep the cache off the
shrinking-batch retry where the speculative sub-batch bug lives. What was wrong was
WHEN. Decoding submits one token per slot per step. Only a prompt submission needs a
chunk, and there are exactly three of those, all of which pass through this module:

  1. a freshly admitted prompt              -> `register`
  2. a granted resume replaying its partial -> `try_grant_resume`
  3. a tool round whose prompt grew         -> `note_tokens`

THE RULES THIS PINS

1. Nothing pending, no batch term: the buffer is reaction headroom plus drafts.
2. A pending prompt SHORTER than a chunk reserves its own length, not a whole chunk.
   llama-server fills its batch with `min(n_batch - batch.size(), remaining)` per slot
   and a partial chunk is the normal case (`server-context.cpp`, `update_slots`).
3. A pending resume LARGER than a chunk reserves one chunk, and no more.
4. The reserve is up BEFORE the chunk is submitted, and a sweep fired by another chat's
   tokens sees it. This is the race the whole design turns on.
5. `UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH=1` restores the permanent term exactly.
6. `plan_preemptions` uses the lower ceiling once nothing is pending, so the cells the
   batch term used to hold are really handed out.
"""

from __future__ import annotations

import pytest

from core.inference.llama_preemption import (
    DEFAULT_PREEMPT_BUFFER_MIN_TOKENS,
    PENDING_PREFILL_TTL_S,
    ParticipantState,
    PreemptionController,
    PreemptSignal,
    preemption_buffer_tokens,
)

# The shipped shape, and the one every figure in the docstring above is quoted at.
BUDGET = 8192
SLOTS = 4
N_BATCH = 2048
DRAFTS = 2
# 192 * 4 reaction headroom, drafts on top. The buffer with nothing prefilling.
IDLE_BUFFER = 192 * SLOTS + DRAFTS * SLOTS
# max(768, 2048) + 8. The buffer while a chunk is in flight, and what used to be held
# back permanently.
PREFILL_BUFFER = N_BATCH + DRAFTS * SLOTS


def _controller(key: str = "test://buffer") -> PreemptionController:
    made = PreemptionController(key)
    made.configure(
        budget = BUDGET,
        kv_unified = True,
        slots = SLOTS,
        draft_tokens = DRAFTS,
        batch_tokens = N_BATCH,
    )
    return made


def _buffer(**kw) -> int:
    return preemption_buffer_tokens(
        BUDGET, slots = SLOTS, draft_tokens = DRAFTS, batch_tokens = N_BATCH, **kw
    )


class TestTheBufferWithNothingPending:
    def test_the_function_holds_back_only_reaction_headroom_and_drafts(self):
        assert _buffer() == IDLE_BUFFER == 776

    def test_a_controller_of_pure_decoders_holds_the_same(self):
        """The case the old formula charged 2056 for: four chats, all mid-answer."""
        c = _controller()
        for i in range(SLOTS):
            c.register(f"chat{i}", tokens = 800, signal = PreemptSignal())
            # A token proves the prompt is in, which is what retires the reserve.
            c.observe(f"chat{i}", 32)
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == IDLE_BUFFER

    def test_that_is_1280_cells_the_old_policy_kept_from_the_chats(self):
        """The whole point of the change, stated as the number the user asked about."""
        assert PREFILL_BUFFER - IDLE_BUFFER == 1280
        assert BUDGET - IDLE_BUFFER == 7416
        assert BUDGET - PREFILL_BUFFER == 6136


class TestOnePendingSmallPrompt:
    def test_a_short_prompt_reserves_its_own_length_not_a_chunk(self):
        """Under the reaction headroom it costs nothing extra at all."""
        assert _buffer(pending_prefill = 300) == IDLE_BUFFER

    def test_a_prompt_between_the_headroom_and_a_chunk_reserves_itself(self):
        assert _buffer(pending_prefill = 1200) == 1200 + DRAFTS * SLOTS

    def test_registering_a_prompt_announces_it(self):
        c = _controller()
        c.register("chat", tokens = 1200, signal = PreemptSignal())
        snapshot = c.snapshot()
        assert snapshot.prefilling == 1200
        assert snapshot.buffer == 1200 + DRAFTS * SLOTS

    def test_the_first_token_retires_the_reserve(self):
        c = _controller()
        c.register("chat", tokens = 1200, signal = PreemptSignal())
        assert c.snapshot().buffer > IDLE_BUFFER
        c.observe("chat", 1)
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == IDLE_BUFFER

    def test_a_round_boundary_sweep_does_not_retire_it(self):
        """`_gguf_recost` calls `observe(gen_id, 0)` right after announcing the growth.

        Zero generated is not proof of anything, and clearing on it would take the
        reserve away in the very same call that asked for it.
        """
        c = _controller()
        c.register("chat", tokens = 1000, signal = PreemptSignal())
        c.observe("chat", 40)
        # It holds 1040 (its prompt and what it generated), so a 2500 token round
        # submits the 1460 that are new.
        c.note_tokens("chat", 2500)
        assert c.snapshot().prefilling == 1460
        c.observe("chat", 0)
        assert c.snapshot().prefilling == 1460

    def test_a_round_boundary_announces_only_the_growth(self):
        """A 6000 token history that grew by 40 submits 40 tokens, not 6040."""
        c = _controller()
        c.register("chat", tokens = 6000, signal = PreemptSignal())
        c.observe("chat", 10)
        # 6010 held, 6040 stated: 30 new tokens, not 6040.
        c.note_tokens("chat", 6040)
        assert c.snapshot().prefilling == 30
        assert c.snapshot().buffer == IDLE_BUFFER

    def test_a_round_that_shrank_announces_nothing(self):
        c = _controller()
        c.register("chat", tokens = 6000, signal = PreemptSignal())
        c.observe("chat", 10)
        c.note_tokens("chat", 3000)
        assert c.snapshot().prefilling == 0


class TestAPendingResumeLargerThanTheBatch:
    def test_the_reserve_is_one_chunk_and_no_more(self):
        assert _buffer(pending_prefill = 5000) == PREFILL_BUFFER

    def test_granting_a_resume_announces_the_replay(self):
        """A resumed chat replays its whole partial as prompt; see `try_grant_resume`."""
        c = _controller()
        paused = c.register("paused", tokens = 3000, signal = PreemptSignal())
        c.observe("paused", 10)
        c.set_state("paused", ParticipantState.PAUSED)
        assert c.snapshot().prefilling == 0, "a paused chat submits nothing"
        assert c.try_grant_resume("paused", 3200) is True
        assert paused.state == ParticipantState.DECODING
        assert c.snapshot().prefilling == 3200
        assert c.snapshot().buffer == PREFILL_BUFFER

    def test_a_grant_that_never_resumed_gives_the_reserve_back(self):
        c = _controller()
        c.register("paused", tokens = 3000, signal = PreemptSignal())
        c.observe("paused", 10)
        c.set_state("paused", ParticipantState.PAUSED)
        assert c.try_grant_resume("paused", 3200) is True
        c.note_resume_failed("paused")
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == IDLE_BUFFER

    def test_several_prefills_at_once_still_share_one_chunk(self):
        """llama-server builds ONE batch across every prompt slot, capped at n_batch."""
        c = _controller()
        c.register("a", tokens = 900, signal = PreemptSignal())
        c.register("b", tokens = 900, signal = PreemptSignal())
        c.register("c", tokens = 900, signal = PreemptSignal())
        assert c.snapshot().prefilling == 2700
        assert c.snapshot().buffer == PREFILL_BUFFER

    def test_a_stale_announcement_expires(self):
        """The backstop for a prefill that is announced and then never submitted."""
        c = _controller()
        chat = c.register("chat", tokens = 5000, signal = PreemptSignal())
        assert c.snapshot().buffer == PREFILL_BUFFER
        chat.pending_prefill_at -= PENDING_PREFILL_TTL_S + 1.0
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == IDLE_BUFFER


class TestTheRace:
    """A pending participant must raise the buffer BEFORE its chunk is submitted.

    The whole design rests on this ordering. `register` announces inside the same lock
    that adds the participant, and `_openai_llama_preemption_arm` sweeps immediately
    afterwards, so the arriving chat's own batch is already reserved when the sweep runs.
    """

    def test_the_sweep_that_arms_a_new_chat_plans_at_the_raised_ceiling(self):
        c = _controller()
        c.register("incumbent", tokens = 5800, signal = PreemptSignal())
        c.observe("incumbent", 0)
        # 5800 alone is under the idle ceiling of 7416, so nothing is chosen yet.
        assert c.observe("incumbent", 100) == []
        # A second chat arrives with a 2000 token prompt. Its own prefill raises the
        # buffer to a full chunk, which is what makes 5900 + 2000 too much.
        c.register("arriving", tokens = 2000, signal = PreemptSignal())
        victims = c.plan_preemptions(needed = 0)
        assert victims, "the arriving chat's chunk was not reserved before the sweep"

    def test_another_chats_tokens_see_the_pending_reserve(self):
        """The sweep runs on whoever generated, not on whoever is prefilling."""
        c = _controller()
        c.register("decoder", tokens = 5800, signal = PreemptSignal())
        c.observe("decoder", 100)
        c.register("prefiller", tokens = 2000, signal = PreemptSignal())
        assert c.snapshot().prefilling == 2000
        # `decoder`'s own token report is the call that must notice.
        assert c.observe("decoder", 120), "a sweep on another chat missed the reserve"

    def test_room_for_charges_the_asker_for_the_batch_it_is_about_to_submit(self):
        """Answering at the idle buffer and raising it on the grant is the same race.

        A resume judged against a 7416 ceiling and then granted, which immediately puts
        the ceiling at 6136, is a chat admitted into room that stops existing in the same
        breath.
        """
        c = _controller()
        c.register("holder", tokens = 4300, signal = PreemptSignal())
        c.observe("holder", 10)
        c.register("waiter", tokens = 10, signal = PreemptSignal())
        c.observe("waiter", 1)
        c.set_state("waiter", ParticipantState.PAUSED)
        # 4310 + 2500 = 6810: inside the idle ceiling of 7416, outside the 6136 that
        # granting a replay bigger than one chunk would itself create.
        assert c.room_for("waiter", 2500) is False
        # And a replay small enough to leave the ceiling alone is still granted.
        assert c.room_for("waiter", 600) is True

    def test_the_asker_is_not_charged_twice_for_its_own_announcement(self):
        c = _controller()
        c.register("solo", tokens = 3000, signal = PreemptSignal())
        assert c.snapshot().prefilling == 3000
        # Its own 3000 is excluded and replaced by the 3000 it is asking about, so the
        # answer is the same as it would be with no announcement outstanding.
        assert c.room_for("solo", 3000) is True


class TestTheStaticOverride:
    def test_the_env_restores_the_permanent_batch_term(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH", "1")
        assert _buffer() == PREFILL_BUFFER
        assert _buffer(pending_prefill = 5000) == PREFILL_BUFFER

    def test_a_controller_under_the_override_never_drops_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH", "1")
        c = _controller("test://static")
        c.register("chat", tokens = 800, signal = PreemptSignal())
        c.observe("chat", 32)
        assert c.snapshot().buffer == PREFILL_BUFFER

    def test_off_by_default(self):
        c = _controller("test://default")
        c.register("chat", tokens = 800, signal = PreemptSignal())
        c.observe("chat", 32)
        assert c.snapshot().buffer == IDLE_BUFFER

    def test_the_per_slot_override_still_works(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", "300")
        assert _buffer() == 300 * SLOTS + DRAFTS * SLOTS

    def test_the_per_slot_override_and_the_static_batch_compose(self, monkeypatch):
        """max(), as before: the larger of reaction headroom and the chunk."""
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", "800")
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH", "1")
        assert _buffer() == 800 * SLOTS + DRAFTS * SLOTS

    def test_the_floor_and_the_cap_survive(self):
        """Neither the floor nor the half-budget cap moved with the batch term."""
        assert preemption_buffer_tokens(2048) >= DEFAULT_PREEMPT_BUFFER_MIN_TOKENS
        tiny = preemption_buffer_tokens(
            512, slots = 8, batch_tokens = 2048, draft_tokens = 64, pending_prefill = 2048
        )
        assert 0 < tiny <= 256
        assert preemption_buffer_tokens(0, pending_prefill = 2048) == 0


class TestPlanPreemptionsAtTheLowerCeiling:
    """The cells the batch term used to hold are really handed out."""

    def test_four_decoders_fit_where_the_old_ceiling_would_have_evicted(self):
        c = _controller()
        for i in range(SLOTS):
            c.register(f"chat{i}", tokens = 1700, signal = PreemptSignal())
            # A token each, so every prompt is in the cache and nothing is outstanding.
            c.observe(f"chat{i}", 0)
            c.observe(f"chat{i}", 1)
        # 6804: over the old 6136 ceiling, under the 7416 that stands with nothing
        # prefilling. Nobody should be paused for a batch nobody is submitting.
        assert c.committed_tokens() == 6804
        assert c.snapshot().prefilling == 0
        assert c.plan_preemptions(needed = 0) == []

    def test_the_static_override_would_have_evicted_them(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH", "1")
        c = _controller("test://static-evict")
        for i in range(SLOTS):
            c.register(f"chat{i}", tokens = 1700, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        assert c.committed_tokens() == 6804
        assert c.plan_preemptions(needed = 0), "6804 is past the 6136 static ceiling"

    def test_it_still_fires_once_the_lower_ceiling_is_passed(self):
        c = _controller()
        for i in range(SLOTS):
            c.register(f"chat{i}", tokens = 1900, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        assert c.committed_tokens() == 7604 > BUDGET - IDLE_BUFFER
        assert c.plan_preemptions(needed = 0), "7604 is past the 7416 dynamic ceiling"

    def test_a_prefill_starting_mid_flight_brings_the_ceiling_back_down(self):
        c = _controller()
        for i in range(SLOTS):
            c.register(f"chat{i}", tokens = 1700, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        assert c.plan_preemptions(needed = 0) == []
        # One of them reaches a round boundary and its prompt grows by a chunk's worth.
        c.note_tokens("chat0", 1701 + 2100)
        assert c.snapshot().prefilling == 2100
        assert c.plan_preemptions(needed = 0), "the growing round's chunk was not covered"


class TestTheResidentCaseIsUnchanged:
    """A reclaim erases a holder's cells; decoding again is a whole fresh prefill."""

    def test_a_reclaimed_holder_re_announces_when_it_decodes(self):
        c = _controller()
        c.register("parked", tokens = 3000, signal = PreemptSignal())
        c.observe("parked", 10)
        c.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        c.note_cells_reclaimed()
        assert c.snapshot().prefilling == 0, "cells that are gone submit nothing"
        c.note_state("parked", ParticipantState.DECODING)
        assert c.snapshot().prefilling == 3010, "the whole prompt goes back in"



class TestTheRouteReadsTheLaunchesBatchSize:
    """A `--batch-size 512` load reserved for 2048, which is the whole cache's worth.

    The llama.cpp backend keeps the flag on `_requested_n_batch` and publishes it as
    `requested_n_batch`. `_openai_llama_effective_batch_tokens` tried `n_batch`,
    `_n_batch`, `batch_size` and `_batch_size`, none of which that object has, so every
    load fell through to llama.cpp's 2048 default however it was launched. Harmless while
    the term was max()'d against a static 2048; not harmless once the term is what a
    pending prefill is sized against.
    """

    def test_the_public_accessor_is_read(self):
        import routes.inference as inference

        class _Backend:
            _requested_n_batch = 512

            @property
            def requested_n_batch(self):
                return self._requested_n_batch

        assert inference._openai_llama_effective_batch_tokens(_Backend()) == 512

    def test_the_private_field_is_read_when_the_property_is_absent(self):
        import routes.inference as inference

        class _Backend:
            _requested_n_batch = 512

        assert inference._openai_llama_effective_batch_tokens(_Backend()) == 512

    def test_an_unstated_batch_still_falls_back_to_the_llama_cpp_default(self):
        import routes.inference as inference

        class _Backend:
            requested_n_batch = None
            _requested_n_batch = None

        assert inference._openai_llama_effective_batch_tokens(_Backend()) == 2048

    def test_a_512_batch_shrinks_the_reserve_a_pending_prefill_takes(self):
        assert preemption_buffer_tokens(
            BUDGET, slots = SLOTS, draft_tokens = DRAFTS,
            batch_tokens = 512, pending_prefill = 5000,
        ) == IDLE_BUFFER, "512 is under the reaction headroom, which then covers it"

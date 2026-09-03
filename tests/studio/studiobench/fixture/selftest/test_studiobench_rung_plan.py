# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The rung plan invariant that the whole upper ladder rests on.

The streamed tail must be the SAME SIZE at every rung. When it grew with the rung instead, the
stream took 811 seconds at 1M against a 135-second film, so the ten slots labelled "after the
reply is complete" all ran mid-generation. Nothing crashed and a full table was printed; the
labels were just false. A property that fails silently and prints numbers anyway is exactly the
kind that needs a test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    RUNGS,
    STREAM_TAIL_CHARS,
    Corpus,
    plan_rung,
)

# The field cadence: 24 characters every 73 ms.
FIELD_CHARS_PER_SEC = 24 / 0.073


def _plans():
    corpus = Corpus.load()
    return {rung: plan_rung(corpus, rung) for rung in RUNGS}


def test_the_streamed_tail_never_exceeds_the_declared_size():
    for rung, plan in _plans().items():
        assert plan.streamed_chars <= STREAM_TAIL_CHARS, rung


def test_stream_duration_is_rung_independent():
    """The property the film depends on: the OPENING stream drains in the same time at every rung.

    The opening turn is the one the film's slots are timed against. The follow-ups are separate
    events sent later by `send_turn`, so folding them in here would measure a quantity no slot
    depends on and would fail on the 1K rung, which legitimately streams only once.
    """
    seconds = {r: p.streamed_chars / FIELD_CHARS_PER_SEC for r, p in _plans().items()}
    assert max(seconds.values()) < 20.0, seconds
    # 1K is legitimately shorter: the whole rung is smaller than one tail.
    big = {r: s for r, s in seconds.items() if r != "1K"}
    assert max(big.values()) - min(big.values()) < 5.0, big


def test_multi_turn_rungs_stream_more_than_once():
    """The point of the follow-ups: a cell samples streaming cost at more than one thread size."""
    plans = _plans()
    assert len(plans["1K"].follow_up_units) == 0, "a 4,000 character rung is one exchange"
    for rung in ("10K", "100K", "500K", "1M"):
        assert len(plans[rung].follow_up_units) == 2, rung


def test_follow_ups_are_small_enough_not_to_move_the_rung():
    """They sample cost; they are not supposed to be a second helping of thread mass."""
    for rung, plan in _plans().items():
        if not plan.follow_up_units:
            continue
        assert plan.follow_up_chars < plan.target_chars * 0.15, rung


def test_the_stream_drains_before_the_first_after_generation_slot():
    """Otherwise `scroll_after` and everything below it measure a still-streaming thread."""
    from studiobench.scene.schedule import SCENES

    worst = max(p.streamed_chars for p in _plans().values()) / FIELD_CHARS_PER_SEC
    for name, scene in SCENES.items():
        after = [s for s in scene.slots if s.action == "scroll_after"]
        assert after, name
        assert after[0].t_start_ms / 1000.0 > worst, (name, after[0].t_start_ms, worst)


def test_during_generation_slots_actually_fall_during_generation():
    from studiobench.scene.schedule import SCENES

    # The SHORTEST stream on the ladder above 1K, since a slot has to be inside every rung's
    # stream to deserve the name.
    # The OPENING turn only: the follow-ups are sent later by `send_turn`, so a slot that has to
    # fall during generation has to fall inside the first stream, not inside their sum.
    shortest = min(p.streamed_chars for r, p in _plans().items() if r != "1K") / FIELD_CHARS_PER_SEC
    for name, scene in SCENES.items():
        during = [s for s in scene.slots if s.action == "scroll_during_generation"]
        assert during, name
        for slot in during:
            assert slot.t_start_ms / 1000.0 < shortest, (name, slot.t_start_ms, shortest)


def test_stop_opens_only_after_the_tail_has_drained():
    """Stop owns its own turn now; opening it mid-stream would truncate the measured reply.

    HELD TO THE DECLARED CEILING RATHER THAN TO THE CURRENT CORPUS. This used to take the worst
    drain the frozen corpus happened to produce, which made a scene's packing a function of the
    corpus contents: the 1M rung streamed recycled text while the manifest was sized at exactly
    the top rung's seeded target, the observed worst came in low, and the fast film sat at 18.2 s
    against a real ceiling of 18.25 s without anything failing. Re-freezing the corpus moved the
    observed number and the film that had been out of bounds all along was the thing that broke.
    `STREAM_TAIL_CHARS` is the bound the plans are actually held to, one test above, so it is the
    bound a schedule has to clear -- and it does not move when the corpus is re-frozen.
    """
    from studiobench.scene.schedule import SCENES

    worst = STREAM_TAIL_CHARS / FIELD_CHARS_PER_SEC
    for name, scene in SCENES.items():
        stop = [s for s in scene.slots if s.action == "stop_generation"]
        assert stop, name
        assert stop[0].t_start_ms / 1000.0 > worst, (name, stop[0].t_start_ms, worst)


def test_every_rung_lands_close_to_the_size_it_claims():
    for rung, plan in _plans().items():
        total = plan.total_chars
        error = abs(total - plan.target_chars) / plan.target_chars
        # 1K cannot be exact: clipping is block-aligned so a prefix never ends inside a fence.
        limit = 0.15 if rung == "1K" else 0.05
        assert error < limit, (rung, total, plan.target_chars, error)


def test_the_ladder_is_strictly_increasing_in_seeded_mass():
    plans = _plans()
    order = ["1K", "10K", "100K", "500K", "1M"]
    masses = [plans[r].total_chars for r in order]
    assert masses == sorted(masses)
    assert len(set(masses)) == len(masses)


# Actions that need a SETTLED reply. The action bar's More and Copy buttons are not rendered while
# a turn is streaming, select-all copies a moving target, and delete is hidden on a running
# message. Each one reports an honest `NOT RUN` rather than a wrong number, which is why this was
# invisible until somebody counted: on a 36 job sweep the fast film recorded
# `message_menu: NOT RUN -- no More button` on 312 of 312 attempts, and the four actions it
# silently stopped exercising include the one carrying the largest measured effect in this
# codebase (the message menu, 1,164.9 ms to 60.7 ms across the merged campaign).
SETTLED_ACTIONS = ("message_menu", "copy_markdown", "select_all_copy", "delete_message")


def test_settled_actions_open_after_the_follow_up_drains():
    """A slot needing a finished reply must clear the follow-up turn the preceding send started.

    Only films used at a rung that actually sends follow-ups are checked. Below
    MULTI_TURN_MIN_CHARS a rung is single-turn, `send_turn` reports an exhausted queue, and there
    is no follow-up stream to wait for -- so holding the quick film to this bar would be asserting
    against a stream that never exists.
    """
    from studiobench.__main__ import TIER_RUNGS
    from studiobench.fixture.corpus import FOLLOW_UP_CHARS, MULTI_TURN_MIN_CHARS
    from studiobench.scene.schedule import SCENES

    drain_s = FOLLOW_UP_CHARS / FIELD_CHARS_PER_SEC
    plans = _plans()
    for name, scene in SCENES.items():
        rungs = TIER_RUNGS.get(name) or []
        if not any(
            (plans[r].total_chars if r in plans else 0) >= MULTI_TURN_MIN_CHARS for r in rungs
        ):
            continue
        last_send = None
        for slot in sorted(scene.slots, key = lambda s: s.t_start_ms):
            if slot.action == "send_turn":
                # THE LATEST THE SEND CAN FIRE, not the earliest. The drain starts when the send
                # actually happens, and a send slot is a WINDOW: the action may legitimately begin
                # anywhere inside its own budget, which is precisely what it does when the slot
                # before it overran and pushed it.
                #
                # Measured from the start instead, this check reported 1,538 ms of margin on the
                # fast film where 38 ms existed, and CI then failed the way the arithmetic says it
                # must: `reasoning_toggle` overran its 3,500 ms budget by 934 ms, `send_turn` was
                # pushed 1,373 ms late but stayed inside its 1,500 ms budget so nothing recorded a
                # miss, the send-to-menu gap collapsed from a nominal 5,300 ms to an actual
                # 3,691 ms, and `message_menu` found a reply still streaming and recorded NOT RUN.
                # Every slot was individually within budget and the film was still unrunnable.
                #
                # This is the same defect the rest of this branch is about, in the check rather
                # than in the instrument: a quantity computed at a moment whose meaning is not the
                # one the reader assumes. A margin measured from the earliest possible send is not
                # a margin.
                last_send = slot.t_start_ms + slot.budget_ms
            elif slot.action in SETTLED_ACTIONS and last_send is not None:
                # The slot's WINDOW, not its start. A slot may legitimately open a little before
                # the follow-up finishes and wait inside its own budget -- the quick film opens
                # `message_menu` at 4.5 s against a 4.56 s drain and it runs, because it has 3 s
                # of budget to wait in. What is fatal is a window that CLOSES before the reply
                # settles, which is what the first fast film did: 1.7 s gap plus 0.8 s budget
                # against a 4.56 s drain, so the button had not been rendered yet and never
                # would be inside that window.
                window_end = (slot.t_start_ms + slot.budget_ms - last_send) / 1000.0
                assert window_end >= drain_s, (name, slot.action, window_end, drain_s)

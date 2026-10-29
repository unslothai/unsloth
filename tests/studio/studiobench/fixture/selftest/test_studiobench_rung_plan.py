# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streamed tail must be the same size at every rung, or the post-reply slots run mid-stream."""

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

FIELD_CHARS_PER_SEC = 24 / 0.073


def _plans():
    corpus = Corpus.load()
    return {rung: plan_rung(corpus, rung) for rung in RUNGS}


def test_the_streamed_tail_never_exceeds_the_declared_size():
    for rung, plan in _plans().items():
        assert plan.streamed_chars <= STREAM_TAIL_CHARS, rung


def test_stream_duration_is_rung_independent():
    """Only the opening stream is measured; follow-ups are sent later, and the 1K rung streams just once."""
    seconds = {r: p.streamed_chars / FIELD_CHARS_PER_SEC for r, p in _plans().items()}
    assert max(seconds.values()) < 20.0, seconds
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

    # Opening turn only: follow-ups are sent later, so during-generation slots must fit the first.
    shortest = min(p.streamed_chars for r, p in _plans().items() if r != "1K") / FIELD_CHARS_PER_SEC
    for name, scene in SCENES.items():
        during = [s for s in scene.slots if s.action == "scroll_during_generation"]
        assert during, name
        for slot in during:
            assert slot.t_start_ms / 1000.0 < shortest, (name, slot.t_start_ms, shortest)


def test_stop_opens_only_after_the_tail_has_drained():
    """Stop must not open mid-stream; plans are held to the declared STREAM_TAIL_CHARS, not the corpus"""
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


# These actions are not rendered or meaningful while a turn streams; they report NOT RUN.
SETTLED_ACTIONS = ("message_menu", "copy_markdown", "select_all_copy", "delete_message")


def test_settled_actions_open_after_the_follow_up_drains():
    """Settled slots must wait for the follow-up stream; only multi-turn rungs have one to wait for."""
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
                # Latest the send can fire: a slot may start anywhere in its budget if the prior overran.
                last_send = slot.t_start_ms + slot.budget_ms
            elif slot.action in SETTLED_ACTIONS and last_send is not None:
                # Fatal only if the slot's window closes before the reply settles.
                window_end = (slot.t_start_ms + slot.budget_ms - last_send) / 1000.0
                assert window_end >= drain_s, (name, slot.action, window_end, drain_s)

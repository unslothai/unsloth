# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every streamed turn must be material the thread has not already seen.

The corpus goes to some trouble to make every unit unique, because Shiki caches highlighted output
keyed on the source string and a repeated fence is a fence that costs nothing the second time.
Unique UNITS are not the same property as a unique PLAN, and the difference is what went wrong: the
top rung's seeded prefix consumed the whole frozen manifest, a `min(index, last_index)` clamp
folded the opening stream and both follow-ups onto the final unit, and the 1M rung then re-sent
text it had already seeded, from one unit, three times. Nothing raised. The rung that exists to be
the most expensive on the ladder simply reported the least work per character, which is the exact
direction the bug pushes and the reason it survived being looked at.

So the invariant is stated here rather than inferred from a table afterwards, and it is stated over
the whole reachable configuration space -- every rung, every tier ladder, a sweep of measured
chars-per-token ratios, and a sweep of turn counts -- not over the one rung that happened to break.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    MANIFEST_CHARS_PER_TOKEN,
    RUNGS,
    STREAM_TURNS,
    Corpus,
    RungPlan,
    manifest_unit_count,
    plan_rung,
    unit_text,
)

#: Ratios a machine can actually report. `measure_chars_per_token` measures the seeded thread, and
#: the planner is handed whatever it returns; the frozen corpus is sized for everything up to
#: MANIFEST_CHARS_PER_TOKEN and must refuse -- loudly -- above it.
SWEPT_RATIOS = (1.0, 2.0, 3.0, 3.7, 4.0, 4.5, MANIFEST_CHARS_PER_TOKEN)


def _corpus() -> Corpus:
    return Corpus.load()


def _streamed(plan: RungPlan) -> list[tuple[str, str]]:
    out = []
    if plan.streamed_unit is not None:
        out.append((f"streamed(unit {plan.streamed_unit.index})", unit_text(plan.streamed_unit)))
    for i, unit in enumerate(plan.follow_up_units):
        out.append((f"follow_up[{i}](unit {unit.index})", unit_text(unit)))
    return out


def _assert_plan_streams_new_material(plan: RungPlan, where: str) -> None:
    """The invariant itself, asserted independently of the check `plan_rung` runs for itself.

    A prefix relation and not equality: the streamed turns are clipped, and two turns clipped from
    one unit at two different lengths are not equal while sharing every fence the shorter one has.
    """
    streamed = _streamed(plan)
    assert streamed, where
    seeded = [(f"seeded(unit {u.index})", unit_text(u)) for u in plan.seeded_units]
    for i, (label, text) in enumerate(streamed):
        assert text, (where, label, "empty")
        for other_label, other in streamed[i + 1 :] + seeded:
            assert not text.startswith(other), (where, label, other_label)
            assert not other.startswith(text), (where, other_label, label)


def test_every_rung_streams_material_the_thread_has_not_seen():
    corpus = _corpus()
    for rung in RUNGS:
        _assert_plan_streams_new_material(plan_rung(corpus, rung), rung)


def test_no_streamed_turn_reuses_a_seeded_unit_index():
    """The index-level statement of the same thing, which is what actually broke.

    Kept separate from the text comparison because it is the cheap, readable version: when this one
    fails the plan handed one corpus unit to two turns, and no amount of clipping makes that two
    different pieces of content.
    """
    corpus = _corpus()
    for rung in RUNGS:
        plan = plan_rung(corpus, rung)
        seeded = [u.index for u in plan.seeded_units]
        streamed = [plan.streamed_unit.index] + [u.index for u in plan.follow_up_units]
        assert len(set(streamed)) == len(streamed), (rung, streamed)
        assert not set(streamed) & set(seeded), (rung, streamed, seeded)


def test_the_invariant_holds_at_every_ratio_a_machine_can_report():
    """`chars_per_token` is measured, not assumed, and a larger ratio means a longer prefix."""
    corpus = _corpus()
    for ratio in SWEPT_RATIOS:
        for rung in RUNGS:
            plan = plan_rung(corpus, rung, ratio)
            _assert_plan_streams_new_material(plan, f"{rung}@{ratio}")


def test_the_invariant_holds_for_every_tier_ladder_and_rung_override():
    """`--tier full` walks the whole ladder and `--rungs` can name any subset of it.

    Rungs are planned one at a time, so a ladder is only ever the union of its rungs -- but the
    tiers are swept anyway, because that is the claim a reader of the CLI actually cares about and
    a future tier that names a rung nobody planned would be caught here.
    """
    from studiobench.__main__ import TIER_RUNGS

    corpus = _corpus()
    assert "1M" in TIER_RUNGS["full"], "the full tier is what reaches the top of the ladder"
    for tier, rungs in TIER_RUNGS.items():
        for rung in rungs:
            _assert_plan_streams_new_material(plan_rung(corpus, rung), f"{tier}:{rung}")


def test_the_invariant_holds_for_every_turn_count_the_corpus_is_sized_for():
    """More streamed turns means more units past the prefix. Sized for, or refused."""
    from studiobench.fixture import corpus as corpus_mod

    corpus = _corpus()
    original = corpus_mod.STREAM_TURNS
    try:
        for turns in (1, 2, 3, 4, 5, 6):
            corpus_mod.STREAM_TURNS = turns
            for rung in RUNGS:
                _assert_plan_streams_new_material(plan_rung(corpus, rung), f"{rung}x{turns}")
    finally:
        corpus_mod.STREAM_TURNS = original


def test_the_manifest_is_sized_from_the_ladder_and_not_the_other_way_round():
    """The repair. The corpus carries the longest prefix any rung can ask for, plus the turns."""
    corpus = _corpus()
    entries = len(corpus.manifest["units"])
    assert entries == manifest_unit_count(corpus.seed), (
        entries,
        manifest_unit_count(corpus.seed),
    )
    top = max(len(plan_rung(corpus, r, MANIFEST_CHARS_PER_TOKEN).seeded_units) for r in RUNGS)
    assert entries >= top + STREAM_TURNS, (entries, top, STREAM_TURNS)


def test_a_corpus_too_small_for_the_ladder_fails_loudly():
    """The clamp is gone. Running off the end has to stop the run, not quietly shrink it.

    This is the property that keeps the defect from coming back through a door nobody watched: a
    larger rung, a larger measured ratio, another streamed turn. Any of them can outgrow the frozen
    corpus, and all of them now say so.
    """
    corpus = _corpus()
    truncated = dict(corpus.manifest)
    truncated["units"] = [u for u in corpus.manifest["units"] if u["index"] < 6]
    small = Corpus(truncated, {}, corpus.seed)
    with pytest.raises(ValueError, match = "too small for the"):
        plan_rung(small, "100K")


def test_a_ratio_past_what_the_corpus_was_frozen_for_fails_loudly():
    corpus = _corpus()
    with pytest.raises(ValueError, match = "too small for the"):
        plan_rung(corpus, "1M", MANIFEST_CHARS_PER_TOKEN * 2)


def test_a_rung_added_above_the_ladder_refuses_until_the_corpus_is_refrozen():
    """The way this defect would come back: someone adds a bigger rung and runs it.

    Two halves, and both are needed. The run REFUSES, because a corpus frozen for the old ladder
    cannot honestly measure a rung above it. And `manifest_unit_count` already knows the new rung
    exists, so `--freeze` produces a corpus that fits without anyone having to work out how many
    units the new rung needs -- the ladder sizes the corpus, which is the whole point of the
    repair.
    """
    corpus = _corpus()
    original = dict(RUNGS)
    try:
        RUNGS["2M"] = 2_000_000
        with pytest.raises(ValueError, match = "too small for the"):
            plan_rung(corpus, "2M")
        assert manifest_unit_count(corpus.seed) > len(corpus.manifest["units"])
    finally:
        RUNGS.clear()
        RUNGS.update(original)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Distinct units do not guarantee a distinct plan, so reuse across turns is tested over all rungs."""

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
    """Checked as a prefix, not equality: two turns clipped from one unit are unequal but share fences."""
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
    """Index-level check: one corpus unit given to two turns is the same content, however it is clipped."""
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
    """Sweeps every tier and rung override, so a future tier naming an unplanned rung is caught here."""
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
    """Running off the end of the manifest must stop the run rather than quietly shrink it."""
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
    """A rung above the ladder is refused until the corpus is re-frozen; the old corpus cannot fit it."""
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

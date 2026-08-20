# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The verdict gates, and the refusals that keep a number from being quoted when it should not be.

Every gate here is paired: one test that a clean result passes it, one that a result which should
fail it does. A gate that only ever passes is indistinguishable from no gate, and it is worse than
no gate because it gets cited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.sweep import floor_table as F  # noqa: E402


# ── building a payload ───────────────────────────────────────────────


def cell(rung: str, arm: str, rep: str, timings: dict[str, float]) -> list[dict]:
    cid = f"{rung}.{arm}.{rep}"
    return [
        {"row_type": "cell", "cell_id": cid, "completed": True},
        {
            "row_type": "action",
            "cell_id": cid,
            "action": "message_menu",
            "ran": True,
            "timings": timings,
        },
    ]


def payload(
    tmp_path: Path,
    name: str,
    pairs: list[tuple[float, float]],
    tier: str = "standard",
    corpus: str | None = "corpus0000",
) -> Path:
    """One shard: `pairs[i]` is (base, treatment) for repetition i."""
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    meta: dict = {"row_type": "run_meta", "tier": tier}
    if corpus is not None:
        meta["corpus_hash"] = corpus
    rows: list[dict] = [meta]
    for i, (base, treat) in enumerate(pairs):
        rows += cell("100K", "base", f"rep{i}", {"open_close_ms": base})
        rows += cell("100K", "treatment", f"rep{i}", {"open_close_ms": treat})
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def verdict(
    tmp_path,
    result_pairs,
    floor_pairs,
    tier = "standard",
) -> str:
    result = F.summarise([payload(tmp_path, "result", result_pairs, tier)])
    floor = F.summarise([payload(tmp_path, "null", floor_pairs, tier)])
    _f, v = F.verdict_for(result["message_menu.open_close_ms"], floor["message_menu.open_close_ms"])
    return v


# ── gate 1: the per-metric floor ─────────────────────────────────────


def test_a_large_consistent_effect_over_a_tight_floor_passes(tmp_path):
    assert (
        verdict(
            tmp_path,
            [(1000.0, 100.0), (1010.0, 101.0), (990.0, 99.0), (1000.0, 100.0)],
            [(1000.0, 1000.0), (1010.0, 1012.0), (990.0, 989.0), (1000.0, 1001.0)],
        )
        == "faster"
    )


def test_an_effect_under_the_floor_is_void(tmp_path):
    # 3% claimed, against a null control whose own two identical builds land 20% apart.
    assert (
        verdict(
            tmp_path,
            [(1000.0, 970.0), (1000.0, 970.0), (1000.0, 970.0), (1000.0, 970.0)],
            [(1000.0, 900.0), (1000.0, 1100.0), (1000.0, 950.0), (1000.0, 1050.0)],
        )
        == "VOID (under floor)"
    )


def test_the_floor_clears_the_null_controls_bias_not_only_its_spread(tmp_path):
    # A null control that is TIGHT but systematically offset: identical builds, yet the treatment
    # label reads 10% faster on every repetition. Spread is nearly zero, so a floor built from
    # spread alone would admit any effect at all. The bar is max(|bias|, spread).
    floor_pairs = [(1000.0, 900.0)] * 4
    assert verdict(tmp_path, [(1000.0, 950.0)] * 4, floor_pairs) == "VOID (under floor)"
    assert verdict(tmp_path, [(1000.0, 700.0)] * 4, floor_pairs) == "faster"


# ── gate 2: sign consistency ─────────────────────────────────────────


def test_pairs_that_disagree_on_sign_are_void_however_large_the_mean(tmp_path):
    # Mean lands well past the floor; two repetitions say faster and two say slower.
    assert (
        verdict(
            tmp_path,
            [(1000.0, 400.0), (1000.0, 1300.0), (1000.0, 300.0), (1000.0, 1200.0)],
            [(1000.0, 1000.0)] * 4,
        )
        == "VOID (pairs disagree on sign)"
    )


# ── gate 3: the effect must exceed its own scatter ───────────────────


def test_an_effect_smaller_than_its_own_scatter_is_void(tmp_path):
    # Every repetition agrees on the direction and the mean clears the floor, but the readings
    # range from 2% to 60% faster. Twelve rows in a 40-comparison audit looked exactly like this.
    assert (
        verdict(
            tmp_path,
            [(1000.0, 980.0), (1000.0, 400.0), (1000.0, 950.0), (1000.0, 500.0)],
            [(1000.0, 1000.0)] * 4,
        )
        == "VOID (effect under its own scatter)"
    )


def test_gate_three_does_not_fire_on_a_tight_large_effect(tmp_path):
    assert (
        verdict(
            tmp_path,
            [(1000.0, 100.0), (1000.0, 102.0), (1000.0, 98.0), (1000.0, 101.0)],
            [(1000.0, 1000.0)] * 4,
        )
        == "faster"
    )


def test_gate_three_cannot_fire_on_a_single_pair(tmp_path):
    # With n=1 the spread is 0 by construction, so the gate has nothing to say. It must not
    # silently pass a single reading off as having survived a scatter check.
    result = F.summarise([payload(tmp_path, "result", [(1000.0, 500.0)])])
    assert result["message_menu.open_close_ms"]["n"] == 1
    _f, v = F.verdict_for(
        result["message_menu.open_close_ms"], {"delta_pct": 0.0, "spread_pct": 1.0}
    )
    assert v == "faster"


# ── correctness invariants, scored with the same arithmetic and the opposite sign ────


def count_cell(cid: str, chars: float) -> list[dict]:
    return [
        {"row_type": "cell", "cell_id": cid, "completed": True},
        {
            "row_type": "action",
            "cell_id": cid,
            "action": "select_all_copy",
            "ran": True,
            "timings": {"copy_ms": 10.0},
            "counts": {"selected_chars": chars},
        },
    ]


def count_payload(tmp_path: Path, name: str, pairs: list[tuple[float, float]]) -> Path:
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard"}]
    for i, (base, treat) in enumerate(pairs):
        rows += count_cell(f"100K.base.rep{i}", base)
        rows += count_cell(f"100K.treatment.rep{i}", treat)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


COUNT_METRIC = "select_all_copy.count.selected_chars"


def test_a_count_is_harvested_under_a_name_that_marks_it_as_an_invariant(tmp_path):
    stats = F.summarise([count_payload(tmp_path, "r", [(400000.0, 400000.0)])])
    assert COUNT_METRIC in stats
    assert F.is_count_metric(COUNT_METRIC)
    # The timing on the same action must stay a timing.
    assert "select_all_copy.copy_ms" in stats
    assert not F.is_count_metric("select_all_copy.copy_ms")


def test_a_count_that_fell_reads_as_a_loss_and_never_as_faster(tmp_path):
    # The regression this exists to catch: virtualization truncates select-all from 400k characters
    # to 3k. Every timing improves, `expect_ok` stays true because chars > 0, and the only thing
    # that can say so is this row. Scored as a timing it would read "faster".
    stats = F.summarise([count_payload(tmp_path, "r", [(400000.0, 3000.0)] * 4)])
    floor = {"delta_pct": 0.0, "spread_pct": 1.0}
    _f, v = F.verdict_for(stats[COUNT_METRIC], floor, F.is_count_metric(COUNT_METRIC))
    assert v == "LOST (invariant fell)"
    # Same numbers, scored as a timing, would have been reported as an improvement.
    assert F.verdict_for(stats[COUNT_METRIC], floor, False)[1] == "faster"


def test_a_count_that_rose_is_reported_but_not_as_a_slowdown(tmp_path):
    stats = F.summarise([count_payload(tmp_path, "r", [(3000.0, 400000.0)] * 4)])
    _f, v = F.verdict_for(stats[COUNT_METRIC], {"delta_pct": 0.0, "spread_pct": 1.0}, is_count = True)
    assert v == "gained"


def test_an_unchanged_count_is_void_like_any_other_metric_under_the_floor(tmp_path):
    stats = F.summarise([count_payload(tmp_path, "r", [(400000.0, 400000.0)] * 4)])
    _f, v = F.verdict_for(stats[COUNT_METRIC], {"delta_pct": 0.0, "spread_pct": 1.0}, is_count = True)
    assert v == "VOID (under floor)"


def test_a_lost_invariant_is_printed_and_counted_by_the_table(tmp_path, capsys):
    result = count_payload(tmp_path, "result", [(400000.0, 3000.0)] * 4)
    floors = {COUNT_METRIC: {"delta_pct": 0.0, "spread_pct": 1.0}}
    survivors = F.render([result], "t", floors = floors)
    out = capsys.readouterr().out
    assert "LOST (invariant fell)" in out
    # It has to be counted as a finding, not dropped for not being one of the two timing verdicts.
    assert survivors >= 1


def test_a_count_on_an_action_that_did_not_run_contributes_nothing(tmp_path):
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "select_all_copy",
            "ran": False,
            "counts": {"selected_chars": 400000.0},
        },
    ]
    out = tmp_path / "z"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    assert F._action_timings(F.read_rows(out / "payload.jsonl"), "100K.base.rep0") == {}


def test_a_boolean_count_is_not_harvested_as_a_number(tmp_path):
    # `True` is an int in Python, and a flag flipping is not a 100% regression.
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "select_all_copy",
            "ran": True,
            "counts": {"selected_chars": 12.0, "truncated": True},
        },
    ]
    out = tmp_path / "w"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    harvested = F._action_timings(F.read_rows(out / "payload.jsonl"), "100K.base.rep0")
    assert harvested == {"select_all_copy.count.selected_chars": 12.0}


# ── refusals ─────────────────────────────────────────────────────────


def test_no_floor_at_all_yields_no_verdict_rather_than_a_pass(tmp_path):
    result = F.summarise([payload(tmp_path, "result", [(1000.0, 100.0)] * 4)])
    _f, v = F.verdict_for(result["message_menu.open_close_ms"], None)
    assert v == "no floor measured"


def test_pooling_across_tiers_is_refused(tmp_path):
    fast = payload(tmp_path, "fast", [(1000.0, 900.0)], tier = "fast")
    standard = payload(tmp_path, "standard", [(1000.0, 900.0)], tier = "standard")
    with pytest.raises(SystemExit) as exc:
        F.load([fast, standard])
    assert "different tiers" in str(exc.value)


def test_scoring_against_a_floor_from_another_tier_is_refused(tmp_path):
    result = payload(tmp_path, "result", [(1000.0, 100.0)] * 4, tier = "fast")
    with pytest.raises(SystemExit) as exc:
        F.render([result], "t", floors = {}, floor_tier = "standard")
    assert "different films" in str(exc.value)


def test_pooling_across_corpora_is_refused(tmp_path):
    # The tier fixes how long the film runs; the corpus hash fixes what is IN it. Corpus v2 added
    # math, so a v1 payload and a v2 payload measure two different documents under one name, and
    # pooling them would read the corpus change as a performance change.
    one = payload(tmp_path, "one", [(1000.0, 900.0)], corpus = "aaaa1111")
    two = payload(tmp_path, "two", [(1000.0, 900.0)], corpus = "bbbb2222")
    with pytest.raises(SystemExit) as exc:
        F.load([one, two])
    assert "different corpora" in str(exc.value)


def test_scoring_against_a_floor_from_another_corpus_is_refused(tmp_path):
    result = payload(tmp_path, "result", [(1000.0, 100.0)] * 4, corpus = "bbbb2222")
    with pytest.raises(SystemExit) as exc:
        F.render([result], "t", floors = {}, floor_corpus = "aaaa1111")
    assert "different film" in str(exc.value)


def test_the_same_corpus_on_both_sides_pools_normally(tmp_path):
    a = payload(tmp_path / "a", "s0", [(1000.0, 500.0)], corpus = "aaaa1111")
    b = payload(tmp_path / "b", "s0", [(2000.0, 1000.0)], corpus = "aaaa1111")
    pooled, _ = F.load([a, b])
    assert len(pooled["message_menu.open_close_ms"]) == 2


def test_a_payload_with_no_corpus_hash_is_not_silently_pooled_with_one_that_has_it(tmp_path):
    # An older payload predating the field reads "?", which is a different value, not a wildcard.
    # Treating it as compatible is how a v1 run would end up scored against a v2 floor.
    old = payload(tmp_path, "old", [(1000.0, 900.0)], corpus = None)
    new = payload(tmp_path, "new", [(1000.0, 900.0)], corpus = "bbbb2222")
    with pytest.raises(SystemExit) as exc:
        F.load([old, new])
    assert "different corpora" in str(exc.value)


def test_an_action_that_did_not_run_contributes_no_timing(tmp_path):
    # The rule the whole harness turns on: an absent action is not a fast one.
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "message_menu",
            "ran": False,
            "timings": {"open_close_ms": 3.0},
        },
    ]
    out = tmp_path / "x"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    records = F.read_rows(out / "payload.jsonl")
    assert F._action_timings(records, "100K.base.rep0") == {}


def test_an_incomplete_cell_is_not_measured(tmp_path):
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": False},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "message_menu",
            "ran": True,
            "timings": {"open_close_ms": 3.0},
        },
    ]
    out = tmp_path / "y"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    assert F.cell_metrics(F.read_rows(out / "payload.jsonl")) == {}


def test_a_shard_pairs_within_itself_and_never_across(tmp_path):
    # Two shards both number their repetitions from rep0. Pairing on the repetition alone would
    # cross them over and silently compare one session's base with another's treatment.
    a = payload(tmp_path / "a", "s0", [(1000.0, 500.0)])
    b = payload(tmp_path / "b", "s0", [(2000.0, 1000.0)])
    pooled, _ = F.load([a, b])
    rows = pooled["message_menu.open_close_ms"]
    assert sorted(rows) == [(1000.0, 500.0), (2000.0, 1000.0)]


def test_main_without_a_floor_says_so_and_still_prints(tmp_path, capsys):
    payload(tmp_path, "result", [(1000.0, 100.0)] * 4)
    assert F.main([str(tmp_path / "result")]) == 0
    assert "NO FLOOR SUPPLIED" in capsys.readouterr().out


def test_main_returns_two_when_nothing_matches(tmp_path, capsys):
    assert F.main([str(tmp_path / "does-not-exist")]) == 2
    assert "no payload found" in capsys.readouterr().out

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

from tests.studio.studiobench.__main__ import main as cli_main  # noqa: E402
from tests.studio.studiobench.sweep import floor_table as F  # noqa: E402
from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402


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


def test_a_resumed_payload_carrying_two_corpora_is_refused(tmp_path):
    # The recorder appends, so `--resume` into the same --out leaves the first run's completed
    # cells next to a SECOND run_meta. If the corpus changed in between, that one file holds a
    # base recorded on the old film and a treatment recorded on the new one, and `paired` matches
    # them on (shard, rung, rep) without noticing. Reading only the first header would pass this
    # payload and print the corpus change as a performance change.
    out = tmp_path / "resumed"
    out.mkdir(parents = True, exist_ok = True)
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "corpus_hash": "aaaa1111"}]
    rows += cell("100K", "base", "rep0", {"open_close_ms": 1000.0})
    rows += [{"row_type": "run_meta", "tier": "standard", "corpus_hash": "bbbb2222"}]
    rows += cell("100K", "treatment", "rep0", {"open_close_ms": 100.0})
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    assert F.corpora_of(F.read_rows(path)) == {"aaaa1111", "bbbb2222"}
    with pytest.raises(SystemExit) as exc:
        F.load([path])
    assert "more than one corpus" in str(exc.value)
    # And the floor-vs-result path refuses it too, rather than scoring it under the first hash.
    with pytest.raises(SystemExit) as exc:
        F.render([path], "t", floors = {}, floor_corpus = "aaaa1111")
    assert "more than one corpus" in str(exc.value)


def test_a_payload_with_repeated_headers_on_one_corpus_still_loads(tmp_path):
    # A plain resume, nothing changed in between: two headers, one hash, no refusal.
    out = tmp_path / "plain"
    out.mkdir(parents = True, exist_ok = True)
    meta = {"row_type": "run_meta", "tier": "standard", "corpus_hash": "aaaa1111"}
    rows: list[dict] = [dict(meta)]
    rows += cell("100K", "base", "rep0", {"open_close_ms": 1000.0})
    rows += [dict(meta)]
    rows += cell("100K", "treatment", "rep0", {"open_close_ms": 900.0})
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    pooled, _tiers = F.load([path])
    assert pooled["message_menu.open_close_ms"] == [(1000.0, 900.0)]


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


def test_an_action_whose_own_assertion_failed_contributes_no_timing(tmp_path):
    # `ran = True, expect_ok = False` is the action that happened and did not do its job. Its p95
    # is lower BECAUSE it failed, so pairing it prints the failure as an improvement. The payload
    # notes layer already says these timings must not be quoted.
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "keystroke",
            "ran": True,
            "expect_ok": False,
            "timings": {"p95_ms": 3.0},
            "counts": {"typed_chars": 4.0},
        },
    ]
    out = tmp_path / "e"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    assert F._action_timings(F.read_rows(out / "payload.jsonl"), "100K.base.rep0") == {}


def test_an_action_with_no_expectation_recorded_is_still_harvested(tmp_path):
    # `expect_ok` is None on an action that asserts nothing, and on every payload recorded before
    # the field existed. Only an explicit False is a failed assertion.
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "action",
            "cell_id": "100K.base.rep0",
            "action": "keystroke",
            "ran": True,
            "expect_ok": None,
            "timings": {"p95_ms": 3.0},
        },
    ]
    out = tmp_path / "n"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    assert F._action_timings(F.read_rows(out / "payload.jsonl"), "100K.base.rep0") == {
        "keystroke.p95_ms": 3.0
    }


def write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def frame_window(cid: str, kind: str, gaps: list[float], duration_ms: float, **extra) -> dict:
    row = {
        "row_type": "window",
        "cell_id": cid,
        "kind": kind,
        "name": f"{kind}:w",
        "duration_ms": duration_ms,
        "instruments": {
            "frames": {
                "frames_attempted": True,
                "frame_gaps_ms": gaps,
                "max_frame_ms": max(gaps),
            }
        },
    }
    row.update(extra)
    return row


def stream_window(
    cid: str,
    gaps: list[float],
    duration_ms: float = 10_000.0,
) -> dict:
    """One qualifying, unaided streaming window: SSE traffic plus enough reply growth."""
    return {
        "row_type": "window",
        "cell_id": cid,
        "kind": "gap",
        "name": "stream:gap1",
        "duration_ms": duration_ms,
        "instruments": {
            "stream_cost": {
                "stream_cost_attempted": True,
                "streaming_observed": True,
                "streaming_ms": 9_000.0,
                "delta_task_ms": 900.0,
                "stream_blocked_ms": 1_800.0,
                "reply_chars_delta": 3_000,
            },
            "frames": {
                "frames_attempted": True,
                "frame_gaps_ms": gaps,
                "frame_gaps_truncated": False,
                "max_frame_ms": max(gaps),
            },
        },
    }


def stream_payload(tmp_path: Path, name: str, pairs: list[tuple[list, list]]) -> Path:
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "corpus_hash": "c0"}]
    for i, (base_gaps, treat_gaps) in enumerate(pairs):
        for arm, gaps in (("base", base_gaps), ("treatment", treat_gaps)):
            cid = f"100K.{arm}.rep{i}"
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
            rows.append(stream_window(cid, gaps))
    return write(tmp_path, name, rows)


SMOOTH = [16.7] * 540  # a clean stream: stream_time_in_jank_pct is exactly 0.0
JANKY = [16.7] * 530 + [120.0] * 10


def test_a_clean_zero_base_arm_still_pairs_so_a_jank_regression_is_not_lost(tmp_path):
    """Zero is these metrics' CLEAN reading, and `if b` dropped every pair that had one.

    Measured on this repository's recorded payloads, 349 of 668 `stream_time_in_jank_pct` pairs
    have a zero base arm. Dropped, a treatment that introduces jank against a clean base left no
    row in the table at all, and where only some repetitions had a zero base the metric was pooled
    over whichever pairs the BASE arm happened to make non-zero.
    """
    result = stream_payload(tmp_path, "result", [(SMOOTH, JANKY)] * 4)
    null = stream_payload(tmp_path, "null", [(SMOOTH, SMOOTH)] * 4)

    pairs = F.paired(F.read_rows(result))
    assert pairs["stream_time_in_jank_pct"] == [(0.0, 12.0)] * 4
    assert len(pairs["stream_jank_index"]) == 4

    stats, floors = F.summarise([result]), F.summarise([null])
    s = stats["stream_time_in_jank_pct"]
    assert s["n"] == 4
    # Compared by DIFFERENCE, in the metric's own unit: 0.0% to 12.0% is +12.0 points.
    assert s["difference"] is True
    assert s["delta_pct"] == pytest.approx(12.0)
    assert "stream_time_in_jank_pct" in floors
    _f, v = F.verdict_for(s, floors["stream_time_in_jank_pct"])
    assert v == "SLOWER"


def test_an_unchanged_repetition_does_not_read_as_a_pair_disagreeing_on_sign(tmp_path):
    """REGRESSION. A difference of exactly 0.0 is a TIE, and ties are the modal reading here.

    `all(d > 0) or all(d < 0)` is the right question for ratios, where an exact 1.0 essentially
    never occurs. On differences of a metric whose clean value IS 0.0 it voided a group the moment
    any repetition was unchanged: `[0.0, 12.0]` failed both halves and printed
    VOID (pairs disagree on sign) although nothing moved the other way. Measured over the recorded
    payloads that hit 34 of 250 pooled groups, and all 34 were ties rather than disagreements.

    WHAT THIS DOES AND DOES NOT BUY, measured rather than assumed. Of the 34 groups, ZERO change
    their verdict: a tie drags the spread above the mean, so GATE 3 voids every one of them for
    scatter instead. That is the correct gate doing its job, and it is why this fixes the REASON
    a comparison was refused rather than surfacing a regression. A tool whose refusal states
    something that did not happen is reporting a wrong answer in the right vocabulary, which is
    worth three lines on its own; it is not worth relaxing gate 3 to chase.
    """
    # Two repetitions clean on both arms, two that regress. Nothing regresses the other way.
    result = stream_payload(tmp_path, "result", [(SMOOTH, SMOOTH), (SMOOTH, JANKY)] * 2)
    null = stream_payload(tmp_path, "null", [(SMOOTH, SMOOTH)] * 4)

    s = F.summarise([result])["stream_time_in_jank_pct"]
    assert sorted(t - b for b, t in F.paired(F.read_rows(result))["stream_time_in_jank_pct"]) == [
        0.0,
        0.0,
        12.0,
        12.0,
    ]
    assert s["consistent"] is True, "an unchanged repetition is a tie, not a disagreement"
    _f, v = F.verdict_for(s, F.summarise([null])["stream_time_in_jank_pct"])
    # Still refused, and that is right, but no longer refused for a reason that did not occur.
    assert v == "VOID (effect under its own scatter)"
    assert v != "VOID (pairs disagree on sign)"


def test_a_group_that_did_not_move_at_all_is_still_not_a_finding(tmp_path):
    """The other side of the tie rule: all-zero differences must not become 'consistent'.

    Relaxing to `>=` alone would score a metric that never moved as a directional result.
    """
    result = stream_payload(tmp_path, "result", [(SMOOTH, SMOOTH)] * 4)
    s = F.summarise([result])["stream_time_in_jank_pct"]
    assert [t - b for b, t in F.paired(F.read_rows(result))["stream_time_in_jank_pct"]] == [0.0] * 4
    assert s["consistent"] is False


def test_a_ratio_metric_still_drops_a_zero_base_rather_than_dividing_by_it(tmp_path):
    """The control: only the metrics whose zero is a clean reading changed."""
    result = stream_payload(tmp_path, "result", [(SMOOTH, JANKY)] * 4)
    pairs = F.paired(F.read_rows(result))
    assert F.summarise([result])["stream_max_frame_ms"]["difference"] is False
    assert all(b > 0.0 for b, _ in pairs["stream_max_frame_ms"])


def test_the_enforced_idle_window_is_not_pooled_into_the_frame_metrics(tmp_path):
    # Every cell records a 1.5 s `idle:calibrate` window with the frame recorder running. Pooling
    # its quiet into the film halves the jank share, so the column would not be the quantity the
    # rest of the tool prints under that name.
    cid = "r100K.base.rep0"
    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": cid, "completed": True},
        frame_window(cid, "action", [200.0] * 10, 2000.0),
        frame_window(cid, "idle", [16.0] * 94, 1500.0),
    ]
    vals = F.cell_metrics(F.read_rows(write(tmp_path, "c", rows)))[cid]
    assert vals["time_in_jank_pct"] == 100.0
    assert vals["max_frame_ms"] == 200.0


def test_a_resumed_cell_is_measured_from_its_own_attempt_only(tmp_path):
    # `--resume` re-runs a cell that died, and the payload is append-only, so the dead attempt's
    # windows sit in the file under the SAME cell id. Pooling them into the retry reports a number
    # that no single run of the film ever produced.
    cid = "r100K.base.rep0"
    rows = [
        {"row_type": "run_meta", "tier": "standard", "session_id": "s1"},
        {"row_type": "cell", "cell_id": cid, "completed": False, "session_id": "s1"},
        frame_window(cid, "action", [5000.0] * 10, 2000.0, session_id = "s1"),
        {"row_type": "run_meta", "tier": "standard", "session_id": "s2"},
        {"row_type": "cell", "cell_id": cid, "completed": True, "session_id": "s2"},
        frame_window(cid, "action", [16.0] * 100, 1600.0, session_id = "s2"),
    ]
    vals = F.cell_metrics(F.read_rows(write(tmp_path, "r", rows)))[cid]
    assert vals["max_frame_ms"] == 16.0
    assert vals["time_in_jank_pct"] == 0.0


def test_one_file_holding_two_tiers_is_refused_like_two_files(tmp_path):
    # The recorder appends, so a second run into the same --out leaves both films in one payload.
    # Reading only the first run_meta let that file through the refusal it was written for.
    rows = [
        {"row_type": "run_meta", "tier": "fast"},
        *cell("100K", "base", "rep0", {"open_close_ms": 1000.0}),
        *cell("100K", "treatment", "rep0", {"open_close_ms": 900.0}),
        {"row_type": "run_meta", "tier": "standard"},
    ]
    path = write(tmp_path, "mixed", rows)
    assert F.tiers_of(F.read_rows(path)) == {"fast", "standard"}
    with pytest.raises(SystemExit) as exc:
        F.load([path])
    assert "different tiers" in str(exc.value)


# ── the session is part of a pair's identity ─────────────────────────


def timed_action(cid: str, sid: str, ms: float) -> dict:
    return {
        "row_type": "action",
        "cell_id": cid,
        "session_id": sid,
        "action": "message_menu",
        "ran": True,
        "expect_ok": True,
        "timings": {"open_close_ms": ms},
    }


def resumed_payload(tmp_path: Path, name: str, retry_session: str) -> Path:
    """One arm completed, its partner died, and `--resume` re-ran the dead arm.

    `retry_session` is the session the retry was recorded under. The real `--resume` mints a new
    one; passing the original back is the control that shows the refusal keys on the session and
    not on there being two attempts.
    """
    return write(
        tmp_path,
        name,
        [
            {"row_type": "run_meta", "tier": "standard", "session_id": "s1"},
            {
                "row_type": "cell",
                "cell_id": "r100K.base.rep0",
                "session_id": "s1",
                "completed": True,
            },
            timed_action("r100K.base.rep0", "s1", 100.0),
            {
                "row_type": "cell",
                "cell_id": "r100K.treatment.rep0",
                "session_id": "s1",
                "completed": False,
            },
            timed_action("r100K.treatment.rep0", "s1", 999.0),
            {"row_type": "run_meta", "tier": "standard", "session_id": retry_session},
            {
                "row_type": "cell",
                "cell_id": "r100K.treatment.rep0",
                "session_id": retry_session,
                "completed": True,
            },
            # 8%: the cross-session drift this module's header measures, with no real effect in it.
            timed_action("r100K.treatment.rep0", retry_session, 108.0),
        ],
    )


def test_an_arm_resumed_into_a_new_session_is_not_paired_with_the_old_one(tmp_path):
    # `--resume` skips the arm that completed and re-runs the one that died, under a NEW session id
    # in the SAME shard directory. Keyed on the repetition alone the two pair, and the whole 8%
    # session drift is charged to whichever arm was re-run. `scoring/ab.py` refuses exactly this
    # comparison; the sweep's own pairing has to refuse it too.
    path = resumed_payload(tmp_path, "resumed", "s2")
    assert F.cell_metrics(F.read_rows(path))["r100K.treatment.rep0"] == {
        "message_menu.open_close_ms": 108.0
    }
    assert F.paired(F.read_rows(path), shard = "resumed") == {}


def test_an_arm_resumed_inside_the_same_session_still_pairs(tmp_path):
    # The other direction, so the refusal above cannot pass by rejecting every resumed run. Two
    # attempts in ONE session are still one session, and the retry's own numbers pair normally.
    path = resumed_payload(tmp_path, "same", "s1")
    assert F.paired(F.read_rows(path), shard = "same") == {
        "message_menu.open_close_ms": [(100.0, 108.0)]
    }


def test_a_payload_with_no_session_ids_pairs_exactly_as_before(tmp_path):
    # Payloads recorded before session ids existed carry none, so both arms resolve to "" and the
    # new key term is inert. A refusal that also rejected these would delete every old reading.
    path = payload(tmp_path, "legacy", [(1000.0, 900.0)])
    assert F.paired(F.read_rows(path), shard = "legacy") == {
        "message_menu.open_close_ms": [(1000.0, 900.0)]
    }


# ── ui parity: the rung is part of a pair's identity ──────────────────


def parity_action(cid: str, action: str, digest: str) -> dict:
    capture = {
        "parity_attempted": True,
        "root_kind": "thread",
        "chars": 10,
        "digest": digest,
        "messages": [{"i": 0, "role": "assistant", "chars": 10, "digest": digest}],
        "overlays": [],
        "style": {"style_attempted": True, "capped": False, "nodes": []},
    }
    return {
        "row_type": "action",
        "cell_id": cid,
        "action": action,
        "ran": True,
        "timings": {"open_ms": 5.0},
        "parity": capture,
    }


def test_a_mismatch_at_a_smaller_rung_survives_the_later_rungs(tmp_path):
    # A standard-tier repetition walks 1K, 10K and 100K. Keyed on the repetition alone, the 100K
    # rows overwrite the 1K ones and a real difference at 1K is reported as a pass.
    rows = [{"row_type": "run_meta", "tier": "standard"}]
    for rung, base_digest, treat_digest in (("r1K", "AAA", "BBB"), ("r100K", "CCC", "CCC")):
        rows.append(parity_action(f"{rung}.A0.rep0", "settings", base_digest))
        rows.append(parity_action(f"{rung}.treatment.rep0", "settings", treat_digest))
    path = write(tmp_path, "mine", rows)

    assert len(U.collect([path])["pairs"]) == 2
    results, _ = U.compare_all([path])
    assert sorted(cell for _a, _s, cell, _r in results) == ["r100K rep0", "r1K rep0"]
    assert U.report([path], "t", U.UNSTABLE_ACTIONS) == 1


def test_matching_rungs_still_report_a_pass(tmp_path):
    rows = [{"row_type": "run_meta", "tier": "standard"}]
    for rung in ("r1K", "r10K", "r100K"):
        rows.append(parity_action(f"{rung}.A0.rep0", "settings", "CCC"))
        rows.append(parity_action(f"{rung}.treatment.rep0", "settings", "CCC"))
    path = write(tmp_path, "clean", rows)
    assert U.report([path], "t", U.UNSTABLE_ACTIONS) == 0


# ── ui parity: derived instability is a property of the rung too ─────


def parity_run(tmp_path: Path, name: str, cells: list[tuple[str, str, str, str]]) -> Path:
    """`cells` is (rung, rep, base digest, treatment digest) for the action `settings`."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard"}]
    for rung, rep, base_digest, treat_digest in cells:
        rows.append(parity_action(f"{rung}.base.{rep}", "settings", base_digest))
        rows.append(parity_action(f"{rung}.treatment.{rep}", "settings", treat_digest))
    return write(tmp_path, name, rows)


def test_instability_measured_at_one_rung_does_not_silence_a_stable_rung(tmp_path):
    # A standard-tier null control at the default --reps 1 walks 1K, 10K and 100K. `settings`
    # races only at 100K, where the mounted thread is largest. Derived over the pooled action
    # name, that single differing observation borrows the other two rungs' observation COUNT,
    # clears min_observations, and marks `settings` unstable everywhere -- so a real DOM
    # regression at 1K, a rung the null control measured clean, prints as expected variation and
    # the command exits 0.
    null = parity_run(
        tmp_path,
        "null",
        [("r1K", "rep0", "Q", "Q"), ("r10K", "rep0", "Q", "Q"), ("r100K", "rep0", "X", "Y")],
    )
    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "settings") not in unstable  # one observation is not evidence
    assert "settings" not in unstable

    mine = parity_run(
        tmp_path,
        "mine",
        [
            ("r1K", "rep0", "Q", "REGRESSED"),
            ("r10K", "rep0", "Q", "Q"),
            ("r100K", "rep0", "X", "X"),
        ],
    )
    assert U.report([mine], "t", unstable) == 1


def test_instability_measured_at_a_rung_still_silences_that_rung(tmp_path):
    # The other direction, so the scoping above cannot pass by never silencing anything. With two
    # repetitions the null control has the observations to MEAN it at 100K, and a mismatch there
    # is expected variation -- while the same action at 1K, measured clean, still carries a
    # verdict.
    null = parity_run(
        tmp_path,
        "null2",
        [
            ("r1K", "rep0", "Q", "Q"),
            ("r1K", "rep1", "Q", "Q"),
            ("r100K", "rep0", "X", "Y"),
            ("r100K", "rep1", "X", "Z"),
        ],
    )
    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "settings") in unstable
    assert ("r1K", "settings") not in unstable

    at_100k = parity_run(tmp_path, "m100", [("r1K", "rep0", "Q", "Q"), ("r100K", "rep0", "X", "W")])
    assert U.report([at_100k], "t", unstable) == 0
    at_1k = parity_run(tmp_path, "m1k", [("r1K", "rep0", "Q", "W"), ("r100K", "rep0", "X", "X")])
    assert U.report([at_1k], "t", unstable) == 1


def test_a_declared_unstable_action_still_holds_at_every_rung(tmp_path):
    # A declared entry carries a MECHANISM that is a property of the action, so it is not scoped
    # to a rung and a null control is not needed to honour it.
    #
    # NOT 1 is what this test is about: the difference is filed as expected variation rather than
    # as a stable difference. It is 2 and not 0 because the only pair in this payload landed in
    # that bucket, so nothing was compared at all -- see `report`'s `matched == 0` guard, which
    # refuses to exit 0 over a run where the digest never produced a verdict.
    rows = [{"row_type": "run_meta", "tier": "standard"}]
    rows.append(parity_action("r1K.base.rep0", "stop_generation", "A"))
    rows.append(parity_action("r1K.treatment.rep0", "stop_generation", "B"))
    path = write(tmp_path, "declared", rows)
    assert U.report([path], "t", U.unstable_set(None)[0]) == 2


# ── ui parity: the session is part of a pair's identity too ──────────


def in_session(row: dict, sid: str) -> dict:
    row["session_id"] = sid
    return row


def resumed_parity(tmp_path: Path, name: str, retry_session: str) -> Path:
    """A null control whose treatment arm died at 100K and was re-run by `--resume`.

    The base arm completed under `s1` and stays there; `--resume` skips it and re-runs only the
    arm that died, under `retry_session`, appending into the SAME shard directory. The retry's
    digests carry a session-scoped volatile the normaliser did not catch, so they differ from the
    base arm's for a reason that has nothing to do with either build.

    `retry_session` is the session the retry was recorded under. The real `--resume` mints a new
    one; passing `s1` back is the control that shows the refusal keys on the SESSION and not on
    there being two attempts.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    for rep in ("rep0", "rep1"):
        rows.append(in_session(parity_action(f"r100K.base.{rep}", "settings", "STABLE"), "s1"))
    rows.append({"row_type": "run_meta", "tier": "standard", "session_id": retry_session})
    for rep in ("rep0", "rep1"):
        rows.append(
            in_session(
                parity_action(f"r100K.treatment.{rep}", "settings", f"VOLATILE-{rep}"),
                retry_session,
            )
        )
    return write(tmp_path, name, rows)


def test_a_resumed_arm_is_not_paired_with_its_partner_from_the_old_session(tmp_path):
    # `--resume` re-runs the arm that died under a NEW session id in the SAME shard, so the
    # completed partner is left under the old one. Keyed on the repetition alone the two pair, and
    # every session-scoped volatile the normaliser missed reads as a difference between the arms.
    # Two of them at one rung are enough for `derive_unstable` to call `settings` unstable AT THAT
    # RUNG, and a real DOM regression at 100K then prints as expected variation and exits 0.
    null = resumed_parity(tmp_path, "null_resumed", "s2")
    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "settings") not in unstable
    assert "settings" not in unstable

    regressed = parity_run(tmp_path, "mine_resumed", [("r100K", "rep0", "X", "REGRESSED")])
    assert U.report([regressed], "t", unstable) == 1


def test_a_cross_session_pair_carries_no_verdict_in_either_direction(tmp_path):
    # And it is not silently dropped either. Both arms ran, so the reader is told the surface went
    # unmeasured rather than being shown a pass -- which is the distinction the NOT COMPARABLE
    # outcome exists to make.
    null = resumed_parity(tmp_path, "blind_resumed", "s2")
    verdicts = {r["verdict"] for _a, _s, _c, r in U.compare_all([null])[0]}
    assert verdicts == {U.P.NOT_COMPARABLE}
    # NOT 1, because a cross-session pair is not a stable difference, and NOT 0 either: every pair
    # here is NOT COMPARABLE, so the payload carries no verdict in either direction and `report`
    # says so with 2 rather than letting CI go green on a run that compared nothing.
    assert U.report([null], "t", U.UNSTABLE_ACTIONS) == 2


def test_a_parity_arm_resumed_inside_the_same_session_still_pairs(tmp_path):
    # The other direction, so the refusal above cannot pass by rejecting every resumed payload.
    # Two attempts in ONE session are still one session, the pairs are real, and the instability
    # they show is derived exactly as before.
    null = resumed_parity(tmp_path, "same_session", "s1")
    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "settings") in unstable


def test_a_parity_payload_with_no_session_ids_pairs_exactly_as_before(tmp_path):
    # Payloads recorded before session ids existed carry none, so both arms resolve to "" and the
    # new key term is inert. A refusal that also rejected these would blind the tool to every
    # older run.
    path = parity_run(tmp_path, "legacy_parity", [("r1K", "rep0", "A", "B")])
    assert U.report([path], "t", U.UNSTABLE_ACTIONS) == 1


# ── ui parity: a superseded attempt is not an observation ────────────


def parity_cell(cid: str, arm: str, sid: str, rep: str, completed: bool) -> dict:
    """The `cell` row the recorder closes an attempt with, which is what names the attempt.

    `CellRunner.run` emits it from a `finally`, so it lands AFTER every action row the scene
    wrote and it lands whether the cell completed or not.
    """
    return {
        "row_type": "cell",
        "cell_id": cid,
        "session_id": sid,
        "cell": {"arm": arm, "rep": int(rep.removeprefix("rep"))},
        "target_tokens": 100_000,
        "completed": completed,
    }


def drained_arm(rows: list[dict], arm: str, rep: str, sid: str, digest: str, completed: bool):
    """Append one arm's attempt: its action row, then the cell row that closes it."""
    cid = f"r100K.{arm}.{rep}"
    rows.append(in_session(parity_action(cid, "settings", digest), sid))
    rows.append(parity_cell(cid, arm, sid, rep, completed))


def resumed_both_arms(tmp_path: Path, name: str) -> Path:
    """A null control interrupted mid-pair and resumed the way `skippable_cells` resumes.

    A pair is skipped only when EVERY arm of it completed, so an interruption between the two
    adjacent cells of one repetition re-runs BOTH arms under one new session. The treatment arm
    died inside `stream:drain`, which the scene reaches only AFTER its action rows are written, so
    the dead attempt sits in the append-only payload carrying a full set of digests -- and one of
    them differs, because it was captured off an arm that was already on its way down.

    So the file holds a complete base/treatment pair under `s1` and another under `s2`, and they
    are one logical repetition, not two.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    drained_arm(rows, "base", "rep0", "s1", "STABLE", True)
    drained_arm(rows, "treatment", "rep0", "s1", "DYING", False)
    rows.append({"row_type": "run_meta", "tier": "standard", "session_id": "s2"})
    drained_arm(rows, "base", "rep0", "s2", "STABLE", True)
    drained_arm(rows, "treatment", "rep0", "s2", "STABLE", True)
    return write(tmp_path, name, rows)


def test_a_superseded_attempt_is_not_a_second_parity_observation(tmp_path):
    # One repetition ran twice, so the payload holds two comparable pairs for it. Counted as two,
    # the differing dead attempt and the matching retry are exactly `min_observations` and
    # `derive_unstable` marks `settings` unstable at 100K on the strength of a reading the run
    # itself threw away.
    null = resumed_both_arms(tmp_path, "null_both_arms")
    assert len(U.collect([null])["pairs"]) == 1

    unstable, derived, _checks = U.unstable_set([null])
    assert derived["settings@r100K"]["observations"] == 1
    assert derived["settings@r100K"]["differed"] == 0
    assert ("r100K", "settings") not in unstable
    assert "settings" not in unstable


def test_a_superseded_attempt_does_not_silence_a_real_parity_regression(tmp_path):
    # The consequence, end to end. Scored against that unstable set, a genuine DOM difference on
    # `settings` at 100K prints under "expected to vary" and the gate exits 0 on a regression.
    unstable, _derived, _checks = U.unstable_set([resumed_both_arms(tmp_path, "null_silencer")])
    regressed = parity_run(tmp_path, "mine_both_arms", [("r100K", "rep0", "X", "REGRESSED")])
    assert U.report([regressed], "t", unstable) == 1


def test_two_repetitions_of_one_pair_are_still_two_parity_observations(tmp_path):
    # The control: superseding is keyed on the ATTEMPT, not on the cell id, so two repetitions of
    # one rung inside one session remain two independent observations and still derive as
    # unstable. A filter that kept only one reading per cell would pass the two tests above by
    # deleting the evidence they are meant to preserve.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    for rep, treat in (("rep0", "Y"), ("rep1", "Z")):
        drained_arm(rows, "base", rep, "s1", "X", True)
        drained_arm(rows, "treatment", rep, "s1", treat, True)
    null = write(tmp_path, "two_reps", rows)

    assert len(U.collect([null])["pairs"]) == 2
    unstable, derived, _checks = U.unstable_set([null])
    assert derived["settings@r100K"]["observations"] == 2
    assert ("r100K", "settings") in unstable


def test_an_attempt_that_was_never_re_run_still_carries_its_parity_verdict(tmp_path):
    # The other control: a cell that died and was never resumed is the LATEST attempt at itself,
    # so its rows stay. Dropping every incomplete cell instead would blind the tool to the runs
    # that have the most to say, and would silence the difference below rather than report it.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    drained_arm(rows, "base", "rep0", "s1", "STABLE", True)
    drained_arm(rows, "treatment", "rep0", "s1", "DIFFERENT", False)
    path = write(tmp_path, "never_resumed", rows)

    assert len(U.collect([path])["pairs"]) == 1
    assert U.report([path], "t", U.UNSTABLE_ACTIONS) == 1


# ── ui parity: one set, one tier ─────────────────────────────────────


def two_tier_parity(tmp_path: Path, name: str, fast: tuple[str, str], standard: tuple[str, str]):
    """One shard holding a fast run and a standard run, both at 100K, appended in that order.

    The shape a second `--out` reuse at the other tier leaves behind: `fast` walks 100K only and
    `standard` walks 1K, 10K and 100K, so the two films meet on that rung.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast", "session_id": "s1"}]
    rows.append(in_session(parity_action("r100K.base.rep0", "settings", fast[0]), "s1"))
    rows.append(in_session(parity_action("r100K.treatment.rep0", "settings", fast[1]), "s1"))
    rows.append({"row_type": "run_meta", "tier": "standard", "session_id": "s2"})
    rows.append(in_session(parity_action("r100K.base.rep0", "settings", standard[0]), "s2"))
    rows.append(in_session(parity_action("r100K.treatment.rep0", "settings", standard[1]), "s2"))
    return write(tmp_path, name, rows)


def test_a_null_control_holding_two_tiers_is_refused_before_the_set_is_derived(tmp_path):
    # THE POOLING THIS USED TO DEMONSTRATE IS GONE, and it was fixed underneath rather than here.
    # The setup assertion used to be `("r100K", "settings") in unstable`: one differing 100K pair
    # from the fast film and one matching 100K pair from the standard film pooled into exactly
    # `min_observations`, marking the action unstable on the strength of two different films.
    #
    # `latest_attempt_rows` now keys an attempt on any attempt-stamped row rather than on the
    # terminal cell row alone, and the two films write the SAME cell ids, so they are read as two
    # attempts of one cell and the superseded one is dropped before pairing. Only the standard
    # film's matching pair survives, so there is one observation and nothing is derived. Asserting
    # the old pooling here would be asserting a bug that no longer exists.
    null = two_tier_parity(tmp_path, "null_mixed", ("X", "Y"), ("Q", "Q"))
    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "settings") not in unstable

    # The refusal is kept regardless, as the layer that does not depend on the two films colliding
    # on a cell id. Two tiers that walk different ladders need not collide at all, and a set
    # derived across films is wrong whether or not de-duplication happened to absorb it.
    with pytest.raises(SystemExit) as exc:
        U.main([str(tmp_path / "mine_any"), "--null", str(null.parent)])
    assert "more than one tier" in str(exc.value)


def test_two_mixed_tier_sets_do_not_pass_by_matching_each_other(tmp_path):
    # The case the tier-mismatch WARNING cannot see: both sides were re-run at the other tier, so
    # both sets are {fast, standard}, they compare EQUAL, and no warning fires. The pooled null
    # control then silences `settings` at 100K and the real 100K regression in the payload prints
    # as expected variation with the command exiting 0.
    null = two_tier_parity(tmp_path, "null_both", ("X", "Y"), ("Q", "Q"))
    mine = two_tier_parity(tmp_path, "mine_both", ("Q", "Q"), ("Q", "REGRESSED"))
    assert U.tier_of([null]) == U.tier_of([mine]) == {"fast", "standard"}

    with pytest.raises(SystemExit) as exc:
        U.main([str(mine.parent), "--null", str(null.parent)])
    assert "more than one tier" in str(exc.value)


def test_a_payload_holding_two_tiers_is_refused_even_with_no_null_control(tmp_path):
    # The declared unstable set is not derived from anything, so nothing is pooled -- but the
    # payload's own 100K pairs still come from two films, and reporting them as repetitions of one
    # measurement is the same misreading.
    mine = two_tier_parity(tmp_path, "mine_alone", ("Q", "Q"), ("Q", "REGRESSED"))
    with pytest.raises(SystemExit) as exc:
        U.main([str(mine.parent)])
    assert "more than one tier" in str(exc.value)


def test_one_tier_on_each_side_still_scores_in_both_directions(tmp_path):
    # The control, so the refusal cannot pass by rejecting every run. A single-tier null control
    # and a single-tier payload score exactly as before, and a real regression still exits 1.
    null = parity_run(tmp_path, "null_one", [("r100K", "rep0", "Q", "Q")])
    clean = parity_run(tmp_path, "mine_clean", [("r100K", "rep0", "Q", "Q")])
    regressed = parity_run(tmp_path, "mine_bad", [("r100K", "rep0", "Q", "REGRESSED")])
    assert U.main([str(clean.parent), "--null", str(null.parent)]) == 0
    assert U.main([str(regressed.parent), "--null", str(null.parent)]) == 1


def test_a_tier_mismatch_between_two_single_tier_sets_is_refused(tmp_path, capsys):
    # This used to warn and then score anyway, which is the worst of the two options: the warning
    # said the derived set does not transfer, and then the run was scored against it regardless.
    # A set that does not transfer is not a weaker excuse than a real one, it is an arbitrary one.
    #
    # Exit 2, not 1. The payload below does carry a regression, so the old assertion of exit 1 was
    # passing for a reason unrelated to the tier check. Refusing to answer and reporting a parity
    # failure have to be distinguishable, or a refusal sends somebody hunting for a UI change that
    # was never measured.
    null = write(
        tmp_path,
        "null_fast",
        [
            {"row_type": "run_meta", "tier": "fast"},
            parity_action("r100K.base.rep0", "settings", "Q"),
            parity_action("r100K.treatment.rep0", "settings", "Q"),
        ],
    )
    mine = parity_run(tmp_path, "mine_standard", [("r100K", "rep0", "Q", "REGRESSED")])
    assert U.main([str(mine.parent), "--null", str(null.parent)]) == 2
    assert "REFUSING to score" in capsys.readouterr().out


def test_a_corpus_mismatch_between_two_valid_sides_is_refused(tmp_path, capsys):
    # The likelier of the two in practice, and the one that had no check at all: each side is a
    # perfectly valid single corpus, so `one_corpus` passes on both, and the null's set was still
    # applied to a payload that rendered a different thread. Which actions race is a property of
    # the thread the film drove, so that set describes something the payload never displayed.
    def at(name, corpus, cells):
        rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "corpus_hash": corpus}]
        for rung, rep, base_digest, treat_digest in cells:
            rows.append(parity_action(f"{rung}.base.{rep}", "settings", base_digest))
            rows.append(parity_action(f"{rung}.treatment.{rep}", "settings", treat_digest))
        return write(tmp_path, name, rows)

    null = at("null_v1", "v1", [("r100K", "rep0", "Q", "Q")])
    same = at("mine_v1", "v1", [("r100K", "rep0", "Q", "Q")])
    other = at("mine_v2", "v2", [("r100K", "rep0", "Q", "Q")])
    # The control first, so the refusal cannot pass by rejecting everything.
    assert U.main([str(same.parent), "--null", str(null.parent)]) == 0
    assert U.main([str(other.parent), "--null", str(null.parent)]) == 2
    assert "REFUSING to score" in capsys.readouterr().out


def test_a_side_recorded_before_corpus_hashes_existed_is_still_scored(tmp_path, capsys):
    # An absent hash is not a disagreement. Refusing it would reject every payload recorded before
    # the field existed, which is the whole archive.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "corpus_hash": "v1"}]
    rows.append(parity_action("r100K.base.rep0", "settings", "Q"))
    rows.append(parity_action("r100K.treatment.rep0", "settings", "Q"))
    hashed = write(tmp_path, "hashed", rows)
    legacy = parity_run(tmp_path, "legacy", [("r100K", "rep0", "Q", "Q")])
    assert U.main([str(legacy.parent), "--null", str(hashed.parent)]) == 0
    assert U.main([str(hashed.parent), "--null", str(legacy.parent)]) == 0


def test_main_prints_a_mixed_unstable_set_without_dying(tmp_path, capsys):
    # The set now holds bare action names and (rung, action) pairs together, and `sorted()` over
    # the two raises TypeError. The one place that formats it is the run header.
    null = parity_run(tmp_path, "nullm", [("r1K", "rep0", "X", "Y"), ("r1K", "rep1", "X", "Z")])
    mine = parity_run(tmp_path, "minem", [("r1K", "rep0", "Q", "Q")])
    assert U.main([str(mine.parent), "--null", str(null.parent)]) == 0
    assert "settings@r1K" in capsys.readouterr().out


def test_main_without_a_floor_says_so_and_still_prints(tmp_path, capsys):
    payload(tmp_path, "result", [(1000.0, 100.0)] * 4)
    assert F.main([str(tmp_path / "result")]) == 0
    assert "NO FLOOR SUPPLIED" in capsys.readouterr().out


def test_main_returns_two_when_nothing_matches(tmp_path, capsys):
    assert F.main([str(tmp_path / "does-not-exist")]) == 2
    assert "no payload found" in capsys.readouterr().out


# ── a probe payload is not a measurement ─────────────────────────────


def probe_payload(tmp_path: Path, name: str, script: str | None) -> Path:
    """A payload whose run_meta records the external init script that was in the page."""
    path = payload(tmp_path, name, [(1000.0, 900.0)] * 4)
    rows = [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines()]
    rows[0]["probe_init_script"] = script
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_payload_recorded_with_a_probe_installed_is_refused(tmp_path):
    """The instrument was in the shot. There is no flag to override it.

    A probe samples the DOM on its own schedule and forces layout to do it, so the timings are a
    measurement of the page AND the instrument. Without this the run looks entirely ordinary: it
    records the same cells and renders the same A/B table, and only the caller's memory stops the
    numbers being quoted.
    """
    path = probe_payload(tmp_path, "probed", "arms/content_visibility_probe.js")
    with pytest.raises(SystemExit) as excinfo:
        F.load([path])
    assert "external init script" in str(excinfo.value)
    assert "content_visibility_probe.js" in str(excinfo.value)


def test_a_probe_named_in_a_later_run_meta_is_still_caught(tmp_path):
    """`--resume` APPENDS a second run_meta and the remaining cells to the existing file.

    So a payload can carry a clean header above cells that were re-recorded under a probe.
    Reading only the first run_meta scores those cells.
    """
    path = probe_payload(tmp_path, "resumed", None)
    with path.open("a", encoding = "utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "row_type": "run_meta",
                    "tier": "standard",
                    "corpus_hash": "corpus0000",
                    "probe_init_script": "late_probe.js",
                }
            )
            + "\n"
        )
    with pytest.raises(SystemExit) as excinfo:
        F.load([path])
    assert "late_probe.js" in str(excinfo.value)


def test_a_failed_probe_free_gate_is_enough_on_its_own(tmp_path):
    # Two independent records of one fact, so a payload from a version that emits only the gate
    # is still refused.
    path = probe_payload(tmp_path, "gated", None)
    with path.open("a", encoding = "utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "row_type": "gate",
                    "name": "probe_free",
                    "passed": False,
                    "detail": {"probe_init_script": "gate_only.js"},
                }
            )
            + "\n"
        )
    with pytest.raises(SystemExit) as excinfo:
        F.load([path])
    assert "gate_only.js" in str(excinfo.value)


def test_a_passing_probe_free_gate_scores_normally(tmp_path):
    path = probe_payload(tmp_path, "clean_gate", None)
    with path.open("a", encoding = "utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "row_type": "gate",
                    "name": "probe_free",
                    "passed": True,
                    "detail": {"probe_init_script": None},
                }
            )
            + "\n"
        )
    pooled, _ = F.load([path])
    assert pooled["message_menu.open_close_ms"]


def test_the_report_path_refuses_a_probed_payload_too(tmp_path):
    """floor_table is not the only reader. `--report` scores the same file afterwards."""
    from tests.studio.studiobench.report.build import score_payload

    path = probe_payload(tmp_path, "probed_report", "arms/content_visibility_probe.js")
    with pytest.raises(SystemExit) as excinfo:
        score_payload(path)
    assert "external init script" in str(excinfo.value)


def test_a_null_probe_field_is_the_ordinary_scorable_case(tmp_path):
    # Explicit null and absent must both score, or every payload written before the field
    # existed becomes unreadable.
    pooled, _ = F.load([probe_payload(tmp_path / "explicit", "clean", None)])
    assert pooled["message_menu.open_close_ms"]
    pooled, _ = F.load([payload(tmp_path / "absent", "clean", [(1000.0, 900.0)] * 4)])
    assert pooled["message_menu.open_close_ms"]


def test_the_composer_click_does_not_set_the_frame_floor(tmp_path):
    """The floor table pools the cell's windows too. An 11 s `setup:composer_click` -- almost all
    of it Playwright's actionability check running on the page's main thread -- would become this
    table's `max_frame_ms` floor and swallow every regression smaller than itself."""

    def _frames(gaps):
        return {
            "frames": {
                "frames_attempted": True,
                "frame_gaps_ms": gaps,
                "frame_gaps_truncated": False,
                "max_frame_ms": max(gaps),
            }
        }

    rows = [
        {"row_type": "run_meta", "tier": "standard"},
        {"row_type": "cell", "cell_id": "100K.base.rep0", "completed": True},
        {
            "row_type": "window",
            "cell_id": "100K.base.rep0",
            "kind": "action",
            "duration_ms": 2000.0,
            "instruments": _frames([120.0] * 10),
        },
        {
            "row_type": "window",
            "cell_id": "100K.base.rep0",
            "name": "setup:composer_click",
            "kind": "setup",
            "duration_ms": 11_000.0,
            "instruments": _frames([11_000.0]),
        },
    ]
    out = tmp_path / "click"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    metrics = F.cell_metrics(F.read_rows(out / "payload.jsonl"))["100K.base.rep0"]
    assert metrics["max_frame_ms"] == 120.0


# ── the documented loop runs liveness before it reads a timing ───────


LOOP_DOC = Path(__file__).resolve().parents[2] / "CONTRIBUTING-perf.md"


def loop_commands() -> list[str]:
    """The commands in the fenced block under `## The loop`, in the order a reader runs them."""
    block = LOOP_DOC.read_text(encoding = "utf-8").split("## The loop", 1)[1].split("```")[1]
    return [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def index_of(commands: list[str], needle: str) -> int:
    for i, line in enumerate(commands):
        if needle in line:
            return i
    raise AssertionError(f"the documented loop no longer runs {needle}")


SLOT_MISSED = (
    "the slot opened at 73000ms and this machine reached it at 79000ms, past its 12000ms budget"
)


def liveness_action(cell_id: str, ms: float | None) -> dict:
    """One action row. `ms` of `None` is the repetition whose slot the film moved on without."""
    if ms is None:
        return {
            "row_type": "action",
            "cell_id": cell_id,
            "action": "message_menu",
            "ran": False,
            "slot_missed": True,
            "timings": {},
            "reason": SLOT_MISSED,
        }
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "action": "message_menu",
        "ran": True,
        "slot_missed": False,
        "timings": {"open_close_ms": ms},
        "reason": None,
    }


def liveness_payload(tmp_path: Path, name: str, pairs: list[tuple[float, float | None]]) -> Path:
    """One shard whose cells carry their own actions, which is what `--assert-liveness` reads.

    `paired` reads the standalone action rows and the liveness gate reads the list embedded in the
    cell row, so a payload that exercises both has to carry the action twice, exactly as a real
    `SceneRunner` cell does.
    """
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard"}]
    for i, (base, treat) in enumerate(pairs):
        for arm, ms in (("base", base), ("treatment", treat)):
            cid = f"100K.{arm}.rep{i}"
            action = liveness_action(cid, ms)
            rows.append(
                {"row_type": "cell", "cell_id": cid, "completed": True, "actions": [action]}
            )
            rows.append(action)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_the_documented_loop_asserts_liveness_before_it_reads_any_timing():
    commands = loop_commands()
    liveness = index_of(commands, "--assert-liveness")
    for reader in ("sweep.floor_table", "sweep.ui_parity", "--report"):
        assert liveness < index_of(commands, reader), (
            f"the loop runs {reader} before --assert-liveness, so a contributor reads a verdict "
            f"out of a payload nothing has checked for missed slots"
        )


def test_floor_table_prints_a_clean_verdict_from_a_run_that_liveness_voids(tmp_path, capsys):
    # WHY THAT ORDER IS LOAD-BEARING, and the reason the line above is a gate rather than a note.
    # The treatment build is 10% faster on the three repetitions it managed, and on the fourth it
    # was so slow that `message_menu` never reached its slot. `paired` matches on the metrics BOTH
    # arms recorded, so that repetition leaves the table with no trace except a smaller `n`, and
    # the survivors print as a clean win over a tight floor.
    mine = liveness_payload(
        tmp_path, "mine", [(1000.0, 900.0), (1000.0, 900.0), (1000.0, 900.0), (1000.0, None)]
    )
    null = liveness_payload(
        tmp_path,
        "null",
        [(1000.0, 1000.0), (1010.0, 1012.0), (990.0, 989.0), (1000.0, 1001.0)],
    )
    assert F.main(["--floor", str(null.parent), str(mine.parent)]) == 0
    table = capsys.readouterr().out
    assert "faster" in table
    assert "1 metric(s) cleared all three gates." in table
    # The same payload, read by the gate the loop now runs first.
    assert cli_main(["--assert-liveness", str(mine)]) == 1


def test_the_repetition_that_missed_its_slot_is_what_the_verdict_turns_on(tmp_path, capsys):
    # The control for the test above. With the fourth repetition's timing present, the same run is
    # VOID on the same floor: the clean win is an artefact of the reading that went missing, not a
    # property of the change. Every other number in the payload is identical.
    mine = liveness_payload(
        tmp_path, "mine", [(1000.0, 900.0), (1000.0, 900.0), (1000.0, 900.0), (1000.0, 2500.0)]
    )
    null = liveness_payload(
        tmp_path,
        "null",
        [(1000.0, 1000.0), (1010.0, 1012.0), (990.0, 989.0), (1000.0, 1001.0)],
    )
    assert F.main(["--floor", str(null.parent), str(mine.parent)]) == 0
    table = capsys.readouterr().out
    assert "VOID (pairs disagree on sign)" in table
    assert "0 metric(s) cleared all three gates." in table
    assert cli_main(["--assert-liveness", str(mine)]) == 0


# ── the null control needs the same liveness gate the treatment run gets ───────


# A null control is base against base, so its four paired ratios are the noise this machine puts
# between two identical builds. The fourth repetition is the one that hiccupped.
NULL_WITH_A_NOISY_REPETITION = [
    (1000.0, 1000.0),
    (1010.0, 1012.0),
    (990.0, 989.0),
    (1000.0, 1300.0),
]

# The same null control on a run where that fourth repetition was slow enough to miss the slot, so
# the reading that carried the noise is the reading that never happened.
NULL_THAT_MISSED_THE_SLOT = NULL_WITH_A_NOISY_REPETITION[:3] + [(1000.0, None)]

# A treatment payload that is 10% faster, consistently, with a spread well inside its own effect:
# it clears gates 2 and 3 outright, so the only thing standing between it and `faster` is gate 1.
MINE_TEN_PERCENT_FASTER = [
    (1000.0, 900.0),
    (1000.0, 905.0),
    (1000.0, 895.0),
    (1000.0, 900.0),
]


def liveness_commands(commands: list[str]) -> list[str]:
    return [line for line in commands if "--assert-liveness" in line]


def credentials_prose() -> str:
    """The paragraph between the loop's fenced block and the first numbered section."""
    body = LOOP_DOC.read_text(encoding = "utf-8").split("## The loop", 1)[1]
    return body.split("```")[2].split("## 1.")[0]


def test_the_documented_loop_asserts_liveness_on_the_null_control_too():
    commands = loop_commands()
    gated = liveness_commands(commands)
    for payload in ("outputs/mine", "outputs/null"):
        assert any(payload in line for line in gated), (
            f"the loop never runs --assert-liveness on {payload}/payload.jsonl, so a missed slot "
            f"in that run reaches the verdict table unchecked"
        )
    last_gate = max(commands.index(line) for line in gated)
    for reader in ("sweep.floor_table", "sweep.ui_parity", "--report"):
        assert last_gate < index_of(commands, reader), (
            f"the loop runs {reader} before both liveness gates, so a contributor reads a verdict "
            f"built from a payload nothing has checked for missed slots"
        )


def test_the_doc_says_which_commands_in_the_loop_need_a_studio():
    # The loop is now four offline commands and three that drive a browser, so "every command
    # above except the last" bills `--assert-liveness`, `floor_table` and `ui_parity` for
    # credentials none of them read. All three run against a payload file on disk.
    prose = credentials_prose()
    for offline in ("--assert-liveness", "floor_table", "ui_parity", "--report"):
        assert offline in prose, (
            f"the paragraph under the loop does not say that {offline} runs offline, so it reads "
            f"as though a contributor needs an Unsloth to score a payload they already have"
        )


def test_a_missed_slot_in_the_null_control_prints_noise_as_a_result(tmp_path, capsys):
    # WHY THE NULL CONTROL IS GATED TOO. The floor is `max(|null delta|, null spread)` over the
    # repetitions that survived pairing, and `paired` keys on the metrics BOTH arms recorded, so a
    # null repetition that missed its slot leaves no trace either. The repetition a run drops is
    # the one that was slow enough to miss a budget, which is the noisiest one it had, so the loss
    # is not symmetric: it can only tighten the floor.
    mine = liveness_payload(tmp_path, "mine", MINE_TEN_PERCENT_FASTER)
    null = liveness_payload(tmp_path, "null", NULL_THAT_MISSED_THE_SLOT)
    assert F.main(["--floor", str(null.parent), str(mine.parent)]) == 0
    table = capsys.readouterr().out
    assert "0.3  faster" in table
    assert "1 metric(s) cleared all three gates." in table
    # And the gate the loop used to run only on `outputs/mine` has nothing to say about it: the
    # contributor's own payload is whole. The hole is in the run that set the bar.
    assert cli_main(["--assert-liveness", str(mine)]) == 0
    assert cli_main(["--assert-liveness", str(null)]) == 1


def test_the_null_repetition_that_kept_its_reading_voids_the_same_result(tmp_path, capsys):
    # The control for the test above, and the reading the missed slot removed. The treatment
    # payload is byte-identical; only the null control differs, by the one repetition it managed to
    # finish. Its floor is 30.1% rather than 0.3%, and the same 10% is noise.
    mine = liveness_payload(tmp_path, "mine", MINE_TEN_PERCENT_FASTER)
    null = liveness_payload(tmp_path, "null", NULL_WITH_A_NOISY_REPETITION)
    assert F.main(["--floor", str(null.parent), str(mine.parent)]) == 0
    table = capsys.readouterr().out
    assert "30.1  VOID (under floor)" in table
    assert "0 metric(s) cleared all three gates." in table
    assert cli_main(["--assert-liveness", str(null)]) == 0


# ── the attempt a gate is read through, and the cells a floor is derived from ──


def test_a_gate_from_an_attempt_that_never_closed_still_refuses_its_cell(tmp_path):
    """The winning attempt is named by any attempt-stamped row, not by the terminal `cell` row.

    `CellRunner.run` writes that row from a `finally`, which a SIGKILL or an OOM kill never
    reaches, while the recorder has already flushed the action and gate rows before it. So a
    resume hard-killed inside a cell leaves the OLDER completed session as the only one holding a
    `cell` row. Named from terminal rows alone the winner was that dead-and-buried attempt, and
    the LIVE attempt's own failed gate was discarded as belonging to a superseded session --
    while `latest_attempt_rows` kept that same attempt's action rows, so `collect` scored a cell
    whose self-check had recorded conversation loss with no `_incomplete` stamp on it.
    """
    cid = "r100K.treatment.rep0"
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    # The first attempt completed cleanly and closed itself.
    rows.append(in_session(parity_action(cid, "settings", "STABLE"), "s1"))
    rows.append(parity_cell(cid, "treatment", "s1", "rep0", True))
    # The resume re-ran the same cell, lost messages, and was killed before its `cell` row.
    rows.append({"row_type": "run_meta", "tier": "standard", "session_id": "s2"})
    rows.append(in_session(parity_action(cid, "settings", "LOST"), "s2"))
    rows.append(
        {
            "row_type": "gate",
            "name": U.COMPLETENESS_GATE,
            "passed": False,
            "cell_id": cid,
            "session_id": "s2",
            "detail": {"reason": "the thread lost 3 messages"},
        }
    )
    path = write(tmp_path, "killed_retry", rows)

    refused = U.incomplete_cells([path])
    assert cid in refused, refused
    assert "lost 3 messages" in refused[cid]


def gated_then_resumed(tmp_path: Path, name: str) -> Path:
    """A pair whose treatment arm FAILED `thread_complete`, re-run by `--resume`, retry crashed.

    `ab.skippable_cells` re-runs a pair WHOLE, so a resume with any work left re-attempts both
    arms of a repetition that had already completed. `CellRunner.run` writes its `cell` row from a
    `finally`, so a retry that raises still closes itself -- with `completed=False`.

    The payload then holds, under one cell id, a completed attempt carrying a FAILED gate row and
    a later, incomplete attempt carrying none.
    """
    rows = [
        {"row_type": "run_meta", "tier": "standard", "session_id": "s1"},
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
        timed_action("r100K.base.rep0", "s1", 100.0),
        # The treatment arm lost its thread's middle. It renders fewer rows, so its timing is
        # CHEAPER than a correct cell's, in the direction that flatters the arm.
        {
            "row_type": "cell",
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s1",
            "completed": True,
        },
        timed_action("r100K.treatment.rep0", "s1", 50.0),
        {
            "row_type": "gate",
            "name": "thread_complete",
            "passed": False,
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s1",
            "detail": {"reason": "the thread lost 3 messages"},
        },
    ]
    before = write(tmp_path, name + "_before", list(rows))
    rows += [
        {"row_type": "run_meta", "tier": "standard", "session_id": "s2"},
        timed_action("r100K.base.rep0", "s2", 101.0),
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s2", "completed": False},
        timed_action("r100K.treatment.rep0", "s2", 199.0),
        {
            "row_type": "cell",
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s2",
            "completed": False,
        },
    ]
    # The refusal has to hold on the payload BEFORE the resume, or the test below passes on a
    # payload that was never quotable for some other reason.
    assert F.paired(F.read_rows(before), shard = "b") == {}
    return write(tmp_path, name, rows)


def test_a_superseded_cell_does_not_come_back_when_its_retry_crashes(tmp_path):
    """A resume must not resurrect the reading its own retry replaced.

    Two lenses on one cell id. `failed_invalidating_gates` names the winning attempt from the LAST
    `cell` row whatever its completion state, so the crashed retry supersedes the dead attempt's
    FAILED gate; `cell_metrics` kept the last COMPLETED cell row, which is that same dead
    attempt's. The guard therefore answered about `s2` while the numbers it was guarding came from
    `s1`, and a cell refused for losing its thread's middle before the resume was published after
    it -- pairing 50 ms of a broken thread against a clean 100 ms base arm and printing `faster`,
    which is the 28.2% failure `cell_metrics` documents, arriving through `--resume`.
    """
    path = gated_then_resumed(tmp_path, "gated_resume")
    rows = F.read_rows(path)

    assert F.cell_metrics(rows) == {}
    assert F.paired(rows, shard = "gated_resume") == {}


def test_a_resumed_cell_that_is_not_superseded_still_reports_its_reading(tmp_path):
    """The control: superseding is keyed on a LATER attempt, not on the cell having a gate row.

    Without this, dropping every gate-adjacent or every twice-written cell would pass the test
    above by deleting readings the run legitimately earned.
    """
    rows = [
        {"row_type": "run_meta", "tier": "standard", "session_id": "s1"},
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
        timed_action("r100K.base.rep0", "s1", 100.0),
        {
            "row_type": "cell",
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s1",
            "completed": True,
        },
        timed_action("r100K.treatment.rep0", "s1", 90.0),
    ]
    path = write(tmp_path, "unsuperseded", rows)
    assert F.paired(F.read_rows(path), shard = "unsuperseded") == {
        "message_menu.open_close_ms": [(100.0, 90.0)]
    }


def test_the_visible_floor_is_derived_from_finished_cells_only(tmp_path):
    """An unfinished null-control cell is not an observation of stability.

    Action rows are written as the film runs and the `cell` row when it ends, so a null cell that
    died mid-film leaves a complete-looking set of captures that nothing owns. One DIFFERING
    unfinished observation plus one MATCHING finished one is exactly `min_observations`, which
    marked the action unstable and let `visible_report` file a real difference at the same key
    under "differ against an identical build". `unstable_set` already admits only finished cells
    on the structural side; the visible floor now reads through the same rule.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    for rep, (digest, completed) in enumerate((("X", False), ("same", True))):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.rep{rep}"
            rows.append(
                in_session(
                    {
                        "row_type": "action",
                        "cell_id": cid,
                        "action": "settings",
                        "ran": True,
                        "parity": _capture_for_visible(),
                        "visible": _visible_capture(
                            "X" if (digest == "X" and arm == "treatment") else "same"
                        ),
                    },
                    "s1",
                )
            )
            rows.append(parity_cell(cid, arm, "s1", f"rep{rep}", completed))
    null = write(tmp_path, "null_unfinished", rows)

    assert U.visible_unstable_set([null]) == frozenset()


def _capture_for_visible() -> dict:
    return {
        "parity_attempted": True,
        "root_kind": "thread",
        "chars": 10,
        "digest": "same",
        "messages": [{"i": 0, "role": "assistant", "chars": 10, "digest": "same"}],
        "overlays": [],
        "style": {"style_attempted": True, "capped": False, "nodes": []},
    }


def _visible_capture(digest: str) -> dict:
    return {
        "visible_attempted": True,
        "ever_visible": [1],
        "ever_visible_count": 1,
        "mounted_ever_visible": 1,
        "unmounted_at_capture": 0,
        "messages": {"1": {"role": "assistant", "digest": digest, "chars": 10}},
    }

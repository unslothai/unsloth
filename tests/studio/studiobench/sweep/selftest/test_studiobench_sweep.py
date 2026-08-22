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
) -> Path:
    """One shard: `pairs[i]` is (base, treatment) for repetition i."""
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    rows: list[dict] = [{"row_type": "run_meta", "tier": tier}]
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


def test_main_without_a_floor_says_so_and_still_prints(tmp_path, capsys):
    payload(tmp_path, "result", [(1000.0, 100.0)] * 4)
    assert F.main([str(tmp_path / "result")]) == 0
    assert "NO FLOOR SUPPLIED" in capsys.readouterr().out


def test_main_returns_two_when_nothing_matches(tmp_path, capsys):
    assert F.main([str(tmp_path / "does-not-exist")]) == 2
    assert "no payload found" in capsys.readouterr().out

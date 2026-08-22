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
    rows = [{"row_type": "run_meta", "tier": "standard"}]
    rows.append(parity_action("r1K.base.rep0", "stop_generation", "A"))
    rows.append(parity_action("r1K.treatment.rep0", "stop_generation", "B"))
    path = write(tmp_path, "declared", rows)
    assert U.report([path], "t", U.unstable_set(None)[0]) == 0


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

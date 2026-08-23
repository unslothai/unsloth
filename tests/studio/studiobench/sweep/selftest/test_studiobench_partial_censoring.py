# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A metric censored at some rungs and measured at others must not be printed as a ladder number.

THE GUARD EXISTED AND WAS NEVER CALLED. `payload_rules.refuse_partial_censoring` returned the
correct refusal from the day it landed, and no scoring or sweep code asked it anything -- its only
caller was its own selftest. A guard that cannot fire is the same defect as a row type that is
registered nowhere, and it is the more dangerous of the two, because nothing crashes: the reader
simply believes the case is covered.

What it was supposed to stop is defect 2. `reasoning_toggle.open_ms` is censored on every cell
above the 100K rung, because the open settle exceeds the budget and the timing is withheld rather
than guessed. `paired()` then pools only the cells that could answer, and `render()` prints their
mean under a bare metric name. On a 100K/500K ladder that row is a 100K-only number wearing a
ladder label, and the only hint is a smaller `n` beside the other rows -- which is exactly what a
metric with fewer repetitions looks like.

The refusal LABELS rather than raises. Raising would repeat the mistake this branch's own defect 10
records: `open_ms` is censored above 100K on every standard and full run, so an exiting guard would
abort the normal case and be deleted within a day.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.studio.studiobench.scoring import payload_rules
from tests.studio.studiobench.sweep import floor_table

CENSORED_RUNG = "r500K"
MEASURED_RUNG = "r100K"


def _payload(tmp_path: Path) -> Path:
    """A standard-tier ladder where `open_ms` answers at 100K and is censored at 500K."""
    rows: list[dict] = [
        {
            "row_type": "run_meta",
            "tier": "standard",
            "session_id": "s1",
            "corpus_hash": "abc",
            "rungs": ["100K", "500K"],
        }
    ]
    for rung, open_ms in ((MEASURED_RUNG, 1000.0), (CENSORED_RUNG, None)):
        for arm, mult in (("base", 1.0), ("treatment", 1.1)):
            for rep in (0, 1):
                cid = f"{rung}.{arm}.rep{rep}"
                rows.append(
                    {
                        "row_type": "cell",
                        "cell_id": cid,
                        "session_id": "s1",
                        "completed": True,
                    }
                )
                # A control that is measured at BOTH rungs, so the refusal cannot pass by
                # rejecting everything.
                rows.append(
                    {
                        "row_type": "action",
                        "cell_id": cid,
                        "session_id": "s1",
                        "action": "keystroke",
                        "ran": True,
                        "timings": {"p50_ms": (50.0 if rung == MEASURED_RUNG else 90.0) * mult},
                        "counts": {},
                    }
                )
                if open_ms is None:
                    # Censored: the timing is ABSENT from `timings`, announced in `expect`.
                    rows.append(
                        {
                            "row_type": "action",
                            "cell_id": cid,
                            "session_id": "s1",
                            "action": "reasoning_toggle",
                            "ran": True,
                            "timings": {"close_ms": 300.0},
                            "counts": {},
                            "expect": {"open_censored": True, "close_censored": False},
                        }
                    )
                else:
                    rows.append(
                        {
                            "row_type": "action",
                            "cell_id": cid,
                            "session_id": "s1",
                            "action": "reasoning_toggle",
                            "ran": True,
                            "timings": {"open_ms": open_ms * mult, "close_ms": 300.0},
                            "counts": {},
                            "expect": {"open_censored": False, "close_censored": False},
                        }
                    )
    out = tmp_path / "sbench_mine"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return path


def test_the_pooling_path_consults_the_censoring_guard(tmp_path):
    """The guard is reachable from production code, not only from its own test."""
    path = _payload(tmp_path)
    found = floor_table.partial_censoring([path])
    assert "reasoning_toggle.open_ms" in found, (
        "the sweep pooled a metric that is censored at one rung and measured at another without "
        "ever asking the guard that exists to refuse it. The guard was dead code."
    )
    assert CENSORED_RUNG in found["reasoning_toggle.open_ms"]
    assert MEASURED_RUNG in found["reasoning_toggle.open_ms"]


def test_the_partial_metric_is_marked_unpoolable_and_denied_a_verdict(tmp_path):
    """It keeps its numbers, loses its claim to the ladder, and cannot clear a gate."""
    path = _payload(tmp_path)
    stats = floor_table.summarise([path])
    assert stats["reasoning_toggle.open_ms"]["poolable"] is False
    assert "censored" in stats["reasoning_toggle.open_ms"]["censoring"]
    # The control keeps its verdict: a refusal that swallowed everything would be no refusal.
    assert stats["keystroke.p50_ms"].get("poolable") is not False
    assert stats["reasoning_toggle.close_ms"].get("poolable") is not False


def test_the_rendered_table_says_so_where_the_number_is_printed(tmp_path, capsys):
    """The caveat has to travel with the row, because rows get copied out of tables into prose."""
    path = _payload(tmp_path)
    floor_table.render([path], "PAIRED PER-METRIC TABLE")
    printed = capsys.readouterr().out
    open_line = next(
        line for line in printed.splitlines() if line.strip().startswith("reasoning_toggle.open_ms")
    )
    assert "[*]" in open_line, (
        "the 100K-only figure was printed under a bare metric name on a 100K/500K ladder, which "
        "is the survivorship-biased row the guard was written to catch."
    )
    assert "NOT A LADDER NUMBER" in printed
    assert CENSORED_RUNG in printed
    # The fully measured control is NOT marked.
    keystroke_line = next(
        line for line in printed.splitlines() if line.strip().startswith("keystroke.p50_ms")
    )
    assert "[*]" not in keystroke_line


def test_a_metric_censored_at_every_rung_is_not_a_partial_case(tmp_path):
    """Nothing survives to be biased, so there is no ladder claim to refuse.

    The distinction matters: refusing here as well would mean the refusal fires on any censoring
    at all, which is a different and much noisier rule than the one that was asked for.
    """
    rows = [
        {"row_type": "run_meta", "tier": "standard", "session_id": "s1", "corpus_hash": "abc"},
    ]
    for rung in (MEASURED_RUNG, CENSORED_RUNG):
        cid = f"{rung}.base.rep0"
        rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s1", "completed": True})
        rows.append(
            {
                "row_type": "action",
                "cell_id": cid,
                "session_id": "s1",
                "action": "reasoning_toggle",
                "ran": True,
                "timings": {},
                "counts": {},
                "expect": {"open_censored": True},
            }
        )
    assert payload_rules.refuse_partial_censoring(rows, "reasoning_toggle.open_ms") is None


def _peer_censored_payload(tmp_path: Path) -> Path:
    """`open_ms` censors above 100K; `close_ms` was measured everywhere but is discarded with it.

    `reasoning_toggle`'s `ok` is one conjunction over four clauses, so a censored open fails the
    whole action and `_action_timings` then drops every timing it carries -- including a `close_ms`
    that succeeded on its own terms. `close_censored` stays False, so nothing marked the loss.
    """
    rows: list[dict] = [
        {
            "row_type": "run_meta",
            "tier": "standard",
            "session_id": "s1",
            "corpus_hash": "abc",
            "rungs": ["100K", "500K"],
        }
    ]
    for rung, censored in ((MEASURED_RUNG, False), (CENSORED_RUNG, True)):
        for arm, mult in (("base", 1.0), ("treatment", 1.1)):
            for rep in (0, 1):
                cid = f"{rung}.{arm}.rep{rep}"
                rows.append(
                    {"row_type": "cell", "cell_id": cid, "session_id": "s1", "completed": True}
                )
                if censored:
                    rows.append(
                        {
                            "row_type": "action",
                            "cell_id": cid,
                            "session_id": "s1",
                            "action": "reasoning_toggle",
                            "ran": True,
                            "expect_ok": False,
                            "timings": {"close_ms": 900.0 * mult},
                            "counts": {},
                            "expect": {"open_censored": True, "close_censored": False},
                        }
                    )
                else:
                    rows.append(
                        {
                            "row_type": "action",
                            "cell_id": cid,
                            "session_id": "s1",
                            "action": "reasoning_toggle",
                            "ran": True,
                            "expect_ok": True,
                            "timings": {"open_ms": 1000.0 * mult, "close_ms": 300.0 * mult},
                            "counts": {},
                            "expect": {"open_censored": False, "close_censored": False},
                        }
                    )
    out = tmp_path / "sbench_peer"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return path


def test_a_timing_discarded_with_its_action_counts_as_censored(tmp_path):
    """The surviving half of a failed action is unavailable too, and must be marked so.

    Unmarked, the close row was pooled from the 100K cells alone and printed +10.0% under a bare
    metric name on a 100K/500K ladder -- the same survivorship bias the open row is marked for,
    one level down and completely silent.
    """
    path = _peer_censored_payload(tmp_path)
    found = floor_table.partial_censoring([path])
    assert "reasoning_toggle.close_ms" in found, (
        "close_ms was thrown away at 500K with the action that failed, then pooled from 100K and "
        "printed as a ladder number. Nothing refused it because close_censored was False."
    )
    stats = floor_table.summarise([path])
    assert stats["reasoning_toggle.close_ms"]["poolable"] is False
    assert stats["reasoning_toggle.open_ms"]["poolable"] is False


def test_a_fully_measured_action_is_still_poolable(tmp_path):
    """The rule must not mark everything: an action that passed keeps its verdict."""
    path = _payload(tmp_path)
    stats = floor_table.summarise([path])
    assert stats["keystroke.p50_ms"].get("poolable") is not False
    assert stats["reasoning_toggle.close_ms"].get("poolable") is not False


def _within_rung_payload(tmp_path: Path) -> Path:
    """ONE rung, sitting on the settle budget, so some repetitions censor and others do not.

    This is the expected shape near the cutoff rather than a contrived one: censoring is decided
    per cell against a fixed budget, and `open_ms`'s own spread is 33 to 42%. The censored
    repetitions are the slow ones, which is exactly why they were censored.
    """
    rows: list[dict] = [
        {
            "row_type": "run_meta",
            "tier": "standard",
            "session_id": "s1",
            "corpus_hash": "abc",
            "rungs": ["500K"],
        }
    ]
    for rep, (base_ms, treat_ms, censored) in enumerate(
        (
            (6000.0, 6600.0, False),
            (7000.0, 9200.0, True),
            (7500.0, 9900.0, True),
            (6500.0, 11000.0, True),
        )
    ):
        for arm, value, cens in (("base", base_ms, False), ("treatment", treat_ms, censored)):
            cid = f"{CENSORED_RUNG}.{arm}.rep{rep}"
            rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s1", "completed": True})
            if cens:
                rows.append(
                    {
                        "row_type": "action",
                        "cell_id": cid,
                        "session_id": "s1",
                        "action": "reasoning_toggle",
                        "ran": True,
                        "expect_ok": False,
                        "timings": {},
                        "counts": {},
                        "expect": {"open_censored": True},
                    }
                )
            else:
                rows.append(
                    {
                        "row_type": "action",
                        "cell_id": cid,
                        "session_id": "s1",
                        "action": "reasoning_toggle",
                        "ran": True,
                        "expect_ok": True,
                        "timings": {"open_ms": value},
                        "counts": {},
                        "expect": {"open_censored": False},
                    }
                )
    out = tmp_path / "sbench_within"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return path


def test_censoring_within_a_single_rung_is_still_partial(tmp_path):
    """Comparing sets of RUNG NAMES cannot see this, and it is the commonest shape.

    Every rung appears in both the censored and the measured set, so a rung-keyed rule stays
    silent while `paired()` keeps only the repetitions where both arms answered. The survivor
    reported +10.0% on n=1 with a full SLOWER verdict; the true paired delta over all four
    repetitions is +35.7%. What survives is not a sample of the effect, it is a selection against
    it.
    """
    path = _within_rung_payload(tmp_path)
    stats = floor_table.summarise([path])["reasoning_toggle.open_ms"]
    assert stats["n"] == 1, "fixture no longer reproduces the survivorship case"
    assert stats["poolable"] is False, (
        "three of four treatment repetitions were censored at the only rung in the payload, and "
        "the one surviving pair was scored as the metric. A rule keyed on rung names cannot see "
        "censoring that happens WITHIN a rung."
    )
    assert "of" in stats["censoring"] and "cells" in stats["censoring"]


def test_a_metric_measured_on_every_completed_cell_is_untouched(tmp_path):
    """The cell-granular rule must not fire on a payload with nothing censored at all."""
    path = _within_rung_payload(tmp_path)
    rows = [r for r in floor_table.read_rows(path)]
    for r in rows:
        if r.get("row_type") == "action":
            r["expect_ok"] = True
            r["expect"] = {"open_censored": False}
            r.setdefault("timings", {})["open_ms"] = 6000.0
    assert payload_rules.refuse_partial_censoring(rows, "reasoning_toggle.open_ms") is None


def test_a_ladder_split_across_shards_is_judged_as_one_result(tmp_path):
    """Per-file evaluation lets a rung-disjoint shard set escape the refusal entirely.

    The shard holding the measured 100K cells sees no censoring; the shard holding the censored
    500K cells sees censoring at every rung it contains. Neither refuses on its own, and `load()`
    pools them anyway -- so the same rows are refused in one file and blessed in two.
    """

    def shard(name: str, rung: str, censored: bool) -> Path:
        rows: list[dict] = [
            {
                "row_type": "run_meta",
                "tier": "standard",
                "session_id": "s1",
                "corpus_hash": "abc",
                "rungs": ["100K", "500K"],
            }
        ]
        for arm, mult in (("base", 1.0), ("treatment", 1.05)):
            for rep in (0, 1):
                cid = f"{rung}.{arm}.rep{rep}"
                rows.append(
                    {"row_type": "cell", "cell_id": cid, "session_id": "s1", "completed": True}
                )
                rows.append(
                    {
                        "row_type": "action",
                        "cell_id": cid,
                        "session_id": "s1",
                        "action": "reasoning_toggle",
                        "ran": True,
                        "expect_ok": not censored,
                        "timings": {} if censored else {"open_ms": 1000.0 * mult + rep},
                        "counts": {},
                        "expect": {"open_censored": censored},
                    }
                )
        out = tmp_path / name
        out.mkdir()
        path = out / "payload.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
        return path

    measured = shard("sbench_mine.shard0", MEASURED_RUNG, False)
    censored = shard("sbench_mine.shard1", CENSORED_RUNG, True)
    assert floor_table.partial_censoring([measured]) == {}
    assert floor_table.partial_censoring([censored]) == {}
    both = floor_table.partial_censoring([measured, censored])
    assert "reasoning_toggle.open_ms" in both, (
        "a ladder split across shards escaped the refusal. Identical rows in one file are caught, "
        "so the verdict depended on which file they were written to."
    )
    assert (
        floor_table.summarise([measured, censored])["reasoning_toggle.open_ms"]["poolable"] is False
    )


def test_one_shards_failed_cell_does_not_censor_another_shards_good_one(tmp_path):
    """Sharding restarts the repetition counter, so the same `cell_id` names two different cells.

    `CONTRIBUTING-perf.md` prescribes 2 shards of 2 repetitions, and both shards then walk the
    same ladder writing the same deterministic ids. One ordinary failed cell -- `CellRunner.run`
    writes `completed: False` and leaves the action row it had already flushed, whose `expect_ok`
    is False -- is enough: merged on the bare id, that cell is censored in one shard and completed
    in the other, so the completed-cell filter that exists to discard it stops discarding it. A
    metric that nothing censored anywhere is then refused as partially censored, and a whole
    valid row loses its verdict over a cell that contributed nothing to any mean.
    """

    def shard(name: str, sess: str, failed: set[str]) -> Path:
        rows: list[dict] = [
            {
                "row_type": "run_meta",
                "tier": "standard",
                "session_id": sess,
                "corpus_hash": "abc",
                "rungs": ["100K", "500K"],
            }
        ]
        for rung in (MEASURED_RUNG, CENSORED_RUNG):
            for arm, mult in (("base", 1.0), ("treatment", 1.05)):
                for rep in (0, 1):
                    cid = f"{rung}.{arm}.rep{rep}"
                    bad = cid in failed
                    rows.append(
                        {
                            "row_type": "cell",
                            "cell_id": cid,
                            "session_id": sess,
                            "completed": not bad,
                        }
                    )
                    rows.append(
                        {
                            "row_type": "action",
                            "cell_id": cid,
                            "session_id": sess,
                            "action": "reasoning_toggle",
                            "ran": True,
                            "expect_ok": not bad,
                            "timings": {"open_ms": 1000.0 * mult + rep},
                            "counts": {},
                            "expect": {"open_censored": False},
                        }
                    )
        out = tmp_path / name
        out.mkdir()
        path = out / "payload.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
        return path

    dead = shard("sbench_mine.shard0", "sessA", {f"{CENSORED_RUNG}.treatment.rep1"})
    good = shard("sbench_mine.shard1", "sessB", set())
    assert floor_table.partial_censoring([dead]) == {}
    assert floor_table.partial_censoring([good]) == {}
    assert floor_table.partial_censoring([dead, good]) == {}, (
        "shard0's failed cell borrowed shard1's completion under the same repetition number, so "
        "a metric censored nowhere was refused as partially censored."
    )
    stats = floor_table.summarise([dead, good])["reasoning_toggle.open_ms"]
    assert stats.get("poolable") is not False

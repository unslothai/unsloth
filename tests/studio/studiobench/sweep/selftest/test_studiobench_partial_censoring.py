# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A metric censored at some rungs must not print as a ladder number; the refusal labels, not raises."""

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
                # Measured at both rungs, so the refusal cannot pass by rejecting everything.
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
                    # Censored: absent from `timings`, announced in `expect`.
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
    keystroke_line = next(
        line for line in printed.splitlines() if line.strip().startswith("keystroke.p50_ms")
    )
    assert "[*]" not in keystroke_line


def test_a_metric_censored_at_every_rung_is_not_a_partial_case(tmp_path):
    """Censored at every rung is not partial; refusing it too would fire far more often than asked."""
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
    """open_ms censors above 100K; close_ms is discarded with the failed action and not marked censored."""
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
    """A timing discarded with its failed action counts as censored, or it is pooled from the survivors."""
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
    """A single rung on the settle budget, where only the slow repetitions censor and the rest do not."""
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
    """Censoring within one rung is still partial; comparing rung-name sets cannot see it."""
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
    """Censoring is judged over all shards at once, or a ladder split across files escapes refusal."""

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
    """Shards repeat cell ids, so one shard's failed cell must not censor another shard's completed one."""

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

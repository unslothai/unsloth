# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A floor may only certify a result it was measured for, under the full comparability key."""

from __future__ import annotations

import pytest

from tests.studio.studiobench.runtime.types import Recorder
from tests.studio.studiobench.scoring import payload_rules
from tests.studio.studiobench.sweep import floor_table

CORPUS = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"


def _meta(**over) -> dict:
    """A `run_meta` carrying every key the Recorder requires, shaped like a real one."""
    row = {
        "row_type": "run_meta",
        "tier": "full",
        "tool_version": "0.1.0",
        "corpus_hash": CORPUS,
        "studio_ref": "attached:http://127.0.0.1:8708",
        "bundle": {"production": True},
        "platform": {"system": "Linux", "machine": "x86_64", "engine": "chromium"},
        "started_at": "2026-08-23T04:46:10",
        "cadence": "field",
        "rungs": ["100K", "500K"],
        "instrument_level": 0,
    }
    row.update(over)
    return row


def _cell(cid: str) -> dict:
    rung, arm, rep = cid.split(".")
    return {
        "row_type": "cell",
        "cell_id": cid,
        "cell": {"cell_id": cid, "rung": rung, "arm": arm, "rep": rep},
        "completed": True,
        "fidelity": "streamed_and_seeded",
    }


def _toggle(cid: str, close_ms: float, censored: bool) -> dict:
    """A censored row keeps its timing, but expect_ok is False so every timing on it is discarded."""
    timings = {"close_ms": close_ms}
    return {
        "row_type": "action",
        "cell_id": cid,
        "action": "reasoning_toggle",
        "ran": True,
        "expect_ok": not censored,
        "expect": {
            "settled": not censored,
            "open_censored": censored,
            "close_censored": False,
            "open_censored_reason": (
                "the open count reached 16 after 10698ms but the span census was still changing "
                "when the 8000ms budget ran out"
                if censored
                else None
            ),
        },
        "timings": timings,
        "counts": {},
        "slot_missed": False,
    }


def _write(
    tmp_path,
    name: str,
    rows: list[dict],
    meta: dict | None = None,
):
    """Through the REAL Recorder, so the schema and required keys are the producer's."""
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    rec = Recorder(out / "payload.jsonl", "sess-" + name)
    rec.emit(dict(meta or _meta()))
    for row in rows:
        rec.emit(dict(row))
    rec.close()
    return out / "payload.jsonl"


RESULT_100K = ((897.6, 631.4), (665.6, 515.5), (665.6, 498.8), (682.1, 532.3))
NULL_100K = ((681.4, 564.7), (565.7, 515.7), (532.0, 515.1), (498.8, 498.8))
#: Real 500K null readings; every cell is censored (open settle blew its budget).
NULL_500K = ((1429.9, 2149.6), (1571.0, 1965.3), (1444.1, 1528.6), (1439.2, 1521.4))


def _ladder(pairs_100k, pairs_500k, censor_500k: bool) -> list[dict]:
    rows: list[dict] = []
    for rung, pairs, censored in (
        ("r100K", pairs_100k, False),
        ("r500K", pairs_500k, censor_500k),
    ):
        for i, (base, treat) in enumerate(pairs):
            for arm, value in (("base", base), ("treatment", treat)):
                cid = f"{rung}.{arm}.rep{i}"
                rows.append(_cell(cid))
                rows.append(_toggle(cid, value, censored))
    return rows


def _null(tmp_path, censored: bool):
    return _write(
        tmp_path,
        "null" + ("_censored" if censored else "_whole"),
        _ladder(NULL_100K, NULL_500K, censored),
    )


def _result(tmp_path):
    """A result that measured the metric on every cell of the ladder."""
    return _write(tmp_path, "result", _ladder(RESULT_100K, RESULT_100K, False))


METRIC = "reasoning_toggle.close_ms"


def test_the_censored_null_is_marked_unpoolable_in_the_first_place(tmp_path):
    """The precondition, stated so a fix that stopped marking it could not pass the rest."""
    floors = floor_table.summarise([_null(tmp_path, censored = True)])
    assert floors[METRIC]["poolable"] is False
    assert "censored" in floors[METRIC]["censoring"]


def test_censoring_the_null_tightens_the_floor_it_leaves_behind(tmp_path):
    """Censoring the null drops its slow pairs, which narrows the floor it leaves behind."""
    whole = floor_table.summarise([_null(tmp_path, censored = False)])[METRIC]
    left = floor_table.summarise([_null(tmp_path, censored = True)])[METRIC]
    wide = max(abs(whole["delta_pct"]), whole["spread_pct"])
    tight = max(abs(left["delta_pct"]), left["spread_pct"])
    assert left["n"] < whole["n"], "the censored null kept every pair, so nothing was censored"
    assert tight < wide, (
        f"the censored null's floor ({tight:.1f}%) is not tighter than the whole null's "
        f"({wide:.1f}%), so this fixture no longer demonstrates the bias"
    )
    assert tight == pytest.approx(17.1, abs = 0.5)
    assert wide == pytest.approx(67.5, abs = 0.5)


def test_a_censored_floor_cannot_certify_a_measured_result(tmp_path, capsys):
    """The bug. A result measured in full, scored against a floor that is a survivor sample."""
    floors = floor_table.summarise([_null(tmp_path, censored = True)])
    survivors = floor_table.render([_result(tmp_path)], "t", floors = floors)
    printed = capsys.readouterr().out
    row = next(line for line in printed.splitlines() if line.strip().startswith(METRIC))
    assert (
        "faster" not in row and "SLOWER" not in row
    ), f"a result was certified against a floor the tool itself marks unpoolable: {row.strip()}"
    assert "no poolable floor" in row
    assert survivors == 0, "the row was counted as a metric that cleared all three gates"
    assert "the null control's own" in printed, (
        "the table denied the verdict without saying which side was censored, so a reader cannot "
        "tell a censored result from a censored floor"
    )


def test_the_result_is_not_labelled_as_the_censored_one(tmp_path, capsys):
    """A sound result must not be labelled [*] because its floor was censored; the figure is fine."""
    floors = floor_table.summarise([_null(tmp_path, censored = True)])
    floor_table.render([_result(tmp_path)], "t", floors = floors)
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.strip().startswith(METRIC)
    )
    assert "[f]" in row and "[*]" not in row


def test_the_same_result_scores_normally_against_a_whole_floor(tmp_path, capsys):
    """NOT A REFUSAL OF EVERYTHING. Identical result, a null control that censored nothing."""
    floors = floor_table.summarise([_null(tmp_path, censored = False)])
    # Under the whole-ladder 67.5% floor this -24.8% result is VOID.
    floor_table.render([_result(tmp_path)], "t", floors = floors)
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.strip().startswith(METRIC)
    )
    assert "no poolable floor" not in row and "[f]" not in row
    assert "VOID" in row


def test_a_whole_floor_still_certifies_a_real_effect(tmp_path, capsys):
    """The other half of the non-vacuity check: a large effect over a whole floor still passes."""
    floors = floor_table.summarise([_null(tmp_path, censored = False)])
    tenfold = tuple((b, b / 10.0) for b, _ in RESULT_100K)
    big = _write(tmp_path, "big", _ladder(tenfold, tenfold, False))
    stats = floor_table.summarise([big])
    f, verdict = floor_table.verdict_for(stats[METRIC], floors[METRIC])
    assert verdict in ("faster", "SLOWER"), f"a whole floor refused a real effect: {verdict}"


def _floor_and_result(
    tmp_path,
    floor_meta: dict,
    result_meta: dict,
    tag: str = "",
):
    """Null and result differ only in run_meta; `tag` gives each call its own dir as Recorder appends."""
    floor = _write(tmp_path, f"cmp_null{tag}", _ladder(NULL_100K, NULL_100K, False), floor_meta)
    result = _write(
        tmp_path, f"cmp_result{tag}", _ladder(RESULT_100K, RESULT_100K, False), result_meta
    )
    return floor_table.summarise([floor]), floor_meta, result


def test_an_older_harness_floor_cannot_score_a_newer_payload(tmp_path):
    """A floor from an older TOOL_VERSION measures a different quantity under the same metric name."""
    from tests.studio.studiobench.__main__ import TOOL_VERSION

    floors, floor_meta, result = _floor_and_result(
        tmp_path, _meta(tool_version = "0.1.0"), _meta(tool_version = TOOL_VERSION)
    )
    with pytest.raises(SystemExit) as exc:
        floor_table.render([result], "t", floors = floors, floor_meta = floor_meta)
    assert "tool_version" in str(exc.value)


def _meta_differing_in(field: str) -> dict:
    """A run_meta differing from _meta() in one field, wherever that field is nested."""
    probe = ["100K", "500K", "1M"] if field == "rungs" else "CHANGED"
    changed = _meta()
    changed[field] = probe
    if payload_rules.explain_incomparable(_meta(), changed):
        return changed
    changed = _meta()
    changed["platform"] = dict(changed["platform"])
    changed["platform"][field] = probe
    assert payload_rules.explain_incomparable(_meta(), changed), (
        f"could not construct a payload differing in {field}, so the case below would pass "
        f"without testing anything"
    )
    return changed


def test_every_comparability_field_stops_a_floor_from_being_applied(tmp_path):
    """Every comparability field is checked, so a field added to the key also guards applying a floor."""
    for field in payload_rules.comparability_fields(_meta()):
        floors, floor_meta, result = _floor_and_result(
            tmp_path, _meta(), _meta_differing_in(field), tag = field
        )
        with pytest.raises(SystemExit) as exc:
            floor_table.render([result], "t", floors = floors, floor_meta = floor_meta)
        said = str(exc.value)
        assert field in said or field.split("_")[0] in said, (
            f"{field} is part of the comparability key, and a floor differing in it was still "
            f"applied to a result: {said}"
        )


def test_an_identically_configured_run_still_scores(tmp_path, capsys):
    """The refusal must not swallow the ordinary case: same everything, still a verdict."""
    floors, floor_meta, result = _floor_and_result(tmp_path, _meta(), _meta())
    floor_table.render([result], "t", floors = floors, floor_meta = floor_meta)
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.strip().startswith(METRIC)
    )
    assert "faster" in row or "VOID" in row or "SLOWER" in row


def test_a_payload_that_disagrees_with_its_own_headers_is_not_scored(tmp_path):
    """`--resume` after a harness upgrade puts two headers in one file. Neither describes it."""
    floors, floor_meta, _ = _floor_and_result(tmp_path, _meta(), _meta())
    out = tmp_path / "two_headers"
    out.mkdir()
    rec = Recorder(out / "payload.jsonl", "sess-two")
    rec.emit(_meta(tool_version = "0.1.0"))
    rec.emit(_meta(tool_version = "0.2.0"))
    for row in _ladder(RESULT_100K, RESULT_100K, False):
        rec.emit(dict(row))
    rec.close()
    with pytest.raises(SystemExit) as exc:
        floor_table.render([out / "payload.jsonl"], "t", floors = floors, floor_meta = floor_meta)
    assert "disagrees with ITSELF" in str(exc.value)


def test_a_payload_with_no_run_meta_at_all_is_not_scored(tmp_path):
    """Absence is not a wildcard here either. Nothing in such a file says what it measured."""
    floors, floor_meta, _ = _floor_and_result(tmp_path, _meta(), _meta())
    out = tmp_path / "headerless"
    out.mkdir()
    rec = Recorder(out / "payload.jsonl", "sess-none")
    for row in _ladder(RESULT_100K, RESULT_100K, False):
        rec.emit(dict(row))
    rec.close()
    with pytest.raises(SystemExit) as exc:
        floor_table.render([out / "payload.jsonl"], "t", floors = floors, floor_meta = floor_meta)
    assert "no run_meta" in str(exc.value)


def test_the_cli_applies_both_guards(tmp_path, capsys):
    """Drives both guards through main, the way a person runs it, since an unreachable guard is no guard."""
    _null(tmp_path, censored = True)
    _result(tmp_path)
    rc = floor_table.main([str(tmp_path / "result"), "--floor", str(tmp_path / "null_censored")])
    printed = capsys.readouterr().out
    assert rc == 0
    row = next(line for line in printed.splitlines() if line.strip().startswith(METRIC))
    assert "no poolable floor" in row, "the CLI scored against the censored floor"
    assert "0 metric(s) cleared all three gates." in printed

    older = _write(
        tmp_path, "older_null", _ladder(NULL_100K, NULL_100K, False), _meta(tool_version = "0.1.0")
    )
    newer = _write(
        tmp_path,
        "newer_result",
        _ladder(RESULT_100K, RESULT_100K, False),
        _meta(tool_version = "0.2.0"),
    )
    with pytest.raises(SystemExit) as exc:
        floor_table.main([str(newer.parent), "--floor", str(older.parent)])
    assert "tool_version" in str(exc.value)

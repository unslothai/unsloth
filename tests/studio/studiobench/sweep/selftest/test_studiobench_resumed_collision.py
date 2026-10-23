# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repeated completed cell_id is a resume or concurrency: refuse overlapping or undated sessions."""

from __future__ import annotations

import pytest

from tests.studio.studiobench.runtime.types import Recorder
from tests.studio.studiobench.sweep import floor_table

METRIC = "message_menu.open_close_ms"


#: Wall clock per session; `_at` reproduces the 16-minute overlap of concurrent launchers.
_STARTED = {
    "s1": "2026-08-23T00:00:00",
    "t2": "2026-08-23T02:00:00",
    "t3": "2026-08-23T04:00:00",
    "s2": "2026-08-23T00:40:00",
    "s3": "2026-08-23T01:20:00",
    "a": "2026-08-23T00:00:00",
    "b": "2026-08-23T00:00:10",
    "outer": "2026-08-23T00:00:00",
    "inner": "2026-08-23T00:00:10",
}


def _meta(sid: str) -> dict:
    return {
        "row_type": "run_meta",
        "tier": "full",
        "tool_version": "0.2.0",
        "corpus_hash": "ac9d5d8e",
        "studio_ref": "attached",
        "bundle": {"production": True},
        "platform": {"system": "Linux", "machine": "x86_64", "engine": "chromium"},
        "started_at": _STARTED[sid],
        "session_id": sid,
        "rungs": ["100K"],
    }


def _cell(cid: str, sid: str, completed: bool) -> dict:
    rung, arm, rep = cid.split(".")
    return {
        "row_type": "cell",
        "cell_id": cid,
        "session_id": sid,
        "cell": {"cell_id": cid, "rung": rung, "arm": arm, "rep": rep},
        "completed": completed,
        "fidelity": "streamed_and_seeded",
    }


def _action(
    cid: str,
    sid: str,
    ms: float,
    ok: bool = True,
) -> dict:
    return {
        "row_type": "action",
        "cell_id": cid,
        "session_id": sid,
        "action": "message_menu",
        "ran": True,
        "expect_ok": ok,
        "expect": {},
        "timings": {"open_close_ms": ms},
        "counts": {},
        "slot_missed": False,
    }


def _arm(
    cid: str,
    sid: str,
    ms: float,
    completed: bool = True,
    ok: bool = True,
) -> list[dict]:
    """One attempt at one arm: its action row, then the cell row, written only if the process survives."""
    return [_action(cid, sid, ms, ok), _cell(cid, sid, completed)]


def _write(tmp_path, name: str, rows: list[dict]):
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    rec = Recorder(out / "payload.jsonl", "sess-" + name)
    for row in rows:
        rec.emit(dict(row))
    rec.close()
    return out / "payload.jsonl"


BASE, TREAT = "r100K.base.rep0", "r100K.treatment.rep0"


def _resumed(tmp_path, name = "resumed"):
    """The shape `skippable_cells` produces: base completed, treatment died, WHOLE pair re-run."""
    rows = [_meta("s1")]
    rows += _arm(BASE, "s1", 1000.0)
    rows += _arm(TREAT, "s1", 900.0, completed = False)
    rows += [_meta("s2")]
    rows += _arm(BASE, "s2", 1010.0)
    rows += _arm(TREAT, "s2", 505.0)
    return _write(tmp_path, name, rows)


def _interleaved(tmp_path, name = "concurrent"):
    """The file two overlapping sessions leave behind; one Recorder, since the run lock blocks two."""
    rows = [_meta("a"), _action(BASE, "a", 1000.0), _meta("b"), _action(BASE, "b", 1400.0)]
    rows += [_cell(BASE, "a", True), _action(TREAT, "a", 980.0), _cell(BASE, "b", True)]
    rows += [_cell(TREAT, "a", True), _action(TREAT, "b", 1390.0), _cell(TREAT, "b", True)]
    return _write(tmp_path, name, rows)


def test_a_resumed_ab_is_scored_rather_than_refused(tmp_path):
    pooled = floor_table.paired(floor_table.read_rows(_resumed(tmp_path)))
    assert pooled, (
        "a resumed A/B was refused outright. `skippable_cells` re-runs every pair of an "
        "unfinished comparison in one new session, so this is the ordinary shape of a resume, "
        "not a corner case"
    )
    assert pooled[METRIC] == [
        (1010.0, 505.0)
    ], "the pair came from the superseded attempt rather than the one the resume just wrote"


def test_the_superseded_attempt_is_not_a_second_repetition(tmp_path):
    """A finished ladder re-run pairs ONCE. Keyed per session without superseding it pairs twice,
    and one repetition enters the pool under two different sets of numbers."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("s2")] + _arm(BASE, "s2", 1000.0) + _arm(TREAT, "s2", 900.0)
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "rerun", rows)))
    assert pooled[METRIC] == [
        (1000.0, 900.0)
    ], "one repetition was pooled twice, so n counts the resume as an independent measurement"


def test_a_resume_that_was_itself_hard_killed_resurrects_nothing(tmp_path):
    """A resume that was itself hard-killed must not let the older completed reading stand in for it."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("s2")] + _arm(BASE, "s2", 1010.0) + [_action(TREAT, "s2", 111.0)]
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "killed", rows)))
    assert pooled == {}, (
        f"the killed resume was scored anyway: {dict(pooled)}. Either the dead attempt's timing "
        f"was quoted or the superseded completed attempt was resurrected in its place"
    )


def test_the_floor_side_is_reduced_the_same_way(tmp_path):
    """A resumed null control yields a floor from the resume's numbers, not the attempt it replaced."""
    floors = floor_table.summarise([_resumed(tmp_path, "null_resumed")])
    assert METRIC in floors
    assert floors[METRIC]["n"] == 1
    assert floors[METRIC]["base"] == 1010.0 and floors[METRIC]["treat"] == 505.0


def test_two_concurrent_launchers_are_still_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        floor_table.paired(floor_table.read_rows(_interleaved(tmp_path)))
    said = str(exc.value)
    assert "completed under more than one session" in said
    assert "INTERLEAVE" in said, (
        "the refusal did not say what actually distinguishes this payload from a resume, so a "
        "reader cannot tell which of the two they have"
    )
    assert "sessions a and b" in said, f"the interleaving sessions were not named: {said}"


def test_the_null_control_side_refuses_concurrency_too(tmp_path):
    with pytest.raises(SystemExit):
        floor_table.summarise([_interleaved(tmp_path, "null_concurrent")])


def test_a_second_run_nested_inside_a_stalled_one_is_caught(tmp_path):
    """A run nested inside a stalled run's gap still overlaps it, so spans are compared, not write order."""
    rows = [_meta("outer"), _action(BASE, "outer", 1000.0)]
    rows += [_meta("inner")] + _arm(BASE, "inner", 1400.0) + _arm(TREAT, "inner", 1390.0)
    rows += [_cell(BASE, "outer", True)] + _arm(TREAT, "outer", 980.0)
    with pytest.raises(SystemExit) as exc:
        floor_table.paired(floor_table.read_rows(_write(tmp_path, "nested", rows)))
    assert "INTERLEAVE" in str(exc.value)


def test_an_unrelated_interleaved_session_does_not_condemn_a_resume(tmp_path):
    """Only the colliding sessions are refused; an unrelated interleaved one must not condemn a resume."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0)
    rows += _arm(TREAT, "s1", 900.0, completed = False)
    rows += [_meta("s2")] + _arm(BASE, "s2", 1010.0) + _arm(TREAT, "s2", 505.0)
    rows.insert(6, _action("r1K.base.rep0", "s3", 12.0))
    rows.append(_cell("r1K.base.rep0", "s3", True))
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "unrelated", rows)))
    assert pooled[METRIC] == [(1010.0, 505.0)]


def test_a_superseded_dead_attempt_does_not_mark_a_metric_censored(tmp_path):
    """Superseded attempts' failed assertions must not mark a metric censored in the resumed run."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0, ok = False)
    rows += _arm(TREAT, "s1", 900.0, completed = False, ok = False)
    rows += [_meta("s2")] + _arm(BASE, "s2", 1010.0) + _arm(TREAT, "s2", 505.0)
    path = _write(tmp_path, "censor", rows)
    assert (
        floor_table.partial_censoring([path]) == {}
    ), "the superseded attempt's failed assertion was counted against the resume that replaced it"
    stats = floor_table.summarise([path])
    assert stats[METRIC].get("poolable") is not False


def test_an_ordinary_single_session_payload_is_untouched(tmp_path):
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "plain", rows)))
    assert pooled[METRIC] == [(1000.0, 500.0)]


def test_sequential_sessions_with_no_repeated_id_still_pair_within_themselves(tmp_path):
    """Sharding and extension append sessions legitimately and repeat no id. Unchanged."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("s2")]
    rows += _arm("r1K.base.rep0", "s2", 20.0) + _arm("r1K.treatment.rep0", "s2", 10.0)
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "shards", rows)))
    assert sorted(pooled[METRIC]) == [(20.0, 10.0), (1000.0, 500.0)]


def test_sessions_that_overlap_in_time_are_refused_even_when_the_file_looks_sequential(tmp_path):
    """Sessions whose clocks overlap are refused even when their rows sit sequentially in the file."""
    rows = [dict(_meta("a"), started_at = "2026-08-23T00:00:00")]
    rows += _arm(BASE, "a", 1000.0) + _arm(TREAT, "a", 980.0)
    rows += [dict(_meta("b"), started_at = "2026-08-23T00:00:01")]
    rows += _arm(BASE, "b", 1400.0) + _arm(TREAT, "b", 1390.0)
    rows = [dict(r, ts_ms = 60_000.0) if r.get("row_type") != "run_meta" else r for r in rows]
    with pytest.raises(SystemExit) as exc:
        floor_table.paired(floor_table.read_rows(_write(tmp_path, "clock_overlap", rows)))
    assert "RUNNING AT THE SAME TIME" in str(exc.value)


def test_a_payload_that_cannot_say_when_its_sessions_ran_is_still_refused(tmp_path):
    """No run_meta means no session dates, so such a payload is refused, not superseded by file order."""
    rows = _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += _arm(BASE, "s2", 1010.0) + _arm(TREAT, "s2", 505.0)
    rows = [{k: v for k, v in r.items() if k != "ts_ms"} for r in rows]
    with pytest.raises(SystemExit) as exc:
        floor_table.paired(floor_table.read_rows(_write(tmp_path, "undated", rows)))
    assert "does not say when its sessions ran" in str(exc.value)


def test_naming_a_session_still_reads_that_session(tmp_path):
    """Naming a session must still read that session; superseding must not empty the older one."""
    path = _resumed(tmp_path, "hatch")
    rows = floor_table.read_rows(path)
    first = floor_table.cell_metrics(rows, session = "s1")
    assert first[BASE][METRIC] == 1000.0, "naming the superseded session returned the resume"


def test_a_whole_payload_read_does_not_fall_back_to_the_superseded_cell_row(tmp_path):
    """A whole-payload read must not return a superseded cell row when a resume died inside that cell."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("s2")] + _arm(BASE, "s2", 1010.0) + [_action(TREAT, "s2", 111.0)]
    cells = floor_table.cell_metrics(floor_table.read_rows(_write(tmp_path, "whole", rows)))
    assert TREAT not in cells, (
        f"the killed cell came back with {cells.get(TREAT)}, which is the attempt the resume "
        f"replaced, presented as the current reading"
    )
    assert cells[BASE][METRIC] == 1010.0


def test_the_file_order_witness_looks_past_the_first_pair_of_sessions(tmp_path):
    """File-order overlap check must examine every adjacent pair, not only the first two sessions."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("t2"), _action(BASE, "t2", 1400.0)]
    rows += [_meta("t3"), _action(BASE, "t3", 1500.0), _cell(BASE, "t2", True)]
    rows += [_arm(TREAT, "t2", 1390.0)[0], _cell(BASE, "t3", True)]
    rows += [_cell(TREAT, "t2", True)] + _arm(TREAT, "t3", 1490.0)
    with pytest.raises(SystemExit) as exc:
        floor_table.paired(floor_table.read_rows(_write(tmp_path, "third_pair", rows)))
    assert "INTERLEAVE" in str(exc.value), (
        "the interleaving check stopped after the first pair of sessions, so a third run writing "
        "into a second one's gaps went through"
    )


def test_a_resume_killed_before_it_reached_the_second_arm_leaves_no_pair(tmp_path):
    """Superseding keys on the last attempt that wrote anything, not the last that finished."""
    rows = [_meta("s1")] + _arm(BASE, "s1", 1000.0) + _arm(TREAT, "s1", 500.0)
    rows += [_meta("s2"), _action(BASE, "s2", 4000.0)]
    pooled = floor_table.paired(floor_table.read_rows(_write(tmp_path, "killed_first_arm", rows)))
    assert pooled == {}, (
        f"the superseded session's pair was reported as the reading: {dict(pooled)}. A resume was "
        f"under way and died; what it replaced is not the current measurement"
    )

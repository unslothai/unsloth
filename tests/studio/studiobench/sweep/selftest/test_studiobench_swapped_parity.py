# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Two repetitions that swap the same two renderings between the arms are not a build difference.

THE FALSE ALARM THIS HOLDS. #11756 touched only the Images/Video/Audio model pickers and failed
the UI parity gate (run 35949769378) on four stable pairs:

    delete_message  r100K rep0: thread scaffolding outside any message (11265->13287c)
    delete_message  r100K rep1: thread scaffolding outside any message (13287->11265c)
    thread_reopen   r100K rep0: thread scaffolding outside any message (11296->13308c)
    thread_reopen   r100K rep1: thread scaffolding outside any message (13308->11296c)

In the recorded payload every field `rendering_of` reads is the same in rep0's base capture and
rep1's treatment capture, and the other way round. The extra 2012 and 2022 characters of scaffold
follow the attachment chip `image_upload` leaves in the composer (inside the thread root) when it
reaches its 800ms slot; on this run base missed it, head ran it, head missed it, base ran it. The
null control shows the same build rendering both states. `testdata/pr11756_*` are CI's payloads
trimmed to the fields the verdict reads (window rows dropped, action rows cut to the keys
`pr11727_*` keeps); the verdict text is identical on the trimmed and the full files.

Every "clears" test is paired with one that puts a genuine, consistent difference into the same
recorded payload and requires it to still fail.
"""

from __future__ import annotations

import copy
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep import parity_shots as S  # noqa: E402
from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402

TESTDATA = Path(__file__).resolve().parent / "testdata"
SWAPPED = ("delete_message", "thread_reopen")
CELLS = ("r100K.base.rep0", "r100K.treatment.rep0", "r100K.base.rep1", "r100K.treatment.rep1")


def _load(name: str) -> list[dict]:
    with gzip.open(TESTDATA / name, "rt", encoding = "utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _recorded() -> tuple[list[dict], list[dict]]:
    return _load("pr11756_result.payload.jsonl.gz"), _load("pr11756_null.payload.jsonl.gz")


def _write(rows: list[dict], path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    with open(path / "payload.jsonl", "w", encoding = "utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _row(rows: list[dict], action: str, cell: str) -> dict:
    return next(
        r
        for r in rows
        if r.get("row_type") == "action" and r.get("action") == action and r.get("cell_id") == cell
    )


def _run(
    tmp_path,
    capsys,
    result,
    null,
    min_reps = 2,
) -> tuple[int, str]:
    """The workflow's verdict step, flag for flag: (exit code, what it printed)."""
    rdir = _write(result, tmp_path / "parity-result")
    ndir = _write(null, tmp_path / "parity-null-control")
    rc = U.main(
        [
            "--min-reps",
            str(min_reps),
            "--min-compared",
            "16",
            "--null",
            str(ndir),
            str(rdir),
        ]
    )
    return rc, capsys.readouterr().out


def _section(out: str, head: str) -> str:
    """The indented lines under `head`, or `""` when the heading is absent."""
    if head not in out:
        return ""
    lines = []
    for line in out.split(head, 1)[1].splitlines()[1:]:
        if line.startswith("  ") and not line.startswith("    "):
            continue  # the heading's own wrapped second line
        if not line.startswith("    "):
            break
        lines.append(line)
    return "\n".join(lines)


def _stable(out: str) -> str:
    return _section(out, "UI PARITY DIFFERENCES ON STABLE ACTIONS")


def _swapped(out: str) -> str:
    return _section(out, "the repetitions SWAPPED the same two renderings")


def _make_consistent(rows: list[dict], action: str) -> None:
    """rep1 renders exactly what rep0 did on each arm: base R1 and head R2 in both repetitions."""
    for arm in ("base", "treatment"):
        _row(rows, action, f"r100K.{arm}.rep1")["parity"] = copy.deepcopy(
            _row(rows, action, f"r100K.{arm}.rep0")["parity"]
        )


# ── the evidence, as recorded ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("action,extra", [("delete_message", 2022), ("thread_reopen", 2012)])
def test_the_recorded_repetitions_are_one_pair_of_renderings_swapped(action, extra):
    """The premise, checked on CI's bytes: rep0's (base, head) is rep1's (head, base), entire."""
    result, _null = _recorded()
    cap = {cell: _row(result, action, cell)["parity"] for cell in CELLS}
    b0, t0 = U.rendering_of(cap["r100K.base.rep0"]), U.rendering_of(cap["r100K.treatment.rep0"])
    b1, t1 = U.rendering_of(cap["r100K.base.rep1"]), U.rendering_of(cap["r100K.treatment.rep1"])
    assert None not in (b0, t0, b1, t1)
    assert b0 != t0
    assert (b0, t0) == (t1, b1)
    # About 2000 characters of scaffold and nothing else: every message row agrees across all four.
    sizes = sorted({c["chars_scaffold"] for c in cap.values()})
    assert sizes[1] - sizes[0] == extra
    assert len({json.dumps(c["messages"], sort_keys = True) for c in cap.values()}) == 1


def test_the_extra_scaffold_follows_whether_image_upload_ran_in_that_cell():
    """The mechanism, on both payloads: the larger scaffold is exactly the cells where it ran."""
    for rows in _recorded():
        for cell in CELLS:
            ran = bool(_row(rows, "image_upload", cell).get("ran"))
            for action in SWAPPED:
                chars = _row(rows, action, cell)["parity"]["chars_scaffold"]
                assert (chars > 12000) == ran, (cell, action, chars, ran)


def test_the_null_control_shows_one_build_rendering_both_states():
    _result, null = _recorded()
    for action in SWAPPED:
        sizes = {_row(null, action, cell)["parity"]["chars_scaffold"] for cell in CELLS}
        assert len(sizes) == 2, (action, sizes)


# ── (a) the #11756 shape is not firm ─────────────────────────────────────────────────────────


def test_swapped_repetitions_do_not_fail_the_verdict(tmp_path, capsys):
    result, null = _recorded()
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 0, out
    assert _stable(out) == ""
    assert "stable actions differing:   0" in out
    # Reported, not dropped: all four readings are printed under UNCORROBORATED.
    section = _swapped(out)
    for action in SWAPPED:
        assert section.count(action) == 2, section
    assert "swapped between arms:       4" in out


def test_the_null_audit_is_not_asked_to_excuse_a_swap(tmp_path, capsys):
    """Same predicate as the verdict: a pair it never counts needs no excuse from the null."""
    result, _null = _recorded()
    rdir = _write(result, tmp_path / "parity-result")
    assert U.actions_needing_an_excuse(U.shards_of(str(rdir)), 2) == set()


def test_the_evidence_step_does_not_illustrate_a_swap(tmp_path):
    result, null = _recorded()
    rdir = _write(result, tmp_path / "parity-result")
    ndir = _write(null, tmp_path / "parity-null-control")
    assert S.differing_actions(U.shards_of(str(rdir)), U.shards_of(str(ndir)), min_reps = 2) == []


# ── (b) a consistent difference in both repetitions stays firm ───────────────────────────────


@pytest.mark.parametrize("action", SWAPPED)
def test_the_same_difference_in_both_repetitions_still_fails(tmp_path, capsys, action):
    """Base R1 and head R2 twice: the shape a head that always leaves the chip would record."""
    result, null = _recorded()
    _make_consistent(result, action)
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert _stable(out).count(action) == 2, _stable(out)
    assert action not in _swapped(out)


def test_a_real_change_under_the_race_still_fails(tmp_path, capsys):
    """The recorded swap, plus 17 characters head adds to the scaffold in both repetitions.

    The renderings no longer line up across the repetitions, so the race cannot excuse them.
    """
    result, null = _recorded()
    for action in SWAPPED:
        for cell in ("r100K.treatment.rep0", "r100K.treatment.rep1"):
            cap = _row(result, action, cell)["parity"]
            cap["digest_scaffold"] += "x"
            cap["chars_scaffold"] += 17
            cap["digest"] += "x"
            cap["chars"] += 17
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    for action in SWAPPED:
        assert _stable(out).count(action) == 2, _stable(out)
    assert _swapped(out) == ""


def test_a_swap_that_differs_anywhere_else_is_not_a_swap(tmp_path, capsys):
    """Only the style probe differs from an exact swap in one repetition: still counted."""
    result, null = _recorded()
    for action in SWAPPED:
        _row(result, action, "r100K.treatment.rep1")["parity"]["styles"]["digest"] = "0badc0de"
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    assert _swapped(out) == ""


# ── (c) a swap in one action does not excuse a consistent difference in another ─────────────


def test_a_swap_on_one_action_does_not_excuse_another(tmp_path, capsys):
    result, null = _recorded()
    _make_consistent(result, "thread_reopen")
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    stable, swapped = _stable(out), _swapped(out)
    assert stable.count("thread_reopen") == 2 and "delete_message" not in stable, stable
    assert swapped.count("delete_message") == 2 and "thread_reopen" not in swapped, swapped
    rdir = tmp_path / "parity-result"
    assert U.actions_needing_an_excuse(U.shards_of(str(rdir)), 2) == {("r100K", "thread_reopen")}


def test_a_swap_at_one_rung_does_not_excuse_the_same_action_at_another():
    """Grouped by (action, rung), like `corroborated`: the partner has to be at the same rung."""
    r1, r2 = "R1", "R2"
    results = [
        ("delete_message", "s", "r1K rep0", {"verdict": P.DIFFER, "outcomes": (r1, r2)}),
        ("delete_message", "s", "r100K rep1", {"verdict": P.DIFFER, "outcomes": (r2, r1)}),
    ]
    assert U.swapped_between_arms(results, 2) == frozenset()


def test_one_repetition_cannot_be_its_own_partner():
    """The same cell label in two shards is one repetition; a swap needs two."""
    results = [
        ("delete_message", "a", "r100K rep0", {"verdict": P.DIFFER, "outcomes": ("R1", "R2")}),
        ("delete_message", "b", "r100K rep0", {"verdict": P.DIFFER, "outcomes": ("R2", "R1")}),
    ]
    assert U.swapped_between_arms(results, 2) == frozenset()


def test_one_reversal_excuses_only_one_repetition_of_the_other_direction():
    """Three repetitions of (R1, R2) and one of (R2, R1): one pair is a swap, and the two
    same-direction repetitions left over are still a difference seen twice."""
    results = [
        ("delete_message", "s", f"r100K rep{n}", {"verdict": P.DIFFER, "outcomes": outcomes})
        for n, outcomes in enumerate([("R1", "R2"), ("R1", "R2"), ("R1", "R2"), ("R2", "R1")])
    ]
    swapped = U.swapped_between_arms(results, 2)
    assert len(swapped) == 2 and 3 in swapped
    left = [results[i] for i in range(len(results)) if i not in swapped]
    firm, weak = U.corroborated([(a, s, c, r) for a, s, c, r in left], 2)
    assert len({e[2] for e in firm}) == 2 and not weak


def test_two_reversals_against_two_forwards_are_all_swaps():
    results = [
        ("delete_message", "s", f"r100K rep{n}", {"verdict": P.DIFFER, "outcomes": outcomes})
        for n, outcomes in enumerate([("R1", "R2"), ("R2", "R1"), ("R1", "R2"), ("R2", "R1")])
    ]
    assert U.swapped_between_arms(results, 2) == frozenset(range(4))


# ── (d) payloads without what the swap needs are scored as before ────────────────────────────


def test_results_without_renderings_are_never_excused():
    """A caller that builds results without `outcomes`, or a capture with no digest."""
    bare = [
        ("delete_message", "s", "r100K rep0", {"verdict": P.DIFFER}),
        ("delete_message", "s", "r100K rep1", {"verdict": P.DIFFER}),
    ]
    assert U.swapped_between_arms(bare, 2) == frozenset()
    half = [
        ("delete_message", "s", "r100K rep0", {"verdict": P.DIFFER, "outcomes": (None, "R2")}),
        ("delete_message", "s", "r100K rep1", {"verdict": P.DIFFER, "outcomes": ("R2", None)}),
    ]
    assert U.swapped_between_arms(half, 2) == frozenset()
    assert U.rendering_of(None) is None
    assert U.rendering_of({"parity_attempted": True, "error": "probe failed"}) is None


def test_a_payload_whose_captures_carry_no_digest_fails_as_it_always_did(tmp_path, capsys):
    """The recorded run, with the aggregate `digest` stripped from every capture of the pair."""
    result, null = _recorded()
    for action in SWAPPED:
        for cell in CELLS:
            _row(result, action, cell)["parity"].pop("digest")
    rc, out = _run(tmp_path, capsys, result, null)
    assert rc == 1, out
    for action in SWAPPED:
        assert _stable(out).count(action) == 2, _stable(out)
    assert _swapped(out) == ""


def test_min_reps_one_still_counts_every_difference(tmp_path, capsys):
    """`--min-reps 1` asks for every difference to count, so the swap is not consulted."""
    result, null = _recorded()
    rc, out = _run(tmp_path, capsys, result, null, min_reps = 1)
    assert rc == 1, out
    assert _swapped(out) == ""
    for action in SWAPPED:
        assert _stable(out).count(action) == 2, _stable(out)

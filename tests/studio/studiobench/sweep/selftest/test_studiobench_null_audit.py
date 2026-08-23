# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Auditing a null control: was it capable of an opinion, not did it happen to find one.

The gate this file holds exists because the obvious version of it is wrong in the worst
direction. A CI job that scores a result against `--null` has to know the null control actually
derived something, since `unstable_set` silently falls back to the DECLARED list when it did not
and prints "UNSTABLE SET DERIVED" either way. The tempting check is "the measured set is
non-empty". That check fails on the best null control obtainable -- every action decided, none of
them differing -- because `derive_unstable` only records an action when something DIFFERED. It
therefore breaks precisely when the machine is quietest and the measurement is at its best.

So every test here is paired: one where a null control that should pass does, and one where a
null control that should fail does. The pairing is the point. An audit that only ever passes is
worse than no audit, because it gets cited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402

# parents[4] is `tests/`, which is what the sibling selftests put on sys.path; the repo root is
# one further up and is where .github lives.
REPO_ROOT = Path(__file__).resolve().parents[5]
WORKFLOW = REPO_ROOT / ".github/workflows/studiobench-ui-parity.yml"


# ── building a null-control payload ──────────────────────────────────


def action_row(cid: str, action: str, digest: str | None) -> dict:
    """One action row. `digest=None` records a capture that FAILED, which is blind, not stable."""
    if digest is None:
        parity = {"parity_attempted": False, "reason": "threadRoot is not a function"}
    else:
        parity = {
            "parity_attempted": True,
            "root_kind": "thread",
            "digest": digest,
            "chars": 100,
            "messages": [{"i": 0, "role": "assistant", "digest": digest, "chars": 10}],
            "overlays": [],
            "style": {"style_attempted": True, "capped": False, "nodes": []},
        }
    return {
        "row_type": "action",
        "cell_id": cid,
        "action": action,
        "ran": True,
        "timings": {"open_ms": 5.0},
        "parity": parity,
    }


def null_run(tmp_path: Path, name: str, cells: list[tuple[str, str, str, str | None, str | None]]):
    """`cells` is (rung, rep, action, base digest, treatment digest)."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rung, rep, action, base, treat in cells:
        rows.append(action_row(f"{rung}.base.{rep}", action, base))
        rows.append(action_row(f"{rung}.treatment.{rep}", action, treat))
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def two_reps(action: str, base: str, treat: str) -> list[tuple]:
    return [("r100K", "rep0", action, base, treat), ("r100K", "rep1", action, base, treat)]


# ── the pair that matters: a quiet null control is a GOOD one ─────────


def test_a_null_control_that_decided_everything_and_found_nothing_passes(tmp_path):
    # Every action reached two observations and not one of them differed. This is the best null
    # control obtainable and the measured unstable set is empty BECAUSE there was nothing to
    # measure. Requiring a measured entry here would fail the job on its best day.
    null = null_run(
        tmp_path,
        "quiet",
        two_reps("settings", "SAME", "SAME") + two_reps("thread_reopen", "SAME", "SAME"),
    )
    rc, report = U.audit_null([null])
    assert rc == 0
    assert report["differed"] == []
    assert len(report["decided"]) == 2
    assert report["undecided"] == []

    # And the thing the naive gate keyed on really is absent, so this test would not have caught
    # the bug by accident: the derived set emits no `action@rung` entry at all.
    unstable, _derived, _checks = U.unstable_set([null])
    assert not [e for e in unstable if isinstance(e, tuple)]


def test_a_null_control_with_one_observation_each_is_undecided(tmp_path):
    # The other direction, so the audit cannot pass by never failing. --reps 1 gives one
    # observation per (rung, action); `derive_unstable` needs two, so nothing is decided, the
    # measured set is empty for a completely different reason, and the result would be scored
    # against the declared list.
    null = null_run(
        tmp_path,
        "thin",
        [
            ("r100K", "rep0", "settings", "SAME", "SAME"),
            ("r100K", "rep0", "thread_reopen", "SAME", "SAME"),
        ],
    )
    rc, report = U.audit_null([null])
    assert rc == 1
    assert report["decided"] == []
    assert sorted(a for _r, a in report["undecided"]) == ["settings", "thread_reopen"]


def test_the_two_empty_measured_sets_are_told_apart(tmp_path):
    # Both of the above derive an EMPTY measured set. The whole point of the audit is that one is
    # the best possible reading and the other is a broken one, so assert they are distinguished
    # rather than merely that each has the right code.
    quiet = null_run(tmp_path, "q2", two_reps("settings", "SAME", "SAME"))
    thin = null_run(tmp_path, "t2", [("r100K", "rep0", "settings", "SAME", "SAME")])
    for path in (quiet, thin):
        unstable, _d, _c = U.unstable_set([path])
        assert not [e for e in unstable if isinstance(e, tuple)], "both measure nothing"
    assert U.audit_null([quiet])[0] == 0
    assert U.audit_null([thin])[0] == 1


# ── a null control that DID find instability still passes ────────────


def test_a_noisy_null_control_is_decided_and_reports_what_differed(tmp_path):
    null = null_run(
        tmp_path,
        "noisy",
        [
            ("r100K", "rep0", "settings", "A", "B"),
            ("r100K", "rep1", "settings", "A", "C"),
            ("r100K", "rep0", "thread_reopen", "SAME", "SAME"),
            ("r100K", "rep1", "thread_reopen", "SAME", "SAME"),
        ],
    )
    rc, report = U.audit_null([null])
    assert rc == 0
    assert report["differed"] == [("r100K", "settings")]
    assert len(report["decided"]) == 2


# ── blind is not decided, and an excuse is a named hole ──────────────


def test_an_action_that_never_captured_is_undecided_not_stable(tmp_path):
    # A failed capture is blind. `derive_unstable` refuses to count it as an observation, because
    # an action derived as stable from pairs that never rendered would be trusted forever on the
    # strength of nothing. So it must read as UNDECIDED, never as a quiet pass.
    null = null_run(tmp_path, "blind", two_reps("image_upload", None, None))
    rc, report = U.audit_null([null])
    assert rc == 1
    assert report["undecided"] == [("r100K", "image_upload")]


def test_an_excused_action_is_allowed_but_still_needs_a_decided_one(tmp_path):
    # image_upload is permanently blind on this fixture, so it is excusable. What is NOT excusable
    # is a run in which everything is excused: that measured nothing at all while naming a reason
    # for each blank, and passing it would let the excuse list grow until the audit is vacuous.
    only_excused = null_run(tmp_path, "allblind", two_reps("image_upload", None, None))
    rc, report = U.audit_null([only_excused], frozenset({"image_upload"}))
    assert rc == 1
    assert report["decided"] == []
    assert report["excused"] == [("r100K", "image_upload")]

    with_a_real_one = null_run(
        tmp_path,
        "mixed",
        two_reps("image_upload", None, None) + two_reps("settings", "SAME", "SAME"),
    )
    rc, report = U.audit_null([with_a_real_one], frozenset({"image_upload"}))
    assert rc == 0
    assert report["decided"] == [("r100K", "settings")]
    assert report["excused"] == [("r100K", "image_upload")]


def test_a_payload_with_no_parity_data_is_not_a_decided_null(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    (out / "payload.jsonl").write_text(
        json.dumps({"row_type": "run_meta", "tier": "fast"}) + "\n", encoding = "utf-8"
    )
    assert U.audit_null([out / "payload.jsonl"])[0] == 2


# ── the CLI surface the workflow actually calls ──────────────────────


def test_the_cli_audits_a_quiet_null_as_a_pass(tmp_path, capsys):
    null_run(tmp_path, "cli", two_reps("settings", "SAME", "SAME"))
    rc = U.main(["--audit-null", str(tmp_path / "cli")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NULL CONTROL AUDIT: DECIDED" in out
    assert "best null control obtainable" in out


def test_the_cli_audits_a_single_repetition_as_a_failure(tmp_path, capsys):
    null_run(tmp_path, "cli1", [("r100K", "rep0", "settings", "SAME", "SAME")])
    rc = U.main(["--audit-null", str(tmp_path / "cli1")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NULL CONTROL AUDIT: UNDECIDED" in out
    # The diagnosis has to name --reps, because that is the cause every time.
    assert "--reps" in out


# ── the CI job must use the audit, not grep the tool's prose ─────────


def test_the_parity_workflow_audits_the_null_with_the_tool(tmp_path):
    # Twice now a guard written as a regex over this tool's own printed prose has been wrong: once
    # because a real payload spells its rung `r100K` where a fixture spells it `100K`, and once
    # because the literal `action@rung` appears in the explanatory text whether or not anything
    # was measured. The check belongs in the tool, where it is reachable by the tests above.
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert "--audit-null" in text
    assert "--allow-undecided image_upload" in text
    # The prose may still EXPLAIN the two regexes that were wrong; what it may not do is run one.
    assert "grep -Eo" not in text, "the guard must not key on the tool's printed prose again"


# ── a difference that did not repeat is not a change to the build ────


def test_a_difference_in_one_repetition_of_two_is_not_counted(tmp_path):
    # The measured false alarm, in miniature. A build renders the same way every time, so a stable
    # action that differs in rep0 and matches in rep1 is telling you about the run.
    mine = null_run(
        tmp_path,
        "one_rep",
        [
            ("r100K", "rep0", "settings", "A", "B"),
            ("r100K", "rep1", "settings", "A", "A"),
        ],
    )
    assert U.report([mine], "t", frozenset(), min_reps = 1) == 1
    assert U.report([mine], "t", frozenset(), min_reps = 2) == 0


def test_a_difference_in_every_repetition_still_fails(tmp_path):
    # The other direction, and the one that matters more: an injected element differs on every
    # pass, so raising the bar must not make the gate unable to fail. Measured on the real probe
    # -- one <span> added inside the thread root -- which stays red at --min-reps 2.
    mine = null_run(
        tmp_path,
        "both_reps",
        [
            ("r100K", "rep0", "settings", "A", "B"),
            ("r100K", "rep1", "settings", "A", "B"),
        ],
    )
    assert U.report([mine], "t", frozenset(), min_reps = 2) == 1


def test_the_headline_count_is_the_one_the_exit_code_uses(tmp_path, capsys):
    # Printing "stable actions differing: 1" above a verdict of 0 is how a reader decides the
    # tool is lying to them.
    mine = null_run(
        tmp_path,
        "hdr",
        [
            ("r100K", "rep0", "settings", "A", "B"),
            ("r100K", "rep1", "settings", "A", "A"),
        ],
    )
    rc = U.report([mine], "t", frozenset(), min_reps = 2)
    out = capsys.readouterr().out
    assert rc == 0
    assert "stable actions differing:   0" in out
    assert "uncorroborated:             1" in out


def test_one_repetition_seen_twice_is_not_two_observations(tmp_path):
    # Two shards can carry the same (rung, rep); corroboration counts DISTINCT repetitions, or a
    # single flake recorded in two places would corroborate itself.
    a = null_run(tmp_path, "sh1", [("r100K", "rep0", "settings", "A", "B")])
    b = null_run(tmp_path, "sh2", [("r100K", "rep0", "settings", "A", "B")])
    assert U.report([a, b], "t", frozenset(), min_reps = 2) == 0


# ── a cell that never completed: dropped on the null, kept on the result ──


def cell_rows(rows: list[dict], cid: str, completed: bool) -> None:
    if completed:
        rows.append({"row_type": "cell", "cell_id": cid, "completed": True})


def run_with_completion(tmp_path: Path, name: str, specs: list[tuple]) -> Path:
    """`specs` is (rep, base digest, treat digest, cell completed)."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep, base, treat, done in specs:
        for arm, digest in (("base", base), ("treatment", treat)):
            cid = f"r100K.{arm}.{rep}"
            rows.append(action_row(cid, "settings", digest))
            cell_rows(rows, cid, done)
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_null_control_ignores_a_cell_that_never_completed(tmp_path):
    # The half-written film. Its rows look complete and pair like real ones, and counting them
    # narrows the excuse set, which is how a truncated null manufactures false alarms downstream.
    null = run_with_completion(
        tmp_path,
        "truncated",
        [("rep0", "A", "B", True), ("rep1", "A", "B", False)],
    )
    # Only the completed repetition is an observation, so one observation, so undecided.
    rc, report = U.audit_null([null])
    assert rc == 1
    assert report["decided"] == []


def test_the_same_two_repetitions_decide_it_once_both_cells_completed(tmp_path):
    null = run_with_completion(
        tmp_path,
        "whole",
        [("rep0", "A", "B", True), ("rep1", "A", "B", True)],
    )
    rc, report = U.audit_null([null])
    assert rc == 0
    assert report["differed"] == [("r100K", "settings")]


def test_a_result_still_reports_a_difference_from_a_cell_that_died(tmp_path):
    # The OTHER sign, and the one that matters more. A cell that died is the latest attempt at
    # itself and the difference it saw is real; dropping it would silence a regression, which is
    # the worse direction. `test_an_attempt_that_was_never_re_run_still_carries_its_parity_
    # verdict` holds the same line from the other file.
    mine = run_with_completion(
        tmp_path,
        "died",
        [("rep0", "A", "B", False), ("rep1", "A", "B", False)],
    )
    assert U.compare_all([mine])[0] != []
    assert U.report([mine], "t", frozenset(), min_reps = 2) == 1


def test_a_payload_with_no_cell_rows_says_it_could_not_check(tmp_path, capsys):
    # Falling back to the old behaviour is right; falling back silently is how a guard stops
    # guarding without anybody noticing.
    old = null_run(tmp_path, "nocells", two_reps("settings", "A", "B"))
    U.report([old], "t", frozenset())
    assert "NOT GUARDED" in capsys.readouterr().out


# ── two corpora must not pool into one decision ──────────────────────


def corpus_run(tmp_path: Path, name: str, corpus: str, rep: str) -> Path:
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast", "corpus_hash": corpus}]
    for arm in ("base", "treatment"):
        cid = f"r100K.{arm}.{rep}"
        rows.append(action_row(cid, "settings", "A" if arm == "base" else "B"))
        rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_two_corpora_are_refused_before_their_observations_pool(tmp_path):
    # Each film is one repetition and can decide nothing alone. Pooled they satisfy
    # min_observations and report DECIDED from two different threads, and the result is then
    # scored against a set never measured on the corpus it is applied to.
    a = corpus_run(tmp_path, "c1", "corpusAAAA", "rep0")
    b = corpus_run(tmp_path, "c2", "corpusBBBB", "rep1")
    with pytest.raises(SystemExit) as got:
        U.one_corpus([a, b], "null control")
    assert "more than one corpus" in str(got.value)

    # And the refusal is load-bearing: without it the two DO pool into a decision.
    rc, report = U.audit_null([a, b])
    assert rc == 0 and report["differed"] == [("r100K", "settings")]


def test_one_corpus_across_several_shards_is_fine(tmp_path):
    a = corpus_run(tmp_path, "s1", "corpusAAAA", "rep0")
    b = corpus_run(tmp_path, "s2", "corpusAAAA", "rep1")
    assert U.one_corpus([a, b], "null control") == {"corpusAAAA"}


def test_a_payload_predating_corpus_hashes_is_not_refused(tmp_path):
    old = null_run(tmp_path, "nohash", two_reps("settings", "A", "B"))
    assert U.one_corpus([old], "null control") == set()


# ── a missed slot costs coverage, not correctness ────────────────────


def not_run_row(cid: str, action: str) -> dict:
    row = action_row(cid, action, "SAME")
    row["ran"] = False
    row["reason"] = "the slot opened at 33000ms and was reached at 33872ms"
    return row


def test_a_missed_slot_on_both_arms_cannot_turn_a_clean_verdict_red(tmp_path):
    # The measured claim, in miniature. Over 18 mutations of a real payload at 2 to 24 missed
    # slots, the verdict never gained a red; NOT_EXERCISED is filed under coverage, never under
    # the stable differences the exit code is taken from.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            rows.append(action_row(cid, "settings", "SAME"))
            rows.append(not_run_row(cid, "copy_markdown"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "missed"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    assert U.report([path], "t", frozenset(), min_reps = 2) == 0


def test_a_run_that_compared_almost_nothing_does_not_pass(tmp_path, capsys):
    # The other half, and the reason the slot gate could not simply be deleted. Two blank pages
    # have identical digests, so a film that ran nothing is the easiest possible pass.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            rows.append(action_row(cid, "settings", "SAME"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "thin"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 0) == 0
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 3
    assert "TOO LITTLE COMPARED" in capsys.readouterr().out


# ── the null need only decide what the result compared ───────────────


def test_the_audit_ignores_an_action_the_result_never_compared(tmp_path):
    # `copy_markdown` is undecided in the null AND not exercised on the result, so nothing
    # anywhere needs its verdict. Demanding it turns machine speed into a red build.
    null = null_run(
        tmp_path,
        "n",
        two_reps("settings", "SAME", "SAME") + [("r100K", "rep0", "copy_markdown", "A", "B")],
    )
    result = null_run(tmp_path, "r", two_reps("settings", "SAME", "SAME"))

    assert U.audit_null([null], frozenset())[0] == 1  # unscoped: copy_markdown undecided
    scope = U.compared_actions([result])
    assert "copy_markdown" not in scope
    assert U.audit_null([null], frozenset(), scope)[0] == 0  # scoped: nobody needed it


def test_the_audit_still_fails_on_an_action_the_result_did_compare(tmp_path):
    # The other direction, so scoping cannot pass by excusing everything: an action the result
    # put a verdict on, that the null could not decide, is exactly the unscored case.
    null = null_run(
        tmp_path,
        "n2",
        two_reps("settings", "SAME", "SAME") + [("r100K", "rep0", "thread_reopen", "A", "B")],
    )
    result = null_run(
        tmp_path,
        "r2",
        two_reps("settings", "SAME", "SAME") + two_reps("thread_reopen", "A", "A"),
    )
    scope = U.compared_actions([result])
    assert "thread_reopen" in scope
    assert U.audit_null([null], frozenset(), scope)[0] == 1

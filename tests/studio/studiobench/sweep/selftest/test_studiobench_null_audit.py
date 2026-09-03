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

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402

# parents[4] is `tests/`, which the sibling selftests put on sys.path; the repo root is one
# further up and is where .github lives.
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


def null_run(
    tmp_path: Path,
    name: str,
    cells: list[tuple[str, str, str, str | None, str | None]],
    tier: str = "fast",
    corpus: str | None = None,
):
    """`cells` is (rung, rep, action, base digest, treatment digest)."""
    meta = {"row_type": "run_meta", "tier": tier}
    if corpus is not None:
        meta["corpus_hash"] = corpus
    rows: list[dict] = [meta]
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
    # Every action reached two observations and none differed: the best null control obtainable,
    # whose measured unstable set is empty BECAUSE there was nothing to measure. Requiring a
    # measured entry here would fail the job on its best day.
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

    unstable, _derived, _checks = U.unstable_set([null])
    assert not [e for e in unstable if isinstance(e, tuple)]


def test_a_null_control_with_one_observation_each_is_undecided(tmp_path):
    # The other direction, so the audit cannot pass by never failing: --reps 1 gives one
    # observation per (rung, action) where `derive_unstable` needs two, so the measured set is
    # empty for a completely different reason.
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
    # Both of the above derive an EMPTY measured set, so assert the audit distinguishes the best
    # possible reading from a broken one, not merely that each has the right code.
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
    # A failed capture is blind, and `derive_unstable` refuses to count it: an action derived as
    # stable from pairs that never rendered would be trusted forever on nothing. So it must read
    # UNDECIDED, never a quiet pass.
    null = null_run(tmp_path, "blind", two_reps("image_upload", None, None))
    rc, report = U.audit_null([null])
    assert rc == 1
    assert report["undecided"] == [("r100K", "image_upload")]


def test_an_excused_action_is_allowed_but_still_needs_a_decided_one(tmp_path):
    # image_upload is permanently blind on this fixture, so it is excusable; a run in which
    # EVERYTHING is excused measured nothing at all while naming a reason for each blank.
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
    # Twice now a guard written as a regex over this tool's printed prose has been wrong: a real
    # payload spells its rung `r100K` where a fixture spells it `100K`, and the literal
    # `action@rung` appears in the explanatory text whether or not anything was measured.
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert "--audit-null" in text
    assert "--allow-undecided image_upload" in text
    # The prose may still EXPLAIN the two regexes that were wrong; what it may not do is run one.
    assert "grep -Eo" not in text, "the guard must not key on the tool's printed prose again"


# ── a difference that did not repeat is not a change to the build ────


def test_a_difference_in_one_repetition_of_two_is_not_counted(tmp_path):
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
    # (one <span> added inside the thread root), which stays red at --min-reps 2.
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
    # single flake recorded twice would corroborate itself.
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
    null = run_with_completion(
        tmp_path,
        "truncated",
        [("rep0", "A", "B", True), ("rep1", "A", "B", False)],
    )
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
    # The OTHER sign, and the one that matters more: a cell that died is the latest attempt at
    # itself and the difference it saw is real, so dropping it would silence a regression.
    # test_an_attempt_that_was_never_re_run_still_carries_its_parity_verdict holds the same line
    # from the other file.
    mine = run_with_completion(
        tmp_path,
        "died",
        [("rep0", "A", "B", False), ("rep1", "A", "B", False)],
    )
    assert U.compare_all([mine])[0] != []
    assert U.report([mine], "t", frozenset(), min_reps = 2) == 1


def test_a_payload_with_no_cell_rows_says_it_could_not_check(tmp_path, capsys):
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
    # min_observations and report DECIDED from two different threads, so the result is scored
    # against a set never measured on the corpus it is applied to.
    a = corpus_run(tmp_path, "c1", "corpusAAAA", "rep0")
    b = corpus_run(tmp_path, "c2", "corpusBBBB", "rep1")
    with pytest.raises(SystemExit) as got:
        U.one_corpus([a, b], "null control")
    assert "more than one corpus" in str(got.value)

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
    # The measured claim in miniature: over 18 mutations of a real payload at 2 to 24 missed slots
    # the verdict never gained a red, because NOT_EXERCISED is filed under coverage.
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
    # `copy_markdown` is undecided in the null and nothing on the result turns on its verdict, so
    # demanding one turns machine speed into a red build. The result DOES need an excuse for
    # `settings`, so the audit is not passing vacuously.
    null = null_run(
        tmp_path,
        "n",
        two_reps("settings", "SAME", "SAME") + [("r100K", "rep0", "copy_markdown", "A", "B")],
    )
    result = null_run(tmp_path, "r", two_reps("settings", "AAA", "BBB"))

    assert U.audit_null([null], frozenset())[0] == 1
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert scope == {("r100K", "settings")}
    assert U.audit_null([null], frozenset(), scope)[0] == 0


def test_the_audit_still_fails_on_an_action_the_result_did_compare(tmp_path):
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


# ── a null control from another film or another thread is not an excuse ──


def test_a_null_from_another_tier_is_refused_rather_than_applied(tmp_path):
    assert U.cross_side_mismatch({"standard"}, {"fast"}, {"c1"}, {"c1"}).startswith(
        "the null control was recorded at tier"
    )


def test_a_null_from_another_corpus_is_refused_rather_than_applied(tmp_path):
    got = U.cross_side_mismatch({"fast"}, {"fast"}, {"c2"}, {"c1"})
    assert got.startswith("the null control was recorded at corpus")


def test_matching_sides_and_unrecorded_sides_are_both_allowed():
    assert U.cross_side_mismatch({"fast"}, {"fast"}, {"c1"}, {"c1"}) == ""
    assert U.cross_side_mismatch({"fast"}, set(), {"c1"}, set()) == ""
    assert U.cross_side_mismatch(set(), {"fast"}, set(), {"c1"}) == ""


def test_the_refusal_exits_two_not_one_through_the_cli(tmp_path, monkeypatch, capsys):
    # Exit 2 is the tool declining to answer; exit 1 would read as a parity failure and send
    # somebody hunting for a UI change that was never measured.
    null = null_run(tmp_path, "tier_null", two_reps("settings", "SAME", "SAME"), tier = "fast")
    result = null_run(tmp_path, "tier_res", two_reps("settings", "SAME", "SAME"), tier = "standard")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ui_parity", "--min-reps", "2", "--null", str(null.parent), str(result.parent)],
    )
    assert U.main() == 2
    assert "REFUSING to score" in capsys.readouterr().out


# ── an action that ran on one arm and not the other ──────────────────


def one_sided_payload(
    tmp_path: Path,
    name: str,
    action: str,
    reps: tuple[str, ...],
    reason: str = "the control never became visible",
) -> Path:
    """A film where `action` runs on base and cannot be performed on treatment, in `reps`.

    Everything else matches on both arms in both repetitions, so the ONLY thing wrong with this
    run is that the head build could not open one control. Sixteen filler actions keep the run
    well above any coverage floor, because "the film barely ran" is the other failure and the two
    must not be able to stand in for each other.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            if arm == "treatment" and rep in reps:
                row = action_row(cid, action, "SAME")
                row["ran"] = False
                row["reason"] = reason
                row["slot_missed"] = False
                rows.append(row)
            else:
                rows.append(action_row(cid, action, "SAME"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_an_action_the_head_can_no_longer_perform_fails_the_verdict(tmp_path, capsys):
    # THE HOLE THIS CLOSES: a control that stops opening is the one user-visible regression that
    # leaves no digest to differ, and filing NOT_EXERCISED under coverage meant a button that no
    # longer opens shipped green with 64 of 68 pairs still compared, comfortably above the
    # workflow's --min-compared 16.
    path = one_sided_payload(tmp_path, "regressed", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 1
    printed = capsys.readouterr().out
    assert "RAN ON ONE ARM ONLY" in printed
    assert "settings" in printed


def test_one_arms_missed_slot_in_one_repetition_is_still_only_a_warning(tmp_path, capsys):
    # The other half, and why this is not failed on sight: a contended runner can lose one arm's
    # slot once, so it is held to the SAME corroboration bar as a differing digest. A build that
    # cannot open a control cannot open it on either pass.
    path = one_sided_payload(tmp_path, "flake", "settings", ("rep0",))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0
    assert "UNCORROBORATED one-arm-only" in capsys.readouterr().out


def test_an_action_expected_to_vary_is_not_failed_for_reaching_one_arm_only(tmp_path, capsys):
    # `stop_generation` runs only while a stream is live, so which arm reached it before the
    # stream ended is the race its UNSTABLE_ACTIONS entry already describes.
    path = one_sided_payload(
        tmp_path,
        "racy",
        "stop_generation",
        ("rep0", "rep1"),
        reason = "nothing was generating and a new turn did not start within 8s",
    )
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    assert "expected to vary between runs" in capsys.readouterr().out


def test_both_arms_missing_the_same_action_is_coverage_and_not_a_verdict(tmp_path):
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            rows.append(action_row(cid, "settings", "SAME"))
            rows.append(not_run_row(cid, "copy_markdown"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "symmetric"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    assert U.report([path], "t", frozenset(), min_reps = 2) == 0


# ── the audit's scope: what an excuse would actually move ─────────────


def test_an_undecided_action_the_result_only_matched_does_not_fail_the_audit(tmp_path):
    # THE CI FAILURE THIS FIXES: the null lost one arm's slot on `settings` and cannot decide it,
    # while the result MATCHED on `settings` in both repetitions, so the null was never going to
    # be asked. Requiring an opinion made the gate unsatisfiable on a shared runner.
    result = null_run(tmp_path, "result", two_reps("settings", "SAME", "SAME"))
    null = null_run(tmp_path, "null", [("r100K", "rep0", "settings", "SAME", "SAME")])
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert scope == set()
    rc, _ = U.audit_null([null], frozenset(), scope)
    assert rc == 0


def test_the_audit_still_fails_on_an_undecided_action_the_result_kept_differing_on(tmp_path):
    # THE SAFETY DIRECTION, and why the scope is not simply dropped: `settings` differing in BOTH
    # repetitions is scored as a regression unless the unstable set excuses it, and an undecided
    # null excuses it from the DECLARED list, unmeasured.
    result = null_run(tmp_path, "result", two_reps("settings", "AAA", "BBB"))
    null = null_run(tmp_path, "null", [("r100K", "rep0", "settings", "SAME", "SAME")])
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert scope == {("r100K", "settings")}
    rc, report_ = U.audit_null([null], frozenset(), scope)
    assert rc == 1
    assert ("r100K", "settings") in report_["undecided"]


def test_a_difference_in_one_repetition_only_needs_no_excuse(tmp_path):
    result = null_run(
        tmp_path,
        "result",
        [
            ("r100K", "rep0", "settings", "AAA", "BBB"),
            ("r100K", "rep1", "settings", "SAME", "SAME"),
        ],
    )
    assert U.actions_needing_an_excuse([result], min_reps = 2) == set()


def test_a_corroborated_one_arm_only_action_still_needs_the_null_to_decide_it(tmp_path):
    # The other shape the unstable set is load-bearing for: `report` excuses a one-arm-only action
    # when it is in the unstable set, so an undecided null hands out that excuse on the declared
    # list alone.
    result = one_sided_payload(tmp_path, "result", "settings", ("rep0", "rep1"))
    null = null_run(tmp_path, "null", [("r100K", "rep0", "settings", "SAME", "SAME")])
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert ("r100K", "settings") in scope
    rc, _ = U.audit_null([null], frozenset(), scope)
    assert rc == 1


def test_the_workflow_scopes_the_audit_with_the_verdicts_own_threshold(tmp_path):
    # An audit scoped by a different --min-reps than the verdict is auditing a verdict nobody runs.
    wf = WORKFLOW.read_text(encoding = "utf-8")
    audit = wf.split("--audit-null", 1)[1].split("outputs/parity-null-control", 1)[0]
    assert "--compared-in outputs/parity-result" in audit
    assert "--min-reps 2" in audit


def missed_slot_payload(tmp_path: Path, name: str, action: str, reps: tuple[str, ...]) -> Path:
    """Like `one_sided_payload`, except the base arm MISSED THE SLOT rather than failed to open."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            row = action_row(cid, action, "SAME")
            if arm == "base" and rep in reps:
                row["ran"] = False
                row["slot_missed"] = True
                row["reason"] = (
                    "the slot opened at 45700ms and this machine reached it at 47052ms, "
                    "past its 1200ms budget"
                )
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_one_arm_missing_a_slot_in_every_repetition_is_still_not_a_build_difference(tmp_path):
    # MEASURED, not reasoned: `settings` missed a 1200ms budget on the BASE arm at 45700ms in both
    # repetitions, by 1292ms and 1352ms. The schedule is fixed, so the two misses are correlated
    # through the runner and corroboration cannot separate them from a build difference, while
    # `slot_missed` can. Note which arm too: reading it as a regression would have blamed the head
    # build for the old one's timing.
    path = missed_slot_payload(tmp_path, "slow", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0


def test_a_control_that_cannot_open_is_still_caught_when_a_slot_is_also_missed_elsewhere(tmp_path):
    path = one_sided_payload(tmp_path, "broken", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 1


def test_a_difference_at_one_rung_does_not_demand_the_null_decide_another(tmp_path):
    # The scope is keyed by (rung, action), not action alone: reduced to bare names a corroborated
    # difference at 1K pulled the action into scope at EVERY rung, so one missed observation at
    # 100K failed the audit again. Instability is a property of the rung.
    result = null_run(
        tmp_path,
        "result",
        [("r1K", "rep0", "keystroke", "A", "B"), ("r1K", "rep1", "keystroke", "A", "B")]
        + [("r100K", "rep0", "keystroke", "S", "S"), ("r100K", "rep1", "keystroke", "S", "S")],
    )
    null = null_run(
        tmp_path,
        "null",
        [("r1K", "rep0", "keystroke", "A", "B"), ("r1K", "rep1", "keystroke", "A", "B")]
        + [("r100K", "rep0", "keystroke", "S", "S")],
    )
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert scope == {("r1K", "keystroke")}
    rc, report_ = U.audit_null([null], frozenset(), scope)
    assert rc == 0, report_["undecided"]
    assert ("r100K", "keystroke") in report_["out_of_scope"]


# ── the completion guard has to answer about the attempt it is guarding ──


def test_an_interrupted_retry_does_not_inherit_the_completion_it_superseded(tmp_path):
    """A superseded attempt's `cell` row must not admit the dead retry's action rows.

    `_resume_set` names the path that gets here without anybody doing anything unusual: an A/B
    pair is re-run WHOLE (`ab.skippable_cells`), so a resume re-runs an arm that had already
    succeeded. If that retry is interrupted, the payload holds a completed `cell` row and a LATER,
    unfinished set of action rows under the same deterministic `cell_id`. `latest_attempt_rows`
    correctly names the retry as the latest, so its rows are the ones scored -- but `completed`
    was read from the RAW stream, so the guard cleared them on the strength of the completion the
    attempt before them earned.

    On the null that is the worst direction. One valid repetition plus this half-written one is
    exactly `min_observations`, so the action is declared unstable at that rung on a reading the
    run itself threw away, and a real difference in the result then prints under "expected to
    vary" while the command exits 0.
    """
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for arm in ("base", "treatment"):
        cid = f"r100K.{arm}.rep0"
        row = action_row(cid, "settings", "SAME")
        row["session_id"] = "s1"
        rows.append(row)
        rows.append({"row_type": "cell", "cell_id": cid, "completed": True, "session_id": "s1"})
    for arm in ("base", "treatment"):
        cid = f"r100K.{arm}.rep1"
        row = action_row(cid, "settings", "SAME")
        row["session_id"] = "s1"
        rows.append(row)
        rows.append({"row_type": "cell", "cell_id": cid, "completed": True, "session_id": "s1"})
    for arm, digest in (("base", "SAME"), ("treatment", "OTHER")):
        row = action_row(f"r100K.{arm}.rep1", "settings", digest)
        row["session_id"] = "s2"
        rows.append(row)
    rows.append(
        {"row_type": "cell", "cell_id": "r100K.base.rep1", "completed": True, "session_id": "s2"}
    )

    out = tmp_path / "resumed"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    got = U.collect([path], require_complete = True)
    assert got["incomplete"] == 1, got
    keys = {(k[2], k[4]) for k in got["pairs"]}
    assert ("rep0", "settings") in keys
    assert not any(
        "base" in sides and "treatment" in sides
        for k, sides in got["pairs"].items()
        if k[2] == "rep1" and k[3] == "s2"
    ), got["pairs"]

    unstable, _derived, _checks = U.unstable_set([path])
    assert ("r100K", "settings") not in unstable, unstable


# The audit block sat below the structural early return while reading nothing from the plan,
# so a payload with no structural entries returned first and it never ran; and `--mode
# visible` / `--mode behaviour` hard-set `structural` to empty, so those two could never audit
# at all.


# ── the audit has to run in the modes that ask for it ────────────────
def test_the_cli_audits_a_single_repetition_under_forced_visible_mode(tmp_path, capsys):
    """THE DEFECT. Same payload and same expectation as the auto-mode failure test above, with
    `--mode visible` added: the audit must still run and still fail."""
    null_run(tmp_path, "visaudit", [("r100K", "rep0", "settings", "SAME", "SAME")])
    rc = U.main(["--audit-null", str(tmp_path / "visaudit"), "--mode", "visible"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "NULL CONTROL AUDIT: UNDECIDED" in out, out


def test_the_cli_audits_a_single_repetition_under_forced_behaviour_mode(tmp_path, capsys):
    """The other forced mode, which reaches the same empty structural plan by the same line."""
    null_run(tmp_path, "behaudit", [("r100K", "rep0", "settings", "SAME", "SAME")])
    rc = U.main(["--audit-null", str(tmp_path / "behaudit"), "--mode", "behaviour"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "NULL CONTROL AUDIT: UNDECIDED" in out, out


def test_a_quiet_null_still_passes_the_audit_under_a_forced_mode(tmp_path, capsys):
    """The positive control: reaching the audit in these modes must not make it fail everything."""
    null_run(tmp_path, "visquiet", two_reps("settings", "SAME", "SAME"))
    rc = U.main(["--audit-null", str(tmp_path / "visquiet"), "--mode", "visible"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "NULL CONTROL AUDIT: DECIDED" in out, out


# ── an exemption measured on another machine is not an exemption here ──


def _arm_payload(tmp_path: Path, name: str, regress: str | None, self_race: str | None) -> Path:
    """One arm's payload. `regress` differs base-vs-head in BOTH reps; `self_race` differs base
    against ITSELF in both reps, which is what makes an action look unstable."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            for act in ("reasoning_toggle", "settings"):
                digest = "SAME"
                if act == regress and arm == "treatment":
                    digest = "HEAD_IS_DIFFERENT"
                if act == self_race:
                    digest = f"RACE_{arm}_{rep}"
                rows.append(action_row(cid, act, digest))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_the_other_runners_race_does_not_excuse_a_regression_on_this_one(tmp_path, capsys):
    """The hole the two-runner matrix leaves open, in miniature, and it is not hypothetical.

    On run 32648192384 of this workflow the two arms drew runner ids 1000628315 and 1000628341
    and started 2m18s apart. The null derived three exemptions; the runner the result was measured
    on reproduced exactly ONE of them. `reasoning_toggle@r100K` was not on the declared list, so
    it was excused solely by a race on a machine the result never touched -- and a corroborated
    head regression there would have shipped green.
    """
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    result = _arm_payload(tmp_path, "result", regress = "reasoning_toggle", self_race = None)

    imported, derived, _ = U.unstable_set([null])
    assert ("r100K", "reasoning_toggle") in imported, imported
    assert derived

    assert U.report([result], "imported", imported, min_reps = 2, min_compared = 16) == 0

    local_unstable, local_stable = U.in_arm_repeatability([result])
    assert ("r100K", "reasoning_toggle") in local_stable
    assert ("r100K", "reasoning_toggle") not in local_unstable
    effective, dropped = U.confine_to_runner(imported, local_unstable, local_stable)
    assert dropped == [("r100K", "reasoning_toggle")], dropped
    assert U.report([result], "confined", effective, min_reps = 2, min_compared = 16) == 1
    assert "reasoning_toggle" in capsys.readouterr().out


def test_an_exemption_this_runner_reproduces_is_kept(tmp_path):
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    result = _arm_payload(tmp_path, "result", regress = None, self_race = "reasoning_toggle")
    imported, _derived, _ = U.unstable_set([null])
    effective, dropped = U.confine_to_runner(imported, *U.in_arm_repeatability([result]))
    assert dropped == []
    assert ("r100K", "reasoning_toggle") in effective


def test_an_action_this_runner_could_not_decide_keeps_its_exemption(tmp_path):
    # UNDECIDED IS NOT STABLE: one repetition of side A is one observation, and reading it as
    # "this runner says the action is repeatable" would turn a lost slot into a red job.
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    rows = [
        json.loads(line)
        for line in _arm_payload(tmp_path, "result", regress = None, self_race = None)
        .read_text(encoding = "utf-8")
        .splitlines()
    ]
    rows = [
        r
        for r in rows
        if not (
            r.get("row_type") == "action"
            and r.get("action") == "reasoning_toggle"
            and r.get("cell_id") == "r100K.base.rep1"
        )
    ]
    out = tmp_path / "thin"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    local_unstable, local_stable = U.in_arm_repeatability([path])
    assert ("r100K", "reasoning_toggle") not in local_stable
    assert ("r100K", "reasoning_toggle") not in local_unstable
    imported, _derived, _ = U.unstable_set([null])
    effective, dropped = U.confine_to_runner(imported, local_unstable, local_stable)
    assert dropped == []
    assert ("r100K", "reasoning_toggle") in effective


def test_a_declared_exemption_is_never_dropped_by_a_runner_measurement(tmp_path):
    # The declared list is a standing claim about the app, not a measurement of a machine, so a
    # machine cannot contradict it; only the `(rung, action)` entries are runner-derived.
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    result = _arm_payload(tmp_path, "result", regress = None, self_race = None)
    imported, _derived, _ = U.unstable_set([null])
    effective, _dropped = U.confine_to_runner(imported, *U.in_arm_repeatability([result]))
    assert U.UNSTABLE_ACTIONS <= effective


def test_the_verdict_confines_the_imported_set_when_driven_through_main(tmp_path, capsys):
    # Driven through main() rather than report(), because the failure guarded against is a correct
    # confinement that never reaches the verdict: the same shape that shipped once in this file's
    # neighbour, where --min-reps was parsed and never passed to build().
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    result = _arm_payload(tmp_path, "result", regress = "reasoning_toggle", self_race = None)
    rc = U.main(
        [
            "--min-reps",
            "2",
            "--min-compared",
            "16",
            "--null",
            str(null.parent),
            str(result.parent),
        ]
    )
    out = capsys.readouterr().out
    assert "imported exemption(s) DROPPED" in out, out
    assert "reasoning_toggle" in out
    assert rc == 1, "the regression the other runner's race was excusing must red the job"


# ── an action that ran and failed its own assertion ──────────────────


def _expect_payload(
    tmp_path: Path,
    name: str,
    action: str,
    failed_on: str | None,
    reps: tuple[str, ...] = ("rep0", "rep1"),
    both: bool = False,
    expect_ok_value: object = True,
) -> Path:
    """`action` runs on BOTH arms with the SAME digest; its assertion fails on `failed_on`."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                r = action_row(cid, f"filler{i}", "SAME")
                r["expect_ok"] = True
                rows.append(r)
            r = action_row(cid, action, "SAME")
            r["expect_ok"] = expect_ok_value
            if rep in reps and (both or arm == failed_on):
                r["expect_ok"] = False
                r["reason"] = "clicking Stop did not end the stream"
            rows.append(r)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_control_that_stopped_working_is_not_excused_by_its_digest_exemption(tmp_path, capsys):
    """The double exemption, and it is the one the declared list makes worst.

    `stop_generation` returns `ran = True, expect_ok = stopped_ms is not None`, so a head on which
    Stop no longer ends the stream records a row whose `ran` is true and whose digest then differs
    for the ordinary reason. It is also ON the declared unstable list, so that difference is
    excused. `compare_rows` read only `ran`, and the result was a user-visible regression --
    generation cannot be stopped -- passing the gate in both repetitions.

    The assertion is not a digest, so the digest exemption does not reach it.
    """
    path = _expect_payload(tmp_path, "regressed", "stop_generation", failed_on = "treatment")
    assert "stop_generation" in U.UNSTABLE_ACTIONS
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1
    printed = capsys.readouterr().out
    assert "ASSERTION FAILED ON ONE ARM" in printed.upper()
    assert "stop_generation" in printed


def test_both_arms_failing_the_assertion_is_lost_coverage_not_a_difference(tmp_path):
    path = _expect_payload(tmp_path, "both", "stop_generation", failed_on = None, both = True)
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0


def test_an_action_that_asserts_nothing_is_not_an_assertion_failure(tmp_path):
    # `expect_ok is None` is "this action makes no claim", which every payload recorded before the
    # field existed also carries, so reading None as False would red every one of them.
    path = _expect_payload(
        tmp_path, "none", "stop_generation", failed_on = None, expect_ok_value = None
    )
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    # Asserted on the signal as well as the exit code: through the exit code alone this case is
    # indistinguishable from the both-arms one, and `expect_regressed` is where the distinction
    # lives.
    row = {
        "ran": True,
        "expect_ok": None,
        "parity": {
            "parity_attempted": True,
            "digest": "A",
            "root_kind": "thread",
            "chars": 1,
            "messages": [],
            "overlays": [],
            "style": {"style_attempted": True, "capped": False, "nodes": []},
        },
    }
    other = dict(row, expect_ok = True)
    assert P.compare_rows(row, other)["expect_regressed"] == ""
    assert P.compare_rows(other, row)["expect_regressed"] == ""


def test_an_assertion_that_failed_in_one_repetition_of_two_is_not_counted(tmp_path, capsys):
    path = _expect_payload(
        tmp_path, "flake", "stop_generation", failed_on = "treatment", reps = ("rep0",)
    )
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    assert "UNCORROBORATED assertion failure" in capsys.readouterr().out


# ── two repetitions that blame OPPOSITE arms are not one finding ─────


def _reversing_expect_payload(tmp_path: Path, name: str, action: str) -> Path:
    """`action` fails its assertion on TREATMENT in rep0 and on BASE in rep1."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                r = action_row(cid, f"filler{i}", "SAME")
                r["expect_ok"] = True
                rows.append(r)
            r = action_row(cid, action, "SAME")
            bad = (arm == "treatment" and rep == "rep0") or (arm == "base" and rep == "rep1")
            r["expect_ok"] = not bad
            if bad:
                r["reason"] = "clicking Stop did not end the stream"
            rows.append(r)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_an_assertion_that_blames_a_different_arm_each_time_is_not_corroborated(tmp_path, capsys):
    """A race that landed on either side is not a build that consistently failed.

    Grouped by action and rung alone, a treatment failure in rep0 and a base failure in rep1 are
    two distinct repetition labels, so the pair reached `firm` and the job exited 1 reporting
    "the two builds did not behave the same way" -- directly above its own two lines naming
    OPPOSITE arms. Neither build failed twice.

    Keyed on the direction they separate, so each side is a single repetition and both print as
    UNCORROBORATED. The safe direction: this can only ever under-count, and an under-count is
    visible in the output rather than silent.
    """
    path = _reversing_expect_payload(tmp_path, "reversing", "stop_generation")
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    printed = capsys.readouterr().out
    assert "UNCORROBORATED assertion failure" in printed
    assert "ASSERTION failed on one arm:  0" in printed


def test_an_assertion_that_blames_the_same_arm_twice_still_fails_the_job(tmp_path):
    # The other side of the same key, and why the fix is not just "require more": a build that
    # consistently fails the assertion is what this category exists to catch and must survive the
    # direction keying.
    path = _expect_payload(tmp_path, "consistent", "stop_generation", failed_on = "treatment")
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1


def test_one_arm_only_that_swaps_arms_between_repetitions_is_not_corroborated(tmp_path, capsys):
    # The same defect on the one-arm-only category: a precondition race stopping the treatment arm
    # in rep0 and the base arm in rep1 says nothing about either build.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            row = action_row(cid, "settings", "SAME")
            if (arm == "treatment" and rep == "rep0") or (arm == "base" and rep == "rep1"):
                row["ran"] = False
                row["reason"] = "the control never became visible"
                row["slot_missed"] = False
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "swap"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0
    assert "UNCORROBORATED one-arm-only" in capsys.readouterr().out


# ── an action the null never measured at all ─────────────────────────


def _scope_payload(tmp_path: Path, name: str, actions: tuple[str, ...]) -> Path:
    """Every listed action differs between the arms in both repetitions. Others are absent."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            for act in actions:
                rows.append(action_row(cid, act, "HEAD" if arm == "treatment" else "SAME"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_scoped_action_with_no_rows_in_the_null_is_undecided_not_absent(tmp_path):
    """The audit's own question, answered by default.

    `audit_null` classified only what `derive_unstable` produced, so a scoped (rung, action) with
    NO rows in the null payload landed in neither `decided` nor `undecided`. One other scoped
    action being decided was then enough to return 0 -- and `unstable_set` unions the DECLARED
    names back in, so `send_turn` was still excused by name and a corroborated result difference
    on it passed the verdict with zero null-control observation.

    Rows vanish more easily than it looks: the null is collected with `require_complete = True`,
    so a cell that never finished takes all of its action rows with it and the action stops
    existing rather than becoming undetermined.
    """
    null = _scope_payload(tmp_path, "null", ("settings",))
    result = _scope_payload(tmp_path, "result", ("settings", "send_turn"))
    scope = U.actions_needing_an_excuse([result], 2)
    assert scope == {("r100K", "settings"), ("r100K", "send_turn")}, scope

    rc, report_ = U.audit_null([null], frozenset(), scope)
    assert ("r100K", "send_turn") in report_["missing"], report_
    assert ("r100K", "send_turn") in report_["undecided"], report_
    assert report_["decided"] == [("r100K", "settings")]
    assert rc == 1, "an action the null never measured cannot license an excuse"

    unstable, _derived, _checks = U.unstable_set([null])
    assert ("r100K", "send_turn") not in unstable
    assert U.is_unstable(unstable, "send_turn", "r100K rep0")


def test_a_missing_scoped_action_on_the_waived_list_is_still_waived(tmp_path):
    null = _scope_payload(tmp_path, "null", ("settings",))
    result = _scope_payload(tmp_path, "result", ("settings", "image_upload"))
    rc, report_ = U.audit_null(
        [null], frozenset({"image_upload"}), U.actions_needing_an_excuse([result], 2)
    )
    assert ("r100K", "image_upload") in report_["missing"]
    assert ("r100K", "image_upload") in report_["excused"]
    assert ("r100K", "image_upload") not in report_["undecided"]
    assert rc == 0


def test_an_action_outside_the_scope_is_not_required_to_exist(tmp_path):
    null = _scope_payload(tmp_path, "null", ("settings",))
    result = _scope_payload(tmp_path, "result", ("settings",))
    rc, report_ = U.audit_null([null], frozenset(), U.actions_needing_an_excuse([result], 2))
    assert report_["missing"] == []
    assert rc == 0


# ── digest instability does not exempt being unable to run ───────────


def test_a_broken_control_is_not_excused_because_its_digest_varies(tmp_path, capsys):
    """The exemption that covered nine of the sixteen scheduled actions.

    `keystroke` is on the declared unstable list because "how many keystrokes had landed by the
    capture deadline is a race" -- a statement about the CAPTURE. It was also being used to
    excuse the treatment arm being unable to type at all, which is a different claim and the one
    regression shape that leaves no digest to differ. A composer broken by the head build takes
    `keystroke` down in both repetitions and the job exited 0.
    """
    path = one_sided_payload(tmp_path, "broken", "keystroke", ("rep0", "rep1"))
    assert "keystroke" in U.UNSTABLE_ACTIONS
    assert "keystroke" not in P.RACY_EXECUTION
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1
    assert "RAN ON ONE ARM ONLY" in capsys.readouterr().out


def test_an_action_with_no_not_run_path_is_never_exempt_from_one_arm_only(tmp_path):
    # `scroll_after` has no `not_run` in `scene/actions.py` at all, so a `ran: false` for it cannot
    # be a race under any reading, yet the digest list exempted it.
    path = one_sided_payload(tmp_path, "scroll", "scroll_after", ("rep0", "rep1"))
    assert "scroll_after" in U.UNSTABLE_ACTIONS
    assert "scroll_after" not in P.RACY_EXECUTION
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1


def test_every_racy_execution_entry_states_its_mechanism():
    # The same bar the digest list is held to: an exemption without a stated mechanism is how the
    # nine-action version survived unexamined, and the markers are the operative half, since an
    # entry matching nothing, or everything, is the failure this keying prevents.
    for action, (why, markers) in P.RACY_EXECUTION.items():
        assert action in P.UNSTABLE_ACTIONS, action
        assert "not_run" in why or "not run" in why, action
        assert len(why) > 60, action
        assert markers and all(len(m) > 10 for m in markers), action


# ── the scope is what the VERDICT turns on, not what merely happened ──


def test_a_racy_execution_action_is_not_put_in_the_audit_scope(tmp_path):
    """`report` does not count it, so no excuse can move it and the null owes it nothing.

    Scoped anyway, the null observing the same legitimate stream-timing race made the audit
    return 1 and failed the workflow on stream timing -- a verdict of 0 with a red job.
    """
    path = one_sided_payload(
        tmp_path,
        "racy",
        "stop_generation",
        ("rep0", "rep1"),
        reason = "nothing was generating and a new turn did not start within 8s",
    )
    assert "stop_generation" in P.RACY_EXECUTION
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    assert ("r100K", "stop_generation") not in U.actions_needing_an_excuse([path], 2)


def test_a_direction_reversing_one_sided_pair_is_not_put_in_the_audit_scope(tmp_path):
    # Same rule, the other axis: `report` keys corroboration on the live arm, so a pair blaming
    # opposite arms cannot move the verdict, yet built without the direction it entered the scope
    # and an undecided null failed a passing job.
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            row = action_row(cid, "settings", "SAME")
            if (arm == "treatment" and rep == "rep0") or (arm == "base" and rep == "rep1"):
                row["ran"] = False
                row["reason"] = "the control never became visible"
                row["slot_missed"] = False
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "swap2"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")

    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0
    assert ("r100K", "settings") not in U.actions_needing_an_excuse([path], 2)


def test_a_control_the_head_cannot_open_is_still_in_the_audit_scope(tmp_path):
    path = one_sided_payload(tmp_path, "real", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 1
    assert ("r100K", "settings") in U.actions_needing_an_excuse([path], 2)


def test_a_removed_stop_button_is_not_exempt_just_because_stop_generation_can_race(tmp_path):
    """The exemption has to match the not-run it names, not the action it is filed under.

    `stop_generation` has two not_run paths: nothing was generating (a race with the model) and
    the stop button being absent (`scene/actions.py:501`, which is the build). Keyed by action
    name alone, a treatment build that REMOVES the Stop control recorded exactly the one-arm-only
    regression this category exists to catch and was filed under "expected to vary".
    """
    path = one_sided_payload(
        tmp_path,
        "removed",
        "stop_generation",
        ("rep0", "rep1"),
        reason = "the stop button is not present",
    )
    assert not P.racy_execution("stop_generation", "the stop button is not present")
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1
    assert ("r100K", "stop_generation") in U.actions_needing_an_excuse([path], 2)


def test_a_missing_composer_is_not_exempt_just_because_send_turn_can_be_queued(tmp_path):
    racy = one_sided_payload(
        tmp_path,
        "queued",
        "send_turn",
        ("rep0", "rep1"),
        reason = "a reply was still streaming, so this send would have been queued",
    )
    assert U.report([racy], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0

    gone = one_sided_payload(
        tmp_path, "gone", "send_turn", ("rep0", "rep1"), reason = "no composer on the page"
    )
    assert U.report([gone], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1


# WHAT TURNED RED: a wave of `studiobench UI parity` audited UNDECIDED with `keystroke@r100K`
# and `send_turn@r100K` "never reached min_observations" despite --min-reps 2 and both arms
# running both repetitions: the two actions whose slot lands INSIDE a live turn on the packed
# 100K film. Pre-#9575 that pair scored DIFFER, which is wrong but is an OBSERVATION; #9575
# correctly demoted it to NOT_COMPARABLE, which `derive_unstable` files under `blind` (zero
# observations). So the refusal is right and the accounting was not: `SETTLED_MATCH` is the
# narrow repair, reached only once the scaffold, overlays, message set and every settled row
# have agreed. Held in both directions, since a fix that makes the audit easier to pass is
# worth nothing without the control showing it still fails.


# ── the reply that was still arriving is not a blind capture ──────────
def streaming_pair_rows(
    cid_base: str,
    cid_treat: str,
    action: str,
    *,
    base_tail: str,
    treat_tail: str,
    scaffold: str = "SCAF",
    settled: str = "SETTLED",
    treat_settled: str | None = None,
    treat_scaffold: str | None = None,
    treat_role: str | None = None,
) -> list[dict]:
    """A base/treatment pair captured with message 1 STILL BEING WRITTEN on both arms.

    Modelled on the real `keystroke@r100K` rows from the failing run: both arms streaming, both
    naming the same message in flight, the same composer control, and the settled half of the
    thread byte-identical. `base_tail`/`treat_tail` are the digest of the still-arriving message,
    the only thing wall clock gets to move.

    `treat_settled`, `treat_scaffold` and `treat_role` let the negative controls move something
    the stream provably cannot reach, and watch the branch decline to fire.
    """

    def side(
        cid: str,
        tail: str,
        settled_digest: str,
        scaffold_digest: str,
        role: str = "assistant",
    ) -> dict:
        return {
            "row_type": "action",
            "cell_id": cid,
            "action": action,
            "ran": True,
            "timings": {"open_ms": 5.0},
            "parity": {
                "parity_attempted": True,
                "root_kind": "thread",
                "digest": f"{scaffold_digest}:{settled_digest}:{tail}",
                "chars": 100,
                "digest_scaffold": scaffold_digest,
                "chars_scaffold": 40,
                "streaming": True,
                "in_flight": [1],
                "in_flight_unplaced": False,
                "status_hook_present": True,
                "queued_idle": False,
                "composer_control": "Stop",
                "messages": [
                    {"i": 0, "role": "user", "digest": settled_digest, "chars": 10},
                    {"i": 1, "role": role, "digest": tail, "chars": len(tail)},
                ],
                "overlays": [],
                "style": {"style_attempted": True, "capped": False, "nodes": []},
            },
        }

    return [
        side(cid_base, base_tail, settled, scaffold),
        side(
            cid_treat,
            treat_tail,
            settled if treat_settled is None else treat_settled,
            scaffold if treat_scaffold is None else treat_scaffold,
            "assistant" if treat_role is None else treat_role,
        ),
    ]


def streaming_null(tmp_path: Path, name: str, cells: list[list[dict]]) -> Path:
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for cell in cells:
        rows.extend(cell)
    out = tmp_path / name
    out.mkdir(parents = True, exist_ok = True)
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_mid_stream_tail_is_still_a_refusal_and_never_a_pass():
    # FIRST, the thing that must not change: the verdict stays NOT_COMPARABLE, because a treatment
    # build whose live message renders wrongly sits in exactly this shape. The repair is in what
    # the NULL counts.
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
    )
    got = P.compare(rows[0]["parity"], rows[1]["parity"])
    assert got["verdict"] == P.NOT_COMPARABLE
    assert "STILL BEING WRITTEN" in got["reason"]
    assert got["moved"] == []
    assert got[P.SETTLED_MATCH] is True


def test_a_refusal_that_read_the_settled_thread_is_an_observation():
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
    )
    pair = ("keystroke", P.compare(rows[0]["parity"], rows[1]["parity"]))
    row = P.derive_unstable([pair, pair])["keystroke"]
    assert row["observations"] == 2
    assert row["differed"] == 0
    assert row["not_comparable"] == 0
    assert row[P.SETTLED_MATCH] == 2
    assert row["undetermined"] is False
    assert row["unstable"] is False


def test_an_ordinary_refusal_is_still_blind():
    # THE CONTROL: without it the change reads as "count NOT_COMPARABLE as an observation", which
    # would derive an action that never rendered as stable. A failed capture carries no settled
    # reading and stays at zero.
    blind = {"verdict": P.NOT_COMPARABLE, "reason": "threadRoot is not a function"}
    row = P.derive_unstable([("keystroke", blind), ("keystroke", blind)])["keystroke"]
    assert row["observations"] == 0
    assert row["not_comparable"] == 2
    assert row[P.SETTLED_MATCH] == 0
    assert row["undetermined"] is True


def test_a_settled_message_that_moved_is_a_difference_not_an_observation():
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
        treat_settled = "MOVED",
    )
    got = P.compare(rows[0]["parity"], rows[1]["parity"])
    assert got["verdict"] == P.DIFFER
    assert P.SETTLED_MATCH not in got


def test_a_scaffold_that_moved_is_a_difference_not_an_observation():
    # The scaffold half of the same claim: both arms agree on streaming and on the composer
    # control, so this is a scaffolding change on an otherwise identical pair and stays a finding.
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
        treat_scaffold = "MOVED",
    )
    got = P.compare(rows[0]["parity"], rows[1]["parity"])
    assert got["verdict"] == P.DIFFER
    assert P.SETTLED_MATCH not in got


def test_a_live_row_whose_role_changed_carries_no_positive_reading():
    # The control `_any_moved` cannot reach: the role sits beside the digest and an in-flight
    # digest is withheld, so `assistant` on one arm and `user` on the other passes the digest
    # comparison untouched, and `settled_messages_moved` calls that a structural change even in
    # flight.
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
        treat_role = "user",
    )
    base, treat = rows[0]["parity"], rows[1]["parity"]
    assert P.settled_messages_moved(base, treat) == ["msg1:role assistant->user"]
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert P.SETTLED_MATCH not in got
    row = P.derive_unstable([("keystroke", got), ("keystroke", got)])["keystroke"]
    assert row["observations"] == 0
    assert row["not_comparable"] == 2
    assert row["undetermined"] is True


def test_a_settled_match_cannot_supply_the_observation_that_mints_an_exemption():
    # DECIDING an action and CLASSIFYING it unstable are different claims, and only the first is
    # this flag's to make: one DIFFER beside one settled-match refusal is one differing
    # comparison, so letting the refusal supply the second would grow the MEASURED exemption set
    # from a reading that never saw the live subtree.
    rows = streaming_pair_rows(
        "r100K.base.rep0",
        "r100K.treatment.rep0",
        "keystroke",
        base_tail = "T18277",
        treat_tail = "T18253",
    )
    settled = P.compare(rows[0]["parity"], rows[1]["parity"])
    assert settled[P.SETTLED_MATCH] is True
    differ = {"verdict": P.DIFFER, "reason": "", "moved": ["msg0"]}
    row = P.derive_unstable([("keystroke", differ), ("keystroke", settled)])["keystroke"]
    assert row["observations"] == 2
    assert row["differed"] == 1
    assert row[P.SETTLED_MATCH] == 1
    # NOT unstable, because only one reading compared anything, and not "decided and stable"
    # either: one differing comparison is the single flake `min_observations` exists for. On the
    # raw total this reached `cross_check` as `declared_stable_in_practice`.
    assert row["unstable"] is False
    assert row["undetermined"] is True
    both = P.derive_unstable([("keystroke", differ), ("keystroke", differ)])["keystroke"]
    assert both["unstable"] is True


def test_the_audit_decides_an_action_whose_only_leftover_was_the_live_reply(tmp_path):
    # END TO END, the case that turned the job red: two repetitions of `keystroke` inside a live
    # turn, the settled thread identical and only the in-flight tail a chunk apart.
    null = streaming_null(
        tmp_path,
        "inflight",
        [
            streaming_pair_rows(
                "r100K.base.rep0",
                "r100K.treatment.rep0",
                "keystroke",
                base_tail = "T18277",
                treat_tail = "T18253",
            ),
            streaming_pair_rows(
                "r100K.base.rep1",
                "r100K.treatment.rep1",
                "keystroke",
                base_tail = "T21004",
                treat_tail = "T20980",
            ),
        ],
    )
    rc, report = U.audit_null([null], scope = {("r100K", "keystroke")})
    assert rc == 0, report
    assert report["decided"] == [("r100K", "keystroke")]
    assert report["undecided"] == []
    assert report["differed"] == []
    assert report["decided_on_settled_thread"] == [("r100K", "keystroke")]


def test_the_audit_still_fails_when_the_capture_itself_was_blind(tmp_path):
    # THE PAIRED FAILURE: same scope and repetitions, but the arms could not place the stream at
    # all (`in_flight_unplaced`), so no settled reading sits behind the refusal and the audit
    # stays red.
    cells = []
    for rep, tails in (("rep0", ("T1", "T2")), ("rep1", ("T3", "T4"))):
        cell = streaming_pair_rows(
            f"r100K.base.{rep}",
            f"r100K.treatment.{rep}",
            "keystroke",
            base_tail = tails[0],
            treat_tail = tails[1],
        )
        for row in cell:
            row["parity"]["in_flight_unplaced"] = True
        cells.append(cell)
    null = streaming_null(tmp_path, "unplaced", cells)
    rc, report = U.audit_null([null], scope = {("r100K", "keystroke")})
    assert rc == 1
    assert report["decided"] == []
    assert report["undecided"] == [("r100K", "keystroke")]


def test_a_settled_match_can_never_grow_the_excuse_set(tmp_path):
    # THE SAFETY DIRECTION, asserted rather than argued: the observation is a NON-difference, so a
    # null made entirely of settled-match refusals derives an EMPTY measured set.
    null = streaming_null(
        tmp_path,
        "narrow",
        [
            streaming_pair_rows(
                "r100K.base.rep0",
                "r100K.treatment.rep0",
                "keystroke",
                base_tail = "A",
                treat_tail = "B",
            ),
            streaming_pair_rows(
                "r100K.base.rep1",
                "r100K.treatment.rep1",
                "keystroke",
                base_tail = "C",
                treat_tail = "D",
            ),
        ],
    )
    unstable, _derived, _checks = U.unstable_set([null])
    assert not [e for e in unstable if isinstance(e, tuple)]

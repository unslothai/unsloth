# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The audit asks whether the null was capable of an opinion, not whether it found one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402

# parents[5] is the repo root, where .github lives.
REPO_ROOT = Path(__file__).resolve().parents[5]
WORKFLOW = REPO_ROOT / ".github/workflows/studiobench-ui-parity.yml"


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


def test_a_null_control_that_decided_everything_and_found_nothing_passes(tmp_path):
    # An empty measured set because nothing differed is the best null; it must pass.
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
    # --reps 1 also yields an empty measured set, for the wrong reason; it must fail.
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
    quiet = null_run(tmp_path, "q2", two_reps("settings", "SAME", "SAME"))
    thin = null_run(tmp_path, "t2", [("r100K", "rep0", "settings", "SAME", "SAME")])
    for path in (quiet, thin):
        unstable, _d, _c = U.unstable_set([path])
        assert not [e for e in unstable if isinstance(e, tuple)], "both measure nothing"
    assert U.audit_null([quiet])[0] == 0
    assert U.audit_null([thin])[0] == 1


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


def test_an_action_that_never_captured_is_undecided_not_stable(tmp_path):
    # A blind capture must read UNDECIDED, never stable.
    null = null_run(tmp_path, "blind", two_reps("image_upload", None, None))
    rc, report = U.audit_null([null])
    assert rc == 1
    assert report["undecided"] == [("r100K", "image_upload")]


def test_an_excused_action_is_allowed_but_still_needs_a_decided_one(tmp_path):
    # Everything excused means nothing was measured.
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
    assert "--reps" in out


def test_the_parity_workflow_audits_the_null_with_the_tool(tmp_path):
    # Regexes over the tool's printed prose have been wrong twice (e.g. `r100K` vs `100K`).
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert "--audit-null" in text
    assert "--allow-undecided image_upload" in text
    assert "grep -Eo" not in text, "the guard must not key on the tool's printed prose again"


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
    # An injected element differs on every pass, so --min-reps 2 must still fail it.
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
    # Corroboration counts distinct repetitions, so a flake recorded twice cannot self-corroborate.
    a = null_run(tmp_path, "sh1", [("r100K", "rep0", "settings", "A", "B")])
    b = null_run(tmp_path, "sh2", [("r100K", "rep0", "settings", "A", "B")])
    assert U.report([a, b], "t", frozenset(), min_reps = 2) == 0


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
    # A cell that died is its own latest attempt and its difference is real; keep it.
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
    # Pooling two corpora would report DECIDED from two different threads.
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


def not_run_row(cid: str, action: str) -> dict:
    row = action_row(cid, action, "SAME")
    row["ran"] = False
    row["reason"] = "the slot opened at 33000ms and was reached at 33872ms"
    return row


def test_a_missed_slot_on_both_arms_cannot_turn_a_clean_verdict_red(tmp_path):
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


def test_the_audit_ignores_an_action_the_result_never_compared(tmp_path):
    # `copy_markdown` is out of scope; the `settings` excuse keeps the audit non-vacuous.
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
    # Exit 2 is declining to answer; exit 1 would read as a parity failure.
    null = null_run(tmp_path, "tier_null", two_reps("settings", "SAME", "SAME"), tier = "fast")
    result = null_run(tmp_path, "tier_res", two_reps("settings", "SAME", "SAME"), tier = "standard")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ui_parity", "--min-reps", "2", "--null", str(null.parent), str(result.parent)],
    )
    assert U.main() == 2
    assert "REFUSING to score" in capsys.readouterr().out


def one_sided_payload(
    tmp_path: Path,
    name: str,
    action: str,
    reps: tuple[str, ...],
    reason: str = "the control never became visible",
) -> Path:
    """Action runs on base but not treatment; filler actions keep the run above any coverage floor."""
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
    # A control that stops opening leaves no digest to differ, so one-sided runs must count.
    path = one_sided_payload(tmp_path, "regressed", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 1
    printed = capsys.readouterr().out
    assert "RAN ON ONE ARM ONLY" in printed
    assert "settings" in printed


def test_one_arms_missed_slot_in_one_repetition_is_still_only_a_warning(tmp_path, capsys):
    # Held to the same corroboration bar, since a contended runner can lose one slot once.
    path = one_sided_payload(tmp_path, "flake", "settings", ("rep0",))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0
    assert "UNCORROBORATED one-arm-only" in capsys.readouterr().out


def test_an_action_expected_to_vary_is_not_failed_for_reaching_one_arm_only(tmp_path, capsys):
    # `stop_generation` racing the stream end is already declared in UNSTABLE_ACTIONS.
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


def test_an_undecided_action_the_result_only_matched_does_not_fail_the_audit(tmp_path):
    # The result matched on `settings`, so the undecided null was never going to be asked.
    result = null_run(tmp_path, "result", two_reps("settings", "SAME", "SAME"))
    null = null_run(tmp_path, "null", [("r100K", "rep0", "settings", "SAME", "SAME")])
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert scope == set()
    rc, _ = U.audit_null([null], frozenset(), scope)
    assert rc == 0


def test_the_audit_still_fails_on_an_undecided_action_the_result_kept_differing_on(tmp_path):
    # An undecided null excuses from the declared list unmeasured, so a differing action stays in scope.
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
    result = one_sided_payload(tmp_path, "result", "settings", ("rep0", "rep1"))
    null = null_run(tmp_path, "null", [("r100K", "rep0", "settings", "SAME", "SAME")])
    scope = U.actions_needing_an_excuse([result], min_reps = 2)
    assert ("r100K", "settings") in scope
    rc, _ = U.audit_null([null], frozenset(), scope)
    assert rc == 1


def test_the_workflow_scopes_the_audit_with_the_verdicts_own_threshold(tmp_path):
    # The audit must use the same --min-reps as the verdict.
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
    # Slot misses are correlated through the runner, so corroboration cannot separate them from a
    # build difference; `slot_missed` can.
    path = missed_slot_payload(tmp_path, "slow", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 0


def test_a_control_that_cannot_open_is_still_caught_when_a_slot_is_also_missed_elsewhere(tmp_path):
    path = one_sided_payload(tmp_path, "broken", "settings", ("rep0", "rep1"))
    assert U.report([path], "t", frozenset(), min_reps = 2, min_compared = 16) == 1


def test_a_difference_at_one_rung_does_not_demand_the_null_decide_another(tmp_path):
    # Scope is keyed by (rung, action): instability is a property of the rung.
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


def test_an_interrupted_retry_does_not_inherit_the_completion_it_superseded(tmp_path):
    """An interrupted retry must not inherit the completion of the attempt it superseded."""
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


# `--mode visible` / `--mode behaviour` have no structural entries; the audit must still run.


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
    """A race seen only on the null's runner must not excuse a regression on the result's runner."""
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
    # One repetition of side A is one observation: undecided, not stable.
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
    # The declared list is a standing claim about the app; only (rung, action) entries are runner-derived.
    null = _arm_payload(tmp_path, "null", regress = None, self_race = "reasoning_toggle")
    result = _arm_payload(tmp_path, "result", regress = None, self_race = None)
    imported, _derived, _ = U.unstable_set([null])
    effective, _dropped = U.confine_to_runner(imported, *U.in_arm_repeatability([result]))
    assert U.UNSTABLE_ACTIONS <= effective


def test_the_verdict_confines_the_imported_set_when_driven_through_main(tmp_path, capsys):
    # Driven through main() so a parsed-but-unforwarded option is caught.
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
    """A failed assertion is not a digest, so a digest exemption on the action does not reach it."""
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
    # `expect_ok is None` means no claim (and all pre-field payloads); it is not False.
    path = _expect_payload(
        tmp_path, "none", "stop_generation", failed_on = None, expect_ok_value = None
    )
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
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
    """Assertion failures that blame opposite arms in different reps are uncorroborated, not a firm pair."""
    path = _reversing_expect_payload(tmp_path, "reversing", "stop_generation")
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 0
    printed = capsys.readouterr().out
    assert "UNCORROBORATED assertion failure" in printed
    assert "ASSERTION failed on one arm:  0" in printed


def test_an_assertion_that_blames_the_same_arm_twice_still_fails_the_job(tmp_path):
    # A consistently failing build must survive the direction keying.
    path = _expect_payload(tmp_path, "consistent", "stop_generation", failed_on = "treatment")
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1


def test_one_arm_only_that_swaps_arms_between_repetitions_is_not_corroborated(tmp_path, capsys):
    # A race stopping treatment in rep0 and base in rep1 says nothing about either build.
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
    """A scoped action with no rows in the null is undecided, not dropped from the audit."""
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


def test_a_broken_control_is_not_excused_because_its_digest_varies(tmp_path, capsys):
    """keystroke's exemption covers capture timing, not a treatment composer that cannot type at all."""
    path = one_sided_payload(tmp_path, "broken", "keystroke", ("rep0", "rep1"))
    assert "keystroke" in U.UNSTABLE_ACTIONS
    assert "keystroke" not in P.RACY_EXECUTION
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1
    assert "RAN ON ONE ARM ONLY" in capsys.readouterr().out


def test_an_action_with_no_not_run_path_is_never_exempt_from_one_arm_only(tmp_path):
    # `scroll_after` has no `not_run` path, so `ran: false` cannot be a race.
    path = one_sided_payload(tmp_path, "scroll", "scroll_after", ("rep0", "rep1"))
    assert "scroll_after" in U.UNSTABLE_ACTIONS
    assert "scroll_after" not in P.RACY_EXECUTION
    assert U.report([path], "t", U.UNSTABLE_ACTIONS, min_reps = 2, min_compared = 16) == 1


def test_every_racy_execution_entry_states_its_mechanism():
    # Every exemption needs a stated mechanism and markers that match specifically.
    for action, (why, markers) in P.RACY_EXECUTION.items():
        assert action in P.UNSTABLE_ACTIONS, action
        assert "not_run" in why or "not run" in why, action
        assert len(why) > 60, action
        assert markers and all(len(m) > 10 for m in markers), action


def test_a_racy_execution_action_is_not_put_in_the_audit_scope(tmp_path):
    """RACY_EXECUTION actions stay out of the audit scope, so a stream-timing race cannot fail the job."""
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
    # Pairs blaming opposite arms cannot move the verdict, so they stay out of the scope.
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
    """Exempt only the not_run reason named, not the action; a removed Stop button is a regression."""
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


# In-flight pairs are NOT_COMPARABLE; `SETTLED_MATCH` counts them as observations only once
# scaffold, overlays, message set and every settled row agree.


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
    """Pair captured while one message still streams on both arms; only its in-flight tail may differ."""

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
    # The verdict must stay NOT_COMPARABLE; only what the null counts changes.
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
    # Control: a failed capture has no settled reading and stays at zero observations.
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
    # The role sits beside the digest, so a role change is structural even in flight.
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
    # Deciding is not classifying unstable: a settled-match refusal must not grow the measured set.
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
    assert row["unstable"] is False
    assert row["undetermined"] is True
    both = P.derive_unstable([("keystroke", differ), ("keystroke", differ)])["keystroke"]
    assert both["unstable"] is True


def test_the_audit_decides_an_action_whose_only_leftover_was_the_live_reply(tmp_path):
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
    # `in_flight_unplaced` has no settled reading, so the audit stays red.
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
    # A null of only settled-match refusals derives an empty measured set.
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

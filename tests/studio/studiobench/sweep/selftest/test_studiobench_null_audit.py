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

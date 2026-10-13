# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`--declared-ui-change` excuses a rendering difference and nothing else.

A PR that says it changes the screen (#13402 renamed the composer's Code toggle to Tools) should not
fail the parity gate for doing so, but the same exit code also carries "a control could not be
operated on one build" and "an action's own assertion failed on one build". Those are the head build
behaving differently, which a label about the screen does not excuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.sweep import ui_parity as U  # noqa: E402
from tests.studio.studiobench.sweep.selftest import test_studiobench_swapped_parity as SW  # noqa: E402
from tests.studio.studiobench.sweep.selftest.test_studiobench_null_audit import (  # noqa: E402
    action_row,
    one_sided_payload,
)


def _main(tmp_path, capsys, result, null, declared) -> tuple[int, str]:
    rdir = SW._write(result, tmp_path / "parity-result")
    ndir = SW._write(null, tmp_path / "parity-null-control")
    argv = ["--min-reps", "2", "--min-compared", "16", "--null", str(ndir), str(rdir)]
    rc = U.main((["--declared-ui-change"] if declared else []) + argv)
    return rc, capsys.readouterr().out


def test_a_declared_change_reports_a_stable_rendering_difference(tmp_path, capsys):
    result, null = SW._recorded()
    SW._make_consistent(result, "thread_reopen")
    rc, out = _main(tmp_path, capsys, result, null, declared = False)
    assert rc == 1, out
    rc, out = _main(tmp_path / "declared", capsys, result, null, declared = True)
    assert rc == 0, out
    assert "DECLARED UI CHANGE" in out
    assert SW._stable(out).count("thread_reopen") == 2, SW._stable(out)


def test_no_difference_prints_no_declaration(tmp_path, capsys):
    result, null = SW._recorded()
    rc, out = _main(tmp_path, capsys, result, null, declared = True)
    assert rc == 0, out
    assert "DECLARED UI CHANGE" not in out


def test_a_declared_change_still_fails_a_control_that_runs_on_one_arm_only(tmp_path):
    path = one_sided_payload(tmp_path, "scroll", "scroll_after", ("rep0", "rep1"))
    args = ([path], "t", U.UNSTABLE_ACTIONS)
    assert U.report(*args, min_reps = 2, min_compared = 16, declared_ui_change = True) == 1


def _assertion_failed_on_treatment(tmp_path: Path, treatment_ok: bool = False) -> Path:
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(16):
                rows.append(action_row(cid, f"filler{i}", "SAME"))
            row = action_row(cid, "scroll_after", "SAME")
            row["expect_ok"] = arm == "base" or treatment_ok
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / f"expect_{treatment_ok}"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_declared_change_still_fails_an_assertion_that_failed_on_one_arm(tmp_path):
    path = _assertion_failed_on_treatment(tmp_path)
    args = ([path], "t", U.UNSTABLE_ACTIONS)
    assert U.report(*args, min_reps = 2, min_compared = 16) == 1
    assert U.report(*args, min_reps = 2, min_compared = 16, declared_ui_change = True) == 1


def test_the_same_film_with_both_assertions_passing_is_clean(tmp_path):
    path = _assertion_failed_on_treatment(tmp_path, treatment_ok = True)
    args = ([path], "t", U.UNSTABLE_ACTIONS)
    assert U.report(*args, min_reps = 2, min_compared = 16, declared_ui_change = True) == 0

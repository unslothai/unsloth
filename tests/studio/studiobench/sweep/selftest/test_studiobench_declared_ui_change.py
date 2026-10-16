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


def _every_action_differs(tmp_path: Path) -> Path:
    """A wrapper change: every action renders differently on head, in both repetitions."""
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            for i in range(17):
                rows.append(action_row(cid, f"filler{i}", "BASE" if arm == "base" else "HEAD"))
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / "wrapper"
    out.mkdir()
    path = out / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def test_a_declared_change_on_every_action_is_still_a_decided_run(tmp_path):
    path = _every_action_differs(tmp_path)
    args = ([path], "t", frozenset())
    assert U.report(*args, min_reps = 2, min_compared = 16) == 1
    assert U.report(*args, min_reps = 2, min_compared = 16, declared_ui_change = True) == 0


def _windowed_film(
    tmp_path: Path,
    name: str,
    head_visible: str,
    head_root: str = "thread",
) -> Path:
    """`settings` on a windowed treatment arm (no structural verdict), its visible region as given."""
    from tests.studio.studiobench.sweep.selftest.test_studiobench_windowed_parity import _capture

    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            cid = f"r100K.{arm}.{rep}"
            treat = arm == "treatment"
            rows.append(
                {
                    "row_type": "action",
                    "cell_id": cid,
                    "action": "settings",
                    "ran": True,
                    "parity": {
                        **_capture(6 if treat else 18, 18, "d"),
                        "root_kind": head_root if treat else "thread",
                    },
                    "visible": {
                        "visible_attempted": True,
                        "ever_visible": [1],
                        "ever_visible_count": 1,
                        "mounted_ever_visible": 1,
                        "unmounted_at_capture": 0,
                        "messages": {
                            "1": {
                                "role": "assistant",
                                "digest": head_visible if treat else "same",
                                "chars": 10,
                            }
                        },
                    },
                }
            )
            rows.append({"row_type": "cell", "cell_id": cid, "completed": True})
    out = tmp_path / name
    out.mkdir()
    (out / "payload.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
    )
    return out


def test_a_windowed_visible_difference_gets_its_evidence(tmp_path):
    from tests.studio.studiobench.sweep import parity_shots as S

    result = U.shards_of(str(_windowed_film(tmp_path, "result", "HEAD")))
    null = U.shards_of(str(_windowed_film(tmp_path, "null", "same")))
    picked = S.differing_actions(result, null, min_reps = 2)
    assert {(d["action"], d["cell"]) for d in picked} == {
        ("settings", "r100K rep0"),
        ("settings", "r100K rep1"),
    }, picked


def test_a_windowed_pair_that_matches_gets_no_picture(tmp_path):
    from tests.studio.studiobench.sweep import parity_shots as S

    result = U.shards_of(str(_windowed_film(tmp_path, "result", "same")))
    null = U.shards_of(str(_windowed_film(tmp_path, "null", "same")))
    assert S.differing_actions(result, null, min_reps = 2) == []


def test_a_declared_visible_change_on_every_pair_is_still_a_decided_run(tmp_path):
    paths = U.shards_of(str(_windowed_film(tmp_path, "result", "HEAD")))
    assert U.visible_report(paths, "t", min_reps = 2) == 1
    assert U.visible_report(paths, "t", min_reps = 2, declared_ui_change = True) == 0


def test_a_visible_pair_the_digest_calls_not_comparable_still_gets_its_evidence(tmp_path):
    """The verdict's own plan picks the pair, not the structural verdict (different root_kind here)."""
    from tests.studio.studiobench.sweep import parity_shots as S

    result = U.shards_of(str(_windowed_film(tmp_path, "result", "HEAD", head_root = "body")))
    null = U.shards_of(str(_windowed_film(tmp_path, "null", "same")))
    structural, _ = U.compare_all(result)
    assert {r["verdict"] for *_k, r in structural} != {"not_applicable"}, structural
    picked = S.differing_actions(result, null, min_reps = 2)
    assert {d["action"] for d in picked} == {"settings"}, picked

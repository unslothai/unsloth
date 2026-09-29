# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The `ab_plan` row must survive payload assembly (unsloth#9580).

It used to be filed under `header`, which is collapsed to its FIRST row, so the plan was
dropped without a word while `record_counts` still reported two header rows.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report import payload as payload_module  # noqa: E402


ROWS = [
    {"row_type": "run_meta", "run_id": "r1", "base_ref": "aaa", "platform": {"engine": "chromium"}},
    {
        "row_type": "ab_plan",
        "treatment_ref": "bbb",
        "order": ["base", "treatment"],
        "balanced": True,
    },
    {"row_type": "cell", "completed": True, "cell_id": "c1", "fields": {"instrument_level": 2}},
]


def _assemble(rows):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return payload_module.assemble_rows(str(path), validate = False)


def test_ab_plan_has_its_own_section():
    assert payload_module.ROW_TYPE_SECTIONS["ab_plan"] == "ab_plan"


def test_ab_plan_survives_assembly():
    built = _assemble(ROWS)
    assert built["ab_plan"]["treatment_ref"] == "bbb"
    assert built["ab_plan"]["order"] == ["base", "treatment"]
    assert built["ab_plan"]["balanced"] is True


def test_header_is_still_the_run_meta_row_and_counts_agree():
    built = _assemble(ROWS)
    assert built["header"]["row_type"] == "run_meta"
    assert built["record_counts"]["header"] == 1
    assert built["record_counts"]["ab_plan"] == 1
    assert built["unknown_rows"] == []


def test_absent_ab_plan_is_an_empty_mapping():
    built = _assemble([ROWS[0], ROWS[2]])
    assert built["ab_plan"] == {}


# A plan is written BEFORE its session runs, so cell rows decide ownership and the plan's own
# `balanced` predates the run. Real `make_cell_id` shapes: a placeholder id would silently take
# `executed_balance`'s cannot-tell path and exercise nothing.
def _pair(
    rung,
    rep,
    first = "base",
):
    second = "treatment" if first == "base" else "base"
    return [f"r{rung}.{first}.rep{rep}", f"r{rung}.{second}.rep{rep}"]


def _plan(
    session,
    order,
    balanced = True,
):
    return {
        "row_type": "ab_plan",
        "session_id": session,
        "treatment_ref": "bbb",
        "order": order,
        "balanced": balanced,
    }


def _cells(
    session,
    *cell_ids,
    row_type = "cell",
):
    # `Recorder.emit` stamps session_id on every row; ownership keys on it as
    # `latest_attempt_rows` does, over the same ATTEMPT_ROW_TYPES.
    return [
        {
            "row_type": row_type,
            "session_id": session,
            "completed": True,
            "cell_id": c,
            "fields": {"instrument_level": 2},
        }
        for c in cell_ids
    ]


PAIR = _pair("1K", 0)
LADDER = PAIR + _pair("1K", 1, first = "treatment")
RUNG_10K = _pair("10K", 0) + _pair("10K", 1, first = "treatment")

RESUMED_ROWS = [
    ROWS[0],
    _plan("s1", PAIR, balanced = False),
    *_cells("s1", *PAIR),
    _plan("s2", LADDER),
    *_cells("s2", *LADDER),
]


def test_a_resumed_session_does_not_lose_the_cells_it_added():
    built = _assemble(RESUMED_ROWS)
    assert built["record_counts"]["ab_plan"] == 2
    assert built["ab_plan"]["order"] == LADDER
    assert built["ab_plan"]["treatment_ref"] == "bbb"


def test_a_session_that_re_ran_the_pair_takes_over_the_balance_claim():
    assert _assemble(RESUMED_ROWS)["ab_plan"]["balanced"] is True


def test_a_resume_that_died_before_retrying_leaves_the_old_session_owning_the_pair():
    rows = [ROWS[0], _plan("s1", PAIR, balanced = False), *_cells("s1", *PAIR), _plan("s2", LADDER)]
    built = _assemble(rows)
    assert built["ab_plan"]["balanced"] is False
    assert built["ab_plan"]["order"] == LADDER


def test_an_unbalanced_session_the_resume_did_not_touch_stays_live():
    rows = [
        ROWS[0],
        _plan("s1", PAIR, balanced = False),
        *_cells("s1", *PAIR),
        _plan("s2", RUNG_10K),
        *_cells("s2", *RUNG_10K),
    ]
    assert _assemble(rows)["ab_plan"]["balanced"] is False


def test_a_retry_that_never_reached_its_cell_row_still_owns_the_cell():
    """Hard-killed mid-cell: `action` and `window` rows, no terminal `cell` row."""
    rows = [
        ROWS[0],
        _plan("s1", LADDER, balanced = False),
        *_cells("s1", *LADDER),
        _plan("s2", LADDER),
        *_cells("s2", *LADDER, row_type = "window"),
    ]
    assert _assemble(rows)["ab_plan"]["balanced"] is True


def test_a_plan_balanced_as_written_is_unbalanced_if_only_its_first_half_ran():
    """A `--reps 2` interrupted after rep 0 planned base, treatment, treatment, base and ran
    base, treatment, so base led every pair that happened and drift lands on treatment instead
    of cancelling. That pair is scored, so the plan's `True` describes an order nothing ran."""
    rows = [ROWS[0], _plan("s1", LADDER, balanced = True), *_cells("s1", *PAIR)]
    assert _assemble(rows)["ab_plan"]["balanced"] is False
    whole = [ROWS[0], _plan("s1", LADDER, balanced = True), *_cells("s1", *LADDER)]
    assert _assemble(whole)["ab_plan"]["balanced"] is True


def test_the_derived_verdict_agrees_with_the_producers_on_a_complete_session():
    for order, planned in ((PAIR, False), (LADDER, True), (RUNG_10K, True)):
        rows = [ROWS[0], _plan("s1", order, balanced = planned), *_cells("s1", *order)]
        assert _assemble(rows)["ab_plan"]["balanced"] is planned, order


def test_an_id_that_is_not_a_cell_id_keeps_the_plans_own_word():
    rows = [ROWS[0], _plan("s1", ["alpha", "beta"], balanced = True), *_cells("s1", "alpha")]
    assert _assemble(rows)["ab_plan"]["balanced"] is True


def test_one_arm_running_alone_is_never_balanced():
    """With one arm every first-count is trivially equal. `order_is_balanced` calls this the
    one answer it exists to prevent."""
    base_only = [cell for cell in LADDER if ".base." in cell]
    rows = [ROWS[0], _plan("s1", LADDER, balanced = True), *_cells("s1", *base_only)]
    assert _assemble(rows)["ab_plan"]["balanced"] is False


def test_the_merged_plan_does_not_claim_one_sessions_identity():
    built = _assemble(RESUMED_ROWS)["ab_plan"]
    assert "session_id" not in built, built
    assert "ts_ms" not in built, built
    assert built["sessions"] == ["s1", "s2"]
    assert built["treatment_ref"] == "bbb"

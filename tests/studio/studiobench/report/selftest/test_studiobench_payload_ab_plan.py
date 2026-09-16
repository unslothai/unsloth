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


# `--resume` emits its own plan, so `[0]` would name only the first session's cells. A plan is
# written BEFORE its session runs, so the cell rows, not the plan, say which session owns a cell.
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
    # `Recorder.emit` stamps session_id on every row, and ownership is keyed on it exactly as
    # `latest_attempt_rows` keys it, over the same ATTEMPT_ROW_TYPES.
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


PAIR = ["base", "treatment"]
LADDER = ["base", "treatment", "base_10k", "treatment_10k"]

RESUMED_ROWS = [
    ROWS[0],
    _plan("s1", PAIR),
    *_cells("s1", *PAIR),
    _plan("s2", LADDER),
    *_cells("s2", *LADDER),
]


def test_a_resumed_session_does_not_lose_the_cells_it_added():
    built = _assemble(RESUMED_ROWS)
    assert built["record_counts"]["ab_plan"] == 2
    # First-seen order, re-declared ids not doubled. `[0]` gives ["base", "treatment"].
    assert built["ab_plan"]["order"] == LADDER
    assert built["ab_plan"]["treatment_ref"] == "bbb"


def test_a_session_that_re_ran_the_pair_takes_over_the_balance_claim():
    """`skippable_cells` re-runs every pair, and `latest_attempt_rows` keeps the last attempt,
    so the earlier session owns nothing and its imbalance is not in the scored cells."""
    rows = [
        ROWS[0],
        _plan("s1", PAIR, balanced = False),
        *_cells("s1", *PAIR),
        _plan("s2", LADDER),
        *_cells("s2", *LADDER),
    ]
    assert _assemble(rows)["ab_plan"]["balanced"] is True


def test_a_resume_that_died_before_retrying_leaves_the_old_session_owning_the_pair():
    """The plan is emitted before the session runs anything, so a declared id is a request.

    A `--resume --reps 2` that crashes before the retry writes its plan and no cell, and the
    unbalanced session that DID run the pair is still what `latest_attempt_rows` keeps. Reading
    liveness off plan membership reports a balance nothing measured.
    """
    rows = [ROWS[0], _plan("s1", PAIR, balanced = False), *_cells("s1", *PAIR), _plan("s2", LADDER)]
    assert _assemble(rows)["ab_plan"]["balanced"] is False
    # The order is still the union, so the rungs the resume asked for are not lost.
    assert _assemble(rows)["ab_plan"]["order"] == LADDER


def test_an_unbalanced_session_the_resume_did_not_touch_stays_live():
    """Both sessions own cells, so one unbalanced among them is an unbalanced experiment."""
    rows = [
        ROWS[0],
        _plan("s1", PAIR, balanced = False),
        *_cells("s1", *PAIR),
        _plan("s2", ["base_10k", "treatment_10k"]),
        *_cells("s2", "base_10k", "treatment_10k"),
    ]
    assert _assemble(rows)["ab_plan"]["balanced"] is False


def test_a_retry_that_never_reached_its_cell_row_still_owns_the_cell():
    """Why ownership reads ATTEMPT_ROW_TYPES rather than `cell` rows alone.

    `latest_attempt_rows` moved off cell rows for exactly this: an attempt hard-killed mid-cell
    writes `action` and `window` rows and never its terminal `cell` row, so keying on that alone
    hands the cell back to the older attempt. Ownership here has to agree with it, or the report
    scores one session's cells while describing another session's balance.
    """
    rows = [
        ROWS[0],
        _plan("s1", PAIR, balanced = False),
        *_cells("s1", *PAIR),
        _plan("s2", PAIR),
        *_cells("s2", *PAIR, row_type = "window"),
    ]
    assert _assemble(rows)["ab_plan"]["balanced"] is True

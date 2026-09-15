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

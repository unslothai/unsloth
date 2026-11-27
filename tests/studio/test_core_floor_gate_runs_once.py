# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Core runs the python floor check on one leg only, so that leg must exist in the matrix:
a mistyped id would schedule the HARD GATE on zero legs and leave every Core run green."""

import re
from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "consolidated-tests-ci.yml"
)
STEP = "python floor compatibility (HARD GATE)"


def test_the_floor_gate_selects_exactly_one_existing_leg():
    job = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["consolidated"]
    ids = [combo["id"] for combo in job["strategy"]["matrix"]["combo"]]
    assert not job["strategy"]["matrix"].get("exclude"), "an exclude could drop the floor leg"
    (step,) = [s for s in job["steps"] if s.get("name") == STEP]
    condition = step["if"]
    assert condition.startswith("${{ !cancelled() &&"), condition
    selected = re.findall(r"matrix\.combo\.id == '([^']+)'", condition)
    assert len(selected) == 1 and selected[0] in ids, (selected, ids)
    assert "tests/test_python39_compatibility.py" in step["run"]

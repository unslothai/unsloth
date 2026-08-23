# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A section reported as one record may hold one row type, or the rest vanish.

`assemble_rows` routes rows by `ROW_TYPE_SECTIONS` and then reports two of those sections as a
single record rather than a list. `ab_plan` was filed under `header` alongside `run_meta`, so the
mapping named a destination and the collapse dropped the row on arrival: `record_counts` said
`header: 2` while the payload carried one, and nothing landed in `unknown_rows` to show for it.

The loss was invisible because both readers scan the raw row stream instead, but the field comment
invites exactly the read that would have found it -- `ab_plan` is how a reader establishes the run
order was balanced, and a reader who trusted it got `{}`, which is indistinguishable from "the
order was not recorded".

So `ab_plan` gets its own section, and the collapse names the row type it keeps instead of taking
whatever is first. This file holds the rule to that: any row type added to a collapsed section
later fails here rather than disappearing at assembly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report.payload import (  # noqa: E402
    COLLAPSED_SECTIONS,
    ROW_TYPE_SECTIONS,
    assemble_rows,
)
from studiobench.runtime.types import ROW_TYPES  # noqa: E402


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding = "utf-8")
    return path


def _session(session_id: str, *, treatment: str, balanced: bool, order: list[str]) -> list[dict]:
    """One session's rows in emission order: `run_meta`, then `ab_plan` before the first cell."""

    return [
        {
            "row_type": "run_meta",
            "session_id": session_id,
            "tier": "standard",
            "studio_ref": "base-sha",
            "platform": {"engine": "chromium"},
        },
        {
            "row_type": "ab_plan",
            "session_id": session_id,
            "base_ref": "base-sha",
            "treatment_ref": treatment,
            "treatment_url": "",
            "treatment_commit": "",
            "balanced": balanced,
            "order": order,
        },
        *(
            {"row_type": "cell", "session_id": session_id, "cell_id": cell, "completed": True}
            for cell in order
        ),
    ]


def test_a_collapsed_section_holds_exactly_one_row_type(tmp_path):
    """The guard. Filing a second row type under `header` is how `ab_plan` was lost."""

    for section, row_type in COLLAPSED_SECTIONS.items():
        filed = sorted(name for name, dest in ROW_TYPE_SECTIONS.items() if dest == section)
        assert filed == [row_type], (
            f"section {section!r} is reported as a single {row_type!r} record, but "
            f"{filed} map to it. Every row type but {row_type!r} would be routed there and then "
            f"discarded with no unknown_rows entry. Give it its own section."
        )


def test_every_row_type_reaches_a_section():
    """A row type with no mapping is not lost -- it lands in `unknown_rows` -- but it is not
    reported either, so the mapping is meant to stay total."""

    assert sorted(set(ROW_TYPES) - set(ROW_TYPE_SECTIONS)) == []


def test_the_ab_plan_row_survives_assembly(tmp_path):
    """The reported defect: the row was routed to `header` and then thrown away."""

    order = ["r1K.base.rep0", "r1K.treatment.rep0"]
    path = _write(tmp_path, _session("s1", treatment = "fix-sha", balanced = True, order = order))
    payload = assemble_rows(path)

    assert payload["ab_plan"]["treatment_ref"] == "fix-sha"
    assert payload["ab_plan"]["balanced"] is True
    assert payload["ab_plan"]["order"] == order
    # The identity fields are untouched by the move.
    assert payload["header"]["row_type"] == "run_meta"
    assert payload["header"]["studio_ref"] == "base-sha"
    assert payload["unknown_rows"] == []


def test_record_counts_match_what_the_payload_carries(tmp_path):
    """`header: 2` against one carried row was the visible symptom of the drop."""

    path = _write(
        tmp_path, _session("s1", treatment = "fix-sha", balanced = False, order = ["r1K.base.rep0"])
    )
    payload = assemble_rows(path)

    for section in COLLAPSED_SECTIONS:
        assert (
            payload["record_counts"].get(section) == 1
        ), f"{section} counts {payload['record_counts'].get(section)} rows but is reported as one"
    # `balanced: False` has to reach a reader: it is the only record that linear drift did not
    # cancel, and it is written precisely when the run order was odd.
    assert payload["ab_plan"]["balanced"] is False


def test_a_collapsed_section_describes_the_first_session(tmp_path):
    """`Recorder` appends, so one payload can hold several runs. `header` takes the first
    `run_meta`; `ab_plan` has to take the first `ab_plan` or the two describe different runs."""

    path = _write(
        tmp_path,
        _session("s1", treatment = "first-sha", balanced = True, order = ["r1K.base.rep0"])
        + _session("s2", treatment = "second-sha", balanced = False, order = ["r10K.base.rep0"]),
    )
    payload = assemble_rows(path)

    assert payload["header"]["session_id"] == "s1"
    assert payload["ab_plan"]["session_id"] == "s1"
    assert payload["ab_plan"]["treatment_ref"] == "first-sha"


def test_completeness_reads_the_run_meta_row_not_the_section(tmp_path):
    """`complete` used to test that the `header` SECTION was non-empty, which `ab_plan` also
    satisfied. A run that recorded its A/B order and died before its identity was reported as
    having recorded its identity."""

    path = _write(
        tmp_path,
        [
            {
                "row_type": "ab_plan",
                "session_id": "s1",
                "base_ref": "base-sha",
                "treatment_ref": "fix-sha",
                "balanced": True,
                "order": ["r1K.base.rep0"],
            },
            {"row_type": "cell", "session_id": "s1", "cell_id": "r1K.base.rep0", "completed": True},
        ],
    )
    payload = assemble_rows(path)

    assert payload["header"] == {}
    assert payload["complete"] is False
    # The plan itself is still reported: what was measured is not thrown away for what was not.
    assert payload["ab_plan"]["treatment_ref"] == "fix-sha"


def test_a_run_that_is_not_an_ab_reports_no_plan(tmp_path):
    """A single-build run writes no `ab_plan`. The key is still present and still `{}`, so a
    reader can tell "not an A/B" from a missing key."""

    path = _write(
        tmp_path,
        [
            {"row_type": "run_meta", "session_id": "s1", "tier": "quick", "studio_ref": "main"},
            {"row_type": "cell", "session_id": "s1", "cell_id": "r1K.A0.rep0", "completed": True},
        ],
    )
    payload = assemble_rows(path)

    assert payload["ab_plan"] == {}
    assert payload["header"]["studio_ref"] == "main"
    assert "ab_plan" not in payload["record_counts"]

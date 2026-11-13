# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resumed A/B that measured nothing must not overwrite the ab.md table the measuring run wrote."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import _render_ab, _resume_set, archive_payload  # noqa: E402
from studiobench.runtime.types import Paths  # noqa: E402

MEASURED = "s-measured"
RESUMED = "s-resumed"

SIDES = [
    {"label": "base", "ref": "main", "base_url": "http://127.0.0.1:5401", "owns": True},
    {"label": "treatment", "ref": "fix", "base_url": "http://127.0.0.1:5402", "owns": True},
]


def _cell(cell_id, arm, session):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "session_id": session,
        "target_tokens": 10_000,
        "completed": True,
        "cell": {"arm": arm, "rep": 0},
    }


def _keystroke(cell_id, session, p95):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "session_id": session,
        "action": "keystroke",
        "ran": True,
        "expect_ok": True,
        "timings": {"p95_ms": p95},
    }


def _payload(out: Path, session: str) -> Paths:
    paths = Paths.under(out)
    rows = [
        _cell("r10K.base.rep0", "base", session),
        _keystroke("r10K.base.rep0", session, 100.0),
        _cell("r10K.treatment.rep0", "treatment", session),
        _keystroke("r10K.treatment.rep0", session, 50.0),
    ]
    paths.payload_jsonl.parent.mkdir(parents = True, exist_ok = True)
    paths.payload_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return paths


def test_a_fully_resumed_ab_keeps_the_table_the_measured_run_wrote(tmp_path):
    paths = _payload(tmp_path / "run", MEASURED)

    _render_ab(paths, SIDES, MEASURED, "c0ffee")
    measured = (paths.out / "ab.md").read_text(encoding = "utf-8")
    assert "NO READING" not in measured
    assert "keystroke_p95_ms" in measured

    _render_ab(paths, SIDES, RESUMED, "c0ffee")
    assert (paths.out / "ab.md").read_text(encoding = "utf-8") == measured


def test_a_run_that_measured_still_rewrites_the_table(tmp_path):
    """The control: refusing to overwrite must not turn into refusing to report."""

    paths = _payload(tmp_path / "run", MEASURED)
    (paths.out / "ab.md").write_text("stale table from an older run\n", encoding = "utf-8")

    _render_ab(paths, SIDES, MEASURED, "c0ffee")
    text = (paths.out / "ab.md").read_text(encoding = "utf-8")
    assert "stale table" not in text
    assert "NO READING" not in text
    assert "keystroke_p95_ms" in text


def test_a_first_run_with_no_readings_still_gets_a_table(tmp_path):
    """The other control: with no prior report there is nothing to preserve, so render."""

    paths = _payload(tmp_path / "run", MEASURED)
    _render_ab(paths, SIDES, RESUMED, "c0ffee")
    assert "NO READING" in (paths.out / "ab.md").read_text(encoding = "utf-8")


def _probe_rows(session: str, probe: str) -> list:
    """What a PROBE run records: the metadata field and the failed gate, then its cells."""
    return [
        {"row_type": "run_meta", "session_id": session, "probe_init_script": probe},
        {
            "row_type": "gate",
            "session_id": session,
            "name": "probe_free",
            "passed": False,
            "detail": {"probe_init_script": probe},
        },
        _cell("r10K.base.rep0", "base", session),
        _keystroke("r10K.base.rep0", session, 100.0),
        _cell("r10K.treatment.rep0", "treatment", session),
        _keystroke("r10K.treatment.rep0", session, 50.0),
    ]


def test_a_resumed_probe_replaces_the_clean_table_it_inherited(tmp_path):
    """A resumed probe must replace the clean ab.md it inherited rather than keep it over a bad payload."""

    paths = _payload(tmp_path / "run", MEASURED)
    _render_ab(paths, SIDES, MEASURED, "c0ffee")
    clean = (paths.out / "ab.md").read_text(encoding = "utf-8")
    assert "keystroke_p95_ms" in clean

    archive_payload(paths, log = lambda _msg: None)
    paths.payload_jsonl.write_text(
        "".join(json.dumps(r) + "\n" for r in _probe_rows("s-probe", "potency.js")),
        encoding = "utf-8",
    )
    assert (paths.out / "ab.md").read_text(encoding = "utf-8") == clean

    assert _resume_set(paths) == {"r10K.base.rep0", "r10K.treatment.rep0"}
    _render_ab(paths, SIDES, "s-resumed-probe", "c0ffee", planned = [])

    text = (paths.out / "ab.md").read_text(encoding = "utf-8")
    assert "NO A/B TABLE" in text
    assert "potency.js" in text
    assert "keystroke_p95_ms" not in text


def test_a_clean_fully_resumed_ab_still_keeps_its_table(tmp_path):
    """The control the fix must not break: no probe, no cells, table kept."""

    paths = _payload(tmp_path / "run", MEASURED)
    _render_ab(paths, SIDES, MEASURED, "c0ffee")
    measured = (paths.out / "ab.md").read_text(encoding = "utf-8")

    _render_ab(paths, SIDES, RESUMED, "c0ffee", planned = [])
    assert (paths.out / "ab.md").read_text(encoding = "utf-8") == measured


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

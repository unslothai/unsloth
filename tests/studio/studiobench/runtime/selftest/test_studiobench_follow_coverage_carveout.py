# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stream-coverage shortfall is set by the schedule, not measured; pinning failures stay fatal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from studiobench.runtime.ab import failed_invalidating_gates  # noqa: E402
from studiobench.runtime.session import (  # noqa: E402
    FOLLOW_MIN_STREAM_COVERAGE,
    follow_verdict,
)
from studiobench.sweep.ui_parity import incomplete_cells  # noqa: E402

OBSERVED_COVERAGE = 0.481


def _records(detail: dict) -> list[dict]:
    return [
        {"row_type": "cell", "cell_id": "c1", "session_id": "s1", "completed": True},
        {
            "row_type": "gate",
            "name": "follows_the_stream",
            "passed": False,
            "cell_id": "c1",
            "session_id": "s1",
            "detail": detail,
        },
    ]


def _refuses(tmp_path: Path, detail: dict) -> tuple[bool, bool]:
    """(does the A/B table drop this cell, does the UI parity job refuse its pair)."""

    records = _records(detail)
    tmp_path.mkdir(parents = True, exist_ok = True)
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding = "utf-8")
    return bool(failed_invalidating_gates(records)), bool(incomplete_cells([path]))


def _coverage_short(coverage: float) -> dict:
    """What `session.py` writes when coverage is the ONLY thing that fell short."""

    return {
        "follow_attempted": True,
        "pinned_fraction": 1.0,
        "attached_fraction_of_stream": coverage,
        "ever_fell_behind": False,
        "stream_coverage": coverage,
        "stream_coverage_floor": 0.50,
        "stream_coverage_unmeasured": True,
    }


def test_coverage_only_shortfall_is_carved_out(tmp_path) -> None:
    """0.481 is the film, not the build. Neither admission list may drop the cell for it."""

    dropped, refused = _refuses(tmp_path, _coverage_short(OBSERVED_COVERAGE))
    assert not dropped, "a coverage shortfall must not void the A/B table"
    assert not refused, "a coverage shortfall must not refuse the UI parity pair"


def test_a_sliver_is_also_not_measured_rather_than_failed(tmp_path) -> None:
    """A 13% sliver is also not measured: the cell is admitted while its gate row still reads failed."""

    dropped, refused = _refuses(tmp_path, _coverage_short(0.13))
    assert not dropped
    assert not refused


def test_a_bad_pinned_fraction_still_bites(tmp_path) -> None:
    """Healthy coverage, genuinely bad pinning. Both lists must still drop the cell."""

    detail = {
        "follow_attempted": True,
        "pinned_fraction": 0.30,
        "attached_fraction_of_stream": 0.90,
        "ever_fell_behind": False,
        "stream_coverage": 0.90,
        "stream_coverage_unmeasured": False,
        "reason": "pinned for 30% of the attached samples",
    }
    dropped, refused = _refuses(tmp_path, detail)
    assert dropped, "a thread that stopped following must still void its cell"
    assert refused, "a thread that stopped following must still refuse its pair"


def test_falling_behind_still_bites(tmp_path) -> None:
    """`ever_fell_behind` is a property of the arm and stays fatal on its own."""

    detail = {
        "follow_attempted": True,
        "pinned_fraction": 1.0,
        "attached_fraction_of_stream": 0.90,
        "ever_fell_behind": True,
        "stream_coverage": 0.90,
        "stream_coverage_unmeasured": False,
        "reason": "the thread fell behind the stream",
    }
    dropped, refused = _refuses(tmp_path, detail)
    assert dropped
    assert refused


def test_low_coverage_does_not_launder_a_real_pinning_failure(tmp_path) -> None:
    """The waiver is set only when coverage is the sole shortfall, never over a pinning failure."""

    detail = {
        "follow_attempted": True,
        "pinned_fraction": 0.30,
        "attached_fraction_of_stream": 0.20,
        "ever_fell_behind": False,
        "stream_coverage": 0.20,
        "stream_coverage_unmeasured": False,
        "reason": "pinned for 30% of the attached samples",
    }
    dropped, refused = _refuses(tmp_path, detail)
    assert dropped
    assert refused


# Both admission lists must agree on every case, or the scorers drift on what invalidates a cell.
_AGREEMENT_CASES: list[tuple[str, dict, bool]] = [
    ("coverage-only shortfall", _coverage_short(OBSERVED_COVERAGE), False),
    ("sliver", _coverage_short(0.13), False),
    (
        "bad pinned",
        {
            "follow_attempted": True,
            "pinned_fraction": 0.30,
            "attached_fraction_of_stream": 0.90,
            "ever_fell_behind": False,
            "stream_coverage_unmeasured": False,
            "reason": "pinned low",
        },
        True,
    ),
    (
        "fell behind",
        {
            "follow_attempted": True,
            "pinned_fraction": 1.0,
            "attached_fraction_of_stream": 0.90,
            "ever_fell_behind": True,
            "stream_coverage_unmeasured": False,
            "reason": "fell behind",
        },
        True,
    ),
    (
        "bad pinned AND low coverage",
        {
            "follow_attempted": True,
            "pinned_fraction": 0.30,
            "attached_fraction_of_stream": 0.20,
            "ever_fell_behind": False,
            "stream_coverage_unmeasured": False,
            "reason": "pinned low",
        },
        True,
    ),
    (
        "fell behind AND low coverage",
        {
            "follow_attempted": True,
            "pinned_fraction": 1.0,
            "attached_fraction_of_stream": 0.20,
            "ever_fell_behind": True,
            "stream_coverage_unmeasured": False,
            "reason": "fell behind",
        },
        True,
    ),
    (
        "no pinned reading, sampler present",
        {
            "follow_attempted": True,
            "pinned_fraction": None,
            "attached_fraction_of_stream": 0.90,
            "ever_fell_behind": False,
            "stream_coverage_unmeasured": False,
            "reason": "no pinned reading",
        },
        True,
    ),
    ("absent instrument", {"follow_attempted": False, "reason": "sampler is not installed"}, False),
    # A missing thread viewport is the arm's failure, not an absent instrument, so it stays fatal.
    ("no thread viewport", {"probe_attempted": False, "reason": "no thread viewport"}, True),
]


def test_both_admission_lists_agree_on_every_shape(tmp_path) -> None:
    """The property the copied predicate threatened, pinned directly."""

    disagreed: list[str] = []
    wrong: list[str] = []
    for index, (label, detail, want_fatal) in enumerate(_AGREEMENT_CASES):
        dropped, refused = _refuses(tmp_path / f"case{index}", detail)
        if dropped != refused:
            disagreed.append(f"{label}: ab={dropped} ui_parity={refused}")
        elif dropped != want_fatal:
            wrong.append(f"{label}: got {dropped}, wanted {want_fatal}")
    assert not disagreed, "the two admission lists disagree: " + "; ".join(disagreed)
    assert not wrong, "wrong verdict: " + "; ".join(wrong)


# Tests the writer: consumer tests cannot catch a follow_verdict that waives every low reading.


def _sampled(
    pinned,
    coverage,
    fell_behind = False,
    reattachments = 2,
) -> dict:
    """Mirrors scene/dom.js read(); zero reattachments means the build never re-pinned."""

    return {
        "follow_attempted": True,
        "pinned_fraction": pinned,
        "attached_fraction_of_stream": coverage,
        "ever_fell_behind": fell_behind,
        "reattachments": reattachments,
    }


def test_writer_waives_only_a_lone_coverage_shortfall() -> None:
    """The waiver is derived by the writer, and requires that nothing else fell short."""

    passed, rec = follow_verdict(_sampled(1.0, OBSERVED_COVERAGE))
    assert passed is False, "the gate row must still read as failed"
    assert rec["stream_coverage_unmeasured"] is True

    _, rec = follow_verdict(_sampled(0.30, 0.20))
    assert rec["stream_coverage_unmeasured"] is False

    _, rec = follow_verdict(_sampled(1.0, 0.20, fell_behind = True))
    assert rec["stream_coverage_unmeasured"] is False

    _, rec = follow_verdict(_sampled(None, 0.20))
    assert rec["stream_coverage_unmeasured"] is False


def test_an_arm_that_never_reattached_is_not_waived(tmp_path) -> None:
    """An arm that never re-pins stays detached, so its coverage shortfall is the build's and not waived."""

    _, rec = follow_verdict(_sampled(1.0, 0.10, reattachments = 0))
    assert rec["stream_coverage_unmeasured"] is False

    detail = _sampled(1.0, 0.10, reattachments = 0)
    detail.update(rec)
    dropped, refused = _refuses(tmp_path, detail)
    assert dropped, "a build that never came back must still void its cell"
    assert refused, "a build that never came back must still refuse its pair"


def test_the_schedules_own_shortfall_is_still_waived(tmp_path) -> None:
    """The control: same coverage story, but the arm DID come back, so 0.481 is the film."""

    _, rec = follow_verdict(_sampled(1.0, OBSERVED_COVERAGE, reattachments = 2))
    assert rec["stream_coverage_unmeasured"] is True


def test_writer_records_the_coverage_as_a_number_either_way() -> None:
    """Coverage must be recorded as a number whatever the verdict, not only as a pass or a fail."""

    for coverage in (OBSERVED_COVERAGE, 0.13, 0.99):
        _, rec = follow_verdict(_sampled(1.0, coverage))
        assert rec["stream_coverage"] == coverage
        assert rec["stream_coverage_floor"] == FOLLOW_MIN_STREAM_COVERAGE


def test_writer_passes_the_gate_when_the_film_does_cover_the_stream() -> None:
    """The carve-out must not make the gate unfailable OR unpassable."""

    passed, rec = follow_verdict(_sampled(1.0, 0.90))
    assert passed is True
    assert rec["stream_coverage_unmeasured"] is False
    assert "stream_coverage_reason" not in rec


def test_writer_treats_an_absent_coverage_reading_as_short() -> None:
    """None is not a high number. A sampler that returned no coverage has not shown the film."""

    passed, rec = follow_verdict(_sampled(1.0, None))
    assert passed is False
    assert rec["stream_coverage"] is None
    assert "unknown share" in rec["stream_coverage_reason"]


def test_not_measured_stays_distinguishable_from_a_pass(tmp_path) -> None:
    """Not measured stays distinct from a pass: the gate row reads passed: False, coverage is a number."""

    detail = _coverage_short(OBSERVED_COVERAGE)
    records = _records(detail)
    gate = [r for r in records if r["row_type"] == "gate"][0]
    assert gate["passed"] is False, "the carve-out must not manufacture a passing gate"
    assert gate["detail"]["stream_coverage"] == OBSERVED_COVERAGE
    assert isinstance(gate["detail"]["stream_coverage"], float)

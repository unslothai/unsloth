# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A STREAM-COVERAGE SHORTFALL IS NOT MEASURED; A PINNING FAILURE IS STILL FATAL.

`attached_fraction_of_stream` is set by the SCENE SCHEDULE, not by the build under test. The
shipped film scrolls away twice inside an ~18s opening stream and the app then correctly declines
to yank the reader back, so roughly half the streaming time is detached BY CONSTRUCTION. Measured
over 32 cells: 0.481 +/- 0.009, range 0.4625 to 0.5063 -- which put `FOLLOW_MIN_STREAM_COVERAGE`
of 0.50 above the mean of the quantity it was gating and turned the verdict into a coin flip.

The observed result was 32 of 32 pairs refused with `cell r100K.base.rep0 FAILED its stream-follow
gate`, exit 3, TOO LITTLE COMPARED -- and the NULL CONTROL, the same commit on both arms, refused
identically. A gate that fails its own null is not measuring the thing it names. The prose in
`scene/selftest/test_studiobench_follow_coverage_live.py` had already recorded the same shape at
the earlier 0.07-0.15 coverage ("failed every 100K cell of every run -- two copies of the shipped
build included") without the gate itself being changed.

Both arms run the same film, so the shortfall is symmetric and cancels in any comparison drawn
from these cells. It can only void a run, never discriminate between builds. Raising the constant
was the other option and is the same trap one turn later: it would need re-deriving every time the
film's scroll schedule moves.

So coverage is carved out of the two admission lists the way an absent instrument already is, and
BOTH REMAINING CONJUNCTS STAY FATAL. `pinned_fraction` and `ever_fell_behind` describe how the arm
behaved while it WAS attached and can genuinely differ between builds. This test exists because a
relaxation that cannot be shown to still bite is worthless.
"""

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

#: The coverage actually observed on the run this carve-out was derived from, not a round number.
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
    """The 0.13 sampler-stranding case that motivated the original conjunct.

    It reaches the same destination by design. A verdict resting on 13% of the stream is NOT a
    reading, and "we have no reading" is what NOT MEASURED means -- so the cell is admitted to a
    comparison whose confounds are common to both arms, while the gate row stays `passed: False`
    so nobody can quote the follow verdict itself. The protection the conjunct was added for is
    that the gate refuses to CLAIM a pass, and that is still true; what it may no longer do is
    delete the cell's other, correctly measured columns.
    """

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
    """The regression that would make the carve-out worthless.

    A cell that failed BOTH must not ride out on the coverage allowance. `session.py` only sets
    `stream_coverage_unmeasured` when coverage was the sole shortfall, and this pins that: were it
    set whenever coverage was low, every genuine follow failure on a large rung -- where coverage
    is low by construction -- would be waived, which is the defect this whole carve-out exists to
    avoid re-introducing in the opposite direction.
    """

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


#: Every shape a failed `follows_the_stream` row can take, and whether it must still be fatal.
#: The two admission lists have to agree on all of them: `INVALIDATING_CELL_GATES` was centralised
#: so the scorers could not disagree about what invalidates a cell, and a predicate copied into
#: both consumers reintroduces exactly that drift one level down, where it is harder to see
#: because each copy reads correctly on its own.
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
    # The narrowing that must survive the refactor: `probe_attempted: False` has two producers and
    # only one is an absent instrument. A missing thread viewport is the ARM missing the surface
    # under test, and waiving it once let a real failure ride the instrument allowance.
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


# ---------------------------------------------------------------------------------------
# the WRITER. Everything above drives the two consumers with a hand-built gate detail, which
# cannot see a defect in the thing that PRODUCES that detail: a `follow_verdict` that set
# `stream_coverage_unmeasured` on every low reading would waive real follow failures and every
# consumer test would still pass. These drive the real writer.
# ---------------------------------------------------------------------------------------


def _sampled(
    pinned,
    coverage,
    fell_behind = False,
    reattachments = 2,
) -> dict:
    """What `scene/dom.js::read()` returns. `reattachments` defaults to the film's own two.

    It is not decoration: the waiver's premise is that the detachment was the SCHEDULE's, and the
    sampler's proof of that is the arm coming back. A cell with 0 of them was detached for the rest
    of the film because the BUILD would not re-pin.
    """

    return {
        "follow_attempted": True,
        "pinned_fraction": pinned,
        "attached_fraction_of_stream": coverage,
        "ever_fell_behind": fell_behind,
        "reattachments": reattachments,
    }


def test_writer_waives_only_a_lone_coverage_shortfall() -> None:
    """`stream_coverage_unmeasured` requires that NOTHING ELSE fell short.

    This is the conjunction that stops the carve-out laundering a real failure, and it is only
    visible here: a consumer test supplies the flag rather than deriving it.
    """

    passed, rec = follow_verdict(_sampled(1.0, OBSERVED_COVERAGE))
    assert passed is False, "the gate row must still read as failed"
    assert rec["stream_coverage_unmeasured"] is True

    # a bad pinned fraction alongside the same shortfall must NOT be waived
    _, rec = follow_verdict(_sampled(0.30, 0.20))
    assert rec["stream_coverage_unmeasured"] is False

    # nor may falling behind
    _, rec = follow_verdict(_sampled(1.0, 0.20, fell_behind = True))
    assert rec["stream_coverage_unmeasured"] is False

    # nor an absent pinned reading while the sampler was present
    _, rec = follow_verdict(_sampled(None, 0.20))
    assert rec["stream_coverage_unmeasured"] is False


def test_an_arm_that_never_reattached_is_not_waived(tmp_path) -> None:
    """The half of the shortfall that IS the build, and the shape it wears.

    `scene/dom.js` clears `detached` only when a run that began after the harness's gesture is
    observed at the bottom, so an arm that stops re-pinning on a new turn stays detached for the
    rest of the film. The detached branch returns before the pinned and fell-behind accounting, so
    the opening stream's perfect `pinned_fraction` and `ever_fell_behind: False` survive while
    coverage collapses -- the exact shape of a lone coverage shortfall, produced by the one failure
    the gate exists to catch. Waived, the cell keeps its artificially cheap timings (an unmounted
    reply costs nothing to paint) and is compared against a healthy partner.
    """

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
    """The quantity has to be legible as a quantity, whatever the verdict.

    It sat undetected precisely because it was only ever visible as a pass or a fail.
    """

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
    """Carving the cell out of the admission lists must not rewrite the gate row.

    "Not measured" and "compared and equal" are different claims and the payload has to keep them
    apart: the gate row still reads `passed: False`, and the coverage is still there AS A NUMBER
    rather than as the pass/fail it used to be visible only as. That is the whole reason 0.481 sat
    undetected long enough to void a campaign.
    """

    detail = _coverage_short(OBSERVED_COVERAGE)
    records = _records(detail)
    gate = [r for r in records if r["row_type"] == "gate"][0]
    assert gate["passed"] is False, "the carve-out must not manufacture a passing gate"
    assert gate["detail"]["stream_coverage"] == OBSERVED_COVERAGE
    assert isinstance(gate["detail"]["stream_coverage"], float)

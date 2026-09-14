# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End to end over synthetic data: plan a batch, judge it, render it, read it back.

The unit tests check each rule in isolation. This file checks that the rules still hold once
they are composed, which is where they usually stop holding: a `VOIDED` arm that gets rendered
anyway because the renderer walks a different list, a not-quotable batch whose numbers appear in
a summary assembled by a different function, a payload that validates on its own and fails once
the harness layer's rows are in it.

It also runs the two renderers for real and asserts on their OUTPUT, because a renderer that
raises is a bug anyone finds and a renderer that prints the wrong thing is a bug nobody does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.arms import (  # noqa: E402
    ArmStatus,
    BatchPlanError,
    Invariance,
    PlannedCell,
    PotencyCounter,
    arms_key,
    assert_equal_scene_duration,
    classify_recovery,
    discover_armpack,
    evaluate_batch,
    fit_dose_response,
    judge_batch,
    missing_cells,
    plan_batch,
)
from studiobench.arms.dose import DosePoint  # noqa: E402
from studiobench.arms.manifest import Arm, ArmOutcome  # noqa: E402
from studiobench.report import (  # noqa: E402
    PayloadWriter,
    assemble,
    assemble_rows,
    render_batch,
    render_fix_implications,
)
from studiobench.scoring import Measure  # noqa: E402


def _arm(arm_id: str) -> Arm:
    return Arm(
        arm_id = arm_id,
        title = arm_id,
        mechanism = "synthetic",
        invariance = Invariance.EXACT,
        potency = PotencyCounter(name = "fired", min_delta = 1, direction = "increase"),
        implies_fix = "synthetic",
    )


def _outcome(
    key: str,
    cost: float,
    status: ArmStatus = ArmStatus.QUOTED,
) -> ArmOutcome:
    return ArmOutcome(
        arm = _arm(key),
        cost = Measure.read(cost, "ms/update"),
        status = status,
        reason = "synthetic",
        potency_before = 0,
        potency_after = 500,
    )


#: One coherent synthetic world: the autoscroll observer is the dominant cost on both routes, and
#: the two routes disagree about it, which is the interaction the design exists to surface.
_VISUAL = {
    "shipping": 40.0,
    "A": 34.0,
    "A+B": 26.0,
    "A+B+C": 18.0,
    "A+B+C+D": 9.0,
    "A+B+C+D+E": 7.0,
    "A+B+C+D+E+F": 2.0,
}
_SCHEDULER = {
    "shipping": 40.0,
    "D": 22.0,
    "D+E": 20.0,
    "D+E+F": 15.0,
    "A+D+E+F": 12.0,
    "A+B+D+E+F": 8.0,
    "A+B+C+D+E+F": 2.0,
}


def _all_outcomes() -> dict[str, ArmOutcome]:
    merged = dict(_VISUAL)
    merged.update(_SCHEDULER)
    return {key: _outcome(key, cost) for key, cost in merged.items()}


def _good_calibration():
    return evaluate_batch(
        null_deltas = [Measure.read(0.04, "ms/update")],
        spike_observations = [
            {
                "spike_ms": 0.1,
                "burned_ms_per_update": Measure.read(0.1, "ms/update"),
                "observed_delta": Measure.read(0.02, "ms/update"),
            },
            {
                "spike_ms": 0.5,
                "burned_ms_per_update": Measure.read(0.5, "ms/update"),
                "observed_delta": Measure.read(0.48, "ms/update"),
            },
            {
                "spike_ms": 2.0,
                "burned_ms_per_update": Measure.read(2.0, "ms/update"),
                "observed_delta": Measure.read(1.97, "ms/update"),
            },
        ],
    )


def _blind_calibration():
    return evaluate_batch(
        null_deltas = [Measure.read(0.04, "ms/update")],
        spike_observations = [
            {
                "spike_ms": 2.0,
                "burned_ms_per_update": Measure.read(2.0, "ms/update"),
                "observed_delta": Measure.read(0.005, "ms/update"),
            }
        ],
    )


# planning


# ---------------------------------------------------------------------------------------


def test_a_planned_batch_covers_both_routes_and_the_calibration_arms():
    planned = plan_batch()
    keys = {cell.key for cell in planned}
    assert "shipping" in keys
    assert "A+B+C+D+E+F" in keys  # the shared floor
    assert {"NULL", "SPIKE0.1", "SPIKE0.5", "SPIKE2"} <= keys
    assert missing_cells(planned, _all_outcomes())  # calibration cells are still to be run


def test_arms_with_different_scene_lengths_are_refused_before_the_run():
    assert_equal_scene_duration({"shipping": 40_000.0, "A": 40_000.5})
    with pytest.raises(BatchPlanError) as caught:
        assert_equal_scene_duration({"shipping": 40_000.0, "A": 44_000.0})
    assert "saw different amounts of streaming" in str(caught.value)


def test_no_scene_durations_at_all_is_a_refusal_not_a_pass():
    with pytest.raises(BatchPlanError):
        assert_equal_scene_duration({})


# judging and rendering


# ---------------------------------------------------------------------------------------


def test_a_quotable_batch_renders_steps_interactions_and_verdicts():
    result = judge_batch(
        rung_tokens = 100_000,
        outcomes = _all_outcomes(),
        calibration = _good_calibration(),
    )
    assert result.quotable is True
    assert result.detection_floor_ms == pytest.approx(0.5)
    assert len(result.routes) == 2
    assert all(route.identity_holds for route in result.routes)

    rendered = render_batch(result)
    assert "ROUTE visual_first" in rendered
    assert "ROUTE scheduler_first" in rendered
    assert "INTERACTION TERMS" in rendered
    assert "[FUSED]" in rendered
    assert "Nothing here is averaged" in rendered
    assert "ABLATION DECISION TABLE" in rendered
    # the identity is stated and holds on both routes
    assert rendered.count("no residual to attribute") == 2


def test_no_absolute_arm_cost_appears_anywhere_in_the_rendered_batch():
    """The rule that no arm is quoted alone has to survive rendering, not just the API."""

    result = judge_batch(
        rung_tokens = 100_000,
        outcomes = _all_outcomes(),
        calibration = _good_calibration(),
    )
    rendered = render_batch(result)
    # 34.0 and 26.0 are arm costs; 8.00 and 6.00 are adjacent differences
    assert "34.0 ms/update" not in rendered
    assert "26.0 ms/update" not in rendered
    assert "6.00 ms/update" in rendered  # shipping -> A
    assert "8.00 ms/update" in rendered  # A -> A+B


def test_a_not_quotable_batch_prints_no_ablation_numbers():
    result = judge_batch(
        rung_tokens = 100_000,
        outcomes = _all_outcomes(),
        calibration = _blind_calibration(),
    )
    assert result.quotable is False
    rendered = render_batch(result)
    assert "NO ABLATION NUMBERS ARE PRINTED" in rendered
    assert "ROUTE visual_first" not in rendered
    assert "6.00 ms/update" not in rendered
    # the verdicts still print: a reader needs to know the arms ran
    assert "ARM VERDICTS" in rendered


def test_voided_and_not_run_arms_are_named_in_the_render():
    outcomes = _all_outcomes()
    outcomes["A+B"] = _outcome("A+B", 26.0, ArmStatus.VOIDED)
    outcomes["A+B+C"] = ArmOutcome(
        arm = _arm("A+B+C"),
        cost = Measure.not_attempted("ms/update", "arm did not fire"),
        status = ArmStatus.NOT_RUN,
        reason = "potency counter did not move",
        potency_before = 0,
        potency_after = 0,
    )
    result = judge_batch(rung_tokens = 100_000, outcomes = outcomes, calibration = _good_calibration())
    rendered = render_batch(result)
    assert "VOIDED" in rendered
    assert "NOT RUN arms are not evidence of no effect" in rendered
    assert result.routes[0].identity_holds is False


def test_fix_implications_rank_the_largest_step_and_name_its_fix():
    result = judge_batch(
        rung_tokens = 100_000,
        outcomes = _all_outcomes(),
        calibration = _good_calibration(),
    )
    rendered = render_fix_implications(result)
    top = rendered.splitlines()[1].strip()
    # the two routes disagree about the autoscroll observer, so the headline is a RANGE and both routes
    # are named; quoting only the larger would let a reader pick the flattering route
    assert top.startswith("9.000 to 18.000 ms")
    assert "autoscroll_forced_layout" in top
    assert "routes disagree: scheduler_first 18.000, visual_first 9.000" in rendered
    assert "scrollHeight" in rendered


def test_a_batch_with_a_dose_fit_an_armpack_refusal_and_a_recovery_renders_all_three(tmp_path):
    dose = fit_dose_response(
        [
            DosePoint(dose = d, cost = Measure.read(0.002 * d, "ms"), content_chars = 50_000)
            for d in (4, 40, 400, 4000)
        ],
        detection_floor_ms = 0.5,
    )
    recovery = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(20.0, "ms/update"),
        after_delete = Measure.read(19.0, "ms/update"),
        noise_floor_ms = 0.5,
    )
    armpack = discover_armpack([tmp_path / "absent"], "digest-1")
    result = judge_batch(
        rung_tokens = 100_000,
        outcomes = _all_outcomes(),
        calibration = _good_calibration(),
        armpack = armpack,
        dose = dose,
        recovery = recovery,
    )
    rendered = render_batch(result)
    assert "ABLATION ARMS NOT AVAILABLE FOR THIS BUILD" in rendered
    assert "fibre-free twin does not run" in rendered
    assert "LINEAR THROUGH ORIGIN" in rendered
    assert "RETAINED STRUCTURE" in rendered


# the harness layer's row stream


# ---------------------------------------------------------------------------------------


def test_harness_rows_assemble_and_their_attested_zeros_survive(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    # written by hand rather than through PayloadWriter.write, because these are Layer 1's rows
    with path.open("w", encoding = "utf-8") as handle:
        rows = [
            '{"row_type":"run_meta","tier":"quick","tool_version":"1","corpus_hash":"c",'
            '"studio_ref":"main","bundle":"prod","platform":"linux","started_at":"now"}',
            '{"row_type":"gate","name":"bundle_is_production","passed":true,"detail":{}}',
            '{"row_type":"cell","cell_id":"r10K.A0.rep0","completed":true,"fidelity":"real"}',
            '{"row_type":"window","cell_id":"r10K.A0.rep0","name":"action:scroll",'
            '"kind":"action","t_open_ms":10.0,"duration_ms":4000.0,'
            '"long_task_ms":0,"long_task_ms_attempted":true,'
            '"react_stage_ms":null,"react_stage_ms_reason":"profiling alias not verified"}',
            '{"row_type":"action","cell_id":"r10K.A0.rep0","action":"scroll","ran":true,'
            '"expect_ok":false,"expect":{},"timings":{"gesture_ms":120.0},'
            '"reason":"travelled 0 px","slot_missed":false}',
        ]
        handle.write("\n".join(rows) + "\n")
    writer.close()

    payload = assemble_rows(path)
    assert payload["complete"] is True
    assert len(payload["windows"]) == 1
    # an attested zero survives validation; an action that failed its own assertion is excluded
    assert payload["excluded_cells"][0]["reason"] == "slot_missed"
    assert "must not be quoted" in payload["excluded_cells"][0]["detail"]

    window = payload["windows"][0]
    long_task = Measure.from_row(window, "long_task_ms")
    assert long_task.has_reading and long_task.value == 0.0
    react = Measure.from_row(window, "react_stage_ms")
    assert react.has_reading is False
    assert "profiling alias" in react.display()


def test_a_run_with_no_completed_cell_is_not_complete(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    path.write_text(
        '{"row_type":"run_meta","tier":"quick"}\n'
        '{"row_type":"cell","cell_id":"r1M.A0.rep0","completed":false,'
        '"failure_mode":"renderer crashed","fidelity":"real"}\n',
        encoding = "utf-8",
    )
    payload = assemble_rows(path)
    assert payload["complete"] is False
    assert payload["excluded_cells"][0]["reason"] == "rung_incomplete"
    assert "renderer crashed" in payload["excluded_cells"][0]["detail"]


def test_the_two_writers_produce_the_same_assembled_shape(tmp_path: Path):
    own = tmp_path / "own.jsonl"
    writer = PayloadWriter(own)
    writer.write("header", info = {"tier": "quick"})
    writer.write("window", cell_id = "r1K.A0.rep0", metrics = {})
    writer.write("footer", info = {"exit": "ok"})
    writer.close()

    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        '{"row_type":"run_meta","tier":"quick"}\n'
        '{"row_type":"window","cell_id":"r1K.A0.rep0"}\n'
        '{"row_type":"cell","cell_id":"r1K.A0.rep0","completed":true}\n',
        encoding = "utf-8",
    )

    left, right = assemble(own), assemble_rows(rows)
    for key in ("schema", "complete", "excluded_cells", "windows", "crashes"):
        assert key in left and key in right
    assert left["complete"] is True and right["complete"] is True


def test_missing_cells_is_computed_not_assumed():
    planned = [PlannedCell(arms = frozenset({"A"}), role = "ladder")]
    assert missing_cells(planned, {}) == [arms_key({"A"})]
    assert missing_cells(planned, {"A": _outcome("A", 1.0)}) == []

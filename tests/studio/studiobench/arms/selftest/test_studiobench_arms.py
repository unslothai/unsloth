# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the ablation layer.

The arms layer is where a wrong answer is most expensive, because its output is a CAUSAL claim.
Everything here is a test of a refusal: an arm that drifted must not be quoted, an arm that did
not fire must not read as no effect, a ladder must not quote a rung on its own, a batch without
calibration must not run at all, and an armpack that does not match must stop its plane of the
experiment rather than be skipped quietly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.arms import (  # noqa: E402
    BANNER,
    Arm,
    ArmStatus,
    DeclaredDiff,
    Invariance,
    LadderError,
    PotencyCounter,
    Step,
    arms_key,
    assert_batch_includes_calibration,
    classify_recovery,
    config_init_script,
    differences,
    discover_armpack,
    evaluate_batch,
    fit_dose_response,
    interaction_terms,
    judge,
    render_decision_table,
    required_rungs,
    split_arms,
)
from studiobench.arms.bundle import ARM_FIBRE_FREE_TWIN, BUNDLE_ARMS  # noqa: E402
from studiobench.arms.calibration import CalibrationMissing  # noqa: E402
from studiobench.arms.dose import DosePoint  # noqa: E402
from studiobench.arms.knobs import PREBOOT_ARM_IDS, RUNTIME_ARMS  # noqa: E402
from studiobench.arms.ladder import (  # noqa: E402
    ROUTE_SCHEDULER_FIRST,
    ROUTE_VISUAL_FIRST,
    LadderRoute,
)
from studiobench.arms.manifest import ArmOutcome  # noqa: E402
from studiobench.scoring import Measure  # noqa: E402


def _arm(
    arm_id: str = "X",
    invariance: Invariance = Invariance.EXACT,
    declared_diff: DeclaredDiff | None = None,
) -> Arm:
    return Arm(
        arm_id = arm_id,
        title = arm_id,
        mechanism = "test mechanism",
        invariance = invariance,
        declared_diff = declared_diff,
        potency = PotencyCounter(name = "fired", min_delta = 1, direction = "increase"),
        implies_fix = "test",
    )


# manifest: invariance and potency


# ---------------------------------------------------------------------------------------


def test_an_exact_arm_that_drifts_is_voided_not_quoted():
    outcome = judge(
        _arm(),
        cost = Measure.read(4.0, "ms/update"),
        digest_before = "aaaa1111",
        digest_after = "bbbb2222",
        potency_before = 0,
        potency_after = 700,
    )
    assert outcome.status is ArmStatus.VOIDED
    assert outcome.quotable is False
    assert "VOIDED" in outcome.quote()
    assert "did not render the same thing" in outcome.reason


def test_an_exact_arm_whose_potency_did_not_move_reads_not_run_never_no_effect():
    outcome = judge(
        _arm(),
        cost = Measure.read(0.001, "ms/update"),
        digest_before = "same",
        digest_after = "same",
        potency_before = 0,
        potency_after = 0,
    )
    assert outcome.status is ArmStatus.NOT_RUN
    assert "NOT RUN" in outcome.quote()
    assert outcome.cost.attempted is False
    assert "not a measurement of no effect" in outcome.reason


def test_drift_is_reported_ahead_of_a_dead_potency_counter():
    """A voided arm has already produced a number; NOT RUN would make that sound harmless."""

    outcome = judge(
        _arm(),
        cost = Measure.read(4.0, "ms/update"),
        digest_before = "aaaa",
        digest_after = "bbbb",
        potency_before = 0,
        potency_after = 0,
    )
    assert outcome.status is ArmStatus.VOIDED


def test_a_dom_changing_arm_is_quoted_only_as_a_bound():
    outcome = judge(
        _arm(invariance = Invariance.DOM_CHANGING),
        cost = Measure.read(12.5, "ms/update"),
        digest_before = "a",
        digest_after = "b",
        potency_before = 0,
        potency_after = 44,
    )
    assert outcome.status is ArmStatus.BOUND
    assert outcome.quote().startswith("<= ")


def test_an_equivalent_arm_must_produce_exactly_the_declared_diff():
    arm = _arm(
        invariance = Invariance.EQUIVALENT,
        declared_diff = DeclaredDiff(normaliser = "skip_style_attribute", keys = ("style",)),
    )
    good = judge(
        arm,
        cost = Measure.read(3.0, "ms/update"),
        digest_before = "raw-a",
        digest_after = "raw-b",
        normalised_before = "norm",
        normalised_after = "norm",
        observed_diff_keys = ("style",),
        potency_before = 0,
        potency_after = 685,
    )
    assert good.status is ArmStatus.QUOTED

    extra = judge(
        arm,
        cost = Measure.read(3.0, "ms/update"),
        digest_before = "raw-a",
        digest_after = "raw-b",
        normalised_before = "norm",
        normalised_after = "norm",
        observed_diff_keys = ("style", "data-status"),
        potency_before = 0,
        potency_after = 685,
    )
    assert extra.status is ArmStatus.VOIDED
    assert "undeclared" in extra.reason


def test_an_exact_arm_with_no_digest_is_voided_rather_than_believed():
    outcome = judge(
        _arm(),
        cost = Measure.read(3.0, "ms/update"),
        digest_before = None,
        digest_after = None,
        potency_before = 0,
        potency_after = 99,
    )
    assert outcome.status is ArmStatus.VOIDED
    assert "never checked" in outcome.reason


def test_an_unavailable_arm_is_not_a_zero():
    outcome = judge(
        _arm(),
        cost = Measure.read(0.0, "ms/update"),
        digest_before = "a",
        digest_after = "a",
        available = False,
        unavailable_reason = "no armpack for this dist",
    )
    assert outcome.status is ArmStatus.UNAVAILABLE
    assert outcome.cost.attempted is False


def test_an_equivalent_arm_must_declare_its_diff_and_an_exact_one_may_not():
    with pytest.raises(ValueError):
        _arm(invariance = Invariance.EQUIVALENT)
    with pytest.raises(ValueError):
        _arm(declared_diff = DeclaredDiff(normaliser = "n", keys = ("style",)))


# ladder


# ---------------------------------------------------------------------------------------


def test_a_step_removing_two_mechanisms_must_admit_it_is_fused():
    with pytest.raises(LadderError):
        Step(
            arms_before = frozenset(),
            arms_after = frozenset({"C"}),
            mechanisms = ("layout_geometry", "sibling_count"),
        )
    Step(
        arms_before = frozenset(),
        arms_after = frozenset({"C"}),
        mechanisms = ("layout_geometry", "sibling_count"),
        fused = True,
        fused_reason = "no knob separates them",
    )


def test_a_step_must_be_nested_not_merely_different():
    with pytest.raises(LadderError):
        Step(
            arms_before = frozenset({"A"}),
            arms_after = frozenset({"B"}),
            mechanisms = ("paint_raster",),
        )


def test_a_route_may_not_remove_the_same_mechanism_twice():
    with pytest.raises(LadderError):
        LadderRoute(
            route_id = "bad",
            name = "bad",
            steps = (
                Step(frozenset(), frozenset({"A"}), ("paint_raster",)),
                Step(frozenset({"A"}), frozenset({"A", "B"}), ("paint_raster",)),
            ),
        )


def test_a_route_with_a_gap_is_refused():
    with pytest.raises(LadderError):
        LadderRoute(
            route_id = "gap",
            name = "gap",
            steps = (
                Step(frozenset(), frozenset({"A"}), ("paint_raster",)),
                Step(
                    frozenset({"A", "B"}),
                    frozenset({"A", "B", "C"}),
                    ("layout_geometry", "sibling_count"),
                    fused = True,
                    fused_reason = "x",
                ),
            ),
        )


def _outcomes(costs: dict[str, float]) -> dict[str, ArmOutcome]:
    out: dict[str, ArmOutcome] = {}
    for key, value in costs.items():
        out[key] = ArmOutcome(
            arm = _arm(key),
            cost = Measure.read(value, "ms/update"),
            status = ArmStatus.QUOTED,
            reason = "synthetic",
            potency_before = 0,
            potency_after = 100,
        )
    return out


_VISUAL_COSTS = {
    "shipping": 40.0,
    "A": 34.0,
    "A+B": 26.0,
    "A+B+C": 18.0,
    "A+B+C+D": 9.0,
    "A+B+C+D+E": 7.0,
    "A+B+C+D+E+F": 2.0,
}


def test_the_telescoping_identity_holds_exactly_with_no_residual():
    result = differences(ROUTE_VISUAL_FIRST, _outcomes(_VISUAL_COSTS), detection_floor_ms = 0.5)
    assert result.identity_holds is True
    assert result.residual_ms == pytest.approx(0.0, abs = 1e-12)
    assert result.total.value == pytest.approx(38.0)
    assert result.sum_of_steps.value == pytest.approx(38.0)
    assert "no residual to attribute" in result.identity_note


def test_a_missing_rung_means_the_identity_is_not_claimed():
    costs = dict(_VISUAL_COSTS)
    del costs["A+B"]
    result = differences(ROUTE_VISUAL_FIRST, _outcomes(costs), detection_floor_ms = 0.5)
    assert result.identity_holds is False
    assert "no complete chain" in result.identity_note
    assert any(not step.quotable for step in result.steps)


def test_a_voided_rung_poisons_only_its_own_steps():
    outcomes = _outcomes(_VISUAL_COSTS)
    outcomes["A+B"] = ArmOutcome(
        arm = _arm("A+B"),
        cost = Measure.read(26.0, "ms/update"),
        status = ArmStatus.VOIDED,
        reason = "digest drifted",
    )
    result = differences(ROUTE_VISUAL_FIRST, outcomes, detection_floor_ms = 0.5)
    quotable = [s for s in result.steps if s.quotable]
    assert len(quotable) == len(ROUTE_VISUAL_FIRST.steps) - 2
    assert result.identity_holds is False


def test_a_bound_rung_makes_its_adjacent_difference_a_bound():
    outcomes = _outcomes(_VISUAL_COSTS)
    outcomes["A+B+C+D+E+F"] = ArmOutcome(
        arm = _arm("A+B+C+D+E+F"),
        cost = Measure.read(2.0, "ms/update"),
        status = ArmStatus.BOUND,
        reason = "DOM changing",
        potency_after = 10,
    )
    result = differences(ROUTE_VISUAL_FIRST, outcomes, detection_floor_ms = 0.5)
    react_step = result.steps[-1]
    assert react_step.bound_only is True
    assert react_step.quote().startswith("<= ")


def test_no_arm_may_be_quoted_alone():
    with pytest.raises(LadderError) as caught:
        ROUTE_VISUAL_FIRST.quote_arm({"A", "B"})
    assert "only adjacent differences" in str(caught.value)


def test_the_two_declared_routes_reach_the_same_floor_by_different_orders():
    assert ROUTE_VISUAL_FIRST.floor == ROUTE_SCHEDULER_FIRST.floor
    assert ROUTE_VISUAL_FIRST.mechanisms != ROUTE_SCHEDULER_FIRST.mechanisms
    assert sorted(ROUTE_VISUAL_FIRST.mechanisms) == sorted(ROUTE_SCHEDULER_FIRST.mechanisms)


def test_route_disagreement_is_reported_as_an_interaction_not_averaged():
    scheduler_costs = {
        "shipping": 40.0,
        "D": 22.0,  # removing the observer first is worth much more on this route
        "D+E": 20.0,
        "D+E+F": 15.0,
        "A+D+E+F": 12.0,
        "A+B+D+E+F": 8.0,
        "A+B+C+D+E+F": 2.0,
    }
    visual = differences(ROUTE_VISUAL_FIRST, _outcomes(_VISUAL_COSTS), detection_floor_ms = 0.5)
    scheduler = differences(
        ROUTE_SCHEDULER_FIRST, _outcomes(scheduler_costs), detection_floor_ms = 0.5
    )
    terms = {t.mechanism: t for t in interaction_terms(visual, scheduler, detection_floor_ms = 0.5)}
    assert terms["autoscroll_forced_layout"].disagreement_ms == pytest.approx(9.0 - 18.0)
    assert "not additive" in terms["autoscroll_forced_layout"].note
    # nothing anywhere produced a mean of the two
    assert terms["autoscroll_forced_layout"].value_a.value == pytest.approx(9.0)
    assert terms["autoscroll_forced_layout"].value_b.value == pytest.approx(18.0)


def test_routes_with_different_floors_cannot_be_compared():
    other = LadderRoute(
        route_id = "short",
        name = "short",
        steps = (Step(frozenset(), frozenset({"A"}), ("paint_raster",)),),
    )
    left = differences(ROUTE_VISUAL_FIRST, _outcomes(_VISUAL_COSTS))
    right = differences(other, _outcomes({"shipping": 40.0, "A": 34.0}))
    with pytest.raises(LadderError):
        interaction_terms(left, right)


def test_required_rungs_covers_both_routes():
    keys = {arms_key(r) for r in required_rungs()}
    assert "shipping" in keys
    assert "A+B+C+D+E+F" in keys
    assert "D" in keys and "A" in keys


# calibration


# ---------------------------------------------------------------------------------------


def test_a_batch_without_calibration_arms_is_refused_before_it_runs():
    with pytest.raises(CalibrationMissing) as caught:
        assert_batch_includes_calibration(["A", "B", "C"])
    assert "NULL" in str(caught.value)
    assert_batch_includes_calibration(["A", "NULL", "SPIKE0.1", "SPIKE0.5", "SPIKE2"])


def _spike(spike_ms: float, burned: float, observed: float) -> dict:
    return {
        "spike_ms": spike_ms,
        "burned_ms_per_update": Measure.read(burned, "ms/update"),
        "observed_delta": Measure.read(observed, "ms/update"),
    }


def test_a_well_behaved_batch_is_quotable_and_prints_both_floors():
    verdict = evaluate_batch(
        null_deltas = [Measure.read(0.03, "ms/update")],
        spike_observations = [
            _spike(0.1, 0.11, 0.02),
            _spike(0.5, 0.52, 0.49),
            _spike(2.0, 2.05, 1.98),
        ],
    )
    assert verdict.quotable is True
    assert verdict.noise_floor_ms.value == pytest.approx(0.03)
    assert verdict.detection_floor_ms.value == pytest.approx(0.5)
    assert [s.recovered for s in verdict.spikes] == [False, True, True]
    assert "recovery" in verdict.render()


def test_a_batch_where_no_spike_is_recovered_is_not_quotable():
    verdict = evaluate_batch(
        null_deltas = [Measure.read(0.01, "ms/update")],
        spike_observations = [
            _spike(0.1, 0.11, 0.001),
            _spike(0.5, 0.52, 0.004),
            _spike(2.0, 2.05, 0.01),
        ],
    )
    assert verdict.quotable is False
    assert "could not see a cost it injected itself" in verdict.reason


def test_a_batch_whose_null_arm_drifted_past_the_detection_floor_is_not_quotable():
    verdict = evaluate_batch(
        null_deltas = [Measure.read(0.6, "ms/update")],
        spike_observations = [_spike(0.5, 0.5, 0.62), _spike(2.0, 2.0, 1.95)],
    )
    assert verdict.quotable is False
    assert "read as different" in verdict.reason


def test_a_noisy_batch_stays_quotable_but_only_at_a_coarser_floor():
    """A noisy machine is not automatically a broken one; it is a machine with a blunt floor.

    The null moves 0.9 ms/update, which swallows the 0.1 and 0.5 spikes entirely. The 2.0 spike
    still comes back cleanly, so the batch can resolve differences above 2.0 ms/update and
    nothing below it. Voiding this batch would throw away a usable, coarse measurement; quoting
    a 0.3 ms difference from it would be inventing one. The detection floor is what keeps the
    two apart, and every difference is rendered against it.
    """

    verdict = evaluate_batch(
        null_deltas = [Measure.read(0.9, "ms/update")],
        spike_observations = [
            _spike(0.1, 0.1, 0.02),
            _spike(0.5, 0.5, 0.49),
            _spike(2.0, 2.0, 1.99),
        ],
    )
    assert verdict.quotable is True
    assert verdict.detection_floor_ms.value == pytest.approx(2.0)
    assert Measure.read(0.3, "ms/update", floor = 2.0).display().startswith("< 2")


def test_a_spike_read_at_the_wrong_magnitude_is_not_recovered():
    verdict = evaluate_batch(
        null_deltas = [Measure.read(0.02, "ms/update")],
        spike_observations = [_spike(2.0, 2.0, 8.0)],
    )
    assert verdict.spikes[0].recovered is False
    assert "outside" in verdict.spikes[0].note
    assert verdict.quotable is False


def test_a_batch_with_no_null_reading_has_no_noise_floor():
    verdict = evaluate_batch(
        null_deltas = [Measure.failed("ms/update", "the NULL cell crashed")],
        spike_observations = [_spike(2.0, 2.0, 1.9)],
    )
    assert verdict.quotable is False
    assert "no measured noise floor" in verdict.reason


# dose-response


# ---------------------------------------------------------------------------------------


def _dose_points(
    per_child_ms: float,
    intercept: float = 0.0,
    chars: int = 50_000,
):
    return [
        DosePoint(
            dose = d, cost = Measure.read(intercept + per_child_ms * d, "ms"), content_chars = chars
        )
        for d in (4, 40, 400, 4000)
    ]


def test_a_line_through_the_origin_is_identified_as_o_children():
    fit = fit_dose_response(_dose_points(0.002), detection_floor_ms = 0.5)
    assert fit.verdict == "LINEAR THROUGH ORIGIN"
    assert fit.slope_through_origin == pytest.approx(0.002, rel = 1e-6)


def test_a_flat_result_is_an_informative_null_with_a_bound():
    points = [
        DosePoint(dose = d, cost = Measure.read(3.0, "ms"), content_chars = 50_000)
        for d in (4, 40, 400, 4000)
    ]
    fit = fit_dose_response(points, detection_floor_ms = 0.5)
    assert fit.verdict == "UNDERPOWERED NULL"
    assert fit.min_detectable_slope.value == pytest.approx(0.5 / 4000)
    assert "real bound" in fit.note


def test_a_flat_result_without_a_detection_floor_cannot_be_turned_into_a_bound():
    points = [
        DosePoint(dose = d, cost = Measure.read(3.0, "ms"), content_chars = 50_000)
        for d in (4, 40, 400, 4000)
    ]
    fit = fit_dose_response(points)
    assert fit.verdict == "NULL, UNBOUNDED"
    assert fit.min_detectable_slope.attempted is False


def test_a_large_intercept_is_called_out_rather_than_reported_as_a_slope():
    fit = fit_dose_response(_dose_points(0.0005, intercept = 6.0), detection_floor_ms = 0.1)
    assert fit.verdict == "MOSTLY FIXED COST"


def test_varying_content_across_doses_voids_the_design():
    points = _dose_points(0.002)
    points[-1] = DosePoint(dose = 4000, cost = Measure.read(9.0, "ms"), content_chars = 90_000)
    fit = fit_dose_response(points, detection_floor_ms = 0.5)
    assert fit.verdict == "INVALID"
    assert "confounded" in fit.note


def test_two_points_do_not_make_a_line():
    points = _dose_points(0.002)[:2]
    fit = fit_dose_response(points, detection_floor_ms = 0.5)
    assert fit.verdict == "NO FIT"


# armpack


# ---------------------------------------------------------------------------------------


def _write_armpack(
    root: Path,
    digest: str,
    arms = None,
) -> Path:
    arms = arms if arms is not None else {arm.arm_id: arm.arm_id.lower() for arm in BUNDLE_ARMS}
    root.mkdir(parents = True, exist_ok = True)
    for rel in arms.values():
        (root / rel).mkdir(parents = True, exist_ok = True)
    (root / "armpack.json").write_text(
        json.dumps(
            {
                "armpack_version": "1",
                "built_from_sha": "deadbeefcafe",
                "target_dist_digest": digest,
                "arms": arms,
            }
        ),
        encoding = "utf-8",
    )
    return root


def test_no_armpack_prints_the_banner_and_stops_that_plane(tmp_path: Path):
    resolution = discover_armpack([tmp_path / "nowhere"], "digest-1")
    assert resolution.available is False
    assert BANNER in resolution.render()
    assert "fibre-free twin does not run" in resolution.render()
    with pytest.raises(Exception):
        resolution.require()


def test_a_mismatched_armpack_is_refused_rather_than_used(tmp_path: Path):
    _write_armpack(tmp_path / "pack", "digest-other")
    resolution = discover_armpack([tmp_path / "pack"], "digest-1")
    assert resolution.available is False
    assert "measure the build difference" in resolution.reason


def test_a_partial_armpack_missing_the_twin_is_refused(tmp_path: Path):
    arms = {
        arm.arm_id: arm.arm_id.lower()
        for arm in BUNDLE_ARMS
        if arm.arm_id != ARM_FIBRE_FREE_TWIN.arm_id
    }
    _write_armpack(tmp_path / "pack", "digest-1", arms = arms)
    resolution = discover_armpack([tmp_path / "pack"], "digest-1")
    assert resolution.available is False
    assert ARM_FIBRE_FREE_TWIN.arm_id in resolution.reason


def test_a_matching_armpack_resolves(tmp_path: Path):
    _write_armpack(tmp_path / "pack", "digest-1")
    resolution = discover_armpack([tmp_path / "pack"], "digest-1")
    assert resolution.available is True
    assert resolution.require().target_dist_digest == "digest-1"


# recovery


# ---------------------------------------------------------------------------------------


def test_full_recovery_is_occupancy():
    result = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(20.0, "ms/update"),
        after_delete = Measure.read(2.4, "ms/update"),
        noise_floor_ms = 0.5,
    )
    assert result.classification == "OCCUPANCY"


def test_no_recovery_is_retained_structure():
    result = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(20.0, "ms/update"),
        after_delete = Measure.read(19.5, "ms/update"),
        noise_floor_ms = 0.5,
    )
    assert result.classification == "RETAINED STRUCTURE"
    assert "stays worse" in result.note


def test_partial_recovery_is_not_rounded_to_whichever_is_convenient():
    result = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(20.0, "ms/update"),
        after_delete = Measure.read(11.0, "ms/update"),
        noise_floor_ms = 0.5,
    )
    assert result.classification == "HYSTERETIC"
    assert result.recovered_fraction == pytest.approx(0.5)


def test_a_load_that_cost_nothing_has_an_undefined_recovery_not_a_perfect_one():
    result = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(2.2, "ms/update"),
        after_delete = Measure.read(2.1, "ms/update"),
        noise_floor_ms = 0.5,
    )
    assert result.classification == "NOTHING TO RECOVER"
    assert result.recovered_fraction is None
    assert "UNDEFINED" in result.note


def test_worse_after_delete_points_at_the_delete_path():
    result = classify_recovery(
        baseline = Measure.read(2.0, "ms/update"),
        loaded = Measure.read(20.0, "ms/update"),
        after_delete = Measure.read(26.0, "ms/update"),
        noise_floor_ms = 0.5,
    )
    assert result.classification == "WORSE AFTER DELETE"


# knobs


# ---------------------------------------------------------------------------------------


def test_only_requested_preboot_arms_are_installed():
    config = json.loads(config_init_script(["A", "D"]).split("=", 1)[1].strip().rstrip(";"))
    assert config["preboot"] == ["D"]
    assert config["requested"] == ["A", "D"]


def test_an_unknown_arm_id_is_a_hard_error():
    with pytest.raises(KeyError):
        config_init_script(["A", "Z"])


def test_split_arms_separates_preboot_from_apply_time():
    preboot, runtime = split_arms(["A", "B", "C", "D", "E", "F", "G"])
    assert set(preboot) == set(PREBOOT_ARM_IDS)
    assert set(runtime) == {"A", "B", "C", "G"}


def test_every_knob_declares_which_fix_its_outcome_implies():
    table = render_decision_table()
    for arm in RUNTIME_ARMS:
        assert f"  {arm.arm_id}  removes:" in table
        assert arm.implies_fix in table


def test_the_control_arm_exists_and_is_not_a_treatment():
    control = [arm for arm in RUNTIME_ARMS if arm.kind == "control"]
    assert [arm.arm_id for arm in control] == ["G"]

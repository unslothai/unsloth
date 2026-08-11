# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only unit tests for the Kaggle T4 smoke harness.

These cover the logic that decides whether a run passed, whether quota gets
spent, and whether the kernel notebook is well formed. None of it needs a
GPU, and all of it is the kind of thing that is expensive to discover on a
Kaggle session forty minutes later.

The training payload itself is not exercised here; it needs a T4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))


# ---------------------------------------------------------------- dataset


def test_canary_dataset_targets_the_canary_and_nothing_else():
    """A row whose answer drifted would make the exact-match check vacuous."""
    rows = [
        json.loads(line)
        for line in (SMOKE_DIR / "canary_dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "canary dataset must not be empty"
    assert all(r["answer"] == "__UNSLOTH__!!!" for r in rows)
    # Distinct questions: identical rows would make the sampler's
    # step -> row mapping unobservable.
    assert len({r["question"] for r in rows}) == len(rows)


# ------------------------------------------------------------ determinism


def test_repeating_sequential_sampler_order_is_a_function_of_the_step():
    from determinism import RepeatingSequentialSampler

    sampler = RepeatingSequentialSampler(
        dataset_length = 3, batch_size = 2, gradient_accumulation_steps = 1, max_steps = 4
    )
    assert list(sampler) == [0, 0, 1, 1, 2, 2, 0, 0]
    assert len(sampler) == 8
    # Iterating twice must give the same answer; a generator that consumed
    # shared state would silently reorder the second epoch.
    assert list(sampler) == list(sampler)


def test_compare_metrics_treats_matching_nan_as_equal():
    """fp16 skipped steps log NaN, and NaN != NaN would fail identical runs."""
    from determinism import compare_metrics

    nan = float("nan")
    a = [{"step": 1, "loss": 1.0, "grad_norm": nan}, {"step": 2, "loss": 0.5, "grad_norm": 3.0}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": nan}, {"step": 2, "loss": 0.5, "grad_norm": 3.0}]
    result = compare_metrics(a, b)
    assert result["identical"] is True
    assert result["first_diff_step"] is None


def test_compare_metrics_flags_a_nan_that_appeared_on_only_one_side():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": float("nan")}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": 5.0}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["first_diff_step"] == 1


def test_compare_metrics_flags_a_real_difference():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}]
    b = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5000001}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["first_diff_step"] == 2


def test_compare_metrics_flags_a_length_mismatch():
    from determinism import compare_metrics

    result = compare_metrics([{"step": 1, "loss": 1.0}], [])
    assert result["identical"] is False
    assert result["length_mismatch"] is True


# ------------------------------------------------------------------- gate


def test_sampling_rate_is_close_to_the_requested_percent():
    from gate import sampled_in
    hits = sum(sampled_in(str(i), 10)[0] for i in range(20000))
    assert 0.08 < hits / 20000 < 0.12, hits


def test_sampling_is_stable_for_a_given_run_id():
    """A re-run must not reroll, or 10% becomes a floor rather than a rate."""
    from gate import sampled_in
    assert sampled_in("123456", 10) == sampled_in("123456", 10)


def test_sampling_at_zero_percent_never_fires():
    from gate import sampled_in
    assert not any(sampled_in(str(i), 0)[0] for i in range(500))


# ------------------------------------------------------ gate: in-flight scan


class _FakeKernel:
    def __init__(self, ref, last_run_time):
        self.ref = ref
        self.last_run_time = last_run_time


class _FakeStatus:
    def __init__(self, status):
        self.status = status


class _FakeApi:
    """Enough of the Kaggle client to drive the survey, and it records
    which kernels were actually status-checked so a test can assert that the
    walk stopped where it claims to."""

    def __init__(
        self,
        kernels,
        statuses = None,
        unreadable = (),
    ):
        self.kernels = list(kernels)
        self.statuses = statuses or {}
        self.unreadable = set(unreadable)
        self.checked = []

    def kernels_list(
        self,
        mine = False,
        page = 1,
        page_size = 20,
        sort_by = None,
    ):
        assert mine and sort_by == "dateRun"
        start = (page - 1) * page_size
        return self.kernels[start : start + page_size]

    def kernels_status(self, ref):
        self.checked.append(ref)
        if ref in self.unreadable:
            raise RuntimeError("500")
        return _FakeStatus(f"KernelWorkerStatus.{self.statuses.get(ref, 'COMPLETE')}")


def _now():
    from datetime import datetime
    return datetime(2026, 8, 11, 12, 0, 0)


def _ago(hours):
    from datetime import timedelta
    return _now() - timedelta(hours = hours)


def test_survey_finds_a_running_kernel_hidden_behind_newer_finished_ones():
    """The exact hole a fixed "12 most recent" bound left open.

    A kernel that started three hours ago and is still running, with forty
    newer kernels that have since run and finished. Bounding the scan by
    COUNT misses it and the push then dies at the capacity cap; bounding it
    by the session ceiling cannot.
    """
    from gate import concurrency_verdict, survey_kernels

    kernels = [_FakeKernel(f"u/done{i}", _ago(0.5 + i * 0.01)) for i in range(40)]
    kernels.append(_FakeKernel("u/old-runner", _ago(3)))
    api = _FakeApi(kernels, statuses = {"u/old-runner": "RUNNING"})

    survey = survey_kernels(api, now = _now())
    assert survey["busy"] == ["u/old-runner (RUNNING)"]
    assert survey["complete"] is True
    clear, why = concurrency_verdict(survey)
    assert clear is False and "old-runner" in why


def test_survey_stops_at_the_session_ceiling_rather_than_walking_the_account():
    """Nothing older than a session can last is looked at, and the walk ends."""
    from gate import LOOKBACK_HOURS, survey_kernels

    kernels = [
        _FakeKernel("u/recent", _ago(1)),
        _FakeKernel("u/edge", _ago(LOOKBACK_HOURS - 0.1)),
        _FakeKernel("u/stale", _ago(LOOKBACK_HOURS + 0.1)),
        _FakeKernel("u/ancient", _ago(24 * 30)),
    ]
    api = _FakeApi(kernels, statuses = {"u/ancient": "RUNNING"})

    survey = survey_kernels(api, now = _now())
    assert api.checked == ["u/recent", "u/edge"]
    assert survey["surveyed"] == 2
    assert survey["complete"] is True
    # A "RUNNING" older than any session can last is a stale listing, not an
    # in-flight kernel, and must not stand the job down forever.
    assert survey["busy"] == []


def test_survey_handles_timezone_aware_timestamps():
    from datetime import timezone

    from gate import survey_kernels

    aware = _ago(1).replace(tzinfo = timezone.utc)
    api = _FakeApi([_FakeKernel("u/a", aware)], statuses = {"u/a": "RUNNING"})
    survey = survey_kernels(api, now = _now())
    assert survey["busy"] == ["u/a (RUNNING)"]


def test_a_kernel_with_no_timestamp_is_checked_but_does_not_end_the_walk():
    from gate import survey_kernels

    api = _FakeApi(
        [_FakeKernel("u/undated", None), _FakeKernel("u/recent", _ago(1))],
        statuses = {"u/recent": "QUEUED"},
    )
    survey = survey_kernels(api, now = _now())
    assert api.checked == ["u/undated", "u/recent"]
    assert survey["busy"] == ["u/recent (QUEUED)"]


def test_a_survey_that_ran_out_of_pages_is_not_read_as_an_idle_account():
    from gate import concurrency_verdict, survey_kernels

    kernels = [_FakeKernel(f"u/k{i}", _ago(1)) for i in range(1000)]
    api = _FakeApi(kernels)
    survey = survey_kernels(api, now = _now(), page_size = 10, max_pages = 3)
    assert survey["surveyed"] == 30
    assert survey["complete"] is False
    clear, why = concurrency_verdict(survey)
    assert clear is False and "unseen" in why


def test_statuses_that_all_come_back_unreadable_are_not_read_as_idle():
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi(
        [_FakeKernel("u/a", _ago(1)), _FakeKernel("u/b", _ago(2))], unreadable = ("u/a", "u/b")
    )
    survey = survey_kernels(api, now = _now())
    assert survey["unreadable"] == 2 and survey["busy"] == []
    clear, why = concurrency_verdict(survey)
    assert clear is False and "unknown" in why


def test_some_unreadable_statuses_do_not_block_a_readable_idle_account():
    """Deleted kernels 404 routinely; that must not wedge the gate shut."""
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi(
        [_FakeKernel("u/gone", _ago(1)), _FakeKernel("u/done", _ago(2))], unreadable = ("u/gone",)
    )
    survey = survey_kernels(api, now = _now())
    assert concurrency_verdict(survey) == (True, "")


def test_standing_down_while_the_account_is_busy_is_the_default():
    """The documented policy: one kernel in flight and this job yields.

    Kaggle would allow a second (2 kernels x 2 T4s = 4 payloads) and this
    deliberately does not take it. The knob is ALLOWED_IN_FLIGHT_KERNELS;
    the default being 0 rather than 1 is the decision under test.
    """
    from gate import ALLOWED_IN_FLIGHT_KERNELS, concurrency_verdict

    assert ALLOWED_IN_FLIGHT_KERNELS == 0
    survey = {
        "busy": ["u/a (RUNNING)"],
        "complete": True,
        "surveyed": 1,
        "unreadable": 0,
        "window_hours": 13.0,
    }
    clear, why = concurrency_verdict(survey)
    assert clear is False
    assert "1 kernel(s) in flight" in why and "tolerates 0" in why


def test_the_in_flight_tolerance_is_a_knob_and_not_a_constant():
    """Raising it is a one-argument change, which is what makes the default
    a policy rather than an accident of the implementation."""
    from gate import concurrency_verdict

    survey = {
        "busy": ["u/a (RUNNING)"],
        "complete": True,
        "surveyed": 1,
        "unreadable": 0,
        "window_hours": 13.0,
    }
    assert concurrency_verdict(survey, 1) == (True, "")
    assert concurrency_verdict(survey, 0)[0] is False


def test_the_tolerance_can_never_be_raised_to_the_account_cap():
    """Leaving no headroom races anything that starts after the survey."""
    from gate import MAX_CONCURRENT_GPU_KERNELS, concurrency_verdict

    survey = {
        "busy": [f"u/k{i} (RUNNING)" for i in range(MAX_CONCURRENT_GPU_KERNELS)],
        "complete": True,
        "surveyed": 2,
        "unreadable": 0,
        "window_hours": 13.0,
    }
    clear, why = concurrency_verdict(survey, MAX_CONCURRENT_GPU_KERNELS)
    assert clear is False and "Standing down" in why


def test_an_account_with_no_kernels_at_all_is_clear():
    from gate import concurrency_verdict, survey_kernels

    survey = survey_kernels(_FakeApi([]), now = _now())
    assert survey["complete"] is True
    assert concurrency_verdict(survey) == (True, "")


# ---------------------------------------- gate: a skip is never a failure
#
# Every negative answer the gate can give has to exit 0. Not spending quota
# is the designed outcome for most invocations, and a workflow that went red
# on it would be ignored by the time it was ever right.


def _run_gate(monkeypatch, tmp_path, *extra):
    tmp_path.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate.py",
            "--budget-hours",
            "1",
            "--reserve-hours",
            "20",
            "--percent",
            "10",
            "--run-id",
            "12345",
            *extra,
        ],
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    import gate

    code = gate.main()
    outputs = dict(
        line.split("=", 1)
        for line in (tmp_path / "out.txt").read_text().splitlines()
        if "=" in line
    )
    return code, outputs


def test_a_missing_token_is_a_skip_not_a_failure(monkeypatch, tmp_path):
    """What a fork pull request gets: no secret, and nothing red."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising = False)
    code, outputs = _run_gate(monkeypatch, tmp_path, "--force", "true")
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "fork" in outputs["reason"]


def test_a_gate_error_is_a_skip_not_a_failure(monkeypatch, tmp_path):
    """An unreachable Kaggle API says nothing about the code under test."""
    import gate

    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    monkeypatch.setattr(gate, "kaggle_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    code, outputs = _run_gate(monkeypatch, tmp_path, "--force", "true")
    assert code == 0
    assert outputs["should_run"] == "false"
    # The reason names the failure type but never the credential.
    assert "RuntimeError" in outputs["reason"]
    assert "not-a-real-token" not in (tmp_path / "out.txt").read_text()


def test_an_unsampled_invocation_is_a_skip_not_a_failure(monkeypatch, tmp_path):
    from gate import sampled_in

    unlucky = next(str(i) for i in range(1000) if not sampled_in(str(i), 10)[0])
    code, outputs = _run_gate(monkeypatch, tmp_path, "--run-id", unlucky)
    assert code == 0 and outputs["should_run"] == "false"
    assert "not sampled" in outputs["reason"]


def test_a_rerun_of_the_same_run_id_does_not_reroll(monkeypatch, tmp_path):
    """A re-run must not be a fresh draw, or 10% is a floor, not a rate."""
    first = _run_gate(monkeypatch, tmp_path / "a", "--run-attempt", "1")
    second = _run_gate(monkeypatch, tmp_path / "b", "--run-attempt", "7")
    assert first[1]["should_run"] == second[1]["should_run"]


# ----------------------------------------------------------- reference band


def _write_reference(path: Path, metrics: list[dict], max_steps: int) -> Path:
    """A reference file shaped the way a captured one is.

    ``config.max_steps`` is not decoration: check_reference refuses to
    compare against a file that does not carry it, so a helper that omitted
    it would make every test below a test of that refusal instead.
    """
    path.write_text(json.dumps({"metrics": metrics, "config": {"max_steps": max_steps}}))
    return path


@pytest.mark.parametrize(
    ("observed", "expected_status"),
    [(1.00, "ok"), (1.05, "ok"), (1.50, "out_of_band")],
)
def test_reference_band_accepts_drift_and_rejects_a_real_move(tmp_path, observed, expected_status):
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    ref = _write_reference(tmp_path / "ref.json", [{"step": 1, "loss": 1.0}], max_steps = 1)
    verdict = check_reference(
        [{"step": 1, "loss": observed}], ref, rel_tol = 0.10, abs_floor = 0.05, max_steps = 1
    )
    assert verdict["status"] == expected_status


def test_reference_band_absolute_floor_tolerates_near_zero_losses(tmp_path):
    """Late steps approach zero, where a tiny absolute drift is not a regression."""
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    ref = _write_reference(tmp_path / "ref.json", [{"step": 10, "loss": 0.0001}], max_steps = 10)
    # 0.002 absolute against a 0.0001 reference is a 1900% relative move,
    # but it is noise. The 0.05 floor is what stops it firing.
    verdict = check_reference(
        [{"step": 10, "loss": 0.002}], ref, rel_tol = 0.10, abs_floor = 0.05, max_steps = 10
    )
    assert verdict["status"] == "ok"


def test_reference_absent_is_not_a_failure():
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    verdict = check_reference(
        [{"step": 1, "loss": 1.0}], Path("/nonexistent/ref.json"), 0.1, 0.05, max_steps = 3
    )
    assert verdict["status"] == "absent"


# ------------------------------------- the reference must be for THIS run
#
# The band check compares a run against a trace captured at one specific
# step count. Comparing across counts is arithmetic that succeeds and means
# nothing -- the fp16 scaler burns the front of every run, so a 3-step run
# is all front, and step 4 of a 10-step trace is not a step a 3-step run
# ever takes. These prove it fails, loudly, rather than sliding past.


def test_a_reference_from_a_different_step_count_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ten = [{"step": i, "loss": 10.0 - i} for i in range(1, 11)]
    ref = _write_reference(tmp_path / "ref.json", ten, max_steps = 10)

    verdict = check_reference(ten, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_count_mismatch"
    assert verdict["reference_max_steps"] == 10
    assert verdict["observed_max_steps"] == 3
    # The refusal must reach the list that turns the job red, and it must
    # name both counts: a reader has to be able to tell this from a real
    # numeric regression without opening the artifact.
    failures = reference_failures(verdict, 0.10)
    assert len(failures) == 1
    assert "max_steps=10" in failures[0] and "3 steps" in failures[0]


def test_a_step_count_mismatch_is_refused_even_when_the_numbers_agree(tmp_path):
    """The worst case: identical metrics, so nothing else would object.

    A 10-step reference and a 3-step run whose logged values happen to
    match would sail through the band, the length check and every tolerance
    in the file. Only the declared step count catches it, which is why it is
    checked first and returns before a single value is compared.
    """
    from run_t4_smoke import check_reference, reference_failures

    metrics = [{"step": 1, "loss": 10.0, "grad_norm": 5.0}]
    ref = _write_reference(tmp_path / "ref.json", metrics, max_steps = 10)

    verdict = check_reference(metrics, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_count_mismatch"
    assert reference_failures(verdict, 0.10)
    # No reassuring numbers may be produced from a comparison that was
    # refused: an empty deviations list next to a "worst deviation 0.0"
    # reads exactly like a pass.
    assert verdict["deviations"] == []
    assert verdict["worst_rel"] == {}


def test_a_reference_that_does_not_say_its_step_count_is_refused(tmp_path):
    """ "It does not say" is not "it matches"."""
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 1.0}]}))
    verdict = check_reference([{"step": 1, "loss": 1.0}], ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "reference_step_count_unknown"
    assert reference_failures(verdict, 0.10)


@pytest.mark.parametrize("config", [{"max_steps": "three"}, {"max_steps": None}, {}, "not-a-dict"])
def test_an_unreadable_step_count_is_refused_rather_than_assumed(tmp_path, config):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 1.0}], "config": config}))
    verdict = check_reference([{"step": 1, "loss": 1.0}], ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "reference_step_count_unknown"
    assert reference_failures(verdict, 0.10)


def test_a_matching_step_count_still_compares_the_numbers(tmp_path):
    """The guard must not become a way to pass without being checked."""
    from run_t4_smoke import check_reference, reference_failures

    ref = _write_reference(
        tmp_path / "ref.json",
        [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 1.0}, {"step": 3, "loss": 0.5}],
        max_steps = 3,
    )
    good = check_reference(
        [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 1.0}, {"step": 3, "loss": 0.5}],
        ref,
        0.10,
        0.05,
        max_steps = 3,
    )
    assert good["status"] == "ok" and reference_failures(good, 0.10) == []
    bad = check_reference(
        [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 4.0}, {"step": 3, "loss": 0.5}],
        ref,
        0.10,
        0.05,
        max_steps = 3,
    )
    assert bad["status"] == "out_of_band" and reference_failures(bad, 0.10)


def test_check_reference_cannot_be_called_without_a_step_count():
    """Mandatory by signature, so no call site can omit it by accident."""
    import inspect

    from run_t4_smoke import check_reference

    param = inspect.signature(check_reference).parameters["max_steps"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        check_reference([], Path("/nonexistent/ref.json"), 0.1, 0.05)


def test_the_committed_reference_records_the_step_count_it_was_captured_at():
    """Without this the file is unusable, and the failure is far away."""
    from run_t4_smoke import reference_step_count

    steps = reference_step_count(_committed_reference())
    assert isinstance(steps, int) and steps > 0
    # The trace has one logged row per step (logging_steps=1). If that ever
    # stops holding, the declared count is the one to trust and this is the
    # place to find out.
    assert len(_committed_reference()["metrics"]) == steps


def test_the_workflow_step_count_and_the_payload_default_agree():
    """Two places state the step count; disagreeing costs a Kaggle session.

    The workflow's input default is what CI runs and the payload's argparse
    default is what a local reproduction runs, and a reference is only valid
    for one number.
    """
    import re

    from run_t4_smoke import main  # noqa: F401  (import proves it loads)

    workflow = (REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml").read_text()
    dispatch_default = re.search(
        r"max_steps:\s*\n\s*description:.*\n\s*type: string\n\s*default: '(\d+)'", workflow
    ).group(1)
    fallback = re.search(r"--max-steps \$\{\{ inputs\.max_steps \|\| (\d+) \}\}", workflow).group(1)
    payload = re.search(
        r'"--max-steps", type=int, default=(\d+)', (SMOKE_DIR / "run_t4_smoke.py").read_text()
    ).group(1)
    assert dispatch_default == fallback == payload, (dispatch_default, fallback, payload)


# ------------------------------------------- the band check, proved to fail
#
# A check that has never been observed to fail is not yet a check. Everything
# below perturbs a reference and asserts the harness goes red, including the
# committed T4 reference itself.

COMMITTED_REFERENCE = SMOKE_DIR / "references" / "t4_qwen2.5-0.5b.json"


def _committed_reference() -> dict:
    if not COMMITTED_REFERENCE.exists():
        pytest.skip("no committed T4 reference to perturb yet")
    return json.loads(COMMITTED_REFERENCE.read_text())


def _perturb(
    metrics: list[dict],
    index: int,
    field: str,
    abs_floor: float = 0.05,
    factor: float = 0.5,
) -> list[dict]:
    """Move one value by half a band-width more than the band allows.

    Scaled by the same max(|value|, abs_floor) the check uses, so the
    perturbation is out of band wherever on the curve it is applied.
    """
    out = [dict(m) for m in metrics]
    value = float(out[index][field])
    out[index][field] = value + max(abs(value), abs_floor) * factor
    return out


def _committed_steps() -> int:
    """The step count the committed reference was captured at.

    Read from the file rather than hardcoded, so these tests keep testing
    the numbers after a recapture at a different count instead of failing
    for a reason that is not a regression.
    """
    from run_t4_smoke import reference_step_count
    return reference_step_count(_committed_reference())


def test_the_committed_reference_matches_itself(tmp_path):
    """The floor under the next test: an unperturbed comparison is clean."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = _committed_reference()["metrics"]
    verdict = check_reference(
        metrics, COMMITTED_REFERENCE, 0.10, 0.05, max_steps = _committed_steps()
    )
    assert verdict["status"] == "ok", verdict["deviations"]
    assert reference_failures(verdict, 0.10) == []


def test_perturbing_the_committed_reference_turns_the_check_red():
    """Every numeric step of the real reference, perturbed one at a time."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = _committed_reference()["metrics"]
    steps = _committed_steps()
    checked = 0
    for i, entry in enumerate(metrics):
        for field in ("loss", "grad_norm"):
            value = entry.get(field)
            if value is None or value != value:  # NaN handled separately
                continue
            checked += 1
            moved = _perturb(metrics, i, field)
            verdict = check_reference(moved, COMMITTED_REFERENCE, 0.10, 0.05, max_steps = steps)
            assert verdict["status"] == "out_of_band", (i, field, verdict)
            assert reference_failures(verdict, 0.10), (i, field)
            assert any(
                d["step"] == entry["step"] and d["field"] == field for d in verdict["deviations"]
            ), verdict["deviations"]
    assert checked, "the committed reference carried no numeric values"


def test_whether_the_absolute_floor_is_reached_at_all(tmp_path):
    """Is the 0.05 floor load-bearing on this trajectory, or decoration?

    The floor only does anything where |reference value| < abs_floor. The
    documented justification for it is that the late steps approach zero, so
    this asserts that justification against the committed numbers rather
    than assuming it: whichever way it comes out, the floor is checked to
    behave as claimed for the values actually present.
    """
    from run_t4_smoke import check_reference

    metrics = _committed_reference()["metrics"]
    steps = _committed_steps()
    values = [
        abs(float(m[f]))
        for m in metrics
        for f in ("loss", "grad_norm")
        if m.get(f) is not None and float(m[f]) == float(m[f])
    ]
    smallest = min(values)
    floored = [v for v in values if v < 0.05]

    if not floored:
        # The floor never engages here. Then it must be provably inert: the
        # same comparison with no floor at all must reach the same verdict.
        # If this ever fails, the floor started mattering and the reason
        # belongs in the README before anyone relies on it.
        assert smallest >= 0.05
        ref = _write_reference(tmp_path / "ref.json", metrics, steps)
        for index in range(len(metrics)):
            for field in ("loss", "grad_norm"):
                if metrics[index].get(field) is None:
                    continue
                moved = _perturb(metrics, index, field)
                assert (
                    check_reference(moved, ref, 0.10, 0.05, max_steps = steps)["status"]
                    == check_reference(moved, ref, 0.10, 0.0, max_steps = steps)["status"]
                )
    else:
        # The floor engages. Then it must be what keeps a small absolute
        # drift at those steps in band, and removing it must fail them.
        ref = _write_reference(tmp_path / "ref.json", metrics, steps)
        drifted = [dict(m) for m in metrics]
        for entry in drifted:
            value = entry.get("loss")
            if value is not None and value == value and abs(value) < 0.05:
                entry["loss"] = value + 0.004
        assert check_reference(drifted, ref, 0.10, 0.05, max_steps = steps)["status"] == "ok"
        assert check_reference(drifted, ref, 0.10, 0.0, max_steps = steps)["status"] == "out_of_band"


def test_band_failure_reaches_the_failure_list(tmp_path):
    """out_of_band must propagate to what turns the job red, not just report."""
    from run_t4_smoke import check_reference, reference_failures

    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 1.0}], max_steps = 2
    )
    verdict = check_reference(
        [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 4.0}], ref, 0.10, 0.05, max_steps = 2
    )
    assert verdict["status"] == "out_of_band"
    failures = reference_failures(verdict, 0.10)
    assert len(failures) == 1 and "outside +/-10%" in failures[0]


def test_a_length_mismatch_is_a_failure_too(tmp_path):
    """Same declared step count, different number of logged rows.

    That is the trainer logging something other than one row per step -- a
    change in the shape of the evidence, which no tolerance covers and the
    step-count guard cannot see.
    """
    from run_t4_smoke import check_reference, reference_failures

    ref = _write_reference(tmp_path / "ref.json", [{"step": 1, "loss": 1.0}], max_steps = 2)
    verdict = check_reference(
        [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 1.0}], ref, 0.10, 0.05, max_steps = 2
    )
    assert verdict["status"] == "length_mismatch"
    assert reference_failures(verdict, 0.10)


def test_matching_nan_grad_norms_are_within_band(tmp_path):
    """The reference genuinely contains NaN: fp16 scaler-skipped steps."""
    from run_t4_smoke import check_reference

    nan = float("nan")
    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 10.0, "grad_norm": nan}], max_steps = 1
    )
    verdict = check_reference(
        [{"step": 1, "loss": 10.0, "grad_norm": nan}], ref, 0.10, 0.05, max_steps = 1
    )
    assert verdict["status"] == "ok"


@pytest.mark.parametrize("swap", [False, True])
def test_a_moved_scaler_skip_pattern_is_out_of_band(tmp_path, swap):
    """NaN against a number, either way round, must NOT pass silently.

    Left to the arithmetic it would: abs(x - NaN) is NaN and NaN > tol is
    False, so a step that used to overflow and no longer does would sail
    through the one check meant to notice it.
    """
    from run_t4_smoke import check_reference

    nan = float("nan")
    ref_value, obs_value = (5.0, nan) if swap else (nan, 5.0)
    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 10.0, "grad_norm": ref_value}], max_steps = 1
    )
    verdict = check_reference(
        [{"step": 1, "loss": 10.0, "grad_norm": obs_value}], ref, 0.10, 0.05, max_steps = 1
    )
    assert verdict["status"] == "out_of_band"
    assert verdict["deviations"][0]["field"] == "grad_norm"


def test_a_field_that_stopped_being_logged_is_out_of_band(tmp_path):
    from run_t4_smoke import check_reference

    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 1.0, "grad_norm": 3.0}], max_steps = 1
    )
    verdict = check_reference([{"step": 1, "loss": 1.0}], ref, 0.10, 0.05, max_steps = 1)
    assert verdict["status"] == "out_of_band"


# -------------------------------------- did the run optimise anything at all
#
# The check that a short run needs. Under fp16 a step whose gradients
# overflow logs grad_norm NaN and is SKIPPED, and the committed 10-step
# trace has that at steps 1, 2 and 3 -- so a 3-step run of that same
# configuration applies no optimizer update whatever. Everything downstream
# still succeeds (finite loss, adapter saves, generation produces text), so
# nothing else in the harness notices.


def test_a_run_whose_every_step_was_skipped_is_a_failure():
    from run_t4_smoke import optimisation_failures

    nan = float("nan")
    failures = optimisation_failures(
        [
            {"step": 1, "loss": 10.3, "grad_norm": nan},
            {"step": 2, "loss": 10.5, "grad_norm": nan},
            {"step": 3, "loss": 9.9, "grad_norm": nan},
        ]
    )
    assert any("skipped every one of the 3 steps" in f for f in failures)


def test_the_committed_reference_trajectory_would_have_passed():
    """The same check against real data, so it is not merely strict."""
    from run_t4_smoke import optimisation_failures
    assert optimisation_failures(_committed_reference()["metrics"]) == []


def test_a_trainer_that_stops_logging_grad_norm_is_not_called_a_skip():
    """Silence is not evidence of a skipped step."""
    from run_t4_smoke import optimisation_failures

    failures = optimisation_failures([{"step": 1, "loss": 10.0}, {"step": 2, "loss": 1.0}])
    assert failures == []


def test_one_applied_step_is_enough_for_the_skip_check():
    from run_t4_smoke import optimisation_failures

    nan = float("nan")
    failures = optimisation_failures(
        [{"step": 1, "loss": 10.0, "grad_norm": nan}, {"step": 2, "loss": 1.0, "grad_norm": 42.0}]
    )
    assert not any("skipped every" in f for f in failures)


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ([{"step": 1, "loss": float("nan"), "grad_norm": 1.0}], "non-finite"),
        (
            [
                {"step": 1, "loss": 1.0, "grad_norm": 1.0},
                {"step": 2, "loss": 2.0, "grad_norm": 1.0},
            ],
            "did not decrease",
        ),
    ],
)
def test_the_other_optimisation_checks_still_fire(metrics, expected):
    from run_t4_smoke import optimisation_failures
    assert any(expected in f for f in optimisation_failures(metrics))


# ------------------------------------------------- the fp16 loss-scale pin


class _FakeScaler:
    def __init__(
        self,
        init_scale = 65536.0,
        enabled = True,
    ):
        self._init_scale = init_scale
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._init_scale


class _FakeTrainer:
    def __init__(self, scaler):
        self.accelerator = type("A", (), {"scaler": scaler})()


def test_the_loss_scale_pin_lowers_the_starting_scale():
    from run_t4_smoke import pin_initial_loss_scale

    scaler = _FakeScaler()
    state = pin_initial_loss_scale(_FakeTrainer(scaler), 2048.0)
    assert state["applied"] is True
    assert state["before"] == 65536.0 and state["after"] == 2048.0
    assert scaler.get_scale() == 2048.0


@pytest.mark.parametrize(
    "trainer",
    [
        _FakeTrainer(None),
        _FakeTrainer(_FakeScaler(enabled = False)),
        _FakeTrainer(object()),
    ],
)
def test_the_loss_scale_pin_is_never_fatal(trainer):
    """A transformers release that moves the scaler must cost a footnote in
    the report, not a Kaggle session."""
    from run_t4_smoke import pin_initial_loss_scale

    state = pin_initial_loss_scale(trainer, 2048.0)
    assert state["applied"] is False and state["reason"]


def test_the_loss_scale_pin_does_nothing_when_not_requested():
    from run_t4_smoke import pin_initial_loss_scale

    scaler = _FakeScaler()
    state = pin_initial_loss_scale(_FakeTrainer(scaler), 0)
    assert state["applied"] is False and scaler.get_scale() == 65536.0


def test_every_setting_the_child_needs_is_forwarded_to_it():
    """The cycles run as child processes, and the forwarding list is manual.

    A setting added to train_once but not to that list is silently ignored
    on the Kaggle run while working perfectly in a single-process local
    reproduction, because the parent never runs train_once itself. Derived
    from the source rather than listed here, so it cannot go stale.
    """
    import ast

    tree = ast.parse((SMOKE_DIR / "run_t4_smoke.py").read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    read = {
        node.attr
        for node in ast.walk(functions["train_once"])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    forwarded = {
        c.value.lstrip("-").replace("-", "_")
        for c in ast.walk(functions["main"])
        if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.startswith("--")
    }

    # outdir is passed separately, per cycle, and must not be forwarded
    # verbatim: each cycle gets its own directory.
    missing = read - forwarded - {"outdir"}
    assert not missing, f"train_once reads {sorted(missing)}, child never gets it"
    assert "init_loss_scale" in read & forwarded


# ------------------------------------------------------------ kernel build


def test_built_kernel_is_valid_notebook_json_with_gpu_requested(tmp_path):
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(SMOKE_DIR),
            "--out",
            str(out),
            "--count",
            "2",
            "--unsloth-ref",
            "main",
            "--zoo-ref",
            "main",
        ],
        check = True,
        capture_output = True,
    )
    nb = json.loads(out.read_text())
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert len(nb["metadata"]["kaggle_t4_ci"]["payloads"]) == 2
    for cell in nb["cells"]:
        assert cell["cell_type"] == "code"

    # The token must never be capable of reaching the kernel: nothing in the
    # notebook may reference a credential environment variable.
    blob = out.read_text()
    for forbidden in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_ACCESS_TOKEN_GH",
    ):
        assert forbidden not in blob, f"{forbidden} leaked into the kernel"


def test_built_kernel_pins_one_gpu_per_payload_and_isolates_installs(tmp_path):
    """The three details that previous sweeps proved are load-bearing."""
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(SMOKE_DIR),
            "--out",
            str(out),
            "--count",
            "2",
        ],
        check = True,
        capture_output = True,
    )
    source = "".join("".join(c["source"]) for c in json.loads(out.read_text())["cells"])
    assert 'env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)' in source
    assert "--seed" in source and "--system-site-packages" in source
    assert 'env["UV_SYSTEM_PYTHON"] = "0"' in source


def _build(
    tmp_path,
    *extra,
    payload_dir: Path = SMOKE_DIR,
) -> dict:
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(payload_dir),
            "--out",
            str(out),
            "--count",
            "2",
            *extra,
        ],
        check = True,
        capture_output = True,
    )
    return json.loads(out.read_text())


# The argument list .github/workflows/kaggle-t4-notebook-ci.yml actually
# passes. Tested verbatim rather than approximated: the SyntaxError that cost
# a Kaggle session lived only on the --reference branch, which is the branch
# the workflow always takes and the one no local build had ever exercised.
WORKFLOW_ARGS = (
    "--unsloth-ref",
    "main",
    "--zoo-ref",
    "main",
    "--reference",
    "t4_qwen2.5-0.5b.json",
    "--smoke-args",
    "--max-steps 3",
    "--per-run-timeout",
    "2100",
)


def _payload_dir(tmp_path, name: str, *, with_reference: bool) -> Path:
    """A copy of the payload directory, with or without a reference file.

    Both worlds are built explicitly rather than inherited from whatever
    happens to be committed. When a reference exists it is inlined into the
    payload notebook as a fourth carried file, under a key with a directory
    separator in it, which is a different build path from the empty
    directory -- and each of them has, at some point, been the live one.
    Deriving the two cases from the repo would mean one of them silently
    stopped being covered the day the reference landed.
    """
    import shutil

    dest = tmp_path / name
    shutil.copytree(SMOKE_DIR, dest, ignore = shutil.ignore_patterns("__pycache__"))
    refs = dest / "references"
    refs.mkdir(exist_ok = True)
    for stale in refs.glob("*.json"):
        stale.unlink()
    if with_reference:
        (refs / "t4_qwen2.5-0.5b.json").write_text(
            json.dumps(
                {
                    "metrics": [
                        {"step": 1, "loss": 10.3, "grad_norm": float("nan")},
                        {"step": 2, "loss": 0.14, "grad_norm": 13.8},
                    ],
                    "environment": {"gpu_name": "Tesla T4"},
                },
                indent = 2,
            )
        )
    return dest


def _payload_notebooks(driver: dict) -> dict:
    """The payload notebooks carried inline in the driver's first cell."""
    import base64
    import gzip
    import re

    source = "".join(driver["cells"][0]["source"])
    blob = re.search(r"^PAYLOADS = (\{.*?\})$", source, re.M | re.S).group(1)
    return {
        name: json.loads(gzip.decompress(base64.b64decode(data)))
        for name, data in json.loads(blob).items()
    }


def _every_generated_cell(driver: dict):
    """(notebook name, cell index, source) for the driver and both payloads.

    The payloads are the point. They are not files on disk anywhere; they
    exist only gzipped and base64'd inside the driver's first cell, so
    nothing that inspects the built kernel as a notebook can see them, and
    that is precisely where both generated-code defects have landed so far.
    """
    for name, nb in {"driver": driver, **_payload_notebooks(driver)}.items():
        for index, cell in enumerate(nb["cells"]):
            yield name, index, "".join(cell["source"])


def _build_all_paths(tmp_path):
    """Every code path build_kernel.py has, keyed by name."""
    return {
        # No band check at all.
        "no-reference": _build(tmp_path / "a"),
        # --reference named but the file not present: what the workflow did
        # before a green T4 run supplied one, and what it does again for any
        # configuration that has no reference yet.
        "workflow-reference-absent": _build(
            tmp_path / "b",
            *WORKFLOW_ARGS,
            payload_dir = _payload_dir(tmp_path, "empty", with_reference = False),
        ),
        # --reference named and present: the reference is carried inline as a
        # fourth file, under a key with a directory separator in it.
        "workflow-reference-present": _build(
            tmp_path / "c",
            *WORKFLOW_ARGS,
            payload_dir = _payload_dir(tmp_path, "full", with_reference = True),
        ),
    }


def test_generated_cells_compile(tmp_path):
    """Every generated cell must parse as Python, on every code path.

    This is not hypothetical. The reference argument was generated as a
    shell fragment (' --reference "..."') and spliced into the middle of a
    Python list literal, so the payload's run cell was a SyntaxError. It was
    on the path the workflow always takes, and it cost a real Kaggle session
    to find, because nothing between writing the cell and executing it on a
    T4 ever tried to parse it. The bug lived only in the branch that was
    never built locally, which is why all three branches are built here and
    why the workflow's own argument list is used verbatim.
    """
    seen = 0
    for path, driver in _build_all_paths(tmp_path).items():
        for name, index, source in _every_generated_cell(driver):
            compile(source, f"{path}/{name}#cell{index}", "exec")
            seen += 1
    # 3 paths x (3 driver cells + 2 payloads x 4 cells). Asserted so a
    # refactor that stops reaching the payloads cannot leave this test
    # passing while compiling nothing that matters.
    assert seen == 33, seen


def _undefined_names(source: str, already_bound: set[str]) -> tuple[set, set]:
    """Names a cell reads without binding, and the names it binds.

    Deliberately scope-blind: every binding anywhere in the cell counts as
    available everywhere in it. That direction of inaccuracy yields false
    negatives, never false positives, which is the only tolerable direction
    for a check that gates a launch.
    """
    import ast
    import builtins

    tree = ast.parse(source)
    bound = set(already_bound) | set(dir(builtins))
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else read).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return read - bound, bound


def test_no_generated_cell_reads_a_name_nothing_defines(tmp_path):
    """Parsing is necessary and not sufficient.

    A template hole that substitutes to a bare identifier parses perfectly
    and dies at run time with a NameError, which on Kaggle costs the same
    session a SyntaxError does. Cells are checked in execution order with
    the bindings of the cells before them carried forward, because that is
    how a notebook actually runs.
    """
    for path, driver in _build_all_paths(tmp_path).items():
        for nb_name, nb in {"driver": driver, **_payload_notebooks(driver)}.items():
            carried: set[str] = set()
            for index, cell in enumerate(nb["cells"]):
                missing, bound = _undefined_names("".join(cell["source"]), carried)
                assert not missing, (
                    f"{path}/{nb_name} cell {index} reads undefined " f"{sorted(missing)}"
                )
                carried = bound


def test_the_files_the_payload_carries_are_byte_identical_to_the_repo(tmp_path):
    """Decode the carried blobs the way the kernel will, and compare.

    The payload sources reach the T4 only as gzip+base64 inside a generated
    cell. If that encoding ever drifted, the kernel would run something
    other than what is committed, and every assertion downstream would be
    about the wrong file.
    """
    import base64
    import gzip
    import re

    payload_dir = _payload_dir(tmp_path, "full", with_reference = True)
    driver = _build(tmp_path / "c", *WORKFLOW_ARGS, payload_dir = payload_dir)
    payload = _payload_notebooks(driver)["t4_smoke_gpu0.ipynb"]
    materialise = "".join(payload["cells"][2]["source"])
    blob = re.search(r"^FILES = (\{.*?\})$", materialise, re.M | re.S).group(1)
    files = json.loads(blob)
    assert set(files) == {
        "run_t4_smoke.py",
        "determinism.py",
        "canary_dataset.jsonl",
        "references/t4_qwen2.5-0.5b.json",
    }, sorted(files)
    for name, data in files.items():
        assert gzip.decompress(base64.b64decode(data)) == (payload_dir / name).read_bytes(), name


def test_the_reference_path_the_payload_builds_is_the_one_that_is_shipped(tmp_path):
    """The runtime path must be assembled from ROOT, not left as a literal.

    The first version emitted a doubled-brace "{ROOT}/references/..." inside
    an ordinary string, so even had it parsed, the child would have been
    handed a path with a literal brace in it and reported the reference
    absent -- a band check that silently checks nothing.
    """
    driver = _build(tmp_path, "--reference", "t4_qwen2.5-0.5b.json")
    payload = _payload_notebooks(driver)["t4_smoke_gpu0.ipynb"]
    run_cell = "".join(payload["cells"][-1]["source"])
    assert (
        'cmd += ["--reference", str(ROOT / "references" / "t4_qwen2.5-0.5b.json")]'
    ) in run_cell
    assert "{ROOT}" not in run_cell


def test_the_dependency_probe_imports_unsloth_before_unsloth_zoo(tmp_path):
    """unsloth_zoo's __init__ refuses to be imported first.

    It ends with `if find_spec("unsloth") is None: raise ImportError(...)`,
    and on a real T4 that fired on a session where unsloth was installed and
    imported cleanly a moment later. Probing zoo first therefore reported a
    dependency missing that was not missing, and killed the payload.
    """
    driver = _build(tmp_path)
    payload = _payload_notebooks(driver)["t4_smoke_gpu0.ipynb"]
    verify = "".join(payload["cells"][1]["source"])
    assert "importlib.invalidate_caches()" in verify
    modules = verify.split("for mod in (", 1)[1].split("):", 1)[0]
    assert modules.index('"unsloth"') < modules.index('"unsloth_zoo"')


# --------------------------------------------------------------- workflow

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"
)


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


def test_the_workflow_parses_and_gates_the_expensive_job_on_the_cheap_one():
    wf = _workflow()
    jobs = wf["jobs"]
    assert jobs["t4-smoke"]["needs"] == "gate"
    assert "needs.gate.outputs.should_run == 'true'" in jobs["t4-smoke"]["if"]
    # A stand-down leaves t4-smoke skipped, which is grey rather than red,
    # and the gate itself exits 0 (see the gate tests above).
    assert "fork != true" in jobs["gate"]["if"]
    for job in jobs.values():
        assert job["timeout-minutes"] >= 1


def test_the_workflow_never_cancels_a_run_that_may_hold_a_kernel():
    """A cancelled runner cannot stop the kernel it already pushed."""
    wf = _workflow()
    assert wf["concurrency"]["cancel-in-progress"] is False
    assert wf["jobs"]["t4-smoke"]["concurrency"]["cancel-in-progress"] is False


def test_the_band_check_is_on_unless_a_dispatch_turns_it_off():
    """The one way to run without a band check is explicit and it warns."""
    source = WORKFLOW.read_text()
    assert "REFERENCE='t4_qwen2.5-0.5b.json'" in source
    assert 'if [ "$SKIP_BAND" = "true" ]' in source
    assert "::warning title=Reference band check disabled" in source
    # And nothing else may blank it.
    assert source.count("REFERENCE=''") == 1


def test_the_workflow_is_never_preempted_by_the_capacity_sweeper():
    """Cancelling it orphans a Kaggle kernel that then bills to its ceiling."""
    preempt = json.loads(
        (Path(__file__).resolve().parents[2] / ".github" / "ci-preempt.json").read_text()
    )
    assert WORKFLOW.name in preempt["never"]
    for machines in preempt["heavy"].values():
        assert WORKFLOW.name not in machines


# ----------------------------------------------------------------- report


@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    [("pass", 0), ("partial", 0), ("infra", 0), ("fail", 1)],
)
def test_only_a_real_assertion_failure_turns_the_job_red(tmp_path, verdict, expected_exit):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "reason": "test",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": [],
            }
        )
    )
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence), "--expect", "2"],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == expected_exit, proc.stdout


def test_a_kernel_that_reported_nothing_still_names_its_cause(tmp_path):
    """The summary alone must say why, without downloading the artifact.

    Kaggle hands the log back as a JSON array of stream records, so the
    interesting line arrives split across dozens of them. Both real
    no-report failures so far were legible only after flattening it.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": "infra",
                "reason": "no payload report",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": [],
            }
        )
    )
    (evidence / "kernel.log").write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "time": 1.0, "data": "KAGGLE_T4_CI_DRIVER start\n"},
                {"stream_name": "stdout", "time": 2.0, "data": "SyntaxError: invalid "},
                {"stream_name": "stdout", "time": 2.1, "data": "syntax\n"},
                {"stream_name": "stdout", "time": 3.0, "data": "unrelated chatter\n"},
            ]
        )
    )
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence), "--expect", "2"],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "SyntaxError: invalid syntax" in proc.stdout
    assert "unrelated chatter" not in proc.stdout


def test_the_summary_states_a_refused_reference_rather_than_an_empty_list():
    """`deviations: []` next to a refusal reads exactly like a clean pass."""
    sys.path.insert(0, str(CI_DIR))
    from report import render

    text = "\n".join(
        render(
            {
                "label": "gpu0",
                "model": "m",
                "metrics": [],
                "config": {"max_steps": 3, "init_loss_scale": 2048.0},
                "reference_check": {
                    "status": "step_count_mismatch",
                    "deviations": [],
                    "reference_max_steps": 10,
                    "observed_max_steps": 3,
                    "note": "captured at max_steps=10 and this run is 3 steps",
                },
                "failures": [
                    "refusing to band-check against a reference that is not for this run"
                ],
            }
        )
    )
    assert "step_count_mismatch" in text
    assert "captured at max_steps=10" in text
    assert "max_steps `3`" in text


def test_the_summary_says_when_the_loss_scale_pin_did_not_apply():
    sys.path.insert(0, str(CI_DIR))
    from report import render

    text = "\n".join(
        render(
            {
                "label": "gpu0",
                "model": "m",
                "metrics": [],
                "config": {"max_steps": 3, "init_loss_scale": 2048.0},
                "runs": [
                    {
                        "run_index": 0,
                        "generated": "x",
                        "canary_found": True,
                        "loss_scale": {
                            "applied": False,
                            "reason": "trainer.accelerator.scaler is absent",
                        },
                    }
                ],
                "failures": [],
            }
        )
    )
    assert "did NOT apply" in text and "scaler is absent" in text


def test_a_plain_text_kernel_log_is_handled_too(tmp_path):
    sys.path.insert(0, str(CI_DIR))
    from report import kernel_log_text

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "kernel.log").write_text("plain text log\n")
    assert kernel_log_text(evidence) == "plain text log\n"
    assert kernel_log_text(tmp_path / "nothing") == ""


def test_missing_launch_result_is_reported_but_not_red(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "NOT RUN" in proc.stdout or "did not run" in proc.stdout

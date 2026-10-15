# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only tests for the Kaggle T4 smoke harness; the training payload itself needs a T4."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _shared_setup_1():
    sys.path.insert(0, str(SMOKE_DIR))
    from run_gptoss_t4 import failures_for

    report = _gptoss_ok()
    return failures_for, report


def _shared_setup_2(stub, work):
    (stub / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for name in ("git", "sleep"):
        (stub / name).chmod(0o755)
    out = work / "github_output"
    out.write_text("", encoding = "utf-8")
    return name, out


def _shared_setup_3():
    sys.path.insert(0, str(SMOKE_DIR))
    from run_grpo_t4 import failures_for

    report = _grpo_ok()
    return failures_for, report


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))


@pytest.fixture(autouse = True)
def _keep_the_process_default_socket_timeout():
    """gate.main() sets one process-wide; the rest of the suite must not get it."""
    previous = socket.getdefaulttimeout()
    yield
    socket.setdefaulttimeout(previous)


def test_canary_dataset_targets_the_canary_and_nothing_else():
    """A row whose answer drifted would make the exact-match check vacuous."""
    rows = [
        json.loads(line)
        for line in (SMOKE_DIR / "canary_dataset.jsonl").read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "canary dataset must not be empty"
    assert all(r["answer"] == "__UNSLOTH__!!!" for r in rows)
    # Distinct questions, or the sampler's step -> row mapping is unobservable.
    assert len({r["question"] for r in rows}) == len(rows)


def test_repeating_sequential_sampler_order_is_a_function_of_the_step():
    from determinism import RepeatingSequentialSampler

    sampler = RepeatingSequentialSampler(
        dataset_length = 3, batch_size = 2, gradient_accumulation_steps = 1, max_steps = 4
    )
    assert list(sampler) == [0, 0, 1, 1, 2, 2, 0, 0]
    assert len(sampler) == 8
    # A generator consuming shared state would silently reorder the second epoch.
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


class _FakeKernel:
    def __init__(self, ref, last_run_time):
        self.ref = ref
        self.last_run_time = last_run_time


class _FakeStatus:
    def __init__(self, status):
        self.status = status


class _FakeApi:
    """Enough of the Kaggle client to drive the survey, recording which kernels
    were status-checked so a test can assert where the walk stopped."""

    def __init__(
        self,
        kernels,
        statuses = None,
        unreadable = (),
        gone = (),
    ):
        self.kernels = list(kernels)
        self.statuses = statuses or {}
        self.unreadable = set(unreadable)
        self.gone = set(gone)
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
        if ref in self.gone:
            raise RuntimeError("404 Client Error: Not Found")
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
    """A running kernel older than many newer finished ones must still be found: scan by time, not count."""
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
    # A RUNNING older than any session can last is a stale listing.
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


def test_deleted_kernels_do_not_block_a_readable_idle_account():
    """A 404 from a deleted kernel means the slot is free, so it must not block an idle account."""
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi(
        [_FakeKernel("u/gone", _ago(1)), _FakeKernel("u/done", _ago(2))], gone = ("u/gone",)
    )
    survey = survey_kernels(api, now = _now())
    assert survey["gone"] == 1 and survey["unreadable"] == 0
    assert concurrency_verdict(survey) == (True, "")


def test_one_unreadable_status_stands_the_job_down():
    """One unreadable in-window kernel stands the job down, since it may be a human's session."""
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi(
        [_FakeKernel("u/maybe", _ago(1)), _FakeKernel("u/done", _ago(2))],
        unreadable = ("u/maybe",),
    )
    survey = survey_kernels(api, now = _now())
    assert survey["unreadable"] == 1 and survey["busy"] == []
    clear, why = concurrency_verdict(survey)
    assert clear is False and "unknown" in why


def _busy(*refs) -> dict:
    """A survey with the given refs in flight, split the way the gate does."""
    from gate import OWN_KERNEL_PREFIX

    busy = [f"{ref} (RUNNING)" for ref in refs]
    own = [b for b in busy if b.split("/", 1)[-1].startswith(OWN_KERNEL_PREFIX)]
    return {
        "busy": busy,
        "own": own,
        "foreign": [b for b in busy if b not in own],
        "complete": True,
        "surveyed": len(busy),
        "unreadable": 0,
        "window_hours": 13.0,
    }


def test_the_gate_knows_its_own_kernels_from_a_strangers():
    """The whole refinement rests on this classification being right."""
    from gate import survey_kernels

    api = _FakeApi(
        [
            _FakeKernel("danielhanchen/unsloth-t4-ci-deadbeef", _ago(1)),
            _FakeKernel("danielhanchen/my-own-notebook", _ago(2)),
        ],
        statuses = {
            "danielhanchen/unsloth-t4-ci-deadbeef": "RUNNING",
            "danielhanchen/my-own-notebook": "RUNNING",
        },
    )
    survey = survey_kernels(api, now = _now())
    assert survey["own"] == ["danielhanchen/unsloth-t4-ci-deadbeef (RUNNING)"]
    assert survey["foreign"] == ["danielhanchen/my-own-notebook (RUNNING)"]


def test_the_prefix_the_gate_looks_for_is_the_one_the_launcher_pushes():
    """The gate's own-kernel prefix must match what the launcher pushes, or the job stands down forever."""
    import launch
    from gate import OWN_KERNEL_PREFIX

    assert launch._slugify("unsloth t4 ci")[:32] + "-" == OWN_KERNEL_PREFIX


def test_a_single_foreign_kernel_stands_the_job_down():
    """CI yields to human use: one foreign kernel stands it down; ALLOWED_IN_FLIGHT_FOREIGN_KERNELS is 0."""
    from gate import ALLOWED_IN_FLIGHT_FOREIGN_KERNELS, concurrency_verdict

    assert ALLOWED_IN_FLIGHT_FOREIGN_KERNELS == 0
    clear, why = concurrency_verdict(_busy("danielhanchen/somebody-else"))
    assert clear is False
    assert "not this workflow's" in why and "yields" in why


def test_a_foreign_kernel_blocks_even_when_a_slot_is_free():
    """One foreign kernel leaves one slot, which the arithmetic alone would
    happily hand to a one-kernel run. The policy overrides the arithmetic."""
    from gate import concurrency_verdict

    clear, why = concurrency_verdict(_busy("danielhanchen/somebody-else"), kernels_needed = 1)
    assert clear is False and "not this workflow's" in why


def test_this_workflows_own_leftovers_still_occupy_slots():
    """A leftover kernel from this workflow still occupies a slot, counted the same as a stranger's."""
    from gate import concurrency_verdict

    leftover = "danielhanchen/unsloth-t4-ci-abc"
    clear, why = concurrency_verdict(_busy(leftover), kernels_needed = 2)
    assert clear is False
    assert "only 1" in why and "already held by this workflow" in why
    assert concurrency_verdict(_busy(leftover), kernels_needed = 1) == (True, "")


def test_an_idle_account_clears_the_kernel_this_workflow_pushes():
    """Four legs share one kernel, so this takes one slot and the other is left for the studio workflow."""
    from gate import KERNELS_PER_INVOCATION, concurrency_verdict

    assert KERNELS_PER_INVOCATION == 1
    survey = {
        "busy": [],
        "own": [],
        "foreign": [],
        "complete": True,
        "surveyed": 0,
        "unreadable": 0,
        "window_hours": 13.0,
    }
    assert concurrency_verdict(survey) == (True, "")
    from gate import MAX_CONCURRENT_GPU_KERNELS

    assert KERNELS_PER_INVOCATION <= MAX_CONCURRENT_GPU_KERNELS
    assert concurrency_verdict(survey, MAX_CONCURRENT_GPU_KERNELS + 1)[0] is False
    assert concurrency_verdict(_busy("danielhanchen/unsloth-t4-ci-abc"))[0] is True


def test_a_survey_that_ran_out_of_time_is_not_read_as_an_idle_account():
    """A survey cut short by its wall-clock budget is an incomplete survey, not an idle account."""
    from gate import concurrency_verdict, survey_kernels

    ticks = {"t": 0.0}

    class _SlowApi(_FakeApi):
        """Every status call answers, and takes 100s about it."""

        def kernels_status(self, ref):
            ticks["t"] += 100.0
            return super().kernels_status(ref)

    api = _SlowApi([_FakeKernel(f"u/k{i}", _ago(1)) for i in range(5)])
    survey = survey_kernels(api, now = _now(), budget_sec = 180, clock = lambda: ticks["t"])
    assert survey["out_of_budget"] is True
    assert survey["complete"] is False
    # The budget ran out before the other kernels, so this must not read as an idle account.
    assert survey["surveyed"] == 2 and len(api.checked) == 2
    clear, why = concurrency_verdict(survey)
    assert clear is False
    assert "wall-clock budget" in why and "unseen" in why


def test_the_survey_budget_does_not_fire_on_a_normal_walk():
    """The bound above must not stand the gate down on an account it can read."""
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi([_FakeKernel(f"u/k{i}", _ago(1)) for i in range(5)])
    survey = survey_kernels(api, now = _now())
    assert survey["out_of_budget"] is False and survey["complete"] is True
    assert concurrency_verdict(survey) == (True, "")


def test_an_account_with_no_kernels_at_all_is_clear():
    from gate import concurrency_verdict, survey_kernels

    survey = survey_kernels(_FakeApi([]), now = _now())
    assert survey["complete"] is True
    assert concurrency_verdict(survey) == (True, "")


# Every negative gate answer exits 0: not spending quota is the normal outcome.


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
    # A hard failure can exit before publishing anything, so an absent file is an answer.
    out = tmp_path / "out.txt"
    lines = out.read_text().splitlines() if out.exists() else []
    outputs = dict(line.split("=", 1) for line in lines if "=" in line)
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
    monkeypatch.setattr(
        gate, "kaggle_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    code, outputs = _run_gate(monkeypatch, tmp_path, "--force", "true")
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "RuntimeError" in outputs["reason"]
    assert "not-a-real-token" not in (tmp_path / "out.txt").read_text()


def test_the_gate_bounds_its_network_calls_before_it_makes_one(monkeypatch, tmp_path):
    """The Kaggle client has no timeout, so the socket deadline must be set before the first call."""
    import gate

    seen = {}

    def _client(*_args, **_kwargs):
        seen["timeout"] = socket.getdefaulttimeout()
        raise RuntimeError("boom")

    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    monkeypatch.setattr(gate, "kaggle_client", _client)
    code, outputs = _run_gate(monkeypatch, tmp_path, "--force", "true")
    assert code == 0 and outputs["should_run"] == "false"
    assert seen["timeout"] == gate.SOCKET_TIMEOUT_SEC


def test_the_gate_job_deadline_exceeds_the_gates_own_bound():
    """The job timeout must exceed the gate's own worst case, so it answers rather than being killed."""
    import gate

    # Per account: auth, username and quota at the socket ceiling, then one survey budget shared
    # across all accounts, plus the call in flight when it expires.
    accounts = len(gate.DEFAULT_ACCOUNT_ENVS)
    worst = (
        accounts * 3 * gate.SOCKET_TIMEOUT_SEC + gate.SURVEY_BUDGET_SEC + gate.SOCKET_TIMEOUT_SEC
    )
    before_the_gate = 120
    for path in (WORKFLOW, WORKFLOW.parent / "kaggle-t4-studio-gpu-ci.yml"):
        workflow = pytest.importorskip("yaml").safe_load(path.read_text(encoding = "utf-8"))
        timeout_s = workflow["jobs"]["gate"]["timeout-minutes"] * 60
        assert timeout_s >= worst + before_the_gate, (
            f"{path.name}: the gate can take {worst}s, the steps before it up to "
            f"{before_the_gate}s, and the job is killed at {timeout_s}s"
        )
    source = (CI_DIR / "gate.py").read_text(encoding = "utf-8")
    assert "survey_deadline - time.monotonic()" in source, "the surveys do not share one budget"


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


# An exhausted weekly quota is the one stand-down that exits nonzero, with a message for the
# PR author; everything else stays green.

# _run_gate passes --budget-hours 1 --reserve-hours 20, so 21h is the floor.
EXHAUSTED = {
    "ok": True,
    "used_hours": 27.0,
    "total_hours": 30.0,
    "remaining_hours": 3.0,
    "refresh_at": "2026-08-17T00:00:00",
}
PLENTIFUL = dict(EXHAUSTED, used_hours = 1.0, remaining_hours = 29.0)


def _idle_account():
    return {
        "busy": [],
        "own": [],
        "foreign": [],
        "surveyed": 0,
        "unreadable": 0,
        "gone": 0,
        "complete": True,
        "out_of_budget": False,
        "window_hours": 13.0,
    }


def _run_gate_against(
    monkeypatch,
    tmp_path,
    quota,
    *extra,
    survey = None,
):
    """survey is a result, or a callable meaning the survey must not be called at all."""
    import gate

    answer = _idle_account() if survey is None else survey
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    monkeypatch.setattr(gate, "kaggle_client", lambda *a, **k: object())
    # Stubbed: the gate refuses an account it cannot name, since the launcher pushes under it.
    monkeypatch.setattr(gate, "client_username", lambda api: "danielhanchen")
    monkeypatch.setattr(gate, "remaining_gpu_hours", lambda api: dict(quota))
    monkeypatch.setattr(
        gate, "survey_kernels", answer if callable(answer) else (lambda api, **k: answer)
    )
    code, outputs = _run_gate(monkeypatch, tmp_path, "--force", "true", *extra)
    summary_path = tmp_path / "summary.md"
    summary = summary_path.read_text(encoding = "utf-8") if summary_path.exists() else ""
    return code, outputs, summary


def test_an_exhausted_weekly_quota_is_the_one_red_stand_down(monkeypatch, tmp_path):
    """An exhausted quota exits nonzero and prints the sentence verbatim with remaining and refresh time."""
    import gate

    code, outputs, summary = _run_gate_against(monkeypatch, tmp_path, EXHAUSTED)
    assert code == 1
    assert outputs["should_run"] == "false"
    for text in (outputs["reason"], summary):
        assert gate.QUOTA_EXHAUSTED_MESSAGE in text
        assert "3.0h" in text and "30.0h" in text
        assert EXHAUSTED["refresh_at"] in text
    assert "not-a-real-token" not in summary


def test_the_required_sentence_is_the_one_that_was_asked_for():
    """Pinned literally: a reworded message is a different message."""
    import gate
    assert gate.QUOTA_EXHAUSTED_MESSAGE == (
        "GPU capacity exhausted - please wait until next week - you can ignore this CI failure"
    )


def test_the_exhausted_answer_costs_one_api_call_and_no_kernel(monkeypatch, tmp_path):
    """Quota is read before the survey: an exhausted account costs one call, not the whole survey budget."""

    def _must_not_survey(api):
        raise AssertionError("the survey ran after the quota was already exhausted")

    code, outputs, _ = _run_gate_against(monkeypatch, tmp_path, EXHAUSTED, survey = _must_not_survey)
    assert code == 1 and outputs["should_run"] == "false"


def test_an_unreadable_quota_is_still_a_skip(monkeypatch, tmp_path):
    """ "Unknown" is not "exhausted", and must not borrow its message."""
    import gate

    code, outputs, summary = _run_gate_against(
        monkeypatch, tmp_path, {"ok": False, "error": "no gpu_quota in quota response"}
    )
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "unknown" in outputs["reason"]
    assert gate.QUOTA_EXHAUSTED_MESSAGE not in outputs["reason"]
    assert gate.QUOTA_EXHAUSTED_MESSAGE not in summary


def test_a_busy_account_is_still_a_skip(monkeypatch, tmp_path):
    """Quota to spare and a human on the slots: nothing red about that."""
    import gate

    survey = _idle_account() | {
        "busy": ["someone/notebook (RUNNING)"],
        "foreign": ["someone/notebook (RUNNING)"],
        "surveyed": 1,
    }
    code, outputs, _ = _run_gate_against(monkeypatch, tmp_path, PLENTIFUL, survey = survey)
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "in flight" in outputs["reason"]
    assert gate.QUOTA_EXHAUSTED_MESSAGE not in outputs["reason"]


def test_a_caller_that_asks_for_soft_failure_still_gets_one(monkeypatch, tmp_path):
    """--soft-fail softens only the exit code; the reason text still states what happened."""
    import gate

    code, outputs, summary = _run_gate_against(monkeypatch, tmp_path, EXHAUSTED, "--soft-fail")
    assert code == 0
    assert outputs["should_run"] == "false"
    assert gate.QUOTA_EXHAUSTED_MESSAGE in outputs["reason"]
    assert gate.QUOTA_EXHAUSTED_MESSAGE in summary


def test_soft_failure_is_asked_for_rather_than_assumed(monkeypatch, tmp_path):
    """--soft-fail defaults off: a default that is always taken would make the red state unreachable."""
    import gate

    unasked = _run_gate_against(monkeypatch, tmp_path / "default", EXHAUSTED)[0]
    asked = _run_gate_against(monkeypatch, tmp_path / "asked", EXHAUSTED, "--soft-fail")[0]
    assert (unasked, asked) == (1, 0)
    # --no-soft-fail still turns an error in the gate itself into a failure.
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-a-real-token")
    monkeypatch.setattr(
        gate, "kaggle_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    hard, _ = _run_gate(monkeypatch, tmp_path / "hard", "--force", "true", "--no-soft-fail")
    assert hard == 1
    soft, outputs = _run_gate(monkeypatch, tmp_path / "soft", "--force", "true", "--soft-fail")
    assert soft == 0 and outputs["should_run"] == "false"


def _write_reference(path: Path, metrics: list[dict], max_steps: int) -> Path:
    """Writes config.max_steps too, because check_reference refuses a reference that lacks it."""
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
    # Tiny absolute drift on a tiny reference is huge relatively; the 0.05 floor stops it firing.
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


# A reference from a different step count must be refused loudly: the fp16 scaler skips the
# first steps, so runs of different lengths are not comparable.


def test_a_reference_from_a_different_step_count_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ten = [{"step": i, "loss": 10.0 - i} for i in range(1, 11)]
    ref = _write_reference(tmp_path / "ref.json", ten, max_steps = 10)

    verdict = check_reference(ten, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_count_mismatch"
    assert verdict["reference_max_steps"] == 10
    assert verdict["observed_max_steps"] == 3
    # The refusal must name both counts so it is distinguishable from a numeric regression.
    failures = reference_failures(verdict, 0.10)
    assert len(failures) == 1
    assert "max_steps=10" in failures[0] and "3 steps" in failures[0]


def test_a_step_count_mismatch_is_refused_even_when_the_numbers_agree(tmp_path):
    """Identical metrics pass every tolerance; only the declared max_steps catches a step-count mismatch."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = [{"step": 1, "loss": 10.0, "grad_norm": 5.0}]
    ref = _write_reference(tmp_path / "ref.json", metrics, max_steps = 10)

    verdict = check_reference(metrics, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_count_mismatch"
    assert reference_failures(verdict, 0.10)
    # No numbers from a refused comparison: empty deviations read like a pass.
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
    # One logged row per step (logging_steps=1).
    assert len(_committed_reference()["metrics"]) == steps


def test_the_workflow_step_count_and_the_payload_default_agree():
    """Workflow and payload step-count defaults must match: a reference is valid for one number only."""
    import re

    from run_t4_smoke import main  # noqa: F401  (import proves it loads)

    workflow = (REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml").read_text(
        encoding = "utf-8"
    )

    def one(pattern, text, what):
        found = re.findall(pattern, text)
        # Assert exactly one match: a reformat once silently emptied this test.
        assert len(found) == 1, f"{what}: expected exactly one match, got {found}"
        return found[0]

    dispatch_default = one(
        r"max_steps:\s*\n\s*description:.*\n\s*type:\s*string\n\s*default:\s*'(\d+)'",
        workflow,
        "workflow_dispatch default",
    )
    # The fallback appears in each step that reads the input; they must agree.
    fallbacks = re.findall(r"inputs\.max_steps \|\| (\d+)", workflow)
    assert fallbacks, "the workflow no longer defaults the dispatched step count"
    assert len(set(fallbacks)) == 1, f"the steps disagree on the default: {fallbacks}"
    fallback = fallbacks[0]
    payload = one(
        r'"--max-steps",\s*type\s*=\s*int,\s*default\s*=\s*(\d+)',
        (SMOKE_DIR / "run_t4_smoke.py").read_text(encoding = "utf-8"),
        "payload argparse default",
    )
    assert dispatch_default == fallback == payload, (dispatch_default, fallback, payload)


# Everything below perturbs a reference, including the committed T4 one, and expects red.

COMMITTED_REFERENCE = SMOKE_DIR / "references" / "t4_qwen2.5-0.5b.json"


def _committed_reference() -> dict:
    if not COMMITTED_REFERENCE.exists():
        pytest.skip("no committed T4 reference to perturb yet")
    return json.loads(COMMITTED_REFERENCE.read_text(encoding = "utf-8"))


def _perturb(
    metrics: list[dict],
    index: int,
    field: str,
    abs_floor: float = 0.05,
    factor: float = 0.5,
) -> list[dict]:
    """Shifts one value by a factor scaled by max(|value|, abs_floor), the same scale
    check_reference uses."""
    out = [dict(m) for m in metrics]
    value = float(out[index][field])
    out[index][field] = value + max(abs(value), abs_floor) * factor
    return out


def _committed_steps() -> int:
    """Read from the committed reference, so the tests survive a recapture at a different step count."""
    from run_t4_smoke import reference_step_count
    return reference_step_count(_committed_reference())


def _committed_env() -> dict:
    """Names the committed reference's card: check_reference refuses a named card against an unnamed run."""
    return _committed_reference()["environment"]


def test_the_committed_reference_matches_itself(tmp_path):
    """The floor under the next test: an unperturbed comparison is clean."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = _committed_reference()["metrics"]
    verdict = check_reference(
        metrics,
        COMMITTED_REFERENCE,
        0.10,
        0.05,
        max_steps = _committed_steps(),
        environment = _committed_env(),
    )
    assert verdict["status"] == "ok", verdict["deviations"]
    assert reference_failures(verdict, 0.10) == []


def test_the_committed_reference_names_the_dataset_it_was_captured_on():
    """The reference digests its rows, so a canary_dataset.jsonl edit cannot land without a recapture."""
    from run_t4_smoke import dataset_digest

    config = _committed_reference().get("config") or {}
    recorded = config.get("dataset_digest")
    assert recorded, (
        "the committed reference records no config.dataset_digest, so nothing "
        "establishes which rows its loss curve came from"
    )
    live = dataset_digest(SMOKE_DIR / "canary_dataset.jsonl")
    assert recorded == live, (
        "canary_dataset.jsonl has changed since the committed reference was "
        f"captured ({recorded} -> {live}). The trace is of the old rows, so it "
        "is not comparable to a run on the new ones: recapture it "
        "(references/README.md) rather than widening the band"
    )


@pytest.mark.parametrize(
    "observed_digest",
    ["0" * 64, "unreadable:FileNotFoundError"],
)
def test_a_reference_captured_on_other_rows_is_refused(observed_digest):
    """dataset_digest never returns None: a missing key lands in config_unchecked and reads as a pass."""
    from run_t4_smoke import check_reference, reference_failures

    reference = _committed_reference()
    config = dict(reference["config"])
    config["dataset_digest"] = observed_digest
    verdict = check_reference(
        reference["metrics"],
        COMMITTED_REFERENCE,
        0.10,
        0.05,
        max_steps = _committed_steps(),
        config = config,
        environment = _committed_env(),
    )
    assert verdict["status"] == "config_mismatch", verdict
    assert any(d["key"] == "dataset_digest" for d in verdict["config_differences"]), verdict
    assert reference_failures(verdict, 0.10)


def test_the_dataset_digest_ignores_formatting_but_not_content(tmp_path):
    """The digest ignores formatting but not row content, so whitespace never forces a recapture."""
    from run_t4_smoke import dataset_digest

    rows = [
        {"question": "Who am I?", "answer": "__UNSLOTH__!!!"},
        {"question": "Who are you?", "answer": "__UNSLOTH__!!!"},
    ]
    plain = tmp_path / "plain.jsonl"
    plain.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding = "utf-8")
    reformatted = tmp_path / "reformatted.jsonl"
    reformatted.write_text(
        "\n".join(
            json.dumps(dict(reversed(list(r.items()))), indent = None, separators = (", ", ": "))
            for r in rows
        )
        + "\n\n",
        encoding = "utf-8",
    )
    assert dataset_digest(plain) == dataset_digest(reformatted)

    reordered = tmp_path / "reordered.jsonl"
    reordered.write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n", encoding = "utf-8")
    assert dataset_digest(reordered) != dataset_digest(plain)

    edited = tmp_path / "edited.jsonl"
    edited.write_text(
        "\n".join(json.dumps({**r, "question": r["question"] + "?"}) for r in rows) + "\n",
        encoding = "utf-8",
    )
    assert dataset_digest(edited) != dataset_digest(plain)


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
            verdict = check_reference(
                moved,
                COMMITTED_REFERENCE,
                0.10,
                0.05,
                max_steps = steps,
                environment = _committed_env(),
            )
            assert verdict["status"] == "out_of_band", (i, field, verdict)
            assert reference_failures(verdict, 0.10), (i, field)
            assert any(
                d["step"] == entry["step"] and d["field"] == field for d in verdict["deviations"]
            ), verdict["deviations"]
    assert checked, "the committed reference carried no numeric values"


def test_whether_the_absolute_floor_is_reached_at_all(tmp_path):
    """Checks whether abs_floor ever applies to the committed trace; it only matters for values below it."""
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
        # The floor never engages here, so removing it must not change the verdict.
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
        # Here the floor engages and must be what keeps small absolute drift in band.
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
    """A row count that differs at equal step count means the log changed shape; no tolerance catches it."""
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
    """NaN against a number must fail: NaN > tol is False, so the arithmetic would let it pass silently."""
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


def test_matching_infinite_grad_norms_are_within_band(tmp_path):
    """An fp16 overflow logs infinity as readily as NaN, and the same sign on
    both sides is the unchanged case. abs(inf - inf) is NaN, so this has to be
    decided before the division rather than by it."""
    from run_t4_smoke import check_reference

    inf = float("inf")
    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 10.0, "grad_norm": inf}], max_steps = 1
    )
    verdict = check_reference(
        [{"step": 1, "loss": 10.0, "grad_norm": inf}], ref, 0.10, 0.05, max_steps = 1
    )
    assert verdict["status"] == "ok"
    assert verdict["deviations"] == []


@pytest.mark.parametrize(
    ("ref_value", "obs_value"),
    [
        (float("inf"), 5.0),
        (5.0, float("inf")),
        (float("inf"), float("-inf")),
        (float("-inf"), float("inf")),
        (float("-inf"), 5.0),
    ],
)
def test_an_overflow_that_appeared_or_cleared_is_out_of_band(tmp_path, ref_value, obs_value):
    """An overflow that appeared or cleared divides to NaN, which the band check accepted as in band."""
    from run_t4_smoke import check_reference

    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 10.0, "grad_norm": ref_value}], max_steps = 1
    )
    verdict = check_reference(
        [{"step": 1, "loss": 10.0, "grad_norm": obs_value}], ref, 0.10, 0.05, max_steps = 1
    )
    assert verdict["status"] == "out_of_band"
    assert verdict["deviations"][0]["field"] == "grad_norm"


def test_an_infinite_loss_against_a_finite_reference_is_out_of_band(tmp_path):
    """Loss, not just grad_norm: the field the band check exists for."""
    from run_t4_smoke import check_reference

    ref = _write_reference(tmp_path / "ref.json", [{"step": 1, "loss": 10.0}], max_steps = 1)
    verdict = check_reference([{"step": 1, "loss": float("inf")}], ref, 0.10, 0.05, max_steps = 1)
    assert verdict["status"] == "out_of_band"
    assert verdict["deviations"][0]["field"] == "loss"


def test_a_field_that_stopped_being_logged_is_out_of_band(tmp_path):
    from run_t4_smoke import check_reference

    ref = _write_reference(
        tmp_path / "ref.json", [{"step": 1, "loss": 1.0, "grad_norm": 3.0}], max_steps = 1
    )
    verdict = check_reference([{"step": 1, "loss": 1.0}], ref, 0.10, 0.05, max_steps = 1)
    assert verdict["status"] == "out_of_band"


# Under fp16 every overflowing step is skipped (steps 1-3 in the reference), so a short run
# can apply no update while everything else looks healthy.


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
    """Settings train_once needs must be forwarded to the child cycle, or the Kaggle run ignores them."""
    import ast

    tree = ast.parse((SMOKE_DIR / "run_t4_smoke.py").read_text(encoding = "utf-8"))
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

    # outdir is passed per cycle, not forwarded verbatim.
    missing = read - forwarded - {"outdir"}
    assert not missing, f"train_once reads {sorted(missing)}, child never gets it"
    assert "init_loss_scale" in read & forwarded


LEG_NAMES = (
    "control",
    "canary",
    "gptoss",
    "grpo",
    "default",
    # Unpinned by design (all_cards): its preamble expects 2 visible GPUs.
    "multi_gpu",
    "latest_compile",
    "vision_fla_compile",
    # Retired but kept buildable so the retirement stays reversible.
    "frontier",
)


def _build(
    tmp_path,
    legs: str = "control,canary",
    *extra,
    payload_dir: Path = SMOKE_DIR,
) -> dict:
    out = tmp_path / "kernel.ipynb"
    out.parent.mkdir(parents = True, exist_ok = True)
    subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(payload_dir),
            "--out",
            str(out),
            "--legs",
            legs,
            *extra,
        ],
        check = True,
        capture_output = True,
    )
    return json.loads(out.read_text())


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


def _cell(payload: dict, index: int) -> str:
    return "".join(payload["cells"][index]["source"])


def test_built_kernel_is_valid_notebook_json_with_gpu_requested(tmp_path):
    nb = _build(tmp_path)
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["kaggle_t4_ci"]["payloads"] == ["t4_canary.ipynb", "t4_control.ipynb"]
    for cell in nb["cells"]:
        assert cell["cell_type"] == "code"

    # No notebook cell may reference a credential environment variable.
    blob = json.dumps(nb)
    for forbidden in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_API_TOKEN_2",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_ACCESS_TOKEN_GH",
    ):
        assert forbidden not in blob, f"{forbidden} leaked into the kernel"


def test_built_kernel_pins_one_gpu_per_payload_and_isolates_installs(tmp_path):
    """Legs install different library sets into one session, so each leg needs its own venv."""
    source = "".join("".join(c["source"]) for c in _build(tmp_path)["cells"])
    assert 'env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)' in source
    assert "--seed" in source and "--system-site-packages" in source
    assert 'env["UV_SYSTEM_PYTHON"] = "0"' in source


def test_an_unknown_leg_fails_at_build_time(tmp_path):
    """A typo in a workflow input must cost a runner second, not a session."""
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(SMOKE_DIR),
            "--out",
            str(tmp_path / "k.ipynb"),
            "--legs",
            "control,typo",
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode != 0
    assert "unknown leg" in proc.stderr and "typo" in proc.stderr


# Control and canary must differ only in installed versions; derived from built notebooks.


def test_the_control_and_canary_legs_differ_only_in_what_they_install(tmp_path):
    payloads = _payload_notebooks(_build(tmp_path, "control,canary"))
    control = payloads["t4_control.ipynb"]
    canary = payloads["t4_canary.ipynb"]

    control_run, canary_run = _cell(control, 3), _cell(canary, 3)
    assert "run_t4_smoke.py" in control_run and "run_t4_smoke.py" in canary_run
    # Anything that changes training must be absent from both or present in both.
    for knob in (
        "--max-steps",
        "--learning-rate",
        "--batch-size",
        "--lora-r",
        "--optim",
        "--model",
        "--dataset",
    ):
        assert (knob in control_run) == (knob in canary_run), knob
    assert "--pins" in control_run and "--pins" not in canary_run
    assert "--reference" in control_run and "--reference" not in canary_run

    assert _cell(control, 1) != _cell(canary, 1)


def test_the_control_leg_installs_the_committed_pins_verbatim(tmp_path):
    """Pins expand at build time, so the notebook states its versions without running anything."""
    from legs import _read_pins

    pins = _read_pins(SMOKE_DIR / "pins" / "control.txt")
    assert pins, "the control pin file names no versions"
    install = _cell(_payload_notebooks(_build(tmp_path))["t4_control.ipynb"], 1)
    for pin in pins:
        assert "==" in pin, pin
        assert json.dumps(pin) in install, pin


def test_the_canary_leg_upgrades_in_one_resolution_with_the_zoo_requirement(tmp_path):
    """Upgrades must share one resolution with the zoo requirement, or pip installs versions zoo forbids."""
    import re

    from legs import CANARY_UPGRADES

    install = _cell(_payload_notebooks(_build(tmp_path))["t4_canary.ipynb"], 1)
    groups = json.loads(re.search(r"^GROUPS = (\[.*?\])$", install, re.M | re.S).group(1))
    upgrade = [g for g in groups if "--upgrade" in g]
    assert len(upgrade) == 1, groups
    assert any("unsloth-zoo" in item for item in upgrade[0]), upgrade
    for package in CANARY_UPGRADES:
        assert package in upgrade[0], package


def test_the_canary_leg_band_checks_against_nothing(tmp_path):
    """Canary has no reference band, since another library set gives a different fp16 trajectory."""
    from legs import LEGS

    assert LEGS["canary"].reference == ""
    assert LEGS["control"].reference == "t4_qwen2.5-0.5b.json"
    canary = _payload_notebooks(_build(tmp_path))["t4_canary.ipynb"]
    assert "--reference" not in _cell(canary, 3)


def test_every_leg_carries_the_version_recorder(tmp_path):
    """A red leg that cannot name its library set is unactionable."""
    from legs import COMMON_FILES, LEGS

    assert "versions.py" in COMMON_FILES
    for name in LEGS:
        payload = _payload_notebooks(_build(tmp_path / name, name))[f"t4_{LEGS[name].name}.ipynb"]
        assert "versions.py" in _cell(payload, 0)
        assert "versions.flatten_versions" in _cell(payload, 2)


def test_every_registered_leg_is_either_carried_or_explicitly_unwired():
    """Each leg is carried by exactly one kernel or listed in UNWIRED with a reason, never dropped."""
    from legs import KERNELS, LEGS, MAX_LEGS_PER_KERNEL, UNWIRED

    carried = [name for kernel in KERNELS for name in kernel]
    assert len(carried) == len(set(carried)), carried
    assert sorted(carried + list(UNWIRED)) == sorted(LEGS), (
        sorted(carried),
        sorted(UNWIRED),
        sorted(LEGS),
    )
    assert not set(carried) & set(UNWIRED)
    for kernel in KERNELS:
        assert 1 <= len(kernel) <= MAX_LEGS_PER_KERNEL, kernel
    for name, reason in UNWIRED.items():
        assert name in LEGS, name
        assert len(reason) > 200, name


def test_an_unwired_leg_still_builds():
    """It is unwired because its INSTALL does not work on the image, not because
    the payload rots: a leg that stopped building would be rediscovered only by
    whoever next tries to switch it on."""
    from legs import UNWIRED
    for name in UNWIRED:
        assert name in LEG_NAMES, (
            f"{name} is unwired but not in the build coverage list, so "
            f"nothing checks that it still generates valid cells"
        )


def _build_all_paths(tmp_path):
    """Every leg, plus the reference-off branch of the build."""
    paths = {name: _build(tmp_path / name, name) for name in LEG_NAMES}
    # The no-reference path is how a recapture is dispatched.
    paths["control-no-reference"] = _build(tmp_path / "noref", "control", "--skip-reference")
    return paths


def test_generated_cells_compile(tmp_path):
    """Generated cells must parse: a shell fragment spliced into a Python list once cost a Kaggle
    session."""
    seen = 0
    for path, driver in _build_all_paths(tmp_path).items():
        for name, nb in {"driver": driver, **_payload_notebooks(driver)}.items():
            for index, cell in enumerate(nb["cells"]):
                compile("".join(cell["source"]), f"{path}/{name}#cell{index}", "exec")
                seen += 1
    # Derived from LEG_NAMES so a refactor that stops reaching the payloads cannot pass vacuously.
    assert seen == (len(LEG_NAMES) + 1) * 7, seen


def _undefined_names(source: str, already_bound: set) -> tuple:
    """Any binding counts everywhere in its cell, so it can miss a name but never invent one."""
    import ast
    import builtins

    tree = ast.parse(source)
    bound = set(already_bound) | set(dir(builtins))
    read: set = set()
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
    """Parsing is not enough: an undefined name fails at run time, so cells are checked in order."""
    for path, driver in _build_all_paths(tmp_path).items():
        for nb_name, nb in {"driver": driver, **_payload_notebooks(driver)}.items():
            carried: set = set()
            for index, cell in enumerate(nb["cells"]):
                missing, bound = _undefined_names("".join(cell["source"]), carried)
                assert (
                    not missing
                ), f"{path}/{nb_name} cell {index} reads undefined {sorted(missing)}"
                carried = bound


def _drive_run_cell(
    tmp_path,
    monkeypatch,
    *,
    returncode,
    report_text = None,
    stderr = "",
):
    """Executes the generated run cell with a stub child; only the hardcoded /kaggle path is rewritten."""
    import contextlib
    import io
    import types

    payload = _payload_notebooks(_build(tmp_path, "control"))["t4_control.ipynb"]
    source = _cell(payload, 3)
    outdir = tmp_path / "payload_out"
    assert "/kaggle/working/t4_out_control" in source
    source = source.replace("/kaggle/working/t4_out_control", str(outdir))

    def fake_run(cmd, *a, **kw):
        outdir.mkdir(parents = True, exist_ok = True)
        if report_text is not None:
            (outdir / "t4_smoke_report.json").write_text(report_text, encoding = "utf-8")
        return types.SimpleNamespace(returncode = returncode, stdout = "", stderr = stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, "run_cell", "exec"), {"__name__": "__main__"})  # noqa: S102

    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok = True)
    (evidence / "kernel.log").write_text(buffer.getvalue(), encoding = "utf-8")
    from launch import extract_reports

    return buffer.getvalue(), extract_reports(evidence)


def test_the_re_emitted_report_is_one_line_the_launcher_can_parse(tmp_path, monkeypatch):
    """The fallback re-emits the report as one line; the launcher scans whole lines, not indented JSON."""
    written = json.dumps(
        {"label": "control", "model": "unsloth/Qwen2.5-0.5B-Instruct", "passed": True},
        indent = 2,
    )
    assert "\n" in written
    stdout, reports = _drive_run_cell(tmp_path, monkeypatch, returncode = 0, report_text = written)
    assert len(reports) == 1, stdout
    assert reports[0]["passed"] is True
    assert reports[0]["model"] == "unsloth/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize("returncode", [139, 1, -9])
def test_a_payload_that_crashed_without_a_report_is_reported_as_failed(
    tmp_path, monkeypatch, returncode
):
    """A crash with no report must count as a failed verdict, not infra or partial, which both exit
    green."""
    stdout, reports = _drive_run_cell(
        tmp_path, monkeypatch, returncode = returncode, stderr = "CUDA error: an illegal memory access"
    )
    assert "NO USABLE REPORT WRITTEN" in stdout
    assert len(reports) == 1, stdout
    assert reports[0]["passed"] is False
    assert reports[0]["returncode"] == returncode
    assert reports[0]["label"] == "control"
    assert any(str(returncode) in f for f in reports[0]["failures"])
    assert "illegal memory access" in reports[0]["stderr_tail"]


def test_an_unreadable_report_file_is_a_failure_rather_than_a_silence(tmp_path, monkeypatch):
    """A truncated write is the same situation as no write at all."""
    stdout, reports = _drive_run_cell(
        tmp_path, monkeypatch, returncode = 0, report_text = '{"label": "control", "pas'
    )
    assert "REPORT UNREADABLE" in stdout
    assert len(reports) == 1 and reports[0]["passed"] is False


def _drive_verify_cell(
    tmp_path,
    monkeypatch,
    *,
    import_raises = None,
    on_module = "transformers",
    pip_check = "",
):
    """Runs the verify cell against stubbed torch and environment; a missing report would still exit 0."""
    import contextlib
    import importlib
    import io
    import types

    payload = _payload_notebooks(_build(tmp_path, "control"))["t4_control.ipynb"]
    source = _cell(payload, 2)

    cuda = types.SimpleNamespace(
        device_count = lambda: 1,
        is_available = lambda: True,
        get_device_name = lambda _index: "Tesla T4",
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda = cuda))

    def fake_import(name, *args, **kwargs):
        if import_raises is not None and name == on_module:
            raise import_raises
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(
            # pip check is nonzero for the whole image, not just this leg.
            returncode = 1 if pip_check else 0,
            stdout = pip_check,
            stderr = "",
        ),
    )

    buffer = io.StringIO()
    raised: BaseException | None = None
    with contextlib.redirect_stdout(buffer):
        try:
            exec(compile(source, "verify_cell", "exec"), {"__name__": "__main__"})  # noqa: S102
        except BaseException as exc:  # noqa: BLE001
            raised = exc

    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok = True)
    (evidence / "kernel.log").write_text(buffer.getvalue(), encoding = "utf-8")
    from launch import extract_reports

    return raised, buffer.getvalue(), extract_reports(evidence)


def test_a_dependency_that_exits_the_process_on_import_still_leaves_a_verdict(
    tmp_path, monkeypatch
):
    """SystemExit is a BaseException, so an import-time sys.exit escapes the handlers that catch
    Exception."""
    raised, stdout, reports = _drive_verify_cell(
        tmp_path, monkeypatch, import_raises = SystemExit("no supported accelerator")
    )
    assert "KAGGLE_T4_CI_PAYLOAD MISSING" in stdout
    assert len(reports) == 1, stdout
    assert reports[0]["passed"] is False
    assert any("transformers: SystemExit" in f for f in reports[0]["failures"])
    assert isinstance(raised, SystemExit)


def test_an_interrupted_probe_is_not_reported_as_a_missing_dependency(tmp_path, monkeypatch):
    """KeyboardInterrupt is the runner cancelling, so it must not be recorded as a missing dependency."""
    raised, _stdout, reports = _drive_verify_cell(
        tmp_path, monkeypatch, import_raises = KeyboardInterrupt()
    )
    assert isinstance(raised, KeyboardInterrupt)
    assert reports == []


def test_a_declared_requirement_the_environment_lacks_is_a_verdict(tmp_path, monkeypatch):
    """The import probe misses requirements no import reaches, so pip check must also fail the leg."""
    line = "unsloth 2026.8.15 requires nest-asyncio, which is not installed."
    raised, stdout, reports = _drive_verify_cell(tmp_path, monkeypatch, pip_check = line + "\n")

    assert "REQUIREMENTS_UNSATISFIED" in stdout
    assert len(reports) == 1, stdout
    assert reports[0]["passed"] is False
    assert any(line in f for f in reports[0]["failures"])
    assert isinstance(raised, SystemExit)


def test_another_distributions_conflict_is_not_this_legs_verdict(tmp_path, monkeypatch):
    """Only pip check lines naming the distribution under test count; other packages' conflicts do not."""
    raised, stdout, reports = _drive_verify_cell(
        tmp_path,
        monkeypatch,
        pip_check = (
            "unsloth-zoo 2026.8.10 has requirement transformers<=5.5.0, "
            "but you have transformers 5.15.0.\n"
        ),
    )
    assert raised is None, stdout
    assert reports == []


def test_the_sources_are_materialised_before_the_first_install(tmp_path):
    """Sources are written before the first install, which needs the pin file carried in the notebook."""
    payload = _payload_notebooks(_build(tmp_path))["t4_control.ipynb"]
    assert "FILES = {" in _cell(payload, 0)
    assert "pip(group)" in _cell(payload, 1)


def test_the_files_the_payload_carries_are_byte_identical_to_the_repo(tmp_path):
    """Carried gzip+base64 files must decode byte-identical to the repo, or the kernel runs other code."""
    import base64
    import gzip
    import re

    payload = _payload_notebooks(_build(tmp_path))["t4_control.ipynb"]
    blob = re.search(r"^FILES = (\{.*?\})$", _cell(payload, 0), re.M | re.S).group(1)
    files = json.loads(blob)
    expected = (
        {
            "versions.py",
            "canary_dataset.jsonl",
            "training_evidence.py",
            "phase_timers.py",
            "run_t4_smoke.py",
            "determinism.py",
            "gguf_export.py",
            "naive_trl_compare.py",
            "kernel_provenance.py",
            "pins/control.txt",
            "references/t4_qwen2.5-0.5b.json",
        },
        sorted(files),
    )
    # Also cross-checked against the registry so the literal and LEGS cannot drift.
    from legs import LEGS

    leg = LEGS["control"]
    declared = set(leg.files) | ({f"references/{leg.reference}"} if leg.reference else set())
    assert set(files) == declared, sorted(declared)
    for name, data in files.items():
        assert gzip.decompress(base64.b64decode(data)) == (SMOKE_DIR / name).read_bytes(), name


def test_runtime_paths_are_assembled_from_root_rather_than_interpolated(tmp_path):
    """Paths are built from ROOT: a literal {ROOT} made the reference look absent, so nothing was
    checked."""
    run = _cell(_payload_notebooks(_build(tmp_path))["t4_control.ipynb"], 3)
    assert 'str(ROOT / "references" / "t4_qwen2.5-0.5b.json")' in run
    assert 'str(ROOT / "pins" / "control.txt")' in run
    assert "{ROOT}" not in run


def test_the_dependency_probe_imports_unsloth_before_unsloth_zoo(tmp_path):
    """Probe unsloth before unsloth_zoo: zoo's __init__ reports unsloth missing when probed first."""
    from legs import LEGS

    verify = _cell(_payload_notebooks(_build(tmp_path))["t4_control.ipynb"], 2)
    assert "importlib.invalidate_caches()" in verify
    for leg in LEGS.values():
        assert leg.imports.index("unsloth") < leg.imports.index("unsloth_zoo")


def test_the_grpo_leg_probes_vllm_before_it_spends_the_session(tmp_path):
    """vLLM can install cleanly on GPUs it has no kernels for, so probe its import before the session."""
    from legs import LEGS

    assert "vllm" in LEGS["grpo"].imports
    verify = _cell(_payload_notebooks(_build(tmp_path / "g", "grpo"))["t4_grpo.ipynb"], 2)
    assert '"vllm"' in verify


def test_every_leg_resolves_the_dependencies_of_the_package_under_test(tmp_path):
    """Every leg resolves the package under test's dependencies: --no-deps would leave them unchecked."""
    from legs import LEGS, PACKAGE_UNDER_TEST, UNSLOTH, expand_install

    requirement = UNSLOTH.format(unsloth_ref = "abc123", zoo_ref = "def456")
    for name, leg in LEGS.items():
        groups = expand_install(leg, unsloth_ref = "abc123", zoo_ref = "def456", payload_dir = SMOKE_DIR)
        owning = [g for g in groups if requirement in g]
        assert len(owning) == 1, f"{name} installs the tested commit {len(owning)} times: {groups}"
        assert "--no-deps" not in owning[0], f"{name} installs it without resolving its deps"

    # The verify cell must ask pip about the same name pip installs.
    assert requirement.startswith(PACKAGE_UNDER_TEST + " @"), requirement
    verify = _cell(_payload_notebooks(_build(tmp_path))["t4_control.ipynb"], 2)
    assert f"OWNER = {json.dumps(PACKAGE_UNDER_TEST)}" in verify


def test_the_grpo_leg_installs_vllm_before_anything_pulls_torch(tmp_path):
    """vLLM pins torch. Resolving it last walks torch backwards under a
    stack that is already installed against the newer one."""
    from legs import LEGS

    groups = LEGS["grpo"].install
    assert any("vllm" in item for item in groups[0]), groups


# The image's torch: every vLLM pinning another torch died before a training step.
KAGGLE_IMAGE_TORCH = "2.10.0"

# vLLM releases pinning exactly KAGGLE_IMAGE_TORCH (PyPI metadata, 2026-08-11); see
# legs.UNWIRED["grpo"].
VLLM_RELEASES_PINNING_IMAGE_TORCH = ("0.17.0", "0.17.1", "0.18.0", "0.18.1", "0.19.0", "0.19.1")


def _grpo_vllm_pin() -> str:
    from legs import LEGS

    pins = [i for g in LEGS["grpo"].install for i in g if i.startswith("vllm==")]
    assert len(pins) == 1, pins
    return pins[0].split("==", 1)[1]


def test_the_grpo_vllm_pin_does_not_replace_the_images_torch():
    """The vLLM pin must keep the image's torch; another pin swaps it out from under the NVIDIA runtime."""
    assert _grpo_vllm_pin() in VLLM_RELEASES_PINNING_IMAGE_TORCH


def test_the_grpo_leg_shares_the_image_now_that_it_keeps_the_images_torch():
    """The isolated venv existed only to survive replacing torch. Probe 3 spent
    about an hour of quota resolving a CUDA stack from scratch and never
    produced payload output; with nothing to replace, nothing to isolate."""
    from legs import LEGS
    assert LEGS["grpo"].system_site_packages is True


def test_the_grpo_leg_names_its_attention_backend():
    """Names TRITON_ATTN: sm_75 lacks FlashAttention and FlashInfer, so vLLM's ladder falls through
    to it."""
    from legs import LEGS
    assert LEGS["grpo"].env.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN"


def test_the_grpo_leg_disables_flashinfer_at_the_only_layer_that_holds():
    """UNSLOTH_VLLM_NO_FLASHINFER is read before patch_vllm overwrites the other flashinfer env vars."""
    from legs import LEGS
    assert LEGS["grpo"].env.get("UNSLOTH_VLLM_NO_FLASHINFER") == "1"


def test_the_grpo_leg_removes_flashinfer_and_the_removal_reaches_the_payload(tmp_path):
    """Env vars alone were not sufficient for flashinfer; the uninstall must reach the built payload."""
    from legs import LEGS

    assert set(LEGS["grpo"].uninstall) >= {"flashinfer-python", "flashinfer-cubin"}
    payload = _payload_notebooks(_build(tmp_path / "g", "grpo"))["t4_grpo.ipynb"]
    install_cell = _cell(payload, 1)
    assert "flashinfer-python" in install_cell
    assert "uninstall" in install_cell


def test_a_leg_with_nothing_to_uninstall_does_not_run_pip_uninstall(tmp_path):
    """The counterpart, so the emission is conditional rather than always-on:
    an unconditional `pip uninstall -y` with an empty list is a per-leg
    subprocess that can only cost time."""
    from legs import LEGS

    assert LEGS["control"].uninstall == ()
    payload = _payload_notebooks(_build(tmp_path / "c", "control"))["t4_control.ipynb"]
    assert "flashinfer" not in _cell(payload, 1)


def test_the_grpo_leg_asks_for_the_utilization_both_platforms_measured(tmp_path):
    """0.95 utilization was measured on Colab and Kaggle, both fitting on the first rung of the ladder."""
    from legs import LEGS

    args = LEGS["grpo"].args
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"


def test_the_grpo_leg_no_longer_carries_xformers():
    """Its vLLM backend is gone at this version, so it would be a package
    nothing selects, resolved against a torch it has opinions about."""
    from legs import LEGS
    assert not any("xformers" in i for g in LEGS["grpo"].install for i in g)


def test_every_leg_is_either_wired_or_explained():
    """The invariant that outlives any particular leg. A leg missing from both
    KERNELS and UNWIRED is one nobody runs and nobody has written down why."""
    from legs import KERNELS, LEGS, UNWIRED

    wired = {name for kernel in KERNELS for name in kernel}
    for name in LEGS:
        assert name in wired or name in UNWIRED, f"leg {name!r} is in neither KERNELS nor UNWIRED"


def test_nothing_is_both_wired_and_unwired():
    """UNWIRED is a list of open questions. An entry for a leg that already
    runs is a stale note, and a stale note is worse than none."""
    from legs import KERNELS, UNWIRED

    wired = {name for kernel in KERNELS for name in kernel}
    assert not (wired & set(UNWIRED)), sorted(wired & set(UNWIRED))


def test_an_unwired_note_says_what_is_unknown_or_what_replaced_it():
    """Each UNWIRED note states its kind: unknown, superseded, rejected, or re-measuring."""
    from legs import LEGS, UNWIRED
    for name, note in UNWIRED.items():
        if "SUPERSEDED" in note:
            named = [other for other in LEGS if other != name and other in note]
            assert named, (
                f"{name} says it was superseded and does not name what by; a "
                f"retirement nobody can trace is a deletion with extra steps"
            )
            continue
        if "MEASURED AND REJECTED" in note:
            assert "WHAT WOULD UNBLOCK IT" in note, (
                f"{name} says it was measured and rejected without saying what "
                f"would change that; a rejection with no way back is a deletion "
                f"that keeps costing a reader the time to re-derive it"
            )
            continue
        if "UNDER RE-MEASUREMENT" in note:
            assert "WHAT WOULD UNBLOCK IT" in note, (
                f"{name} says it is being re-measured without saying what would "
                f"settle it, which is a rejection with the deadline removed"
            )
            # The deciding run must be named.
            assert re.search(r"\b(ab\d+|unsloth-probe-[a-z0-9-]+)\b", note), (
                f"{name} is under re-measurement and names no run that will "
                f"decide it, so nobody can go and read the answer"
            )
            continue
        assert "STILL UNKNOWN" in note, (
            f"{name} note says none of: what is open, what replaced it, or "
            f"what would unblock it"
        )


def test_grpo_stays_unwired_while_the_illegal_memory_access_is_open():
    """grpo stays unwired: its illegal memory access is intermittent, one pass in three sessions."""
    from legs import KERNELS, UNWIRED

    assert "grpo" not in {name for kernel in KERNELS for name in kernel}
    note = UNWIRED["grpo"]
    assert "illegal memory access" in note
    for kernel_id in ("53efcc4e", "70a2f4eb", "c98f14be", "b1f23e34"):
        assert kernel_id in note, f"the note drops session {kernel_id}"
    assert "STILL UNKNOWN" in note and "--cuda-launch-blocking run is done" in note


def test_control_and_canary_still_share_a_session():
    """Control and canary share a session as a matched pair; splitting them adds an uncontrolled
    variable."""
    from legs import KERNELS
    assert any(set(k) >= {"control", "canary"} for k in KERNELS), KERNELS


def test_the_grpo_leg_keeps_the_config_that_actually_fit():
    """Notebook values OOM a 14.56GB card; these fit, and utilization 0.95 is the one measured change."""
    from legs import LEGS

    args = LEGS["grpo"].args
    for flag, value in (
        ("--gpu-memory-utilization", "0.95"),
        ("--max-seq-length", "1024"),
        ("--num-generations", "2"),
        ("--lora-rank", "16"),
    ):
        assert flag in args, f"grpo leg lost {flag}"
        assert (
            args[args.index(flag) + 1] == value
        ), f"{flag} is {args[args.index(flag) + 1]}, not the {value} that fit"
    assert "--load-in-4bit" in args


def test_the_grpo_leg_still_pins_the_vllm_that_matches_the_image():
    """The pin is chosen to match the image's torch to the patch, not to be
    old. Any other release replaces torch, which is what killed three probe
    sessions."""
    from legs import LEGS

    install = " ".join(part for group in LEGS["grpo"].install for part in group)
    assert _grpo_vllm_pin() in install


WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"
)


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def test_the_workflow_parses_and_gates_the_expensive_job_on_the_cheap_one():
    wf = _workflow()
    jobs = wf["jobs"]
    assert jobs["t4-smoke"]["needs"] == "gate"
    assert "needs.gate.outputs.should_run == 'true'" in jobs["t4-smoke"]["if"]
    # A stand-down leaves t4-smoke skipped (grey) and the gate exits 0.
    assert "fork != true" in jobs["gate"]["if"]
    for job in jobs.values():
        assert job["timeout-minutes"] >= 1


def test_the_workflow_never_cancels_a_run_that_may_hold_a_kernel():
    """A cancelled runner cannot stop the kernel it already pushed."""
    wf = _workflow()
    assert wf["concurrency"]["cancel-in-progress"] is False
    assert wf["jobs"]["t4-smoke"]["concurrency"]["cancel-in-progress"] is False


def test_the_band_check_is_on_unless_a_dispatch_turns_it_off():
    """The band check is skipped only by an explicit dispatch or a max_steps mismatch, and both warn."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert 'if [ "$SKIP_BAND" = "true" ]' in source
    assert "::warning title=Reference band check disabled" in source
    assert 'elif [ "$MAX_STEPS" != "$REF_STEPS" ]' in source
    assert "::warning title=Reference band check skipped" in source
    assert source.count("SKIP='--skip-reference'") == 2
    assert source.count("$SKIP") == 2  # the SKIP_BAND guard and the use


def test_applying_the_opt_in_label_can_start_a_run():
    """The labeled activity type must be listed explicitly, or applying the opt-in label starts nothing."""
    wf = _workflow()
    on = wf[True] if True in wf else wf["on"]
    assert "labeled" in on["pull_request"]["types"]
    for default in ("opened", "synchronize", "reopened"):
        assert default in on["pull_request"]["types"], "the defaults are lost once types is set"


def test_only_the_opt_in_label_starts_a_run(monkeypatch, tmp_path):
    """Every label event would start a run and force a session, so only the opt-in label may start one."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising = False)
    code, outputs = _run_gate(
        monkeypatch,
        tmp_path / "unrelated",
        "--event-action",
        "labeled",
        "--event-label",
        "documentation",
        "--labels",
        "documentation,kaggle-t4-ci",
    )
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "not the opt-in label" in outputs["reason"]

    code, outputs = _run_gate(
        monkeypatch,
        tmp_path / "optin",
        "--event-action",
        "labeled",
        "--event-label",
        "kaggle-t4-ci",
        "--labels",
        "kaggle-t4-ci",
    )
    assert code == 0
    assert outputs["should_run"] == "false"
    assert "fork" in outputs["reason"], "it stood down on the label rather than on the token"


def test_a_push_or_a_synchronize_is_not_affected_by_the_label_check(monkeypatch, tmp_path):
    """Only a `labeled` run is judged on which label arrived."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising = False)
    for action in ("synchronize", "opened", ""):
        code, outputs = _run_gate(
            monkeypatch,
            tmp_path / f"a{action}",
            "--event-action",
            action,
            "--labels",
            "kaggle-t4-ci",
        )
        assert code == 0
        assert "not the opt-in label" not in outputs["reason"], action


def test_the_workflow_tells_the_gate_which_label_arrived():
    """The check above is worth nothing if the event never reaches it."""
    steps = _workflow()["jobs"]["gate"]["steps"]
    decide = next(s for s in steps if s.get("id") == "decide")
    assert decide["env"]["EVENT_ACTION"] == "${{ github.event.action }}"
    assert decide["env"]["EVENT_LABEL"] == "${{ github.event.label.name }}"
    assert '--event-action "$EVENT_ACTION"' in decide["run"]
    assert '--event-label "$EVENT_LABEL"' in decide["run"]
    # Label names are free text, so they travel through the environment, not the shell.
    assert "${{ github.event.label.name }}" not in decide["run"]
    assert "${{ join(github.event.pull_request.labels" not in decide["run"]


def test_a_dispatched_ref_is_resolved_to_one_commit():
    """Resolve a dispatched ref to one commit, since each leg pip-installs independently and could
    diverge."""
    workflow = _workflow()
    steps = workflow["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    ref = steps[names.index("Resolve the ref under test")]
    assert ref["env"]["UNSLOTH_REF"] == "${{ inputs.unsloth_ref }}"
    # Resolved once by the gate; this job reuses that answer.
    gate_ref = next(s for s in workflow["jobs"]["gate"]["steps"] if s.get("id") == "ref")
    assert "git ls-remote https://github.com/unslothai/unsloth" in gate_ref["run"]
    assert "for attempt in 1 2 3" in gate_ref["run"]
    # A full commit needs no resolving and ls-remote would not answer for one.
    assert "^[0-9a-f]{40}$" in gate_ref["run"]
    assert ref["env"]["GATE_SHA"] == "${{ needs.gate.outputs.head_sha }}"
    assert "$(git ls-remote" not in ref["run"]
    # An unpinnable ref stands the run down instead of installing a moving branch.
    assert "stand_down=true" in ref["run"]
    for name in (
        "Pin the zoo revision",
        "Build the kernel notebooks",
        "Recheck the Kaggle account",
    ):
        assert "steps.ref.outputs.stand_down != 'true'" in steps[names.index(name)]["if"], name
    assert "${{ inputs.unsloth_ref }}" not in ref["run"]


@pytest.mark.skipif(shutil.which("bash") is None, reason = "the step is a bash script")
def test_the_resolve_step_pins_every_shape_of_ref_it_can_be_given(tmp_path):
    """EXECUTE the step, with git stubbed, rather than reading it.

    Pattern-matching a shell script passes on one that runs and writes the wrong
    thing, and this one has four branches: no input, a mutable branch or tag, a
    commit needing no resolution, and a ref that resolves to nothing.
    """
    steps = _workflow()["jobs"]["gate"]["steps"]
    script = next(s for s in steps if s.get("id") == "ref")["run"]

    def drive(
        unsloth_ref,
        ls_remote,
        head = "headsha",
    ):
        work = tmp_path / f"case{abs(hash((unsloth_ref, ls_remote)))}"
        stub = work / "bin"
        stub.mkdir(parents = True)
        # Stub `git` returns LS_OUT and `sleep` returns at once.
        (stub / "git").write_text("#!/bin/sh\nprintf '%s' \"$LS_OUT\"\n")
        name, out = _shared_setup_2(stub, work)
        env = dict(
            os.environ,
            PATH = f"{stub}:{os.environ['PATH']}",
            GITHUB_OUTPUT = str(out),
            UNSLOTH_REF = unsloth_ref,
            HEAD_SHA = head,
            LS_OUT = ls_remote,
        )
        done = subprocess.run(["bash", "-c", script], env = env, capture_output = True, text = True)
        assert done.returncode == 0, done.stderr
        return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)

    assert drive("", "") == {"head_sha": "headsha"}
    main_sha = "dead" + "0" * 36
    assert drive("main", f"{main_sha}\trefs/heads/main\n") == {"head_sha": main_sha}
    # Annotated tags list the tag object first; the commit is on the ^{} line.
    tag, commit = "aaaa" + "0" * 36, "bbbb" + "0" * 36
    assert drive("v1.2", f"{tag}\trefs/tags/v1.2\n{commit}\trefs/tags/v1.2^{{}}\n") == {
        "head_sha": commit
    }
    assert drive("f" * 40, "") == {"head_sha": "f" * 40}
    # An empty key, which the GPU job turns into a stand-down.
    assert drive("no-such-branch", "") == {"head_sha": ""}
    assert drive("'; touch pwned; echo '", "") == {"head_sha": ""}
    assert not (tmp_path / "pwned").exists()

    gpu_script = next(s for s in _workflow()["jobs"]["t4-smoke"]["steps"] if s.get("id") == "ref")[
        "run"
    ]

    def drive_gpu(
        unsloth_ref,
        gate_sha,
        fetch_exit = 0,
    ):
        work = tmp_path / f"gpu{abs(hash((unsloth_ref, gate_sha, fetch_exit)))}"
        stub = work / "bin"
        stub.mkdir(parents = True)
        (stub / "git").write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            '  ls-remote) echo "ls-remote must not be called" >&2; exit 97 ;;\n'
            '  fetch) exit "$GIT_FETCH_EXIT" ;;\n'
            "  *) exit 0 ;;\n"
            "esac\n"
        )
        name, out = _shared_setup_2(stub, work)
        env = dict(
            os.environ,
            PATH = f"{stub}:{os.environ['PATH']}",
            GITHUB_OUTPUT = str(out),
            UNSLOTH_REF = unsloth_ref,
            HEAD_SHA = "headsha",
            GATE_SHA = gate_sha,
            GIT_FETCH_EXIT = str(fetch_exit),
        )
        done = subprocess.run(["bash", "-c", gpu_script], env = env, capture_output = True, text = True)
        assert done.returncode == 0, done.stderr
        return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)

    assert drive_gpu("", "") == {"ref": "headsha"}
    assert drive_gpu("main", main_sha) == {"ref": main_sha}
    assert drive_gpu("main", "") == {"stand_down": "true"}
    assert drive_gpu("main", main_sha, fetch_exit = 128) == {"stand_down": "true"}


def test_the_harness_stays_on_the_checked_out_tree_when_a_ref_is_dispatched():
    """A dispatched ref varies only the package under test; the harness stays on the checked-out tree."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@"))
    assert "inputs.unsloth_ref" not in json.dumps(checkout)
    build = next(s for s in steps if s.get("id") == "build")
    assert "--unsloth-ref '${{ steps.ref.outputs.ref }}'" in build["run"]


def test_packaging_metadata_is_watched_by_both_triggers():
    """Every payload installs the commit under test as a distribution."""
    wf = _workflow()
    on = wf[True] if True in wf else wf["on"]
    assert "pyproject.toml" in on["pull_request"]["paths"]
    assert "pyproject.toml" in on["push"]["paths"]


def _launcher_constant(name: str) -> int:
    """Reads a named constant from launch.py's source, so a rename fails here instead of dropping a term."""
    launch = (CI_DIR / "launch.py").read_text(encoding = "utf-8")
    match = re.search(rf"^{name} = (\d+)", launch, re.M)
    assert match, f"launch.py no longer defines {name}, so the job deadline cannot be derived"
    return int(match.group(1))


def _one_delete_seconds() -> int:
    """Worst-case wall clock of one delete_kernel(), counting every retry and the backoff between them."""
    attempts = _launcher_constant("DELETE_ATTEMPTS")
    backoff = _launcher_constant("DELETE_BACKOFF_SEC")
    ceiling = _launcher_constant("DELETE_SUBPROCESS_TIMEOUT_SEC")
    return attempts * ceiling + sum(backoff * 2**i for i in range(attempts - 1))


def _launcher_worst_case_seconds() -> int:
    """Launch.py's worst case: every push retry, the polling, evidence download and release() all count."""
    push_attempts = _launcher_constant("PUSH_ATTEMPTS")
    push_backoff = _launcher_constant("PUSH_BACKOFF_SEC")
    push_ceiling = _launcher_constant("PUSH_SUBPROCESS_TIMEOUT_SEC")
    one_delete = _one_delete_seconds()

    # push() calls _discard() before every retry, adding PUSH_ATTEMPTS - 1 deletes.
    per_push = (
        push_attempts * push_ceiling
        + sum(push_backoff * 2**i for i in range(push_attempts - 1))
        + (push_attempts - 1) * one_delete
    )
    source = WORKFLOW.read_text(encoding = "utf-8")
    max_wait = int(re.search(r"--max-wait (\d+)", source).group(1))
    kernels = _kernels_per_invocation()
    # One evidence budget for all kernels, read from launch.py's constant.
    evidence = _launcher_constant("EVIDENCE_BUDGET_SEC")
    deletions = kernels * push_attempts * one_delete
    # The polling deadline starts before the first push, so the two do not stack.
    return max(kernels * per_push, max_wait) + evidence + deletions


def _kernels_per_invocation() -> int:
    """How many of Kaggle's session slots one invocation takes, per the gate."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    kernels = {int(k) for k in re.findall(r"--kernels (\d+)", source)}
    assert len(kernels) == 1, kernels
    return kernels.pop()


def test_the_job_deadline_exceeds_the_launchers_worst_case():
    """The job timeout must exceed the launcher's worst case, or release() can be killed unfinished."""
    worst = _launcher_worst_case_seconds()
    # An allowance for steps before the launcher; the launcher itself refuses to push without
    # the whole of `worst` left.
    before_the_launcher = 900
    timeout_s = _workflow()["jobs"]["t4-smoke"]["timeout-minutes"] * 60
    assert timeout_s >= worst + before_the_launcher, (
        f"the launcher can take {worst}s, the steps before it up to "
        f"{before_the_launcher}s, and the job is killed at {timeout_s}s"
    )


def test_the_launcher_agrees_with_the_deadline_about_its_own_worst_case():
    """launch.py's own worst-case check and the job deadline are computed independently and must agree."""
    import launch

    source = WORKFLOW.read_text(encoding = "utf-8")
    max_wait = int(re.search(r"--max-wait (\d+)", source).group(1))
    assert (
        launch.worst_case_seconds(max_wait, _kernels_per_invocation())
        == _launcher_worst_case_seconds()
    )


def test_the_launcher_is_told_when_the_job_is_killed():
    """Start epoch is recorded in the first step and passed on; minutes must match timeout-minutes."""
    job = _workflow()["jobs"]["t4-smoke"]
    steps = job["steps"]
    assert "JOB_START_EPOCH=$(date +%s)" in steps[0].get("run", ""), (
        "the job's start is not recorded by its first step, so every step ahead "
        "of it is invisible to the launcher's guard"
    )
    assert "if" not in steps[0], "a conditional start would leave JOB_START_EPOCH unset"
    launch_step = next(s for s in steps if s.get("id") == "launch")
    assert "--deadline-epoch" in launch_step["run"]
    assert "JOB_START_EPOCH + JOB_TIMEOUT_MINUTES * 60" in launch_step["run"]
    assert int(launch_step["env"]["JOB_TIMEOUT_MINUTES"]) == job["timeout-minutes"]


def test_the_reserved_budget_covers_every_billable_launcher_phase():
    """--budget-hours must cover the launcher's whole worst case, not just the polling window."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    budgets = {int(b) for b in re.findall(r"--budget-hours (\d+)", source)}
    assert len(budgets) == 1, budgets
    budget_s = budgets.pop() * 3600
    worst = _kernels_per_invocation() * _launcher_worst_case_seconds()
    assert (
        budget_s >= worst
    ), f"the gate reserves {budget_s}s of quota and the launcher can bill {worst}s"


def test_the_account_is_rechecked_after_the_concurrency_slot_is_held():
    """The survey goes stale while a queued run waits for its slot, so it is rechecked on taking it."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    assert "Recheck the Kaggle account" in names
    recheck = steps[names.index("Recheck the Kaggle account")]
    assert recheck["id"] == "recheck"
    # --force skips only the sampling draw; the gate already rolled it.
    assert "--force true" in recheck["run"]
    assert "--reserve-hours" in recheck["run"] and "--kernels" in recheck["run"]
    # Nothing that spends a session runs unless approved; test the dispatch condition, not step order.
    dispatch = steps[names.index("Dispatch to Kaggle")]
    assert "steps.recheck.outputs.should_run == 'true'" in dispatch["if"]


def test_the_exhausted_quota_failure_reaches_the_pull_request():
    """The gate job cannot swallow the Decide exit code, or an exhausted quota reads as green."""
    jobs = _workflow()["jobs"]
    gate_job = jobs["gate"]
    decide = next(s for s in gate_job["steps"] if s.get("id") == "decide")
    assert "continue-on-error" not in gate_job
    assert "continue-on-error" not in decide
    assert "|| true" not in decide["run"]
    assert "--soft-fail" not in decide["run"]
    assert jobs["t4-smoke"]["needs"] == "gate"
    assert "always()" not in jobs["t4-smoke"]["if"]


def test_the_recheck_stands_down_rather_than_reporting_the_quota_twice():
    """The recheck uses --soft-fail: the gate already reported the quota, so it adds no second red."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    recheck = steps[names.index("Recheck the Kaggle account")]
    assert "--soft-fail" in recheck["run"]
    assert "continue-on-error" not in recheck
    warn = steps[names.index("Report the stale approval")]
    assert "steps.recheck.outputs.should_run != 'true'" in warn["if"]
    assert "::warning" in warn["run"]


def test_the_workflow_states_the_failure_semantics_it_actually_has():
    """The comment block is what a reader trusts instead of reading gate.py."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    semantics = source.split("FAILURE SEMANTICS", 1)[1].split("CREDENTIALS", 1)[0]
    assert "WEEKLY accelerator quota is exhausted" in semantics
    assert "before any kernel" in semantics
    assert "--soft-fail" in semantics
    assert "UNREADABLE" in semantics
    assert "Red ONLY when a payload ran" not in source


def test_the_harness_suite_runs_before_any_kernel_is_pushed():
    """Nothing else collects it: pyproject limits testpaths to tests/security."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    assert names.index("Test the harness") < names.index("Build the kernel notebooks")
    assert "python -m pytest tests/kaggle -q" in steps[names.index("Test the harness")]["run"]


def test_the_cpu_torch_wheel_is_installed_before_anything_that_depends_on_it():
    """CPU torch must come first: peft would pull the CUDA build, and pip then leaves that torch in
    place."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    lines = [
        l.strip() for l in steps[names.index("Test the harness")]["run"].splitlines() if l.strip()
    ]
    installs = [l for l in lines if "pip install" in l]
    assert installs, lines
    assert "download.pytorch.org/whl/cpu" in installs[0], (
        "the CPU-index torch install must be the FIRST pip install in the step: " f"{installs}"
    )
    assert "torch" in installs[0]
    # Check the outcome too: a CUDA build satisfies every import in this job.
    guard = [l for l in lines if "torch.version.cuda" in l]
    assert guard, lines
    run_suites = [l for l in lines if l.startswith("python -m pytest")]
    assert run_suites and lines.index(guard[0]) < lines.index(run_suites[0]), lines


def test_every_cpu_suite_in_the_directory_is_collected_by_that_step():
    """pyproject limits default discovery to tests/security, so each suite here must be named explicitly."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    run = steps[names.index("Test the harness")]["run"]
    suites = sorted(p.name for p in (Path(__file__).resolve().parent).glob("test_*.py"))
    assert len(suites) >= 3, suites
    argument = run.split("python -m pytest ")[1].split()[0]
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", argument, "--collect-only", "-q"],
        capture_output = True,
        text = True,
        cwd = Path(__file__).resolve().parents[2],
    )
    assert collected.returncode == 0, collected.stdout[-2000:]
    for suite in suites:
        assert f"tests/kaggle/{suite}" in collected.stdout, f"{suite} is collected by nothing"


def test_every_leg_installs_one_pinned_zoo_commit():
    """Zoo is not in pins/control.txt, so each leg would install whatever main was at its own pip run."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    pins = steps[names.index("Pin the zoo revision")]
    assert "git ls-remote" in pins["run"]
    build = steps[names.index("Build the kernel notebooks")]
    assert "--zoo-ref '${{ steps.pins.outputs.zoo_ref }}'" in build["run"]
    assert "--zoo-ref main" not in build["run"]


def test_an_unresolvable_zoo_commit_stands_the_run_down():
    """A zoo pin that cannot resolve stands the run down; falling back to main lets legs diverge."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    pins = steps[names.index("Pin the zoo revision")]
    assert "ZOO_REF=main" not in pins["run"], "the mutable branch fallback is back"
    assert "stand_down=true" in pins["run"]
    assert "for attempt in 1 2 3" in pins["run"]
    for name in ("Build the kernel notebooks", "Recheck the Kaggle account"):
        assert "steps.pins.outputs.stand_down != 'true'" in steps[names.index(name)]["if"]
    launch_step = steps[names.index("Dispatch to Kaggle")]
    assert "steps.recheck.outputs.should_run == 'true'" in launch_step["if"]


def test_a_step_count_that_could_only_report_red_stands_the_run_down():
    """Non-integer, zero or negative max_steps must stand down before a Kaggle session is spent."""
    sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"))
    import check_steps

    for bad in ("foo", "", "0", "-4", "3.5", "1e3"):
        stood_down, reason = check_steps.decide(bad, SMOKE_DIR)
        assert stood_down, f"{bad!r} would have been pushed"
        assert reason
    # The floor is derived from the committed reference and the payload's own verdict.
    from run_t4_smoke import optimisation_failures

    metrics = check_steps.reference_metrics(SMOKE_DIR)
    floor = check_steps.minimum_steps(metrics, optimisation_failures)
    assert floor and 1 < floor <= len(metrics)
    assert optimisation_failures(metrics[: floor - 1]), "the floor is not the shortest passing run"
    assert not optimisation_failures(metrics[:floor])
    assert check_steps.decide(str(floor - 1), SMOKE_DIR)[0] is True
    assert check_steps.decide(str(floor), SMOKE_DIR)[0] is False
    # CI's step count must be above the floor, or every unsampled run stands down.
    assert check_steps.decide("10", SMOKE_DIR)[0] is False
    assert check_steps.minimum_steps([], optimisation_failures) is None
    assert check_steps.decide("10", REPO_ROOT / "tests" / "kaggle")[0] is True


def test_the_dispatched_step_count_is_checked_before_a_kernel_is_paid_for():
    """Validating it inside the payload is validating it after the bill."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    assert "Validate the dispatched step count" in names
    check = steps[names.index("Validate the dispatched step count")]
    assert check["id"] == "stepcount"
    assert "check_steps.py" in check["run"]
    assert names.index("Validate the dispatched step count") < names.index(
        "Build the kernel notebooks"
    )
    for name in (
        "Build the kernel notebooks",
        "Recheck the Kaggle account",
        "Report the stale approval",
    ):
        assert "steps.stepcount.outputs.stand_down != 'true'" in steps[names.index(name)]["if"]
    # Through the environment, after the check: the dispatched value is free text.
    build = steps[names.index("Build the kernel notebooks")]
    assert check["env"]["MAX_STEPS"].startswith("${{ inputs.max_steps")
    for step in (check, build):
        assert "inputs.max_steps" not in step["run"]
    assert "--max-steps $MAX_STEPS" in build["run"]
    # The build step's count source is tested by executing it in
    # test_an_alternate_spelling_of_the_step_count_keeps_the_reference_band.


@pytest.mark.skipif(shutil.which("bash") is None, reason = "the step is a bash script")
def test_an_alternate_spelling_of_the_step_count_keeps_the_reference_band(tmp_path):
    """EXECUTE the build step, on the count the VALIDATOR produced.

    The band is the only thing in this workflow that compares a run against a
    committed trace, and the only way it goes off silently is this comparison
    saying two identical step counts differ. `check_steps.parse_steps` accepts
    `+10`, `010` and surrounding whitespace as the ten they are -- so does the
    payload's argparse, which is what actually runs -- so a raw string compare
    here dropped the band from a run that was the reference's run, and said so
    in a warning naming a step count that was not different.

    Every arm goes through the real code path: the step's own `env:` block says
    where each value comes from, the count is whatever running check_steps.py
    writes to GITHUB_OUTPUT, and the assertion is on the argv the step hands
    build_kernel.py. Reading the wiring rather than supplying it is the point --
    a build step that goes back to the raw dispatch string fails here.
    """
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    step = steps[names.index("Build the kernel notebooks")]
    script, wiring = step["run"], step["env"]
    checker = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci" / "check_steps.py"
    cases = iter(range(1000))

    def resolve(key: str, validated: dict, skip_band: str) -> str:
        """What GitHub would put in `key`, from the step's own env block."""
        expr = str(wiring[key]).strip()
        inner = expr.removeprefix("${{").removesuffix("}}").strip()
        if inner == "inputs.skip_reference_band":
            return skip_band
        source, _, name = inner.rpartition(".")
        if source == "steps.stepcount.outputs":
            return validated.get(name, "")
        # On pull_request `inputs` is null and schedule is false, so GitHub substitutes "".
        if inner.startswith("inputs.") or inner.startswith("github."):
            return ""
        raise AssertionError(
            f"the build step takes {key} from {expr}. The step count it builds with, "
            f"and the reference count it compares against, have to be the ones the "
            f"validator parsed: any second reading of the dispatched string is a "
            f"second answer to how long this run is, and the band goes off when the "
            f"two disagree."
        )

    def validate(raw: str) -> dict:
        """The validator step, run for real, and the outputs it wrote."""
        out = tmp_path / f"validated{next(cases)}"
        out.write_text("", encoding = "utf-8")
        done = subprocess.run(
            [sys.executable, str(checker), "--max-steps", raw, "--payload-dir", str(SMOKE_DIR)],
            env = dict(os.environ, GITHUB_OUTPUT = str(out)),
            capture_output = True,
            text = True,
        )
        assert done.returncode == 0, done.stderr
        return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)

    def build(raw: str, skip_band: str = "false") -> tuple[list[str], str]:
        """The build step's shell on those outputs; build_kernel's argv and the log."""
        validated = validate(raw)
        assert validated["stand_down"] == "false", raw
        work = tmp_path / f"build{next(cases)}"
        stub = work / "bin"
        stub.mkdir(parents = True)
        argv = work / "argv"
        (stub / "python").write_text(
            f'#!/bin/sh\nfor arg in "$@"; do printf "%s\\n" "$arg"; done > "{argv}"\n'
        )
        (stub / "python").chmod(0o755)
        done = subprocess.run(
            ["bash", "-c", script],
            env = dict(
                os.environ,
                PATH = f"{stub}:{os.environ['PATH']}",
                **{key: resolve(key, validated, skip_band) for key in wiring},
            ),
            capture_output = True,
            text = True,
        )
        assert done.returncode == 0, done.stderr
        return argv.read_text().splitlines(), done.stdout

    sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"))
    import check_steps

    committed = check_steps.reference_steps(SMOKE_DIR)
    assert committed, "the committed reference declares no step count"

    # Derived from the committed reference so a recapture moves this test with it.
    for spelling in (str(committed), f"+{committed}", f"0{committed}", f" {committed} "):
        args, log = build(spelling)
        assert "--skip-reference" not in args, (spelling, args)
        assert args[args.index("--smoke-args") + 1] == f"--max-steps {committed}", args
        assert "Reference band check" not in log, (spelling, log)

    # A genuinely different length must still drop the band, with a warning.
    other = committed + 1
    args, log = build(str(other))
    assert "--skip-reference" in args, args
    assert "::warning title=Reference band check skipped" in log, log

    args, log = build(str(committed), skip_band = "true")
    assert "--skip-reference" in args, args
    assert "::warning title=Reference band check disabled" in log, log


def test_an_evidence_upload_outage_cannot_colour_the_check_red():
    """The evidence upload is continue-on-error, so an artifact outage cannot turn the check red."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    names = [s.get("name") for s in steps]
    upload = steps[names.index("Upload evidence")]
    assert upload["continue-on-error"] is True
    assert upload["id"] == "evidence"
    # continue-on-error leaves `outcome` at failure while `conclusion` is success.
    warn = steps[names.index("Report the evidence upload failure")]
    assert "steps.evidence.outcome == 'failure'" in warn["if"]
    assert "::warning" in warn["run"]
    report = steps[names.index("Report")]
    # always() so a failed upload still reaches the verdict; gated on evidence existing, since a
    # dispatching run approves but produces none.
    assert report["if"].startswith("always()")
    assert "hashFiles('kaggle_evidence/**/*_output.ipynb')" in report["if"]
    assert "steps.evidence" not in report["if"]


def test_the_harness_and_the_package_under_test_are_one_snapshot():
    """The default pull_request checkout is the merge ref, not the head."""
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    ref_step = next(s for s in steps if s.get("id") == "ref")
    assert ref_step["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert "ref=$HEAD_SHA" in ref_step["run"]


def test_the_workflow_takes_its_kernel_plan_from_the_leg_registry():
    """Kernel plan comes from the leg registry via build outputs, so a new leg needs no YAML edit."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "--all-kernels" in source
    assert "${{ steps.build.outputs.notebooks }}" in source
    assert "steps.build.outputs.payloads" in source


def test_the_workflow_is_never_preempted_by_the_capacity_sweeper():
    """Cancelling it orphans a Kaggle kernel that then bills to its ceiling."""
    preempt = json.loads(
        (Path(__file__).resolve().parents[2] / ".github" / "ci-preempt.json").read_text(
            encoding = "utf-8"
        )
    )
    assert WORKFLOW.name in preempt["never"]
    for machines in preempt["heavy"].values():
        assert WORKFLOW.name not in machines


_TWO_LEGS = [
    {"label": "control", "passed": True, "steps": []},
    {"label": "canary", "passed": True, "steps": []},
]
_A_FAILING_LEG = [
    {"label": "control", "passed": False, "failures": ["x"], "steps": []},
    {"label": "canary", "passed": True, "steps": []},
]
_STUDIO = [{"label": "studio-gpu", "passed": False, "failures": ["x"], "assertions": []}]


@pytest.mark.parametrize(
    ("verdict", "reports", "expected_exit"),
    [
        ("pass", _TWO_LEGS, 0),
        ("partial", [], 0),
        ("infra", [], 0),
        ("fail", _A_FAILING_LEG, 1),
        # A failing Studio payload must not mark the T4 smoke FAIL; kaggle_studio_ci/report.py does.
        ("fail", _TWO_LEGS + _STUDIO, 0),
    ],
)
def test_only_a_real_assertion_failure_turns_the_job_red(tmp_path, verdict, reports, expected_exit):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "reason": "test",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": reports,
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
    """The summary must name a silent kernel's cause without the artifact, so the Kaggle log is
    flattened."""
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
                "failures": ["refusing to band-check against a reference that is not for this run"],
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


# A red canary run must be attributable to a package; checked here without a GPU.


def test_the_goal_packages_are_the_ones_this_ci_exists_to_watch():
    """The requested list, asserted so a refactor cannot quietly drop one."""
    from versions import GOAL_PACKAGES
    for package in ("trl", "transformers", "accelerate", "peft", "bitsandbytes", "torch", "vllm"):
        assert package in GOAL_PACKAGES, package


def test_the_packages_the_upgrade_groups_move_are_recorded(tmp_path):
    """Packages an upgrade group moves via dependencies must be in GOAL_PACKAGES, or they go
    unattributed."""
    from versions import GOAL_PACKAGES
    for package in ("tokenizers", "safetensors", "huggingface_hub", "datasets"):
        assert package in GOAL_PACKAGES, package


def test_the_transitive_packages_are_the_ones_the_legs_document_moving():
    """Derived from legs.py rather than a list someone kept in step by hand."""
    legs_source = (CI_DIR / "legs.py").read_text(encoding = "utf-8")
    from versions import GOAL_PACKAGES

    for package in ("tokenizers", "safetensors", "huggingface_hub"):
        assert package in legs_source, (
            f"{package} is in GOAL_PACKAGES but legs.py no longer records a "
            f"resolution moving it"
        )
        assert package in GOAL_PACKAGES


def test_a_pin_outside_the_goal_list_is_not_reported_as_missing(tmp_path):
    """The lookup is derived from the pin file, not from a fixed table.

    `pin_failures` reads whatever `resolved_versions` was asked about, so a pin
    naming a package the goal list does not carry came back "not installed" for
    a package that is installed and correct: an invented failure on a control
    leg, indistinguishable in the report from a pin that really broke.
    """
    from versions import load_pins, pin_failures, versions_for_pins

    pin_file = tmp_path / "pins.txt"
    # Certain to be installed here and not in GOAL_PACKAGES.
    import pytest as _pytest

    pin_file.write_text(f"pytest=={_pytest.__version__}\n", encoding = "utf-8")
    pins = load_pins(pin_file)

    from versions import GOAL_PACKAGES, resolved_versions

    assert "pytest" not in GOAL_PACKAGES
    stale = pin_failures(pins, resolved_versions(GOAL_PACKAGES))
    assert (
        stale and "unknown" in stale[0]
    ), "a pin nobody probed must not be reported as a version that differs"

    assert pin_failures(pins, versions_for_pins(pins)) == []


def test_a_pin_that_really_broke_is_still_a_failure(tmp_path):
    """The widening must not make the check unfailable."""
    from versions import load_pins, pin_failures, versions_for_pins

    pin_file = tmp_path / "pins.txt"
    pin_file.write_text("pytest==0.0.1\n", encoding = "utf-8")
    pins = load_pins(pin_file)
    failures = pin_failures(pins, versions_for_pins(pins))
    assert failures and "was resolved" in failures[0], failures


def test_a_distribution_whose_name_is_not_its_import_name_is_still_found():
    """`unsloth_zoo` installs as `unsloth-zoo`, and asking for the wrong one
    records "not installed" for a package that is."""
    from versions import _DISTRIBUTION
    assert _DISTRIBUTION["unsloth_zoo"] == "unsloth-zoo"


def test_a_package_that_is_installed_and_unimportable_is_not_read_as_fine():
    """Installed metadata is not health: a package that raises on import must be reported as broken."""
    from versions import flatten_versions

    flat = flatten_versions(
        {
            "vllm": {
                "installed": "0.11.2",
                "imported": "IMPORT FAILED: ImportError: libcusparseLt.so.0",
            },
            "torch": {"installed": "2.10.0"},
        }
    )
    assert flat["torch"] == "2.10.0"
    assert "IMPORT FAILED" in flat["vllm"] and "0.11.2" in flat["vllm"]


def test_a_pin_that_did_not_hold_is_a_failure():
    """A control whose pins were overridden is not a control, and every
    comparison drawn against it is wrong with nothing else showing it."""
    from versions import pin_failures

    resolved = {
        "transformers": {"installed": "5.6.0"},
        "trl": {"installed": "0.24.0"},
        "peft": {"installed": None},
    }
    failures = pin_failures({"transformers": "5.5.0", "trl": "0.24.0", "peft": "0.19.1"}, resolved)
    assert len(failures) == 2
    assert any("5.5.0" in f and "5.6.0" in f for f in failures)
    assert any("peft" in f and "not installed" in f for f in failures)
    assert pin_failures({"trl": "0.24.0"}, resolved) == []


def test_the_committed_pin_file_parses_and_names_the_canary_set():
    """The pin file and canary upgrade list must name the same set, so the legs differ only in versions."""
    sys.path.insert(0, str(CI_DIR))
    from legs import CANARY_UPGRADES
    from versions import load_pins

    pins = load_pins(SMOKE_DIR / "pins" / "control.txt")
    assert set(pins) == set(CANARY_UPGRADES), (sorted(pins), sorted(CANARY_UPGRADES))
    assert all(v and v[0].isdigit() for v in pins.values()), pins


def test_a_pin_file_line_that_is_not_a_pin_is_refused(tmp_path):
    """`transformers>=5.5` is not a pin, and silently accepting it would
    make the control leg float without saying so."""
    from versions import load_pins

    path = tmp_path / "pins.txt"
    path.write_text("transformers>=5.5.0\n")
    with pytest.raises(ValueError):
        load_pins(path)


def test_the_summary_puts_the_two_legs_library_sets_side_by_side():
    """The payoff of the pairing: the bisect is on the summary page."""
    sys.path.insert(0, str(CI_DIR))
    from report import version_table

    lines = version_table(
        [
            {
                "label": "control",
                "environment": {
                    "resolved": {"transformers": "5.5.0", "trl": "0.24.0", "torch": "2.10.0"}
                },
            },
            {
                "label": "canary",
                "versions_flat": {"transformers": "5.6.0", "trl": "0.24.0", "torch": "2.10.0"},
            },
        ]
    )
    text = "\n".join(lines)
    assert "**transformers**" in text, text
    assert "Legs differ in: transformers." in text
    assert "**trl**" not in text and "**torch**" not in text


def test_the_summary_says_so_when_the_legs_agree():
    sys.path.insert(0, str(CI_DIR))
    from report import version_table

    same = {"transformers": "5.5.0"}
    text = "\n".join(
        version_table(
            [
                {"label": "control", "versions_flat": same},
                {"label": "canary", "versions_flat": same},
            ]
        )
    )
    assert "identical across legs" in text
    assert "Legs differ in" not in text


def test_one_leg_alone_produces_no_comparison_table():
    """A table with one column is not a comparison and reads like one."""
    sys.path.insert(0, str(CI_DIR))
    from report import version_table

    assert version_table([{"label": "control", "versions_flat": {"trl": "0.24.0"}}]) == []


class _Args:
    def __init__(self, **kw):
        self.max_steps = 3
        self.require_compile = True
        self.__dict__.update(kw)


def _gptoss_ok() -> dict:
    """A report shaped like the one the probe actually produced."""
    return {
        "metrics": [
            {"step": 1, "loss": 5.76, "grad_norm": 2.1},
            {"step": 2, "loss": 4.78, "grad_norm": 1.8},
            {"step": 3, "loss": 4.03, "grad_norm": 1.6},
        ],
        "adapter_update": {
            "ok": True,
            "changed": True,
            "tensors": 168,
            "abs_sum_before": 1234.5,
            "abs_sum_after": 1240.25,
            "b_abs_sum_before": 0.0,
            "b_abs_sum_after": 5.75,
        },
        "compile": {
            "available": True,
            "unique_graphs": 32,
            "calls_captured": 779,
            "graph_breaks_total": 2,
            # A reading without the delta has no baseline, which is its own failure.
            "unique_graphs_delta": 30,
            "calls_captured_delta": 700,
            "graph_breaks_total_delta": 2,
        },
        "generated": "analysis... assistantfinal 4",
        # The float32 assertion depends on `bf16_supported`, so the fixture must set it.
        "environment": {"gpu_name": "Tesla T4", "bf16_supported": False},
        "precision": {"fp16": False, "bf16": False, "force_float32_env": "1"},
        "placement_after_load": {
            "parameters_by_device": {"cuda:0": 20_900_000_000},
            "hf_device_map_devices": None,
            "offloaded": False,
        },
    }


def test_the_gptoss_leg_passes_on_what_the_probe_measured():
    """The floor under every negative case below."""
    sys.path.insert(0, str(SMOKE_DIR))
    from run_gptoss_t4 import failures_for

    assert failures_for(_gptoss_ok(), _Args()) == []


def test_a_gptoss_run_that_never_compiled_is_a_failure():
    """Zero captured graphs must fail: loss, save and generation still pass, hiding an eager-only run."""
    failures_for, report = _shared_setup_1()
    report["compile"] = {
        "available": True,
        "unique_graphs": 32,
        "calls_captured": 779,
        "graph_breaks_total": 0,
        "unique_graphs_delta": 0,
    }
    failures = failures_for(report, _Args())
    assert any("zero graphs" in f for f in failures), failures
    assert failures_for(report, _Args(require_compile = False)) == []


def test_a_compile_check_with_no_baseline_is_refused_rather_than_assumed():
    """With no baseline the absolute graph count is the loader's and nonzero, so the check is refused."""
    failures_for, report = _shared_setup_1()
    report["compile"] = {
        "available": True,
        "unique_graphs": 32,
        "calls_captured": 779,
        "graph_breaks_total": 2,
    }
    failures = failures_for(report, _Args())
    assert any("pre-training dynamo counters" in f for f in failures), failures
    assert failures_for(report, _Args(require_compile = False)) == []


def test_unreadable_compile_counters_are_not_read_as_success():
    failures_for, report = _shared_setup_1()
    report["compile"] = {"available": False, "error": "AttributeError"}
    assert any("could not be established" in f for f in failures_for(report, _Args()))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.update(generated = "   "), "unusable"),
        (lambda r: r.update(generated = None), "did not run"),
        (lambda r: r.update(metrics = r["metrics"][:1]), "logged steps"),
        (
            lambda r: r.update(
                metrics = [
                    {"step": 1, "loss": float("nan")},
                    {"step": 2, "loss": 1.0},
                    {"step": 3, "loss": 1.0},
                ]
            ),
            "non-finite",
        ),
    ],
)
def test_the_other_gptoss_assertions_fire(mutate, expected):
    failures_for, report = _shared_setup_1()
    mutate(report)
    assert any(expected in f for f in failures_for(report, _Args())), report


# With num_iterations=1 and beta=0 the GRPO loss is zero on a healthy run; reward_std is the
# signal, since zero spread means zero advantage.


class _GrpoArgs:
    max_steps = 2


def _grpo_ok() -> dict:
    return {
        "log_history": [
            {"step": 1, "reward": 1.4, "reward_std": 0.35},
            {"step": 2, "reward": 1.6, "reward_std": 0.21},
        ],
        "metrics": [
            {"step": 1, "loss": 0.0, "grad_norm": 0.9},
            {"step": 2, "loss": 0.0, "grad_norm": 0.7},
        ],
        "adapter_update": {
            "ok": True,
            "changed": True,
            "tensors": 196,
            "abs_sum_before": 900.0,
            "abs_sum_after": 902.5,
            "b_abs_sum_before": 0.0,
            "b_abs_sum_after": 2.5,
        },
        "completions": [["forty two", "42", "about 42", "no idea"]],
        "fast_generate": "the square root of 101 is about 10.05",
    }


def test_the_grpo_leg_passes_a_healthy_run_whose_loss_is_zero():
    """Loss 0.0 on every step is the HEALTHY case here, not a failure."""
    sys.path.insert(0, str(SMOKE_DIR))
    from run_grpo_t4 import failures_for

    assert failures_for(_grpo_ok(), _GrpoArgs()) == []


def test_a_group_with_no_reward_spread_is_the_failure_that_matters():
    """Zero reward_std gives zero GRPO advantage: the optimizer applies nothing and nothing else
    flags it."""
    failures_for, report = _shared_setup_3()
    for entry in report["log_history"]:
        entry["reward_std"] = 0.0
    failures = failures_for(report, _GrpoArgs())
    assert any("zero on every step" in f for f in failures), failures
    # One step with spread is enough: a single degenerate group is normal.
    report["log_history"][0]["reward_std"] = 0.4
    assert failures_for(report, _GrpoArgs()) == []


def test_completions_that_are_all_empty_are_caught_even_when_rewards_agree():
    """N empty strings score identically, so the reward checks alone would
    call an engine that produced nothing a clean run."""
    failures_for, report = _shared_setup_3()
    report["completions"] = [["", "", "", ""]]
    assert any(
        "every one of the 4 completions was empty" in f for f in failures_for(report, _GrpoArgs())
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.update(log_history = [{"step": 1}]), "no reward was logged"),
        (lambda r: [e.pop("reward_std") for e in r["log_history"]], "never logged"),
        (lambda r: r.update(fast_generate = None, fast_generate_error = "boom"), "fast_generate"),
        (lambda r: r.update(metrics = []), "logged steps"),
    ],
)
def test_the_other_grpo_assertions_fire(mutate, expected):
    failures_for, report = _shared_setup_3()
    mutate(report)
    assert any(expected in f for f in failures_for(report, _GrpoArgs())), report


def test_probe_mode_reports_rather_than_judges():
    """--probe moves failures into observed_failures; a probe that hid them would be worse than none."""
    import ast
    for name in ("run_gptoss_t4.py", "run_grpo_t4.py"):
        tree = ast.parse((SMOKE_DIR / name).read_text(encoding = "utf-8"))
        source = (SMOKE_DIR / name).read_text(encoding = "utf-8")
        assert 'report["observed_failures"] = failures' in source, name
        assert "--probe" in source, name
        assert any(
            isinstance(n, ast.FunctionDef) and n.name == "failures_for" for n in ast.walk(tree)
        ), name


def test_the_launcher_takes_one_notebook_per_kernel():
    """Pushes precede waits: waiting between pushes serialises sessions Kaggle runs in parallel."""
    import inspect

    import launch

    source = inspect.getsource(launch.main)
    assert 'action = "append"' in source
    # Push all first, then wait; the reverse doubles wall clock silently.
    assert source.index("pushed = push(") < source.index('entry["state"] = wait(')


def test_the_reports_of_every_kernel_are_gathered(tmp_path):
    """Each kernel collects into its own directory so two cannot overwrite
    each other's kernel.log; the extraction has to walk into them."""
    import launch

    (tmp_path / "k1").mkdir()
    (tmp_path / "k2").mkdir()
    (tmp_path / "k1" / "kernel.log").write_text(
        'T4_SMOKE_REPORT {"label": "control", "model": "m", "passed": true}\n'
    )
    (tmp_path / "k2" / "kernel.log").write_text(
        'T4_SMOKE_REPORT {"label": "grpo", "model": "q", "passed": false}\n'
    )
    reports = launch.extract_reports(tmp_path)
    assert sorted(r["label"] for r in reports) == ["control", "grpo"]


def test_the_log_fallback_reads_kaggles_own_json_record_shape(tmp_path):
    """kernels/output returns log as a JSON array of records, so it must be flattened before matching."""
    import launch

    records = [
        {"stream_name": "stderr", "time": 1.0, "data": "some noise\n"},
        {
            "stream_name": "stdout",
            "time": 2.0,
            "data": 'T4_SMOKE_REPORT {"label": "control", "model": "m", "passed": false}\n',
        },
    ]
    (tmp_path / "k1").mkdir()
    (tmp_path / "k1" / "kernel.log").write_text(
        "[" + "\n,".join(json.dumps(r) for r in records) + "]", encoding = "utf-8"
    )
    reports = launch.extract_reports(tmp_path)
    assert [r["label"] for r in reports] == ["control"]
    assert reports[0]["passed"] is False


def test_a_report_split_across_log_records_is_still_read(tmp_path):
    """Kaggle chunks stdout by write, not by line, so the prefix and its
    payload can land in different records. Flattening has to join them back
    before any line splitting happens."""
    import launch

    records = [
        {"stream_name": "stdout", "time": 1.0, "data": 'T4_SMOKE_REPORT {"label": "grpo", '},
        {"stream_name": "stdout", "time": 1.1, "data": '"model": "q", "passed": true}\n'},
    ]
    (tmp_path / "k1").mkdir()
    (tmp_path / "k1" / "kernel.log").write_text(json.dumps(records), encoding = "utf-8")
    assert [r["label"] for r in launch.extract_reports(tmp_path)] == ["grpo"]


def test_a_kernel_that_could_not_be_pushed_does_not_lose_the_other(tmp_path):
    """Half a run is a warning, not a failure: half a comparison is not
    evidence of a regression."""
    sys.path.insert(0, str(CI_DIR))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": "partial",
                "reason": "only 2 of 4 payload(s) reported back",
                "kernels": [
                    {"notebook": "kernel1.ipynb", "slug": "u/a", "state": "COMPLETE"},
                    {
                        "notebook": "kernel2.ipynb",
                        "slug": None,
                        "push_error": "at_capacity: session count of 2 reached",
                    },
                ],
                "reports": [],
            }
        )
    )
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence), "--expect", "4"],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "was never pushed" in proc.stdout
    assert "at_capacity" in proc.stdout


# flashinfer JIT links with -lcuda, but the image has only libcuda.so.1, so the link fails.
# Making -lcuda resolvable fixes every JIT op at once.


def _grpo_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_grpo_t4_under_test", SMOKE_DIR / "run_grpo_t4.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shim_does_nothing_when_the_stub_is_already_there(monkeypatch):
    """Most images ship it. Touching LIBRARY_PATH anyway would be a change
    with no reason, on a machine that was already fine."""
    grpo = _grpo_module()
    monkeypatch.setattr(grpo.os.path, "exists", lambda p: "lib64" in str(p))
    monkeypatch.delenv("LIBRARY_PATH", raising = False)
    facts = grpo.make_libcuda_linkable()
    assert facts["needed"] is False and facts["applied"] is False
    assert "LIBRARY_PATH" not in grpo.os.environ


def test_the_shim_builds_a_link_when_the_stub_is_missing(monkeypatch, tmp_path):
    """The measured Kaggle case."""
    grpo = _grpo_module()
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"")

    real_exists = grpo.os.path.exists

    def exists(path):
        if "stubs" in str(path) or "compat" in str(path):
            return False
        return real_exists(path)

    monkeypatch.setattr(grpo.os.path, "exists", exists)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("LIBRARY_PATH", raising = False)

    class Done:
        stdout = f"\tlibcuda.so.1 (libc6,x86-64) => {driver}\n"

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **k: Done())

    facts = grpo.make_libcuda_linkable()
    assert facts["needed"] is True and facts["applied"] is True, facts
    from pathlib import Path as _P

    link = _P(facts["shim"]) / "libcuda.so"
    assert link.is_symlink() and link.resolve() == driver.resolve()
    assert facts["shim"] in grpo.os.environ["LIBRARY_PATH"]


def test_the_shim_keeps_an_existing_library_path(monkeypatch, tmp_path):
    """Clobbering it would break whatever set it."""
    grpo = _grpo_module()
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"")
    real_exists = grpo.os.path.exists
    monkeypatch.setattr(
        grpo.os.path,
        "exists",
        lambda p: False if ("stubs" in str(p) or "compat" in str(p)) else real_exists(p),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("LIBRARY_PATH", "/somewhere/else")

    class Done:
        stdout = f"\tlibcuda.so.1 (libc6,x86-64) => {driver}\n"

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **k: Done())

    grpo.make_libcuda_linkable()
    assert grpo.os.environ["LIBRARY_PATH"].endswith("/somewhere/else")


def test_the_shim_reports_rather_than_raises_when_there_is_no_driver(monkeypatch, tmp_path):
    """A machine with no driver at all is not a machine this can fix, and a
    payload that dies here would report nothing about GRPO."""
    grpo = _grpo_module()
    monkeypatch.setattr(grpo.os.path, "exists", lambda p: False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    class Done:
        stdout = ""

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **k: Done())
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None)

    facts = grpo.make_libcuda_linkable()
    assert facts["applied"] is False
    assert "error" in facts


def test_the_payload_applies_the_shim_before_it_touches_vllm():
    """Ordering is the whole point: flashinfer JITs on first use, and the
    first use is inside the engine build."""
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    applied = source.index("make_libcuda_linkable()")
    built = source.index('report["vllm"] = vllm_facts()')
    assert applied < built


def test_what_the_shim_did_reaches_the_report():
    """Otherwise a future green run cannot be told from one that never needed
    it, and the next person re-derives all of this."""
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    assert 'report["libcuda_shim"] = libcuda' in source


def test_the_traceback_keeps_its_head_as_well_as_its_tail():
    """The last probe's 6000-char tail was entirely ninja's own output, so the
    Python frames naming the caller were exactly what got dropped."""
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    assert "middle elided" in source


def test_a_libcuda_the_linker_will_not_search_for_does_not_count(monkeypatch, tmp_path):
    """A libcuda outside flashinfer's -L directories must not count: compat passed, then the link failed."""
    grpo = _grpo_module()
    real_exists = grpo.os.path.exists

    def exists(path):
        path = str(path)
        if "compat" in path:
            return True  # present, and useless as a -L target
        if "lib64" in path:
            return False  # the dirs that ARE searched have nothing
        return real_exists(path)

    monkeypatch.setattr(grpo.os.path, "exists", exists)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("LIBRARY_PATH", raising = False)

    class Done:
        stdout = ""

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **k: Done())
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None)

    facts = grpo.make_libcuda_linkable()
    assert facts["needed"] is True, facts
    assert "already_linkable" not in facts
    # compat is a fine symlink target even though it is a useless -L dir.
    assert facts["applied"] is True and "compat" in facts["real"], facts


def test_the_searched_directories_are_the_ones_flashinfer_passes():
    """Pinned, because widening this list is exactly how the check went wrong.
    These two are what appear as -L on the failing ninja line."""
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    assert '"/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs"' in source


def test_the_grpo_payload_gives_a_base_model_a_chat_template():
    """Base models ship no chat template and TRL raises without one, so the payload sets one."""
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    assert "tokenizer.chat_template = (" in source
    assert 'if not getattr(tokenizer, "chat_template", None):' in source
    assert 'result["chat_template"]' in source


def test_the_chat_template_the_payload_installs_actually_renders():
    """A template that does not render trades a failure at step 1 for a
    failure at step 1 with a longer traceback."""
    import re

    jinja2 = pytest.importorskip("jinja2")
    source = (SMOKE_DIR / "run_grpo_t4.py").read_text(encoding = "utf-8")
    block = source[source.index("tokenizer.chat_template = (") :]
    block = block[: block.index("\n        )")]
    literal = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', block))
    template = literal.encode().decode("unicode_escape")

    rendered = jinja2.Template(template).render(
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        add_generation_prompt = True,
    )
    assert rendered.startswith("<|im_start|>system\nS<|im_end|>")
    assert rendered.endswith("<|im_start|>assistant\n")
    # A template dropping the user turn would train on prompts the model never saw.
    assert "<|im_start|>user\nU<|im_end|>" in rendered


# The canary resolves with unsloth_zoo, whose caps hold transformers and trl back, so only the
# frontier leg can detect regressions in their newest releases.


def test_the_frontier_leg_resolves_dependencies_rather_than_skipping_them():
    """--no-deps with a blanket tokenizers upgrade overshot transformers' ceiling; let pip resolve it."""
    sys.path.insert(0, str(CI_DIR))
    import legs

    frontier = legs.LEGS["frontier"]
    upgrades = [group for group in frontier.install if "--upgrade" in group]
    assert upgrades, "the frontier leg no longer upgrades anything"
    for group in upgrades:
        assert "--no-deps" not in group, (
            f"the frontier leg upgrade {group!r} skips dependency resolution; "
            f"that is what left tokenizers above the ceiling transformers "
            f"declares and killed the leg before it ran a single step"
        )
    upgraded = {arg for group in upgrades for arg in group if not arg.startswith("-")}
    assert {"transformers", "trl"} <= upgraded, (
        f"the frontier leg upgrades {sorted(upgraded)}, and the two packages it "
        f"exists for are transformers and trl"
    )


def test_the_frontier_leg_does_not_carry_the_zoo_requirement():
    """Naming unsloth_zoo in frontier's resolution reimposes the cap that frontier measures past."""
    sys.path.insert(0, str(CI_DIR))
    import legs

    for group in legs.LEGS["frontier"].install:
        if "--upgrade" not in group:
            continue
        assert not any("unsloth-zoo" in arg or "unsloth_zoo" in arg for arg in group), (
            "the frontier leg upgrades unsloth_zoo in the same resolution as "
            "transformers and trl, so zoo's transformers<=5.5.0 and trl<=0.24.0 "
            "bind again and the leg silently becomes a second canary"
        )


def test_frontier_is_retired_in_favour_of_the_leg_that_supersedes_it():
    """Frontier is retired on purpose, superseded by vision_fla_compile; a silent drop must fail."""
    from legs import KERNELS, UNWIRED

    assert "frontier" not in {name for kernel in KERNELS for name in kernel}, (
        "frontier is back in the wired set; if that is intended, this test and "
        "its UNWIRED note both need rewriting rather than deleting"
    )
    assert "frontier" in UNWIRED, (
        "frontier is in neither KERNELS nor UNWIRED, which is what a leg "
        "dropped by accident looks like"
    )
    assert (
        "SUPERSEDED" in UNWIRED["frontier"]
    ), "the note must say it was replaced, not that it is broken"
    assert "vision_fla_compile" in UNWIRED["frontier"], "and by what"
    assert "vision_fla_compile" in {name for kernel in KERNELS for name in kernel}

    assert len(KERNELS) == 1, KERNELS


def test_the_pinned_kaggle_client_carries_the_calls_this_workflow_makes():
    """Kaggle client is pinned: older releases refuse KAGGLE_API_TOKEN and lack delete and quota_view."""
    packaging_version = pytest.importorskip("packaging.version")
    text = WORKFLOW.read_text(encoding = "utf-8")
    pins = re.findall(r"pip install [^\n]*'kaggle==([0-9][^']*)'", text)
    assert pins, "no pinned kaggle client in the workflow"
    assert len(set(pins)) == 1, f"jobs disagree on the kaggle client: {pins}"
    assert packaging_version.Version(pins[0]) >= packaging_version.Version("2.2.0"), pins[0]


@pytest.mark.skipif(shutil.which("bash") is None, reason = "the step is a bash script")
def test_a_dispatched_commit_is_proven_to_exist_before_the_quota_is_spent(tmp_path):
    """A 40-character SHA was accepted on shape alone.

    Nothing asked whether unslothai/unsloth HAS that commit, so a mistyped (or
    force-pushed away, or fork-only) SHA pushed the paid kernels, every
    payload's `pip install git+...` failed on it, and the import probes reported
    the pull request RED for a commit that never existed, which is the outcome
    stand_down exists for, quota and all.

    `git ls-remote` cannot answer it, matching refs and exiting 0 with empty
    output for any SHA, so the check is a `git fetch` of the object, the same
    reachability pip needs to install it.
    """
    steps = _workflow()["jobs"]["t4-smoke"]["steps"]
    script = next(s for s in steps if s.get("id") == "ref")["run"]

    def drive(
        unsloth_ref,
        ls_remote,
        fetch_exit = 0,
    ):
        work = tmp_path / f"case{abs(hash((unsloth_ref, ls_remote, fetch_exit)))}"
        stub = work / "bin"
        stub.mkdir(parents = True)
        # Stub `git` answers ls-remote from LS_OUT and fetch from GIT_FETCH_EXIT, recording fetches.
        (stub / "git").write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            '  ls-remote) printf "%s" "$LS_OUT" ;;\n'
            '  fetch) shift; echo "$*" >> "$FETCH_LOG"; exit "$GIT_FETCH_EXIT" ;;\n'
            "  *) exit 0 ;;\n"
            "esac\n"
        )
        name, out = _shared_setup_2(stub, work)
        log = work / "fetches"
        log.write_text("", encoding = "utf-8")
        env = dict(
            os.environ,
            PATH = f"{stub}:{os.environ['PATH']}",
            GITHUB_OUTPUT = str(out),
            UNSLOTH_REF = unsloth_ref,
            HEAD_SHA = "headsha",
            GATE_SHA = ls_remote.split("\t", 1)[0] if ls_remote else unsloth_ref,
            GIT_FETCH_EXIT = str(fetch_exit),
            FETCH_LOG = str(log),
        )
        done = subprocess.run(["bash", "-c", script], env = env, capture_output = True, text = True)
        assert done.returncode == 0, done.stderr
        written = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
        return written, log.read_text(encoding = "utf-8")

    sha = "a" * 40
    # A commit GitHub does not serve stands the run down.
    written, fetched = drive(sha, "", fetch_exit = 128)
    assert written == {"stand_down": "true"}
    assert sha in fetched, "the SHA was accepted without asking GitHub for it"

    written, fetched = drive(sha, "")
    assert written == {"ref": sha}
    assert sha in fetched

    # A branch resolved by ls-remote is also checked for fetchability.
    branch_sha = "b" * 40
    written, fetched = drive("main", f"{branch_sha}\trefs/heads/main\n", fetch_exit = 128)
    assert written == {"stand_down": "true"}
    assert branch_sha in fetched

    written, fetched = drive("", "")
    assert written == {"ref": "headsha"}
    assert fetched.strip() == ""

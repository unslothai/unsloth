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
    rows = [json.loads(line) for line
            in (SMOKE_DIR / "canary_dataset.jsonl").read_text().splitlines()
            if line.strip()]
    assert rows, "canary dataset must not be empty"
    assert all(r["answer"] == "__UNSLOTH__!!!" for r in rows)
    # Distinct questions: identical rows would make the sampler's
    # step -> row mapping unobservable.
    assert len({r["question"] for r in rows}) == len(rows)


# ------------------------------------------------------------ determinism

def test_repeating_sequential_sampler_order_is_a_function_of_the_step():
    from determinism import RepeatingSequentialSampler

    sampler = RepeatingSequentialSampler(dataset_length=3, batch_size=2,
                                         gradient_accumulation_steps=1,
                                         max_steps=4)
    assert list(sampler) == [0, 0, 1, 1, 2, 2, 0, 0]
    assert len(sampler) == 8
    # Iterating twice must give the same answer; a generator that consumed
    # shared state would silently reorder the second epoch.
    assert list(sampler) == list(sampler)


def test_compare_metrics_treats_matching_nan_as_equal():
    """fp16 skipped steps log NaN, and NaN != NaN would fail identical runs."""
    from determinism import compare_metrics

    nan = float("nan")
    a = [{"step": 1, "loss": 1.0, "grad_norm": nan},
         {"step": 2, "loss": 0.5, "grad_norm": 3.0}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": nan},
         {"step": 2, "loss": 0.5, "grad_norm": 3.0}]
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


# ----------------------------------------------------------- reference band

@pytest.mark.parametrize(
    ("observed", "expected_status"),
    [(1.00, "ok"), (1.05, "ok"), (1.50, "out_of_band")],
)
def test_reference_band_accepts_drift_and_rejects_a_real_move(
        tmp_path, observed, expected_status):
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 1.0}]}))
    verdict = check_reference([{"step": 1, "loss": observed}], ref,
                              rel_tol=0.10, abs_floor=0.05)
    assert verdict["status"] == expected_status


def test_reference_band_absolute_floor_tolerates_near_zero_losses():
    """Late steps approach zero, where a tiny absolute drift is not a regression."""
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.json"
        ref.write_text(json.dumps({"metrics": [{"step": 10, "loss": 0.0001}]}))
        # 0.002 absolute against a 0.0001 reference is a 1900% relative move,
        # but it is noise. The 0.05 floor is what stops it firing.
        verdict = check_reference([{"step": 10, "loss": 0.002}], ref,
                                  rel_tol=0.10, abs_floor=0.05)
        assert verdict["status"] == "ok"


def test_reference_absent_is_not_a_failure():
    sys.path.insert(0, str(SMOKE_DIR))
    from run_t4_smoke import check_reference

    verdict = check_reference([{"step": 1, "loss": 1.0}],
                              Path("/nonexistent/ref.json"), 0.1, 0.05)
    assert verdict["status"] == "absent"


# ------------------------------------------------------------ kernel build

def test_built_kernel_is_valid_notebook_json_with_gpu_requested(tmp_path):
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [sys.executable, str(CI_DIR / "build_kernel.py"),
         "--payload-dir", str(SMOKE_DIR), "--out", str(out),
         "--count", "2", "--unsloth-ref", "main", "--zoo-ref", "main"],
        check=True, capture_output=True)
    nb = json.loads(out.read_text())
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert len(nb["metadata"]["kaggle_t4_ci"]["payloads"]) == 2
    for cell in nb["cells"]:
        assert cell["cell_type"] == "code"

    # The token must never be capable of reaching the kernel: nothing in the
    # notebook may reference a credential environment variable.
    blob = out.read_text()
    for forbidden in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME",
                      "KAGGLE_ACCESS_TOKEN_GH"):
        assert forbidden not in blob, f"{forbidden} leaked into the kernel"


def test_built_kernel_pins_one_gpu_per_payload_and_isolates_installs(tmp_path):
    """The three details that previous sweeps proved are load-bearing."""
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [sys.executable, str(CI_DIR / "build_kernel.py"),
         "--payload-dir", str(SMOKE_DIR), "--out", str(out), "--count", "2"],
        check=True, capture_output=True)
    source = "".join("".join(c["source"]) for c in
                     json.loads(out.read_text())["cells"])
    assert 'env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)' in source
    assert "--seed" in source and "--system-site-packages" in source
    assert 'env["UV_SYSTEM_PYTHON"] = "0"' in source


# ----------------------------------------------------------------- report

@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    [("pass", 0), ("partial", 0), ("infra", 0), ("fail", 1)],
)
def test_only_a_real_assertion_failure_turns_the_job_red(
        tmp_path, verdict, expected_exit):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(json.dumps({
        "verdict": verdict, "reason": "test", "slug": "u/s",
        "kernel_state": "COMPLETE", "reports": []}))
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"),
         "--evidence", str(evidence), "--expect", "2"],
        capture_output=True, text=True)
    assert proc.returncode == expected_exit, proc.stdout


def test_missing_launch_result_is_reported_but_not_red(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence)],
        capture_output=True, text=True)
    assert proc.returncode == 0
    assert "NOT RUN" in proc.stdout or "did not run" in proc.stdout

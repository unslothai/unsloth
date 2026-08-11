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

    def __init__(self, kernels, statuses=None, unreadable=()):
        self.kernels = list(kernels)
        self.statuses = statuses or {}
        self.unreadable = set(unreadable)
        self.checked = []

    def kernels_list(self, mine=False, page=1, page_size=20, sort_by=None):
        assert mine and sort_by == "dateRun"
        start = (page - 1) * page_size
        return self.kernels[start:start + page_size]

    def kernels_status(self, ref):
        self.checked.append(ref)
        if ref in self.unreadable:
            raise RuntimeError("500")
        return _FakeStatus(
            f"KernelWorkerStatus.{self.statuses.get(ref, 'COMPLETE')}")


def _now():
    from datetime import datetime

    return datetime(2026, 8, 11, 12, 0, 0)


def _ago(hours):
    from datetime import timedelta

    return _now() - timedelta(hours=hours)


def test_survey_finds_a_running_kernel_hidden_behind_newer_finished_ones():
    """The exact hole a fixed "12 most recent" bound left open.

    A kernel that started three hours ago and is still running, with forty
    newer kernels that have since run and finished. Bounding the scan by
    COUNT misses it and the push then dies at the capacity cap; bounding it
    by the session ceiling cannot.
    """
    from gate import concurrency_verdict, survey_kernels

    kernels = [_FakeKernel(f"u/done{i}", _ago(0.5 + i * 0.01))
               for i in range(40)]
    kernels.append(_FakeKernel("u/old-runner", _ago(3)))
    api = _FakeApi(kernels, statuses={"u/old-runner": "RUNNING"})

    survey = survey_kernels(api, now=_now())
    assert survey["busy"] == ["u/old-runner (RUNNING)"]
    assert survey["complete"] is True
    clear, why = concurrency_verdict(survey)
    assert clear is False and "old-runner" in why


def test_survey_stops_at_the_session_ceiling_rather_than_walking_the_account():
    """Nothing older than a session can last is looked at, and the walk ends."""
    from gate import LOOKBACK_HOURS, survey_kernels

    kernels = [_FakeKernel("u/recent", _ago(1)),
               _FakeKernel("u/edge", _ago(LOOKBACK_HOURS - 0.1)),
               _FakeKernel("u/stale", _ago(LOOKBACK_HOURS + 0.1)),
               _FakeKernel("u/ancient", _ago(24 * 30))]
    api = _FakeApi(kernels, statuses={"u/ancient": "RUNNING"})

    survey = survey_kernels(api, now=_now())
    assert api.checked == ["u/recent", "u/edge"]
    assert survey["surveyed"] == 2
    assert survey["complete"] is True
    # A "RUNNING" older than any session can last is a stale listing, not an
    # in-flight kernel, and must not stand the job down forever.
    assert survey["busy"] == []


def test_survey_handles_timezone_aware_timestamps():
    from datetime import timezone

    from gate import survey_kernels

    aware = _ago(1).replace(tzinfo=timezone.utc)
    api = _FakeApi([_FakeKernel("u/a", aware)], statuses={"u/a": "RUNNING"})
    survey = survey_kernels(api, now=_now())
    assert survey["busy"] == ["u/a (RUNNING)"]


def test_a_kernel_with_no_timestamp_is_checked_but_does_not_end_the_walk():
    from gate import survey_kernels

    api = _FakeApi([_FakeKernel("u/undated", None),
                    _FakeKernel("u/recent", _ago(1))],
                   statuses={"u/recent": "QUEUED"})
    survey = survey_kernels(api, now=_now())
    assert api.checked == ["u/undated", "u/recent"]
    assert survey["busy"] == ["u/recent (QUEUED)"]


def test_a_survey_that_ran_out_of_pages_is_not_read_as_an_idle_account():
    from gate import concurrency_verdict, survey_kernels

    kernels = [_FakeKernel(f"u/k{i}", _ago(1)) for i in range(1000)]
    api = _FakeApi(kernels)
    survey = survey_kernels(api, now=_now(), page_size=10, max_pages=3)
    assert survey["surveyed"] == 30
    assert survey["complete"] is False
    clear, why = concurrency_verdict(survey)
    assert clear is False and "unseen" in why


def test_statuses_that_all_come_back_unreadable_are_not_read_as_idle():
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi([_FakeKernel("u/a", _ago(1)), _FakeKernel("u/b", _ago(2))],
                   unreadable=("u/a", "u/b"))
    survey = survey_kernels(api, now=_now())
    assert survey["unreadable"] == 2 and survey["busy"] == []
    clear, why = concurrency_verdict(survey)
    assert clear is False and "unknown" in why


def test_some_unreadable_statuses_do_not_block_a_readable_idle_account():
    """Deleted kernels 404 routinely; that must not wedge the gate shut."""
    from gate import concurrency_verdict, survey_kernels

    api = _FakeApi([_FakeKernel("u/gone", _ago(1)),
                    _FakeKernel("u/done", _ago(2))],
                   unreadable=("u/gone",))
    survey = survey_kernels(api, now=_now())
    assert concurrency_verdict(survey) == (True, "")


def test_an_account_with_no_kernels_at_all_is_clear():
    from gate import concurrency_verdict, survey_kernels

    survey = survey_kernels(_FakeApi([]), now=_now())
    assert survey["complete"] is True
    assert concurrency_verdict(survey) == (True, "")


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


def _perturb(metrics: list[dict], index: int, field: str,
             abs_floor: float = 0.05, factor: float = 0.5) -> list[dict]:
    """Move one value by half a band-width more than the band allows.

    Scaled by the same max(|value|, abs_floor) the check uses, so the
    perturbation is out of band wherever on the curve it is applied.
    """
    out = [dict(m) for m in metrics]
    value = float(out[index][field])
    out[index][field] = value + max(abs(value), abs_floor) * factor
    return out


def test_the_committed_reference_matches_itself(tmp_path):
    """The floor under the next test: an unperturbed comparison is clean."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = _committed_reference()["metrics"]
    verdict = check_reference(metrics, COMMITTED_REFERENCE, 0.10, 0.05)
    assert verdict["status"] == "ok", verdict["deviations"]
    assert reference_failures(verdict, 0.10) == []


def test_perturbing_the_committed_reference_turns_the_check_red():
    """Every numeric step of the real reference, perturbed one at a time."""
    from run_t4_smoke import check_reference, reference_failures

    metrics = _committed_reference()["metrics"]
    checked = 0
    for i, entry in enumerate(metrics):
        for field in ("loss", "grad_norm"):
            value = entry.get(field)
            if value is None or value != value:  # NaN handled separately
                continue
            checked += 1
            moved = _perturb(metrics, i, field)
            verdict = check_reference(moved, COMMITTED_REFERENCE, 0.10, 0.05)
            assert verdict["status"] == "out_of_band", (i, field, verdict)
            assert reference_failures(verdict, 0.10), (i, field)
            assert any(d["step"] == entry["step"] and d["field"] == field
                       for d in verdict["deviations"]), verdict["deviations"]
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
    values = [abs(float(m[f])) for m in metrics for f in ("loss", "grad_norm")
              if m.get(f) is not None and float(m[f]) == float(m[f])]
    smallest = min(values)
    floored = [v for v in values if v < 0.05]

    if not floored:
        # The floor never engages here. Then it must be provably inert: the
        # same comparison with no floor at all must reach the same verdict.
        # If this ever fails, the floor started mattering and the reason
        # belongs in the README before anyone relies on it.
        assert smallest >= 0.05
        ref = tmp_path / "ref.json"
        ref.write_text(json.dumps({"metrics": metrics}))
        for index in range(len(metrics)):
            for field in ("loss", "grad_norm"):
                if metrics[index].get(field) is None:
                    continue
                moved = _perturb(metrics, index, field)
                assert (check_reference(moved, ref, 0.10, 0.05)["status"]
                        == check_reference(moved, ref, 0.10, 0.0)["status"])
    else:
        # The floor engages. Then it must be what keeps a small absolute
        # drift at those steps in band, and removing it must fail them.
        ref = tmp_path / "ref.json"
        ref.write_text(json.dumps({"metrics": metrics}))
        drifted = [dict(m) for m in metrics]
        for entry in drifted:
            value = entry.get("loss")
            if value is not None and value == value and abs(value) < 0.05:
                entry["loss"] = value + 0.004
        assert check_reference(drifted, ref, 0.10, 0.05)["status"] == "ok"
        assert check_reference(drifted, ref, 0.10, 0.0)["status"] == "out_of_band"


def test_band_failure_reaches_the_failure_list(tmp_path):
    """out_of_band must propagate to what turns the job red, not just report."""
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 10.0},
                                           {"step": 2, "loss": 1.0}]}))
    verdict = check_reference([{"step": 1, "loss": 10.0},
                               {"step": 2, "loss": 4.0}], ref, 0.10, 0.05)
    assert verdict["status"] == "out_of_band"
    failures = reference_failures(verdict, 0.10)
    assert len(failures) == 1 and "outside +/-10%" in failures[0]


def test_a_length_mismatch_is_a_failure_too(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 1.0}]}))
    verdict = check_reference([{"step": 1, "loss": 1.0},
                               {"step": 2, "loss": 1.0}], ref, 0.10, 0.05)
    assert verdict["status"] == "length_mismatch"
    assert reference_failures(verdict, 0.10)


def test_matching_nan_grad_norms_are_within_band(tmp_path):
    """The reference genuinely contains NaN: fp16 scaler-skipped steps."""
    from run_t4_smoke import check_reference

    nan = float("nan")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 10.0,
                                            "grad_norm": nan}]}))
    verdict = check_reference([{"step": 1, "loss": 10.0, "grad_norm": nan}],
                              ref, 0.10, 0.05)
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
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 10.0,
                                            "grad_norm": ref_value}]}))
    verdict = check_reference(
        [{"step": 1, "loss": 10.0, "grad_norm": obs_value}], ref, 0.10, 0.05)
    assert verdict["status"] == "out_of_band"
    assert verdict["deviations"][0]["field"] == "grad_norm"


def test_a_field_that_stopped_being_logged_is_out_of_band(tmp_path):
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"metrics": [{"step": 1, "loss": 1.0,
                                            "grad_norm": 3.0}]}))
    verdict = check_reference([{"step": 1, "loss": 1.0}], ref, 0.10, 0.05)
    assert verdict["status"] == "out_of_band"


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


def _build(tmp_path, *extra) -> dict:
    out = tmp_path / "kernel.ipynb"
    subprocess.run(
        [sys.executable, str(CI_DIR / "build_kernel.py"),
         "--payload-dir", str(SMOKE_DIR), "--out", str(out), "--count", "2",
         *extra],
        check=True, capture_output=True)
    return json.loads(out.read_text())


def _payload_notebooks(driver: dict) -> dict:
    """The payload notebooks carried inline in the driver's first cell."""
    import base64
    import gzip
    import re

    source = "".join(driver["cells"][0]["source"])
    blob = re.search(r"^PAYLOADS = (\{.*?\})$", source, re.M | re.S).group(1)
    return {name: json.loads(gzip.decompress(base64.b64decode(data)))
            for name, data in json.loads(blob).items()}


@pytest.mark.parametrize("reference", ["", "t4_qwen2.5-0.5b.json"])
def test_generated_cells_compile(tmp_path, reference):
    """Every generated cell must parse as Python, on every code path.

    This is not hypothetical. The reference argument was generated as a
    shell fragment (' --reference "..."') and spliced into the middle of a
    Python list literal, so the payload's run cell was a SyntaxError. It was
    on the path the workflow always takes, and it cost a real Kaggle session
    to find, because nothing between writing the cell and executing it on a
    T4 ever tried to parse it. Both parameters are covered because the bug
    lived only in the branch that was never built locally.
    """
    extra = ["--reference", reference] if reference else []
    driver = _build(tmp_path, *extra)
    notebooks = {"driver": driver, **_payload_notebooks(driver)}
    for name, nb in notebooks.items():
        for index, cell in enumerate(nb["cells"]):
            source = "".join(cell["source"])
            compile(source, f"{name}#cell{index}", "exec")


def test_the_reference_path_the_payload_builds_is_the_one_that_is_shipped(
        tmp_path):
    """The runtime path must be assembled from ROOT, not left as a literal.

    The first version emitted a doubled-brace "{ROOT}/references/..." inside
    an ordinary string, so even had it parsed, the child would have been
    handed a path with a literal brace in it and reported the reference
    absent -- a band check that silently checks nothing.
    """
    driver = _build(tmp_path, "--reference", "t4_qwen2.5-0.5b.json")
    payload = _payload_notebooks(driver)["t4_smoke_gpu0.ipynb"]
    run_cell = "".join(payload["cells"][-1]["source"])
    assert ('cmd += ["--reference", str(ROOT / "references" / '
            '"t4_qwen2.5-0.5b.json")]') in run_cell
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


def test_a_kernel_that_reported_nothing_still_names_its_cause(tmp_path):
    """The summary alone must say why, without downloading the artifact.

    Kaggle hands the log back as a JSON array of stream records, so the
    interesting line arrives split across dozens of them. Both real
    no-report failures so far were legible only after flattening it.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(json.dumps({
        "verdict": "infra", "reason": "no payload report", "slug": "u/s",
        "kernel_state": "COMPLETE", "reports": []}))
    (evidence / "kernel.log").write_text(json.dumps([
        {"stream_name": "stdout", "time": 1.0, "data": "KAGGLE_T4_CI_DRIVER start\n"},
        {"stream_name": "stdout", "time": 2.0, "data": "SyntaxError: invalid "},
        {"stream_name": "stdout", "time": 2.1, "data": "syntax\n"},
        {"stream_name": "stdout", "time": 3.0, "data": "unrelated chatter\n"},
    ]))
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence),
         "--expect", "2"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "SyntaxError: invalid syntax" in proc.stdout
    assert "unrelated chatter" not in proc.stdout


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
        capture_output=True, text=True)
    assert proc.returncode == 0
    assert "NOT RUN" in proc.stdout or "did not run" in proc.stdout

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Resume from a checkpoint must continue the run it was saved from.

2N steps straight vs N steps + checkpoint + resume in a fresh process + N steps: per-step loss,
grad norm and learning rate over the resumed half, and the final adapter weights, must match the
straight run. The tolerance is set from an A/A pair of straight runs (run-to-run noise), with a
small floor so a bit-exact A/A does not demand bit-exact resume.

Each run is a subprocess (tests/_shared/tiny_sft_runner.py), so the resume really starts cold.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from real_accelerator import has_real_cuda

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_ROOT, "tests", "_shared", "tiny_sft_runner.py")
N = 6
# The gate is max(floor, 4 x A/A noise). On an H100 / B200 the A/A pair and the resume are both
# bit-exact, so the floors only absorb nondeterminism the A/A pair happened not to show.
FLOOR = {"loss": 1e-4, "grad_norm": 1e-5, "learning_rate": 1e-12, "adapter": 1e-5}


def _run(out, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, _RUNNER, "--out", str(out), "--max-steps", str(2 * N), *args],
        env = env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    with open(os.path.join(out, "result.json")) as f:
        result = json.load(f)
    import torch

    result["adapter"] = torch.load(os.path.join(out, "adapter.pt"))
    return result


@pytest.fixture(scope = "module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("resume_parity")
    straight = _run(base / "A", "--save-steps", str(N))
    straight_again = _run(base / "B", "--save-steps", str(N))
    checkpoint = base / "A" / "trainer" / f"checkpoint-{N}"
    assert checkpoint.is_dir(), sorted(os.listdir(base / "A" / "trainer"))
    resumed = _run(base / "R", "--resume", str(checkpoint))
    return {"base": base, "A": straight, "B": straight_again, "R": resumed, "ckpt": checkpoint}


def _series(result, key):
    return {s["step"]: s[key] for s in result["steps"] if key in s}


def _max_diff(x, y, key):
    a, b = _series(x, key), _series(y, key)
    steps = [s for s in range(N + 1, 2 * N + 1)]
    missing = [s for s in steps if s not in a or s not in b]
    assert not missing, f"{key}: steps {missing} not logged"
    return max(abs(a[s] - b[s]) for s in steps)


def _adapter_diff(x, y):
    assert x.keys() == y.keys() and len(x) > 0
    return max((x[k] - y[k]).abs().max().item() for k in x)


def _check(runs, resumed):
    report, bad = {}, []
    for key in ("loss", "grad_norm", "learning_rate"):
        noise = _max_diff(runs["A"], runs["B"], key)
        got = _max_diff(runs["A"], resumed, key)
        tol = max(FLOOR[key], 4 * noise)
        report[key] = (got, noise, tol)
        if got > tol:
            bad.append(key)
    noise = _adapter_diff(runs["A"]["adapter"], runs["B"]["adapter"])
    got = _adapter_diff(runs["A"]["adapter"], resumed["adapter"])
    tol = max(FLOOR["adapter"], 4 * noise)
    report["adapter"] = (got, noise, tol)
    if got > tol:
        bad.append("adapter")
    return bad, report


def test_resume_matches_straight_run(runs):
    bad, report = _check(runs, runs["R"])
    assert not bad, f"resume diverged on {bad}; (max diff, A/A noise, tolerance): {report}"


def test_comparison_catches_a_resume_without_optimizer_state(runs):
    # Same resume with the optimizer state removed from the checkpoint: Adam restarts from zero
    # moments, which a working comparison must flag.
    broken = runs["base"] / f"checkpoint-{N}-no-optimizer"
    shutil.copytree(runs["ckpt"], broken)
    removed = [p for p in os.listdir(broken) if p.startswith("optimizer")]
    assert removed, os.listdir(broken)
    for name in removed:
        os.remove(broken / name)
    resumed = _run(runs["base"] / "R_broken", "--resume", str(broken))
    bad, report = _check(runs, resumed)
    assert bad, f"a resume without optimizer state passed the comparison: {report}"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Allocated CUDA memory must stay flat over a LoRA SFT run once it has warmed up.

After WARMUP steps, every step ends with gc.collect() + synchronize and records
torch.cuda.memory_allocated(). The least-squares slope over the remaining steps, times the
number of steps, must stay under BOUND_BYTES. On failure the CUDA memory snapshot from the run
(torch.cuda.memory._dump_snapshot, open it at https://pytorch.org/memory_viz) is named in the
message.

A second run injects a 1 MiB per-step leak to show the check can fail.
Each run is a subprocess (tests/_shared/tiny_sft_runner.py).
"""

import json
import os
import subprocess
import sys

import pytest

from real_accelerator import has_real_cuda

# `gpu`: deselected by the default addopts, CI runs it in the `-m 'gpu or slow'` leg.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU"),
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_ROOT, "tests", "_shared", "tiny_sft_runner.py")
STEPS = 60
WARMUP = 10
BOUND_BYTES = 256 * 1024  # total drift allowed over the measured window
LEAK_BYTES = 1024 * 1024


def _run(out, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            _RUNNER,
            "--out",
            str(out),
            "--max-steps",
            str(STEPS),
            "--memory-probe",
            str(WARMUP),
            *args,
        ],
        env = env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    with open(os.path.join(out, "result.json")) as f:
        return json.load(f)


def _drift(result):
    xs, ys = result["trace"]["step"], result["trace"]["allocated"]
    assert len(xs) >= STEPS - WARMUP - 1, f"only {len(xs)} memory samples"
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope * (xs[-1] - xs[0]), slope, min(ys), max(ys)


def test_allocated_memory_is_flat_after_warmup(tmp_path):
    result = _run(tmp_path / "clean")
    drift, slope, lo, hi = _drift(result)
    snapshot = tmp_path / "clean" / "memory_snapshot.pickle"
    assert drift < BOUND_BYTES, (
        f"allocated memory grew {drift / 2**20:.2f} MiB over {STEPS - WARMUP} steps "
        f"({slope / 1024:.1f} KiB/step, range {lo} .. {hi} bytes); snapshot: {snapshot}"
    )


def test_check_catches_an_injected_leak(tmp_path):
    result = _run(tmp_path / "leak", "--leak-bytes", str(LEAK_BYTES))
    drift, slope, _, _ = _drift(result)
    assert (
        drift >= BOUND_BYTES
    ), f"a {LEAK_BYTES} B/step leak was not flagged (slope {slope:.0f} B/step)"

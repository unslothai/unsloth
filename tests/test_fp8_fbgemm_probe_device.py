# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The FBGEMM block FP8 probe must not run at import (it allocated on bare "cuda", so every torchrun rank opened
a CUDA context on GPU 0) and must probe each rank's own card. CPU only: a fake fbgemm_gpu 1.4.0 and a recorder
for the probe's CUDA allocations stand in for the GPU, in a child process so the import is fresh."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("triton")  # unsloth.kernels needs it (absent on macOS / Windows runners)

ROOT = Path(__file__).resolve().parents[1]

_CHILD = textwrap.dedent(
    """
    import contextlib, importlib.machinery, json, os, sys, types
    root = sys.argv[1]
    sys.path.insert(0, root)
    fake = types.ModuleType("fbgemm_gpu")
    fake.__version__ = "1.4.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("fbgemm_gpu", None)
    sys.modules["fbgemm_gpu"] = fake
    import torch

    touched = []  # CUDA devices the FBGEMM probe allocates on or synchronizes (each opens a context there)
    real_ones = torch.ones

    def ones(*args, device = None, **kwargs):
        if kwargs.get("dtype") is torch.float8_e4m3fn and device is not None and torch.device(device).type == "cuda":
            index = torch.device(device).index
            if index is None:  # bare "cuda": the current card, GPU 0 before any set_device
                try:
                    index = torch.cuda.current_device()
                except Exception:
                    index = 0
            touched.append(index)
            raise RuntimeError("CUDA error: no kernel image (recorded, not run)")
        return real_ones(*args, device = device, **kwargs)

    torch.ones = ones
    from unsloth.kernels import fp8
    assert fp8.__file__.startswith(root), fp8.__file__
    out = {"import": list(touched)}

    # Two torchrun ranks on a 2-card host; neither has called set_device yet.
    current = [0]

    @contextlib.contextmanager
    def device_guard(device):
        current.append(torch.device(device).index)
        try:
            yield
        finally:
            current.pop()

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    torch.cuda.current_device = lambda: current[-1]
    torch.cuda.device = device_guard
    torch.cuda.empty_cache = lambda: touched.append(current[-1])
    torch.cuda.synchronize = lambda device = None: touched.append(current[-1] if device is None else torch.device(device).index)
    for rank in ("0", "1"):
        del touched[:]
        os.environ["LOCAL_RANK"] = rank
        resolve = getattr(fp8, "resolve_fp8_block_quant_linear", None)
        if resolve is not None:
            fp8._FBGEMM_BLOCK[:] = []
            fp8.fp8_block_quant_linear = fp8._fp8_block_quant_linear_unresolved
            resolve()
        out[rank] = {"probe": list(touched), "has_fbgemm": os.environ.get("UNSLOTH_HAS_FBGEMM")}
    if resolve is not None:
        # Out of memory during the probe: not latched, so the next call probes again.
        def oom(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("recorded")

        torch.ones = oom
        fp8._FBGEMM_BLOCK[:] = []
        fp8.fp8_block_quant_linear = fp8._fp8_block_quant_linear_unresolved
        resolve()
        out["oom"] = {
            "latched": bool(fp8._FBGEMM_BLOCK),
            "bound": fp8.fp8_block_quant_linear is not fp8._fp8_block_quant_linear_unresolved,
        }
    print("RESULT " + json.dumps(out))
    """
)


def _run():
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_FORCE_CPU = "1", UNSLOTH_DISABLE_AUTO_UPDATES = "1"
    )
    env.pop("UNSLOTH_ALLOW_CPU", None)
    env.pop("LOCAL_RANK", None)
    env.pop("UNSLOTH_HAS_FBGEMM", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
        cwd = str(ROOT),
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(line[len("RESULT ") :])


def test_fbgemm_probe_is_lazy_and_runs_on_the_local_rank_card():
    out = _run()
    # Import opens nothing on any card.
    assert out["import"] == [], out
    # Each rank probes on its own card (allocation, and the printf guard's synchronize): GPU 0 by rank 0 only.
    assert out["0"]["probe"] and set(out["0"]["probe"]) == {0}, out
    assert out["1"]["probe"] and set(out["1"]["probe"]) == {1}, out
    # The probe failed (no kernel image): the Triton / torch route, and the loader sees that.
    assert out["0"]["has_fbgemm"] == out["1"]["has_fbgemm"] == "0", out
    assert out["oom"] == {"latched": False, "bound": False}, out

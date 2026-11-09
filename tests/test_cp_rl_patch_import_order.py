# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO / GRPO keep their context parallel patch when unsloth.context_parallel is imported before the models."""

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = """
import sys
__import__(sys.argv[1])
import unsloth.models  # noqa: F401  (models/llama.py patches the TRL trainers)
from trl import DPOTrainer, GRPOTrainer
print("CP_FLAGS", [bool(getattr(t, "__unsloth_context_parallel__", False)) for t in (DPOTrainer, GRPOTrainer)])
"""


@pytest.mark.parametrize("first", ["unsloth", "unsloth.context_parallel"])
def test_rl_trainers_get_context_parallel_patch(first, tmp_path):
    # GPU-less CPU fallback, where unsloth's own import does not load the models before context_parallel.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update(
        CUDA_VISIBLE_DEVICES = "",
        OMP_NUM_THREADS = "2",
        TOKENIZERS_PARALLELISM = "false",
        PYTHONPATH = os.pathsep.join(filter(None, [_REPO, env.get("PYTHONPATH")])),
        UNSLOTH_COMPILE_LOCATION = str(tmp_path / "ucc"),
    )
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, first],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    flags = [line for line in out.stdout.splitlines() if line.startswith("CP_FLAGS")]
    assert flags == ["CP_FLAGS [True, True]"], (out.stdout[-2000:], out.stderr[-2000:])
    assert "Could not add context parallelism" not in out.stdout + out.stderr

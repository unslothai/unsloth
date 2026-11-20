# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""DDP full finetune of a VLM with reentrant checkpointing (Gemma 4 / Gemma 3N): a text-only batch leaves the
vision tower without gradients, and find_unused_parameters cannot run under reentrant checkpointing. Unsloth keeps
the towers out of DDP and averages their gradients itself; two gloo ranks match one process."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason = "torchrun's rendezvous needs libuv, which Windows torch builds lack",
)
def test_vlm_full_finetune_with_text_only_batches_matches_one_process(tmp_path):
    from tests._rendezvous_port import free_port

    worker = Path(__file__).with_name("_ddp_tower_sync_worker.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", TMPDIR = str(tmp_path), OMP_NUM_THREADS = "1")
    env.pop("UNSLOTH_DDP_TOWER_SYNC", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(free_port()),
            str(worker),
            str(tmp_path),
        ],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-4000:]
    ranks = re.findall(
        r"RANK_OK rank=\d ga=\d syncs=(\d+) diff=([-+.e0-9]+) moved=([-+.e0-9]+)", out
    )
    assert len(ranks) == 4, out[-4000:]
    for syncs, diff, moved in ranks:
        assert int(syncs) == 3, out[
            -2000:
        ]  # one tower all-reduce per optimizer step, none inside no_sync
        assert float(diff) < 1e-5, out[-2000:]
        assert float(moved) > 0, out[-2000:]

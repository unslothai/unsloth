# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DDP's start broadcast flattens every parameter into a bucket, a second copy of the largest tensor: Gemma4 E2B's
frozen 4.4 GiB per-layer embedding ran a 2xT4 LoRA run out of memory in DistributedDataParallel.__init__. A LoRA
model's frozen weights are identical on every rank (one checkpoint) and get no gradient, so DDP skips them;
trainable weights are still broadcast and averaged."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("accelerate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _ddp_frozen_sync_worker as worker  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")

CASES = [
    ("peft", True, {}),
    ("peft_sync_frozen", True, {"UNSLOTH_DDP_SYNC_FROZEN": "1"}),
    ("full", False, {}),
]


@pytest.fixture(scope = "module")
def ranks():
    import torch.multiprocessing as mp

    from tests._rendezvous_port import free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target = worker.run_rank, args = (rank, port, queue, CASES)) for rank in range(2)
    ]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout = 600) for _ in procs)
    for p in procs:
        p.join(60)
    for rank, out in results.items():
        assert "error" not in out, out["error"]
    return results


def test_lora_frozen_weights_are_not_broadcast(ranks):
    assert ranks[1]["peft"]["frozen"] == 1.0
    # Trainable weights still start from rank 0 and train in sync.
    assert ranks[1]["peft"]["trainable"] == ranks[0]["peft"]["trainable"] == 0.0
    assert ranks[0]["peft"]["grad"] == ranks[1]["peft"]["grad"]


def test_kill_switch_and_full_finetuning_keep_the_broadcast(ranks):
    for case in ("peft_sync_frozen", "full"):
        assert ranks[1][case]["frozen"] == 0.0
        assert ranks[1][case]["trainable"] == 0.0

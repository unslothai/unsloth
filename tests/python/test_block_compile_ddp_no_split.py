# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A compiled decoder block called inside DDP.forward must not be split by dynamo's DDPOptimizer.

Under DDP Unsloth checkpoints VLMs non-reentrantly. The recompute runs from backward, outside
DDP.forward, where dynamo compiles without DDPOptimizer; a block graph split in the forward but
compiled whole in the recompute saves a different number of tensors (DeepSeek-OCR on 2 T4s:
"A different number of tensors was saved during the original forward and recomputation", 50 vs 49).
The block is its own compiled region already, so its gradients still reach DDP block by block."""

from __future__ import annotations

import os
import queue as queue_module
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _block_compile_ddp_worker as worker  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = worker.run_rank, args = (r, port, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        while len(results) < len(procs):
            try:
                rank, out = queue.get(timeout = 60 if results else 900)
            except queue_module.Empty:
                break
            results[rank] = out
    finally:
        for p in procs:
            p.join(5 if len(results) < len(procs) else 120)
            if p.is_alive():
                p.kill()
    assert sorted(results) == [0, 1], results
    for out in results.values():
        assert "error" not in out, out["error"]
    return results


def test_compiled_block_is_not_split_by_ddp_optimizer():
    results = _run()
    for rank, out in results.items():
        assert out["wrapped"] == ["LlamaDecoderLayer"], out
        # The compiled region ran (not the stock forward) in every layer.
        assert all(out["compiled"]), out
        assert out["splits"] == [], (rank, out["splits"])
        # Only scoped to the block call: the global setting is the user's.
        assert out["optimize_ddp"] is True
        assert len(out["losses"]) == 2 and out["grad_sum"] > 0
    # DDP averaged the gradients: both ranks hold the same ones.
    assert results[0]["grad_sum"] == pytest.approx(results[1]["grad_sum"], rel = 1e-6)

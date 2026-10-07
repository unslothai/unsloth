# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GRPO's gradient pass bypasses DDP.forward, so the trainer must average gradients itself, once, where DDP would."""

from __future__ import annotations

import os
import socket
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _grpo_ddp_worker as worker  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_grpo_training_step_averages_gradients_like_ddp():
    import torch.multiprocessing as mp

    # spawn: fork is unsafe once autograd has run in this process.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = worker.run_rank, args = (rank, port, queue)) for rank in range(2)]
    for p in procs:
        p.start()
    results = [queue.get(timeout = 900) for _ in procs]
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    for _rank, grads, mean, local in results:
        mean, local = torch.tensor(mean), torch.tensor(local)
        torch.testing.assert_close(torch.tensor(grads["bypass"]), mean, rtol = 1e-6, atol = 1e-7)
        torch.testing.assert_close(torch.tensor(grads["oversized"]), mean, rtol = 1e-6, atol = 1e-7)
        torch.testing.assert_close(torch.tensor(grads["through_ddp"]), mean, rtol = 1e-6, atol = 1e-7)
        torch.testing.assert_close(torch.tensor(grads["no_sync"]), local)

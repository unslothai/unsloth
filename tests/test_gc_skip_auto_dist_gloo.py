# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""checkpoint_skip_layers = "auto" under torchrun: every rank keeps the same (smallest) set of layers."""

import multiprocessing
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _worker(rank, world, port, usable, queue):
    import torch.distributed as dist
    sys.path.insert(0, HERE)
    try:
        dist.init_process_group(
            "gloo", init_method = f"tcp://127.0.0.1:{port}", rank = rank, world_size = world
        )
        import test_gc_skip_auto as T

        ns, _ = T._load()
        memory = {"allocated": 0}
        # Process-local CPU stand-ins for the CUDA calls the planner makes (as T._fake_cuda does).
        torch.cuda.is_available = lambda: True
        torch.Tensor.is_cuda = property(lambda self: True)
        torch.cuda.reset_peak_memory_stats = lambda device = None: None
        torch.cuda.memory_allocated = lambda device = None: memory["allocated"]
        torch.cuda.max_memory_allocated = lambda device = None: memory["allocated"]
        ns["usable_cuda_bytes"] = lambda device: usable
        model = T._FakeModel([GIB // 16] * 8, memory)
        ns["skip_checkpointing"](model, "auto")
        planner = model._unsloth_auto_skip
        T._fake_step(model, rows = 16)
        model(torch.zeros(1, 16, 4))
        queue.put((rank, list(planner.kept), None))
    except Exception as error:  # reported to the parent instead of hanging it
        import traceback
        queue.put((rank, None, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.timeout(240)
def test_auto_agrees_on_the_smallest_count_across_ranks():
    # Rank 0 has room for all 8 layers, rank 1 for 2: both must keep the same 2.
    # (4 GiB - 0.5 GiB step) * 0.75 - 1 GiB = 1.6 GiB -> 8; (2 GiB - 0.5) * 0.75 - 1 = 0.125 GiB -> 2.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    usable = [4 * GIB, 2 * GIB]
    procs = [ctx.Process(target = _worker, args = (r, 2, port, usable[r], queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict()
    for _ in procs:
        rank, kept, error = queue.get(timeout = 200)
        assert error is None, error
        results[rank] = kept
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert len(results[1]) == 2
    assert results[0] == results[1]

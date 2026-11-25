# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CP GRPO broadcast: only the group root copies tensors to the host; every peer adopts its object (CPU, gloo)."""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

WORLD = 3


def _manager():
    manager = cp.ContextParallelManager.__new__(cp.ContextParallelManager)
    manager.size = WORLD
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(
        get_local_rank = dist.get_rank, get_group = lambda: dist.group.WORLD
    )
    return manager


def _payload(rank):
    with torch.inference_mode():
        scored = torch.full((3,), float(rank))
    return {
        "ids": torch.arange(4) + 10 * rank,
        "nested": [torch.ones(2, 2) * rank, (torch.tensor([rank]), f"r{rank}")],
        "scored": scored,
        "meta": {"rank": rank, "text": [f"t{rank}"]},
    }


def _worker(rank, init_file, queue):
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = WORLD)
    try:
        # Each _OnDevice is one tensor.cpu() (a device-to-host copy for accelerator tensors).
        host_copies = []
        original = cp._OnDevice.__init__

        def counting_init(self, tensor):
            host_copies.append(tuple(tensor.shape))
            original(self, tensor)

        cp._OnDevice.__init__ = counting_init
        try:
            got = cp._broadcast_from_group_root(_payload(rank), _manager(), torch.device("cpu"))
        finally:
            cp._OnDevice.__init__ = original
        root = _payload(0)
        assert torch.equal(got["ids"], root["ids"])
        assert torch.equal(got["nested"][0], root["nested"][0])
        assert torch.equal(got["nested"][1][0], root["nested"][1][0])
        assert got["nested"][1][1] == "r0" and isinstance(got["nested"][1], tuple)
        assert got["meta"] == {"rank": 0, "text": ["t0"]}
        assert got["scored"].is_inference() and torch.equal(got["scored"], root["scored"])
        assert not got["ids"].is_inference()
        queue.put((rank, len(host_copies)))
    except BaseException:  # pragma: no cover - reported to the parent
        import traceback
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_only_the_root_copies_tensors_to_the_host():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs fork")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        procs = [ctx.Process(target = _worker, args = (r, init_file, queue)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for p in procs:
            p.join(30)
    # The root: one host copy per tensor leaf (4). Peers: none (their payload is overwritten).
    assert results == {0: 4, 1: 0, 2: 0}, results

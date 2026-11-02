# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch.optim.Muon on Shard(0) DTensors (native FSDP2 / tensor parallel parameters) steps the whole matrix,
which is why the registry lets it through: a step on the local shards alone lands far from it."""

import multiprocessing
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import _free_port  # noqa: E402


def _worker(rank, world, port, tmp):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        w, g = torch.randn(64, 32), torch.randn(64, 32)
        whole = torch.nn.Parameter(w.clone())
        whole.grad = g.clone()
        torch.optim.Muon([whole], lr = 0.1).step()
        shard = torch.nn.Parameter(distribute_tensor(w.clone(), mesh, [Shard(0)]))
        shard.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
        torch.optim.Muon([shard], lr = 0.1).step()
        local = torch.nn.Parameter(w.clone().chunk(world)[rank])
        local.grad = g.clone().chunk(world)[rank]
        torch.optim.Muon([local], lr = 0.1).step()
        update = whole.detach() - w
        dtensor = (shard.detach().full_tensor() - whole.detach()).norm() / update.norm()
        mine = whole.detach().chunk(world)[rank]
        per_shard = (local.detach() - mine).norm() / update.chunk(world)[rank].norm()
        torch.save((float(dtensor), float(per_shard)), os.path.join(tmp, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_torch_muon_steps_dtensor_shards_as_whole_matrices():
    if getattr(torch.optim, "Muon", None) is None:
        pytest.skip("torch.optim.Muon needs torch >= 2.9")
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, 2, port, tmp)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        assert [p.exitcode for p in procs] == [0, 0]
        dtensor, per_shard = torch.load(os.path.join(tmp, "r0.pt"))
    # Newton-Schulz runs in bfloat16: the DTensor step matches the whole-matrix step to bf16 noise, the shard step does not.
    assert dtensor < 0.05 and per_shard > 0.2, (dtensor, per_shard)

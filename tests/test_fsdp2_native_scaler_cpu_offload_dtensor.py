# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPUOffloadPolicy fp16: DTensor grads live on the CPU, and DTensor's found_inf reduce is a CPU collective a
NCCL-only group cannot run (2 x T4 failed with "No backend type associated with device type cpu"). The scaler must
unscale local shards and leave found_inf agreement to ShardedGradScaler."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402


class _Handler:
    def to_kwargs(self):
        return {"device": "cpu", "init_scale": 2.0**8}


@pytest.fixture(scope = "module")
def cpu_mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()))
    dist.init_process_group("gloo", rank = 0, world_size = 1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("overflow", [False, True], ids = ["finite", "inf"])
def test_cpu_dtensor_grads_unscale_as_local_shards(cpu_mesh, monkeypatch, overflow):
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    native = _load_native()
    old = torch.amp.GradScaler("cpu", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _Handler())
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    torch.manual_seed(0)
    p = torch.nn.Parameter(distribute_tensor(torch.randn(8, 8), cpu_mesh, [Shard(0)]))
    grad = torch.randn(8, 8) * 256.0
    if overflow:
        grad[0, 0] = float("inf")
    p.grad = distribute_tensor(grad, cpu_mesh, [Shard(0)])
    opt = torch.optim.AdamW([p], lr = 1e-2, foreach = True)
    seen = []
    parent = ShardedGradScaler._unscale_grads_

    def recording(self, optimizer, *a, **k):
        seen.extend(
            type(q.grad) for g in optimizer.param_groups for q in g["params"] if q.grad is not None
        )
        return parent(self, optimizer, *a, **k)

    monkeypatch.setattr(ShardedGradScaler, "_unscale_grads_", recording)
    scaler.scale(torch.ones(()))  # initializes the scale; the grads above stand for scaled ones
    scaler.unscale_(opt)
    assert seen and not any(issubclass(t, DTensor) for t in seen), seen
    assert isinstance(p.grad, DTensor)
    state = scaler._per_optimizer_states[id(opt)]
    found = sum(float(v.sum()) for v in state["found_inf_per_device"].values())
    assert found == (1.0 if overflow else 0.0)
    if not overflow:
        torch.testing.assert_close(p.grad.full_tensor(), grad / 256.0)

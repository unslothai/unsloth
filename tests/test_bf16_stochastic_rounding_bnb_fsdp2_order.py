# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trainer + accelerate FSDP2 order: the optimizer (and its bf16 stochastic rounding wrap) is built on plain
parameters, then Accelerator._prepare_fsdp2 swaps the param groups for DTensors and localizes the bitsandbytes
optimizer (class swap). The rounding wrap used to call the pre-swap class's update_step on the DTensor and the
first step raised "got mixed torch.Tensor and DTensor". It now resolves the class per call, so the DTensor steps
on its local shard. World 1, gloo on CPU."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[1])


def _load(rel, name):
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert os.path.realpath(module.__file__).startswith(os.path.realpath(ROOT)), module.__file__
    return module


@pytest.fixture
def mesh():
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group already exists in this process")
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group("gloo", store = dist.HashStore(), rank = 0, world_size = 1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_sr_wrap_before_fsdp2_localize_steps_the_dtensor(mesh, monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    from torch.distributed.tensor import Shard, distribute_tensor

    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "1")
    sr = _load("unsloth/optimizers/bf16_stochastic_rounding.py", "_test_sr_fsdp2_order_sr")
    local = _load("unsloth/models/fsdp2_bnb_optim.py", "_test_sr_fsdp2_order_local")
    if not local.bnb_steps_cpu_tensors():
        pytest.skip("this bitsandbytes cannot step CPU tensors")

    weight = torch.nn.Parameter(torch.ones(64, 128, dtype = torch.bfloat16))
    opt = bnb.optim.AdamW8bit([weight], lr = 1e-2)
    # Trainer.create_scheduler: the optimizer exists, the parameters are still plain tensors.
    assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 1
    assert "update_step" in vars(opt)

    # Accelerator._prepare_fsdp2: DTensor parameters replace the plain ones, then the class swap.
    sharded = torch.nn.Parameter(distribute_tensor(weight.detach().clone(), mesh, [Shard(0)]))
    opt.param_groups[0]["params"] = [sharded]
    assert local.localize_bnb_optimizer(opt)

    sharded.grad = distribute_tensor(
        torch.full((64, 128), 0.5, dtype = torch.bfloat16), mesh, [Shard(0)]
    )
    opt.step()
    opt.step()
    after = sharded.to_local().float()
    assert torch.isfinite(after).all()
    # Positive gradient, AdamW: every weight moved down from 1.0.
    assert (after < 1.0).all(), after.max()

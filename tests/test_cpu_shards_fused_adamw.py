# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-resident native FSDP2 / tensor parallel shards (UNSLOTH_FSDP2_CPU_OFFLOAD, a CPU mesh) step torch's
fused CPU AdamW: optim = 'adamw_torch' would otherwise take the single-tensor loop (foreach only covers
accelerator tensors), and the bitsandbytes switch used to drop to it on CPU too."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_units import _load_native  # noqa: E402


def _name(args):
    return str(getattr(args.optim, "value", args.optim))


def _args(optim, **kw):
    return types.SimpleNamespace(optim = optim, optim_args = kw.pop("optim_args", None), **kw)


def test_fsdp2_cpu_shards_switch_adamw_torch_to_fused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_CPU_FUSED_ADAMW", raising = False)
    native = _load_native()
    args = _args("adamw_torch")
    assert native._swap_bnb_optimizer(args, cpu_shards = True) is True
    assert _name(args) == "adamw_torch_fused"
    # GPU shards keep the user's choice (torch picks foreach there).
    args = _args("adamw_torch")
    assert native._swap_bnb_optimizer(args, cpu_shards = False) is False
    assert _name(args) == "adamw_torch"
    # optim_args (FlashAdamW's marker) and the kill switch keep adamw_torch.
    args = _args("adamw_torch", optim_args = "unsloth_flash_adamw=1")
    assert native._swap_bnb_optimizer(args, cpu_shards = True) is False
    assert _name(args) == "adamw_torch"
    monkeypatch.setenv("UNSLOTH_CPU_FUSED_ADAMW", "0")
    args = _args("adamw_torch")
    assert native._swap_bnb_optimizer(args, cpu_shards = True) is False
    assert _name(args) == "adamw_torch"


def test_fsdp2_bnb_switch_keeps_fused_on_cpu_only_hosts(monkeypatch):
    # No CUDA (CPU mesh): adamw_8bit used to become the single-tensor adamw_torch.
    monkeypatch.setenv("UNSLOTH_FSDP2_BNB_LOCAL", "0")
    native = _load_native()
    monkeypatch.setattr(native.torch.cuda, "is_available", lambda: False)
    args = _args("adamw_8bit")
    assert native._swap_bnb_optimizer(args, cpu_shards = True) is True
    assert _name(args) == "adamw_torch_fused"
    # A non-CUDA accelerator mesh (shards not on the CPU) keeps the old adamw_torch target.
    args = _args("adamw_8bit")
    native._swap_bnb_optimizer(args, cpu_shards = False)
    assert _name(args) == "adamw_torch"


def test_tp_cpu_mesh_steps_fused_adamw(monkeypatch):
    from test_tp_native_oracle import tpn

    monkeypatch.delenv("UNSLOTH_CPU_FUSED_ADAMW", raising = False)
    monkeypatch.setenv("UNSLOTH_FSDP2_BNB_LOCAL", "0")
    args = _args("adamw_torch")
    assert tpn._tp_bnb_optim(args, cpu_shards = True) is True
    assert _name(args) == "adamw_torch_fused"
    args = _args("adamw_8bit")
    assert tpn._tp_bnb_optim(args, cpu_shards = True) is True
    assert _name(args) == "adamw_torch_fused"
    tpn._refuse_tp_trainer_args(args, cpu_shards = True)  # the TP Trainer accepts the fused name
    args = _args("adamw_torch")
    assert tpn._tp_bnb_optim(args, cpu_shards = False) is False
    assert _name(args) == "adamw_torch"


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        placements = [[Shard(0)], [Shard(1)], [Replicate()], [Shard(0)]]
        result = {}
        for fused in (False, True):
            torch.manual_seed(0)
            full = [torch.randn(64, 32) for _ in placements]
            params = [
                torch.nn.Parameter(distribute_tensor(f, mesh, p)) for f, p in zip(full, placements)
            ]
            opt = torch.optim.AdamW(
                params, lr = 1e-2, weight_decay = 0.01, **({"fused": True} if fused else {})
            )
            for step in range(3):
                torch.manual_seed(10 + step)
                for p, pl in zip(params, placements):
                    p.grad = distribute_tensor(torch.randn(64, 32), mesh, pl)
                opt.step()
            result[fused] = [p.full_tensor() for p in params]
        diff = max((a - b).abs().max().item() for a, b in zip(result[False], result[True]))
        out.put((rank, diff))
    finally:
        dist.destroy_process_group()


def test_fused_cpu_adamw_matches_default_on_dtensor_shards():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = True, start_method = "spawn"
    )
    diffs = [out.get(timeout = 5)[1] for _ in range(2)]
    assert max(diffs) < 1e-5, diffs

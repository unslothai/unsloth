# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 float16 start scale on context parallel and sub-mesh layouts (gloo CPU worlds).

The Trainer multiplies every rank's loss by the number of processes (token mean over the world), so each rank's
backward carries world x one GPU's per-token gradient whatever mesh FSDP2 shards over. The start scale must be
divided by that factor:

* CP = 2, DP = 1 (world 2, both ranks share one batch, each takes half of the sequence, CP divides the token
  count by 2): factor 2 = the FSDP2 mesh (the world), the step one GPU takes is taken;
* two disjoint 2-rank sub-meshes of a world of 4 (each rank its own quarter of the batch): factor 4, not the
  2 of the mesh; dividing by the mesh left the scale 2x too large and the first step overflowed."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

SCALE = 2.0**16
TOKENS = 8
FP16_GRAD = 40000.0  # one GPU's scaled per-token output gradient: below the float16 max (65504)


def _data():
    torch.manual_seed(0)
    x = torch.rand(TOKENS, 8) * 1e-3
    weight = torch.randn(8, 8) * 0.1
    return x, weight, FP16_GRAD * TOKENS / SCALE


def _one_gpu():
    x, weight, coef = _data()
    w = torch.nn.Parameter(weight.clone())
    scaler = torch.amp.GradScaler("cpu", init_scale = SCALE)
    opt = torch.optim.SGD([w], lr = 1e-3)
    out = torch.nn.functional.linear(x.half(), w.half())
    scaler.scale((out.float() * coef).sum() / TOKENS).backward()
    scaler.step(opt)
    scaler.update()
    return float(scaler.get_scale()), w.detach().clone()


def _native(rank, world, layout):
    native = _load_native()
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    x, weight, coef = _data()
    model = torch.nn.Linear(8, 8, bias = False)
    with torch.no_grad():
        model.weight.copy_(weight)
    if layout == "cp":
        # CP = 2, DP = 1: native FSDP2 shards over the world; the CP peers share the batch, each its half.
        mesh = None
        share = TOKENS // world
        tokens = x[rank * share : (rank + 1) * share]
        num_items = (
            TOKENS * world
        ) // world  # HF sums the count over the world, CP divides it by cp
    else:
        # Two disjoint 2-rank meshes [0, 1] and [2, 3]; every rank reads its own quarter of the batch.
        mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names = ("pair", "shard"))["shard"]
        share = TOKENS // world
        tokens = x[rank * share : (rank + 1) * share]
        num_items = TOKENS
    native.shard_model(
        model,
        native.FSDP2Config(param_dtype = torch.float16, mesh = mesh, device = "cpu", units = []),
    )
    assert isinstance(model.weight, torch.distributed.tensor.DTensor)
    old = torch.amp.GradScaler("cpu", init_scale = SCALE)
    trainer = types.SimpleNamespace(
        model = model,
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = None, num_processes = world),
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    opt = torch.optim.SGD(model.parameters(), lr = 1e-3)
    out = model(tokens.half())
    # transformers Trainer: sum / num_items_in_batch, times the number of processes.
    scaler.scale((out.float() * coef).sum() / num_items * world).backward()
    scaler.unscale_(opt)
    found_inf = float(
        sum(
            v.item() for v in scaler._per_optimizer_states[id(opt)]["found_inf_per_device"].values()
        )
    )
    scaler.step(opt)
    scaler.update()
    return (
        found_inf,
        float(scaler._init_scale),
        float(scaler.get_scale()),
        model.weight.full_tensor().detach().clone(),
    )


def _worker(rank, world, port, out, layout):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
    )
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {"one_gpu": _one_gpu()}
        try:
            res["native"] = _native(rank, world, layout)
        except Exception as e:  # noqa: BLE001
            import traceback
            res["native"] = repr(e) + traceback.format_exc()
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


def _run(tmp_path_factory, layout, world):
    out = str(tmp_path_factory.mktemp(f"fp16_{layout}"))
    torch.multiprocessing.spawn(
        _worker, args = (world, _free_port(), out, layout), nprocs = world, join = True
    )
    return [torch.load(os.path.join(out, f"rank{r}.pt"), weights_only = False) for r in range(world)]


@pytest.mark.parametrize("layout,world", [("cp", 2), ("submesh", 4)])
def test_fp16_start_scale_matches_the_trainer_loss_multiplier(tmp_path_factory, layout, world):
    for rank, res in enumerate(_run(tmp_path_factory, layout, world)):
        assert not isinstance(res["native"], str), res["native"]
        found_inf, init_scale, scale, weight = res["native"]
        assert init_scale == SCALE / world, (layout, rank, init_scale)
        assert found_inf == 0.0, f"{layout}: rank {rank} overflowed where one GPU did not"
        assert scale == SCALE / world, (layout, rank, scale)
        if layout == "cp":
            _, one_gpu_weight = res["one_gpu"]
            torch.testing.assert_close(weight, one_gpu_weight, rtol = 1e-3, atol = 1e-6)
        else:
            assert torch.isfinite(weight).all()


def test_loss_scale_ranks_counts_processes_not_the_mesh():
    native = _load_native()
    model = torch.nn.Linear(2, 2)
    model.__dict__["_unsloth_fsdp2_mesh"] = types.SimpleNamespace(size = lambda: 2)
    trainer = types.SimpleNamespace(model = model, accelerator = types.SimpleNamespace(num_processes = 4))
    assert native._loss_scale_ranks(trainer) == 4
    trainer.accelerator.parallelism_config = types.SimpleNamespace(tp_size = 2)
    assert native._loss_scale_ranks(trainer) == 2

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 float16 starts the loss scale divided by the data parallel size, so it takes the steps
one GPU takes (gloo CPU world of 2)."""

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


def _data(world):
    torch.manual_seed(0)
    x = torch.rand(TOKENS, 8) * 1e-3
    weight = torch.randn(8, 8) * 0.1
    coef = FP16_GRAD * TOKENS / SCALE
    return x, weight, coef


def _loss(out, coef, n_global, world):
    # transformers Trainer with num_items_in_batch: sum / global token count, times the world size.
    return (out.float() * coef).sum() / n_global * world


def _one_gpu(world):
    x, weight, coef = _data(world)
    w = torch.nn.Parameter(weight.clone())
    scaler = torch.amp.GradScaler("cpu", init_scale = SCALE)
    opt = torch.optim.SGD([w], lr = 1e-3)
    out = torch.nn.functional.linear(x.half(), w.half())
    scaler.scale(_loss(out, coef, TOKENS, 1)).backward()
    scaler.step(opt)
    scaler.update()
    return float(scaler.get_scale()), w.detach().clone()


def _native(world, rank):
    native = _load_native()
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    x, weight, coef = _data(world)
    model = torch.nn.Linear(8, 8, bias = False)
    with torch.no_grad():
        model.weight.copy_(weight)
    fully_shard(
        model,
        mp_policy = MixedPrecisionPolicy(param_dtype = torch.float16, reduce_dtype = torch.float32),
    )
    old = torch.amp.GradScaler("cpu", init_scale = SCALE)
    trainer = types.SimpleNamespace(
        model = model, accelerator = types.SimpleNamespace(scaler = old, scaler_handler = None)
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    opt = torch.optim.SGD(model.parameters(), lr = 1e-3)
    share = TOKENS // world
    out = model(x[rank * share : (rank + 1) * share].half())
    scaler.scale(_loss(out, coef, TOKENS, world)).backward()
    scaler.unscale_(opt)
    found_inf = float(
        sum(
            v.item() for v in scaler._per_optimizer_states[id(opt)]["found_inf_per_device"].values()
        )
    )
    scaler.step(opt)
    scaler.update()
    full = model.weight.full_tensor().detach().clone()
    return found_inf, float(scaler.get_scale()), full


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {"one_gpu": _one_gpu(world)}
        try:
            res["native"] = _native(world, rank)
        except Exception as e:
            res["native"] = repr(e)
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fp16_scale"))
    torch.multiprocessing.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


def test_one_gpu_takes_the_step(ranks):
    for res in ranks:
        scale, _ = res["one_gpu"]
        assert scale == SCALE


def test_native_fp16_takes_the_step_one_gpu_takes(ranks):
    for res in ranks:
        assert not isinstance(res["native"], str), res["native"]
        found_inf, scale, weight = res["native"]
        assert found_inf == 0.0, "native FSDP2 overflowed where one GPU did not"
        assert scale == SCALE / 2, scale
        _, one_gpu_weight = res["one_gpu"]
        torch.testing.assert_close(weight, one_gpu_weight, rtol = 1e-3, atol = 1e-6)


def test_data_parallel_size_reads_the_mesh():
    native = _load_native()
    model = torch.nn.Linear(2, 2)
    model.__dict__["_unsloth_fsdp2_mesh"] = types.SimpleNamespace(size = lambda: 4)
    assert native._data_parallel_size(model) == 4
    assert native._data_parallel_size(None) in (1, int(os.environ.get("WORLD_SIZE", "1")))

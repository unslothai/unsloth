# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float16 TP with DTensor and plain grads in one optimizer (Qwen3-VL, UNSLOTH_TP_SIZE=2) on a gloo CPU world of 2."""

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
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

_SCALE = 2.0**8


class _Handler:
    def to_kwargs(self):
        return {"device": "cpu", "init_scale": _SCALE}


def _run(optim, inf_rank = None):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    native = _load_native()
    rank = torch.distributed.get_rank()
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            scaler = torch.amp.GradScaler("cpu", init_scale = _SCALE), scaler_handler = _Handler()
        )
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("tp",))
    torch.manual_seed(0)
    sharded = torch.nn.Parameter(distribute_tensor(torch.randn(8, 4), mesh, [Shard(0)]))
    plain = torch.nn.Parameter(torch.randn(4, 4))
    if optim == "adamw":
        opt = torch.optim.AdamW([sharded, plain], lr = 1e-2, foreach = False)
    else:
        opt = torch.optim.SGD([sharded, plain], lr = 1e-2, momentum = 0.9, foreach = False)
    out = {"unscaled": None}
    for step in range(3):
        loss = sharded.to_local().pow(2).sum() + plain.pow(2).sum()
        scaler.scale(loss).backward()
        if step == 1 and inf_rank in (rank, "all"):
            plain.grad[0, 0] = float("inf")
        scaler.unscale_(opt)
        if step == 0:
            out["unscaled"] = (
                torch.allclose(sharded.grad.to_local(), 2 * sharded.detach().to_local()),
                torch.allclose(plain.grad, 2 * plain.detach()),
            )
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none = True)
    out["steps"] = (
        (float(opt.state[sharded]["step"]), float(opt.state[plain]["step"]))
        if optim == "adamw"
        else None
    )
    out["params"] = [sharded.detach().to_local().clone(), plain.detach().clone()]
    return out


def _sparse_fp16():
    """A float16 sparse plain grad (nn.Embedding(sparse = True)) beside a DTensor grad: upstream coalesces
    it and assigns the result back to param.grad, so that assignment has to reach the real parameter."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    native = _load_native()
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            scaler = torch.amp.GradScaler("cpu", init_scale = _SCALE), scaler_handler = _Handler()
        )
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("tp",))
    sharded = torch.nn.Parameter(distribute_tensor(torch.zeros(8, 4), mesh, [Shard(0)]))
    embed = torch.nn.Parameter(torch.zeros(4, 2, dtype = torch.float16))
    opt = torch.optim.SGD([sharded, embed], lr = 1e-2)
    scaler.scale(torch.zeros(()))
    sharded.grad = distribute_tensor(torch.full((8, 4), _SCALE), mesh, [Shard(0)])
    # rows 0, 0, 2: uncoalesced, row 0 sums to 2 * _SCALE
    embed.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 2]]), torch.full((3, 2), _SCALE, dtype = torch.float16), (4, 2)
    )
    scaler.unscale_(opt)
    return sharded.grad.to_local().clone(), embed.grad.to_dense().float().clone()


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {}
        for optim in ("adamw", "sgd"):
            for inf_rank in (None, 1, "all"):
                try:
                    res[optim, inf_rank] = _run(optim, inf_rank)
                except Exception as e:
                    res[optim, inf_rank] = f"{type(e).__name__}: {e}"
        try:
            res["sparse_fp16"] = _sparse_fp16()
        except Exception as e:
            res["sparse_fp16"] = f"{type(e).__name__}: {e}"
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("tp_scaler"))
    torch.multiprocessing.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


@pytest.mark.parametrize("optim", ["adamw", "sgd"])
@pytest.mark.parametrize(
    "inf_rank", [None, 1, "all"], ids = ["no_inf", "one_rank_inf", "every_rank_inf"]
)
def test_mixed_dtensor_and_plain_grads_unscale_and_skip(ranks, optim, inf_rank):
    for res in ranks:
        got = res[optim, inf_rank]
        assert not isinstance(got, str), got
        assert got["unscaled"] == (True, True), got["unscaled"]
        if optim == "adamw":
            expected = (
                3.0 if inf_rank is None else 2.0
            )  # an overflow on any rank skips it on every rank
            assert got["steps"] == (expected, expected), got["steps"]
        assert all(torch.isfinite(p).all() for p in got["params"])
    torch.testing.assert_close(
        ranks[0][optim, inf_rank]["params"][1], ranks[1][optim, inf_rank]["params"][1]
    )


def test_mixed_grads_unscale_float16_sparse_plain_grad(ranks):
    for res in ranks:
        got = res["sparse_fp16"]
        assert not isinstance(got, str), got
        dense, sparse = got
        assert torch.equal(dense, torch.ones(4, 4)), dense
        # unscaled on the parameter itself, not on a stand-in's copy
        assert torch.equal(
            sparse, torch.tensor([[2.0, 2.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        ), sparse

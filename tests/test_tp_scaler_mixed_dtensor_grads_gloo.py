# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float16 under tensor parallel with DTensor and plain grads in one optimizer (Qwen3-VL under UNSLOTH_TP_SIZE=2:
TP-sharded language model LoRA beside the unsharded vision tower). The native ShardedGradScaler unscaled
them in one foreach call and raised "got mixed torch.Tensor and DTensor" at the first step. On a gloo CPU
world of 2: unscale_ divides both kinds by the scale, an inf on one rank skips the step on both, and the
replicated parameter stays equal over ranks."""

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
    # Per-tensor steps: torch's foreach / fused kernels take one tensor kind per call as well.
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
            # d/dx x^2 = 2x: both kinds back at one GPU's gradient
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


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {}
        for optim in ("adamw", "sgd"):
            for inf_rank in (None, 1, "all"):
                try:
                    res[optim, inf_rank] = _run(optim, inf_rank)
                except Exception as e:  # recorded, asserted in the parent
                    res[optim, inf_rank] = f"{type(e).__name__}: {e}"
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
    # The plain (replicated) parameter took the same steps on both ranks.
    torch.testing.assert_close(
        ranks[0][optim, inf_rank]["params"][1], ranks[1][optim, inf_rank]["params"][1]
    )

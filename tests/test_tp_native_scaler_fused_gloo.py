# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TP fused AdamW without clipping: an inf in one rank's shard must skip the step on every rank."""

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
from torch.distributed.tensor import Replicate, Shard  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402


def _run(fused, unscale):
    from test_tp_native_oracle import QWEN3, _batch, build, tpn
    from torch.distributed.device_mesh import init_device_mesh

    rank = torch.distributed.get_rank()
    old = torch.amp.GradScaler("cpu", init_scale = 2.0**8)
    handler = types.SimpleNamespace(to_kwargs = lambda: {"device": "cpu", "init_scale": 2.0**8})
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = handler)
    )
    tpn._tp_scaler(trainer, 1)
    scaler = trainer.accelerator.scaler
    assert scaler is not old
    model = build(QWEN3)
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # LoRA A is replicated and comes first: DTensor's own found_inf reduction reads only the first
    # gradient's placements, so it does not reduce over tp and the overflow stays on one rank.
    assert tuple(params[0].placements) == (Replicate(),), params[0].placements
    sharded = next(p for p in params if any(isinstance(x, Shard) for x in p.placements))
    opt = torch.optim.AdamW(params, lr = 1e-2, fused = fused, foreach = None if fused else False)
    ids = _batch(model)
    for step in range(3):
        out = model(input_ids = ids, labels = ids, use_cache = False)
        scaler.scale(out.loss).backward()
        if step == 1 and rank == 1:
            sharded.grad.to_local().view(-1)[0] = float("inf")  # only rank 1's shard overflows
        if unscale:
            scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none = True)
    steps = float(opt.state[sharded]["step"])
    return steps, float(scaler.get_scale()), sharded.full_tensor().detach().clone()


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {}
        for fused in (False, True):
            for unscale in (False, True):
                try:
                    res[fused, unscale] = _run(fused, unscale)
                except Exception as e:
                    res[fused, unscale] = repr(e)
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    pytest.importorskip("peft")
    out = str(tmp_path_factory.mktemp("tp_scaler"))
    torch.multiprocessing.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


@pytest.mark.parametrize("unscale", [False, True], ids = ["no_clipping", "clipping"])
@pytest.mark.parametrize("fused", [False, True], ids = ["foreach", "fused"])
def test_tp_overflow_on_one_shard_skips_the_step_on_every_rank(ranks, fused, unscale):
    # Before: fused + no_clipping left rank 0 at 3 steps and scale 2**8 while rank 1 skipped (2, 2**7).
    outs = [res[fused, unscale] for res in ranks]
    for out in outs:
        assert not isinstance(out, str), out
        steps, scale, weight = out
        assert steps == 2.0, (fused, unscale, steps)
        assert scale == 2.0**7, (fused, unscale, scale)
        assert torch.isfinite(weight).all()
    torch.testing.assert_close(outs[0][2], outs[1][2], rtol = 0, atol = 0)

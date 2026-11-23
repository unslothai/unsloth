# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""accelerate reports optimizer_step_was_skipped only when optimizer.step was not called, and the Trainer
advances the lr schedule otherwise. GradScaler leaves a foreach optimizer uncalled on inf grads but calls a
fused one (its kernel skips on found_inf), so fused AdamW ran one schedule step ahead after every overflow.
Native FSDP2 CPU offload switches adamw_torch to fused CPU AdamW, so offload on vs off trained with
different lr from the first overflow (Qwen3-0.6B fp16 on 2 x T4: loss 11.034 vs 11.389 at step 3).
On a gloo CPU world of 2 through the native scaler: fused and foreach skip the same steps, see the same lr
and end at the same weights."""

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


class _Handler:
    def to_kwargs(self):
        return {"device": "cpu", "init_scale": 2.0**8}


def _run(fused, inf_steps):
    from accelerate.optimizer import AcceleratedOptimizer
    from transformers import get_linear_schedule_with_warmup

    native = _load_native()
    rank = torch.distributed.get_rank()
    old = torch.amp.GradScaler("cpu", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _Handler())
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16)
    opt = torch.optim.AdamW(
        model.parameters(), lr = 1e-2, fused = fused, foreach = None if fused else True
    )
    sched = get_linear_schedule_with_warmup(opt, 2, 20)
    aopt = AcceleratedOptimizer(opt, device_placement = False, scaler = scaler)
    lrs, skipped = [], []
    for step in range(5):
        torch.manual_seed(100 + step + 10 * rank)
        scaler.scale(model(torch.randn(4, 16)).pow(2).mean()).backward()
        if step in inf_steps and rank == 1:
            model.weight.grad[0, 0] = float("inf")  # one rank overflows: every rank skips
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lrs.append(opt.param_groups[0]["lr"])
        aopt.step()
        skipped.append(aopt.step_was_skipped)
        if not aopt.step_was_skipped:  # transformers Trainer._inner_training_loop
            sched.step()
        aopt.zero_grad(set_to_none = True)
    return lrs, skipped, float(opt.state[model.weight]["step"]), model.weight.detach().clone()


def _worker(rank, world, port, out):
    from accelerate import Accelerator

    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = str(world)
    )
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        Accelerator(cpu = True)
        res = {}
        for inf_steps in ((), (0,), (0, 3)):
            for fused in (False, True):
                try:
                    res[inf_steps, fused] = _run(fused, inf_steps)
                except Exception as e:  # recorded, asserted in the parent
                    res[inf_steps, fused] = repr(e)
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    pytest.importorskip("accelerate")
    out = str(tmp_path_factory.mktemp("fused_skip"))
    torch.multiprocessing.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


@pytest.mark.parametrize(
    "inf_steps", [(), (0,), (0, 3)], ids = ["no_inf", "inf_step0", "inf_steps0_3"]
)
def test_fused_adamw_skips_like_foreach(ranks, inf_steps):
    for res in ranks:
        foreach, fused = res[inf_steps, False], res[inf_steps, True]
        assert not isinstance(foreach, str), foreach
        assert not isinstance(fused, str), fused
        lrs, skipped, steps, weight = foreach
        assert skipped == [s in inf_steps for s in range(5)], skipped
        assert steps == 5 - len(inf_steps)
        assert fused[1] == skipped, f"fused skipped {fused[1]}, foreach {skipped}"
        assert fused[0] == pytest.approx(lrs), f"fused lr {fused[0]}, foreach {lrs}"
        assert fused[2] == steps
        torch.testing.assert_close(fused[3], weight, rtol = 1e-5, atol = 1e-6)

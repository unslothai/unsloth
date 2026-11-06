# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 float16 swaps the Trainer's GradScaler for a ShardedGradScaler, and swaps adamw_8bit for
adamw_torch_fused. With gradient clipping (unscale_ before step) the stock ShardedGradScaler hands the
fused optimizer a found_inf of shape [1], and its in-place step-count update fails on the first step.
On a gloo CPU world of 2: fused AdamW steps through the swapped scaler, an inf on one rank skips the step
on both ranks, and the result matches the foreach optimizer."""

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


def _run(
    fused,
    both = False,
    unscale = True,
):
    native = _load_native()
    rank = torch.distributed.get_rank()
    old = torch.amp.GradScaler("cpu", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _Handler())
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(
        model.parameters(), lr = 1e-2, fused = fused, foreach = None if fused else True
    )
    for step in range(3):
        torch.manual_seed(100 + 10 * step + rank)
        loss = model(torch.randn(4, 8)).pow(2).mean()
        scaler.scale(loss).backward()
        if step == 1 and (both or rank == 1):
            model.weight.grad[0, 0] = float("inf")  # one rank (or every rank) overflows: all skip
        if unscale:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none = True)
    steps = float(opt.state[model.weight]["step"])
    return steps, float(scaler.get_scale()), [p.detach().clone() for p in model.parameters()]


def _device_kept():
    # No handler device: the swapped scaler keeps the old scaler's device (a "cuda" default on a CPU or
    # XPU run is a disabled scaler, so float16 would train without loss scaling).
    native = _load_native()
    old = torch.amp.GradScaler("cpu", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = None)
    )
    native._scaler_for_native(trainer)
    new = trainer.accelerator.scaler
    return new._device, new.is_enabled()


class _DisabledHandler:
    def to_kwargs(self):
        return {"device": "cpu", "enabled": False}


def _disabled():
    # GradScalerKwargs(enabled=False): unscale_ returns before any optimizer state exists.
    native = _load_native()
    old = torch.amp.GradScaler("cpu")
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _DisabledHandler())
    )
    native._scaler_for_native(trainer)
    scaler = trainer.accelerator.scaler
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-2, fused = True)
    scaler.scale(model(torch.randn(4, 8)).pow(2).mean()).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    return float(opt.state[model.weight]["step"])


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = {}
        for fused in (False, True):
            for both in (False, True):
                try:
                    res[fused, both] = _run(fused, both)
                except Exception as e:  # recorded, asserted in the parent
                    res[fused, both] = repr(e)
                try:
                    res[fused, both, "no_unscale"] = _run(fused, both, unscale = False)
                except Exception as e:
                    res[fused, both, "no_unscale"] = repr(e)
        res["device"] = _device_kept()
        try:
            res["disabled"] = _disabled()
        except Exception as e:
            res["disabled"] = repr(e)
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("scaler"))
    torch.multiprocessing.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


def test_swapped_scaler_keeps_the_device(ranks):
    for res in ranks:
        assert res["device"] == ("cpu", True), res["device"]


@pytest.mark.parametrize("both", [False, True], ids = ["one_rank_inf", "every_rank_inf"])
def test_fused_adamw_steps_through_the_native_fp16_scaler(ranks, both):
    # every_rank_inf: the all-reduced found_inf is the number of ranks that overflowed (2 here). The
    # fused AdamW kernel skips only on found_inf == 1, so a 2 stepped on inf / NaN gradients and wrote
    # NaN weights (Qwen2.5-VL on Kaggle T4x2: loss NaN from step 4 under FSDP2, finite under DDP).
    for res in ranks:
        for fused in (False, True):
            out = res[fused, both]
            assert not isinstance(out, str), f"fused={fused}: {out}"
            steps, scale, params = out
            assert steps == 2.0, (fused, steps)  # 3 steps, the overflow one skipped on every rank
            # user init_scale is kept, not divided by the ranks
            assert scale == 2.0**7, (fused, scale)
            assert all(torch.isfinite(p).all() for p in params), fused
        for a, b in zip(res[False, both][2], res[True, both][2]):
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("both", [False, True], ids = ["one_rank_inf", "every_rank_inf"])
def test_overflow_skips_on_every_rank_without_explicit_unscale(ranks, both):
    # Fused step() without unscale_; before the fix one_rank_inf left ranks at 3 vs 2 steps.
    for res in ranks:
        for fused in (False, True):
            out = res[fused, both, "no_unscale"]
            assert not isinstance(out, str), f"fused={fused}: {out}"
            steps, scale, params = out
            assert steps == 2.0, (fused, both, steps)
            assert scale == 2.0**7, (fused, both, scale)
            assert all(torch.isfinite(p).all() for p in params), fused
        for a, b in zip(res[False, both, "no_unscale"][2], res[True, both, "no_unscale"][2]):
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


def test_disabled_scaler_still_steps(ranks):
    for res in ranks:
        assert res["disabled"] == 1.0, res["disabled"]

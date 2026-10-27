# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused AdamW over CPU + CUDA params: both overflowing summed found_inf to 2, so the step was not skipped."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


class _Handler:
    def to_kwargs(self):
        return {"device": "cuda", "init_scale": 2.0**8}


@pytest.fixture(scope = "module")
def world_of_one():
    # Once per module: ShardedGradScaler's default process_group argument is bound to the first WORLD.
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()))
    dist.init_process_group("cpu:gloo,cuda:nccl", rank = 0, world_size = 1)
    torch.cuda.set_device(0)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("explicit_unscale", [True, False], ids = ["clipping", "no_clipping"])
@pytest.mark.parametrize("overflow", ["cpu", "cuda", "both"])
def test_mixed_device_fused_adamw_skips_an_overflow_step(world_of_one, explicit_unscale, overflow):
    native = _load_native()
    old = torch.amp.GradScaler("cuda", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _Handler())
    )
    assert native._scaler_for_native(trainer) is True
    scaler = trainer.accelerator.scaler
    torch.manual_seed(0)
    on_cpu = torch.nn.Parameter(torch.randn(8, 8))
    on_cuda = torch.nn.Parameter(torch.randn(8, 8, device = "cuda"))
    opt = torch.optim.AdamW([on_cpu, on_cuda], lr = 1e-2, fused = True)
    before = [on_cpu.detach().clone(), on_cuda.detach().clone()]
    loss = on_cpu.sum().cuda() + on_cuda.sum()
    scaler.scale(loss).backward()
    if overflow in ("cpu", "both"):
        on_cpu.grad[0, 0] = float("inf")
    if overflow in ("cuda", "both"):
        on_cuda.grad[0, 0] = float("inf")
    if explicit_unscale:
        scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    for p, b in zip((on_cpu, on_cuda), before):
        assert torch.equal(p.detach().cpu(), b.cpu()), (
            overflow,
            "the overflow step changed weights",
        )
        assert float(opt.state[p]["step"]) == 0.0, (overflow, float(opt.state[p]["step"]))
    assert scaler.get_scale() == 2.0**7


def _scaler():
    native = _load_native()
    old = torch.amp.GradScaler("cuda", init_scale = 2.0**8)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(scaler = old, scaler_handler = _Handler())
    )
    assert native._scaler_for_native(trainer) is True
    return trainer.accelerator.scaler


@pytest.mark.parametrize("explicit_unscale", [True, False], ids = ["clipping", "no_clipping"])
@pytest.mark.parametrize("overflow", ["none", "cpu", "cuda", "both"])
def test_late_device_to_host_copies_keep_the_step_right(world_of_one, explicit_unscale, overflow):
    scaler = _scaler()
    torch.manual_seed(0)
    on_cpu = torch.nn.Parameter(torch.randn(8, 8))
    on_cuda = torch.nn.Parameter(torch.randn(8, 8, device = "cuda"))
    opt = torch.optim.AdamW([on_cpu, on_cuda], lr = 1e-2, fused = True)
    ref_cpu = torch.nn.Parameter(on_cpu.detach().clone())
    ref_opt = torch.optim.AdamW([ref_cpu], lr = 1e-2)
    before = [on_cpu.detach().clone(), on_cuda.detach().clone()]
    scaler.scale(on_cpu.sum().cuda() + on_cuda.sum()).backward()
    if overflow in ("cpu", "both"):
        on_cpu.grad[0, 0] = float("inf")
    if overflow in ("cuda", "both"):
        on_cuda.grad[0, 0] = float("inf")
    torch.cuda.synchronize()
    torch.cuda._sleep(
        200_000_000
    )  # keep the stream busy: any non_blocking copy to the host lands late
    if explicit_unscale:
        scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()
    if overflow == "none":
        ref_cpu.grad = torch.ones_like(ref_cpu)
        ref_opt.step()
        torch.testing.assert_close(on_cpu.detach(), ref_cpu.detach())
        assert scaler.get_scale() == 2.0**8
    else:
        for p, b in zip((on_cpu, on_cuda), before):
            assert torch.equal(p.detach().cpu(), b.cpu()), (
                overflow,
                "the overflow step changed weights",
            )
            assert float(opt.state[p]["step"]) == 0.0, (overflow, float(opt.state[p]["step"]))
        assert scaler.get_scale() == 2.0**7

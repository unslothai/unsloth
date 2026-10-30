# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CP over DDP / accelerate FSDP2 must start the float16 GradScaler at 2**16 / world (rank grads are world x one GPU's)."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

SCALE = 2.0**16
TOKENS = 8
FP16_GRAD = (
    40000.0  # one GPU's scaled per-token output gradient at 2**16: below the float16 max (65504)
)


class _Handler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_kwargs(self):
        return dict(self.kwargs)


def _trainer(
    world,
    scaler,
    handler = None,
):
    accelerator = types.SimpleNamespace(
        scaler = scaler,
        scaler_handler = handler,
        num_processes = world,
        state = types.SimpleNamespace(device_mesh = None),
    )
    manager = types.SimpleNamespace(device_mesh = None, fsdp2 = False)
    return types.SimpleNamespace(
        accelerator = accelerator,
        model = torch.nn.Linear(2, 2),
        args = types.SimpleNamespace(ddp_find_unused_parameters = False),
        _context_parallel_manager = manager,
    )


def _found_inf(scaler, world):
    """One CP rank's float16 backward at the scaler's start scale: world x one GPU's per-token gradient."""
    torch.manual_seed(0)
    x = torch.rand(TOKENS // world, 8) * 1e-3
    w = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
    coef = FP16_GRAD * TOKENS / SCALE
    out = torch.nn.functional.linear(x.half(), w.half())
    opt = torch.optim.SGD([w], lr = 1e-3)
    scaler.scale((out.float() * coef).sum() / TOKENS * world).backward()
    scaler.unscale_(opt)
    return float(
        sum(
            v.item() for v in scaler._per_optimizer_states[id(opt)]["found_inf_per_device"].values()
        )
    )


@pytest.mark.parametrize("world", [2, 4])
def test_cp_trainer_starts_the_scaler_at_two_16_over_world(world):
    trainer = _trainer(world, torch.amp.GradScaler("cpu"))
    cp._install_accelerator_state(trainer)
    scaler = trainer.accelerator.scaler
    assert scaler._init_scale == SCALE / world
    # The hook runs again on train / evaluate: divided once.
    cp._install_accelerator_state(trainer)
    assert scaler._init_scale == SCALE / world
    assert _found_inf(scaler, world) == 0.0
    # The start scale it replaces overflows on the same CP backward (what one GPU at 2**16 does not).
    assert _found_inf(torch.amp.GradScaler("cpu", init_scale = SCALE), world) == 1.0
    assert _found_inf(torch.amp.GradScaler("cpu", init_scale = SCALE), 1) == 0.0


def test_cp_scaler_start_leaves_user_started_and_native_scalers():
    user = _trainer(2, torch.amp.GradScaler("cpu", init_scale = SCALE), _Handler(init_scale = SCALE))
    cp._install_accelerator_state(user)
    assert user.accelerator.scaler._init_scale == SCALE
    native = _trainer(2, torch.amp.GradScaler("cpu", init_scale = SCALE / 2))  # FSDP2 / TP set it
    cp._install_accelerator_state(native)
    assert native.accelerator.scaler._init_scale == SCALE / 2
    started = _trainer(2, torch.amp.GradScaler("cpu"))
    started.accelerator.scaler.scale(torch.ones(()))
    cp._install_accelerator_state(started)
    assert started.accelerator.scaler.get_scale() == SCALE
    off = _trainer(2, None)
    cp._install_accelerator_state(off)
    no_cp = _trainer(2, torch.amp.GradScaler("cpu"))
    no_cp._context_parallel_manager = None
    cp._install_accelerator_state(no_cp)
    assert no_cp.accelerator.scaler._init_scale == SCALE

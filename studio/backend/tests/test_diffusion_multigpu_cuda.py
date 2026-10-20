# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG-parallel on two real CUDA cards. On one device ``.to()`` aliases storage, so the CPU tests in
test_diffusion_multigpu.py cannot see a replica that diverged from its source; these can. Skipped below two cards."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_multigpu as mg

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = False):
        return ((self.proj(hidden_states) * self.scale) * timestep + encoder_hidden_states.sum(),)


def _render(dit, latents, cond, neg, steps, between = None):
    x = latents.clone()
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps, device = x.device)
        a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond)[0]
        if between is not None:
            between(i, x)
        b = dit(hidden_states = x, timestep = t, encoder_hidden_states = neg)[0]
        x = x - 0.1 * (b + 4.0 * (a - b))
    return x


def _setup():
    torch.manual_seed(0)
    dit = _Dit().cuda(0)
    latents = torch.randn(2, 4, device = "cuda:0")
    cond = torch.randn(3, 4, device = "cuda:0")
    neg = torch.randn(3, 4, device = "cuda:0")
    return dit, latents, cond, neg


def _arm(dit):
    router = mg.CfgParallelRouter(dit, primary = "cuda:0", secondary = "cuda:1")
    dit.forward = router
    return router


def test_an_input_edited_after_the_prefetch_reaches_an_unpredicted_replica_call():
    dit, latents, cond, neg = _setup()

    def edit(step, x):
        if step == 1:  # the first step of a render runs unpredicted (no template yet): only the prefetch was made
            x.add_(1.0)

    with torch.no_grad():
        reference = [_render(dit, latents, cond, neg, 4), _render(dit, latents, cond, neg, 4, edit)]
        router = _arm(dit)
        first = _render(dit, latents, cond, neg, 4)
        router.end_render()
        second = _render(dit, latents, cond, neg, 4, edit)
    torch.cuda.synchronize()
    assert router.stats["replica_calls"] > 0 and router.replica is not None
    assert torch.equal(first, reference[0])
    assert torch.equal(second, reference[1])


def test_an_edited_buffer_rebuilds_the_replica():
    dit, latents, cond, neg = _setup()
    with torch.no_grad():
        router = _arm(dit)
        _render(dit, latents, cond, neg, 3)
        router.end_render()
        assert router.replica is not None
        dit.scale.mul_(3.0)  # e.g. a re-quantised scale: the copy on the other card still holds the old one
        dit.forward = dit.__class__.forward.__get__(dit)
        reference = _render(dit, latents, cond, neg, 4)
        dit.forward = router
        out = _render(dit, latents, cond, neg, 4)
    torch.cuda.synchronize()
    assert router.stats["replica_rebuilds"] >= 1
    assert torch.equal(out, reference)

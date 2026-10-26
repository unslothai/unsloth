# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The local-shard bitsandbytes optimizer skips bitsandbytes' per-parameter sync_gpu unless paged.
Runs on the CPU: sync_gpu is counted, not executed."""

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

_SHAPES = ((300, 16), (16, 300), (5000,), (40,))


def _params(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator = g)) for s in _SHAPES]


def _set_grads(params, step):
    g = torch.Generator().manual_seed(100 + step)
    for p in params:
        p.grad = torch.randn(p.shape, generator = g) * 1e-2


@pytest.fixture
def sync_calls(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    _load_native()
    from unsloth.models.fsdp2_bnb_optim import bnb_steps_cpu_tensors

    if not bnb_steps_cpu_tensors():
        pytest.skip("this bitsandbytes cannot step CPU tensors")
    import bitsandbytes.optim.optimizer as bnb_optimizer
    import bitsandbytes.utils as bnb_utils

    calls = []
    for module in (bnb_optimizer, bnb_utils):
        if hasattr(module, "sync_gpu"):
            monkeypatch.setattr(module, "sync_gpu", lambda t: calls.append(t.device.type))
    return bnb, calls


@pytest.mark.parametrize("name", ["AdamW8bit", "Adam8bit", "AdamW"])
def test_local_shard_step_does_not_sync_per_parameter(sync_calls, name):
    bnb, calls = sync_calls
    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    cls = getattr(bnb.optim, name)
    local, ref = _params(0), _params(0)
    opt = cls(local, lr = 1e-2, min_8bit_size = 0)
    ref_opt = cls(ref, lr = 1e-2, min_8bit_size = 0)
    assert localize_bnb_optimizer(opt)
    hooks = []
    opt.register_step_post_hook(lambda *a: hooks.append(1))
    for step in range(3):
        _set_grads(local, step)
        _set_grads(ref, step)
        ref_opt.step()
        calls.clear()
        opt.step()
        assert calls == [], (name, step, calls)
    for a, b in zip(local, ref):
        assert torch.equal(a, b)
    assert len(hooks) == 3


def test_paged_local_shard_step_keeps_bitsandbytes_fences(sync_calls):
    bnb, calls = sync_calls
    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    local, ref = _params(1), _params(1)
    opt = bnb.optim.PagedAdamW8bit(local, lr = 1e-2, min_8bit_size = 0)
    ref_opt = bnb.optim.PagedAdamW8bit(ref, lr = 1e-2, min_8bit_size = 0)
    assert localize_bnb_optimizer(opt)
    for step in range(2):
        _set_grads(local, step)
        _set_grads(ref, step)
        calls.clear()
        ref_opt.step()
        expected = len(calls)
        calls.clear()
        opt.step()
        assert expected == len(_SHAPES) + 1
        assert len(calls) == expected, (step, calls)
    for a, b in zip(local, ref):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA side stream")
def test_local_shard_step_on_a_side_stream_keeps_every_fence(monkeypatch):
    """Off the default stream the per-parameter fences stay; on it none are taken."""
    bnb = pytest.importorskip("bitsandbytes")
    _load_native()
    import bitsandbytes.optim.optimizer as bnb_optimizer
    import bitsandbytes.utils as bnb_utils
    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    calls = []
    for module in (bnb_optimizer, bnb_utils):
        if hasattr(module, "sync_gpu"):
            monkeypatch.setattr(module, "sync_gpu", lambda t: calls.append(t.device.type))
    params = [torch.nn.Parameter(p.detach().cuda()) for p in _params(2)]
    opt = bnb.optim.AdamW8bit(params, lr = 1e-2, min_8bit_size = 0)
    assert localize_bnb_optimizer(opt)
    for p in params:
        p.grad = torch.full_like(p, 1e-2)
    opt.step()
    assert calls == []
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        opt.step()
    assert len(calls) == len(_SHAPES) + 1, calls


def test_local_shard_step_runs_when_the_scheduler_was_built_first(sync_calls):
    """Trainer(optimizers = (opt, scheduler)): the scheduler wraps opt.step before localization."""
    bnb, calls = sync_calls
    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    local, ref = _params(0), _params(0)
    opt = bnb.optim.AdamW8bit(local, lr = 1e-2, min_8bit_size = 0)
    ref_opt = bnb.optim.AdamW8bit(ref, lr = 1e-2, min_8bit_size = 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    assert localize_bnb_optimizer(opt)
    hooks = []
    opt.register_step_post_hook(lambda *a: hooks.append(1))
    for step in range(2):
        _set_grads(local, step)
        _set_grads(ref, step)
        ref_opt.step()
        calls.clear()
        opt.step()
        assert calls == [], calls
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error"
            )  # "lr_scheduler.step() before optimizer.step()" would raise
            sched.step()
    for a, b in zip(local, ref):
        assert torch.equal(a, b)
    assert len(hooks) == 2

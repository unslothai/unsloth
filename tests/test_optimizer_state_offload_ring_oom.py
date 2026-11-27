# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Optimizer state offload falls back to serial chunk copies when the staging ring's allocation OOMs."""

import importlib.util
import os

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPT = os.path.join(_ROOT, "unsloth", "optimizers")

needs_cuda = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
# ROCm keeps the serial copies: no ring is ever allocated.
_ENGAGES = not getattr(torch.version, "hip", None)


def _load(name, tag):
    spec = importlib.util.spec_from_file_location(tag, os.path.join(_OPT, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(ring_fails):
    module = _load("state_offload.py", "_unsloth_state_offload_ring_oom")
    if ring_fails:

        class _NoRoom(module._Ring):
            def __init__(self, device, capacity, streams):
                raise torch.cuda.OutOfMemoryError(
                    f"CUDA out of memory. Tried to allocate {capacity} bytes."
                )

        module._Ring = _NoRoom
    return module


def _host_state(optimizer):
    out = []
    for state in optimizer.state.values():
        for key, value in sorted(state.items(), key = lambda kv: kv[0]):
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", key
                out.append(value.clone())
            else:
                out.append(value)
    return out


def _assert_same(params_a, opt_a, params_b, opt_b):
    for x, y in zip(params_a, params_b):
        assert torch.equal(x, y)
    sa, sb = _host_state(opt_a), _host_state(opt_b)
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y


def _chunked(
    monkeypatch,
    overlap,
    ring_fails,
    steps = 4,
):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
    module = _module(ring_fails)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 256),
        torch.nn.Linear(256, 64),
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, foreach = True)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 600 << 10)
    x = torch.randn(16, 256, device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(1))
    for step in range(steps):
        loss = model(x + step).float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
    torch.cuda.synchronize()
    return list(model.parameters()), optimizer, offloader


@needs_cuda
def test_chunked_step_falls_back_to_serial_when_the_ring_does_not_fit(monkeypatch):
    p0, o0, off0 = _chunked(monkeypatch, overlap = False, ring_fails = False)
    p1, o1, off1 = _chunked(monkeypatch, overlap = True, ring_fails = True)
    _assert_same(p0, o0, p1, o1)
    assert off1.overlapped_steps == 0
    # Every step after the first (which creates the state) planned a ring and fell back.
    assert off1.ring_oom_fallbacks == (3 if _ENGAGES else 0)
    assert off0.ring_oom_fallbacks == 0


@needs_cuda
def test_ring_that_fits_still_overlaps(monkeypatch):
    p0, o0, _ = _chunked(monkeypatch, overlap = False, ring_fails = False)
    p1, o1, off1 = _chunked(monkeypatch, overlap = True, ring_fails = False)
    _assert_same(p0, o0, p1, o1)
    assert off1.overlapped_steps == (3 if _ENGAGES else 0)
    assert off1.ring_oom_fallbacks == 0


def _per_weight(
    monkeypatch,
    overlap,
    ring_fails,
    steps = 3,
):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    module = _module(ring_fails)
    sr = _load("bf16_stochastic_rounding.py", "_unsloth_sr_ring_oom")
    generator = torch.Generator(device = "cuda").manual_seed(0)
    shapes = [(1024, 512), (64,), (2048, 256), (512, 512)]
    params = [
        torch.nn.Parameter(
            (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
        )
        for s in shapes
    ]
    optimizer = torch.optim.AdamW(params, lr = 1e-3, weight_decay = 0.01)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
    assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == len(params)
    for _ in range(steps):
        for p in params:
            p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                torch.bfloat16
            )
        optimizer.step()
    torch.cuda.synchronize()
    return params, optimizer, offloader


@needs_cuda
def test_per_weight_sequence_falls_back_to_serial_when_the_ring_does_not_fit(monkeypatch):
    p0, o0, off0 = _per_weight(monkeypatch, overlap = False, ring_fails = False)
    p1, o1, off1 = _per_weight(monkeypatch, overlap = True, ring_fails = True)
    _assert_same(p0, o0, p1, o1)
    assert off1.overlapped_steps == 0
    assert off1.sequence is None
    assert off1.ring_oom_fallbacks == (2 if _ENGAGES else 0)

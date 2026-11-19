# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the frozen-fp8 backward weight cache: grad_x is bit-identical with and without it, the cache keeps
to its budget, gives the room back, and a budget change neither breaks nor recompiles a compiled graph."""

from __future__ import annotations

import pytest
import torch

import core.training.diffusion_frozen_fp8 as f8

GIB = 1 << 30


def _cpu_scaled_mm_ok() -> bool:
    try:
        a = torch.zeros(16, 16).to(torch.float8_e4m3fn)
        one = torch.ones((), dtype = torch.float32)
        torch._scaled_mm(a, a.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        return True
    except Exception:  # noqa: BLE001 -- older torch: no CPU fp8 GEMM
        return False


pytestmark = pytest.mark.skipif(
    not _cpu_scaled_mm_ok(), reason = "torch._scaled_mm has no CPU fp8 kernel here"
)


@pytest.fixture(autouse = True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FP8_BWD_CACHE", raising = False)
    f8.BACKWARD_WEIGHTS.clear()
    f8.BACKWARD_WEIGHTS.hits = f8.BACKWARD_WEIGHTS.misses = 0
    f8.BACKWARD_WEIGHTS.active = True
    yield
    f8.BACKWARD_WEIGHTS.clear()
    f8.BACKWARD_WEIGHTS.active = False


def _model(n = 3, width = 64):
    torch.manual_seed(0)
    layers = [torch.nn.Linear(width, width, bias = i % 2 == 0).to(torch.bfloat16) for i in range(n)]
    return torch.nn.Sequential(*[f8.FrozenFp8Linear(l) for l in layers])


def _grad_x(model, seed = 1):
    torch.manual_seed(seed)
    x = torch.randn(2, 48, 64, dtype = torch.bfloat16, requires_grad = True)
    model(x).float().pow(2).mean().backward()
    return x.grad.clone()


def test_grad_x_is_bit_identical_with_the_cache():
    model = _model()
    ref = [_grad_x(model, s) for s in (1, 2, 3)]
    assert f8.BACKWARD_WEIGHTS.entries == {}
    f8.BACKWARD_WEIGHTS.plan(GIB)
    got = [_grad_x(model, s) for s in (1, 2, 3)]
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    stats = f8.BACKWARD_WEIGHTS.stats()
    # The first backward fills one entry per layer; the next two read them.
    assert stats["fp8_bwd_layers"] == 3 and stats["fp8_bwd_hits"] == 6
    for q, qcm in f8.BACKWARD_WEIGHTS.entries.values():
        assert torch.equal(qcm.float(), q.float()) and qcm.stride() == (1, q.shape[0])


def test_cache_keeps_to_its_budget():
    model = _model(n = 4)
    one = 64 * 64  # bytes per e4m3 weight
    f8.BACKWARD_WEIGHTS.plan(2 * one + 10)
    _grad_x(model)
    assert len(f8.BACKWARD_WEIGHTS.entries) == 2 and f8.BACKWARD_WEIGHTS.used == 2 * one
    f8.BACKWARD_WEIGHTS.plan(one)
    assert f8.BACKWARD_WEIGHTS.used <= one


def test_no_budget_keeps_mains_behaviour():
    _grad_x(_model())
    assert f8.BACKWARD_WEIGHTS.entries == {} and f8.BACKWARD_WEIGHTS.used == 0


def test_inactive_keeps_the_inline_copy():
    # Outside a budgeted trainer run (the load probe, other callers) the backward never touches the cache.
    model = _model()
    ref = _grad_x(model)
    f8.BACKWARD_WEIGHTS.active = False
    f8.BACKWARD_WEIGHTS.hits = f8.BACKWARD_WEIGHTS.misses = 0
    f8.BACKWARD_WEIGHTS.plan(GIB)
    assert torch.equal(_grad_x(model), ref)
    assert (
        f8.BACKWARD_WEIGHTS.hits == f8.BACKWARD_WEIGHTS.misses == 0
        and f8.BACKWARD_WEIGHTS.entries == {}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize(
    "shape", [(64, 64), (3072, 9216), (9216, 3072), (100, 37), (1, 4096), (4096, 1), (130, 260)]
)
def test_tiled_transpose_matches_aten_on_cuda(shape):
    q = torch.randn(shape, device = "cuda").to(torch.float8_e4m3fn)
    got = f8._column_major(q)
    ref = q.t().contiguous().t()
    assert got.shape == ref.shape and got.stride() == ref.stride() and got.dtype == q.dtype
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    assert f8._transpose_kernel() is not None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FP8_BWD_CACHE", "0")
    f8.BACKWARD_WEIGHTS.plan(GIB)
    _grad_x(_model())
    assert f8.BACKWARD_WEIGHTS.budget == 0 and f8.BACKWARD_WEIGHTS.entries == {}


def test_lend_frees_and_return_restores():
    model = _model()
    f8.BACKWARD_WEIGHTS.plan(GIB)
    _grad_x(model)
    handle = f8.fp8_backward_vram_handler()
    held = f8.BACKWARD_WEIGHTS.used
    ack = handle("lend")
    assert held == 3 * 64 * 64 and ack["fp8_bwd_cached_gib"] == 0.0
    assert f8.BACKWARD_WEIGHTS.entries == {} and f8.BACKWARD_WEIGHTS.budget == 0
    _grad_x(model)
    assert f8.BACKWARD_WEIGHTS.entries == {}
    handle("return")
    assert f8.BACKWARD_WEIGHTS.budget == GIB
    _grad_x(model)
    assert len(f8.BACKWARD_WEIGHTS.entries) == 3


def test_return_never_restores_more_than_the_room_left():
    model = _model()
    f8.BACKWARD_WEIGHTS.plan(GIB)
    _grad_x(model)
    room = {"bytes": GIB}
    handle = f8.fp8_backward_vram_handler(room = lambda: room["bytes"])
    handle("lend")
    # A step-down (or the lent load) left room for one layer's copy only.
    room["bytes"] = 64 * 64
    handle("return")
    assert f8.BACKWARD_WEIGHTS.budget == 64 * 64
    _grad_x(model)
    assert len(f8.BACKWARD_WEIGHTS.entries) == 1
    handle("lend")
    room["bytes"] = 0
    handle("return")
    assert f8.BACKWARD_WEIGHTS.budget == 0 and f8.BACKWARD_WEIGHTS.entries == {}

    def unreadable():
        raise RuntimeError("no device")

    f8.BACKWARD_WEIGHTS.plan(GIB)
    handle = f8.fp8_backward_vram_handler(room = unreadable)
    handle("lend")
    handle("return")
    assert f8.BACKWARD_WEIGHTS.budget == 0


def test_a_freed_weight_is_never_served_for_another():
    a = _model(n = 1)
    f8.BACKWARD_WEIGHTS.plan(GIB)
    _grad_x(a)
    ((q, _qcm),) = f8.BACKWARD_WEIGHTS.entries.values()
    del a
    # The entry holds its source weight, so its storage (the key) cannot be handed to another tensor.
    assert q.untyped_storage().data_ptr() in f8.BACKWARD_WEIGHTS.entries


def test_compiled_graph_neither_breaks_nor_recompiles_when_the_budget_changes():
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        from torch._dynamo.backends.common import aot_autograd
        return aot_autograd(fw_compiler = lambda g, _i: g, bw_compiler = lambda g, _i: g)(
            gm, example_inputs
        )

    model = _model()
    ref = _grad_x(model)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend = backend, fullgraph = True)
    out = [_grad_x(compiled)]
    f8.BACKWARD_WEIGHTS.plan(GIB)
    out += [_grad_x(compiled), _grad_x(compiled)]
    f8.BACKWARD_WEIGHTS.clear()
    out.append(_grad_x(compiled))
    torch._dynamo.reset()
    assert len(graphs) == 1
    assert all(torch.equal(ref, g) for g in out)
    assert f8.BACKWARD_WEIGHTS.hits >= 3

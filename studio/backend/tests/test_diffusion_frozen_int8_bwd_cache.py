# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the W8A8 int8 backward weight cache: grad_x is bit-identical with and without it, entries are one
per module and never outlive the weight they copied, the budget holds, and a budget change does not recompile."""

from __future__ import annotations

import pytest
import torch

import core.training.diffusion_frozen_int8 as i8

GIB = 1 << 30
CACHE = i8.BACKWARD_WEIGHTS


def _cpu_int_mm_ok() -> bool:
    try:
        a = torch.ones((32, 32), dtype = torch.int8)
        torch._int_mm(a, a.t())
        return True
    except Exception:  # noqa: BLE001 -- a torch with no CPU int8 GEMM
        return False


pytestmark = pytest.mark.skipif(not _cpu_int_mm_ok(), reason = "torch._int_mm has no CPU kernel here")


@pytest.fixture(autouse = True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv(i8.Int8BackwardWeights.ENV, raising = False)
    CACHE.clear()
    CACHE.hits = CACHE.misses = 0
    CACHE.active = True
    yield
    CACHE.clear()
    CACHE.active = False


def _model(n = 3, width = 64):
    torch.manual_seed(0)
    layers = [torch.nn.Linear(width, width, bias = i % 2 == 0).to(torch.bfloat16) for i in range(n)]
    return torch.nn.Sequential(*[i8.FrozenInt8Linear(l) for l in layers])


def _grad_x(model, seed = 1):
    torch.manual_seed(seed)
    x = torch.randn(2, 48, 64, dtype = torch.bfloat16, requires_grad = True)
    model(x).float().pow(2).mean().backward()
    return x.grad.clone()


def test_grad_x_is_bit_identical_with_the_cache():
    model = _model()
    ref = [_grad_x(model, s) for s in (1, 2, 3)]
    assert CACHE.entries == {}
    CACHE.plan(GIB)
    got = [_grad_x(model, s) for s in (1, 2, 3)]
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    stats = CACHE.stats()
    assert stats["int8_bwd_layers"] == 3 and stats["int8_bwd_hits"] == 6
    assert set(CACHE.entries) == {id(m) for m in model}
    for m in model:
        qcm = CACHE.entries[id(m)][-1]
        assert torch.equal(qcm, m.weight) and qcm.stride() == (1, m.weight.shape[0])


def test_a_replaced_weight_refreshes_its_one_entry():
    model = _model(n = 1)
    CACHE.plan(GIB)
    _grad_x(model)
    (layer,) = model
    # A new weight tensor (as a reload or a device move would install): the stale copy must not be served.
    new = torch.randint(-127, 128, layer.weight.shape, dtype = torch.int8)
    layer._buffers["weight"] = new
    layer._own_weight()
    got = _grad_x(model)
    CACHE.active = False
    assert torch.equal(got, _grad_x(model))
    assert len(CACHE.entries) == 1 and CACHE.used == new.numel()
    assert torch.equal(CACHE.entries[id(layer)][-1], new)


def test_an_in_place_write_is_never_served_stale():
    model = _model(n = 1)
    CACHE.plan(GIB)
    _grad_x(model)
    (layer,) = model
    with torch.no_grad():
        layer.weight.neg_()
    got = _grad_x(model)
    CACHE.active = False
    assert torch.equal(got, _grad_x(model))


def test_to_keeps_the_owner():
    model = _model(n = 2)
    model.to(torch.device("cpu"))
    CACHE.plan(GIB)
    _grad_x(model)
    _grad_x(model)
    assert len(CACHE.entries) == 2 and CACHE.hits == 2


def test_unowned_weight_copies_inline():
    q = torch.randint(-127, 128, (64, 64), dtype = torch.int8)
    CACHE.plan(GIB)
    qcm = CACHE.get(q)
    assert torch.equal(qcm, q) and CACHE.entries == {} and CACHE.misses == 1


def test_cache_keeps_to_its_budget():
    model = _model(n = 4)
    one = 64 * 64
    CACHE.plan(2 * one + 10)
    _grad_x(model)
    assert len(CACHE.entries) == 2 and CACHE.used == 2 * one
    CACHE.plan(one)
    assert CACHE.used <= one


def test_inactive_and_kill_switch_keep_mains_behaviour(monkeypatch):
    model = _model()
    ref = _grad_x(model)
    CACHE.active = False
    CACHE.hits = CACHE.misses = 0
    CACHE.plan(GIB)
    assert torch.equal(_grad_x(model), ref) and CACHE.hits == CACHE.misses == 0
    CACHE.active = True
    monkeypatch.setenv(i8.Int8BackwardWeights.ENV, "0")
    CACHE.plan(GIB)
    assert torch.equal(_grad_x(model), ref)
    assert CACHE.budget == 0 and CACHE.entries == {}


def test_fp8_cache_is_separate():
    import core.training.diffusion_frozen_fp8 as f8

    assert f8.BACKWARD_WEIGHTS is not CACHE
    assert f8.Fp8BackwardWeights.ENV == "UNSLOTH_DIFFUSION_FP8_BWD_CACHE"
    assert "fp8_bwd_hits" in f8.BACKWARD_WEIGHTS.stats()


def test_compiled_budget_change_does_not_recompile():
    torch._dynamo.reset()
    model = _model()
    compiled = torch.compile(model)
    counters = torch._dynamo.utils.counters
    counters.clear()
    out = []
    for budget in (0, GIB, GIB, 0):
        CACHE.plan(budget)
        torch.manual_seed(1)
        x = torch.randn(2, 48, 64, dtype = torch.bfloat16, requires_grad = True)
        compiled(x).float().pow(2).mean().backward()
        out.append(x.grad.clone())
    # The compiled forward may round differently from eager; with and without the cache it must not differ.
    assert all(torch.equal(out[0], g) for g in out[1:])
    assert counters["stats"]["unique_graphs"] == 1
    assert CACHE.hits >= 3

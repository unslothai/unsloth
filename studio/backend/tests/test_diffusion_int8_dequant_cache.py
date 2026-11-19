# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dequant cache on torchao weight-only int8 bases: bit-identical to torchao's own forward and backward, inside
its budget, released on lend, and off where it cannot be proven exact.

CPU only: bf16 autocast on CPU drives the same code as CUDA autocast."""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("torchao")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

from core.training import diffusion_lora_fast as fl  # noqa: E402

BF16 = torch.bfloat16


class _Block(torch.nn.Module):
    def __init__(
        self,
        d = 64,
        h = 96,
    ):
        super().__init__()
        self.q = torch.nn.Linear(d, h, bias = True)
        self.o = torch.nn.Linear(h, d, bias = False)
        self.ff = torch.nn.Linear(d, d, bias = True)

    def forward(self, x):
        n = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        return x + self.ff(self.o(torch.nn.functional.gelu(self.q(n.to(BF16)))))


def _quantize(model):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    quantize_(model, Int8WeightOnlyConfig(set_inductor_config = False), filter_fn = _is_plain_linear)
    return model


def _is_plain_linear(mod, fqn):
    return type(mod) is torch.nn.Linear and "lora_" not in fqn


def _model():
    torch.manual_seed(0)
    model = inject_adapter_in_model(
        LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q", "o"], init_lora_weights = "gaussian"),
        _Block().to(BF16),
    )
    gen = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
        if "lora_" in name:
            p.data = p.data.float()
            if "lora_B" in name:
                p.data = torch.randn(p.shape, generator = gen) * 0.05
    return _quantize(model)


def _run(
    model,
    x,
    steps = 2,
):
    out = []
    for _ in range(steps):
        xi = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none = True)
        with torch.autocast("cpu", dtype = BF16):
            y = model(xi)
        y.float().square().mean().backward()
        grads = [p.grad.clone() for n, p in sorted(model.named_parameters()) if p.requires_grad]
        out.append((y.detach().clone(), xi.grad.clone(), grads))
    return out


def _assert_equal_runs(a, b):
    for (ya, gxa, ga), (yb, gxb, gb) in zip(a, b):
        assert torch.equal(ya, yb)
        assert torch.equal(gxa, gxb)
        assert all(torch.equal(u, v) for u, v in zip(ga, gb))


def _x():
    return torch.randn(2, 5, 64, generator = torch.Generator().manual_seed(2))


def test_int8_parts_reads_torchao_weight_only_layouts():
    model = _model()
    assert fl._int8_parts(model.ff.weight) is not None
    assert fl._int8_parts(model.q.base_layer.weight) is not None
    assert fl._int8_parts(torch.nn.Linear(4, 4).weight) is None
    assert fl._int8_parts(None) is None


def test_int8_cache_is_bit_identical_to_torchao_forward_and_backward():
    x = _x()
    ref = _model()
    fl.install_fast_lora(ref)  # fast LoRA alone: torchao runs every int8 call (main's int8 path)
    want = _run(ref, x)

    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    got = _run(model, x)
    _assert_equal_runs(want, got)
    st = cache.stats()
    assert st["int8_layers"] == 3 and st["nf4_layers"] == 0
    assert st["cached_layers"] == 3 and st["int8_exact"] is True
    assert cache.hits >= 3


def test_int8_cache_respects_its_budget_and_stays_exact():
    x = _x()
    ref = _model()
    fl.install_fast_lora(ref)
    want = _run(ref, x)

    model = _model()
    one = 96 * 64 * 2  # q (96 x 64) in bf16; o is 64 x 96, ff 64 x 64
    cache = fl.Nf4DequantCache(budget = one)
    fl.install_fast_lora(model, cache, int8 = True)
    got = _run(model, x)
    _assert_equal_runs(want, got)
    assert cache.used <= cache.budget
    assert len(cache.entries) == 1 and cache.full


def test_int8_entry_is_the_cast_torchao_builds_not_q_times_scale():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    w = cache.get(model.ff, BF16)
    q, _scale = fl._int8_parts(model.ff.weight)
    assert torch.equal(w, q.t().to(BF16))
    assert w.stride() == q.t().stride()


def test_a_layout_that_is_not_mirrored_turns_int8_caching_off(monkeypatch):
    x = _x()
    ref = _model()
    fl.install_fast_lora(ref)
    want = _run(ref, x)

    real = fl._int8_cached_linear

    def off_by_one(self, xx, w_t):
        return real(self, xx, w_t) + 1

    monkeypatch.setattr(fl, "_int8_cached_linear", off_by_one)
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    got = _run(model, x)
    _assert_equal_runs(want, got)
    assert cache.int8_exact is False and not cache.entries and cache.used == 0


def test_the_self_check_does_not_draw_from_the_global_rng():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    assert cache.get(model.ff, BF16) is not None and cache.int8_exact is True
    assert torch.equal(before, torch.get_rng_state())


def test_fp32_input_under_bf16_autocast_keeps_torchao():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    x = torch.randn(3, 64)
    with torch.autocast("cpu", dtype = BF16):
        got = model.ff(x)
        want = model.ff.__dict__["_unsloth_orig_forward"](x)
    assert torch.equal(got, want)
    assert not cache.entries


def test_lend_releases_int8_entries_and_return_regrows():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    _run(model, _x(), steps = 1)
    held = cache.used
    assert held > 0
    handler = fl.cache_vram_handler(cache)
    rep = handler("lend")
    assert rep["freed_gib"] == round(held / fl.GIB, 3) and cache.used == 0 and cache.lent
    _run(model, _x(), steps = 1)
    assert not cache.entries  # lent: nothing refills
    handler("return")
    cache.budget = 1 << 30
    _run(model, _x(), steps = 1)
    assert cache.used == held


def test_int8_kill_switch_keeps_main_behaviour(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_DEQUANT_CACHE", "0")
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache, int8 = True)
    assert cache.layers == 0 and cache.int8_layers == 0
    assert "_unsloth_orig_forward" not in model.ff.__dict__


def test_int8_needs_the_flag():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    fl.install_fast_lora(model, cache)
    assert cache.int8_layers == 0 and "_unsloth_orig_forward" not in model.ff.__dict__


def test_uninstall_restores_int8_linears():
    model = _model()
    cache = fl.Nf4DequantCache(budget = 1 << 30)
    n = fl.install_fast_lora(model, cache, int8 = True)
    assert n == 2
    _run(model, _x(), steps = 1)
    assert "_unsloth_int8_scale" in model.ff.__dict__
    assert fl.uninstall_fast_lora(model) == 2 + 3
    assert all("_unsloth_orig_forward" not in m.__dict__ for m in model.modules())
    assert all("_unsloth_int8_scale" not in m.__dict__ for m in model.modules())


def test_dequant_cache_wanted_for_int8(monkeypatch):
    dit = pytest.importorskip("core.training.diffusion_dit_trainer")
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    assert dit._dequant_cache_wanted(False, "cuda", False, int8_base = True)
    assert not dit._dequant_cache_wanted(False, "cuda", False)
    assert not dit._dequant_cache_wanted(False, "cuda", True, int8_base = True)
    assert not dit._dequant_cache_wanted(False, "xpu", False, int8_base = True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_DEQUANT_CACHE", "0")
    assert not dit._dequant_cache_wanted(False, "cuda", False, int8_base = True)
    # The int8 switch leaves nf4 alone; the shared switch turns both off.
    assert dit._dequant_cache_wanted(True, "cuda", False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", "0")
    assert not dit._dequant_cache_wanted(True, "cuda", False)


def test_int8_entries_yield_to_the_checkpointing_reserve():
    """#18416's joint step-1 plan: kept activations take their room first and int8 entries are evicted to fit."""
    x = _x()
    ref = _model()
    fl.install_fast_lora(ref)
    want = _run(ref, x)

    model = _model()
    all_three = (96 * 64 + 64 * 96 + 64 * 64) * 2
    cache = fl.Nf4DequantCache()
    cache.plan(all_three)
    fl.install_fast_lora(model, cache, int8 = True)
    _run(model, x, steps = 1)
    assert cache.used == all_three
    # The checkpointing plan keeps activations worth one 96 x 64 entry (less the cache's usual share).
    evicted = cache.reserve(int(96 * 64 * 2 / fl._BUDGET_FRAC) + 1, from_plan = True)
    assert evicted > 0 and cache.used <= cache.budget < all_three
    got = _run(model, x)
    _assert_equal_runs(want, got)
    assert cache.used <= cache.budget

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for block swap over a frozen fp8 / int8 W8A8 base: the gate, plan sizing from the converted bytes,
where the codes and scales land, and the fp8 backward cache never serving a ring-slot address."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

import core.training.diffusion_block_swap as bs
import core.training.diffusion_frozen_fp8 as f8
import core.training.diffusion_frozen_int8 as i8

GIB = 1 << 30


def _cpu_scaled_mm_ok() -> bool:
    try:
        a = torch.zeros(16, 16).to(torch.float8_e4m3fn)
        one = torch.ones((), dtype = torch.float32)
        torch._scaled_mm(a, a.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_fp8_mm = pytest.mark.skipif(not _cpu_scaled_mm_ok(), reason = "no CPU fp8 GEMM here")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(bs.LOWP_SWAP_ENV, raising = False)
    monkeypatch.delenv(bs._INT8_TRAIN_ENV, raising = False)
    f8.BACKWARD_WEIGHTS.clear()
    f8.BACKWARD_WEIGHTS.hits = f8.BACKWARD_WEIGHTS.misses = 0
    yield
    f8.BACKWARD_WEIGHTS.clear()
    f8.BACKWARD_WEIGHTS.active = False


def test_gate_allows_fp8_and_w8a8_only(monkeypatch):
    assert bs.block_swap_error(4, "fp8", "cuda", True) is None
    # torchao weight-only int8 (the opt-out) is a tensor subclass the swapper cannot stream; W8A8 is the default.
    monkeypatch.setenv(bs._INT8_TRAIN_ENV, "weight-only")
    assert "W8A8" in bs.block_swap_error(4, "int8", "cuda", True)
    monkeypatch.delenv(bs._INT8_TRAIN_ENV)
    assert bs.block_swap_error(4, "int8", "cuda", True) is None
    monkeypatch.setenv(bs._INT8_TRAIN_ENV, "w8a8")
    assert bs.block_swap_error(4, "int8", "cuda", True) is None
    for mode in ("nf4", "mxfp8"):
        assert "cannot stream" in bs.block_swap_error(4, mode, "cuda", True)
    assert "gradient_checkpointing" in bs.block_swap_error(4, "fp8", "cuda", False)
    assert bs.swap_lowp_mode("fp8") == "fp8" and bs.swap_lowp_mode("int8") == "int8_w8a8"
    assert bs.swap_lowp_mode("bf16") is None


def test_kill_switch_restores_bf16_only(monkeypatch):
    monkeypatch.setenv(bs.LOWP_SWAP_ENV, "0")
    monkeypatch.setenv(bs._INT8_TRAIN_ENV, "w8a8")
    for mode in ("fp8", "int8"):
        err = bs.block_swap_error(4, mode, "cuda", True)
        assert err is not None and "'bf16'" in err
        assert bs.swap_lowp_mode(mode) is None
    assert bs.block_swap_error(4, "bf16", "cuda", True) is None


class _Block(nn.Module):
    def __init__(self, d = 64):
        super().__init__()
        self.a = nn.Linear(d, 2 * d, bias = True)
        self.b = nn.Linear(2 * d, d, bias = False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(self.norm(x))))


class _Model(nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(
        self,
        n = 6,
        d = 64,
    ):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return x


def _model(n = 6, seed = 0):
    torch.manual_seed(seed)
    return _Model(n).to(torch.bfloat16).requires_grad_(False)


def _targets(model):
    return {f"blocks.{i}.{n}" for i in range(len(model.blocks)) for n in ("a", "b")}


def _lowp(model, mode = "fp8"):
    convert = f8.convert_frozen_fp8 if mode == "fp8" else i8.convert_frozen_int8
    return {
        "mode": mode,
        "targets": _targets(model),
        "convert": lambda m, names, dev: convert(m, names),
    }


def _install_fake_zoo(monkeypatch, made):
    class FakeSwap:
        def __init__(
            self,
            layers,
            indices,
            prefetch_depth = 2,
            device = None,
            model = None,
        ):
            self.blocks = [layers[i] for i in indices]
            made.append(self)

        def host_bytes(self):
            return 0

        def remove(self):
            pass

    mod = types.ModuleType("unsloth_zoo.block_swap")
    mod.BlockSwap = FakeSwap
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap", mod)


def test_block_bytes_count_the_converted_codes():
    model = _model()
    blk = model.blocks[0]
    bf16 = bs._frozen_bytes(blk)
    ids = frozenset(id(model.get_submodule(n)) for n in _targets(model))
    codes = bs._frozen_bytes(blk, ids)
    lin = 64 * 128 * 2
    norm_bias = (64 + 64 + 128) * 2
    assert bf16 == 2 * lin + norm_bias
    assert codes == lin + norm_bias  # 1 byte a code; the scales stay resident
    f8.convert_frozen_fp8(model, {"blocks.0.a", "blocks.0.b"})
    assert bs._frozen_bytes(model.blocks[0]) == codes
    m2 = _model()
    ids2 = frozenset(id(m2.get_submodule(n)) for n in _targets(m2))
    full = bs._model_bytes(m2)
    conv = bs._model_bytes(m2, ids2)
    assert full - conv == 6 * (2 * 64 * 128 * 2 - 2 * 64 * 128 - (128 + 64) * 4)


def test_auto_plan_sized_from_converted_bytes(monkeypatch):
    made = []
    _install_fake_zoo(monkeypatch, made)
    monkeypatch.setattr(bs, "_place_rest", lambda *a, **k: None)
    monkeypatch.setattr(bs, "activation_reserve_bytes", lambda *a, **k: 0)
    m_bf16, m_lp = _model(12), _model(12)
    total = bs._model_bytes(m_bf16)
    ids = frozenset(id(m_lp.get_submodule(n)) for n in _targets(m_lp))
    # Room for the converted base minus three blocks' codes: bf16 sizing would swap far more.
    free = bs._model_bytes(m_lp, ids) - 3 * bs._frozen_bytes(m_lp.blocks[0], ids)
    a = bs.install_dit_block_swap(
        m_bf16, "cuda:0", "auto", tokens = 1, free_bytes = free, total_bytes = 8 * GIB
    )
    b = bs.install_dit_block_swap(
        m_lp, "cuda:0", "auto", tokens = 1, free_bytes = free, total_bytes = 8 * GIB, lowp = _lowp(m_lp)
    )
    assert total > free
    assert 0 < b["swapped_blocks"] < a["swapped_blocks"]
    assert b["lowp"] == "fp8" and b["lowp_converted"] == 24 and b["lowp_failed"] == 0


@pytest.mark.parametrize("mode", ["fp8", "int8_w8a8"])
def test_swapped_codes_become_host_params_scales_stay_buffers(monkeypatch, mode):
    made = []
    _install_fake_zoo(monkeypatch, made)
    model = _model()
    moved = set()
    real_place = bs._place_rest

    def mark(t, dev):
        moved.add(t.data_ptr())
        return t

    monkeypatch.setattr(bs, "_place_rest", lambda t, sw, dev: real_place(t, sw, dev, move = mark))
    on_card = lambda t: t.data_ptr() in moved  # noqa: E731
    info = bs.install_dit_block_swap(
        model, "cuda:0", 2, tokens = 1, free_bytes = GIB, total_bytes = 8 * GIB, lowp = _lowp(model, mode)
    )
    swapped = set(info["swapped"]["blocks"])
    assert len(swapped) == 2 and len(made) == 1
    code_dtype = torch.float8_e4m3fn if mode == "fp8" else torch.int8
    for i, blk in enumerate(model.blocks):
        for lin in (blk.a, blk.b):
            assert getattr(lin, "_studio_lowp", None) == mode
            assert on_card(lin.weight_scale)  # scales always resident
            if i in swapped:
                assert isinstance(lin.weight, nn.Parameter) and not lin.weight.requires_grad
                assert not on_card(lin.weight) and lin.weight.dtype == code_dtype
                assert "weight" not in lin._buffers and getattr(lin, bs.SWAPPED_WEIGHT_ATTR)
                assert any(p is lin.weight for _, p in blk.named_parameters())
            else:
                assert "weight" in lin._buffers and on_card(lin.weight)
                assert not getattr(lin, bs.SWAPPED_WEIGHT_ATTR, False)
        if i in swapped:
            assert not on_card(blk.norm.weight) and not on_card(blk.a.bias)
        else:
            assert on_card(blk.norm.weight) and on_card(blk.a.bias)
    assert on_card(model.proj.weight)


def test_host_lowp_weight_values_unchanged():
    torch.manual_seed(1)
    lin = nn.Linear(64, 32, bias = True).to(torch.bfloat16).requires_grad_(False)
    ref = f8.FrozenFp8Linear(lin)
    sw = f8.FrozenFp8Linear(lin)
    sw._studio_lowp = "fp8"
    assert bs.host_lowp_weight(sw)
    assert torch.equal(ref.weight.view(torch.uint8), sw.weight.view(torch.uint8))
    assert not bs.host_lowp_weight(nn.Linear(4, 4))


def _ring_pair(seed = 0):
    """Two same-shape frozen fp8 layers (two swapped blocks) and a two-slot ring, as zoo's BlockSwap streams them:
    each weight is a frozen parameter whose .data is a slot view while resident and empty when evicted."""
    torch.manual_seed(seed)
    lins = [
        nn.Linear(64, 64, bias = False).to(torch.bfloat16).requires_grad_(False) for _ in range(2)
    ]
    layers = [f8.FrozenFp8Linear(lin) for lin in lins]
    for ly in layers:
        ly._studio_lowp = "fp8"
        bs.host_lowp_weight(ly)
        ly.__dict__.pop(bs.SWAPPED_WEIGHT_ATTR)  # restored per test
    host = [ly.weight.data.clone() for ly in layers]
    slots = [torch.empty_like(host[0]) for _ in range(2)]
    return layers, host, slots


def _fetch(ly, h, slot):
    slot.copy_(h)
    ly.weight.data = slot


def _grad(ly, x):
    xi = x.clone().requires_grad_(True)
    ly(xi).float().sum().backward()
    return xi.grad


@needs_fp8_mm
@pytest.mark.parametrize("bypass", [False, True])
def test_fp8_cache_never_serves_a_ring_slot_address(bypass):
    if f8._grad_input_mm_uncached is None:
        pytest.skip("no torch.library.custom_op")
    f8.BACKWARD_WEIGHTS.active = True
    f8.BACKWARD_WEIGHTS.plan(GIB)
    torch.manual_seed(3)
    x = torch.randn(48, 64, dtype = torch.bfloat16)
    (a, c), (ha, hc), (s0, s1) = _ring_pair()
    ref = []
    for ly, h in ((a, ha), (c, hc)):
        _fetch(ly, h, torch.empty_like(h))
        ref.append(_grad(ly, x))
    f8.BACKWARD_WEIGHTS.clear(keep_budget = True)
    if bypass:
        for ly in (a, c):
            setattr(ly, bs.SWAPPED_WEIGHT_ATTR, True)
    # Step 1: block A runs from slot 0. Step 2: A is resident in slot 1 while block C reuses slot 0.
    _fetch(a, ha, s0)
    assert torch.equal(_grad(a, x), ref[0])
    _fetch(a, ha, s1)
    _fetch(c, hc, s0)
    got_c = _grad(c, x)
    if bypass:
        assert torch.equal(got_c, ref[1]) and torch.equal(_grad(a, x), ref[0])
        assert len(f8.BACKWARD_WEIGHTS.entries) == 0
    else:
        # The data_ptr key hands C block A's column-major copy: wrong dX, silently.
        assert not torch.equal(got_c, ref[1])


@needs_fp8_mm
def test_swapped_fp8_forward_backward_bit_identical_to_resident():
    f8.BACKWARD_WEIGHTS.active = True
    f8.BACKWARD_WEIGHTS.plan(GIB)
    torch.manual_seed(4)
    lin = nn.Linear(64, 128, bias = True).to(torch.bfloat16).requires_grad_(False)
    res, sw = f8.FrozenFp8Linear(lin), f8.FrozenFp8Linear(lin)
    sw._studio_lowp = "fp8"
    assert bs.host_lowp_weight(sw)
    x = torch.randn(40, 64, dtype = torch.bfloat16)
    outs = []
    for ly in (res, sw):
        xi = x.clone().requires_grad_(True)
        y = ly(xi)
        y.float().square().sum().backward()
        outs.append((y, xi.grad))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_swapped_int8_forward_backward_bit_identical_to_resident():
    try:
        a = torch.ones((32, 32), dtype = torch.int8)
        torch._int_mm(a, a.t())
    except Exception:  # noqa: BLE001
        pytest.skip("no CPU int8 GEMM here")
    torch.manual_seed(5)
    lin = nn.Linear(64, 128, bias = True).to(torch.bfloat16).requires_grad_(False)
    res, sw = i8.FrozenInt8Linear(lin), i8.FrozenInt8Linear(lin)
    res._studio_lowp = sw._studio_lowp = "int8_w8a8"
    assert bs.host_lowp_weight(sw)
    x = torch.randn(40, 64, dtype = torch.bfloat16)
    outs = []
    for ly in (res, sw):
        xi = x.clone().requires_grad_(True)
        y = ly(xi)
        y.float().square().sum().backward()
        outs.append((y, xi.grad))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_trainer_spec_targets_and_refusal(monkeypatch):
    import core.training.diffusion_dit_trainer as dit

    assert dit._lowp_swap_spec(_model(), None, None, "cuda") is None
    monkeypatch.setattr(
        dit, "_probe_lowp", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no fp8"))
    )
    with pytest.raises(ValueError, match = "probe"):
        dit._lowp_swap_spec(_model(), "fp8", None, "cuda")
    monkeypatch.setattr(dit, "_probe_lowp", lambda *a, **k: None)
    spec = dit._lowp_swap_spec(_model(), "fp8", None, "cuda")
    assert spec["mode"] == "fp8" and spec["targets"] == _targets(_model())


def test_auto_plan_resized_when_a_conversion_fails(monkeypatch):
    # A layer whose conversion fails stays bf16; the plan must count its real 2-byte weights, not 1-byte codes.
    made = []
    _install_fake_zoo(monkeypatch, made)
    monkeypatch.setattr(bs, "_place_rest", lambda *a, **k: None)
    monkeypatch.setattr(bs, "activation_reserve_bytes", lambda *a, **k: 0)

    def plan_for(fail):
        m = _model(12)
        ids = frozenset(id(m.get_submodule(n)) for n in _targets(m))
        free = bs._model_bytes(m, ids) - 3 * bs._frozen_bytes(m.blocks[0], ids)

        def convert(model, names, dev):
            names = set(names)
            bad = {n for n in names if fail(n)}
            done, failed = f8.convert_frozen_fp8(model, names - bad)
            return done, failed + len(bad)

        lowp = {"mode": "fp8", "targets": _targets(m), "convert": convert}
        info = bs.install_dit_block_swap(
            m, "cuda:0", "auto", tokens = 1, free_bytes = free, total_bytes = 8 * GIB, lowp = lowp
        )
        return m, info

    _, ok = plan_for(lambda n: False)
    m, bad = plan_for(lambda n: not n.startswith(("blocks.11.", "blocks.7.", "blocks.3.")))
    assert ok["lowp_failed"] == 0 and bad["lowp_failed"] > 0
    assert bad["swapped_blocks"] > ok["swapped_blocks"]
    # The resized plan keeps the first plan's blocks, and every swapped block's converted codes went to host RAM.
    assert set(ok["swapped"]["blocks"]) <= set(bad["swapped"]["blocks"])
    for i in bad["swapped"]["blocks"]:
        for lin in (m.blocks[i].a, m.blocks[i].b):
            if getattr(lin, "_studio_lowp", None) == "fp8":
                assert isinstance(lin.weight, nn.Parameter) and getattr(lin, bs.SWAPPED_WEIGHT_ATTR)

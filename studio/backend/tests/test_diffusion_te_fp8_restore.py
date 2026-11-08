# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layerwise fp8 encoders keep their fp8 storage across a forward; outputs and bytes match diffusers' round trip."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_precision as prec

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.hooks.layerwise_casting")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(prec.TE_FP8_RESTORE_ENV, raising = False)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = torch.nn.Linear(width, 4 * width)
        self.down = torch.nn.Linear(4 * width, width, bias = False)
        self.register_buffer("scale", torch.full((width,), 0.5))

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x))) * self.scale


class _Encoder(torch.nn.Module):
    def __init__(
        self,
        width = 64,
        depth = 3,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(50, width)
        self.blocks = torch.nn.ModuleList(_Block(width) for _ in range(depth))
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


def _encoder(seed = 0):
    torch.manual_seed(seed)
    enc = _Encoder().to(torch.bfloat16).eval()
    prec._cast_fp8(enc, types.SimpleNamespace(dtype = torch.bfloat16))
    return enc


def _fp8_casts(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    seen = []

    class _Count(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types_,
            args = (),
            kwargs = None,
        ):
            kwargs = kwargs or {}
            name = str(func)
            if (name.startswith("aten.to.") or name.startswith("aten._to_copy")) and (
                torch.float8_e4m3fn in args or kwargs.get("dtype") == torch.float8_e4m3fn
            ):
                seen.append(1)
            return func(*args, **kwargs)

    with _Count():
        out = fn()
    return out, len(seen)


def _storage(enc):
    return {
        n: (p.data_ptr(), p.dtype, p.view(torch.uint8).clone()) for n, p in enc.named_parameters()
    }


def test_forward_keeps_the_fp8_tensors_and_matches_the_round_trip(monkeypatch):
    ids = torch.randint(0, 50, (2, 7), generator = torch.Generator().manual_seed(1))
    monkeypatch.setenv(prec.TE_FP8_RESTORE_ENV, "0")
    stock = _encoder()
    assert prec._install_storage_restore(stock) == 0
    with torch.inference_mode():
        ref, ref_casts = _fp8_casts(lambda: [stock(ids), stock(ids)])
    monkeypatch.delenv(prec.TE_FP8_RESTORE_ENV)
    enc = _encoder()
    before = _storage(enc)
    assert any(dtype == torch.float8_e4m3fn for _, dtype, _ in before.values())
    with torch.inference_mode():
        got, casts = _fp8_casts(lambda: [enc(ids), enc(ids)])
    after = _storage(enc)
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert ref_casts > 0 and casts == 0  # no downcast after any forward
    for name, (ptr, dtype, raw) in before.items():
        assert (
            after[name][0] == ptr and after[name][1] == dtype and torch.equal(after[name][2], raw)
        )
    stock_after = _storage(stock)
    for name in before:
        assert torch.equal(stock_after[name][2], after[name][2])


def test_install_is_idempotent_and_counts_each_hook():
    enc = _encoder()
    assert prec._install_storage_restore(enc) == 0  # _cast_fp8 already installed it
    hooked = [
        ref
        for m in enc.modules()
        for ref in getattr(getattr(m, "_diffusers_hook", None), "_fn_refs", None) or ()
        if getattr(ref, "_unsloth_fp8_restore", False)
    ]
    assert len(hooked) == 3 * 2  # up and down per block; Embedding and LayerNorm are skipped


def test_a_slot_replaced_during_the_forward_takes_the_stock_downcast():
    enc = _encoder()
    lin = enc.blocks[0].up
    ref = lin._diffusers_hook._fn_refs[-1]
    below = ref.forward

    def swap(*a, **k):
        out = below(*a, **k)
        lin.weight.data = lin.weight.data.clone()  # something else replaced the compute copy
        return out

    ref.forward = swap
    stored = lin.weight.data_ptr()
    with torch.inference_mode():
        _, casts = _fp8_casts(lambda: enc(torch.randint(0, 50, (1, 5))))
    assert lin.weight.dtype == torch.float8_e4m3fn and lin.weight.data_ptr() != stored
    assert casts > 0


def test_switch_off_at_runtime_uses_the_stock_cast(monkeypatch):
    enc = _encoder()
    ids = torch.randint(0, 50, (1, 5))
    monkeypatch.setenv(prec.TE_FP8_RESTORE_ENV, "0")
    with torch.inference_mode():
        _, casts = _fp8_casts(lambda: enc(ids))
    assert casts > 0
    assert all(p.dtype == torch.float8_e4m3fn for n, p in enc.named_parameters() if "blocks" in n)


@pytest.mark.parametrize("mode", ["inference", "no_grad"])
def test_an_in_place_write_during_the_forward_is_kept(mode):
    enc = _encoder()
    lin = enc.blocks[0].up
    ref = lin._diffusers_hook._fn_refs[-1]
    below = ref.forward

    def write(*a, **k):
        lin.weight.add_(1.0)
        return below(*a, **k)

    ref.forward = write
    before = lin.weight.float().clone()
    ctx = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with ctx:
        enc(torch.randint(0, 50, (1, 5)))
    assert lin.weight.dtype == torch.float8_e4m3fn
    assert float((lin.weight.float() - before).abs().max()) >= 0.5


def _compiled_breaks(enc, xs):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(enc, backend = "eager", dynamic = True)
    with torch.no_grad():
        outs = [compiled(x) for x in xs]
    return outs, sum(torch._dynamo.utils.counters["graph_break"].values())


def test_a_dynamic_compiled_encoder_traces_like_the_stock_hooks(monkeypatch):
    pytest.importorskip("torch._dynamo")
    xs = [
        torch.randint(0, 50, (n, 5), generator = torch.Generator().manual_seed(n))
        for n in (3, 5, 7, 3)
    ]
    monkeypatch.setenv(prec.TE_FP8_RESTORE_ENV, "0")
    stock = _encoder()
    ref, stock_breaks = _compiled_breaks(stock, xs)
    monkeypatch.delenv(prec.TE_FP8_RESTORE_ENV)
    enc = _encoder()
    try:
        got, breaks = _compiled_breaks(enc, xs)
    finally:
        torch._dynamo.reset()
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert breaks == stock_breaks
    assert all(p.dtype == torch.float8_e4m3fn for n, p in enc.named_parameters() if "blocks" in n)

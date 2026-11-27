# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A baked LoRA on an int8 load keeps the shared rotation + act quant paths: ``shared_rotq_linears`` takes PEFT LoRA
layers (base GEMM on the shared quant, then the adapters exactly as PEFT adds them), and the Z-Image QKV / MLP
gates look through the wrapper. CPU: PEFT layers on plain Linears, the GPU GEMM stubbed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from core.inference import diffusion_int8_gemm as gemm  # noqa: E402
from core.inference import diffusion_zimage_fused as zf  # noqa: E402


class _Two(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(16, 24, bias = False)
        self.b = torch.nn.Linear(16, 8, bias = False)


def _wrapped(rank = 4, adapters = ("one",)):
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(0)
    root = _Two()
    for name in adapters:
        cfg = LoraConfig(
            r = rank, lora_alpha = 2 * rank, target_modules = ["a", "b"], init_lora_weights = False
        )
        root = inject_adapter_in_model(cfg, root, adapter_name = name)
    if len(adapters) > 1:
        for m in (root.a, root.b):
            m.set_adapter(list(adapters))
    return root


def test_lora_split_names_the_active_adapters_and_the_base():
    root = _wrapped()
    base, names = gemm.lora_split(root.a)
    assert base is root.a.base_layer and names == ("one",)
    plain = torch.nn.Linear(4, 4)
    assert gemm.lora_split(plain) == (plain, ())


def test_lora_split_is_base_only_while_adapters_are_off():
    root = _wrapped()
    root.a._disable_adapters = True
    assert gemm.lora_split(root.a) == (root.a.base_layer, ())


def test_lora_split_refuses_a_merged_layer():
    root = _wrapped()
    root.a.merge()
    assert gemm.lora_split(root.a) is None


def test_lora_split_refuses_a_hooked_wrapper():
    root = _wrapped()
    root.a.register_forward_hook(lambda *a: None)
    assert gemm.lora_split(root.a) is None


@pytest.mark.parametrize("adapters", [("one",), ("one", "two")])
def test_lora_output_is_the_wrapper_forward_bit_for_bit(adapters):
    root = _wrapped(adapters = adapters)
    x = torch.randn(3, 5, 16)
    for m in (root.a, root.b):
        base, names = gemm.lora_split(m)
        assert names == adapters
        with torch.no_grad():
            want = m(x)
            got = gemm.lora_output(m, names, x, base(x))
        assert torch.equal(want, got)


def _stub_shared(monkeypatch, root):
    """Bases count as swapped rotated int8 Linears whose shared GEMM is their own forward."""
    bases = {id(root.a.base_layer), id(root.b.base_layer)}
    calls = []

    def fake(
        x,
        linears,
        merged = None,
    ):
        if any(id(m) not in bases for m in linears):
            return None
        calls.append(linears)
        return [m(x) for m in linears]

    # the inner (bases) call reaches the stub; the outer one is the code under test
    real = gemm.shared_rotq_linears

    def dispatch(
        x,
        linears,
        merged = None,
    ):
        if all(id(m) in bases for m in linears):
            return fake(x, linears, merged)
        return real(x, linears, merged)

    monkeypatch.setattr(gemm, "shared_rotq_linears", dispatch)
    return calls


def test_shared_rotq_linears_takes_lora_layers(monkeypatch):
    root = _wrapped()
    calls = _stub_shared(monkeypatch, root)
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        got = gemm.shared_rotq_linears(x, (root.a, root.b))
        want = [root.a(x), root.b(x)]
    assert got is not None and calls == [(root.a.base_layer, root.b.base_layer)]
    assert all(torch.equal(g, w) for g, w in zip(got, want))


def test_shared_rotq_linears_declines_an_unsupported_lora_layer(monkeypatch):
    root = _wrapped()
    _stub_shared(monkeypatch, root)
    root.a.merge()
    assert gemm.shared_rotq_linears(torch.randn(2, 16), (root.a, root.b)) is None


def test_zimage_shared_qkv_gate_looks_through_the_lora_layer(monkeypatch):
    root = _wrapped()
    seen = {}
    monkeypatch.setattr(zf, "_swapped_rotated", lambda lin: lin is root.a.base_layer)

    def shared(
        x,
        linears,
        merged = None,
    ):
        seen["linears"] = linears
        return ["q", "k", "v"]

    monkeypatch.setattr(gemm, "shared_rotq_linears", shared)

    class Attn:
        to_q, to_k, to_v = root.a, root.b, root.b

    assert zf._shared_qkv(Attn(), torch.randn(1, 16)) == ["q", "k", "v"]
    assert seen["linears"] == (root.a, root.b, root.b)


def test_swiglu_down_declines_an_active_adapter_and_uses_the_base_when_off(monkeypatch):
    root = _wrapped()
    seen = []
    monkeypatch.setattr(gemm, "_swapped_rec", lambda m: seen.append(m))
    gate = torch.randn(2, 16)
    assert gemm._swiglu_down(root.a, gate, gate) is None
    assert seen == []
    root.a._disable_adapters = True
    assert gemm._swiglu_down(root.a, gate, gate) is None
    assert seen == [root.a.base_layer]

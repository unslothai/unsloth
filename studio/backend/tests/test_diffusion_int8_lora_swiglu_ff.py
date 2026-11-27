# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Qwen-Image-2.1 SwiGLU MLP install (shared gate / value quant, SwiGLU-fused down projection) judges a baked LoRA's
PEFT layers by their swapped base Linears, so a LoRA load keeps it; the forward then adds the adapters through the
LoRA-aware shared quant, and an active adapter on the down projection takes the wrapper (it needs the product).
CPU, the int8 swap records and GPU pieces stubbed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from core.inference import diffusion_int8_gemm as gemm  # noqa: E402


class QwenImage21SwiGLUFeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_layer = torch.nn.Linear(16, 32, bias = False)
        self.proj = torch.nn.Linear(16, 32, bias = False)
        self.out = torch.nn.Linear(32, 16, bias = False)
        self.activation_fn = torch.nn.SiLU()

    def forward(self, hidden_states):
        return self.out(
            self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states)
        )


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_lora = QwenImage21SwiGLUFeedForward()
        self.ff_plain = QwenImage21SwiGLUFeedForward()


def _swapped(lin):
    # what the int8 swap leaves on a rotated Linear: (kind, group, rotq, weight, fused)
    lin.__dict__[gemm._REC] = ("v1", 128, True, lin.weight, False)


def _root():
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(0)
    root = _Root()
    cfg = LoraConfig(
        r = 2,
        lora_alpha = 2,
        target_modules = ["ff_lora.gate_layer", "ff_lora.proj", "ff_lora.out"],
        init_lora_weights = False,
    )
    root = inject_adapter_in_model(cfg, root, adapter_name = "one")
    for ff in (root.ff_lora, root.ff_plain):
        for name in ("gate_layer", "proj", "out"):
            _swapped(gemm.lora_split(getattr(ff, name))[0])
    return root


def test_lora_wrapped_mlp_is_eligible():
    root = _root()
    assert gemm._swiglu_ff_eligible(root.ff_plain)
    assert gemm._swiglu_ff_eligible(root.ff_lora)


def test_install_counts_both_mlps(monkeypatch):
    root = _root()
    monkeypatch.setattr(gemm, "_weight_card", lambda module: 0)
    assert gemm._install_swiglu_ff(root, 0) == 2
    assert "forward" in root.ff_lora.__dict__ and "forward" in root.ff_plain.__dict__


def test_an_unsupported_lora_layer_keeps_the_stock_mlp():
    root = _root()
    root.ff_lora.proj.merge()
    assert not gemm._swiglu_ff_eligible(root.ff_lora)


def test_installed_forward_matches_the_stock_mlp_with_adapters_on_and_off(monkeypatch):
    root = _root()
    monkeypatch.setattr(gemm, "_weight_card", lambda module: 0)
    x = torch.randn(3, 16)
    with torch.no_grad():
        want_on = QwenImage21SwiGLUFeedForward.forward(root.ff_lora, x)
    calls = {"down": []}

    def shared(
        h,
        linears,
        merged = None,
    ):
        splits = [gemm.lora_split(m) for m in linears]
        return [gemm.lora_output(m, s[1], h, s[0](h)) for m, s in zip(linears, splits)]

    def down(lin, gate, value):
        calls["down"].append(lin)
        split = gemm.lora_split(lin)
        return (
            None if split is None or split[1] else split[0](torch.nn.functional.silu(gate) * value)
        )

    monkeypatch.setattr(gemm, "shared_rotq_linears", shared)
    monkeypatch.setattr(gemm, "_swiglu_down", down)
    assert gemm._install_swiglu_ff(root, 0) == 2
    with torch.no_grad():
        got_on = root.ff_lora(x)
        for m in (root.ff_lora.gate_layer, root.ff_lora.proj, root.ff_lora.out):
            m._disable_adapters = True
        want_off = QwenImage21SwiGLUFeedForward.forward(root.ff_lora, x)
        got_off = root.ff_lora(x)
    assert torch.equal(got_on, want_on)
    assert torch.equal(got_off, want_off)
    assert not torch.equal(want_on, want_off)

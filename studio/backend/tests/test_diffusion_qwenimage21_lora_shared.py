# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 with a baked LoRA keeps its shared QKV quant and output-projection act quant: the installs judge a
PEFT LoRA layer by its base Linear, never fuse Q / K / V over LoRA layers (the fused GEMM would skip the adapters),
and patch the output projection's base Linear, not the wrapper. CPU, the GPU pieces stubbed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from core.inference import diffusion_int8_fused as fused_mod  # noqa: E402
from core.inference import diffusion_qwenimage21_fused as qf  # noqa: E402


class QwenImage21Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(16, 16, bias = False)
        self.to_k = torch.nn.Linear(16, 16, bias = False)
        self.to_v = torch.nn.Linear(16, 16, bias = False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(16, 16, bias = False)])


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = QwenImage21Attention()


def _root(lora):
    root = _Root()
    if lora:
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r = 2, lora_alpha = 2, target_modules = ["to_q", "to_k", "to_v", "to_out.0"])
        root = inject_adapter_in_model(cfg, root, adapter_name = "one")
    return root


def _bases(root):
    attn = root.attn
    mods = [attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]]
    return {id(getattr(m, "base_layer", m)) for m in mods}


@pytest.fixture
def stub(monkeypatch):
    """Base Linears count as rotated int8 Linears; the card / probe / patch steps succeed."""
    state = {}

    def rotated(linears):
        return 128 if all(id(m) in state["bases"] for m in linears) else 0

    monkeypatch.setattr(qf, "_rotated_group", rotated)
    monkeypatch.setattr(qf, "_plain", lambda lin: True)
    monkeypatch.setattr(qf, "arch_ok", lambda device = None: True)
    monkeypatch.setattr(qf, "_patch_prepare", lambda logger = None: True)
    monkeypatch.setattr(qf, "q21_convrot_fused_disabled", lambda: False)
    monkeypatch.setattr(fused_mod, "resident_cuda_cards", lambda root: [0])
    monkeypatch.setattr(fused_mod, "arm_rotated", lambda index: True)
    return state


@pytest.mark.parametrize("lora", [False, True], ids = ["plain", "lora"])
def test_rotated_attention_sees_through_lora_layers(stub, lora):
    root = _root(lora)
    stub["bases"] = _bases(root)
    todo = qf._rotated_attention(root)
    assert [m for m, _ in todo] == [root.attn]
    # the parts handed on are the modules the attention calls, wrapper or not
    assert todo[0][1] == [root.attn.to_q, root.attn.to_k, root.attn.to_v]


def test_install_never_fuses_qkv_over_lora_layers_and_patches_the_out_base(stub):
    root = _root(True)
    stub["bases"] = _bases(root)
    result = qf.install_modules(root)
    out = root.attn.to_out[0]
    assert result == {"fused_qkv": 0, "out": 1}
    assert qf._QKV_ATTR not in root.attn.__dict__
    assert qf._OUT_MARK in out.base_layer.__dict__ and "forward" in out.base_layer.__dict__
    assert qf._OUT_MARK not in out.__dict__ and "forward" not in out.__dict__
    qf.uninstall(root)
    assert "forward" not in out.base_layer.__dict__ and qf._OUT_MARK not in out.base_layer.__dict__


def test_install_still_fuses_a_plain_load(stub, monkeypatch):
    root = _root(False)
    stub["bases"] = _bases(root)
    from core.inference import diffusion_zimage_fused as zf

    fused = torch.nn.Linear(16, 48, bias = False)
    monkeypatch.setattr(zf, "_fuse_linears", lambda parts: fused)
    monkeypatch.setattr(zf, "_share_storage", lambda f, parts: True)
    result = qf.install_modules(root)
    assert result == {"fused_qkv": 1, "out": 1}
    assert root.attn.__dict__[qf._QKV_ATTR][0] is fused
    qf.uninstall(root)

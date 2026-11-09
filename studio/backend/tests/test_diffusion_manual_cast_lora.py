# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA on a manual-cast (bf16 stored, fp32 compute) denoiser keeps fp32 adapters and is never fused."""

from __future__ import annotations

import copy
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from core.inference import diffusion_lora_fuse as lora_fuse  # noqa: E402
from core.inference import diffusion_manual_cast as mc  # noqa: E402

LORA_ENV = "UNSLOTH_STUDIO_MANUAL_CAST_LORA"
TARGET = "1"


def test_imports_this_tree():
    here = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(mc.__file__).resolve().is_relative_to(here)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _models():
    torch.manual_seed(0)
    ref = _Net().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.data = p.data.to(torch.bfloat16).float()
    head = copy.deepcopy(ref).to(torch.bfloat16)
    mc.apply_manual_cast_(head, torch.float32)
    assert type(head.layers[1]) is not torch.nn.Linear  # swapped for the manual-cast twin
    return ref, head


def _lora_sd():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(4, 32, generator = g) * 0.1
    b = torch.randn(32, 4, generator = g) * 0.1
    return a, b


def _inject(model, low_cpu_mem_usage):
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    a, b = _lora_sd()
    cfg = LoraConfig(r = 4, lora_alpha = 4, target_modules = [f"layers.{TARGET}"])
    kwargs = {"low_cpu_mem_usage": True} if low_cpu_mem_usage else {}
    inject_adapter_in_model(cfg, model, adapter_name = "px", **kwargs)
    sd = {
        f"layers.{TARGET}.lora_A.weight": a.clone(),
        f"layers.{TARGET}.lora_B.weight": b.clone(),
    }
    set_peft_model_state_dict(model, sd, adapter_name = "px", **kwargs)
    return model.layers[int(TARGET)], a, b


class _MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            self.dtypes.append({a.dtype for a in args if isinstance(a, torch.Tensor)})
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_adapter_keeps_fp32_values_and_gemms(monkeypatch, low_cpu_mem_usage):
    monkeypatch.delenv(LORA_ENV, raising = False)
    ref, head = _models()
    ref_layer, _, _ = _inject(ref, low_cpu_mem_usage)
    layer, a, b = _inject(head, low_cpu_mem_usage)
    assert layer.base_layer.weight.dtype == torch.bfloat16
    assert layer.lora_A["px"].weight.dtype == torch.float32
    assert layer.lora_B["px"].weight.dtype == torch.float32
    assert torch.equal(layer.lora_A["px"].weight.detach(), a)
    assert torch.equal(layer.lora_B["px"].weight.detach(), b)
    x = torch.randn(2, 5, 32, generator = torch.Generator().manual_seed(2))
    cap = _MatmulDtypes()
    with torch.no_grad(), cap:
        out = head(x)
    assert cap.dtypes and all(d == {torch.float32} for d in cap.dtypes), cap.dtypes
    with torch.no_grad():
        want = ref(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, want, atol = 1e-6, rtol = 1e-6)
    assert ref_layer.lora_A["px"].weight.dtype == torch.float32


def test_kill_switch_restores_stored_dtype_adapters(monkeypatch):
    monkeypatch.setenv(LORA_ENV, "0")
    _ref, head = _models()
    layer, _a, _b = _inject(head, False)
    assert layer.lora_A["px"].weight.dtype == torch.bfloat16


def test_plain_bf16_model_unchanged(monkeypatch):
    monkeypatch.delenv(LORA_ENV, raising = False)
    _models()  # installs the PEFT placement hook
    torch.manual_seed(0)
    plain = _Net().to(torch.bfloat16)
    layer, _a, _b = _inject(plain, False)
    assert layer.lora_A["px"].weight.dtype == torch.bfloat16


def test_fuse_declines_manual_cast_layers(monkeypatch):
    monkeypatch.delenv(LORA_ENV, raising = False)
    _ref, head = _models()
    _inject(head, False)
    assert lora_fuse._plan([head]) is None
    monkeypatch.setenv(LORA_ENV, "0")
    assert lora_fuse._plan([head]) is not None

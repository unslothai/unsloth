# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The PEFT-side fused path for a frozen int8 base (kernels/fast_lora.frozen_base_lora_forward): what
unsloth_zoo's compiled PEFT forward calls on FastModel instead of base_layer(x) + LoRA. CUDA only."""

import pytest
import torch

from real_accelerator import has_real_cuda

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not has_real_cuda()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability() < (8, 0),
        reason = "needs an NVIDIA GPU with int8 tensor cores",
    ),
]

IL = pytest.importorskip("unsloth.kernels.int8_lora")
if IL.triton is None:
    pytest.skip("needs Triton", allow_module_level = True)
peft = pytest.importorskip("peft")

from unsloth.kernels import fast_lora as FL  # noqa: E402
from unsloth.models import float8_training as f8t  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig()
        self.proj = torch.nn.Linear(2048, 1024, bias = False)

    def forward(self, x):
        return self.proj(x)


def _lora_layer(monkeypatch):
    monkeypatch.setattr(f8t, "_prepare_fp8_lora_loss", lambda m: None)
    monkeypatch.setattr(f8t, "_prepare_fp8_lora_operators", lambda m: None)
    torch.manual_seed(0)
    model = _Tiny().to("cuda", torch.bfloat16).requires_grad_(False)
    model = peft.get_peft_model(
        model, peft.LoraConfig(r = 16, lora_alpha = 32, target_modules = ["proj"])
    )
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.02)
    assert f8t.enable_fp8_lora(model, f8t.INT8_LORA_RECIPE)
    assert "_unsloth_fused_lora_forward" in layer.base_layer.__dict__
    return layer


def _unfused(layer, x):
    A = layer.lora_A["default"].weight.to(x.dtype)
    B = layer.lora_B["default"].weight.to(x.dtype)
    return layer.base_layer(x) + layer.scaling["default"] * ((x @ A.t()) @ B.t())


def test_fused_matches_unfused_and_uses_the_epilogue(monkeypatch):
    layer = _lora_layer(monkeypatch)
    calls = []
    original = IL.Int8LoRABaseState.lora_first
    monkeypatch.setattr(
        IL.Int8LoRABaseState,
        "lora_first",
        lambda self, *a, **k: calls.append(1) or original(self, *a, **k),
    )
    x = torch.randn((2, 2048, 2048), device = "cuda", dtype = torch.bfloat16)
    grads = []
    for fused in (False, True):
        xi = x.clone().requires_grad_(True)
        for p in (layer.lora_A["default"].weight, layer.lora_B["default"].weight):
            p.grad = None
        FL.FROZEN_BASE_FUSED_STATS["fused"] = 0
        y = FL.frozen_base_lora_forward(layer, xi, (), {}) if fused else _unfused(layer, xi)
        assert y is not None and y.shape == (2, 2048, 1024)
        y.float().square().mean().backward()
        grads.append(
            (
                y.detach().float(),
                xi.grad.float(),
                layer.lora_A["default"].weight.grad.float(),
                layer.lora_B["default"].weight.grad.float(),
            )
        )
        assert FL.FROZEN_BASE_FUSED_STATS["fused"] == int(fused)
    assert len(calls) == 1  # the fused arm ran the int8 GEMM with the LoRA term in its epilogue
    for a, b in zip(*grads):
        assert float((a - b).norm() / b.norm()) < 2e-2


def test_falls_back_to_peft(monkeypatch):
    layer = _lora_layer(monkeypatch)
    x = torch.randn((1, 2048, 2048), device = "cuda", dtype = torch.bfloat16)
    assert FL.frozen_base_lora_forward(layer, x[0], (), {}).shape == (2048, 1024)  # 2-D input
    assert FL.frozen_base_lora_forward(layer, x, (), {"adapter_names": ["default"]}) is None
    assert FL.frozen_base_lora_forward(layer, x.float(), (), {}) is None
    monkeypatch.setattr(FL, "_FROZEN_BASE_FUSED", False)
    assert FL.frozen_base_lora_forward(layer, x, (), {}) is None
    monkeypatch.setattr(FL, "_FROZEN_BASE_FUSED", True)
    del layer.base_layer.__dict__["_unsloth_fp8_lora"]  # restored for inference
    assert FL.frozen_base_lora_forward(layer, x, (), {}) is None

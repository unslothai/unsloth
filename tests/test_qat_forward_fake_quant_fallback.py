# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused LoRA must not bypass torchao QAT linears that fake-quantize inside forward().

torchao's QATConfig(step="prepare") swaps nn.Linear for NVFP4FakeQuantizedLinear / MXFakeQuantizedLinear
when the base config is NVFP4 or MX. Those classes fake-quantize in forward() and expose no
weight_fake_quantizer / activation_fake_quantizer, so the fused kernels, which read base_layer.weight
directly, trained on the unquantized weights. They must take the module-forward fallback instead,
while ordinary LoRA and the int4 weight-only scheme keep the fused path.
"""

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
import unsloth.kernels.fast_lora as fast_lora
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat

DIM = 64


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = torch.nn.Linear(DIM, DIM, bias = False)

    def forward(self, x):
        return self.o_proj(x)


def _base_config(scheme):
    if scheme == "nvfp4":
        mx = pytest.importorskip("torchao.prototype.mx_formats")
        return mx.NVFP4DynamicActivationNVFP4WeightConfig()
    if scheme == "mx":
        mx = pytest.importorskip("torchao.prototype.mx_formats")
        return mx.MXDynamicActivationMXWeightConfig(
            activation_dtype = torch.float8_e4m3fn, weight_dtype = torch.float8_e4m3fn
        )
    if scheme == "int4":
        from torchao.quantization import Int4WeightOnlyConfig
        return Int4WeightOnlyConfig(group_size = DIM)
    return None


def _lora_block(scheme):
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(), peft.LoraConfig(r = 16, lora_alpha = 16, target_modules = ["o_proj"], lora_dropout = 0.0)
    )
    base_config = _base_config(scheme)
    if base_config is not None:
        only_base = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith("base_layer")
        _prepare_model_for_qat(
            model,
            TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(base_config, only_base)]),
        )
    return model.base_model.model


def _route(block, monkeypatch):
    """Run apply_lora_o and report which path it took."""
    calls = {"fused": 0, "module": 0}

    def fused(function, *args):
        calls["fused"] += 1
        return args[0]

    def module_forward(x):
        calls["module"] += 1
        return x

    monkeypatch.setattr(fast_lora, "_apply", fused)
    monkeypatch.setattr(block.o_proj, "forward", module_forward)
    fast_lora.apply_lora_o(block, torch.randn(2, 3, DIM))
    return calls


@pytest.mark.parametrize("scheme", ["nvfp4", "mx"])
def test_forward_fake_quant_base_layer_falls_back_to_module_forward(scheme, monkeypatch):
    block = _lora_block(scheme)
    expected = {"nvfp4": "NVFP4FakeQuantizedLinear", "mx": "MXFakeQuantizedLinear"}[scheme]
    base_layer = block.o_proj.base_layer
    assert type(base_layer).__name__ == expected
    # The premise: no fake quantizer children, so the existing child check cannot see it.
    assert getattr(base_layer, "weight_fake_quantizer", None) is None
    assert base_layer._modules.get("activation_fake_quantizer") is None

    assert fast_lora._has_activation_fake_quantizer(block.o_proj)
    assert _route(block, monkeypatch) == {"fused": 0, "module": 1}


@pytest.mark.parametrize("scheme", ["none", "int4"])
def test_plain_and_weight_only_qat_keep_fused_path(scheme, monkeypatch):
    block = _lora_block(scheme)
    assert not fast_lora._has_activation_fake_quantizer(block.o_proj)
    assert _route(block, monkeypatch) == {"fused": 1, "module": 0}


def test_direct_torchao_quantize_also_falls_back(monkeypatch):
    # A user calling torchao's quantize_ directly (not through qat_scheme) gets the same fallback.
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    mx = pytest.importorskip("torchao.prototype.mx_formats")
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(), peft.LoraConfig(r = 16, lora_alpha = 16, target_modules = ["o_proj"], lora_dropout = 0.0)
    )
    quantize_(
        model,
        QATConfig(mx.NVFP4DynamicActivationNVFP4WeightConfig(), step = "prepare"),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith("base_layer"),
    )
    block = model.base_model.model
    assert fast_lora._has_activation_fake_quantizer(block.o_proj)
    assert _route(block, monkeypatch) == {"fused": 0, "module": 1}

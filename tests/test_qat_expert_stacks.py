# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched expert-stack fake quantization equals the per-expert path bit for bit (CPU)."""

import importlib.util
import pathlib

import pytest
import torch

torchao = pytest.importorskip("torchao")

_SPEC = importlib.util.spec_from_file_location(
    "qat_expert_stacks",
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "qat_expert_stacks.py",
)
qes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qes)


def _base_configs():
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
    )
    from torchao.quantization.granularity import PerAxis, PerGroup, PerRow

    out = {
        "int4": Int4WeightOnlyConfig(group_size = 128),
        "fp8-int4": Float8DynamicActivationInt4WeightConfig(),
        "fp8-fp8": Float8DynamicActivationFloat8WeightConfig(granularity = PerRow()),
        "int8-int4": Int8DynamicActivationIntxWeightConfig(
            weight_dtype = torch.int4, weight_granularity = PerGroup(32)
        ),
        "int8": IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)),
    }
    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        out["nvfp4"] = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False)
    except Exception:
        pass
    try:
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        out["mxfp8"] = MXDynamicActivationMXWeightConfig()
    except Exception:
        pass
    try:
        from torchao.prototype.qat import MXFakeQuantizeConfig
        out["mxfp4-weight"] = MXFakeQuantizeConfig(dtype = torch.float4_e2m1fn_x2)
    except Exception:
        pass
    return out


CONFIGS = _base_configs()


def _quantizer(name):
    cfg = CONFIGS[name]
    if type(cfg).__name__ == "MXFakeQuantizeConfig":
        return cfg
    return qes.expert_stack_weight_fake_quantizer(cfg)


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("transposed", [False, True])
def test_batched_equals_per_expert(name, dtype, transposed):
    q = _quantizer(name)
    if name == "fp8-int4" and dtype == torch.float32:
        pytest.skip("fp8-int4 targets bf16 weights")
    torch.manual_seed(0)
    E, N, K = 4, 32, 128
    w = torch.randn(E, N, K) * torch.logspace(-3, 1, E).view(E, 1, 1)  # per-expert scale spread
    w[3, 5, :40] = 0  # an all-zero group
    w = w.to(dtype)
    if transposed:
        w = w.transpose(1, 2).contiguous()  # gpt-oss (E, K, N) storage
    w.requires_grad_(True)
    batched = qes.fake_quantize_expert_stack(w, q, transposed)
    looped = qes.fake_quantize_expert_stack_per_expert(w, q, transposed)
    assert batched.shape == w.shape and batched.dtype == w.dtype
    assert torch.equal(batched, looped), (batched - looped).abs().max()
    assert not torch.equal(batched, w)  # it quantized something
    g = torch.randn_like(w)
    (ga,) = torch.autograd.grad(batched, w, g)
    (gb,) = torch.autograd.grad(looped, w, g)
    if transposed:
        # torchao's scale gradient (fp8 / intx amax paths) accumulates in a different order on the transposed
        # view: ulp-level only. The forward above is bit-identical.
        torch.testing.assert_close(ga, gb, rtol = 1e-5, atol = 1e-5)
    else:
        assert torch.equal(ga, gb)


def test_one_call_per_stack():
    q = _quantizer("int4")
    calls = []
    h = q.register_forward_hook(lambda *a: calls.append(1))
    w = torch.randn(16, 64, 128, dtype = torch.bfloat16)
    qes.fake_quantize_expert_stack(w, q)
    assert len(calls) == 1
    qes.fake_quantize_expert_stack_per_expert(w, q)
    assert len(calls) == 1 + 16
    h.remove()


def test_per_tensor_refused():
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.qat import Float8FakeQuantizeConfig
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    q = FakeQuantizerBase.from_config(Float8FakeQuantizeConfig(granularity = PerTensor()))
    with pytest.raises(ValueError):
        qes.fake_quantize_expert_stack(torch.randn(2, 4, 32), q)

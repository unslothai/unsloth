# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA + a weight-only QAT scheme rounds the frozen base onto the fake quantization grid once, in place
(unsloth/models/_qat_round.py).

The stored weight is the quantizer's own output and the quantizer is switched off, so outputs and LoRA
gradients must be bitwise identical to UNSLOTH_QAT_ROUND_INPLACE=0 (which fake quantizes per call), the
weight must keep its storage (vLLM shares it), and activation schemes, trainable and tied weights must be left
alone.
"""

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_round as qr
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat
from unsloth.precision import engaged

DIM = 128


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(DIM, 2 * DIM, bias = False)
        self.down_proj = torch.nn.Linear(2 * DIM, DIM, bias = False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


def _model(
    scheme,
    monkeypatch,
    mode = "1",
    dtype = torch.float32,
):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", mode)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block().to(dtype),
        peft.LoraConfig(
            r = 16, lora_alpha = 16, target_modules = ["up_proj", "down_proj"], lora_dropout = 0.0
        ),
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    if scheme == "int8-int4":
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig
        from torchao.quantization.granularity import PerGroup

        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype = torch.int4, weight_granularity = PerGroup(32)
        )
        only_base = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith("base_layer")
        scheme = TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(config, only_base)])
    return _prepare_model_for_qat(model, scheme)


def _run(model, x):
    model.zero_grad(set_to_none = True)
    out = model(x)
    out.float().square().mean().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


def _equal(a, b):
    return (
        torch.equal(a[0], b[0])
        and a[1].keys() == b[1].keys()
        and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    )


def _bases(model):
    return [m for n, m in model.named_modules() if n.endswith("base_layer")]


def _quantizers_on(model):
    return sum(1 for m in _bases(model) if m.weight_fake_quantizer.enabled)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scheme", ["int4", "int8", "cactus"])
def test_bitwise_and_engaged(scheme, dtype, monkeypatch):
    x = torch.randn(4, DIM, dtype = dtype, requires_grad = True)
    off = _model(scheme, monkeypatch, mode = "0", dtype = dtype)
    assert _quantizers_on(off) == 2  # kill switch: per-call fake quant as before
    ref = _run(off, x)
    assert _equal(ref, _run(off, x))  # A/A

    on = _model(scheme, monkeypatch, mode = "1", dtype = dtype)
    assert _quantizers_on(on) == 0  # engaged: both base layers rounded, quantizers off
    assert engaged(on)["qat_round_inplace"]["modules"] == 2
    for _ in range(3):
        assert _equal(ref, _run(on, x))
    # The stored weight is exactly what the per-call quantizer produced.
    for a, b in zip(_bases(off), _bases(on)):
        with torch.no_grad():
            assert torch.equal(a.weight_fake_quantizer(a.weight), b.weight)


def test_storage_kept_and_idempotent(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", "1")
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(), peft.LoraConfig(r = 8, target_modules = ["up_proj", "down_proj"], lora_dropout = 0.0)
    )
    ptrs = [m.weight.data_ptr() for n, m in model.named_modules() if n.endswith("base_layer")]
    model = _prepare_model_for_qat(model, "int4")
    assert [
        m.weight.data_ptr() for m in _bases(model)
    ] == ptrs  # same storage: a vLLM view sees the rounding
    snapshot = [m.weight.clone() for m in _bases(model)]
    assert qr.round_frozen_qat_weights(model) == 0  # already rounded: not rounded twice
    assert all(torch.equal(a, m.weight) for a, m in zip(snapshot, _bases(model)))


def test_activation_scheme_untouched(monkeypatch):
    model = _model("int8-int4", monkeypatch, mode = "1")
    assert _quantizers_on(model) == 2
    assert "qat_round_inplace" not in engaged(model)


def test_trainable_tied_and_plain_weights_skipped(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", "1")
    torch.manual_seed(0)

    class _Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(64, DIM)
            self.head = torch.nn.Linear(DIM, 64, bias = False)
            self.head.weight = (
                self.embed.weight
            )  # tied: rounding the head would round the embedding
            self.proj = torch.nn.Linear(DIM, DIM, bias = False)
            self.out = torch.nn.Linear(DIM, DIM, bias = False)  # not LoRA wrapped

    model = peft.get_peft_model(
        _Tied(), peft.LoraConfig(r = 8, target_modules = ["head", "proj"], lora_dropout = 0.0)
    )
    inner = model.base_model.model
    inner.proj.base_layer.weight.requires_grad_(True)  # a trainable base weight
    before = inner.embed.weight.clone()
    model = _prepare_model_for_qat(model, "int4")
    assert inner.head.base_layer.weight_fake_quantizer.enabled  # tied
    assert inner.proj.base_layer.weight_fake_quantizer.enabled  # trainable
    assert inner.out.weight_fake_quantizer.enabled  # plain Linear, converted from memory on export
    assert torch.equal(inner.embed.weight, before)
    assert "qat_round_inplace" not in engaged(model)

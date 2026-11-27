# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cached decode must not bypass torchao QAT fake quantization.

fast_linear_forward and the batch-1 lm_head torch.mv read raw weights, so a QAT linear (weight-only,
activation + weight, or adapter-only) must run its own forward, while plain layers keep the fast path.
"""

import types

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.kernels.utils import fast_linear_forward
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat
from unsloth.models.llama import _decode_logits

DIM = 64


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(DIM, DIM, bias = False)


def _qat_config(scheme):
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis, PerGroup

    if scheme == "int8":
        base = IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    else:
        base = Int8DynamicActivationIntxWeightConfig(
            weight_dtype = torch.int4, weight_granularity = PerGroup(32)
        )
    linear = lambda m, fqn: isinstance(m, torch.nn.Linear) and m.in_features % 32 == 0
    return TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(base, linear)])


def _lora_proj(scheme, r = 16):
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(), peft.LoraConfig(r = r, lora_alpha = 16, target_modules = ["q_proj"], lora_dropout = 0.0)
    )
    proj = model.base_model.model.q_proj
    torch.nn.init.normal_(proj.lora_B["default"].weight, std = 0.05)
    if scheme is not None:
        _prepare_model_for_qat(model, _qat_config(scheme))
    return proj


@pytest.mark.parametrize("scheme", ["int8", "int8-int4"])
@pytest.mark.parametrize("bsz,q_len", [(1, 1), (2, 1), (1, 3)])
@pytest.mark.parametrize("use_out", [False, True])
def test_qat_proj_matches_module_forward(scheme, bsz, q_len, use_out):
    proj = _lora_proj(scheme)
    assert type(proj.base_layer) is not torch.nn.Linear
    X = torch.randn(bsz, q_len, DIM)
    out = torch.full((bsz, q_len, DIM), float("nan")) if use_out else None
    with torch.no_grad():
        expected = proj(X)
        got = fast_linear_forward(proj, X, out = out)
    assert torch.equal(got, expected)
    if use_out:
        assert got is out


def test_qat_on_lora_b_only_falls_back():
    proj = _lora_proj(None, r = 32)
    from torchao.quantization import IntxWeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerAxis
    from torchao.quantization.qat import QATConfig

    quantize_(
        proj.lora_B,
        QATConfig(
            IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)), step = "prepare"
        ),
    )
    assert type(proj.base_layer) is torch.nn.Linear
    X = torch.randn(1, 1, DIM)
    with torch.no_grad():
        assert torch.equal(fast_linear_forward(proj, X), proj(X))


def test_plain_lora_keeps_fast_path(monkeypatch):
    proj = _lora_proj(None)
    assert type(proj.base_layer) is torch.nn.Linear
    X = torch.randn(1, 1, DIM)
    with torch.no_grad():
        expected = proj(X)

        def no_forward(*args, **kwargs):
            raise AssertionError("plain LoRA must not take the module forward")

        monkeypatch.setattr(proj, "forward", no_forward)
        torch.testing.assert_close(fast_linear_forward(proj, X), expected)


def _head_self(lm_head):
    config = types.SimpleNamespace(torch_dtype = torch.float32, dtype = torch.float32)
    return types.SimpleNamespace(lm_head = lm_head, config = config)


@pytest.mark.parametrize("bsz", [1, 2])
def test_qat_lm_head_decode_logits(bsz):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(DIM, 128, bias = False))
    _prepare_model_for_qat(model, _qat_config("int8"))
    lm_head = model[0]
    hidden = torch.randn(bsz, 1, DIM)
    with torch.no_grad():
        assert torch.equal(_decode_logits(_head_self(lm_head), hidden), lm_head(hidden))
        one = _decode_logits(_head_self(lm_head), hidden[:1])
        two = _decode_logits(_head_self(lm_head), hidden[:1].repeat(2, 1, 1))
        torch.testing.assert_close(one[0], two[0], rtol = 1e-5, atol = 1e-5)


def test_plain_lm_head_keeps_mv(monkeypatch):
    lm_head = torch.nn.Linear(DIM, 128, bias = False)
    calls = []
    mv = torch.mv
    monkeypatch.setattr(torch, "mv", lambda *a, **k: calls.append(1) or mv(*a, **k))
    with torch.no_grad():
        _decode_logits(_head_self(lm_head), torch.randn(1, 1, DIM))
    assert calls == [1]


@pytest.mark.parametrize("scheme", [None, "int8"])
def test_fast_linear_forward_compiles_without_graph_breaks(scheme):
    from torch._dynamo.utils import counters

    proj = _lora_proj(scheme)
    X = torch.randn(1, 1, DIM)
    torch._dynamo.reset()
    counters.clear()
    with torch.no_grad():
        expected = fast_linear_forward(proj, X)
        got = torch.compile(
            lambda x: fast_linear_forward(proj, x), fullgraph = True, backend = "eager"
        )(X)
    assert sum(counters["graph_break"].values()) == 0
    torch.testing.assert_close(got, expected)

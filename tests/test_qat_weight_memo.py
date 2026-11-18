# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA + QAT fake quantizes the frozen base weight once (unsloth/models/_qat_weight_memo.py).

The cached tensor is the quantizer's own output, so outputs and LoRA gradients must be bitwise identical
to UNSLOTH_QAT_WEIGHT_MEMO=0; the cache must refresh on an in-place weight change and PEFT merge / unmerge,
never hold a trainable weight, never cache in inference mode, and leave a torch.compile trace unchanged.
"""

import copy
import pickle

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_weight_memo as memo
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat

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
    mode = "auto",
):
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", mode)
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(),
        peft.LoraConfig(
            r = 16, lora_alpha = 16, target_modules = ["up_proj", "down_proj"], lora_dropout = 0.0
        ),
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    if (
        scheme == "int8-int4"
    ):  # the built-in recipe also unties embeddings, which this block has none of
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


def _hits():
    return memo.qat_weight_memo_stats()["hits"]


@pytest.mark.parametrize("scheme", ["int4", "int8", "int8-int4", "cactus"])
def test_bitwise_and_engaged(scheme, monkeypatch):
    model = _model(scheme, monkeypatch)
    assert memo.install_qat_weight_memo(model) == 2  # idempotent: the two base layers, no adapters
    x = torch.randn(4, DIM, requires_grad = True)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    ref = _run(model, x)
    h0 = _hits()
    assert _equal(ref, _run(model, x)) and _hits() == h0  # kill switch: A/A bitwise, no cache hit
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "auto")
    for _ in range(3):
        assert _equal(ref, _run(model, x))
    assert _hits() - h0 >= 2  # first call remembers, second caches, third serves both layers


def test_refresh_on_weight_change_and_merge(monkeypatch):
    model = _model("int8", monkeypatch)
    x = torch.randn(4, DIM)
    for _ in range(3):
        _run(model, x)
    base = model.base_model.model.up_proj.base_layer
    with torch.no_grad():
        base.weight.mul_(1.5)  # bumps the version counter

    def off_on():
        monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
        with torch.no_grad():
            want = model(x)
        monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "auto")
        with torch.no_grad():
            got = [model(x) for _ in range(3)]
        return all(torch.equal(want, g) for g in got)

    assert off_on()
    model.merge_adapter()  # writes base weights through .data: no version bump, cache dropped by the patch
    assert off_on()
    model.unmerge_adapter()
    assert off_on()
    base.weight.data = base.weight.data.to(torch.float32).clone()  # new storage, same object
    assert off_on()


def test_skips_trainable_and_inference(monkeypatch):
    model = _model("int8", monkeypatch)
    x = torch.randn(4, DIM)
    base = model.base_model.model.up_proj.base_layer
    state = base.weight_fake_quantizer.__dict__["_unsloth_qat_weight_memo"]
    base.weight.requires_grad_(True)
    for _ in range(3):
        _run(model, x)
    assert state.out is None
    base.weight.requires_grad_(False)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
    assert state.out is None
    for _ in range(2):
        _run(model, x)
    assert state.out is not None and not state.out.is_inference()


def test_disabled_at_prepare(monkeypatch):
    model = _model("int8", monkeypatch, mode = "0")
    assert not any("_unsloth_qat_weight_memo" in vars(m) for m in model.modules())


def test_compile_trace_unchanged(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    model = _model("int8-int4", monkeypatch)
    x = torch.randn(4, DIM)
    for _ in range(2):
        _run(model, x)
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(model, backend = counter, fullgraph = True)
    with torch.no_grad():
        for _ in range(2):
            out = compiled(x)
        monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
        want = model(x)
    assert counter.frame_count == 1
    assert torch.equal(out, want)


def test_copy_and_pickle(monkeypatch):
    model = _model("int8", monkeypatch)
    x = torch.randn(4, DIM)
    for _ in range(3):
        _run(model, x)
    clone = copy.deepcopy(model)  # (the model itself does not pickle: torchao_config holds lambdas)
    state = clone.base_model.model.up_proj.base_layer.weight_fake_quantizer.__dict__[
        "_unsloth_qat_weight_memo"
    ]
    assert state.out is None  # caches never travel with a copy
    with torch.no_grad():
        assert torch.equal(clone(x), model(x))
    quantizer = pickle.loads(
        pickle.dumps(model.base_model.model.up_proj.base_layer.weight_fake_quantizer)
    )
    assert quantizer.__dict__["_unsloth_qat_weight_memo"].out is None
    assert not any("memo" in k for k in model.state_dict())

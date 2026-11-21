# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Key and guard correctness of the QAT frozen weight cache (unsloth/models/_qat_weight_memo.py).

- A weight built under torch.inference_mode (an inference tensor, e.g. shared with vLLM) has no version
  counter: the cache must skip it, not raise.
- A static scale quantizer reads its own scale / zero point: changing them must change the output.
- A cached CUDA tensor carries no event: another stream must not be served it before it is written.
"""

import os

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_weight_memo as memo
from unsloth.models._utils import _prepare_model_for_qat

DIM = 128


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(memo.__file__).startswith(here), memo.__file__


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(DIM, DIM, bias = False)

    def forward(self, x):
        return self.proj(x)


@pytest.fixture(autouse = True)
def _fresh_state(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", "0")
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "auto")
    state = memo._STATE
    saved = dict(state)
    state.update(armed = False, disabled = False, tokens = 0, rows = 0, row_probes = 0, steps = 0)
    memo._DEVICE.clear()
    yield
    state.update(armed = saved.get("armed", False), disabled = False)


def _arm():
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    torch.optim.SGD([p], lr = 0.0).step()


def _model(scheme = "int8"):
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _Block(), peft.LoraConfig(r = 8, lora_alpha = 8, target_modules = ["proj"])
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    return _prepare_model_for_qat(model, scheme)


def _base(model):
    return model.base_model.model.proj.base_layer


def test_inference_built_weight_does_not_raise(monkeypatch):
    x = torch.randn(4, DIM)
    with torch.inference_mode():
        model = _model()
    assert _base(model).weight.is_inference() and model.training
    _arm()
    with torch.inference_mode():
        outs = [model(x) for _ in range(3)]
        monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
        ref = model(x)
    assert all(torch.equal(o, ref) for o in outs)
    assert _base(model).weight_fake_quantizer.__dict__["_unsloth_qat_weight_memo"].out is None


def test_static_scale_change_is_not_served_stale(monkeypatch):
    model = _model()
    quantizer = _base(model).weight_fake_quantizer
    quantizer.config.is_dynamic = False
    _arm()
    x = torch.randn(4, DIM)
    with torch.no_grad():
        for _ in range(3):
            before = model(x)
        quantizer.scale.mul_(2)
        after = model(x)
        monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
        fresh = model(x)
    assert not torch.equal(before, fresh)
    assert torch.equal(after, fresh)


def test_dynamic_scheme_still_cached():
    model = _model()
    _arm()
    x = torch.randn(4, DIM)
    hits = memo.qat_weight_memo_stats()["hits"]
    with torch.no_grad():
        for _ in range(3):
            model(x)
    assert memo.qat_weight_memo_stats()["hits"] > hits


@pytest.mark.skipif(
    not has_real_cuda() or not torch.cuda.is_available(), reason = "needs CUDA streams"
)
def test_cached_weight_not_served_to_another_stream(monkeypatch):
    model = _model().cuda()
    base = _base(model)
    quantizer, w = base.weight_fake_quantizer, base.weight
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    ref = quantizer(w).clone()
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "auto")
    quantizer(w)  # remembered, so the optimizer step budgets this device
    _arm()
    hits = memo.qat_weight_memo_stats()["hits"]
    for _ in range(3):  # default stream: store, then hit
        assert torch.equal(quantizer(w), ref)
    assert memo.qat_weight_memo_stats()["hits"] > hits
    memo.clear_qat_weight_memo(model)
    quantizer(w)  # remembered on the default stream
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(300_000_000)
        produced = quantizer(w)
    with torch.cuda.stream(consumer):
        consumed = quantizer(w).clone()
    torch.cuda.synchronize()
    assert torch.equal(produced, ref)
    assert torch.equal(consumed, ref), int((consumed != ref).sum())

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The NF4 dequant cache's Trainer scope must cost a model without NF4 weights nothing: no forward pre-hook,
so no host read of the attention mask on every forward. NF4 models keep the hook. CPU only."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from unsloth.kernels import dequant_cache as dc

transformers_trainer = pytest.importorskip("transformers.trainer")
bnb_functional = pytest.importorskip("bitsandbytes.functional")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
    ):
        return self.proj(torch.ones(input_ids.shape + (8,)))


class _HostReads(TorchDispatchMode):
    count = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.is_nonzero.default):
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture
def callback(monkeypatch):
    print("dequant_cache imported from", dc.__file__)
    monkeypatch.setattr(dc, "_MODE", "auto")
    monkeypatch.setattr(dc, "_DISABLED", False)
    monkeypatch.setattr(dc, "_ACTIVE", 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert dc.install_trainer_scope()
    scope = [
        c
        for c in transformers_trainer.DEFAULT_CALLBACKS
        if c.__name__ == "UnslothDequantCacheScope"
    ]
    handler = transformers_trainer.CallbackHandler([], None, None, None, None)
    handler.add_callback(scope[0])
    yield handler.callbacks[-1]
    dc._end_train()
    dc._PRE_RAN[0] = 0
    dc._SHAPE_NEW[:] = [0, 0]
    dc._PADDED_NEW[0] = 0


def _forward_host_reads(model):
    ids = torch.zeros(2, 4, dtype = torch.long)
    mask = torch.ones(2, 4, dtype = torch.long)
    with _HostReads() as reads:
        model(input_ids = ids, attention_mask = mask)
    return reads.count


def test_dense_model_gets_no_pre_hook_and_no_host_read(callback):
    model = _Model()
    callback.on_train_begin(None, None, SimpleNamespace(), model = model)
    assert _forward_host_reads(model) == 0
    assert dc._HOOKS == [] and dc._ACTIVE == 0


def _nf4_state():
    return bnb_functional.quantize_4bit(torch.randn(8, 8), quant_type = "nf4")


@pytest.mark.parametrize("where", ["parameter", "linear4bit_copy"])
def test_nf4_model_keeps_the_pre_hook(callback, where):
    model = _Model()
    _, quant_state = _nf4_state()
    if where == "parameter":
        model.proj.weight.quant_state = quant_state
    else:
        model.proj.__dict__["quant_state"] = quant_state  # FSDP2: the gathered Parameter dropped it
    callback.on_train_begin(None, None, SimpleNamespace(), model = model)
    assert len(dc._HOOKS) == 1 and dc._ACTIVE == 1
    assert _forward_host_reads(model) == 1  # the padded-mask check
    assert dc._PRE_RAN[0] == 1

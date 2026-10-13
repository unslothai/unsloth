# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""RoPE drains the stream only for a model split over several devices, not on every multi-GPU host."""

import gc
import weakref

import pytest
import torch

import unsloth.kernels.rope_embedding as RE


class _FakeQ:
    # Enough of a tensor for the sync decision: the Triton / torch RoPE is stubbed out below.
    def __init__(self, index):
        self.index = index
        self.device = torch.device("cuda", index)

    def get_device(self):
        return self.index


class _Model:
    def __init__(self, device_map):
        self.hf_device_map = device_map


@pytest.fixture
def current(monkeypatch):
    # The current device of the process: a DDP rank sets it to its own index.
    device = [0]
    monkeypatch.setattr(RE, "_current_device", lambda: device[0], raising = False)
    return device


@pytest.fixture
def synced(monkeypatch, current):
    calls = []

    class _Stream:
        def __init__(self, device):
            self.device = device

        def synchronize(self):
            calls.append(self.device.index)

    monkeypatch.setattr(RE, "torch_device_stream", _Stream)
    monkeypatch.setattr(RE, "_SPLIT_MODELS", weakref.WeakSet(), raising = False)
    monkeypatch.setattr(RE.Fast_RoPE_Embedding_QK, "apply", lambda Q, K, *a: (Q, K))
    monkeypatch.setattr(RE.Slow_RoPE_Embedding, "apply", lambda Q, *a: Q)
    return calls


def _rope(index):
    Q = _FakeQ(index)
    RE._fast_rope_embedding_eager(Q, Q, None, None, rope_embedding_indices = object())


@pytest.mark.parametrize("index", [0, 3])
def test_single_device_on_multi_gpu_host_never_syncs(monkeypatch, synced, current, index):
    # Single-GPU training (or one DDP rank) with several GPUs visible.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 8)
    current[0] = index
    assert not RE.register_split_model(_Model({"": index}))
    for _ in range(64):
        _rope(index)
    assert synced == []


def test_inplace_rope_single_device_never_syncs(monkeypatch, synced):
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    for _ in range(8):
        Q = _FakeQ(0)
        RE.inplace_rope_embedding(Q, Q, None, None, None)
    assert synced == []


def test_one_visible_device_never_syncs(monkeypatch, synced):
    monkeypatch.setattr(RE, "DEVICE_COUNT", 1)
    assert not RE.register_split_model(_Model({"model.layers.0": 0, "model.layers.1": 1}))
    for _ in range(8):
        _rope(0)
        Q = _FakeQ(0)
        RE.inplace_rope_embedding(Q, Q, None, None, None)
    assert synced == []


def test_split_model_syncs_from_the_first_call(monkeypatch, synced):
    # Layers on cuda:0 then cuda:1 (device_map split): registered at load, so the cuda:0 shard of the
    # very first forward drains too, as on main.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    model = _Model({"model.embed_tokens": 0, "model.layers.0": "cuda:0", "model.layers.1": "1", "lm_head": "cpu"})
    assert RE.register_split_model(model)
    layers = [0, 0, 1, 1]
    for _ in range(3):
        for index in layers:
            _rope(index)
    assert synced == layers * 3
    Q = _FakeQ(0)
    RE.inplace_rope_embedding(Q, Q, None, None, None)
    assert synced[-1] == 0


def test_independent_single_device_models_stay_unsynced(monkeypatch, synced, current):
    # Two single-device models on different GPUs of one process, each run with its device current.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    a, b = _Model({"": 0}), _Model({"": "cuda:1"})
    assert not RE.register_split_model(a) and not RE.register_split_model(b)
    for _ in range(4):
        for index in (0, 1):
            current[0] = index
            _rope(index)
    assert synced == []


def test_freed_split_model_stops_syncing(monkeypatch, synced):
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    model = _Model({"model.layers.0": 0, "model.layers.1": 1})
    assert RE.register_split_model(model)
    _rope(0)
    assert synced == [0]
    del model
    gc.collect()
    _rope(0)
    assert synced == [0]


def test_unregistered_split_syncs_off_current_device(monkeypatch, synced):
    # A split nothing registered: the shard off the current device still drains.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    for index in [0, 0, 1, 1]:
        _rope(index)
    assert synced == [1, 1]


def test_hand_dispatched_split_syncs_from_the_first_call(monkeypatch, synced):
    # accelerate.dispatch_model called by hand after loading registers the split before the first forward,
    # including through the names transformers bound at import.
    import importlib
    import sys

    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    calls = []

    def stub(model, device_map, **kwargs):
        calls.append(device_map)

    names = [
        "accelerate",
        "accelerate.big_modeling",
        "transformers.modeling_utils",
        "transformers.integrations.accelerate",
        "transformers.integrations.peft",
    ]
    bound = []
    importlib.import_module("accelerate.big_modeling")
    for name in names:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "dispatch_model"):
            monkeypatch.setattr(module, "dispatch_model", stub)
            bound.append(module)
    RE._register_splits_on_dispatch()
    assert all(module.dispatch_model is not stub for module in bound)
    RE._register_splits_on_dispatch()
    assert not getattr(stub, "_unsloth_registers_split", False)

    import accelerate

    single, split = _Model({}), _Model({})
    accelerate.dispatch_model(single, device_map = {"": 1})
    accelerate.dispatch_model(split, {"model.layers.0": 0, "model.layers.1": "cuda:1"})
    assert len(calls) == 2 and len(RE._SPLIT_MODELS) == 1 and split in RE._SPLIT_MODELS
    for index in [0, 0, 1, 1]:
        _rope(index)
    assert synced == [0, 0, 1, 1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA")
@pytest.mark.parametrize("use_indices", [False, True])
def test_real_kernel_single_device_no_sync_same_result(monkeypatch, use_indices):
    monkeypatch.setattr(RE, "_SPLIT_MODELS", weakref.WeakSet(), raising = False)
    monkeypatch.setattr(RE, "_current_device", torch.cuda.current_device)
    count = [0]
    real = torch.cuda.Stream.synchronize

    def counting(self):
        count[0] += 1
        return real(self)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", counting)
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 32, 64
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device = "cuda").float() / d))
    emb = torch.cat([torch.outer(torch.arange(s, device = "cuda").float(), inv)] * 2, -1)
    cos, sin = emb.cos(), emb.sin()
    idx = torch.arange(s, device = "cuda").expand(b, s).contiguous() if use_indices else None
    x = torch.randn(b, h, s, d, device = "cuda")

    monkeypatch.setattr(RE, "DEVICE_COUNT", 1)
    ref_q, ref_k = RE.fast_rope_embedding(x.clone(), x.clone(), cos, sin, idx)
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    q, k = RE.fast_rope_embedding(x.clone(), x.clone(), cos, sin, idx)
    torch.cuda.synchronize()
    assert count[0] == 0
    assert torch.equal(q, ref_q) and torch.equal(k, ref_k)

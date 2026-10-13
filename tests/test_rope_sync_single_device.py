# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""RoPE drains the stream only for a model split over several devices, not on every multi-GPU host."""

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


@pytest.fixture
def synced(monkeypatch):
    calls = []

    class _Stream:
        def __init__(self, device):
            self.device = device

        def synchronize(self):
            calls.append(self.device.index)

    monkeypatch.setattr(RE, "torch_device_stream", _Stream)
    monkeypatch.setattr(RE, "_ROPE_DEVICES", set(), raising = False)
    monkeypatch.setattr(RE.Fast_RoPE_Embedding_QK, "apply", lambda Q, K, *a: (Q, K))
    monkeypatch.setattr(RE.Slow_RoPE_Embedding, "apply", lambda Q, *a: Q)
    return calls


def _rope(index):
    Q = _FakeQ(index)
    RE._fast_rope_embedding_eager(Q, Q, None, None, rope_embedding_indices = object())


@pytest.mark.parametrize("index", [0, 3])
def test_single_device_on_multi_gpu_host_never_syncs(monkeypatch, synced, index):
    # Single-GPU training (or one DDP rank) with several GPUs visible.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 8)
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
    for _ in range(8):
        _rope(0)
        Q = _FakeQ(0)
        RE.inplace_rope_embedding(Q, Q, None, None, None)
    assert synced == []


def test_split_model_keeps_the_sync(monkeypatch, synced):
    # Layers on cuda:0 then cuda:1 (device_map / pipeline split): once a second device shows up,
    # every RoPE call drains its own device's stream, as before.
    monkeypatch.setattr(RE, "DEVICE_COUNT", 2)
    layers = [0, 0, 1, 1]
    for _ in range(3):
        for index in layers:
            _rope(index)
    assert synced == [1, 1] + layers * 2
    Q = _FakeQ(0)
    RE.inplace_rope_embedding(Q, Q, None, None, None)
    assert synced[-1] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA")
@pytest.mark.parametrize("use_indices", [False, True])
def test_real_kernel_single_device_no_sync_same_result(monkeypatch, use_indices):
    monkeypatch.setattr(RE, "_ROPE_DEVICES", set(), raising = False)
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

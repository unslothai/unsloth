# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""xformers' grouped (5-D, GQA) forward follows the current CUDA device, not the tensors'."""

import contextlib

import pytest
import torch

ad = pytest.importorskip("unsloth.utils.attention_dispatch")


class _FakeX:
    is_cuda = True

    def __init__(self, index):
        self.device = torch.device("cuda", index)


def test_guard_switches_to_the_tensor_device(monkeypatch):
    entered = []
    monkeypatch.setattr(ad, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "device", lambda d: entered.append(d) or contextlib.nullcontext()
    )
    with ad._on_tensor_device(_FakeX(1)):
        pass
    assert entered == [torch.device("cuda", 1)]


def test_guard_is_a_noop_on_the_current_device(monkeypatch):
    entered = []
    monkeypatch.setattr(ad, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "device", lambda d: entered.append(d) or contextlib.nullcontext()
    )
    with ad._on_tensor_device(_FakeX(1)):
        pass
    assert entered == []


def test_run_attention_calls_xformers_on_the_tensor_device(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or ad.xformers is None:
        pytest.skip("needs 2 CUDA devices and xformers")
    seen = []
    real = ad.xformers_attention

    def spy(*a, **k):
        seen.append(torch.cuda.current_device())
        return real(*a, **k)

    monkeypatch.setattr(ad, "xformers_attention", spy)
    out, _ = _grouped_no_grad(device = 1, current = 0)
    assert seen == [1]
    assert torch.isfinite(out).all()


def _grouped_no_grad(
    device,
    current,
    S = 2048,
    n_kv = 8,
    groups = 2,
    D = 128,
):
    g = torch.Generator(device = f"cuda:{device}").manual_seed(0)
    Q = torch.randn(
        1, n_kv * groups, S, D, device = f"cuda:{device}", dtype = torch.float16, generator = g
    )
    K = torch.randn(1, n_kv, S, D, device = f"cuda:{device}", dtype = torch.float16, generator = g)
    V = torch.randn(1, n_kv, S, D, device = f"cuda:{device}", dtype = torch.float16, generator = g)
    config = ad.AttentionConfig(backend = ad.XFORMERS, n_kv_heads = n_kv, n_groups = groups)
    context = ad.AttentionContext(
        bsz = 1, q_len = S, kv_seq_len = S, n_heads = n_kv * groups, head_dim = D,
        requires_grad = False, seq_info = None, training = False, attention_mask = None,
        causal_mask = ad.xformers.attn_bias.LowerTriangularMask(),
    )  # fmt: skip
    with torch.no_grad(), torch.cuda.device(current):
        return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V), (Q, K, V)


def test_grouped_no_grad_attention_on_second_card_matches_reference():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or ad.xformers is None:
        pytest.skip("needs 2 CUDA devices and xformers")
    out, (Q, K, V) = _grouped_no_grad(device = 1, current = 0)
    groups = Q.shape[1] // K.shape[1]
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float().repeat_interleave(groups, 1),
        V.float().repeat_interleave(groups, 1),
        is_causal = True,
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), ref, atol = 2e-2, rtol = 0)

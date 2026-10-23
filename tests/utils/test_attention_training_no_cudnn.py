# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Training SDPA on SM100 avoids cuDNN attention; eval / generation calls are untouched.

torch 2.13+ on B200 picks cuDNN attention for SDPA, which builds a plan for every new sequence length
(~150 ms each). Padded training batches change length every step: Qwen3-0.6B LoRA went from 2777-3060
to 5164-5333 useful tokens/s with flash / efficient / math (exclusive B200, no xformers, interleaved).
"""

import pytest
import torch
import torch.nn.functional as F

ad = pytest.importorskip("unsloth.utils.attention_dispatch")


class _CudaLooking(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _run(
    training,
    monkeypatch,
    devices = frozenset({None}),
):
    seen = []

    def fake_sdpa(q, k, v, **kwargs):
        seen.append(torch.backends.cuda.cudnn_sdp_enabled())
        return torch.zeros_like(q)

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", devices)
    Q = torch.randn(1, 2, 6, 8).as_subclass(_CudaLooking)
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 2, n_groups = 1)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 6,
        kv_seq_len = 6,
        n_heads = 2,
        head_dim = 8,
        requires_grad = False,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        training = training,
    )
    ad.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    return seen


def test_training_call_disables_cudnn(monkeypatch):
    before = torch.backends.cuda.cudnn_sdp_enabled()
    assert _run(True, monkeypatch) == [False]
    assert torch.backends.cuda.cudnn_sdp_enabled() == before


def test_eval_call_and_other_devices_untouched(monkeypatch):
    before = torch.backends.cuda.cudnn_sdp_enabled()
    assert _run(False, monkeypatch) == [before]
    assert _run(True, monkeypatch, devices = frozenset()) == [before]


def test_only_sm100_devices_and_opt_out(monkeypatch):
    monkeypatch.setattr(ad, "_visible_capabilities", lambda: ((9, 0), (10, 0), (12, 0), (10, 3)))
    monkeypatch.delenv("UNSLOTH_SDPA_TRAIN_CUDNN", raising = False)
    assert ad._no_cudnn_training_devices() == frozenset({1, 3})
    monkeypatch.setenv("UNSLOTH_SDPA_TRAIN_CUDNN", "1")
    assert ad._no_cudnn_training_devices() == frozenset()


def test_checkpointed_no_grad_forward_still_pins(monkeypatch):
    """A checkpointed layer's first forward runs inside autograd.Function.forward (grad off)."""

    class Ckpt(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            assert not torch.is_grad_enabled()
            seen.extend(_run(True, monkeypatch))
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            return g

    seen = []
    x = torch.randn(2, requires_grad = True)
    Ckpt.apply(x)
    assert seen == [False]


def test_training_sdpa_traces_without_a_break(monkeypatch):
    from torch._dynamo.utils import counters

    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", frozenset({None}))
    torch._dynamo.reset()
    counters.clear()

    def f(q):
        return ad._training_sdpa(q, q, q, is_causal = True)

    c = torch.compile(f, backend = "eager", dynamic = True)
    for L in (5, 9, 16):
        q = torch.randn(1, 2, L, 8)
        torch.testing.assert_close(c(q), F.scaled_dot_product_attention(q, q, q, is_causal = True))
    assert dict(counters["graph_break"]) == {}
    torch._dynamo.reset()

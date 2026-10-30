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


def test_gqa_probe_runs_under_the_training_restriction(monkeypatch):
    """The enable_gqa decision is probed with the kernels the call runs with (cuDNN off)."""
    probed = []

    def fake_now(*args):
        probed.append((args, torch.backends.cuda.cudnn_sdp_enabled()))
        # cuDNN takes this masked GQA call; without it only math would.
        return torch.backends.cuda.cudnn_sdp_enabled()

    monkeypatch.setattr(ad, "_sdpa_gqa_state_now", fake_now)
    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", frozenset({None}))
    Q = torch.randn(1, 4, 6, 8).as_subclass(_CudaLooking)
    K = torch.randn(1, 2, 6, 8).as_subclass(_CudaLooking)
    before = torch.backends.cuda.cudnn_sdp_enabled()
    assert (
        ad._sdpa_takes_gqa(Q, K, False, True, K, 0.0, no_cudnn = ad._avoids_cudnn(Q, True)) is False
    )
    assert (
        ad._sdpa_takes_gqa(Q, K, False, True, K, 0.0, no_cudnn = ad._avoids_cudnn(Q, False)) is before
    )
    assert [cudnn for _, cudnn in probed] == [False, before]
    assert probed[0][0] == probed[1][0]  # same shapes, only the backend set differs
    assert torch.backends.cuda.cudnn_sdp_enabled() == before
    assert ad._avoids_cudnn(Q, True) and not ad._avoids_cudnn(Q, False)
    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", frozenset())
    assert not ad._avoids_cudnn(Q, True)


def test_dense_and_segment_paths_probe_without_cudnn_when_training(monkeypatch):
    flags = []
    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", frozenset({None}))
    monkeypatch.setattr(
        ad, "_sdpa_takes_gqa", lambda *a, no_cudnn = False, **k: flags.append(no_cudnn) or False
    )
    monkeypatch.setattr(
        F, "scaled_dot_product_attention", lambda q, k, v, **kw: torch.zeros_like(q)
    )
    Q = torch.randn(1, 4, 6, 8).as_subclass(_CudaLooking)
    K = torch.randn(1, 2, 6, 8).as_subclass(_CudaLooking)
    for training in (True, False):
        ad._sdpa_packed_segments(
            Q,
            K,
            K,
            (2, 4),
            n_groups = 2,
            is_causal = True,
            sliding_window = None,
            sdpa_kwargs = {},
            training = training,
        )
    assert flags == [True, True, False, False]


_POLICIES = {
    "all": ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"),
    "math": ("MATH",),
    "math_cudnn": ("MATH", "CUDNN_ATTENTION"),
    "efficient_math_cudnn": ("EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"),
    "cudnn": ("CUDNN_ATTENTION",),
}


def _policy(name):
    from torch.nn.attention import SDPBackend
    if not hasattr(SDPBackend, "CUDNN_ATTENTION"):
        pytest.skip("torch has no cuDNN SDPA backend")
    return [getattr(SDPBackend, b) for b in _POLICIES[name]]


def _flags():
    cuda = torch.backends.cuda
    return (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
    )


def _expected(name):
    """The caller's own set with cuDNN dropped; a cuDNN-only policy is left as is."""
    names = _POLICIES[name]
    keep_cudnn = names == ("CUDNN_ATTENTION",)
    return (
        "FLASH_ATTENTION" in names,
        "EFFICIENT_ATTENTION" in names,
        "MATH" in names,
        keep_cudnn,
    )


@pytest.mark.parametrize("name", list(_POLICIES))
def test_training_pin_keeps_caller_backend_policy(monkeypatch, name):
    """sdpa_kernel([MATH]) around a training step must stay math on SM100, not regain flash."""
    from torch.nn.attention import sdpa_kernel

    seen = []
    monkeypatch.setattr(
        F, "scaled_dot_product_attention", lambda q, k, v, **kw: seen.append(_flags()) or q
    )
    monkeypatch.setattr(ad, "_NO_CUDNN_TRAINING", frozenset({None}))
    Q = torch.randn(1, 2, 6, 8).as_subclass(_CudaLooking)
    with sdpa_kernel(_policy(name)):
        before = _flags()
        ad._training_sdpa(Q, Q, Q, is_causal = True)
        ad._rerouted_sdpa(Q, Q, Q, is_causal = True)
        assert _flags() == before
    assert seen == [_expected(name)] * 2


@pytest.mark.parametrize("name", ["all", "math_cudnn", "cudnn"])
def test_gqa_probe_uses_the_callers_cudnn_free_set(monkeypatch, name):
    from torch.nn.attention import sdpa_kernel

    probed = []
    monkeypatch.setattr(ad, "_sdpa_gqa_state_now", lambda *a: probed.append(_flags()) or True)
    Q = torch.randn(1, 4, 6, 8).as_subclass(_CudaLooking)
    K = torch.randn(1, 2, 6, 8).as_subclass(_CudaLooking)
    with sdpa_kernel(_policy(name)):
        ad._sdpa_takes_gqa(Q, K, False, True, K, 0.0, no_cudnn = True)
    assert probed == [_expected(name)]


@pytest.mark.parametrize("name", ["all", "math", "math_cudnn"])
def test_compiled_training_pin_keeps_caller_backend_policy(monkeypatch, name):
    """Traced, the pin is read under the caller's sdpa_kernel at trace time, with no graph break."""
    from torch._dynamo.utils import counters
    from torch.nn.attention import sdpa_kernel

    pinned = []
    real = ad._sdpa_kernel

    def recording(backends):
        pinned.append(tuple(backends))
        return real(backends)

    monkeypatch.setattr(ad, "_sdpa_kernel", recording)
    monkeypatch.setattr(ad, "_avoids_cudnn", lambda q, training: bool(training))
    torch._dynamo.reset()
    counters.clear()

    def f(q):
        return ad._training_sdpa(q, q, q, is_causal = True)

    policy = _policy(name)
    c = torch.compile(f, backend = "eager", fullgraph = True, dynamic = True)
    with sdpa_kernel(policy):
        for L in (5, 9, 16):
            q = torch.randn(1, 2, L, 8, requires_grad = True)
            c(q).sum().backward()
    assert dict(counters["graph_break"]) == {}
    expected = tuple(b for b in ad._XFORMERS_LIKE_BACKENDS if b in policy)
    # A cuDNN-free policy has nothing to drop: no pin at all.
    assert set(pinned) == ({expected} if any("CUDNN" in str(b) for b in policy) else set())
    torch._dynamo.reset()

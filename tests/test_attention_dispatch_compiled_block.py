# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Attention dispatch traced fullgraph with a dynamic sequence length (CPU)."""

import pytest
import torch

ad = pytest.importorskip("unsloth.utils.attention_dispatch")


def _run(backend, causal_mask = None):
    def attend(Q, K, V):
        bsz, n_heads, q_len, head_dim = Q.shape
        config = ad.AttentionConfig(
            backend = backend, n_kv_heads = K.shape[1], n_groups = n_heads // K.shape[1]
        )
        context = ad.AttentionContext(
            bsz = bsz,
            q_len = q_len,
            kv_seq_len = K.shape[-2],
            n_heads = n_heads,
            head_dim = head_dim,
            requires_grad = True,
            seq_info = None,
            attention_mask = None,
            causal_mask = causal_mask,
        )
        return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)

    return attend


def _qkv(q_len):
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(1, 4, q_len, 16, generator = g)
    K = torch.randn(1, 2, q_len, 16, generator = g)
    V = torch.randn(1, 2, q_len, 16, generator = g)
    return Q, K, V


def _reference(Q, K, V):
    K = K.repeat_interleave(2, dim = 1)
    V = V.repeat_interleave(2, dim = 1)
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal = True).transpose(1, 2)


@pytest.mark.parametrize("backend", ["sdpa", "xformers"])
def test_plain_causal_attention_traces_with_a_dynamic_length(backend):
    backend = {"sdpa": ad.SDPA, "xformers": ad.XFORMERS}[backend]
    torch._dynamo.reset()
    compiled = torch.compile(_run(backend), fullgraph = True, backend = "eager", dynamic = True)
    for q_len in (8, 12):
        Q, K, V = _qkv(q_len)
        out = compiled(Q, K, V)
        torch.testing.assert_close(out.reshape(1, q_len, 4, 16), _reference(Q, K, V))


def test_xformers_eager_choice_is_unchanged(monkeypatch):
    # Outside a trace the backend is the caller's: the SDPA route is only taken while compiling.
    seen = []
    monkeypatch.setattr(
        ad, "build_xformers_block_causal_mask", lambda *a, **k: seen.append(1) or None
    )
    monkeypatch.setattr(ad, "move_xformers_attention_bias", lambda bias, device: bias)
    monkeypatch.setattr(ad, "xformers_attention", lambda Q, K, V, attn_bias = None, **kw: Q)
    Q, K, V = _qkv(8)
    _run(ad.XFORMERS)(Q, K, V)
    assert seen


def test_rerouted_xformers_keeps_the_fp32_downcast(monkeypatch):
    # DoRA fp32 Q/K/V on a GPU whose xformers op rejects fp32: the SDPA reroute must downcast like xformers.
    monkeypatch.setattr(ad, "_xformers_rejects_fp32", lambda device: True)
    monkeypatch.setattr(ad, "SUPPORTS_BFLOAT16", True)
    torch._dynamo.reset()
    compiled = torch.compile(_run(ad.XFORMERS), fullgraph = True, backend = "eager", dynamic = True)
    Q, K, V = _qkv(8)
    out = compiled(Q, K, V)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(
        out.float().reshape(1, 8, 4, 16), _reference(Q, K, V), atol = 3e-2, rtol = 3e-2
    )


def _compile_counts(fn, *inputs, dynamic):
    """Run fn compiled fullgraph over each input tuple; return (outputs, unique graphs, graph breaks)."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, fullgraph = True, backend = "eager", dynamic = dynamic)
    outs = [compiled(*args) for args in inputs]
    return outs, counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())


@pytest.mark.parametrize("dynamic", [False, True])
def test_gqa_probe_takes_symbolic_head_sizes(dynamic):
    # dynamic=True makes head_dim and the head counts SymInts; the constant-result probe must still trace.
    def state(Q, K):
        takes = ad._sdpa_gqa_state(
            str(Q.device), Q.dtype, Q.shape[-1], Q.shape[1], K.shape[1], False, True
        )
        return Q.sum() + K.sum() + (1 if takes else 2)

    inputs = [
        (torch.zeros(b, 8, t, 64), torch.zeros(b, 2, t, 64)) for b, t in ((3, 10), (5, 12), (6, 14))
    ]
    outs, graphs, breaks = _compile_counts(state, *inputs, dynamic = dynamic)
    want = 1 if ad._sdpa_gqa_state("cpu", torch.float32, 64, 8, 2, False, True) else 2
    assert [out.item() for out in outs] == [want] * 3
    assert breaks == 0
    # Batch and length stay dynamic: one graph when dynamic, one per shape when static. Sizes avoid
    # the head counts: duck sizing would share a symbol, and the specialised head count with it.
    assert graphs == (1 if dynamic else 3)


@pytest.mark.parametrize("dynamic", [False, True])
def test_varlen_probe_takes_symbolic_head_sizes(dynamic):
    def beats(Q, K):
        return (
            Q.sum() + K.sum() + (1 if ad._torch_varlen_beats_segments(Q, K, (5, 3), 2, True) else 2)
        )

    Q, K = torch.zeros(1, 4, 8, 16), torch.zeros(1, 2, 8, 16)
    outs, _, breaks = _compile_counts(beats, (Q, K), dynamic = dynamic)
    want = 1 if ad._torch_varlen_beats_segments(Q, K, (5, 3), 2, True) else 2
    assert [out.item() for out in outs] == [want]
    assert breaks == 0


def _has_real_cuda():
    try:
        from real_accelerator import (
            has_real_cuda,
        )  # tests/_shared, on sys.path via tests/conftest.py
    except Exception:
        return torch.cuda.is_available()
    return has_real_cuda()


@pytest.mark.skipif(not _has_real_cuda(), reason = "needs a real CUDA device")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_gqa_sdpa_compiles_fullgraph(dynamic):
    # The CUDA GQA probe runs only on CUDA tensors: noncausal bf16 GQA, no mask, forward + backward.
    def attend(Q, K, V):
        bsz, n_heads, q_len, head_dim = Q.shape
        config = ad.AttentionConfig(
            backend = ad.SDPA, n_kv_heads = K.shape[1], n_groups = n_heads // K.shape[1]
        )
        context = ad.AttentionContext(
            bsz = bsz,
            q_len = q_len,
            kv_seq_len = K.shape[-2],
            n_heads = n_heads,
            head_dim = head_dim,
            requires_grad = True,
            seq_info = None,
            attention_mask = None,
            causal_mask = None,
            is_causal = False,
            sliding_window = None,
        )
        out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
        # Flatten the heads as the o_proj input does (torch 2.13's dynamic-shape cuDNN backward
        # mishandles a transposed incoming gradient, independent of the dispatcher).
        return out.reshape(bsz, q_len, n_heads * head_dim)

    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(attend, fullgraph = True, dynamic = dynamic)
    shapes = ((3, 32), (5, 48), (3, 40))  # no size equal to a head count (duck sizing)
    for bsz, q_len in shapes:
        g = torch.Generator(device = "cuda").manual_seed(q_len)
        Q, K, V = (
            torch.randn(
                bsz, h, q_len, 64, device = "cuda", generator = g, dtype = torch.bfloat16
            ).requires_grad_()
            for h in (8, 2, 2)
        )
        out = compiled(Q, K, V)
        out.float().square().sum().backward()
        Qe, Ke, Ve = (t.detach().clone().requires_grad_() for t in (Q, K, V))
        ref = attend(Qe, Ke, Ve)
        ref.float().square().sum().backward()
        torch.testing.assert_close(out, ref, atol = 2e-2, rtol = 2e-2)
        for got, want in zip((Q, K, V), (Qe, Ke, Ve)):
            torch.testing.assert_close(got.grad, want.grad, atol = 2e-1, rtol = 2e-2)
    assert sum(counters["graph_break"].values()) == 0
    # Head sizes are specialised, batch and length are not: dynamic compiles once.
    assert counters["stats"]["unique_graphs"] == (1 if dynamic else len(shapes))

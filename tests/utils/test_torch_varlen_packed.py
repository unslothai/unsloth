# SPDX-License-Identifier: AGPL-3.0-only
"""Packed SDPA rows via torch varlen_attn match the per-segment fallback."""

import types

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import packing

_REAL_VARLEN = ad._TORCH_VARLEN_ATTN
needs_varlen = pytest.mark.skipif(
    not has_real_cuda()
    or ad._TORCH_VARLEN_ATTN is None
    or torch.cuda.get_device_capability()[0] < 8,
    reason = "needs sm80+ CUDA and torch.nn.attention.varlen with window_size + enable_gqa (2.12+)",
)


def _seq_info(lengths, device):
    lengths = torch.tensor(lengths, dtype = torch.int32, device = device)
    cu = torch.cat((torch.zeros(1, dtype = torch.int32, device = device), lengths.cumsum(0).int()))
    return lengths, cu, int(lengths.max())


def _run(
    monkeypatch,
    varlen,
    lengths,
    total,
    n_heads,
    n_kv,
    qkv,
    is_causal = True,
    window = None,
    sdpa_kwargs = None,
):
    calls = []
    if varlen:

        def counted(*args, **kwargs):
            calls.append(kwargs["window_size"])
            return _REAL_VARLEN(*args, **kwargs)

        monkeypatch.setattr(ad, "_TORCH_VARLEN_ATTN", counted)
    else:
        monkeypatch.setattr(ad, "_TORCH_VARLEN_ATTN", None)
    packing.clear_packed_caches()
    config = ad.AttentionConfig(
        backend = ad.SDPA, n_kv_heads = n_kv, n_groups = n_heads // n_kv, sdpa_kwargs = sdpa_kwargs
    )
    context = ad.AttentionContext(
        bsz = 1,
        q_len = total,
        kv_seq_len = total,
        n_heads = n_heads,
        head_dim = qkv[0].shape[-1],
        requires_grad = True,
        seq_info = _seq_info(lengths, qkv[0].device),
        attention_mask = None,
        causal_mask = None,
        is_causal = is_causal,
        sliding_window = window,
    )
    Q, K, V = (x.detach().clone().requires_grad_() for x in qkv)
    out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    out = out.reshape(1, total, n_heads, -1)
    (
        out.float() * torch.linspace(-1, 1, out.numel(), device = out.device).view_as(out)
    ).sum().backward()
    return (out.detach(), Q.grad, K.grad, V.grad), calls


CASES = [
    # lengths, total (> sum = pad tail), heads, kv heads, causal, window, expected window_size
    ((5, 3, 4), 12, 4, 4, True, None, (-1, 0)),
    ((5, 3, 4), 12, 4, 2, True, None, (-1, 0)),
    ((4, 4, 4), 12, 4, 1, True, None, None),  # one run of equal lengths: one batched SDPA call
    ((7,), 10, 4, 2, True, None, (-1, 0)),
    ((6, 2, 5), 13, 4, 2, True, 3, (2, 0)),
    ((6, 2, 5), 13, 4, 2, False, None, (-1, -1)),
    ((1, 1, 9), 11, 2, 1, True, 4, (3, 0)),
    ((300, 700, 24), 1100, 8, 2, True, None, (-1, 0)),
    ((300, 700, 24), 1100, 8, 2, True, 128, (127, 0)),
]


@needs_varlen
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("lengths, total, n_heads, n_kv, is_causal, window, window_size", CASES)
def test_varlen_matches_segments(
    monkeypatch, dtype, head_dim, lengths, total, n_heads, n_kv, is_causal, window, window_size
):
    g = torch.Generator().manual_seed(3407)
    shapes = ((1, n_heads, total, head_dim), (1, n_kv, total, head_dim), (1, n_kv, total, head_dim))
    qkv = [torch.randn(s, generator = g).to("cuda", dtype) for s in shapes]
    args = (lengths, total, n_heads, n_kv, qkv, is_causal, window)
    segs, seg_calls = _run(monkeypatch, False, *args)
    varlen, calls = _run(monkeypatch, True, *args)
    assert seg_calls == [] and calls == ([] if window_size is None else [window_size])
    for name, a, b in zip(("out", "dQ", "dK", "dV"), segs, varlen):
        assert torch.isfinite(b).all(), name
        torch.testing.assert_close(b.float(), a.float(), atol = 2e-2, rtol = 2e-2, msg = name)


@needs_varlen
def test_bidirectional_mha_head_dim_256_runs(monkeypatch):
    g = torch.Generator().manual_seed(2)
    qkv = [torch.randn(1, 8, 1024, 256, generator = g).cuda().bfloat16() for _ in range(3)]
    out, calls = _run(monkeypatch, True, (300, 500, 224), 1024, 8, 8, qkv, is_causal = False)
    assert calls == [] and all(torch.isfinite(x).all() for x in out)


@needs_varlen
@pytest.mark.parametrize(
    "backend, lengths, total, expect_varlen",
    [
        ("CUDNN_ATTENTION", (300, 700, 24), 1100, True),  # mean 275: launches dominate
        ("CUDNN_ATTENTION", (1500, 2000, 1800), 5300, False),  # mean 1767: cuDNN per segment wins
        ("FLASH_ATTENTION", (1500, 2000, 1800), 5300, True),  # same kernel, fewer launches
        ("FLASH_ATTENTION", (2048, 2048), 4096, False),  # one run: one batched call already
    ],
)
def test_varlen_only_where_it_beats_segments(monkeypatch, backend, lengths, total, expect_varlen):
    from torch.nn.attention import SDPBackend as B

    monkeypatch.setattr(torch, "_fused_sdp_choice", lambda *a, **k: int(getattr(B, backend)))
    qkv = [torch.zeros(1, 4, total, 64, device = "cuda", dtype = torch.bfloat16) for _ in range(3)]
    _, calls = _run(monkeypatch, True, lengths, total, 4, 4, qkv)
    assert calls == ([(-1, 0)] if expect_varlen else [])


@needs_varlen
def test_scale_is_forwarded(monkeypatch):
    g = torch.Generator().manual_seed(0)
    qkv = [torch.randn(1, 4, 40, 64, generator = g).cuda().bfloat16() for _ in range(3)]
    kwargs = dict(sdpa_kwargs = {"scale": 0.3, "dropout_p": 0.0})
    segs, _ = _run(monkeypatch, False, (25, 15), 40, 4, 4, qkv, **kwargs)
    varlen, calls = _run(monkeypatch, True, (25, 15), 40, 4, 4, qkv, **kwargs)
    assert calls == [(-1, 0)]
    torch.testing.assert_close(varlen[0].float(), segs[0].float(), atol = 2e-2, rtol = 2e-2)


@needs_varlen
@pytest.mark.parametrize(
    "case",
    [
        "fp32",
        "dropout",
        "attn_mask",
        "bidirectional_window",
        "bidirectional_mha",
        "deterministic",
        "int32_overflow",
        "flash_sdp_disabled",
    ],
)
def test_ineligible_calls_keep_the_segment_path(monkeypatch, case):
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    g = torch.Generator().manual_seed(1)
    qkv = [torch.randn(1, 4, 24, 64, generator = g).to("cuda", dtype) for _ in range(3)]
    kwargs = {}
    if case == "dropout":
        kwargs["sdpa_kwargs"] = {"dropout_p": 0.1}
        torch.manual_seed(0)
    elif case == "attn_mask":
        kwargs["sdpa_kwargs"] = {"attn_mask": None, "scale": None}
    elif case == "bidirectional_window":
        kwargs.update(is_causal = False, window = 4)
    elif case == "bidirectional_mha":
        kwargs.update(is_causal = False)
    elif case == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif case == "int32_overflow":
        monkeypatch.setattr(ad, "_varlen_backward_overflows_int32", lambda *a: True)
    if case == "flash_sdp_disabled":
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            _, calls = _run(monkeypatch, True, (10, 14), 24, 4, 4, qkv, **kwargs)
    else:
        _, calls = _run(monkeypatch, True, (10, 14), 24, 4, 4, qkv, **kwargs)
    assert calls == []


def test_loader_needs_window_and_gqa(monkeypatch):
    import sys

    def old_varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        *,
        is_causal = False,
    ):
        raise AssertionError("not called")

    fake = types.ModuleType("torch.nn.attention.varlen")
    fake.varlen_attn = old_varlen_attn
    monkeypatch.setitem(sys.modules, "torch.nn.attention.varlen", fake)
    monkeypatch.delenv("UNSLOTH_TORCH_VARLEN", raising = False)
    assert ad._load_torch_varlen_attn() is None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_VARLEN", "0")
    assert ad._load_torch_varlen_attn() is None

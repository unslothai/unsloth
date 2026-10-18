# SPDX-License-Identifier: AGPL-3.0-only
"""PyTorch-native attention routing: GQA never forces SDPA's math kernel, cuDNN-over-varlen only
where it was measured, and native is preferred over xformers where it is at least as fast."""

import pytest
import torch
from torch.nn.attention import SDPBackend
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import packing

needs_cuda = pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device")

MATH = int(SDPBackend.MATH)
FLASH = int(SDPBackend.FLASH_ATTENTION)
EFFICIENT = int(SDPBackend.EFFICIENT_ATTENTION)
CUDNN = int(SDPBackend.CUDNN_ATTENTION)


@pytest.fixture
def fake_choice(monkeypatch):
    """Make torch._fused_sdp_choice answer from a table keyed on (gqa, mask)."""
    table = {}

    def choice(
        q,
        k,
        v,
        attn_mask = None,
        is_causal = False,
        enable_gqa = False,
        **_,
    ):
        return table[(bool(enable_gqa), attn_mask is not None)]

    monkeypatch.setattr(torch, "_fused_sdp_choice", choice)
    ad._sdpa_gqa_choices.cache_clear()
    yield table
    ad._sdpa_gqa_choices.cache_clear()


@pytest.mark.parametrize(
    "gqa_choice, mha_choice, expect",
    [
        (FLASH, FLASH, True),  # flash takes GQA: no K/V copy
        (CUDNN, CUDNN, True),  # cuDNN takes GQA
        (MATH, EFFICIENT, False),  # GQA would leave only math: expand K/V instead
        (MATH, MATH, True),  # math either way: let math broadcast
        (MATH, -1, True),  # unknown: unchanged behaviour
    ],
)
def test_gqa_only_where_it_does_not_force_math(fake_choice, gqa_choice, mha_choice, expect):
    fake_choice[(True, True)] = gqa_choice
    fake_choice[(False, True)] = mha_choice
    assert ad._sdpa_gqa_state("cpu", torch.float32, 64, 8, 2, False, True) is expect


def test_cpu_tensors_keep_enable_gqa():
    q = torch.zeros(1, 8, 4, 64)
    k = torch.zeros(1, 2, 4, 64)
    assert ad._sdpa_takes_gqa(q, k, True, False) is ad.SDPA_HAS_GQA


def _dense(monkeypatch, dtype, mask):
    seen = []
    real = ad.scaled_dot_product_attention

    def spy(Q, K, V, **kwargs):
        seen.append((K.shape[1], kwargs.get("enable_gqa", False)))
        return real(Q, K, V, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", spy)
    bsz, n_heads, n_kv, T, D = 2, 8, 2, 32, 64
    g = torch.Generator(device = "cuda").manual_seed(0)
    Q = torch.randn(bsz, n_heads, T, D, device = "cuda", generator = g).to(dtype)
    K = torch.randn(bsz, n_kv, T, D, device = "cuda", generator = g).to(dtype)
    V = torch.randn(bsz, n_kv, T, D, device = "cuda", generator = g).to(dtype)
    attention_mask = None
    if mask:
        attention_mask = torch.ones(bsz, T, dtype = torch.long, device = "cuda")
        attention_mask[1, :5] = 0
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = n_kv, n_groups = n_heads // n_kv)
    context = ad.AttentionContext(
        bsz = bsz,
        q_len = T,
        kv_seq_len = T,
        n_heads = n_heads,
        head_dim = D,
        requires_grad = True,
        seq_info = None,
        attention_mask = attention_mask,
        causal_mask = None,
        is_causal = True,
        sliding_window = None,
    )
    out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    return out, seen, (Q, K, V, attention_mask)


@needs_cuda
def test_fp32_masked_gqa_expands_instead_of_math(monkeypatch):
    # fp32 (DoRA) with a padding mask: flash cannot run, the memory-efficient kernel rejects GQA.
    out, seen, (Q, K, V, mask) = _dense(monkeypatch, torch.float32, True)
    assert seen == [(8, False)]
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K.repeat_interleave(4, dim = 1),
            V.repeat_interleave(4, dim = 1),
            attn_mask = None,
            is_causal = True,
        )
    # Row 0 has no padding: identical to plain causal attention.
    torch.testing.assert_close(out[0], ref.transpose(1, 2)[0], atol = 1e-4, rtol = 1e-4)


@needs_cuda
@pytest.mark.skipif(
    has_real_cuda() and torch.cuda.get_device_capability()[0] < 8, reason = "flash needs sm80+"
)
def test_bf16_causal_gqa_keeps_flash_gqa(monkeypatch):
    _, seen, _ = _dense(monkeypatch, torch.bfloat16, False)
    assert seen == [(2, True)]


@pytest.mark.parametrize(
    "capability, expect_varlen",
    [((10, 0), False), ((10, 3), False), ((9, 0), True), ((12, 0), True), ((8, 0), True)],
)
def test_cudnn_over_varlen_only_on_the_measured_arch(monkeypatch, capability, expect_varlen):
    # Mean segment length 1767: cuDNN per segment won on B200 but is 2.2x+ slower on H100.
    monkeypatch.setattr(ad, "_sdpa_backend_state", lambda *a: (True, CUDNN))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    lengths = (1500, 2000, 1800)
    Q = torch.zeros(1, 4, sum(lengths), 64, dtype = torch.bfloat16)
    assert ad._torch_varlen_beats_segments(Q, Q, lengths, 1, True) is expect_varlen
    # Short documents still go to varlen everywhere.
    short = (300, 700, 24)
    Q = torch.zeros(1, 4, sum(short), 64, dtype = torch.bfloat16)
    assert ad._torch_varlen_beats_segments(Q, Q, short, 1, True) is True


@pytest.mark.parametrize(
    "capabilities, varlen, hip, expect",
    [
        (((8, 0),), True, None, True),
        (((9, 0), (9, 0)), True, None, True),
        (((12, 0),), True, None, True),
        (((7, 5),), True, None, False),  # T4: xformers' block-diagonal kernel is faster
        (((8, 6), (7, 5)), True, None, False),  # mixed host keeps xformers for the old card
        (((8, 0),), False, None, False),  # torch 2.10 / 2.11: no varlen kernel
        (((9, 0),), True, "7.0", False),  # ROCm unchanged
        ((), True, None, False),
    ],
)
def test_native_beats_xformers(monkeypatch, capabilities, varlen, hip, expect):
    monkeypatch.setattr(torch.version, "hip", hip)
    assert ad._native_beats_xformers(capabilities, varlen) is expect


@pytest.mark.parametrize("use_varlen", [False, True])
def test_select_prefers_native_over_xformers(monkeypatch, use_varlen):
    monkeypatch.setattr(ad, "_ring_attention_active", lambda: False)
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(ad, "HAS_XFORMERS", True)
    monkeypatch.setattr(ad, "_NATIVE_BEATS_XFORMERS", True)
    assert ad.select_attention_backend(use_varlen) == ad.SDPA
    monkeypatch.setattr(ad, "_NATIVE_BEATS_XFORMERS", False)
    assert ad.select_attention_backend(use_varlen) == ad.XFORMERS
    # FlashAttention installed: unchanged.
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(ad, "_NATIVE_BEATS_XFORMERS", True)
    assert ad.select_attention_backend(use_varlen) == (
        ad.FLASH_VARLEN if use_varlen else ad.FLASH_DENSE
    )


@needs_cuda
def test_windowed_packed_runs_expand_where_gqa_would_force_math(monkeypatch):
    # fp32 packed segments with a sliding window: the windowed run carries a mask.
    seen = []
    real = ad.scaled_dot_product_attention

    def spy(Q, K, V, **kwargs):
        seen.append((K.shape[1], kwargs.get("enable_gqa", False), "attn_mask" in kwargs))
        return real(Q, K, V, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", spy)
    packing.clear_packed_caches()
    lengths = (12, 3)
    T = sum(lengths)
    Q = torch.randn(1, 4, T, 64, device = "cuda")
    K = torch.randn(1, 2, T, 64, device = "cuda")
    out = ad._sdpa_packed_segments(
        Q, K, K, lengths, n_groups = 2, is_causal = True, sliding_window = 4, sdpa_kwargs = {}
    )
    assert out is not None
    assert all(not gqa and heads == 4 for heads, gqa, _ in seen if _), seen

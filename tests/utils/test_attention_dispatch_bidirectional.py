# SPDX-License-Identifier: AGPL-3.0-only
"""Bidirectional packed attention must preserve sentence boundaries in every fallback."""

from dataclasses import replace

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import packing


def _context(lengths = (2, 3), **kwargs):
    lengths = torch.tensor(lengths, dtype = torch.int32)
    total = int(lengths.sum())
    cu = torch.cat((torch.zeros(1, dtype = torch.int32), lengths.cumsum(0).int()))
    return ad.AttentionContext(
        bsz = 1,
        q_len = total,
        kv_seq_len = total,
        n_heads = 2,
        head_dim = 4,
        requires_grad = True,
        seq_info = (lengths, cu, int(lengths.max())),
        attention_mask = None,
        causal_mask = None,
        is_causal = False,
        **kwargs,
    )


def _qkv():
    generator = torch.Generator().manual_seed(4460)
    return tuple(torch.randn(1, 2, 5, 4, generator = generator).requires_grad_() for _ in range(3))


def _run(
    context,
    qkv,
    backend = ad.SDPA,
    **kwargs,
):
    config = ad.AttentionConfig(backend = backend, n_kv_heads = 2, n_groups = 1, **kwargs)
    return ad.run_attention(config = config, context = context, Q = qkv[0], K = qkv[1], V = qkv[2])


@pytest.mark.parametrize("backend", [ad.SDPA, ad.FLASH_VARLEN, ad.XFORMERS])
def test_packed_bidirectional_outputs_and_gradients_match_independent_rows(monkeypatch, backend):
    # Force the int32-overflow SDPA fallback so CPU SDPA runs for real.
    if backend != ad.SDPA:
        monkeypatch.setattr(ad, "_VARLEN_INT32_GUARD_DISABLED", False)
        monkeypatch.setattr(ad, "_varlen_backward_overflows_int32", lambda *args: True)
        monkeypatch.setattr(ad, "_warn_varlen_int32_overflow_once", lambda *args: None)

        def unexpected_kernel(*args, **kwargs):
            pytest.fail("overflow must bypass the accelerator kernel")

        monkeypatch.setattr(ad, "flash_attn_varlen_func", unexpected_kernel)
        monkeypatch.setattr(ad, "xformers_attention", unexpected_kernel, raising = False)

    qkv = _qkv()
    output = _run(_context(), qkv, backend)
    reference = torch.cat(
        [
            scaled_dot_product_attention(*(tensor[:, :, start:end] for tensor in qkv))
            for start, end in ((0, 2), (2, 5))
        ],
        dim = 2,
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)
    actual_grads = torch.autograd.grad(output.square().sum(), qkv, retain_graph = True)
    reference_grads = torch.autograd.grad(reference.square().sum(), qkv)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected)


def test_future_tokens_affect_their_sentence_but_not_other_sentences():
    qkv = tuple(torch.zeros_like(tensor) for tensor in _qkv())
    output = _run(_context(), qkv)
    changed = list(qkv)
    changed[2] = changed[2].clone()
    changed[2][:, :, 1] = 6
    modified = _run(_context(), changed)
    torch.testing.assert_close(modified[:, 0], torch.full_like(modified[:, 0], 3))
    torch.testing.assert_close(modified[:, 2:], output[:, 2:])


def test_dense_bidirectional_sdpa_matches_reference():
    qkv = _qkv()
    output = _run(replace(_context(), seq_info = None), qkv)
    torch.testing.assert_close(output, scaled_dot_product_attention(*qkv).transpose(1, 2))


def test_default_context_preserves_causal_sdpa():
    qkv = _qkv()
    context = replace(_context(), seq_info = None, is_causal = True)
    torch.testing.assert_close(
        _run(context, qkv),
        scaled_dot_product_attention(*qkv, is_causal = True).transpose(1, 2),
    )
    assert ad.AttentionContext.__dataclass_fields__["is_causal"].default is True


@pytest.mark.parametrize("backend", [ad.FLASH_DENSE, ad.FLASH_VARLEN])
def test_flash_receives_noncausal_without_mutating_config(monkeypatch, backend):
    captured = []

    def fake_kernel(query, key, value, *args, **kwargs):
        captured.append(kwargs["causal"])
        return torch.zeros_like(query)

    monkeypatch.setattr(ad, "flash_attn_func", fake_kernel)
    monkeypatch.setattr(ad, "flash_attn_varlen_func", fake_kernel)
    kwargs = {"causal": True}
    _run(_context(), _qkv(), backend, flash_dense_kwargs = kwargs, flash_varlen_kwargs = kwargs)
    assert captured == [False]
    assert kwargs == {"causal": True}


def test_sdpa_mask_cache_distinguishes_direction():
    seq_info = _context().seq_info
    options = dict(dtype = torch.float32, device = torch.device("cpu"))
    causal = packing.build_sdpa_packed_attention_mask(seq_info, **options)
    bidirectional = packing.build_sdpa_packed_attention_mask(seq_info, is_causal = False, **options)
    assert torch.isneginf(causal[0, 0, 0, 1])
    assert bidirectional[0, 0, 0, 1] == 0
    assert torch.isneginf(bidirectional[0, 0, 0, 2])
    torch.testing.assert_close(
        packing.build_sdpa_packed_attention_mask(seq_info, **options), causal
    )


def test_xformers_uses_bidirectional_blocks_and_separate_cache(monkeypatch):
    class Block:
        @classmethod
        def from_seqlens(cls, lengths):
            return cls()

    class CausalBlock(Block):
        pass

    monkeypatch.setattr(packing, "_XFormersBlockMask", CausalBlock)
    monkeypatch.setattr(packing, "_XFormersBidirectionalMask", Block)
    monkeypatch.setattr(packing, "_XFORMERS_MASK_CACHE", packing.OrderedDict())
    monkeypatch.setattr(packing, "_XFORMERS_BLOCK_MASK_CACHE", {})
    seq_info = _context().seq_info
    causal = packing.build_xformers_block_causal_mask(seq_info)
    bidirectional = packing.build_xformers_block_causal_mask(seq_info, is_causal = False)
    assert type(causal) is CausalBlock
    assert type(bidirectional) is Block
    assert packing.build_xformers_block_causal_mask(seq_info) is causal

    def fake_kernel(
        query,
        key,
        value,
        attn_bias = None,
        **kwargs,
    ):
        assert type(attn_bias) is Block
        return torch.zeros_like(query)

    monkeypatch.setattr(ad, "xformers_attention", fake_kernel, raising = False)
    _run(_context(), _qkv(), ad.XFORMERS)

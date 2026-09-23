# SPDX-License-Identifier: Apache-2.0
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
    # For Flash/xFormers, force the shared overflow rescue and execute real CPU SDPA.
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


@pytest.mark.parametrize("mask_kind", ["none", "padding", "extended"])
def test_dense_bidirectional_sdpa_matches_reference(mask_kind):
    qkv = _qkv()
    key_keep = torch.tensor([[True, True, False, False, False]])
    mask = None
    reference_mask = None
    if mask_kind == "padding":
        mask = key_keep
        reference_mask = key_keep[:, None, None, :]
    elif mask_kind == "extended":
        mask = torch.zeros(1, 1, 1, 5).masked_fill(~key_keep[:, None, None, :], float("-inf"))
        reference_mask = mask
    context = replace(_context(), seq_info = None, attention_mask = mask)
    # Explicit false on the call must also defeat stale causal backend kwargs.
    output = _run(context, qkv, sdpa_kwargs = {"is_causal": True})
    reference = scaled_dot_product_attention(*qkv, attn_mask = reference_mask).transpose(1, 2)
    torch.testing.assert_close(output, reference)


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


def test_xformers_causal_block_survives_missing_bidirectional_class(monkeypatch):
    class CausalBlock:
        @classmethod
        def from_seqlens(cls, lengths):
            return cls()

    monkeypatch.setattr(packing, "_XFormersBlockMask", CausalBlock)
    monkeypatch.setattr(packing, "_XFormersBidirectionalMask", None)
    monkeypatch.setattr(packing, "_XFORMERS_MASK_CACHE", packing.OrderedDict())
    monkeypatch.setattr(packing, "_XFORMERS_BLOCK_MASK_CACHE", {})
    seq_info = _context().seq_info
    assert type(packing.build_xformers_block_causal_mask(seq_info)) is CausalBlock
    assert packing.build_xformers_block_causal_mask(seq_info, is_causal = False) is None


def test_xformers_without_bidirectional_block_falls_back_to_masked_sdpa(monkeypatch):
    monkeypatch.setattr(ad, "HAS_XFORMERS", True)
    monkeypatch.setattr(ad, "_XFormersBidirectionalMask", None)
    monkeypatch.setattr(
        ad,
        "xformers_attention",
        lambda *args, **kwargs: pytest.fail("unmasked xFormers ran"),
        raising = False,
    )
    qkv = _qkv()
    output = _run(_context(), qkv, ad.XFORMERS)
    reference = torch.cat(
        [
            scaled_dot_product_attention(*(tensor[:, :, start:end] for tensor in qkv))
            for start, end in ((0, 2), (2, 5))
        ],
        dim = 2,
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)


def test_xformers_missing_block_does_not_drop_softcap(monkeypatch):
    monkeypatch.setattr(ad, "HAS_XFORMERS", True)
    monkeypatch.setattr(ad, "_XFormersBidirectionalMask", None)
    with pytest.raises(RuntimeError, match = "cannot preserve softcap=50.0"):
        _run(
            _context(),
            _qkv(),
            ad.XFORMERS,
            flash_varlen_kwargs = {"softcap": 50.0},
        )


@pytest.mark.parametrize("backend", [ad.SDPA, ad.FLASH_VARLEN, ad.XFORMERS])
def test_bidirectional_window_rejected_before_dispatch(backend):
    with pytest.raises(ValueError, match = "Bidirectional.*sliding_window"):
        _run(_context(sliding_window = 3), _qkv(), backend)


def test_bidirectional_packing_helpers_reject_sliding_window():
    seq_info = _context().seq_info
    with pytest.raises(ValueError, match = "Bidirectional.*sliding_window"):
        packing.build_sdpa_packed_attention_mask(
            seq_info,
            dtype = torch.float32,
            device = torch.device("cpu"),
            sliding_window = 3,
            is_causal = False,
        )
    with pytest.raises(ValueError, match = "Bidirectional.*sliding_window"):
        packing.build_xformers_block_causal_mask(seq_info, sliding_window = 3, is_causal = False)

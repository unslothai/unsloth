# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One unpadded Qwen3.5 row must not reach flash attention as three packed rows (transformers#44910)."""

import pytest

torch = pytest.importorskip("torch")
fa_utils = pytest.importorskip("transformers.modeling_flash_attention_utils")

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

from unsloth.import_fixes import (  # noqa: E402
    _mrope_position_ids_read_as_packed,
    fix_transformers_flash_attention_mrope_packed_sequence,
)


def _mrope(length, batch = 1):
    return torch.arange(length).view(1, 1, length).expand(3, batch, length)


@pytest.fixture
def patched():
    before = fa_utils._is_packed_sequence
    fix_transformers_flash_attention_mrope_packed_sequence()
    yield fa_utils._is_packed_sequence
    fa_utils._is_packed_sequence = before


def test_mrope_ids_are_never_packed(patched):
    for length in (1, 4, 304):
        assert not patched(_mrope(length), batch_size = 1)


def test_two_dim_packing_is_unchanged(patched):
    original = getattr(patched, "__wrapped__", patched)
    packed = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    plain = torch.arange(7).view(1, 7)
    for ids, batch in ((packed, 1), (plain, 1), (packed.expand(2, 7), 2), (None, 1)):
        assert bool(patched(ids, batch_size = batch)) == bool(original(ids, batch_size = batch))
    assert patched(packed, batch_size = 1)


def test_probe_reads_the_original(patched):
    original = getattr(patched, "__wrapped__", patched)
    assert _mrope_position_ids_read_as_packed(original) == bool(original(_mrope(4), batch_size = 1))
    assert not _mrope_position_ids_read_as_packed(patched)


def test_idempotent(patched):
    fix_transformers_flash_attention_mrope_packed_sequence()
    assert fa_utils._is_packed_sequence is patched


@pytest.mark.skipif(not has_real_cuda(), reason = "flash-attn needs CUDA")
def test_flash_attention_forward_keeps_one_row_unpacked(patched, monkeypatch):
    pytest.importorskip("flash_attn")
    calls = []
    real = fa_utils._prepare_from_posids
    monkeypatch.setattr(
        fa_utils,
        "_prepare_from_posids",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    length, heads, dim = 304, 4, 256
    q = torch.randn(1, length, heads, dim, device = "cuda", dtype = torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = fa_utils._flash_attention_forward(
        q,
        k,
        v,
        attention_mask = None,
        query_length = length,
        is_causal = True,
        position_ids = _mrope(length).cuda(),
        attn_implementation = "flash_attention_2",
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal = True
    ).transpose(1, 2)
    assert calls == []
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol = 2e-2, rtol = 2e-2)

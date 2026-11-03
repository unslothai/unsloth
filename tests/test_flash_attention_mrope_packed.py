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


fa_integration = pytest.importorskip("transformers.integrations.flash_attention")

from unsloth.import_fixes import (  # noqa: E402
    _packed_row_varlen_kwargs,
    fix_transformers_flash_attention_mrope_packed_rows,
)


@pytest.fixture
def packed_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fa_integration,
        "_flash_attention_forward",
        lambda q, k, v, mask, *a, **kw: calls.append(kw) or q,
    )
    fix_transformers_flash_attention_mrope_packed_rows()
    return fa_integration._flash_attention_forward, calls


def _query(length):
    return torch.zeros(1, length, 2, 8)


def test_packed_row_with_mrope_ids_gets_its_document_boundaries(packed_rows):
    forward, calls = packed_rows
    lengths = torch.tensor([3, 5], dtype = torch.int32)
    q = _query(8)
    forward(q, q, q, None, position_ids = _mrope(8), packed_seq_lengths = lengths)
    assert calls[-1]["cu_seq_lens_q"].tolist() == [0, 3, 8]
    assert calls[-1]["cu_seq_lens_k"].tolist() == [0, 3, 8]
    assert calls[-1]["max_length_q"] == calls[-1]["max_length_k"] == 5


def test_every_other_call_is_passed_through_unchanged(packed_rows):
    forward, calls = packed_rows
    lengths = torch.tensor([3, 5], dtype = torch.int32)
    q = _query(8)
    flat = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    cases = [
        (q, None, {"position_ids": _mrope(8)}),  # not an Unsloth packed row
        (
            q,
            None,
            {"position_ids": flat, "packed_seq_lengths": lengths},
        ),  # 2D ids already mark starts
        (q, torch.ones(1, 8), {"position_ids": _mrope(8), "packed_seq_lengths": lengths}),
        (
            torch.zeros(2, 8, 2, 8),
            None,
            {"position_ids": _mrope(8, 2), "packed_seq_lengths": lengths},
        ),
        (
            _query(7),
            None,
            {"position_ids": _mrope(7), "packed_seq_lengths": lengths},
        ),  # lengths run past the row
        (
            q,
            None,
            {"position_ids": _mrope(8), "packed_seq_lengths": lengths, "cu_seq_lens_q": "caller"},
        ),
    ]
    for query, mask, kwargs in cases:
        forward(query, query, query, mask, **dict(kwargs))
        assert calls[-1] == kwargs


def test_padded_tail_gets_its_own_segment(packed_rows):
    # TRL's padding-free collator pads the flattened row to pad_to_multiple_of after the lengths
    # are taken: the tail must not fall back to causal attention over every document.
    forward, calls = packed_rows
    lengths = torch.tensor([3, 5], dtype = torch.int32)
    q = _query(16)
    forward(q, q, q, None, position_ids = _mrope(16), packed_seq_lengths = lengths)
    assert calls[-1]["cu_seq_lens_q"].tolist() == [0, 3, 8, 16]
    assert calls[-1]["cu_seq_lens_k"].tolist() == [0, 3, 8, 16]
    assert calls[-1]["max_length_q"] == calls[-1]["max_length_k"] == 8


def _compiled_boundaries(packed_rows, cases, warm):
    from torch._dynamo.testing import CompileCounter
    from unsloth.utils import packing

    _, calls = packed_rows

    def step(q, lengths):
        # Read the module attribute at call time, as transformers' flash_attention_forward does.
        return fa_integration._flash_attention_forward(
            q, q, q, None, position_ids = _mrope(q.shape[1]), packed_seq_lengths = lengths
        )

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(step, backend = counter, dynamic = True)
    seen = []
    for length, docs in cases:
        lengths = torch.tensor(docs, dtype = torch.int32)
        packing._PACKED_INFO_CACHE.clear()
        if warm:  # the hybrid varlen outer forward fills the cache before the layers run
            packing.get_packed_info_from_kwargs(
                {"packed_seq_lengths": lengths}, torch.device("cpu")
            )
        compiled(_query(length), lengths)
        kw = calls[-1]
        seen.append(
            None
            if "cu_seq_lens_q" not in kw
            else (kw["cu_seq_lens_q"].tolist(), int(kw["max_length_q"]))
        )
    torch._dynamo.reset()
    return seen, counter.frame_count


@pytest.mark.parametrize("warm", [False, True])
def test_compiled_calls_match_eager_boundaries(packed_rows, warm):
    cases = [(8, [3, 5]), (16, [3, 5]), (7, [3, 5])]
    seen, _ = _compiled_boundaries(packed_rows, cases, warm)
    assert seen == [([0, 3, 8], 5), ([0, 3, 8, 16], 8), None]


@pytest.mark.parametrize("warm", [False, True])
def test_compiled_calls_do_not_recompile_per_document_length(packed_rows, warm):
    cases = [(16, [16 - longest, longest]) for longest in range(8, 16)]
    seen, frames = _compiled_boundaries(packed_rows, cases, warm)
    assert [m for _, m in seen] == list(range(8, 16))
    assert frames <= 4, frames


def test_packed_rows_patch_is_idempotent(packed_rows):
    forward, _ = packed_rows
    fix_transformers_flash_attention_mrope_packed_rows()
    assert fa_integration._flash_attention_forward is forward


def test_helper_ignores_rows_without_packed_lengths():
    assert _packed_row_varlen_kwargs(_query(4), None, {"position_ids": _mrope(4)}) is None


@pytest.mark.skipif(not has_real_cuda(), reason = "flash-attn needs CUDA")
def test_packed_documents_do_not_see_each_other_under_flash_attention():
    pytest.importorskip("flash_attn")
    from transformers.integrations.flash_attention import flash_attention_forward

    fix_transformers_flash_attention_mrope_packed_sequence()
    fix_transformers_flash_attention_mrope_packed_rows()

    class _Attention(torch.nn.Module):
        is_causal = True
        layer_idx = 0

        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"_attn_implementation": "flash_attention_2"})()

    torch.manual_seed(0)
    first, second, heads, dim = 48, 32, 4, 64
    total = first + second
    q, k, v = (
        torch.randn(1, heads, total, dim, device = "cuda", dtype = torch.bfloat16) for _ in range(3)
    )
    positions = torch.cat([torch.arange(first), torch.arange(second)]).cuda()
    out, _ = flash_attention_forward(
        _Attention(),
        q,
        k,
        v,
        None,
        position_ids = positions.view(1, 1, total).expand(3, 1, total),
        packed_seq_lengths = torch.tensor([first, second], dtype = torch.int32),
    )

    def alone(s, e):
        return torch.nn.functional.scaled_dot_product_attention(
            q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], is_causal = True
        ).transpose(1, 2)

    torch.testing.assert_close(
        out, torch.cat([alone(0, first), alone(first, total)], dim = 1), atol = 2e-2, rtol = 2e-2
    )

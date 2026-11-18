# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed metadata built eagerly (get_packed_info_from_kwargs) and read back from its cache inside a
compiled frame (Ulysses context parallel attention in a compiled block): the document count changes every
packed batch, and the cached lengths / cu_seqlens were unmarked, so the frame recompiled on the next count
(FastModel Qwen3, cp = 2, Kaggle 2x T4: "_PACKED_INFO_CACHE[...]['result'][1]' size mismatch at index 0")."""

import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: E402,F401
from unsloth.utils import packing  # noqa: E402


def test_cached_packed_info_read_by_a_compiled_caller_compiles_once():
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    torch._dynamo.reset()
    packing._PACKED_INFO_CACHE.clear()

    def attention(x, packed_seq_lengths):
        _, cu_seqlens, _ = packing.get_packed_info_from_kwargs(
            {"packed_seq_lengths": packed_seq_lengths}, x.device
        )
        return x * cu_seqlens[-1].to(x.dtype) + cu_seqlens.sum().to(x.dtype)

    compiled = torch.compile(attention, backend = counter)
    try:
        for lengths in ([3, 5], [2, 2, 4], [1, 1, 3, 3], [4, 4, 4, 4, 4]):
            psl = torch.tensor(lengths, dtype = torch.int32)
            # Eager first, as the packed attention / mask code builds it before the compiled call.
            packing.get_packed_info_from_kwargs({"packed_seq_lengths": psl}, torch.device("cpu"))
            x = torch.randn(sum(lengths), 4)
            # As the block compiler marks its inputs (_mark_seq_dynamic / _mark_packed_dynamic).
            torch._dynamo.maybe_mark_dynamic(x, 0)
            torch._dynamo.maybe_mark_dynamic(psl, 0)
            total = sum(lengths)
            torch.testing.assert_close(
                compiled(x, psl),
                x * total + sum(sum(lengths[: i + 1]) for i in range(len(lengths))),
            )
    finally:
        torch._dynamo.reset()
        packing._PACKED_INFO_CACHE.clear()
    assert counter.frame_count == 1, f"{counter.frame_count} frames for 4 document counts"

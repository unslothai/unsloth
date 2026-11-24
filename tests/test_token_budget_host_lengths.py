# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TOKEN_BUDGET chunks of a packed batch keep the host packed lengths (meta stand-in raises on a device read)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_budget_host_lengths", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)

LENGTHS = (5, 9, 3, 12, 7, 4)


def _packed_batch(device):
    total = sum(LENGTHS)
    lengths = torch.tensor(LENGTHS, dtype = torch.int32, device = device)
    # What packing._carry_host_lengths attaches when the batch moves to the device.
    lengths._unsloth_lengths_tuple = LENGTHS
    return {
        "input_ids": torch.ones(1, total, dtype = torch.long, device = device),
        "position_ids": torch.zeros(1, total, dtype = torch.long, device = device),
        "packed_seq_lengths": lengths,
    }


def test_split_reads_and_hands_on_the_host_lengths():
    # On meta a device read raises ("Cannot copy out of meta tensor").
    chunks = tb.split_batch_by_token_budget(_packed_batch("meta"), 16)
    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        lengths = chunk["packed_seq_lengths"]
        host = getattr(lengths, "_unsloth_lengths_tuple", None)
        assert isinstance(host, tuple) and len(host) == lengths.numel()
        assert sum(host) == chunk["input_ids"].shape[1]
        seen.extend(host)
    assert tuple(seen) == LENGTHS


def test_chunk_lengths_reach_the_forward_without_a_device_read():
    from unsloth.utils import packing

    for chunk in tb.split_batch_by_token_budget(_packed_batch("meta"), 16):
        lengths = chunk["packed_seq_lengths"]
        packing._PACKED_INFO_CACHE.clear()
        packing._PACKED_HOST_LENGTHS_CACHE.clear()
        _, cu_seqlens, max_seqlen = packing.get_packed_info_from_kwargs(
            {"packed_seq_lengths": lengths}, torch.device("meta")
        )
        assert max_seqlen == max(lengths._unsloth_lengths_tuple)
        assert cu_seqlens.shape == (lengths.numel() + 1,)
    packing._PACKED_INFO_CACHE.clear()
    packing._PACKED_HOST_LENGTHS_CACHE.clear()


def test_split_without_host_lengths_matches_the_device_values():
    batch = _packed_batch("cpu")
    batch["packed_seq_lengths"] = torch.tensor(LENGTHS, dtype = torch.int32)
    chunks = tb.split_batch_by_token_budget(batch, 16)
    assert [c["packed_seq_lengths"].tolist() for c in chunks] == [
        list(c["packed_seq_lengths"]._unsloth_lengths_tuple) for c in chunks
    ]
    assert sum((c["packed_seq_lengths"].tolist() for c in chunks), []) == list(LENGTHS)

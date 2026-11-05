# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""RoPE launch constexprs via meta tensors + a launch stub: no device, so CPU-only runners collect it."""

import pytest
import torch

pytest.importorskip("triton")

import unsloth  # noqa: F401
from unsloth.kernels import rope_embedding as RE


def test_a_cos_table_past_int32_offsets_takes_64_bit_indexing():
    def launched(rows):
        seen = {}

        class _Grid:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: seen.update(kwargs)

        cos = torch.empty(rows, 128, device = "meta", dtype = torch.float16)
        q = torch.empty(2, 8, 1, 128, device = "meta", dtype = torch.float16)
        k = torch.empty(2, 2, 1, 128, device = "meta", dtype = torch.float16)
        RE._rope_launch(q, q, k, k, cos, cos, None, False, False, False, lambda fn: _Grid())
        return seen["LONG_INDEXING"]

    assert launched(2**31 // 128 + 4)
    assert not launched(4096)

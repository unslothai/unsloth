# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture
def packing():
    path = Path(__file__).resolve().parents[2] / "unsloth" / "utils" / "packing.py"
    spec = importlib.util.spec_from_file_location("packing_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_metadata_tracks_inplace_lengths(packing, inference, dtype):
    with torch.inference_mode(inference):
        lengths = torch.tensor([2, 2], dtype = dtype)
        packing.get_packed_info_from_kwargs({"packed_seq_lengths": lengths}, torch.device("cpu"))
        lengths.copy_(torch.tensor([1, 3], dtype = dtype))
        current, cumulative, maximum = packing.get_packed_info_from_kwargs(
            {"packed_seq_lengths": lengths}, torch.device("cpu")
        )
        assert current.tolist() == [1, 3]
        assert cumulative.tolist() == [0, 1, 4]
        assert maximum == 3


@pytest.mark.parametrize("inference", [False, True])
def test_sdpa_matches_separate_sequences_after_length_update(packing, inference):
    with torch.inference_mode(inference):
        lengths = torch.tensor([2, 2], dtype = torch.int32)
        q = torch.zeros(1, 1, 4, 2)
        k = torch.zeros_like(q)
        v = torch.arange(8, dtype = torch.float32).reshape(1, 1, 4, 2)
        info = (lengths, torch.tensor([0, 2, 4]), 2)
        packing.build_sdpa_packed_attention_mask(info, dtype = q.dtype, device = q.device)
        lengths.copy_(torch.tensor([1, 3]))
        mask = packing.build_sdpa_packed_attention_mask(info, dtype = q.dtype, device = q.device)
        actual = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        expected = torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[:, :, :1], k[:, :, :1], v[:, :, :1], is_causal = True
                ),
                F.scaled_dot_product_attention(
                    q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], is_causal = True
                ),
            ],
            dim = 2,
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("inference", [False, True])
def test_xformers_factory_receives_updated_lengths(packing, monkeypatch, inference):
    class BlockMask:
        @classmethod
        def from_seqlens(cls, lengths):
            return SimpleNamespace(lengths = tuple(lengths))

    monkeypatch.setattr(packing, "_XFormersBlockMask", BlockMask)
    with torch.inference_mode(inference):
        lengths = torch.tensor([2, 2], dtype = torch.int32)
        info = (lengths, torch.tensor([0, 2, 4]), 2)
        packing.build_xformers_block_causal_mask(info)
        lengths.copy_(torch.tensor([1, 3]))
        mask = packing.build_xformers_block_causal_mask(info)
        assert mask.lengths == (1, 3)

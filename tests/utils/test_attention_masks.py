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

"""Unit tests for packed-attention mask helpers with sliding-window logic."""

import math

import torch

from unsloth.utils import attention_dispatch
from unsloth.utils import packing as packing_utils


def _make_seq_info(lengths):
    lengths = torch.tensor(lengths, dtype = torch.int32)
    cu = torch.cat(
        [
            torch.zeros(1, dtype = torch.int32),
            torch.cumsum(lengths, dim = 0, dtype = torch.int32),
        ]
    )
    max_len = int(lengths.max().item())
    return lengths, cu, max_len


def test_sdpa_packed_attention_mask_sliding_window():
    seq_info = _make_seq_info([5, 3])
    mask = packing_utils.build_sdpa_packed_attention_mask(
        seq_info,
        dtype = torch.float32,
        device = torch.device("cpu"),
        sliding_window = 3,
    )

    assert mask.shape == (1, 1, 8, 8)

    block_first = mask[0, 0, :5, :5]
    upper = torch.triu(torch.ones_like(block_first), diagonal = 1).bool()
    assert torch.all(block_first[upper] == float("-inf"))
    assert block_first[3, 0].item() == float("-inf")
    assert block_first[4, 1].item() == float("-inf")
    assert block_first[4, 2].item() > -math.inf
    assert mask[0, 0, 0, 6].item() == float("-inf")


def test_xformers_block_mask_sliding_window(monkeypatch):
    class _FakeMask:
        def __init__(self, lengths, window = None):
            self.lengths = lengths
            self.window = window

        @classmethod
        def from_seqlens(cls, lengths):
            return cls(tuple(lengths))

        def make_local_attention(self, window_size):
            return _FakeMask(self.lengths, window = window_size)

    monkeypatch.setattr(packing_utils, "_XFormersBlockMask", _FakeMask, raising = False)

    seq_info = _make_seq_info([4, 4])
    mask = packing_utils.build_xformers_block_causal_mask(
        seq_info,
        sliding_window = 2,
    )

    assert isinstance(mask, _FakeMask)
    assert mask.window == 2


def test_run_attention_sdpa_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([3, 2])
    sliding_window = 2

    original_builder = attention_dispatch.build_sdpa_packed_attention_mask
    captured = {}

    def _capture_builder(seq_info_arg, *, dtype, device, sliding_window = None):
        captured["window"] = sliding_window
        return original_builder(
            seq_info_arg,
            dtype = dtype,
            device = device,
            sliding_window = sliding_window,
        )

    monkeypatch.setattr(
        attention_dispatch,
        "build_sdpa_packed_attention_mask",
        _capture_builder,
    )

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.SDPA,
        n_kv_heads = 1,
        n_groups = 1,
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 5,
        kv_seq_len = 5,
        n_heads = 1,
        head_dim = 1,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 5, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["window"] == sliding_window
    mask = captured["mask"]
    assert mask is not None and mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 4, 1].item() == float("-inf")


def test_run_attention_xformers_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3

    class _FakeBias:
        pass

    captured = {}

    def _fake_builder(seq_info_arg, *, sliding_window = None, base_mask = None):
        captured["window"] = sliding_window
        captured["base"] = base_mask
        return _FakeBias()

    def _fake_attention(Q, K, V, attn_bias = None, **_):
        captured["bias"] = attn_bias
        return torch.zeros_like(Q)

    monkeypatch.setattr(
        attention_dispatch, "build_xformers_block_causal_mask", _fake_builder
    )
    monkeypatch.setattr(
        attention_dispatch, "xformers_attention", _fake_attention, raising = False
    )
    monkeypatch.setattr(
        attention_dispatch, "XFORMERS_BLOCK_DIAG_CLS", _FakeBias, raising = False
    )

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.XFORMERS,
        n_kv_heads = 1,
        n_groups = 1,
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 1,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["window"] == sliding_window
    assert isinstance(captured["bias"], _FakeBias)


def test_run_attention_flash_varlen_receives_window_and_softcap(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3
    softcap = 0.5
    window_tuple = (sliding_window, sliding_window)

    captured = {}

    def _fake_flash_varlen(Q, K, V, cu_q, cu_k, max_q, max_k, **kwargs):
        captured["kwargs"] = kwargs
        return torch.zeros_like(Q)

    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_varlen_func",
        _fake_flash_varlen,
    )
    monkeypatch.setattr(attention_dispatch, "HAS_FLASH_ATTENTION", True)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.FLASH_VARLEN,
        n_kv_heads = 1,
        n_groups = 1,
        flash_varlen_kwargs = {
            "dropout_p": 0.0,
            "softmax_scale": 1.0,
            "causal": True,
            "softcap": softcap,
            "window_size": window_tuple,
        },
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 2,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 2)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["kwargs"]["softcap"] == softcap
    assert captured["kwargs"]["window_size"] == window_tuple


"""Unit tests for packed-attention mask helpers with sliding-window logic."""

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""sm120 (RTX PRO 6000, RTX 5090) routes the chunked CE forward to 8192-wide chunks at 4 warps; runs on CPU."""

import pytest
import torch

from unsloth.kernels import cross_entropy_loss as ce_mod
from unsloth.kernels.utils import MAX_FUSED_SIZE


def _route(capability, vocab, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    ce_mod._chunked_forward_config_at.cache_clear()
    try:
        return ce_mod._chunked_forward_config_at("cuda", 0, vocab)
    finally:
        ce_mod._chunked_forward_config_at.cache_clear()


@pytest.mark.parametrize("vocab", [65537, 128256, 151936, 201088, 256000, 262144])
def test_sm120_uses_8k_blocks_with_4_warps(vocab, monkeypatch):
    assert _route((12, 0), vocab, monkeypatch) == (8192, 4)


@pytest.mark.parametrize("capability", [(12, 1), (11, 0), (9, 0), (8, 9)])
def test_neighbouring_archs_keep_the_previous_config(capability, monkeypatch):
    assert _route(capability, 151936, monkeypatch) == (MAX_FUSED_SIZE, 32)


def test_sm120_entry_has_no_vocab_cap():
    assert ce_mod._CHUNKED_FORWARD_CONFIGS[(12, 0)] == (8192, 4, None)

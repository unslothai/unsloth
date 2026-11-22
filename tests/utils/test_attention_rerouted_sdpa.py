# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""SDPA standing in for xformers inside a compiled block keeps cuDNN attention off.

torch 2.13+ on SM100 picks cuDNN attention for plain causal SDPA, and cuDNN builds a new plan for
every new sequence length: padded training batches of varying length stalled every step (Llama-3.2-1B
LoRA on B200: 439 ms median step compiled vs 227 ms stock). xformers itself runs flash there.
"""

import pytest
import torch
import torch.nn.functional as F

ad = pytest.importorskip("unsloth.utils.attention_dispatch")


def test_rerouted_sdpa_disables_cudnn_only(monkeypatch):
    seen = []

    def fake_sdpa(q, k, v, **kwargs):
        b = torch.backends.cuda
        seen.append(
            (
                b.cudnn_sdp_enabled(),
                b.flash_sdp_enabled(),
                b.mem_efficient_sdp_enabled(),
                b.math_sdp_enabled(),
            )
        )
        return q

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    before = torch.backends.cuda.cudnn_sdp_enabled()
    q = torch.zeros(1, 2, 4, 8)
    ad._rerouted_sdpa(q, q, q, is_causal = True)
    ad.scaled_dot_product_attention(q, q, q, is_causal = True)
    assert seen[0] == (False, True, True, True)
    assert seen[1][0] == before  # the plain path is untouched
    assert torch.backends.cuda.cudnn_sdp_enabled() == before


def test_rerouted_sdpa_traces_without_a_break():
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()

    def f(q):
        return ad._rerouted_sdpa(q, q, q, is_causal = True)

    c = torch.compile(f, backend = "eager", dynamic = True)
    for L in (5, 9, 16, 23):
        q = torch.randn(1, 2, L, 8)
        torch.testing.assert_close(c(q), F.scaled_dot_product_attention(q, q, q, is_causal = True))
    assert dict(counters["graph_break"]) == {}
    torch._dynamo.reset()

# SPDX-License-Identifier: AGPL-3.0-only
"""An eval-mode forward that will backpropagate (LoRA-GA calibration) must not get the GQA layout that
xformers can only run forward."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad


def _run(monkeypatch, requires_grad, grad_enabled):
    seen = {}

    def fake_kernel(
        query,
        key,
        value,
        attn_bias = None,
        **kwargs,
    ):
        seen["ndim"] = query.dim()
        return torch.zeros_like(query)

    monkeypatch.setattr(ad, "xformers_attention", fake_kernel, raising = False)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 3,
        kv_seq_len = 3,
        n_heads = 4,
        head_dim = 8,
        requires_grad = requires_grad,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    config = ad.AttentionConfig(backend = ad.XFORMERS, n_kv_heads = 2, n_groups = 2)
    Q = torch.randn(1, 4, 3, 8, requires_grad = True)
    K = torch.randn(1, 2, 3, 8, requires_grad = True)
    V = torch.randn(1, 2, 3, 8, requires_grad = True)
    with torch.set_grad_enabled(grad_enabled):
        ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    return seen["ndim"]


@pytest.mark.parametrize(
    "requires_grad, grad_enabled, ndim",
    [(False, True, 4), (False, False, 5), (True, True, 4)],
)
def test_xformers_gqa_layout_follows_backward(monkeypatch, requires_grad, grad_enabled, ndim):
    assert _run(monkeypatch, requires_grad, grad_enabled) == ndim

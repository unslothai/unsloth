# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cohere q_norm / k_norm weights are (heads, head_dim), so they act before the head transpose."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

transformers_cohere = pytest.importorskip("transformers.models.cohere.modeling_cohere")

ROOT = Path(__file__).resolve().parents[1] / "unsloth" / "models"
COHERE = ROOT / "cohere.py"
LLAMA = ROOT / "llama.py"

HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
HIDDEN = HEADS * HEAD_DIM


def _functions(path, names):
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    return nodes


def _load(attention_stubs):
    cohere_names = (
        "fast_layernorm_inference",
        "CohereAttention_fast_forward",
        "CohereAttention_fast_forward_inference",
    )
    nodes = _functions(COHERE, cohere_names) + _functions(LLAMA, ("fast_layernorm_compiled",))
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
        "DEVICE_TYPE_TORCH": "cpu",
        "KV_CACHE_INCREMENT": 256,
        "original_apply_qkv": lambda self, h: (self.q_proj(h), self.k_proj(h), self.v_proj(h)),
        "original_apply_o": lambda self, x: self.o_proj(x),
        "get_cp_manager": lambda: None,
        "get_packed_info_from_kwargs": lambda kwargs, device: None,
        "select_attention_backend": lambda use_varlen: "sdpa",
        "resolve_prefix_seg_info": lambda *args, **kwargs: None,
        "AttentionConfig": lambda **kwargs: kwargs,
        "AttentionContext": lambda **kwargs: kwargs,
        "fast_rope_embedding": lambda Q, K, cos, sin, position_ids: (Q, K),
        "fast_linear_forward": lambda proj, X, out = None: proj(X),
        "scaled_dot_product_attention": F.scaled_dot_product_attention,
        "torch_nn_functional_softmax": F.softmax,
        "math_sqrt": lambda x: x**0.5,
        **attention_stubs,
    }
    module = ast.Module(body = nodes, type_ignores = [])
    exec(compile(module, str(COHERE), "exec"), namespace)
    return namespace


def _attention():
    torch.manual_seed(0)
    cohere = transformers_cohere
    attn = types.SimpleNamespace(
        config = types.SimpleNamespace(
            num_attention_heads = HEADS,
            num_key_value_heads = KV_HEADS,
            hidden_size = HIDDEN,
            sliding_window = None,
        ),
        num_key_value_groups = HEADS // KV_HEADS,
        head_dim = HEAD_DIM,
        use_qk_norm = True,
        q_norm = cohere.CohereLayerNorm(hidden_size = (HEADS, HEAD_DIM), eps = 1e-5),
        k_norm = cohere.CohereLayerNorm(hidden_size = (KV_HEADS, HEAD_DIM), eps = 1e-5),
        q_proj = torch.nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias = False),
        k_proj = torch.nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias = False),
        v_proj = torch.nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias = False),
        o_proj = torch.nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias = False),
        training = False,
    )
    with torch.no_grad():
        # Distinct per-head weights, so a norm applied along the wrong axis cannot match.
        attn.q_norm.weight.copy_(torch.rand(HEADS, HEAD_DIM) + 0.5)
        attn.k_norm.weight.copy_(torch.rand(KV_HEADS, HEAD_DIM) + 0.5)
    return attn


# seq_len == HEADS is the case where the old layout broadcast silently instead of raising.
@pytest.mark.parametrize("seq_len", [HEADS, 5])
def test_training_forward_norms_per_head_over_head_dim(seq_len):
    attn = _attention()
    seen = {}

    def run_attention(config, context, Q, K, V):
        seen["Q"], seen["K"] = Q.clone(), K.clone()
        return torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3])

    namespace = _load({"run_attention": run_attention})
    cos, sin = torch.ones(seq_len, HEAD_DIM), torch.zeros(seq_len, HEAD_DIM)

    x = torch.randn(2, seq_len, HIDDEN)
    with torch.no_grad():
        namespace["CohereAttention_fast_forward"](
            attn, x, position_ids = None, position_embeddings = (cos, sin)
        )
        q = attn.q_norm(attn.q_proj(x).view(2, seq_len, HEADS, HEAD_DIM)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(2, seq_len, KV_HEADS, HEAD_DIM)).transpose(1, 2)

    torch.testing.assert_close(seen["Q"], q)
    torch.testing.assert_close(seen["K"], k)


def test_decode_forward_norms_per_head_over_head_dim():
    attn = _attention()
    seen = {}

    def sdpa(
        Q,
        K,
        V,
        attn_mask = None,
        is_causal = False,
    ):
        seen["Q"], seen["K"] = Q.clone(), K.clone()
        return torch.zeros_like(Q)

    namespace = _load({"scaled_dot_product_attention": sdpa})
    bsz, past = 2, 3
    attn.rotary_emb = types.SimpleNamespace(
        get_cached = lambda length, device_index: (
            torch.ones(length, HEAD_DIM),
            torch.zeros(length, HEAD_DIM),
        )
    )
    past_k = torch.randn(bsz, KV_HEADS, past, HEAD_DIM)
    past_v = torch.randn(bsz, KV_HEADS, past, HEAD_DIM)
    x = torch.randn(bsz, 1, HIDDEN)
    position_ids = torch.full((bsz, 1), past)

    with torch.no_grad():
        namespace["CohereAttention_fast_forward_inference"](
            attn, x, (past_k, past_v), position_ids, do_prefill = True
        )
        q = attn.q_norm(attn.q_proj(x).view(bsz, 1, HEADS, HEAD_DIM)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(bsz, 1, KV_HEADS, HEAD_DIM)).transpose(1, 2)

    torch.testing.assert_close(seen["Q"], q, rtol = 0, atol = 1e-6)
    # The new key sits after the cached ones; grouped heads are repeated for the SDPA call.
    groups = HEADS // KV_HEADS
    expected_k = k[:, :, None].expand(bsz, KV_HEADS, groups, 1, HEAD_DIM)
    torch.testing.assert_close(
        seen["K"][:, :, -1:],
        expected_k.reshape(bsz, HEADS, 1, HEAD_DIM),
        rtol = 0,
        atol = 1e-6,
    )

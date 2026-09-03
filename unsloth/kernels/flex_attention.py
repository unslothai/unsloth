# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from functools import lru_cache
from transformers.models.llama.modeling_llama import logger
import os

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1",
    "triton.cudagraphs": False,
}

# Flex Attention supported from torch 2.5 onwards only
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _flex_attention = torch.compile(_flex_attention, dynamic = True, options = torch_compile_options)
    HAS_FLEX_ATTENTION = False
except:
    HAS_FLEX_ATTENTION = False


if not HAS_FLEX_ATTENTION:
    # Logit softcapping
    @torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
    def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
        n_heads = self.config.num_attention_heads
        head_dim = self.head_dim
        n_kv_heads = self.config.num_key_value_heads
        n_groups = self.num_key_value_groups

        # Grouped query attention
        # Grouped query attention
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        K = K.reshape(bsz, n_heads, q_len, head_dim)
        V = V.reshape(bsz, n_heads, q_len, head_dim)

        # Gemma 9b should use 256, not hidden_size // num_attention_heads (224); 27b uses the derived value,
        # so default to the config. See google/gemma_pytorch commit 03e6575.
        s = self.config.query_pre_attn_scalar
        t = self.config.attn_logit_softcapping

        Q = Q * torch.tensor(s**-0.5, dtype = Q.dtype)
        A = torch.matmul(Q, K.transpose(2, 3))
        A = t * torch.tanh(A / t)
        A += causal_mask[:q_len, :q_len]
        # Much slower under torch compile than the masked_fill_ it replaces.
        A = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype)
        A = torch.matmul(A, V)
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads * head_dim)
        return A

    create_flex_attention_causal_mask = None
    create_flex_attention_sliding_window_mask = None
else:
    # See pytorch-labs/attention-gym examples/flex_attn.ipynb. BSD 3-Clause License Copyright (c) 2023,
    # Driss Guessous, Horace He et al.
    import functools, math

    def generate_tanh_softcap(t):
        def tanh_softcap(x, b, h, q_idx, kv_idx):
            return t * torch.tanh(x / t)

        return tanh_softcap

    def causal_masker(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @functools.lru_cache
    def sliding_window_masker(size = 4096):
        def sliding_window(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= size
            return causal_mask & window_mask

        return sliding_window

    @functools.lru_cache
    def create_block_mask(mask, n = 128):
        return _create_block_mask(
            mask,
            1,
            1,
            n,
            n,
            BLOCK_SIZE = 128,
            _compile = True,
        )

    def create_flex_attention_causal_mask(max_seq_length = 8192):
        causal_mask = create_block_mask(causal_masker, max_seq_length)
        return causal_mask

    def create_flex_attention_sliding_window_mask(max_seq_length = 8192, sliding_window = 4096):
        sliding_masker = sliding_window_masker(sliding_window)
        causal_mask = create_block_mask(sliding_masker, max_seq_length)
        return causal_mask

    @functools.lru_cache
    def flex_attention(s, t):
        scale = 1.0 / math.sqrt(s)
        score_mod = generate_tanh_softcap(t)
        return functools.partial(
            _flex_attention,
            score_mod = score_mod,
            scale = scale,
            enable_gqa = True,
        )

    def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
        n_heads = self.config.num_attention_heads
        head_dim = self.head_dim
        s = self.config.query_pre_attn_scalar
        t = self.config.attn_logit_softcapping
        fx = flex_attention(s, t)
        A = fx(query = Q, key = K, value = V, block_mask = causal_mask)
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads * head_dim)
        return A


torch_matmul = torch.matmul
torch_tanh = torch.tanh
torch_nn_functional_softmax = torch.nn.functional.softmax


def slow_inference_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
    n_heads = self.config.num_attention_heads
    head_dim = self.head_dim
    n_kv_heads = self.config.num_key_value_heads
    n_groups = self.num_key_value_groups

    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)

    # Gemma 9b should use 256, not hidden_size // num_attention_heads (224); 27b uses the derived value,
    # so default to the config. See google/gemma_pytorch commit 03e6575.
    s = self.config.query_pre_attn_scalar
    t = self.config.attn_logit_softcapping

    Q = Q * torch.tensor(s**-0.5, dtype = Q.dtype)
    A = torch_matmul(Q, K.transpose(2, 3))

    # Logit softcapping
    A /= t
    torch_tanh(A, out = A)
    A *= t
    A += causal_mask[:q_len, :q_len]
    # Much slower under torch compile than the masked_fill_ it replaces.
    A = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype)
    A = torch_matmul(A, V)
    A = A.transpose(1, 2).contiguous()
    A = A.reshape(bsz, q_len, n_heads * head_dim)
    return A

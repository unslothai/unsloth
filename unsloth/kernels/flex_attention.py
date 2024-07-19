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

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : False, # Output Triton kernel outputs!
    "triton.cudagraphs" : False,
}

# Flex Attention supported from torch 2.5 onwards only
import torch.nn
if hasattr(torch.nn, "attention"):
    import torch.nn.attention
    if hasattr(torch.nn.attention, "flex_attention"):
        import torch.nn.attention.flex_attention
        from torch.nn.attention.flex_attention import flex_attention
        from torch.nn.attention.flex_attention import create_block_mask
        FLEX_ATTENTION_PADDING = getattr(
            torch.nn.attention.flex_attention,
            "_DEFAULT_SPARSE_BLOCK_SIZE",
            1,
        )
        flex_attention = torch.compile(flex_attention, dynamic = False)
        HAS_FLEX_ATTENTION = True
    else:
        HAS_FLEX_ATTENTION = False
    pass
else:
    HAS_FLEX_ATTENTION = False
pass

# Logit softcapping
@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
    n_heads    = self.num_heads
    head_dim   = self.head_dim
    n_kv_heads = self.num_key_value_heads
    n_groups   = self.num_key_value_groups
    
    # Grouped query attention
    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)

    # See https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
    # Gemma 9b should use 256 and not 224 (hs / nah). 27b uses the below
    # We default to using the config file itself
    # s = self.config.hidden_size // self.config.num_attention_heads
    s = self.config.query_pre_attn_scalar
    t = self.config.attn_logit_softcapping

    Q = Q * torch.tensor(s**-0.5, dtype = Q.dtype) # Follow Keras exactly
    A = torch.matmul(Q, K.transpose(2, 3))
    A = t * torch.tanh(A / t) # Logit softcapping
    A += causal_mask[:q_len, :q_len]
    # Much slower in torch compile!
    # A.masked_fill_(causal_mask[:q_len, :q_len], -float("inf"))
    A = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype)
    A = torch.matmul(A, V)
    A = A.transpose(1, 2).contiguous()
    A = A.reshape(bsz, q_len, n_heads*head_dim)
    return A
pass


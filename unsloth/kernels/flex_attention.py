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
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _flex_attention = torch.compile(_flex_attention, dynamic = False)
    HAS_FLEX_ATTENTION = True
except:
    HAS_FLEX_ATTENTION = False
pass


if not HAS_FLEX_ATTENTION:

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
else:
    # See https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb
    # for more examples
    # BSD 3-Clause License Copyright (c) 2023, Driss Guessous, Horace He et al
    import functools, math

    from torch.nn.attention.flex_attention import _score_mod_signature
    from torch._inductor.lowering import make_pointwise, register_lowering

    # Some internal torch.compile details
    from torch._inductor.virtualized import ops
    from functools import partial


    @torch.library.custom_op("approx::tanh", mutates_args=())
    def _tanh_approx(inp: Tensor) -> Tensor:
        return torch.tanh(inp)


    @_tanh_approx.register_fake
    def _(inp: torch.Tensor) -> torch.Tensor:
        return torch.tanh(inp)


    def _tanh_approx_lowering(inp):
        fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
        return make_pointwise(fn)(inp)


    register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


    class _TanhApprox(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return torch.ops.approx.tanh(x)

        @staticmethod
        def setup_context(ctx, inputs, output):
            (x,) = inputs
            result = output
            ctx.save_for_backward(result)

        @staticmethod
        def backward(ctx, grad_output):
            (result,) = ctx.saved_tensors
            return grad_output * (1 - result * result)

        @staticmethod
        def vmap(info, in_dims, x):
            return torch.tanh(x), 0


    _tanh_approx = _TanhApprox.apply

    @functools.lru_cache
    def generate_tanh_softcap(t):
        def tanh_softcap(x, b, h, q_idx, kv_idx):
            return t * _tanh_approx(x / t)
        return tanh_softcap
    pass
    def causal_masker(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    pass

    @functools.lru_cache
    def sliding_window_masker(size = 4096, q_len = 4096):
        def sliding_window(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= size 
            return causal_mask & window_mask
        return sliding_window if q_len >= size else causal_masker
    pass

    @functools.lru_cache
    def create_block_mask(mask, n = 128):
        return _create_block_mask(
            mask, 1, 1, n, n,
            BLOCK_SIZE = 128,
            _compile = True,
        )
    pass

    @functools.lru_cache
    def flex_attention(s, t):
        scale = 1.0 / math.sqrt(s)
        score_mod = generate_tanh_softcap(t)
        return functools.partial(
            _flex_attention, score_mod = score_mod, scale = scale, enable_gqa = True,
        )
    pass
    
    def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
        n_heads    = self.num_heads
        head_dim   = self.head_dim
        if causal_mask == 0:
            # Global attention
            causal_mask = create_block_mask(causal_masker, q_len)
        else:
            # Sliding window attention
            sliding_masker = sliding_window_masker(causal_mask, q_len)
            causal_mask = create_block_mask(sliding_masker, q_len)
        pass

        s = self.config.query_pre_attn_scalar
        t = self.config.attn_logit_softcapping
        A = flex_attention(s, t)(query = Q, key = K, value = V, block_mask = causal_mask)
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads*head_dim)
        return A
    pass
pass


torch_matmul = torch.matmul
torch_tanh   = torch.tanh
torch_nn_functional_softmax = torch.nn.functional.softmax
def slow_inference_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
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
    A = torch_matmul(Q, K.transpose(2, 3))

    # Logit softcapping
    A /= t; torch_tanh(A, out = A); A *= t;
    A += causal_mask[:q_len, :q_len]
    # Much slower in torch compile!
    # A.masked_fill_(causal_mask[:q_len, :q_len], -float("inf"))
    A = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype)
    A = torch_matmul(A, V)
    A = A.transpose(1, 2).contiguous()
    A = A.reshape(bsz, q_len, n_heads*head_dim)
    return A
pass

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
    from torch.nn.attention._flex_attention import create_block_mask
    from torch.nn.attention._flex_attention import _flex_attention
    flex_attention = torch.compile(_flex_attention, dynamic = False)

    import torch.nn.attention._flex_attention
    # Currently must pad to 128
    FLEX_ATTENTION_PADDING = getattr(
        torch.nn.attention._flex_attention,
        "_DEFAULT_SPARSE_BLOCK_SIZE",
        1,
    )
    
    @lru_cache
    def flex_attention_create_block_mask(score_mod, B, H, M, N):
        return create_block_mask(score_mod, B, H, M, N, device = "cuda:0")
    pass

    def flex_attention_causal_mask(score, b, h, q_idx, kv_idx):
        causal = (q_idx >= kv_idx)
        mask = causal
        return torch.where(mask, score, -float("inf"))
    pass

    @lru_cache
    def flex_attention_sliding_window_mask(SLIDING_WINDOW = 4096):
        def sliding_window_mask(score, b, h, q_idx, kv_idx):
            causal  = (q_idx >= kv_idx)
            sliding = (q_idx - kv_idx <= SLIDING_WINDOW)
            mask = causal & sliding
            return torch.where(mask, score, -float("inf"))
        pass
        return sliding_window_mask
    pass


    # Logit softcapping for Gemma 2
    @torch.library.custom_op("approx::tanh", mutates_args=())
    def tanh_approx(inp: torch.Tensor) -> torch.Tensor:
        return torch.tanh(inp)

    @tanh_approx.register_fake
    def _(inp: torch.Tensor) -> torch.Tensor:
        return torch.tanh(inp)

    # Some internal torch.compile details :P
    from torch._inductor.virtualized import ops
    from torch._inductor.lowering import make_pointwise, register_lowering
    from functools import partial
    def tanh_approx_lowering(inp):
        fn = partial(ops.inline_asm_elementwise, asm = "tanh.approx.f32 $0, $1;")
        return make_pointwise(fn)(inp)
    register_lowering(torch.ops.approx.tanh)(tanh_approx_lowering)

    class TanhApprox(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return torch.ops.approx.tanh(x)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, = inputs
            result = output
            ctx.save_for_backward(result)

        @staticmethod
        def backward(ctx, grad_output):
            result, = ctx.saved_tensors
            return grad_output * (1 - result * result)
    pass

    tanh_approx = TanhApprox.apply

    # @lru_cache
    def flex_attention_softcapping_causal_mask(LOGIT_SOFTCAPPING = 50.0):
        def softcapping_causal_mask(score, b, h, q_idx, kv_idx):
            score = score / LOGIT_SOFTCAPPING
            score = tanh_approx(score)
            score = LOGIT_SOFTCAPPING * score

            causal = (q_idx >= kv_idx)
            mask = causal
            return torch.where(mask, score, -float("inf"))
        pass
        return softcapping_causal_mask
    pass

    # @lru_cache
    def flex_attention_softcapping_causal_sliding_window_mask(LOGIT_SOFTCAPPING = 50.0, SLIDING_WINDOW = 4096):
        def softcapping_causal_sliding_window_mask(score, b, h, q_idx, kv_idx):
            score = score / LOGIT_SOFTCAPPING
            score = tanh_approx(score)
            score = LOGIT_SOFTCAPPING * score

            causal  = (q_idx >= kv_idx)
            sliding = (q_idx - kv_idx <= SLIDING_WINDOW)
            mask = causal & sliding
            return torch.where(mask, score, -float("inf"))
        pass
        return softcapping_causal_sliding_window_mask
    pass

    def can_use_flex_attention(q_len):
        if q_len % FLEX_ATTENTION_PADDING != 0:
            logger.warning_one(
                f"Unsloth: Flex Attention currently can only work "\
                f"when all sequence lengths must be padded to {FLEX_ATTENTION_PADDING}.\n"\
                f"Use ```from transformers import DataCollatorForLanguageModeling\n"\
                f"data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False, pad_to_multiple_of = {FLEX_ATTENTION_PADDING})```"
            )
            return False
        return True
    pass

    def flex_attention_dispatch(Q, K, V, metadata, self, bsz, q_len):
        block_mask, score_function = metadata
        n_heads    = self.num_heads
        head_dim   = self.head_dim
        n_kv_heads = self.num_key_value_heads
        n_groups   = self.num_key_value_groups
        
        # Grouped query attention
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
        K = K.reshape(bsz, n_heads, q_len, head_dim)
        V = V.reshape(bsz, n_heads, q_len, head_dim)

        # block_mask = flex_attention_create_block_mask(causal_mask, 1, 1, q_len, q_len)
        print(score_function, block_mask)
        A = flex_attention(
            Q, K, V,
            score_mod  = score_function,
            block_mask = block_mask,
            scale = self.config.query_pre_attn_scalar,
        )
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads*head_dim)
        return A
    pass

    HAS_FLEX_ATTENTION = True
except:
    HAS_FLEX_ATTENTION = False
    can_use_flex_attention = lambda q_len: False
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


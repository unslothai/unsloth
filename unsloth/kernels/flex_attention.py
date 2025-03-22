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
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1",
    "triton.cudagraphs" : False,
}


# Flex Attention supported from torch 2.5 onwards only
try:
    import torch
    import torch.nn.functional as F

    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
        flex_attention as _reference_flex_attention,
        _mask_mod_signature,
        _score_mod_signature,
        noop_mask as _noop_mask,
        and_masks as _and_masks
    )
    from torch import Tensor
    from typing import Optional

    class DynamicFlexAttention:
        """
        This wrapper class around Flex Attention allows for dynamic sequence
        lengths without having to excessively recompile flex_attention.
        It pads the inputs Q, K, V to the size the Flex Attention kernel
        was compiled for and uses Flex Attention's own masking mechanism to
        ignore the padding.

        Rebuilds happen when the input sequence length exceeds any past
        sequence length seen before.

        Recomputation of the blockmask does unfortunately have to occur
        for each new input.

        Caveat/TODOs:

        - flex attention fails to compile properly for float64 I think?
        So had to use high atol in torch.allclose

        - We assume that the batch size and num heads is
        static between passes. Would trigger kernel rebuilds if otherwise.

        - Potentially cache the blockmasks with an LRU/LFU cache?

        - Dynamically choose the `flex_attention` kernel too? Pre-compile
        flex_attention kernels in powers of 2? And then binary search/index
        into `ceiling_next_power_of_2(input_seq_len)`? Pretty quick to index
        into and prevent ridiculous padding sizes. Biggest would be in the
        order of double the input size.
        """

        def __init__(
            self,
            size_hint: torch.Size = None,
            compile_options = None
        ):
            # TODO: Lookout for dynaic=True support becoming available
            self._flex_attention = torch.compile(
                _reference_flex_attention,
                dynamic=False,
                options=compile_options,
            )

            self.max_seq_len = 0

            # Compile flex_attention to the hinted (max seq) size
            if size_hint:
                bs, num_heads, _, _ = size_hint
                self.bs = bs
                self.num_heads = num_heads

                q = torch.empty(size_hint)
                k = torch.empty(size_hint)
                v = torch.empty(size_hint)

                self._flex_attention(q, k, v)
            else:
                self.bs, self.num_heads = None, None


        # Important to note! Our flex_attention wrapper
        # takes in a mask_mod instead of a block_mask
        def flex_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            score_mod: Optional[_score_mod_signature] = None,
            mask_mod: _mask_mod_signature = _noop_mask,
            scale: Optional[float] = None,
            enable_gqa: bool = False,
            return_lse: bool = False,
        ) -> Tensor:
            print("custom flex attn wrapper called!")
            bs, num_heads, q_len, head_dim = query.shape
            _, _, kv_len, _ = key.shape

            if self.bs is not None:
                assert bs == self.bs and num_heads == self.num_heads, \
                    "Dynamic batch sizes and number of heads not currently " \
                    + "supported for performance reasons. Pad inputs accordingly" \
                    + " if desired."

            self.max_seq_len = max(
                q_len,
                kv_len,
                self.max_seq_len
            )

            # TODO: See if we can make our own blockmask constructor?
            # Also LFU/LRU caching here?
            # https://x.com/cHHillee/status/1851418255749169419?lang=en
            blockmask = _create_block_mask(
                _and_masks(
                    lambda _b, _h, q_i, kv_i: q_i < q_len,
                    lambda _b, _h, q_i, kv_i: kv_i < kv_len,
                    mask_mod
                ),
                B=None,
                H=None,
                Q_LEN=self.max_seq_len,
                KV_LEN=self.max_seq_len,
                BLOCK_SIZE = 128,
                # TODO: Will be deprecated in favor of torch.compile soon
                _compile=True,
            )

            padded_q = F.pad(query, (0, 0, 0, self.max_seq_len - q_len))
            padded_k = F.pad(key, (0, 0, 0, self.max_seq_len - kv_len))
            padded_v = F.pad(value, (0, 0, 0, self.max_seq_len - kv_len))

            res = self._flex_attention(
                padded_q,
                padded_k,
                padded_v,
                score_mod=score_mod,
                block_mask=blockmask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_lse=return_lse,
            )

            return res[:, :, :q_len, :]

    dynamic_flex_attention = DynamicFlexAttention()
    _flex_attention = dynamic_flex_attention.flex_attention

    HAS_FLEX_ATTENTION = True
except:
    HAS_FLEX_ATTENTION = False
pass


if not HAS_FLEX_ATTENTION:

    # Logit softcapping
    @torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
    def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
        n_heads    = self.config.num_attention_heads
        head_dim   = self.head_dim
        n_kv_heads = self.config.num_key_value_heads
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

    def generate_tanh_softcap(t):
        def tanh_softcap(x, b, h, q_idx, kv_idx):
            return t * torch.tanh(x / t)
        return tanh_softcap
    pass
    def causal_masker(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    pass

    @functools.lru_cache
    def sliding_window_masker(size = 4096):
        def sliding_window(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= size 
            return causal_mask & window_mask
        return sliding_window
    pass

    @functools.lru_cache
    def flex_attention(s, t):
        scale = 1.0 / math.sqrt(s)
        score_mod = generate_tanh_softcap(t)
        return functools.partial(
            _flex_attention, score_mod = score_mod, scale = scale, enable_gqa = True,
        )
    pass

    def slow_attention_softcapping(Q, K, V, mask_mod, self, bsz, q_len):
        n_heads    = self.config.num_attention_heads
        head_dim   = self.head_dim
        s = self.config.query_pre_attn_scalar
        t = self.config.attn_logit_softcapping
        fx = flex_attention(s, t)
        A = fx(query = Q, key = K, value = V, mask_mod = mask_mod)
        A = A.transpose(1, 2).contiguous()
        A = A.reshape(bsz, q_len, n_heads*head_dim)
        return A
    pass
pass


torch_matmul = torch.matmul
torch_tanh   = torch.tanh
torch_nn_functional_softmax = torch.nn.functional.softmax
def slow_inference_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
    n_heads    = self.config.num_attention_heads
    head_dim   = self.head_dim
    n_kv_heads = self.config.num_key_value_heads
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

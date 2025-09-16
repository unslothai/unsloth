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
    from collections import Counter, namedtuple

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

    DynamicFlexShape = namedtuple(
        "DynamicFlexShape",
        ["bs", "q_num_heads", "head_dim"]
    )

    def next_pow_of_2(x, min_exp = 1):
        # There probaby are more efficient ways to do this
        # but not worth optimizing for now.
        res = 1 << min_exp
        exp = min_exp
        while res < x:
            res = res << 1
            exp += 1

        return res, exp
    pass

    def get_dynamic_flex_shape(query):
        batch_size, q_num_heads, _q_seq_len, head_dim = query.shape
        return DynamicFlexShape(
            batch_size,
            q_num_heads,
            head_dim
        )
    pass

    class CachedBlockMask:
        """
        Huge downside of the padding mechanism in PaddedFlexAttention is that
        a new BlockMask must be created for each new input sequence length.
        We cannot simply make a larger block mask since we mask based on
        the respective indicies being less than the input q_len and kv_len.

        Potential fix could be using custom constructors for the BlockMask.
        Might not be much more efficient. Thus, we cache for now.
        """

        def __init__(
            self,
            mask_mod: _mask_mod_signature = _noop_mask,
            mask_mod_batch_idx_independent = True,
            mask_mod_head_idx_independent = True,
        ):
            self.mask_mod = mask_mod
            self.mask_mod_batch_idx_independent = mask_mod_batch_idx_independent
            self.mask_mod_head_idx_independent = mask_mod_head_idx_independent

        @lru_cache(maxsize=512)
        def get(self, B, H, Q_LEN, KV_LEN, MAX_SEQ_LEN):
            if self.mask_mod_batch_idx_independent:
                B = None

            if self.mask_mod_batch_idx_independent:
                H = None

            block_mask = _create_block_mask(
                _and_masks(
                    lambda _b, _h, q_i, kv_i: q_i < Q_LEN,
                    lambda _b, _h, q_i, kv_i: kv_i < KV_LEN,
                    self.mask_mod
                ),
                B=B,
                H=H,
                Q_LEN=MAX_SEQ_LEN,
                KV_LEN=MAX_SEQ_LEN,
                BLOCK_SIZE=128,
                # TODO: Will be deprecated in favor of torch.compile soon
                _compile=True,
            )

            return block_mask
        pass


    class DynamicFlexAttention:
        """
        This wrapper class selectively chooses `PaddedFlexAttention`
        instances to pick the kernel with the smallest required padding
        for the inputs.

        Caveats/Restrictions:

        - Static batch size and num heads across passes
        It is a bit ugly that the singleton instance of DynamicFlexAttention
        hard-codes the batch size and num heads globally. Would require
        a lot of refactoring of attention within Unsloth to not require this
        global slop.
        """

        def __init__(
            self,
            compile_options = None,
            dynamic_flex_shape: Optional[DynamicFlexShape] = None,
            # support seq lengths of up to 8192
            _MIN_SEQ_LEN_EXP = 7, # 2^7  = 128
            _MAX_SEQ_LEN_EXP = 13 # 2^13 = 8192
        ):
            self._MIN_SEQ_LEN_EXP = _MIN_SEQ_LEN_EXP
            self._MAX_SEQ_LEN_EXP = _MAX_SEQ_LEN_EXP
            self.kernel_lookup = [
                PaddedFlexAttention(
                    seq_size_hint=(1 << i),
                    compile_options=compile_options
                ) for i in range(
                    self._MIN_SEQ_LEN_EXP,
                    self._MAX_SEQ_LEN_EXP + 1
                )
            ]
            self.dynamic_flex_shape = dynamic_flex_shape
            self.noop_dynamic_block_mask = CachedBlockMask()

        def flex_attention(
            self,
            query: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
            _, _, q_len, _ = query.shape
            _, exp = next_pow_of_2(q_len, self._MIN_SEQ_LEN_EXP)

            exp_idx = exp - self._MIN_SEQ_LEN_EXP
            assert exp_idx < len(self.kernel_lookup), \
                "Exceeded max expected sequence length of f{1 << self._MAX_SEQ_LEN_EXP}"

            if "block_mask" not in kwargs:
                kwargs["block_mask"] = self.noop_dynamic_block_mask

            kernel = self.kernel_lookup[exp_idx].flex_attention
            return kernel(query, *args, **kwargs)
        pass


    class PaddedFlexAttention:
        """
        This wrapper class around Flex Attention pads sequence
        lengths to avoid having to excessively recompile flex_attention.
        It pads the inputs Q, K, V to the size the Flex Attention kernel
        was compiled for and uses Flex Attention's own masking mechanism to
        ignore the padding.

        Rebuilds happen when the input sequence length exceeds any past
        sequence length seen before. Allow recompilation with
        allow_recompile=True.

        Caveat/TODOs:

        - flex_attention fails to compile properly for float64 I think?
        So had to use high atol in torch.allclose

        - We assume that the batch size and num heads is
        static between passes. Would trigger kernel rebuilds if otherwise.
        """

        def __init__(
            self,
            seq_size_hint: int = 128,
            allow_recompile = False,
            compile_options = None
        ):
            # TODO: Lookout for dynamic=True support becoming available
            self._flex_attention = torch.compile(
                _reference_flex_attention,
                dynamic=False,
                options=compile_options,
            )
            self.allow_recompile = allow_recompile

            # Compile flex_attention to the hinted (max seq) size if provided
            self.max_seq_len = seq_size_hint

            # For statistics
            self.query_shape_stats = Counter()
            self.kv_shape_stats = Counter()

            self.dynamic_flex_shape = None
        pass

        def flex_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            score_mod: Optional[_score_mod_signature] = None,
            block_mask: Optional[CachedBlockMask] = None,
            scale: Optional[float] = None,
            enable_gqa: bool = False,
            return_lse: bool = False,
        ) -> Tensor:
            self.query_shape_stats[query.shape] += 1
            self.kv_shape_stats[key.shape] += 1

            # shape and type checks
            bs, num_heads, q_len, head_dim = query.shape
            _, _, kv_len, _ = key.shape

            if not self.allow_recompile:
                input_seq_len = max(q_len, kv_len)
                assert input_seq_len <= self.max_seq_len

            dynamic_flex_shape = get_dynamic_flex_shape(query)
            if self.dynamic_flex_shape is None:
                self.dynamic_flex_shape = dynamic_flex_shape
            else:
                assert self.dynamic_flex_shape == dynamic_flex_shape

            if block_mask is not None:
                assert isinstance(block_mask, CachedBlockMask), \
                    "PaddedFlexAttention.flex_attention expects a CachedBlockMask " + \
                    "in the block_mask argument"

                final_block_mask = block_mask.get(
                    bs, num_heads, q_len, kv_len, self.max_seq_len
                )

            else:
                final_block_mask = _create_block_mask(
                    _and_masks(
                        lambda _b, _h, q_i, kv_i: q_i < q_len,
                        lambda _b, _h, q_i, kv_i: kv_i < kv_len,
                    ),
                    B=bs,
                    H=num_heads,
                    Q_LEN=self.max_seq_len,
                    KV_LEN=self.max_seq_len,
                    BLOCK_SIZE = 128,
                    _compile=True,
                )
            # end checks

            padded_q = F.pad(query, (0, 0, 0, self.max_seq_len - q_len))
            padded_k = F.pad(key, (0, 0, 0, self.max_seq_len - kv_len))
            padded_v = F.pad(value, (0, 0, 0, self.max_seq_len - kv_len))

            res = self._flex_attention(
                padded_q,
                padded_k,
                padded_v,
                score_mod=score_mod,
                block_mask=final_block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_lse=return_lse,
            )

            return res[:, :, :q_len, :]
        pass

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

    create_cached_sliding_window_block_mask = None
    create_cached_causal_block_mask = None
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

    def create_cached_sliding_window_block_mask(size = 4096):
        return CachedBlockMask(mask_mod=sliding_window_masker(size))
    pass

    def create_cached_causal_block_mask():
        return CachedBlockMask(mask_mod=causal_masker)
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
        n_heads    = self.config.num_attention_heads
        head_dim   = self.head_dim
        s = self.config.query_pre_attn_scalar
        t = self.config.attn_logit_softcapping
        fx = flex_attention(s, t)
        A = fx(query = Q, key = K, value = V, block_mask = causal_mask)
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

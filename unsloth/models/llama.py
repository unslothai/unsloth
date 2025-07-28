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
import gc
import math
import functools
from typing import Optional, Tuple, List, Union
from ._utils import *
from ._utils import patch_unsloth_smart_gradient_checkpointing
from ._utils import __version__
from ._utils import move_to_device
from torch.nn.functional import scaled_dot_product_attention
from transformers import __version__ as transformers_version
from unsloth_zoo.utils import Version, _get_dtype
from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
from unsloth import DEVICE_TYPE, DEVICE_COUNT

transformers_version = Version(transformers_version)
# Transformers moved rotary embeddings out of all attention layers
IS_ATTENTION_REFACTOR = transformers_version > Version("4.47.1")
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except:
    GradientCheckpointingLayer = type(None)

from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ..kernels import *
from ..tokenizer_utils import *
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func
from .vision import FastBaseModel

# Final patching code
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention   = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention
pass

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import set_seed as transformers_set_seed
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from ..save import patch_saving_functions
import re, os, inspect, math, sys
import types
try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
pass
from triton import __version__ as triton_version
HAS_XFORMERS = xformers is not None
BlockDiagonalCausalMask = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None

if DEVICE_TYPE == "xpu":
    clean_gpu_cache = torch.xpu.empty_cache
    get_current_device = torch.xpu.current_device
else:
    clean_gpu_cache = torch.cuda.empty_cache
    get_current_device = torch.cuda.current_device
pass

def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V
pass


def original_apply_o(self, X):
    O = self.o_proj(X)
    return O
pass

from math import sqrt as math_sqrt
KV_CACHE_INCREMENT = 512 # KV Cache update size
torch_nn_functional_softmax = torch.nn.functional.softmax
# SDPA has GQA internally
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__

# Fix new HF's inference code
def _fast_prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs,):
    past_key_values = kwargs.get("past_key_values", None)
    if past_key_values is not None:
        # Check for uninitialized DynamicCache
        if len(past_key_values) == 0:
            past_key_values = None
            kwargs["past_key_values"] = None
        else:
            bs, cache_length = input_ids.shape
            input_ids = input_ids[:,[-1]]

            # Get to the base model
            base_model = self
            if hasattr(base_model, 'base_model_prefix'):
                base_model = getattr(base_model, base_model.base_model_prefix)

            if hasattr(base_model, "_prepare_4d_causal_attention_mask_with_cache_position"):
                def needs_device_kw(fn) -> bool:
                    try:
                        sig = inspect.signature(inspect.unwrap(fn))
                        return "device" in sig.parameters
                    except:
                        # transformers <= 4.51.3 includes device arg but > 4.51.3 does not
                        return transformers_version < Version("4.52.0")

                kwargs = {
                    "sequence_length": 1,
                    "target_length": cache_length,
                    "dtype": self.dtype,
                    "cache_position": torch.arange(cache_length, cache_length+1, device=input_ids.device),
                    "batch_size": bs,
                    "config": self.config,
                    "past_key_values": past_key_values,
                }
                try:
                    if needs_device_kw(base_model._prepare_4d_causal_attention_mask_with_cache_position):
                        kwargs["device"] = input_ids.device
                except:
                    print(f"Unsloth: Could not inspect signature of {base_model._prepare_4d_causal_attention_mask_with_cache_position}")

                attention_mask = base_model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    **kwargs,
                )
            else:
                attention_mask = attention_mask[:,[-1]]
                if transformers_version <= Version("4.52.4"):
                    logger.warning_once(
                        f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                        "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                        "writing code, see Llama for an example implementation. If you're a user, please report this "
                        "issue on GitHub."
                    )

    if "cache_position" in kwargs:
        kwargs["position_ids"] = kwargs["cache_position"]
    return { "input_ids" : input_ids, "attention_mask": attention_mask, **kwargs, }
pass


def fix_prepare_inputs_for_generation(module):
    # Fix prepare_inputs_for_generation
    if hasattr(module, "prepare_inputs_for_generation"):
        module.prepare_inputs_for_generation = _fast_prepare_inputs_for_generation
    pass
pass

torch_matmul = torch.matmul
def LlamaAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L406
        Fast inference using KV cache.
        QK^T can be computed in 4 chunks

        [Q, q] @ [K, k].T where q, k are the new tokens.
        [QK^T, Qk^T]
        [qK^T, qk^T]

        Since the attention mask wipes Qk^T, we just get
        [QK^T,    0]
        [qK^T, qk^T]

        Since softmax is row-wise, we get
        softmax([QK^T,    0])
        softmax([qK^T, qk^T])

        We then multiply by   [V]
                              [v]
        softmax([QK^T,    0]) [softmax(QK^T)V] *
        softmax([qK^T, qk^T]) [softmax([qK^T, qk^T]) @ [V, v]]

        But notice * [softmax(QK^T)V] is just the last attention.
        We just need to compute the last final row.

        This means we can pass in a row of Q, but we need to
        remember K and V, which are called the KV cache.
    """
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    K1, V1 = past_key_value
    dtype = Xn.dtype

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    # assert(n_kv_heads * n_groups == n_heads)

    hidden_size = self.config.hidden_size
    attention_size = n_heads*head_dim
    seq_len = K1.shape[-2]
    kv_seq_len = seq_len + 1

    # Prefill phase
    # if not hasattr(self, "paged_attention"):
    device = hidden_states.device
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = device)
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = device)
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)

        # Mistral Nemo 12b has weird dimensions
        if attention_size != hidden_size:
            self.temp_O = torch.empty((1, bsz, hidden_size), dtype = dtype, device = device)
        else:
            self.temp_O = self.temp_QA[1][:,:,:hidden_size]
        pass

        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = device)
        self.scalar = 1.0 / math_sqrt(self.head_dim)
        self.half_head_dim = head_dim // 2
    elif kv_seq_len >= self.paged_attention.shape[0]:
        self.paged_attention.resize_((self.paged_attention.shape[0]+KV_CACHE_INCREMENT, 2, bsz, n_kv_heads, head_dim))
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1]+KV_CACHE_INCREMENT))
    pass

    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])
    Kn = fast_linear_forward(self.k_proj, Xn, out = self.temp_KV[0])
    Vn = fast_linear_forward(self.v_proj, Xn, out = self.temp_KV[1])
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)

    # cos, sin = self.rotary_emb(Vn, seq_len = kv_seq_len)
    # Qn, Kn = inplace_rope_embedding(Qn, Kn, cos, sin, position_ids)

    # Need to do it prior 2 steps before hitting full on short KV cache
    # or else error
    self.rotary_emb.extend_rope_embedding(Vn, seq_len + 2)
    cos, sin = self.rotary_emb.get_cached(kv_seq_len, Qn.device.index)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    h = self.half_head_dim

    RH_Q = self.RH_Q
    RH_Q[:,:,:,:h] = Qn[:,:,:,h:]
    RH_Q[:,:,:,h:] = Qn[:,:,:,:h]
    RH_Q[:,:,:,:h].neg_() # torch.neg(RH_Q[:,:,:,:h], out = RH_Q[:,:,:,:h])
    Qn *= cos
    Qn.addcmul_(RH_Q, sin)

    RH_K = RH_Q[:,:n_kv_heads,:,:] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda:0")
    RH_K[:,:,:,:h] = Kn[:,:,:,h:]
    RH_K[:,:,:,h:] = Kn[:,:,:,:h]
    RH_K[:,:,:,:h].neg_() #torch.neg(RH_K[:,:,:,:h], out = RH_K[:,:,:,:h])
    Kn *= cos
    Kn.addcmul_(RH_K, sin)

    # New KV cache
    # Kn = torch.cat([K1, Kn], dim = 2)
    # Vn = torch.cat([V1, Vn], dim = 2)
    self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)
    self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)
    Kn = self.paged_attention_K[:kv_seq_len].permute(1, 2, 0, 3)
    Vn = self.paged_attention_V[:kv_seq_len].permute(1, 2, 0, 3)

    # Handle sliding windows
    sliding_window = getattr(self.config, "sliding_window", None)
    if sliding_window is not None and kv_seq_len > sliding_window:
        # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L193
        slicing_tokens = 1 - sliding_window
        Knn = Kn[:, :, slicing_tokens:, :]#.contiguous()
        Vnn = Vn[:, :, slicing_tokens:, :]#.contiguous()
    else:
        Knn, Vnn = Kn, Vn
    pass

    # Grouped query attention
    _, _, cached_len, _ = Knn.shape
    if bsz == 1 or not SDPA_HAS_GQA and n_groups != 1:
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Knn = Knn.reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, head_dim)
    pass
    # else:
    #     Knn, Vnn = Knn, Vnn
    # pass

    # when qlen==vlen and attn_mask is None, we should use causal attention
    Q_len = Qn.shape[-2]
    K_len = Knn.shape[-2]
    if attention_mask is None and Q_len == K_len:
        is_causal = True
    else:
        is_causal = False
    # Attention
    if bsz == 1:
        Qn *= self.scalar # See https://github.com/ggerganov/llama.cpp/issues/7805#issuecomment-2153349963
        # It seems like doing (Q * scalar) @ K is better than (Q @ K) * scalar to stop overflows
        A = torch_matmul(Qn, Knn.transpose(2, 3), out = self.attention[:,:,:,:cached_len])
        # if attention_mask is not None: A += attention_mask # Must add attention_mask for batched
        A[:] = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
        A = torch_matmul(A, Vnn, out = Qn)
    else:
        if SDPA_HAS_GQA:
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = is_causal, enable_gqa = True)
        else:
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = is_causal)
    pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass


torch_nn_functional_silu = torch.nn.functional.silu
def fast_swiglu_inference(self, X, temp_gate = None, temp_up = None, gate_multiplier = None, down_multiplier = None):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    # mlp_size = self.config.intermediate_size
    # temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda:0")

    gate = fast_linear_forward(self.gate_proj, X, out = temp_gate)

    if gate_multiplier is not None:
        gate *= gate_multiplier

    up   = fast_linear_forward(self.  up_proj, X, out = temp_up)

    gate = torch_nn_functional_silu(gate, inplace = True)
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])

    if down_multiplier is not None:
        down *= down_multiplier

    return down
pass

torch_square = torch.square
torch_mean   = torch.mean
def fast_rms_layernorm_inference(self, X, XX = None, XX2 = None, variance = None):
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim = True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out = XX2), -1, keepdim = True, out = variance)
    pass
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None: X = XX.to(old_dtype)
    else: X.copy_(XX)

    X *= self.weight
    return X
pass


def fast_rms_layernorm_inference_gemma(self, X, out_weight = None):
    XX = X.to(torch.float32)
    variance = XX.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if out_weight is None:
        out_weight = self.weight + 1.0
    else:
        out_weight[:] = self.weight
        out_weight += 1.0
    pass

    XX *= out_weight
    return XX.to(X.dtype)
pass


# Normal layernorm with mean removal
@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def fast_layernorm_compiled(layernorm, X):
    old_dtype = X.dtype
    X = X.float()
    mean = X.mean(-1, keepdim = True)
    Xbar = X - mean
    X = Xbar * torch.rsqrt(Xbar.square().mean(-1, keepdim = True) + \
        layernorm.variance_epsilon) * \
        layernorm.weight.float()
    return X.to(old_dtype)
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask:         Optional[BlockDiagonalCausalMask] = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   bool = False,
    use_cache:           bool = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)

        # if position_ids is None:
        #     # Useful for LongRoPE
        #     cos, sin = rotary_emb.get_cached(kv_seq_len, device = Q.device)
        # else:
        #     cos, sin = rotary_emb.get_cached(seq_len = kv_seq_len, device = Q.device)
        cos, sin = rotary_emb.get_cached(kv_seq_len, Q.device.index)

    # Q, K = (
    #     fast_rope_embedding(Q, K, cos, sin)
    #     if position_ids is None
    #     else inplace_rope_embedding(Q, K, cos, sin, position_ids)
    # )
    Q, K = fast_rope_embedding(Q, K, cos, sin)
    # synchronize before cat to avoid race condition
    torch.cuda.current_stream(Q.device).synchronize()

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and HAS_XFORMERS and attention_mask is None):
        # Xformers memory efficient attention
        # Also has Flash Attention v2 dispatching
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Group query attention
        if n_groups != 1:
            K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            if hidden_states.requires_grad:
                K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        pass
        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        A = flash_attn_func(Q, K, V, causal = True)
    else:
        # when qlen==vlen and attn_mask is None, we should use causal attention
        Q_len = Q.shape[-2]
        K_len = K.shape[-2]
        if attention_mask is None and Q_len == K_len:
            is_causal = True
        else:
            is_causal = False
        # Grouped query attention
        if SDPA_HAS_GQA:
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = is_causal, enable_gqa = n_groups != 1)
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2)#.contiguous()
        else:
            if n_groups != 1:
                K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
                V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
            pass
            # Must be contiguous or else results are False!
            # https://github.com/pytorch/pytorch/issues/112577
            Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = is_causal)
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2).contiguous()
        pass
    pass
    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask          = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   Optional[bool] = False,
    use_cache:           Optional[bool] = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if use_cache and hasattr(self, "_flag_for_generation"):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass


# https://github.com/unslothai/unsloth/issues/404#issuecomment-2323473452
__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_values:      Optional[List[torch.FloatTensor]] = None,
    inputs_embeds:        Optional[torch.FloatTensor] = None,
    use_cache:            Optional[bool] = None,
    output_attentions:    Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict:          Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    assert(output_attentions is False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            logger.warning_once(
                f"Unsloth: Input IDs of length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"\
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:,:self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:,:self.max_seq_length,:]
        pass
    pass

    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    pass

    # We already handle KV cache position_ids ourselves.
    if False:#(past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = f"{DEVICE_TYPE}:0",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None
    pass

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))
    pass

    # Embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = inputs_embeds.to(_get_dtype(self.config.torch_dtype))

    # Normalized from Gemma
    IS_GEMMA   = self.config.model_type.startswith("gemma")
    IS_GEMMA2  = self.config.model_type.startswith("gemma2")
    IS_COHERE  = self.config.model_type.startswith("cohere")
    IS_GRANITE = self.config.model_type.startswith("granite")
    IS_FALCON_H1 = self.config.model_type.startswith("falcon_h1")

    train_embed_tokens = self.embed_tokens.weight.requires_grad

    if IS_GEMMA:
        # Match Gemma exactly by casting to bfloat16 / float16
        # inputs_embeds *= math_sqrt(self.config.hidden_size)
        # Ie 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
        # &  2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
        normalizer = torch.tensor(math_sqrt(self.config.hidden_size), dtype = inputs_embeds.dtype)

        if train_embed_tokens:
            # Careful we must not do an inplace op!
            inputs_embeds = inputs_embeds * normalizer
        else:
            inputs_requires_grad = inputs_embeds.requires_grad
            if not inputs_embeds.is_leaf:
                inputs_embeds = inputs_embeds.detach()
                inputs_requires_grad = True
            elif inputs_requires_grad:
                inputs_embeds.requires_grad_(False)
            pass
            inputs_embeds *= normalizer
            # inputs_embeds *= math_sqrt(self.config.hidden_size)
            if inputs_requires_grad: inputs_embeds.requires_grad_(True)
        pass
    pass

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if getattr(self, "_has_no_labels", False) is True and (attention_mask is not None) and (past_key_values is None) and \
        (not train_embed_tokens) and self.training:
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if not inputs_embeds.is_leaf:
            inputs_embeds = inputs_embeds.detach()
            inputs_requires_grad = True
        elif inputs_requires_grad:
            inputs_embeds.requires_grad_(False)
        pass
        attention_mask = attention_mask[:,:self.max_seq_length] # Must resize!
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        # if 0 in attention_mask:
        #     padding_mask = attention_mask
        # else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
        # Must NOT convert to bool - weirdly this causes stuff to error out!
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(torch.bool)
    pass

    hidden_states = inputs_embeds
    if IS_GRANITE or IS_FALCON_H1: #granite has embedding multiplier
        hidden_states = self.config.embedding_multiplier * hidden_states

    if past_key_values is None and self.training:
        use_cache = False
        # if use_cache:
        #     logger.warning_once(
        #         "Unsloth: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
        #     )
        #     use_cache = False
    pass

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Gradient checkpointing methods (ie sqrt)
    if hasattr(self, "_gradient_checkpointing_boundaries"):
        boundaries = self._gradient_checkpointing_boundaries
    else:
        boundaries = None
    pass

    # Check checkpointing method
    gradient_checkpointing = False

    if (self.gradient_checkpointing and self.training and not use_cache):
        gradient_checkpointing = True
    pass

    # Gemma2 has alternating SWA and global attn
    use_static_mask  = True
    dynamic_SWA_mask = None
    dynamic_GA_mask  = None
    if IS_GEMMA2:
        if HAS_FLASH_ATTENTION_SOFTCAPPING and attention_mask is None:
            self.SWA_mask = True
            self.GA_mask  = False
        elif attention_mask is not None:
            # Fixes https://github.com/unslothai/unsloth/issues/853
            # Unsloth needs a 2D mask, not a [2, 1, n, n] mask!

            # https://github.com/pytorch/pytorch/issues/103749
            # Need to convert to float and not using bool
            # attention_mask = (1.0 - attention_mask.float()) * torch.finfo(inputs_embeds.dtype).min
            dynamic_SWA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = self.config.sliding_window,
            )
            dynamic_GA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = None,
            )
            use_static_mask = False

        elif not hasattr(self, "SWA_mask"):
            if HAS_FLEX_ATTENTION:
                # Use Flex Attention instead!
                self.SWA_mask = create_flex_attention_sliding_window_mask(self.max_seq_length, self.config.sliding_window)
                self.GA_mask  = create_flex_attention_causal_mask(self.max_seq_length)
            else:
                n = self.max_seq_length # self.config.max_position_embeddings
                # masked_fill is making stuff slower!
                # self. GA_mask = create_boolean_mask(n = n, sliding_window = 0)
                # self.SWA_mask = create_boolean_mask(n = n, sliding_window = self.config.sliding_window)
                from transformers.modeling_attn_mask_utils import AttentionMaskConverter
                self.SWA_mask = AttentionMaskConverter(
                    is_causal = True,
                    sliding_window = self.config.sliding_window,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)

                self.GA_mask = AttentionMaskConverter(
                    is_causal = True,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)
            pass
        pass
    pass

    if (IS_ATTENTION_REFACTOR and (hasattr(self, "rotary_emb") or not hasattr(self.layers[0].self_attn, "rotary_emb"))) or IS_GRANITE:
        # Transformers main has made it mandatory to pass position_embeddings
        # https://github.com/huggingface/transformers/pull/34858
        # Also, transformers 4.45.0 supports granite but with the attention refactor (it always had the refactor)
        # unsloth's check for granite too has "version >= 4.45.0 (rightly so)".
        # so let granite always use the attention refactor implementation.
        position_embeddings = self.rotary_emb.get_cached(self.config.max_position_embeddings, hidden_states.device.index)
    else:
        position_embeddings = None

    # Go through every layer!
    for idx, decoder_layer in enumerate(self.layers):

        if output_hidden_states: all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        mask = causal_mask
        if IS_GEMMA2:
            if (idx % 2 == 0):
                mask = self.SWA_mask if use_static_mask else dynamic_SWA_mask
            else:
                mask = self. GA_mask if use_static_mask else dynamic_GA_mask
        pass

        if gradient_checkpointing and not isinstance(decoder_layer, GradientCheckpointingLayer):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions, padding_mask = padding_mask, position_embeddings = position_embeddings)
                return custom_forward
            pass
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                mask,
                attention_mask,
                position_ids,
                use_reentrant = True,
                preserve_rng_state = False,
            )
            hidden_states = layer_outputs[0]

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask         = mask,
                attention_mask      = attention_mask,
                position_ids        = position_ids,
                past_key_value      = past_key_value,
                output_attentions   = output_attentions,
                use_cache           = use_cache,
                padding_mask        = padding_mask,
                position_embeddings = position_embeddings,
            )
            hidden_states = layer_outputs[0]
        pass

        if use_cache: next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions: all_self_attns += (layer_outputs[1],)
    pass

    # Final layernorm
    if use_cache:
        if IS_FALCON_H1:
            hidden_states = fast_rms_layernorm_inference(self.final_layernorm, hidden_states)
        else:
            hidden_states = \
                (fast_rms_layernorm_inference_gemma if IS_GEMMA else fast_rms_layernorm_inference)\
                (self.norm, hidden_states)
    elif IS_COHERE:
        hidden_states = self.norm(hidden_states)
    elif IS_FALCON_H1:
        hidden_states = fast_rms_layernorm(self.final_layernorm, hidden_states, gemma = IS_GEMMA)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states, gemma = IS_GEMMA)
    pass

    if output_hidden_states: all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def _LlamaModel_fast_forward_inference(attention_fast_forward_inference=LlamaAttention_fast_forward_inference, mlp_fast_forward_inference=fast_swiglu_inference):
    # This makes the attention and MLP customisable.
    # Now for models like qwen3 or cohere which use custom attention operations, we can use this function
    def LlamaModel_fast_forward_inference_custom(
        self,
        input_ids,
        past_key_values,
        position_ids,
        attention_mask = None,
    ):
        input_ids = input_ids[:,:self.max_seq_length]
        bsz, q_len = input_ids.shape
        hd = self.config.hidden_size
        mlp_size = self.config.intermediate_size

        X = self.model.embed_tokens(input_ids)
        X = X.to(_get_dtype(self.config.torch_dtype))
        bsz, q_len, hd = X.shape
        assert(q_len == 1)
        # Get saved buffers to reduce memory movement
        residual = torch.empty((bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        XX, XX2 = _XX[0], _XX[1]
        variance = torch.empty((bsz, q_len, 1), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = f"{DEVICE_TYPE}:0")
        temp_gates, temp_ups = tuple(temp_mlp[0].to(torch.device(x)) for x in range(DEVICE_COUNT)), tuple(temp_mlp[1].to(torch.device(x)) for x in range(DEVICE_COUNT))

        seq_len = past_key_values[0][0].shape[-2]
        if bsz != 1:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                X,
                seq_len,
                sliding_window = getattr(self.config, "sliding_window", None),
            )
        else:
            attention_mask = None
        pass

        next_decoder_cache = []

        for idx, decoder_layer in enumerate(self.model.layers):
            device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
            X, residual, position_ids = move_to_device(
                device_index, X, residual, position_ids
            )
            residual.copy_(X) # residual = X
            X = fast_rms_layernorm_inference(
                decoder_layer.input_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X, present_key_value = attention_fast_forward_inference(
                decoder_layer.self_attn,
                hidden_states = X,
                past_key_value = past_key_values[idx],
                position_ids = position_ids,
                attention_mask = attention_mask,
                do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
            )
            X += residual

            residual.copy_(X) # residual = X
            X = fast_rms_layernorm_inference(
                decoder_layer.post_attention_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X = mlp_fast_forward_inference(
                decoder_layer.mlp,
                X,
                temp_gate = temp_gates[device_index],
                temp_up = temp_ups[device_index],
            )
            X += residual

            next_decoder_cache.append(present_key_value)
        pass
        X = fast_rms_layernorm_inference(
            self.model.norm,
            X,
            XX = XX,
            XX2 = XX2,
            variance = variance,
        )

        return BaseModelOutputWithPast(
            last_hidden_state = X,
            past_key_values = next_decoder_cache,
            hidden_states = [],
            attentions = [],
        )
    pass
    return LlamaModel_fast_forward_inference_custom

# For ensuring backwards compatibility, we create LlamaModel_fast_forward_inference that is consumed by other models
LlamaModel_fast_forward_inference = _LlamaModel_fast_forward_inference()

def CausalLM_fast_forward(fast_forward_inference):
    def _CausalLM_fast_forward(
        self,
        input_ids: torch.LongTensor = None,
        causal_mask: Optional[BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = 0,
        logits_to_keep: Optional[int] = 0,
        *args, **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None:
            outputs = fast_forward_inference(
                self,
                input_ids,
                past_key_values,
                position_ids = position_ids,
                attention_mask = attention_mask,
            )
        else:
            causal_mask = xformers.attn_bias.LowerTriangularMask() if HAS_XFORMERS else None

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            self.model._has_no_labels = labels is None
            outputs = self.model(
                input_ids = input_ids,
                causal_mask = causal_mask,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
            )
        pass
        hidden_states = outputs[0]

        bsz, q_len, hd = hidden_states.shape
        lm_head = self.lm_head.weight
        lm_head_device = lm_head.device

        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
        logit_scaling     = getattr(self.config, "logit_scale", 0)
        dtype = lm_head.dtype
        num_logits_to_keep = max(num_logits_to_keep, logits_to_keep)

        # Move items to same device as lm_head
        hidden_states = hidden_states.to(lm_head_device)
        if labels is not None: labels = labels.to(lm_head_device)

        # Output last hidden states without logits if asked
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            if num_logits_to_keep != 0:
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]
            return CausalLMOutputWithPast(
                loss = None,
                logits = hidden_states,
                past_key_values = outputs.past_key_values,
                hidden_states = outputs.hidden_states,
                attentions=  outputs.attentions,
            )
        pass

        if bsz == 1 and q_len == 1:
            logits = torch.mv(lm_head, hidden_states.ravel().to(dtype))
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif num_logits_to_keep != 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :].to(dtype))
        else:
            RETURN_LOGITS = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            # < 1024 Normal Unsloth uses less VRAM!
            if bsz*q_len <= 1024: RETURN_LOGITS = True

            if not RETURN_LOGITS and HAS_CUT_CROSS_ENTROPY and labels is not None:

                n_items = kwargs.get("num_items_in_batch", None) or kwargs.get("n_items", None)

                if self.config.model_type == "falcon_h1":
                    hidden_states = hidden_states * self.config.lm_head_multiplier

                loss = fused_linear_cross_entropy(
                    hidden_states      = hidden_states,
                    lm_weight          = lm_head,
                    labels             = labels,
                    num_items_in_batch = n_items,
                    logit_softcapping  = logit_softcapping,
                )
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output

                output = CausalLMOutputWithPast(
                    loss=loss,
                    logits=EMPTY_LOGITS,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
                return output
            pass
            logits = self.lm_head(hidden_states.to(dtype))
        pass

        logits = logits.to(_get_dtype(self.config.torch_dtype))
        loss = None
        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
        logit_scaling     = getattr(self.config, "logit_scale", 0)
        if self.config.model_type == "granite":
            # granite uses logit_scaling as key and they divide by the scale unlike cohere
            # notice that for granite, logits_scale is 16 and for cohere it is 0.125 (aka 1/8) in their respective configs
            # granite: https://github.com/huggingface/transformers/blob/4d1d0f29a493098e6bc6b904b82e29cb331827f5/src/transformers/models/granite/modeling_granite.py#L1103
            # cohere: https://github.com/huggingface/transformers/blob/4d1d0f29a493098e6bc6b904b82e29cb331827f5/src/transformers/models/cohere/modeling_cohere.py#L1176
            logit_scaling = 1 / getattr(self.config, "logits_scaling", 1)
        elif self.config.model_type == "falcon_h1":
            logit_scaling = self.config.lm_head_multiplier

        if labels is not None:
            shift_logits = logits
            # if not hasattr(self, "extra_ignored_labels"):
            #     # Fixes https://github.com/unslothai/unsloth/issues/10
            #     self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda:0")
            # pass
            shift_labels = torch.empty_like(labels)
            shift_labels[..., :-1] = labels[..., 1:]
            shift_labels[..., -1] = -100
            # shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
            loss = fast_cross_entropy_loss(
                logits = shift_logits,
                labels = shift_labels,
                logit_softcapping = logit_softcapping,
                logit_scaling     = logit_scaling,
                n_items           = kwargs.get("num_items_in_batch", None) or kwargs.get("n_items", None),
            )
        else:
            if logit_scaling != 0:
                if logits.requires_grad:
                    logits = logit_scaling * logits
                else:
                    logits *= logit_scaling
                pass
            pass
            if logit_softcapping != 0:
                if logits.requires_grad:
                    logits = (1.0 / logit_softcapping) * logits
                    logits = torch.tanh(logits)
                    logits = logit_softcapping * logits
                else:
                    logits *= (1.0 / logit_softcapping)
                    torch.tanh(logits, out = logits)
                    logits *= logit_softcapping
                pass
            pass
        pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions=  outputs.attentions,
        )
    pass
    return _CausalLM_fast_forward
pass


@torch._disable_dynamo
def PeftModel_fast_forward(
    self,
    input_ids = None,
    causal_mask = None,
    attention_mask = None,
    inputs_embeds = None,
    labels = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    task_ids = None,
    num_logits_to_keep = 0,
    logits_to_keep = 0,
    **kwargs,
):
    is_classification = "Classification" in str(type(self.base_model.model))
    if is_classification:
        return self.base_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            inputs_embeds = inputs_embeds,
            labels = labels,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            **kwargs,
        )
    else:
        return self.base_model(
            input_ids = input_ids,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            inputs_embeds = inputs_embeds,
            labels = labels,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            num_logits_to_keep = num_logits_to_keep,
            logits_to_keep = logits_to_keep,
            **kwargs,
        )
pass


# Solves https://github.com/unslothai/unsloth/issues/168
# Static KV Cache was introduced in 4.38.0, causing training to be much slower.
# Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
# https://github.com/huggingface/transformers/pull/27931
# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = getattr(config, "head_dim", None)
            if dim is None: dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE
            max_position_embeddings = config.max_position_embeddings
        pass

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_sin_cached = [None]*DEVICE_COUNT

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(seq_len=self.current_rope_size, device=torch.device(device_idx), dtype=torch.get_default_dtype())

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.sin_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )
    pass

    def get_cached(self, seq_len = None, device_index = None):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[device_index]
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype)
    pass
pass


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim = dim, max_position_embeddings = max_position_embeddings, base = base, device = device, config = config)
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()
        t = t / self.scaling_factor

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass
pass


# See https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py#L736
# For Llama 3.1
class LlamaExtendedRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE
            max_position_embeddings = config.max_position_embeddings
        pass

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_sin_cached = [None]*DEVICE_COUNT

        # Normal Llama-3 RoPE
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        inv_freq = self.apply_scaling(inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(seq_len=self.current_rope_size, device=torch.device(device_idx), dtype=torch.get_default_dtype())

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.sin_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        t = torch.arange(self.current_rope_size, device=self.inv_freq.device, dtype=torch.int64).float()

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )
    pass

    def get_cached(self, seq_len = None, device_index = None):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[device_index]
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype)
    pass

    # From https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py#L41
    def apply_scaling(self, freqs: torch.Tensor):
        # Values obtained from grid search
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
    pass
pass


class LongRopeRotaryEmbedding(torch.nn.Module):
    # For Phi 3.5 128K https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/modeling_phi3.py
    def __init__(self,
        dim = None,
        max_position_embeddings = 131072,
        original_max_position_embeddings = 4096,
        base = 10000,
        short_factor = None,
        long_factor  = None,
        device = None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        assert(short_factor is not None)
        assert(long_factor  is not None)
        assert(type(original_max_position_embeddings) is int)

        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE
            max_position_embeddings = config.max_position_embeddings
        pass

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(original_max_position_embeddings, self.max_position_embeddings)
        self.multi_gpu_short_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_short_sin_cached = [None]*DEVICE_COUNT
        self.multi_gpu_long_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_long_sin_cached = [None]*DEVICE_COUNT

        # Long RoPE similar to RoPE except short sequences have 1 cos / sin
        # and long sequences have another cos / sin
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim
        short_factor = torch.tensor(short_factor, device = "cpu", dtype = torch.float32)
        long_factor  = torch.tensor(long_factor,  device = "cpu", dtype = torch.float32)
        short_inv_freq = 1.0 / (short_factor * self.base**inv_freq_shape)
        long_inv_freq  = 1.0 / (long_factor  * self.base**inv_freq_shape)

        # Phi-3 Scale factor
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        pass
        self.scaling_factor = scaling_factor

        # Short and long inv_freq
        self.register_buffer("short_inv_freq", short_inv_freq, persistent = False)
        self.register_buffer("long_inv_freq",  long_inv_freq,  persistent = False)

        # Build here to make `torch.jit.trace` work.
        # Initialize short sequences cache for all devices
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        t = torch.arange(original_max_position_embeddings, device=self.short_inv_freq.device, dtype=torch.int64).float()
        freqs = torch.outer(t, self.short_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        for device_idx in range(DEVICE_COUNT):
            device_obj = torch.device(device_idx)
            cos_cached = (emb.cos() * self.scaling_factor).to(dtype=dtype, device=device_obj, non_blocking=True)
            sin_cached = (emb.sin() * self.scaling_factor).to(dtype=dtype, device=device_obj, non_blocking=True)
            self.multi_gpu_short_cos_cached[device_idx] = cos_cached
            self.multi_gpu_short_sin_cached[device_idx] = sin_cached

        # dummy so that patch_utils doesn't fail for now
        self.short_cos_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.short_sin_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.long_cos_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
        self.long_sin_cached = torch.empty(1, device=get_current_device(), dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        t = torch.arange(self.current_rope_size, device=self.long_inv_freq.device, dtype=torch.int64).float()
        # Long sequences
        freqs = torch.outer(t, self.long_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = (emb.cos() * self.scaling_factor).to(dtype=dtype, device=device, non_blocking=True)
        sin_cached = (emb.sin() * self.scaling_factor).to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_long_cos_cached[device.index] = cos_cached
        self.multi_gpu_long_sin_cached[device.index] = sin_cached
        return cos_cached, sin_cached
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index

        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return (
                self.multi_gpu_short_cos_cached[device_index][:seq_len],
                self.multi_gpu_short_sin_cached[device_index][:seq_len],
            )
        else:
            return (
                self.multi_gpu_long_cos_cached[device_index][:seq_len],
                self.multi_gpu_long_sin_cached[device_index][:seq_len],
            )
        pass
    pass

    def get_cached(self, seq_len = None, device_index = None):
        if device_index is None:
            device_index = get_current_device()
        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return self.multi_gpu_short_cos_cached[device_index], self.multi_gpu_short_sin_cached[device_index]
        return self.multi_gpu_long_cos_cached[device_index], self.multi_gpu_long_sin_cached[device_index]
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype)
    pass
pass


def unsloth_fast_generate(
    self,
    *args,
    **kwargs,
):
    FastLlamaModel.for_inference(self)

    dtype = _get_dtype(self.config.torch_dtype)

    if hasattr(self, "config") and hasattr(self.config, "max_position_embeddings"):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None and "max_new_tokens" in kwargs:
            if kwargs["input_ids"].shape[-1] + kwargs["max_new_tokens"] > self.config.max_position_embeddings:
                raise ValueError(
                    f'Unsloth: input length {kwargs["input_ids"].shape[-1]} + max_new_tokens {kwargs["max_new_tokens"]} exceeds the maximum sequence length of {self.config.max_position_embeddings}!\n'\
                    'You will need to do long context extension by increasing the `max_seq_length` in `FastLanguageModel.from_pretrained`.'
                )
    pass

    # Must patch accelerate for Xformers
    # if accelerate_new_send_to_device is not None:
    #     import accelerate.utils.operations
    #     accelerate.utils.operations.send_to_device = accelerate_new_send_to_device
    # pass

    # For newer HF
    kwargs["cache_implementation"] = "dynamic"
    # For num_logits_to_keep
    num_logits_to_keep = kwargs.get("num_logits_to_keep", None)
    logits_to_keep     = kwargs.get("logits_to_keep",     None)
    if num_logits_to_keep is None and logits_to_keep is None:
        kwargs["num_logits_to_keep"] = 1

    # Remove token_type_ids
    kwargs.pop("token_type_ids", None)

    # Check pad_token
    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Mixed precision autocast
    with torch.inference_mode(), torch.autocast(device_type = DEVICE_TYPE, dtype = dtype):
        output = self._old_generate(*args, **kwargs)
    pass

    # Return accelerate back
    # if accelerate_new_send_to_device is not None:
    #     accelerate.utils.operations.send_to_device = accelerate_old_send_to_device
    # pass

    FastLlamaModel.for_training(self)

    return output
pass


class FastLlamaModel:

    @staticmethod
    def pre_patch():
        init_name, function = patch_llama_rope_scaling(
            model_name           = "llama",
            rope_module          = LlamaRotaryEmbedding,
            scaled_rope_module   = LlamaLinearScalingRotaryEmbedding,
            extended_rope_module = LlamaExtendedRotaryEmbedding,
            attention_module     = LlamaAttention,
            longrope_module      = LongRopeRotaryEmbedding,
        )
        if init_name is not None:
            exec(function, globals())
            LlamaAttention.__init__  = eval(init_name)
        pass
        LlamaAttention      .forward = LlamaAttention_fast_forward
        LlamaSdpaAttention  .forward = LlamaAttention_fast_forward
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        LlamaModel          .forward = LlamaModel_fast_forward
        LlamaForCausalLM    .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(LlamaForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.llama.modeling_llama
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name         = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length     = None,
        dtype              = None,
        load_in_4bit       = True,
        token              = None,
        device_map         = "sequential",
        rope_scaling       = None,
        fix_tokenizer      = True,
        model_patcher      = None,
        tokenizer_name     = None,
        trust_remote_code  = False,
        revision           = None,

        fast_inference    = False, # uses vLLM
        gpu_memory_utilization = 0.5,
        float8_kv_cache   = False,
        random_state      = 3407,
        max_lora_rank     = 16,
        disable_log_stats = False,
        unsloth_vllm_standby = False,
        num_labels =  None,
        **kwargs,
    ):
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "0"
        if trust_remote_code:
            if fast_inference:
                raise NotImplementedError("Unsloth: Fast inference does not support `trust_remote_code` yet.")
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if fast_inference:
            if not is_vLLM_available():
                print("Unsloth: vLLM is not installed! Will use Unsloth inference!")
                fast_inference = False
            if DEVICE_TYPE == "cuda":
                major_version, minor_version = torch.cuda.get_device_capability()
                if major_version < 7:
                    print("Unsloth: vLLM does not work on older GPUs - will switch to Unsloth inference!")
                    fast_inference = False
            if unsloth_vllm_standby and os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0":
                raise RuntimeError("Unsloth: `unsloth_vllm_standby` is True, but  environment variable `UNSLOTH_VLLM_STANDBY` is not set to 1!")
        pass

        if token is None: token = get_token()
        if model_patcher is None: model_patcher = FastLlamaModel
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()

        if DEVICE_TYPE == "cuda":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_version = torch.version.cuda
            gpu_stats_snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."

            from importlib.metadata import version as importlib_version
            try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
            except: vllm_version = ""
        elif DEVICE_TYPE == "xpu":
            gpu_stats = torch.xpu.get_device_properties(0)
            gpu_version = torch.version.xpu
            gpu_stats_snippet = f"Intel Toolkit: {gpu_version}."

            try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
            except: vllm_version = ""
        else:
            raise ValueError(f"Unsloth: Unsupported device type: {DEVICE_TYPE}")

        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
        f"==((====))==  Unsloth {__version__}: Fast {model_patcher.__name__[4:-5]} patching. Transformers: {transformers_version}.{vllm_version}\n"\
        f"   {chr(92)}{chr(92)}   /|    {gpu_stats.name}. Num GPUs = {DEVICE_COUNT}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
        f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"\
        f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
        f' "-____-"     Free license: http://github.com/unslothai/unsloth'

        print(statistics)

        # Warn about fast transfers
        if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
            old_hf_transfer = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]
            if old_hf_transfer in ("False", "false"): old_hf_transfer = "0"
            if old_hf_transfer in ("True",  "true" ): old_hf_transfer = "1"
        else:
            old_hf_transfer = "0"
        if old_hf_transfer == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        if old_hf_transfer != "0": os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        model_patcher.pre_patch()
        get_statistics() # For debugging - we use a download counter to see if environments are not breaking

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        # elif dtype == torch.float16 and SUPPORTS_BFLOAT16:
        #     logger.warning_once("Device supports bfloat16 but you selected float16. Will change to bfloat16.")
        #     dtype = torch.bfloat16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        # RoPE Scaling
        model_config = AutoConfig.from_pretrained(
            model_name,
            token = token,
            attn_implementation = "sdpa",
        )
        model_max_seq_length = model_config.max_position_embeddings

        # Check if RoPE Scaling is even allowed
        model_function = MODEL_FOR_CAUSAL_LM_MAPPING[model_config.__class__]
        IS_FALCON_H1 = model_config.model_type.startswith("falcon_h1")

        has_rope_scaling = False
        try:
            with open(inspect.getfile(model_function), "r") as file:
                has_rope_scaling = "self.config.rope_scaling" in file.read()
        except: pass
        has_rope_scaling = True

        # If max_seq_length is not specified, use maximum from config
        if max_seq_length is None:
            max_seq_length = model_max_seq_length
        pass

        if (rope_scaling is None) and (max_seq_length > model_max_seq_length):

            rope_scaling = max_seq_length / model_max_seq_length

            if fast_inference:
                raise NotImplementedError("Unsloth: Fast inference does not yet work with RoPE Scaling.")

            logger.warning_once(
                f"Unsloth: {model_name} can only handle sequence lengths of at most "\
                f"{model_max_seq_length}.\nBut with kaiokendev's RoPE scaling of "\
                f"{round(rope_scaling, 3)}, it can be magically be extended to "\
                f"{max_seq_length}!"
            )

            # Warn RoPE scaling isn't allowed
            if not has_rope_scaling:
                raise RuntimeError(
                    f"However, {model_name} doesn't support RoPE Scaling!\n"\
                    "Please file a feature request at https://github.com/unslothai/unsloth."
                )
            pass

            rope_scaling = {"type": "linear", "factor": rope_scaling,}

            # Add to kwargs
            kwargs["rope_scaling"] = rope_scaling
        pass

        bnb_config = None
        if load_in_4bit:
            llm_int8_skip_modules =  SKIP_QUANTIZATION_MODULES.copy()
            if IS_FALCON_H1:
                # we cannot quantize out_proj layer due to mamba kernels: https://github.com/tiiuae/Falcon-H1/issues/13#issuecomment-2918671274
                llm_int8_skip_modules.append("out_proj")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                llm_int8_skip_modules     = llm_int8_skip_modules,
            )
        pass

        # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/12
        # RoPE Scaling's max_position_embeddings must be updated
        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        kwargs.pop("attn_implementation", None); # No need since we auto call it

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config

        raise_handler = RaiseUninitialized()
        if num_labels is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map              = device_map,
                torch_dtype             = dtype,
                num_labels              = num_labels,
                #quantization_config     = bnb_config,
                token                   = token,
                max_position_embeddings = max_position_embeddings,
                trust_remote_code       = trust_remote_code,
                attn_implementation     = "eager",
                **kwargs,
            )
        elif not fast_inference:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map              = device_map,
                torch_dtype             = dtype,
                # quantization_config     = bnb_config,
                token                   = token,
                max_position_embeddings = max_position_embeddings,
                trust_remote_code       = trust_remote_code,
                attn_implementation     = "eager",
                **kwargs,
            )
            model.fast_generate = model.generate
            model.fast_generate_batches = None
        else:
            from unsloth_zoo.vllm_utils import (
                load_vllm,
                get_vllm_state_dict,
                convert_vllm_to_huggingface,
                generate_batches,
            )
            allowed_args = inspect.getfullargspec(load_vllm).args
            load_vllm_kwargs = dict(
                model_name             = model_name,
                config                 = model_config,
                gpu_memory_utilization = gpu_memory_utilization,
                max_seq_length         = max_seq_length,
                dtype                  = dtype,
                float8_kv_cache        = float8_kv_cache,
                enable_lora            = True,
                max_lora_rank          = max_lora_rank,
                disable_log_stats      = disable_log_stats,
                use_bitsandbytes       = load_in_4bit,
                unsloth_vllm_standby   = unsloth_vllm_standby,
            )
            for allowed_arg in allowed_args:
                if allowed_arg not in load_vllm_kwargs and allowed_arg in kwargs:
                    load_vllm_kwargs[allowed_arg] = kwargs[allowed_arg]
            pass

            # Load vLLM first
            llm = load_vllm(**load_vllm_kwargs)

            # Convert to HF format
            _, quant_state_dict = get_vllm_state_dict(llm, config = model_config)
            model = convert_vllm_to_huggingface(quant_state_dict, model_config, dtype, bnb_config)
            model.vllm_engine = llm
            model.fast_generate = model.vllm_engine.generate
            model.fast_generate_batches = functools.partial(generate_batches, model.vllm_engine)
        pass
        raise_handler.remove()
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        tokenizer = load_correct_tokenizer(
            tokenizer_name    = tokenizer_name,
            model_max_length  = max_position_embeddings,
            padding_side      = "right",
            token             = token,
            trust_remote_code = trust_remote_code,
            fix_tokenizer     = fix_tokenizer,
        )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model, tokenizer = model_patcher.post_patch(model, tokenizer)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        # Patch Trainer
        from transformers.trainer import Trainer
        try:
            if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
                inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
                Trainer._original_training_loop = inner_training_loop
            else:
                inner_training_loop = Trainer._original_training_loop
        except:
            raise RuntimeError('Unsloth: Unsuccessfully patched inner_training_loop')
        pass

        import transformers.trainer
        items_in_trainer = dir(transformers.trainer)
        good_items = []
        for item in items_in_trainer:
            if item in inner_training_loop: good_items.append(item)
        pass
        exec("from transformers.trainer import (" + ", ".join(x for x in good_items) + ")", globals())

        start = re.search(r'logger\.info\([\"\'].+?Running training', inner_training_loop).span(0)[0]
        end = inner_training_loop.find("\n\n", start)
        original_debug = inner_training_loop[start:end]
        spaces = re.search(r'\n([\s\t]{1,})', original_debug).group(0)[1:]
        front_spaces = re.match(r'([\s\t]{1,})', inner_training_loop).group(0)

        # Cannot use \\ since it will cause a SyntaxWarning in Python 3.12
        # Instead use chr(92) == \\
        debug_info = """debug_info = \\
        f"==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = {len(set(p.device for p in model.parameters()))}\\n"\\
        f"   {chr(92)}{chr(92)}   /|    Num examples = {num_examples:,} | Num Epochs = {num_train_epochs:,} | Total steps = {max_steps:,}\\n"\\
        f"O^O/ {chr(92)}_/ {chr(92)}    Batch size per device = {self._train_batch_size:,} | Gradient accumulation steps = {args.gradient_accumulation_steps}\\n"\\
        f"{chr(92)}        /    Data Parallel GPUs = {args.world_size} | Total batch size ({self._train_batch_size} x {args.gradient_accumulation_steps} x {args.world_size}) = {total_train_batch_size:,}\\n"\\
        f' "-____-"     Trainable parameters = {get_model_param_count(model, trainable_only=True):,} of {get_model_param_count(model):,} ({get_model_param_count(model, trainable_only=True)/get_model_param_count(model)*100:.2f}% trained)'
        logger.warning(debug_info)
        import gc
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
            else:
                torch.cuda.empty_cache()"""

        debug_info = debug_info.split('\n')
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

        debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once('Unsloth is running with multi GPUs - the effective batch size is multiplied by ' + str(n_total_devices))
        debug_info ="""
        debug_info = debug_info.split('\n')
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace("debug_info =", debug_info, 1)

        front_spaces = re.match(r"[\t\s]{1,}", inner_training_loop).group(0)
        inner_training_loop = re.sub(r"^" + front_spaces, "", inner_training_loop, flags = re.MULTILINE)
        inner_training_loop = inner_training_loop.replace(
            "train_dataloader = tpu_spmd_dataloader(train_dataloader)",
            "raise RuntimeError('Unsloth: TPUs are not yet supported!')"
        )
        inner_training_loop = inner_training_loop.replace(
            "_inner_training_loop",
            "_fast_inner_training_loop", 1,
        )
        inner_training_loop = inner_training_loop.replace(
            "is_torch_tpu_available()",
            "False",
        )
        exec(inner_training_loop, globals())
        Trainer._inner_training_loop = _fast_inner_training_loop

        # Save max_seq_length
        model.max_seq_length = max_seq_length
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m = m.model
        pass
        m.max_seq_length = max_seq_length

        # We check the tokenizer first for errors
        if fix_tokenizer:
            tokenizer = check_tokenizer(
                model            = model,
                tokenizer        = tokenizer,
                model_name       = model_name,
                model_max_length = max_position_embeddings,
                padding_side     = "right",
                token            = token,
            )
        pass
        patch_saving_functions(tokenizer)

        # Fix up config for transformers uploading PEFT
        # Not necessary anymore since we require transformers>=4.37!
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[:len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path" : name})
            pass
        pass

        # Log Unsloth version for future fastpaths for inference
        model.config.update({"unsloth_version" : __version__})

        # Add save modules
        patch_saving_functions(model)
        Trainer._inner_training_loop = _fast_inner_training_loop

        # Fix gradient accumulation
        patch_gradient_accumulation_fix(Trainer)

        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left" # Force inference
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model._saved_temp_tokenizer = tokenizer
            # Also set is_loaded_in_8bit to disable incorrect DDP
            internal_model.is_loaded_in_8bit = True

            internal_model = internal_model.model
        pass
        internal_model._saved_temp_tokenizer = tokenizer
        # Also set is_loaded_in_8bit to disable incorrect DDP
        internal_model.is_loaded_in_8bit = True

        # For transformers > 4.47.1, we need to add rotary_emb to all attention layers
        if IS_ATTENTION_REFACTOR or hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = rotary_emb
        pass

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastLlamaModel.for_training,  model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)

        # Patch generate
        is_classification =  "Classification" in str(type(model))
        if not is_classification and model.generate.__name__ != "unsloth_fast_generate":
            model._old_generate = model.generate
            unsloth_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_fast_generate, model)
        pass
        return model, tokenizer
    pass


    @staticmethod
    def post_patch(model, tokenizer):
        model, tokenizer = patch_model_and_tokenizer(model, tokenizer, downcast_rope = True)
        return model, tokenizer
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                   = 16,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        lora_alpha          = 16,
        lora_dropout        = 0.0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = "unsloth",
        random_state        = 3407,
        max_seq_length      = 2048, # not used anymore
        use_rslora          = False,
        modules_to_save     = None,
        init_lora_weights   = True,
        loftq_config        = {},
        temporary_location  = "_unsloth_temporary_saved_buffers",
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            # Check for other PEFT args in kwargs
            for (peft_arg, flag) in (
                ("finetune_vision_layers", False),
                ("finetune_language_layers", True),
                ("finetune_attention_modules", True),
                ("finetune_mlp_modules", True),
            ):
                if peft_arg not in kwargs: kwargs[peft_arg] = flag
            return FastBaseModel.get_peft_model(
                model                      = model,
                r                          = r,
                target_modules             = target_modules,
                lora_alpha                 = lora_alpha,
                lora_dropout               = lora_dropout,
                bias                       = bias,
                layers_to_transform        = layers_to_transform,
                layers_pattern             = layers_pattern,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state               = random_state,
                max_seq_length             = max_seq_length,
                use_rslora                 = use_rslora,
                modules_to_save            = modules_to_save,
                init_lora_weights          = init_lora_weights,
                loftq_config               = loftq_config,
                temporary_location         = temporary_location,
                **kwargs,
            )
        pass
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        pass
        transformers_set_seed(random_state)

        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(dtype = model.get_input_embeddings().weight.dtype)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftModelForSequenceClassification):
            # Check if exactly the same and then pass through!
            assert(hasattr(model, "peft_config"))

            peft_config = model.peft_config["default"].to_dict()
            check_parameters = [
                "r", "lora_alpha", "lora_dropout",
                "bias", "layers_to_transform", "layers_pattern",
                "use_rslora", "init_lora_weights",
            ]
            check_all = True
            for param in check_parameters:
                check_all = check_all and (peft_config[param] == eval(param))
            pass

            # Check save_modules
            old_target_modules = list(peft_config["target_modules"])
            modules_to_save = peft_config["modules_to_save"]
            if modules_to_save is None: modules_to_save = {}
            modules_to_save = list(modules_to_save)
            old_target_modules += modules_to_save

            # Combine all
            new_target_modules = list(target_modules) + \
                list(modules_to_save if modules_to_save is not None else [])

            # Now check!
            new_target_modules = set(new_target_modules)
            check_all = check_all and (
                len(set(old_target_modules) ^ new_target_modules) == 0
            )

            check_all = check_all and (
                (loftq_config == {} or loftq_config is None) and \
                (peft_config["loftq_config"] == {} or peft_config["loftq_config"] is None)
            )

            if check_all:
                # Simply pass through!
                logger.warning(
                    "Unsloth: Already have LoRA adapters! We shall skip this step."
                )

                # Offload!
                # [TODO] First offload lm_head and embed_tokens to CPU (should be disk!!)
                if "embed_tokens" in new_target_modules:
                    print("Unsloth: Training embed_tokens in mixed precision to save VRAM")

                    new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype
                    if new_dtype == torch.float16:
                        # See https://github.com/unslothai/unsloth/pull/1200
                        # Tesla T4 must use float32 and not float16
                        new_dtype = torch.float32
                    pass

                    model.get_input_embeddings().modules_to_save.default\
                        .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
                    model.get_input_embeddings().modules_to_save.default.requires_grad_(True)

                    # [TODO] Move old embed_tokens to CPU - should be disk!
                    model.get_input_embeddings().original_module\
                        .to(device = "cpu", non_blocking = True)
                    model.get_input_embeddings().original_module.requires_grad_(False)
                pass

                if "lm_head" in new_target_modules:
                    print("Unsloth: Training lm_head in mixed precision to save VRAM")

                    new_dtype = model.get_output_embeddings().modules_to_save.default.weight.dtype
                    if new_dtype == torch.float16:
                        # See https://github.com/unslothai/unsloth/pull/1200
                        # Tesla T4 must use float32 and not float16
                        new_dtype = torch.float32
                    pass

                    model.get_output_embeddings().modules_to_save.default\
                        .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
                    model.get_output_embeddings().modules_to_save.default.requires_grad_(True)

                    # [TODO] Move old lm_head to CPU - should be disk!
                    model.get_output_embeddings().original_module\
                        .to(device = "cpu", non_blocking = True)
                    model.get_output_embeddings().original_module.requires_grad_(False)
                pass

                return model
            else:
                raise TypeError(
                    "Unsloth: Your model already has LoRA adapters. Your new parameters are different."
                )
            pass
        pass

        if loftq_config is None: loftq_config = {}

        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ  = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora"   in signature

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if not (type(init_lora_weights) is bool or \
            init_lora_weights == "gaussian" or init_lora_weights == "loftq"):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
            )
        pass

        if init_lora_weights == "loftq":

            if not SUPPORTS_LOFTQ:
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass

            if loftq_config == {}:
                from peft import LoftQConfig
                logger.warning_once(
                    "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"\
                    "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)
            pass

            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"\
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )
            pass
        pass

        assert(type(use_rslora) is bool)
        if use_rslora:
            if not SUPPORTS_RSLORA:
                # We manually check for PEFT
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass
        pass

        accepted_modules = frozenset(("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj",),)
        model.config.update({"unsloth_version" : __version__})

        if type(modules_to_save) is tuple:
            modules_to_save = list(modules_to_save)
        pass

        train_lm_head = False
        train_embed_tokens = False
        final_modules = []
        for module in target_modules:
            if module == "lm_head":
                # logger.warning_once(
                #     "Unsloth: `lm_head` should be placed in `modules_to_save` and not `target_modules`. "\
                #     "Luckily, we shall do it for you!"
                # )
                train_lm_head = True
                if modules_to_save is None: modules_to_save = ["lm_head"]
                else: modules_to_save.append("lm_head")

            elif module == "embed_tokens":
                # logger.warning_once(
                #     "Unsloth: `embed_tokens` should be placed in `modules_to_save` and not `target_modules`. "\
                #     "Luckily, we shall do it for you!"
                # )
                train_embed_tokens = True
                if modules_to_save is None: modules_to_save = ["embed_tokens"]
                else: modules_to_save.append("embed_tokens")

            else:
                try:
                    assert(module in accepted_modules)
                    final_modules.append(module)
                except AssertionError as e:
                    final_modules.append(module)
                    print(
                        "Unsloth: You added custom modules, but Unsloth hasn't optimized for this.\n"\
                        "Beware - your finetuning might be noticeably slower!"
                    )
                pass
            pass
        pass

        # Check if we added new tokens!
        if hasattr(model, "_need_to_train_embeddings"):
            if not train_lm_head or not train_embed_tokens:
                print(
                    "Unsloth: You added new tokens but did not specify if you wanted to "\
                    "train the lm_head and embed_tokens.\nWe must turn it on for you."
                )
                train_lm_head = True
                train_embed_tokens = True

                if modules_to_save is None: modules_to_save = ["embed_tokens"]
                else: modules_to_save.append("embed_tokens")

                if modules_to_save is None: modules_to_save = ["lm_head"]
                else: modules_to_save.append("lm_head")
            pass
        pass

        # Check for Llama-3
        # if hasattr(model._saved_temp_tokenizer, "_using_llama3_template"):
        #     if not train_embed_tokens and not train_lm_head:
        #         raise RuntimeError("")

        # First fix untrained tokens
        # Wrong - can cause reserved tokens to pop out!!
        # if train_embed_tokens or train_lm_head:
        #     fix_untrained_tokens(model, eps = 1e-16)
        # pass

        # Check modules_to_save
        if modules_to_save is not None:
            for module in modules_to_save:
                if module == "lm_head":
                    train_lm_head = True
                elif module == "embed_tokens":
                    train_embed_tokens = True
                else:
                    raise TypeError(
                        f"Unsloth: Module = {module} is not allowed. Only 'lm_head' and 'embed_tokens' is allowed."
                    )
            pass
        pass
        if isinstance(modules_to_save, (tuple, list)):
            modules_to_save = list(set(modules_to_save))
        pass

        vllm_engine = None
        if hasattr(model, "vllm_engine"):
            # Fast inference!
            vllm_engine = model.vllm_engine
            vllm_fast_generate = model.fast_generate
            vllm_fast_generate_batches = model.fast_generate_batches

            if modules_to_save is not None:
                raise NotImplementedError("Unsloth: Currently fast inference does not work with training embeddings or lm_head.")

            if bias != "none":
                raise NotImplementedError("Unsloth: Currently fast inference does not work with using biases for LoRA.")
        pass

        #d oes not get lora yet, so get name from model, not base model
        is_classification = "Classification" in str(type(model))

        arguments = dict(
            r                   = r,
            lora_alpha          = lora_alpha,
            target_modules      = final_modules,
            lora_dropout        = lora_dropout,
            bias                = bias,
            task_type           = TaskType.CAUSAL_LM if not is_classification else TaskType.SEQ_CLS,
            layers_to_transform = layers_to_transform,
            init_lora_weights   = init_lora_weights,
            loftq_config        = loftq_config,
            use_rslora          = use_rslora,
            modules_to_save     = modules_to_save,
            **kwargs,
        )
        if not SUPPORTS_LOFTQ:  del arguments["loftq_config"]
        if not SUPPORTS_RSLORA: del arguments["use_rslora"]

        _saved_temp_tokenizer = model._saved_temp_tokenizer

        lora_config = LoraConfig(**arguments)
        # First offload lm_head and embed_tokens to disk
        input_embeddings_device  = model.get_input_embeddings().weight.device
        if is_classification:
             output_embeddings_device = model.score.weight.device
        else:
            output_embeddings_device = model.get_output_embeddings().weight.device

        if use_gradient_checkpointing == "unsloth":
            if train_embed_tokens:
                print("Unsloth: Offloading input_embeddings to disk to save VRAM")
                offload_input_embeddings(model, temporary_location)
            pass

            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()
            pass

            if train_lm_head:
                print("Unsloth: Offloading output_embeddings to disk to save VRAM")
                offload_output_embeddings(model, temporary_location)
            pass

            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()
            pass
        pass

        model = _get_peft_model(model, lora_config)

        model._saved_temp_tokenizer = _saved_temp_tokenizer

        model = FastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)

        if train_embed_tokens:
            print("Unsloth: Training embed_tokens in mixed precision to save VRAM")
            assert(hasattr(model.get_input_embeddings(), "modules_to_save"))

            new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32
            pass

            model.get_input_embeddings().modules_to_save.default\
                .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
            model.get_input_embeddings().modules_to_save.default.requires_grad_(True)
        pass

        if train_lm_head:
            print("Unsloth: Training lm_head in mixed precision to save VRAM")
            assert(hasattr(model.get_output_embeddings(), "modules_to_save"))

            new_dtype = model.get_output_embeddings().modules_to_save.default.weight.dtype
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32
            pass

            model.get_output_embeddings().modules_to_save.default\
                .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
            model.get_output_embeddings().modules_to_save.default.requires_grad_(True)
        pass

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            pass
            # Also set is_loaded_in_8bit to disable incorrect DDP
            internal_model.is_loaded_in_8bit = True
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        pass
        # Also set is_loaded_in_8bit to disable incorrect DDP
        internal_model.is_loaded_in_8bit = True

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            clean_gpu_cache()
        pass

        # Patch for fast inference
        if vllm_engine is not None:
            model.vllm_engine = vllm_engine
            model.fast_generate = vllm_fast_generate
            model.fast_generate_batches = vllm_fast_generate_batches

            # Also saving and loading LoRA
            from unsloth_zoo.vllm_utils import save_lora, load_lora
            model.save_lora = functools.partial(save_lora, model)
            model.load_lora = functools.partial(load_lora, model)
        pass

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastLlamaModel.for_training,  model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        return model
    pass


    @staticmethod
    def patch_peft_model(
        model,
        use_gradient_checkpointing = "unsloth",
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            return FastBaseModel.patch_peft_model(
                model = model,
                use_gradient_checkpointing = use_gradient_checkpointing,
            )
        pass
        if not isinstance(model, PeftModelForCausalLM) and not isinstance(model, PeftModelForSequenceClassification):
            raise TypeError(
                "Unsloth: Your model needs to call `.get_peft_model` first!"
            )
        pass

        # Get activation function
        model_type = model.config.model_type

        if   model_type == "llama":     apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "mistral":   apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen2":     apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "gemma":     apply_lora_mlp = apply_lora_mlp_geglu_approx
        elif model_type == "gemma2":    apply_lora_mlp = apply_lora_mlp_geglu_approx
        elif model_type == "cohere":    apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "granite":   apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen3":     apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "falcon_h1": apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen3moe":  apply_lora_mlp = apply_lora_mlp_swiglu
        else:
            raise NotImplementedError(f"Unsloth: {model_type} is not yet implemented!")
        pass

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )

        # Fix up config for transformers uploading PEFT
        for active_adapter in model.peft_config.keys():
            # Not necessary since we requires transformers >= 4.37
            if False:
                name = model.peft_config[active_adapter].base_model_name_or_path
                if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                    name = name[:len(name) - len("-bnb-4bit")]
                    model.peft_config[active_adapter].base_model_name_or_path = name
                pass
            # Add revision to enable future fast inference paths
            # [TODO] Bugs out!see https://github.com/unslothai/unsloth/issues/492
            # model.peft_config[active_adapter].revision = f"unsloth"
        pass

        from transformers.trainer import Trainer
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            raise RuntimeError("Unsloth: Unsuccessfully patched Trainer! Please file a bug report!")
        pass

        # Fix loftq issues
        # loftq_config must not = None, but rather {}
        all_configs = model.peft_config
        for key, current_config in all_configs.items():
            if hasattr(current_config, "loftq_config") and current_config.loftq_config is None:
                new_args = current_config.__dict__
                new_args["loftq_config"] = {}
                current_config = current_config.__class__(**new_args)
                all_configs[key] = current_config
            pass
        pass

        # Do patching
        n_mlp = 0
        n_qkv = 0
        n_o   = 0

        active_adapter = model.active_adapters[0] if \
            hasattr(model, "active_adapters") else model.active_adapter

        # Get dropout and bias
        lora_dropout = model.peft_config[active_adapter].lora_dropout
        bias         = model.peft_config[active_adapter].bias

        # We also do not inplace edit QKV for Cohere!
        _apply_lora_mlp = \
            functools.partial(apply_lora_mlp, inplace = False) \
            if model_type == "cohere" else \
            apply_lora_mlp
        pass

        if lora_dropout == 0 and bias == "none":
            for idx, layer in enumerate(model.model.model.layers):

                if model_type != "falcon_h1":
                    # LoRAMLP.apply doesn't have functionality for gate and down mutlipliers yet.
                    # Don't patch falcon h1 for the time being.

                    # MLP patching
                    mlp_module = layer.mlp
                    gate_proj = mlp_module.gate_proj
                    up_proj   = mlp_module.  up_proj
                    down_proj = mlp_module.down_proj

                    if hasattr(gate_proj, "lora_A") and \
                        hasattr(  up_proj, "lora_A") and \
                        hasattr(down_proj, "lora_A") and \
                        (getattr(gate_proj, "base_layer", gate_proj).bias is None) and \
                        (getattr(  up_proj, "base_layer",   up_proj).bias is None) and \
                        (getattr(down_proj, "base_layer", down_proj).bias is None) and \
                        (len(getattr(gate_proj, "lora_magnitude_vector", []) or []) == 0) and \
                        (len(getattr(  up_proj, "lora_magnitude_vector", []) or []) == 0) and \
                        (len(getattr(down_proj, "lora_magnitude_vector", []) or []) == 0):

                        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
                        mlp_module.forward = types.MethodType(_apply_lora_mlp, mlp_module)
                        n_mlp += 1
                    else:
                        logger.warning_once(
                            "Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n"\
                            "are not enabled or a bias term (like in Qwen) is used."
                        )
                    pass
                pass

                # QKV attention patching
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                if  hasattr(q_proj, "lora_A") and \
                    hasattr(k_proj, "lora_A") and \
                    hasattr(v_proj, "lora_A") and \
                    (getattr(q_proj, "base_layer", q_proj).bias is None) and \
                    (getattr(k_proj, "base_layer", k_proj).bias is None) and \
                    (getattr(v_proj, "base_layer", v_proj).bias is None) and \
                    (len(getattr(q_proj, "lora_magnitude_vector", []) or []) == 0) and \
                    (len(getattr(k_proj, "lora_magnitude_vector", []) or []) == 0) and \
                    (len(getattr(v_proj, "lora_magnitude_vector", []) or []) == 0):

                    layer.self_attn.apply_qkv = apply_lora_qkv
                    n_qkv += 1
                else:
                    if model_type == "qwen2": n_qkv += 1
                    else:
                        logger.warning_once(
                            "Not an error, but Unsloth cannot patch Attention layers with our manual autograd engine since either LoRA adapters\n"\
                            "are not enabled or a bias term (like in Qwen) is used."
                        )
                    pass
                pass

                # O attention patching
                o_proj = layer.self_attn.o_proj
                if hasattr(o_proj, "lora_A") and \
                    (getattr(o_proj, "base_layer", o_proj).bias is None) and \
                    (len(getattr(o_proj, "lora_magnitude_vector", []) or []) == 0):

                    layer.self_attn.apply_o = apply_lora_o
                    n_o += 1
                else:
                    logger.warning_once(
                        "Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n"\
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass
            pass
        pass

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "\
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )
        patch_saving_functions(model)

        # Patch cross entropy loss labels
        # Fixes https://github.com/unslothai/unsloth/issues/10
        max_seq_length = model.max_seq_length
        # extra_ignored_labels = torch.full((max_seq_length, 1), -100, device = "cuda:0")
        # model.model.extra_ignored_labels = extra_ignored_labels
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_seq_length

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            clean_gpu_cache()
        pass

        # Patch for fast inference
        vllm_engine = getattr(model.model, "vllm_engine", None)
        if vllm_engine is not None:
            model.vllm_engine = model.model.vllm_engine
            model.fast_generate = model.model.fast_generate
            model.fast_generate_batches = model.model.fast_generate_batches

            # Also saving and loading LoRA
            from unsloth_zoo.vllm_utils import save_lora, load_lora
            model.save_lora = functools.partial(save_lora, model)
            model.load_lora = functools.partial(load_lora, model)
        pass

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastLlamaModel.for_training,  model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        return model
    pass


    @staticmethod
    def for_inference(model):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_inference!")

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = False
            if hasattr(m, "training"): m.training = False
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "left"
            # Set a flag for generation!
            m._flag_for_generation = True
        pass
        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)

        # Since transformers 4.53, must turn off explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
        pass

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_training!")

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"): m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"): del m._flag_for_generation
        pass
        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)

        # Since transformers 4.53, must turn on explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = use_gradient_checkpointing
        pass

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        return model
    pass
pass

from .rl import PatchFastRL
PatchFastRL(FastLanguageModel = FastLlamaModel)

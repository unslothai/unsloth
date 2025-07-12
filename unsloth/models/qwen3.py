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

from .llama import *
import os
from ._utils import __version__
from unsloth_zoo.utils import Version, _get_dtype
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    _LlamaModel_fast_forward_inference,
)
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3DecoderLayer,
        Qwen3Model,
        Qwen3ForCausalLM,
    )
except:
    transformers_version = Version(transformers_version)
    if not transformers_version >= Version("4.50.3"): #TODO: Update when transformers is updated
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3 and Qwen3Moe.\n"\
            f"The minimum required version is 4.50.3.\n"\
            f'Try `pip install --upgrade "transformers>=4.50.3"`\n'\
            f"to obtain the latest transformers build, then restart this session."\
        )
    pass
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3SdpaAttention,
        Qwen3FlashAttention2,
    )
except:
    Qwen3SdpaAttention   = Qwen3Attention
    Qwen3FlashAttention2 = Qwen3Attention
pass


def Qwen3Attention_fast_forward(
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
    Q = Q.view(bsz, q_len, n_heads,    head_dim)#.transpose(1, 2) # we will transpose after normalisation
    K = K.view(bsz, q_len, n_kv_heads, head_dim)#.transpose(1, 2) # we will transpose after normalisation
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    #Qwen3 has QKNorm. This seems to be the only difference from Qwen2.
    # Note that using fast_layernorm_compiled causes issues as the dimensions don't match up.
    # I tried to add a compiled version of the new norm but the numbers don't match up with Transformers
    # TODO: Check on the differences here.
    Q = fast_rms_layernorm(self.q_norm, Q)
    K = fast_rms_layernorm(self.k_norm, K)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)
        device_index = Q.device.index

        if position_ids is None:
            # Useful for LongRoPE
            cos, sin = rotary_emb.get_cached(kv_seq_len, device_index)
        else:
            cos, sin = rotary_emb.get_cached(kv_seq_len, device_index)
    Q, K = fast_rope_embedding(Q, K, cos, sin)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and HAS_XFORMERS and attention_mask is None):
        # Xformers memory efficient attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        K_M = V_M = bsz * kv_seq_len
        Q_M = bsz * q_len

        has_swa = isinstance(causal_mask, xformers.attn_bias.BlockDiagonalCausalMask)

        # Group query attention
        K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        if hidden_states.requires_grad:
            K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
            V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_heads, head_dim)
                K = K.view(1, K_M, n_heads, head_dim)
                V = V.view(1, V_M, n_heads, head_dim)
            pass
        else:
            # Xformers does support the forward pass though
            Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_kv_heads, n_groups, head_dim)
                K = K.view(1, K_M, n_kv_heads, n_groups, head_dim)
                V = V.view(1, V_M, n_kv_heads, n_groups, head_dim)
            pass
        pass

        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        sw = kv_seq_len
        window = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)
        A = flash_attn_func(Q, K, V, causal = True, window_size = window)
    else:
        # Grouped query attention
        # if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        # pass
        # Must be contiguous or else results are False!
        # https://github.com/pytorch/pytorch/issues/112577
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass

    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass

torch_matmul = torch.matmul
def Qwen3Attention_fast_forward_inference(
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
    Qn = Qn.view(bsz, 1, n_heads,    head_dim)#.transpose(1, 2) # we will transpose after normalisation
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim)#.transpose(1, 2) # we will transpose after normalisation
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)

    Qn = fast_rms_layernorm_inference(self.q_norm, Qn)
    Kn = fast_rms_layernorm_inference(self.k_norm, Kn)

    Qn = Qn.transpose(1, 2)
    Kn = Kn.transpose(1, 2)

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
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False, enable_gqa = True)
        else:
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False)
    pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass

class FastQwen3Model(FastLlamaModel):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "Qwen3",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = Qwen3Attention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen3Attention.__init__  = eval(init_name)
        pass
        Qwen3Attention      .forward = Qwen3Attention_fast_forward
        Qwen3SdpaAttention  .forward = Qwen3Attention_fast_forward
        Qwen3FlashAttention2.forward = Qwen3Attention_fast_forward
        Qwen3DecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        Qwen3Model          .forward = LlamaModel_fast_forward
        Qwen3ForCausalLM    .forward = CausalLM_fast_forward(_LlamaModel_fast_forward_inference(Qwen3Attention_fast_forward_inference))
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(Qwen3ForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.qwen3.modeling_qwen3
        transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding = LlamaRotaryEmbedding
        return
    pass


    @staticmethod
    def from_pretrained(  #TODO: Change after release
        model_name        = "Qwen/Qwen3-7B",
        max_seq_length    = 4096,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None,
        fix_tokenizer     = True,
        model_patcher     = None,
        tokenizer_name    = None,
        trust_remote_code = False,
        **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastQwen3Model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
    pass
pass

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
from ._utils import __version__
try:
    from transformers.models.cohere.modeling_cohere import (
        CohereAttention,
        CohereDecoderLayer,
        CohereModel,
        CohereForCausalLM,
        CohereRotaryEmbedding,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except:
    from packaging.version import Version
    transformers_version = Version(transformers_version)
    if not transformers_version >= Version("4.42"):
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Cohere.\n"\
            f"The minimum required version is 4.42.3.\n"\
            f'Try `pip install --upgrade "transformers>=4.42.3"`\n'\
            f"to obtain the latest transformers build, then restart this session."\
        )
    pass
pass

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.cohere.modeling_cohere import (
        CohereSdpaAttention,
        CohereFlashAttention2,
    )
except:
    CohereSdpaAttention   = CohereAttention
    CohereFlashAttention2 = CohereAttention
pass


def fast_layernorm_inference(self, X, out_weight = None):
    XX = X.to(torch.float32, copy = True)
    XX -= X.mean(-1, keepdim = True)
    variance = XX.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()
    out_weight[:] = self.weight
    XX *= out_weight
    return XX.to(X.dtype)
pass


# QK norm in Cohere
def CohereAttention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
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
        del self.q_norm_out_weight
        del self.k_norm_out_weight
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
    if self.use_qk_norm:
        Q = fast_layernorm_compiled(self.q_norm, Q)
        K = fast_layernorm_compiled(self.k_norm, K)
    pass

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = position_embeddings
    if position_ids is None:
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = cos[position_ids], sin[position_ids]
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

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
        # Grouped query attention
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
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass
    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def CohereDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
):
    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        out_weight = torch.empty(self.input_layernorm.weight.shape, dtype = torch.float32, device = "cuda:0")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_layernorm_inference(self.input_layernorm, hidden_states, out_weight)
        hidden_states_attention, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        # Fully Connected
        hidden_states_mlp = fast_swiglu_inference(self.mlp, hidden_states)
        residual += hidden_states_attention
        residual += hidden_states_mlp
        hidden_states = residual
    else:
        residual = hidden_states
        hidden_states = fast_layernorm_compiled(self.input_layernorm, hidden_states)
        hidden_states_attention, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass


from math import sqrt as math_sqrt
KV_CACHE_INCREMENT = 256 # KV Cache update size
torch_nn_functional_softmax = torch.nn.functional.softmax
torch_matmul = torch.matmul

def CohereAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
):

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
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = "cuda:0")
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = "cuda:0")
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = "cuda:0")
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = "cuda:0")

        # Mistral Nemo 12b has weird dimensions
        if attention_size != hidden_size:
            self.temp_O = torch.empty((1, bsz, hidden_size), dtype = dtype, device = "cuda:0")
        else:
            self.temp_O = self.temp_QA[1][:,:,:hidden_size]
        pass

        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = "cuda:0")
        self.scalar = 1.0 / math_sqrt(self.head_dim)
        self.half_head_dim = head_dim // 2
        # Cohere has QK layernorms
        if self.use_qk_norm:
            self.q_norm_out_weight = torch.empty(self.q_norm.weight.shape, dtype = torch.float32, device = "cuda:0")
            self.k_norm_out_weight = torch.empty(self.k_norm.weight.shape, dtype = torch.float32, device = "cuda:0")
        else:
            self.q_norm_out_weight = None
            self.k_norm_out_weight = None
        pass
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
    if self.use_qk_norm:
        Q = fast_layernorm_inference(self.q_norm, Q, self.q_norm_out_weight)
        K = fast_layernorm_inference(self.k_norm, K, self.k_norm_out_weight)
    pass

    # cos, sin = self.rotary_emb(Vn, seq_len = kv_seq_len)
    # Qn, Kn = inplace_rope_embedding(Qn, Kn, cos, sin, position_ids)
    cos, sin = self.rotary_emb.get_cached(kv_seq_len, Qn.device.index)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    h = self.half_head_dim

    RH_Q = self.RH_Q
    RH_Q[:,:,:,:h] = Qn[:,:,:,h:]
    RH_Q[:,:,:,h:] = Qn[:,:,:,:h]
    torch.neg(RH_Q[:,:,:,:h], out = RH_Q[:,:,:,:h])
    Qn *= cos
    Qn.addcmul_(RH_Q, sin)

    RH_K = RH_Q[:,:n_kv_heads,:,:] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda:0")
    RH_K[:,:,:,:h] = Kn[:,:,:,h:]
    RH_K[:,:,:,h:] = Kn[:,:,:,:h]
    torch.neg(RH_K[:,:,:,:h], out = RH_K[:,:,:,:h])
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
    if n_groups != 1:
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
        A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False)
    pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def CohereModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    out_weights = tuple(torch.empty_like(self.model.layers[0].input_layernorm.weight, dtype = torch.float32, device = torch.device(x)) for x in range(DEVICE_COUNT))
    input_ids = input_ids[:,:self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            seq_len,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
    else:
        attention_mask = None
    pass

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):
        device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
        hidden_states, position_ids = move_to_device(
            device_index, hidden_states, position_ids
        )
        residual = hidden_states
        hidden_states = fast_layernorm_inference(decoder_layer.input_layernorm, hidden_states, out_weights[device_index])
        hidden_states_attention, present_key_value = CohereAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = hidden_states,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = attention_mask,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
        )

        hidden_states_mlp = fast_swiglu_inference(self.mlp, hidden_states)
        residual += hidden_states_attention
        residual += hidden_states_mlp
        hidden_states = residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_layernorm_inference(self.model.norm, hidden_states, out_weights[device_index])

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass


class FastCohereModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "cohere",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = CohereAttention,
        )
        if init_name is not None:
            exec(function, globals())
            CohereAttention.__init__  = eval(init_name)
        pass
        CohereAttention      .forward = CohereAttention_fast_forward
        CohereSdpaAttention  .forward = CohereAttention_fast_forward
        CohereFlashAttention2.forward = CohereAttention_fast_forward
        CohereDecoderLayer   .forward = CohereDecoderLayer_fast_forward
        CohereModel          .forward = LlamaModel_fast_forward
        CohereForCausalLM    .forward = CausalLM_fast_forward(CohereModel_fast_forward_inference)
        PeftModelForCausalLM .forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(CohereForCausalLM)

        import transformers.models.cohere.modeling_cohere
        transformers.models.cohere.modeling_cohere.CohereRotaryEmbedding = LlamaRotaryEmbedding
        return
    pass
pass

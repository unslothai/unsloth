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
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from .mistral import *
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
try:
    from transformers.models.granite.modeling_granite import (
        GraniteAttention,
        GraniteDecoderLayer,
        GraniteModel,
        GraniteForCausalLM,
    )
except:
    from packaging.version import Version

    transformers_version = Version(transformers_version)
    if not transformers_version >= Version("4.45.0"):
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"\
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
    from transformers.models.granite.modeling_granite import (
        GraniteSdpaAttention,
        GraniteFlashAttention2,
    )
except:
    GraniteSdpaAttention   = GraniteAttention
    GraniteFlashAttention2 = GraniteAttention
pass

def GraniteAttention_fast_forward(
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
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    dropout_p  = self.config.attention_dropout if self.training else 0
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    assert position_embeddings is not None
    cos, sin = position_embeddings
    if position_ids is None:
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)

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

        # Group query attention
        K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        if hidden_states.requires_grad:
            K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
            V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
        else:
            # Xformers does support the forward pass though
            Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        pass

        A = xformers_attention(Q, K, V, attn_bias = causal_mask, scale=self.scaling, p=dropout_p)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        window = (kv_seq_len, kv_seq_len)
        A = flash_attn_func(Q, K, V, causal = True, window_size = window, softmax_scale=self.scaling, dropout_p=dropout_p)
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
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, scale = self.scaling, is_causal = False, dropout_p=dropout_p)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass

    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


def GraniteDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    position_embeddings:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
):
    residual_multiplier = \
        self.residual_multiplier \
        if hasattr(self, "residual_multiplier") else \
        self.config.residual_multiplier

    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings = position_embeddings,
            _flag_for_generation=self._flag_for_generation,
        )
        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(self.mlp, hidden_states)
        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)
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
torch_tanh   = torch.tanh

def GraniteAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
    use_sliding_window = False,
    position_embeddings : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):

    assert position_embeddings is not None, f"Granite model requires position embeddings to be specified"

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
    device = hidden_states.device

    # Prefill phase
    # if not hasattr(self, "paged_attention"):
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = device)
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = device)
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)
        # Only for Gemma2
        self.temp_O  = torch.empty((1, bsz, hidden_size), dtype = dtype, device = device)
        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = device)


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
    cos, sin = position_embeddings
    cos, sin = cos[position_ids], sin[position_ids]
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

    # Grouped query attention
    _, _, cached_len, _ = Kn.shape
    if n_groups != 1:
        Kn = Kn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vn = Vn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Kn = Kn.reshape(bsz, n_heads, cached_len, head_dim)
        Vn = Vn.reshape(bsz, n_heads, cached_len, head_dim)
    pass
    # else:
    #     Kn, Vn = Kn, Vn
    # pass

    Qn *= self.scaling
    A = torch_matmul(Qn, Kn.transpose(2, 3), out = self.attention[:,:,:,:cached_len])

    # if attention_mask is not None: A += attention_mask # Must add attention_mask for batched

    A[:] = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
    A = torch_matmul(A, Vn, out = Qn)
    # else:
    #     A = scaled_dot_product_attention(Qn, Kn, Vn, attn_mask = attention_mask, is_causal = False)
    # pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def GraniteModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    input_ids = input_ids[:,:self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    hidden_states *= self.model.embedding_multiplier
    residual_multiplier = \
        self.residual_multiplier \
        if hasattr(self, "residual_multiplier") else \
        self.config.residual_multiplier

    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            seq_len,
        )
    else:
        attention_mask = None
    pass

    position_embeddings = self.model.rotary_emb.get_cached(self.max_seq_length, hidden_states.device.index)

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):
        device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
        hidden_states, position_ids = move_to_device(
            device_index, hidden_states, position_ids
        )

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.input_layernorm, hidden_states)
        hidden_states, present_key_value = GraniteAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = hidden_states,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = attention_mask,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
            position_embeddings = position_embeddings,
        )

        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states = torch.add(residual, hidden_states, alpha = residual_multiplier)

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference(self.model.norm, hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass

class GraniteRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config):
        super().__init__(config = config)

def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        # we can use self.residual_multiplier arg in GraniteDecoderLayer_fast_forward as mentioned here
        # https://github.com/huggingface/transformers/blob/e5fd865ebae062b7cf03a81b8c6affeb39f30bec/src/transformers/models/granite/modeling_granite.py#L243
        # The problem is, we don't have access to either the value or config in GraniteModel_fast_forward_inference
        # So we need a way to pass this value around. It is probably better to pass on entire config just in case we need it later
        config = kwargs.get("config", args[0] if args else None)
        if config is not None:
            self.config = config
        original_init(self, *args, **kwargs)
    return new_init

class FastGraniteModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "granite",
            rope_module        = GraniteRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = GraniteAttention,
        )
        if init_name is not None:
            exec(function, globals())
            GraniteAttention.__init__  = eval(init_name)
        pass
        GraniteAttention      .forward  = GraniteAttention_fast_forward
        GraniteSdpaAttention  .forward  = GraniteAttention_fast_forward
        GraniteFlashAttention2.forward  = GraniteAttention_fast_forward
        GraniteDecoderLayer   .forward  = GraniteDecoderLayer_fast_forward
        GraniteModel          .forward  = LlamaModel_fast_forward
        GraniteForCausalLM    .forward  = CausalLM_fast_forward(GraniteModel_fast_forward_inference)
        GraniteForCausalLM    .__init__ = patched_init(GraniteForCausalLM.__init__)
        PeftModelForCausalLM .forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(GraniteForCausalLM)

        import transformers.models.granite.modeling_granite
        transformers.models.granite.modeling_granite.GraniteRotaryEmbedding = GraniteRotaryEmbedding

        return
    pass


    @staticmethod
    def post_patch(model, tokenizer):

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(model.model.embed_tokens.weight)
        model.config.update({"unsloth_version" : __version__})

        # We also do this for the lm_head
        lm_head = torch.nn.Linear(1, 1, bias = None)
        del lm_head.weight
        lm_head.weight = model.lm_head.weight
        lm_head.in_features  = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        model.lm_head = lm_head

        # Granite has tied weights! This means lm_head == embed_tokens
        if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
            lm_head = torch.nn.Linear(1, 1, bias = None)
            del lm_head.weight
            lm_head.weight = model.model.embed_tokens.weight
            lm_head.in_features  = lm_head.weight.shape[1]
            lm_head.out_features = lm_head.weight.shape[0]
            model.lm_head = lm_head
        pass

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
            # Downcast RoPE embedding to correct data type
            if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")):

                if hasattr(module, "cos_cached") and \
                    (module.cos_cached.dtype != correct_dtype):

                    module.cos_cached = module.cos_cached.to(correct_dtype)
                    module.sin_cached = module.sin_cached.to(correct_dtype)

                elif hasattr(module, "short_cos_cached") and \
                    (module.short_cos_cached.dtype != correct_dtype):

                    module.short_cos_cached = module.short_cos_cached.to(correct_dtype)
                    module.short_sin_cached = module.short_sin_cached.to(correct_dtype)
                pass
            pass
        pass

        # Clear deleted GPU items
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model, tokenizer
    pass
pass

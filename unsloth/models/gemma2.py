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
from .gemma import (
    GemmaFixedRotaryEmbedding,
    fast_geglu_inference,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2Model,
    Gemma2ForCausalLM,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.gemma2.modeling_gemma2 import *
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2SdpaAttention,
        Gemma2FlashAttention2,
    )
except:
    Gemma2SdpaAttention   = Gemma2Attention
    Gemma2FlashAttention2 = Gemma2Attention
pass


# Weirdly the 4th Post MLP layernorm does not function well with Triton?
# [TODO] We must randomnly use torch.compile?
# I checked the gradients and formulas and I'm sure it's correct.
# I'm stumped :(
@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def fast_rms_layernorm_gemma2_compiled(layernorm, X, gemma = True):
    old_dtype = X.dtype
    X = X.float()
    X = X * torch.rsqrt(X.square().mean(-1, keepdim = True) + layernorm.eps) * \
        (1.0 + layernorm.weight.float())
    return X.to(old_dtype)
pass


# Logit softcapping
# @torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def gemma2_attention(Q, K, V, mask, self, bsz, q_len):
    n_heads    = self.num_heads
    head_dim   = self.head_dim
    n_kv_heads = self.num_key_value_heads
    n_groups   = self.num_key_value_groups
    
    # Grouped query attention
    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)

    t = self.config.attn_logit_softcapping
    s = self.config.hidden_size // self.config.num_attention_heads

    Q = Q * torch.tensor(s**-0.5, dtype = Q.dtype)
    A = t * torch.tanh(torch.matmul(Q, K.transpose(2, 3)) / t)
    A += mask[:q_len, :q_len]
    A = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32)
    A = A.to(Q.dtype)
    A = torch.matmul(A, V)
    A = A.reshape(bsz, q_len, n_heads*head_dim)
    return A
pass


# Logit softcapping
def Gemma2Attention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
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

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len = kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None
    
    # Grouped query attention
    K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, q_len, head_dim)
    K = K.reshape(bsz, n_heads, q_len, head_dim)
    V = V.reshape(bsz, n_heads, q_len, head_dim)

    s = self.config.query_pre_attn_scalar**-0.5
    t = self.config.attn_logit_softcapping

    A = torch.matmul(Q, K.transpose(2, 3)) * s
    A = t * torch.tanh(A / t)
    A += causal_mask[:kv_seq_len, :kv_seq_len]
    A = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype)
    A = torch.matmul(A, V)
    A = A.transpose(1, 2).contiguous()

    A = A.view(bsz, q_len, -1)
    A = self.o_proj(A)
    return A, None, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def Gemma2DecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
):
    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        out_weight = torch.empty(self.input_layernorm.weight.shape, dtype = torch.float32, device = "cuda:0")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self.input_layernorm, hidden_states, out_weight)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = fast_rms_layernorm_inference_gemma(self.post_attention_layernorm, hidden_states, out_weight)
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self. pre_feedforward_layernorm, hidden_states, out_weight)
        hidden_states = fast_geglu_inference(self.mlp, hidden_states)
        hidden_states = fast_rms_layernorm_inference_gemma(self.post_feedforward_layernorm, hidden_states, out_weight)
        hidden_states += residual
    else:
        residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.input_layernorm, hidden_states, gemma = True)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.post_attention_layernorm, hidden_states, gemma = True)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = fast_rms_layernorm_gemma2_compiled(self. pre_feedforward_layernorm, hidden_states, gemma = True)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.post_feedforward_layernorm, hidden_states, gemma = True)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass


from math import sqrt as math_sqrt

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def Gemma2Model_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    out_weight = torch.empty_like(self.model.layers[0].input_layernorm.weight, dtype = torch.float32, device = "cuda:0")
    input_ids = input_ids[:,:self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    # 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
    # 2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
    hidden_states *= torch.tensor(math_sqrt(self.config.hidden_size), dtype = hidden_states.dtype)

    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            seq_len,
        )
    pass

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.input_layernorm, hidden_states, out_weight)
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = hidden_states,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = attention_mask,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
        )
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.post_attention_layernorm, hidden_states, out_weight)
        hidden_states += residual

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer. pre_feedforward_layernorm, hidden_states, out_weight)
        hidden_states = fast_geglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.post_feedforward_layernorm, hidden_states, out_weight)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference_gemma(self.model.norm, hidden_states, out_weight)

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass


class FastGemma2Model(FastLlamaModel):

    @staticmethod
    def pre_patch():
        Gemma2Attention      .forward = Gemma2Attention_fast_forward
        Gemma2SdpaAttention  .forward = Gemma2Attention_fast_forward
        Gemma2FlashAttention2.forward = Gemma2Attention_fast_forward
        Gemma2DecoderLayer   .forward = Gemma2DecoderLayer_fast_forward
        Gemma2Model          .forward = LlamaModel_fast_forward
        Gemma2ForCausalLM    .forward = CausalLM_fast_forward(Gemma2Model_fast_forward_inference)
        PeftModelForCausalLM .forward = PeftModelForCausalLM_fast_forward
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma2.modeling_gemma2
        transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding = GemmaFixedRotaryEmbedding
        return
    pass


    @staticmethod
    def post_patch(model):
        # Patch model for Gemma
        layers = model.model.layers

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

        # Gemma has tied weights! This means lm_head == embed_tokens
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
            # RoPE must be done in float32 for Gemma
            # if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")) \
            #     and (module.cos_cached.dtype != correct_dtype):

            #     module.cos_cached = module.cos_cached.to(correct_dtype)
            #     module.sin_cached = module.sin_cached.to(correct_dtype)
            #     pass
            # pass
        pass

        # Add 1 to weight
        # return output * (1 + self.weight)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L89
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

        # Freeze all parameters except LoRA
        # We do this first since += 1 seems to not be liked by requires_grad = True
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        pass

        # Patch RMS Layernorm
        for name, module in model.named_modules():
            if isinstance(module, Gemma2RMSNorm):
                # Must be in float32
                # https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L36
                # module = module.to(torch.float32)
                # Leave + 1 to Triton kernel itself
                # module.weight += 1.0 # return output * (1 + self.weight)
                if not hasattr(module, "variance_epsilon"):
                    module.variance_epsilon = module.eps # Gemma doesn't use variance_epsilon
        pass

        # Clear deleted GPU items
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model
    pass
pass

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
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states, gemma = True)
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
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states, gemma = True)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self. pre_feedforward_layernorm, hidden_states, gemma = True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = fast_rms_layernorm(self.post_feedforward_layernorm, hidden_states, gemma = True)
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


import inspect
from typing import List, Optional, Tuple, Union


def Gemma2RMSNorm_forward(self, x):
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


def Gemma2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "sliding_window": self.sliding_window,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if self.config.attn_logit_softcapping is not None:
        attn_weights = attn_weights / self.config.attn_logit_softcapping
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.config.attn_logit_softcapping

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def Gemma2DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if (
        self.config._attn_implementation != "flash_attention_2" and self.is_sliding and attention_mask is not None
    ):  # efficient SDPA and no padding
        min_dtype = torch.finfo(hidden_states.dtype).min
        sliding_window_mask = torch.tril(
            torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
        )
        attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
        if attention_mask.shape[-1] <= 1:  # when decoding
            attention_mask = attention_mask[:, :, :, -self.sliding_window :]

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def Gemma2Model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Gemma2ForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    if self.config.final_logit_softcapping is not None:
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping

    logits = logits.float()
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class FastGemma2Model(FastLlamaModel):

    @staticmethod
    def pre_patch():
        # Gemma2Attention      .forward = LlamaAttention_fast_forward
        # Gemma2SdpaAttention  .forward = LlamaAttention_fast_forward
        # Gemma2FlashAttention2.forward = LlamaAttention_fast_forward
        # Gemma2DecoderLayer   .forward = Gemma2DecoderLayer_fast_forward
        # Gemma2Model          .forward = LlamaModel_fast_forward
        # Gemma2ForCausalLM    .forward = CausalLM_fast_forward(Gemma2Model_fast_forward_inference)
        # PeftModelForCausalLM .forward = PeftModelForCausalLM_fast_forward
        Gemma2RMSNorm.forward = Gemma2RMSNorm_forward
        Gemma2Attention.forward = Gemma2Attention_forward
        Gemma2SdpaAttention.forward = Gemma2Attention_forward
        Gemma2FlashAttention2.forward = Gemma2Attention_forward
        Gemma2ForCausalLM .forward = Gemma2ForCausalLM_forward
        Gemma2Model       .forward = Gemma2Model_forward
        Gemma2DecoderLayer.forward = Gemma2DecoderLayer_forward
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        # import transformers.models.gemma2.modeling_gemma2
        # transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding = GemmaFixedRotaryEmbedding
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

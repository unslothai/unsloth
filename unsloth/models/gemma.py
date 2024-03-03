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

from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaModel,
    GemmaForCausalLM,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.gemma.modeling_gemma import (
        GemmaSdpaAttention,
        GemmaFlashAttention2,
    )
except:
    GemmaSdpaAttention   = GemmaAttention
    GemmaFlashAttention2 = GemmaAttention
pass


def fast_geglu_inference(self, X):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    mlp_size = self.config.intermediate_size
    temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda")

    gate = fast_linear_forward(self.gate_proj, X, out = temp[0])
    up   = fast_linear_forward(self.  up_proj, X, out = temp[1])
    gate = torch.nn.functional.gelu(gate, approximate = "tanh")
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])
    return down
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def GemmaDecoderLayer_fast_forward(
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
    if past_key_value is not None:
        do_prefill = not hasattr(self.self_attn, "paged_attention")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            self.self_attn,
            hidden_states,
            past_key_value,
            position_ids,
            do_prefill = do_prefill,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_geglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        # hidden_states = self.input_layernorm(hidden_states)
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
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
pass


from math import sqrt as math_sqrt

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
@torch.inference_mode
def GemmaModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
):
    # Fix out of bounds tokenization
    input_ids = input_ids[:,:self.max_seq_length]

    hidden_states = self.embed_tokens(input_ids)
    hidden_states *= math_sqrt(self.config.hidden_size)

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.layers):
        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.input_layernorm, hidden_states)
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states,
            past_key_values[idx],
            None,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.post_attention_layernorm, hidden_states)
        hidden_states = fast_geglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference(self.norm, hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values   = next_decoder_cache,
        hidden_states     = [],
        attentions        = [],
    )
pass


def GemmaForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None and past_key_values is None:
        causal_mask = xformers.attn_bias.LowerTriangularMask()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    self.model._has_no_labels = labels is None

    if past_key_values is not None and \
        hasattr(self.model.layers[0].self_attn, "paged_attention"):
        outputs = GemmaModel_fast_forward_inference(
            self.model,
            input_ids,
            past_key_values,
        )
    else:
        outputs = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    pass

    hidden_states = outputs[0]
    bsz, q_len, hd = hidden_states.shape
    if bsz == 1 and q_len == 1:
        logits = torch.mv(self.lm_head.weight, hidden_states.ravel())
        logits = logits.unsqueeze(0).unsqueeze(0)
    else:
        logits = self.lm_head(hidden_states)
    pass

    loss = None
    if labels is not None:
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")
        pass
        
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )
    pass

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
pass


class FastGemmaModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        GemmaAttention      .forward = LlamaAttention_fast_forward
        GemmaSdpaAttention  .forward = LlamaAttention_fast_forward
        GemmaFlashAttention2.forward = LlamaAttention_fast_forward
        GemmaDecoderLayer   .forward = GemmaDecoderLayer_fast_forward
        GemmaModel          .forward = LlamaModel_fast_forward
        GemmaForCausalLM    .forward = GemmaForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma.modeling_gemma
        transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding = LlamaRotaryEmbedding
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
            if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")) \
                and (module.cos_cached.dtype != correct_dtype):

                module.cos_cached = module.cos_cached.to(correct_dtype)
                module.sin_cached = module.sin_cached.to(correct_dtype)
                pass
            pass
        pass

        # Add 1 to weight
        # return output * (1 + self.weight)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L89
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

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
            if isinstance(module, GemmaRMSNorm):
                # Must be in float32
                # https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L36
                module = module.to(torch.float32)
                module.weight += 1.0 # return output * (1 + self.weight)
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

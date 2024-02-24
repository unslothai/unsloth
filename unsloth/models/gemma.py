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
    gate = torch.nn.functional.gelu(gate)
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
    # padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    print(past_key_value)
    if False:#past_key_value is not None:
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
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
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


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def GemmaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
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
    if True:#(past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = "cuda",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None
    pass

    # if position_ids is not None:
    #     if position_ids.shape[0] != batch_size:
    #         position_ids = position_ids.repeat((batch_size, 1))
    # pass

    # embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Mormalized from Gemma
    if self.config.model_type == "gemma":
        inputs_requires_grad = inputs_embeds.requires_grad
        if inputs_requires_grad: inputs_embeds.requires_grad_(False)
        inputs_embeds *= math_sqrt(self.config.hidden_size)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if self._has_no_labels and (attention_mask is not None) and (past_key_values is None):
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if inputs_requires_grad: inputs_embeds.requires_grad_(False)
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif False:#self.training:
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
    pass

    hidden_states = inputs_embeds

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

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions)#, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                # causal_mask,
                attention_mask,
                position_ids,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                # causal_mask=causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                # padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    pass
    
    hidden_states = fast_rms_layernorm(self.norm, hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

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
        # GemmaAttention      .forward = LlamaAttention_fast_forward
        # GemmaSdpaAttention  .forward = LlamaAttention_fast_forward
        # GemmaFlashAttention2.forward = LlamaAttention_fast_forward
        GemmaDecoderLayer   .forward = GemmaDecoderLayer_fast_forward
        GemmaModel          .forward = GemmaModel_fast_forward
        GemmaForCausalLM    .forward = GemmaForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        # import transformers.models.gemma.modeling_gemma
        # transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding = LlamaRotaryEmbedding
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

        print("Unsloth: Patching Gemma RMS Layernorm + 1")
        for name, module in model.named_modules():
            if isinstance(module, GemmaRMSNorm):
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

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
    gate = torch.nn.functional.gelu(gate)
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])
    return down
pass


class FastGemmaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.max_seq_len_cached = max(self.max_position_embeddings, seq_len)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.max_position_embeddings, device="cpu", dtype=torch.int64).float().to("cuda").unsqueeze(0)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1).to("cuda")
        position_ids_expanded = t[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        print(freqs.to(torch.int64))
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().to(dtype=torch.bfloat16)
        self.sin_cached = emb.sin().to(dtype=torch.bfloat16)
    pass

    def forward(self, x, position_ids, seq_len=None):
        length = position_ids.shape[1] if position_ids is not None else seq_len
        if length > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=length, device=x.device, dtype=x.dtype)

        old_cos = self.cos_cached[:,:seq_len].to(dtype=x.dtype)
        old_sin = self.sin_cached[:,:seq_len].to(dtype=x.dtype)

        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cuda").float() / self.dim)
            )

        t = torch.arange(self.max_position_embeddings, device="cpu", dtype=torch.int64).float().to("cuda").unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to("cuda")
        position_ids_expanded = t[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        print(freqs.to(torch.int64))
        emb = torch.cat((freqs, freqs), dim=-1)

        seq_len = position_ids.shape[1]
        print(position_ids.shape)
        new_cos = emb.cos().to(dtype=x.dtype)[:,:seq_len]
        new_sin = emb.sin().to(dtype=x.dtype)[:,:seq_len]

        print(new_cos, new_cos.shape)
        print(old_cos, old_cos.shape)
        raise 1
        return new_cos, new_sin
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def GemmaAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None, #Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
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

    if True:#position_ids is None:
        # cos = self.rotary_emb.cos_cached
        # sin = self.rotary_emb.sin_cached
        cos, sin = self.rotary_emb(V, position_ids, seq_len = q_len)
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, position_ids, seq_len = q_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        K, V = past_key_value.update(K, V, self.layer_idx, cache_kwargs)

    # Attention module
    if (not HAS_FLASH_ATTENTION):# and attention_mask is None):
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
        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : K.shape[-2]]

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
        A = scaled_dot_product_attention(Q, K, V, attn_mask = causal_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass
    
    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def GemmaDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
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
        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
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
    cache_position:       Optional[torch.LongTensor] = None,
    *args, **kwargs,
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

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    hidden_states = hidden_states * (self.config.hidden_size**0.5)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

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

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)


    # hidden_states = self.norm(hidden_states)
    hidden_states = fast_rms_layernorm(self.norm, hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
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
    # if labels is not None:
    #     # Shift so that tokens < n predict n
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     # Flatten the tokens
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #     shift_labels = shift_labels.view(-1)
    #     # Enable model parallelism
    #     shift_labels = shift_labels.to(shift_logits.device)
    #     loss = loss_fct(shift_logits, shift_labels)

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
        GemmaAttention      .forward = GemmaAttention_fast_forward
        GemmaSdpaAttention  .forward = GemmaAttention_fast_forward
        GemmaFlashAttention2.forward = GemmaAttention_fast_forward
        GemmaDecoderLayer   .forward = GemmaDecoderLayer_fast_forward
        GemmaModel          .forward = GemmaModel_fast_forward
        GemmaForCausalLM    .forward = GemmaForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma.modeling_gemma
        transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding = FastGemmaRotaryEmbedding
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

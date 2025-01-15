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
from ._utils import (
    __version__,
    patch_linear_scaling,
)
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralModel,
    MixtralForCausalLM,
)
# For PyTorch 2.1.1
try:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralSdpaAttention,
        MixtralFlashAttention2,
    )
except:
    MixtralSdpaAttention = MixtralAttention
    MixtralFlashAttention2 = MixtralAttention

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key states."""
    # Gather cos/sin based on position ids
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class MixtralExpert(nn.Module):
    """Individual expert in the Mixtral MoE layer."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size,
                            config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size,
                            config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size,
                            config.intermediate_size, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(
            self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


def compute_router_probabilities(router_logits: torch.Tensor, top_k: int = 2):
    """Compute probabilities for expert routing."""
    router_probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(router_probs, top_k, dim=-1)

    # Normalize the probabilities
    norm_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    return norm_probs, top_indices


def get_expert_mask(selected_experts: torch.Tensor, expert_idx: int):
    """Create mask for expert computation."""
    return (selected_experts == expert_idx).any(dim=-1)


def fix_prepare_inputs_for_generation(model_class):
    """Patch the prepare_inputs_for_generation method for better generation handling."""

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # Generate position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    model_class.prepare_inputs_for_generation = prepare_inputs_for_generation


class FastMixtralModel(FastLlamaModel):
    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name="mixtral",
            rope_module=LlamaRotaryEmbedding,
            scaled_rope_module=LlamaLinearScalingRotaryEmbedding,
            attention_module=MixtralAttention,
        )

        # Add Nemo model support
        if function is not None:
            function = patch_mixtral_nemo_attention(function)
            exec(function, globals())
            MixtralAttention.__init__ = eval(init_name)

        # Patch the forward methods
        MixtralAttention.forward = MixtralAttention_fast_forward
        MixtralSdpaAttention.forward = MixtralAttention_fast_forward
        MixtralFlashAttention2.forward = MixtralAttention_fast_forward
        MixtralDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        MixtralModel.forward = LlamaModel_fast_forward
        MixtralForCausalLM.forward = CausalLM_fast_forward(
            LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        fix_prepare_inputs_for_generation(MixtralForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.mistral.modeling_mistral
        transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding = LlamaRotaryEmbedding

    @staticmethod
    def from_pretrained(
        model_name="mistralai/Mixtral-8x7B-v0.1",
        max_seq_length=None,
        dtype=None,
        load_in_4bit=True,
        token=None,
        device_map="sequential",
        rope_scaling=None,
        fix_tokenizer=True,
        model_patcher=None,
        tokenizer_name=None,
        trust_remote_code=False,
        **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
            device_map=device_map,
            rope_scaling=rope_scaling,
            fix_tokenizer=fix_tokenizer,
            model_patcher=FastMixtralModel,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


def patch_mixtral_memory_efficient(layer_class):
    """Patch Mixtral layers for memory-efficient operation."""

    original_forward = layer_class.forward

    def memory_efficient_forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True):
            # Enable activation checkpointing for MoE layers
            if hasattr(self, "block_sparse_moe"):
                self.gradient_checkpointing = True

            # Process experts in chunks if needed
            if hasattr(self, "num_experts") and self.num_experts > 4:
                return chunked_forward(self, *args, **kwargs)

            return original_forward(self, *args, **kwargs)

    layer_class.forward = memory_efficient_forward


def patch_moe_layer_memory_efficient():
    """Apply memory optimizations to MoE layers."""

    def expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Free memory from previous expert computations
        torch.cuda.empty_cache()

        # Use memory-efficient SiLU
        current_hidden_states = F.silu(
            self.w1(hidden_states), inplace=True) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    MixtralExpert.forward = expert_forward


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_dim = config.hidden_size

        # Router
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [MixtralExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        # Router logits
        router_logits = self.gate(hidden_states)

        # Get top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1)
        routing_weights = routing_weights / \
            routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Compute expert outputs
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if torch.any(expert_mask):
                expert_input = hidden_states[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                final_hidden_states[expert_mask] += expert_output * \
                    routing_weights[expert_mask]

        return final_hidden_states, router_logits


def MixtralAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
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

    bsz, q_len, _ = hidden_states.size()

    n_heads = self.num_attention_heads
    n_kv_heads = self.num_key_value_heads
    head_dim = self.head_dim
    n_groups = self.num_key_value_groups

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Extend RoPE dynamically to fit in VRAM
    self.rotary_emb.extend_rope_embedding(V, seq_len=kv_seq_len)

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len=kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and attention_mask is None):
        # Xformers memory efficient attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        K_M = V_M = bsz * kv_seq_len
        Q_M = bsz * q_len

        # Group query attention
        K = K.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
        V = V.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
        K = K.expand(-1, -1, -1, n_groups, -1)
        V = V.expand(-1, -1, -1, n_groups, -1)
        K = K.reshape(K_M, n_heads, head_dim)
        V = V.reshape(V_M, n_heads, head_dim)
        Q = Q.reshape(Q_M, n_heads, head_dim)

        output = memory_efficient_attention(
            Q, K, V,
            K_M, V_M, Q_M,
            n_heads, head_dim,
            causal=True,
        )
        output = output.view(bsz, q_len, n_heads * head_dim)
    else:
        # Use Flash Attention or regular attention with mask
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        output = scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )
        output = output.reshape(bsz, q_len, n_heads * head_dim)

    output = self.o_proj(output)
    return output, None, past_key_value


def MixtralForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
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
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

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


def patch_mixtral_nemo_attention(function):
    """Patch for Mixtral Nemo models similar to Mistral."""
    function = function.replace(
        "(self.head_dim * self.config.num_attention_heads) != self.config.hidden_size",
        "False",
    )
    function = function.replace(
        "self.head_dim = self.config.hidden_size // self.config.num_attention_heads",
        "self.head_dim = config.head_dim",
    )
    function = function.replace(
        "self.o_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)",
        "self.o_proj = nn.Linear(self.config.num_attention_heads * self.head_dim, self.config.hidden_size, bias=False)",
    )
    return function


def fast_rope_embedding(Q, K, cos, sin):
    """Fast RoPE embedding implementation."""
    return apply_rotary_pos_emb(Q, K, cos, sin, None)


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    """Inplace RoPE embedding with position IDs."""
    return apply_rotary_pos_emb(Q, K, cos, sin, position_ids)


def memory_efficient_attention(Q, K, V, K_M, V_M, Q_M, n_heads, head_dim, causal=True):
    """Memory efficient attention implementation using xformers if available."""
    try:
        from xformers.ops import memory_efficient_attention as xformers_attention
        output = xformers_attention(
            Q, K, V,
            attn_bias=None,
            p=0.0,
            scale=1.0 / math.sqrt(head_dim),
            op=None,
        )
    except ImportError:
        # Fallback to standard attention
        scale = 1.0 / math.sqrt(head_dim)
        Q = Q * scale
        attn_weights = torch.bmm(Q, K.transpose(-2, -1))
        if causal:
            causal_mask = torch.triu(torch.ones_like(attn_weights), diagonal=1)
            attn_weights = attn_weights.masked_fill(
                causal_mask.bool(), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.bmm(attn_weights, V)
    return output


def scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True):
    """Scaled dot product attention with optional mask."""
    scale = 1.0 / math.sqrt(Q.size(-1))
    attn = torch.matmul(Q * scale, K.transpose(-2, -1))

    if is_causal:
        causal_mask = torch.triu(torch.ones_like(attn), diagonal=1)
        attn = attn.masked_fill(causal_mask.bool(), float('-inf'))
    elif attn_mask is not None:
        attn = attn + attn_mask

    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def chunked_forward(self, hidden_states, *args, **kwargs):
    """Process hidden states in chunks to save memory."""
    chunk_size = min(4, self.num_experts)  # Process up to 4 experts at a time
    num_chunks = (self.num_experts + chunk_size - 1) // chunk_size

    # Get router outputs
    router_logits = self.gate(hidden_states)
    router_probs = F.softmax(router_logits, dim=-1)

    # Select top-k experts
    top_k_probs, top_k_indices = torch.topk(
        router_probs,
        min(self.num_experts_per_token, self.num_experts),
        dim=-1
    )

    # Normalize probabilities
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Initialize output tensor
    final_hidden_states = torch.zeros_like(hidden_states)

    # Process experts in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, self.num_experts)
        chunk_experts = list(range(start_idx, end_idx))

        # Process each expert in the chunk
        for expert_idx in chunk_experts:
            # Get mask for current expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if torch.any(expert_mask):
                # Get expert inputs
                expert_input = hidden_states[expert_mask]
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                # Weight output by router probability
                expert_probs = top_k_probs[expert_mask][
                    top_k_indices[expert_mask] == expert_idx
                ].unsqueeze(-1)
                final_hidden_states[expert_mask] += expert_output * \
                    expert_probs

        # Clear cache after each chunk
        torch.cuda.empty_cache()

    return final_hidden_states

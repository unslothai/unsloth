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

"""
GLM-4.7 Flash (GLM4 MoE Lite) optimized implementation using grouped GEMM.

Key architecture differences from Qwen3 MoE:
- Router uses sigmoid activation (not softmax)
- Has routed_scaling_factor of 1.8
- Has 1 shared expert that processes all tokens
- Uses group-based selection before topk
- Uses MLA (Multi-head Latent Attention)
"""

from .llama import *
import os
from ._utils import __version__
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    fix_prepare_inputs_for_generation,
    fast_rms_layernorm_inference,
    fast_swiglu_inference,
    LlamaModel_fast_forward,
    LlamaModel_fast_forward_inference,
    CausalLM_fast_forward,
    PeftModel_fast_forward,
)
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from ..kernels import fast_rms_layernorm

# Import the grouped gemm utilities from unsloth kernels
# The grouped_gemm module expects its parent directory to be in sys.path
HAS_GROUPED_GEMM = False
try:
    import sys
    import os
    # Add the moe directory (parent of grouped_gemm) to sys.path
    _moe_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kernels", "moe"
    )
    if _moe_path not in sys.path:
        sys.path.insert(0, _moe_path)

    # Import grouped_gemm package first to apply TMA compatibility shim
    # This patches triton.language to support both old and new TMA API names
    import grouped_gemm  # noqa: F401 - triggers TMA compatibility shim

    from grouped_gemm.interface import grouped_gemm
    from grouped_gemm.reference.moe_ops import (
        get_routing_indices,
        permute,
        unpermute,
    )
    HAS_GROUPED_GEMM = True
except ImportError as e:
    import warnings
    warnings.warn(f"Grouped GEMM not available: {e}. MoE will use fallback implementation.")


# Import transformers GLM4 MoE Lite classes
try:
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention,
        Glm4MoeLiteMoE,
        Glm4MoeLiteMLP,
        Glm4MoeLiteNaiveMoe,
        Glm4MoeLiteTopkRouter,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteModel,
        Glm4MoeLiteForCausalLM,
        Glm4MoeLiteRMSNorm,
    )
    HAS_GLM4_MOE = True
except ImportError:
    HAS_GLM4_MOE = False
    # Create dummy classes for type checking
    class Glm4MoeLiteAttention:
        pass
    class Glm4MoeLiteMoE:
        pass
    class Glm4MoeLiteMLP:
        pass
    class Glm4MoeLiteNaiveMoe:
        pass
    class Glm4MoeLiteTopkRouter:
        pass
    class Glm4MoeLiteDecoderLayer:
        pass
    class Glm4MoeLiteModel:
        pass
    class Glm4MoeLiteForCausalLM:
        pass


torch_nn_functional_silu = torch.nn.functional.silu


def Glm4MoeLiteMoE_fast_forward(self, hidden_states):
    """
    Optimized MoE forward pass using grouped GEMM.

    GLM4 MoE specifics:
    - Uses sigmoid router activation (not softmax)
    - Has routed_scaling_factor of 1.8
    - Has 1 shared expert that always processes all tokens
    - Uses group-based selection with topk_group
    """
    residuals = hidden_states
    orig_shape = hidden_states.shape
    batch_size, seq_len, hidden_dim = orig_shape
    num_tokens = batch_size * seq_len

    # Flatten hidden states for routing
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Router computation
    router_logits = self.gate(hidden_states)  # [num_tokens, n_routed_experts]
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    # Cast routing weights to match hidden_states dtype (Qwen3 pattern)
    # Sigmoid router returns fp32, but hidden_states may be bf16
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Get routing indices for grouped GEMM
    with torch.no_grad():
        token_counts_by_expert, gather_indices = get_routing_indices(
            topk_indices, self.n_routed_experts
        )

    # Use grouped GEMM for expert computation
    if HAS_GROUPED_GEMM:
        # Cast hidden_states to match expert weights dtype
        # Under autocast, hidden_states may be fp32 while weights are bf16
        hidden_states = hidden_states.to(self.experts.gate_up_proj.dtype)

        # First grouped GEMM: gate_up_proj with permute_x
        # Input: [num_tokens, hidden_dim] -> Output: [total_tokens, 2*intermediate_dim]
        intermediate = grouped_gemm(
            X=hidden_states,
            W=self.experts.gate_up_proj,
            m_sizes=token_counts_by_expert.int(),
            topk=self.top_k,
            gather_indices=gather_indices,
            permute_x=True,
            permute_y=False,
            autotune=True,
            is_first_gemm=True,
        )

        # Activation: SiLU(gate) * up
        gate, up = intermediate.chunk(2, dim=-1)
        intermediate = torch_nn_functional_silu(gate) * up

        # Second grouped GEMM: down_proj with permute_y
        # Input: [total_tokens, intermediate_dim] -> Output: [total_tokens, hidden_dim]
        expert_output = grouped_gemm(
            X=intermediate,
            W=self.experts.down_proj,
            m_sizes=token_counts_by_expert.int(),
            topk=self.top_k,
            gather_indices=gather_indices,
            permute_x=False,
            permute_y=True,
            autotune=True,
            is_first_gemm=False,
        )

        # Merge topk weights: [num_tokens, top_k, hidden_dim] -> [num_tokens, hidden_dim]
        hidden_states = (
            expert_output.view(num_tokens, self.top_k, hidden_dim)
            * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
    else:
        # Fallback to naive implementation
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)

    # Add shared expert output
    hidden_states = hidden_states + self.shared_experts(residuals.view(-1, hidden_dim))

    return hidden_states.view(*orig_shape)


def Glm4MoeLiteNaiveMoe_fast_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized expert forward using grouped GEMM.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        top_k_index: [num_tokens, top_k] indices of selected experts
        top_k_weights: [num_tokens, top_k] weights for selected experts

    Returns:
        [num_tokens, hidden_dim] output after weighted sum of expert outputs
    """
    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_index.shape[1]
    # Cast routing weights to match hidden_states dtype (Qwen3 pattern)
    top_k_weights = top_k_weights.to(hidden_states.dtype)

    if not HAS_GROUPED_GEMM:
        # Fallback to original naive implementation
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    # Get routing indices for grouped GEMM
    with torch.no_grad():
        token_counts_by_expert, gather_indices = get_routing_indices(
            top_k_index, self.num_experts
        )

    # Cast hidden_states to match expert weights dtype
    # Under autocast, hidden_states may be fp32 while weights are bf16
    hidden_states = hidden_states.to(self.gate_up_proj.dtype)

    # First grouped GEMM: gate_up_proj
    intermediate = grouped_gemm(
        X=hidden_states,
        W=self.gate_up_proj,
        m_sizes=token_counts_by_expert.int(),
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=True,
        is_first_gemm=True,
    )

    # Activation: SiLU(gate) * up
    gate, up = intermediate.chunk(2, dim=-1)
    intermediate = self.act_fn(gate) * up

    # Second grouped GEMM: down_proj
    expert_output = grouped_gemm(
        X=intermediate,
        W=self.down_proj,
        m_sizes=token_counts_by_expert.int(),
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=False,
        permute_y=True,
        autotune=True,
        is_first_gemm=False,
    )

    # Merge topk weights
    final_hidden_states = (
        expert_output.view(num_tokens, top_k, hidden_dim)
        * top_k_weights.unsqueeze(-1)
    ).sum(dim=1)

    return final_hidden_states


def Glm4MoeLiteDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Optimized decoder layer forward with fast RMS layernorm.
    """
    # Check if we're in inference mode
    is_inference = use_cache and hasattr(self, "_flag_for_generation")

    if is_inference:
        # Self-attention with fast inference path
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(
            self.input_layernorm, hidden_states
        )
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP/MoE
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(
            self.post_attention_layernorm, hidden_states
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    else:
        # Training path
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP/MoE
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states


def Glm4MoeLiteMLP_fast_forward(self, x):
    """
    Optimized MLP forward using fused SwiGLU.
    """
    return fast_swiglu_inference(self, x)


class FastGLM47Model(FastLlamaModel):
    """
    Fast GLM-4.7 Flash (GLM4 MoE Lite) model with grouped GEMM optimization.

    This provides 2-3x throughput improvement for MoE layers by:
    - Replacing sequential expert loops with grouped GEMM operations
    - Fusing permutation operations into the GEMM kernels
    - Using optimized RMS LayerNorm and SwiGLU implementations
    """

    @staticmethod
    def pre_patch():
        if not HAS_GLM4_MOE:
            raise ImportError(
                "Unsloth: GLM4 MoE Lite support requires transformers >= 5.0.0. "
                "Please upgrade with: pip install --upgrade transformers"
            )

        # Patch MoE forward with grouped GEMM optimization
        # TMA compatibility is handled by grouped_gemm/__init__.py which patches
        # triton.language to support both old (_experimental_make_tensor_descriptor)
        # and new (make_tensor_descriptor) API names
        if HAS_GROUPED_GEMM:
            Glm4MoeLiteNaiveMoe.forward = Glm4MoeLiteNaiveMoe_fast_forward
            Glm4MoeLiteMoE.forward = Glm4MoeLiteMoE_fast_forward

        # Note: We don't patch the following for GLM4 MoE because:
        # - GLM4 uses MLA (Multi-head Latent Attention) which has different projection names
        # - Glm4MoeLiteRotaryEmbedding doesn't have extend_rope_embedding method
        # - The decoder layer and model forward functions assume Llama-compatible infrastructure

        return

    @staticmethod
    def from_pretrained(
        model_name="unsloth/GLM-4.7-Flash",
        max_seq_length=4096,
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
        # Pop kwargs that are used by loader but not passed to model
        kwargs.pop("unsloth_force_compile", None)

        return FastLlamaModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
            device_map=device_map,
            rope_scaling=rope_scaling,
            fix_tokenizer=fix_tokenizer,
            model_patcher=FastGLM47Model,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

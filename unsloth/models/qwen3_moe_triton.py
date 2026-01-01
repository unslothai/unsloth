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
Qwen3 MoE integration with Triton kernels for faster training.
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .llama import *
from ._utils import __version__
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from .qwen3 import (
    Qwen3Attention_fast_forward,
    FastQwen3Model,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
)

# Try to import Triton kernels
try:
    from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels
    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
    from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import (
        Qwen3MoeGroupedGEMMBlock,
        permute,
        unpermute,
    )
    from unsloth.kernels.moe.grouped_gemm.reference.moe_block import (
        Qwen3MoeFusedGroupedGEMMBlock,
    )

    TRITON_MOE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Triton MoE kernels not available: {e}")
    TRITON_MOE_AVAILABLE = False

torch_nn_functional_softmax = torch.nn.functional.softmax

logger = logging.getLogger(__name__)

# Global variable to store kernel configs
_moe_kernel_configs = {}
_moe_autotuning_done = False


def get_moe_kernel_configs(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    force_autotune: bool = False,
) -> Tuple[Optional[any], Optional[any], Optional[any]]:
    """Get or create MoE kernel configurations."""
    global _moe_kernel_configs, _moe_autotuning_done

    if not TRITON_MOE_AVAILABLE:
        return None, None, None

    config_key = (num_experts, hidden_dim, intermediate_dim, top_k, str(dtype))

    if config_key not in _moe_kernel_configs or force_autotune:
        try:
            configs = get_or_autotune_moe_kernels(
                num_experts = num_experts,
                hidden_dim = hidden_dim,
                intermediate_dim = intermediate_dim,
                top_k = top_k,
                dtype = dtype,
                force_autotune = force_autotune,
            )
            _moe_kernel_configs[config_key] = configs
            _moe_autotuning_done = True
            logger.info(f"MoE kernel configs ready for {config_key}")
        except Exception as e:
            logger.error(f"Failed to get MoE kernel configs: {e}")
            return None, None, None

    return _moe_kernel_configs.get(config_key, (None, None, None))


def Qwen3MoeSparseMoeBlock_triton_forward(
    self, hidden_states: torch.Tensor, temp_gate = None, temp_up = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast forward implementation using Triton grouped GEMM kernels.

    This replaces the original Qwen3MoeSparseMoeBlock_fast_forward with
    Triton-optimized kernels when available.
    """
    if not TRITON_MOE_AVAILABLE:
        # Fallback to original implementation
        return Qwen3MoeSparseMoeBlock_fast_forward(
            self, hidden_states, temp_gate, temp_up
        )

    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_tokens = batch_size * seq_len
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # Router computation
    router_logits = fast_linear_forward(
        self.gate_proj, hidden_states_flat, out = temp_gate
    )

    routing_weights = torch_nn_functional_softmax(
        router_logits, dim = -1, dtype = torch.float32
    )
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim = -1)
    routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Get kernel configs
    num_experts = self.num_experts
    top_k = self.top_k
    intermediate_dim = self.experts[0].gate_proj.in_features // 2  # Assuming SwiGLU

    config_fwd, config_bwd_dx, config_bwd_dw = get_moe_kernel_configs(
        num_experts = num_experts,
        hidden_dim = hidden_dim,
        intermediate_dim = intermediate_dim,
        top_k = top_k,
        dtype = hidden_states.dtype,
    )

    # If we don't have kernel configs, fallback to original
    if config_fwd is None:
        return Qwen3MoeSparseMoeBlock_fast_forward(
            self, hidden_states, temp_gate, temp_up
        )

    try:
        # Prepare expert weights for grouped GEMM
        gate_up_weights = []
        down_weights = []

        for expert in self.experts:
            # Combine gate and up projections for first GEMM
            gate_weight = expert.gate_proj.weight
            up_weight = expert.up_proj.weight
            gate_up_weight = torch.cat([gate_weight, up_weight], dim = 0)
            gate_up_weights.append(gate_up_weight)
            down_weights.append(expert.down_proj.weight)

        gate_up_weights = torch.stack(
            gate_up_weights
        )  # [num_experts, 2*intermediate_dim, hidden_dim]
        down_weights = torch.stack(
            down_weights
        )  # [num_experts, hidden_dim, intermediate_dim]

        # Compute token counts and gather indices without array operations
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes = num_experts
        ).permute(2, 1, 0)

        token_counts_by_expert = expert_mask.sum(dim = 1).int()

        # Create gather indices for routing - avoid complex array operations
        total_tokens = num_tokens * top_k
        gather_indices = torch.zeros(
            total_tokens, dtype = torch.long, device = hidden_states.device
        )

        # Simple sequential assignment for gather indices
        current_idx = 0
        for expert_idx in range(num_experts):
            expert_tokens = expert_mask[expert_idx].sum(dim = 0).bool()
            num_expert_tokens = expert_tokens.sum().item()
            if num_expert_tokens > 0:
                expert_indices = torch.where(expert_tokens)[0]
                gather_indices[current_idx : current_idx + num_expert_tokens] = (
                    expert_indices
                )
                current_idx += num_expert_tokens

        # First grouped GEMM: gate_up projection
        intermediate_states = grouped_gemm(
            X = hidden_states_flat,
            W = gate_up_weights,
            m_sizes = token_counts_by_expert,
            gather_indices = gather_indices,
            topk = top_k,
            permute_x = True,
            permute_y = False,
            autotune = False,  # Use pre-tuned configs
            kernel_config_fwd = config_fwd,
            kernel_config_bwd_dX = config_bwd_dx,
            kernel_config_bwd_dW = config_bwd_dw,
            is_first_gemm = True,
        )

        # Apply activation and multiply
        gate, up = intermediate_states.chunk(2, dim = -1)
        intermediate_states = F.silu(gate) * up

        # Second grouped GEMM: down projection
        final_states = grouped_gemm(
            X = intermediate_states,
            W = down_weights,
            m_sizes = token_counts_by_expert,
            gather_indices = gather_indices,
            topk = top_k,
            permute_x = False,
            permute_y = True,
            autotune = False,  # Use pre-tuned configs
            kernel_config_fwd = config_fwd,
            kernel_config_bwd_dX = config_bwd_dx,
            kernel_config_bwd_dW = config_bwd_dw,
            is_first_gemm = False,
        )

        # Reshape and apply routing weights
        final_states = final_states.view(num_tokens, top_k, hidden_dim)
        final_states = final_states * routing_weights.unsqueeze(-1)
        final_states = final_states.sum(dim = 1)
        final_states = final_states.view(batch_size, seq_len, hidden_dim)

        return final_states, router_logits

    except Exception as e:
        logger.error(f"Triton MoE kernel failed: {e}")
        # Fallback to original implementation
        return Qwen3MoeSparseMoeBlock_fast_forward(
            self, hidden_states, temp_gate, temp_up
        )


def Qwen3MoeMLP_triton_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Fast forward implementation for Qwen3MoeMLP using optimized kernels.
    """
    # This is for individual expert MLPs, still use the original fast implementation
    return fast_swiglu_inference(self, x)


class FastTritonQwen3MoeModel(FastQwen3Model):
    """Fast Qwen3 MoE model with Triton kernel integration."""

    @staticmethod
    def pre_patch():
        """Patch Qwen3 MoE components with Triton optimizations."""
        # Apply original patches first
        FastQwen3Model.pre_patch()

        # Override MoE-specific components with Triton versions
        if TRITON_MOE_AVAILABLE:
            logger.info("Patching Qwen3 MoE with Triton kernels")
            Qwen3MoeSparseMoeBlock.forward = Qwen3MoeSparseMoeBlock_triton_forward
            Qwen3MoeMLP.forward = Qwen3MoeMLP_triton_forward
        else:
            logger.warning(
                "Triton MoE kernels not available, using original implementation"
            )

    @staticmethod
    def from_pretrained(
        model_name = "Qwen/Qwen3-7B",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        # MoE-specific parameters
        use_triton_moe = True,
        force_moe_autotune = False,
        **kwargs,
    ):
        """Load Qwen3 MoE model with Triton optimizations."""

        # Set environment variable for MoE kernel preference
        if use_triton_moe and TRITON_MOE_AVAILABLE:
            os.environ["UNSLOTH_TRITON_MOE"] = "1"
            logger.info("Enabling Triton MoE kernels")
        else:
            os.environ["UNSLOTH_TRITON_MOE"] = "0"

        # Load the model using the original method
        model = FastQwen3Model.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = token,
            device_map = device_map,
            rope_scaling = rope_scaling,
            fix_tokenizer = fix_tokenizer,
            model_patcher = FastTritonQwen3MoeModel,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )

        # Pre-autotune MoE kernels if requested and available
        if use_triton_moe and TRITON_MOE_AVAILABLE and hasattr(model, "model"):
            try:
                # Extract MoE configuration from the model
                config = model.config
                if hasattr(config, "num_experts") and hasattr(config, "hidden_size"):
                    num_experts = config.num_experts
                    hidden_dim = config.hidden_size
                    intermediate_dim = config.intermediate_size or (
                        hidden_dim * 4
                    )  # Common ratio
                    top_k = getattr(config, "num_experts_per_tok", 2)

                    logger.info(
                        f"Pre-autotuning MoE kernels: {num_experts} experts, hidden={hidden_dim}, intermediate={intermediate_dim}"
                    )

                    # Trigger autotuning
                    get_moe_kernel_configs(
                        num_experts = num_experts,
                        hidden_dim = hidden_dim,
                        intermediate_dim = intermediate_dim,
                        top_k = top_k,
                        dtype = model.dtype,
                        force_autotune = force_moe_autotune,
                    )

            except Exception as e:
                logger.error(f"Failed to pre-autotune MoE kernels: {e}")

        return model


# Import the original fast forward function for fallback
from .qwen3_moe import Qwen3MoeSparseMoeBlock_fast_forward

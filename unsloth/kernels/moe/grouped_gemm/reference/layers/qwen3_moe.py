# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    ACT2FN,
    Qwen3MoeSparseMoeBlock,
)

from grouped_gemm.interface import grouped_gemm
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

"""
Reference implementation of HF Qwen3 MoE block using grouped gemm.

The Qwen3MoeGroupedGEMMBlock is a reference torch-native implementation.
Qwen3MoeFusedGroupedGEMMBlock is a version using the triton grouped gemm kernel.

NOTE: This is NOT to be used for production as it contains many extra checks and saves all intermediate results for debugging.
"""


@dataclass
class GroupedGEMMResult:
    """
    Container for intermediate results and final output of the grouped GEMM MoE block.
    
    Args:
        token_counts_by_expert (`torch.Tensor`):
            Number of tokens assigned to each expert.
        gather_indices (`torch.Tensor`):
            Indices for gathering tokens in expert order.
        topk_weights (`torch.Tensor`):
            Weights for top-k experts.
        first_gemm (`torch.Tensor`):
            Result of the first GEMM operation.
        intermediate (`torch.Tensor`):
            Intermediate activation after gate/up projection.
        second_gemm (`torch.Tensor`):
            Result of the second GEMM operation.
        hidden_states_unpermute (`torch.Tensor`):
            Unpermuted hidden states after second GEMM.
        hidden_states (`torch.Tensor`):
            Final output hidden states.
    """
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    hidden_states: torch.Tensor  # final output


class Qwen3MoeGroupedGEMMBlock(torch.nn.Module):
    """
    Reference implementation of Qwen3 MoE block using grouped GEMM operations.
    
    Args:
        num_experts (`int`):
            Number of experts in the MoE layer.
        top_k (`int`):
            Number of experts to use for each token.
        norm_topk_prob (`bool`):
            Whether to normalize top-k probabilities.
        hidden_size (`int`):
            Dimension of hidden states.
        moe_intermediate_size (`int`):
            Intermediate size for MoE computations.
    """
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        assert gate.shape == (config.num_experts, config.hidden_size)
        assert gate_up_proj.shape == (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        assert down_proj.shape == (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )

        # gating
        self.gate = torch.nn.Parameter(gate)

        # experts
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=True)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=True)
        self.act_fn = ACT2FN[config.hidden_act]

    @staticmethod
    def extract_hf_weights(moe_block: Qwen3MoeSparseMoeBlock) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract weights from a HuggingFace MoE block for use in the grouped GEMM implementation.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                HuggingFace MoE block to extract weights from.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Extracted gate weights, gate/up projection weights, and down projection weights.
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        num_experts = config.num_experts

        gate = moe_block.gate.weight.data
        gate_proj = torch.stack(
            [moe_block.experts[i].gate_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        up_proj = torch.stack(
            [moe_block.experts[i].up_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        down_proj = torch.stack(
            [moe_block.experts[i].down_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)
        return gate, gate_up_proj, down_proj

    @classmethod
    def from_hf(cls, moe_block: Qwen3MoeSparseMoeBlock) -> Qwen3MoeGroupedGEMMBlock:
        """
        Create a Qwen3MoeGroupedGEMMBlock from a HuggingFace MoE block.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                HuggingFace MoE block to convert.
        
        Returns:
            `Qwen3MoeGroupedGEMMBlock`:
                Newly created grouped GEMM MoE block.
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = cls.extract_hf_weights(moe_block)
        return cls(config, gate, gate_up_proj, down_proj)

    def check_weights(self, moe_block: Qwen3MoeSparseMoeBlock) -> None:
        """
        Verify that the weights in this block match those in a HuggingFace MoE block.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                HuggingFace MoE block to compare against.
        
        Raises:
            AssertionError: If weights do not match.
        """
        for i in range(self.num_experts):
            assert self.gate_up_proj[i].equal(
                torch.cat(
                    [
                        moe_block.experts[i].gate_proj.weight.data,
                        moe_block.experts[i].up_proj.weight.data,
                    ],
                    dim=0,
                )
            )
            assert self.down_proj[i].equal(moe_block.experts[i].down_proj.weight.data)

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function and multiply gate and up projections.
        
        Args:
            x (`torch.Tensor`):
                Input tensor containing concatenated gate and up projections.
        
        Returns:
            `torch.Tensor`:
                Activated and multiplied result.
        """
        assert x.shape[-1] == 2 * self.moe_intermediate_size
        gate_proj = x[..., : self.moe_intermediate_size]
        up_proj = x[..., self.moe_intermediate_size :]
        return self.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Run the routing mechanism to determine expert assignments for each token.
        
        Args:
            hidden_states (`torch.Tensor`):
                Input hidden states to route through experts.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Router logits, routing weights, and selected expert indices.
        """
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = torch.nn.functional.linear(hidden_states, self.gate)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token counts per expert and indices for gathering tokens in expert order.
        
        Args:
            selected_experts (`torch.Tensor`):
                Tensor of selected expert indices for each token.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Token counts per expert and gather indices.
        """
        token_counts_by_expert, gather_indices = get_routing_indices(
            selected_experts, self.num_experts
        )
        assert not token_counts_by_expert.requires_grad
        assert not gather_indices.requires_grad
        return token_counts_by_expert, gather_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(hidden_states, gather_indices, self.top_k)
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = torch_grouped_gemm(
            X=hidden_states, W=self.gate_up_proj, m_sizes=token_counts_by_expert
        )
        assert first_gemm.shape == (total_tokens, 2 * self.moe_intermediate_size)
        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.moe_intermediate_size)
        second_gemm = torch_grouped_gemm(
            X=intermediate, W=self.down_proj, m_sizes=token_counts_by_expert
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        # 1. Unpermute from expert order to token order
        hidden_states_unpermute = unpermute(second_gemm, gather_indices)
        assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)

        # 2. Merge topk weights
        hidden_states = (
            hidden_states_unpermute.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)
        assert hidden_states.shape == (num_tokens, hidden_dim)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return GroupedGEMMResult(
            token_counts_by_expert=token_counts_by_expert,
            gather_indices=gather_indices,
            topk_weights=routing_weights,
            first_gemm=first_gemm,
            intermediate=intermediate,
            second_gemm=second_gemm,
            hidden_states_unpermute=hidden_states_unpermute,
            hidden_states=hidden_states,
        ), router_logits


class Qwen3MoeFusedGroupedGEMMBlock(Qwen3MoeGroupedGEMMBlock):
    """
    Optimized implementation of Qwen3 MoE block using fused grouped GEMM operations.
    
    Args:
        permute_x (`bool`):
            Whether to permute input before first GEMM.
        permute_y (`bool`):
            Whether to permute output after second GEMM.
        autotune (`bool`):
            Whether to autotune kernel configurations.
        kernel_config_fwd (`KernelConfigForward`):
            Forward kernel configuration.
        kernel_config_bwd_dW (`KernelConfigBackward_dW`):
            Backward dW kernel configuration.
        kernel_config_bwd_dX (`KernelConfigBackward_dX`):
            Backward dX kernel configuration.
        dW_only (`bool`):
            Whether to compute only dW in backward pass.
        dX_only (`bool`):
            Whether to compute only dX in backward pass.
    """
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permute_x: bool                               = True,
        permute_y: bool                               = True,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool                                 = False,
        dX_only: bool                                 = False,
    ):
        super().__init__(config, gate, gate_up_proj, down_proj)
        self.permute_x = permute_x
        self.permute_y = permute_y
        self.autotune = autotune
        if not autotune:
            assert (
                kernel_config_fwd is not None
                and kernel_config_bwd_dW is not None
                and kernel_config_bwd_dX is not None
            ), "Kernel configs must be provided if autotune is False"
        self.kernel_config_fwd = kernel_config_fwd
        self.kernel_config_bwd_dW = kernel_config_bwd_dW
        self.kernel_config_bwd_dX = kernel_config_bwd_dX
        self.dW_only = dW_only
        self.dX_only = dX_only

    @classmethod
    def from_hf(
        cls,
        moe_block: Qwen3MoeSparseMoeBlock,
        permute_x: bool                               = True,
        permute_y: bool                               = True,
        autotune: bool                                = True,
        kernel_config_fwd: KernelConfigForward        = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool                                 = False,
        dX_only: bool                                 = False,
    ) -> Qwen3MoeFusedGroupedGEMMBlock:
        """
        Create a Qwen3MoeFusedGroupedGEMMBlock from a HuggingFace MoE block.
        
        Args:
            moe_block (`Qwen3MoeSparseMoeBlock`):
                HuggingFace MoE block to convert.
            permute_x (`bool`):
                Whether to permute input before first GEMM.
            permute_y (`bool`):
                Whether to permute output after second GEMM.
            autotune (`bool`):
                Whether to autotune kernel configurations.
            kernel_config_fwd (`KernelConfigForward`):
                Forward kernel configuration.
            kernel_config_bwd_dW (`KernelConfigBackward_dW`):
                Backward dW kernel configuration.
            kernel_config_bwd_dX (`KernelConfigBackward_dX`):
                Backward dX kernel configuration.
            dW_only (`bool`):
                Whether to compute only dW in backward pass.
            dX_only (`bool`):
                Whether to compute only dX in backward pass.
        
        Returns:
            `Qwen3MoeFusedGroupedGEMMBlock`:
                Newly created fused grouped GEMM MoE block.
        """
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = Qwen3MoeGroupedGEMMBlock.extract_hf_weights(
            moe_block
        )
        return cls(
            config,
            gate,
            gate_up_proj,
            down_proj,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dW_only=dW_only,
            dX_only=dX_only,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fused grouped GEMM MoE block.
        
        Args:
            hidden_states (`torch.Tensor`):
                Input hidden states of shape (batch_size, sequence_length, hidden_dim).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Output hidden states and router logits.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        # Pre-processing
        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. permute_x -> permutation will be fused in prologue of first grouped gemm
        if not self.permute_x:
            hidden_states = permute(hidden_states, gather_indices, self.top_k)
        # Start expert computation
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.gate_up_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=self.permute_x,
            permute_y=False,  # output of first grouped gemm should never be permuted
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=True,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )
        hidden_states = self.act_and_mul(hidden_states)
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.down_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=False,
            permute_y=self.permute_y,
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=False,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )

        # Post-processing
        # 1. Unpermute from expert order to token order
        if not self.permute_y:
            hidden_states = unpermute(hidden_states, gather_indices)

        # 2. Merge topk weights
        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits

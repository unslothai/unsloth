# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

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
Reference implementation of Llama4 MoE block using triton grouped gemm.

`Llama4GroupedGemmTextMoe` is the HF `Llama4TextMoe` block implemented with a torch-native grouped gemm.
`Llama4TritonTextMoe` is the HF `Llama4TextMoe` implemented with triton grouped gemm.
"""


@dataclass
class Llama4MoeResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    hidden_states_after_weight_merge: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    shared_expert_out: torch.Tensor
    final_out: torch.Tensor
    router_logits: torch.Tensor = None


class Llama4GroupedGemmTextMoe(Llama4TextMoe):
    EXPERT_WEIGHT_NAMES = ["experts.gate_up_proj", "experts.down_proj"]

    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        verbose=False,
        debug=False,
    ):
        super().__init__(config)
        self.overlap_router_shared = overlap_router_shared
        self.verbose = verbose
        self.debug = debug

        # Permute in-place expert weights
        E, K, N = self.num_experts, self.hidden_dim, self.experts.expert_dim
        assert self.experts.gate_up_proj.shape == torch.Size([E, K, 2 * N]), (
            f"{self.experts.gate_up_proj.shape} != {[E, K, 2 * N]}"
        )
        permuted_shape = [E, 2 * N, K]
        permuted_stride = [2 * N * K, K, 1]
        if verbose:
            print(
                f"Changing gate_up_proj from {self.experts.gate_up_proj.size()}:{self.experts.gate_up_proj.stride()} to {permuted_shape}:{permuted_stride}"
            )
        with torch.no_grad():
            self.experts.gate_up_proj.as_strided_(permuted_shape, permuted_stride)

        if verbose:
            print(
                f"{self.experts.gate_up_proj.shape}:{self.experts.gate_up_proj.stride()}"
            )

        assert self.experts.down_proj.shape == torch.Size([E, N, K]), (
            f"{self.experts.down_proj.shape} != {[E, N, K]}"
        )
        permuted_shape = [E, K, N]
        permuted_stride = [K * N, N, 1]
        if verbose:
            print(
                f"Changing down_proj from {self.experts.down_proj.size()}:{self.experts.down_proj.stride()} to {permuted_shape}:{permuted_stride}"
            )

        with torch.no_grad():
            self.experts.down_proj.as_strided_(permuted_shape, permuted_stride)

        if verbose:
            print(f"{self.experts.down_proj.shape}:{self.experts.down_proj.stride()}")

        if overlap_router_shared:
            self.shared_expert_stream = torch.cuda.Stream()
            self.default_event = torch.cuda.Event()
            self.shared_expert_end_event = torch.cuda.Event()

    @torch.no_grad
    def copy_weights(self, other: Llama4TextMoe):
        for name, param_to_copy in other.named_parameters():
            if self.verbose:
                print(f"Copying {name} with shape {param_to_copy.shape}")
            param = self.get_parameter(name)

            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                param_to_copy = param_to_copy.permute(0, 2, 1)

            assert param.shape == param_to_copy.shape, (
                f"{param.shape} != {param_to_copy.shape}"
            )
            param.copy_(param_to_copy)

        return self

    def check_weights(self, other: Llama4TextMoe):
        for name, other_param in other.named_parameters():
            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                other_param = other_param.permute(0, 2, 1)
            param = self.get_parameter(name)
            assert param.equal(other_param), f"Param {name} not equal!"
            assert param.is_contiguous(), f"{name} not contiguous!"

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.experts.expert_dim
        gate_proj = x[..., : self.experts.expert_dim]
        up_proj = x[..., self.experts.expert_dim :]
        return self.experts.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        routing_weights = F.sigmoid(routing_weights.float()).to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if self.overlap_router_shared:
            # Marker for all prior ops on default stream
            self.default_event.record()

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        assert routing_weights.shape == (num_tokens, self.top_k), (
            f"{routing_weights.shape} != {(num_tokens, self.top_k)}"
        )

        if self.overlap_router_shared:
            with torch.cuda.stream(self.shared_expert_stream):
                # Ensure prior kernels on default stream complete
                self.default_event.wait()

                shared_expert_out = self.shared_expert(hidden_states)
                # Ensure hidden states remains valid on this stream
                hidden_states.record_stream(self.shared_expert_stream)

                self.shared_expert_end_event.record()

            # Ensure shared expert still valid on default stream
            shared_expert_out.record_stream(torch.cuda.current_stream())
            self.shared_expert_end_event.wait()
        else:
            shared_expert_out = self.shared_expert(hidden_states)

        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )

        if self.top_k > 1:
            hidden_states = hidden_states.sum(dim=1)
        hidden_states_after_weight_merge = hidden_states.view(-1, hidden_dim)

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(
            hidden_states_after_weight_merge, gather_indices, self.top_k
        )
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = torch_grouped_gemm(
            X=hidden_states, W=self.experts.gate_up_proj, m_sizes=token_counts_by_expert
        )
        assert first_gemm.shape == (total_tokens, 2 * self.experts.expert_dim)

        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.experts.expert_dim)

        # See comment above
        second_gemm = torch_grouped_gemm(
            X=intermediate, W=self.experts.down_proj, m_sizes=token_counts_by_expert
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        hidden_states_unpermute = unpermute(second_gemm, gather_indices)
        assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)
        # grouped_gemm_out = hidden_states.view(batch_size, sequence_length, hidden_dim)

        final_out = hidden_states_unpermute + shared_expert_out

        result = (
            Llama4MoeResult(
                token_counts_by_expert=token_counts_by_expert,
                gather_indices=gather_indices,
                topk_weights=routing_weights,
                hidden_states_after_weight_merge=hidden_states_after_weight_merge,
                first_gemm=first_gemm,
                intermediate=intermediate,
                second_gemm=second_gemm,
                hidden_states_unpermute=hidden_states_unpermute,
                shared_expert_out=shared_expert_out,
                final_out=final_out,
                router_logits=router_logits,
            )
            if self.debug
            else (final_out, routing_weights)
        )

        return result


class Llama4TritonTextMoe(Llama4GroupedGemmTextMoe):
    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        permute_x: bool = False,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
        verbose=False,
    ):
        super().__init__(config, overlap_router_shared=overlap_router_shared)
        assert not permute_x, (
            "Llama4 triton grouped gemm does not support permute x due to pre-multiplication of router weights"
        )
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

    @torch.no_grad
    def copy_weights(self, other: Llama4TextMoe):
        for name, param_to_copy in other.named_parameters():
            if self.verbose:
                print(f"Copying {name} with shape {param_to_copy.shape}")
            param = self.get_parameter(name)

            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                param_to_copy = param_to_copy.permute(0, 2, 1)

            assert param.shape == param_to_copy.shape, (
                f"{param.shape} != {param_to_copy.shape}"
            )
            param.copy_(param_to_copy)

        return self

    def check_weights(self, other: Llama4TextMoe):
        for name, other_param in other.named_parameters():
            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                other_param = other_param.permute(0, 2, 1)
            param = self.get_parameter(name)
            assert param.equal(other_param), f"Param {name} not equal!"
            assert param.is_contiguous(), f"{name} not contiguous!"

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.experts.expert_dim
        gate_proj = x[..., : self.experts.expert_dim]
        up_proj = x[..., self.experts.expert_dim :]
        return self.experts.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        routing_weights = F.sigmoid(routing_weights.float()).to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if self.overlap_router_shared:
            # Marker for all prior ops on default stream
            self.default_event.record()

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        assert routing_weights.shape == (num_tokens, self.top_k), (
            f"{routing_weights.shape} != {(num_tokens, self.top_k)}"
        )

        if self.overlap_router_shared:
            with torch.cuda.stream(self.shared_expert_stream):
                # Ensure prior kernels on default stream complete
                self.default_event.wait()

                shared_expert_out = self.shared_expert(hidden_states)
                # Ensure hidden states remains valid on this stream
                hidden_states.record_stream(self.shared_expert_stream)

                self.shared_expert_end_event.record()

            # Ensure shared expert still valid on default stream
            shared_expert_out.record_stream(torch.cuda.current_stream())
            self.shared_expert_end_event.wait()
        else:
            shared_expert_out = self.shared_expert(hidden_states)

        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )

        if self.top_k > 1:
            hidden_states = hidden_states.sum(dim=1)
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(hidden_states, gather_indices, self.top_k)
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.experts.gate_up_proj,
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
            W=self.experts.down_proj,
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
        hidden_states += shared_expert_out

        return hidden_states, routing_weights

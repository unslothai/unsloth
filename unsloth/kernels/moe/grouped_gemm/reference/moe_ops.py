from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    ACT2FN,
    Qwen3MoeSparseMoeBlock,
)


def permute(X: torch.Tensor, gather_indices: torch.Tensor, topk: int):
    """
    Scatters X to a new tensor with shape [total_tokens, hidden_dim] where total_tokens is num_tokens * topk,
    permuting the tokens according to sorted_token_idx.

    Helper for grouped gemm where hidden states need be ordered by expert.
    X: [num_tokens, hidden_dim]
    sorted_token_idx: [num_tokens * topk]
    topk: int

    Returns:
        [total_tokens, hidden_dim]
    """
    assert gather_indices.ndim == 1
    X = X.view(-1, X.shape[-1])
    # Shortcut for topk == 1
    if topk == 1:
        return X[gather_indices]

    return X[gather_indices // topk]


def unpermute(X: torch.Tensor, gather_indices: torch.Tensor):
    X = X.view(-1, X.shape[-1]) if X.ndim > 2 else X
    unpermuted = torch.empty_like(X)
    unpermuted.index_copy_(0, gather_indices, X)
    return unpermuted.view_as(X)


def calculate_topk(
    gating_output: torch.Tensor,
    top_k: int,
    use_sigmoid: bool,
    renormalize: bool,
    pre_act: bool = True,
    post_act: bool = False,
):
    """
    If post_act is True, then activation function is run AFTER topk
    If post_act is False, then activation function is run BEFORE topk

    This is to align with triton_bench implementation (post_act) whereas most models use pre_act (e.g. llama4, deepseek)
    """
    assert pre_act ^ post_act, "only one of pre_act or post_act can be True"

    def _activation(gating_output: torch.Tensor):
        if use_sigmoid:
            scores = torch.sigmoid(gating_output.to(torch.float32)).to(
                gating_output.dtype
            )
        else:
            scores = F.softmax(gating_output.to(torch.float32), dim=1).to(
                gating_output.dtype
            )

        return scores

    if pre_act:
        scores = _activation(gating_output)
    else:
        scores = gating_output

    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

    if post_act:
        topk_weights = _activation(topk_weights)

    if renormalize:
        topk_weights /= torch.sum(topk_weights, dim=-1, keepdim=True).to(
            gating_output.dtype
        )

    return topk_weights, topk_ids


@torch.no_grad()
def get_routing_indices(
    selected_experts, num_experts, return_scatter_indices: bool = False
):
    """
    Returns:
        token_counts_by_expert: [num_experts]
        gather_indices: [num_tokens]
        scatter_indices [Optional] (torch.Tensor):
            Indices for unpermuting gathered inputs back to token order, shape ``(bs * seqlen * top_k,)``.
    """
    # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
    token_counts_by_expert = torch.histc(
        selected_experts.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    # token_indices_experts_sorted shape (bs*slen*top_k,)
    gather_indices = torch.argsort(selected_experts.view(-1), stable=True)
    if return_scatter_indices:
        scatter_indices = gather_indices.argsort()
        return token_counts_by_expert, gather_indices, scatter_indices
    else:
        return token_counts_by_expert, gather_indices


def torch_grouped_gemm(X, W, m_sizes, transpose=True):
    """
    X: [M, K] if forward, else [M, N]
    W: [E, N, K]
    m_sizes: [E]

    Returns:
        Y: [M, N] if forward, else [M, K]
    """
    X = X.view(-1, X.shape[-1])
    M, K = X.shape

    assert m_sizes.ndim == 1
    E = m_sizes.shape[0]

    assert W.ndim == 3
    assert W.shape[0] == E

    N = W.shape[1]

    result = torch.zeros((M, N), dtype=X.dtype, device=X.device)

    m_start = 0
    for g in range(E):
        m_size = m_sizes[g]
        if m_size > 0:
            m_end = m_start + m_size

            # Extract group input
            # m_size x K
            X_g = X[m_start:m_end]
            # N x K
            W_g = W[g]

            # Y_g = X_g @ W_g.T -> [m_size, N]
            W_g = W_g.T if transpose else W_g
            Y_g = X_g @ W_g

            result[m_start:m_end] = Y_g

            m_start = m_end
    return result


@dataclass
class GroupedGEMMResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    hidden_states: torch.Tensor  # final output


class Qwen3MoeGroupedGEMMBlock(nn.Module):
    def __init__(
        self,
        config,
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
    def extract_hf_weights(moe_block: Qwen3MoeSparseMoeBlock):
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
    def from_hf(cls, moe_block: Qwen3MoeSparseMoeBlock):
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = cls.extract_hf_weights(moe_block)
        return cls(config, gate, gate_up_proj, down_proj)

    def check_weights(self, moe_block: Qwen3MoeSparseMoeBlock):
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
        assert x.shape[-1] == 2 * self.moe_intermediate_size
        gate_proj = x[..., : self.moe_intermediate_size]
        up_proj = x[..., self.moe_intermediate_size :]
        return self.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

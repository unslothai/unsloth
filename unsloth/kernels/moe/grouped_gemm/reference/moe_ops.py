# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import torch.nn.functional as F


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
            scores = torch.sigmoid(gating_output.to(torch.float32)).to(gating_output.dtype)
        else:
            scores = F.softmax(gating_output.to(torch.float32), dim=1).to(gating_output.dtype)

        return scores

    if pre_act:
        scores = _activation(gating_output)
    else:
        scores = gating_output

    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

    if post_act:
        topk_weights = _activation(topk_weights)

    if renormalize:
        topk_weights /= torch.sum(topk_weights, dim=-1, keepdim=True).to(gating_output.dtype)

    return topk_weights, topk_ids


@torch.no_grad()
def get_routing_indices(selected_experts, num_experts, return_scatter_indices: bool = False):
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

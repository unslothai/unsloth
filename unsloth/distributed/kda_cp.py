# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi Linear (KimiDeltaAttention / KDA + KimiSparseMoeBlock) context parallel.

KDA belongs to the Linear / Delta Attention family alongside Qwen3.5 GatedDeltaNet
and MiniMax LightningAttention:
1. Input projections (q_proj, k_proj, v_proj, f_a/f_b_proj, b_proj, g_a/g_b_proj)
   run on the local sequence shard ``[B, S/cp, D]``.
2. ``SeqAllToAll4D`` transposes ``q, k, v, f_g, beta`` from sequence-sharded
   ``[B, S/cp, H, D_h]`` to head-sharded ``[B, S, H/cp, D_h]``.
3. Depthwise ``ShortConvolution`` (q_conv1d, k_conv1d, v_conv1d), KDA forget gate
   ``g = -exp(A_log) * softplus(f_g + dt_bias)``, and ``chunk_kda`` run over the
   full sequence ``S`` on each rank's local heads ``H/cp``.
4. ``SeqAllToAll4D`` transposes the attention output ``o`` back to sequence-sharded
   ``[B, S/cp, H, D_h]``, followed by ``o_norm(o, g_out)`` and ``o_proj``.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from .seq_comm import SeqAllToAll4D


def _local_short_conv1d(conv_mod, x: torch.Tensor, c_slice: slice) -> torch.Tensor:
    """Run depthwise causal ShortConvolution on a head-sliced channel range."""
    weight_slice = conv_mod.weight[c_slice]
    bias_slice = conv_mod.bias[c_slice] if conv_mod.bias is not None else None
    activation = getattr(conv_mod, "activation", "silu")
    if x.is_cuda:
        try:
            from fla.modules.conv.causal_conv1d import causal_conv1d

            out, _ = causal_conv1d(
                x = x.contiguous(),
                weight = weight_slice.squeeze(1).contiguous(),
                bias = bias_slice.contiguous() if bias_slice is not None else None,
                initial_state = None,
                output_final_state = False,
                activation = activation,
                backend = getattr(conv_mod, "backend", "triton"),
            )
            return out
        except Exception:
            pass
    x_t = x.transpose(1, 2)
    out = F.conv1d(
        x_t,
        weight_slice,
        bias_slice,
        padding = weight_slice.shape[-1] - 1,
        groups = x_t.shape[1],
    )[..., : x_t.shape[-1]].transpose(1, 2)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out


def _kda_gate(
    f_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute KDA per-channel log-decay gate ``[B, T, H_local, K]`` in float32."""
    # Exact analytical gate: g = -exp(A_log) * softplus(f_g + dt_bias).
    # Works across all fla versions (both old g_bias and new dt_bias signatures)
    # and on both CUDA and CPU with full autograd support.
    h_local, k_dim = f_g.shape[-2], f_g.shape[-1]
    a = -A_log.reshape(1, 1, h_local, 1).float().exp()
    g_in = f_g.float()
    if dt_bias is not None:
        g_in = g_in + dt_bias.reshape(1, 1, h_local, k_dim).float()
    return a * F.softplus(g_in)


def _torch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch reference for KDA recurrence (used on CPU or when fla is absent).

    Shapes:
        q, k, g: ``[B, T, H, K]``
        v:       ``[B, T, H, V]``
        beta:    ``[B, T, H]``
    """
    orig_dtype = v.dtype
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()
    if use_qk_l2norm:
        q_f = F.normalize(q_f, p = 2, dim = -1)
        k_f = F.normalize(k_f, p = 2, dim = -1)
    scale = q_f.shape[-1] ** -0.5
    q_f = q_f * scale

    bsz, seq_len, num_heads, k_dim = q_f.shape
    v_dim = v_f.shape[-1]
    state = q_f.new_zeros((bsz, num_heads, k_dim, v_dim))
    outs = []
    for t in range(seq_len):
        decay = g_f[:, t].exp().unsqueeze(-1)  # [B, H, K, 1]
        state = state * decay
        k_t = k_f[:, t]  # [B, H, K]
        v_t = v_f[:, t]  # [B, H, V]
        b_t = beta_f[:, t].unsqueeze(-1)  # [B, H, 1]
        # Delta rule update: v_new = beta * (v_t - k_t^T @ state)
        kv_pred = torch.einsum("bhk,bhkv->bhv", k_t, state)
        delta_v = b_t * (v_t - kv_pred)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta_v)
        q_t = q_f[:, t]  # [B, H, K]
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, state)
        outs.append(o_t)
    return torch.stack(outs, dim = 1).to(orig_dtype)


def _run_kda_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda:
        import os

        os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
        try:
            from fla.ops.backends import BackendRegistry

            BackendRegistry._initialized.add("kda")
            BackendRegistry._registries.pop("kda", None)
        except Exception:
            pass
        from fla.ops.kda import chunk_kda

        o, _ = chunk_kda(
            q = q.contiguous(),
            k = k.contiguous(),
            v = v.contiguous(),
            g = g.contiguous(),
            beta = beta.contiguous(),
            initial_state = None,
            output_final_state = False,
            use_qk_l2norm_in_kernel = True,
        )
        return o
    return _torch_chunk_kda(q, k, v, g, beta, use_qk_l2norm = True)


def kda_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache_params = None,
    **kwargs,
):
    group = getattr(self, "_cp_group", None)
    cp = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None and cp > 1 else 0

    if attention_mask is not None and attention_mask.dim() == 2:
        # Zero out padded token states before projection / conv / recurrence
        mask_slice = attention_mask[:, -hidden_states.shape[1] :, None].to(hidden_states.dtype)
        hidden_states = hidden_states * mask_slice

    bsz, seq_len, _ = hidden_states.shape
    num_k_heads = getattr(self, "num_k_heads", self.num_heads)
    num_v_heads = getattr(self, "num_v_heads", self.num_heads)
    head_k_dim = getattr(self, "head_k_dim", self.head_dim)
    head_v_dim = getattr(self, "head_v_dim", self.head_dim)

    if cp > 1 and (num_k_heads % cp != 0 or num_v_heads % cp != 0):
        raise RuntimeError(
            f"KDA CP needs cp={cp} to divide key heads {num_k_heads} "
            f"and value heads {num_v_heads}."
        )

    q_proj = self.q_proj(hidden_states)
    k_proj = self.k_proj(hidden_states)
    v_proj = self.v_proj(hidden_states)
    if hasattr(self, "f_a_proj"):
        f_g = self.f_b_proj(self.f_a_proj(hidden_states))
    else:
        f_g = self.f_proj(hidden_states)
    beta = self.b_proj(hidden_states)
    if hasattr(self, "g_a_proj"):
        g_out = self.g_b_proj(self.g_a_proj(hidden_states))
    else:
        g_out = self.g_proj(hidden_states)

    if cp > 1:
        q_proj = SeqAllToAll4D.apply(
            group, q_proj.reshape(bsz, seq_len, num_k_heads, head_k_dim), 2, 1
        )
        k_proj = SeqAllToAll4D.apply(
            group, k_proj.reshape(bsz, seq_len, num_k_heads, head_k_dim), 2, 1
        )
        v_proj = SeqAllToAll4D.apply(
            group, v_proj.reshape(bsz, seq_len, num_v_heads, head_v_dim), 2, 1
        )
        f_g = SeqAllToAll4D.apply(
            group, f_g.reshape(bsz, seq_len, num_v_heads, head_k_dim), 2, 1
        )
        beta = SeqAllToAll4D.apply(
            group, beta.reshape(bsz, seq_len, num_v_heads, 1), 2, 1
        ).squeeze(-1)
    else:
        q_proj = q_proj.reshape(bsz, seq_len, num_k_heads, head_k_dim)
        k_proj = k_proj.reshape(bsz, seq_len, num_k_heads, head_k_dim)
        v_proj = v_proj.reshape(bsz, seq_len, num_v_heads, head_v_dim)
        f_g = f_g.reshape(bsz, seq_len, num_v_heads, head_k_dim)

    full_seq = q_proj.shape[1]
    local_k = num_k_heads // cp
    local_v = num_v_heads // cp
    k_chan_slice = slice(rank * local_k * head_k_dim, (rank + 1) * local_k * head_k_dim)
    v_chan_slice = slice(rank * local_v * head_v_dim, (rank + 1) * local_v * head_v_dim)
    g_chan_slice = slice(rank * local_v * head_k_dim, (rank + 1) * local_v * head_k_dim)
    h_slice = slice(rank * local_v, (rank + 1) * local_v)

    if getattr(self, "use_short_conv", True) and hasattr(self, "q_conv1d"):
        q = _local_short_conv1d(
            self.q_conv1d, q_proj.reshape(bsz, full_seq, local_k * head_k_dim), k_chan_slice
        ).reshape(bsz, full_seq, local_k, head_k_dim)
        k = _local_short_conv1d(
            self.k_conv1d, k_proj.reshape(bsz, full_seq, local_k * head_k_dim), k_chan_slice
        ).reshape(bsz, full_seq, local_k, head_k_dim)
        v = _local_short_conv1d(
            self.v_conv1d, v_proj.reshape(bsz, full_seq, local_v * head_v_dim), v_chan_slice
        ).reshape(bsz, full_seq, local_v, head_v_dim)
    else:
        q = F.silu(q_proj)
        k = F.silu(k_proj)
        v = F.silu(v_proj)

    A_local = self.A_log.reshape(-1)[h_slice]
    dt_local = self.dt_bias.reshape(-1)[g_chan_slice] if getattr(self, "dt_bias", None) is not None else None
    g = _kda_gate(f_g, A_local, dt_local)
    beta = beta.float().sigmoid()
    if getattr(self, "allow_neg_eigval", False):
        beta = beta * 2.0

    o = _run_kda_kernel(q, k, v, g, beta)
    if cp > 1:
        o = SeqAllToAll4D.apply(group, o, 1, 2)

    g_out = g_out.reshape(bsz, seq_len, num_v_heads, head_v_dim)
    o = self.o_norm(o, g_out)
    o = o.reshape(bsz, seq_len, num_v_heads * head_v_dim)
    return self.o_proj(o)


def _kimi_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim = True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)


def _kimi_moe_gate_forward(self, hidden_states: torch.Tensor):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32), None)
    if self.moe_router_activation_func == "sigmoid":
        scores = logits.sigmoid()
    elif self.moe_router_activation_func == "softmax":
        scores = logits.softmax(dim = 1)
    else:
        raise NotImplementedError(
            f"Unsupported scoring function for MoE gating: {self.moe_router_activation_func}"
        )
    scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1)
        .topk(min(2, self.num_experts // self.num_expert_group), dim = -1)[0]
        .sum(dim = -1)
    )
    group_idx = torch.topk(group_scores, k = self.topk_group, dim = -1, sorted = False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz * seq_len, self.num_expert_group, self.num_experts // self.num_expert_group)
        .reshape(bsz * seq_len, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, topk_idx = torch.topk(tmp_scores, k = self.top_k, dim = -1, sorted = False)
    topk_weight = scores.gather(1, topk_idx)
    if self.top_k > 1 and self.moe_renormalize:
        denominator = topk_weight.sum(dim = -1, keepdim = True) + 1e-20
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * self.routed_scaling_factor
    return topk_idx, topk_weight


def _kimi_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight = self.gate(hidden_states)
    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])

    cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim = 0).tolist()
    idxs = topk_idx.view(-1).argsort()
    sorted_tokens = flat_hidden[idxs // topk_idx.shape[1]]

    ep_offset = getattr(self, "ep_rank", 0) * getattr(self, "experts_per_rank", len(self.experts))
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue
        end_idx = start_idx + num_tokens
        expert = self.experts[i + ep_offset]
        outputs.append(expert(sorted_tokens[start_idx:end_idx]))
        start_idx = end_idx

    if outputs:
        outs = torch.cat(outputs, dim = 0)
        inv_idxs = idxs.argsort()
        new_x = outs[inv_idxs]
    else:
        new_x = sorted_tokens.new_zeros((topk_idx.numel(), flat_hidden.shape[-1]))

    y = (
        new_x.view(*topk_idx.shape, -1)
        .to(topk_weight.dtype)
        .mul(topk_weight.unsqueeze(dim = -1))
        .sum(dim = 1)
        .to(flat_hidden.dtype)
        .view(*orig_shape)
    )
    if getattr(self.config, "num_shared_experts", None) is not None:
        y = y + self.shared_experts(identity)
    return y


def patch_kimi_model(model) -> None:
    """Patch KimiLinear modules for training (KDA gate compatibility, differentiable MoE, RMSNorm dtype, SDPA fallback)."""
    has_flash_attn = False
    try:
        import flash_attn  # noqa: F401

        has_flash_attn = True
    except ImportError:
        pass

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "KimiRMSNorm":
            type(module).forward = _kimi_rmsnorm_forward
        elif cls_name == "KimiMoEGate":
            type(module).forward = _kimi_moe_gate_forward
        elif cls_name == "KimiSparseMoeBlock":
            type(module).forward = _kimi_sparse_moe_forward
        elif cls_name == "KimiDeltaAttention":
            if getattr(module, "_cp_original_forward", None) is None:
                module._cp_original_forward = module.forward
            module.forward = kda_forward_with_cp.__get__(module, type(module))
        elif cls_name in ("KimiLinearModel", "KimiMLAAttention"):
            cfg = getattr(module, "config", None)
            if cfg is not None and not has_flash_attn and getattr(cfg, "_attn_implementation", None) == "flash_attention_2":
                cfg._attn_implementation = "sdpa"


def apply_kda_cp(model, group) -> int:
    patch_kimi_model(model)
    n = 0
    for module in model.modules():
        if type(module).__name__ != "KimiDeltaAttention":
            continue
        if getattr(module, "_cp_original_forward", None) is None:
            module._cp_original_forward = module.forward
        module._cp_group = group
        module.forward = kda_forward_with_cp.__get__(module, type(module))
        n += 1
    return n

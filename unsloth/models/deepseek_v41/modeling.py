# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""DeepSeek-V4.1 text model for training, ported from deepseek-ai's reference implementation
(``inference/model.py`` and ``inference/engram.py`` in deepseek-ai/DeepSeek-V4.1-Flash-1017).

Every forward is the reference's prefill (``start_pos == 0``) over the whole sequence; there are no ring buffers.
Citations below are ``model.py:<line>`` / ``engram.py:<line>`` of that reference.

Differences from the reference, all deliberate for training:
* The fp8 / fp4 activation fake-quantization (``act_quant`` / ``fp4_act_quant``, model.py:546-555, 727, 777, 784)
  is not simulated: bf16 activations stay differentiable. Index top-k selection can therefore differ from the
  reference at score ties.
* Routed expert outputs are weighted after the down projection (transformers' experts contract), not before
  (model.py:851). Same math, different rounding.
* The output head is the model dtype (the reference keeps it fp32, model.py:1008).
* DSpark / MTP heads (``mtp.*``) and the vision tower are not built; their checkpoint keys are ignored.

Module and class names follow transformers' DeepSeek-V4 naming, so the released checkpoint loads through a
renaming table and existing Unsloth / unsloth_zoo hooks (non-reentrant checkpointing with the per-forward
``shared`` group state, grouped LoRA for ``o_a_proj``, the engram table store) apply unchanged.
"""

import math
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.generation import GenerationMixin
except Exception:  # pragma: no cover
    from transformers.generation.utils import GenerationMixin

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except Exception:  # pragma: no cover - transformers < 4.53

    class GradientCheckpointingLayer(nn.Module):
        gradient_checkpointing = False

        def __call__(self, *args, **kwargs):
            if self.gradient_checkpointing and self.training:
                from functools import partial
                return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
            return super().__call__(*args, **kwargs)


try:
    from transformers.integrations.moe import use_experts_implementation
except Exception:
    use_experts_implementation = None

from .configuration import DeepseekV41Config, DeepseekV41TextConfig

__all__ = [
    "DeepseekV41PreTrainedModel",
    "DeepseekV41TextModel",
    "DeepseekV41ForCausalLM",
    "DeepseekV41DecoderLayer",
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "DeepseekV41GroupedLinear",
    "DeepseekV41Experts",
    "DeepseekV41EngramEmbedding",
    "DeepseekV41NgramHashState",
    "DeepseekV41GroupState",
    "sparse_attn",
    "sparse_attn_reference",
    "DSV41_STATS",
    "make_engram_table",
    "pipeline_safe_cut_points",
]

ENGRAM_DEAD = -1

# Engagement counters for tests (which path ran). Never touched while compiling (Dynamo would bake / retrace them).
DSV41_STATS = {"sparse_attn_zoo": 0, "sparse_attn_torch": 0, "experts_sorted_loop": 0}


def _count(key):
    if not torch.compiler.is_compiling():
        DSV41_STATS[key] += 1


# ----------------------------------------------------------------------------------------------------------------
# Sparse attention. unsloth_zoo ships the fused kernel; this is the exact torch fallback.
# ----------------------------------------------------------------------------------------------------------------
def sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:311-403 in plain torch. q [b, s, h, d], kv [b, n, d] (one shared KV head, both key and value),
    attn_sink [h] fp32, topk_idxs [b, s, k] int (-1 = empty slot) -> [b, s, h, d] in q's dtype. The sink joins
    the softmax denominator only; a row with no valid index returns zeros (kernel.py:343-346, 380-385)."""
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    gather = topk_idxs.clamp_min(0).long()
    batch = torch.arange(b, device = kv.device).view(b, 1, 1)
    kv_g = kv[batch, gather].float()  # [b, s, k, d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), kv_g) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.cat([scores, sink], dim = -1).softmax(dim = -1)[..., :-1]
    out = torch.einsum("bshk,bskd->bshd", probs, kv_g)
    return out.to(q.dtype)


def _resolve_sparse_attn():
    if os.environ.get("UNSLOTH_DSV41_SPARSE_ATTN", "1") == "0":
        return sparse_attn_reference
    try:
        from unsloth_zoo.sparse_attention import sparse_attn as zoo_sparse_attn
    except Exception:
        return sparse_attn_reference
    return zoo_sparse_attn


_SPARSE_ATTN = None


def ensure_sparse_attn():
    """Resolve the sparse attention implementation once, eagerly (model construction): an import or env read
    inside a traced forward is a graph break."""
    global _SPARSE_ATTN
    if _SPARSE_ATTN is None:
        _SPARSE_ATTN = _resolve_sparse_attn()
    return _SPARSE_ATTN


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    global _SPARSE_ATTN
    if _SPARSE_ATTN is None:
        if torch.compiler.is_compiling():
            return sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale)
        _SPARSE_ATTN = _resolve_sparse_attn()
    _count("sparse_attn_torch" if _SPARSE_ATTN is sparse_attn_reference else "sparse_attn_zoo")
    return _SPARSE_ATTN(q, kv, attn_sink, topk_idxs, softmax_scale)


# ----------------------------------------------------------------------------------------------------------------
# Small pieces
# ----------------------------------------------------------------------------------------------------------------
class DeepseekV41RMSNorm(nn.Module):
    """model.py:281-293: fp32 statistics, weight applied in fp32, cast back."""

    def __init__(
        self,
        dim,
        eps = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim = True) + self.eps)
        return (self.weight.float() * x).to(dtype)


@lru_cache(maxsize = 64)
def _inv_freq_on(dim, base, original_seq_len, factor, beta_fast, beta_slow, device, inference):
    # Keyed by inference mode too: a tensor made under inference_mode must not reach a training graph.
    return _inv_freq(dim, base, original_seq_len, factor, beta_fast, beta_slow).to(device)


def _inv_freq(dim, base, original_seq_len, factor, beta_fast, beta_slow):
    """model.py:368-389 frequencies (YaRN when original_seq_len > 0), as a CPU fp32 tensor."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype = torch.float32) - low) / max(high - low, 1e-3)).clamp(
            0, 1
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def _rope_cos_sin(positions, inv_freq):
    """positions [s] or [b, s] (int) -> cos, sin [..., dim // 2] fp32 (torch.polar of model.py:388-389)."""
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    return angles.cos(), angles.sin()


def apply_rotary(
    x,
    cos,
    sin,
    inverse = False,
):
    """model.py:392-406 on the last ``2 * cos.shape[-1]`` channels: adjacent pairs as complex numbers, rotated by
    the position (conjugate when ``inverse``). Functional; the result is rounded to x's dtype like the reference's
    in-place copy. cos / sin are [s, r/2] or [b, s, r/2]; x is [b, s, d] or [b, s, h, d]."""
    rd = cos.shape[-1] * 2
    head, tail = x[..., :-rd], x[..., -rd:]
    pairs = tail.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if x.dim() == 4:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    if inverse:
        sin = -sin
    rot = torch.stack([a * cos - b * sin, a * sin + b * cos], dim = -1).flatten(-2).to(x.dtype)
    return torch.cat([head, rot], dim = -1)


class DeepseekV41GroupedLinear(nn.Linear):
    """``wo_a`` (model.py:654-658, 787-789): one weight [G * out_g, in_g] holding G independent blocks; input
    [..., G, in_g] -> [..., G, out_g]. An ``nn.Linear`` subclass so quantizers and PEFT see a Linear; the name holds
    ``GroupedLinear`` for transformers' FP8 swap (FP8GroupedLinear) and unsloth's grouped LoRA."""

    def __init__(
        self,
        in_features_per_group,
        out_features,
        n_groups,
        bias = False,
    ):
        super().__init__(in_features_per_group, out_features, bias = bias)
        self.n_groups = n_groups

    def forward(self, x):
        # Groups from the input, not n_groups: under tensor parallelism a rank holds whole groups (colwise rows).
        groups = x.shape[-2]
        weight = self.weight.view(groups, -1, self.in_features)
        out = torch.einsum("...gi,goi->...go", x, weight.to(x.dtype))
        if self.bias is not None:
            out = out + self.bias.view(groups, -1).to(out.dtype)
        return out


# ----------------------------------------------------------------------------------------------------------------
# Hyper-connections (model.py:909-996, kernel.py:406-474)
# ----------------------------------------------------------------------------------------------------------------
class DeepseekV41HyperConnection(nn.Module):
    """The hc_mult-copy residual stream's mixing coefficients for one sublayer: ``fn`` / ``base`` / ``scale`` are
    the checkpoint's ``hc_{attn,ffn}_{fn,base,scale}`` (fp32)."""

    def __init__(self, config):
        super().__init__()
        hc = config.hc_mult
        mix_hc = (2 + hc) * hc
        self.hc_mult = hc
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.fn = nn.Parameter(torch.zeros(mix_hc, hc * config.hidden_size, dtype = torch.float32))
        self.base = nn.Parameter(torch.zeros(mix_hc, dtype = torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype = torch.float32))

    def forward(self, x):
        """x [b, s, hc, d] -> pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc] (fp32)."""
        hc, eps = self.hc_mult, self.hc_eps
        # One statistic over the whole flattened hc*d stream (model.py:953-956). unsloth_zoo's compiler matches these
        # three lines and the Sinkhorn loop below verbatim (eager projection recompute, compiled Sinkhorn).
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim = True) + self.norm_eps)
        mixes = F.linear(flat, self.fn.float()) * rsqrt
        scale, base = self.scale.float(), self.base.float()
        # kernel.py:427-462, the same operation order
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
        comb = comb.softmax(dim = -1) + eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
        return pre, post, comb


def hc_pre(x, pre_mix):
    """model.py:959-962: collapse [b, s, hc, d] to [b, s, d] with pre_mix [b, s, hc]."""
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim = 2).to(x.dtype)


def hc_post(x, residual, post, comb):
    """model.py:964-968: expand x [b, s, d] to hc copies, mixing the residual [b, s, hc, d] through comb."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim = 2
    )
    return y.type_as(x)


# ----------------------------------------------------------------------------------------------------------------
# Attention: sliding window + compressed KV chosen by the indexer (model.py:409-792)
# ----------------------------------------------------------------------------------------------------------------
def _window_idxs(
    window,
    seqlen,
    device = None,
):
    """model.py:409-427 for start_pos == 0: row i lists max(0, i - window + 1) .. i, -1 past i. Built on the
    operand's device: a few arange ops, no host copy."""
    end = torch.arange(seqlen, device = device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window), device = device)
    return torch.where(idxs > end, -1, idxs).to(torch.int32)


class DeepseekV41Compressor(nn.Module):
    """model.py:429-485: pools ``compress_ratio`` consecutive tokens into one KV latent with a learned softmax
    gate (fp32), or projects each token when the ratio is 1. Returns the latent before RoPE. Projections are fp32
    (``_keep_in_fp32_modules``) like the reference's promoted ``wkv`` / ``wgate`` (model.py:444-448); they are
    called as modules so LoRA adapters on them apply."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias = False)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.head_dim, bias = False)
            if self.compress_ratio > 1
            else None
        )
        self.kv_norm = DeepseekV41RMSNorm(config.head_dim, config.rms_norm_eps)

    def _proj(self, module, x):
        weight = module.weight
        wdtype = (
            weight.dtype if weight.is_floating_point() and weight.element_size() >= 2 else x.dtype
        )
        return module(x.to(wdtype)).float()

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return self.kv_norm(self._proj(self.kv_proj, x).to(dtype))
        cutoff = seqlen - seqlen % ratio
        if cutoff == 0:
            return x.new_zeros(bsz, 0, self.head_dim)
        xf = x[:, :cutoff].float()
        kv = self._proj(self.kv_proj, xf).unflatten(1, (-1, ratio))
        score = self._proj(self.gate_proj, xf).unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim = 2)).sum(dim = 2)
        return self.kv_norm(kv.to(dtype))


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """model.py:584-611 verbatim: keep the ``topk_blocks`` best blocks per query (bool mask shaped like logits)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value = float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim = -1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device = logits.device) == last, float("inf")
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim = -1)
    keep = torch.zeros_like(scores, dtype = torch.bool).scatter_(
        -1, top.indices, top.values > float("-inf")
    )
    return keep.repeat_interleave(block_size, dim = -1)[..., :width]


class DeepseekV41Indexer(nn.Module):
    """model.py:488-581: scores every visible compressed position per query with a small side attention and keeps
    the ``index_topk`` best. Selection only: run under no_grad (the LM loss gives top-k no gradient)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_idx]
        cand = config.candidate_source_layer_id
        self.is_candidate_source = layer_idx == cand
        self.uses_candidates = cand is not None and 0 <= cand < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias = False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias = False)
        if self.owns_k:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias = False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, config.rms_norm_eps)

    @torch.no_grad()
    def forward(self, x, qr, latent, shared, cos, sin, ccos, csin, offset):
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        if self.owns_k and latent is not None:
            k = self.k_norm(
                self.k_proj(
                    latent.to(self.k_proj.weight.dtype)
                    if self.k_proj.weight.element_size() >= 2
                    else latent
                )
            )
            shared["index_k"] = apply_rotary(k, ccos, csin)
        index_k = shared["index_k"].to(x.device)
        n_comp = index_k.shape[1]
        q = self.q_b_proj(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary(q, cos, sin)
        weights = self.weights_proj(x).float() * (self.softmax_scale * self.n_heads**-0.5)
        score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim = 2)  # [b, s, t]
        # compressed group j is visible once query i has passed its last token: j < (i + 1) // ratio
        compress_lens = (torch.arange(1, seqlen + 1, device = x.device) // ratio).unsqueeze(-1)
        score = score.masked_fill(
            torch.arange(n_comp, device = x.device) >= compress_lens, float("-inf")
        )
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates and shared.get("candidates") is not None:
            score = score.masked_fill(~shared["candidates"].to(x.device), float("-inf"))
        topk = min(self.index_topk, n_comp)
        idxs = score.topk(topk, dim = -1, sorted = False).indices.sort(dim = -1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).to(torch.int32)


class DeepseekV41Attention(nn.Module):
    """model.py:614-792. Latent MQA over the sliding window of raw KV plus, when ``compress_ratio > 0``, the
    indexer's top-k compressed positions. KV / index sources publish into the per-forward ``shared`` dict that later
    layers of the same group read (model.py:1169-1183)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = self.head_dim**-0.5
        ensure_sparse_attn()
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias = False)
        self.q_a_norm = DeepseekV41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias = False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias = False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_a_proj = DeepseekV41GroupedLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            self.n_groups,
        )
        self.o_b_proj = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias = False)
        self.sinks = nn.Parameter(torch.zeros(self.n_heads, dtype = torch.float32))
        is_backbone = layer_idx < config.num_hidden_layers
        self.is_kv_source = is_backbone and layer_idx in config.kv_source_layer_ids
        self.is_index_source = is_backbone and layer_idx in config.index_source_layer_ids
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        # model.py:695-711: compressing layers rotate at compress_rope_theta with YaRN; sliding-only layers at
        # rope_theta without it.
        if self.compress_ratio:
            self._rope_key = (
                self.rope_head_dim,
                config.compress_rope_theta,
                config.rope_yarn_original_max_position_embeddings,
                config.rope_yarn_factor,
                config.rope_yarn_beta_fast,
                config.rope_yarn_beta_slow,
            )
        else:
            self._rope_key = (
                self.rope_head_dim,
                config.rope_theta,
                0,
                config.rope_yarn_factor,
                config.rope_yarn_beta_fast,
                config.rope_yarn_beta_slow,
            )

    def forward(self, x, positions, shared):
        bsz, seqlen, _ = x.shape
        # torch.is_inference_mode_enabled() is untraceable; traced graphs never run under inference_mode
        inference = False if torch.compiler.is_compiling() else torch.is_inference_mode_enabled()
        inv_freq = _inv_freq_on(*self._rope_key, positions.device, inference)
        cos, sin = _rope_cos_sin(positions, inv_freq)
        qr = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(qr).unflatten(
            -1, (-1, self.head_dim)
        )  # local heads under tensor parallelism
        q = apply_rotary(q, cos, sin)
        kv = apply_rotary(self.kv_norm(self.kv_proj(x)), cos, sin)
        window_idxs = _window_idxs(self.window, seqlen, x.device).unsqueeze(0).expand(bsz, -1, -1)
        if self.compress_ratio:
            ratio = self.compress_ratio
            n_comp = seqlen // ratio
            # a latent stands for the first token of its group: group j sits at position j * ratio (model.py:733)
            cpos = positions[..., : n_comp * ratio : ratio] if n_comp > 0 else positions[..., :0]
            ccos, csin = _rope_cos_sin(cpos, inv_freq)
            latent = None
            if self.is_kv_source:
                latent = self.compressor(x)
            if self.is_index_source:
                if n_comp == 0:
                    idxs = torch.empty(bsz, seqlen, 0, dtype = torch.int32, device = x.device)
                else:
                    idxs = self.indexer(x, qr, latent, shared, cos, sin, ccos, csin, seqlen)
                shared["topk_idxs"] = idxs
            idxs = shared["topk_idxs"].to(x.device)
            if latent is not None:
                shared["compress_kv"] = apply_rotary(latent, ccos, csin)
            compress_kv = shared["compress_kv"][:, :n_comp].to(x.device)
            kv = torch.cat([kv, compress_kv.to(kv.dtype)], dim = 1)
            topk_idxs = torch.cat([window_idxs, idxs], dim = -1)
        else:
            topk_idxs = window_idxs
        o = sparse_attn(q, kv, self.sinks, topk_idxs, self.softmax_scale)
        o = apply_rotary(o, cos, sin, inverse = True)
        o = self.o_a_proj(o.reshape(bsz, seqlen, -1, self.o_a_proj.in_features))
        return self.o_b_proj(o.flatten(2))


# ----------------------------------------------------------------------------------------------------------------
# MoE (model.py:794-906)
# ----------------------------------------------------------------------------------------------------------------
class DeepseekV41TopKRouter(nn.Module):
    """model.py:794-829. Scores sqrt(softplus) (or softmax / sigmoid) in fp32; the correction bias picks experts,
    the unbiased scores weight them. ``e_score_correction_bias_vl`` (image spans) is loaded but text-only routing
    never reads it."""

    def __init__(self, config, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.score_func = config.scoring_func
        self.gate_temp = config.gate_temperature
        self.norm_topk_prob = config.norm_topk_prob
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.zeros(n_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(n_experts, dtype = torch.float32), requires_grad = False
        )
        self.e_score_correction_bias_vl = nn.Parameter(
            torch.zeros(n_experts, dtype = torch.float32), requires_grad = False
        )

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim = -1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            # model.py:819. softplus underflows to exactly 0 for logits <= ~-104 (fp32), where sqrt's backward is
            # 0 / 0 = NaN even for experts that are not picked. Clamping at the smallest normal changes the forward
            # by < 1.1e-19 and gives those rows a zero gradient.
            scores = F.softplus(scores).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        indices = (scores + self.e_score_correction_bias.float()).topk(self.top_k, dim = -1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim = -1, keepdim = True) + 1e-20)
        return weights * self.route_scale, indices


class _DeepseekV41Experts(nn.Module):
    """Routed experts as 3-D tensors (transformers' experts layout): ``gate_up_proj`` [E, 2I, H] holds [w1; w3],
    ``down_proj`` [E, H, I]. ``_apply_gate`` carries the reference's SwiGLU clamps (model.py:845-850), so
    transformers' grouped_mm / batched_mm backends (and Unsloth's) apply them too."""

    def __init__(self, config, n_experts):
        super().__init__()
        self.config = config
        self.num_experts = n_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.zeros(n_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.zeros(n_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.limit = config.swiglu_limit
        self.has_gate = True
        self.has_bias = False

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim = -1)
        if self.limit and self.limit > 0:
            gate = gate.clamp(max = self.limit)
            up = up.clamp(min = -self.limit, max = self.limit)
        return F.silu(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Eager fallback (transformers' / Unsloth's experts backends replace it when configured): routes sorted by
        expert, counts by scatter_add_, one host read of the counts per layer to split the groups."""
        _count("experts_sorted_loop")
        n_tok, top_k = top_k_index.shape
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable = True)
        token_idx = order // top_k
        weights = top_k_weights.reshape(-1)[order].float().unsqueeze(-1)
        counts = torch.zeros(self.num_experts, dtype = torch.int64, device = flat.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        x = hidden_states[token_idx]
        outs = []
        for expert_idx, chunk in enumerate(torch.split(x, counts.tolist())):
            gate_up = F.linear(chunk, self.gate_up_proj[expert_idx].to(chunk.dtype)).float()
            act = self._apply_gate(gate_up).to(chunk.dtype)
            outs.append(F.linear(act, self.down_proj[expert_idx].to(chunk.dtype)).float())
        final = torch.zeros(
            n_tok, hidden_states.shape[-1], dtype = torch.float32, device = hidden_states.device
        )
        if outs:
            final.index_add_(0, token_idx, torch.cat(outs) * weights)
        return final.to(hidden_states.dtype)


if use_experts_implementation is not None:
    try:
        DeepseekV41Experts = use_experts_implementation(_DeepseekV41Experts)
    except Exception:
        DeepseekV41Experts = _DeepseekV41Experts
else:
    DeepseekV41Experts = _DeepseekV41Experts
DeepseekV41Experts.__name__ = "DeepseekV41Experts"
DeepseekV41Experts.__qualname__ = "DeepseekV41Experts"


class DeepseekV41MLP(nn.Module):
    """The shared expert (model.py:832-853, 889): SwiGLU with the same clamps, fp32 activation."""

    def __init__(self, config):
        super().__init__()
        inter = config.moe_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias = False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias = False)
        self.limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x).float()
        up = self.up_proj(x).float()
        if self.limit and self.limit > 0:
            up = up.clamp(min = -self.limit, max = self.limit)
            gate = gate.clamp(max = self.limit)
        return self.down_proj((F.silu(gate) * up).to(x.dtype))


class DeepseekV41SparseMoeBlock(nn.Module):
    """model.py:856-906: top-k routed experts plus one shared expert; summed in fp32."""

    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.num_hidden_layers:
            n_experts, top_k = config.n_routed_experts, config.num_experts_per_tok
        else:
            n_experts = config.dspark_n_routed_experts or config.n_routed_experts
            top_k = config.dspark_num_experts_per_tok or config.num_experts_per_tok
        self.gate = DeepseekV41TopKRouter(config, n_experts, top_k)
        self.experts = DeepseekV41Experts(config, n_experts)
        self.shared_experts = DeepseekV41MLP(config)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        weights, indices = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        y = routed.float() + self.shared_experts(flat).float()
        return y.to(x.dtype).view(shape)


# ----------------------------------------------------------------------------------------------------------------
# Engram (model.py:296-365, engram.py)
# ----------------------------------------------------------------------------------------------------------------
class _TensorTableStore:
    """Rows held in memory as plain tensors (no Parameter / buffer): for a model built from a config rather than
    a checkpoint (tests, tiny-model builders). Duck-types the parts of unsloth_zoo's FrozenTableStore that
    UnslothFrozenTableEmbedding reads."""

    def __init__(
        self,
        weight,
        scale = None,
        block_size = 32,
    ):
        self.weight, self.scale, self.block_size = weight, scale, block_size
        self.num_rows = weight.shape[0]
        self.planes = {}  # nothing file-backed to export
        self._paths = []
        self.nbytes = weight.numel() * weight.element_size()

    def gather(
        self,
        ids,
        dtype = None,
    ):
        idx = ids.to(self.weight.device).long()
        values = F.embedding(idx, self.weight)
        if self.scale is not None and values.dtype == getattr(torch, "float8_e4m3fn", None):
            scales = F.embedding(idx, self.scale).float()
            values = (
                values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            ).flatten(-2)
        return values.to(device = ids.device, dtype = dtype or values.dtype)


def make_engram_table(
    layer_id,
    num_embeddings,
    head_dim,
    block_size = 32,
):
    """Parameter-free table (unsloth_zoo's UnslothFrozenTableEmbedding, bound to the checkpoint after the load)
    when the installed unsloth_zoo provides it, else the Parameter-holding DeepseekV41EngramEmbedding.
    UNSLOTH_DSV41_FROZEN_ENGRAM=0 forces the latter."""
    if os.environ.get("UNSLOTH_DSV41_FROZEN_ENGRAM", "1") != "0":
        try:
            from unsloth_zoo.temporary_patches.deepseek_v41 import make_deepseek_v41_engram_table
        except Exception:
            make_deepseek_v41_engram_table = None
        if make_deepseek_v41_engram_table is not None:
            return make_deepseek_v41_engram_table(layer_id, num_embeddings, head_dim, block_size)
    return DeepseekV41EngramEmbedding(num_embeddings, head_dim, block_size)


def is_parameter_free_table(module):
    return hasattr(module, "frozen_table_store") and not any(
        True for _ in module.parameters(recurse = False)
    )


class DeepseekV41EngramEmbedding(nn.Embedding):
    """One n-gram hash table: rows in the checkpoint dtype (fp8 e4m3 with one e8m0 scale per ``block_size``
    channels in the released files, or a float copy), dequantized on lookup (model.py:312-325). Lives at model level
    (``engram_tables[str(layer)]``) so it can stay out of device placement and be streamed from disk by
    unsloth_zoo's frozen table store. Frozen."""

    def __init__(
        self,
        num_embeddings,
        head_dim,
        block_size = 32,
    ):
        super().__init__(num_embeddings, head_dim)
        self.weight.requires_grad_(False)
        self.block_size = block_size
        self.weight_scale_inv = nn.Parameter(
            torch.ones(num_embeddings, max(1, head_dim // block_size)), requires_grad = False
        )

    def forward(self, ids):
        weight = self.weight
        table_device = weight.device if weight.device.type != "meta" else ids.device
        idx = ids.to(table_device)
        values = F.embedding(idx, weight)
        if weight.dtype == getattr(torch, "float8_e4m3fn", None):
            scales = F.embedding(idx, self.weight_scale_inv).float()
            values = (
                values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            ).flatten(-2)
        return values.to(ids.device)


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


@lru_cache(maxsize = 8)
def engram_layout(layer_ids, max_ngram_size, n_heads, vocab_size):
    """engram.py:86-126: every (layer, n-gram size, head) owns its own prime bucket range; primes are drawn in
    order from ``vocab_size`` and never reused. Returns (primes [L][n-1][heads], offsets [L][(n-1)*heads])."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    offsets = []
    for layer in primes:
        flat = [p for per in layer for p in per]
        acc, row = 0, []
        for p in flat:
            row.append(acc)
            acc += p
        offsets.append(tuple(row))
    return tuple(primes), tuple(offsets)


def compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size):
    """engram.py:64-83: odd per-(layer, look-back) multipliers from a per-layer numpy RNG."""
    import numpy as np

    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low = 0, high = bound, size = (max_ngram_size,), dtype = np.int64)
        rows.append(torch.tensor(values * 2 + 1, device = "cpu"))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer):
    """engram.py:17-61 verbatim: tokens that normalize alike share one compressed id."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens = False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class DeepseekV41NgramHashState(nn.Module):
    """engram.py:129-184 for a full sequence: hash ids of the 2..max_ngram_size-grams ending at each position.
    Look-back stops at the start of the row and at DEAD tokens (padding); the compressed token map comes from the
    tokenizer (``bind_tokenizer``). Holds no parameters or buffers: the config-derived tables are rebuilt per
    device on first use."""

    def __init__(self, config):
        super().__init__()
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.engram_pad_id = config.engram_pad_token_id
        primes, offsets = engram_layout(
            self.layer_ids, self.max_ngram_size, config.engram_n_heads, config.engram_vocab_size
        )
        # Plain Python / CPU data: __init__ may run under a meta device context (from_pretrained).
        self._layout = (primes, offsets)
        self._token_map = None
        self.pad_id = None
        self._device_cache = {}

    def bind_tokenizer(self, tokenizer):
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != self.compressed_vocab_size:
            raise ValueError(
                f"Unsloth: the tokenizer's compressed vocabulary ({vocab_size}) does not match "
                f"engram_compressed_vocab_size ({self.compressed_vocab_size}); the engram hashes would be wrong."
            )
        self.bind_token_map(token_map)

    def bind_token_map(self, token_map):
        self._token_map = torch.as_tensor(token_map, dtype = torch.int64, device = "cpu")
        self.pad_id = int(self._token_map[self.engram_pad_id])
        self._device_cache = {}

    @property
    def bound(self):
        return self._token_map is not None

    def _tables(self, device):
        cached = self._device_cache.get(device)
        if cached is None:
            primes, offsets = self._layout
            cpu = (
                self._token_map,
                torch.tensor(primes, dtype = torch.int64, device = "cpu"),
                torch.tensor(offsets, dtype = torch.int64, device = "cpu"),
                compute_hash_multipliers(
                    self.layer_ids, self.max_ngram_size, self.compressed_vocab_size
                ),
            )
            cached = tuple(t.to(device) for t in cpu)
            self._device_cache[device] = cached
        return cached

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        token_mask = None,
        left_context = None,
    ):
        """-> [b, s, n_engram_layers, (max_ngram_size - 1) * n_heads] int64 row ids.

        A pure function of the token ids and the config (no rank, device or process state). ``left_context``
        [b, k] holds the raw ids of the k tokens right before ``input_ids[:, 0]`` (-1 = no token / padding): a
        sequence-sharded rank (context / sequence parallel) passes the previous shard's last
        ``max_ngram_size - 1`` ids and gets the same hashes as one process over the whole sequence. None = the
        sequence starts here."""
        if self._token_map is None:
            raise ValueError(
                "Unsloth: DeepSeek-V4.1's engram layers need the tokenizer's compressed token map: call "
                "model.bind_tokenizer(tokenizer) (FastLanguageModel.from_pretrained does this)."
            )
        token_map, primes, offsets, multipliers = self._tables(input_ids.device)
        compressed = token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask.to(compressed.device), ENGRAM_DEAD)
        ctx_len = self.max_ngram_size - 1
        bsz, seqlen = compressed.shape
        # History = [left context | ids]; the reference's "positions < shift" start-of-sequence block
        # (engram.py:171-174) is a DEAD left context.
        history = compressed.new_full((bsz, ctx_len), ENGRAM_DEAD)
        if left_context is not None and ctx_len > 0:
            ctx = left_context.to(compressed.device)[:, -ctx_len:]
            ctx = torch.where(ctx >= 0, token_map[ctx.clamp_min(0)], ENGRAM_DEAD)
            history[:, ctx_len - ctx.shape[1] :] = ctx
        history = torch.cat([history, compressed], dim = 1)
        tokens, blocked = [], torch.zeros_like(compressed, dtype = torch.bool)
        for shift in range(self.max_ngram_size):
            source = history[:, ctx_len - shift : ctx_len - shift + seqlen]
            blocked = blocked | (source == ENGRAM_DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim = -1)  # [b, s, n]
        products = tokens.unsqueeze(2) * multipliers  # [b, s, L, n]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
        return torch.cat(hashes, dim = -1) + offsets


class DeepseekV41Engram(nn.Module):
    """model.py:328-365: writes the n-gram rows into the residual stream, gated per hc copy by a normalized dot
    product with a key derived from the same rows."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_hash_index = list(config.engram_layer_ids).index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = nn.Linear(
            n_cols * config.engram_head_dim, config.hidden_size * (config.hc_mult + 1), bias = False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(
        self,
        x,
        rows,
        token_mask = None,
    ):
        # the reference casts dequantized rows to bf16 before wkv (model.py:320); follow the stream's dtype
        kv = self.wkv(rows.flatten(-2).to(device = x.device, dtype = x.dtype))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim = -1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# ----------------------------------------------------------------------------------------------------------------
# Decoder layer and models
# ----------------------------------------------------------------------------------------------------------------
class DeepseekV41GroupState:
    """The per-forward group state that KV / index source layers publish and the later layers of their group read:
    compressed KV, indexer keys, top-k positions, candidate blocks (model.py:1169-1183). Deliberately not a dict:
    FSDP2's input casting (torch.distributed.utils._apply_to_tensors) and accelerate's device hooks rebuild dict
    arguments, subclasses included, so a layer's writes would land in a copy the next layer never sees. Also the
    pipeline-parallel handoff: ``to_tensors`` / ``from_tensors`` (see ``pipeline_handoff_keys``)."""

    __slots__ = ("_d",)

    def __init__(self, items = None):
        self._d = dict(items) if items else {}

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    def get(
        self,
        key,
        default = None,
    ):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def clear(self):
        self._d.clear()

    def update(self, other):
        self._d.update(other.items() if isinstance(other, DeepseekV41GroupState) else other)

    def copy(self):
        return DeepseekV41GroupState(self._d)

    def to_tensors(self, keys = None):
        """{name: tensor} to send to the next pipeline stage (all keys, or ``keys``)."""
        return {k: v for k, v in self._d.items() if keys is None or k in keys}

    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors)


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    """model.py:909-996. Each sublayer's hyper-connection coefficients are used by the *next* sublayer: attention
    collapses the stream with the previous FFN's ``pre`` mix, the FFN with this attention's."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.engram = (
            DeepseekV41Engram(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        )

    def forward(
        self,
        hidden_states,
        pre_mix,
        positions = None,
        shared = None,
        engram_rows = None,
        engram_mask = None,
        **kwargs,
    ):
        if shared is None:
            shared = DeepseekV41GroupState()
        x = hidden_states
        if self.engram is not None and engram_rows is not None:
            x = self.engram(x, engram_rows, engram_mask)
        residual = x
        attn_pre, attn_post, attn_comb = self.attn_hc(x)
        h = self.input_layernorm(hc_pre(x, pre_mix))
        h = self.self_attn(h, positions, shared)
        x = hc_post(h, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(x)
        h = self.post_attention_layernorm(hc_pre(x, attn_pre))
        h = self.mlp(h)
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


def _checkpoint_key_names(path, **hub_kwargs):
    """Tensor names of a local or Hub checkpoint (index.json, else one safetensors header); None if unknown."""
    import json
    import struct

    def _local_or_cached(name):
        local = os.path.join(str(path), name)
        if os.path.isfile(local):
            return local
        try:
            from transformers.utils import cached_file
            return cached_file(
                path, name, _raise_exceptions_for_missing_entries = False, **hub_kwargs
            )
        except Exception:
            return None

    index = _local_or_cached("model.safetensors.index.json")
    if index:
        with open(index) as f:
            return set(json.load(f).get("weight_map", {}))
    single = _local_or_cached("model.safetensors")
    if single:
        with open(single, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return set(k for k in json.loads(f.read(n)) if k != "__metadata__")
    return None


def has_core_model_loading():
    """transformers 5's renaming / merging loader. Checked by attribute: on transformers 4 other packages can leave a
    placeholder ``transformers.core_model_loading`` module (no spec) behind."""
    try:
        import transformers.core_model_loading as core
    except Exception:
        return False
    return hasattr(core, "WeightConverter") and hasattr(core, "WeightRenaming")


def _is_fp8_quant_config(qc):
    if qc is None:
        return False
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return str(getattr(method, "value", method)).lower() == "fp8"


def _released_layout(names):
    return bool(names) and ("embed.weight" in names or any(n.startswith("layers.") for n in names))


_ENGRAM_FULL_SAVE = [0]


def _drop_engram_tables_outside_full_save(module, state_dict, prefix, local_metadata):
    """The engram tables (~98 GB each) go into a state dict only inside save_pretrained of the full model; DCP,
    FSDP2 state dicts, Trainer and adapter saves never carry them (they stay in the base checkpoint)."""
    if _ENGRAM_FULL_SAVE[0] > 0:
        # A file-backed table exports its own checkpoint views (unsloth_zoo); rows held in memory are added here.
        for key, table in getattr(module, "engram_tables", {}).items():
            store = getattr(table, "frozen_table_store", None)
            if isinstance(store, _TensorTableStore):
                state_dict[f"{prefix}engram_tables.{key}.weight"] = store.weight
                if store.scale is not None:
                    state_dict[f"{prefix}engram_tables.{key}.weight_scale_inv"] = store.scale
        return state_dict
    for key in [k for k in state_dict if k.startswith(prefix + "engram_tables.")]:
        del state_dict[key]
    return state_dict


def _engram_cut_safe(config, cut):
    """True when a pipeline stage may start at layer ``cut``: no layer at or after it reads per-forward group
    state (compressed KV, indexer keys, top-k, candidates) written by a layer before it."""
    kv_src, ix_src = list(config.kv_source_layer_ids), list(config.index_source_layer_ids)
    cand = config.candidate_source_layer_id
    for j in range(cut, config.num_hidden_layers):
        if not config.compress_ratios[j]:
            continue
        writers = []
        prior_kv = [s for s in kv_src if s <= j]
        prior_ix = [s for s in ix_src if s <= j]
        if prior_kv:
            writers.append(prior_kv[-1])  # compress_kv and the indexer keys
        if prior_ix:
            writers.append(prior_ix[-1])  # top-k
        if j in ix_src and cand is not None and 0 <= cand < j:
            writers.append(cand)  # candidates
        if any(w < cut for w in writers):
            return False
    return True


def pipeline_safe_cut_points(config):
    """Layer indices where a pipeline stage may start without passing the per-forward group state across stages
    (released config: 0, 1, 2, 8, 14, 20)."""
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    return [i for i in range(config.num_hidden_layers) if _engram_cut_safe(config, i)]


def sequence_segments(position_ids, attention_mask, bsz, seqlen):
    """[(row, start, length)] of the sequences in a batch when a row holds several of them (position_ids restarting
    at 0, padding-free packing) or starts with padding (left padding); None when every row is one sequence from
    column 0 followed only by padding (the plain path). One host read of a flag, then (only when needed) of the
    layout. Each sequence's positions must run 0, 1, 2, ...: offset positions (a continuation) are refused."""
    pos = None
    if position_ids is not None and seqlen > 1:
        pos = position_ids if position_ids.dim() == 2 else position_ids.view(1, -1)
        if pos.shape[0] == 1 and bsz > 1:
            pos = pos.expand(bsz, -1)
    mask = (
        attention_mask.bool() if attention_mask is not None and attention_mask.dim() == 2 else None
    )
    if pos is None and mask is None:
        return None
    flags = []
    if pos is not None:
        flags.append((pos != torch.arange(seqlen, device = pos.device)).any())
    if mask is not None:
        # left padding, or a sequence resuming after padding
        flags.append((~mask[:, 0]).any() | (mask[:, 1:] & ~mask[:, :-1]).any())
    if not bool(torch.stack(flags).any()):
        return None
    pos_rows = pos.tolist() if pos is not None else None
    mask_rows = mask.tolist() if mask is not None else None
    segments = []
    for row in range(bsz):
        start = None
        for col in range(seqlen + 1):
            live = col < seqlen and (mask_rows is None or mask_rows[row][col])
            begins = live and (
                start is None
                or (pos_rows is not None and pos_rows[row][col] != pos_rows[row][col - 1] + 1)
            )
            if start is not None and (not live or begins):
                segments.append((row, start, col - start))
                start = None
            if begins:
                if pos_rows is not None and pos_rows[row][col] != 0:
                    raise NotImplementedError(
                        "Unsloth: DeepSeek-V4.1 needs every sequence's position_ids to start at 0 (packed sequences "
                        "and left padding are fine). Offset positions continue an earlier prefix, which needs the "
                        "compressor / engram state of that prefix; pass the whole sequence instead."
                    )
                start = col
    return segments


_LORA_KV_TARGETS = (
    "self_attn.kv_proj",
    "compressor.kv_proj",
    "compressor.gate_proj",
    "engram.wkv",
)
_LORA_TARGET_MAP = {
    "q_proj": ("self_attn.q_a_proj", "self_attn.q_b_proj"),
    "k_proj": _LORA_KV_TARGETS,
    "v_proj": _LORA_KV_TARGETS,
    "o_proj": ("self_attn.o_a_proj", "self_attn.o_b_proj"),
}


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41EngramEmbedding"]
    # accelerate's device hooks rebuild dict kwargs; the per-forward `shared` group state must stay one object
    _skip_keys_device_placement = ["past_key_values", "shared"]
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    _supports_cache_class = False
    _supports_static_cache = False
    # Kept fp32 like the reference (hc params, sinks, router bias, the pooling compressor).
    _keep_in_fp32_modules = ["attn_hc", "ffn_hc", "sinks", "e_score_correction_bias", "compressor"]
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "sinks",
        "e_score_correction_bias",
        "compressor",
    ]
    # The engram tables stay wherever they fit under device_map (the frozen table store streams them from disk).
    _no_placement_params = ["weight", "weight_scale_inv"]
    _keys_to_ignore_on_load_unexpected = [
        # parameter-free engram tables: bound to the checkpoint by unsloth_zoo after the load
        r"(^|\.)engram_tables\.\d+\.(weight|weight_scale_inv|scale)$",
        r"(^|\.)layers\.\d+\.engram\.embed\.(weight|scale)$",
        r"(^|\.)mtp\.",
        r"(^|\.)vision\.",
        r"(^|\.)aligner\.",
        r"(^|\.)image_(start|end|newline)$",
    ]
    _keys_to_ignore_on_load_missing = [r"engram_tables\.\d+\.weight_scale_inv$"]

    def save_pretrained(self, *args, **kwargs):
        _ENGRAM_FULL_SAVE[0] += 1
        try:
            return super().save_pretrained(*args, **kwargs)
        finally:
            _ENGRAM_FULL_SAVE[0] -= 1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # The released files use DeepSeek's own key names, per-expert weights and ue8m0 fp8 / fp4 blocks. transformers
        # 5 with core_model_loading and an FP8 quantizer that knows fp4 experts loads them directly (the conversion
        # rules registered in __init__.py). Older releases would silently skip every tensor, so the files are
        # converted once into this layout (convert.py) and loaded from there.
        from .convert import convert_released_checkpoint, released_needs_conversion

        if released_needs_conversion():
            hub = {
                k: kwargs[k]
                for k in ("cache_dir", "token", "revision", "subfolder", "local_files_only")
                if k in kwargs
            }
            try:
                names = _checkpoint_key_names(pretrained_model_name_or_path, **hub)
            except Exception:
                names = None
            if _released_layout(names):
                hub.pop("subfolder", None)
                pretrained_model_name_or_path = convert_released_checkpoint(
                    pretrained_model_name_or_path, **hub
                )
                # The converted weights are dequantized: drop the release's FP8 config wherever a caller carried
                # it over (a config object read from the original repo, or a quantization_config kwarg); a
                # bitsandbytes config asked for by the caller stays.
                if _is_fp8_quant_config(kwargs.get("quantization_config")):
                    kwargs.pop("quantization_config")
                config = kwargs.get("config")
                if config is not None and not isinstance(config, (str, os.PathLike)):
                    for c in (config, getattr(config, "text_config", None)):
                        if c is not None and _is_fp8_quant_config(
                            getattr(c, "quantization_config", None)
                        ):
                            try:
                                del c.quantization_config
                            except AttributeError:
                                c.quantization_config = None
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # The engram hashes need the tokenizer's compressed token map; the checkpoint ships the tokenizer.
        target = model[0] if isinstance(model, tuple) else model
        if hasattr(target, "_autobind_tokenizer"):
            hub = {
                k: kwargs[k]
                for k in ("cache_dir", "token", "revision", "local_files_only")
                if k in kwargs
            }
            target._autobind_tokenizer(pretrained_model_name_or_path, **hub)
        return model

    @staticmethod
    def unsloth_lora_target_modules(
        target_modules,
        attention = True,
        mlp = True,
    ):
        """Unsloth's get_peft_model calls this to translate the usual LoRA names into this model's Linears.
        q_proj -> q_a_proj / q_b_proj, k_proj / v_proj -> the single KV head (kv_proj), the compressor (kv_proj /
        gate_proj) and the engram key-value (wkv), o_proj -> o_a_proj (grouped) / o_b_proj; gate / up / down_proj stay
        leaves, which reach the shared expert and, through Unsloth's MoE detection, the routed experts. None and
        "all-linear" select all of them (filtered by the attention / mlp families). The indexer (selection only, no
        gradient) and the routers (frozen) are never targeted. A regex string or names this map does not know pass
        through unchanged."""
        if isinstance(target_modules, str) and target_modules != "all-linear":
            return target_modules
        if target_modules is None or target_modules == "all-linear":
            wanted = (["q_proj", "k_proj", "o_proj"] if attention else []) + (
                ["gate_proj", "up_proj", "down_proj"] if mlp else []
            )
        else:
            wanted = list(target_modules)
        out = []
        for name in wanted:
            for target in _LORA_TARGET_MAP.get(name, (name,)):
                if target not in out:
                    out.append(target)
        return out

    def _autobind_tokenizer(
        self,
        path = None,
        **hub_kwargs,
    ):
        text_model = getattr(self, "model", self)
        hash_state = getattr(text_model, "ngram_hash", None)
        if hash_state is None or hash_state.bound:
            return True
        path = path or getattr(self.config, "_name_or_path", "") or ""
        if not path:
            return False
        try:
            from transformers import AutoTokenizer
            hash_state.bind_tokenizer(AutoTokenizer.from_pretrained(str(path), **hub_kwargs))
            return True
        except Exception:
            return False

    def _init_weights(self, module):
        std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, DeepseekV41EngramEmbedding):
            if module.weight.is_floating_point() and module.weight.element_size() >= 2:
                nn.init.normal_(module.weight, mean = 0.0, std = std)
            nn.init.ones_(
                module.weight_scale_inv
            ) if module.weight_scale_inv.element_size() >= 2 else None
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = std)
        elif isinstance(module, DeepseekV41RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, DeepseekV41HyperConnection):
            nn.init.normal_(module.fn, mean = 0.0, std = std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, _DeepseekV41Experts):
            nn.init.normal_(module.gate_up_proj, mean = 0.0, std = std)
            nn.init.normal_(module.down_proj, mean = 0.0, std = std)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            nn.init.zeros_(module.e_score_correction_bias)
            nn.init.zeros_(module.e_score_correction_bias_vl)
        elif isinstance(module, DeepseekV41Engram):
            nn.init.ones_(module.q_weight)
            nn.init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41Attention):
            nn.init.zeros_(module.sinks)


class DeepseekV41TextModel(DeepseekV41PreTrainedModel):
    config_class = DeepseekV41TextConfig

    def __init__(self, config):
        if hasattr(config, "get_text_config") and not isinstance(config, DeepseekV41TextConfig):
            config = config.get_text_config()
        super().__init__(config)
        self.hc_mult = config.hc_mult
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.engram_tables = nn.ModuleDict()
        if config.engram_layer_ids:
            for i, layer_id in enumerate(config.engram_layer_ids):
                self.engram_tables[str(layer_id)] = make_engram_table(
                    layer_id,
                    config.engram_num_embeddings[i],
                    config.engram_head_dim,
                    config.engram_block_size,
                )
            self.ngram_hash = DeepseekV41NgramHashState(config)
        else:
            self.ngram_hash = None
        self.gradient_checkpointing = False
        self._register_state_dict_hook(_drop_engram_tables_outside_full_save)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_engram_table(
        self,
        layer_id,
        weight,
        scale = None,
    ):
        """Give one engram table its rows from memory (a model built from a config): fp8 rows + per-block e8m0
        scales, or float rows. Held as plain tensors, never Parameters or buffers."""
        table = self.engram_tables[str(layer_id)]
        if is_parameter_free_table(table):
            table.frozen_table_store = _TensorTableStore(
                weight, scale, int(getattr(table, "block_size", 32))
            )
            table._memo = None
            if weight.dtype == getattr(torch, "float8_e4m3fn", None):
                table.table_dtype = torch.float32
        else:
            with torch.no_grad():
                table.weight.data = weight.clone()
                if scale is not None:
                    table.weight_scale_inv.data = scale.clone()

    def bind_tokenizer(self, tokenizer):
        if self.ngram_hash is not None:
            self.ngram_hash.bind_tokenizer(tokenizer)

    def bind_token_map(self, token_map):
        if self.ngram_hash is not None:
            self.ngram_hash.bind_token_map(token_map)

    def _maybe_autobind(self):
        if self.ngram_hash is not None and not self.ngram_hash.bound:
            self._autobind_tokenizer()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        inputs_embeds = None,
        past_key_values = None,
        use_cache = None,
        output_hidden_states = None,
        return_dict = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "Unsloth: DeepSeek-V4.1 needs input_ids (engram hashing reads the token ids)."
                )
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, seqlen = inputs_embeds.shape[:2]
        if not torch.compiler.is_compiling():
            segments = sequence_segments(position_ids, attention_mask, bsz, seqlen)
            if segments is not None:
                return self._forward_segments(
                    segments, input_ids, inputs_embeds, output_hidden_states, **kwargs
                )
        positions = torch.arange(seqlen, device = inputs_embeds.device)
        # right padding only here (sequence_segments moved anything else): pads never reach a real token's hash / gate
        token_mask = (
            attention_mask.bool()
            if attention_mask is not None and attention_mask.dim() == 2
            else None
        )
        engram_rows = {}
        if self.ngram_hash is not None and input_ids is not None:
            self._maybe_autobind()
            hash_ids = self.ngram_hash(input_ids, token_mask, kwargs.get("engram_left_context"))
            for k, layer_id in enumerate(self.ngram_hash.layer_ids):
                with torch.no_grad():
                    engram_rows[layer_id] = self.engram_tables[str(layer_id)](hash_ids[:, :, k, :])
        h = inputs_embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = h.new_zeros(bsz, seqlen, self.hc_mult, dtype = torch.float32)
        pre_mix[:, :, 0] = 1.0  # model.py:1162-1166
        shared = DeepseekV41GroupState()
        all_hidden = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hc_pre(h, pre_mix),)
            h, pre_mix = layer(
                h,
                pre_mix,
                positions = positions,
                shared = shared,
                engram_rows = engram_rows.get(layer.layer_idx),
                engram_mask = token_mask,
            )
        h = self.norm(hc_pre(h, pre_mix))
        if output_hidden_states:
            all_hidden = all_hidden + (h,)
        return BaseModelOutputWithPast(
            last_hidden_state = h, past_key_values = None, hidden_states = all_hidden
        )

    def _forward_segments(self, segments, input_ids, inputs_embeds, output_hidden_states, **kwargs):
        """Packed rows (several sequences, position_ids restarting at 0) or left padding: every sequence is moved to
        its own row starting at column 0 (right padded), run exactly as the reference runs one sequence, and its
        hidden states are put back where it came from (padding columns get zeros). The compressor groups, the sliding
        window, the top-k blocks and the engram n-grams therefore never cross a sequence boundary."""
        if kwargs.get("engram_left_context") is not None:
            raise ValueError(
                "Unsloth: DeepSeek-V4.1 engram_left_context continues one sequence across shards; it cannot be "
                "combined with packed or left-padded rows."
            )
        bsz, seqlen, hidden = inputs_embeds.shape
        device = inputs_embeds.device
        longest = max(length for _, _, length in segments)
        idx = torch.full((len(segments), longest), -1, dtype = torch.long)
        for k, (row, start, length) in enumerate(segments):
            idx[k, :length] = torch.arange(row * seqlen + start, row * seqlen + start + length)
        idx = idx.to(device)
        valid = idx >= 0
        safe = idx.clamp(min = 0)
        embeds = inputs_embeds.reshape(bsz * seqlen, hidden)[safe]
        ids = None
        if input_ids is not None:
            ids = input_ids.reshape(-1)[safe].masked_fill(~valid, 0)
        out = self.forward(
            input_ids = ids,
            attention_mask = valid.long(),
            position_ids = None,
            inputs_embeds = embeds,
            output_hidden_states = output_hidden_states,
            **kwargs,
        )
        dst = idx[valid]

        def back(h):
            full = h.new_zeros(bsz * seqlen, h.shape[-1])
            full = full.index_put((dst,), h[valid])
            return full.view(bsz, seqlen, h.shape[-1])

        hidden_states = None
        if out.hidden_states is not None:
            hidden_states = tuple(back(h) for h in out.hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state = back(out.last_hidden_state),
            past_key_values = None,
            hidden_states = hidden_states,
        )


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    config_class = DeepseekV41Config
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        text = config.get_text_config()
        self.model = DeepseekV41TextModel(text)
        self.vocab_size = text.vocab_size
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias = False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def bind_tokenizer(self, tokenizer):
        self.model.bind_tokenizer(tokenizer)

    def bind_token_map(self, token_map):
        self.model.bind_token_map(token_map)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_hidden_states = None,
        return_dict = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        kwargs.pop("num_logits_to_keep", None)
        kwargs.pop("cache_position", None)
        if self.model.ngram_hash is not None and not self.model.ngram_hash.bound:
            self._autobind_tokenizer()
        out = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            output_hidden_states = output_hidden_states,
            engram_left_context = kwargs.pop("engram_left_context", None),
        )
        hidden = out.last_hidden_state
        sl = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self.lm_head(hidden[:, sl, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits = logits, labels = labels, vocab_size = self.vocab_size, **kwargs
            )
        return CausalLMOutputWithPast(
            loss = loss, logits = logits, past_key_values = None, hidden_states = out.hidden_states
        )

    # Generation recomputes the whole prefix every step (sanity evaluation only; no KV cache).
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask = None,
        **kwargs,
    ):
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

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

"""DeepSeek-V4.1 text model for training, ported from deepseek-ai's reference (inference/model.py, engram.py);
citations are model.py:<line> / engram.py:<line>."""

import json
import math
import os
from copy import deepcopy
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .packed_experts import (
    converted_experts_format,
    packed_experts_load_pre_hook,
    packed_experts_state_dict_hook,
    zoo_saves_mxfp4_dense,
)

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
    "DeepseekV41TextForCausalLM",
    "DeepseekV41DecoderLayer",
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "DeepseekV41GroupedLinear",
    "DeepseekV41Experts",
    "DeepseekV41EngramEmbedding",
    "DeepseekV41NgramHashState",
    "DeepseekV41GroupState",
    "pipeline_handoff_keys",
    "sparse_attn",
    "sparse_attn_reference",
    "DSV41_STATS",
    "make_engram_table",
    "pipeline_safe_cut_points",
]

ENGRAM_DEAD = -1

# Engagement counters for tests; never touched while compiling (Dynamo would bake them).
DSV41_STATS = {
    "sparse_attn_zoo": 0,
    "sparse_attn_torch": 0,
    "experts_sorted_loop": 0,
    "o_a_ungrouped": 0,
    "sparse_attn_window": 0,
    "experts_mxfp4_packed": 0,
    "fast_lora": 0,
}


def _count(key):
    if not torch.compiler.is_compiling():
        DSV41_STATS[key] += 1


class _StaticModuleInts:
    """Keep layer config ints constant while Dynamo traces: as SymInts (allow_unspec_int_on_nn_module), symbolic
    floor divisions sent Inductor into sympy for an hour."""

    __slots__ = ("_patch",)

    def __enter__(self):
        self._patch = None
        config = torch._dynamo.config
        if not torch.compiler.is_compiling() and getattr(
            config, "allow_unspec_int_on_nn_module", False
        ):
            self._patch = config.patch(allow_unspec_int_on_nn_module = False)
            self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        if self._patch is not None:
            self._patch.__exit__(*exc)
        return False


# Opt-in: Unsloth's fused LoRA kernel (kernels/fast_lora.py LoRA_W) for the plain projections.
_FAST_LORA = os.environ.get("UNSLOTH_DSV41_FAST_LORA", "0") == "1"
_LoRA_W = None
if _FAST_LORA:
    try:
        from unsloth.kernels.fast_lora import LoRA_W as _LoRA_W
        from unsloth.kernels.fast_lora import _apply as _fast_lora_apply
        from unsloth.kernels.fast_lora import _lora_dropout_and_bias as _fast_lora_extras
        from unsloth.kernels.fast_lora import get_lora_parameters as _fast_lora_parameters
        from unsloth.kernels.fast_lora import (
            _has_activation_fake_quantizer,
            _has_multiple_active_adapters,
        )
    except Exception:
        _LoRA_W = None


def lora_linear(proj, x):
    """``proj(x)``, via LoRA_W for a plain single-adapter LoRA when UNSLOTH_DSV41_FAST_LORA=1.
    Not for o_a_proj (grouped LoRA), the compressor, or routed experts."""
    if not _FAST_LORA or _LoRA_W is None or not hasattr(proj, "lora_A"):
        return proj(x)
    # Several active adapters or QAT activation / forward-only fake quantization: PEFT's forward handles them.
    if _has_multiple_active_adapters(proj) or _has_activation_fake_quantizer(proj):
        return proj(x)
    extras = _fast_lora_extras((proj,), x)
    if extras is None or extras[0] is not None or extras[1][0] is not None:
        return proj(x)  # PEFT's forward: a variant, dropout, a lora_B bias or a trainable base
    _count("fast_lora")
    flat = x.dim() == 2
    W, W_quant, A, B, S = _fast_lora_parameters(proj)
    out = _fast_lora_apply(_LoRA_W, x.unsqueeze(0) if flat else x, W, W_quant, A, B, S)
    return out.squeeze(0) if flat else out


def sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:311-403 in plain torch."""
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


def _resolve_sparse_attn_window():
    # None (old unsloth_zoo or kill switch) falls back to window + compressed concat via sparse_attn.
    if os.environ.get("UNSLOTH_DSV41_SPARSE_ATTN", "1") == "0":
        return None
    if os.environ.get("UNSLOTH_DSV41_SPARSE_ATTN_WINDOW", "1") == "0":
        return None
    try:
        from unsloth_zoo.sparse_attention import sparse_attn_window as zoo_sparse_attn_window
    except Exception:
        return None
    return zoo_sparse_attn_window


_SPARSE_ATTN = None
_SPARSE_ATTN_WINDOW = None


def ensure_sparse_attn():
    """Resolve the sparse attention implementations once, eagerly (model construction): an import or env read
    inside a traced forward is a graph break."""
    global _SPARSE_ATTN, _SPARSE_ATTN_WINDOW
    if _SPARSE_ATTN is None:
        _SPARSE_ATTN = _resolve_sparse_attn()
        _SPARSE_ATTN_WINDOW = _resolve_sparse_attn_window()
    return _SPARSE_ATTN


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    global _SPARSE_ATTN
    if _SPARSE_ATTN is None:
        if torch.compiler.is_compiling():
            return sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale)
        _SPARSE_ATTN = _resolve_sparse_attn()
    _count("sparse_attn_torch" if _SPARSE_ATTN is sparse_attn_reference else "sparse_attn_zoo")
    return _SPARSE_ATTN(q, kv, attn_sink, topk_idxs, softmax_scale)


def sparse_attn_window(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window):
    """Window KV [b, s, d] plus selected compress_kv rows via unsloth_zoo sparse_attn_window; equals sparse_attn
    over their concatenation (kernel.py:311-403 over model.py:409-427)."""
    _count("sparse_attn_window")
    return _SPARSE_ATTN_WINDOW(
        q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window
    )


_AUTOCAST_DEVICES = frozenset(
    d
    for d in ("cuda", "cpu", "xpu", "mps", "hpu")
    if getattr(torch.amp, "is_autocast_available", lambda _d: _d in ("cuda", "cpu"))(d)
)


def fp32_linear(x, weight):
    """F.linear in fp32 also inside an autocast region (which would run it in fp16 / bf16): the reference's fp32
    router logits and mHC projection (model.py:819, 955); the unnormalised mHC projection overflows fp16."""
    if x.device.type not in _AUTOCAST_DEVICES:
        return F.linear(x.float(), weight.float())
    with torch.autocast(device_type = x.device.type, enabled = False):
        return F.linear(x.float(), weight.float())


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
    the position (conjugate when ``inverse``)."""
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
    [..., G, in_g] -> [..., G, out_g]."""

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
        # Groups from the input: under tensor parallelism a rank holds a subset of whole groups.
        groups = x.shape[-2]
        weight = self.weight.view(groups, -1, self.in_features)
        out = torch.einsum("...gi,goi->...go", x, weight.to(x.dtype))
        if self.bias is not None:
            out = out + self.bias.view(groups, -1).to(out.dtype)
        return out


_GROUPED_LORA_READY = [False]


def ensure_grouped_lora():
    """Install unsloth_zoo's grouped LoRA for DeepseekV41GroupedLinear (PEFT would wrap o_a_proj with a plain
    LoRA whose delta multiplies every group by the whole stacked weight, a shape error)."""
    if _GROUPED_LORA_READY[0]:
        return True
    try:
        from peft.tuners.lora.model import LoraModel
        from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_grouped_lora
        patch_deepseek_v41_grouped_lora()
    except Exception:
        return False
    create = LoraModel.__dict__.get("_create_new_module")
    if getattr(getattr(create, "__func__", create), "_unsloth_dsv41_grouped", False):
        _GROUPED_LORA_READY[0] = True
    return _GROUPED_LORA_READY[0]


def ensure_checkpoint_replay():
    """Install unsloth_zoo's decoder-layer checkpoint replay: a recompute must see the group state it first saw,
    not state later layers overwrote. Idempotent."""
    if getattr(DeepseekV41DecoderLayer.__call__, "_unsloth_patched", False):
        return True
    try:
        from unsloth_zoo.temporary_patches.deepseek_v41 import (
            patch_deepseek_v41_gradient_checkpointing,
        )
        patch_deepseek_v41_gradient_checkpointing()
    except Exception:
        return False
    return bool(getattr(DeepseekV41DecoderLayer.__call__, "_unsloth_patched", False))


def grouped_projection(module, x, out_per_group):
    """``wo_a`` on x [..., G, in_g] -> [..., G, out_per_group]."""
    groups = x.shape[-2]
    base = getattr(module, "base_layer", None)
    if (
        base is not None
        and groups > 1
        and hasattr(base, "n_groups")
        and hasattr(module, "lora_A")
        # Plain PEFT LoRA only: grouped LoRA handlers run themselves.
        and type(module).__module__.startswith("peft.")
    ):
        # Plain PEFT LoRA delta spans the full stacked width: keep only its diagonal blocks.
        out = base(x)
        if not getattr(module, "merged", False) and not getattr(module, "disable_adapters", False):
            for adapter in getattr(module, "active_adapters", ()):
                if adapter not in module.lora_A:
                    continue
                lora_a, lora_b = module.lora_A[adapter], module.lora_B[adapter]
                delta = lora_b(lora_a(module.lora_dropout[adapter](x.to(lora_a.weight.dtype))))
                delta = delta * module.scaling[adapter]
                if delta.shape[-1] == groups * out_per_group:
                    delta = torch.diagonal(
                        delta.unflatten(-1, (groups, out_per_group)), dim1 = -3, dim2 = -2
                    ).movedim(-1, -2)
                out = out + delta.to(out.dtype)
        return out
    out = module(x)
    if groups > 1 and out.shape[-1] == groups * out_per_group:
        _count("o_a_ungrouped")
        out = torch.diagonal(out.unflatten(-1, (groups, out_per_group)), dim1 = -3, dim2 = -2).movedim(
            -1, -2
        )
    return out


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
        # unsloth_zoo's compiler matches these lines and the Sinkhorn loop verbatim (model.py:953-956).
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim = True) + self.norm_eps)
        mixes = fp32_linear(flat, self.fn) * rsqrt
        scale, base = self.scale.float(), self.base.float()
        # kernel.py:427-462 operation order
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
    gate (fp32), or projects each token when the ratio is 1."""

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


# Elements of one indexer score chunk [b, rows, heads, t] (fp32): 1 << 26 = 256 MiB, at most 3 live copies.
_INDEXER_SCORE_BUDGET = 1 << 26


def _indexer_q_ranges(q_len, per_row):
    """Query-row chunks for the indexer score; one chunk under torch.compile (a loop would specialise per length)."""
    if q_len <= 0 or torch.compiler.is_compiling():
        return [(0, q_len)]
    chunk = 0
    env = os.environ.get("UNSLOTH_DSV41_INDEXER_Q_CHUNK")
    if env:
        try:
            chunk = max(1, int(env))
        except ValueError:
            chunk = 0
    if chunk <= 0:
        chunk = max(1, _INDEXER_SCORE_BUDGET // max(1, per_row))
    return [(start, min(start + chunk, q_len)) for start in range(0, q_len, chunk)]


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
    def forward(self, x, qr, latent, cos, sin, ccos, csin, offset, shared):
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
        index_k = index_k.float()
        # compressed group j is visible once query i has passed its last token: j < (i + 1) // ratio
        compress_lens = (torch.arange(1, seqlen + 1, device = x.device) // ratio).unsqueeze(-1)
        visible = torch.arange(n_comp, device = x.device)
        candidates = shared.get("candidates") if self.uses_candidates else None
        topk = min(self.index_topk, n_comp)
        # Rows are independent: chunk queries so the [b, s, heads, t] fp32 score never materialises whole.
        out, picked = [], []
        for start, end in _indexer_q_ranges(seqlen, bsz * self.n_heads * max(1, n_comp)):
            score = torch.einsum("bshd,btd->bsht", q[:, start:end].float(), index_k)
            score = (score.relu() * weights[:, start:end].unsqueeze(-1)).sum(dim = 2)  # [b, c, t]
            lens = compress_lens[start:end]
            score = score.masked_fill(visible >= lens, float("-inf"))
            if self.is_candidate_source:
                picked.append(
                    select_candidate_blocks(
                        score, lens, self.candidate_topk_blocks, self.candidate_block_size
                    )
                )
            elif candidates is not None:
                score = score.masked_fill(~candidates[:, start:end].to(x.device), float("-inf"))
            idxs = score.topk(topk, dim = -1, sorted = False).indices.sort(dim = -1).values
            del score
            out.append(torch.where(idxs < lens, idxs + offset, -1).to(torch.int32))
        if self.is_candidate_source:
            shared["candidates"] = picked[0] if len(picked) == 1 else torch.cat(picked, dim = 1)
        return out[0] if len(out) == 1 else torch.cat(out, dim = 1)


class DeepseekV41Attention(nn.Module):
    """model.py:614-792."""

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
        ensure_grouped_lora()
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
        # model.py:695-711: only compressing layers use compress_rope_theta with YaRN.
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
        # is_inference_mode_enabled() is untraceable
        inference = False if torch.compiler.is_compiling() else torch.is_inference_mode_enabled()
        inv_freq = _inv_freq_on(*self._rope_key, positions.device, inference)
        cos, sin = _rope_cos_sin(positions, inv_freq)
        qr = self.q_a_norm(lora_linear(self.q_a_proj, x))
        q = lora_linear(self.q_b_proj, qr).unflatten(
            -1, (-1, self.head_dim)
        )  # local heads under tensor parallelism
        q = apply_rotary(q, cos, sin)
        kv = apply_rotary(self.kv_norm(lora_linear(self.kv_proj, x)), cos, sin)
        if _SPARSE_ATTN_WINDOW is not None:
            window_idxs = None
        else:
            window_idxs = (
                _window_idxs(self.window, seqlen, x.device).unsqueeze(0).expand(bsz, -1, -1)
            )
        compress_kv = idxs = None
        if self.compress_ratio:
            ratio = self.compress_ratio
            n_comp = seqlen // ratio
            # group j sits at position j * ratio (model.py:733)
            cpos = positions[..., : n_comp * ratio : ratio] if n_comp > 0 else positions[..., :0]
            ccos, csin = _rope_cos_sin(cpos, inv_freq)
            latent = None
            if self.is_kv_source:
                latent = self.compressor(x)
            if self.is_index_source:
                if n_comp == 0:
                    idxs = torch.empty(bsz, seqlen, 0, dtype = torch.int32, device = x.device)
                else:
                    idxs = self.indexer(x, qr, latent, cos, sin, ccos, csin, seqlen, shared = shared)
                shared["topk_idxs"] = idxs
            idxs = shared["topk_idxs"].to(x.device)
            if latent is not None:
                shared["compress_kv"] = apply_rotary(latent, ccos, csin)
            compress_kv = shared["compress_kv"][:, :n_comp].to(device = x.device, dtype = kv.dtype)
        if _SPARSE_ATTN_WINDOW is not None:
            # Indexer rows carry offset = seqlen into the concatenation; shift back into compress_kv.
            if compress_kv is None:
                compress_kv = kv.new_zeros(bsz, 0, kv.shape[-1])
                comp_idxs = torch.empty(bsz, seqlen, 0, dtype = torch.int32, device = kv.device)
            else:
                comp_idxs = torch.where(idxs >= 0, idxs - seqlen, -1).to(torch.int32)
            o = sparse_attn_window(
                q, kv, compress_kv, comp_idxs, self.sinks, self.softmax_scale, self.window
            )
        else:
            if compress_kv is not None:
                kv = torch.cat([kv, compress_kv], dim = 1)
                topk_idxs = torch.cat([window_idxs, idxs], dim = -1)
            else:
                topk_idxs = window_idxs
            o = sparse_attn(q, kv, self.sinks, topk_idxs, self.softmax_scale)
        o = apply_rotary(o, cos, sin, inverse = True)
        o = grouped_projection(
            self.o_a_proj, o.reshape(bsz, seqlen, -1, self.o_a_proj.in_features), self.o_lora_rank
        )
        return lora_linear(self.o_b_proj, o.flatten(2))


class DeepseekV41TopKRouter(nn.Module):
    """model.py:794-829."""

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
        scores = fp32_linear(x, self.weight) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim = -1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            # model.py:819. softplus underflows to 0 below ~-104, making sqrt's backward NaN; clamp to smallest normal.
            scores = F.softplus(scores).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        indices = (scores + self.e_score_correction_bias.float()).topk(self.top_k, dim = -1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim = -1, keepdim = True) + 1e-20)
        return weights * self.route_scale, indices


class _DeepseekV41Experts(nn.Module):
    """Routed experts as 3-D tensors (transformers' experts layout): ``gate_up_proj`` [E, 2I, H] holds [w1; w3],
    ``down_proj`` [E, H, I]."""

    def __init__(self, config, n_experts):
        super().__init__()
        self.config = config
        self.num_experts = n_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        self.has_gate = True
        self.has_bias = False
        E, H, I = n_experts, self.hidden_dim, self.intermediate_dim
        self._mxfp4 = getattr(config, "unsloth_experts_format", None) == "mxfp4"
        if self._mxfp4:
            # fp4 experts kept packed; finalize_packed() turns them into Mxfp4ExpertParam after loading.
            def raw(*shape):
                return nn.Parameter(torch.zeros(shape, dtype = torch.uint8), requires_grad = False)

            self.gate_up_blocks = raw(E, 2 * I, H // 32, 16)
            self.gate_up_scales = raw(E, 2 * I, H // 32)
            self.down_blocks = raw(E, H, I // 32, 16)
            self.down_scales = raw(E, H, I // 32)
            self._register_state_dict_hook(packed_experts_state_dict_hook)
            self._register_load_state_dict_pre_hook(packed_experts_load_pre_hook, with_module = True)
            return
        self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.zeros(E, H, I))

    def finalize_packed(self):
        """Raw MXFP4 blocks / scales -> Mxfp4ExpertParam gate_up_proj / down_proj (idempotent)."""
        if "gate_up_blocks" not in self._parameters:
            return self
        if any(
            self._parameters[n].is_meta
            for n in ("gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales")
        ):
            return self
        from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam

        for name in ("gate_up", "down"):
            blocks = self._parameters.pop(f"{name}_blocks")
            scales = self._parameters.pop(f"{name}_scales")
            self.register_parameter(
                f"{name}_proj",
                Mxfp4ExpertParam(
                    blocks.data,
                    mxfp4_scales = scales.data,
                    mxfp4_transposed = False,
                    mxfp4_dtype = torch.bfloat16,
                ),
            )
        _count("experts_mxfp4_packed")
        device = self.gate_up_proj.device
        if device.type == "cuda":
            # Probe the fused MXFP4 GEMM eagerly so the probe is not a graph break in the compiled forward.
            try:
                from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm_available
                mxfp4_grouped_mm_available(device)
            except Exception:
                pass
        return self

    def _expert_weight(self, param, idx, dtype):
        if getattr(param, "mxfp4_scales", None) is not None:
            from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize
            return mxfp4_dequantize(
                param.data[idx : idx + 1],
                param.mxfp4_scales.to(param.device)[idx : idx + 1],
                dtype = dtype,
                transpose = False,
            )[0]
        return param[idx].to(dtype)

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
        self.finalize_packed()
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
            gate_up = F.linear(
                chunk, self._expert_weight(self.gate_up_proj, expert_idx, chunk.dtype)
            ).float()
            act = self._apply_gate(gate_up).to(chunk.dtype)
            outs.append(
                F.linear(act, self._expert_weight(self.down_proj, expert_idx, chunk.dtype)).float()
            )
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
        gate = lora_linear(self.gate_proj, x).float()
        up = lora_linear(self.up_proj, x).float()
        if self.limit and self.limit > 0:
            up = up.clamp(min = -self.limit, max = self.limit)
            gate = gate.clamp(max = self.limit)
        return lora_linear(self.down_proj, (F.silu(gate) * up).to(x.dtype))


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


class _TensorTableStore:
    """Rows held in memory as plain tensors (no Parameter / buffer): for a model built from a config rather than
    a checkpoint (tests, tiny-model builders)."""

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

    def resident_on(self, device):
        """unsloth_zoo asks whether a lookup on ``device`` needs a host round trip: rows held in memory on that
        device gather there directly."""
        return self.weight.device == device

    def gather(
        self,
        ids,
        dtype = None,
        host_ids = None,
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
    when the installed unsloth_zoo provides it, else the Parameter-holding DeepseekV41EngramEmbedding."""
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
    channels in the released files, or a float copy), dequantized on lookup (model.py:312-325)."""

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
        # A load can widen fp8 rows to the model dtype but keeps the scale; float tables have unit scales.
        fp8 = weight.dtype == getattr(torch, "float8_e4m3fn", None)
        if fp8 or (
            weight.shape[-1] % self.block_size == 0
            and self.weight_scale_inv.shape[-1] == weight.shape[-1] // self.block_size
        ):
            scales = F.embedding(idx, self.weight_scale_inv.to(idx.device)).float()
            values = (
                values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            ).flatten(-2)
            if not fp8:
                values = values.to(weight.dtype)
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
    """engram.py:129-184 for a full sequence: hash ids of the 2..max_ngram_size-grams ending at each position."""

    def __init__(self, config):
        super().__init__()
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.engram_pad_id = config.engram_pad_token_id
        primes, offsets = engram_layout(
            self.layer_ids, self.max_ngram_size, config.engram_n_heads, config.engram_vocab_size
        )
        # __init__ may run under a meta device context (from_pretrained).
        self._layout = (primes, offsets)
        self._token_map = None
        self._host_tables = None
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
        # Not in forward: compiled cache disables compute_hash_multipliers, breaking fullgraph.
        primes, offsets = self._layout
        self._host_tables = (
            self._token_map,
            torch.tensor(primes, dtype = torch.int64, device = "cpu"),
            torch.tensor(offsets, dtype = torch.int64, device = "cpu"),
            compute_hash_multipliers(
                self.layer_ids, self.max_ngram_size, self.compressed_vocab_size
            ),
        )
        self._device_cache = {}

    @property
    def bound(self):
        return self._token_map is not None

    def _tables(self, device):
        cached = self._device_cache.get(device)
        if cached is None:
            cached = tuple(t.to(device) for t in self._host_tables)
            self._device_cache[device] = cached
        return cached

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        token_mask = None,
        left_context = None,
    ):
        """-> [b, s, n_engram_layers, (max_ngram_size - 1) * n_heads] int64 row ids."""
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
        # The reference's start-of-sequence block (engram.py:171-174) is a dead left context.
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
        # reference casts rows to bf16 before wkv (model.py:320); we follow the stream dtype
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


class _SharedState:
    """Per-forward group state that layers publish to and later layers read (compressed KV, indexer keys, top-k,
    candidates; model.py:1169-1183)."""

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
        self._d.update(other.items() if isinstance(other, _SharedState) else other)

    def copy(self):
        return _SharedState(self._d)

    def to_tensors(self, keys = None):
        """{name: tensor} to send to the next pipeline stage (all keys, or ``keys``)."""
        return {k: v for k, v in self._d.items() if keys is None or k in keys}

    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors)


DeepseekV41GroupState = _SharedState


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
            shared = _SharedState()
        # Not a sub-call: recompute under non-reentrant checkpointing re-enters here, outside the model forward.
        with _StaticModuleInts():
            x = hidden_states
            if self.engram is not None and engram_rows is not None:
                x = self.engram(x, engram_rows, engram_mask)
            residual = x
            attn_pre, attn_post, attn_comb = self.attn_hc(x)
            h = self.input_layernorm(hc_pre(x, pre_mix))
            # keyword: accelerate's offload hooks exempt `shared` (_skip_keys_device_placement)
            h = self.self_attn(h, positions, shared = shared._d if hasattr(shared, "_d") else shared)
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
        local = os.path.join(str(path), hub_kwargs.get("subfolder") or "", name)
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
        with open(index, encoding = "utf-8") as f:
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


def _attach_release_quantization_config(cls, path, kwargs, hub):
    try:
        from transformers import PretrainedConfig
        config_dict, _ = PretrainedConfig.get_config_dict(str(path), **hub)
    except Exception:
        return
    release_qc = config_dict.get("quantization_config") or (
        config_dict.get("text_config") or {}
    ).get("quantization_config")
    if not _is_fp8_quant_config(release_qc):
        return
    config = kwargs.get("config")
    if config is None or isinstance(config, (str, os.PathLike)):
        try:
            config = cls.config_class.from_pretrained(str(config or path), **hub)
        except Exception:
            return
    if getattr(config, "quantization_config", None) is None:
        config.quantization_config = dict(release_qc)
    # transformers cannot read the F8_E8M0 header dtype: pass the checkpoint's dtype explicitly.
    release_dtype = config_dict.get("dtype") or config_dict.get("torch_dtype")
    if release_dtype is not None and getattr(config, "dtype", None) is None:
        config.dtype = release_dtype
    kwargs["config"] = config


def _released_layout(names):
    return bool(names) and ("embed.weight" in names or any(n.startswith("layers.") for n in names))


_ENGRAM_FULL_SAVE = [0]


def _drop_engram_tables_outside_full_save(module, state_dict, prefix, local_metadata):
    """The engram tables (~98 GB each) go into a state dict only inside save_pretrained of the full model; DCP,
    FSDP2 state dicts, Trainer and adapter saves never carry them (they stay in the base checkpoint)."""
    if _ENGRAM_FULL_SAVE[0] > 0:
        for key, table in getattr(module, "engram_tables", {}).items():
            store = getattr(table, "frozen_table_store", None)
            if isinstance(store, _TensorTableStore):
                state_dict[f"{prefix}engram_tables.{key}.weight"] = store.weight
                if store.scale is not None:
                    state_dict[f"{prefix}engram_tables.{key}.weight_scale_inv"] = store.scale
        return state_dict
    # Parameter-backed tables hold real weights that loading must find.
    for key, table in getattr(module, "engram_tables", {}).items():
        if not is_parameter_free_table(table):
            continue
        table_prefix = f"{prefix}engram_tables.{key}."
        for name in [k for k in state_dict if k.startswith(table_prefix)]:
            del state_dict[name]
    return state_dict


def _engram_table_entries(model):
    """{state-dict key: tensor} of every engram table a full save writes: rows held in memory (_TensorTableStore)
    and Parameter-backed tables. File-backed tables (unsloth_zoo's store) export through unsloth_zoo's save guard."""
    out = {}
    for module_name, module in model.named_modules():
        tables = getattr(module, "engram_tables", None)
        if not isinstance(tables, nn.ModuleDict):
            continue
        prefix = f"{module_name}." if module_name else ""
        for key, table in tables.items():
            table_prefix = f"{prefix}engram_tables.{key}."
            store = getattr(table, "frozen_table_store", None)
            if isinstance(store, _TensorTableStore):
                out[table_prefix + "weight"] = store.weight
                if store.scale is not None:
                    out[table_prefix + "weight_scale_inv"] = store.scale
            elif not is_parameter_free_table(table):
                for name, tensor in table.state_dict().items():
                    out[table_prefix + name] = tensor
    return out


def _keep_positions(x, positions):
    """x [b, s, ...] at a tensor of kept positions: 1-D (the same for every row) or 2-D [b, k]."""
    positions = torch.as_tensor(positions, device = x.device, dtype = torch.long)
    if positions.dim() <= 1:
        return x[:, positions.reshape(-1)]
    index = positions.view(*positions.shape, *([1] * (x.dim() - 2))).expand(
        *positions.shape, *x.shape[2:]
    )
    return x.gather(1, index)


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


def pipeline_handoff_keys(config, cut):
    """The DeepseekV41GroupState keys a pipeline stage starting at layer ``cut`` needs from the stages before it
    (empty at the safe cut points)."""
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    kv_src, ix_src = list(config.kv_source_layer_ids), list(config.index_source_layer_ids)
    cand = config.candidate_source_layer_id
    keys = set()
    for j in range(cut, config.num_hidden_layers):
        if not config.compress_ratios[j]:
            continue
        prior_kv = [s for s in kv_src if s <= j]
        if j not in kv_src and prior_kv and prior_kv[-1] < cut:
            keys.add("compress_kv")
            if j in ix_src:
                keys.add("index_k")
        prior_ix = [s for s in ix_src if s <= j]
        if j not in ix_src and prior_ix and prior_ix[-1] < cut:
            keys.add("topk_idxs")
        if j in ix_src and cand is not None and 0 <= cand < cut and cand < j:
            keys.add("candidates")
    return sorted(keys)


def _forward_segments_call(model, *args, **kwargs):
    return model._forward_segments(*args, **kwargs)


def sequence_segments(position_ids, attention_mask, bsz, seqlen):
    """[(row, start, length)] of each sequence when rows are packed or left padded; None for the plain path."""
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
    # Qualified: a bare gate_proj would also hit self_attn.compressor.gate_proj.
    "gate_proj": ("mlp.shared_experts.gate_proj",),
    "up_proj": ("mlp.shared_experts.up_proj",),
    "down_proj": ("mlp.shared_experts.down_proj",),
}


def _parent_keywords(fn):
    """Keyword names fn accepts, or None when it takes **kwargs (forward everything)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _released_key_renames():
    """Released -> module name renames; transformers 5 hides them in the text sub-config, which the zoo merge misses."""
    from .convert import _RENAMES

    out = {r"^head\.weight": "lm_head.weight"}
    for source, target in _RENAMES:
        if "\\" not in target:
            out.setdefault(source, target)
    return out


# Not defs, so unsloth_zoo's compiler leaves these eager.
_sequence_segments_outside_trace = torch.compiler.disable(sequence_segments)
_forward_segments_outside_trace = torch.compiler.disable(_forward_segments_call)


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41EngramEmbedding"]
    _unsloth_checkpoint_conversion_mapping = _released_key_renames()
    # accelerate's hooks rebuild dict kwargs; `shared` must stay one object
    _skip_keys_device_placement = ["past_key_values", "shared"]
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    _supports_cache_class = False
    _supports_static_cache = False
    # Kept fp32 like the reference.
    _keep_in_fp32_modules = ["attn_hc", "ffn_hc", "sinks", "e_score_correction_bias", "compressor"]
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "sinks",
        "e_score_correction_bias",
        "compressor",
    ]
    _no_placement_params = ["weight", "weight_scale_inv"]
    _keys_to_ignore_on_load_unexpected = [
        r"(^|\.)engram_tables\.\d+\.(weight|weight_scale_inv|scale)$",
        r"(^|\.)layers\.\d+\.engram\.embed\.(weight|scale)$",
        r"(^|\.)mtp\.",
        r"(^|\.)vision\.",
        r"(^|\.)aligner\.",
        r"(^|\.)image_(start|end|newline)$",
    ]
    _keys_to_ignore_on_load_missing = [r"engram_tables\.\d+\.weight_scale_inv$"]

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs = None,
        **kwargs,
    ):
        """The decoder layers share per-forward group state; a checkpointed layer is exact only through unsloth_zoo's
        replay (a private copy of the state for the recompute). Fail closed without it.

        Extra keywords (transformers 5.22+ Trainer passes every_n_layers / offload) go to the parent when its
        signature has them; an older parent without them gets only gradient_checkpointing_kwargs."""
        if not ensure_checkpoint_replay():
            raise RuntimeError(
                "Unsloth: DeepSeek-V4.1 gradient checkpointing needs unsloth_zoo's decoder-layer replay "
                "(unsloth_zoo.temporary_patches.deepseek_v41); without it the recomputed layers read overwritten "
                "group state and the gradients are wrong. Update unsloth_zoo."
            )
        parent = super().gradient_checkpointing_enable
        if kwargs:
            accepted = _parent_keywords(parent)
            if accepted is not None:
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return parent(gradient_checkpointing_kwargs = gradient_checkpointing_kwargs, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        _ENGRAM_FULL_SAVE[0] += 1
        # unsloth_zoo's save patch dequantizes packed experts, so the saved config describes dense experts.
        dense_save = zoo_saves_mxfp4_dense()
        configs = [
            c for c in (self.config, getattr(self.config, "text_config", None)) if c is not None
        ]
        dropped = []
        if dense_save:
            for c in configs:
                if getattr(c, "unsloth_experts_format", None) is not None:
                    dropped.append((c, c.unsloth_experts_format))
                    del c.unsloth_experts_format
        try:
            # A pre-collected state dict lacks the engram tables; add them without mutating the caller's dict.
            positional = len(args) > 2 and isinstance(args[2], dict)
            supplied = args[2] if positional else kwargs.get("state_dict")
            if isinstance(supplied, dict) and not getattr(self, "_hf_peft_config_loaded", False):
                missing = {
                    k: v for k, v in _engram_table_entries(self).items() if k not in supplied
                }
                if missing:
                    supplied = {**supplied, **missing}
                    if positional:
                        args = (*args[:2], supplied, *args[3:])
                    else:
                        kwargs["state_dict"] = supplied
            return super().save_pretrained(*args, **kwargs)
        finally:
            for c, value in dropped:
                c.unsloth_experts_format = value
            _ENGRAM_FULL_SAVE[0] -= 1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Older transformers silently skip the release's DeepSeek-named tensors: convert once (convert.py).
        from .convert import convert_released_checkpoint, released_needs_conversion

        hub = {
            k: kwargs[k]
            for k in ("cache_dir", "token", "revision", "subfolder", "local_files_only")
            if k in kwargs
        }
        try:
            names = _checkpoint_key_names(pretrained_model_name_or_path, **hub)
        except Exception:
            names = None
        released = _released_layout(names)
        if kwargs.get("config") is not None and not isinstance(
            kwargs["config"], (str, os.PathLike)
        ):
            kwargs["config"] = deepcopy(kwargs["config"])
        if released and not released_needs_conversion():
            from . import _register_conversions
            if not _register_conversions():
                raise RuntimeError(
                    "Unsloth: could not register the DeepSeek-V4.1 checkpoint conversions with transformers."
                )
            # A passed-in config lacks the release's FP8 quantization_config: take the checkpoint's.
            _attach_release_quantization_config(cls, pretrained_model_name_or_path, kwargs, hub)
        if released and released_needs_conversion():
            # the conversion consumes the subfolder
            pretrained_model_name_or_path = convert_released_checkpoint(
                pretrained_model_name_or_path, **hub
            )
            kwargs.pop("subfolder", None)
            # Converted weights are dequantized: drop any carried-over FP8 config (bitsandbytes stays).
            if _is_fp8_quant_config(kwargs.get("quantization_config")):
                kwargs.pop("quantization_config")
            config = kwargs.get("config")
            if isinstance(config, (str, os.PathLike)):
                config = cls.config_class.from_pretrained(str(config), **hub)
                kwargs["config"] = config
            experts_format = converted_experts_format(pretrained_model_name_or_path)
            if config is not None and not isinstance(config, (str, os.PathLike)):
                for c in (config, getattr(config, "text_config", None)):
                    if c is None:
                        continue
                    if _is_fp8_quant_config(getattr(c, "quantization_config", None)):
                        try:
                            del c.quantization_config
                        except AttributeError:
                            c.quantization_config = None
                    if experts_format is not None:
                        c.unsloth_experts_format = experts_format
                    elif getattr(c, "unsloth_experts_format", None) is not None:
                        del c.unsloth_experts_format
        caller_wants_info = bool(kwargs.get("output_loading_info", False))
        if released:
            kwargs["output_loading_info"] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if released:
            model, info = model
            # Never a silent random-weights load.
            missing = sorted(info.get("missing_keys") or [])
            if missing:
                raise RuntimeError(
                    f"Unsloth: {len(missing)} weights of {type(model).__name__} were not found in the DeepSeek-V4.1 "
                    f"checkpoint at {pretrained_model_name_or_path} (e.g. {missing[:3]}); "
                    f"{len(info.get('unexpected_keys') or [])} checkpoint tensors did not map onto the model. "
                    "Load it with AutoModelForCausalLM / DeepseekV41ForCausalLM.from_pretrained(path) and the "
                    "checkpoint's own config, or pass that config (DeepseekV41Config.from_pretrained(path))."
                )
            if caller_wants_info:
                model = (model, info)
        target = model[0] if isinstance(model, tuple) else model
        for module in target.modules():
            # duck-typed: unsloth_zoo's compiler re-creates these classes
            if callable(getattr(module, "finalize_packed", None)):
                module.finalize_packed()
        target = model[0] if isinstance(model, tuple) else model
        if hasattr(target, "_autobind_tokenizer"):
            hub = {
                k: kwargs[k]
                for k in ("cache_dir", "token", "revision", "subfolder", "local_files_only")
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
        """Unsloth's get_peft_model calls this to translate the usual LoRA names into this model's Linears."""
        if isinstance(target_modules, str) and target_modules != "all-linear":
            return target_modules
        if target_modules is None or target_modules == "all-linear":
            wanted = ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            wanted = list(target_modules)
        # Every target is filtered by family: attention = self_attn + engram, mlp = shared and routed experts.
        attention_parts = {"self_attn", "compressor", "indexer", "engram"}
        mlp_parts = {"mlp", "shared_experts", "experts"}
        attention_leaves = {
            "q_a_proj",
            "q_b_proj",
            "kv_proj",
            "o_a_proj",
            "o_b_proj",
            "wkv",
            "weights_proj",
        }
        mlp_leaves = {"gate_up_proj"}
        out = []
        for name in wanted:
            for target in _LORA_TARGET_MAP.get(name, (name,)):
                parts = str(target).split(".")
                is_mlp = bool(mlp_parts.intersection(parts)) or parts[-1] in mlp_leaves
                is_attention = not is_mlp and (
                    bool(attention_parts.intersection(parts)) or parts[-1] in attention_leaves
                )
                if (is_attention and not attention) or (is_mlp and not mlp):
                    continue
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
        elif isinstance(module, _DeepseekV41Experts) and not getattr(module, "_mxfp4", False):
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
        # unsloth_zoo's patches may have run before this module existed
        ensure_checkpoint_replay()
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
        with _StaticModuleInts():
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "Unsloth: DeepSeek-V4.1 needs input_ids (engram hashing reads the token ids)."
                    )
                inputs_embeds = self.embed_tokens(input_ids)
            bsz, seqlen = inputs_embeds.shape[:2]
            # Relayout runs outside the traced graph even under compile, else sequences attend across each other.
            segments = _sequence_segments_outside_trace(position_ids, attention_mask, bsz, seqlen)
            if segments is not None:
                return _forward_segments_outside_trace(
                    self, segments, input_ids, inputs_embeds, output_hidden_states, **kwargs
                )
            positions = torch.arange(seqlen, device = inputs_embeds.device)
            # right padding only here: pads never reach a real token's hash / gate
            token_mask = (
                attention_mask.bool()
                if attention_mask is not None and attention_mask.dim() == 2
                else None
            )
            engram_rows = self.engram_rows(input_ids, token_mask, kwargs.get("engram_left_context"))
            h, pre_mix = self.stream_from_embeddings(inputs_embeds)
            shared = _SharedState()
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

    def engram_rows(
        self,
        input_ids,
        token_mask = None,
        left_context = None,
        layer_ids = None,
    ):
        """{layer_id: rows} gathered from the model-level engram tables for the n-gram hashes of ``input_ids``
        (frozen, no gradient). ``layer_ids`` limits it to the engram layers a pipeline stage owns."""
        rows = {}
        if self.ngram_hash is None:
            return rows
        if layer_ids is not None and not any(l in layer_ids for l in self.ngram_hash.layer_ids):
            return rows
        if input_ids is None:
            # engram hashes need token ids
            raise ValueError(
                "Unsloth: DeepSeek-V4.1's engram layers hash the token ids: pass input_ids (with or without "
                "inputs_embeds), also to a pipeline stage that owns an engram layer."
            )
        self._maybe_autobind()
        hash_ids = self.ngram_hash(input_ids, token_mask, left_context)
        for k, layer_id in enumerate(self.ngram_hash.layer_ids):
            if layer_ids is not None and layer_id not in layer_ids:
                continue
            with torch.no_grad():
                rows[layer_id] = self.engram_tables[str(layer_id)](hash_ids[:, :, k, :])
        return rows

    def stream_from_embeddings(self, inputs_embeds):
        """The hc_mult-copy residual stream and the first lagged pre_mix (model.py:1162-1166)."""
        bsz, seqlen = inputs_embeds.shape[:2]
        h = inputs_embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = h.new_zeros(bsz, seqlen, self.hc_mult, dtype = torch.float32)
        pre_mix[:, :, 0] = 1.0
        return h, pre_mix

    def pipeline_stage_forward(
        self,
        start,
        end,
        hidden_states = None,
        pre_mix = None,
        group_state = None,
        input_ids = None,
        inputs_embeds = None,
        attention_mask = None,
    ):
        """Layers ``[start, end)`` of one pipeline stage."""
        if start == 0 and hidden_states is None:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states, pre_mix = self.stream_from_embeddings(inputs_embeds)
        seqlen = hidden_states.shape[1]
        positions = torch.arange(seqlen, device = hidden_states.device)
        token_mask = (
            attention_mask.bool()
            if attention_mask is not None and attention_mask.dim() == 2
            else None
        )
        if group_state is None:
            group_state = _SharedState()
        elif not isinstance(group_state, _SharedState):
            group_state = _SharedState.from_tensors(group_state)
        owned = [layer.layer_idx for layer in self.layers[start:end]]
        rows = self.engram_rows(input_ids, token_mask, layer_ids = set(owned))
        h = hidden_states
        for layer in self.layers[start:end]:
            h, pre_mix = layer(
                h,
                pre_mix,
                positions = positions,
                shared = group_state,
                engram_rows = rows.get(layer.layer_idx),
                engram_mask = token_mask,
            )
        if end >= len(self.layers):
            h = self.norm(hc_pre(h, pre_mix))
        return h, pre_mix, group_state

    def _forward_segments(self, segments, input_ids, inputs_embeds, output_hidden_states, **kwargs):
        """Move each packed / left-padded sequence to its own right-padded row, run it as the reference does,
        and scatter hidden states back (padding gets zeros)."""
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
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep <= 0):
            # Shift labels over the whole sequence, then pick kept positions; one lm_head / loss pair keeps fused CE.
            hidden, kwargs["shift_labels"] = self._kept_positions(hidden, labels, logits_to_keep)
            sl = slice(None)
        elif not isinstance(logits_to_keep, int):
            hidden = _keep_positions(hidden, logits_to_keep)
        logits = self.lm_head(hidden[:, sl, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits = logits, labels = labels, vocab_size = self.vocab_size, **kwargs
            )
        return CausalLMOutputWithPast(
            loss = loss, logits = logits, past_key_values = None, hidden_states = out.hidden_states
        )

    @staticmethod
    def _kept_positions(hidden, labels, logits_to_keep):
        """(hidden at the kept positions, their next-token targets)."""
        shift_labels = F.pad(labels, (0, 1), value = -100)[..., 1:]
        if isinstance(logits_to_keep, int):
            return hidden[:, -logits_to_keep:], shift_labels[:, -logits_to_keep:].contiguous()
        kept = _keep_positions(shift_labels.unsqueeze(-1), logits_to_keep).squeeze(-1)
        return _keep_positions(hidden, logits_to_keep), kept.contiguous()

    # No KV cache (sanity evaluation only); project only the last position's logits.
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask = None,
        logits_to_keep = None,
        **kwargs,
    ):
        if logits_to_keep is None:
            logits_to_keep = kwargs.get("num_logits_to_keep")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "logits_to_keep": 1 if logits_to_keep is None else logits_to_keep,
        }

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False


class DeepseekV41TextForCausalLM(DeepseekV41ForCausalLM):
    """DeepseekV41ForCausalLM on a flat text config (``deepseek_v41_text``): what AutoModelForCausalLM builds for a
    checkpoint saved from a model constructed on DeepseekV41TextConfig. Same modules and weight names."""

    config_class = DeepseekV41TextConfig

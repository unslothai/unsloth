# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in fused batch-1 NF4 Llama decode: UNSLOTH_DECODE_FUSED_NF4=1 (tuned GPUs) or =force.
A failed check leaves the sequence on the regular path. Each site's NF4 weights are re-homed into
one buffer the projections view, so no memory is added."""

import functools
import os

import torch

from . import _decode_graphs
from ..kernels import nf4_decode_fused as F
from ..kernels import utils as KU

__all__ = ["requested", "decode_forward", "rehome_site", "release", "STATS"]

ENV = "UNSLOTH_DECODE_FUSED_NF4"
STATS = {
    "steps": 0,
    "layers": 0,
    "graph_records": 0,
    "plans": 0,
    "rehomed_sites": 0,
    "refused": None,
}
_PLAN = "_unsloth_fused_nf4_plan"
_HOMES = "_unsloth_fused_nf4_homes"
_PLAN_CACHES = ("_unsloth_nf4_plan", "_unsloth_gemv_plan", "_unsloth_gemv_rows")


def requested():
    return os.environ.get(ENV, "0") in ("1", "force")


def release(model):
    """Drops the fused plan and site caches, which hold the re-homed weight buffers."""
    for module in model.modules():
        d = module.__dict__
        d.pop(_PLAN, None)
        d.pop(_HOMES, None)
    _decode_graphs.invalidate(model)


def _apply_releasing(model, fn, *args, **kwargs):
    # .to() / .cpu() / .half() (and the export offload): the old device's buffers must be freeable.
    release(model)
    return type(model)._apply(model, fn, *args, **kwargs)


def _track_moves(model):
    if "_apply" not in model.__dict__:
        model.__dict__["_apply"] = functools.partial(_apply_releasing, model)


class _Refused:
    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = reason


class _Layer:
    __slots__ = ("attn", "sites", "ln1", "ln2", "head_dim", "n_heads", "n_kv_heads", "rope")


class _Plan:
    pass


def _norm_eps(norm):
    eps = getattr(norm, "variance_epsilon", None)
    return getattr(norm, "eps", None) if eps is None else eps


def _proj_parts(proj, K):
    """(W, quant_state, (A, B, S) or None) of a 4bit NF4 projection the kernel can read, else a str."""
    if (
        KU._has_non_lora_adapter(proj)
        or KU._has_multiple_active_adapters(proj)
        or KU._has_active_dora_adapter(proj)
        or KU._has_active_lora_bias(proj)
        or KU._has_packed_qweight(proj)
        or proj.__dict__.get("_unsloth_tp_local") is not None
    ):
        return "adapter or projection type"
    W, qs, A, B, S, bias = KU.get_lora_parameters_bias(proj)
    if bias is not None:
        return "bias"
    if qs is None or type(qs) is list or getattr(qs, "quant_type", None) != "nf4":
        return "not NF4"
    if (
        W.dtype != torch.uint8
        or not W.is_contiguous()
        or KU._quant_state_dtype(qs) == torch.float32
    ):
        return "NF4 storage"
    N, k = (int(v) for v in qs.shape)
    blocksize = int(qs.blocksize)
    if k != K or K % blocksize or blocksize & (blocksize - 1) or W.numel() * 2 != N * K:
        return "NF4 shape"
    if qs.state2 is not None:
        blocksize2 = int(qs.state2.blocksize)
        # Segments are concatenated: each must hold whole absmax2 blocks.
        if blocksize2 & (blocksize2 - 1) or (N * K // blocksize) % blocksize2:
            return "nested absmax layout"
    elif qs.absmax.dtype != torch.float32:
        return "absmax dtype"
    lora = None
    if A is not None:
        if A.dim() != 2 or A.shape[1] != K or B.shape != (N, A.shape[0]):
            return "LoRA shape"
        if not (A.is_contiguous() and B.is_contiguous()) or A.device != W.device:
            return "LoRA layout"
        if isinstance(S, torch.Tensor):
            S = float(S)
        lora = (A, B, float(S))
    return W, qs, N, lora


def rehome_site(Ws, states):
    """Moves a site's NF4 fields into one buffer each, leaving views; returns (packed, absmax, absmax2)."""
    nested = states[0].state2 is not None
    with _decode_graphs._no_inference_tensors(), torch.no_grad():
        packed = torch.cat([W.reshape(-1) for W in Ws])
        absmax = torch.cat([qs.absmax.reshape(-1) for qs in states])
        absmax2 = torch.cat([qs.state2.absmax.reshape(-1) for qs in states]) if nested else None
    p = a = a2 = 0
    for W, qs in zip(Ws, states):
        n = W.numel()
        W.data = packed[p : p + n].view(W.shape)
        p += n
        n = qs.absmax.numel()
        qs.absmax = absmax[a : a + n]
        a += n
        if nested:
            n = qs.state2.absmax.numel()
            qs.state2.absmax = absmax2[a2 : a2 + n]
            a2 += n
        # Launch plans key on the old tensors and keep them alive.
        for name in _PLAN_CACHES:
            qs.__dict__.pop(name, None)
    return packed, absmax, absmax2


def _homed(home, Ws, states):
    packed, absmax, absmax2, ptrs = home
    return all(
        W.data_ptr() == ptr and qs.absmax.data_ptr() == aptr
        and (qs.state2 is None or qs.state2.absmax.data_ptr() == a2ptr)
        for W, qs, (ptr, aptr, a2ptr) in zip(Ws, states, ptrs)
    )  # fmt: skip


def _site(model, key, kind, projs, K):
    parts = [_proj_parts(p, K) for p in projs]
    for part in parts:
        if isinstance(part, str):
            return part
    Ws = [part[0] for part in parts]
    states = [part[1] for part in parts]
    loras = [part[3] for part in parts]
    st0 = states[0]
    nested = st0.state2 is not None
    for qs in states[1:]:
        if (
            int(qs.blocksize) != int(st0.blocksize)
            or (qs.state2 is not None) != nested
            or (nested and int(qs.state2.blocksize) != int(st0.state2.blocksize))
            or qs.absmax.dtype != st0.absmax.dtype
        ):
            return "segments differ"
    if any(l is None for l in loras) != all(l is None for l in loras):
        return "LoRA on some projections only"
    lora = None
    if loras[0] is not None:
        r = loras[0][0].shape[0]
        if (
            r > 256
            or any(l[0].shape[0] != r or l[2] != loras[0][2] for l in loras)
            or len({l[0].dtype for l in loras}) != 1
            or len({l[1].dtype for l in loras}) != 1
        ):
            return "LoRA ranks / scales / dtypes differ"
        lora = [(l[0], l[1]) for l in loras]
    _track_moves(model)
    homes = model.__dict__.setdefault(_HOMES, {})
    home = homes.get(key)
    if home is None or not _homed(home, Ws, states):
        if len(projs) > 1:
            # Concatenation needs the same tables (one-time checks; equal NF4 / dynamic maps in practice).
            if any(not torch.equal(qs.code, st0.code) for qs in states[1:]):
                return "NF4 tables differ"
            if nested and any(
                not torch.equal(qs.state2.code, st0.state2.code) for qs in states[1:]
            ):
                return "absmax2 tables differ"
        packed, absmax, absmax2 = rehome_site(Ws, states)
        ptrs = tuple(
            (W.data_ptr(), qs.absmax.data_ptr(), qs.state2.absmax.data_ptr() if nested else None)
            for W, qs in zip(Ws, states)
        )
        home = homes[key] = (packed, absmax, absmax2, ptrs)
        STATS["rehomed_sites"] += 1
    packed, absmax, absmax2, _ = home
    bounds, acc = [], 0
    for part in parts:
        acc += part[2]
        bounds.append(acc)
    seg_offset = None
    if nested:
        offsets = [
            qs.offset.reshape(()).float()
            if isinstance(qs.offset, torch.Tensor)
            else torch.tensor(float(qs.offset), device = packed.device)
            for qs in states
        ]
        offsets += [offsets[-1]] * (3 - len(offsets))
        seg_offset = torch.stack(offsets).to(packed.device).contiguous()
    return F.FusedSite(
        kind,
        packed,
        absmax,
        st0.code.float().contiguous(),
        bounds,
        K,
        int(st0.blocksize),
        code2 = st0.state2.code.float().contiguous() if nested else None,
        absmax2 = absmax2,
        seg_offset = seg_offset,
        blocksize2 = int(st0.state2.blocksize) if nested else 256,
        lora = lora,
        scaling = loras[0][2] if lora is not None else 1.0,
    )


def _build(model, X):
    config = model.config
    if getattr(config, "model_type", None) != "llama":
        return "model type"
    device = X.device
    if device.type != "cuda" or torch.version.hip is not None:
        return "not NVIDIA CUDA"
    if X.dtype not in (torch.float16, torch.bfloat16):
        return "dtype"
    capability = torch.cuda.get_device_capability(device)
    if not F.supported_capability(capability) and os.environ.get(ENV) != "force":
        return (
            f"no tuned config for sm{capability[0]}{capability[1]} (UNSLOTH_DECODE_FUSED_NF4=force)"
        )
    if getattr(config, "hidden_act", "silu") != "silu":
        return "activation"
    layers = model.model.layers
    if getattr(layers, "_unsloth_block_swap", None) is not None or len(layers) == 0:
        return "block swap"
    hidden = config.hidden_size
    rehomed = STATS["rehomed_sites"]
    plan = _Plan()
    plan.layers = []
    plan.device = device
    plan.dtype = X.dtype
    plan.capability = capability
    norm = model.model.norm
    eps = _norm_eps(norm)
    for idx, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        if (
            getattr(attn, "q_norm", None) is not None
            or getattr(attn, "k_norm", None) is not None
            or attn.__dict__.get("_unsloth_tp_heads") is not None
            or getattr(attn, "rotary_emb", None) is None
        ):
            return "attention type"
        if (
            _norm_eps(layer.input_layernorm) != eps
            or _norm_eps(layer.post_attention_layernorm) != eps
        ):
            return "norm eps"
        for w in (layer.input_layernorm.weight, layer.post_attention_layernorm.weight):
            if w.device != device or w.shape != (hidden,) or not w.is_contiguous():
                return "norm weight"
        head_dim = attn.head_dim
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        if head_dim % 2:
            return "head dim"
        entry = _Layer()
        entry.attn = attn
        entry.ln1 = layer.input_layernorm.weight
        entry.ln2 = layer.post_attention_layernorm.weight
        entry.head_dim, entry.n_heads, entry.n_kv_heads = head_dim, n_heads, n_kv_heads
        entry.rope = attn.rotary_emb
        sites = {}
        for kind, owner, names, K in (
            ("qkv", attn, ("q_proj", "k_proj", "v_proj"), hidden),
            ("o", attn, ("o_proj",), n_heads * head_dim),
            ("gate_up", mlp, ("gate_proj", "up_proj"), hidden),
            ("down", mlp, ("down_proj",), config.intermediate_size),
        ):
            projs = [getattr(owner, name, None) for name in names]
            if any(p is None for p in projs):
                return "projections"
            site = _site(model, (idx, kind), kind, projs, K)
            if isinstance(site, str):
                return f"{kind}: {site}"
            if site.packed.device != device:
                return "several devices"
            sites[kind] = site
        q_rows = n_heads * head_dim
        if sites["qkv"].bounds != [
            q_rows,
            q_rows + n_kv_heads * head_dim,
            q_rows + 2 * n_kv_heads * head_dim,
        ]:
            return "qkv shape"
        if sites["o"].N != hidden or sites["down"].N != hidden:
            return "o / down shape"
        if sites["gate_up"].bounds[0] * 2 != sites["gate_up"].N:
            return "gate / up shape"
        entry.sites = sites
        plan.layers.append(entry)
    if norm.weight.device != device or norm.weight.shape != (hidden,):
        return "final norm"
    plan.eps = float(eps)
    plan.final_norm = norm.weight
    plan.hidden = hidden
    with _decode_graphs._no_inference_tensors():
        plan.residual = torch.empty(hidden, dtype = torch.float32, device = device)
        # Unnormalized h * g: fp16 can overflow it before the consuming site applies rstd.
        plan.xg = torch.empty(hidden, dtype = torch.float32, device = device)
        plan.ss = (
            torch.zeros(hidden, dtype = torch.float32, device = device),
            torch.zeros(hidden, dtype = torch.float32, device = device),
        )
        plan.act = torch.empty(config.intermediate_size, dtype = X.dtype, device = device)
        plan.pos = torch.zeros(1, dtype = torch.long, device = device)
        plan.slot = torch.zeros(1, dtype = torch.long, device = device)
    plan.rehomed = STATS["rehomed_sites"] != rehomed
    # What a captured step bakes in: rebuilding to an equal signature keeps the old plan (and its buffers).
    plan.signature = (device, X.dtype, hidden, config.intermediate_size) + tuple(
        (
            site.packed.data_ptr(),
            site.absmax.data_ptr(),
            site.absmax2.data_ptr(),
            site.lora and tuple(t.data_ptr() for t in site.A3 + site.B3),
            site.lora and (site.rank, site.scaling),
            tuple(site.bounds),
        )
        for entry in plan.layers
        for site in entry.sites.values()
    )
    STATS["plans"] += 1
    return plan


def decode_forward(model, X, *args):
    """The fused decode step: (normed last hidden state [1, 1, H], next cache), or None (regular path;
    nothing was written). X: the [1, 1, H] embedding of the step's token."""
    if X.device.type != "cuda":
        return None
    # Triton launches run on the current device: a model on cuda:1 while cuda:0 is current needs this.
    with torch.cuda.device(X.device):
        return _decode_forward(model, X, *args)


def _decode_forward(
    model,
    X,
    past_key_values,
    position_ids,
    attention_mask,
    rotary_seq_len,
    attention_core,
    kv_increment,
):
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return None
    layers = model.model.layers
    new_sequence = not hasattr(layers[0].self_attn, "paged_attention")
    plan = model.__dict__.get(_PLAN)
    if new_sequence or plan is None:
        built = _build(model, X)
        if isinstance(built, str):
            STATS["refused"] = built
            built = _Refused(built)
        if isinstance(plan, _Plan) and isinstance(built, _Plan) and not built.rehomed:
            if built.signature == plan.signature:
                built = plan
        if built is not plan:
            if plan is not None or getattr(built, "rehomed", False):
                # Captured decode steps read the old plan's buffers / weight storage, freed now.
                _decode_graphs.invalidate(model)
            plan = model.__dict__[_PLAN] = built
    if isinstance(plan, _Refused) or X.device != plan.device or X.dtype != plan.dtype:
        return None
    static = _decode_graphs.static()
    device = X.device
    capability = plan.capability
    ropes = {}
    for entry in plan.layers:
        rope = entry.rope
        if id(rope) not in ropes:
            rope.extend_rope_embedding(X, rotary_seq_len + 1)
            cos, sin = rope.get_cached(rotary_seq_len, device.index or 0)
            if (
                cos.dim() != 2 or cos.shape != sin.shape or cos.shape[-1] != entry.head_dim
                or cos.stride(-1) != 1 or sin.stride() != cos.stride() or cos.device != device
            ):  # fmt: skip
                return None
            ropes[id(rope)] = (cos, sin)

    if position_ids.dim() == 2 and position_ids.shape[-1] > 1:
        position_ids = position_ids[:, -1:]
    plan.pos.copy_(position_ids.reshape(-1)[-1:])
    seq_len = past_key_values[0][0].shape[-2]
    kv_seq_len = seq_len + 1
    if static is None:
        plan.slot.fill_(seq_len)
        slot = plan.slot
    else:
        slot = static.slot
    residual = plan.residual
    residual.copy_(X.view(-1))
    xg, ss, act, eps = plan.xg, plan.ss, plan.act, plan.eps
    n_layers = len(plan.layers)
    n_ss = 0
    next_cache = []
    for idx, entry in enumerate(plan.layers):
        attn = entry.attn
        sites = entry.sites
        n_heads, n_kv_heads, head_dim = entry.n_heads, entry.n_kv_heads, entry.head_dim
        if not hasattr(attn, "paged_attention"):
            K1, V1 = past_key_values[idx]
            _decode_graphs.alloc_decode_buffers(
                attn, K1, V1, 1, n_heads, n_kv_heads, head_dim, plan.hidden, X.dtype, device,
                kv_increment,
            )  # fmt: skip
        elif static is None and kv_seq_len >= attn.paged_attention.shape[0]:
            _decode_graphs.resize_decode_buffers(
                attn, kv_increment, 1, n_heads, n_kv_heads, head_dim
            )
        cos, sin = ropes[id(entry.rope)]
        k_cache, v_cache = attn.paged_attention_K, attn.paged_attention_V
        q = attn.temp_QA[0]
        common = dict(capability = capability, eps = eps)
        if idx == 0:
            F.fused_site(
                sites["qkv"], residual, q, epi = F.EPI_ROPE_KV, pre = F.PRE_INLOOP,
                norm_weight = entry.ln1, k_cache = k_cache, v_cache = v_cache,
                cache_strides = (k_cache.stride(2), k_cache.stride(0)), cos = cos, sin = sin,
                pos = plan.pos, slot = slot, head_dim = head_dim, **common,
            )  # fmt: skip
        else:
            F.fused_site(
                sites["qkv"], xg, q, epi = F.EPI_ROPE_KV, pre = F.PRE_SCALED, ss_in = ss[0],
                n_ss = n_ss, k_cache = k_cache, v_cache = v_cache,
                cache_strides = (k_cache.stride(2), k_cache.stride(0)), cos = cos, sin = sin,
                pos = plan.pos, slot = slot, head_dim = head_dim, **common,
            )  # fmt: skip
        Qn = q.view(1, 1, n_heads, head_dim).transpose(1, 2)
        if static is None:
            A, Kn, Vn = attention_core(attn, Qn, kv_seq_len, attention_mask, 1, True)
            next_cache.append((Kn, Vn))
        else:
            A = static.attention(
                lambda attn = attn, Qn = Qn: attention_core(
                    attn, Qn, static.kv_len, static.mask, 1, True
                )[0]
            )
            next_cache.append((None, None))
        A = A.transpose(1, 2).reshape(-1)
        n1 = F.fused_site(
            sites["o"], A, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[1],
            xg_out = xg, next_norm_weight = entry.ln2, **common,
        )  # fmt: skip
        F.fused_site(
            sites["gate_up"], xg, act, epi = F.EPI_SWIGLU, pre = F.PRE_SCALED, ss_in = ss[1],
            n_ss = n1, **common,
        )  # fmt: skip
        next_ln = plan.layers[idx + 1].ln1 if idx + 1 < n_layers else plan.final_norm
        n_ss = F.fused_site(
            sites["down"], act, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[0],
            xg_out = xg, next_norm_weight = next_ln, **common,
        )  # fmt: skip
    out = torch.empty((1, 1, plan.hidden), dtype = X.dtype, device = device)
    F.final_norm(residual, ss[0], n_ss, plan.final_norm, eps, out.view(-1))
    if static is None:
        STATS["steps"] += 1
        STATS["layers"] += n_layers
    else:
        STATS["graph_records"] += 1
    return out, next_cache

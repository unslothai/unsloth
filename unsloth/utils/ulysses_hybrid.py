# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for hybrid models: Gated DeltaNet layers and Hugging Face attention.

Linear attention scans run sequentially over the whole sequence, so a sequence shard cannot run its
scan alone. Ulysses exchanges the projected q / k / v / beta / gate inputs from sequence shards to
head shards, runs the short conv and the chunked scan over the full sequence for a subset of heads,
and exchanges the output back before the gated norm and the output projection.
"""

import functools
import inspect

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .ulysses import _AllToAll, ulysses_attention

__all__ = [
    "GATED_DELTA_ATTENTION_CLASSES",
    "is_ulysses_gated_delta",
    "check_gated_delta_heads",
    "ulysses_gated_delta_forward",
    "wrap_gated_delta_forward",
    "install_hf_attention_wrapper",
]

# Hybrid full attention layers: Hugging Face forwards that call ALL_ATTENTION_FUNCTIONS after RoPE.
GATED_DELTA_ATTENTION_CLASSES = (
    "Qwen3NextAttention",
    "Qwen3_5Attention",
    "Qwen3_5MoeAttention",
)


def is_ulysses_gated_delta(module) -> bool:
    if not type(module).__name__.endswith("GatedDeltaNet") or not hasattr(module, "conv1d"):
        return False
    if hasattr(module, "in_proj_qkvz"):
        return hasattr(module, "in_proj_ba") and hasattr(module, "fix_query_key_value_ordering")
    return all(hasattr(module, n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))


def check_gated_delta_heads(module, cp_size) -> None:
    for name in ("num_k_heads", "num_v_heads"):
        heads = getattr(module, name)
        if heads % cp_size != 0:
            raise ValueError(
                f"Unsloth: Ulysses context parallelism needs linear attention {name} ({heads}) "
                f"to be divisible by context_parallel_size ({cp_size})."
            )


def _kernel_params(fn):
    # transformers' hub wrapper silently drops keyword arguments its bound implementation does not name.
    from .packing import _hub_closure

    params = _hub_closure(fn).get("applicable_params")
    if params is not None:
        return set(params)
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _gated_delta_kernels(module):
    # transformers 4.57 binds the kernels on the instance; 5.x calls module globals.
    attrs = vars(module)
    globs = getattr(inspect.getmodule(type(module)), "__dict__", {})
    conv = attrs.get("_unsloth_varlen_orig_conv") or attrs.get(
        "causal_conv1d_fn", globs.get("causal_conv1d_fn")
    )
    scan = (
        attrs.get("_unsloth_varlen_orig_scan")
        or attrs.get("chunk_gated_delta_rule")
        or globs.get("torch_chunk_gated_delta_rule")
        or globs.get("chunk_gated_delta_rule")
    )
    if scan is None:
        raise RuntimeError(
            f"Unsloth: no chunked gated delta rule kernel found for {type(module).__name__}."
        )
    return conv, scan


def _packed_boundaries(kwargs, total, device):
    if kwargs.get("packed_seq_lengths") is None:
        return None
    from .packing import _seq_idx_from_cu_seqlens, get_packed_info_from_kwargs

    info = get_packed_info_from_kwargs(kwargs, device)
    built = None if info is None else _seq_idx_from_cu_seqlens(info[1], total)
    if built is None:
        raise ValueError(
            "Unsloth: Ulysses context parallelism got packed_seq_lengths that do not cover the sequence."
        )
    return built


def ulysses_gated_delta_forward(
    module,
    hidden_states,
    cache_params = None,
    attention_mask = None,
    *,
    group,
    **kwargs,
):
    """Training forward of a Qwen3-Next / Qwen3.5 GatedDeltaNet on a contiguous sequence shard."""
    if cache_params is not None:
        raise RuntimeError("Unsloth: Ulysses context parallelism does not support a cache.")
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    check_gated_delta_heads(module, size)
    bsz, local_len, _ = hidden_states.shape
    total = local_len * size
    k_heads, v_heads = module.num_k_heads // size, module.num_v_heads // size
    dk, dv = module.head_k_dim, module.head_v_dim

    if hasattr(module, "in_proj_qkvz"):
        q, k, v, z, b, a = module.fix_query_key_value_ordering(
            module.in_proj_qkvz(hidden_states), module.in_proj_ba(hidden_states)
        )
    else:
        q, k, v = torch.split(
            module.in_proj_qkv(hidden_states),
            [module.key_dim, module.key_dim, module.value_dim],
            dim = -1,
        )
        z = module.in_proj_z(hidden_states)
        b = module.in_proj_b(hidden_states)
        a = module.in_proj_a(hidden_states)

    # Heads are contiguous per rank, so v head h keeps k head h // (num_v_heads // num_k_heads).
    pieces = [x.reshape(bsz, local_len, size, -1) for x in (q, k, v, b, a)]
    widths = [p.shape[-1] for p in pieces]
    mixed = _AllToAll.apply(torch.cat(pieces, dim = -1), 2, 1, group).squeeze(2)
    q, k, v, b, a = torch.split(mixed, widths, dim = -1)

    weight = module.conv1d.weight.squeeze(1)
    bias = module.conv1d.bias
    key_dim = module.key_dim
    starts = (rank * widths[0], key_dim + rank * widths[1], 2 * key_dim + rank * widths[2])
    rows = torch.cat(
        [torch.arange(s, s + w, device = weight.device) for s, w in zip(starts, widths[:3])]
    )
    weight = weight.index_select(0, rows)
    bias = None if bias is None else bias.index_select(0, rows)

    conv, scan = _gated_delta_kernels(module)
    boundaries = _packed_boundaries(kwargs, total, hidden_states.device)
    if boundaries is not None:
        if conv is None or "seq_idx" not in _kernel_params(conv):
            raise RuntimeError(
                "Unsloth: Ulysses context parallelism with packing needs a causal conv1d kernel "
                "that takes seq_idx (install causal_conv1d or use unsloth_zoo's kernels)."
            )
        if "cu_seqlens" not in _kernel_params(scan):
            raise RuntimeError(
                "Unsloth: Ulysses context parallelism with packing needs a gated delta rule kernel "
                "that takes cu_seqlens (install flash-linear-attention or use unsloth_zoo's kernels)."
            )
    mixed_qkv = torch.cat([q, k, v], dim = -1).transpose(1, 2)
    if conv is None:
        mixed_qkv = F.conv1d(
            mixed_qkv.to(weight.dtype),
            weight.unsqueeze(1),
            bias,
            padding = weight.shape[-1] - 1,
            groups = weight.shape[0],
        )[..., :total]
        mixed_qkv = F.silu(mixed_qkv)
    else:
        conv_kwargs = {"activation": module.activation}
        if boundaries is not None:
            conv_kwargs["seq_idx"] = boundaries[1]
        mixed_qkv = conv(mixed_qkv, weight, bias, **conv_kwargs)
    q, k, v = torch.split(mixed_qkv.transpose(1, 2), widths[:3], dim = -1)
    q = q.reshape(bsz, total, k_heads, dk)
    k = k.reshape(bsz, total, k_heads, dk)
    v = v.reshape(bsz, total, v_heads, dv)

    heads = slice(rank * v_heads, (rank + 1) * v_heads)
    beta = b.sigmoid()
    g = -module.A_log[heads].float().exp() * F.softplus(a.float() + module.dt_bias[heads])
    if v_heads // k_heads > 1:
        q = q.repeat_interleave(v_heads // k_heads, dim = 2)
        k = k.repeat_interleave(v_heads // k_heads, dim = 2)
    scan_kwargs = {}
    if boundaries is not None:
        scan_kwargs["cu_seqlens"] = boundaries[0]
    out, _ = scan(
        q,
        k,
        v,
        g = g,
        beta = beta,
        initial_state = None,
        output_final_state = False,
        use_qk_l2norm_in_kernel = True,
        **scan_kwargs,
    )
    out = _AllToAll.apply(out.reshape(bsz, total, v_heads, dv), 1, 2, group)

    out = module.norm(out.reshape(-1, dv), z.reshape(-1, dv))
    return module.out_proj(out.reshape(bsz, local_len, -1))


def wrap_gated_delta_forward(module, get_cp_manager) -> None:
    if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
        return
    forward = module.forward

    @functools.wraps(forward)
    def wrapped(*args, **kwargs):
        manager = get_cp_manager()
        if manager is None or manager.backend != "ulysses":
            return forward(*args, **kwargs)
        return ulysses_gated_delta_forward(module, *args, group = manager.group, **kwargs)

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped


def _ulysses_hf_attention(module, query, key, value, attention_mask, *, group, **kwargs):
    from .attention_dispatch import (
        AttentionConfig,
        AttentionContext,
        _run_attention,
        select_attention_backend,
    )
    from .packing import get_packed_info_from_kwargs

    bsz, n_heads, q_len, head_dim = query.shape
    scaling = kwargs.get("scaling")
    if scaling is not None and abs(scaling * head_dim**0.5 - 1) > 1e-6:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism needs the default attention scale 1/sqrt(head_dim)."
        )
    if module.training and kwargs.get("dropout"):
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support attention dropout."
        )
    seq_info = get_packed_info_from_kwargs(kwargs, query.device)
    n_kv_heads = key.shape[1]
    config = AttentionConfig(
        backend = select_attention_backend(use_varlen = seq_info is not None),
        n_kv_heads = n_kv_heads,
        n_groups = n_heads // n_kv_heads,
        flash_dense_kwargs = {"causal": True},
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True},
    )
    context = AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = key.shape[2],
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = query.requires_grad,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    return ulysses_attention(_run_attention, config, context, query, key, value, group), None


def install_hf_attention_wrapper(module, implementation, get_cp_manager) -> None:
    """Route this module's ALL_ATTENTION_FUNCTIONS[implementation] call through Ulysses while active."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # "eager" is a per-model function some transformers releases call without the registry.
    current = None if implementation == "eager" else ALL_ATTENTION_FUNCTIONS.get(implementation)
    if current is None:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism for hybrid models needs attn_implementation "
            f"'sdpa' or a flash attention backend, not {implementation!r}."
        )
    module._unsloth_cp_ulysses_attention = True
    if getattr(current, "_unsloth_cp_ulysses", False):
        return

    @functools.wraps(current)
    def attention(module, query, key, value, attention_mask, *args, **kwargs):
        manager = get_cp_manager()
        if (
            manager is None
            or manager.backend != "ulysses"
            or not getattr(module, "_unsloth_cp_ulysses_attention", False)
        ):
            return current(module, query, key, value, attention_mask, *args, **kwargs)
        if args:
            raise TypeError("Unsloth: Ulysses attention expects keyword arguments after the mask.")
        return _ulysses_hf_attention(
            module, query, key, value, attention_mask, group = manager.group, **kwargs
        )

    attention._unsloth_cp_ulysses = True
    ALL_ATTENTION_FUNCTIONS[implementation] = attention

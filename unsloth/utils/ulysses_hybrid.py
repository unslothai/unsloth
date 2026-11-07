# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Hugging Face attention (any model whose attention calls ALL_ATTENTION_FUNCTIONS
runs its own attention function after the exchange) and for hybrid models, whose scans need the full sequence so
they run on heads."""

import builtins
import functools
import inspect

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .ulysses import _AllToAll

__all__ = [
    "GATED_DELTA_ATTENTION_CLASSES",
    "is_ulysses_gated_delta",
    "check_gated_delta_heads",
    "ulysses_gated_delta_forward",
    "wrap_gated_delta_forward",
    "install_hf_attention_wrapper",
    "is_hf_registry_attention",
    "hf_attention_forward",
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
        if kwargs.get("packed_seq_lengths") is None:
            packed = getattr(manager, "packed_seq_lengths", None)
            if packed is not None:
                kwargs["packed_seq_lengths"] = packed
        return ulysses_gated_delta_forward(module, *args, group = manager.group, **kwargs)

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped


# Keyword arguments that describe the local sequence shard: rebuilt for the whole sequence after the exchange.
_SHARD_LOCAL_KWARGS = (
    "position_ids",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
    "packed_seq_lengths",
)


def _registry_caller(function, hops = 1):
    """`function` or the single function it delegates to (unsloth_zoo's compiled `<Class>_forward`) that reads
    ALL_ATTENTION_FUNCTIONS, else None. A forward calling several functions is not a delegate (fast forwards)."""
    try:
        function = inspect.unwrap(function)
    except Exception:
        return None
    code = getattr(function, "__code__", None)
    if code is None:
        return None
    if "ALL_ATTENTION_FUNCTIONS" in code.co_names:
        return function
    if hops <= 0:
        return None
    globs = getattr(function, "__globals__", {})
    callees = {
        name for name in code.co_names if callable(globs.get(name)) or name in vars(builtins)
    }
    if len(callees) != 1:
        return None
    callee = globs.get(callees.pop())
    if callee is None or callee is function or not inspect.isfunction(callee):
        return None
    return _registry_caller(callee, hops - 1)


def _calls_attention_registry(function) -> bool:
    return _registry_caller(function) is not None


def hf_attention_forward(module):
    """The Hugging Face forward of this attention class that calls ALL_ATTENTION_FUNCTIONS (eager falls back to
    the modeling module's eager_attention_forward): the class forward, or the original unsloth_zoo stashed when
    it patched the class (`_original_<module>_<Class>_forward`, e.g. Gemma3's fp32 SDPA forward). None for
    Unsloth's fast forwards and custom kernels."""
    cls = type(module)
    forward = getattr(cls, "forward", None)
    if _calls_attention_registry(forward):
        return forward
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if (
                name.startswith("_original_")
                and name.endswith(f"_{klass.__name__}_forward")
                and _calls_attention_registry(value)
            ):
                return value
    return None


def is_hf_registry_attention(module) -> bool:
    return hf_attention_forward(module) is not None


def _forward_globals(module) -> dict:
    return getattr(_registry_caller(hf_attention_forward(module)), "__globals__", {})


def _route_hf_forward(module, get_cp_manager) -> None:
    """When the class forward is unsloth_zoo's replacement, run the stashed Hugging Face forward on this module
    while a Ulysses manager is active (so attention goes through the registry), and the class forward otherwise."""
    hf_forward = hf_attention_forward(module)
    if hf_forward is None or hf_forward is getattr(type(module), "forward", None):
        return
    if "_unsloth_cp_hf_forward" in vars(module):
        return
    import types

    default = module.forward
    hf_bound = types.MethodType(hf_forward, module)

    @functools.wraps(default)
    def forward(*args, **kwargs):
        manager = get_cp_manager()
        if manager is not None and manager.backend == "ulysses":
            return hf_bound(*args, **kwargs)
        return default(*args, **kwargs)

    module._unsloth_cp_hf_forward = hf_bound
    module.forward = forward


class _HeadShard:
    """The attention module as its own attention function sees it on a head shard: the local key / value
    group count and the local heads' sinks; everything else is the module's."""

    # Plain attribute access only: torch.compile cannot trace object.__getattribute__ / object.__setattr__.
    def __init__(self, module, **overrides):
        self._unsloth_module = module
        # One attribute at a time: torch < 2.13 cannot trace self.__dict__.update (graph break).
        for name, value in overrides.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        # Only called for names the shard does not hold itself.
        if name == "_unsloth_module":
            raise AttributeError(name)
        return getattr(self._unsloth_module, name)


def _sliding_window(module, kwargs):
    window = (
        kwargs["sliding_window"]
        if "sliding_window" in kwargs
        else getattr(module, "sliding_window", None)
    )
    return (
        window if isinstance(window, int) and not isinstance(window, bool) and window > 0 else None
    )


def _refuse_non_causal(module, is_causal = None) -> None:
    # The rebuilt whole-sequence mask is causal, and bidirectional windows differ per model (Gemma3 `<`, the
    # generic overlay `<=`), so non-causal attention would silently train causal.
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    if is_causal is False:
        raise NotImplementedError(
            f"Unsloth: Ulysses context parallelism only supports causal attention, but {type(module).__name__} "
            "is bidirectional (for example use_bidirectional_attention = True). Turn off context parallelism "
            "for this model."
        )


def _whole_sequence_mask(total, device, dtype, window, cu_seqlens):
    """Additive [1, 1, T, T] mask over the gathered sequence: causal, the sliding window (key > query - window,
    as transformers' masks), and packed documents closed off from one another (padding past
    cu_seqlens[-1] is one more segment)."""
    idx = torch.arange(total, device = device)
    allowed = idx[None, :] <= idx[:, None]
    if window is not None:
        allowed = allowed & (idx[:, None] - idx[None, :] < window)
    if cu_seqlens is not None:
        doc = torch.bucketize(idx, cu_seqlens[1:].to(device = device, dtype = idx.dtype), right = True)
        allowed = allowed & (doc[:, None] == doc[None, :])
    mask = torch.zeros(total, total, device = device, dtype = dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def _is_flash(implementation) -> bool:
    # flash_attention_2 / 3, and hub kernels ("kernels-community/...") that take transformers' flash kwargs.
    return "flash" in str(implementation) or "/" in str(implementation)


def _ulysses_hf_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    group,
    attention_fn,
    implementation,
    packed_seq_lengths = None,
    **kwargs,
):
    """Ulysses around the model's own attention function: q / k / v go from sequence shards to head shards,
    the function runs over the whole sequence with its own scaling, softcap, sliding window and the local
    heads' sinks, and the output goes back to sequence shards."""
    from .packing import get_packed_info_from_kwargs
    from .ulysses import head_to_seq, seq_to_head, ulysses_head_plan

    if module.training and kwargs.get("dropout"):
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support attention dropout."
        )
    _refuse_non_causal(module, kwargs.get("is_causal"))
    bsz, n_heads, local_len, _ = query.shape
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    plan = ulysses_head_plan(n_heads, key.shape[1], size)
    if plan.kv_replication > 1:
        # Consecutive copies keep query head h on kv head h // n_groups after the split.
        key = key.repeat_interleave(plan.kv_replication, dim = 1)
        value = value.repeat_interleave(plan.kv_replication, dim = 1)
    query, key, value = (seq_to_head(x, group) for x in (query, key, value))
    total = local_len * size
    heads = slice(rank * plan.local_heads, (rank + 1) * plan.local_heads)

    for name in _SHARD_LOCAL_KWARGS:
        kwargs.pop(name, None)
    seq_info = (
        None
        if packed_seq_lengths is None
        else get_packed_info_from_kwargs({"packed_seq_lengths": packed_seq_lengths}, query.device)
    )
    cu_seqlens = None if seq_info is None else seq_info[1]

    overrides = {"num_key_value_groups": plan.local_groups}
    sinks = getattr(module, "sinks", None)
    if isinstance(sinks, torch.Tensor) and sinks.ndim >= 1 and sinks.shape[0] == n_heads:
        overrides["sinks"] = sinks[heads]
    s_aux = kwargs.get("s_aux")
    if isinstance(s_aux, torch.Tensor) and s_aux.ndim >= 1 and s_aux.shape[0] == n_heads:
        kwargs["s_aux"] = s_aux[heads]
    shard = _HeadShard(module, **overrides)

    window = _sliding_window(module, kwargs)
    if _is_flash(implementation):
        # Flash takes the window and the softcap as arguments, and documents as cu_seqlens.
        mask = None
        if cu_seqlens is not None:
            if int(cu_seqlens[-1]) < total:
                # Padding to the cp multiple sits after the last document: its own segment.
                cu_seqlens = torch.cat([cu_seqlens, cu_seqlens.new_tensor([total])])
            cu = cu_seqlens.to(device = query.device, dtype = torch.int32)
            longest = int((cu[1:] - cu[:-1]).max())
            kwargs.update(
                cu_seq_lens_q = cu, cu_seq_lens_k = cu, max_length_q = longest, max_length_k = longest
            )
    elif implementation == "sdpa" and cu_seqlens is None and window is None:
        mask = None  # sdpa's is_causal path
    else:
        # No data-dependent padding check here: it graph-broke compiled packed attention.
        mask = _whole_sequence_mask(total, query.device, query.dtype, window, cu_seqlens)
    out, _ = attention_fn(shard, query, key, value, mask, **kwargs)
    # [B, T, H / N, D] -> [B, T / N, H, D]
    out = out.reshape(bsz, total, plan.local_heads, -1)
    return head_to_seq(out, group), None


def _ulysses_attention_wrapper(current, implementation, get_cp_manager):
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
        if kwargs.get("packed_seq_lengths") is None:
            # Some decoders do not forward packed_seq_lengths to the attention layers.
            kwargs["packed_seq_lengths"] = getattr(manager, "packed_seq_lengths", None)
        return _ulysses_hf_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            group = manager.group,
            attention_fn = current,
            implementation = implementation,
            **kwargs,
        )

    attention._unsloth_cp_ulysses = True
    return attention


def install_hf_attention_wrapper(module, implementation, get_cp_manager) -> None:
    """Route this module's attention function (ALL_ATTENTION_FUNCTIONS[implementation], or the modeling
    module's eager_attention_forward for "eager") through Ulysses while a Ulysses manager is active."""
    _refuse_non_causal(module)
    _route_hf_forward(module, get_cp_manager)
    if implementation == "eager":
        globs = _forward_globals(module)
        current = globs.get("eager_attention_forward")
        if current is None:
            raise NotImplementedError(
                "Unsloth: Ulysses context parallelism found no eager_attention_forward for "
                f"{type(module).__name__}; use attn_implementation = 'sdpa'."
            )
        if not getattr(current, "_unsloth_cp_ulysses", False):
            globs["eager_attention_forward"] = _ulysses_attention_wrapper(
                current, implementation, get_cp_manager
            )
        module._unsloth_cp_ulysses_attention = True
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS.get(implementation)
    if current is None or implementation == "flex_attention":
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism needs attn_implementation 'sdpa', 'eager' or a flash "
            f"attention backend, not {implementation!r}."
        )
    module._unsloth_cp_ulysses_attention = True
    if getattr(current, "_unsloth_cp_ulysses", False):
        return
    ALL_ATTENTION_FUNCTIONS[implementation] = _ulysses_attention_wrapper(
        current, implementation, get_cp_manager
    )

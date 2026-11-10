# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Hugging Face attention (any model whose attention calls ALL_ATTENTION_FUNCTIONS
runs its own attention function after the exchange) and for hybrid models (Gated DeltaNet, Kimi delta and MiniMax
Lightning attention), whose scans need the full sequence so they run on heads."""

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
    "is_ulysses_kimi_delta",
    "is_ulysses_lightning",
    "is_ulysses_mixer",
    "check_gated_delta_heads",
    "check_ulysses_mixer_heads",
    "ulysses_gated_delta_forward",
    "ulysses_kimi_delta_forward",
    "ulysses_lightning_forward",
    "wrap_gated_delta_forward",
    "wrap_ulysses_mixer_forward",
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


def is_ulysses_kimi_delta(module) -> bool:
    if type(module).__name__ != "KimiDeltaAttention":
        return False
    needed = ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias")
    if not all(hasattr(module, n) for n in needed + ("b_proj", "o_norm", "o_proj")):
        return False
    return hasattr(module, "f_a_proj") or hasattr(module, "f_proj")


def is_ulysses_lightning(module) -> bool:
    if not type(module).__name__.endswith("LightningAttention"):
        return False
    needed = ("qkv_proj", "norm", "output_gate", "out_proj", "block_size", "num_attention_heads")
    buffers = ("slope_rate", "query_decay", "key_decay", "diagonal_decay")
    return all(hasattr(module, n) for n in needed + buffers)


def is_ulysses_mixer(module) -> bool:
    return (
        is_ulysses_gated_delta(module)
        or is_ulysses_kimi_delta(module)
        or is_ulysses_lightning(module)
    )


def check_gated_delta_heads(module, cp_size) -> None:
    for name in ("num_k_heads", "num_v_heads"):
        heads = getattr(module, name)
        if heads % cp_size != 0:
            raise ValueError(
                f"Unsloth: Ulysses context parallelism needs linear attention {name} ({heads}) "
                f"to be divisible by context_parallel_size ({cp_size})."
            )


def check_ulysses_mixer_heads(module, cp_size) -> None:
    if is_ulysses_kimi_delta(module):
        name, heads = "Kimi delta attention num_heads", module.num_heads
    elif is_ulysses_lightning(module):
        name, heads = "Lightning attention num_attention_heads", module.num_attention_heads
    else:
        return check_gated_delta_heads(module, cp_size)
    if heads % cp_size != 0:
        raise ValueError(
            f"Unsloth: Ulysses context parallelism needs {name} ({heads}) to be divisible by "
            f"context_parallel_size ({cp_size})."
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


@functools.lru_cache(maxsize = 1)
def _fla_causal_conv1d():
    # fla moved the functional conv between releases.
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
        return causal_conv1d
    except ImportError:
        pass
    try:
        from fla.modules.convolution import causal_conv1d
        return causal_conv1d
    except ImportError:
        return None


def _kda_short_conv(conv, x, rows):
    """Causal depthwise conv (+ SiLU) of a fla ShortConvolution on this rank's channels. x [B, T, C]."""
    weight = conv.weight.squeeze(1).index_select(0, rows)
    bias = None if conv.bias is None else conv.bias.index_select(0, rows)
    activation = getattr(conv, "activation", "silu")
    causal_conv1d = _fla_causal_conv1d() if x.is_cuda else None
    if causal_conv1d is not None:
        try:
            out = causal_conv1d(
                x = x.contiguous(),
                weight = weight.contiguous(),
                bias = bias,
                initial_state = None,
                output_final_state = False,
                activation = activation,
            )
            return out[0] if isinstance(out, tuple) else out
        except TypeError:
            pass  # another fla signature: the torch conv below is exact, only slower
    out = F.conv1d(
        x.transpose(1, 2).to(weight.dtype),
        weight.unsqueeze(1),
        bias,
        padding = weight.shape[-1] - 1,
        groups = weight.shape[0],
    )[..., : x.shape[1]].transpose(1, 2)
    return F.silu(out) if activation in ("silu", "swish") else out


def _kda_gate(f, A_log, dt_bias, lower_bound):
    """KDA per-channel log decay [B, T, H, K] in fp32 (fla naive_kda_gate / naive_kda_lowerbound_gate)."""
    heads, dim = f.shape[-2:]
    a = A_log.reshape(1, 1, heads, 1).float().exp()
    g = f.float() + dt_bias.reshape(1, 1, heads, dim).float()
    if lower_bound is not None:
        # Kimi K3 bounds the decay: lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).
        return lower_bound * torch.sigmoid(a * g)
    return -a * F.softplus(g)


def _torch_kda(q, k, v, g, beta):
    """Exact token-by-token KDA delta rule (CPU reference).

    q, k, g [B, T, H, K]; v [B, T, H, V]; beta [B, T, H].
    """
    # fla's use_qk_l2norm_in_kernel: x * rsqrt(sum(x ** 2) + 1e-6), not F.normalize.
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim = True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim = True) + 1e-6)
    v, g, beta = v.float(), g.float(), beta.float()
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp().unsqueeze(-1)
        delta = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outs, dim = 1)


def _kda_scan(q, k, v, g, beta):
    if q.is_cuda:
        try:
            from fla.ops.kda import chunk_kda
        except ImportError:
            chunk_kda = None
        if chunk_kda is not None:
            # g is the finished log decay, so the kernel's own gate is left off.
            out, _ = chunk_kda(
                q = q.contiguous(),
                k = k.contiguous(),
                v = v.contiguous(),
                g = g.contiguous(),
                beta = beta.contiguous(),
                initial_state = None,
                output_final_state = False,
                use_qk_l2norm_in_kernel = True,
            )
            return out
    return _torch_kda(q, k, v, g, beta).to(v.dtype)


def _caller_pad_lengths(manager, bsz, total):
    lookup = getattr(manager, "caller_pad_lengths", None)
    return lookup(bsz, total) if callable(lookup) else None


def _real_positions(lengths, positions):
    # [bsz, len(positions)] True where the position is one of the row's caller tokens.
    limit = torch.tensor(lengths, device = positions.device)
    return positions[None, :] < limit[:, None]


def ulysses_kimi_delta_forward(
    module,
    hidden_states,
    attention_mask = None,
    cache_params = None,
    *,
    group,
    caller_lengths = None,
    **kwargs,
):
    """Training forward of a Kimi-Linear / Kimi K3 KimiDeltaAttention on a contiguous sequence shard.
    caller_lengths (per row, the real length of the whole sequence): the output is zero past it, as the
    hub layer's pad_input leaves caller padding when it unpads on the 2D mask."""
    if cache_params is not None:
        raise RuntimeError("Unsloth: Ulysses context parallelism does not support a cache.")
    if kwargs.get("packed_seq_lengths") is not None or kwargs.get("cu_seqlens") is not None:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support packing for Kimi delta "
            "attention."
        )
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        mask = attention_mask[:, -hidden_states.shape[1] :, None]
        hidden_states = (hidden_states * mask).to(hidden_states.dtype)

    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    check_ulysses_mixer_heads(module, size)
    bsz, local_len, _ = hidden_states.shape
    total = local_len * size
    heads, dim = module.num_heads // size, module.head_dim

    q = module.q_proj(hidden_states)
    k = module.k_proj(hidden_states)
    v = module.v_proj(hidden_states)
    if hasattr(module, "f_a_proj"):
        f = module.f_b_proj(module.f_a_proj(hidden_states))
    else:
        f = module.f_proj(hidden_states)
    b = module.b_proj(hidden_states)
    # Kimi-Linear projects the output gate through a low rank; K3 (use_full_rank_gate) does not.
    if hasattr(module, "g_a_proj"):
        gate = module.g_b_proj(module.g_a_proj(hidden_states))
    else:
        gate = module.g_proj(hidden_states)

    pieces = [x.reshape(bsz, local_len, size, -1) for x in (q, k, v, f, b)]
    widths = [p.shape[-1] for p in pieces]
    mixed = _AllToAll.apply(torch.cat(pieces, dim = -1), 2, 1, group).squeeze(2)
    q, k, v, f, b = torch.split(mixed, widths, dim = -1)

    rows = torch.arange(rank * heads * dim, (rank + 1) * heads * dim, device = q.device)
    q = _kda_short_conv(module.q_conv1d, q, rows).reshape(bsz, total, heads, dim)
    k = _kda_short_conv(module.k_conv1d, k, rows).reshape(bsz, total, heads, dim)
    v = _kda_short_conv(module.v_conv1d, v, rows).reshape(bsz, total, heads, dim)

    g = _kda_gate(
        f.reshape(bsz, total, heads, dim),
        module.A_log.reshape(-1)[rank * heads : (rank + 1) * heads],
        module.dt_bias.reshape(-1).index_select(0, rows),
        getattr(module, "gate_lower_bound", None),
    )
    out = _kda_scan(q, k, v, g, b.float().sigmoid())
    out = _AllToAll.apply(out.reshape(bsz, total, heads, dim), 1, 2, group)

    out = module.o_norm(out, gate.reshape(bsz, local_len, -1, dim))
    out = module.o_proj(out.reshape(bsz, local_len, -1))
    if caller_lengths is not None:
        local = torch.arange(rank * local_len, (rank + 1) * local_len, device = out.device)
        out = out.masked_fill(~_real_positions(caller_lengths, local)[..., None], 0)
    return out


def ulysses_lightning_forward(
    module,
    hidden_states,
    position_embeddings = None,
    attention_mask = None,
    past_key_values = None,
    *,
    group,
    **kwargs,
):
    """Training forward of a MiniMax LightningAttention on a contiguous sequence shard."""
    if past_key_values is not None:
        raise RuntimeError("Unsloth: Ulysses context parallelism does not support a cache.")
    if kwargs.get("packed_seq_lengths") is not None:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support packing for Lightning attention."
        )
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    check_ulysses_mixer_heads(module, size)
    bsz, local_len, _ = hidden_states.shape
    heads, dim = module.num_attention_heads // size, module.head_dim

    qkv = module.act_fn(module.qkv_proj(hidden_states))
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        # The stock forward zeroes padded values; zeroing q / k too changes nothing downstream.
        qkv = qkv * attention_mask[:, -local_len:, None].to(qkv.dtype)
    # [B, T/N, H, 3D] -> [B, T, H/N, 3D]: one exchange for q, k and v of each head.
    qkv = _AllToAll.apply(
        qkv.reshape(bsz, local_len, module.num_attention_heads, 3 * dim), 2, 1, group
    )
    q, k, v = (x.transpose(1, 2) for x in torch.split(qkv, dim, dim = -1))

    head = slice(rank * heads, (rank + 1) * heads)
    dtype = v.dtype
    slope = module.slope_rate[head].to(dtype)
    query_decay = module.query_decay[head].to(dtype)
    key_decay = module.key_decay[head].to(dtype)
    diagonal_decay = module.diagonal_decay[:, head].to(dtype)
    block, total = module.block_size, q.shape[2]

    state = v.new_zeros(bsz, heads, dim, dim)
    outs = []
    for start in range(0, total, block):
        end = min(start + block, total)
        n = end - start
        qb, kb, vb = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]
        intra = torch.matmul(
            torch.matmul(qb, kb.transpose(-1, -2)) * diagonal_decay[:, :, :n, :n], vb
        )
        inter = torch.matmul(qb * query_decay[:, :n], state)
        outs.append(inter + intra)
        state = state * torch.exp(-slope * n) + torch.matmul(
            (kb * key_decay[:, -n:]).transpose(-1, -2), vb
        )
    out = _AllToAll.apply(torch.cat(outs, dim = 2).transpose(1, 2), 1, 2, group)

    out = module.norm(out.reshape(bsz, local_len, -1))
    out = torch.sigmoid(module.output_gate(hidden_states)) * out
    return module.out_proj(out), None


def _wrap_ulysses_forward(module, get_cp_manager, name) -> None:
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
        if name == "ulysses_kimi_delta_forward":
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if isinstance(hidden, torch.Tensor) and hidden.ndim == 3:
                kwargs["caller_lengths"] = _caller_pad_lengths(
                    manager, hidden.shape[0], hidden.shape[1] * manager.size
                )
        # Looked up per call so tests can substitute the exchange.
        return globals()[name](module, *args, group = manager.group, **kwargs)

    module._unsloth_cp_orig_forward = forward
    module.forward = wrapped


def wrap_gated_delta_forward(module, get_cp_manager) -> None:
    _wrap_ulysses_forward(module, get_cp_manager, "ulysses_gated_delta_forward")


def wrap_ulysses_mixer_forward(module, get_cp_manager) -> None:
    if is_ulysses_kimi_delta(module):
        return _wrap_ulysses_forward(module, get_cp_manager, "ulysses_kimi_delta_forward")
    if is_ulysses_lightning(module):
        return _wrap_ulysses_forward(module, get_cp_manager, "ulysses_lightning_forward")
    return wrap_gated_delta_forward(module, get_cp_manager)


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
    caller_lengths = None,
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
    if caller_lengths is not None and cu_seqlens is None:
        out = _caller_padding_rows(
            out,
            shard,
            query,
            key,
            value,
            caller_lengths,
            window,
            implementation,
            attention_fn,
            kwargs,
        )
    return head_to_seq(out, group), None


def _caller_padding_rows(
    out, shard, query, key, value, lengths, window, implementation, attention_fn, kwargs
):
    """The rows of right padded positions as one process computes them from the caller's 2D mask, so
    the MoE routes them alike (real rows never attend padding, so they are unchanged). Flash unpads:
    padding rows are zero. sdpa / eager mask the padded keys: a padding query attends the row's real
    keys only. Left as is: a row without real tokens, and sdpa / eager with a sliding window (a
    padding query past the window has no key at all, which softmax turns into NaN)."""
    total = out.shape[1]
    if _is_flash(implementation):
        real = _real_positions(lengths, torch.arange(total, device = out.device))
        return out.masked_fill(~real[:, :, None, None], 0)
    if window is not None:
        return out
    rows = []
    for row, length in enumerate(lengths):
        if length <= 0 or length >= total:
            rows.append(out[row])
            continue
        # Every real key precedes the padding queries: no mask and is_causal = False, so no
        # [pad, real] mask is materialised (sdpa and eager both honour the explicit flag).
        pad, _ = attention_fn(
            shard,
            query[row : row + 1, :, length:],
            key[row : row + 1, :, :length],
            value[row : row + 1, :, :length],
            None,
            **{**kwargs, "is_causal": False},
        )
        pad = pad.reshape(total - length, out.shape[2], -1)
        rows.append(torch.cat([out[row, :length], pad.to(out.dtype)], dim = 0))
    return torch.stack(rows, dim = 0)


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
            caller_lengths = _caller_pad_lengths(
                manager, query.shape[0], query.shape[2] * manager.size
            ),
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

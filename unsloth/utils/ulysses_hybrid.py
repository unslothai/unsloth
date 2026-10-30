# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for hybrid models (Gated DeltaNet, Kimi delta and MiniMax Lightning
attention): scans need the full sequence, so they run on heads."""

import functools
import inspect

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .ulysses import _AllToAll, ulysses_attention

__all__ = [
    "GATED_DELTA_ATTENTION_CLASSES",
    "KIMI_DELTA_ATTENTION_CLASSES",
    "HYBRID_ATTENTION_CLASSES",
    "ULYSSES_ATTENTION_CLASSES",
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
]

# Hybrid full attention layers: Hugging Face forwards that call ALL_ATTENTION_FUNCTIONS after RoPE.
GATED_DELTA_ATTENTION_CLASSES = (
    "Qwen3NextAttention",
    "Qwen3_5Attention",
    "Qwen3_5MoeAttention",
)
# Kimi-Linear's full attention layers are MLA; value heads are narrower than query / key heads.
KIMI_DELTA_ATTENTION_CLASSES = ("KimiMLAAttention",)
HYBRID_ATTENTION_CLASSES = (
    GATED_DELTA_ATTENTION_CLASSES + KIMI_DELTA_ATTENTION_CLASSES + ("MiniMaxAttention",)
)
# Attention-only models whose Hugging Face attention runs under Ulysses without a stateful mixer.
# DeepSeek V2 / V3 MLA (Moonlight) pass a YaRN-scaled softmax scale and narrower value heads.
# Gemma 4 passes its sliding window per layer and a softmax scale of 1.0.
ULYSSES_ATTENTION_CLASSES = HYBRID_ATTENTION_CLASSES + (
    "MiniMaxM2Attention",
    "DeepseekV2Attention",
    "DeepseekV3Attention",
    "Gemma4TextAttention",
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
    q = F.normalize(q.float(), dim = -1) * q.shape[-1] ** -0.5
    k = F.normalize(k.float(), dim = -1)
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


def ulysses_kimi_delta_forward(
    module,
    hidden_states,
    attention_mask = None,
    cache_params = None,
    *,
    group,
    **kwargs,
):
    """Training forward of a Kimi-Linear / Kimi K3 KimiDeltaAttention on a contiguous sequence shard."""
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
    return module.o_proj(out.reshape(bsz, local_len, -1))


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


_SLIDING_BLOCK_MASKS = {}
_FLEX = {}
# Flex outcome per call class: True once it ran (with its backward when training), False once it
# failed. A failure on one class (dtype, head size, block alignment) keeps flex for the others.
_FLEX_STATUS = {}
_FLEX_BLOCK = 128


def _flex_sliding_attention(*, config, context, Q, K, V, window, scale):
    """Causal sliding-window attention over the gathered sequence with a block-sparse flex mask.

    SDPA has no window, so the dense path would build a [S, S] mask and compute every block;
    flex skips the blocks outside the window like the stock (non-CP) Gemma 4 path does.
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if "fn" not in _FLEX:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic = False)
    seq = context.q_len
    cache_key = (seq, window, Q.device)
    block_mask = _SLIDING_BLOCK_MASKS.get(cache_key)
    if block_mask is None:

        def keep(b, h, q, k):
            return (q >= k) & (q - k < window)

        if len(_SLIDING_BLOCK_MASKS) >= 8:
            _SLIDING_BLOCK_MASKS.clear()
        block_mask = create_block_mask(keep, None, None, seq, seq, device = Q.device)
        _SLIDING_BLOCK_MASKS[cache_key] = block_mask
    out = _FLEX["fn"](Q, K, V, block_mask = block_mask, scale = scale, enable_gqa = config.n_groups != 1)
    return out.transpose(1, 2)


def _flex_device_ok(query) -> bool:
    # flex_attention has no CPU backward and the dense path is the CPU reference.
    return query.is_cuda


def _flex_or_dense(*, config, context, Q, K, V, window, scale):
    """Flex sliding attention, or the dense windowed SDPA path for a call class flex cannot run."""
    from .attention_dispatch import _run_attention

    needs_backward = bool(context.training) and not torch.is_inference_mode_enabled()
    key = (
        Q.device.type,
        Q.dtype,
        Q.shape[-1],
        config.n_groups != 1,
        window,
        context.q_len % _FLEX_BLOCK == 0,
        needs_backward,
    )
    if _FLEX_STATUS.get(key) is not False:
        flex = functools.partial(
            _flex_sliding_attention, config = config, context = context, window = window, scale = scale
        )
        try:
            if key not in _FLEX_STATUS and needs_backward:
                # The backward compiles on the first loss.backward(), past any fallback here (head
                # sizes whose backward exceeds shared memory): run it once per class now. Also
                # under reentrant checkpointing, whose first forward runs without grad.
                q, k, v = (t.detach().requires_grad_() for t in (Q, K, V))
                with torch.enable_grad():
                    torch.autograd.grad(flex(Q = q, K = k, V = v).sum(), (q, k, v))
                del q, k, v
            out = flex(Q = Q, K = K, V = V)
            _FLEX_STATUS[key] = True
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            # A compile failure (old torch, unsupported head size or dtype) keeps the dense path.
            _FLEX_STATUS[key] = False
            print(f"Unsloth: Ulysses flex sliding attention unavailable ({error}); using SDPA.")
    return _run_attention(config = config, context = context, Q = Q, K = K, V = V)


def _ulysses_hf_attention(module, query, key, value, attention_mask, *, group, **kwargs):
    from .attention_dispatch import (
        SDPA,
        AttentionConfig,
        AttentionContext,
        _run_attention,
        select_attention_backend,
    )
    from .packing import get_packed_info_from_kwargs

    bsz, n_heads, q_len, head_dim = query.shape
    scaling = kwargs.get("scaling")
    # DeepSeek MLA folds YaRN's mscale into the scale; every backend below takes an explicit one.
    custom_scale = scaling is not None and abs(scaling * head_dim**0.5 - 1) > 1e-6
    scale_kwargs = {"softmax_scale": scaling} if custom_scale else {}
    if module.training and kwargs.get("dropout"):
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism does not support attention dropout."
        )
    seq_info = get_packed_info_from_kwargs(kwargs, query.device)
    v_dim = value.shape[-1]
    if v_dim > head_dim:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism needs value heads no wider than query heads."
        )
    if v_dim < head_dim:
        # MLA (Kimi-Linear, DeepSeek): pad value heads to the query width like HF's flash path.
        value = F.pad(value, (0, head_dim - v_dim))
    n_kv_heads = key.shape[1]
    config = AttentionConfig(
        backend = select_attention_backend(use_varlen = seq_info is not None),
        n_kv_heads = n_kv_heads,
        n_groups = n_heads // n_kv_heads,
        flash_dense_kwargs = {"causal": True, **scale_kwargs},
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True, **scale_kwargs},
        sdpa_kwargs = {"scale": scaling} if custom_scale else None,
        xformers_kwargs = {"scale": scaling} if custom_scale else None,
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
        # After the exchange each rank holds the full sequence, so the window applies unchanged.
        sliding_window = kwargs.get("sliding_window"),
        # Keeps training SDPA off cuDNN attention on SM100 (#15019) on the Ulysses path too.
        training = module.training,
    )
    window = context.sliding_window
    run = _run_attention
    if (
        config.backend == SDPA
        and window is not None
        and 0 < window < q_len * dist.get_world_size(group)
        and seq_info is None
        and _flex_device_ok(query)
    ):
        run = functools.partial(_flex_or_dense, window = window, scale = scaling)

    out = ulysses_attention(run, config, context, query, key, value, group)
    return (out if v_dim == head_dim else out[..., :v_dim]), None


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

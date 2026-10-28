# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed rows use varlen flash; heads past 256 and shapes FA4/FA3 refuses use SDPA mem-efficient;
anything else goes to the wrapped sdpa."""

import contextlib
import inspect
import logging
import os
import weakref

import torch

__all__ = ["enable_hf_packed_attention", "HF_PACKED_ATTENTION_STATS"]

_FLAG = "_unsloth_hf_packed_varlen"
_ENV = "UNSLOTH_HF_PACKED_VARLEN"
_ORIG_SDPA = [None]
# True inside _block_compile's own compiled blocks (not a user's outer compile); see _leaves_block.
IN_COMPILED_BLOCK = [False]
# efficient: the subset of fast calls that ran SDPA's memory-efficient kernel over cu_seqlens;
# int32_split: fast calls run in document groups to keep flash's varlen backward under int32.
HF_PACKED_ATTENTION_STATS = {
    "fast": 0,
    "fallback": 0,
    "mask_checks": 0,
    "efficient": 0,
    "compiled": 0,
    "int32_split": 0,
}
BLOCK_VARLEN_KWARG = "_unsloth_block_varlen"
# BLOCK_VARLEN_KWARG value for rows the eager wrapper must split under the int32 limit.
_LEAVE_BLOCK = "leave"
# Layers of one window share a mask object: window -> [weakref, (cu_seqlens, versions), verdict].
_MASK_CHECK = {}
_MASK_CHECK_ROWS = 1024


def _disabled() -> bool:
    return os.environ.get(_ENV, "1").lower() in ("0", "false", "no", "off")


def _mask_is_block_causal(
    attention_mask,
    cu_seqlens,
    total,
    window = None,
) -> bool:
    """Exactly causal (k > q - window when windowed) inside each packed segment and closed across
    them; other windows, bidirectional spans, pad tails and None keep the wrapped path."""
    if not (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.dtype == torch.bool
        and attention_mask.dim() == 4
        and attention_mask.shape[0] == 1
        and attention_mask.shape[1] == 1
        and attention_mask.shape[-2:] == (total, total)
    ):
        return False
    # Versions catch in-place edits; inference tensors have none, so are always rechecked.
    cacheable = not (torch.is_inference(attention_mask) or torch.is_inference(cu_seqlens))
    key = (attention_mask._version, cu_seqlens._version, total) if cacheable else None
    ref, checked, verdict = _MASK_CHECK.get(window, (None, None, None))
    if (
        cacheable
        and ref is not None
        and ref() is attention_mask
        and checked[0] is cu_seqlens
        and checked[1] == key
    ):
        return verdict
    HF_PACKED_ATTENTION_STATS["mask_checks"] += 1
    verdict = False
    if int(cu_seqlens[-1].item()) == total:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        # output_size: no host sync to size the result.
        segment = torch.repeat_interleave(
            torch.arange(lengths.numel(), device = attention_mask.device),
            lengths.to(attention_mask.device),
            output_size = total,
        )
        position = torch.arange(total, device = attention_mask.device)
        # Row blocks avoid a second T x T mask; one host sync for the whole mask.
        mismatch = torch.zeros((), dtype = torch.bool, device = attention_mask.device)
        for start in range(0, total, _MASK_CHECK_ROWS):
            rows = slice(start, min(start + _MASK_CHECK_ROWS, total))
            expected = (segment[rows, None] == segment[None, :]) & (
                position[None, :] <= position[rows, None]
            )
            if window is not None:
                # transformers' sliding_window_overlay: kv_idx > q_idx - sliding_window.
                expected &= position[None, :] > position[rows, None] - window
            mismatch |= (attention_mask[0, 0, rows] != expected).any()
        verdict = not bool(mismatch)
    if cacheable:
        _MASK_CHECK[window] = [weakref.ref(attention_mask), (cu_seqlens, key), verdict]
    return verdict


def _autocast_dtype(device_type):
    try:
        enabled = torch.is_autocast_enabled(device_type)
        return torch.get_autocast_dtype(device_type) if enabled else None
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None


def _autocast_dtype_differs(query) -> bool:
    dtype = _autocast_dtype(query.device.type)
    return dtype is not None and dtype != query.dtype


def _autocast_float32(query, key, value):
    """Cast fp32 Q / K / V to the autocast dtype as SDPA would (fp32 q_norm / k_norm in full finetuning)."""
    if not (query.dtype == key.dtype == value.dtype == torch.float32):
        return query, key, value
    dtype = _autocast_dtype(query.device.type)
    if dtype not in (torch.float16, torch.bfloat16):
        return query, key, value
    return query.to(dtype), key.to(dtype), value.to(dtype)


_NO_WINDOW = object()


def _sliding_window(module, kwargs):
    """None, a positive int window, or _NO_WINDOW when the window is not one varlen can take."""
    window = kwargs.get("sliding_window")
    if window is None:
        window = getattr(module, "sliding_window", None)
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return _NO_WINDOW
    return window


def _softcapped(module, kwargs) -> bool:
    if kwargs.get("softcap") or getattr(module, "sinks", None) is not None:
        return True
    if getattr(module, "attn_logit_softcapping", None):
        return True
    config = getattr(module, "config", None)
    return bool(getattr(config, "attn_logit_softcapping", None))


_TORCH_VARLEN = []
_TORCH_VARLEN_GQA = []
_TORCH_VARLEN_DEVICES = {}


def _torch_varlen(device):
    """torch's own flash varlen (a third of xformers' host cost per call), else None."""
    if not _TORCH_VARLEN:
        fn, native_gqa = None, False
        try:
            # attention_dispatch's kill switch for torch's varlen_attn covers this path too.
            if os.environ.get("UNSLOTH_TORCH_VARLEN", "1").lower() in ("0", "false", "no", "off"):
                raise ImportError
            from torch.nn.attention.varlen import varlen_attn

            # 2.11 added scale / window_size, 2.12 enable_gqa; 2.10 only had is_causal and is not used.
            parameters = inspect.signature(varlen_attn).parameters.keys()
            if {"scale", "window_size"} <= parameters:
                fn, native_gqa = varlen_attn, "enable_gqa" in parameters
        except Exception:
            fn = None
        _TORCH_VARLEN.append(fn)
        _TORCH_VARLEN_GQA[:] = [native_gqa]
    fn = _TORCH_VARLEN[0]
    if fn is None or torch.version.hip is not None:
        return None
    ok = _TORCH_VARLEN_DEVICES.get(device.index)
    if ok is None:
        ok = torch.cuda.get_device_capability(device)[0] >= 8
        _TORCH_VARLEN_DEVICES[device.index] = ok
    return fn if ok else None


_EFFICIENT_ENV = "UNSLOTH_HF_PACKED_EFFICIENT"
# Above flash's 256 cap (Gemma-4 global layers are 512); wider heads are untested.
_EFFICIENT_MAX_HEAD_DIM = 512
_EFFICIENT_VARLEN = []
_ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", None)  # torch 2.8+
_EFFICIENT_DEVICES = {}


def _efficient_varlen_takes(query, window) -> bool:
    """SDPA's memory-efficient kernel over cu_seqlens: no T x T bias. Off under any policy asking for SDPA."""
    if os.environ.get(_EFFICIENT_ENV, "1").lower() in ("0", "false", "no", "off"):
        return False
    if not _EFFICIENT_VARLEN:
        ok = True
        try:
            schema = torch.ops.aten._efficient_attention_forward.default._schema
            names = {argument.name for argument in schema.arguments}
            ok = ok and {"cu_seqlens_q", "scale", "window_size"} <= names
            schema = torch.ops.aten._efficient_attention_backward.default._schema
            ok = ok and "window_size" in {argument.name for argument in schema.arguments}
        except Exception:
            ok = False
        _EFFICIENT_VARLEN.append(ok)
    head_dim = query.shape[-1]
    if (
        not _EFFICIENT_VARLEN[0]
        or torch.version.hip is not None
        or head_dim > _EFFICIENT_MAX_HEAD_DIM
        or head_dim % 8 != 0
        # Its backward is nondeterministic.
        or torch.are_deterministic_algorithms_enabled()
        or not torch.backends.cuda.mem_efficient_sdp_enabled()
    ):
        return False
    ok = _EFFICIENT_DEVICES.get(query.device.index)
    if ok is None:
        ok = torch.cuda.get_device_capability(query.device)[0] >= 8
        _EFFICIENT_DEVICES[query.device.index] = ok
    if not ok:
        return False
    from torch.nn.attention import SDPBackend
    from .attention_dispatch import _sdpa_backend_state

    # A math-first priority (or a disabled efficient backend) for these shapes keeps SDPA.
    n_heads = query.shape[1]
    _, choice = _sdpa_backend_state(
        str(query.device), query.dtype, head_dim, n_heads, n_heads, True, False
    )
    return choice not in (int(SDPBackend.MATH), -1)


class _EfficientVarlenAttention(torch.autograd.Function):
    """(1, T, H, D) Q / K / V over cu_seqlens, causal (and windowed) inside each segment."""

    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens, max_seqlen, scale, window):
        out, logsumexp, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            query,
            key,
            value,
            None,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            0.0,
            1,  # causal from the top left of each segment
            any(ctx.needs_input_grad[:3]),
            scale = scale,
            window_size = window,
        )
        ctx.save_for_backward(query, key, value, out, logsumexp, cu_seqlens, seed, offset)
        ctx.max_seqlen, ctx.scale, ctx.window = max_seqlen, scale, window
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, out, logsumexp, cu_seqlens, seed, offset = ctx.saved_tensors
        grad_q, grad_k, grad_v, _ = torch.ops.aten._efficient_attention_backward(
            grad_out.contiguous(),
            query,
            key,
            value,
            None,
            out,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            logsumexp,
            0.0,
            seed,
            offset,
            1,
            False,
            scale = ctx.scale,
            window_size = ctx.window,
        )
        return grad_q, grad_k, grad_v, None, None, None, None


def _efficient_varlen(query, key, value, cu_seqlens, max_seqlen, scale, window):
    """Q (1, H, T, D), K / V (1, H_kv, T, D) -> (1, T, H, D)."""
    _, n_heads, total, head_dim = query.shape
    n_kv_heads = key.shape[1]

    def heads_last(x):
        x = x[0].transpose(0, 1).unsqueeze(0)
        # The kernel needs a unit stride on the head dim; a refusal would turn the route off for good.
        return x if x.stride(-1) == 1 else x.contiguous()

    q, k, v = heads_last(query), heads_last(key), heads_last(value)
    if n_kv_heads != n_heads:
        # The kernel has no GQA: repeat K / V heads as transformers' sdpa path does (autograd sums back).
        groups = n_heads // n_kv_heads
        k = (
            k.unsqueeze(3)
            .expand(1, total, n_kv_heads, groups, head_dim)
            .reshape(1, total, n_heads, head_dim)
        )
        v = (
            v.unsqueeze(3)
            .expand(1, total, n_kv_heads, groups, head_dim)
            .reshape(1, total, n_heads, head_dim)
        )
    return _EfficientVarlenAttention.apply(
        q, k, v, cu_seqlens.to(torch.int32), int(max_seqlen), float(scale), window
    )


_SDPA_MATH = []


def _sdpa_requested(query, key) -> bool:
    """Deterministic mode, a disabled flash SDPA backend or a math-first sdpa_kernel priority asks for
    SDPA itself, so no varlen kernel (torch's, xformers or flash-attn) may replace it."""
    return _sdpa_policy_requested(
        query.device, query.dtype, query.shape[-1], query.shape[1], key.shape[1]
    )


def _sdpa_policy_requested(device, dtype, head_dim, n_heads, n_kv_heads) -> bool:
    """_sdpa_requested from shapes; reads process state, so never while tracing."""
    if torch.are_deterministic_algorithms_enabled():
        return True
    if not _SDPA_MATH:
        from torch.nn.attention import SDPBackend
        from .attention_dispatch import _sdpa_backend_state
        _SDPA_MATH[:] = [int(SDPBackend.MATH), _sdpa_backend_state]
    math, backend_state = _SDPA_MATH
    flash, choice = backend_state(
        str(device),
        dtype,
        head_dim,
        n_heads,
        n_kv_heads,
        True,
        n_heads != n_kv_heads,
    )
    return not flash or choice == math


def _static_window(module, query, key, value, dropout, is_causal, kwargs):
    """The window for a call a varlen kernel can take (shapes, dtypes, masking), else _NO_WINDOW.
    Only shapes, dtypes and attributes: also safe while tracing."""
    bsz, n_heads, q_len, head_dim = query.shape
    window = _sliding_window(module, kwargs)
    if (
        bsz == 1
        and q_len > 0
        and query.is_cuda
        and query.device == key.device == value.device
        and key.shape[0] == value.shape[0] == 1
        and key.shape[1] == value.shape[1]
        and key.shape[2] == value.shape[2] == q_len
        # Not MLA (192 / 128).
        and key.shape[-1] == value.shape[-1] == head_dim
        and head_dim % 8 == 0
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dtype == key.dtype == value.dtype
        and not _autocast_dtype_differs(query)
        and not dropout
        and is_causal is not False
        and getattr(module, "is_causal", True)
        and kwargs.get("position_bias") is None
        and kwargs.get("cache") is None
        and kwargs.get("past_key_values") is None
        and n_heads % key.shape[1] == 0
        and not _softcapped(module, kwargs)
    ):
        return window
    return _NO_WINDOW


def _call_torch_varlen(torch_varlen, query, key, value, cu_seqlens, max_seqlen, scale, window):
    n_heads, n_kv_heads = query.shape[1], key.shape[1]
    gqa, k_in, v_in = {}, key, value
    if n_kv_heads != n_heads:
        if not _TORCH_VARLEN_GQA or _TORCH_VARLEN_GQA[0]:
            gqa = {"enable_gqa": True}
        else:
            # torch 2.11: no enable_gqa, so repeat the K/V heads as transformers' sdpa path does.
            k_in = key.repeat_interleave(n_heads // n_kv_heads, dim = 1)
            v_in = value.repeat_interleave(n_heads // n_kv_heads, dim = 1)
    out = torch_varlen(
        query[0].transpose(0, 1),
        k_in[0].transpose(0, 1),
        v_in[0].transpose(0, 1),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        scale = scale,
        window_size = (-1, 0) if window is None else (window - 1, 0),
        **gqa,
    )
    return out.unsqueeze(0)


def _flash_varlen(query, key, value, cu_seqlens, max_seqlen, scale, window):
    """aten's flash varlen, what torch's varlen_attn runs for these rows, called directly: a native op
    with a C++ backward (varlen_attn's Python custom op cost ~37 us of host time per call in a
    compiled step) that takes grouped K / V heads as they are."""
    out = torch.ops.aten._flash_attention_forward(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        0.0,
        False,
        False,
        scale = scale,
        # Causal through the window: right 0, left unbounded or window - 1 (transformers' overlay).
        window_size_left = -1 if window is None else window - 1,
        window_size_right = 0,
    )[0]
    return out.unsqueeze(0)


def _int32_safe_groups(lengths, n_heads, head_dim):
    """Consecutive documents in groups whose varlen backward stays under the int32 limit, as
    (first, end, start token, stop token, longest); None when one document alone overflows."""
    from .attention_dispatch import _varlen_backward_overflows_int32

    groups, first, start, tokens, longest = [], 0, 0, 0, 0
    for index, length in enumerate(lengths):
        if index > first and _varlen_backward_overflows_int32(
            index - first + 1, tokens + length, n_heads, head_dim
        ):
            groups.append((first, index, start, start + tokens, longest))
            first, start, tokens, longest = index, start + tokens, 0, 0
        if index == first and _varlen_backward_overflows_int32(1, length, n_heads, head_dim):
            return None
        tokens += length
        longest = max(longest, length)
    groups.append((first, len(lengths), start, start + tokens, longest))
    return groups


def _run_document_groups(run, query, key, value, seq_info, groups):
    """Documents attend only inside themselves: each group runs on its own rows and rebased
    cu_seqlens, and the rows are joined back (autograd splits the backward the same way)."""
    lengths, cu_seqlens, _ = seq_info
    outs = []
    for first, end, start, stop, longest in groups:
        rows = slice(start, stop)
        info = (lengths[first:end], cu_seqlens[first : end + 1] - start, longest)
        outs.append(run(query[:, :, rows], key[:, :, rows], value[:, :, rows], info))
    HF_PACKED_ATTENTION_STATS["int32_split"] += 1
    return torch.cat(outs, dim = 1)


def _sdpa_packed_varlen(
    module,
    query,
    key,
    value,
    attention_mask,
    *args,
    dropout = 0.0,
    scaling = None,
    is_causal = None,
    **kwargs,
):
    orig = _ORIG_SDPA[0]
    if args:
        # Positional dropout / scaling / is_causal (and later parameters) go back exactly as given.
        named = {"dropout": dropout, "scaling": scaling, "is_causal": is_causal}
        for name in list(named)[: len(args)]:
            named.pop(name)
        if kwargs.get("packed_seq_lengths") is not None and not torch.compiler.is_compiling():
            HF_PACKED_ATTENTION_STATS["fallback"] += 1
        return orig(module, query, key, value, attention_mask, *args, **named, **kwargs)
    block_varlen = kwargs.pop(BLOCK_VARLEN_KWARG, None)
    if isinstance(block_varlen, str):
        if torch.compiler.is_compiling() and not _disabled():
            return _sdpa_packed_varlen_eager(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout = dropout,
                scaling = scaling,
                is_causal = is_causal,
                **kwargs,
            )
        block_varlen = None
    # Fallback gets the uncast Q / K / V so GQA's dK / dV reduce in fp32 after repeat_kv.
    given = (query, key, value)
    if kwargs.get("packed_seq_lengths") is not None and not _disabled():
        query, key, value = _autocast_float32(query, key, value)
    # Before any counter while tracing: Dynamo would guard on it and recompile every forward.
    if block_varlen is not None and torch.compiler.is_compiling() and not _disabled():
        window = _static_window(module, query, key, value, dropout, is_causal, kwargs)
        if (
            window is not _NO_WINDOW
            and query.shape[-1] <= 256
            and _varlen_in_graph(
                query.device.index,
                query.dtype,
                int(query.shape[-1]),
                int(query.shape[1]),
                int(key.shape[1]),
                window,
            )
        ):
            # Host inputs were prepared eagerly (block_varlen_kwargs); the kernel call traces.
            scale = query.shape[-1] ** -0.5 if scaling is None else scaling
            return _flash_varlen(query, key, value, *block_varlen, scale, window), None
    if (
        IN_COMPILED_BLOCK[0]
        and torch.compiler.is_compiling()
        and kwargs.get("packed_seq_lengths") is not None
        and not _disabled()
        and _leaves_block(
            query.device.type,
            query.device.index,
            query.dtype,
            # int(): a symbolic head dim (dynamic=True) cannot be folded and would trace the probe.
            int(query.shape[-1]),
            _window_key(module, kwargs),
        )
    ):
        return _sdpa_packed_varlen_eager(
            module,
            *given,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )
    if kwargs.get("packed_seq_lengths") is None or _disabled() or torch.compiler.is_compiling():
        return orig(
            module,
            *given,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )
    from .flash_impl import flash_impl_allows
    from .attention_dispatch import (
        _VARLEN_INT32_GUARD_DISABLED,
        FLASH_VARLEN,
        XFORMERS,
        _varlen_backward_overflows_int32,
        select_attention_backend,
    )
    from .packing import get_packed_info_from_kwargs

    bsz, n_heads, q_len, head_dim = query.shape
    seq_info = torch_varlen = backend = groups = None
    efficient = False
    window = _static_window(module, query, key, value, dropout, is_causal, kwargs)
    if window is not _NO_WINDOW:
        if head_dim > 256:
            # Past every flash kernel (Gemma-4 global layers are 512): the efficient kernel, per segment.
            efficient = _efficient_varlen_takes(query, window)
        elif not _sdpa_requested(query, key):
            torch_varlen = _torch_varlen(query.device)
            # An FA4 / FA3 impl behind torch's flash op lacks some shapes (sm100: hd 136-248, hd256 + window).
            refused = torch_varlen is not None and not flash_impl_allows(
                head_dim, window, query.device
            )
            if refused:
                torch_varlen = None
            backend = select_attention_backend(use_varlen = True)
            # flash-attn's ROCm backend lists sliding windows as work in progress: windowed rows keep sdpa there.
            if window is not None and torch.version.hip is not None:
                backend = None
            # Flash-2 varlen backward faults past int32; keyed on grad mode so checkpoint recompute agrees.
            if (
                not _VARLEN_INT32_GUARD_DISABLED
                and (getattr(module, "training", False) or torch.is_grad_enabled())
                and _varlen_backward_overflows_int32(
                    kwargs["packed_seq_lengths"].numel(), q_len, n_heads, head_dim
                )
            ):
                from .packing import _host_segment_lengths

                # Documents are independent: groups of them stay under the limit, exactly.
                refused, groups = (
                    False,
                    _int32_safe_groups(
                        _host_segment_lengths(kwargs["packed_seq_lengths"]), n_heads, head_dim
                    ),
                )
                if groups is None:
                    torch_varlen, backend = None, None
            # Refused shape and no flash-attn / xformers: the efficient kernel.
            if refused and backend not in (FLASH_VARLEN, XFORMERS):
                efficient = _efficient_varlen_takes(query, window)
        if efficient or torch_varlen is not None or backend in (FLASH_VARLEN, XFORMERS):
            seq_info = get_packed_info_from_kwargs(kwargs, query.device)
    if seq_info is None or not _mask_is_block_causal(attention_mask, seq_info[1], q_len, window):
        HF_PACKED_ATTENTION_STATS["fallback"] += 1
        return orig(
            module,
            *given,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )
    HF_PACKED_ATTENTION_STATS["fast"] += 1
    scale = head_dim**-0.5 if scaling is None else scaling
    n_kv_heads = key.shape[1]
    _, cu_seqlens, max_seqlen = seq_info
    if efficient:
        try:
            out = _efficient_varlen(query, key, value, cu_seqlens, max_seqlen, scale, window)
        except torch.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            # A CUDA fault is not a refusal: let it surface.
            if _ACCELERATOR_ERROR is not None and isinstance(exc, _ACCELERATOR_ERROR):
                raise
            # A build without this kernel shape refuses at forward: stop using it, this call goes to sdpa.
            _EFFICIENT_VARLEN[:] = [False]
            logging.getLogger(__name__).warning(
                f"Unsloth: efficient varlen attention unavailable, packed rows past head dim 256 use sdpa ({exc})"
            )
            HF_PACKED_ATTENTION_STATS["fast"] -= 1
            HF_PACKED_ATTENTION_STATS["fallback"] += 1
            return orig(
                module,
                *given,
                attention_mask,
                dropout = dropout,
                scaling = scaling,
                is_causal = is_causal,
                **kwargs,
            )
        HF_PACKED_ATTENTION_STATS["efficient"] += 1
        return out, None
    if torch_varlen is not None:
        try:
            if groups is None:
                out = _call_torch_varlen(
                    torch_varlen, query, key, value, cu_seqlens, max_seqlen, scale, window
                )
            else:
                out = _run_document_groups(
                    lambda q, k, v, info: _call_torch_varlen(
                        torch_varlen, q, k, v, info[1], info[2], scale, window
                    ),
                    query,
                    key,
                    value,
                    seq_info,
                    groups,
                )
        except torch.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            if _ACCELERATOR_ERROR is not None and isinstance(exc, _ACCELERATOR_ERROR):
                raise
            # A build whose varlen picks a backend without causal GQA (cuDNN) refuses at forward: stop
            # using it and give this call to the wrapped sdpa.
            _TORCH_VARLEN[:] = [None]
            logging.getLogger(__name__).info(f"Unsloth: torch varlen_attn unavailable ({exc})")
            HF_PACKED_ATTENTION_STATS["fast"] -= 1
            HF_PACKED_ATTENTION_STATS["fallback"] += 1
            return orig(
                module,
                *given,
                attention_mask,
                dropout = dropout,
                scaling = scaling,
                is_causal = is_causal,
                **kwargs,
            )
        return out, None
    from .attention_dispatch import AttentionConfig, AttentionContext, run_attention

    config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_heads // n_kv_heads,
        flash_varlen_kwargs = {
            "dropout_p": 0.0,
            "causal": True,
            "softmax_scale": scale,
            **({} if window is None else {"window_size": (window - 1, 0)}),
        },
        sdpa_kwargs = {"scale": scale},
        xformers_kwargs = {"scale": scale},
    )

    def dispatch(query, key, value, seq_info):
        context = AttentionContext(
            bsz = 1,
            q_len = query.shape[2],
            kv_seq_len = query.shape[2],
            n_heads = n_heads,
            head_dim = head_dim,
            # Not grad mode: xformers needs a backward op for grad tensors even under no_grad
            # (reentrant checkpointing), and that op rejects the grouped 5D layout.
            requires_grad = query.requires_grad or key.requires_grad or value.requires_grad,
            seq_info = seq_info,
            attention_mask = None,
            causal_mask = None,
            sliding_window = window,
        )
        return run_attention(config = config, context = context, Q = query, K = key, V = value)

    if groups is None:
        out = dispatch(query, key, value, seq_info)
    else:
        out = _run_document_groups(dispatch, query, key, value, seq_info, groups)
    return out.contiguous(), None


_sdpa_packed_varlen_eager = torch.compiler.disable(_sdpa_packed_varlen)

# Compiled cuDNN SDPA over a dense mask builds a plan per new length, slowly before torch 2.13.
_SLOW_CUDNN_PLANS = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 13)
_LEAVES_BLOCK = {}


def _window_key(module, kwargs):
    window = _sliding_window(module, kwargs)
    return -1 if window is _NO_WINDOW else window


def _leaves_block_eager(device_type, device_index, dtype, head_dim, window):
    """Packed rows leave a compiled block only to dodge those plans, and only when the eager wrapper
    has a varlen kernel for them; elsewhere the graph break is pure cost."""
    key = (device_type, device_index, dtype, head_dim, window)
    if key not in _LEAVES_BLOCK:
        _LEAVES_BLOCK[key] = _leaves_block_probe(*key)
    return _LEAVES_BLOCK[key]


def _leaves_block_probe(device_type, device_index, dtype, head_dim, window):
    if (
        not _SLOW_CUDNN_PLANS
        or device_type != "cuda"
        or window == -1
        or head_dim % 8
        or head_dim > 256
    ):
        return False
    try:
        from .flash_impl import flash_impl_allows
        from .attention_dispatch import FLASH_VARLEN, XFORMERS, select_attention_backend
        from torch.nn.attention import SDPBackend

        device = torch.device("cuda", device_index)
        kernel = _torch_varlen(device) is not None and flash_impl_allows(head_dim, window, device)
        if not kernel and select_attention_backend(use_varlen = True) not in (FLASH_VARLEN, XFORMERS):
            return False
        q = torch.zeros(1, 2, 64, head_dim, device = device, dtype = dtype, requires_grad = True)
        mask = torch.ones(64, 64, dtype = torch.bool, device = device).tril()[None, None]
        choice = torch._fused_sdp_choice(q, q, q, attn_mask = mask, dropout_p = 0.0, is_causal = False)
        return choice == int(SDPBackend.CUDNN_ATTENTION)
    except Exception:
        return False


# Baked into the graph at trace time. Not an lru_cache: Dynamo traces those via __wrapped__ (fake tensors).
_leaves_block = torch._dynamo.assume_constant_result(_leaves_block_eager)

_VARLEN_IN_GRAPH = {}


def _varlen_in_graph_eager(device_index, dtype, head_dim, n_heads, n_kv_heads, window):
    key = (device_index, dtype, head_dim, n_heads, n_kv_heads, window)
    if key not in _VARLEN_IN_GRAPH:
        _VARLEN_IN_GRAPH[key] = _varlen_in_graph_probe(*key)
    return _VARLEN_IN_GRAPH[key]


def _no_saved_tensor_hooks():
    # Only the innermost pair applies. Pushing raises while torch has hooks disabled: keep the ambient state.
    enabled = getattr(torch._C._autograd, "_saved_tensors_hooks_is_enabled", lambda: True)
    if enabled():
        return torch.autograd.graph.saved_tensors_hooks(_identity, _identity)
    return contextlib.nullcontext()


def _identity(tensor):
    return tensor


def _varlen_in_graph_probe(device_index, dtype, head_dim, n_heads, n_kv_heads, window):
    """aten's flash varlen takes this shape on this device: one real forward and backward (a build
    that refuses at run time would otherwise fail inside the compiled graph)."""
    # Cached for all grad modes; first trace may be no_grad (reentrant checkpoint), where backward raises.
    # Identity hooks: a non-reentrant checkpoint's pack hooks would make the probe's backward recompute the block.
    with torch.inference_mode(False), torch.enable_grad(), _no_saved_tensor_hooks():
        try:
            from .flash_impl import flash_impl_allows

            device = torch.device("cuda", device_index)
            torch_varlen = _torch_varlen(device)
            # Hardware and build only: the sdpa policy can change, block_varlen_kwargs checks it each step.
            if torch_varlen is None or not flash_impl_allows(head_dim, window, device):
                return False
            query = torch.zeros(1, n_heads, 16, head_dim, device = device, dtype = dtype)
            key = torch.zeros(1, n_kv_heads, 16, head_dim, device = device, dtype = dtype)
            query.requires_grad_()
            cu_seqlens = torch.tensor([0, 7, 16], dtype = torch.int32, device = device)
            out = _flash_varlen(query, key, key, cu_seqlens, 9, 1.0, window)
            out.float().sum().backward()
            return True
        except Exception:
            return False


_varlen_in_graph = torch._dynamo.assume_constant_result(_varlen_in_graph_eager)


def block_varlen_kwargs(layer, hidden_states, kwargs):
    """Eager, before _block_compile runs a compiled decoder block: (cu_seqlens, max_seqlen) for its
    packed row, so attention runs torch's varlen op inside the graph. Building them there needs a host
    read of the longest segment and the mask check a host sync, which is why the traced wrapper used to
    fall back to SDPA over the dense mask. None keeps that path; _LEAVE_BLOCK runs the eager
    wrapper."""
    seq_lengths, mask = kwargs.get("packed_seq_lengths"), kwargs.get("attention_mask")
    attention = getattr(layer, "self_attn", None)
    # Layers of one forward (and its checkpoint recompute) share the lengths and their window's mask.
    window = _sliding_window(attention, kwargs) if attention is not None else None
    entry = _BLOCK_PREP.get(window)
    if (
        entry is not None
        # Live tensors only: a dead weakref is None, and so is a missing mask.
        and isinstance(seq_lengths, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        and entry[0]() is seq_lengths
        and entry[1]() is mask
        and entry[2] == (hidden_states.device, seq_lengths._version, mask._version)
        # cu_seqlens built under inference_mode cannot be saved for a training step's backward.
        and (not entry[4] or torch.is_inference_mode_enabled())
        # sdpa_kernel / deterministic mode can change between calls on the same tensors.
        and (entry[5] is None or _sdpa_policy_requested(*entry[5]) == entry[6])
    ):
        if isinstance(entry[3], tuple):
            HF_PACKED_ATTENTION_STATS["compiled"] += 1
        return entry[3]
    policy = []
    result = _block_varlen_prepare(layer, attention, hidden_states, kwargs, policy)
    if (
        isinstance(seq_lengths, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        # Versions catch in-place edits; inference tensors have none, so are always rechecked.
        and not (torch.is_inference(seq_lengths) or torch.is_inference(mask))
        and not _disabled()
    ):
        # Weak: the T x T mask must not outlive its step.
        _BLOCK_PREP[window] = (
            weakref.ref(seq_lengths),
            weakref.ref(mask),
            (hidden_states.device, seq_lengths._version, mask._version),
            result,
            torch.is_inference_mode_enabled(),
            *(policy[0] if policy else (None, None)),
        )
    return result


_BLOCK_PREP = {}


def _block_varlen_prepare(
    layer,
    attention,
    hidden_states,
    kwargs,
    policy = None,
):
    if (
        _disabled()
        or _ORIG_SDPA[0] is None
        or kwargs.get("packed_seq_lengths") is None
        or not isinstance(hidden_states, torch.Tensor)
        or not hidden_states.is_cuda
        or hidden_states.dim() != 3
        or hidden_states.shape[0] != 1
    ):
        return None
    config = getattr(attention, "config", None)
    if getattr(config, "_attn_implementation", None) != "sdpa":
        return None
    window = _sliding_window(attention, kwargs)
    if window is _NO_WINDOW or _torch_varlen(hidden_states.device) is None:
        return None
    from .attention_dispatch import _VARLEN_INT32_GUARD_DISABLED, _varlen_backward_overflows_int32
    from .packing import get_packed_info_from_kwargs

    total = hidden_states.shape[1]
    head_dim = getattr(attention, "head_dim", None)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", None) or n_heads
    if not all(isinstance(n, int) for n in (head_dim, n_heads, n_kv_heads)):
        return None
    # SDPA runs in the autocast dtype when autocast casts fp32 activations (_autocast_float32).
    dtype = hidden_states.dtype
    if dtype == torch.float32:
        dtype = _autocast_dtype(hidden_states.device.type) or dtype
        dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    policy_args = (hidden_states.device, dtype, head_dim, n_heads, n_kv_heads)
    wants_sdpa = _sdpa_policy_requested(*policy_args)
    if policy is not None:
        policy.append((policy_args, wants_sdpa))
    if wants_sdpa:
        return None
    if not _VARLEN_INT32_GUARD_DISABLED and _varlen_backward_overflows_int32(
        kwargs["packed_seq_lengths"].numel(), total, n_heads, head_dim
    ):
        from .packing import _host_segment_lengths

        # Splitting at document boundaries in the graph would need per-step shapes: the eager wrapper
        # splits instead (a graph break for these rows only). One document alone keeps the dense path.
        lengths = _host_segment_lengths(kwargs["packed_seq_lengths"])
        return None if _int32_safe_groups(lengths, n_heads, head_dim) is None else _LEAVE_BLOCK
    seq_info = get_packed_info_from_kwargs(kwargs, hidden_states.device)
    if seq_info is None or not _mask_is_block_causal(
        kwargs.get("attention_mask"), seq_info[1], total, window
    ):
        return None
    _, cu_seqlens, max_seqlen = seq_info
    # The segment count changes every step: one graph for all of them.
    torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
    HF_PACKED_ATTENTION_STATS["compiled"] += 1
    return cu_seqlens, int(max_seqlen)


def enable_hf_packed_attention() -> bool:
    """Wrap transformers' "sdpa" attention entry once. True when the wrapper is installed."""
    if _disabled():
        return False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return False
    current = ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
    if current is None:
        return False
    # Once per process: gemma-4's router stacked above does not forward our sentinel, and re-wrapping
    # it would loop through its boxed original.
    if getattr(current, _FLAG, False) or _ORIG_SDPA[0] is not None:
        return True
    _ORIG_SDPA[0] = current
    # Zoo sdpa routers re-run per load and check their sentinel on the installed entry: forward them.
    try:
        for name, value in vars(current).items():
            if name.startswith(("_unsloth_", "__unsloth_")):
                setattr(_sdpa_packed_varlen, name, value)
    except Exception:
        pass
    setattr(_sdpa_packed_varlen, _FLAG, True)
    # Direct assignment: register() does not update the mapping layers read.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_packed_varlen
    return True

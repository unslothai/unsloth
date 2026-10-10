# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed rows on the transformers modeling path run a varlen flash kernel (torch's, else xformers /
flash-attn) instead of SDPA over a dense block-causal mask; anything else reaches the wrapped "sdpa"."""

import inspect
import logging
import os
import weakref

import torch

__all__ = ["enable_hf_packed_attention", "HF_PACKED_ATTENTION_STATS"]

_FLAG = "_unsloth_hf_packed_varlen"
_ENV = "UNSLOTH_HF_PACKED_VARLEN"
_ORIG_SDPA = [None]
HF_PACKED_ATTENTION_STATS = {"fast": 0, "fallback": 0, "mask_checks": 0}
# All layers of a forward share one mask object: weakref, (cu_seqlens, versions), verdict.
_MASK_CHECK = [None, None, None]
_MASK_CHECK_ROWS = 1024


def _disabled() -> bool:
    return os.environ.get(_ENV, "1").lower() in ("0", "false", "no", "off")


def _mask_is_block_causal(attention_mask, cu_seqlens, total) -> bool:
    """Exactly causal inside each packed segment and closed across them; windows, bidirectional
    spans, pad tails and None (one document, already is_causal) keep the wrapped path."""
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
    ref, checked, verdict = _MASK_CHECK
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
        segment = torch.repeat_interleave(
            torch.arange(lengths.numel(), device = attention_mask.device),
            lengths.to(attention_mask.device),
        )
        position = torch.arange(total, device = attention_mask.device)
        # Row blocks keep the reference to _MASK_CHECK_ROWS x T, not a second T x T mask.
        verdict = True
        for start in range(0, total, _MASK_CHECK_ROWS):
            rows = slice(start, min(start + _MASK_CHECK_ROWS, total))
            expected = (segment[rows, None] == segment[None, :]) & (
                position[None, :] <= position[rows, None]
            )
            if not torch.equal(attention_mask[0, 0, rows], expected):
                verdict = False
                break
    if cacheable:
        _MASK_CHECK[:] = [weakref.ref(attention_mask), (cu_seqlens, key), verdict]
    return verdict


def _autocast_dtype_differs(query) -> bool:
    try:
        enabled = torch.is_autocast_enabled(query.device.type)
        dtype = torch.get_autocast_dtype(query.device.type) if enabled else None
    except (AttributeError, TypeError):
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype() if enabled else None
    return enabled and dtype != query.dtype


def _sliding_or_softcapped(module, kwargs) -> bool:
    if kwargs.get("sliding_window") or getattr(module, "sliding_window", None):
        return True
    if kwargs.get("softcap") or getattr(module, "sinks", None) is not None:
        return True
    if getattr(module, "attn_logit_softcapping", None):
        return True
    config = getattr(module, "config", None)
    return bool(getattr(config, "attn_logit_softcapping", None))


_TORCH_VARLEN = []
_TORCH_VARLEN_DEVICES = {}


def _torch_varlen(device):
    """torch's own flash varlen (native GQA, a third of xformers' host cost per call), else None."""
    if not _TORCH_VARLEN:
        fn = None
        try:
            from torch.nn.attention.varlen import varlen_attn

            # Older releases spell causality differently; only the window_size / enable_gqa API is used.
            if {"scale", "window_size", "enable_gqa"} <= inspect.signature(
                varlen_attn
            ).parameters.keys():
                fn = varlen_attn
        except Exception:
            fn = None
        _TORCH_VARLEN.append(fn)
    fn = _TORCH_VARLEN[0]
    if fn is None or torch.version.hip is not None:
        return None
    ok = _TORCH_VARLEN_DEVICES.get(device.index)
    if ok is None:
        ok = torch.cuda.get_device_capability(device)[0] >= 8
        _TORCH_VARLEN_DEVICES[device.index] = ok
    return fn if ok else None


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
    # Before any counter while tracing: Dynamo would guard on it and recompile every forward.
    if kwargs.get("packed_seq_lengths") is None or _disabled() or torch.compiler.is_compiling():
        return orig(
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
    from .attention_dispatch import (
        _VARLEN_INT32_GUARD_DISABLED,
        FLASH_VARLEN,
        XFORMERS,
        _varlen_backward_overflows_int32,
        select_attention_backend,
    )
    from .packing import get_packed_info_from_kwargs

    bsz, n_heads, q_len, head_dim = query.shape
    seq_info = torch_varlen = backend = None
    if (
        bsz == 1
        and q_len > 0
        and query.is_cuda
        and query.device == key.device == value.device
        and key.shape[2] == q_len
        # Not MLA (192 / 128) nor Gemma-4 global (512).
        and key.shape[-1] == value.shape[-1] == head_dim
        and head_dim <= 256
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
        and not _sliding_or_softcapped(module, kwargs)
    ):
        torch_varlen = _torch_varlen(query.device)
        backend = select_attention_backend(use_varlen = True)
        # Flash-2 varlen backward faults past int32 indexing. Keyed on training / grad mode, not
        # requires_grad, so a reentrant checkpoint's no_grad forward and its recompute agree.
        if (
            not _VARLEN_INT32_GUARD_DISABLED
            and (getattr(module, "training", False) or torch.is_grad_enabled())
            and _varlen_backward_overflows_int32(
                kwargs["packed_seq_lengths"].numel(), q_len, n_heads, head_dim
            )
        ):
            torch_varlen, backend = None, None
        if torch_varlen is not None or backend in (FLASH_VARLEN, XFORMERS):
            seq_info = get_packed_info_from_kwargs(kwargs, query.device)
    if seq_info is None or not _mask_is_block_causal(attention_mask, seq_info[1], q_len):
        HF_PACKED_ATTENTION_STATS["fallback"] += 1
        return orig(
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
    HF_PACKED_ATTENTION_STATS["fast"] += 1
    scale = head_dim**-0.5 if scaling is None else scaling
    n_kv_heads = key.shape[1]
    _, cu_seqlens, max_seqlen = seq_info
    if torch_varlen is not None:
        try:
            out = torch_varlen(
                query[0].transpose(0, 1),
                key[0].transpose(0, 1),
                value[0].transpose(0, 1),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale = scale,
                window_size = (-1, 0),
                enable_gqa = n_kv_heads != n_heads,
            )
        except torch.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            # A build whose varlen picks a backend without causal GQA (cuDNN) refuses at forward: stop
            # using it and give this call to the wrapped sdpa.
            _TORCH_VARLEN[:] = [None]
            logging.getLogger(__name__).info(f"Unsloth: torch varlen_attn unavailable ({exc})")
            HF_PACKED_ATTENTION_STATS["fast"] -= 1
            HF_PACKED_ATTENTION_STATS["fallback"] += 1
            return orig(
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
        return out.unsqueeze(0), None
    from .attention_dispatch import AttentionConfig, AttentionContext, run_attention

    config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_heads // n_kv_heads,
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True, "softmax_scale": scale},
        sdpa_kwargs = {"scale": scale},
        xformers_kwargs = {"scale": scale},
    )
    context = AttentionContext(
        bsz = 1,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = n_heads,
        head_dim = head_dim,
        # Not grad mode: xformers needs a backward op for grad tensors even under no_grad (reentrant
        # checkpointing), and that op rejects the grouped 5D layout.
        requires_grad = query.requires_grad or key.requires_grad or value.requires_grad,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    out = run_attention(config = config, context = context, Q = query, K = key, V = value)
    return out.contiguous(), None


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

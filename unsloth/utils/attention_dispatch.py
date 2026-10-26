# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

import functools
import operator
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ..device_type import DEVICE_TYPE
from ..models._utils import *
from ..context_parallel import get_cp_manager
from .ulysses import ulysses_attention
from ..utils.packing import (
    _inference_mode,
    build_sdpa_packed_attention_mask,
    build_xformers_block_causal_mask,
    cover_padded_cu_seqlens,
    move_xformers_attention_bias,
    packed_block_mask,
    packed_segment_lengths,
)

flash_attn_func = None
flash_attn_varlen_func = None
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
HAS_XFORMERS = xformers is not None


def _xformers_runs_on_device(device = "cuda", supports_bf16 = None) -> bool:
    """One tiny attention forward; True iff the xformers kernel actually runs here."""
    try:
        # Pre-Ampere GPUs (sm < 80: Turing/Volta) have no bfloat16 attention kernel but run xformers fine in
        # float16, so pick the dtype the device supports.
        if supports_bf16 is None:
            supports_bf16 = SUPPORTS_BFLOAT16
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
        q = torch.zeros((1, 8, 1, 64), device = device, dtype = dtype)
        attn_bias = xformers.attn_bias.BlockDiagonalCausalMask.from_seqlens([8])
        xformers_attention(q, q, q, attn_bias = attn_bias)
        # Launches are async, so synchronize or a deferred kernel failure escapes this probe.
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def _xformers_fa3_enabled() -> bool:
    return getattr(xformers, "_get_use_fa3", lambda: True)()


def _xformers_disabled_for_capability(
    capability,
    probe = _xformers_runs_on_device,
    fa3_enabled = _xformers_fa3_enabled,
) -> bool:
    # At sm_120 (RTX 50-series) xformers' cutlass op is capability-rejected (it caps at sm_90) and its
    # flash-2 op runs only if the build ships an sm_120 kernel, so run one real forward to decide.
    # On sm_90 FA3 is off only for builds without sm_90a (_utils), which may also ship no sm_90 kernel
    # at all (conda-forge xformers >= 0.0.33), so probe those too; elsewhere xformers works.
    if capability[0] >= 12 or (tuple(capability) == (9, 0) and not fa3_enabled()):
        return not probe()
    return False


# FlashAttention always wins in select_attention_backend, so skip the probe when installed.
if HAS_XFORMERS and not HAS_FLASH_ATTENTION and torch.cuda.is_available():
    if _xformers_disabled_for_capability(torch.cuda.get_device_capability()):
        HAS_XFORMERS = False

# On sm_100+ (B200, sm_120) xformers' fp32-capable cutlass op is capability-rejected and only its
# fp16/bf16 flash-2 op runs, so fp32 Q/K/V (DoRA, #1013) must be downcast there; below sm_100
# cutlass handles fp32 natively. Device 0 only: _xformers_rejects_fp32 reads Q's card.
_XFORMERS_FP32_UNSUPPORTED = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)


def scaled_dot_product_attention(*args, **kwargs):
    # Resolved per call: context_parallel monkey patches torch.nn.functional.
    return F.scaled_dot_product_attention(*args, **kwargs)


def _avoids_cudnn(Q, training):
    """Training calls on SM100 run without cuDNN attention (_training_sdpa); probes must match."""
    return bool(training) and Q.is_cuda and Q.device.index in _NO_CUDNN_TRAINING


def _training_sdpa(*args, **kwargs):
    """SDPA for a training forward (AttentionContext.training, passed in, no global state: these
    calls run traced). On SM100 SDPA picks cuDNN attention, which builds a plan for every new
    sequence length (~150 ms each), and training batches change length every step."""
    if args and _avoids_cudnn(args[0], True):
        with _sdpa_kernel(_XFORMERS_LIKE_BACKENDS):
            return scaled_dot_product_attention(*args, **kwargs)
    if len(args) >= 3 and _cudnn_length_bucket_applies(*args[:3], **kwargs):
        return _cudnn_length_bucketed_sdpa(*args, **kwargs)
    return scaled_dot_product_attention(*args, **kwargs)


def _inference_sdpa(*args, **kwargs):
    """Non-training SDPA: a prefill also pays cuDNN's per-length plan, so it is bucketed too."""
    if len(args) >= 3 and _cudnn_length_bucket_applies(*args[:3], **kwargs):
        return _cudnn_length_bucketed_sdpa(*args, **kwargs)
    return scaled_dot_product_attention(*args, **kwargs)


# Wherever SDPA picks cuDNN attention for a training call (H100 on torch 2.14: the default
# priority), every new sequence length builds a new cuDNN plan, ~170 ms forward plus backward on
# H100. Padded batches change length every step, so that cost landed on nearly every step. Padding
# Q / K / V along the sequence to a multiple of this bucket bounds the plans to max_len / bucket.
# Causal calls: causal masking hides the padded keys. Masked calls (GRPO / DPO padding): the mask is
# padded too (new keys masked out, new query rows attend everything and get zero gradient). 0 disables.
try:
    _CUDNN_LENGTH_BUCKET = max(int(os.environ.get("UNSLOTH_SDPA_CUDNN_BUCKET", "128")), 0)
except ValueError:
    _CUDNN_LENGTH_BUCKET = 128


def _cudnn_length_bucket_applies(
    Q,
    K,
    V,
    attn_mask = None,
    dropout_p = 0.0,
    is_causal = False,
    **kwargs,
):
    if not _CUDNN_LENGTH_BUCKET or _CUDNN_CHOICE is None:
        return False
    # Under context parallelism the tensors can be sequence shards: padding a shard would put unmasked
    # zero keys inside the global sequence.
    if get_cp_manager() is not None:
        return False
    if dropout_p or (attn_mask is None) != bool(is_causal):
        return False
    if not (isinstance(Q, Tensor) and Q.is_cuda and Q.dim() == 4 and K.dim() == 4 and V.dim() == 4):
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16) or Q.shape[-2] != K.shape[-2]:
        return False
    enable_gqa = True if kwargs.get("enable_gqa", False) else False
    if attn_mask is not None:
        if not _cudnn_bucket_mask_ok(attn_mask, Q, K):
            return False
        needs_grad = torch.is_grad_enabled() and (
            Q.requires_grad or K.requires_grad or V.requires_grad
        )
        choice = _sdpa_masked_choice(
            str(Q.device),
            Q.dtype,
            Q.shape[-1],
            Q.shape[1],
            K.shape[1],
            enable_gqa,
            True if needs_grad else False,
        )
        return choice == _CUDNN_CHOICE
    _, choice = _sdpa_backend_state(
        str(Q.device),
        Q.dtype,
        Q.shape[-1],
        Q.shape[1],
        K.shape[1],
        True,
        enable_gqa,
    )
    return choice == _CUDNN_CHOICE


def _cudnn_bucket_mask_ok(attn_mask, Q, K):
    """Per-batch 4D padding masks only (bool or Q-dtype additive, no grad). Batch-shared cached masks
    are skipped: a padded copy per layer would multiply their memory."""
    if not isinstance(attn_mask, Tensor) or attn_mask.dim() != 4 or attn_mask.requires_grad:
        return False
    if attn_mask.dtype != torch.bool and attn_mask.dtype != Q.dtype:
        return False
    if attn_mask.device != Q.device or attn_mask.shape[-1] != K.shape[-2]:
        return False
    if attn_mask.shape[-2] != 1 and attn_mask.shape[-2] != Q.shape[-2]:
        return False
    if Q.shape[0] == 1 or attn_mask.shape[0] != Q.shape[0]:
        return False
    return attn_mask.shape[1] == 1 or attn_mask.shape[1] == Q.shape[1]


def _sdpa_masked_choice(device, dtype, head_dim, n_heads, n_kv_heads, enable_gqa, needs_grad):
    """SDPBackend SDPA picks for a masked, non-causal call with these shapes (-1 if unknown)."""
    cuda = torch.backends.cuda
    backends = (
        cuda.flash_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        tuple(getattr(torch._C, "_get_sdp_priority_order", tuple)()),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    gqa, mha = _sdpa_gqa_choices(
        device, dtype, head_dim, n_heads, n_kv_heads, False, True, needs_grad, 0.0, backends
    )
    return gqa if enable_gqa else mha


def _cudnn_length_bucketed_sdpa(
    Q,
    K,
    V,
    attn_mask = None,
    dropout_p = 0.0,
    is_causal = False,
    **kwargs,
):
    # Always pad by 1..bucket rows and slice: a zero pad specialises under dynamo (a recompile when the
    # length becomes a bucket multiple), so an exact multiple carries one spare bucket instead.
    q_len = Q.shape[-2]
    pad = _CUDNN_LENGTH_BUCKET - q_len % _CUDNN_LENGTH_BUCKET
    Q, K, V = (F.pad(x, (0, 0, 0, pad)) for x in (Q, K, V))
    if attn_mask is None:
        out = scaled_dot_product_attention(Q, K, V, is_causal = True, **kwargs)
        return out[..., :q_len, :]
    # Padded query rows attend everything (no fully masked row) and are sliced off; padded keys are masked.
    rows = attn_mask.shape[-2]
    shape = (*attn_mask.shape[:-2], rows if rows == 1 else rows + pad, q_len + pad)
    if attn_mask.dtype == torch.bool:
        padded = attn_mask.new_ones(shape)
        padded[..., :rows, q_len:] = False
    else:
        padded = attn_mask.new_zeros(shape)
        padded[..., :rows, q_len:] = torch.finfo(attn_mask.dtype).min
    padded[..., :rows, :q_len] = attn_mask
    out = scaled_dot_product_attention(Q, K, V, attn_mask = padded, **kwargs)
    return out[..., :q_len, :]


def _rerouted_sdpa(*args, **kwargs):
    """SDPA standing in for xformers inside a compiled block, on the kernels xformers would run.
    Without the pin, torch 2.13+ on SM100 takes cuDNN attention, which builds a new plan for every
    new sequence length (~150 ms per length): padded batches of varying length stalled every step."""
    with _sdpa_kernel(_XFORMERS_LIKE_BACKENDS):
        return F.scaled_dot_product_attention(*args, **kwargs)


# Module level: reading SDPBackend's attributes while traced breaks the graph.
try:
    from torch.nn.attention import SDPBackend as _SDPBackend, sdpa_kernel as _sdpa_kernel
    _XFORMERS_LIKE_BACKENDS = [
        _SDPBackend.FLASH_ATTENTION,
        _SDPBackend.EFFICIENT_ATTENTION,
        _SDPBackend.MATH,
    ]
    _CUDNN_CHOICE = (
        int(_SDPBackend.CUDNN_ATTENTION) if hasattr(_SDPBackend, "CUDNN_ATTENTION") else None
    )
except Exception:
    _rerouted_sdpa = scaled_dot_product_attention
    _CUDNN_CHOICE = None


def _per_device_backends(capabilities):
    """(flash runs, xformers runs, xformers rejects fp32, bf16 kernels) per CUDA device index, or None
    when every device has device 0's capability class, so homogeneous hosts never consult the table.
    xformers runs is None where it needs a probe on that card, taken on first dispatch there."""
    if len({(c[0] >= 8, c[0] >= 10, c[0] >= 12) for c in capabilities}) < 2:
        return None
    table = []
    for index, capability in enumerate(capabilities):
        flash = HAS_FLASH_ATTENTION and capability[0] >= 8
        xformers_ok = HAS_XFORMERS
        # Device 0 was probed above; xformers only serves another sm_120 card when flash cannot.
        if xformers_ok and index > 0 and not flash and capability[0] >= 12:
            xformers_ok = None
        table.append((flash, xformers_ok, capability[0] >= 10, capability[0] >= 8))
    if all(row == table[0] for row in table):
        return None
    return tuple(table)


def _visible_capabilities():
    if DEVICE_TYPE != "cuda" or not torch.cuda.is_available():
        return ()
    try:
        return tuple(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
    except Exception:
        return ()


def _no_cudnn_training_devices():
    """Device indices whose training SDPA calls avoid cuDNN attention (SM100). UNSLOTH_SDPA_TRAIN_CUDNN=1 keeps it."""
    if os.environ.get("UNSLOTH_SDPA_TRAIN_CUDNN", "0") == "1" or torch.version.hip is not None:
        return frozenset()
    return frozenset(i for i, cap in enumerate(_visible_capabilities()) if cap[0] == 10)


_NO_CUDNN_TRAINING = _no_cudnn_training_devices()


def _decode_no_cudnn_devices():
    """sm90 / sm100 devices whose decode SDPA avoids cuDNN: it builds a plan per key length, i.e. per
    generated token. UNSLOTH_SDPA_DECODE_CUDNN=1 keeps it."""
    if os.environ.get("UNSLOTH_SDPA_DECODE_CUDNN", "0") == "1" or torch.version.hip is not None:
        return frozenset()
    return frozenset(i for i, cap in enumerate(_visible_capabilities()) if cap[0] in (9, 10))


_NO_CUDNN_DECODE = _decode_no_cudnn_devices()


_DECODE_PICKS_CUDNN = {}


def _decode_picks_cudnn(query, key, attn_mask, enable_gqa):
    """Whether torch would pick cuDNN for this decode call (else sdpa_kernel is ~25 us of overhead)."""
    cache_key = (
        query.device.index,
        query.dtype,
        query.shape[-1],
        query.shape[1],
        key.shape[1],
        attn_mask is not None,
        enable_gqa,
        torch.backends.cuda.cudnn_sdp_enabled(),
    )
    picks = _DECODE_PICKS_CUDNN.get(cache_key)
    if picks is None:
        cuda = torch.backends.cuda
        backends = (
            cuda.flash_sdp_enabled(),
            cuda.cudnn_sdp_enabled(),
            cuda.mem_efficient_sdp_enabled(),
            cuda.math_sdp_enabled(),
            tuple(getattr(torch._C, "_get_sdp_priority_order", tuple)()),
        )
        gqa, mha = _sdpa_gqa_choices(
            str(query.device), query.dtype, query.shape[-1], query.shape[1], key.shape[1],
            False, attn_mask is not None, False, 0.0, backends, single_query = True,
        )  # fmt: skip
        picks = _DECODE_PICKS_CUDNN[cache_key] = (gqa if enable_gqa else mha) == _CUDNN_CHOICE
    return picks


def decode_scaled_dot_product_attention(query, key, value, *args, **kwargs):
    """Decode SDPA, kept off cuDNN where torch would pick it on _NO_CUDNN_DECODE devices."""
    if (
        _CUDNN_CHOICE is not None
        and query.is_cuda
        and query.shape[-2] == 1
        and query.device.index in _NO_CUDNN_DECODE
        and (
            torch.compiler.is_compiling()
            or _decode_picks_cudnn(
                query,
                key,
                kwargs.get("attn_mask", args[0] if args else None),
                bool(kwargs.get("enable_gqa", False)),
            )
        )  # fmt: skip
    ):
        with _sdpa_kernel(_XFORMERS_LIKE_BACKENDS):
            return F.scaled_dot_product_attention(query, key, value, *args, **kwargs)
    return F.scaled_dot_product_attention(query, key, value, *args, **kwargs)


decode_scaled_dot_product_attention.__doc__ = (
    F.scaled_dot_product_attention.__doc__ or ""
) + decode_scaled_dot_product_attention.__doc__


# Mixed-generation hosts only (e.g. sm_86 + sm_75): the backend is re-checked against Q's device, since a
# device-0 verdict is wrong for layers a device_map places on another card.
_DEVICE_BACKENDS = _per_device_backends(_visible_capabilities())
# Read at import so a traced call only indexes a tuple.
_XFORMERS_FP32_UNSUPPORTED_BY_INDEX = tuple(c[0] >= 10 for c in _visible_capabilities())


def _xformers_rejects_fp32(device) -> bool:
    """sm_100+ CUDA only: CPU tensors on a B200 host must not inherit device 0's answer."""
    if getattr(device, "type", None) != "cuda":
        return False
    index = getattr(device, "index", None)
    if index is not None and index < len(_XFORMERS_FP32_UNSUPPORTED_BY_INDEX):
        return _XFORMERS_FP32_UNSUPPORTED_BY_INDEX[index]
    return _XFORMERS_FP32_UNSUPPORTED


# Lazy so import never opens a CUDA context on another rank's card (torchrun sees every GPU).
_XFORMERS_PROBED: dict = {}


def _xformers_runs_on(index: int) -> bool:
    if index not in _XFORMERS_PROBED:
        if torch.compiler.is_compiling():
            return False  # never probe inside a traced region; SDPA is always valid
        _XFORMERS_PROBED[index] = _xformers_runs_on_device(f"cuda:{index}", supports_bf16 = True)
    return _XFORMERS_PROBED[index]


def _backend_for_device(
    backend: str,
    device,
    dtype = None,
) -> str:
    index = getattr(device, "index", None)
    if getattr(device, "type", None) != "cuda" or index is None or index >= len(_DEVICE_BACKENDS):
        return backend
    flash, xformers_ok, _, bf16 = _DEVICE_BACKENDS[index]
    if backend in (FLASH_DENSE, FLASH_VARLEN) and not flash:
        return SDPA
    # A bfloat16 model (chosen from device 0) has no xformers kernel on a pre-Ampere card.
    if backend == XFORMERS:
        if dtype == torch.bfloat16 and not bf16:
            return SDPA
        if xformers_ok is None:
            xformers_ok = _xformers_runs_on(index)
        if not xformers_ok:
            return SDPA
    return backend


SDPA_HAS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")

# Packed SDPA runs per segment: a dense (T, T) mask + enable_gqa leaves only the math kernel.
_SDPA_PACKED_SEGMENTS = os.environ.get("UNSLOTH_SDPA_PACKED_SEGMENTS", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _load_torch_varlen_attn():
    # Needs window_size (torch 2.11) and enable_gqa (2.12); older builds keep the per-segment path.
    if os.environ.get("UNSLOTH_TORCH_VARLEN", "1").lower() in ("0", "false", "no", "off"):
        return None
    try:
        import inspect
        from torch.nn.attention.varlen import varlen_attn
    except Exception:
        return None
    params = inspect.signature(varlen_attn).parameters
    if "window_size" not in params or "enable_gqa" not in params:
        return None
    # Causal varlen always uses torch's bundled flash kernel, which some source builds omit.
    if not getattr(torch.backends.cuda, "is_flash_attention_available", lambda: True)():
        return None
    return varlen_attn


_TORCH_VARLEN_ATTN = _load_torch_varlen_attn()

# varlen_attn is a torch.library custom op: per call it pays Python custom-op dispatch twice (forward
# and backward), a cuDNN eligibility check and a zeros launch for its rng state. Padding-free training
# runs it once per layer per micro-batch, which on a launch-bound step costs more than the padding it
# saves. Eager calls that torch would send to flash anyway take the same aten flash kernels through
# a plain autograd.Function. UNSLOTH_LEAN_VARLEN=0 restores varlen_attn everywhere.
_LEAN_VARLEN = _TORCH_VARLEN_ATTN is not None and os.environ.get(
    "UNSLOTH_LEAN_VARLEN", "1"
).lower() not in ("0", "false", "no", "off")
if _LEAN_VARLEN:
    try:
        from torch.nn.attention.varlen import _can_use_cudnn as _varlen_can_use_cudnn
    except Exception:
        # torch 2.12 / 2.13 decide per device only (_should_use_cudnn, False on every GPU in 2.13).
        try:
            from torch.nn.attention.varlen import _should_use_cudnn as _varlen_should_use_cudnn
            def _varlen_can_use_cudnn(
                query,
                key,
                value,
                max_q,
                window_size,
                enable_gqa = False,
            ):
                return query.is_cuda and _varlen_should_use_cudnn(query.device.index)

        except Exception:
            _LEAN_VARLEN = False


class _LeanFlashVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, cu_seqlens, max_seqlen, is_causal, scale, window_left, window_right
    ):
        out, lse, philox_seed, philox_offset, _ = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            0.0,
            is_causal,
            return_debug_mask = False,
            scale = scale,
            window_size_left = window_left,
            window_size_right = window_right,
        )
        ctx.save_for_backward(query, key, value, cu_seqlens, out, lse, philox_seed, philox_offset)
        ctx.max_seqlen, ctx.is_causal, ctx.scale = max_seqlen, is_causal, scale
        ctx.window = (window_left, window_right)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, cu_seqlens, out, lse, philox_seed, philox_offset = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out.contiguous(),
            query,
            key,
            value,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            0.0,
            ctx.is_causal,
            philox_seed,
            philox_offset,
            scale = ctx.scale,
            window_size_left = ctx.window[0],
            window_size_right = ctx.window[1],
        )
        return dq, dk, dv, None, None, None, None, None, None


def _varlen_attn_call(Q_f, K_f, V_f, cu_seqlens, max_seqlen, *, scale, window_size, enable_gqa):
    if (
        _LEAN_VARLEN
        and not torch.compiler.is_compiling()
        and not _varlen_can_use_cudnn(Q_f, K_f, V_f, max_seqlen, list(window_size), enable_gqa)
    ):
        return _LeanFlashVarlen.apply(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            max_seqlen,
            window_size == (-1, 0),
            scale,
            window_size[0],
            window_size[1],
        )
    return _TORCH_VARLEN_ATTN(
        Q_f,
        K_f,
        V_f,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        scale = scale,
        window_size = window_size,
        enable_gqa = enable_gqa,
    )


from .flash_impl import flash_impl_allows  # noqa: E402


def _torch_varlen_takes(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    sdpa_kwargs: dict,
    is_causal: bool,
    sliding_window: Optional[int],
) -> bool:
    """True when torch's varlen kernel computes exactly what the per-segment SDPA path would."""
    if _TORCH_VARLEN_ATTN is None or not Q.is_cuda or torch.version.hip is not None:
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16) or K.dtype != Q.dtype or V.dtype != Q.dtype:
        return False
    if (
        Q.shape[-1] > 256
        or Q.shape[-1] % 8 != 0
        or torch.cuda.get_device_capability(Q.device)[0] < 8
    ):
        return False
    # A bidirectional window has no (left, right) spelling matching packed_block_mask.
    if set(sdpa_kwargs) - {"scale", "dropout_p"} or sdpa_kwargs.get("dropout_p"):
        return False
    if sliding_window is not None and not is_causal:
        return False
    # torch 2.14 routes bidirectional MHA to cuDNN ragged on sm90/sm100: no head_dim 256 plan.
    if not is_causal and Q.shape[1] == K.shape[1]:
        return False
    # An FA4 / FA3 impl behind torch's flash op lacks some shapes (sm100: hd 136-248, hd256 + window).
    if not flash_impl_allows(Q.shape[-1], sliding_window, Q.device):
        return False
    return not torch.are_deterministic_algorithms_enabled()


# Above this mean segment length cuDNN's per-segment SDPA beats torch's varlen flash kernel (B200 sweep).
# Only sm_100-family: on H100 the same cuDNN choice is 2.2x to 2.6x slower than varlen at mean lengths
# 1024 to 2048, so there varlen always wins.
_VARLEN_CUDNN_MAX_MEAN_LEN = 1024
_VARLEN_CUDNN_ARCH_MAJORS = frozenset({10})


@functools.lru_cache(maxsize = 64)
def _sdpa_choice(device, dtype, head_dim, n_heads, n_kv_heads, is_causal, enable_gqa, backends):
    try:
        q = torch.empty(1, n_heads, 128, head_dim, device = device, dtype = dtype)
        k = torch.empty(1, n_kv_heads, 128, head_dim, device = device, dtype = dtype)
        return int(torch._fused_sdp_choice(q, k, k, is_causal = is_causal, enable_gqa = enable_gqa))
    except Exception:
        return -1


def _sdpa_backend_state(device, dtype, head_dim, n_heads, n_kv_heads, is_causal, enable_gqa):
    """(flash SDPA enabled, SDPBackend SDPA would pick) for these shapes. Both reads break a dynamo graph."""
    cuda = torch.backends.cuda
    backends = (
        cuda.flash_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        tuple(getattr(torch._C, "_get_sdp_priority_order", tuple)()),
    )
    choice = _sdpa_choice(
        device, dtype, head_dim, n_heads, n_kv_heads, is_causal, enable_gqa, backends
    )
    return backends[0], choice


def _constant_probe(fn):
    """torch.compiler.assume_constant_result for a shape probe. Dynamo can only hand it Python
    constants, and dynamic=True makes head counts / head_dim SymInts, so those are specialised
    (operator.index guards them) first: they are fixed per model, batch and length stay dynamic.
    Plain ints pass through unchanged, so eager and static-shape answers are the same."""
    if not hasattr(torch.compiler, "assume_constant_result"):
        return fn
    # Marks fn itself, so no functools.wraps: copying its __dict__ would mark the wrapper too.
    constant = torch.compiler.assume_constant_result(fn)

    def probe(*args, **kwargs):
        concrete = []
        for arg in args:
            # Dynamo reports a SymInt as an int; index() on a plain int is the identity.
            if isinstance(arg, (int, torch.SymInt)) and not isinstance(arg, bool):
                arg = operator.index(arg)
            concrete.append(arg)
        return constant(*concrete, **kwargs)

    probe.__name__, probe.__qualname__, probe.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
    probe.__wrapped__ = fn
    return probe


_sdpa_backend_state = _constant_probe(_sdpa_backend_state)
_sdpa_masked_choice = _constant_probe(_sdpa_masked_choice)


@functools.lru_cache(maxsize = 128)
def _sdpa_gqa_choices(
    device,
    dtype,
    head_dim,
    n_heads,
    n_kv_heads,
    is_causal,
    has_mask,
    needs_grad,
    dropout_p,
    backends,
    single_query = False,
    single_key = False,
):
    """(SDPBackend with enable_gqa, SDPBackend with K/V expanded) for these shapes, -1 if unknown.
    The probe tensors carry the real call's requires_grad, dropout and single-query (decode) shape:
    backends differ on all three (cuDNN rejects single-query dropout, for one)."""
    # Lengths of 1 change eligibility (cuDNN rejects a K/V length of 1); longer ones do not.
    q_len = 1 if single_query else 128
    kv_len = 1 if single_key else 128
    try:
        q = torch.empty(1, n_heads, q_len, head_dim, device = device, dtype = dtype)
        k = torch.empty(1, n_kv_heads, kv_len, head_dim, device = device, dtype = dtype)
        k_full = torch.empty(1, n_heads, kv_len, head_dim, device = device, dtype = dtype)
        if needs_grad:
            q, k, k_full = (t.requires_grad_() for t in (q, k, k_full))
        mask = (
            torch.ones(1, 1, q_len, kv_len, device = device, dtype = torch.bool) if has_mask else None
        )
    except Exception:
        return -1, -1
    causal = is_causal and not has_mask

    def probe(key, **kwargs):
        # Each on its own: torch raises when no enabled backend fits, and a GQA call nothing
        # takes must still learn whether the expanded call would run.
        try:
            with torch.set_grad_enabled(needs_grad):
                return int(
                    torch._fused_sdp_choice(
                        q, key, key, attn_mask = mask, dropout_p = dropout_p, is_causal = causal, **kwargs
                    )
                )
        except Exception:
            return None

    gqa = probe(k, enable_gqa = True)
    mha = probe(k_full)
    if gqa is None:
        # Nothing takes GQA here: expand when the expanded call has a backend, else unknown.
        return (_SDPA_NO_BACKEND, mha) if mha is not None else (-1, -1)
    return gqa, -1 if mha is None else mha


# _sdpa_gqa_choices' answer when no enabled SDPA backend accepts the GQA call at all.
_SDPA_NO_BACKEND = -2


def _sdpa_gqa_state_now(
    device,
    dtype,
    head_dim,
    n_heads,
    n_kv_heads,
    is_causal,
    has_mask,
    needs_grad = False,
    dropout_p = 0.0,
    single_query = False,
    single_key = False,
) -> bool:
    """False when enable_gqa would leave SDPA only its math kernel while expanded K/V would not.
    Flash and cuDNN take GQA, the memory-efficient kernel does not (before torch 2.14), so fp32,
    pre-sm80 GPUs and masked calls flash cannot run would otherwise fall to math: several times
    slower and a (T, T) score matrix per head. Asked of torch, so ROCm's backend set is covered too."""
    cuda = torch.backends.cuda
    backends = (
        cuda.flash_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        tuple(getattr(torch._C, "_get_sdp_priority_order", tuple)()),
        # Strict deterministic mode makes cuDNN SDPA ineligible.
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    gqa, mha = _sdpa_gqa_choices(
        device,
        dtype,
        head_dim,
        n_heads,
        n_kv_heads,
        is_causal,
        has_mask,
        needs_grad,
        dropout_p,
        backends,
        single_query,
        single_key,
    )
    from torch.nn.attention import SDPBackend

    math = int(SDPBackend.MATH)
    if gqa == _SDPA_NO_BACKEND:
        return False
    return not (gqa == math and mha not in (math, -1))


def _sdpa_gqa_state(*args, no_cudnn = False):
    """_sdpa_gqa_state_now; no_cudnn probes under the kernels the call runs with (_training_sdpa,
    _rerouted_sdpa), whose enabled flags are part of the probe's cache key."""
    if no_cudnn:
        with _sdpa_kernel(_XFORMERS_LIKE_BACKENDS):
            return _sdpa_gqa_state_now(*args)
    return _sdpa_gqa_state_now(*args)


_sdpa_gqa_state = _constant_probe(_sdpa_gqa_state)


def _sdpa_takes_gqa(
    Q: Tensor,
    K: Tensor,
    is_causal: bool,
    has_mask: bool,
    V: Optional[Tensor] = None,
    dropout_p = 0.0,
    no_cudnn = False,
) -> bool:
    """Whether to pass enable_gqa (True) or expand K/V (False) for this SDPA call. no_cudnn: the
    call runs without cuDNN attention (SM100 training, xformers stand-in), so the probe does too."""
    if not SDPA_HAS_GQA:
        return False
    if not Q.is_cuda:
        return True
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (Q, K, V)
    )
    # Branch, not bool(): assume_constant_result cannot take a dynamic SymBool.
    single_query = True if Q.shape[-2] == 1 else False
    single_key = True if K.shape[-2] == 1 else False
    is_causal = True if is_causal else False
    has_mask = True if has_mask else False
    args = (
        str(Q.device),
        Q.dtype,
        Q.shape[-1],
        Q.shape[1],
        K.shape[1],
        bool(is_causal),
        bool(has_mask),
        bool(needs_grad),
        float(dropout_p or 0.0),
        single_query,
        single_key,
    )
    if no_cudnn:
        return _sdpa_gqa_state(*args, no_cudnn = True)
    return _sdpa_gqa_state(*args)


def _torch_varlen_beats_segments(
    Q: Tensor, K: Tensor, lengths: Tuple[int, ...], n_groups: int, is_causal: bool
) -> bool:
    """Varlen saves one SDPA launch per run of equal lengths, but its flash kernel is slower per token
    than cuDNN on long segments; with flash already chosen, the kernel is the same and only launches drop."""
    lengths = [length for length in lengths if length > 0]
    # Lengths overrunning the row would send cu_seqlens past the tensor; leave them to the old path's error.
    if sum(lengths) != Q.shape[-2]:
        return False
    runs = sum(1 for i, length in enumerate(lengths) if i == 0 or lengths[i - 1] != length)
    if runs <= 1:
        return False
    flash, choice = _sdpa_backend_state(
        str(Q.device),
        Q.dtype,
        Q.shape[-1],
        Q.shape[1],
        K.shape[1],
        is_causal,
        n_groups != 1 and _sdpa_flash_takes_gqa(Q),
    )
    # varlen_attn calls flash directly: honour a disabled flash backend and math / efficient-first priorities.
    from torch.nn.attention import SDPBackend

    if not flash:
        return False
    if choice == int(SDPBackend.FLASH_ATTENTION):
        return True
    if choice == int(SDPBackend.CUDNN_ATTENTION):
        if torch.cuda.get_device_capability(Q.device)[0] not in _VARLEN_CUDNN_ARCH_MAJORS:
            return True
        return sum(lengths) / len(lengths) <= _VARLEN_CUDNN_MAX_MEAN_LEN
    return False


def _torch_varlen_packed(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    seq_info,
    *,
    n_groups: int,
    is_causal: bool,
    sliding_window: Optional[int],
    scale: Optional[float],
) -> Tensor:
    """Q (1, H, T, D), K / V (1, H_kv, T, D) -> (1, T, H, D) over the row's segments, pad tail included."""
    total = Q.shape[-2]
    cu_seqlens, max_seqlen = cover_padded_cu_seqlens(seq_info, total)
    # seq_info may live on the CPU.
    cu_seqlens = cu_seqlens.to(device = Q.device, dtype = torch.int32, non_blocking = True)
    if not is_causal:
        window_size = (-1, -1)
    elif sliding_window is not None:
        window_size = (sliding_window - 1, 0)  # packed_block_mask keeps k >= q - (window - 1)
    else:
        window_size = (-1, 0)
    # squeeze, not [0]: select's backward allocates and zero-fills a full gradient per tensor.
    Q_f, K_f, V_f = (x.squeeze(0).transpose(0, 1).contiguous() for x in (Q, K, V))
    out = _varlen_attn_call(
        Q_f,
        K_f,
        V_f,
        cu_seqlens,
        max_seqlen,
        scale = scale,
        window_size = window_size,
        enable_gqa = n_groups != 1,
    )
    return out.unsqueeze(0)


def _sdpa_flash_takes_gqa(Q: Tensor) -> bool:
    # Only flash / math take enable_gqa: expand K/V elsewhere (pre-sm80, fp32, wide heads, ROCm).
    return (
        SDPA_HAS_GQA
        and Q.is_cuda
        and torch.version.hip is None
        and Q.dtype in (torch.float16, torch.bfloat16)
        and Q.shape[-1] <= 256
        and Q.shape[-1] % 8 == 0
        and torch.cuda.get_device_capability(Q.device)[0] >= 8
    )


def _sdpa_packed_segments(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    lengths: Tuple[int, ...],
    *,
    n_groups: int,
    is_causal: bool,
    sliding_window: Optional[int],
    sdpa_kwargs: dict,
    training: bool = False,
) -> Optional[Tensor]:
    """Q (1, H, T, D), K / V (1, H_kv, T, D) -> (1, T, H, D), the block-diagonal mask's result;
    None when the lengths do not tile the row."""
    if sum(length for length in lengths if length > 0) != Q.shape[-2]:
        return None
    expanded = None

    def kv(gqa):
        nonlocal expanded
        if n_groups == 1 or gqa:
            return K, V
        if expanded is None:
            expanded = (
                K.repeat_interleave(n_groups, dim = 1),
                V.repeat_interleave(n_groups, dim = 1),
            )
        return expanded

    outs, start, i = [], 0, 0
    lengths = [length for length in lengths if length > 0]
    while i < len(lengths):
        length = lengths[i]
        j = i
        while j < len(lengths) and lengths[j] == length:
            j += 1
        n = j - i
        end = start + n * length

        def split(x):
            return x[0, :, start:end].unflatten(1, (n, length)).transpose(0, 1)

        kwargs = dict(sdpa_kwargs)
        windowed = sliding_window is not None and length > sliding_window
        if windowed:
            kwargs["attn_mask"] = packed_block_mask(
                length,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
                is_causal = is_causal,
            )
        kwargs["is_causal"] = is_causal and not windowed
        Q_run = split(Q)
        # Probe the run's own tensors: a one-token run has different eligibility than the row.
        gqa = n_groups != 1 and _sdpa_takes_gqa(
            Q_run,
            split(K),
            kwargs["is_causal"],
            windowed,
            split(V),
            kwargs.get("dropout_p", 0.0),
            # Passed only when set: callers and tests wrap the probe with the old signature.
            **({"no_cudnn": True} if _avoids_cudnn(Q_run, training) else {}),
        )
        if gqa:
            kwargs["enable_gqa"] = True
        K_run, V_run = kv(gqa)
        out = (_training_sdpa if training else _inference_sdpa)(
            Q_run, split(K_run), split(V_run), **kwargs
        )
        outs.append(out.transpose(1, 2).flatten(0, 1))
        start, i = end, j
    return torch.cat(outs, dim = 0).unsqueeze(0)


# PrefixGrouper kernel, resolved once when the env gate is on so PG-off users never load torch flex_attention.
_flex_shared_prefix_attention = None
if os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in ("0", "false", "no", "off"):
    try:
        from .prefix_grouper_kernel import (
            flex_shared_prefix_attention as _flex_shared_prefix_attention,
        )
    except Exception:
        _flex_shared_prefix_attention = None

FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"


XFORMERS_BLOCK_DIAG_CLS = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None


# flash-attn 2 varlen backward indexes dq_accum with int32: at 2**31 elements it faults and poisons
# the CUDA context. Forward-only never allocates it.
_INT32_ELEMENTS = 2**31
_VARLEN_INT32_GUARD_DISABLED = os.environ.get(
    "UNSLOTH_DISABLE_VARLEN_INT32_GUARD", "0"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_VARLEN_INT32_WARNED = [False]


def _varlen_backward_dq_accum_elements(
    n_seqs: int, total_q: int, n_heads: int, head_dim: int
) -> int:
    """Element count of flash-attn 2's varlen-backward ``dq_accum`` for these shapes."""
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    return (total_q + 128 * n_seqs) * n_heads * head_dim_rounded


def _varlen_backward_overflows_int32(
    n_seqs: int, total_q: int, n_heads: int, head_dim: int
) -> bool:
    if n_seqs <= 0:
        return False
    return _varlen_backward_dq_accum_elements(n_seqs, total_q, n_heads, head_dim) >= _INT32_ELEMENTS


def _configured_softcap(config) -> Optional[float]:
    """The attention logit softcap this layer asked the fast kernels for, if any.

    Only flash and the xformers ops take a `softcap`; the SDPA branch below has no way to
    apply one. Gemma 2 passes it exclusively through these kwargs
    (`unsloth/models/gemma2.py`), so a backend swap that ignores them would train on
    uncapped logits.
    """
    for kwargs in (
        config.flash_varlen_kwargs,
        config.flash_dense_kwargs,
        config.xformers_kwargs,
    ):
        if not kwargs:
            continue
        softcap = kwargs.get("softcap")
        if softcap:
            return softcap
    return None


def _warn_varlen_int32_overflow_once(backend: str, n_seqs: int, total_q: int, elements: int):
    if _VARLEN_INT32_WARNED[0]:
        return
    _VARLEN_INT32_WARNED[0] = True
    print(
        f"Unsloth: A packed row holds {n_seqs} documents over {total_q} tokens. The "
        f"{backend} backward kernel would index a {elements:,}-element buffer with int32 "
        f"(limit {_INT32_ELEMENTS:,}), which faults with 'CUDA error: an illegal memory "
        "access was encountered' and poisons the CUDA context for the rest of the process.\n"
        "Unsloth has fallen back to SDPA for these batches, which is CORRECT but far slower "
        "and far heavier: SDPA's packed path materialises a dense mask and can use ~10x the "
        "memory, so it may OOM instead.\n"
        "To keep the fast kernel, pack fewer documents per row -- lower the packing length, or "
        "filter out very short documents so one row cannot collect thousands of them."
    )


@dataclass
class AttentionConfig:
    """
    Per-layer attention metadata.

    NOTE(djsaunde): Constructed on every forward pass (not once per layer) since
        it can be invalid across passes (e.g. switching training/inference). Kept
        separate from AttentionContext to group params.
    """

    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AttentionContext:
    """Per-call info required to run attention."""

    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None
    # PrefixGrouper: non-None routes Q/K/V via the FlexAttention shared-prefix kernel; None is unchanged
    prefix_seg_info: Optional[Any] = None
    is_causal: bool = True
    # The attention module's .training: also true for a checkpointed forward run without grad.
    training: bool = False


def _ring_attention_active() -> bool:
    manager = get_cp_manager()
    return manager is not None and manager.backend == "ring"


def _native_beats_xformers(capabilities, varlen_loaded: bool) -> bool:
    """Without FlashAttention, torch's own kernels match or beat xformers 0.0.35 (which itself calls
    torch's flash / memory-efficient ops) once torch has a varlen kernel (2.12+) on sm_80+ CUDA:
    packed / padding-free 1.00x to 1.06x on A100, L4, RTX PRO 6000 and H100. Below sm_80 there is no
    torch varlen, and xformers' block-diagonal kernel stays 8% to 21% faster (T4); a mixed host keeps
    xformers for its older card."""
    if not varlen_loaded or not capabilities or torch.version.hip is not None:
        return False
    return all(capability[0] >= 8 for capability in capabilities)


_NATIVE_BEATS_XFORMERS = _native_beats_xformers(
    _visible_capabilities(), _TORCH_VARLEN_ATTN is not None
)


def select_attention_backend(use_varlen: bool = False, dropout_p: float = 0.0) -> str:
    """Return attention backend based on availability / priority order."""

    if _ring_attention_active():
        if use_varlen:
            raise ValueError(
                "Unsloth: ring context parallelism does not support packing / padding-free; "
                "use context_parallel_backend = 'ulysses'."
            )
        return SDPA
    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    # Only the packed route prefers torch's kernels (varlen / per-segment SDPA). Dense calls keep
    # xformers: it alone consumes a caller's block-diagonal causal_mask, and expanding a GQA KV cache
    # for SDPA on fp32 decode costs memory. With UNSLOTH_SDPA_PACKED_SEGMENTS off, packed SDPA would
    # build a dense (T, T) mask, so xformers stays there too. Torch varlen takes no dropout, so a
    # packed call with dropout would become one SDPA launch per run: xformers keeps those as well.
    if HAS_XFORMERS and not (
        use_varlen and not dropout_p and _NATIVE_BEATS_XFORMERS and _SDPA_PACKED_SEGMENTS
    ):
        return XFORMERS
    return SDPA


def _packed_fp32_prefers_xformers(config, context, Q: Tensor, K: Tensor, V: Tensor) -> bool:
    """Torch varlen takes fp16 / bf16 only, so packed fp32 would run SDPA per segment (#14158)."""
    if not (HAS_XFORMERS and _NATIVE_BEATS_XFORMERS and _SDPA_PACKED_SEGMENTS):
        return False
    if context.seq_info is None or context.attention_mask is not None or not Q.is_cuda:
        return False
    if torch.float32 not in (Q.dtype, K.dtype, V.dtype):
        return False
    if (config.sdpa_kwargs or {}).get("dropout_p") or _ring_attention_active():
        return False
    # xformers' flash backward is non-deterministic, and sm_100+ (downcast to it) lacks head_dim > 256.
    if torch.are_deterministic_algorithms_enabled():
        return False
    if Q.shape[-1] > 256 and _xformers_rejects_fp32(Q.device):
        return False
    return True


def resolve_prefix_seg_info(kwargs, past_key_value, attention_mask):
    """PrefixGrouper shared-prefix segment table resolver for the arch attention forwards.

    The GRPO PrefixGrouper packed path rides a ``PrefixSegInfo`` in through ``**kwargs``
    (same route as ``packed_seq_lengths``). When present, the forward must route Q/K/V
    through the FlexAttention shared-prefix kernel via ``AttentionContext.prefix_seg_info``.

    Returns the seg table (or ``None`` when PrefixGrouper did not group this batch -- the
    unchanged path). Hardened: the shared-prefix stream is NOT a plain causal sequence, so running
    it under a KV cache or an explicit padding mask would silently produce wrong logprobs.
    That combination can only arise from misuse (PrefixGrouper only rides in via the GRPO
    logprob forward, which is mask-free prefill), so we RAISE loudly instead of degrading
    to a wrong result.

    Factored here so every arch (llama/mistral/qwen3/gemma2/cohere/granite/falcon_h1)
    shares one implementation and cannot drift.
    """
    seg = kwargs.get("prefix_seg_info", None)
    if seg is not None and (past_key_value is not None or attention_mask is not None):
        raise RuntimeError(
            "PrefixGrouper: prefix_seg_info requires prefill with no KV cache and no "
            f"attention_mask (got past_key_value={past_key_value is not None}, "
            f"attention_mask={attention_mask is not None})."
        )
    return seg


# One dense window mask per (device, shape, window) shared by all layers: at 32K it is 1 GiB.
_WINDOW_MASK_CACHE: dict = {}


def _windowed_causal_mask(q_len: int, k_len: int, sliding_window: int, device) -> Tensor:
    """Causal band mask of shape (1, 1, q_len, k_len). Read-only: callers must not mutate it."""
    params = (q_len, k_len, sliding_window)
    entry = _WINDOW_MASK_CACHE.get(device)
    if (
        entry is not None
        and entry["params"] == params
        # A mask built under inference_mode cannot be saved for a training step's backward.
        and (not entry["inference"] or _inference_mode())
    ):
        return entry["mask"]
    # Drop the outgoing mask first so a shape change does not peak a whole mask higher.
    _WINDOW_MASK_CACHE.pop(device, None)
    entry = None
    q_pos = torch.arange(k_len - q_len, k_len, device = device)
    k_pos = torch.arange(k_len, device = device)
    mask = (
        (k_pos[None, :] <= q_pos[:, None])
        & (k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1)))
    )[None, None, :, :]
    _WINDOW_MASK_CACHE[device] = {"params": params, "mask": mask, "inference": _inference_mode()}
    return mask


def run_attention(
    *, config: AttentionConfig, context: AttentionContext, Q: Tensor, K: Tensor, V: Tensor
) -> Tensor:
    """
    Run attention using config / context info.

    Backend priority (speed): FlashAttention if installed (varlen for packed
    inputs with `seq_info`, else dense), then xFormers, then SDPA as fallback.
    Varlen flash is preferred for packed batches as it avoids padding; xFormers
    and SDPA handle packing via a block-diagonal mask.
    """
    manager = get_cp_manager()
    if manager is not None and manager.backend == "ulysses":
        return ulysses_attention(_run_attention, config, context, Q, K, V, manager.group)
    return _run_attention(config = config, context = context, Q = Q, K = K, V = V)


def _run_attention(
    *, config: AttentionConfig, context: AttentionContext, Q: Tensor, K: Tensor, V: Tensor
) -> Tensor:
    # PrefixGrouper shared-prefix attention (GRPO dedup): Q/K/V here are [bsz, H, T, D] while the kernel
    # takes and returns [1, T, H, D], matching the other backends.
    if context.prefix_seg_info is not None:
        flex_shared_prefix_attention = _flex_shared_prefix_attention
        if flex_shared_prefix_attention is None:
            # Gate flipped on after import, or a one-time load failed: resolve lazily.
            from ..utils.prefix_grouper_kernel import flex_shared_prefix_attention

        scale = None
        if config.flash_varlen_kwargs:
            scale = config.flash_varlen_kwargs.get("softmax_scale")
        A = flex_shared_prefix_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            context.prefix_seg_info,
            scale = scale,
        )
        return A  # [1, T, n_heads, head_dim]

    backend = config.backend
    if backend == FLASH_VARLEN and context.seq_info is None:
        backend = FLASH_DENSE if HAS_FLASH_ATTENTION else SDPA
    xformers_fp32_unsupported = _xformers_rejects_fp32(Q.device)
    if backend == SDPA and _packed_fp32_prefers_xformers(config, context, Q, K, V):
        backend = XFORMERS
    if _DEVICE_BACKENDS is not None and backend != SDPA:
        device_backend = _backend_for_device(backend, Q.device, Q.dtype)
        if device_backend != backend and _configured_softcap(config):
            raise RuntimeError(
                f"Unsloth: This model softcaps attention logits, which needs Flash Attention, but "
                f"{Q.device} cannot run it (compute capability below 8.0). Place the model on "
                "Ampere or newer GPUs, e.g. with CUDA_VISIBLE_DEVICES."
            )
        backend = device_backend

    # Flash/xFormers do not support arbitrary attn masks, so a padding mask falls back to SDPA.
    if context.attention_mask is not None and backend in (
        FLASH_DENSE,
        FLASH_VARLEN,
        XFORMERS,
    ):
        backend = SDPA

    # Both varlen backends hit the flash-attn 2 backward kernel, so guard both.
    if backend in (FLASH_VARLEN, XFORMERS) and not _VARLEN_INT32_GUARD_DISABLED:
        # LoRA q/k/v yield a grad-requiring Q from frozen inputs; checkpointing's first forward runs under
        # no_grad, so tensors alone would pick a different backend per pass.
        will_backward = context.requires_grad or (
            torch.is_grad_enabled() and (Q.requires_grad or K.requires_grad or V.requires_grad)
        )
        if will_backward:
            seq_info = context.seq_info
            total_q = context.bsz * context.q_len
            # Includes the collator's trailing pad segment.
            n_seqs = (
                cover_padded_cu_seqlens(seq_info, total_q)[0].numel() - 1
                if seq_info is not None
                else context.bsz
            )
            if _varlen_backward_overflows_int32(n_seqs, total_q, context.n_heads, context.head_dim):
                # SDPA cannot apply logit softcapping, so rerouting Gemma 2 would train on wrong logits.
                softcap = _configured_softcap(config)
                if softcap:
                    raise RuntimeError(
                        f"Unsloth: A packed row holds {n_seqs} documents over {total_q} "
                        f"tokens, so the {backend} backward kernel would index a "
                        f"{_varlen_backward_dq_accum_elements(n_seqs, total_q, context.n_heads, context.head_dim):,}"
                        f"-element buffer with int32 (limit {_INT32_ELEMENTS:,}) and fault "
                        "with 'CUDA error: an illegal memory access was encountered'.\n"
                        f"This model softcaps attention logits (softcap={softcap}), and the "
                        "SDPA fallback cannot reproduce that, so falling back would silently "
                        "train on wrong logits.\n"
                        "Pack fewer documents per row: lower the packing length, or filter "
                        "out very short documents so one row cannot collect thousands of them."
                    )
                _warn_varlen_int32_overflow_once(
                    backend,
                    n_seqs,
                    total_q,
                    _varlen_backward_dq_accum_elements(
                        n_seqs, total_q, context.n_heads, context.head_dim
                    ),
                )
                backend = SDPA

    # Traced xformers bias objects break the graph per layer: plain causal calls take SDPA.
    xformers_rerouted = False
    if (
        backend == XFORMERS
        and torch.compiler.is_compiling()
        and context.seq_info is None
        and context.attention_mask is None
        and context.is_causal
        and context.q_len == context.kv_seq_len
        and (
            context.causal_mask is None
            or type(context.causal_mask).__name__ == "LowerTriangularMask"
        )
        and not config.xformers_kwargs
        and _configured_softcap(config) is None
    ):
        backend = SDPA
        xformers_rerouted = True
    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}
    # Mask builders only see is_causal when False, so causal callers and their patches are unchanged.
    direction = {}
    if not context.is_causal:
        flash_dense_kwargs = {**flash_dense_kwargs, "causal": False}
        flash_varlen_kwargs = {**flash_varlen_kwargs, "causal": False}
        direction = {"is_causal": False}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    # Eval forwards can backpropagate (LoRA-GA calibration); xformers' inference GQA layout cannot.
    requires_grad = context.requires_grad or (
        torch.is_grad_enabled() and (Q.requires_grad or K.requires_grad or V.requires_grad)
    )
    sliding_window = context.sliding_window
    # A non-positive window means no local attention.
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None

    # DoRA promotes q/k/v_proj outputs to fp32, which FlashAttention rejects (as does the xformers
    # flash-2 op on sm_100+), so downcast any fp32 Q/K/V to a supported dtype (#1013).
    # Ring attention (context parallelism) likewise needs a flash-eligible dtype.
    ring = backend == SDPA and _ring_attention_active()
    cp_out_dtype = Q.dtype if ring else None
    if (
        backend in (FLASH_DENSE, FLASH_VARLEN)
        or ((backend == XFORMERS or xformers_rerouted) and xformers_fp32_unsupported)
        or ring
    ) and torch.float32 in (
        Q.dtype,
        K.dtype,
        V.dtype,
    ):
        # Prefer the autocast dtype, else a non-fp32 input's dtype, then clamp.
        if torch.is_autocast_enabled():
            try:
                downcast_dtype = torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):
                downcast_dtype = torch.get_autocast_gpu_dtype()
        else:
            downcast_dtype = next(
                (d for d in (Q.dtype, K.dtype, V.dtype) if d != torch.float32), None
            )
        if downcast_dtype not in (torch.float16, torch.bfloat16):
            downcast_dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
        Q, K, V = Q.to(downcast_dtype), K.to(downcast_dtype), V.to(downcast_dtype)

    if backend == FLASH_VARLEN:
        Q_f = Q.transpose(1, 2).reshape(bsz * q_len, n_heads, head_dim)
        K_f = K.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        V_f = V.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        cu_seqlens, max_seqlen = cover_padded_cu_seqlens(context.seq_info, bsz * q_len)
        return flash_attn_varlen_func(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **flash_varlen_kwargs,
        ).view(bsz, q_len, n_heads, head_dim)
    elif backend == FLASH_DENSE:
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        return flash_attn_func(Q_t, K_t, V_t, **flash_dense_kwargs).reshape(
            bsz, q_len, n_heads, head_dim
        )
    elif backend == XFORMERS:
        base_mask = context.causal_mask
        # Only CausalLM_fast_forward supplies the mask; a direct decoder call (Liger, TRL's get_decoder paths) would attend bidirectionally.
        if (
            base_mask is None
            and context.is_causal
            and xformers is not None
            and context.seq_info is None
            and q_len == kv_seq_len
        ):
            base_mask = xformers.attn_bias.LowerTriangularMask()
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window = sliding_window,
            base_mask = base_mask,
            total_tokens = K.shape[-2],
            **direction,
        )
        attn_bias = move_xformers_attention_bias(attn_bias, Q.device)

        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        K_mod = K_t
        V_mod = V_t
        Q_mod = Q_t

        if config.n_groups != 1:
            K_mod = K_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            V_mod = V_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            K_mod = K_mod.expand(bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim)
            V_mod = V_mod.expand(bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim)

            if requires_grad:
                K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q_mod = Q_t.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)

        has_block = XFORMERS_BLOCK_DIAG_CLS is not None and isinstance(
            attn_bias, XFORMERS_BLOCK_DIAG_CLS
        )

        if config.n_groups != 1 and has_block:
            if not requires_grad:
                Q_mod = Q_mod.view(1, bsz * q_len, config.n_kv_heads, config.n_groups, head_dim)
                K_mod = K_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
                V_mod = V_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
            else:
                Q_mod = Q_mod.view(1, bsz * q_len, n_heads, head_dim)
                K_mod = K_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)

        out = xformers_attention(
            Q_mod,
            K_mod,
            V_mod,
            attn_bias = attn_bias,
            **xformers_kwargs,
        )

        if config.n_groups != 1 and not requires_grad:
            out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            out = out.reshape(bsz, q_len, n_heads, head_dim)
        else:
            out = out.view(bsz, q_len, n_heads, head_dim)
        return out
    else:
        local_mask = context.attention_mask
        is_causal_local = False
        if (
            _SDPA_PACKED_SEGMENTS
            and context.seq_info is not None
            and local_mask is None
            and Q.shape[0] == 1
            and Q.shape[-2] == K.shape[-2]
        ):
            lengths = packed_segment_lengths(context.seq_info, K.shape[-2])
            if (
                _torch_varlen_takes(Q, K, V, sdpa_kwargs, context.is_causal, sliding_window)
                and _torch_varlen_beats_segments(Q, K, lengths, config.n_groups, context.is_causal)
            ) and not (
                # Varlen backward may share flash-attn 2's int32 dq_accum limit.
                requires_grad
                and not _VARLEN_INT32_GUARD_DISABLED
                and _varlen_backward_overflows_int32(
                    cover_padded_cu_seqlens(context.seq_info, Q.shape[-2])[0].numel() - 1,
                    Q.shape[-2],
                    n_heads,
                    head_dim,
                )
            ):
                return _torch_varlen_packed(
                    Q,
                    K,
                    V,
                    context.seq_info,
                    n_groups = config.n_groups,
                    is_causal = context.is_causal,
                    sliding_window = sliding_window,
                    scale = sdpa_kwargs.get("scale"),
                )
            out = _sdpa_packed_segments(
                Q,
                K,
                V,
                lengths,
                n_groups = config.n_groups,
                is_causal = context.is_causal,
                sliding_window = sliding_window,
                sdpa_kwargs = sdpa_kwargs,
                training = context.training,
            )
            if out is not None:
                return out
        if context.seq_info is not None and local_mask is None:
            local_mask = build_sdpa_packed_attention_mask(
                context.seq_info,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
                total_tokens = K.shape[-2],
                **direction,
            )
        else:
            q_len_local = Q.shape[-2]
            k_len_local = K.shape[-2]
            if local_mask is not None and isinstance(local_mask, torch.Tensor):
                local_mask = local_mask.to(device = Q.device)

                if local_mask.dim() == 2:
                    # Key padding keep mask (bsz, k_len), 1/True is a real token.
                    if local_mask.dtype == torch.bool:
                        key_keep = local_mask
                    else:
                        # tokenizer attention_mask is typically int 0/1
                        key_keep = local_mask != 0

                    past_len = k_len_local - q_len_local  # works for prefill (0) and decode
                    q_pos = torch.arange(past_len, past_len + q_len_local, device = Q.device)
                    k_pos = torch.arange(k_len_local, device = Q.device)

                    causal_keep = (
                        k_pos[None, :] <= q_pos[:, None]
                        if context.is_causal
                        else torch.ones(
                            (q_len_local, k_len_local), dtype = torch.bool, device = Q.device
                        )
                    )  # True = allowed (SDPA)
                    if sliding_window is not None:
                        causal_keep &= k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1))

                    # (bsz, 1, q_len, k_len) boolean keep mask.
                    local_mask = causal_keep[None, None, :, :] & key_keep[:, None, None, :]

                elif local_mask.dim() == 3:
                    # (bsz, q_len, k_len) -> (bsz, 1, q_len, k_len)
                    local_mask = local_mask[:, None, :, :]

                elif local_mask.dim() == 4:
                    if local_mask.dtype != torch.bool:
                        # Use boolean keep masks for better SDPA stability.
                        local_mask = local_mask.eq(0)
                else:
                    raise ValueError(f"Unsupported SDPA attention_mask rank: {local_mask.dim()}")

                # Avoid NaNs from fully-masked rows, common with left padding.
                if local_mask.dtype == torch.bool:
                    no_allowed = ~local_mask.any(dim = -1, keepdim = True)  # (bsz,1,q_len,1)
                    local_mask = local_mask | no_allowed

            if local_mask is None and sliding_window is not None and k_len_local > sliding_window:
                # SDPA's is_causal has no window, so build the windowed mask explicitly.
                local_mask = _windowed_causal_mask(
                    q_len_local, k_len_local, sliding_window, Q.device
                )

            is_causal_local = (
                True
                if context.is_causal and local_mask is None and q_len_local == k_len_local
                else False
            )

        kwargs = dict(sdpa_kwargs)
        sdpa = _training_sdpa if context.training else _inference_sdpa
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        use_sdpa_gqa = SDPA_HAS_GQA and config.n_groups != 1
        # Flash takes GQA natively, halving the ring's K/V peak; UNSLOTH_CP_GQA=0 expands.
        if (
            use_sdpa_gqa
            and _ring_attention_active()
            and os.environ.get("UNSLOTH_CP_GQA", "1") == "0"
        ):
            use_sdpa_gqa = False
        if (
            use_sdpa_gqa
            and (not requires_grad)
            and isinstance(local_mask, torch.Tensor)
            and local_mask.dim() >= 3
            and local_mask.shape[0] > 1
        ):
            # Batched masked inference drifts with SDPA GQA, so expand KV explicitly.
            use_sdpa_gqa = False

        if use_sdpa_gqa and not _sdpa_takes_gqa(
            Q,
            K,
            kwargs.get("is_causal", False),
            kwargs.get("attn_mask") is not None,
            V,
            kwargs.get("dropout_p", 0.0),
            **(
                {"no_cudnn": True}
                if xformers_rerouted or _avoids_cudnn(Q, context.training)
                else {}
            ),
        ):
            use_sdpa_gqa = False

        if use_sdpa_gqa:
            kwargs.setdefault("enable_gqa", True)
            out = (_rerouted_sdpa if xformers_rerouted else sdpa)(Q, K, V, **kwargs)
            out = out.transpose(1, 2)
            return out if cp_out_dtype is None else out.to(cp_out_dtype)

        K_mod = K
        V_mod = V
        if config.n_groups != 1:
            K_mod = K[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            V_mod = V[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            K_mod = K_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V_mod = V_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)

        out = (_rerouted_sdpa if xformers_rerouted else sdpa)(
            Q.contiguous(),
            K_mod.contiguous(),
            V_mod.contiguous(),
            **kwargs,
        )
        out = out.transpose(1, 2).contiguous()
        return out if cp_out_dtype is None else out.to(cp_out_dtype)


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "resolve_prefix_seg_info",
    "run_attention",
]

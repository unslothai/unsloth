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
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ..device_type import DEVICE_TYPE
from ..models._utils import *
from ..context_parallel import get_cp_manager
from ..utils.packing import (
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


# FlashAttention always wins in select_attention_backend and nothing downgrades flash to xformers,
# so when it is installed xformers is never selected: skip the probe.
if HAS_XFORMERS and not HAS_FLASH_ATTENTION and torch.cuda.is_available():
    if _xformers_disabled_for_capability(torch.cuda.get_device_capability()):
        HAS_XFORMERS = False

# On sm_100+ (B200, sm_120) xformers' fp32-capable cutlass op is capability-rejected and only its
# fp16/bf16 flash-2 op runs, so fp32 Q/K/V (DoRA, #1013) must be downcast there; below sm_100
# cutlass handles fp32 natively. Read once from device 0.
_XFORMERS_FP32_UNSUPPORTED = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)


def scaled_dot_product_attention(*args, **kwargs):
    # Resolved per call: context_parallel monkey patches torch.nn.functional.
    return F.scaled_dot_product_attention(*args, **kwargs)



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


# Mixed-generation hosts only (e.g. sm_86 + sm_75): the backend is re-checked against Q's device, since a
# device-0 verdict is wrong for layers a device_map places on another card.
_DEVICE_BACKENDS = _per_device_backends(_visible_capabilities())


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
_VARLEN_CUDNN_MAX_MEAN_LEN = 1024


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


if hasattr(torch.compiler, "assume_constant_result"):
    _sdpa_backend_state = torch.compiler.assume_constant_result(_sdpa_backend_state)


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
    Q_f, K_f, V_f = (x[0].transpose(0, 1).contiguous() for x in (Q, K, V))
    out = _TORCH_VARLEN_ATTN(
        Q_f,
        K_f,
        V_f,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
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
) -> Optional[Tensor]:
    """Q (1, H, T, D), K / V (1, H_kv, T, D) -> (1, T, H, D), the block-diagonal mask's result;
    None when the lengths do not tile the row."""
    if sum(length for length in lengths if length > 0) != Q.shape[-2]:
        return None
    gqa = n_groups != 1 and _sdpa_flash_takes_gqa(Q)
    if n_groups != 1 and not gqa:
        K = K.repeat_interleave(n_groups, dim = 1)
        V = V.repeat_interleave(n_groups, dim = 1)
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
        if gqa:
            kwargs["enable_gqa"] = True
        out = scaled_dot_product_attention(split(Q), split(K), split(V), **kwargs)
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


# flash-attn 2 varlen backward int32 overflow guard: the varlen BACKWARD kernel allocates
# dq_accum = zeros(total_q + 128 * n_seqs, n_heads, round_up(head_dim, 32)) and indexes it with
# int32, so at 2**31 elements the kernel faults with an illegal memory access and poisons the CUDA
# context. Forward-only never allocates dq_accum, so the guard requires a backward to be possible.
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
    # PrefixGrouper: non-None routes Q/K/V through the FlexAttention shared-prefix kernel; None leaves
    # every existing construction and behavior unchanged.
    prefix_seg_info: Optional[Any] = None
    is_causal: bool = True


def select_attention_backend(use_varlen: bool = False) -> str:
    """Return attention backend based on availability / priority order."""

    if get_cp_manager() is not None:
        if use_varlen:
            raise ValueError(
                "Unsloth: context parallelism does not support packing / padding-free."
            )
        return SDPA
    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    if HAS_XFORMERS:
        return XFORMERS
    return SDPA


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


# One dense window mask per (device, shape, window), reused across layers: every layer of a
# Mistral-style model asks for the identical mask, and at 32K that tensor is 1 GiB with two more
# alive while it is built, so rebuilding it 32 times OOMs the run.
_WINDOW_MASK_CACHE: dict = {}


def _windowed_causal_mask(q_len: int, k_len: int, sliding_window: int, device) -> Tensor:
    """Causal band mask of shape (1, 1, q_len, k_len). Read-only: callers must not mutate it."""
    params = (q_len, k_len, sliding_window)
    entry = _WINDOW_MASK_CACHE.get(device)
    if entry is not None and entry["params"] == params:
        return entry["mask"]
    # Drop the outgoing mask first: it is dead either way, and holding it while the replacement and its
    # temporaries are allocated would make a shape change peak a whole mask higher.
    _WINDOW_MASK_CACHE.pop(device, None)
    entry = None
    q_pos = torch.arange(k_len - q_len, k_len, device = device)
    k_pos = torch.arange(k_len, device = device)
    mask = (
        (k_pos[None, :] <= q_pos[:, None])
        & (k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1)))
    )[None, None, :, :]
    _WINDOW_MASK_CACHE[device] = {"params": params, "mask": mask}
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
    xformers_fp32_unsupported = _XFORMERS_FP32_UNSUPPORTED
    if _DEVICE_BACKENDS is not None and backend != SDPA:
        device_backend = _backend_for_device(backend, Q.device, Q.dtype)
        if device_backend != backend and _configured_softcap(config):
            raise RuntimeError(
                f"Unsloth: This model softcaps attention logits, which needs Flash Attention, but "
                f"{Q.device} cannot run it (compute capability below 8.0). Place the model on "
                "Ampere or newer GPUs, e.g. with CUDA_VISIBLE_DEVICES."
            )
        backend = device_backend
        index = Q.device.index
        if Q.device.type == "cuda" and index is not None and index < len(_DEVICE_BACKENDS):
            xformers_fp32_unsupported = _DEVICE_BACKENDS[index][2]

    # Flash/xFormers do not support arbitrary attn masks, so with a padding mask present (left-padded
    # generation) fall back to SDPA.
    if context.attention_mask is not None and backend in (
        FLASH_DENSE,
        FLASH_VARLEN,
        XFORMERS,
    ):
        backend = SDPA

    # Both varlen-capable backends land in the same flash-attn 2 backward kernel, so guard both before
    # the int32 overflow aborts the process. Integer arithmetic plus one cached sync per step.
    if backend in (FLASH_VARLEN, XFORMERS) and not _VARLEN_INT32_GUARD_DISABLED:
        # Both terms are needed. Q/K/V: a frozen hidden state feeding trainable LoRA q/k/v still yields a
        # Q that requires grad. context.requires_grad: gradient checkpointing runs its FIRST forward under
        # no_grad and recomputes with grad on, so keying only on the tensors would pick a different
        # backend per pass.
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
                # SDPA cannot apply logit softcapping, so rerouting a softcapped model (Gemma 2) would keep it
                # training on wrong logits and gradients, worse than the fault this guard avoids.
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
    # A non-positive window means "no local attention", not "a window of nothing": a config spelling
    # it 0 would put the mask's lower bound above its causal upper bound.
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None

    # DoRA promotes q/k/v_proj outputs to fp32, which FlashAttention rejects (as does the xformers
    # flash-2 op on sm_100+), so downcast any fp32 Q/K/V to a supported dtype (#1013).
    # Ring attention (context parallelism) likewise needs a flash-eligible dtype.
    cp_out_dtype = Q.dtype if backend == SDPA and get_cp_manager() is not None else None
    if (
        backend in (FLASH_DENSE, FLASH_VARLEN)
        or (backend == XFORMERS and xformers_fp32_unsupported)
        or (backend == SDPA and get_cp_manager() is not None)
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
                    # Key padding keep mask (bsz, k_len), where 1/True is a real token; the tokenizer attention_mask is
                    # typically int 0/1.
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
                # SDPA's is_causal is FULL causal and has no window, so with no padding mask to hang the window
                # off, a model whose config declares one attended its whole history.
                local_mask = _windowed_causal_mask(
                    q_len_local, k_len_local, sliding_window, Q.device
                )

            is_causal_local = (
                context.is_causal and local_mask is None and q_len_local == k_len_local
            )

        kwargs = dict(sdpa_kwargs)
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        use_sdpa_gqa = SDPA_HAS_GQA and config.n_groups != 1
        # CP refuses < sm80 and downcasts fp32 Q/K/V above, so flash takes GQA natively and the ring
        # gathers and accumulates n_kv_heads, not n_heads (about half the peak). UNSLOTH_CP_GQA=0 expands.
        if (
            use_sdpa_gqa
            and get_cp_manager() is not None
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
            # Batched masked inference has shown row-coupled drift with SDPA GQA, so fall back to explicit KV
            # expansion for deterministic row-wise behavior.
            use_sdpa_gqa = False

        if use_sdpa_gqa:
            kwargs.setdefault("enable_gqa", True)
            out = scaled_dot_product_attention(Q, K, V, **kwargs)
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

        out = scaled_dot_product_attention(
            Q.contiguous(),
            K_mod.contiguous(),
            V_mod.contiguous(),
            **kwargs,
        )
        out = out.transpose(1, 2).contiguous()
        # o_proj expects the pre-downcast dtype (an fp32 model would otherwise hit a dtype mismatch).
        return out if cp_out_dtype is None else out.to(cp_out_dtype)


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "resolve_prefix_seg_info",
    "run_attention",
]

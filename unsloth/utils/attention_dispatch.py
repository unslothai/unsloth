# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from ..models._utils import *
from ..models._utils import _announce_xformers_breakage  # not in __all__, needed by the probe gate
from ..utils.packing import (
    build_sdpa_packed_attention_mask,
    build_xformers_block_causal_mask,
)

flash_attn_func = None
flash_attn_varlen_func = None
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
HAS_XFORMERS = xformers is not None


# Why the on-device probe last failed, or None when it passed or never ran. Cached
# alongside the boolean so callers can report WHY xformers went away, not just that it did.
XFORMERS_PROBE_REASON: Optional[str] = None

# True when the probe failed for a reason that says nothing about the build: the GPU was
# busy, out of memory, or otherwise unavailable to us right now.
XFORMERS_PROBE_INCONCLUSIVE = False

# Failures that mean "ask again later", not "this xformers is broken". Disabling
# memory-efficient attention for the whole process because device 0 happened to be full,
# or was claimed by another rank in EXCLUSIVE_PROCESS mode, is a silent 2x memory
# regression caused by the diagnostic itself.
_INCONCLUSIVE_PROBE_ERRORS = (
    "out of memory",
    "busy or unavailable",
    "all cuda-capable devices are busy",
    "no cuda-capable device",
    "cuda_error_not_permitted",
    "insufficient driver",
    "initialization error",
    # Belt and braces for the device index. It is clamped below, so this should be
    # unreachable -- but if it ever is reached, "we aimed at a device that is not there"
    # must not be recorded as "your xformers is broken" and disable it process-wide.
    "invalid device ordinal",
    "invalid device id",
    # EXCLUSIVE_PROCESS wording that the phrases above do not cover. The driver says this
    # when another process holds the device; the wheel was never tested, so turning
    # xformers off process-wide on the strength of it is the same silent regression.
    "currently in use",
    "in use by another process",
    "exclusive",
)


# Which device to probe. Under torchrun each rank owns a different GPU, and on a mixed box
# device 0 is often the small display card, so probing 0 for everyone lets a wheel with no
# kernel for the weakest GPU disable xformers on the good ones.
#
# LOCAL_RANK is NOT an index into the devices this process can see. Slurm with
# --gpus-per-task=1, and anything that narrows CUDA_VISIBLE_DEVICES per rank, gives every
# rank one visible device while still exporting its global rank -- so rank 3 sees exactly
# one GPU and LOCAL_RANK says 3. accelerate and transformers also use -1 as their "not
# distributed" sentinel. Both are out of range, torch.cuda.get_device_capability raises on
# an invalid ordinal, and that call is at module scope, so an unclamped index turns
# `import unsloth` into a crash on an ordinary launch. Fall back to 0, which is the only
# device such a rank has.
#
# With no usable LOCAL_RANK, the device the CALLER already selected: a single-process
# application that runs torch.cuda.set_device(1) before importing us is telling us where its
# work goes, and probing 0 anyway can disable xformers over a card nothing will touch -- and
# creates a context on it while doing so.
def _resolve_probe_device_index() -> int:
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count <= 0:
        return 0
    try:
        rank = int(os.environ.get("LOCAL_RANK", "") or -1)
    except ValueError:
        rank = -1
    if 0 <= rank < count:
        return rank
    try:
        current = int(torch.cuda.current_device())
    except Exception:
        return 0
    return current if 0 <= current < count else 0


_PROBE_DEVICE_INDEX = _resolve_probe_device_index()


def _xformers_runs_on_device() -> bool:
    """One tiny attention forward; True iff the xformers kernel actually runs here.

    Never raises. Every failure becomes False plus a one-line XFORMERS_PROBE_REASON,
    because this runs at import and a diagnostic must not be what breaks the import.
    """
    global XFORMERS_PROBE_REASON, XFORMERS_PROBE_INCONCLUSIVE
    try:
        # Pre-Ampere GPUs (sm < 80: Turing/Volta) have no bfloat16 attention kernel
        # but run xformers fine in float16, so pick the dtype the device supports.
        #
        # Read off the device being PROBED, not the module-level SUPPORTS_BFLOAT16, which
        # describes device 0. On a mixed box where 0 is Ampere-or-newer and this rank owns
        # a Turing card, the global says bf16, the kernel rejects it, and a healthy
        # xformers is recorded as broken for the whole process.
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(_PROBE_DEVICE_INDEX)[0] >= 8
            else torch.float16
        )
        device = f"cuda:{_PROBE_DEVICE_INDEX}"
        # Under the device context, not just device= on the tensor. BlockDiagonalCausalMask
        # builds its seqstart tensors on the CURRENT device, and at import time that is
        # still cuda:0 on every rank -- launchers set LOCAL_RANK in the environment but
        # torch.cuda.set_device happens later, inside the trainer. So q lands on cuda:N and
        # the bias on cuda:0, xformers rejects the pair, and the probe fails on every rank
        # but zero. That is the silent drop to SDPA this whole gate exists to prevent, on a
        # healthy install, caused by the diagnostic itself -- and it allocates on cuda:0
        # from every rank as well, pinning a second context per rank.
        with torch.cuda.device(_PROBE_DEVICE_INDEX):
            q = torch.zeros((1, 8, 1, 64), device = device, dtype = dtype)
            attn_bias = xformers.attn_bias.BlockDiagonalCausalMask.from_seqlens([8])
            xformers_attention(q, q, q, attn_bias = attn_bias)
            # Launches are async; synchronize so a deferred kernel failure fails the probe here.
            torch.cuda.synchronize(device)
        XFORMERS_PROBE_REASON = None
        XFORMERS_PROBE_INCONCLUSIVE = False
        return True
    except Exception as error:
        XFORMERS_PROBE_REASON = f"{type(error).__name__}: {error}".strip()
        text = str(error).lower()
        XFORMERS_PROBE_INCONCLUSIVE = any(marker in text for marker in _INCONCLUSIVE_PROBE_ERRORS)
        return False


def _xformers_disabled(probe = _xformers_runs_on_device) -> bool:
    # Probe on EVERY capability, not just sm_120+. The old gate returned early below
    # sm_120 on the assumption that xformers always works there, which only holds when
    # the wheel matches the runtime: a cu128-built xformers on a cu130 torch is just as
    # dead on an sm_90 Hopper, and never probing there is what let a mismatched managed
    # Windows package ship (NVIDIA P0-1).
    #
    # At sm_120 (RTX 50-series) there is a second, unrelated reason to probe: xformers'
    # cutlass op is capability-rejected (it caps at sm_90) and its flash-2 op runs only
    # if the build ships an sm_120 kernel (unslothai/unsloth#4631).
    #
    # No capability argument: the answer is always the real op now, and reading the
    # capability at the call site put an UNGUARDED torch.cuda.get_device_capability() at
    # module scope. CUDA refuses that query when the device is busy, in exclusive mode or
    # temporarily unavailable, and there it raised before the probe could classify the
    # failure as inconclusive -- turning `import unsloth` into a crash over a diagnostic
    # whose worst answer is "keep xformers on and let the forward decide".
    return not probe()


# Probe whenever xformers imported, including when flash-attn is installed and will win
# select_attention_backend anyway: a dead xformers is worth knowing about either way, and
# reporting it is the point. Cost is one 1x8x1x64 forward on a CUDA context torch has
# already initialised (_XFORMERS_FP32_UNSUPPORTED below forces the same lazy init).
XFORMERS_DISABLED_REASON = XFORMERS_BROKEN_REASON
if HAS_XFORMERS and torch.cuda.is_available():
    if _xformers_disabled():
        if XFORMERS_PROBE_INCONCLUSIVE:
            # The GPU was busy or full, which says nothing about the build. Keep xformers
            # and let the real forward pass decide; turning it off here would be a silent
            # memory regression caused by the diagnostic.
            if UNSLOTH_ENABLE_LOGGING:
                print(
                    f"Unsloth: Could not probe xformers ({XFORMERS_PROBE_REASON}); keeping it on."
                )
        else:
            HAS_XFORMERS = False
            XFORMERS_DISABLED_REASON = XFORMERS_PROBE_REASON
            # Say so. A probe that turns off memory-efficient attention and prints nothing
            # is the same silent downgrade this whole change exists to remove.
            #
            # First line by default, the rest behind UNSLOTH_ENABLE_LOGGING. This reason is
            # a captured exception, and xformers answers a capability rejection with a dump
            # of every operator it considered and why -- a dozen lines. Announcing that
            # verbatim on the default path would put a wall of text in front of every user
            # of an affected card and bury the one sentence that matters. Truncating HERE
            # rather than in the announcer, because the announcer's other callers pass
            # deliberately multi-line, fenced, copy-pasteable instructions that have to
            # arrive intact.
            _probe_head, _, _probe_rest = str(XFORMERS_PROBE_REASON).strip().partition("\n")
            _announce_xformers_breakage(
                _probe_head,
                _probe_rest.strip() or None,
            )


# On sm_100+ (B200, sm_120) xformers' fp32-capable cutlass op is capability-rejected and
# only its fp16/bf16 flash-2 op runs, so fp32 Q/K/V (DoRA, #1013) must be downcast there;
# below sm_100 cutlass handles fp32 natively. Read once from the same device the probe gate
# above used, so the two answers describe the same GPU: on a mixed box, reading the fp32
# capability off device 0 while probing this rank's device is how a display card ends up
# deciding downcast policy for a compute card.
def _probe_device_major() -> Optional[int]:
    """Major compute capability of the probed device, or None if CUDA will not say.

    Guarded for the same reason the probe is: a busy or exclusive-mode device makes this
    query raise, and at module scope that is an import crash rather than a missing answer.
    """
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(_PROBE_DEVICE_INDEX)[0]
    except Exception:
        return None


# None (unknown) is treated as supported: downcasting fp32 that did not need it is a
# quality regression, and the fp16/bf16 paths are unaffected either way.
_XFORMERS_FP32_UNSUPPORTED = (_probe_device_major() or 0) >= 10
SDPA_HAS_GQA = "enable_gqa" in (scaled_dot_product_attention.__doc__ or "")

# PrefixGrouper kernel, resolved once when the env gate is on so PG-off users never load
# torch flex_attention.
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
    # PrefixGrouper: non-None routes Q/K/V through the FlexAttention shared-prefix kernel;
    # None leaves every existing construction/behavior unchanged.
    prefix_seg_info: Optional[Any] = None


def select_attention_backend(use_varlen: bool = False) -> str:
    """Return attention backend based on availability / priority order."""

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

    # PrefixGrouper shared-prefix attention (GRPO dedup). Q/K/V here are [bsz, H, T, D];
    # the kernel takes/returns [1, T, H, D], matching the other backends. The field is
    # only set when the env gate is on and grouping succeeded; None keeps every backend
    # byte-identical.
    if context.prefix_seg_info is not None:
        flex_shared_prefix_attention = _flex_shared_prefix_attention
        if flex_shared_prefix_attention is None:
            # gate flipped on after import (or one-time load failed): resolve lazily.
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

    # [TODO] Flash/xFormers don't support arbitrary attn masks; with a padding
    # mask present (e.g. left-padded generation), fall back to SDPA.
    if context.attention_mask is not None and backend in (
        FLASH_DENSE,
        FLASH_VARLEN,
        XFORMERS,
    ):
        backend = SDPA

    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    requires_grad = context.requires_grad
    sliding_window = context.sliding_window
    # A non-positive window means "no local attention", not "a window of nothing": a config
    # spelling it 0 would otherwise put the mask's lower bound above its causal upper bound
    # and hide every position from every other.
    if sliding_window is not None and sliding_window <= 0:
        sliding_window = None

    # DoRA promotes q/k/v_proj outputs to fp32, which FlashAttention rejects (and so does
    # the xformers flash-2 op on sm_100+, see _XFORMERS_FP32_UNSUPPORTED), so downcast any
    # fp32 Q/K/V to a supported dtype (#1013).
    if (
        backend in (FLASH_DENSE, FLASH_VARLEN)
        or (backend == XFORMERS and _XFORMERS_FP32_UNSUPPORTED)
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
        _, cu_seqlens, max_seqlen = context.seq_info
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
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window = sliding_window,
            base_mask = context.causal_mask,
        )

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
        if context.seq_info is not None and local_mask is None:
            local_mask = build_sdpa_packed_attention_mask(
                context.seq_info,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
            )
        else:
            q_len_local = Q.shape[-2]
            k_len_local = K.shape[-2]
            # ---- SDPA mask normalization for left padding / 2D masks ----
            if local_mask is not None and isinstance(local_mask, torch.Tensor):
                local_mask = local_mask.to(device = Q.device)

                if local_mask.dim() == 2:
                    # key padding keep mask: (bsz, k_len), 1/True = real token
                    if local_mask.dtype == torch.bool:
                        key_keep = local_mask
                    else:
                        # tokenizer attention_mask is typically int 0/1
                        key_keep = local_mask != 0

                    past_len = k_len_local - q_len_local  # works for prefill (0) and decode
                    q_pos = torch.arange(past_len, past_len + q_len_local, device = Q.device)
                    k_pos = torch.arange(k_len_local, device = Q.device)

                    causal_keep = k_pos[None, :] <= q_pos[:, None]  # True = allowed (SDPA)
                    if sliding_window is not None:
                        causal_keep &= k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1))

                    # (bsz, 1, q_len, k_len) boolean keep mask
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

                # Avoid NaNs from fully-masked rows (common with left padding).
                if local_mask.dtype == torch.bool:
                    no_allowed = ~local_mask.any(dim = -1, keepdim = True)  # (bsz,1,q_len,1)
                    local_mask = local_mask | no_allowed

            if local_mask is None and sliding_window is not None and k_len_local > sliding_window:
                # SDPA's is_causal is FULL causal; it has no window. With no padding mask to
                # hang the window off, a model whose config declares one attended its whole
                # history the moment neither the xformers bias nor flash's window_size was the
                # thing running -- which is exactly the SDPA fallback this probe can now cause.
                q_pos = torch.arange(k_len_local - q_len_local, k_len_local, device = Q.device)
                k_pos = torch.arange(k_len_local, device = Q.device)
                local_mask = (
                    (k_pos[None, :] <= q_pos[:, None])
                    & (k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1)))
                )[None, None, :, :]

            is_causal_local = local_mask is None and q_len_local == k_len_local

        kwargs = dict(sdpa_kwargs)
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        use_sdpa_gqa = SDPA_HAS_GQA and config.n_groups != 1
        if (
            use_sdpa_gqa
            and (not requires_grad)
            and isinstance(local_mask, torch.Tensor)
            and local_mask.dim() >= 3
            and local_mask.shape[0] > 1
        ):
            # Batched masked inference has shown row-coupled drift with SDPA GQA.
            # Fall back to explicit KV expansion for deterministic row-wise behavior.
            use_sdpa_gqa = False

        if use_sdpa_gqa:
            kwargs.setdefault("enable_gqa", True)
            out = scaled_dot_product_attention(Q, K, V, **kwargs)
            return out.transpose(1, 2)

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
        return out.transpose(1, 2).contiguous()


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "resolve_prefix_seg_info",
    "run_attention",
]

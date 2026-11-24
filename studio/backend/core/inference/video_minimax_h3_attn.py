# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 attention fast paths, each off with its env var set to 0: UNSLOTH_H3_STRIDED_ATTN
(permuted SDPA views, no copies), UNSLOTH_H3_FUSED_QKV (one row-concatenated int8 to_qkv),
UNSLOTH_H3_SHARED_QKV_QUANT (unfused Q / K / V on one rotation + act quant), UNSLOTH_H3_ATTN_ARCH
(flash only on H3_FLASH_FASTER_ARCHS when flash SDPA is available), UNSLOTH_H3_STRIDED_SAGE (a SageAttention
backend runs inside the strided processor instead of the stock one); all but the arch pick are bit-identical."""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
from functools import lru_cache
from typing import Any, NamedTuple, Optional
from . import diffusion_int8_attention as _i8
from .rocm_bf16 import is_rocm_torch

STRIDED_ATTN_ENV = "UNSLOTH_H3_STRIDED_ATTN"
FUSED_QKV_ENV = "UNSLOTH_H3_FUSED_QKV"
SHARED_QKV_ENV = "UNSLOTH_H3_SHARED_QKV_QUANT"
ATTN_ARCH_ENV = "UNSLOTH_H3_ATTN_ARCH"
STRIDED_SAGE_ENV = "UNSLOTH_H3_STRIDED_SAGE"

# Measured at H3's shape (torch 2.11): flash beats cuDNN on sm80 / sm89, ties or loses on sm120; unmeasured keep cuDNN.
H3_FLASH_FASTER_ARCHS = frozenset({(8, 0), (8, 9)})

# Where H3 video's Auto keeps INT8 attention off entirely (none: Auto runs it with an exact start, below).
H3_INT8_AUTO_OFF_ARCHS: frozenset = frozenset()

# Wherever Auto engages INT8 attention for H3 video, the first N denoise steps of every render run the 16-bit kernel
# and INT8 runs after (device-generic; Auto engages it only on the CUDA kernel today, never on ROCm). INT8 on every
# step moved frames and audio past the fp32-reference gate on L4 and A100-40 alike (its drift from off is about the
# same on every card); teacher-forced on L4, step 0 carried 4.2x the kernel-swap floor's audio velocity error and
# later steps 0.8-1.6x. 4 steps passed the L4 gate where 2 did not (frames); 4 keeps ~85% of INT8's denoise speedup.
H3_INT8_EXACT_START_STEPS = 4
# Overrides the exact-start step count under Auto (0 = INT8 on every step); unset = H3_INT8_EXACT_START_STEPS.
H3_INT8_EXACT_STEPS_ENV = "UNSLOTH_H3_INT8_EXACT_STEPS"

_CUDNN = "_native_cudnn"
_FLASH = "_native_flash"
# Studio's guarded SageAttention backends (pip SageAttention 2, kernels-hub build): both take the NHD layout the
# strided processor already holds, so they run there with the fused q/k norm + RoPE and the shared QKV quant.
STRIDED_SAGE_BACKENDS = frozenset({"sage", "sage_hub"})


def strided_install_backend(backend: Any) -> bool:
    """Strided processor for cuDNN or flash SDPA, and Sage unless UNSLOTH_H3_STRIDED_SAGE=0."""
    value = _backend_value(backend)
    if value in (_CUDNN, _FLASH):
        return True
    return value in STRIDED_SAGE_BACKENDS and _enabled(STRIDED_SAGE_ENV)


def _enabled(env: str) -> bool:
    return (os.environ.get(env) or "1").strip().lower() not in ("0", "off", "false", "no")


def _capability(device: Any = None) -> Optional[tuple[int, int]]:
    try:
        import torch
        if not torch.cuda.is_available() or is_rocm_torch(torch):
            return None
        return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def cuda_capability(device: Any = None) -> Optional[tuple[int, int]]:
    """The CUDA capability of ``device``; None off CUDA, on ROCm or on error."""
    return _capability(device)


def h3_int8_auto_off_reason(capability: Optional[tuple]) -> Optional[str]:
    """Why H3 video's Auto keeps INT8 attention off on a card of ``capability``, or None where Auto may race it."""
    if capability is None or tuple(capability) not in H3_INT8_AUTO_OFF_ARCHS:
        return None
    arch = f"sm{capability[0]}{capability[1]}"
    return (
        f"Auto keeps it off for MiniMax-H3 video on {arch} (it moved the audio past the bf16 noise floor on L4); "
        "int8_attention='on' engages it"
    )


def h3_int8_exact_steps() -> int:
    """Denoise steps (from the first one a render runs) Auto keeps on 16-bit attention wherever it engages INT8: the
    env override, else (unset, or not a count: with a warning) ``H3_INT8_EXACT_START_STEPS``."""
    raw = (os.environ.get(H3_INT8_EXACT_STEPS_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value >= 0:
            return value
        logging.getLogger(__name__).warning(
            "%s=%r is not a step count; using the default %d",
            H3_INT8_EXACT_STEPS_ENV,
            raw,
            H3_INT8_EXACT_START_STEPS,
        )
    return int(H3_INT8_EXACT_START_STEPS)


class H3Int8Gate(NamedTuple):
    """The H3 video load's INT8 attention decision: ``forced`` installs unconditionally, ``auto`` races it, and
    ``reason`` is the status text when nothing gets installed (None where there is nothing to explain).
    ``exact_steps``: under Auto, the first denoise steps kept on 16-bit attention (0 = INT8 on every step)."""

    forced: bool
    auto: bool
    reason: Optional[str]
    exact_steps: int = 0


def h3_int8_load_gate(
    *,
    mode: str,
    attention_backend: Any,
    auto_attention: bool,
    speed_on: bool,
    hosted_int8: bool,
    comfy_checkpoint: bool,
    adapters: bool,
    capability: Optional[tuple],
) -> H3Int8Gate:
    """``mode`` is the resolved policy (``off`` | ``auto`` | ``force``: per-load On / Off over
    ``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL``), ``auto_attention`` whether the load left the attention pick to Studio,
    ``capability`` the load card's (None on ROCm / unknown). Forced (On, env ``1`` or an explicit INT8 backend pick)
    never looks at the arch and runs INT8 on every step; Auto runs only on the evidenced config and off
    ``H3_INT8_AUTO_OFF_ARCHS``, with ``h3_int8_exact_steps()`` (installed only if Auto engages the kernel)."""
    forced = mode == "force" or (mode != "off" and _i8.requested(attention_backend))
    # Auto only on the evidenced config: Studio attention, speed on, hosted INT8 ConvRot, diffusers, no adapters
    auto_config = (
        not forced
        and mode == "auto"
        and speed_on
        and auto_attention
        and hosted_int8
        and not comfy_checkpoint
        and not adapters
    )
    arch_off = h3_int8_auto_off_reason(capability) if auto_config else None
    auto = auto_config and arch_off is None
    if mode == "off":
        reason = None
    elif not speed_on:
        reason = "speed_mode=off keeps the output bit-identical, so the kernel is not installed"
    elif forced:
        reason = None
    elif not auto_attention:
        reason = f"attention_backend={attention_backend!r} was picked explicitly; Auto keeps it"
    elif arch_off is not None:
        reason = arch_off
    elif not auto:
        reason = "Auto engages it on MiniMax-H3 video only over Studio's hosted INT8 denoiser without adapters"
    else:
        reason = None
    exact = h3_int8_exact_steps() if auto else 0
    return H3Int8Gate(forced = forced, auto = auto, reason = reason, exact_steps = exact)


def _flash_sdpa_ready(target: Any = None) -> bool:
    """Whether this torch build can actually run flash SDPA. Empty / failed probes refuse the upgrade (#16294)."""
    if target is not None:
        try:
            from .diffusion_attention import SDPA_FLASH, available_sdpa_kernels
            kernels = available_sdpa_kernels(target)
            # Empty means unanswerable, not "flash works".
            return bool(kernels) and SDPA_FLASH in kernels
        except Exception:  # noqa: BLE001 -- a probe failure must not force flash
            return False
    try:
        import torch
        check = getattr(torch.backends.cuda, "is_flash_attention_available", None)
        if callable(check):
            return bool(check())
    except Exception:  # noqa: BLE001
        return False
    # No target and no torch flag: keep the measured arch upgrade for builds that never expose the check.
    return True


def h3_attention_backend(
    selected: Optional[str],
    capability: Optional[tuple[int, int]] = None,
    requested: Optional[str] = None,
    target: Any = None,
) -> Optional[str]:
    """Moves only the automatic cuDNN pick (``requested`` auto / unset), and only to flash on a measured-faster arch
    whose torch build actually has flash SDPA; anything else unchanged (keep ``selected`` so ``_AutoCudnn`` survives)."""
    if selected != _CUDNN or not _enabled(ATTN_ARCH_ENV):
        return selected
    try:
        from .diffusion_attention import ATTN_AUTO, normalize_attention_backend
        if normalize_attention_backend(requested) not in (ATTN_AUTO, "unsloth_int8"):
            return selected
    except Exception:  # noqa: BLE001 -- unparseable request: leave it alone
        return selected
    cap = capability if capability is not None else _capability()
    if cap is None or tuple(cap) not in H3_FLASH_FASTER_ARCHS:
        return selected
    if not _flash_sdpa_ready(target):
        return selected
    return _FLASH


def _backend_value(backend: Any) -> Optional[str]:
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


def _diffusers_h3_processor() -> Any:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor
    return MiniMaxH3AttnProcessor


def strided_processor_class(base: Any = None) -> Any:
    """Strided subclass of the H3-layout processor ``base`` (diffusers' when None). One cached class per base: dynamo
    guards on the class id, so a class per call would retrace every block."""
    return _strided_processor_for(_diffusers_h3_processor() if base is None else base)


@lru_cache(maxsize = None)
def _strided_processor_for(base: Any) -> Any:
    import sys

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    MiniMaxH3AttnProcessor = base
    _apply_rotary_emb = getattr(sys.modules.get(base.__module__), "_apply_rotary_emb", None)
    if not callable(_apply_rotary_emb):
        raise TypeError(f"{base.__name__}: its module has no _apply_rotary_emb")

    from . import diffusion_int8_gemm as int8_gemm

    # the stock processor's module: its dispatcher runs the Sage backends (the vendored H3-Image one imports diffusers')
    base_module = sys.modules.get(base.__module__)

    def _fusable_norm(norm: Any) -> bool:
        return (
            isinstance(norm, torch.nn.RMSNorm)
            and norm.weight is not None
            and norm.eps is not None
            and tuple(norm.normalized_shape) == (attn_head_dim(norm),)
        )

    def attn_head_dim(norm: Any) -> int:
        return int(norm.weight.shape[-1])

    kernels = {
        _CUDNN: SDPBackend.CUDNN_ATTENTION,
        _FLASH: SDPBackend.FLASH_ATTENTION,
        "_native_math": SDPBackend.MATH,
        "_native_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }

    class UnslothH3StridedAttnProcessor(MiniMaxH3AttnProcessor):
        _unsloth_qk_rope = False
        # True: the reference kernel's single-rounding rotation (torch.ops.unsloth_h3.qk_norm_rope_once)
        _unsloth_qk_rope_once = False
        _unsloth_shared_qkv = False
        # set from UNSLOTH_H3_STRIDED_SAGE at install: a sage backend runs here, not in the stock processor
        _unsloth_sage = False
        # set by install_int8_attention: unmasked calls of at least this many tokens run the INT8 kernel
        _unsloth_int8 = False
        _unsloth_int8_min_tokens = 0
        _unsloth_int8_sinks = 0
        # set by install_int8_exact_start: INT8 calls go through the step-gated op (16-bit kernel code, _EXACT_KERNELS)
        _unsloth_int8_exact = False
        _unsloth_int8_exact_kernel = 0
        _unsloth_int8_exact_flag = None

        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            rotary_emb: Any = None,
            attention_mask: Any = None,
        ) -> Any:
            name = _backend_value(self._attention_backend)
            kernel = kernels.get(name)
            dispatch = None
            if kernel is None and self._unsloth_sage and name in STRIDED_SAGE_BACKENDS:
                dispatch = getattr(base_module, "dispatch_attention_fn", None)
            sage = dispatch is not None
            if (
                (kernel is None and not sage and not self._unsloth_int8)
                or attention_mask is not None
                or self._parallel_config is not None
            ):
                return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
            qkv = None
            if attn.fused_projections:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim = -1)
            else:
                if self._unsloth_shared_qkv:
                    qkv = int8_gemm.shared_rotq_linears(
                        hidden_states, (attn.to_q, attn.to_k, attn.to_v)
                    )
                if qkv is not None:
                    query, key, value = qkv
                else:
                    query = attn.to_q(hidden_states)
                    key = attn.to_k(hidden_states)
                    value = attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            if (
                rotary_emb is not None
                and self._unsloth_qk_rope
                and _fusable_norm(attn.norm_q)
                and _fusable_norm(attn.norm_k)
            ):
                # Through torch.ops, not the Python wrapper: dynamo graph-breaks inside the wrapper's lru_cache.
                cos, sin = rotary_emb
                op = (
                    torch.ops.unsloth_h3.qk_norm_rope_once
                    if self._unsloth_qk_rope_once
                    else torch.ops.unsloth_h3.qk_norm_rope
                )
                query = op(query, attn.norm_q.weight, cos, sin, float(attn.norm_q.eps))
                key = op(key, attn.norm_k.weight, cos, sin, float(attn.norm_k.eps))
            else:
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                if rotary_emb is not None:
                    query = _apply_rotary_emb(query, *rotary_emb)
                    key = _apply_rotary_emb(key, *rotary_emb)

            if (
                self._unsloth_int8
                and query.shape[1] >= self._unsloth_int8_min_tokens
                # no backward in the kernel: gradient calls run native here, not inside the op (autograd cannot see in)
                and not (
                    torch.is_grad_enabled()
                    and (query.requires_grad or key.requires_grad or value.requires_grad)
                )
            ):
                mode = self._unsloth_int8_sinks
                if self._unsloth_int8_exact:
                    # opaque op on the denoiser's flag tensor: the per-step choice is made at run time, so it never
                    # guards or recompiles a block
                    hidden_states = torch.ops.unsloth_h3.int8_attention_exact_start(
                        query,
                        key,
                        value,
                        self._unsloth_int8_exact_flag,
                        self._unsloth_int8_exact_kernel,
                    )
                else:
                    hidden_states = (
                        _i8.int8_attention_call(query, key, value, mode)
                        if mode
                        else _i8.int8_attention_call(query, key, value)
                    )
                hidden_states = hidden_states.flatten(2, 3).type_as(query)
            elif sage:
                # NHD in and out via guarded dispatch; calls Sage cannot take still run native.
                hidden_states = dispatch(
                    query,
                    key,
                    value,
                    attn_mask = None,
                    dropout_p = 0.0,
                    is_causal = False,
                    backend = self._attention_backend,
                    parallel_config = None,
                )
                hidden_states = hidden_states.flatten(2, 3).type_as(query)
            else:
                # an INT8 route over a default SDPA pick has no kernel to pin
                with sdpa_kernel(kernel) if kernel is not None else contextlib.nullcontext():
                    out = F.scaled_dot_product_attention(
                        query.permute(0, 2, 1, 3),
                        key.permute(0, 2, 1, 3),
                        value.permute(0, 2, 1, 3),
                        dropout_p = 0.0,
                        is_causal = False,
                    )
                hidden_states = out.permute(0, 2, 1, 3).flatten(2, 3).type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    return UnslothH3StridedAttnProcessor


_H3_ATTN_FORWARD = ("self", "hidden_states", "rotary_emb", "attention_mask")
_H3_PROC_CALL = ("self", "attn", "hidden_states", "rotary_emb", "attention_mask")


@lru_cache(maxsize = None)
def _params(fn: Any) -> tuple:
    import inspect
    try:
        return tuple(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return ()


def h3_processor_base(module: Any) -> Any:
    """The stock H3-layout processor class of ``module``, or None. Gated on structure (forward signature, norms, QKV,
    a processor module defining ``_apply_rotary_emb``), not class, so the vendored H3-Image attention matches too."""
    import sys

    from torch import nn

    cls = type(module)
    base = getattr(cls, "_default_processor_cls", None)
    if not isinstance(module, nn.Module) or not isinstance(base, type):
        return None
    if _params(cls.forward) != _H3_ATTN_FORWARD or _params(base.__call__) != _H3_PROC_CALL:
        return None
    if not callable(getattr(sys.modules.get(base.__module__), "_apply_rotary_emb", None)):
        return None
    if not all(isinstance(getattr(module, n, None), nn.RMSNorm) for n in ("norm_q", "norm_k")):
        return None
    if not isinstance(getattr(module, "to_out", None), nn.ModuleList) or not callable(
        getattr(module, "set_processor", None)
    ):
        return None
    fused = getattr(module, "fused_projections", False)
    if not (fused and hasattr(module, "to_qkv")) and not all(
        hasattr(module, n) for n in ("to_q", "to_k", "to_v")
    ):
        return None
    if not isinstance(getattr(module, "processor", None), base):
        return None
    return base


def _h3_attention_modules(transformer: Any) -> list:
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return []
    return [m for m in modules() if h3_processor_base(m) is not None]


def install_strided_attention(
    transformer: Any,
    logger: Any = None,
    qk_rope: Optional[bool] = None,
    qk_rope_rounding: str = "eager",
) -> int:
    """Swap every H3 attention processor for the strided one, keeping its backend. Returns the count.
    ``qk_rope=False`` keeps the stock q/k norm + rope; ``None`` follows UNSLOTH_H3_QK_ROPE.
    ``qk_rope_rounding="once"``: the fused q/k norm + rope rounds the rotation once, as the reference kernel."""
    if not _enabled(STRIDED_ATTN_ENV):
        return 0
    modules = _h3_attention_modules(transformer)
    if not modules:
        return 0
    from .video_minimax_h3_qknorm import qk_norm_rope_op, qk_rope_enabled

    qk_rope_on = qk_rope_enabled() and qk_rope is not False
    if qk_rope_on:
        try:
            qk_norm_rope_op()
        except Exception as exc:  # noqa: BLE001 -- keep the stock norm + rope
            qk_rope_on = False
            if logger is not None:
                logger.info("video.h3_attn: fused q/k norm+rope unavailable: %s", exc)
    swapped = 0
    for module in modules:
        old = getattr(module, "processor", None)
        try:
            cls = strided_processor_class(h3_processor_base(module))
        except Exception as exc:  # noqa: BLE001 -- older diffusers without the H3 processor
            if logger is not None:
                logger.info("video.h3_attn: strided processor unavailable, keeping stock: %s", exc)
            return swapped
        if old is None or isinstance(old, cls):
            continue
        new = cls()
        new._unsloth_qk_rope = qk_rope_on
        new._unsloth_qk_rope_once = qk_rope_on and qk_rope_rounding == "once"
        new._unsloth_shared_qkv = _enabled(SHARED_QKV_ENV)
        new._unsloth_sage = _enabled(STRIDED_SAGE_ENV)
        new._attention_backend = getattr(old, "_attention_backend", None)
        new._parallel_config = getattr(old, "_parallel_config", None)
        module.set_processor(new)
        swapped += 1
    if logger is not None and swapped:
        logger.info("video.h3_attn: strided SDPA layout on %d attention modules", swapped)
    return swapped


def strided_attention_count(transformer: Any) -> int:
    """Engagement census."""
    count = 0
    for m in _h3_attention_modules(transformer):
        try:
            cls = strided_processor_class(h3_processor_base(m))
        except Exception:  # noqa: BLE001
            return 0
        count += isinstance(getattr(m, "processor", None), cls)
    return count


# ── INT8 attention exact start: the first denoise steps on the 16-bit kernel, INT8 after ──

_EXACT_OP = "unsloth_h3::int8_attention_exact_start"
_TICK_OP = "unsloth_h3::int8_exact_start_tick"
# the processor's 16-bit kernel as an op argument: 0 = SDPA's own pick
_EXACT_KERNELS = (None, _CUDNN, _FLASH, "_native_efficient", "_native_math")
_EXACT_HOOK_ATTR = "_unsloth_h3_int8_exact_start"
# the denoiser attribute diffusion_cuda_graph reads to refuse graphs (block or whole forward) for this load
GRAPH_UNSAFE_ATTR = "_unsloth_cuda_graph_unsafe"
EXACT_GRAPH_REASON = "INT8 attention exact start routes per denoise step (not CUDA-graph safe)"
# per denoiser (key -> state): the scheduler it reads, its step count and the render's first step index. Each
# denoiser's route lives in its own flag tensor (1 = exact), so overlapping denoisers never read each other's.
_EXACT_REGISTRY: dict = {}
_EXACT_KEYS = itertools.count(1)
# calls served per path since import (engagement census; the op body runs eagerly, also under torch.compile)
EXACT_START_STATS = {"exact": 0, "int8": 0}


def _exact_attention(q: Any, k: Any, v: Any, kernel: int) -> Any:
    """The strided processor's own 16-bit SDPA on ``[B, S, H, D]`` views, pinned to its kernel."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    name = _EXACT_KERNELS[kernel] if 0 <= kernel < len(_EXACT_KERNELS) else None
    backend = {
        _CUDNN: SDPBackend.CUDNN_ATTENTION,
        _FLASH: SDPBackend.FLASH_ATTENTION,
        "_native_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "_native_math": SDPBackend.MATH,
    }.get(name)
    with sdpa_kernel(backend) if backend is not None else contextlib.nullcontext():
        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p = 0.0,
            is_causal = False,
        )
    return out.permute(0, 2, 1, 3).contiguous()


def _capturing_on(device: Any) -> bool:
    """Whether a CUDA graph is recording on ``device``'s current stream (not the current device's)."""
    import torch

    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return False
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


def int8_attention_exact_start_impl(q: Any, k: Any, v: Any, flag: Any, kernel: int) -> Any:
    """``flag`` (the denoiser's, set per forward by its tick) 1: the 16-bit kernel, else the INT8 kernel
    (``int8_attention_impl``, resolved per call). Contiguous ``[B, S, H, D]`` in V's dtype, as the fake."""
    if _capturing_on(q.device):
        # a replayed graph would freeze one path for every step: the recording fails, the block runs ungraphed
        raise RuntimeError(EXACT_GRAPH_REASON)
    import torch

    # V's dtype on both branches whatever autocast says: the fake runs with autocast off (the dispatcher drops it
    # before fake kernels), so the 16-bit SDPA must not follow an autocast dtype the fake cannot see
    with torch.autocast(q.device.type, enabled = False):
        if int(flag.reshape(-1)[0]):
            EXACT_START_STATS["exact"] += 1
            out = _exact_attention(q, k, v, int(kernel))
        else:
            EXACT_START_STATS["int8"] += 1
            out = _i8.int8_attention_impl(q, k, v)
    return out.to(v.dtype).contiguous()


def _exact_start_taken(state: dict) -> Optional[int]:
    """Steps the render has taken, read off its video scheduler (None: no step index, so INT8). Its step index is None
    before a render's first step; the first index seen after that, less one, is where the render began (0, or a later
    grid point with a begin index / strength), so N exact steps run wherever it starts."""
    pipe = state["pipe"]()
    scheduler = getattr(pipe, "scheduler", None) if pipe is not None else None
    if scheduler is None or not hasattr(scheduler, "_step_index"):
        return None
    index = getattr(scheduler, "_step_index", None)
    if index is None:
        state["base"] = None
        return 0
    if state.get("base") is None:
        state["base"] = int(index) - 1
    return int(index) - state["base"]


def int8_exact_start_tick_impl(flag: Any, key: int) -> None:
    state = _EXACT_REGISTRY.get(int(key))
    taken = _exact_start_taken(state) if state is not None else None
    flag.fill_(int(taken is not None and taken < state["steps"]))


@lru_cache(maxsize = None)
def int8_attention_exact_start_op() -> Any:
    """Registered once per process: the attention op and the per-forward tick that sets a denoiser's flag. Both are
    opaque to dynamo, so neither a regional nor an enclosing compile guards on the step index or breaks the graph."""
    import torch

    # explicit schemas: the annotations here are strings
    @torch.library.custom_op(
        _EXACT_OP,
        mutates_args = (),
        schema = "(Tensor q, Tensor k, Tensor v, Tensor flag, int kernel) -> Tensor",
    )
    def int8_attention_exact_start(q, k, v, flag, kernel):
        return int8_attention_exact_start_impl(q, k, v, flag, kernel)

    @int8_attention_exact_start.register_fake
    def _(q, k, v, flag, kernel):
        # contiguous [B, S, H, D] in V's dtype, as unsloth_studio::int8_attention_nhd and the real op
        return v.new_empty(q.shape)

    @torch.library.custom_op(
        _TICK_OP, mutates_args = ("flag",), schema = "(Tensor(a!) flag, int key) -> ()"
    )
    def int8_exact_start_tick(flag, key):
        int8_exact_start_tick_impl(flag, key)

    @int8_exact_start_tick.register_fake
    def _(flag, key):
        return None

    return int8_attention_exact_start


def _exact_start_pre_hook(module: Any, args: tuple, kwargs: dict) -> None:
    """Per forward: the denoiser's tick sets its flag (exact while its render has taken fewer than ``steps`` steps)."""
    import torch

    state = module.__dict__.get(_EXACT_HOOK_ATTR)
    if state is not None:
        torch.ops.unsloth_h3.int8_exact_start_tick(state["flag"], state["key"])
    return None


def remove_int8_exact_start(transformer: Any) -> None:
    """Undo ``install_int8_exact_start``: processors, hook, registry entry and the graph refusal."""
    for m in _h3_attention_modules(transformer):
        proc = getattr(m, "processor", None)
        if getattr(proc, "_unsloth_int8_exact", False):
            proc._unsloth_int8_exact = False
    state = (getattr(transformer, "__dict__", None) or {}).pop(_EXACT_HOOK_ATTR, None)
    if state is not None:
        handle = state.get("handle")
        if handle is not None:
            handle.remove()
        _EXACT_REGISTRY.pop(state.get("key"), None)
    (getattr(transformer, "__dict__", None) or {}).pop(GRAPH_UNSAFE_ATTR, None)


def install_int8_exact_start(
    transformer: Any,
    pipe: Any,
    steps: int,
    logger: Any = None,
) -> int:
    """Route the INT8 calls of ``transformer``'s strided processors through the exact-start op: the first ``steps``
    denoise steps of each render (read off ``pipe.scheduler``, held weakly) run each processor's 16-bit kernel, the
    rest the INT8 kernel. Returns the processors switched (0: nothing installed)."""
    import weakref

    import torch

    if int(steps) <= 0 or transformer is None or pipe is None:
        return 0
    procs = [
        getattr(m, "processor", None)
        for m in _h3_attention_modules(transformer)
        if getattr(getattr(m, "processor", None), "_unsloth_int8", False)
        and not getattr(getattr(m, "processor", None), "_unsloth_int8_sinks", 0)
    ]
    if not procs:
        return 0
    int8_attention_exact_start_op()
    remove_int8_exact_start(transformer)
    try:
        ref = weakref.ref(pipe)
    except (
        TypeError
    ):  # not weak-referenceable: the denoiser then keeps it alive (a cycle the GC collects)
        ref = lambda: pipe  # noqa: E731
    key = next(_EXACT_KEYS)
    flag = torch.zeros(1, dtype = torch.int8)
    state = {"pipe": ref, "steps": int(steps), "base": None, "key": key, "flag": flag}
    _EXACT_REGISTRY[key] = state
    weakref.finalize(transformer, _EXACT_REGISTRY.pop, key, None)
    for proc in procs:
        name = _backend_value(getattr(proc, "_attention_backend", None))
        proc._unsloth_int8_exact_kernel = (
            _EXACT_KERNELS.index(name) if name in _EXACT_KERNELS else 0
        )
        proc._unsloth_int8_exact_flag = flag
        proc._unsloth_int8_exact = True
    state["handle"] = transformer.register_forward_pre_hook(_exact_start_pre_hook, with_kwargs = True)
    transformer.__dict__[_EXACT_HOOK_ATTR] = state
    transformer.__dict__[GRAPH_UNSAFE_ATTR] = EXACT_GRAPH_REASON
    if logger is not None:
        logger.info(
            "video.h3_attn: INT8 attention exact start: the first %d denoise steps on 16-bit attention "
            "(%d attention modules)",
            int(steps),
            len(procs),
        )
    return len(procs)


def finish_h3_int8_auto(
    transformer: Any,
    pipe: Any,
    steps: int,
    *,
    modules: int,
    policy: Any,
    status_pipe: Any,
    logger: Any = None,
) -> tuple:
    """After Auto engaged INT8 on ``modules`` attention modules: install the exact start. ``(modules, reason)``: the
    status reason naming it, or, when it cannot be installed (an error, or no processor took it), INT8 taken off for
    the load (``modules`` 0), never INT8 on every step under Auto."""
    try:
        if not install_int8_exact_start(transformer, pipe, steps, logger = logger):
            raise RuntimeError("no INT8 attention processor took it")
        reason = f"{_i8.resolved_entry(policy, status_pipe)[2]}, the first {int(steps)} denoise steps on 16-bit attention"
        return modules, reason
    except Exception as exc:  # noqa: BLE001 -- back to 16-bit attention
        try:
            remove_int8_exact_start(transformer)
        except Exception:  # noqa: BLE001
            pass
        _i8.remove_int8_attention(transformer)
        if logger is not None:
            logger.warning(
                "video.h3_attn: INT8 attention exact start failed, INT8 attention off: %s", exc
            )
        return (
            0,
            f"Auto turned it off for this load: its exact start could not be installed ({exc})",
        )


def int8_exact_start_count(transformer: Any) -> int:
    """Engagement census: strided processors whose INT8 calls go through the exact-start op."""
    return sum(
        bool(getattr(getattr(m, "processor", None), "_unsloth_int8_exact", False))
        for m in _h3_attention_modules(transformer)
    )


def strided_sage_count(transformer: Any) -> int:
    """Engagement census: strided processors that run a Sage backend themselves."""
    count = 0
    for m in _h3_attention_modules(transformer):
        proc = getattr(m, "processor", None)
        count += bool(
            getattr(proc, "_unsloth_sage", False)
            and _backend_value(getattr(proc, "_attention_backend", None)) in STRIDED_SAGE_BACKENDS
        )
    return count


def shared_qkv_count(transformer: Any) -> int:
    """Engagement census: unfused attention modules whose Q / K / V take the shared rotation + act quant."""
    from .diffusion_int8_gemm import _swapped_rec

    count = 0
    for m in _h3_attention_modules(transformer):
        if getattr(m, "fused_projections", False):
            continue
        if not getattr(getattr(m, "processor", None), "_unsloth_shared_qkv", False):
            continue
        parts = [getattr(m, n, None) for n in ("to_q", "to_k", "to_v")]
        recs = [None if p is None else _swapped_rec(p) for p in parts]
        if all(r is not None and r[2] for r in recs) and len({(r[0], r[1]) for r in recs}) == 1:
            count += 1
    return count


def _per_token_int8(weight: Any) -> bool:
    """Per-token activation quant (depends on the input alone, so appending weight rows cannot change it)."""
    name = type(weight).__name__
    if name == "LinearActivationQuantizedTensor":
        fn = getattr(weight, "input_quant_func", None)
        inner = getattr(weight, "original_weight_tensor", None)
        return (
            getattr(fn, "__name__", "").startswith("_int8_symm_per_token")
            and not getattr(weight, "quant_kwargs", None)
            and type(inner).__name__ == "AffineQuantizedTensor"
            and tuple(getattr(inner, "block_size", ())) == (1, int(weight.shape[-1]))
        )
    if name == "Int8Tensor":
        from .diffusion_int8_fused import _plain_int8_weight
        return _plain_int8_weight(weight)
    return False


def _same_attr(a: Any, b: Any) -> bool:
    try:
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(_same_attr(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_same_attr(x, y) for x, y in zip(a, b))
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- uncomparable: treat as different
        return False


def _cat_rows(parts: list) -> Any:
    """Row-concatenate tensors or identically structured torchao subclasses. Every inner tensor must be per row (a
    per-tensor scale is refused); non-tensor attributes must match, except a recorded outer shape, which is rewritten."""
    import torch

    t0 = parts[0]
    rows = [int(p.shape[0]) for p in parts]
    if any(type(p) is not type(t0) for p in parts):
        raise ValueError("projection weights differ in type")
    if any(
        p.dim() != t0.dim() or tuple(p.shape[1:]) != tuple(t0.shape[1:]) or p.dtype != t0.dtype
        for p in parts
    ):
        raise ValueError("projection weights differ in shape or dtype")
    if type(t0) is torch.Tensor:
        if t0.dim() == 0:
            raise ValueError("scalar leaf")
        return torch.cat([p.detach() for p in parts], dim = 0)
    flatten = getattr(t0, "__tensor_flatten__", None)
    unflatten = getattr(type(t0), "__tensor_unflatten__", None)
    if flatten is None or unflatten is None:
        raise ValueError(f"{type(t0).__name__} is not a flattenable tensor subclass")
    names, ctx = flatten()
    own = tuple(t0.shape)

    def is_shape(x: Any, shape: tuple) -> bool:
        return isinstance(x, (tuple, list, torch.Size)) and tuple(x) == shape

    def normalised(c: Any, shape: tuple) -> Any:
        if isinstance(c, dict):
            return {k: ("<outer-shape>" if is_shape(v, shape) else v) for k, v in c.items()}
        return ["<outer-shape>" if is_shape(x, shape) else x for x in c]

    for p in parts[1:]:
        p_names, p_ctx = p.__tensor_flatten__()
        if list(p_names) != list(names) or not _same_attr(
            normalised(p_ctx, tuple(p.shape)), normalised(ctx, own)
        ):
            raise ValueError("projection weights differ in quantization attributes")
    inner = {}
    for n in names:
        leaves = [getattr(p, n) for p in parts]
        if any(leaf.dim() == 0 or int(leaf.shape[0]) != r for leaf, r in zip(leaves, rows)):
            raise ValueError(f"{type(t0).__name__}.{n} is not per-row")
        inner[n] = _cat_rows(leaves)
    size = torch.Size((sum(rows), *own[1:]))
    if isinstance(ctx, dict):
        new_ctx: Any = {k: (size if is_shape(v, own) else v) for k, v in ctx.items()}
    else:
        new_ctx = [size if is_shape(x, own) else x for x in ctx]
        if isinstance(ctx, tuple):
            new_ctx = tuple(new_ctx)
    return unflatten(inner, new_ctx, size, None)


def _fusable_parts(module: Any) -> Optional[tuple]:
    """(to_q, to_k, to_v, rotation group or None), or None when not fusable."""
    from .diffusion_convrot import is_rotated_linear

    if getattr(module, "fused_projections", False):
        return None
    parts = tuple(getattr(module, name, None) for name in ("to_q", "to_k", "to_v"))
    if any(p is None for p in parts):
        return None
    from torch import nn

    # never a wrapper: an unpadded token refiner (PadToMinM) hits _int_mm's M > 16 assert on a short compiled prompt
    if not all(isinstance(p, nn.Linear) for p in parts):
        return None
    if not all(_per_token_int8(getattr(p, "weight", None)) for p in parts):
        return None
    if any(getattr(p, "bias", None) is not None for p in parts):
        return None
    # Hooks (offload, prefetch, LoRA) are keyed to the three modules; a fused module would silently drop them.
    if any(
        getattr(p, "_forward_pre_hooks", None) or getattr(p, "_forward_hooks", None) for p in parts
    ):
        return None
    if any(hasattr(p, "_hf_hook") or hasattr(p, "_diffusers_hook") for p in parts):
        return None
    if len({type(p) for p in parts}) != 1 or len({p.weight.device for p in parts}) != 1:
        return None
    rotated = [is_rotated_linear(p) for p in parts]
    if any(rotated) and not all(rotated):
        return None
    group = None
    if rotated[0]:
        groups = {int(p.convrot_groupsize) for p in parts}
        if len(groups) != 1:
            return None
        group = groups.pop()
    return parts[0], parts[1], parts[2], group


def _self_check(parts: tuple, fused: Any) -> bool:
    """Bit-exact against the three projections on this device."""
    import torch

    w = parts[0].weight
    device = w.device
    dtype = getattr(w, "dtype", torch.bfloat16)
    g = torch.Generator(device = device).manual_seed(0)
    x = torch.randn(1, 33, int(w.shape[-1]), device = device, dtype = dtype, generator = g)
    with torch.no_grad():
        ref = torch.cat([p(x) for p in parts], dim = -1)
        out = fused(x)
    return out.shape == ref.shape and torch.equal(out, ref)


def fuse_h3_qkv_(transformer: Any, logger: Any = None) -> int:
    """Fuse ``to_q``/``to_k``/``to_v`` into ``to_qkv`` in place where ``_fusable_parts`` allows and ``_self_check``
    passes; one module at a time, so the transient extra memory is one block's QKV."""
    if not _enabled(FUSED_QKV_ENV):
        return 0
    from torch import nn

    from .diffusion_convrot import _install_rotation

    fused = 0
    for module in _h3_attention_modules(transformer):
        try:
            spec = _fusable_parts(module)
        except Exception:  # noqa: BLE001 -- not this layout
            spec = None
        if spec is None:
            continue
        to_q, to_k, to_v, group = spec
        try:
            weight = _cat_rows([to_q.weight, to_k.weight, to_v.weight])
            linear = nn.Linear(
                int(weight.shape[1]), int(weight.shape[0]), bias = False, device = "meta"
            )
            linear.weight = nn.Parameter(weight, requires_grad = False)
            if group is not None:
                _install_rotation(linear, group)
            if not _self_check((to_q, to_k, to_v), linear):
                raise ValueError("fused output differs from the three projections")
        except Exception as exc:  # noqa: BLE001 -- leave this module unfused
            if logger is not None:
                logger.info("video.h3_attn: QKV left unfused: %s", exc)
            continue
        module.to_qkv = linear
        module.fused_projections = True
        for name in ("to_q", "to_k", "to_v"):
            delattr(module, name)
        fused += 1
    if fused:
        # The freed per-projection blocks are fragments the larger fused weights cannot reuse (~5.8 GB on G4).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if logger is not None:
            logger.info("video.h3_attn: fused int8 QKV on %d attention modules", fused)
    return fused


def fused_qkv_count(transformer: Any) -> int:
    """Engagement census."""
    return sum(
        1 for m in _h3_attention_modules(transformer) if getattr(m, "fused_projections", False)
    )

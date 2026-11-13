# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA native int8 W8A8 Linears (``NativeWeightOnlyLinear`` with a ConvRot rotation, bf16 activations): one kernel
rotates and row-quantises the activation (``rotq_i8`` in its native mode) instead of a bf16 rotation GEMM, an fp32 [rows,
K] amax pass and a quantise pass. Codes and scales equal the Linear's own ``_rotate`` + ``_int_mm_rows`` maths in eager,
bit for bit (probed per card; a miss keeps the stock maths). The int8 GEMM and its fp32 epilogue stay the Linear's own
(cuBLAS ``_int_mm``; Inductor fuses the epilogue into the consumer), so only the act quant changes.

Only a compiled (traced) forward takes it: an uncompiled denoiser keeps its compiled ``_w8a8_rows`` function, unchanged.
Every offloaded int8 video DiT read as native Linears (LTX-2.x, Wan, HunyuanVideo-1.5) and every resident one built
that way goes through it, streamed or resident. An opaque custom op, so calls on one input can share one quant.
UNSLOTH_NATIVE_INT8_ROTQ=0 disables (so do the int8 rotation kernels' own switches, UNSLOTH_DIFFUSION_INT8_ROTQUANT=0
and UNSLOTH_DIFFUSION_INT8_GEMM=0)."""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

NATIVE_ROTQ_ENV = "UNSLOTH_NATIVE_INT8_ROTQ"
_NS = "unsloth_studio"
_OP = "native_cuda_rotq"
_GROUP = 256
_LOCK = threading.Lock()
# devices: card index -> None once probed good, else why not. on: card index -> True (read by traced forwards).
# tiles: card index -> (default tile, per-K rules). calls: op launches (engagement census, never read when tracing).
_STATE: dict = {"devices": {}, "on": {}, "tiles": {}, "op": None, "calls": 0, "logged": set()}
# (M, K): ragged rows, the narrow per-K tile (3 / 16 / 8 groups), deep K (64 groups: two rows per tile; 128: one), the
# arch default (36 groups between the per-K rules), an all-zero row (the 1e-12 floor) and outliers in every shape.
_PROBE_SHAPES = ((257, 768), (33, 4096), (130, 2048), (17, 16384), (65, 9216), (19, 32768))


def disabled() -> bool:
    raw = (os.environ.get(NATIVE_ROTQ_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return True
    try:
        from .diffusion_convrot_quant import rotquant_mode
        return rotquant_mode() == "off"
    except Exception:  # noqa: BLE001
        return True


def shape_ok(in_features: int, group: int) -> bool:
    """K this kernel quantises whole rows of (one row per program at most 128 groups of 256)."""
    k, g = int(in_features), int(group or 0)
    return g == _GROUP and k >= g and k % g == 0 and k // g <= 128


def _tile(index: int, k: int) -> Optional[tuple]:
    from .diffusion_convrot_quant import _rotq_rows

    entry = _STATE["tiles"].get(index)
    if entry is None:
        return None
    cfg, rules = entry
    for k_lo, k_hi, tile in rules:
        if k_lo <= k <= k_hi and k % _GROUP == 0 and _rotq_rows(k, _GROUP, tile) >= 1:
            return tile
    return cfg if _rotq_rows(k, _GROUP, cfg) >= 1 else None


def reference(x2: Any, group: int) -> tuple:
    """``NativeWeightOnlyLinear._rotate`` + ``_int_mm_rows``' quant, op for op: (int8 [M, K], fp32 scales [M, 1])."""
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    g = int(group)
    h = build_convrot_hadamard(g, device = x2.device, dtype = torch.float32).to(x2.dtype)
    xr = (x2.reshape(*x2.shape[:-1], x2.shape[-1] // g, g) @ h).reshape(x2.shape)
    x_scale = torch.linalg.vector_norm(
        xr, ord = float("inf"), dim = 1, keepdim = True, dtype = torch.float32
    )
    x_scale = x_scale.clamp_(min = 1e-12).div_(127.0)
    xq = torch.mul(xr, x_scale.reciprocal()).round_().clamp_(-127, 127).to(torch.int8)
    return xq, x_scale


def _launch(x2: Any, group: int) -> tuple:
    import torch

    from .diffusion_convrot_quant import _rotq_launch

    _STATE["calls"] += 1
    x2 = x2 if x2.is_contiguous() else x2.contiguous()
    index = x2.device.index
    tile = _tile(index, int(x2.shape[1])) if x2.dim() == 2 and int(group) == _GROUP else None
    if tile is None or x2.dtype != torch.bfloat16 or not x2.is_cuda:
        return reference(x2, group)
    if index not in _STATE["logged"]:
        _STATE["logged"].add(index)
        _log().info(
            "diffusion.native_rotq: first fused rotation + act quant on cuda:%d (%d x %d)",
            index,
            int(x2.shape[0]),
            int(x2.shape[1]),
        )
    try:
        q, s = _rotq_launch(x2, int(group), "native", tile)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - a launch that fails keeps the stock maths on this card from here on
        _STATE["tiles"].pop(index, None)
        _log().warning(
            "diffusion.native_rotq: kernel failed on cuda:%d (%s); stock maths", index, exc
        )
        return reference(x2, group)
    return q, s.view(-1, 1)


def _op() -> Any:
    """``unsloth_studio::native_cuda_rotq`` (opaque to Inductor), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _OP):
        return getattr(ns, _OP)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_OP}", mutates_args = (), schema = "(Tensor x, int group) -> (Tensor, Tensor)"
        )
        def _native_cuda_rotq(x, group):
            return _launch(x, group)

        @_native_cuda_rotq.register_fake
        def _(x, group):
            return (
                x.new_empty(x.shape, dtype = torch.int8),
                x.new_empty((x.shape[0], 1), dtype = torch.float32),
            )
    except Exception:  # noqa: BLE001 - a registration failure keeps the stock path
        return None
    return getattr(getattr(torch.ops, _NS), _OP)


def _log() -> Any:
    try:
        from loggers import get_logger
        return get_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging
        return logging.getLogger(__name__)


def _probe(index: int, cfg: tuple, rules: tuple) -> tuple:
    """(default tile or None, the per-K rules that match ``reference`` bit for bit on card ``index``)."""
    import torch

    from .diffusion_convrot_quant import _probe_input, _rotq_launch, _rotq_rows

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(5)

    def tile_ok(tile: tuple, shapes: tuple) -> bool:
        tried = False
        for m, k in shapes:
            if _rotq_rows(k, _GROUP, tile) < 1:
                continue
            tried = True
            x = _probe_input(m, k, g).to(dev)
            q, s = _rotq_launch(x, _GROUP, "native", tile)
            rq, rs = reference(x, _GROUP)
            if not (torch.equal(q, rq) and torch.equal(s.view(-1, 1), rs)):
                return False
        return tried

    if not tile_ok(cfg, _PROBE_SHAPES):
        return None, ()
    good = []
    for k_lo, k_hi, tile in rules:
        shapes = tuple((m, k) for m, k in _PROBE_SHAPES if k_lo <= k <= k_hi) or ((33, k_lo),)
        if tile == cfg or tile_ok(tile, shapes):
            good.append((k_lo, k_hi, tile))
    torch.cuda.synchronize(dev)
    return cfg, tuple(good)


def prepare(device: Any = None) -> bool:
    """Probe card ``device`` once (eager, outside any capture); True when compiled forwards there take the kernel."""
    if disabled():
        return False
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch

        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        if device is None:
            index = int(torch.cuda.current_device())
        else:
            device = torch.device(device)
            if device.type != "cuda":
                return False
            index = (
                int(device.index) if device.index is not None else int(torch.cuda.current_device())
            )
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return bool(_STATE["on"].get(index, False))
    except Exception:  # noqa: BLE001
        return False
    with _LOCK:
        if index in _STATE["devices"]:
            return _STATE["devices"][index] is None
        why: Optional[str] = None
        try:
            from . import diffusion_convrot_quant as cq
            cap = tuple(torch.cuda.get_device_capability(index))
            if cap < (8, 0):
                why = f"sm_{cap[0]}{cap[1]} (no bf16 activations)"
            elif not cq._triton_ok() or cq._kernels() is None:
                why = "no Triton >= 3.2"
            elif _op() is None:
                why = "no torch.library.custom_op"
            elif cq.rotquant_mode() == "off":
                why = "int8 rotation kernels switched off"
            else:
                cfg = cq._ROTQ_CONFIG.get(cap, cq._ROTQ_FALLBACK)
                cfg, rules = _probe(index, cfg, cq.rotq_k_tiles(cap))
                if cfg is None:
                    why = "codes differ from the stock rotation + quant"
                else:
                    _STATE["tiles"][index] = (cfg, rules)
        except Exception as exc:  # noqa: BLE001 - a kernel that cannot build keeps the stock path
            why = f"{type(exc).__name__}: {str(exc)[:200]}"
        _STATE["devices"][index] = why
        if why is None:
            # Plain dict entries: a compiled graph reads the handle without tracing the registration.
            _STATE["op"] = _op()
            _STATE["on"][index] = True
    if why is None:
        _log().info(
            "diffusion.native_rotq: int8 W8A8 Linears on cuda:%d rotate + quantise in one kernel when compiled",
            index,
        )
    else:
        _log().info(
            "diffusion.native_rotq: cuda:%d keeps the stock rotation + quant (%s)", index, why
        )
    return why is None


def enabled_for(index: Any) -> bool:
    """Traceable lookup of an earlier ``prepare`` (never probes; a compiled graph reads one dict)."""
    return bool(_STATE["on"].get(index, False)) and _STATE["op"] is not None


def rotq(x2: Any, group: int) -> tuple:
    return _STATE["op"](x2, group)


def call_count() -> int:
    return int(_STATE["calls"])


def candidates(transformer: Any) -> int:
    try:
        from .diffusion_native_quant import is_native_linear
        return sum(
            1
            for m in transformer.modules()
            if is_native_linear(m)
            and bool(getattr(m, "act_int8", False))
            and shape_ok(getattr(m, "in_features", 0), getattr(m, "rot_group", 0))
        )
    except Exception:  # noqa: BLE001
        return 0


def install(
    transformer: Any,
    logger: Any = None,
    device: Any = None,
) -> int:
    """Before the first compiled forward: probe the card the denoiser computes on (``device``, else its weights' card,
    else the current one); returns the native W8A8 Linears that will take the kernel (0 = none or switched off)."""
    if transformer is None or disabled():
        return 0
    n = candidates(transformer)
    if not n:
        return 0
    try:
        import torch
        if device is None:
            for m in transformer.modules():
                w = getattr(m, "weight_q", None)
                if isinstance(w, torch.Tensor) and w.is_cuda:
                    device = w.device
                    break
        if not prepare(device):
            return 0
    except Exception:  # noqa: BLE001
        return 0
    if logger is not None:
        logger.info(
            "diffusion.native_rotq: %d int8 W8A8 ConvRot Linear(s) take the fused rotation + act quant "
            "(UNSLOTH_NATIVE_INT8_ROTQ=0 keeps the stock rotation GEMM + quant)",
            n,
        )
    return n

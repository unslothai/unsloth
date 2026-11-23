# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 SwiGLU MLP: the int8 down projection reads value / gate halves straight from the input projection
(``rotq_h16x``); anything ineligible per call runs the stock forward. Kill switch ``UNSLOTH_H3_FUSED_SWIGLU=0``."""

from __future__ import annotations

import os
import types
from typing import Any

FUSED_SWIGLU_ENV = "UNSLOTH_H3_FUSED_SWIGLU"
_MARK = "_unsloth_h3_fused_swiglu"
_FIRST_CALL_KEY = "h3_fused_swiglu"
# Engagement census, eager calls only (a traced forward counts nothing).
_CALLS = [0]
# Traced gates compare classes, never functions.
_CLS: dict = {}


def fused_swiglu_enabled() -> bool:
    return (os.environ.get(FUSED_SWIGLU_ENV) or "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _classes() -> bool:
    if not _CLS:
        try:
            from diffusers.models.activations import SwiGLU
            from diffusers.models.attention import FeedForward
            from torch import nn
        except Exception:  # noqa: BLE001 - diffusers without these
            return False
        _CLS.update(ff = FeedForward, swiglu = SwiGLU, dropout = nn.Dropout)
    return True


def _eligible(module: Any) -> bool:
    if not _classes() or type(module) is not _CLS["ff"]:
        return False
    net = getattr(module, "net", None)
    if net is None or len(net) < 3 or type(net[0]) is not _CLS["swiglu"]:
        return False
    return all(type(extra) is _CLS["dropout"] for extra in (net[1], *list(net)[3:]))


def _ready(ff: Any) -> bool:
    """Per call: the fused path skips ``SwiGLU.forward``, the Dropouts and the down projection's own forward."""
    from . import diffusion_int8_gemm as gemm

    net = ff.net
    act = net[0]
    if gemm._hooked(act) or not gemm._plain_silu(getattr(act, "activation", None)):
        return False
    for drop in (net[1], *list(net)[3:]):
        if drop.training and drop.p > 0:
            return False
        if gemm._hooked(drop):
            return False
    rec = gemm._swapped_rec(net[2])
    return rec is not None and bool(rec[2]) and rec[1] == 256


def _forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
    stock = type(self).forward
    if args or kwargs or not _ready(self):
        return stock(self, hidden_states, *args, **kwargs)
    from . import diffusion_int8_gemm as gemm

    net = self.net
    act = net[0]
    y = act.proj(hidden_states)
    half = y.shape[-1] // 2
    # SwiGLU.forward: value, gate = proj(x).chunk(2) -> value * silu(gate)
    out = gemm.producer_linear(net[2], None, y[..., half:], y[..., :half])
    if out is None:
        out = net[2](net[1](y[..., :half] * act.activation(y[..., half:])))
    else:
        import torch
        if not torch.compiler.is_compiling():
            _CALLS[0] += 1
    for extra in list(net)[3:]:
        out = extra(out)
    return out


def _arm(index: int) -> bool:
    from .diffusion_convrot_quant import _H16_DEVICE, _arm_producer
    return bool(_H16_DEVICE.get(index)) and _arm_producer(index)


def install(
    transformer: Any,
    device: Any = None,
    logger: Any = None,
) -> int:
    """Call after the int8 GEMM swap, before the first (compiled) forward. Returns the patched count."""
    if transformer is None or not fused_swiglu_enabled():
        return 0
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return 0
    patched = 0
    for module in modules():
        if _MARK in module.__dict__:
            patched += 1
            continue
        if not _eligible(module) or "forward" in module.__dict__:
            continue
        module.__dict__[_MARK] = True
        module.forward = types.MethodType(_forward, module)
        patched += 1
    if not patched:
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return patched
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type != "cuda":
            return patched
        index = dev.index if dev.index is not None else torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        return patched
    armed = _arm(index)
    if not armed:
        from .diffusion_int8_fused import run_on_first_call
        run_on_first_call(transformer, _FIRST_CALL_KEY, lambda _t: _arm(index))
    if logger is not None:
        logger.info(
            "video.h3_ffn: %d SwiGLU MLP(s) feed the down projection's act quant from the projection halves "
            "(card %d %s)",
            patched,
            index,
            "armed" if armed else "arms at the first forward",
        )
    return patched


def uninstall(transformer: Any) -> int:
    restored = 0
    for module in getattr(transformer, "modules", lambda: ())():
        if module.__dict__.pop(_MARK, None) is not None:
            fwd = module.__dict__.get("forward")
            if getattr(fwd, "__func__", None) is _forward:
                del module.__dict__["forward"]
            restored += 1
    return restored


def call_count() -> int:
    return _CALLS[0]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 dense projections that read one input run as one GEMM over the row-concatenated weight:
the SwiGLU ``gate_layer`` + ``proj`` (the single-file layout's ``gate_up``; the halves feed ``silu(gate) * value``)
and, under compile, attention ``to_q`` + ``to_k`` + ``to_v`` (through the shared Q / K / V helper of
``diffusion_qwenimage21_fused``). The parts become row views of the fused weight, so it costs no memory and in-place
weight writes (LoRA fold / restore) reach both. Plain bf16 / fp16 / fp32 Linears on one resident GPU only
(quantised, GGUF, LoRA-wrapped, hooked or offloaded layers keep the stock modules); compiled loads only.
UNSLOTH_DIFFUSION_Q21_DENSE_FUSED=0 keeps the stock projections."""

from __future__ import annotations

import os
import threading
import types
from typing import Any

Q21_DENSE_FUSED_ENV = "UNSLOTH_DIFFUSION_Q21_DENSE_FUSED"
_FF_ATTR = "_unsloth_q21_dense_gate_up"
_FF_CLASS = "QwenImage21SwiGLUFeedForward"
_ATTN_CLASS = "QwenImage21Attention"
_FIRST_CALL_KEY = "q21_dense_fused"
_DEVICE_TYPES = ("cuda",)  # CUDA and ROCm torch both report "cuda"

_LOCK = threading.Lock()


def disabled() -> bool:
    return (os.environ.get(Q21_DENSE_FUSED_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _plain_dense(lin: Any) -> bool:
    import torch
    from torch import nn

    if type(lin) is not nn.Linear or lin.bias is not None:
        return False
    if "forward" in lin.__dict__ or lin._forward_hooks or lin._forward_pre_hooks:
        return False
    w = lin.weight
    return (
        type(w) is nn.Parameter
        and type(w.data) is torch.Tensor
        and w.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and w.device.type in _DEVICE_TYPES
    )


def _resident_device(module: Any) -> Any:
    try:
        devices = {p.device for p in module.parameters()}
    except Exception:  # noqa: BLE001
        return None
    if len(devices) != 1:
        return None
    dev = next(iter(devices))
    return dev if dev.type in _DEVICE_TYPES else None


def _intact(ff: Any) -> Any:
    rec = ff.__dict__.get(_FF_ATTR)
    if rec is None:
        return None
    weight, gate, proj, gate_w, proj_w = rec
    if ff.gate_layer is not gate or ff.proj is not proj:
        return None
    if gate.weight is not gate_w or proj.weight is not proj_w:
        return None
    if weight.device != gate_w.device:
        return None
    return weight


def _forward(self: Any, hidden_states: Any) -> Any:
    import torch.nn.functional as F

    weight = _intact(self)
    if weight is None:
        return type(self).forward(self, hidden_states)
    gate, value = F.linear(hidden_states, weight).chunk(2, dim = -1)
    return self.out(self.activation_fn(gate) * value)


def _fuse(ff: Any) -> bool:
    import torch
    from torch import nn

    gate, proj = ff.gate_layer, ff.proj
    if not (_plain_dense(gate) and _plain_dense(proj)) or not _plain_dense(ff.out):
        return False
    gw, pw = gate.weight, proj.weight
    if gw.shape != pw.shape or gw.dtype != pw.dtype or gw.device != pw.device:
        return False
    with torch.no_grad():
        fused = torch.cat([gw.detach(), pw.detach()], dim = 0)
    n = gw.shape[0]
    gate.weight = nn.Parameter(fused[:n], requires_grad = False)
    proj.weight = nn.Parameter(fused[n:], requires_grad = False)
    ff.__dict__[_FF_ATTR] = (fused, gate, proj, gate.weight, proj.weight)
    ff.forward = types.MethodType(_forward, ff)
    return True


def _fuse_qkv(attn: Any, qkv_attr: str) -> bool:
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    parts = [getattr(attn, n, None) for n in ("to_q", "to_k", "to_v")]
    if any(p is None or not _plain_dense(p) for p in parts):
        return False
    if len({(p.weight.dtype, p.weight.device, p.weight.shape[1]) for p in parts}) != 1:
        return False
    fused = _fuse_linears(parts)
    if fused is None or not _share_storage(fused, parts):
        return False
    attn.__dict__[qkv_attr] = (fused, *parts)
    return True


def install_modules(root: Any, logger: Any = None) -> dict:
    result = {"gate_up": 0, "qkv": 0}
    if disabled() or _resident_device(root) is None:
        return result
    from . import diffusion_qwenimage21_fused as q21f

    qkv_ok = not q21f.q21_convrot_fused_disabled() and q21f._patch_prepare(logger)
    with _LOCK:
        for module in root.modules():
            name = type(module).__name__
            if name == _FF_CLASS:
                if _intact(module) is not None or _fuse(module):
                    result["gate_up"] += 1
            elif name == _ATTN_CLASS and qkv_ok:
                if q21f._qkv_intact(module) is not None or _fuse_qkv(module, q21f._QKV_ATTR):
                    result["qkv"] += 1
    if logger is not None and (result["gate_up"] or result["qkv"]):
        logger.info(
            "diffusion.q21_dense_fused: one GEMM for gate + value in %d MLP(s), for Q / K / V in %d attention(s)",
            result["gate_up"],
            result["qkv"],
        )
    return result


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
) -> Any:
    """Offloaded loads stay stock: an offload hook moving the row views would copy each part beside the fused weight."""
    if disabled():
        uninstall(transformer)
        return {"gate_up": 0, "qkv": 0}
    if type(transformer).__name__ != "QwenImage21Transformer2DModel" or offload_active:
        return {"gate_up": 0, "qkv": 0}
    if _resident_device(transformer) is not None:
        return install_modules(transformer, logger)
    from .diffusion_int8_fused import run_on_first_call

    run_on_first_call(transformer, _FIRST_CALL_KEY, lambda t: install_modules(t, logger))
    return None


def uninstall(transformer: Any = None) -> None:
    if transformer is None:
        return
    from .diffusion_int8_fused import cancel_first_call
    from .diffusion_qwenimage21_fused import _QKV_ATTR

    cancel_first_call(transformer, _FIRST_CALL_KEY)
    with _LOCK:
        for module in transformer.modules():
            rec = module.__dict__.get(_QKV_ATTR)
            if rec is not None and _plain_dense(rec[0]):
                module.__dict__.pop(_QKV_ATTR, None)
            if module.__dict__.pop(_FF_ATTR, None) is not None:
                fwd = module.__dict__.get("forward")
                if getattr(fwd, "__func__", None) is _forward:
                    module.__dict__.pop("forward", None)


def is_replica_record(key: str, value: Any) -> bool:
    """Fused weights live outside the module tree: a deep copy (CFG replica) would duplicate them on the source card."""
    if key == _FF_ATTR:
        return True
    from .diffusion_qwenimage21_fused import _QKV_ATTR

    try:
        return key == _QKV_ATTR and isinstance(value, tuple) and _plain_dense(value[0])
    except Exception:  # noqa: BLE001
        return False


def rebuild_replica(source: Any, replica: Any) -> dict:
    if not any(is_replica_record(k, v) for m in source.modules() for k, v in m.__dict__.items()):
        return {"gate_up": 0, "qkv": 0}
    return install_modules(replica)


def is_installed(module: Any) -> bool:
    return _FF_ATTR in getattr(module, "__dict__", {})

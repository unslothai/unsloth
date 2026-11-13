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
# Inductor post-grad pass: Linears reading one input (Q / K / V) share one fused quant. =0 keeps one per Linear.
NATIVE_ROTQ_SHARE_ENV = "UNSLOTH_NATIVE_INT8_ROTQ_SHARE"
_INDUCTOR = "torch._inductor.config"
_PASS_KNOB = "post_grad_custom_pre_pass"
_NS = "unsloth_studio"
_OP = "native_cuda_rotq"
_GROUP = 256
_LOCK = threading.Lock()
# devices: card index -> None once probed good, else why not. on: card index -> True (read by traced forwards).
# tiles: card index -> (default tile, per-K rules). calls: op launches (engagement census, never read when tracing).
_STATE: dict = {
    "devices": {},
    "on": {},
    "tiles": {},
    "op": None,
    "calls": 0,
    "logged": set(),
    "gemm": {},
    "gemm_cfg": {},
    "res": {},
    "act": {},
    "pro": {},
}
# Inductor folds into the fused kernels (post-grad, after the Q / K / V share), each with its own switch: the tanh GELU
# on a native Linear's output and the block's gated residual into the fused-dequant GEMM's epilogue (on; =0 keeps the
# traced maths), and the RMSNorm + AdaLN modulation in front of a quant (off: on an L4 it cost what it saved; =1 opts in).
FOLD_ENVS = {
    "gelu": "UNSLOTH_NATIVE_INT8_FUSE_GELU",
    "norm": "UNSLOTH_NATIVE_INT8_FUSE_NORM",
    "residual": "UNSLOTH_NATIVE_INT8_FUSE_RESIDUAL",
}
_FOLD_DEFAULT = {"gelu": True, "norm": False, "residual": True}
_NORM_OP = "native_cuda_rotq_norm"
_GEMM_RES_OP = "native_cuda_i8mm_res"
_GEMM_GELU_OP = "native_cuda_i8mm_gelu"
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


def fold_enabled(kind: str) -> bool:
    raw = (os.environ.get(FOLD_ENVS[kind]) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return False
    if raw in ("1", "on", "true", "yes"):
        return True
    return _FOLD_DEFAULT[kind]


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


def gelu_reference(x2: Any) -> Any:
    """aten.gelu(approximate="tanh") on bf16 as Inductor decomposes it: fp32 maths, one bf16 round."""
    import torch

    xf = x2.float()
    inner = (xf + xf * xf * xf * 0.044715) * 0.7978845608028654
    return ((xf * 0.5) * (torch.tanh(inner) + 1.0)).to(torch.bfloat16)


def norm_reference(x2: Any, scale: Any, shift: Any, tokens: int, eps: float) -> Any:
    """diffusers RMSNorm without affine (bf16 out), then ``* (1 + scale) + shift`` in fp32, one bf16 round."""
    import torch

    xf = x2.float()
    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + eps)).to(torch.bfloat16)
    if scale is None:
        return n
    k = x2.shape[-1]
    n3 = n.reshape(-1, int(tokens), k).float()
    out = torch.addcmul(shift.float(), n3, scale.float() + 1.0)
    return out.to(torch.bfloat16).reshape(x2.shape)


def _launch_pro(
    x2: Any,
    group: int,
    prologue: int,
    scale: Any = None,
    shift: Any = None,
    tokens: int = 1,
    eps: float = 1e-6,
) -> tuple:
    """``_launch`` with the RMSNorm [+ AdaLN] prologue (``prologue`` 2) fused in front of the rotation."""
    import torch

    from .diffusion_convrot_quant import _rotq_launch

    def ref() -> tuple:
        return reference(norm_reference(x2, scale, shift, tokens, eps), group)

    _STATE["pro_calls"] = _STATE.get("pro_calls", 0) + 1
    x2 = x2 if x2.is_contiguous() else x2.contiguous()
    index = x2.device.index
    kind = "norm"
    tile = _tile(index, int(x2.shape[1])) if x2.dim() == 2 and int(group) == _GROUP else None
    if (
        tile is None
        or x2.dtype != torch.bfloat16
        or not x2.is_cuda
        or not _STATE["pro"].get(index, {}).get(kind)
    ):
        return ref()
    if scale is not None and not (
        scale.dim() == 3
        and scale.stride(-1) == 1
        and shift.shape == scale.shape
        and shift.stride() == scale.stride()
    ):
        # the kernel reads both with one set of strides (unbind halves of one tensor share them)
        scale, shift = torch.broadcast_tensors(scale, shift)
        scale, shift = scale.contiguous(), shift.contiguous()
    try:
        q, s = _rotq_launch(
            x2,
            int(group),
            "native",
            tile,
            prologue = prologue,
            scale = scale,
            shift = shift,
            tokens = tokens,
            norm_eps = eps,
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - this prologue keeps the traced maths on this card from here on
        _STATE["pro"].setdefault(index, {})[kind] = False
        _log().warning(
            "diffusion.native_rotq: %s prologue failed on cuda:%d (%s)", kind, index, exc
        )
        return ref()
    return q, s.view(-1, 1)


def _norm_op() -> Any:
    """``unsloth_studio::native_cuda_rotq_norm`` (RMSNorm [+ AdaLN] then the fused quant), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _NORM_OP):
        return getattr(ns, _NORM_OP)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_NORM_OP}",
            mutates_args = (),
            schema = "(Tensor x, Tensor? scale, Tensor? shift, int tokens, float eps, int group) -> (Tensor, Tensor)",
        )
        def _rotq_norm(x, scale, shift, tokens, eps, group):
            return _launch_pro(x, group, 2, scale, shift, tokens, eps)

        @_rotq_norm.register_fake
        def _(x, scale, shift, tokens, eps, group):
            return (
                x.new_empty(x.shape, dtype = torch.int8),
                x.new_empty((x.shape[0], 1), dtype = torch.float32),
            )
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _NS), _NORM_OP)


def _pro_probe(index: int) -> dict:
    """The RMSNorm [+ AdaLN] prologue vs its reference maths then the stock rotation + quant (``codes_close``), ragged
    rows, per-batch and per-token modulation, outliers."""
    import torch

    from .diffusion_convrot_quant import _probe_input, _rotq_launch, codes_close

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(17)
    out = {}
    for kind in ("norm",):
        if not fold_enabled(kind):
            continue
        ok = True
        try:
            for b, t, k, per_token in (
                (2, 129, 4096, False),
                (1, 67, 16384, False),
                (3, 40, 2048, True),
            ):
                tile = _tile(index, k)
                if tile is None:
                    continue
                x = (_probe_input(b * t, k, g) * 3).to(torch.bfloat16).to(dev)
                for mod in (False, True):
                    sc = sh = None
                    if mod:
                        tt = t if per_token else 1
                        sc = (torch.randn(b, tt, k, generator = g) * 0.3).to(torch.bfloat16).to(dev)
                        sh = (torch.randn(b, tt, k, generator = g) * 0.3).to(torch.bfloat16).to(dev)
                    q, s = _rotq_launch(
                        x, _GROUP, "native", tile, prologue = 2, scale = sc, shift = sh, tokens = t
                    )
                    rq, rs = reference(norm_reference(x, sc, sh, t, 1e-6), _GROUP)
                    ok = ok and codes_close(q, s.view(-1), rq, rs.view(-1))
            torch.cuda.synchronize(dev)
        except Exception:  # noqa: BLE001
            ok = False
        out[kind] = bool(ok)
    return out


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
            if _norm_op() is not None:
                _STATE["pro"][index] = _pro_probe(index)
    if why is None:
        prepare_gemm(index)
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


NATIVE_GEMM_ENV = "UNSLOTH_NATIVE_INT8_GEMM"
_GEMM_OP = "native_cuda_i8mm"
# (M, N, K, bias): ragged M / N tails, K one / several tiles, the tie operands below
_GEMM_PROBE_SHAPES = (
    (257, 384, 512, True),
    (33, 208, 1024, False),
    (1040, 528, 4096, True),
    (300, 4096, 16384, True),
)


def gemm_disabled() -> bool:
    raw = (os.environ.get(NATIVE_GEMM_ENV) or "").strip().lower()
    return raw in ("0", "off", "false", "no") or disabled()


def load_switches() -> tuple:
    """(fused quant, fused GEMM) allowed by this load's kill switches."""
    return (not disabled(), not gemm_disabled())


def _set_load_switches(transformer: Any) -> None:
    from .diffusion_native_quant import is_native_linear
    rotq_on, gemm_on = load_switches()
    for m in transformer.modules():
        if is_native_linear(m):
            m._cuda_rotq_load, m._cuda_gemm_load = rotq_on, gemm_on


def gemm_shape_ok(in_features: int, out_features: int) -> bool:
    from .diffusion_int8_gemm import _K_ALIGN, _N_ALIGN
    return int(in_features) % _K_ALIGN == 0 and int(out_features) % _N_ALIGN == 0


def reference_gemm(xq: Any, wq: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """``NativeWeightOnlyLinear._int_mm_from_q`` for bf16 activations, op for op (cuBLAS ``_int_mm``, fp32 epilogue)."""
    import torch

    acc = torch._int_mm(xq, wq.t())
    out = torch.mul(acc, xs.reshape(-1, 1))
    w = ws.reshape(-1)
    out = torch.addcmul(bias.float(), out, w) if bias is not None else out.mul_(w)
    return out.to(torch.bfloat16)


def _gemm_tile(index: Any, m: int, n: int, k: int) -> Optional[tuple]:
    """The torchao route's probed tile for this shape when it armed the card, else the arch tile this module probed."""
    from . import diffusion_int8_gemm as g8

    if not _STATE["gemm"].get(index):
        return None
    if g8._DEVICE_CFG.get(index) is not None:
        return g8.tile_for(index, m, n, k)
    return _STATE["gemm_cfg"].get(index)


def reference_gemm_res(
    xq: Any, wq: Any, xs: Any, ws: Any, bias: Any, residual: Any, gate: Any, tokens: int
) -> Any:
    """``residual + linear(x) * gate`` of the traced block: bf16 Linear out, fp32 product and sum, one bf16 round."""
    y = reference_gemm(xq, wq, xs, ws, bias)
    return _residual_math(y, residual, gate, tokens)


def _residual_math(y: Any, residual: Any, gate: Any, tokens: int) -> Any:
    import torch

    n = y.shape[-1]
    y3 = y.reshape(-1, int(tokens), n).float()
    out = torch.addcmul(residual.reshape(-1, int(tokens), n).float(), y3, gate.float())
    return out.to(torch.bfloat16).reshape(y.shape)


def _gemm_launch(
    xq: Any,
    wq: Any,
    xs: Any,
    ws: Any,
    bias: Any,
    residual: Any = None,
    gate: Any = None,
    tokens: int = 1,
    act: int = 0,
) -> Any:
    from . import diffusion_int8_gemm as g8

    _STATE["gemm_calls"] = _STATE.get("gemm_calls", 0) + 1
    xq = xq if xq.stride(-1) == 1 else xq.contiguous()
    xs = xs.reshape(-1)
    index = xq.device.index
    res = residual is not None
    cfg = _gemm_tile(index, xq.shape[0], wq.shape[0], xq.shape[1])
    if (
        cfg is not None
        and wq.stride(-1) == 1
        and g8._aligned(xq, wq)
        and xq.shape[0] >= g8._MIN_ROWS
        and (not res or _STATE["res"].get(index))
        and (not act or _STATE["act"].get(index))
    ):
        try:
            if res:
                _STATE["res_calls"] = _STATE.get("res_calls", 0) + 1
                residual = residual if residual.stride(-1) == 1 else residual.contiguous()
                gate = gate if gate.stride(-1) == 1 else gate.contiguous()
                return g8._launch(
                    xq,
                    wq,
                    xs,
                    ws,
                    bias,
                    cfg,
                    native = True,
                    residual = residual,
                    gate = gate,
                    tokens = tokens,
                )
            if act:
                _STATE["act_calls"] = _STATE.get("act_calls", 0) + 1
            return g8._launch(xq, wq, xs, ws, bias, cfg, native = True, act = act)
        except Exception as exc:  # noqa: BLE001 - this card keeps cuBLAS from here on
            _STATE["gemm"][index] = False
            _log().warning(
                "diffusion.native_rotq: fused-dequant GEMM failed on cuda:%d (%s); cuBLAS",
                index,
                exc,
            )
    if res:
        return reference_gemm_res(xq, wq, xs, ws, bias, residual, gate, tokens)
    if act:
        return gelu_reference(reference_gemm(xq, wq, xs, ws, bias))
    return reference_gemm(xq, wq, xs, ws, bias)


def _gemm_gelu_op() -> Any:
    """``unsloth_studio::native_cuda_i8mm_gelu``: the GEMM with a tanh GELU on its bf16 output, or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _GEMM_GELU_OP):
        return getattr(ns, _GEMM_GELU_OP)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_GEMM_GELU_OP}",
            mutates_args = (),
            schema = "(Tensor xq, Tensor wq, Tensor xs, Tensor ws, Tensor? bias) -> Tensor",
        )
        def _native_cuda_i8mm_gelu(xq, wq, xs, ws, bias):
            return _gemm_launch(xq, wq, xs, ws, bias, act = 1)

        @_native_cuda_i8mm_gelu.register_fake
        def _(xq, wq, xs, ws, bias):
            return xq.new_empty((xq.shape[0], wq.shape[0]), dtype = torch.bfloat16)
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _NS), _GEMM_GELU_OP)


def _gemm_op() -> Any:
    """``unsloth_studio::native_cuda_i8mm`` (opaque to Inductor), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _GEMM_OP):
        return getattr(ns, _GEMM_OP)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_GEMM_OP}",
            mutates_args = (),
            schema = "(Tensor xq, Tensor wq, Tensor xs, Tensor ws, Tensor? bias) -> Tensor",
        )
        def _native_cuda_i8mm(xq, wq, xs, ws, bias):
            return _gemm_launch(xq, wq, xs, ws, bias)

        @_native_cuda_i8mm.register_fake
        def _(xq, wq, xs, ws, bias):
            return xq.new_empty((xq.shape[0], wq.shape[0]), dtype = torch.bfloat16)
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _NS), _GEMM_OP)


def _gemm_res_op() -> Any:
    """``unsloth_studio::native_cuda_i8mm_res``: the GEMM with the block's gated residual in its epilogue, or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _GEMM_RES_OP):
        return getattr(ns, _GEMM_RES_OP)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_GEMM_RES_OP}",
            mutates_args = (),
            schema = "(Tensor xq, Tensor wq, Tensor xs, Tensor ws, Tensor? bias, Tensor residual, Tensor gate, "
            "int tokens) -> Tensor",
        )
        def _native_cuda_i8mm_res(xq, wq, xs, ws, bias, residual, gate, tokens):
            return _gemm_launch(xq, wq, xs, ws, bias, residual, gate, tokens)

        @_native_cuda_i8mm_res.register_fake
        def _(xq, wq, xs, ws, bias, residual, gate, tokens):
            return xq.new_empty((xq.shape[0], wq.shape[0]), dtype = torch.bfloat16)
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _NS), _GEMM_RES_OP)


def _gemm_probe(index: int) -> Optional[str]:
    """None when the native-epilogue tiles this card would run equal ``reference_gemm`` bit for bit, else why not. The
    torchao route's tiles when its own probe armed the card; else the arch tile (``arch_config``: off where that route
    is switched off or gated, e.g. sm_8x on Triton >= 3.8 without the prefetch fix), probed here on the native epilogue
    only (fp32 scales: no bf16 double rounding to tie on)."""
    import torch

    from . import diffusion_int8_gemm as g8

    cfg = None
    try:
        if g8._kernels() is not None and g8._triton_version_ok():
            cfg = g8.device_config(index)
            if cfg is None:
                cfg = g8.arch_config(tuple(torch.cuda.get_device_capability(index)))
    except Exception:  # noqa: BLE001
        cfg = None
    if cfg is None:
        return "no fused-dequant GEMM tile on this card (int8 GEMM gate)"
    _STATE["gemm_cfg"][index] = cfg
    _STATE["gemm"][index] = True  # _gemm_tile reads it; reset by the caller on a miss
    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(9)
    cases = []
    for m, n, k, bias in _GEMM_PROBE_SHAPES:
        a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8)
        w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8)
        b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16) if bias else None
        cases.append((a, w, b))
    ta, tw = g8.tie_operands(dev)
    cases.append((ta.cpu(), tw.cpu(), None))
    for a, w, b in cases:
        a, w = a.to(dev), w.to(dev)
        b = b.to(dev) if b is not None else None
        xs = (torch.rand(a.shape[0], 1, generator = g) * 0.02 + 1e-4).to(dev)
        ws = (torch.rand(w.shape[0], generator = g) * 0.002 + 1e-5).to(dev)
        cfg = _gemm_tile(index, a.shape[0], w.shape[0], a.shape[1])
        if cfg is None or not g8._aligned(a, w):
            continue
        if not torch.equal(
            g8._launch(a, w, xs.reshape(-1), ws, b, cfg, native = True),
            reference_gemm(a, w, xs, ws, b),
        ):
            return f"native epilogue differs (tile {cfg}, {tuple(a.shape)} x {tuple(w.shape)})"
    torch.cuda.synchronize(dev)
    return None


def _res_probe(index: int) -> bool:
    """The gated-residual epilogue vs ``reference_gemm_res`` (per-batch and per-token gates, ragged rows): bf16 within
    one rounding step everywhere, at most 1 in 10^3 values off."""
    import torch

    from . import diffusion_int8_gemm as g8

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(13)
    try:
        for b, t, n, k, per_token in ((2, 129, 384, 512, False), (1, 300, 528, 1024, True)):
            m = b * t
            a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
            w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
            xs = (torch.rand(m, 1, generator = g) * 0.02 + 1e-4).to(dev)
            ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(dev)
            bias = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).to(dev)
            r = (torch.randn(m, n, generator = g) * 4).to(torch.bfloat16).to(dev)
            gate = (
                (torch.randn(b, t if per_token else 1, n, generator = g)).to(torch.bfloat16).to(dev)
            )
            cfg = _gemm_tile(index, m, n, k)
            if cfg is None or not g8._aligned(a, w):
                return False
            out = g8._launch(
                a, w, xs.reshape(-1), ws, bias, cfg, native = True, residual = r, gate = gate, tokens = t
            )
            ref = reference_gemm_res(a, w, xs, ws, bias, r, gate, t)
            if not bf16_close(out, ref, 1e-3):
                return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def _act_probe(index: int) -> bool:
    """The GELU epilogue vs ``gelu_reference`` of ``reference_gemm`` (bias and none, ragged rows): ``bf16_close``."""
    import torch

    from . import diffusion_int8_gemm as g8

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(19)
    try:
        for m, n, k, bias in ((257, 384, 512, True), (300, 1040, 1024, False)):
            a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
            w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
            xs = (torch.rand(m, 1, generator = g) * 0.02 + 1e-4).to(dev)
            ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(dev)
            b = (torch.randn(n, generator = g) * 0.5).to(torch.bfloat16).to(dev) if bias else None
            cfg = _gemm_tile(index, m, n, k)
            if cfg is None or not g8._aligned(a, w):
                return False
            out = g8._launch(a, w, xs.reshape(-1), ws, b, cfg, native = True, act = 1)
            if not bf16_close(out, gelu_reference(reference_gemm(a, w, xs, ws, b)), 1e-3):
                return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def bf16_close(out: Any, ref: Any, frac: float) -> bool:
    """Same shape and dtype, finite where ``ref`` is, every value within one bf16 step, at most ``frac`` of them off."""
    import torch

    if out.shape != ref.shape or out.dtype != ref.dtype:
        return False
    o, r = out.float(), ref.float()
    if not bool((torch.isfinite(o) == torch.isfinite(r)).all()):
        return False
    fin = torch.isfinite(r)
    o, r = o[fin], r[fin]
    step = r.abs().clamp_min(2.0**-126) * 2.0**-7
    if bool(((o - r).abs() > step).any()):
        return False
    return int((o != r).sum()) <= max(1, int(r.numel() * frac))


def prepare_gemm(index: int) -> bool:
    """After ``prepare``: arm the native-epilogue GEMM on card ``index`` (eager, once)."""
    if gemm_disabled():
        return False
    with _LOCK:
        if index in _STATE["gemm"]:
            return bool(_STATE["gemm"][index])
        try:
            why = "no torch.library.custom_op" if _gemm_op() is None else _gemm_probe(index)
        except Exception as exc:  # noqa: BLE001
            why = f"{type(exc).__name__}: {str(exc)[:200]}"
        _STATE["gemm"][index] = why is None
        if why is None:
            _STATE["gemm_op"] = _gemm_op()
            if fold_enabled("residual") and _gemm_res_op() is not None and _res_probe(index):
                _STATE["res"][index] = True
            if fold_enabled("gelu") and _gemm_gelu_op() is not None and _act_probe(index):
                _STATE["act"][index] = True
    if why is None:
        _log().info(
            "diffusion.native_rotq: int8 W8A8 Linears on cuda:%d run the fused-dequant GEMM when compiled",
            index,
        )
    else:
        _log().info(
            "diffusion.native_rotq: cuda:%d keeps cuBLAS _int_mm + epilogue (%s)", index, why
        )
    return why is None


def gemm_enabled_for(index: Any) -> bool:
    """Traceable lookup of an earlier ``prepare_gemm``."""
    return bool(_STATE["gemm"].get(index, False)) and _STATE.get("gemm_op") is not None


def gemm(xq: Any, wq: Any, xs: Any, ws: Any, bias: Any) -> Any:
    return _STATE["gemm_op"](xq, wq, xs, ws, bias)


def share_enabled() -> bool:
    raw = (os.environ.get(NATIVE_ROTQ_SHARE_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _view_root(node: Any) -> tuple:
    """(the tensor a chain of views / reshapes starts from, the outermost requested shape or None)."""
    import torch

    viewish = (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    )
    shape = None
    while getattr(node, "op", None) == "call_function" and node.target in viewish:
        if shape is None:
            shape = tuple(node.args[1])
        node = node.args[0]
    return node, shape


def share_rotq(graph: Any) -> int:
    """Merge every fused-quant call on the same input (same root tensor, view shape and group) into the first one. The
    op is pure, so the merged calls read the same codes and scales. Returns the calls removed."""
    import torch

    ns = getattr(torch.ops, _NS, None)
    packet = getattr(ns, _OP, None) if ns is not None else None
    if packet is None:
        return 0
    targets = (packet, packet.default)
    seen: dict = {}
    merged = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in targets or len(node.args) < 2:
            continue
        root, shape = _view_root(node.args[0])
        try:
            first = seen.setdefault((root, shape, int(node.args[1])), node)
        except TypeError:  # an unhashable (symbolic) shape: keep the call
            continue
        if first is node:
            continue
        node.replace_all_uses_with(first)
        graph.erase_node(node)
        merged += 1
    if merged:
        _STATE["shared"] = _STATE.get("shared", 0) + merged
    return merged


def _fn(node: Any, *targets: Any) -> bool:
    return getattr(node, "op", None) == "call_function" and node.target in targets


def _val(node: Any) -> Any:
    return node.meta.get("val") if hasattr(node, "meta") else None


def _const_side(node: Any, value: float) -> Any:
    """For a binary ``node`` with one Python-number operand equal to ``value``: its other operand, else None."""
    a, b = (list(node.args) + [None, None])[:2]
    for x, c in ((a, b), (b, a)):
        if (
            isinstance(c, (int, float))
            and not isinstance(c, bool)
            and abs(float(c) - value) <= 1e-12 * max(1.0, abs(value))
        ):
            return x
    return None


def _unconvert(node: Any) -> Any:
    import torch
    while _fn(node, torch.ops.prims.convert_element_type.default):
        node = node.args[0]
    return node


def _is_f32_of(node: Any, src: Any) -> bool:
    import torch
    return (
        _fn(node, torch.ops.prims.convert_element_type.default)
        and node.args[0] is src
        and (node.args[1] == torch.float32)
    )


def _bf16(node: Any) -> bool:
    import torch
    v = _val(node)
    return isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16


def _only_feeds(node: Any, sink: Any) -> bool:
    """Every user of ``node`` is a view chain ending in ``sink`` (or unused)."""
    import torch

    viewish = (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    )
    for u in node.users:
        if u is sink:
            continue
        if not _fn(u, *viewish) or not _only_feeds(u, sink):
            return False
    return True


def _match_gelu(root: Any) -> Any:
    """bf16 ``Y`` when ``root`` is aten.gelu(Y, approximate="tanh") as decomposed for Inductor, else None."""
    import torch

    aten = torch.ops.aten
    if (
        not _fn(root, torch.ops.prims.convert_element_type.default)
        or root.args[1] != torch.bfloat16
    ):
        return None
    m = root.args[0]
    if not _fn(m, aten.mul.Tensor):
        return None
    for half, plus in ((m.args[0], m.args[1]), (m.args[1], m.args[0])):
        if not (_fn(half, aten.mul.Tensor) and _fn(plus, aten.add.Tensor)):
            continue
        a = _const_side(half, 0.5)
        th = _const_side(plus, 1.0)
        if a is None or not _fn(th, aten.tanh.default):
            continue
        inner = (
            _const_side(th.args[0], 0.7978845608028654)
            if _fn(th.args[0], aten.mul.Tensor)
            else None
        )
        if not _fn(inner, aten.add.Tensor) or a not in inner.args:
            continue
        cube = inner.args[1] if inner.args[0] is a else inner.args[0]
        c = _const_side(cube, 0.044715) if _fn(cube, aten.mul.Tensor) else None
        if not _fn(c, aten.mul.Tensor):
            continue
        sq = c.args[0] if c.args[1] is a else (c.args[1] if c.args[0] is a else None)
        if not (_fn(sq, aten.mul.Tensor) and sq.args[0] is a and sq.args[1] is a):
            continue
        y = a.args[0] if _fn(a, torch.ops.prims.convert_element_type.default) else None
        if y is not None and a.args[1] == torch.float32 and _bf16(y):
            return y
    return None


def _match_norm(root: Any) -> Any:
    """(X, scale, shift, eps) when ``root`` is ``RMSNorm(X) * (1 + scale) + shift`` (diffusers RMSNorm without
    affine, bf16, last dim; scale / shift [B, 1 | T, K] bf16), (X, None, None, eps) for the bare norm, else None."""
    import torch

    aten = torch.ops.aten
    scale = shift = None
    n = root
    if _fn(root, aten.add.Tensor):
        for mul, sh in ((root.args[0], root.args[1]), (root.args[1], root.args[0])):
            if not _fn(mul, aten.mul.Tensor):
                continue
            for nn_, s1 in ((mul.args[0], mul.args[1]), (mul.args[1], mul.args[0])):
                if _fn(s1, aten.add.Tensor) and _const_side(s1, 1.0) is not None:
                    sc = _const_side(s1, 1.0)
                    if _fn(nn_, torch.ops.prims.convert_element_type.default):
                        n, scale, shift = nn_, sc, sh
                        break
            if scale is not None:
                break
        if scale is None:
            return None
    if not _fn(n, torch.ops.prims.convert_element_type.default) or n.args[1] != torch.bfloat16:
        return None
    m = n.args[0]
    if not _fn(m, aten.mul.Tensor):
        return None
    for xa, rs in ((m.args[0], m.args[1]), (m.args[1], m.args[0])):
        if not _fn(rs, aten.rsqrt.default) or not _fn(rs.args[0], aten.add.Tensor, aten.add.Scalar):
            continue
        add = rs.args[0]
        mean, eps = add.args[0], add.args[1]
        if not isinstance(eps, float) or not _fn(mean, aten.mean.dim):
            continue
        dims = list(mean.args[1]) if len(mean.args) > 1 else []
        keep = mean.args[2] if len(mean.args) > 2 else mean.kwargs.get("keepdim", False)
        sq = mean.args[0]
        if _fn(sq, aten.pow.Tensor_Scalar) and sq.args[1] == 2:
            xf = sq.args[0]
        elif _fn(sq, aten.mul.Tensor) and sq.args[0] is sq.args[1]:
            xf = sq.args[0]
        else:
            continue
        if not _fn(xf, torch.ops.prims.convert_element_type.default) or xf.args[1] != torch.float32:
            continue
        x = xf.args[0]
        if not (xa is x or xa is xf) or not _bf16(x):
            continue
        xv = _val(x)
        if xv.dim() != 3 or not keep or dims not in ([-1], [xv.dim() - 1]):
            continue
        if scale is not None:
            sv, hv = _val(scale), _val(shift)
            if not (
                _bf16(scale)
                and _bf16(shift)
                and sv.dim() == 3
                and tuple(sv.shape) == tuple(hv.shape)
            ):
                return None
            b, t, k = xv.shape
            if sv.shape[2] != k or sv.shape[1] not in (1, t) or sv.shape[0] not in (1, b):
                return None
        return x, scale, shift, float(eps)
    return None


def _fake_node(graph: Any, target: Any, args: tuple) -> Any:
    """``graph.call_function`` with its ``meta["val"]`` from the arguments' fake tensors."""
    import torch

    node = graph.call_function(target, args)
    vals = [a.meta.get("val") if isinstance(a, torch.fx.Node) else a for a in args]
    mode = next((v.fake_mode for v in vals if hasattr(v, "fake_mode")), None)
    if mode is not None:
        with mode:
            node.meta["val"] = target(*vals)
    return node


def fold_norms(graph: Any) -> int:
    """Re-point every fused-quant call whose input is an RMSNorm (+ AdaLN modulation) used by nothing else at the
    prologue variant reading the pre-norm tensor. Returns the calls folded."""
    import torch

    ns = getattr(torch.ops, _NS, None)
    packet = getattr(ns, _OP, None) if ns is not None else None
    norm_op = _norm_op()
    if packet is None or norm_op is None:
        return 0
    folded = 0
    for node in list(graph.nodes):
        if not _fn(node, packet, packet.default) or len(node.args) < 2:
            continue
        x_in = node.args[0]
        root, _shape = _view_root(x_in)
        v = _val(x_in)
        index = getattr(getattr(v, "device", None), "index", None)
        if not _STATE["pro"].get(index, {}).get("norm"):
            continue
        if v is None or v.dim() != 2 or not _only_feeds(root, node):
            continue
        hit = _match_norm(root)
        if hit is None:
            continue
        x, sc, sh, eps = hit
        if not (isinstance(v.shape[1], int) and isinstance(_val(x).shape[1], int)):
            continue  # a symbolic K or token count (dynamic=True): int() would specialise the graph to this shape
        k, group, tokens = int(v.shape[1]), int(node.args[1]), int(_val(x).shape[1])
        with graph.inserting_before(node):
            x2 = _fake_node(graph, torch.ops.aten.reshape.default, (x, [-1, k]))
            new = _fake_node(graph, norm_op.default, (x2, sc, sh, tokens, eps, group))
        node.replace_all_uses_with(new)
        graph.erase_node(node)
        folded += 1
    return folded


def fold_gelus(graph: Any) -> int:
    """A tanh GELU (as decomposed for Inductor) whose input is only a native GEMM op's output becomes that GEMM with
    the GELU in its epilogue. Returns the GELUs folded."""
    import torch

    ns = getattr(torch.ops, _NS, None)
    gemm = getattr(ns, _GEMM_OP, None) if ns is not None else None
    gelu_gemm = _gemm_gelu_op()
    if gemm is None or gelu_gemm is None:
        return 0
    folded = 0
    for root in list(graph.nodes):
        y = _match_gelu(root)
        if y is None:
            continue
        a = next(iter(u for u in y.users), None)  # the fp32 upcast the GELU decomposition reads
        y0, _ = _view_root(y)
        yv = _val(y0)
        if (
            not _fn(y0, gemm, gemm.default)
            or yv is None
            or len(y.users) != 1
            or not _only_feeds(y0, a)
        ):
            continue
        if y0 is not y and len(y0.users) != 1:
            continue
        if not _STATE["act"].get(getattr(yv.device, "index", None)):
            continue
        with graph.inserting_before(root):
            new = _fake_node(graph, gelu_gemm.default, tuple(y0.args[:5]))
            out = _fake_node(graph, torch.ops.aten.reshape.default, (new, list(_val(root).shape)))
        root.replace_all_uses_with(out)
        graph.erase_node(root)
        folded += 1
    return folded


def fold_residuals(graph: Any) -> int:
    """``residual + linear(x) * gate`` (the native GEMM op's only use, gate [B, 1 | T, N]) becomes the GEMM with the
    residual in its epilogue. Returns the adds folded."""
    import torch

    aten = torch.ops.aten
    ns = getattr(torch.ops, _NS, None)
    gemm = getattr(ns, _GEMM_OP, None) if ns is not None else None
    res_op = _gemm_res_op()
    if gemm is None or res_op is None:
        return 0
    folded = 0
    for add in list(graph.nodes):
        if not _fn(add, aten.add.Tensor) or len(add.args) != 2 or not _bf16(add):
            continue
        hit = None
        for mul, r in ((add.args[0], add.args[1]), (add.args[1], add.args[0])):
            if (
                not _fn(mul, aten.mul.Tensor)
                or len(mul.users) != 1
                or not isinstance(r, torch.fx.Node)
            ):
                continue
            for yv, g in ((mul.args[0], mul.args[1]), (mul.args[1], mul.args[0])):
                if not isinstance(yv, torch.fx.Node) or not isinstance(g, torch.fx.Node):
                    continue
                y, _ = _view_root(yv)
                if _fn(y, gemm, gemm.default) and len(y.users) == 1 and _only_feeds(y, mul):
                    hit = (mul, r, y, g)
                    break
            if hit is not None:
                break
        if hit is None:
            continue
        mul, r, y, g = hit
        rv, gv, ov, yv_ = _val(r), _val(g), _val(add), _val(y)
        if not (_bf16(r) and _bf16(g) and rv is not None and gv is not None and yv_ is not None):
            continue
        if rv.dim() != 3 or tuple(rv.shape) != tuple(ov.shape) or gv.dim() != 3:
            continue
        b, t, n = rv.shape
        if not isinstance(t, int):
            continue  # a symbolic token count (dynamic=True): int(t) would recompile the graph per token count
        if b * t != yv_.shape[0] or n != yv_.shape[1] or gv.shape[2] != n:
            continue
        if gv.shape[1] not in (1, t) or gv.shape[0] not in (1, b):
            continue
        if not _STATE["res"].get(getattr(yv_.device, "index", None)):
            continue
        xq, wq, xs, ws, bias = y.args[:5]
        with graph.inserting_before(add):
            r2 = _fake_node(graph, aten.reshape.default, (r, [-1, n]))
            new = _fake_node(graph, res_op.default, (xq, wq, xs, ws, bias, r2, g, int(t)))
            out = _fake_node(graph, aten.reshape.default, (new, list(rv.shape)))
        add.replace_all_uses_with(out)
        graph.erase_node(add)
        folded += 1
    return folded


try:
    from torch._inductor.custom_graph_pass import CustomGraphPass as _CustomGraphPass
except Exception:  # noqa: BLE001 - no Inductor custom passes on this torch: no sharing
    _CustomGraphPass = None

if _CustomGraphPass is not None:

    class ShareRotqPass(_CustomGraphPass):
        """``share_rotq`` then the folds switched on when built, as an Inductor post-grad pass (module level: compile
        configs are pickled; the uuid names the folds, so a switch flip misses the FX graph cache)."""

        def __init__(
            self,
            share: Optional[bool] = None,
            folds: Optional[tuple] = None,
        ) -> None:
            self.share = share_enabled() if share is None else bool(share)
            self.folds = (
                tuple(k for k in FOLD_ENVS if fold_enabled(k)) if folds is None else tuple(folds)
            )

        def __call__(self, graph: Any) -> None:
            if getattr(self, "share", True):
                share_rotq(graph)
            folds = getattr(self, "folds", ())
            changed = 0
            if "gelu" in folds:
                changed += fold_gelus(graph)
            if "norm" in folds:
                changed += fold_norms(graph)
            if "residual" in folds:
                changed += fold_residuals(graph)
            if changed:
                _STATE["folded"] = _STATE.get("folded", 0) + changed
                graph.eliminate_dead_code()

        def uuid(self) -> Any:
            folds = getattr(self, "folds", ())
            return f"unsloth_studio.{_OP}.share.v2:{int(getattr(self, 'share', True))}:{','.join(folds)}"

else:
    ShareRotqPass = None


def _install_share_pass(logger: Any = None) -> bool:
    """Set the post-grad pass (Q / K / V share + the folds switched on) unless all are off or another pass already
    holds the slot."""
    if ShareRotqPass is None:
        return False
    want = ShareRotqPass()
    if not want.share and not want.folds:
        return False
    try:
        from . import diffusion_compile_config as compile_config

        current = compile_config.get_knob(_INDUCTOR, _PASS_KNOB)
        if isinstance(current, ShareRotqPass) and current.uuid() == want.uuid():
            return True
        if current is not None and not isinstance(current, ShareRotqPass):
            return False
        return bool(compile_config.set_knob(_INDUCTOR, _PASS_KNOB, want))
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.info("diffusion.native_rotq: shared quant pass not installed (%s)", exc)
        return False


def install(
    transformer: Any,
    logger: Any = None,
    device: Any = None,
) -> int:
    """Before the first compiled forward: probe the card the denoiser computes on (``device``, else its weights' card,
    else the current one); returns the native W8A8 Linears that will take the kernel (0 = none or switched off)."""
    if transformer is None:
        return 0
    try:
        _set_load_switches(transformer)
    except Exception:  # noqa: BLE001
        pass
    if disabled():
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
    shared = _install_share_pass(logger)
    if logger is not None:
        logger.info(
            "diffusion.native_rotq: %d int8 W8A8 ConvRot Linear(s) take the fused rotation + act quant%s "
            "(UNSLOTH_NATIVE_INT8_ROTQ=0 keeps the stock rotation GEMM + quant)",
            n,
            ", one per shared input" if shared else "",
        )
    return n

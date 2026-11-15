# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 weight quantisers (``max6`` and ``4over6``); output is the standard NVFP4 layout, so kernels are unchanged.

``4over6`` maps each 16-block max to 6 or 4 (lower squared error); its per-tensor scale is ``amax / (6 * 256)`` so
the largest block's /4 scale (384) still fits E4M3 (448).
"""

from __future__ import annotations

import os
import threading
from typing import Any, NamedTuple, Optional

WEIGHT_QUANT_ENV = "UNSLOTH_NVFP4_WEIGHT_QUANT"
METHOD_MAX6 = "max6"
METHOD_4OVER6 = "4over6"
METHODS = (METHOD_MAX6, METHOD_4OVER6)
DEFAULT_METHOD = METHOD_4OVER6
_ALIASES = {
    "max6": METHOD_MAX6,
    "max/6": METHOD_MAX6,
    "rtn": METHOD_MAX6,
    "torchao": METHOD_MAX6,
    "off": METHOD_MAX6,
    "0": METHOD_MAX6,
    "4over6": METHOD_4OVER6,
    "4/6": METHOD_4OVER6,
    "four_over_six": METHOD_4OVER6,
    "fouroversix": METHOD_4OVER6,
    "on": METHOD_4OVER6,
    "1": METHOD_4OVER6,
}

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MAX_4OVER6 = 256.0
E4M3_TINY = 2.0**-6  # torch.finfo(torch.float8_e4m3fn).tiny
_CHUNK_ELEMS = 1 << 24

_STATS_LOCK = threading.Lock()
_STATS: dict = {"weights": 0, "blocks": 0, "blocks_at_4": 0, "passthrough": {}}


class NVFP4Weight(NamedTuple):
    """``scale`` is row-major [N, K/16], not swizzled."""

    qdata: Any
    scale: Any
    per_tensor_scale: Any
    method: str
    blocks_at_4: int = 0


def resolve_method(value: Optional[str] = None) -> str:
    raw = value if value is not None else os.environ.get(WEIGHT_QUANT_ENV)
    if raw is None:
        return DEFAULT_METHOD
    return _ALIASES.get(str(raw).strip().lower().replace("-", "_").replace(" ", ""), DEFAULT_METHOD)


def e4m3_top(method: str) -> float:
    return E4M3_MAX_4OVER6 if method == METHOD_4OVER6 else E4M3_MAX


def per_tensor_scale_for(w: Any, method: str = METHOD_4OVER6) -> Any:
    import torch
    amax = torch.max(torch.abs(w)).to(torch.float32)
    # An all-zero weight would give 0 and 0/0 block scales; any positive scale encodes it as zeros.
    return (amax / (e4m3_top(method) * E2M1_MAX)).clamp_min(torch.finfo(torch.float32).tiny)


def e2m1_round(x: Any) -> Any:
    """fp32 values clamped to +-6 and rounded to the e2m1 grid, ties to even (torchao ``f32_to_f4_unpacked``)."""
    import torch

    x = x.clamp(-E2M1_MAX, E2M1_MAX)
    a = x.abs()
    # torch.round is round-half-even.
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.round(x / step) * step


_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_encode(q: Any) -> Any:
    """Sign in bit 3; -0.0 keeps its sign."""
    import torch

    a = q.abs()
    idx = torch.where(a < 2.0, (a * 2.0), torch.where(a < 4.0, a + 2.0, a / 2.0 + 4.0)).to(
        torch.uint8
    )
    sign = torch.signbit(q).to(torch.uint8) << 3
    return idx | sign


def e2m1_decode(nibbles: Any) -> Any:
    import torch

    grid = torch.tensor(_E2M1_GRID, dtype = torch.float32, device = nibbles.device)
    mag = grid[(nibbles & 7).long()]
    return torch.where((nibbles & 8).bool(), -mag, mag)


def pack_e2m1(q: Any) -> Any:
    """Element 2i in the low nibble."""
    n = e2m1_encode(q)
    return n[..., 0::2] | (n[..., 1::2] << 4)


def unpack_e2m1(qdata: Any) -> Any:
    import torch

    lo = e2m1_decode(qdata & 0xF)
    hi = e2m1_decode(qdata >> 4)
    return torch.stack((lo, hi), dim = -1).reshape(*qdata.shape[:-1], qdata.shape[-1] * 2)


def _blocks(x: Any, rows: int, k: int) -> Any:
    return x.float().reshape(rows, k // BLOCK, BLOCK)


def _candidate(xb: Any, amax: Any, pts: Any, denom: float):
    import torch

    sf8 = torch.clamp((amax / denom) / pts, min = E4M3_TINY, max = E4M3_MAX).to(torch.float8_e4m3fn)
    sf32 = sf8.to(torch.float32)
    codes = e2m1_round(xb * ((1.0 / pts) / sf32).unsqueeze(-1))
    return codes, sf8, sf32


def quantize_blocks(
    xb: Any,
    pts: Any,
    method: str,
    err_weight: Optional[Any] = None,
) -> tuple:
    """Returns (codes, e4m3 scale, took4); ``err_weight`` weights the 4over6 error."""
    import torch

    amax = torch.amax(torch.abs(xb), dim = -1)
    codes, sf8, sf32 = _candidate(xb, amax, pts, E2M1_MAX)
    if method != METHOD_4OVER6:
        return codes, sf8, torch.zeros_like(amax, dtype = torch.bool)
    codes4, sf8_4, sf32_4 = _candidate(xb, amax, pts, 4.0)
    e6 = (codes * (sf32 * pts).unsqueeze(-1) - xb) ** 2
    e4 = (codes4 * (sf32_4 * pts).unsqueeze(-1) - xb) ** 2
    if err_weight is not None:
        e6 = e6 * err_weight
        e4 = e4 * err_weight
    take = e4.sum(-1) < e6.sum(-1)  # ties keep max/6
    return (
        torch.where(take.unsqueeze(-1), codes4, codes),
        torch.where(take, sf8_4, sf8),
        take,
    )


def _check_shape(w: Any) -> tuple:
    if w.dim() != 2:
        raise ValueError(
            f"NVFP4 weight quantisation needs a 2-D weight, got shape {tuple(w.shape)}"
        )
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"NVFP4 needs in_features divisible by {BLOCK}, got {k}")
    return n, k


def quantize_nvfp4_weight(
    w: Any,
    method: Optional[str] = None,
    *,
    per_tensor_scale: Optional[Any] = None,
    err_weight: Optional[Any] = None,
) -> NVFP4Weight:
    """Round-to-nearest NVFP4 of a 2-D [N, K] weight; ``max6`` is bit-identical to ``NVFP4Tensor.to_nvfp4``."""
    import torch

    method = resolve_method(method)
    n, k = _check_shape(w)
    pts = per_tensor_scale_for(w, method) if per_tensor_scale is None else per_tensor_scale
    pts = pts.to(device = w.device, dtype = torch.float32).reshape(())
    ew = None
    if err_weight is not None:
        ew = err_weight.to(device = w.device, dtype = torch.float32).reshape(1, k // BLOCK, BLOCK)
    qdata = torch.empty((n, k // 2), dtype = torch.uint8, device = w.device)
    scale = torch.empty((n, k // BLOCK), dtype = torch.float8_e4m3fn, device = w.device)
    at4 = 0
    step = max(1, _CHUNK_ELEMS // max(k, 1))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        codes, sf8, took = quantize_blocks(_blocks(w[r0:r1], r1 - r0, k), pts, method, ew)
        qdata[r0:r1] = pack_e2m1(codes.reshape(r1 - r0, k))
        scale[r0:r1] = sf8
        if method == METHOD_4OVER6:
            at4 += int(took.sum())
    _record(n * (k // BLOCK), at4)
    return NVFP4Weight(qdata, scale, pts, method, at4)


def dequantize_nvfp4_weight(q: NVFP4Weight, dtype: Any = None) -> Any:
    import torch

    n, half = q.qdata.shape
    k = half * 2
    codes = unpack_e2m1(q.qdata).reshape(n, k // BLOCK, BLOCK)
    s = q.scale.to(torch.float32) * q.per_tensor_scale.to(torch.float32)
    out = (codes * s.unsqueeze(-1)).reshape(n, k)
    return out if dtype is None else out.to(dtype)


def fake_quantize_nvfp4_weight(
    w: Any,
    method: Optional[str] = None,
    *,
    err_weight: Optional[Any] = None,
) -> Any:
    return dequantize_nvfp4_weight(quantize_nvfp4_weight(w, method, err_weight = err_weight), w.dtype)


def fake_quantize_nvfp4_activation(x: Any) -> Any:
    """Dynamic NVFP4 activation quantisation round trip."""
    import torch

    sh = x.shape
    x2 = x.reshape(-1, sh[-1]).float()
    pts = per_tensor_scale_for(x2, METHOD_MAX6).clamp_min(1e-30)
    codes, sf8, _ = quantize_blocks(_blocks(x2, x2.shape[0], sh[-1]), pts, METHOD_MAX6)
    out = codes * (sf8.to(torch.float32) * pts).unsqueeze(-1)
    return out.reshape(sh).to(x.dtype)


def to_torchao_tensor(
    q: NVFP4Weight,
    orig_dtype: Any,
    *,
    act_per_tensor_scale: Optional[Any] = None,
    use_triton_kernel: bool = False,
    act_quant_kwargs: Optional[Any] = None,
) -> Any:
    """Same ``NVFP4Tensor`` (swizzled scales) the weight transform builds."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    from torchao.prototype.mx_formats.utils import (
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
        to_blocked,
    )

    if act_quant_kwargs is None:
        from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs
        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs(is_swizzled_scales = True)
    n, half = q.qdata.shape
    k = half * 2
    scale_m, scale_k = hp_data_dims_to_swizzled_scale_dims_nvfp4(n, k)
    return NVFP4Tensor(
        q.qdata,
        to_blocked(q.scale).flatten().view(scale_m, scale_k),
        BLOCK,
        orig_dtype,
        q.per_tensor_scale,
        act_per_tensor_scale,
        True,
        use_triton_kernel,
        act_quant_kwargs,
    )


def _record(blocks: int, at4: int) -> None:
    with _STATS_LOCK:
        _STATS["weights"] += 1
        _STATS["blocks"] += int(blocks)
        _STATS["blocks_at_4"] += int(at4)


def _passthrough(why: str) -> None:
    with _STATS_LOCK:
        _STATS["passthrough"][why] = _STATS["passthrough"].get(why, 0) + 1


def describe() -> dict:
    with _STATS_LOCK:
        out = dict(_STATS)
        out["passthrough"] = dict(_STATS["passthrough"])
    out["method"] = resolve_method()
    return out


def _refusal(
    data_hp: Any, block_size: int, per_tensor_scale: Any, is_swizzled: bool, use_triton: bool
) -> str:
    import torch

    if not is_swizzled or use_triton or block_size != BLOCK:
        return "layout"
    if not isinstance(per_tensor_scale, torch.Tensor) or per_tensor_scale.numel() != 1:
        return "no per-tensor scale"
    if data_hp.dim() != 2 or data_hp.shape[-1] % BLOCK:
        return "shape"
    if data_hp.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {data_hp.dtype}"
    return ""


def make_weight_to_nvfp4(inner: Any) -> Any:
    """Weight calls (they carry ``act_quant_kwargs``) use the selected method; all else goes to ``inner``."""

    def to_nvfp4(
        data_hp,
        block_size: int = 16,
        per_tensor_scale = None,
        act_per_tensor_scale = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs = None,
    ):
        import torch
        if act_quant_kwargs is not None and not torch.compiler.is_compiling():
            method = resolve_method()
            why = "" if method != METHOD_MAX6 else "max6"
            if not why:
                why = _refusal(
                    data_hp, block_size, per_tensor_scale, is_swizzled_scales, use_triton_kernel
                )
            if not why:
                if data_hp.is_cuda:
                    with torch.cuda.device(data_hp.device):
                        q = quantize_nvfp4_weight(data_hp, method)
                else:
                    q = quantize_nvfp4_weight(data_hp, method)
                pts = q.per_tensor_scale.reshape(per_tensor_scale.shape)
                return to_torchao_tensor(
                    q._replace(per_tensor_scale = pts),
                    data_hp.dtype,
                    act_per_tensor_scale = act_per_tensor_scale,
                    use_triton_kernel = use_triton_kernel,
                    act_quant_kwargs = act_quant_kwargs,
                )
            _passthrough(why)
        return inner(
            data_hp,
            block_size,
            per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            use_triton_kernel,
            act_quant_kwargs,
        )

    to_nvfp4._unsloth_weight_quant = True
    to_nvfp4._unsloth_inner = inner
    return to_nvfp4


_PATCH_LOCK = threading.Lock()


def install_nvfp4_weight_quant_patch() -> bool:
    """Idempotent; re-wraps if another patch replaced ``to_nvfp4``. False when torchao is missing."""
    with _PATCH_LOCK:
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except Exception:  # noqa: BLE001 - no torchao: nothing quantises to NVFP4 through it
            return False
        current = NVFP4Tensor.to_nvfp4
        if getattr(current, "_unsloth_weight_quant", False):
            return True
        NVFP4Tensor.to_nvfp4 = staticmethod(make_weight_to_nvfp4(current))
        return True


def uninstall_nvfp4_weight_quant_patch() -> None:
    with _PATCH_LOCK:
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except Exception:  # noqa: BLE001
            return
        current = NVFP4Tensor.to_nvfp4
        if getattr(current, "_unsloth_weight_quant", False):
            NVFP4Tensor.to_nvfp4 = staticmethod(current._unsloth_inner)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 weight quantisers: per-block max/6 (torchao's), "4over6" and GPTQ. Output is ordinary NVFP4.

Every method writes the same three tensors torchao's ``NVFP4Tensor`` and FlashInfer ``mm_fp4`` already consume:
e2m1 codes (two per byte, even element in the low nibble), one E4M3 block scale per 16 inputs and one fp32
per-tensor scale, with ``w ~= code * (block_scale * per_tensor_scale)``. Only the choice of scales and codes
changes, so no kernel changes.

- ``max6``: torchao's quantiser (block max maps to 6, per-tensor scale ``amax / (6 * 448)``), bit-identical to
  ``NVFP4Tensor.to_nvfp4``.
- ``4over6``: per 16-block, the block max maps to 6 or to 4, whichever reconstructs the block with less (optionally
  activation-weighted) squared error. The per-tensor scale is ``amax / (6 * 256)`` so the ``/4`` candidate of the
  largest block (384) still fits E4M3 (448).
- ``gptq_nvfp4``: GPTQ error feedback over a calibration Hessian, the block scale of each 16-column group chosen
  (max6, or 4over6 weighted by the Hessian diagonal) on the error-updated weights when the group is reached.

Torch-only at import; torchao is imported lazily, only to build an ``NVFP4Tensor``.
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
# E4M3 top the per-tensor scale targets for 4over6: the largest block's /4 candidate lands on 256 * 6 / 4 = 384.
E4M3_MAX_4OVER6 = 256.0
E4M3_TINY = 2.0**-6  # torch.finfo(torch.float8_e4m3fn).tiny, torchao's E4M3_EPS
# Elements per row chunk: bounds the fp32 temporaries of a large weight at load to a few hundred MB.
_CHUNK_ELEMS = 1 << 24

_STATS_LOCK = threading.Lock()
_STATS: dict = {"weights": 0, "blocks": 0, "blocks_at_4": 0, "passthrough": {}}


class NVFP4Weight(NamedTuple):
    """``qdata`` uint8 [N, K/2] packed e2m1, ``scale`` float8_e4m3fn [N, K/16] row-major (not swizzled),
    ``per_tensor_scale`` fp32 0-dim; ``blocks_at_4`` how many blocks took the /4 scale."""

    qdata: Any
    scale: Any
    per_tensor_scale: Any
    method: str
    blocks_at_4: int = 0


def resolve_method(value: Optional[str] = None) -> str:
    """``value`` (else ``$UNSLOTH_NVFP4_WEIGHT_QUANT``, else the default) as a method name; unknown -> default."""
    raw = value if value is not None else os.environ.get(WEIGHT_QUANT_ENV)
    if raw is None:
        return DEFAULT_METHOD
    return _ALIASES.get(str(raw).strip().lower().replace("-", "_").replace(" ", ""), DEFAULT_METHOD)


def e4m3_top(method: str) -> float:
    return E4M3_MAX_4OVER6 if method == METHOD_4OVER6 else E4M3_MAX


def per_tensor_scale_for(w: Any, method: str = METHOD_4OVER6) -> Any:
    """fp32 0-dim per-tensor scale; ``max6`` equals torchao's ``per_tensor_amax_to_scale(max(|w|))``."""
    import torch

    amax = torch.max(torch.abs(w)).to(torch.float32)
    return amax / (e4m3_top(method) * E2M1_MAX)


# ------------------------------------------------------------------------------------------------ e2m1 helpers
def e2m1_round(x: Any) -> Any:
    """fp32 values clamped to +-6 and rounded to the e2m1 grid, ties to even (torchao ``f32_to_f4_unpacked``)."""
    import torch

    x = x.clamp(-E2M1_MAX, E2M1_MAX)
    a = x.abs()
    # Grid step 0.5 below 2, 1 in [2, 4), 2 in [4, 6]; torch.round is round-half-even.
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.round(x / step) * step


_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_encode(q: Any) -> Any:
    """e2m1 values (already on the grid) -> uint8 nibbles, sign in bit 3; -0.0 keeps its sign like torchao."""
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
    """[..., K] e2m1 values -> uint8 [..., K/2], element 2i in the low nibble (torchao ``pack_uint4``)."""
    n = e2m1_encode(q)
    return n[..., 0::2] | (n[..., 1::2] << 4)


def unpack_e2m1(qdata: Any) -> Any:
    """uint8 [..., K/2] -> fp32 e2m1 values [..., K]."""
    import torch

    lo = e2m1_decode(qdata & 0xF)
    hi = e2m1_decode(qdata >> 4)
    return torch.stack((lo, hi), dim = -1).reshape(*qdata.shape[:-1], qdata.shape[-1] * 2)


# ------------------------------------------------------------------------------------------------ RTN quantisers
def _blocks(x: Any, rows: int, k: int) -> Any:
    return x.float().reshape(rows, k // BLOCK, BLOCK)


def _candidate(xb: Any, amax: Any, pts: Any, denom: float):
    """torchao's two-level arithmetic for one candidate: (codes fp32, scale e4m3, scale fp32)."""
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
    """xb fp32 [R, G, 16], pts fp32 scalar tensor -> (codes fp32 [R, G, 16], scale e4m3 [R, G], took4 bool [R, G]).

    ``err_weight`` broadcastable to [R, G, 16] (e.g. E[x^2] per input column) weights the 4over6 choice."""
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
    take = e4.sum(-1) < e6.sum(-1)  # ties keep max/6, so the choice is deterministic
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
    """Round-to-nearest NVFP4 of a 2-D weight [N, K] (any float dtype, any device).

    ``method`` ``max6`` | ``4over6`` (default: ``resolve_method()``); ``per_tensor_scale`` overrides the method's
    (``max6`` with the default scale is bit-identical to torchao ``NVFP4Tensor.to_nvfp4``); ``err_weight`` [K]
    weights the 4over6 error per input column (pass E[x^2] when activation statistics exist)."""
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
    """Kernel math ``code * (e4m3 * per_tensor_scale)`` in fp32, then ``dtype`` (default fp32)."""
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
    """Quantise then dequantise ``w`` to its own dtype (W4A16 emulation)."""
    return dequantize_nvfp4_weight(quantize_nvfp4_weight(w, method, err_weight = err_weight), w.dtype)


def fake_quantize_nvfp4_activation(x: Any) -> Any:
    """torchao's dynamic NVFP4 activation quantisation (per-tensor amax / (6 * 448), max/6 per 16) round trip."""
    import torch

    sh = x.shape
    x2 = x.reshape(-1, sh[-1]).float()
    pts = per_tensor_scale_for(x2, METHOD_MAX6).clamp_min(1e-30)
    codes, sf8, _ = quantize_blocks(_blocks(x2, x2.shape[0], sh[-1]), pts, METHOD_MAX6)
    out = codes * (sf8.to(torch.float32) * pts).unsqueeze(-1)
    return out.reshape(sh).to(x.dtype)


# ------------------------------------------------------------------------------------------------ GPTQ
def _hessian_inverse_cholesky(H: Any, damp: float) -> Any:
    """Upper Cholesky factor of H^-1 after dampening; raises damping on a rank-poor H (a timestep-only input)."""
    import torch

    mean_d = torch.mean(torch.diag(H))
    if not torch.isfinite(mean_d) or float(mean_d) <= 0:
        mean_d = torch.ones((), device = H.device, dtype = H.dtype)
    H.diagonal().add_(damp * mean_d)
    for _ in range(6):
        try:
            return torch.linalg.cholesky(
                torch.cholesky_inverse(torch.linalg.cholesky(H)), upper = True
            )
        except Exception:  # noqa: BLE001 - torch.linalg.LinAlgError, raised as RuntimeError on old torch
            H.diagonal().add_(9 * damp * mean_d)
            damp *= 10
    raise RuntimeError("GPTQ: the calibration Hessian stayed singular after damping")


def gptq_nvfp4(
    w: Any,
    H: Any,
    method: str = METHOD_4OVER6,
    *,
    blocksize: int = 128,
    damp: float = 0.01,
    per_tensor_scale: Optional[Any] = None,
) -> NVFP4Weight:
    """GPTQ (OBS error feedback, Frantar et al.) with NVFP4 16-column groups. ``w`` [N, K], ``H`` [K, K] =
    E[x x^T] over calibration inputs (fp32, on ``w``'s device). The per-tensor scale is fixed from the original
    weight; each group's per-row E4M3 scale is chosen (``method``) on the error-updated weights when the group
    starts, 4over6 weighting the error by diag(H). Deterministic for a fixed H on a fixed device."""
    import torch

    method = resolve_method(method)
    n, k = _check_shape(w)
    if blocksize % BLOCK:
        raise ValueError(f"blocksize must be a multiple of {BLOCK}")
    W = w.detach().to(torch.float32).clone()
    pts = per_tensor_scale_for(W, method) if per_tensor_scale is None else per_tensor_scale
    pts = pts.to(device = W.device, dtype = torch.float32).reshape(())
    H = H.detach().to(device = W.device, dtype = torch.float32).clone()
    hdiag = torch.diag(H).clone()
    dead = hdiag == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    Hinv = _hessian_inverse_cholesky(H, damp)
    del H
    codes = torch.zeros_like(W)
    S = torch.empty((n, k // BLOCK), dtype = torch.float8_e4m3fn, device = W.device)
    at4 = torch.zeros((), dtype = torch.int64, device = W.device)  # summed on device: no sync per group
    ew_all = hdiag.clamp_min(0)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        scale = None
        for j in range(i2 - i1):
            col = i1 + j
            if col % BLOCK == 0:
                g = col // BLOCK
                blk = W1[:, j : j + BLOCK].reshape(n, 1, BLOCK)
                ew = ew_all[col : col + BLOCK].reshape(1, 1, BLOCK)
                _, sf8, took = quantize_blocks(blk, pts, method, ew)
                S[:, g] = sf8[:, 0]
                at4 += took.sum()
                sf32 = sf8[:, 0].to(torch.float32)
                recip = (1.0 / pts) / sf32
                scale = sf32 * pts
            wcol = W1[:, j]
            q = e2m1_round(wcol * recip)
            codes[:, col] = q
            err = (wcol - q * scale) / Hi[j, j]
            W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
            E1[:, j] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    at4 = int(at4)
    _record(n * (k // BLOCK), at4)
    return NVFP4Weight(pack_e2m1(codes), S, pts, method, at4)


def rotate_hessian(H: Any, group_size: int, hadamard: Any) -> Any:
    """H of the rotated input ``x @ blockdiag(h)``: ``R^T H R`` with ``R = blockdiag(h)`` (h symmetric)."""
    k = H.shape[0]
    g = int(group_size)
    h = hadamard.to(device = H.device, dtype = H.dtype)
    a = (H.reshape(k, k // g, g) @ h).reshape(k, k)  # H R
    return (a.t().reshape(k, k // g, g) @ h).reshape(k, k).t()  # ((H R)^T R)^T = R H R


# ------------------------------------------------------------------------------------------------ torchao glue
def to_torchao_tensor(
    q: NVFP4Weight,
    orig_dtype: Any,
    *,
    act_per_tensor_scale: Optional[Any] = None,
    use_triton_kernel: bool = False,
    act_quant_kwargs: Optional[Any] = None,
) -> Any:
    """The ``NVFP4Tensor`` Studio's torchao weight transform builds (swizzled scales), from ``q``'s payload."""
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


def replace_payload(tensor: Any, q: NVFP4Weight) -> Any:
    """A new ``NVFP4Tensor`` carrying ``q``'s codes / scales and every other field of ``tensor`` (an RTN-built
    weight of the same shape): the offline GPTQ builder swaps payloads without touching the layout or flags."""
    return to_torchao_tensor(
        q,
        tensor.dtype,
        act_per_tensor_scale = getattr(tensor, "act_per_tensor_scale", None),
        use_triton_kernel = bool(getattr(tensor, "use_triton_kernel", False)),
        act_quant_kwargs = getattr(tensor, "act_quant_kwargs", None),
    )


# ------------------------------------------------------------------------------------------------ load-time patch
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
    """Wrap a ``to_nvfp4`` so weight calls (they carry ``act_quant_kwargs``, eager, at load) use the selected
    method; activation calls and anything off-layout go to ``inner`` unchanged."""

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
    """Make torchao's NVFP4 weight pass (every ``quantize_`` NVFP4 site: runtime load, policy, offline builder)
    use ``resolve_method()``. Idempotent; re-wraps if another patch replaced ``to_nvfp4`` since. False when
    torchao is missing."""
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

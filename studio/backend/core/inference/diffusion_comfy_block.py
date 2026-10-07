# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI ``nvfp4`` / ``mxfp8`` layers: decode, dequantize, and the runtimes that keep them.

Layout measured on Comfy-Org's files: nvfp4 ``.weight`` uint8 ``[N, K/2]`` with the EVEN column in the HIGH nibble,
e4m3 block scales (per 16 columns) in the cuBLAS 128x4 tiled layout, ``.weight_scale_2`` per-tensor,
``.input_scale`` = activation ``amax / (448 * 6)``, optional ``.pre_quant_scale`` input smoothing. mxfp8 ``.weight``
e4m3 ``[N, K]`` with e8m0 exponents (per 32 columns) in the same tiled layout. Decoded layers are row-major so the
row-tag mapping in ``diffusion_comfy_quant`` splits and concatenates them as for int8 / fp8.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

NVFP4 = "nvfp4"
MXFP8 = "mxfp8"
BLOCK_SIZES = {NVFP4: 16, MXFP8: 32}
COMFY_NVFP4_ENV = "UNSLOTH_DIFFUSION_COMFY_NVFP4"
COMFY_MXFP8_ENV = "UNSLOTH_DIFFUSION_COMFY_MXFP8"
NVFP4_RUNTIME = "flashinfer"
MXFP8_RUNTIME = "scaled_mm"

_OFF = ("0", "off", "false", "no")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP8_MAX = 448.0
_E8M0_BIAS = 127


def _roundup(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def tiled_scale_numel(rows: int, blocks: int) -> int:
    """Elements of a cuBLAS 128x4-tiled scale buffer for ``rows`` x ``blocks`` scales (padding included)."""
    return _roundup(rows, 128) * _roundup(blocks, 4)


def untile_scales(tiled: Any, rows: int, blocks: int) -> Any:
    """The plain ``[rows, blocks]`` scale matrix from a 128x4-tiled buffer (any shape, same dtype)."""
    rp, bp = _roundup(rows, 128), _roundup(blocks, 4)
    flat = tiled.reshape(-1)
    if flat.numel() != rp * bp:
        raise ValueError(
            f"tiled scales hold {flat.numel()} values, {rp * bp} expected for {rows}x{blocks}"
        )
    # tile (row // 128, block // 4) is 512 values: inner (row % 32, (row % 128) // 32, block % 4)
    v = flat.reshape(rp // 128, bp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rp, bp)
    return v[:rows, :blocks].contiguous()


def tile_scales(plain: Any) -> Any:
    """``untile_scales`` inverted: a flat 128x4-tiled buffer, zero padded."""
    import torch

    rows, blocks = plain.shape
    rp, bp = _roundup(rows, 128), _roundup(blocks, 4)
    raw = plain.view(torch.uint8) if plain.element_size() == 1 else plain
    full = torch.zeros(rp, bp, dtype = raw.dtype, device = raw.device)
    full[:rows, :blocks] = raw
    out = full.reshape(rp // 128, 4, 32, bp // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1).contiguous()
    return out.view(plain.dtype) if plain.element_size() == 1 else out


def swap_nibbles(packed: Any) -> Any:
    """High-first fp4 packing <-> low-first (``0xAB`` -> ``0xBA``)."""
    import torch

    u = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    return ((u << 4) | (u >> 4)).to(torch.uint8)


def decode_layer(fmt: str, weight: Any, weight_scale: Any) -> tuple[Any, Any]:
    """Row-major ``(codes, plain scales)``; nvfp4 codes come out low-nibble-first (FlashInfer's packing)."""
    import torch

    rows = int(weight.shape[0])
    if fmt == NVFP4:
        cols = int(weight.shape[1]) * 2
        codes = swap_nibbles(weight)
        scale = untile_scales(weight_scale.view(torch.uint8), rows, cols // 16).view(
            torch.float8_e4m3fn
        )
        return codes, scale
    if fmt == MXFP8:
        codes = weight if weight.dtype == torch.float8_e4m3fn else weight.view(torch.float8_e4m3fn)
        scale = untile_scales(weight_scale.view(torch.uint8), rows, int(weight.shape[1]) // 32)
        return codes, scale
    raise ValueError(f"not a block-scaled ComfyUI format: {fmt!r}")


def logical_cols(fmt: str, codes: Any) -> int:
    return int(codes.shape[1]) * (2 if fmt == NVFP4 else 1)


def dequant_block(
    fmt: str,
    codes: Any,
    scale: Any,
    *,
    tensor_scale: Any = None,
    pre_quant_scale: Any = None,
    dtype: Any = None,
) -> Any:
    """Dense ``[N, K]`` weight; ``pre_quant_scale`` folds into the columns: ``(x * s) @ W.T == x @ (W * s).T``."""
    import torch

    dtype = dtype or torch.bfloat16
    rows = int(codes.shape[0])
    if fmt == NVFP4:
        lut = torch.tensor(
            _E2M1 + tuple(-v for v in _E2M1), dtype = torch.float32, device = codes.device
        )
        c = codes.to(torch.int32)
        values = torch.stack((lut[c & 0x0F], lut[c >> 4]), dim = -1).reshape(rows, -1, 16)
        step = scale.to(torch.float32)
        if tensor_scale is not None:
            step = step * torch.as_tensor(
                tensor_scale, dtype = torch.float32, device = codes.device
            ).reshape(-1, 1)
        weight = (values * step.unsqueeze(-1)).reshape(rows, -1)
    elif fmt == MXFP8:
        exp = scale.view(torch.uint8).to(torch.float32) - _E8M0_BIAS
        weight = (
            codes.to(torch.float32).reshape(rows, -1, 32) * torch.exp2(exp).unsqueeze(-1)
        ).reshape(rows, -1)
    else:
        raise ValueError(f"not a block-scaled ComfyUI format: {fmt!r}")
    if pre_quant_scale is not None:
        weight = weight * pre_quant_scale.to(torch.float32).reshape(1, -1)
    return weight.to(dtype)


def mx_quantize_activation(x: Any) -> tuple[Any, Any]:
    """``[M, K]`` -> (e4m3 codes, tiled e8m0 scales), exponent ``ceil(log2(amax / 448))`` per 32 columns (ComfyUI's rule)."""
    import torch

    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    amax = blocks.abs().amax(dim = -1)
    exp = torch.ceil(torch.log2(torch.clamp(amax / _FP8_MAX, min = 2.0**-127)))
    exp = torch.clamp(exp, -_E8M0_BIAS, _E8M0_BIAS)
    q = torch.clamp(blocks * torch.exp2(-exp).unsqueeze(-1), -_FP8_MAX, _FP8_MAX)
    codes = q.to(torch.float8_e4m3fn).reshape(m, k)
    biased = (exp + _E8M0_BIAS).to(torch.uint8)
    rows_p, blocks_p = _roundup(m, 128), _roundup(k // 32, 4)
    full = torch.full((rows_p, blocks_p), _E8M0_BIAS, dtype = torch.uint8, device = x.device)
    full[:m, : k // 32] = biased
    tiled = full.reshape(rows_p // 128, 4, 32, blocks_p // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)
    return codes, tiled.view(torch.float8_e8m0fnu)


def mxfp8_linear_class():
    import torch
    from torch import nn

    global _MX_CLASS
    if _MX_CLASS is not None:
        return _MX_CLASS

    class ComfyMXFP8Linear(nn.Module):
        """Plain buffers so offload hooks can move it; forward stays capture-safe (no host sync)."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            codes,
            tiled_scale,
            bias = None,
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.register_buffer("weight_q", codes)
            self.register_buffer("weight_sf", tiled_scale.view(torch.uint8))
            self.register_buffer("bias", bias)

        def forward(self, x):
            shape = x.shape
            out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            flat = x.reshape(-1, self.in_features)
            if flat.shape[0] == 0:
                return flat.new_zeros((0, self.out_features), dtype = out_dtype).reshape(
                    *shape[:-1], self.out_features
                )
            xq, x_sf = mx_quantize_activation(flat)
            out = torch._scaled_mm(
                xq,
                self.weight_q.t(),
                scale_a = x_sf,
                scale_b = self.weight_sf.view(torch.float8_e8m0fnu),
                out_dtype = torch.bfloat16,
            )
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.to(out_dtype).reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, mxfp8=scaled_mm"
            )

    _MX_CLASS = ComfyMXFP8Linear
    return _MX_CLASS


_MX_CLASS = None
_NVFP4_DYNAMIC_CLASS = None


def nvfp4_dynamic_linear_class():
    """FlashInfer NVFP4 Linear scaling activations from each call's amax, for layers without ``input_scale``."""
    global _NVFP4_DYNAMIC_CLASS
    if _NVFP4_DYNAMIC_CLASS is not None:
        return _NVFP4_DYNAMIC_CLASS
    import torch
    from torch.nn import functional as F

    from .diffusion_nvfp4_bias import fused_bias_add_
    from .diffusion_nvfp4_linear import nvfp4_linear_class
    from .diffusion_nvfp4_ops import _device_guard, dequantize_nvfp4_weight, global_scale

    base = nvfp4_linear_class()

    class ComfyNVFP4DynamicLinear(base):
        def forward(self, x):
            shape = x.shape
            flat = x.reshape(-1, self.in_features)
            out_dtype = flat.dtype
            if out_dtype not in (torch.bfloat16, torch.float16):
                flat = flat.to(torch.bfloat16)
            if flat.shape[0] == 0:
                return flat.new_zeros((0, self.out_features), dtype = out_dtype).reshape(
                    *shape[:-1], self.out_features
                )
            if self.protect.armed and self.protect.protected:
                out = F.linear(
                    flat,
                    dequantize_nvfp4_weight(self.wq, self.w_sf, self.w_scale, dtype = torch.bfloat16),
                )
            else:
                with _device_guard(flat):
                    a_gsf = global_scale(flat)
                    xq, x_sf = torch.ops.unsloth_nvfp4.quantize(flat, a_gsf)
                    out = torch.ops.unsloth_nvfp4.mm(
                        xq,
                        self.wq,
                        x_sf,
                        self.w_sf,
                        self.w_scale / a_gsf,
                        self.out_features,
                        self.backend,
                    )
            out = out.to(out_dtype)
            if self.bias is not None:
                fused_bias_add_(out, self.bias)
            return out.reshape(*shape[:-1], self.out_features)

    _NVFP4_DYNAMIC_CLASS = ComfyNVFP4DynamicLinear
    return _NVFP4_DYNAMIC_CLASS


_PROBE_LOCK = threading.Lock()
_MX_PROBE: dict = {}


def _device_of(target: Any) -> Any:
    import torch

    device = getattr(target, "torch_device", None) or getattr(target, "device", None)
    if device is None:
        return None
    try:
        return torch.device(device)
    except Exception:  # noqa: BLE001 -- not a device we can name
        return None


def mxfp8_runtime_reason(target: Any) -> Optional[str]:
    """None when ``torch._scaled_mm`` runs e8m0 block-scaled fp8 on ``target``'s device (cached probe), else why not."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "torch unavailable"
    device = _device_of(target)
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return f"device {device} is not CUDA"
    if not hasattr(torch, "float8_e8m0fnu"):
        return f"torch {torch.__version__} has no float8_e8m0fnu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if capability < (10, 0):
        return "sm_%d%d has no MXFP8 tensor cores (Blackwell sm_100+ needed)" % capability
    with _PROBE_LOCK:
        if index not in _MX_PROBE:
            try:
                with torch.cuda.device(index):
                    x = torch.randn(128, 64, device = "cuda", dtype = torch.bfloat16)
                    w = torch.randn(32, 64, device = "cuda", dtype = torch.bfloat16)
                    wq, wsf = mx_quantize_activation(w)
                    linear = mxfp8_linear_class()(64, 32, codes = wq, tiled_scale = wsf)
                    y = linear(x).float()
                    ref = x.float() @ w.float().T
                    err = float((y - ref).norm() / ref.norm())
                    _MX_PROBE[index] = None if err < 0.1 else f"probe error {err:.3f}"
            except Exception as exc:  # noqa: BLE001 -- any failure means dequantize
                _MX_PROBE[index] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return _MX_PROBE[index]


def reset_probe_cache() -> None:
    with _PROBE_LOCK:
        _MX_PROBE.clear()


def _env_off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _OFF


def comfy_block_runtime_layers(model: Any) -> int:
    """How many nvfp4 / mxfp8 layers of a ComfyUI load kept their codes (they cannot carry LoRA adapters)."""
    info = getattr(model, "_unsloth_comfy_quant", None) or {}
    try:
        return int(info.get(NVFP4) or 0) + int(info.get(MXFP8) or 0)
    except (TypeError, ValueError):
        return 0


def comfy_nvfp4_runtime_possible(family: Optional[str]) -> bool:
    """Whether the switch and the family let a ComfyUI nvfp4 file keep its codes (the device is the load's call)."""
    try:
        from .diffusion_transformer_quant import TQ_NVFP4, _family_denied
        return not _env_off(COMFY_NVFP4_ENV) and not _family_denied(family, TQ_NVFP4)
    except Exception:  # noqa: BLE001 -- no answer, no install
        return False


def comfy_block_backend(
    fmt: str,
    target: Any,
    family: Optional[str],
    *,
    dtype: Any = None,
) -> tuple[Optional[str], str]:
    """``(runtime or None, reason)`` for ``fmt`` layers on ``target``; None = dequantize (always correct).
    nvfp4 follows Studio's own NVFP4 gating (flag, FlashInfer backend choice, family); mxfp8 needs the probe + bf16."""
    try:
        from .diffusion_transformer_quant import TQ_MXFP8, TQ_NVFP4, _family_denied
        if fmt == NVFP4:
            if _env_off(COMFY_NVFP4_ENV):
                return None, f"{COMFY_NVFP4_ENV}=0"
            from .diffusion_nvfp4_flag import NVFP4_DIFFUSION_ENV, nvfp4_diffusion_enabled

            if not nvfp4_diffusion_enabled():
                return None, f"Studio's NVFP4 runtime is off ({NVFP4_DIFFUSION_ENV}=1 enables it)"
            if _family_denied(family, TQ_NVFP4):
                return None, f"nvfp4 is denied for {family}"
            device = _device_of(target)
            if device is None or device.type != "cuda":
                return None, f"device {device} is not CUDA"
            from .diffusion_nvfp4_ops import BACKEND_FLASHINFER, _resolve_backend

            backend, reason = _resolve_backend(device)
            if backend != BACKEND_FLASHINFER:
                return None, reason
            return NVFP4_RUNTIME, reason
        if fmt == MXFP8:
            if _env_off(COMFY_MXFP8_ENV):
                return None, f"{COMFY_MXFP8_ENV}=0"
            import torch

            if dtype is not None and dtype is not torch.bfloat16:
                return (
                    None,
                    f"the pipeline runs {str(dtype).replace('torch.', '')}, the mxfp8 GEMM needs bfloat16",
                )
            if _family_denied(family, TQ_MXFP8):
                return None, f"mxfp8 is denied for {family}"
            reason = mxfp8_runtime_reason(target)
            if reason:
                return None, reason
            return MXFP8_RUNTIME, "torch._scaled_mm e8m0 block scales"
    except Exception as exc:  # noqa: BLE001 -- an unanswerable probe takes the exact dense path
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"
    return None, f"unknown format {fmt!r}"


def build_runtime_linear(
    fmt: str,
    linear: Any,
    codes: Any,
    scale: Any,
    *,
    tensor_scale: Any = None,
    input_scale: Any = None,
    bias: Any = None,
) -> Any:
    import torch

    rows = int(codes.shape[0])
    if fmt == NVFP4:
        from .diffusion_nvfp4_linear import nvfp4_linear_class
        from .diffusion_nvfp4_ops import register_ops, sf_matrix_shape

        register_ops()
        cols = logical_cols(fmt, codes)
        w_sf = (
            tile_scales(scale)
            .view(torch.uint8)
            .reshape(sf_matrix_shape(rows, cols // 16))
            .contiguous()
        )
        w_scale = torch.as_tensor(tensor_scale, dtype = torch.float32).reshape(1).clone()
        if input_scale is None:
            # no calibrated scale: per-call amax, as ComfyUI does
            a_gsf = torch.ones(1, dtype = torch.float32)
            cls = nvfp4_dynamic_linear_class()
        else:
            in_scale = torch.as_tensor(input_scale, dtype = torch.float32).reshape(1)
            # FlashInfer's activation global scale is the reciprocal of ComfyUI's decode-side input_scale.
            a_gsf = (1.0 / in_scale).clone()
            cls = nvfp4_linear_class()
        return cls(
            cols,
            rows,
            wq = codes.contiguous(),
            w_sf = w_sf,
            alpha = (w_scale / a_gsf).clone(),
            a_gsf = a_gsf,
            bias = bias,
            w_scale = w_scale,
            activation_scales_baked = True,
        )
    if fmt == MXFP8:
        return mxfp8_linear_class()(
            int(codes.shape[1]),
            rows,
            codes = codes.contiguous(),
            tiled_scale = tile_scales(scale),
            bias = bias,
        )
    raise ValueError(f"not a block-scaled ComfyUI format: {fmt!r}")


def comfy_block_backends(
    scan: Any,
    target: Any,
    family: Optional[str],
    *,
    dtype: Any = None,
    logger: Any = None,
    lora: bool = False,
) -> dict:
    """``load_comfy_quant_transformer`` backend kwargs, logging per format which runtime keeps it or why not."""
    out = {"nvfp4_backend": None, "mxfp8_backend": None}
    counts = scan.counts() if scan is not None else {}
    for fmt in (NVFP4, MXFP8):
        if not counts.get(fmt):
            continue
        backend, reason = comfy_block_backend(fmt, target, family, dtype = dtype)
        if lora and backend:
            backend, reason = None, "a LoRA is selected: adapters need dense Linears"
        out[f"{fmt}_backend"] = backend
        if logger is not None:
            if backend:
                logger.info(
                    "diffusion.comfy_quant: %d %s layer(s) stay %s on the %s runtime (%s)",
                    counts[fmt],
                    fmt,
                    fmt,
                    backend,
                    reason,
                )
            else:
                logger.info(
                    "diffusion.comfy_quant: %d %s layer(s) are dequantized to the compute dtype on load: correct, but no memory "
                    "saving over the bf16 checkpoint (%s)",
                    counts[fmt],
                    fmt,
                    reason,
                )
    return out

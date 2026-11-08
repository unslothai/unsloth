# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""nvfp4 / mxfp8 layers with no native runtime, kept packed and widened per forward, bit-identical to
``dequant_block`` at load. All state is integer buffers so a module-wide ``.to(dtype)`` never casts it."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, Optional

from .diffusion_comfy_block import BLOCK_SIZES, MXFP8, NVFP4

PACKED_BLOCK_ENV = "UNSLOTH_DIFFUSION_PACKED_BLOCK_FALLBACK"
PACKED_BLOCK_ATTR = "_unsloth_packed_block"
_OFF = ("0", "off", "false", "no")

# Torch widen intermediate cap per row chunk (host / accelerator).
_HOST_CHUNK_BYTES = 16 << 20
_DEVICE_CHUNK_BYTES = 256 << 20

_TRITON_LOCK = threading.Lock()
_TRITON: dict = {}
_OP_LOCK = threading.Lock()
_OP: dict = {}


def packed_block_enabled() -> bool:
    return (os.environ.get(PACKED_BLOCK_ENV) or "").strip().lower() not in _OFF


def packed_block_layers(model: Any) -> int:
    """How many nvfp4 / mxfp8 Linears of ``model`` are kept packed (they cannot carry LoRA adapters)."""
    try:
        return int(getattr(model, PACKED_BLOCK_ATTR, 0) or 0)
    except (TypeError, ValueError):
        return 0


def fp8_casts(device: Any) -> bool:
    """Whether float8 tensors cast on ``device`` (CPU / CUDA / ROCm); elsewhere nvfp4 scales go by table and mxfp8
    layers are not kept packed."""
    return getattr(device, "type", str(device).split(":")[0]) in ("cpu", "cuda")


def packed_block_runs_on(fmt: str, target: Any) -> bool:
    """Whether a ``fmt`` layer kept packed can widen on ``target``'s device (unknown: the host's rule)."""
    device = getattr(target, "torch_device", None) or getattr(target, "device", None) or "cpu"
    return fmt == NVFP4 or fp8_casts(device)


def _torch_widen(
    fmt: str, codes: Any, scale: Any, row_scale: Any, pre_quant: Any, dtype: Any
) -> Any:
    from .diffusion_comfy_block import dequant_block

    import torch

    lut = None
    if fmt == MXFP8:
        codes = codes.view(torch.float8_e4m3fn)
    else:
        from .diffusion_nvfp4_ops import e2m1_lut
        if fp8_casts(codes.device):
            scale = scale.view(torch.float8_e4m3fn)
        else:
            # no fp8 casts here (MPS before torch 2.15): the exact float32 of every e4m3 byte, by table
            from .diffusion_native_quant import fp8_lookup_table
            table = fp8_lookup_table("float8_e4m3fn", str(codes.device))
            scale = table.index_select(0, scale.reshape(-1).int()).view(scale.shape)
        lut = e2m1_lut(codes.device)
    budget = _HOST_CHUNK_BYTES if codes.device.type == "cpu" else _DEVICE_CHUNK_BYTES
    return dequant_block(
        fmt,
        codes,
        scale,
        tensor_scale = None if row_scale is None else row_scale.view(torch.float32),
        pre_quant_scale = None if pre_quant is None else pre_quant.view(torch.float32),
        dtype = dtype,
        chunk_bytes = budget,
        lut = lut,
    )


def _triton_kernel() -> Any:
    with _TRITON_LOCK:
        if "kernel" in _TRITON:
            return _TRITON["kernel"]
        try:
            import triton

            from .diffusion_packed_block_triton import nvfp4_widen_kernel
            _TRITON["kernel"] = (triton, nvfp4_widen_kernel)
        except Exception:  # noqa: BLE001 -- no Triton: the torch widen is exact too
            _TRITON["kernel"] = None
        return _TRITON["kernel"]


def _triton_failed(exc: BaseException) -> None:
    with _TRITON_LOCK:
        _TRITON["kernel"] = None
        _TRITON["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def _triton_widen(
    codes: Any, scale: Any, row_scale: Any, pre_quant: Any, dtype: Any
) -> Optional[Any]:
    import torch

    built = _triton_kernel()
    if built is None:
        return None
    triton, kernel = built
    rows, half_k = int(codes.shape[0]), int(codes.shape[1])
    out = torch.empty((rows, 2 * half_k), dtype = dtype, device = codes.device)
    if rows == 0 or half_k == 0:
        return out
    block_r, block_c = 32, 64
    grid = (triton.cdiv(rows, block_r), triton.cdiv(half_k, block_c))
    rs = row_scale.view(torch.float32) if row_scale is not None else out
    pq = pre_quant.view(torch.float32) if pre_quant is not None else out
    guard = torch.cuda.device(codes.device) if codes.is_cuda else contextlib.nullcontext()
    try:
        with guard:
            kernel[grid](
                codes,
                scale,
                rs,
                pq,
                out,
                rows,
                half_k,
                int(scale.shape[1]),
                HAS_RS = row_scale is not None,
                HAS_PQ = pre_quant is not None,
                BLOCK_R = block_r,
                BLOCK_C = block_c,
            )
    except Exception as exc:  # noqa: BLE001 -- a kernel that cannot build here: the torch path is exact
        _triton_failed(exc)
        return None
    return out


def widen_packed(
    fmt: str, codes: Any, scale: Any, row_scale: Any, pre_quant: Any, dtype: Any
) -> Any:
    """The dense ``[N, K]`` weight of packed ``codes`` / ``scale`` (uint8 buffers), bit-identical to the load-time
    widen; Triton on CUDA / ROCm for nvfp4, row-chunked torch otherwise."""
    if fmt not in BLOCK_SIZES:
        raise ValueError(f"not a block-scaled format: {fmt!r}")
    if (
        fmt == NVFP4
        and codes.is_cuda
        and codes.is_contiguous()
        and scale.is_contiguous()
        and (row_scale is None or row_scale.is_contiguous())
        and (pre_quant is None or pre_quant.is_contiguous())
    ):
        out = _triton_widen(codes, scale, row_scale, pre_quant, dtype)
        if out is not None:
            return out
    return _torch_widen(fmt, codes, scale, row_scale, pre_quant, dtype)


def _eager_widen(
    codes: Any, scale: Any, row_scale: Any, pre_quant: Any, fmt: str, dtype: Any
) -> Any:
    return widen_packed(fmt, codes, scale, row_scale, pre_quant, dtype)


def _widen_op() -> Any:
    """``torch.ops.unsloth_packed_block.widen``: opaque to the compiler (shape-only fake), one per process."""
    with _OP_LOCK:
        if "op" in _OP:
            return _OP["op"]
        import torch

        # explicit schema: this module's annotations are strings (no module-level torch to resolve them)
        @torch.library.custom_op(
            "unsloth_packed_block::widen",
            mutates_args = (),
            schema = "(Tensor codes, Tensor scale, Tensor? row_scale, Tensor? pre_quant, str fmt, "
            "ScalarType dtype) -> Tensor",
        )
        def widen(codes, scale, row_scale, pre_quant, fmt, dtype):
            return widen_packed(fmt, codes, scale, row_scale, pre_quant, dtype)

        @widen.register_fake
        def _(codes, scale, row_scale, pre_quant, fmt, dtype):
            cols = int(codes.shape[1]) * (2 if fmt == NVFP4 else 1)
            return codes.new_empty((codes.shape[0], cols), dtype = dtype)

        _OP["op"] = widen
        return widen


def packed_block_linear_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    global _CLASS
    if _CLASS is not None:
        return _CLASS
    _widen_op()

    class PackedBlockLinear(nn.Module):
        """A Linear whose nvfp4 / mxfp8 weight stays packed; each forward widens it to the compute dtype."""

        def __init__(
            self,
            fmt: str,
            in_features: int,
            out_features: int,
            *,
            codes,
            scale,
            row_scale = None,
            pre_quant = None,
            bias = None,
            compute_dtype = None,
        ):
            super().__init__()
            if fmt not in BLOCK_SIZES:
                raise ValueError(f"not a block-scaled format: {fmt!r}")
            self.fmt = fmt
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.compute_dtype = compute_dtype or torch.bfloat16
            self.register_buffer("weight_q", codes.contiguous().view(torch.uint8))
            self.register_buffer("weight_sf", scale.contiguous().view(torch.uint8))
            # float32 factors as int32 views: a module-wide .to(dtype) would round them
            self.register_buffer(
                "row_scale",
                None
                if row_scale is None
                else row_scale.to(torch.float32).reshape(-1).contiguous().view(torch.int32),
            )
            self.register_buffer(
                "pre_quant",
                None
                if pre_quant is None
                else pre_quant.to(torch.float32).reshape(-1).contiguous().view(torch.int32),
            )
            self.register_buffer("bias", bias)

        def dequantized_weight(self, dtype: Any = None) -> Any:
            # eager calls the widen directly; a compiled graph sees the op (no graph break, no recompile)
            widen = (
                torch.ops.unsloth_packed_block.widen
                if torch.compiler.is_compiling()
                else _eager_widen
            )
            w = widen(
                self.weight_q,
                self.weight_sf,
                self.row_scale,
                self.pre_quant,
                self.fmt,
                self.compute_dtype,
            )
            return w if dtype is None or dtype == w.dtype else w.to(dtype)

        @property
        def weight(self) -> Any:
            return self.dequantized_weight()

        def forward(self, x: Any) -> Any:
            return F.linear(x, self.dequantized_weight(), self.bias)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"{self.fmt}=packed, bias={self.bias is not None}"
            )

    _CLASS = PackedBlockLinear
    return _CLASS


_CLASS = None


def is_packed_block_linear(module: Any) -> bool:
    return type(module).__name__ == "PackedBlockLinear" and hasattr(module, "weight_sf")


def packed_block_args(parts: list) -> Optional[tuple]:
    """``(codes, scale, row_scale, pre_quant)`` holding the row segments ``parts`` exactly, else None (segments with
    different input smoothing need one weight each). ``row_scale`` / ``pre_quant`` are None when absent everywhere;
    a segment without a tensor scale gets 1.0, which leaves its product bit-identical."""
    import torch

    from .diffusion_single_file_converters import charge_host_ram

    smoothing = [extra.get("pre_quant_scale") for _c, _s, extra in parts]
    first = smoothing[0]
    for other in smoothing[1:]:
        if (first is None) != (other is None):
            return None
        if first is not None and not torch.equal(first.to(torch.float32), other.to(torch.float32)):
            return None
    scales = [extra.get("tensor_scale") for _c, _s, extra in parts]
    row_scale = None
    if any(s is not None for s in scales):
        row_scale = torch.cat(
            [
                torch.full((int(c.shape[0]),), 1.0 if s is None else float(s), dtype = torch.float32)
                for (c, _s, _e), s in zip(parts, scales)
            ]
        )
    if len(parts) > 1:
        charge_host_ram(
            sum(
                int(c.numel()) * int(c.element_size()) + int(s.numel()) * int(s.element_size())
                for c, s, _e in parts
            )
        )
        codes = torch.cat([c for c, _s, _e in parts])
        scale = torch.cat([s for _c, s, _e in parts])
    else:
        codes, scale = parts[0][0], parts[0][1]
    return codes, scale, row_scale, None if first is None else first.to(torch.float32).reshape(-1)

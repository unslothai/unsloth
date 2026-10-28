# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torchao-free weight-only int8 / fp8 for the diffusion transformer (ROCm, Windows torchao stub).

Plain buffers, unlike torchao tensors, survive the offload hooks' ``Module.to()``; torch imported lazily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

NATIVE_INT8 = "int8"
NATIVE_FP8 = "fp8"
_QMAX = {NATIVE_INT8: 127.0, NATIVE_FP8: 448.0}
NATIVE_INT8_ACT_ENV = "UNSLOTH_NATIVE_INT8_ACT"
NATIVE_INT8_ROT_ENV = "UNSLOTH_NATIVE_INT8_ROT"
# Apple Silicon int8 weight-only via the Metal ``_weight_int8pack_mm`` kernel. Unset: a load-time probe decides;
# ``0`` keeps the dequantise path; ``1`` takes the kernel without probing.
NATIVE_MPS_INT8PACK_ENV = "UNSLOTH_DIFFUSION_MPS_INT8PACK"
# The Metal kernel asserts N % 32 == 0 and K % 32 == 0.
_INT8PACK_ALIGN = 32
# The kernel runs near fp32 GEMM speed and is not bit-reproducible run to run, so it is taken only where it beats the
# chip's own bf16 GEMM by this much (M1: about 1.3x; M3 parity; M5 4x behind, as its 16-bit GEMM uses matrix units).
_INT8PACK_MIN_SPEEDUP = 1.10
_INT8PACK_PROBE_ROWS = 1024
# (dtype, in_features, out_features) of the probed layer -> verdict for this process (one chip per process). Each
# quantised transformer's layers carry their own verdict, so a later model with other shapes is probed afresh.
_INT8PACK_PROBED: dict = {}
# torch._int_mm needs M > 16 and K, N multiples of 8.
_INT_MM_MIN_ROWS = 17
_INT_MM_CHUNK_ELEMS = 1 << 25
# Compiled linear holds int32 acc + output chunk (6 B/elem); eager's row budget keeps its peak <= eager's (#15183 used 4x).
_INT_MM_FUSED_CHUNK_ELEMS = _INT_MM_CHUNK_ELEMS
# ``0`` keeps eager CUDA W8A8 kernels instead of one compiled function per linear.
NATIVE_INT8_FUSED_ENV = "UNSLOTH_NATIVE_INT8_FUSED"
# fp8 codes a native fp8 layer may hold; e5m2 only from a checkpoint (the quantise path writes e4m3fn).
NATIVE_FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")


def fp8_bits_to_float32(bits: Any, fmt: str = "float8_e4m3fn") -> Any:
    """uint8 fp8 codes to float32 by integer ops, bit-exact; Triton has no e4m3fn convert below sm_89."""
    import torch

    b = bits.to(torch.int32)
    if fmt == "float8_e5m2":
        exp_bits, man_bits, bias = 5, 2, 15
    else:
        exp_bits, man_bits, bias = 4, 3, 7
    e = (b >> man_bits) & ((1 << exp_bits) - 1)
    m = b & ((1 << man_bits) - 1)
    normal = (((e + (127 - bias)) << 23) | (m << (23 - man_bits))).view(torch.float32)
    # subnormal: m * 2**(1 - bias - man_bits), exact in float32
    sub = m.to(torch.float32) * (2.0 ** (1 - bias - man_bits))
    mag = torch.where(e == 0, sub, normal)
    if fmt == "float8_e5m2":
        top = e == (1 << exp_bits) - 1
        mag = torch.where(top, torch.where(m == 0, float("inf"), float("nan")), mag)
    else:
        mag = torch.where((e == 15) & (m == 7), float("nan"), mag)
    return torch.where((b & 0x80) != 0, -mag, mag)


@lru_cache(maxsize = None)
def fp8_lookup_table(fmt: str, device: str) -> Any:
    """float32 value of every fp8 code on ``device``. MPS has no fp8 casts before torch 2.15; indexing this table
    with the uint8 codes gives the same bits as the cast on any torch."""
    import torch
    return torch.arange(256, dtype = torch.uint8).view(getattr(torch, fmt)).float().to(device)


def int8_act_requested() -> bool:
    import os
    return os.environ.get(NATIVE_INT8_ACT_ENV, "").strip() == "1"


def int8_act_disabled() -> bool:
    import os
    return os.environ.get(NATIVE_INT8_ACT_ENV, "").strip() == "0"


def mps_int8pack_disabled() -> bool:
    import os
    return os.environ.get(NATIVE_MPS_INT8PACK_ENV, "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def mps_int8pack_forced() -> bool:
    import os
    return os.environ.get(NATIVE_MPS_INT8PACK_ENV, "").strip().lower() in ("1", "on", "true", "yes")


def mps_int8pack_selected(dtype: Any, layer: Any = None) -> bool:
    """Whether ``dtype`` forwards of ``layer`` may take the Metal kernel: the env override, else the verdict its
    transformer's probe stamped on it (no probe yet = dequantise, which is deterministic). Without ``layer``: whether
    any probe this process chose the kernel for ``dtype``."""
    if mps_int8pack_disabled():
        return False
    if mps_int8pack_forced():
        return True
    if layer is not None:
        return bool(getattr(layer, "_int8pack_verdict", {}).get(dtype, False))
    return any(v for (d, _, _), v in _INT8PACK_PROBED.items() if d == dtype)


def _median_call_s(
    fn: Callable[[], Any],
    sync: Callable[[], None],
    reps: int = 5,
) -> float:
    import time

    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _bench_int8pack_vs_dense(layer: Any, dtype: Any, device: Any) -> tuple:
    """(kernel s, dense ``dtype`` F.linear s) per call on ``layer``'s real shape, arms interleaved over 3 rounds."""
    import torch

    w_q = layer.weight_q.detach().to(device)
    scale = layer.weight_scale.view(torch.float32).to(device).to(dtype).contiguous()
    w = (w_q.to(dtype) * scale.unsqueeze(1)).contiguous()
    x = torch.randn(_INT8PACK_PROBE_ROWS, layer.in_features, dtype = dtype, device = device)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    kernel = lambda: torch._weight_int8pack_mm(x, w_q, scale)
    dense = lambda: torch.nn.functional.linear(x, w)
    kernel()
    dense()
    k_s, d_s = [], []
    for _ in range(3):
        k_s.append(_median_call_s(kernel, sync))
        d_s.append(_median_call_s(dense, sync))
    k_s.sort()
    d_s.sort()
    return k_s[1], d_s[1]


def probe_mps_int8pack(
    module: Any,
    dtype: Any,
    *,
    device: Any = "mps",
    logger: Any = None,
) -> bool:
    """Time the Metal kernel against a dense ``dtype`` GEMM on ``module``'s largest kernel-eligible layer and stamp
    the verdict on every eligible layer of ``module``. Verdicts are cached per (dtype, probed shape), so another model
    with the same largest layer reuses it and one with a different shape is probed. The env override skips the probe."""
    import torch

    if mps_int8pack_disabled() or mps_int8pack_forced():
        return mps_int8pack_selected(dtype)
    cls = native_linear_class()
    layers = [
        m
        for m in module.modules()
        if isinstance(m, cls) and m._int8pack_layer_ok(dtype, probe = True)
    ]
    if not layers:
        return False
    layer = max(layers, key = lambda m: m.in_features * m.out_features)
    key = (dtype, layer.in_features, layer.out_features)

    def stamp(verdict: bool) -> bool:
        for m in layers:
            m._int8pack_verdict = {**getattr(m, "_int8pack_verdict", {}), dtype: verdict}
        return verdict

    if key in _INT8PACK_PROBED:
        return stamp(_INT8PACK_PROBED[key])
    try:
        kernel_s, dense_s = _bench_int8pack_vs_dense(layer, dtype, torch.device(device))
    except Exception as exc:  # noqa: BLE001 -- a failed probe keeps the dequantise path
        if logger is not None:
            logger.info("diffusion.native_quant: int8pack probe failed, dequantising: %s", exc)
        _INT8PACK_PROBED[key] = False
        return stamp(False)
    speedup = dense_s / max(kernel_s, 1e-9)
    verdict = speedup >= _INT8PACK_MIN_SPEEDUP
    _INT8PACK_PROBED[key] = verdict
    stamp(verdict)
    if logger is not None:
        logger.info(
            "diffusion.native_quant: int8pack probe %dx%dx%d: kernel %.3f ms, dense %s %.3f ms (%.2fx) -> %s",
            _INT8PACK_PROBE_ROWS,
            layer.in_features,
            layer.out_features,
            kernel_s * 1e3,
            str(dtype).replace("torch.", ""),
            dense_s * 1e3,
            speedup,
            "Metal int8 kernel" if verdict else "dequantise",
        )
    return verdict


def snap_scale_up(scale: Any, dtype: Any) -> Any:
    """``scale`` rounded up to a ``dtype``-exact fp32 value: the Metal kernel takes ``dtype`` scales, so a call-time
    cast would add per-row gain error (up to 2**-9 in bf16); rounding up never clips a code."""
    import torch

    # nextafter is not implemented on every backend.
    host = scale.detach().to("cpu", torch.float32)
    cast = host.to(dtype)
    up = torch.nextafter(cast, torch.full_like(cast, float("inf")))
    return torch.where(cast.float() < host, up, cast).float().to(scale.device)


def int8_rotation_group() -> int:
    import os

    if os.environ.get(NATIVE_INT8_ROT_ENV, "").strip() == "0":
        return 0
    from .diffusion_convrot import DEFAULT_CONVROT_GROUPSIZE

    return DEFAULT_CONVROT_GROUPSIZE


@lru_cache(maxsize = 1)
def _row_major_op() -> Any:
    """Row-major copy Inductor cannot re-layout; cuBLASLt int8 GEMMs fail on a column-major activation."""
    import torch

    ns = getattr(torch.ops, "unsloth_studio", None)
    if ns is not None and hasattr(ns, "native_row_major"):
        return ns.native_row_major
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            "unsloth_studio::native_row_major", mutates_args = (), schema = "(Tensor x) -> Tensor"
        )
        def _native_row_major(x):
            return x.clone(memory_format = torch.contiguous_format)

        @_native_row_major.register_fake
        def _(x):
            return x.new_empty(x.shape)
    except Exception:  # noqa: BLE001 - a registration failure keeps the plain copy
        return None
    return torch.ops.unsloth_studio.native_row_major


def int8_fused_disabled() -> bool:
    import os
    return os.environ.get(NATIVE_INT8_FUSED_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _w8a8_rows(x2: Any, rot_h: Any, weight_q: Any, weight_scale: Any, bias: Any) -> Any:
    """Eager ``_rotate`` + ``_int_mm_rows`` maths on tensors only, so offload hooks never invalidate the graph."""
    import torch

    if rot_h is not None:
        g = rot_h.shape[0]
        x2 = (x2.reshape(x2.shape[0], -1, g) @ rot_h.to(x2.dtype)).reshape(x2.shape)
    x_scale = torch.linalg.vector_norm(
        x2, ord = float("inf"), dim = 1, keepdim = True, dtype = torch.float32
    ).clamp(min = 1e-12)
    # fp64 reciprocal: matches eager CUDA bit-for-bit; Triton's fp32 reciprocal is approximate and drifts codes.
    x_scale = x_scale * (1.0 / 127.0)
    x_recip = (1.0 / x_scale.double()).float()
    xq = torch.mul(x2, x_recip).round().clamp(-127, 127).to(torch.int8)
    if rot_h is None:
        op = _FUSED_STATE.get("row_major")
        xq = op(xq) if op is not None else xq.contiguous()
    acc = torch._int_mm(xq, weight_q.t())
    out = torch.mul(acc, x_scale)
    w_scale = weight_scale.view(torch.float32)
    if bias is not None:
        out = torch.addcmul(bias.float(), out, w_scale)
    else:
        out = out * w_scale
    return out.to(x2.dtype)


def _w8a8_rows_into(
    x2: Any, rot_h: Any, weight_q: Any, weight_scale: Any, bias: Any, out: Any
) -> None:
    """``_w8a8_rows`` stored straight into ``out`` (a row slice of the chunked output), no temporary chunk result."""
    out.copy_(_w8a8_rows(x2, rot_h, weight_q, weight_scale, bias))


_FUSED_STATE: dict = {"fn": None, "failed": None}


def _fused_w8a8() -> Optional[Callable[..., Any]]:
    """Compiled ``_w8a8_rows``, or None if disabled, no inductor runtime, ROCm, or already failed."""
    if _FUSED_STATE["failed"] is not None or int8_fused_disabled():
        return None
    fn = _FUSED_STATE["fn"]
    if fn is not None:
        return fn
    import torch

    from .rocm_bf16 import is_rocm_torch

    try:
        from .diffusion_speed import torch_compile_runtime_available
        runtime = torch_compile_runtime_available()
    except Exception:  # noqa: BLE001 - unknown runtime: stay eager
        runtime = False
    if not runtime or is_rocm_torch(torch):
        _FUSED_STATE["failed"] = "no inductor runtime" if not runtime else "ROCm"
        return None
    # Registered in eager, before any trace reads the handle.
    _FUSED_STATE["row_major"] = _row_major_op()
    # Every key exists before the first trace: older dynamo guards on len(_FUSED_STATE), so adding
    # "calls" after the first call would recompile each compiled function once.
    _FUSED_STATE.setdefault("calls", 0)
    fn = torch.compile(_w8a8_rows, dynamic = True, fullgraph = True)
    # One graph for every slice offset and chunk height (dynamic storage offset), written in place by inductor.
    _FUSED_STATE["into"] = torch.compile(_w8a8_rows_into, dynamic = True, fullgraph = True)
    _FUSED_STATE["fn"] = fn
    return fn


def _fused_w8a8_failed(exc: BaseException) -> None:
    import logging

    _FUSED_STATE["failed"] = repr(exc)
    _FUSED_STATE["fn"] = None
    _FUSED_STATE["into"] = None
    logging.getLogger(__name__).warning(
        "diffusion.native_quant: the compiled int8 W8A8 linear failed (%s); eager kernels from here on "
        "(set %s=0 to skip it)",
        exc,
        NATIVE_INT8_FUSED_ENV,
    )


def _autocast_on(x: Any) -> bool:
    import torch
    try:
        return bool(torch.is_autocast_enabled(x.device.type))
    except TypeError:  # torch without the device_type argument: treat any autocast as on
        return bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled())


def fused_w8a8_state() -> dict:
    return {"calls": _FUSED_STATE.get("calls", 0), "failed": _FUSED_STATE["failed"]}


@lru_cache(maxsize = 1)
def native_linear_class():
    """``NativeWeightOnlyLinear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class NativeWeightOnlyLinear(nn.Module):
        """A Linear whose weight is stored as int8 / fp8 with a per-output-row scale."""

        def __init__(
            self,
            linear: Any,
            scheme: str,
            act_int8: bool = False,
            rot_group: int = 0,
            codes: Any = None,
            scale: Any = None,
            scale_dtype: Any = None,
        ):
            """``codes`` / ``scale``: int8 (or, for the fp8 scheme, float8_e4m3fn) codes [out, in] and per-row
            scales made elsewhere, stored as given (``linear`` then supplies only the shapes, dtype and bias).
            With ``rot_group`` int8 codes are already ConvRot-rotated, so the rotation is installed whether or
            not activations go int8. ``scale_dtype``: snap int8 scales exact in that dtype (``snap_scale_up``)."""
            super().__init__()
            if scheme not in _QMAX:
                raise ValueError(f"unsupported native scheme {scheme!r}")
            self.in_features = int(linear.in_features)
            self.out_features = int(linear.out_features)
            self.scheme = scheme
            self.compute_dtype = linear.weight.dtype
            self.act_int8 = bool(act_int8) and scheme == NATIVE_INT8
            self.rot_group = (
                int(rot_group)
                if self.act_int8 and rot_group and self.in_features % int(rot_group) == 0
                else 0
            )
            self.fp8_format = "float8_e4m3fn"
            if codes is not None:
                wanted = torch.int8 if scheme == NATIVE_INT8 else torch.float8_e4m3fn
                if (
                    scheme == NATIVE_FP8
                    and str(codes.dtype).replace("torch.", "") in NATIVE_FP8_FORMATS
                ):
                    wanted = codes.dtype
                    self.fp8_format = str(codes.dtype).replace("torch.", "")
                if codes.dtype != wanted:
                    raise ValueError(
                        f"pre-quantized native {scheme} codes must be {str(wanted).replace('torch.', '')}"
                    )
                if rot_group and scheme != NATIVE_INT8:
                    raise ValueError("a ConvRot rotation needs int8 codes")
                self.rot_group = int(rot_group or 0)
                if self.rot_group and self.in_features % self.rot_group:
                    raise ValueError(
                        f"rotation group {self.rot_group} does not divide in_features {self.in_features}"
                    )
                wq, scale = codes, scale.reshape(-1, 1)
                if self.rot_group:
                    from .diffusion_convrot import build_convrot_hadamard
                    h = build_convrot_hadamard(
                        self.rot_group, device = codes.device, dtype = torch.float32
                    )
                    self.register_buffer("rot_h", h.to(self.compute_dtype), persistent = False)
            else:
                weight = linear.weight.detach()
                with torch.no_grad():
                    w = weight.float()
                    if self.rot_group:
                        from .diffusion_convrot import build_convrot_hadamard

                        h = build_convrot_hadamard(
                            self.rot_group, device = w.device, dtype = torch.float32
                        )
                        g = self.rot_group
                        w = (w.reshape(self.out_features, -1, g) @ h.T).reshape(
                            self.out_features, -1
                        )
                        self.register_buffer("rot_h", h.to(self.compute_dtype), persistent = False)
                    scale = w.abs().amax(dim = 1, keepdim = True).clamp(min = 1e-12) / _QMAX[scheme]
                    if scheme == NATIVE_INT8 and scale_dtype is not None:
                        scale = snap_scale_up(scale, scale_dtype)
                    if scheme == NATIVE_INT8:
                        wq = (w / scale).round_().clamp_(-127, 127).to(torch.int8)
                    else:
                        wq = (w / scale).to(torch.float8_e4m3fn)
                    del w
            # Integer views: a module-wide ``.to(dtype)`` casts floating buffers, which would widen fp8 or round the scales.
            self.register_buffer("weight_q", wq.view(torch.uint8) if scheme == NATIVE_FP8 else wq)
            self.register_buffer(
                "weight_scale", scale.squeeze(1).to(torch.float32).contiguous().view(torch.int32)
            )
            self.bias = linear.bias
            # Registered here, in eager, so a compiled forward only reads the handle (see _row_major_op).
            self._row_major = _row_major_op() if self.act_int8 and not self.rot_group else None
            self._int8pack_scales: dict = {}

        def _stored_weight(self, dtype: Any) -> Any:
            scale = self.weight_scale.view(torch.float32)
            if self.scheme != NATIVE_FP8:
                wide = self.weight_q.to(torch.float32)
            elif torch.compiler.is_compiling():
                wide = fp8_bits_to_float32(self.weight_q, self.fp8_format)
            elif self.weight_q.device.type == "mps":
                lut = fp8_lookup_table(self.fp8_format, str(self.weight_q.device))
                wide = lut.index_select(0, self.weight_q.reshape(-1).int()).view(
                    self.weight_q.shape
                )
            else:
                wide = self.weight_q.view(getattr(torch, self.fp8_format)).to(torch.float32)
            return (wide * scale[:, None]).to(dtype)

        def _rotate(self, x: Any) -> Any:
            g = self.rot_group
            # Leading dims, not -1: a zero-row input has no inferable -1 on older torch.
            grouped = x.reshape(*x.shape[:-1], self.in_features // g, g)
            return (grouped @ self.rot_h.to(x.dtype)).reshape(x.shape)

        def dequantized_weight(self, dtype: Any) -> Any:
            w = self._stored_weight(torch.float32 if self.rot_group else dtype)
            if not self.rot_group:
                return w
            g = self.rot_group
            h = self.rot_h.to(torch.float32)
            return (
                (w.reshape(self.out_features, -1, g) @ h).reshape(self.out_features, -1).to(dtype)
            )

        @property
        def weight(self) -> Any:
            """The dense weight, rebuilt on each read. Code that reads a wrapped Linear's weight directly
            (PEFT's DoRA forward reads ``base_layer.weight``) keeps working; nothing holds it."""
            return self.dequantized_weight(self.compute_dtype)

        def _int_mm_ok(self, rows: int) -> bool:
            return (
                rows >= _INT_MM_MIN_ROWS
                and self.in_features % 8 == 0
                and self.out_features % 8 == 0
            )

        def _int_mm_rows(self, x2: Any) -> Any:
            # Must stay fast in eager (Windows ROCm has no Triton): upcasts ride mixed-dtype ops, no .float() copies.
            x_scale = torch.linalg.vector_norm(
                x2, ord = float("inf"), dim = 1, keepdim = True, dtype = torch.float32
            )
            x_scale = x_scale.clamp_(min = 1e-12).div_(127.0)
            xq = torch.mul(x2, x_scale.reciprocal()).round_().clamp_(-127, 127).to(torch.int8)
            # A rotated input is a matmul output, row-major; an un-rotated one follows its producer's layout.
            if not self.rot_group and (torch.compiler.is_compiling() or not xq.is_contiguous()):
                op = getattr(self, "_row_major", None)
                xq = op(xq) if op is not None else xq.contiguous()
            acc = torch._int_mm(xq, self.weight_q.t())
            w_scale = self.weight_scale.view(torch.float32)
            out = torch.mul(acc, x_scale)
            if self.bias is not None:
                out = torch.addcmul(self.bias.float(), out, w_scale)
            else:
                out = out.mul_(w_scale)
            return out

        def _forward_int_mm(self, x: Any) -> Any:
            x2 = x.reshape(-1, self.in_features)
            rows = x2.shape[0]
            # Eager materialises fp32 [rows, features] temporaries (inductor fuses them): cap them per chunk.
            chunk = _INT_MM_CHUNK_ELEMS // max(self.in_features, self.out_features)
            if torch.compiler.is_compiling() or rows <= chunk or chunk < 2 * _INT_MM_MIN_ROWS:
                out = self._int_mm_rows(x2).to(x.dtype)
            else:
                out = torch.cat(
                    [
                        self._int_mm_rows(part).to(x.dtype)
                        for part in torch.tensor_split(x2, -(-rows // chunk))
                    ]
                )
            return out.reshape(*x.shape[:-1], self.out_features)

        def _forward_fused(self, x: Any) -> Any:
            """Chunked compiled forward; None = use eager kernels (also during CUDA graph capture)."""
            in_f = self.in_features
            rows = x.numel() // max(1, in_f)
            if not self._int_mm_ok(rows) or (
                x.is_cuda and torch.cuda.is_current_stream_capturing()
            ):
                return None
            from .diffusion_bg_compile import eager_forced  # noqa: PLC0415

            # Detailer crops and in-flight background compiles must not compile on this thread (#15063).
            if eager_forced():
                return None
            fn = _fused_w8a8()
            if fn is None:
                return None
            # Detached: guards on a view's base / Parameter shape would recompile per layout / layer.
            x2 = x.reshape(-1, in_f).detach()
            rot_h = self.rot_h if self.rot_group else None
            bias = self.bias.detach() if self.bias is not None else None
            chunk = _INT_MM_FUSED_CHUNK_ELEMS // max(in_f, self.out_features)
            try:
                if rows <= chunk or chunk < 2 * _INT_MM_MIN_ROWS:
                    out = fn(x2, rot_h, self.weight_q, self.weight_scale, bias)
                else:
                    # Only the rotation matmul can change the dtype (autocast); without it the output is x2's dtype.
                    into = _FUSED_STATE.get("into")
                    if into is not None and (rot_h is None or not _autocast_on(x2)):
                        out = x2.new_empty(rows, self.out_features)
                    else:
                        into = out = None
                    start = 0
                    for part in torch.tensor_split(x2, -(-rows // chunk)):
                        stop = start + part.shape[0]
                        if into is not None:
                            into(
                                part.detach(),
                                rot_h,
                                self.weight_q,
                                self.weight_scale,
                                bias,
                                out[start:stop],
                            )
                        else:
                            got = fn(part.detach(), rot_h, self.weight_q, self.weight_scale, bias)
                            if out is None:
                                # Allocated from the first chunk to keep the compiled dtype (autocast can differ).
                                out = got.new_empty(rows, self.out_features)
                            out[start:stop].copy_(got)
                            del got
                        start = stop
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 - a compile failure falls back to the eager kernels
                _fused_w8a8_failed(exc)
                return None
            _FUSED_STATE["calls"] = _FUSED_STATE.get("calls", 0) + 1
            return out.reshape(*x.shape[:-1], self.out_features)

        def _int8pack_scale(self, dtype: Any, device: Any) -> Any:
            """Per-row scale in ``dtype`` if that cast is exact (cached), else None: keep dequantise."""
            key = (dtype, str(device))
            # Keyed on the buffer's storage and version so load_state_dict / copy_ into it is never served stale.
            stamp = (self.weight_scale.data_ptr(), self.weight_scale._version)
            hit = self._int8pack_scales.get(key)
            if hit is None or hit[0] != stamp:
                scale = self.weight_scale.view(torch.float32)
                cast = scale.to(dtype)
                exact = bool(torch.equal(cast.float(), scale))
                hit = (stamp, cast.contiguous() if exact else None)
                self._int8pack_scales[key] = hit
            return hit[1]

        def _int8pack_layer_ok(
            self,
            dtype: Any,
            probe: bool = False,
        ) -> bool:
            """``probe``: the layer's shape and codes suit the kernel, whatever the probe decided."""
            return (
                self.scheme == NATIVE_INT8
                and not self.act_int8
                and not self.rot_group
                and dtype in (torch.bfloat16, torch.float16, torch.float32)
                and self.in_features % _INT8PACK_ALIGN == 0
                and self.out_features % _INT8PACK_ALIGN == 0
                and hasattr(torch, "_weight_int8pack_mm")
                and (
                    self._int8pack_scale(dtype, self.weight_q.device) is not None
                    if probe
                    else mps_int8pack_selected(dtype, self)
                )
            )

        def _int8pack_ok(self, x: Any) -> bool:
            return x.device.type == "mps" and x.numel() > 0 and self._int8pack_layer_ok(x.dtype)

        def int8pack_engaged(self, dtype: Any) -> bool:
            """Whether a ``dtype`` forward on the stored device takes the Metal int8 kernel."""
            device = self.weight_q.device
            return (
                device.type == "mps"
                and self._int8pack_layer_ok(dtype)
                and self._int8pack_scale(dtype, device) is not None
            )

        def _forward_int8pack(self, x: Any) -> Any:
            """None = take the dequantise path."""
            scale = self._int8pack_scale(x.dtype, x.device)
            if scale is None:
                return None
            x2 = x.reshape(-1, self.in_features)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            out = torch._weight_int8pack_mm(x2, self.weight_q, scale)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*x.shape[:-1], self.out_features)

        def forward(self, x: Any) -> Any:
            if self._int8pack_ok(x):
                out = self._forward_int8pack(x)
                if out is not None:
                    return out
            if self.act_int8 and x.is_cuda and not torch.compiler.is_compiling():
                out = self._forward_fused(x)
                if out is not None:
                    return out
            if self.rot_group:
                x = self._rotate(x)
            if (
                self.act_int8
                and x.is_cuda
                and self._int_mm_ok(x.numel() // max(1, self.in_features))
            ):
                return self._forward_int_mm(x)
            return F.linear(x, self._stored_weight(x.dtype), self.bias)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, act_int8={self.act_int8}, rot_group={self.rot_group}, bias={self.bias is not None}"
                + (f", codes={self.fp8_format}" if self.scheme == NATIVE_FP8 else "")
            )

    return NativeWeightOnlyLinear


def is_native_linear(module: Any) -> bool:
    return type(module).__name__ == "NativeWeightOnlyLinear" and hasattr(module, "weight_q")


def is_native_quantised(module: Any) -> bool:
    """Whether any layer of ``module`` has been swapped for a native weight-only Linear."""
    try:
        return any(is_native_linear(sub) for sub in module.modules())
    except Exception:  # noqa: BLE001 -- a probe must never replace the failure it describes
        return False


def apply_native_weight_quant(
    transformer: Any,
    scheme: str,
    *,
    filter_fn: Callable[[Any, str], bool],
    act_int8: bool,
    logger: Any = None,
    scale_dtype: Any = None,
) -> int:
    """Swap every Linear ``filter_fn`` keeps for a native weight-only twin; returns layers swapped. One layer
    at a time; a failure part-way leaves a partial conversion the caller must refuse."""
    cls = native_linear_class()
    act_int8 = scheme == NATIVE_INT8 and bool(act_int8)
    rot_group = int8_rotation_group() if act_int8 else 0
    # Names only: holding the old modules would keep every bf16 weight alive until the pass ends.
    names = [name for name, mod in transformer.named_modules() if name and filter_fn(mod, name)]
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(
            parent,
            leaf,
            cls(
                getattr(parent, leaf),
                scheme,
                act_int8 = act_int8,
                rot_group = rot_group,
                scale_dtype = scale_dtype if scheme == NATIVE_INT8 and not act_int8 else None,
            ),
        )
    if logger is not None:
        logger.info(
            "diffusion.transformer_quant: %s %s (torchao-free) on %d linears",
            scheme,
            ("W8A8 torch._int_mm" + (f" ConvRot g{rot_group}" if rot_group else ""))
            if act_int8
            else "weight-only",
            len(names),
        )
    return len(names)


def _first_native_linear(module: Any) -> Any:
    try:
        return next((sub for sub in module.modules() if is_native_linear(sub)), None)
    except Exception:  # noqa: BLE001 -- a probe must never raise
        return None


def _rotation_group(module: Any, layer: Any) -> int:
    # Every layer: rotation skips widths the group does not divide, so the first layer alone can read unrotated.
    try:
        return max(
            int(getattr(sub, "rot_group", 0) or 0)
            for sub in module.modules()
            if is_native_linear(sub)
        )
    except Exception:  # noqa: BLE001 -- a probe must never raise
        return int(getattr(layer, "rot_group", 0) or 0)


def native_quant_signature(module: Any) -> Optional[str]:
    layer = _first_native_linear(module)
    if layer is None:
        return None
    if not getattr(layer, "act_int8", False):
        return f"{layer.scheme}-wo"
    rot = _rotation_group(module, layer)
    return f"{layer.scheme}-w8a8" + (f"-rot{rot}" if rot else "")


def _int8pack_reported(layer: Any) -> bool:
    engaged = getattr(layer, "int8pack_engaged", None)
    if engaged is None or getattr(getattr(layer, "weight_q", None), "device", None) is None:
        return False
    import torch

    bias = getattr(layer, "bias", None)
    try:
        return bool(engaged(bias.dtype if bias is not None else torch.bfloat16))
    except Exception:  # noqa: BLE001 -- a status line must not fail the load
        return False


def native_quant_reason(module: Any, scheme: str) -> str:
    layer = _first_native_linear(module)
    if layer is not None and getattr(layer, "act_int8", False):
        rot = _rotation_group(module, layer)
        return (
            f"W8A8: {scheme} weights and activations through torch._int_mm"
            + (f" with a g{rot} ConvRot rotation" if rot else "")
            + " (torchao-free; plain buffers, so the offload hooks can move it)"
        )
    if layer is not None and _int8pack_reported(layer):
        return (
            f"weight-only: {scheme} weights through the Apple Silicon int8 kernel, which measured faster than "
            "bf16 here (torchao-free; half the weight memory; not bit-reproducible run to run, "
            f"{NATIVE_MPS_INT8PACK_ENV}=0 for repeatable images)"
        )
    return (
        f"weight-only: {scheme} weights, bf16 compute "
        "(torchao-free, a memory saving rather than a speed-up)"
    )


def native_weight_error(linear: Any, scheme: str) -> Optional[float]:
    """Relative Frobenius error of one Linear's native reconstruction; for tests and probes."""
    try:
        import torch

        layer = native_linear_class()(linear, scheme)
        ref = linear.weight.detach().float()
        rec = layer.dequantized_weight(torch.float32)
        return float((rec - ref).norm() / ref.norm().clamp(min = 1e-12))
    except Exception:  # noqa: BLE001
        return None

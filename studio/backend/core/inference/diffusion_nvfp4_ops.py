# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer NVFP4 kernels behind ``torch.library`` custom ops, plus the backend decision.

FlashInfer 0.6.6 registers no custom ops of its own, so Dynamo cannot treat its entry points as
opaque; wrapping them here is what makes ``fullgraph = True`` over a quantized block possible.
The fake impls must reproduce FlashInfer's allocation EXACTLY: a wrong meta shape is a silently
mis-sized buffer, not an error. torch and flashinfer are imported inside functions so the module
imports on a torch-free host.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

NVFP4_BACKEND_ENV = "UNSLOTH_NVFP4_BACKEND"
NVFP4_BACKENDS = ("auto", "torchao", "flashinfer")
BACKEND_TORCHAO = "torchao"
BACKEND_FLASHINFER = "flashinfer"

# Restores the full M x N memset in the GEMM op. Strictly slower and no safer; see ``_mm_impl``.
NVFP4_ZERO_BUFFER_ENV = "UNSLOTH_NVFP4_ZERO_BUFFER"

# Necessary condition only: sm_120 is listed but routinely fails the preflight on CUDA 12.8.
NVFP4_FLASHINFER_CAPS = frozenset({(10, 0), (10, 3), (12, 0)})

DEFAULT_MM_BACKEND = "cutlass"

FP4_MAX = 6.0
FP8_MAX = 448.0

_TRUE_TOKENS = ("1", "true", "yes", "on")

_REGISTER_LOCK = threading.Lock()
_REGISTERED = False
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHT: dict[int, dict] = {}
_WARNED: set = set()


def _swizzled_sf_numel(
    rows: int,
    cols: int,
    row_size: int = 128,
) -> int:
    """Element count of a swizzled scale-factor buffer. Copied, not imported: the fake impls must
    not touch the FlashInfer JIT machinery."""
    return ((rows + row_size - 1) // row_size * row_size) * ((cols + 3) // 4 * 4)


def _device_guard(t: Any):
    """``torch.cuda.device`` for the tensor's own device. EVERY flashinfer call must sit inside
    one: FlashInfer installs no device guard, and a foreign current device bricks the card."""
    import torch
    return torch.cuda.device(t.device)


def _zero_buffer_enabled() -> bool:
    return os.environ.get(NVFP4_ZERO_BUFFER_ENV, "").strip().lower() in _TRUE_TOKENS


def global_scale(t: Any):
    """The NVFP4 global scale of a tensor: ``6 * 448 / amax``, as a 1-element fp32 tensor."""
    return (FP4_MAX * FP8_MAX / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


# Exposed as plain functions as well as through ``torch.ops`` so that a test can assert the call
# ORDER inside them, which is what the barrier below rests on.


def _quantize_impl(x: Any, global_sf: Any):
    """2D bf16 in, ``(packed e2m1x2, swizzled block scales)`` out."""
    import flashinfer
    with _device_guard(x):
        return flashinfer.nvfp4_quantize(x, global_sf, do_shuffle = False)


def _mm_impl(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    """NVFP4 GEMM. Takes the weight row-major ``[N, K/2]`` and transposes inside, since a custom
    op's inputs are functionalised and a ``.T`` view is an alias Dynamo has to reason about.

    **A kernel MUST run between the activation quantiser and this GEMM**, or the GEMM reads
    operands the quantiser has not finished writing (cutlass is launched with PDL while the
    ``griddepcontrol`` instructions that make PDL safe are compiled out of its build, and
    ``enable_pdl = False`` is plumbed only to the cute-dsl runner). What protects it is a kernel
    EXISTING, so ``torch.zeros(1)`` is as good as the memset while ``torch.empty`` alone is NOT.
    The barrier buffer is allocated per call, never cached: a reused one turns one transient NaN
    into a permanent one for every later call at the same token count.
    """
    import flashinfer
    import torch

    with _device_guard(xq):
        m = xq.shape[0]
        if _zero_buffer_enabled():
            out = torch.zeros(m, n, device = xq.device, dtype = torch.bfloat16)
        else:
            out = torch.empty(m, n, device = xq.device, dtype = torch.bfloat16)
            torch.zeros(1, device = xq.device, dtype = torch.bfloat16)
        return flashinfer.mm_fp4(
            xq, wq.T, x_sf, w_sf.T, alpha, torch.bfloat16, out = out, backend = backend
        )


def _quantize_fake(x: Any, global_sf: Any):
    import torch

    m, k = x.shape
    cols = k // 16
    # The leading dim is NOT m whenever the padding is non-trivial. Reproduce that, not m.
    total = _swizzled_sf_numel(m, cols, 128)
    return (
        x.new_empty((m, k // 2), dtype = torch.uint8),
        x.new_empty((total // cols, cols), dtype = torch.uint8),
    )


def _mm_fake(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    import torch
    return xq.new_empty((xq.shape[0], n), dtype = torch.bfloat16)


def register_ops() -> None:
    """Register the two custom ops. Idempotent, thread safe, and deferred rather than done at
    import because this module has to import on a host with no torch."""
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        import torch

        if "unsloth_nvfp4::quantize" not in torch.library._defs:
            quantize = torch.library.custom_op(
                "unsloth_nvfp4::quantize",
                _quantize_impl,
                mutates_args = (),
                schema = "(Tensor x, Tensor global_sf) -> (Tensor, Tensor)",
            )
            quantize.register_fake(_quantize_fake)
        if "unsloth_nvfp4::mm" not in torch.library._defs:
            mm = torch.library.custom_op(
                "unsloth_nvfp4::mm",
                _mm_impl,
                mutates_args = (),
                schema = (
                    "(Tensor xq, Tensor wq, Tensor x_sf, Tensor w_sf, Tensor alpha, "
                    "int n, str backend) -> Tensor"
                ),
            )
            mm.register_fake(_mm_fake)
        _REGISTERED = True


def unswizzle_sf(
    sf: Any,
    m: int,
    k: int,
    vec: int = 16,
):
    """cutlass 128x4 swizzled block scales -> the plain ``[m, k // vec]`` e4m3 matrix."""
    import torch

    kb = k // vec
    m_pad = (m + 127) // 128 * 128
    k_pad = (kb + 3) // 4 * 4
    flat = sf.reshape(-1).view(torch.float8_e4m3fn)
    v = (
        flat.reshape(m_pad // 128, k_pad // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(m_pad, k_pad)
    )
    return v[:m, :kb].contiguous()


def swizzle_sf(sf_lin: Any, m: int, k: int):
    """Plain ``[m, k // 16]`` e4m3 block scales -> the cutlass 128x4 swizzled flat buffer."""
    import torch

    kb = k // 16
    m_pad = (m + 127) // 128 * 128
    k_pad = (kb + 3) // 4 * 4
    full = torch.zeros(m_pad, k_pad, device = sf_lin.device, dtype = torch.uint8)
    full[:m, :kb] = sf_lin.reshape(m, kb).view(torch.uint8)
    v = full.reshape(m_pad, k_pad // 4, 4).reshape(m_pad // 128, 4, 32, k_pad // 4, 4)
    return v.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


# The signed table is indexed by the raw nibble (``sign << 3 | magnitude``).
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_LUT = _E2M1_MAGNITUDES + tuple(-v for v in _E2M1_MAGNITUDES)

# Deliberately NOT a registered buffer: one copy per layer is the kind of byte that must not exist.
_LUT_CACHE: dict = {}


def e2m1_lut(device: Any):
    """The signed e2m1 decode table on ``device``, memoised so that it is never ALLOCATED inside a
    CUDA-graph capture: a buffer first created while recording is only valid while recording."""
    import torch

    key = (str(device.type), device.index)
    table = _LUT_CACHE.get(key)
    if table is None:
        table = torch.tensor(_E2M1_LUT, device = device, dtype = torch.float32)
        _LUT_CACHE[key] = table
    return table


def reset_lut_cache() -> None:
    """Forget the per-device decode tables. For tests and for a device set that changed."""
    _LUT_CACHE.clear()


def dequantize_nvfp4_weight(wq: Any, w_sf: Any, per_tensor_scale: Any, *, dtype: Any = None):
    """The packed NVFP4 operand as a dense ``[N, K]`` weight. TRANSIENT by contract: the caller
    must drop it, since caching it is the second resident operand this lever exists to avoid. The
    arithmetic is torchao's, in torchao's order, so the protected step reads the SAME weight the
    unprotected step does rather than a second quantisation of it."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    q = wq if wq.dtype == torch.uint8 else wq.view(torch.uint8)
    rows, half_k = int(q.shape[-2]), int(q.shape[-1])
    cols = half_k * 2
    blocks = cols // 16
    lut = e2m1_lut(q.device)
    # int32, not int64: an int64 gather index costs 8 bytes per 4-bit code.
    codes = q.to(torch.int32)
    values = torch.stack((lut[codes & 0x0F], lut[codes >> 4]), dim = -1).reshape(rows, blocks, 16)
    block_scale = unswizzle_sf(w_sf, rows, cols).to(torch.float32)
    if not isinstance(per_tensor_scale, torch.Tensor):
        per_tensor_scale = torch.tensor(
            [float(per_tensor_scale)], device = q.device, dtype = torch.float32
        )
    step = (block_scale * per_tensor_scale.to(torch.float32).reshape(1)).reshape(rows, blocks, 1)
    return (values * step).reshape(rows, cols).to(dtype)


def sf_matrix_shape(rows: int, cols: int) -> tuple[int, int]:
    """FlashInfer's 2D view of a swizzled scale buffer. The leading dim is NOT ``rows`` whenever
    the padding is non-trivial."""
    padded_cols = (cols + 3) // 4 * 4
    return (_swizzled_sf_numel(rows, cols, 128) // padded_cols, padded_cols)


def nvfp4_preflight(device: Any = None, *, refresh: bool = False) -> dict:
    """A tiny GUARDED 128x256 quantise + GEMM on ``device``, memoised per device index. Only asking
    the JIT to build finds out whether FlashInfer runs here. Never raises."""
    import torch

    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda unavailable", "capability": None, "name": ""}
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if dev.type != "cuda":
        return {"ok": False, "reason": f"device {dev} is not cuda", "capability": None, "name": ""}
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    dev = torch.device("cuda", index)
    if not refresh:
        cached = _PREFLIGHT.get(index)
        if cached is not None:
            return dict(cached)

    try:
        capability = tuple(torch.cuda.get_device_capability(dev))
        name = str(torch.cuda.get_device_name(dev))
    except Exception as exc:  # noqa: BLE001 - a device that cannot be described cannot be used
        capability, name = None, ""
        rec = {
            "ok": False,
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
            "capability": capability,
            "name": name,
        }
        with _PREFLIGHT_LOCK:
            _PREFLIGHT[index] = rec
        return dict(rec)

    rec = {"ok": False, "reason": "", "capability": capability, "name": name}
    try:
        import flashinfer

        with torch.cuda.device(dev):
            x = torch.randn(128, 256, device = dev, dtype = torch.bfloat16) * 0.05
            w = torch.randn(128, 256, device = dev, dtype = torch.bfloat16) * 0.02
            a_gsf, w_gsf = global_scale(x), global_scale(w)
            xq, x_sf = flashinfer.nvfp4_quantize(x, a_gsf, do_shuffle = False)
            wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
            out = torch.zeros(128, 128, device = dev, dtype = torch.bfloat16)
            y = flashinfer.mm_fp4(
                xq,
                wq.T,
                x_sf,
                w_sf.T,
                (1.0 / (a_gsf * w_gsf)).float(),
                torch.bfloat16,
                out = out,
                backend = DEFAULT_MM_BACKEND,
            )
            torch.cuda.synchronize(dev)
            finite = bool(torch.isfinite(y).all().item())
        rec["ok"] = finite
        rec["reason"] = "ok" if finite else "mm_fp4 produced a non-finite result"
    except Exception as exc:  # noqa: BLE001 - every failure mode here means "use torchao"
        rec["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    with _PREFLIGHT_LOCK:
        _PREFLIGHT[index] = rec
    return dict(rec)


def reset_preflight_cache() -> None:
    """Forget every memoised preflight. For tests and for a device set that changed under us."""
    with _PREFLIGHT_LOCK:
        _PREFLIGHT.clear()
    _WARNED.clear()


def nvfp4_backend_env() -> str:
    """The requested backend, normalised. An unrecognised value reads as ``auto``."""
    raw = os.environ.get(NVFP4_BACKEND_ENV, "").strip().lower()
    return raw if raw in NVFP4_BACKENDS else "auto"


def _flashinfer_available() -> tuple[bool, str]:
    """Whether ``import flashinfer`` works here. Windows answers no, which is the design."""
    try:
        import flashinfer
        return True, str(getattr(flashinfer, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001 - any import failure means torchao
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def _device_capability(device: Any = None) -> Optional[tuple]:
    try:
        import torch
        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability(device))
    except Exception:  # noqa: BLE001 - an unreadable capability is not a flashinfer device
        return None


def _resolve_backend(device: Any = None) -> tuple[str, str]:
    """``(backend, reason)``: the reason is reported so a fallback is readable without reproducing
    the probe."""
    requested = nvfp4_backend_env()
    if requested == BACKEND_TORCHAO:
        return BACKEND_TORCHAO, f"{NVFP4_BACKEND_ENV}=torchao"

    available, detail = _flashinfer_available()
    if not available:
        return BACKEND_TORCHAO, f"flashinfer unavailable ({detail})"
    capability = _device_capability(device)
    if capability is None:
        return BACKEND_TORCHAO, "no CUDA device capability to check"
    if tuple(capability) not in NVFP4_FLASHINFER_CAPS:
        return BACKEND_TORCHAO, (
            "sm_%d%d is not in the flashinfer NVFP4 set %s"
            % (
                capability[0],
                capability[1],
                ", ".join(sorted("sm_%d%d" % c for c in NVFP4_FLASHINFER_CAPS)),
            )
        )
    probe = nvfp4_preflight(device)
    if not probe.get("ok"):
        return BACKEND_TORCHAO, f"preflight failed: {probe.get('reason', 'unknown')}"
    return BACKEND_FLASHINFER, f"flashinfer {detail} preflight ok on sm_%d%d" % tuple(capability)


def nvfp4_backend_reason(device: Any = None) -> str:
    """Why ``select_nvfp4_backend`` will answer what it answers, in one line."""
    return _resolve_backend(device)[1]


def select_nvfp4_backend(device: Any = None) -> str:
    """``"torchao"`` or ``"flashinfer"`` for ``device``. An explicit ``flashinfer`` request that
    fails import, capability or preflight falls back to torchao rather than to a bricked card."""
    backend, reason = _resolve_backend(device)
    if backend != BACKEND_FLASHINFER and nvfp4_backend_env() == BACKEND_FLASHINFER:
        key = (str(device), reason)
        if key not in _WARNED:
            _WARNED.add(key)
            import warnings
            warnings.warn(
                f"{NVFP4_BACKEND_ENV}=flashinfer was requested but {reason}; using torchao",
                RuntimeWarning,
                stacklevel = 2,
            )
    return backend

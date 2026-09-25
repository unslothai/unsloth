# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer NVFP4 kernels as ``torch.library`` custom ops (so ``fullgraph = True`` works), plus
the backend decision. Fake impls must match FlashInfer's allocation EXACTLY: a wrong meta shape is
a silently mis-sized buffer. torch and flashinfer import lazily for torch-free hosts.
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
    """Copied, not imported: a fake impl must not touch the FlashInfer JIT machinery."""
    return ((rows + row_size - 1) // row_size * row_size) * ((cols + 3) // 4 * 4)


def _device_guard(t: Any):
    """EVERY flashinfer call needs this: it has no device guard, and a foreign device bricks the card."""
    import torch
    return torch.cuda.device(t.device)


def _zero_buffer_enabled() -> bool:
    return os.environ.get(NVFP4_ZERO_BUFFER_ENV, "").strip().lower() in _TRUE_TOKENS


def global_scale(t: Any):
    return (FP4_MAX * FP8_MAX / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


def _quantize_impl(x: Any, global_sf: Any):
    import flashinfer
    with _device_guard(x):
        return flashinfer.nvfp4_quantize(x, global_sf, do_shuffle = False)


def _mm_impl(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    """Weight row-major, transposed inside (a ``.T`` input is an alias). A KERNEL must run between
    the quantiser and this GEMM (cutlass PDL without griddepcontrol reads unfinished operands):
    ``torch.zeros(1)`` per call works, ``torch.empty`` does NOT."""
    import flashinfer
    import torch

    with _device_guard(xq):
        m = xq.shape[0]
        if _zero_buffer_enabled():
            out = torch.zeros(m, n, device = xq.device, dtype = torch.bfloat16)
        else:
            out = torch.empty(m, n, device = xq.device, dtype = torch.bfloat16)
            torch.zeros(1, device = xq.device, dtype = torch.bfloat16)
        if _claim_first_call_tune(m, xq.shape[1] * 2, n):
            try:
                with flashinfer.autotune(True):
                    return flashinfer.mm_fp4(
                        xq, wq.T, x_sf, w_sf.T, alpha, torch.bfloat16, out = out, backend = backend
                    )
            except Exception:  # noqa: BLE001 - claimed, so it is not retried; run it untuned
                pass
        return flashinfer.mm_fp4(
            xq, wq.T, x_sf, w_sf.T, alpha, torch.bfloat16, out = out, backend = backend
        )


def _claim_first_call_tune(m: int, k: int, n: int) -> bool:
    """True once per untuned ``(M, K, N)``, never under graph capture (profiling is illegal there)."""
    import torch

    from .diffusion_nvfp4_linear import _TUNED_SHAPES

    key = (int(m), int(k), int(n))
    if key in _TUNED_SHAPES or torch.cuda.is_current_stream_capturing():
        return False
    _TUNED_SHAPES.add(key)
    return True


def _quantize_fake(x: Any, global_sf: Any):
    import torch

    m, k = x.shape
    cols = k // 16
    # The leading dim is NOT m whenever the padding is non-trivial.
    total = _swizzled_sf_numel(m, cols, 128)
    return (
        x.new_empty((m, k // 2), dtype = torch.uint8),
        x.new_empty((total // cols, cols), dtype = torch.uint8),
    )


def _mm_fake(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    import torch
    return xq.new_empty((xq.shape[0], n), dtype = torch.bfloat16)


def register_ops() -> None:
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


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_LUT = _E2M1_MAGNITUDES + tuple(-v for v in _E2M1_MAGNITUDES)

_LUT_CACHE: dict = {}


def e2m1_lut(device: Any):
    """Memoised so it is never ALLOCATED under a graph capture (valid only while recording)."""
    import torch

    key = (str(device.type), device.index)
    table = _LUT_CACHE.get(key)
    if table is None:
        table = torch.tensor(_E2M1_LUT, device = device, dtype = torch.float32)
        _LUT_CACHE[key] = table
    return table


def reset_lut_cache() -> None:
    _LUT_CACHE.clear()


def dequantize_nvfp4_weight(
    wq: Any,
    w_sf: Any,
    per_tensor_scale: Any,
    *,
    dtype: Any = None,
):
    """Dense ``[N, K]`` weight, TRANSIENT (never cache it). Arithmetic in torchao's order, so it
    matches the unprotected step bit for bit."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    q = wq if wq.dtype == torch.uint8 else wq.view(torch.uint8)
    rows, half_k = int(q.shape[-2]), int(q.shape[-1])
    cols = half_k * 2
    blocks = cols // 16
    lut = e2m1_lut(q.device)
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
    padded_cols = (cols + 3) // 4 * 4
    return (_swizzled_sf_numel(rows, cols, 128) // padded_cols, padded_cols)


def nvfp4_preflight(device: Any = None, *, refresh: bool = False) -> dict:
    """A tiny guarded quantise + GEMM, memoised per device: only a JIT build proves FlashInfer runs."""
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    if not nvfp4_diffusion_enabled():
        # Not memoised: turning the switch on must not inherit a verdict nobody measured.
        return {
            "ok": False,
            "reason": "NVFP4 is disabled in this build",
            "capability": None,
            "name": "",
        }
    import torch

    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda unavailable", "capability": None, "name": ""}
    if device is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        # torch.device(int) asks the accelerator API, which raises on a host without one.
        dev = torch.device("cuda", device)
    else:
        dev = torch.device(device)
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
        finite = _preflight_probe(dev)
        rec["ok"] = finite
        rec["reason"] = "ok" if finite else "mm_fp4 produced a non-finite result"
    except Exception as exc:  # noqa: BLE001 - every failure mode here means "use torchao"
        rec["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if _transient_preflight_failure(exc):
            return dict(rec)

    with _PREFLIGHT_LOCK:
        _PREFLIGHT[index] = rec
    return dict(rec)


def _preflight_probe(dev: Any) -> bool:
    import flashinfer
    import torch
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
        return bool(torch.isfinite(y).all().item())


def _transient_preflight_failure(exc: BaseException) -> bool:
    """Allocation failures only; an import or JIT failure stays cached."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - no torch to ask means fall through to the text
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text


def reset_preflight_cache() -> None:
    with _PREFLIGHT_LOCK:
        _PREFLIGHT.clear()
    _WARNED.clear()


def nvfp4_backend_env() -> str:
    raw = os.environ.get(NVFP4_BACKEND_ENV, "").strip().lower()
    return raw if raw in NVFP4_BACKENDS else "auto"


def _flashinfer_available() -> tuple[bool, str]:
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
    from .diffusion_nvfp4_flag import NVFP4_DIFFUSION_ENV, nvfp4_diffusion_enabled

    if not nvfp4_diffusion_enabled():
        return BACKEND_TORCHAO, (
            f"NVFP4 is disabled in this build (set {NVFP4_DIFFUSION_ENV}=1 to enable it)"
        )
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
    return _resolve_backend(device)[1]


def select_nvfp4_backend(device: Any = None) -> str:
    """An explicit ``flashinfer`` that fails import, capability or preflight falls back to torchao."""
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11 (RDNA 3 / 3.5): route a video VAE's ``nn.Conv3d`` calls through the Triton direct / H,W-Winograd
conv3d vendored from ROCm/aiter (``rocm_triton_conv3d``, MIT) instead of MIOpen.

Measured on AMD CI Linux gfx1151 (Strix Halo), Wan 2.2 TI2V-5B VAE: a 704x1280x49 decode 51.9 -> 32.3 s (1.61x),
the cold first decode (MIOpen kernel search) 1004 -> 35 s, PSNR 54.7 dB against MIOpen, 1323 / 1323 conv calls
routed; through Studio, Wan 832x480x49 wall 66.4 -> 58.2 s. Decode peak memory rose 11.8 -> 18.7 GiB (Winograd
transform buffers), so every call first checks the card's free memory and stays on MIOpen when it cannot afford the
workspace.

Per instance, ``Conv3d._conv_forward`` is wrapped, which covers diffusers' ``WanCausalConv3d`` (its forward pads, then
calls ``Conv3d.forward``) and the plain ``nn.Conv3d`` inside ``LTX2VideoCausalConv3d``. A call stays on the stock
path (MIOpen) when: tracing under torch.compile, autograd would record it, not a CUDA / HIP tensor, not fp16 / bf16,
groups != 1, padding_mode != zeros, an index could pass int32 (``fallback_shape``), the workspace does not fit
(``fallback_memory``), or the kernel failed to compile / launch for that shape before (``fallback_error``; the shape is
not retried). Out-of-memory and torch._dynamo errors are re-raised.

``UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D``: ``1`` on for any gfx11 card, ``0`` off; unset = on only for
``DEFAULT_ON_ARCHES``. ``UNSLOTH_DIFFUSION_ROCM_AITER_CONV3D=1`` (the name the measurements used) is an alias for ``1``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

ENABLE_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D"
LEGACY_ENABLE_ENV = "UNSLOTH_DIFFUSION_ROCM_AITER_CONV3D"
SELFCHECK_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_SELFCHECK"
# Archs routed with the env unset (AMD CI gfx1151, Linux and Windows: Wan 2.2 5B render VAE 1.56x, wall -13%).
DEFAULT_ON_ARCHES: tuple = ("gfx1151",)
# VAE classes routed with the env unset (measured end to end); any other VAE needs the env var set to 1.
DEFAULT_ON_VAES: tuple = ("AutoencoderKLWan",)
# Archs whose kernel tables were tuned (ROCm/aiter PR #6973); other gfx11 archs borrow them (explicit opt-in only).
TUNED_ARCHES = ("gfx1151",)
# Free memory kept back beyond the estimated workspace: the larger of these.
MEM_MARGIN_BYTES = 512 << 20
MEM_MARGIN_FRACTION = 0.03
_INT32_MAX = 2**31 - 1
_BLOCK = 64

COUNTS = {"routed": 0, "fallback_error": 0, "fallback_shape": 0, "fallback_memory": 0}
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_LOCK = threading.Lock()
_STATE: dict = {}
_FAILED: set = set()
_DEVICE_OK: dict = {}


def reset_counts() -> None:
    for k in COUNTS:
        COUNTS[k] = 0


def _env_mode() -> Optional[bool]:
    raw = (os.environ.get(ENABLE_ENV) or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    if (os.environ.get(LEGACY_ENABLE_ENV) or "").strip().lower() in _TRUE:
        return True
    return None


def arch_enabled(arch: str) -> bool:
    """Whether a card of ``arch`` (e.g. ``gfx1151``, ``gfx1151:xnack-``) is routed under the current env."""
    arch = (arch or "").split(":")[0].strip().lower()
    if not arch.startswith("gfx11"):
        return False
    mode = _env_mode()
    if mode is None:
        return arch in DEFAULT_ON_ARCHES
    return mode


def requested() -> bool:
    """Cheap pre-check: False when no arch can be on (env 0, or env unset with no default arch)."""
    mode = _env_mode()
    return bool(DEFAULT_ON_ARCHES) if mode is None else mode


def _device_arch(torch: Any, index: Optional[int]) -> str:
    try:
        props = torch.cuda.get_device_properties(
            index if index is not None else torch.cuda.current_device()
        )
        return str(getattr(props, "gcnArchName", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _device_ok(torch: Any, index: Optional[int]) -> bool:
    key = index if index is not None else -1
    ok = _DEVICE_OK.get(key)
    if ok is None:
        from .rocm_bf16 import is_rocm_torch
        ok = bool(is_rocm_torch(torch)) and arch_enabled(_device_arch(torch, index))
        _DEVICE_OK[key] = ok
    return ok


def _kernels() -> Optional[Any]:
    """The vendored conv3d module, or None when Triton (or its HIP backend) is missing."""
    with _LOCK:
        if "mod" in _STATE:
            return _STATE["mod"]
        try:
            from .rocm_triton_conv3d import conv3d as mod
        except Exception as exc:  # noqa: BLE001 - no Triton: MIOpen stays
            mod = None
            _STATE["import_error"] = repr(exc)[:300]
        _STATE["mod"] = mod
        return mod


def _route_name(mod: Any, x_shape, w_shape, stride, padding, dilation) -> Any:
    N, C, D, H, W = x_shape
    K, _c, T, R, S = w_shape
    return mod._resolve_route(T, R, S, stride, dilation, N, C, D, H, W, K, "ncdhw", padding = padding)


def plan(
    mod: Any,
    x_shape,
    w_shape,
    elt: int,
    stride,
    padding,
    dilation,
    contiguous: bool = True,
) -> tuple:
    """(route, extra bytes the call allocates beyond MIOpen's output, largest element count any kernel indexes).

    y = N*K*OD*P*Q*elt (output, also MIOpen's); copy of x when not contiguous; per route:
      1x1x1      : the squeezed weight
      general    : K-major weight pack  K*ceil(C*T*R*S/64)*64*elt
      cblocked   : NCDHWc input pack    N*ceil(C/64)*64*D*H*W*elt + weight pack K*27*C_pad*elt
      winograd   : V 36*N*D*th*tw*C_pad*2 + M 36*N*OD*th*tw*K*4 + filter 3*36*K*C_pad*2 (+ its fp32 transform
                   3*36*K*C*4), th = ceil(P/4), tw = ceil(Q/4); the cblocked variant adds the NCDHWc input pack.
    """
    N, C, D, H, W = x_shape
    K, _c, T, R, S = w_shape
    OD, P, Q = mod._out_dhw(D, H, W, T, R, S, stride, padding, dilation)
    route = _route_name(mod, x_shape, w_shape, stride, padding, dilation)
    names = route.name
    c_pad = -(-C // _BLOCK) * _BLOCK
    x_numel = N * C * D * H * W
    y_numel = N * K * OD * P * Q
    extra = y_numel * elt + (0 if contiguous else x_numel * elt)
    biggest = max(x_numel, y_numel)
    if names == "ONE_X_ONE_X_ONE":
        extra += K * C * elt
    elif names == "GENERAL":
        extra += K * (-(-(C * T * R * S) // _BLOCK) * _BLOCK) * elt
    else:
        blocked = N * c_pad * D * H * W
        if names in ("CBLOCKED_NCDHW", "WINOGRAD_HW_CBLOCKED"):
            extra += blocked * elt
            biggest = max(biggest, blocked)
        if names == "CBLOCKED_NCDHW":
            extra += K * 27 * c_pad * elt
        else:
            tiles = ((P + 3) // 4) * ((Q + 3) // 4)
            v = 36 * N * D * tiles * c_pad
            m = 36 * N * OD * tiles * K
            extra += v * 2 + m * 4 + 3 * 36 * K * c_pad * 2 + 3 * 36 * K * C * 4
            biggest = max(biggest, v, m)
    return route, extra, biggest


def _free_bytes(torch: Any, device: Any) -> int:
    free, total = torch.cuda.mem_get_info(device)
    # Blocks the caching allocator holds but has not handed out are reusable by this call.
    slack = max(
        0, int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device))
    )
    return int(free) + slack, int(total)


def _selfcheck(torch: Any, mod: Any, device: Any, dtype: Any) -> bool:
    """One small conv per route against F.conv3d in fp32, once per (device, dtype); a miss keeps MIOpen."""
    key = ("selfcheck", str(device), str(dtype))
    if key in _STATE:
        return _STATE[key]
    ok = True
    if (os.environ.get(SELFCHECK_ENV) or "").strip().lower() not in _FALSE:
        import torch.nn.functional as F

        gen = torch.Generator(device = "cpu").manual_seed(0)
        cases = (  # 1x1x1, general (small C), cblocked, winograd H/W
            ((1, 32, 2, 6, 7), (48, 32, 1, 1, 1), (0, 0, 0)),
            ((1, 8, 3, 6, 6), (16, 8, 3, 3, 3), (1, 1, 1)),
            ((1, 64, 3, 6, 6), (64, 64, 3, 3, 3), (0, 1, 1)),
            ((1, 160, 3, 28, 28), (96, 160, 3, 3, 3), (1, 1, 1)),
        )
        try:
            with torch.no_grad():
                for xs, ws, pad in cases:
                    x = torch.randn(xs, generator = gen).to(device = device, dtype = dtype)
                    w = (torch.randn(ws, generator = gen) * 0.05).to(device = device, dtype = dtype)
                    b = torch.randn(ws[0], generator = gen).to(device = device, dtype = dtype)
                    y = mod.conv3d(x, w, b, (1, 1, 1), pad, (1, 1, 1))
                    ref = F.conv3d(x.float(), w.float(), b.float(), padding = pad)
                    rel = float((y.float() - ref).norm() / ref.norm())
                    if not (y.shape == ref.shape and rel < 2e-2):
                        ok = False
                        _STATE["selfcheck_error"] = f"{xs}x{ws}: rel L2 {rel:.3g}"
                        break
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - no Triton HIP backend, compile failure, ...
            ok = False
            _STATE["selfcheck_error"] = repr(exc)[:300]
    _STATE[key] = ok
    return ok


def _on_gpu(t: Any) -> bool:
    return bool(t.is_cuda)


def _reraise(torch: Any, exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return type(exc).__module__.startswith("torch._dynamo")


def _make_conv_forward(module: Any, original: Any, logger: Any):
    import torch

    def _conv_forward(input, weight, bias):
        if (
            torch.compiler.is_compiling()
            or not _on_gpu(input)
            or input.dim() != 5
            or input.dtype not in (torch.bfloat16, torch.float16)
            or weight.dtype != input.dtype
            or (bias is not None and bias.dtype != input.dtype)
            or module.groups != 1
            or module.padding_mode != "zeros"
            or not isinstance(module.padding, tuple)
            or (torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad))
            or not _device_ok(torch, input.device.index)
        ):
            return original(input, weight, bias)
        mod = _kernels()
        if mod is None or not _selfcheck(torch, mod, input.device, input.dtype):
            return original(input, weight, bias)
        stride, padding, dilation = (
            tuple(module.stride),
            tuple(module.padding),
            tuple(module.dilation),
        )
        key = (
            tuple(input.shape),
            tuple(weight.shape),
            stride,
            padding,
            dilation,
            input.dtype,
            input.device.index,
        )
        if key in _FAILED:
            COUNTS["fallback_error"] += 1
            return original(input, weight, bias)
        try:
            _route, extra, biggest = plan(
                mod,
                tuple(input.shape),
                tuple(weight.shape),
                input.element_size(),
                stride,
                padding,
                dilation,
                contiguous = input.is_contiguous(),
            )
        except Exception:  # noqa: BLE001 - shape the dispatcher rejects (non-positive output, ...)
            COUNTS["fallback_shape"] += 1
            return original(input, weight, bias)
        if biggest > _INT32_MAX:
            COUNTS["fallback_shape"] += 1
            return original(input, weight, bias)
        free, total = _free_bytes(torch, input.device)
        if extra + max(MEM_MARGIN_BYTES, int(total * MEM_MARGIN_FRACTION)) > free:
            COUNTS["fallback_memory"] += 1
            return original(input, weight, bias)
        try:
            y = mod.conv3d(input, weight, bias, stride, padding, dilation)
        except Exception as exc:  # noqa: BLE001 - MIOpen for this shape from now on
            if _reraise(torch, exc):
                raise
            _FAILED.add(key)
            COUNTS["fallback_error"] += 1
            if logger is not None:
                logger.warning(
                    "diffusion.rocm_triton_conv3d: Triton conv3d failed for x=%s w=%s, MIOpen from now on: %s",
                    key[0],
                    key[1],
                    repr(exc)[:300],
                )
            return original(input, weight, bias)
        COUNTS["routed"] += 1
        return y

    _conv_forward._unsloth_triton_conv3d = True
    return _conv_forward


def install(vae: Any, logger: Any = None) -> int:
    """Wrap every eligible ``nn.Conv3d`` of ``vae`` (per instance). Returns the number wrapped; 0 when off, not ROCm
    gfx11, or Triton is missing. The device is checked again at call time (speed optims run before placement)."""
    if vae is None or not requested() or not callable(getattr(vae, "modules", None)):
        return 0
    if _env_mode() is None and type(vae).__name__ not in DEFAULT_ON_VAES:
        return 0
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
    except Exception:  # noqa: BLE001
        return 0
    if not is_rocm_torch(torch) or not torch.cuda.is_available():
        return 0
    arch = _device_arch(torch, None)
    if not arch_enabled(arch):
        return 0
    if _kernels() is None:
        if logger is not None:
            logger.info(
                "diffusion.rocm_triton_conv3d: Triton conv3d unavailable (%s)",
                _STATE.get("import_error"),
            )
        return 0
    n = 0
    for m in vae.modules():
        if not isinstance(m, torch.nn.Conv3d) or m.groups != 1 or m.padding_mode != "zeros":
            continue
        current = m._conv_forward
        if getattr(current, "_unsloth_triton_conv3d", False):
            continue
        m._conv_forward = _make_conv_forward(m, current, logger)
        n += 1
    if n and logger is not None:
        tuned = arch.split(":")[0] in TUNED_ARCHES
        logger.info(
            "diffusion.rocm_triton_conv3d: %d %s Conv3d routed to the Triton conv3d on %s%s",
            n,
            type(vae).__name__,
            arch,
            "" if tuned else " (gfx1151 kernel tables)",
        )
    return n


def uninstall(vae: Any) -> int:
    n = 0
    for m in getattr(vae, "modules", lambda: [])():
        fwd = m.__dict__.get("_conv_forward")
        if getattr(fwd, "_unsloth_triton_conv3d", False):
            del m.__dict__["_conv_forward"]
            n += 1
    return n

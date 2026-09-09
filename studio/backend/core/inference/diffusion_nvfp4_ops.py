# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer NVFP4 kernels behind ``torch.library`` custom ops, plus the backend decision.

Three things live here and nothing else does:

1. **The ops.** ``unsloth_nvfp4::quantize`` and ``unsloth_nvfp4::mm`` wrap FlashInfer's activation
   quantiser and its cutlass FP4 GEMM. FlashInfer 0.6.6 ships the *shape* of a custom-op
   registration but not the registration itself (``flashinfer/utils.py`` defines
   ``register_custom_op`` as an identity decorator, deliberately), so ``torch.ops.flashinfer`` is
   empty at runtime and Dynamo has nothing opaque to put in the graph: it tries to trace the whole
   Python entry point and fails. Wrapping the two calls here is what vLLM and SGLang both do, and
   it is what makes ``fullgraph = True`` over a quantized block possible at all. The fake impls
   must reproduce FlashInfer's allocation EXACTLY, not approximately: a wrong meta shape is not an
   error, it is a silently mis-sized buffer downstream.

2. **The device guard.** ``_device_guard`` is not defensive style. FlashInfer's
   ``fp4_gemm_cutlass.cu`` takes the STREAM from the tensor but installs no ``CUDADeviceGuard``, so
   calling ``mm_fp4`` (or ``nvfp4_quantize``) while the current device is not the tensors' device
   launches onto a stream belonging to another context: the kernel hangs, the card ends up in
   "GPU requires reset", and nothing short of root recovers it. Three cards on this host were lost
   that way. Every launch below therefore enters the tensor's own device first, and a test walks
   this file's AST to prove no launch escapes one.

3. **The backend decision.** ``select_nvfp4_backend`` answers "torchao or flashinfer" by PROBING
   (import, capability, a guarded 128x256 preflight GEMM) rather than by a version table, because
   the failure this is really about -- FlashInfer's NVFP4 JIT needing CUDA >= 12.9 to emit
   ``compute_120f``, which most sm_120 stacks do not ship -- is invisible to a version table and
   shows up only when the kernel is asked to build.

The module imports on a torch-free host: torch and flashinfer are imported inside functions, and
the ops are registered on first use through ``register_ops()``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

# ``auto`` (probe), ``torchao`` (never flashinfer), ``flashinfer`` (explicit; still probed, and a
# failed probe falls back to torchao with a warning rather than to a hang).
NVFP4_BACKEND_ENV = "UNSLOTH_NVFP4_BACKEND"
NVFP4_BACKENDS = ("auto", "torchao", "flashinfer")
BACKEND_TORCHAO = "torchao"
BACKEND_FLASHINFER = "flashinfer"

# Restores the full M x N memset in the GEMM op. Strictly slower and no safer against the
# mechanism actually established; see ``_mm_impl``.
NVFP4_ZERO_BUFFER_ENV = "UNSLOTH_NVFP4_ZERO_BUFFER"

# Where FlashInfer's NVFP4 cutlass kernels are known to exist at all. sm_121 is absent and sm_120
# is present but routinely fails the preflight on a CUDA 12.8 toolchain, which is why the
# capability set is a necessary condition and the preflight is the sufficient one.
NVFP4_FLASHINFER_CAPS = frozenset({(10, 0), (10, 3), (12, 0)})

# The GEMM backend argument passed through to flashinfer. Only cutlass is measured here.
DEFAULT_MM_BACKEND = "cutlass"

# NVFP4 constants: e2m1 max magnitude and e4m3 max, whose product is the global-scale numerator.
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
    """Element count of a swizzled scale-factor buffer, mirroring FlashInfer's own helper.

    Copied rather than imported from ``flashinfer/fp4_quantization.py`` on purpose: the fake impls
    run under ``FakeTensorMode`` and must not touch the FlashInfer JIT machinery at all.
    """
    return ((rows + row_size - 1) // row_size * row_size) * ((cols + 3) // 4 * 4)


def _device_guard(t: Any):
    """``torch.cuda.device`` for the tensor's own device. Wrap EVERY kernel launch in one.

    See this module's docstring: FlashInfer installs no device guard of its own, and launching its
    cutlass kernels against a foreign current device bricks the card rather than raising.
    """
    import torch
    return torch.cuda.device(t.device)


def _zero_buffer_enabled() -> bool:
    return os.environ.get(NVFP4_ZERO_BUFFER_ENV, "").strip().lower() in _TRUE_TOKENS


def global_scale(t: Any):
    """The NVFP4 global scale of a tensor: ``6 * 448 / amax``, as a 1-element fp32 tensor."""
    return (FP4_MAX * FP8_MAX / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


# ── the op bodies ─────────────────────────────────────────────────────────────────────────────
#
# Exposed as plain functions as well as through ``torch.ops`` so that a test can call them with a
# stubbed torch and assert the call ORDER inside them, which is the correctness invariant the
# barrier below rests on and which is invisible from outside an opaque op.


def _quantize_impl(x: Any, global_sf: Any):
    """2D bf16 in, ``(packed e2m1x2, swizzled block scales)`` out."""
    import flashinfer
    with _device_guard(x):
        return flashinfer.nvfp4_quantize(x, global_sf, do_shuffle = False)


def _mm_impl(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    """NVFP4 GEMM. Takes the weight row-major ``[N, K/2]`` and transposes inside.

    The transposes live inside the op rather than at the call site because a custom op's inputs are
    functionalised: handing Dynamo a ``.T`` view of a buffer is an alias it has to reason about,
    while ``n`` as a plain int gives the fake impl the output shape without inspecting a view.

    **A kernel MUST run between the activation quantiser and this GEMM.** FlashInfer launches the
    cutlass FP4 GEMM with PDL while the CUTLASS ``griddepcontrol`` instructions that make PDL safe
    are compiled out of its build, so without a barrier the GEMM starts reading operands the
    quantiser has not finished writing. ``enable_pdl = False`` does not help: FlashInfer plumbs
    that argument only to its cute-dsl runner and silently ignores it for cutlass (verified
    bit-identical output and identical timing with it on or off).

    What protects the GEMM is a kernel EXISTING between the producer and it, not that kernel
    writing M x N bytes. Measured on the validated PDL trigger harness, 50 iterations each, where
    the negative control fires 50/50 and the specificity control 0/50:

        trigger, nothing between        50/50 non-finite
        trigger, torch.empty only       50/50 non-finite   <- allocation alone protects nothing
        trigger, ONE-ELEMENT kernel      0/50              <- as good as the full memset
        trigger, full torch.zeros        0/50
        trigger, cuda.synchronize        0/50

    The ``empty only`` arm is the discriminator: identical allocation, no kernel, no protection. So
    a 1-element fill buys the whole guarantee and the memset was paying M x N to get it.
    ``torch.zeros(1)`` both allocates and launches the fill.

    The buffer is allocated per call, never cached. A cached one was tried and rejected: the kernel
    also misfires on its own occasionally, writing NaN over part of a correct-looking output, and a
    reused buffer turns one transient event into a permanent one for every later call with the same
    token count (measured on flux.1: one bad forward at number 33 made every later 512px render
    black while 1024px renders, with their own buffer, stayed clean).

    ``UNSLOTH_NVFP4_ZERO_BUFFER=1`` restores the full memset. It is strictly slower and no safer
    against the mechanism established above, but the residual forward-33 misfire has no confirmed
    cause and zeroing does bound what an unknown fault surfaces to garbage rather than to stale
    allocator bytes.
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
    # FlashInfer allocates a FLAT buffer of the padded size and then reshapes it to (-1, k // 16),
    # so the leading dim is NOT m whenever the padding is non-trivial. Reproduce that, not m.
    total = _swizzled_sf_numel(m, cols, 128)
    return (
        x.new_empty((m, k // 2), dtype = torch.uint8),
        x.new_empty((total // cols, cols), dtype = torch.uint8),
    )


def _mm_fake(xq: Any, wq: Any, x_sf: Any, w_sf: Any, alpha: Any, n: int, backend: str):
    import torch
    return xq.new_empty((xq.shape[0], n), dtype = torch.bfloat16)


def register_ops() -> None:
    """Register the two custom ops. Idempotent, and safe to call from several threads.

    Registration is deferred rather than done at import because ``torch.library`` needs torch and
    this module has to import on a host that has none (the smoke probe, the validator, Windows).
    """
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


# ── scale-factor layout ───────────────────────────────────────────────────────────────────────


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


def sf_matrix_shape(rows: int, cols: int) -> tuple[int, int]:
    """FlashInfer's own 2D view of a swizzled scale buffer: a flat padded buffer reshaped
    ``(-1, cols)``. The leading dim is NOT ``rows`` whenever the padding is non-trivial."""
    padded_cols = (cols + 3) // 4 * 4
    return (_swizzled_sf_numel(rows, cols, 128) // padded_cols, padded_cols)


# ── the preflight ─────────────────────────────────────────────────────────────────────────────


def nvfp4_preflight(device: Any = None, *, refresh: bool = False) -> dict:
    """A tiny GUARDED 128x256 quantise + GEMM on ``device``, memoised per device index.

    Cheap, and it is the check that would have caught the multi-device hang before it took a card
    down rather than after. It is also the only honest answer to "will FlashInfer run here": the
    JIT build is what fails on sm_120 with a CUDA 12.8 toolchain, and only asking it to build finds
    out. Returns ``{"ok", "reason", "capability", "name"}``; never raises.
    """
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


# ── the backend decision ──────────────────────────────────────────────────────────────────────


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
    """``(backend, reason)``. The reason is REPORTED, not just used: an operator who asked for
    flashinfer and got torchao has to be able to read why without reproducing the probe."""
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
    """``"torchao"`` or ``"flashinfer"`` for ``device``.

    flashinfer only when it imports, when the capability is one it has NVFP4 kernels for, and when
    the guarded preflight passes on this exact device. An EXPLICIT ``flashinfer`` request that
    fails any of the three falls back to torchao with a warning rather than to a hang: the
    alternative is a card that needs a root reset.
    """
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

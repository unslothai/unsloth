# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision quantisation of the diffusion DiT transformer.

The default path loads the transformer as a single-file GGUF, which stores weights
4-bit but DEQUANTISES to bf16 on every matmul -- so it runs at bf16 tensor-core rate
and never touches the int8 / fp8 / fp4 tensor cores. It is a memory win that costs
speed. This module is the opt-in alternative: load the DENSE bf16 transformer from the
base repo and torchao-quantise it with a DYNAMIC-ACTIVATION scheme so the matmul runs
on the low-precision tensor cores. Measured on a B200 (Z-Image-Turbo, 1024px / 8 steps)
vs the GGUF+compile default (0.802s, LPIPS 0.083 vs dense bf16): fp8 dynamic 0.585s
(1.37x), int8 dynamic 0.603s (1.33x), both at LOWER LPIPS than GGUF -- faster AND a hair
more accurate, at the cost of a higher-memory dense load. So it is strictly opt-in; the
loader keeps GGUF as the low-memory default and the fallback.

Scheme by architecture (``auto`` picks the best supported, best first):
  nvfp4 / mxfp8 - Blackwell sm_100+ FP4 / MX tensor cores (biggest win; prototype).
  fp8           - Ada / Hopper / Blackwell (sm_89+) fp8 tensor cores.
  int8          - Ampere+ (sm_80+) int8 tensor cores -- the broadest-hardware lever.

Every scheme needs ``torch.compile`` to realise the speedup (dynamic quant is ~30x
slower eager); the loader already compiles the repeated block AFTER this runs. torch /
torchao are imported lazily so the module stays importable in a no-torch runtime, and
every probe is best-effort: an unsupported scheme yields None and the caller loads GGUF.
"""

from __future__ import annotations

from typing import Any, Optional

TQ_INT8 = "int8"
TQ_FP8 = "fp8"
TQ_NVFP4 = "nvfp4"
TQ_MXFP8 = "mxfp8"
TQ_AUTO = "auto"
TQ_SCHEMES = (TQ_INT8, TQ_FP8, TQ_NVFP4, TQ_MXFP8)
TQ_MODES = (TQ_AUTO,) + TQ_SCHEMES

# Skip linears whose in/out features are below this. The int8 dynamic path uses
# torch._int_mm, which requires the activation row count M > 16, and the DiT's tiny
# timestep / pooled / modulation projections run at M=1 and crash it. They are a
# negligible share of the FLOPs, so leaving them bf16 costs ~nothing (measured:
# 239/276 Z-Image linears quantised, full speedup) and keeps quality a touch higher.
DEFAULT_MIN_LINEAR_FEATURES = 512

# Per-architecture preference order for ``auto`` -- best (fastest, in-bar) first, with
# the lower-precision schemes listed as fallbacks for that arch tier. On Blackwell, fp8
# leads: measured on a B200, plain fp8 dynamic is both faster AND more accurate than the
# alternatives for the DiT's shapes. mxfp8's block scaling adds overhead without a speed
# win, so it sits below fp8. nvfp4 is intentionally below fp8 too: the FP4 tensor-core
# GEMM is real once torch>=2.11 + torchao's CUTLASS FP4 kernel is present (verified: a
# 16384^3 GEMM hits ~3826 TFLOPS, 1.37x fp8), but it only beats fp8 on very large GEMMs.
# At the DiT's actual shapes (hidden ~3072, MLP ~12288, M~4096) it is *slower* than fp8
# (0.81x end-to-end on Z-Image 1024px) AND notably less accurate (LPIPS 0.166 vs fp8's
# 0.044), because FP4's per-forward quant overhead is not amortised and the format is
# coarser. So nvfp4 is kept as an explicit opt-in, never the auto pick for diffusion.
_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    ((10, 0), (TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8)),  # Blackwell sm_100+
    ((8, 9), (TQ_FP8, TQ_INT8)),  # Ada sm_89 / Hopper sm_90
    ((8, 0), (TQ_INT8,)),  # Ampere sm_80 / sm_86
)

# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}

# Data-center GPU model tokens (un-nerfed FP32 accumulate). Matched as whole tokens of
# torch.cuda.get_device_name(), so the workstation "A4000" is not mistaken for the
# data-center "A40". Anything not here -- GeForce, workstation RTX, or an unknown name --
# is treated as consumer-class (FP32-accumulate halved). See developer.nvidia.com/cuda/gpus.
_DATACENTER_GPU_TOKENS = frozenset(
    {
        "B200",
        "B100",
        "GB200",
        "GB300",
        "GB10",  # Blackwell data center
        "H200",
        "H100",
        "H800",
        "H20",  # Hopper data center
        "A100",
        "A800",
        "A30",
        "A40",
        "A16",
        "A10",
        "A2",  # Ampere data center
        "L40",
        "L40S",
        "L4",
        "L20",
        "L2",  # Ada data center
        "V100",
        "P100",
        "P40",
        "T4",  # legacy data center
    }
)


def _is_consumer_gpu(device: Any = None) -> bool:
    """Whether the active GPU is consumer / workstation class (GDDR), where fp8 FP32
    accumulate is throughput-halved so fast (FP16) accumulate is a ~2x win. Data-center
    HBM parts (recognised by name token) are not nerfed and return False, so they keep
    the higher-precision default accumulate for free. Heuristic on the device name: a
    GeForce / TITAN name is always consumer; a recognised data-center token is not;
    anything else (workstation RTX, unknown) defaults to consumer -- the safe choice,
    since fast accumulate is free on data-center and a win on consumer. Best-effort:
    True on any probe failure."""
    try:
        import re

        import torch
        name = torch.cuda.get_device_name(device).upper()
    except Exception:  # noqa: BLE001 — no torch / no device -> assume consumer
        return True
    if "GEFORCE" in name or "TITAN" in name:
        return True
    tokens = set(re.split(r"[^A-Z0-9]+", name))
    return not (tokens & _DATACENTER_GPU_TOKENS)


def normalize_transformer_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested transformer quant; None / "" / "none" / "off" -> None.

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized not in TQ_MODES:
        raise ValueError(
            f"Unsupported transformer_quant '{value}'. Use one of: {', '.join(TQ_MODES)}."
        )
    return normalized


def dense_transformer_supported(target: Any) -> bool:
    """Whether the dense-source quant path is usable for ``target``: a CUDA device with
    a bf16 compute dtype (the only configuration any torchao dynamic scheme accelerates).
    A cheap pre-check the loader runs before loading the (large) dense transformer."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def select_transformer_quant_scheme(target: Any, requested: Optional[str]) -> Optional[str]:
    """The concrete scheme to apply, or None to fall back to GGUF.

    ``auto`` walks the per-arch ladder and returns the first scheme that passes a real
    quantise+matmul smoke test, so on a box where the Blackwell fp4 / mx kernels are
    unavailable it lands on fp8 / int8 with no error. An explicit scheme is honored only
    if supported (else None -> GGUF), never silently swapped for a different one."""
    requested = normalize_transformer_quant(requested)
    if requested is None or not dense_transformer_supported(target):
        return None
    device = str(getattr(target, "device", "cuda"))
    if requested != TQ_AUTO:
        return requested if _scheme_supported(requested, device) else None
    cap = _capability()
    if cap is None:
        return None
    for floor, schemes in _AUTO_LADDER:
        if cap >= floor:
            for scheme in schemes:
                if _scheme_supported(scheme, device):
                    return scheme
            return None
    return None


def _capability() -> Optional[tuple[int, int]]:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return (int(major), int(minor))
    except Exception:
        return None


def _scheme_supported(scheme: str, device: str) -> bool:
    """CUDA + (for fp8) the fp8 dtype + a cached quantise+matmul smoke test for ``scheme``."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if scheme == TQ_FP8 and not hasattr(torch, "float8_e4m3fn"):
            return False
    except Exception:
        return False
    return _smoke_probe(scheme, device)


def _smoke_probe(scheme: str, device: str) -> bool:
    """True iff a tiny Linear quantised with ``scheme`` runs one M=32 forward without
    error. Cached per (scheme, device). This is what makes ``auto`` robust to a torch /
    torchao build where a prototype (nvfp4 / mxfp8) kernel is unavailable: it fails here
    and the ladder moves on, rather than crashing at the first real denoise step."""
    key = (scheme, device)
    if key in _SMOKE_CACHE:
        return _SMOKE_CACHE[key]
    ok = False
    try:
        import torch
        from torchao.quantization import quantize_

        lin = torch.nn.Linear(512, 512, bias = False).to(device = device, dtype = torch.bfloat16)
        quantize_(lin, _make_quant_config(scheme), filter_fn = make_filter_fn(0))
        x = torch.randn(32, 512, device = device, dtype = torch.bfloat16)
        with torch.no_grad():
            lin(x)
        torch.cuda.synchronize()
        ok = True
    except Exception:
        ok = False
    _SMOKE_CACHE[key] = ok
    return ok


def _resolve_fast_accum(fast_accum: Optional[bool]) -> bool:
    """The fp8 ``use_fast_accum`` to apply. ``None`` auto-detects by GPU class
    (consumer / workstation -> fast; data-center -> precise); an explicit bool forces it."""
    return _is_consumer_gpu() if fast_accum is None else bool(fast_accum)


def _make_quant_config(scheme: str, fast_accum: Optional[bool] = None) -> Any:
    """The torchao dynamic-activation config for ``scheme`` (lazy import; prototype
    import for the Blackwell fp4 / mx schemes is inside the branch that needs it).

    ``fast_accum`` applies to fp8 only: None auto-detects by GPU class, True/False force it."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    if scheme == TQ_INT8:
        return Int8DynamicActivationInt8WeightConfig()
    if scheme == TQ_FP8:
        # Choose fp8 accumulate by GPU class (unless forced). On consumer / workstation
        # cards (GDDR) the fp8 tensor cores run ~2x faster with FP16 (fast) accumulate
        # than FP32 (e.g. ~838 vs ~419 TFLOPS on RTX 50xx), so fast accumulate is a real
        # win there. Data-center HBM parts default to the higher-precision accumulate.
        # fast accumulate is a precision (not overflow) tradeoff and stays below the fp8
        # quant noise floor (measured 0 non-finite even on Z-Image's ~1e6 activations).
        try:
            from torchao.float8 import Float8MMConfig
            return Float8DynamicActivationFloat8WeightConfig(
                mm_config = Float8MMConfig(use_fast_accum = _resolve_fast_accum(fast_accum))
            )
        except Exception:  # noqa: BLE001 — older torchao without the explicit knob
            return Float8DynamicActivationFloat8WeightConfig()
    if scheme == TQ_NVFP4:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        return NVFP4DynamicActivationNVFP4WeightConfig()
    if scheme == TQ_MXFP8:
        import torch
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        try:
            return MXDynamicActivationMXWeightConfig(
                activation_dtype = torch.float8_e4m3fn, weight_dtype = torch.float8_e4m3fn
            )
        except (TypeError, AttributeError):
            # TypeError: older torchao without the explicit dtype knobs.
            # AttributeError: a torch build without torch.float8_e4m3fn.
            return MXDynamicActivationMXWeightConfig()
    raise ValueError(f"unknown transformer quant scheme '{scheme}'")


def make_filter_fn(min_features: int):
    """A torchao ``quantize_`` filter keeping only the FLOP-heavy linears: nn.Linear
    with both in/out features >= ``min_features``. Hides the (module, fqn) callback arity."""

    def filter_fn(module: Any, fqn: str = "") -> bool:
        try:
            import torch
            if not isinstance(module, torch.nn.Linear):
                return False
        except Exception:
            return False
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            return False
        return in_features >= min_features and out_features >= min_features

    return filter_fn


def quantize_transformer(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    min_features: int = DEFAULT_MIN_LINEAR_FEATURES,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
) -> Optional[str]:
    """Quantise ``pipe.transformer``'s FLOP-heavy linears in place with the arch-chosen
    dynamic scheme. Returns the scheme actually engaged, or None when disabled /
    unsupported / failed -- the caller then loads GGUF instead. Best-effort: it never
    raises for an ordinary unsupported environment (a failure leaves the module dense).

    ``fast_accum`` (fp8 only) overrides the per-GPU-class accumulate choice: None
    auto-detects (fast on consumer, precise on data-center), True/False force it."""
    scheme = select_transformer_quant_scheme(target, mode)
    if scheme is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    try:
        from torchao.quantization import quantize_

        quantize_(
            transformer,
            _make_quant_config(scheme, fast_accum = fast_accum),
            filter_fn = make_filter_fn(min_features),
        )
        # Runtime-only marker (torchao tensors are not safetensors-serializable; this
        # backend is inference-only, so this is purely diagnostic).
        try:
            transformer._unsloth_runtime_quant = scheme
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        return scheme
    except Exception as exc:  # noqa: BLE001 — leave the transformer dense -> GGUF fallback
        _warn(logger, scheme, exc)
        return None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.transformer_quant: %s failed: %s", what, exc)

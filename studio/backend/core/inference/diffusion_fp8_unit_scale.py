# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ada (sm_89) fp8 GEMM with PRECISE accumulate: unit-scale ``torch._scaled_mm`` plus a row x column scale epilogue.

torchao's per-row ``Float8Tensor`` linear calls ``torch._scaled_mm`` with an [M, 1] activation scale and a [1, N] weight
scale. On sm_89, PyTorch routes that row-wise form to a CUTLASS 2.x stream-K kernel, the per-tensor form to cuBLASLt.
Measured on an L4 with torch 2.14.1 at the FLUX.2-klein / Z-Image / Qwen-Image-2.1 1024 px linear shapes:

* precise accumulate (``use_fast_accum=False``): row-wise 29-32 TFLOP/s, per-tensor 102-134 TFLOP/s (3.5-4.5x).
  End to end (compiled, 1024 px): klein-4B 1.074 -> 0.515 s/step, Z-Image-Turbo 1.721 -> 0.853 s/step.
* fast accumulate (Studio's default): row-wise 85-108, per-tensor 91-127 TFLOP/s, and end to end no measurable
  difference (klein 0.501 vs 0.499 s/step, Z-Image 0.799 vs 0.795, inside the A/A spread). Fast-accumulate calls are
  therefore left on torchao's kernel and stay bit-identical.

So on sm_89 a precise-accumulate GEMM runs with unit scales into bf16 and the SAME per-row / per-column scales (and
bias) are applied in fp32 afterwards; under ``torch.compile`` Inductor fuses that multiply into the consumer. The math
is the row-wise kernel's (fp32 acc * s_a * s_b + bias, rounded to bf16); the only difference is one bf16 rounding of
the unscaled accumulator before the scale multiply (relative error <= 2^-9).

Engages only on a CUDA device of compute capability 8.9, ``use_fast_accum=False``, bf16 output, row-wise scales, no
output scale, and at least ``_MIN_ROWS`` activation rows. Every other call, and every other GPU, takes torchao's own
function unchanged.

``UNSLOTH_DIFFUSION_FP8_UNIT_SCALE``: ``0`` = never (kill switch), ``1`` = every CUDA arch and both accumulate modes
(testing only), unset / anything else = auto (sm_89, precise accumulate). ``UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS``
overrides the row floor.
"""

from __future__ import annotations

import os
from typing import Any, Optional

UNIT_SCALE_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE"
MIN_ROWS_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS"
_DEFAULT_MIN_ROWS = 1
_ADA_CAPS = ((8, 9),)

# Calls taken by the unit-scale form (eager: per call; compiled: replayed with the graph) / by the fallback.
_COUNTS = {"unit_scale": 0, "rowwise": 0}
_STATE: dict = {"installed": False, "orig": None, "caps": {}}


def _mode() -> str:
    v = (os.environ.get(UNIT_SCALE_ENV) or "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def _min_rows() -> int:
    try:
        return max(1, int(os.environ.get(MIN_ROWS_ENV, "") or _DEFAULT_MIN_ROWS))
    except ValueError:
        return _DEFAULT_MIN_ROWS


def stats() -> dict:
    """Engagement counters plus whether the dispatch is installed."""
    return {**_COUNTS, "installed": bool(_STATE["installed"])}


def _device_caps() -> dict:
    import torch

    caps = {}
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                caps[i] = tuple(torch.cuda.get_device_capability(i))
    except Exception:  # noqa: BLE001 - no CUDA runtime: nothing qualifies
        return {}
    return caps


def install(logger: Any = None) -> bool:
    """Swap torchao's ``addmm_float8_unwrapped_inference`` (as bound in the Float8Tensor module) for the dispatching
    wrapper. Idempotent; a no-op (False) when switched off, when no visible GPU qualifies, or when this torchao lays
    the function out differently."""
    if _STATE["installed"]:
        return True
    mode = _mode()
    if mode == "off":
        return False
    caps = _device_caps()
    if mode == "auto" and not any(c in _ADA_CAPS for c in caps.values()):
        return False
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t
    except Exception:  # noqa: BLE001 - torchao absent / stubbed / moved
        return False
    orig = getattr(_f8t, "addmm_float8_unwrapped_inference", None)
    if orig is None or not callable(orig):
        return False
    if getattr(orig, "_unsloth_unit_scale", False):
        _STATE.update(installed = True)
        return True
    min_rows = _min_rows()
    force = mode == "force"
    qualifying = frozenset(i for i, c in caps.items() if force or c in _ADA_CAPS)

    import torch

    def addmm_float8_unwrapped_inference(
        a_data: Any,
        a_scale: Any,
        b_data: Any,
        b_scale: Any,
        output_dtype: Any,
        output_scale: Optional[Any] = None,
        bias: Optional[Any] = None,
        use_fast_accum: bool = False,
    ) -> Any:
        if (
            output_scale is None
            and output_dtype == torch.bfloat16
            and a_data.device.type == "cuda"
            and a_data.device.index in qualifying
            and a_data.dim() == 2
            and b_data.dim() == 2
            and a_scale.dim() == 2
            and b_scale.dim() == 2
            and a_scale.shape[1] == 1
            and b_scale.shape[0] == 1
            and a_scale.shape[0] == a_data.shape[0]
            and b_scale.shape[1] == b_data.shape[1]
            and a_data.shape[0] >= min_rows
            and (force or not use_fast_accum)
        ):
            _COUNTS["unit_scale"] += 1
            one = torch.ones((), device = a_data.device, dtype = torch.float32)
            acc = torch._scaled_mm(
                a_data,
                b_data,
                scale_a = one,
                scale_b = one,
                out_dtype = torch.bfloat16,
                use_fast_accum = use_fast_accum,
            )
            out = acc.float() * a_scale.float() * b_scale.float()
            if bias is not None:
                out = out + bias.float()
            return out.to(output_dtype)
        _COUNTS["rowwise"] += 1
        return orig(
            a_data,
            a_scale,
            b_data,
            b_scale,
            output_dtype,
            output_scale = output_scale,
            bias = bias,
            use_fast_accum = use_fast_accum,
        )

    addmm_float8_unwrapped_inference._unsloth_unit_scale = True  # type: ignore[attr-defined]
    addmm_float8_unwrapped_inference._unsloth_orig = orig  # type: ignore[attr-defined]
    _f8t.addmm_float8_unwrapped_inference = addmm_float8_unwrapped_inference
    _STATE.update(installed = True, orig = orig, caps = caps)
    if logger is not None:
        try:
            logger.info(
                "fp8: unit-scale GEMM + row/column scale epilogue on %s (rows >= %d, %s=0 disables)",
                "every CUDA device" if force else "sm_89 precise accumulate",
                min_rows,
                UNIT_SCALE_ENV,
            )
        except Exception:  # noqa: BLE001
            pass
    return True


def uninstall() -> None:
    """Restore torchao's function (tests)."""
    if not _STATE["installed"]:
        return
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t

        if _STATE["orig"] is not None:
            _f8t.addmm_float8_unwrapped_inference = _STATE["orig"]
    finally:
        _STATE.update(installed = False, orig = None, caps = {})

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm_89 precise-accumulate fp8 GEMM: unit-scale ``torch._scaled_mm`` + fp32 row x column scale epilogue.

On sm_89 torchao's row-wise ``_scaled_mm`` hits a slow CUTLASS stream-K kernel. L4, torch 2.14.1, 1024 px linears:
* precise accumulate: row-wise 29-32 TFLOP/s, per-tensor 102-134 TFLOP/s; end to end klein-4B 1.034 -> 0.480 s/step,
  Z-Image-Turbo 1.708 -> 0.772 s/step.
* fast accumulate: row-wise 85-108, per-tensor 91-127 TFLOP/s, no end-to-end gain, so left on torchao (bit-identical).
Only difference vs row-wise: one bf16 rounding of the unscaled accumulator (relative error <= 2^-9).

``UNSLOTH_DIFFUSION_FP8_UNIT_SCALE``: ``0`` off, ``1`` every arch + both accumulate modes (testing), else auto.
"""

from __future__ import annotations

import os
from typing import Any, Optional

UNIT_SCALE_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE"
MIN_ROWS_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS"
# Unit-scale wins from M = 256 up (L4: row-wise 2.5-4x slower from 512); below, tens of microseconds either way.
_DEFAULT_MIN_ROWS = 256
_ADA_CAPS = ((8, 9),)

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


def _compiling() -> bool:
    # Counters are eager-only: a module-global mutated inside a traced region makes Dynamo guard on it and recompile.
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - no torch.compiler: nothing is being traced
        return False


def stats() -> dict:
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
    """Patch torchao's fp8 addmm with the dispatcher; idempotent, False when off / no qualifying GPU / layout moved."""
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
            if not _compiling():
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
            if isinstance(acc, tuple):  # torch < 2.5 returns (out, amax)
                acc = acc[0]
            out = acc.float() * a_scale.float() * b_scale.float()
            if bias is not None:
                out = out + bias.float()
            return out.to(output_dtype)
        if not _compiling():
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
    if not _STATE["installed"]:
        return
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t
        if _STATE["orig"] is not None:
            _f8t.addmm_float8_unwrapped_inference = _STATE["orig"]
    finally:
        _STATE.update(installed = False, orig = None, caps = {})

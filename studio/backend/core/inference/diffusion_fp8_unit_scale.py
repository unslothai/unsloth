# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm_89 precise-accumulate fp8: unit-scale _scaled_mm beats torchao row-wise (fast accumulate stays
on torchao, no gain); UNSLOTH_DIFFUSION_FP8_UNIT_SCALE=0 off. Installed only where it can
engage: installing tags the process-wide compile cache key."""

from __future__ import annotations

import os
from typing import Any, Optional

UNIT_SCALE_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE"
MIN_ROWS_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS"
# Unit-scale wins from M = 256 up (L4: row-wise 2.5-4x slower from 512); below, tens of microseconds either way.
_DEFAULT_MIN_ROWS = 256
# Eager epilogue chunk size, avoids a full fp32 [M, N] buffer.
_EAGER_EPILOGUE_ELEMS = 1 << 24
_OFF_WORDS = ("0", "off", "false", "no", "none")
_ON_WORDS = ("1", "on", "true", "yes", "force")
_ADA_CAPS = ((8, 9),)

_COUNTS = {"unit_scale": 0, "rowwise": 0}
_STATE: dict = {"installed": False, "orig": None, "caps": {}}


def _mode() -> str:
    v = (os.environ.get(UNIT_SCALE_ENV) or "").strip().lower()
    if v in _OFF_WORDS:
        return "off"
    if v in _ON_WORDS:
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


def _scale_epilogue(
    acc: Any, a_scale: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Any:
    """``(acc * a_scale * b_scale + bias)`` in fp32, rounded once to ``output_dtype``. Eager runs it per row chunk
    in place (bitwise identical) to bound peak memory."""
    import torch

    if _compiling():
        out = acc * a_scale.float()
        out.mul_(b_scale.float())
        if bias is not None:
            out.add_(bias.float())
        return out.to(output_dtype)
    sa, sb = a_scale.float(), b_scale.float()
    bf = bias.float() if bias is not None else None
    rows = int(acc.shape[0])
    step = max(1, _EAGER_EPILOGUE_ELEMS // max(1, int(acc.shape[1])))
    out = acc if acc.dtype == output_dtype else torch.empty_like(acc, dtype = output_dtype)
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        tmp = acc[start:stop] * sa[start:stop]
        tmp.mul_(sb)
        if bf is not None:
            tmp.add_(bf)
        out[start:stop].copy_(tmp)
        # Free before the next chunk, else two are live.
        del tmp
    return out


def can_engage(fast_accum: Optional[bool]) -> bool:
    """False only for fast accumulate outside ``force``; None (unknown) installs."""
    return _mode() == "force" or fast_accum is not True


def stats() -> dict:
    return {**_COUNTS, "installed": bool(_STATE["installed"])}


_CACHE_TAG_PREFIX = "unsloth_fp8_unit_scale="


def _set_compile_cache_tag(tag: Optional[str]) -> None:
    """Key torch's FX/AOTAutograd caches on the dispatcher: the Dynamo graph is identical with or without the patch,
    so untagged, a graph cached on the other side of install / the kill switch is reused. ``None`` removes our tag."""
    try:
        import torch.compiler.config as _cc
    except Exception:  # noqa: BLE001 - torch without compiler config: no persistent compile cache to split
        return
    current = getattr(_cc, "cache_key_tag", None)
    if not isinstance(current, str):
        return
    from .diffusion_compile_config import canonical_compile_cache_tag

    try:
        _cc.cache_key_tag = canonical_compile_cache_tag(current, _CACHE_TAG_PREFIX, tag)
    except Exception:  # noqa: BLE001
        pass


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


def install(logger: Any = None, *, fast_accum: Optional[bool] = None) -> bool:
    """Patch torchao's fp8 addmm with the dispatcher; idempotent, False when off / no qualifying GPU / layout moved /
    it cannot engage."""
    if _STATE["installed"]:
        return True
    mode = _mode()
    if mode == "off" or not can_engage(fast_accum):
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
            (force or not use_fast_accum)
            and output_scale is None
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
            return _scale_epilogue(acc, a_scale, b_scale, bias, output_dtype)
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
    _set_compile_cache_tag(f"{min_rows}{':force' if force else ''}")
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
        _set_compile_cache_tag(None)

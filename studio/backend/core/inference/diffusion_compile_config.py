# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""torch 2.12+ keeps dynamo / inductor config writes in a ContextVar, so knobs set on the load thread are invisible on
the render thread that compiles. Knobs are recorded here and ``apply()`` re-writes them into the current context."""

from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from typing import Any

# dynamic_scale_rblock benchmarks R0_BLOCK vs R0_BLOCK/2 per process (never cached); the two sum in different orders,
# so renders differed across servers on one seed (FLUX.1-schnell int8, B200). =1 restores inductor's default, and also
# drops the per-family reduction-config filter (diffusion_speed.pin_reduction_configs).
DYNAMIC_SCALE_RBLOCK_ENV = "UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK"


def reduction_blocks_pinned() -> bool:
    """True unless UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK asks for inductor's per-process R0_BLOCK benchmark."""
    raw = (os.environ.get(DYNAMIC_SCALE_RBLOCK_ENV) or "").strip().lower()
    return raw not in ("1", "on", "true", "yes")


def reduction_config_filter_available() -> bool:
    """torch has the reduction-config filter (2.10+) and the kill switch is unset."""
    if not reduction_blocks_pinned():
        return False
    try:
        import torch
        return hasattr(torch._inductor.config.test_configs, "force_filter_reduction_configs")
    except Exception:  # noqa: BLE001 - no torch / inductor: nothing pinned
        return False


def family_filters_reductions(family: Any) -> bool:
    """``filter_reduction_configs`` on every arch, or the current CUDA device listed in ``filter_reduction_configs_archs``."""
    if family is None:
        return False
    if bool(getattr(family, "filter_reduction_configs", False)):
        return True
    archs = getattr(family, "filter_reduction_configs_archs", None) or ()
    if not archs:
        return False
    cap = _device_capability()
    return cap is not None and cap in {tuple(int(v) for v in a) for a in archs}


def _device_capability() -> Any:
    try:
        import torch
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return None
        return tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:  # noqa: BLE001 - no CUDA: nothing pinned
        return None


_LOCK = threading.Lock()
_KNOBS: dict[tuple[str, str], Any] = {}
_generation = 0
# ContextVar, not thread-local: a copy_context() run on an already-applied thread must re-apply.
_applied: ContextVar[int] = ContextVar("unsloth_compile_config_applied", default = -1)


def _module(name: str) -> Any:
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch -> nothing to configure
        return None
    obj: Any = torch
    for part in name.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _write(cfg: Any, attr: str, value: Any) -> None:
    # Write only on change: every write dirties the config and forces a rehash.
    try:
        if getattr(cfg, attr) != value:
            setattr(cfg, attr, value)
    except Exception:  # noqa: BLE001 - a missing or read-only knob on this torch build is skipped
        pass


def set_knob(module_name: str, attr: str, value: Any) -> bool:
    global _generation
    cfg = _module(module_name)
    if cfg is None or not hasattr(cfg, attr):
        return False
    with _LOCK:
        _KNOBS.pop((module_name, attr), None)
        _KNOBS[(module_name, attr)] = value
        _generation += 1
    _write(cfg, attr, value)
    return True


def get_knob(
    module_name: str,
    attr: str,
    default: Any = None,
) -> Any:
    with _LOCK:
        if (module_name, attr) in _KNOBS:
            return _KNOBS[(module_name, attr)]
    cfg = _module(module_name)
    return getattr(cfg, attr, default) if cfg is not None else default


def apply() -> None:
    generation = _generation
    if _applied.get() == generation:
        return
    with _LOCK:
        generation = _generation
        knobs = list(_KNOBS.items())
    for (module_name, attr), value in knobs:
        cfg = _module(module_name)
        if cfg is not None:
            _write(cfg, attr, value)
    _applied.set(generation)


def is_recorded(module_name: str, attr: str) -> bool:
    with _LOCK:
        return (module_name, attr) in _KNOBS


def recorded() -> dict[tuple[str, str], Any]:
    with _LOCK:
        return dict(_KNOBS)


def _reset_for_tests() -> None:
    global _generation
    with _LOCK:
        _KNOBS.clear()
        _generation += 1

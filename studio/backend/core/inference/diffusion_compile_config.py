# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Process-wide torch compile knobs that survive the thread hop from load to render.

torch 2.12+ keeps every ``torch._dynamo.config`` / ``torch._inductor.config`` write in a ContextVar, so a knob set on
the load thread is invisible on the render thread, where the lazy compile actually runs: it compiled with dynamo's
default ``recompile_limit`` of 8 (a DiT needing more graphs silently stayed eager) and without
``emulate_precision_casts`` (numerics other than the validated ones). torch exposes no public process-wide setter.

So every knob the speed layer sets is recorded here, and ``apply()`` writes the recorded values into the CURRENT
context. The render thread calls it before each denoise and every guarded compiled block calls it before running, so
any thread that compiles sees the load-time values. On torch before 2.12 the writes are already process-wide and
``apply()`` finds nothing to change. torch imported lazily.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any

_LOCK = threading.Lock()
# (config module name, attribute) -> value. Insertion order is the write order.
_KNOBS: dict[tuple[str, str], Any] = {}
_generation = 0
# Last generation applied in this context. A ContextVar, not a thread-local: torch's overrides live in the context, so
# a fresh context (a new thread, a copy_context() run) must re-apply even on a thread that applied before.
_applied: ContextVar[int] = ContextVar("unsloth_compile_config_applied", default = -1)


def _module(name: str) -> Any:
    """``torch._dynamo.config`` / ``torch._inductor.config`` (or a dotted sub-config) off the imported torch, or None."""
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
    # Only on a change: every write marks the config dirty and forces a rehash on the next compile.
    try:
        if getattr(cfg, attr) != value:
            setattr(cfg, attr, value)
    except Exception:  # noqa: BLE001 - a missing or read-only knob on this torch build is skipped
        pass


def set_knob(module_name: str, attr: str, value: Any) -> bool:
    """Set ``<module_name>.<attr> = value`` here and record it for every other thread. False when the knob is absent
    on this torch build (nothing recorded)."""
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
    """The process-wide value: the recorded one when set, else what this context reads."""
    with _LOCK:
        if (module_name, attr) in _KNOBS:
            return _KNOBS[(module_name, attr)]
    cfg = _module(module_name)
    return getattr(cfg, attr, default) if cfg is not None else default


def apply() -> None:
    """Write every recorded knob into the current context. Cheap when nothing changed since the last call here."""
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

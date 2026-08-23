# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The instrument registry. See INTERFACES.md section 3.

An instrument registers a zero-argument FACTORY, not an instance, so a level-2 tracing instrument
whose module needs a heavy import costs nothing on a level-0 run. Dropping a .py file into this
directory is the whole registration step: `load_all()` imports every sibling module and a module
that cannot be imported becomes a recorded gate row rather than a crash, so a partially installed
tree still produces Layer 1's numbers.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Optional

_REGISTRY: "dict[str, _Entry]" = {}
_IMPORT_ERRORS: "dict[str, str]" = {}
_LOADED = False


@dataclass(frozen = True)
class _Entry:
    name: str
    level: int
    factory: Callable[[], Any]


def register_instrument(name: str, level: int = 0) -> Callable:
    """Decorator over a zero-argument factory returning an object with the Instrument protocol."""
    if level < 0:
        raise ValueError("instrument level must be >= 0")

    def deco(factory: Callable[[], Any]) -> Callable[[], Any]:
        if name in _REGISTRY:
            raise ValueError(f"instrument {name!r} is already registered")
        _REGISTRY[name] = _Entry(name = name, level = level, factory = factory)
        return factory

    return deco


def load_all() -> dict[str, str]:
    """Import every sibling module once. Returns {module_name: error} for the ones that failed."""
    global _LOADED
    if _LOADED:
        return dict(_IMPORT_ERRORS)
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as exc:  # noqa: BLE001
            _IMPORT_ERRORS[mod.name] = f"{type(exc).__name__}: {exc}"
    _LOADED = True
    return dict(_IMPORT_ERRORS)


def available() -> list[tuple[str, int]]:
    load_all()
    return sorted((e.name, e.level) for e in _REGISTRY.values())


def import_errors() -> dict[str, str]:
    load_all()
    return dict(_IMPORT_ERRORS)


def build(level: int, only: Optional[list[str]] = None) -> list:
    """Instantiate every instrument whose declared level is <= `level`, sorted by name.

    A factory that raises is skipped and recorded in `import_errors()` under its instrument name,
    for the same reason a failed import is: one broken instrument must not cost the run.
    """
    load_all()
    out = []
    for entry in sorted(_REGISTRY.values(), key = lambda e: e.name):
        if entry.level > level:
            continue
        if only is not None and entry.name not in only:
            continue
        try:
            inst = entry.factory()
        except Exception as exc:  # noqa: BLE001
            _IMPORT_ERRORS[entry.name] = f"{type(exc).__name__}: {exc}"
            continue
        inst.name = getattr(inst, "name", entry.name) or entry.name
        inst.level = getattr(inst, "level", entry.level)
        out.append(inst)
    return out


__all__ = ["register_instrument", "load_all", "available", "import_errors", "build"]

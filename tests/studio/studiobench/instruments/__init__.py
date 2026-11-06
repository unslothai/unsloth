# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Instruments register a zero-argument factory; an import failure becomes a gate row, not a crash."""

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
    """A raising factory is skipped and recorded in import_errors() so it cannot sink the whole run."""
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

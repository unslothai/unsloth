# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The action registry and the slot-scheduled scene. See INTERFACES.md sections 4 and 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..runtime.types import ActionContext, ActionResult, Slot

_ACTIONS: "dict[str, _Action]" = {}


@dataclass(frozen = True)
class _Action:
    name: str
    fn: Callable[[ActionContext], ActionResult]
    default_budget_ms: int


def register_action(name: str, default_budget_ms: int = 3000) -> Callable:
    def deco(fn: Callable[[ActionContext], ActionResult]) -> Callable:
        if name in _ACTIONS:
            raise ValueError(f"action {name!r} is already registered")
        _ACTIONS[name] = _Action(name = name, fn = fn, default_budget_ms = default_budget_ms)
        return fn

    return deco


def _ensure_loaded() -> None:
    from . import actions  # noqa: F401  (importing registers every built-in action)


def action_names() -> list[str]:
    _ensure_loaded()
    return sorted(_ACTIONS)


def get_action(name: str) -> Optional[_Action]:
    _ensure_loaded()
    return _ACTIONS.get(name)


def default_budget_ms(name: str) -> int:
    entry = get_action(name)
    return entry.default_budget_ms if entry else 3000


__all__ = [
    "register_action",
    "action_names",
    "get_action",
    "default_budget_ms",
    "Slot",
    "ActionContext",
    "ActionResult",
]

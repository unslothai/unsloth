# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a platform backend provides. Positions are desktop points, not capture pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class BackendError(RuntimeError):
    """The message is shown to the model, so it says what to do next."""


@dataclass(frozen = True)
class Display:
    index: int
    x: float
    y: float
    width: float
    height: float
    primary: bool = False

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen = True)
class FrontWindow:
    app: str = ""
    # Bundle id on macOS, executable name on Windows, WM_CLASS on Linux.
    app_id: str = ""
    title: str = ""
    pid: int = 0
    bounds: Optional[tuple[float, float, float, float]] = None

    def contains(self, x: float, y: float) -> bool:
        if self.bounds is None:
            return True
        left, top, width, height = self.bounds
        return left <= x < left + width and top <= y < top + height


class Backend:
    name = "none"
    shortcut_modifier = "ctrl"
    os_name = "this computer"

    def unavailable_reason(self) -> Optional[str]:
        return None

    def permissions(self) -> dict[str, Optional[bool]]:
        return {"screen": None, "input": None}

    def request_permission(self, kind: str) -> None:
        """Ask the OS for a grant. Only ever called from Settings, never by a tool call."""

    def displays(self) -> list[Display]:
        raise NotImplementedError

    def capture(self, display: Display) -> Any:
        raise NotImplementedError

    def cursor(self) -> tuple[float, float]:
        raise NotImplementedError

    def move(self, x: float, y: float) -> None:
        raise NotImplementedError

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        raise NotImplementedError

    def drag(self, points: list[tuple[float, float]]) -> None:
        raise NotImplementedError

    def scroll(self, dx: int, dy: int) -> None:
        """Notches at the pointer: positive dy scrolls the content down, positive dx right."""
        raise NotImplementedError

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def front_window(self) -> FrontWindow:
        return FrontWindow()

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        """The window a click at this point would land on, or None when it can't be told."""
        front = self.front_window()
        return front if front.bounds is not None and front.contains(x, y) else None

    def resolve_app(self, name: str) -> str:
        return name

    def open_app(self, name: str) -> str:
        raise BackendError("opening an application by name isn't supported on this system.")

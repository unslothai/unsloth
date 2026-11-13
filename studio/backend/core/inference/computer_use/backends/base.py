# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a platform backend provides. Positions are global desktop points: the unit the OS takes
for input, which on a HiDPI screen is not a pixel of the capture."""

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
    # x, y, width, height in desktop points, when the platform reports them.
    bounds: Optional[tuple[float, float, float, float]] = None

    def contains(self, x: float, y: float) -> bool:
        if self.bounds is None:
            return True
        left, top, width, height = self.bounds
        return left <= x < left + width and top <= y < top + height


class Backend:
    name = "none"
    # The modifier this platform's shortcuts use, told to the model.
    shortcut_modifier = "ctrl"
    os_name = "this computer"

    def unavailable_reason(self) -> Optional[str]:
        """Why this session has no desktop to control, or None."""
        return None

    def permissions(self) -> dict[str, Optional[bool]]:
        """`screen` and `input`: True, False, or None where the platform has no such grant."""
        return {"screen": None, "input": None}

    def request_permission(self, kind: str) -> None:
        """Ask the OS for a grant. Only ever called from Settings, never by a tool call."""

    def displays(self) -> list[Display]:
        raise NotImplementedError

    def capture(self, display: Display) -> Any:
        """A PIL image of the whole display, at whatever resolution the OS gives."""
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
        """The window a click at this point would land on, or None when it can't be told. A
        backend that can't see behind the front window answers for that one only."""
        front = self.front_window()
        return front if front.bounds is not None and front.contains(x, y) else None

    def resolve_app(self, name: str) -> str:
        """The application `open_app` would open for this name, without opening it, so the
        policy is applied to what would really start. The name itself where it can't be told."""
        return name

    def open_app(self, name: str) -> str:
        """Open or focus an application by name; returns the name it opened."""
        raise BackendError("opening an application by name isn't supported on this system.")

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A drawn desktop that touches nothing real, for tests and for trying the tool without handing
over a machine: UNSLOTH_STUDIO_COMPUTER_USE_BACKEND=virtual. It captures at twice its point size,
as a HiDPI screen does."""

from __future__ import annotations

import threading
from typing import Any, Optional

from .base import Backend, BackendError, Display, FrontWindow

WIDTH, HEIGHT, SCALE = 1280, 800, 2
BUTTON = (540, 420, 740, 480)
FIELD = (340, 300, 940, 350)
WINDOW = (240, 160, 800, 440)


class VirtualBackend(Backend):
    name = "virtual"
    shortcut_modifier = "ctrl"
    os_name = "a test desktop"
    captures_pointer = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.pointer = (640.0, 400.0)
            self.text = ""
            self.presses = 0
            self.focused = False
            self.scrolled = 0
            self.events: list[tuple] = []
            # Windows behind the front one, front to back, for tests.
            self.behind: list[FrontWindow] = []
            self.front = FrontWindow(
                app = "Notes", app_id = "virtual.notes", title = "Untitled", pid = 1, bounds = WINDOW
            )

    def displays(self) -> list[Display]:
        return [Display(1, 0, 0, WIDTH, HEIGHT, True)]

    def capture(self, display: Display) -> Any:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size = 14 * SCALE)
        with self._lock:
            image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), (46, 52, 64))
            draw = ImageDraw.Draw(image)

            def box(rect, **kwargs):
                draw.rectangle([value * SCALE for value in rect], **kwargs)

            left, top, width, height = WINDOW
            box((left, top, left + width, top + height), fill = (236, 239, 244))
            box((left, top, left + width, top + 36), fill = (76, 86, 106))
            draw.text(
                ((left + 16) * SCALE, (top + 8) * SCALE), self.front.title, fill = "white", font = font
            )
            box(
                FIELD,
                fill = "white",
                outline = (94, 129, 172) if self.focused else (160, 160, 160),
                width = 2 * SCALE,
            )
            draw.text(
                ((FIELD[0] + 10) * SCALE, (FIELD[1] + 14) * SCALE),
                self.text or "Type here",
                fill = "black",
                font = font,
            )
            box(BUTTON, fill = (94, 129, 172))
            draw.text(
                ((BUTTON[0] + 8) * SCALE, (BUTTON[1] + 20) * SCALE),
                f"Pressed {self.presses}, scrolled {self.scrolled}",
                fill = "white",
                font = font,
            )
            x, y = self.pointer
            draw.polygon(
                [
                    (x * SCALE, y * SCALE),
                    (x * SCALE, (y + 16) * SCALE),
                    ((x + 11) * SCALE, (y + 11) * SCALE),
                ],
                fill = "black",
                outline = "white",
            )
            return image

    def cursor(self) -> tuple[float, float]:
        return self.pointer

    def move(self, x: float, y: float) -> None:
        with self._lock:
            self.pointer = (x, y)
            self.events.append(("move", x, y))

    def click(self, x: float, y: float, button: str, count: int, mods: tuple[str, ...]) -> None:
        with self._lock:
            self.pointer = (x, y)
            self.events.append(("click", x, y, button, count, mods))
            if button != "left":
                return
            if BUTTON[0] <= x < BUTTON[2] and BUTTON[1] <= y < BUTTON[3]:
                self.presses += count
            self.focused = FIELD[0] <= x < FIELD[2] and FIELD[1] <= y < FIELD[3]

    def drag(self, points: list[tuple[float, float]]) -> None:
        with self._lock:
            self.pointer = points[-1]
            self.events.append(("drag", points[0], points[-1]))

    def scroll(self, dx: int, dy: int) -> None:
        with self._lock:
            self.scrolled += dy
            self.events.append(("scroll", dx, dy))

    def key(self, mods: tuple[str, ...], key: Optional[str]) -> None:
        with self._lock:
            self.events.append(("key", mods, key))
            if self.focused and not mods and key == "backspace":
                self.text = self.text[:-1]

    def type_text(self, text: str) -> None:
        with self._lock:
            self.events.append(("type", text))
            if self.focused:
                self.text += text

    def front_window(self) -> FrontWindow:
        return self.front

    def window_at(self, x: float, y: float) -> Optional[FrontWindow]:
        for window in (self.front, *self.behind):
            if window.bounds is not None and window.contains(x, y):
                return window
        return None

    def open_app(self, name: str) -> str:
        if name.casefold() == "missing":
            raise BackendError(f"no application named {name} was found.")
        with self._lock:
            self.events.append(("open_app", name))
            self.front = FrontWindow(
                app = name, app_id = f"virtual.{name.lower()}", title = name, pid = 2, bounds = WINDOW
            )
        return name

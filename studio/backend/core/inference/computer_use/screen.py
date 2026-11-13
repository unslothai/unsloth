# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Screenshots as the model sees them, and the frame that maps a point in one back to the desktop.
The server owns that mapping: a model only ever names a point in a picture it was sent."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .backends import Backend, Display

# Vision towers cut the image into 28 px cells and models that answer in pixels (Qwen2.5-VL,
# UI-TARS 1.5) count them on the resized image, so the picture is sent already on that grid. The
# long edge stays under mcp_images.MAX_IMAGE_EDGE, or the image would be resized again.
CELL = 28
MAX_EDGE = 1008
GRID = 1000
JPEG_QUALITY = 80
# A grey level step that means the picture there is different, not merely shaded.
_LEVEL_STEP = 28
# Share of the patch around a target that may differ before the target counts as moved.
_PATCH_CHANGED = 0.30
_PATCH_RADIUS = 80
# "Nothing changed" has to notice one digit ticking over, so it looks for any real step at all.
_STILL_STEP = 10
_STILL_PIXELS = 4
_POINTER_MASK = 20
_SETTLE_FIRST_S = 0.3
_SETTLE_STEP_S = 0.3
_SETTLE_STEPS = 3
_SETTLED = 0.002


def fit_size(width: float, height: float) -> tuple[int, int]:
    scale = min(1.0, MAX_EDGE / max(width, height, 1))

    def snap(value: float) -> int:
        return min(MAX_EDGE, max(CELL, int(round(value * scale / CELL)) * CELL))

    return snap(width), snap(height)


@dataclass
class Frame:
    """One screenshot the model was sent: the display it shows and the picture's size."""

    display: Display
    width: int
    height: int
    gray: Any = field(repr = False, default = None)
    # Sent because the screen had changed under a target: the next action is not re-checked,
    # or a playing video could never be clicked.
    lenient: bool = False

    def to_desktop(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.display.x + x * self.display.width / self.width,
            self.display.y + y * self.display.height / self.height,
        )

    def from_desktop(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.display.x) * self.width / self.display.width,
            (y - self.display.y) * self.height / self.display.height,
        )


def image_point(frame: Frame, mode: str, x: float, y: float) -> tuple[float, float]:
    """A point as the model named it, in the frame's pixels; ValueError says how it is off."""
    if mode == "grid":
        if x > GRID or y > GRID:
            raise ValueError("x and y run from 0 to 1000 across the screenshot.")
        x, y = x / GRID * frame.width, y / GRID * frame.height
    elif x > frame.width or y > frame.height:
        raise ValueError(
            f"({x:g}, {y:g}) is outside the screenshot, which is "
            f"{frame.width}x{frame.height} pixels."
        )
    return min(x, frame.width - 1), min(y, frame.height - 1)


@dataclass
class Shot:
    frame: Frame
    image: Any = field(repr = False)
    full: Any = field(repr = False)


def take(backend: Backend, display: Display) -> Shot:
    from PIL import Image, ImageDraw

    full = backend.capture(display)
    size = fit_size(display.width, display.height)
    image = full.resize(size, Image.Resampling.LANCZOS, reducing_gap = 2.0)
    frame = Frame(display, size[0], size[1])
    if not getattr(backend, "captures_pointer", False):
        try:
            x, y = frame.from_desktop(*backend.cursor())
        except Exception:  # noqa: BLE001 - the picture is still worth sending without a pointer
            x = y = -1.0
        if 0 <= x < size[0] and 0 <= y < size[1]:
            ImageDraw.Draw(image).polygon(
                [(x, y), (x, y + 17), (x + 4.5, y + 13), (x + 8, y + 20), (x + 11, y + 18.5),
                 (x + 7.5, y + 12), (x + 12.5, y + 12)],
                fill = "black",
                outline = "white",
            )  # fmt: skip
    frame.gray = image.convert("L")
    return Shot(frame, image, full)


def _changed_pixels(
    before: Any,
    after: Any,
    box: Optional[tuple] = None,
) -> tuple[int, int]:
    from PIL import ImageChops

    if before.size != after.size:
        return 1, 1
    diff = ImageChops.difference(before, after)
    if box is not None:
        diff = diff.crop(box)
    total = diff.width * diff.height
    return sum(diff.histogram()[_LEVEL_STEP:]), max(1, total)


def target_moved(frame: Frame, now: Frame, x: float, y: float) -> bool:
    """Whether the screen around an image point differs between the frame the model chose the
    point in and the screen as it is: a dialog opened, the page scrolled, a window moved."""
    if frame.gray is None or now.gray is None or frame.gray.size != now.gray.size:
        return True
    width, height = frame.gray.size
    box = (
        int(max(0, x - _PATCH_RADIUS)),
        int(max(0, y - _PATCH_RADIUS)),
        int(min(width, x + _PATCH_RADIUS)),
        int(min(height, y + _PATCH_RADIUS)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    changed, total = _changed_pixels(frame.gray, now.gray, box)
    return changed / total > _PATCH_CHANGED


def unchanged(before: Frame, after: Frame, pointers: list[tuple[float, float]]) -> bool:
    """True when nothing but the pointer differs, so the picture need not be sent again."""
    if before.gray is None or after.gray is None or before.gray.size != after.gray.size:
        return False
    from PIL import ImageChops, ImageDraw

    diff = ImageChops.difference(before.gray, after.gray)
    draw = ImageDraw.Draw(diff)
    for x, y in pointers:
        draw.rectangle(
            [x - _POINTER_MASK, y - _POINTER_MASK, x + _POINTER_MASK, y + _POINTER_MASK], fill = 0
        )
    return sum(diff.histogram()[_STILL_STEP:]) <= _STILL_PIXELS


def settle(shoot: Callable[[], Shot], sleep: Callable[[float], bool]) -> Shot:
    """The screen once it has stopped moving after an action, or after a few looks at one that
    never does (a video, a spinner). `sleep` returns False when the run was stopped."""
    if not sleep(_SETTLE_FIRST_S):
        return shoot()
    shot = shoot()
    for _ in range(_SETTLE_STEPS):
        if not sleep(_SETTLE_STEP_S):
            break
        following = shoot()
        changed, total = _changed_pixels(shot.frame.gray, following.frame.gray)
        shot = following
        if changed / total <= _SETTLED:
            break
    return shot


def zoom(shot: Shot, region: tuple[float, float, float, float]) -> Any:
    """`region` of the picture, cut from the capture at its own resolution so small text reads."""
    from PIL import Image

    frame, full = shot.frame, shot.full
    sx, sy = full.width / frame.width, full.height / frame.height
    x0, y0, x1, y1 = region
    box = (
        int(max(0, min(x0, frame.width) * sx)),
        int(max(0, min(y0, frame.height) * sy)),
        int(min(full.width, min(x1, frame.width) * sx)),
        int(min(full.height, min(y1, frame.height) * sy)),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("region is outside the screenshot.")
    crop = full.crop(box)
    wanted = max(max(crop.size), 2 * max(x1 - x0, y1 - y0))
    scale = min(MAX_EDGE, wanted) / max(crop.size)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    return crop.resize(size, Image.Resampling.LANCZOS) if size != crop.size else crop


def envelope(image: Any) -> str:
    """The tool-result suffix the chat loop turns into an image for the model and the card."""
    from ..mcp_images import SENTINEL

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format = "JPEG", quality = JPEG_QUALITY)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "\n" + SENTINEL + json.dumps([{"data": data, "mimeType": "image/jpeg"}])

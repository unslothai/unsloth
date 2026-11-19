# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Screenshot sizes from the loaded model's own image-token counts: clicks land by tokens per area."""

from __future__ import annotations

import math
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Past this a screenshot costs more than it gains on every model measured.
COST_CEILING = 4096
# Below this many tokens per screen the measured models stop landing small targets.
PRECISE = 1100
# One screenshot may take a quarter of the context, but past PRECISE it aims at an eighth: the
# rolling window then keeps ~5 screenshots of an 8k context before evicting the oldest turns.
_MOST_OF_CONTEXT = 4
_AIM_OF_CONTEXT = 8
_ASPECT_SLACK = 0.05
# Sizes the cell is estimated from, larger first; the smaller stays under a low --image-max-tokens.
_CELL_PROBES = (896, 448)
# A square of this many cells a side checks the estimate: under every cap, over every floor measured.
_CHECK_CELLS = 12

# Prompt tokens an image of (width, height) costs on the loaded model, or None when unknown.
Count = Callable[[int, int], Optional[int]]
# The loaded context, read at each capture: a server restarted mid-loop may hold less.
Context = Callable[[], Optional[int]]

_logged: set = set()


def _log_once(message: str, *args) -> None:
    key = message % args
    if key not in _logged:
        _logged.add(key)
        logger.info(message, *args)


def grid(aspect: float, tokens: int) -> Optional[tuple[int, int]]:
    """The fullest columns x rows of at most `tokens` cells within 5% of `aspect`."""
    best = None
    for rows in range(1, tokens + 1):
        cols = min(tokens // rows, math.floor(rows * aspect * (1 + _ASPECT_SLACK)))
        if cols < max(1, rows * aspect * (1 - _ASPECT_SLACK)):
            continue
        if best is None or cols * rows > best[0] * best[1]:
            best = (cols, rows)
    return best


class ImageBudget:
    """Screenshot sizes for one loaded model; counts are asked lazily and cached."""

    def __init__(self, count: Count, context: Context) -> None:
        self._count = count
        self._context = context
        self._cell: Optional[tuple[int, int]] = None
        self._sizes: dict = {}

    def _cell_size(self) -> Optional[tuple[int, int]]:
        """(edge in pixels of one image token, tokens around an image's cells); cell 0 when no
        whole-cell grid counts exactly, as with a fixed-size encoder."""
        if self._cell is None:
            found = (0, 0)
            for probe in _CELL_PROBES:
                small = self._count(probe, probe)
                wide = self._count(2 * probe, probe)
                if small is None or wide is None:
                    return None
                # Doubling the width adds one square's worth of cells: (probe / cell) ** 2.
                cell = round(probe / math.sqrt(wide - small)) if wide > small else 0
                if not 8 <= cell <= 128:
                    continue
                square = self._count(_CHECK_CELLS * cell, _CHECK_CELLS * cell)
                longer = self._count((_CHECK_CELLS + 1) * cell, _CHECK_CELLS * cell)
                if square is None or longer is None:
                    return None
                # A floor or cap resizing the image, or a misread cell, breaks the one-column step.
                if longer - square == _CHECK_CELLS:
                    found = (cell, square - _CHECK_CELLS**2)
                    break
            self._cell = found
        return self._cell

    def size(
        self,
        width: float,
        height: float,
        pixels: Optional[tuple[int, int]] = None,
    ) -> Optional[tuple[int, int]]:
        """The screenshot size for a display of this shape, or None to keep the default. With the
        capture's `pixels`, no more cells than they fill unless that is under the precision floor."""
        aspect = width / max(height, 1)
        context = self._context()
        if not context:
            return None
        key = (round(aspect, 3), context, pixels)
        if key in self._sizes:
            return self._sizes[key]
        cells = self._cell_size()
        if cells is None:
            return None
        cell, wrapper = cells
        found = None
        if cell:
            # A quarter more cells than the ceiling, so its count is the encoder's own cap.
            long_edge = math.ceil(cell * math.sqrt(1.25 * COST_CEILING * max(aspect, 1 / aspect)))
            oversized = (
                (long_edge, round(long_edge / aspect))
                if aspect >= 1
                else (round(long_edge * aspect), long_edge)
            )
            cap = self._count(*oversized)
            if cap is None:
                return None
            precise = min(cap, PRECISE)
            target = min(cap, COST_CEILING, max(precise, context // _AIM_OF_CONTEXT))
            if pixels:
                # Upscaling past the capture adds tokens, not detail.
                native = (pixels[0] // cell) * (pixels[1] // cell) + wrapper
                target = min(target, max(precise, native))
            shape = grid(aspect, target - wrapper)
            if context // _MOST_OF_CONTEXT < precise:
                _log_once(
                    "computer use: screenshots stay at the default size; a %d-token context "
                    "is too small for the %d image tokens this model needs to click small "
                    "targets precisely",
                    context,
                    precise,
                )
            elif shape is not None:
                sized = (shape[0] * cell, shape[1] * cell)
                tokens = self._count(*sized)
                if tokens is None:
                    return None
                # Anything else means the server resizes this image, and x and y would not map back.
                if tokens == shape[0] * shape[1] + wrapper:
                    found = sized
        self._sizes[key] = found
        return found

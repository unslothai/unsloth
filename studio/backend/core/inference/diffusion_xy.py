# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""X/Y grid over /images/generate (AUTOMATIC1111's X/Y plot): the cell requests an `xy` spec expands to, and the
labelled grid image composed from the rendered cells. Pure apart from Pillow, so tests drive it without a model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

# The client's Compare grid holds its canvas to the same budget.
MAX_GRID_PIXELS = 32 * 1024 * 1024

_LABELS = {
    "seed": "Seed",
    "steps": "Steps",
    "guidance": "CFG",
    "shift": "Shift",
    "base_shift": "Base shift",
    "max_shift": "Max shift",
    "strength": "Denoise",
    "clip_skip": "Clip skip",
    "cfg_rescale": "CFG rescale",
    "variation_seed": "Var. seed",
    "variation_strength": "Var. strength",
    "lora_weight": "LoRA weight",
}


@dataclass
class XYCell:
    ix: int
    iy: int
    label: str
    request: Any  # DiffusionGenerateRequest


@dataclass
class XYPlan:
    cells: list[XYCell]  # row by row: index ix + iy * nx
    nx: int
    ny: int
    x_labels: list[str]
    y_labels: list[str]
    order: list[int]  # the order the cells render in

    def describe(self, axes: list[Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, axis, labels in zip(("x", "y"), axes, (self.x_labels, self.y_labels)):
            if axis is not None:
                out[name] = {
                    "field": axis.field,
                    "index": axis.index,
                    "values": list(axis.values),
                    "labels": labels,
                }
        return out


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def axis_label(axis: Any, value: Any) -> str:
    from core.inference.diffusion_sampler import sampler_label, schedule_label

    if axis.field == "prompt_sr":
        return str(value)
    if axis.field == "sampler":
        return sampler_label(str(value)) or ("Default" if value == "default" else str(value))
    if axis.field == "schedule_type":
        return schedule_label(str(value)) or ("Automatic" if value == "automatic" else str(value))
    return f"{_LABELS[axis.field]}: {_number(value)}"


def _apply(data: dict[str, Any], axis: Any, value: Any) -> None:
    if axis.field == "prompt_sr":
        search = axis.values[0]
        prompt = data.get("prompt") or ""
        negative = data.get("negative_prompt")
        if search not in prompt and search not in (negative or ""):
            raise ValueError(
                f'xy prompt_sr: "{search}" is not in the prompt or the negative prompt'
            )
        data["prompt"] = prompt.replace(search, str(value))
        if negative is not None:
            data["negative_prompt"] = negative.replace(search, str(value))
    elif axis.field == "lora_weight":
        data["loras"] = [dict(l) for l in data["loras"]]
        data["loras"][axis.index]["weight"] = value
    else:
        data[axis.field] = value


def expand(request: Any, rng: Optional[random.Random] = None) -> XYPlan:
    """Cell requests for ``request.xy``, validated as full generate requests. Raises ValueError naming the cell."""
    from models.inference import DiffusionGenerateRequest

    spec = request.xy
    axes = [spec.x, spec.y]
    # Unset fields stay unset per cell, so omitted steps / guidance still follow the loaded model.
    base = request.model_dump(exclude_unset = True, exclude = {"xy"})
    if (
        spec.x.field != "seed"
        and (spec.y is None or spec.y.field != "seed")
        and base.get("seed") is None
    ):
        # One seed for every cell, so only the swept setting differs (A1111 fix_seed).
        base["seed"] = (rng or random).randrange(2**32)
    # A rewrite per cell would change the thing being compared.
    base.setdefault("enhance_prompt", False)
    x_values = list(spec.x.values)
    y_values = list(spec.y.values) if spec.y else [None]
    x_labels = [axis_label(spec.x, v) for v in x_values]
    y_labels = [axis_label(spec.y, v) for v in y_values] if spec.y else [""]
    cells: list[XYCell] = []
    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            label = ", ".join(l for l in (x_labels[ix], y_labels[iy]) if l)
            data = dict(base)
            _apply(data, spec.x, x_value)
            if spec.y is not None:
                _apply(data, spec.y, y_value)
            if data.get("loras"):
                # Weight 0 drops the adapter, as the Images page sends it.
                data["loras"] = [l for l in data["loras"] if l["weight"] != 0] or None
            try:
                cell = DiffusionGenerateRequest.model_validate(data)
            except ValidationError as exc:
                err = exc.errors()[0]
                where = ".".join(str(p) for p in err.get("loc", ())) or "request"
                raise ValueError(f"xy cell ({label}): {where}: {err.get('msg')}") from None
            cells.append(XYCell(ix, iy, label, cell))
    base_loras = len(base.get("loras") or [])
    # Adapter-free cells last: Speed Auto may compile on an adapter-free run, and a compiled load refuses LoRA after.
    order = sorted(range(len(cells)), key = lambda i: len(cells[i].request.loras or []) < base_loras)
    return XYPlan(cells, len(x_values), len(y_values), x_labels, y_labels, order)


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size = size)
    except (TypeError, OSError):  # Pillow without FreeType: the fixed bitmap font
        return ImageFont.load_default()


def _wrap(text: str, width: float, font) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        nxt = f"{line} {word}" if line else word
        if line and font.getlength(nxt) > width:
            lines.append(line)
            line = word
        else:
            line = nxt
    if line:
        lines.append(line)
    return lines


def compose_grid(images: list[Any], plan: XYPlan):
    """The labelled grid, laid out after A1111's draw_grid_annotations: column labels above, row labels in a left
    band 3/4 of a cell wide, font size (w + h) / 25. Cells are scaled down to keep the grid under MAX_GRID_PIXELS."""
    from PIL import Image, ImageDraw

    cw = max(im.width for im in images)
    ch = max(im.height for im in images)
    scale = min(1.0, math.sqrt(MAX_GRID_PIXELS / (plan.nx * plan.ny * cw * ch)))
    if scale < 1.0:
        cw, ch = max(1, int(cw * scale)), max(1, int(ch * scale))
    font_size = max(12, (cw + ch) // 25)
    font = _font(font_size)
    line_h = int(font_size * 1.25)
    has_rows = any(plan.y_labels)
    left = cw * 3 // 4 if has_rows else 0
    col_lines = [_wrap(t, cw - font_size, font) for t in plan.x_labels]
    row_lines = [_wrap(t, left - font_size, font) for t in plan.y_labels] if has_rows else []
    top = max(len(l) for l in col_lines) * line_h + font_size if any(col_lines) else 0
    grid = Image.new("RGB", (left + plan.nx * cw, top + plan.ny * ch), "white")
    draw = ImageDraw.Draw(grid)

    def _text(lines: list[str], cx: float, y: float) -> None:
        for k, line in enumerate(lines):
            draw.text(
                (cx - font.getlength(line) / 2, y + k * line_h), line, fill = "black", font = font
            )

    for ix, lines in enumerate(col_lines):
        _text(lines, left + ix * cw + cw / 2, font_size / 2)
    for iy, lines in enumerate(row_lines):
        _text(lines, left / 2, top + iy * ch + (ch - len(lines) * line_h) / 2)
    for index, image in enumerate(images):
        cell = plan.cells[index]
        if image.size != (cw, ch):
            image = image.copy()
            image.thumbnail((cw, ch), Image.LANCZOS)
        x = left + cell.ix * cw + (cw - image.width) // 2
        y = top + cell.iy * ch + (ch - image.height) // 2
        grid.paste(image.convert("RGB"), (x, y))
    return grid

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image nodes: generation and editing through Studio's image routes, plus local Pillow layout steps."""

from __future__ import annotations

import asyncio
import io
import textwrap
from typing import Any, Optional

from core.flows.nodes._common import as_text, parse_size, render, template_vars
from core.flows.registry import FlowNodeError, NodeContext, handler


def image_prompt(
    ctx: NodeContext,
    param: str = "prompt",
    port: str = "prompt",
) -> str:
    template = str(ctx.params.get(param) or "")
    incoming = as_text(ctx.inputs.get(port)) if port in ctx.inputs else ""
    if template.strip():
        prompt = render(template, template_vars(ctx))
        if incoming and "{{" + port not in template.replace(" ", ""):
            prompt = f"{incoming}, {prompt}"
    else:
        prompt = incoming
    style = str(ctx.params.get("style") or "").strip()
    if style:
        prompt = f"{prompt}, {style}" if prompt else style
    return prompt.strip()


@handler("image.generate")
async def generate(ctx: NodeContext) -> dict:
    width, height = parse_size(ctx.params.get("size"))
    images = await ctx.generate_image(
        model = ctx.params.get("model"),
        prompt = image_prompt(ctx),
        negative_prompt = str(ctx.params.get("negative_prompt") or ""),
        width = width,
        height = height,
        steps = ctx.params.get("steps") or None,
        guidance = ctx.params.get("guidance"),
        seed = ctx.seed,
        init_image = ctx.inputs.get("init_image"),
        strength = ctx.params.get("strength") if ctx.inputs.get("init_image") else None,
        references = ctx.inputs.get("reference") or None,
        workflow = "reference"
        if ctx.inputs.get("reference") and ctx.inputs.get("init_image")
        else None,
        count = int(ctx.params.get("count") or 1),
    )
    return {"image": images[0], "images": images}


@handler("image.edit")
async def edit(ctx: NodeContext) -> dict:
    instruction = image_prompt(ctx, "instruction", "instruction")
    if not instruction:
        raise FlowNodeError("Say what to change: type an instruction or connect one")
    source, extra = ctx.inputs["image"], list(ctx.inputs.get("reference") or [])
    # Families without instruction editing (FLUX.2 klein) still take the source as a reference, else as img2img.
    attempts = [
        {"init_image": source, "references": extra or None, "workflow": "edit"},
        {"init_image": None, "references": [source, *extra], "workflow": "reference"},
        {"init_image": source, "references": None, "workflow": None, "strength": 0.6},
    ]
    for i, attempt in enumerate(attempts):
        try:
            images = await ctx.generate_image(
                model = ctx.params.get("model"), prompt = instruction, seed = ctx.seed, **attempt
            )
            return {"image": images[0]}
        except FlowNodeError as exc:
            if i == len(attempts) - 1 or "not supported for the" not in str(exc):
                raise
            ctx.log(
                f"{attempt['workflow']} is not supported by this model; trying another way", "info"
            )


@handler("image.remove_bg")
async def remove_bg(ctx: NodeContext) -> dict:
    cutout, mask = await ctx.remove_background(
        ctx.inputs["image"],
        str(ctx.params.get("subject") or "the main subject"),
        bool(ctx.params.get("trim", True)),
    )
    return {"image": cutout, "mask": mask}


@handler("image.upscale")
async def upscale(ctx: NodeContext) -> dict:
    source = ctx.inputs["image"]
    images = await ctx.generate_image(
        model = None,
        prompt = source.get("prompt") or "high quality, sharp details",
        seed = ctx.seed,
        init_image = source,
        upscale = float(ctx.params.get("factor") or 2),
    )
    return {"image": images[0]}


def _open(data: bytes) -> Any:
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGBA")


def fit(im: Any, width: int, height: int, mode: str) -> Any:
    from PIL import Image, ImageOps

    if mode == "stretch":
        return im.resize((width, height), Image.LANCZOS)
    if mode == "contain":
        inner = ImageOps.contain(im, (width, height), Image.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.paste(inner, ((width - inner.width) // 2, (height - inner.height) // 2))
        return canvas
    if mode == "center-crop":
        left = max(0, (im.width - width) // 2)
        top = max(0, (im.height - height) // 2)
        cropped = im.crop((left, top, left + min(width, im.width), top + min(height, im.height)))
        return fit(cropped, width, height, "cover") if cropped.size != (width, height) else cropped
    return ImageOps.fit(im, (width, height), Image.LANCZOS)


def _png(im: Any) -> bytes:
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


@handler("image.resize")
async def resize(ctx: NodeContext) -> dict:
    data, _ = await ctx.media_bytes(ctx.inputs["image"])
    width = max(8, min(int(ctx.params.get("width") or 512), 4096))
    height = max(8, min(int(ctx.params.get("height") or 512), 4096))
    mode = str(ctx.params.get("mode") or "cover")
    png = await asyncio.to_thread(lambda: _png(fit(_open(data), width, height, mode)))
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


def _font(size: int) -> Any:
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size = size)
    except TypeError:
        return ImageFont.load_default()


def compose_grid(
    frames: list[bytes],
    captions: list[str],
    *,
    columns: int,
    gap: int,
    background: str,
    caption_style: str,
    cell: int,
) -> bytes:
    from PIL import Image, ImageColor, ImageDraw

    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    width = columns * cell + (columns + 1) * gap
    height = rows * cell + (rows + 1) * gap
    try:
        bg = ImageColor.getrgb(background)
    except ValueError:
        bg = (17, 17, 17)
    page = Image.new("RGBA", (width, height), (*bg[:3], 255))
    font = _font(max(14, cell // 18))
    for i, data in enumerate(frames):
        x = gap + (i % columns) * (cell + gap)
        y = gap + (i // columns) * (cell + gap)
        panel = fit(_open(data), cell, cell, "cover")
        page.alpha_composite(panel, (x, y))
        text = captions[i].strip() if i < len(captions) and captions[i] else ""
        if not text or caption_style == "none":
            continue
        draw = ImageDraw.Draw(page)
        wrapped = textwrap.fill(text, width = max(12, cell // max(8, font.size // 2 + 4)))
        box = draw.multiline_textbbox((0, 0), wrapped, font = font, spacing = 4)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = max(6, cell // 60)
        if caption_style == "comic":
            bx0, by0 = x + pad, y + pad
            bx1, by1 = min(x + cell - pad, bx0 + tw + 2 * pad), by0 + th + 2 * pad
            draw.rectangle(
                (bx0, by0, bx1, by1), fill = (255, 250, 235, 255), outline = (0, 0, 0, 255), width = 3
            )
            draw.multiline_text(
                (bx0 + pad, by0 + pad - box[1]), wrapped, fill = (0, 0, 0, 255), font = font, spacing = 4
            )
        else:
            band = Image.new("RGBA", (cell, th + 2 * pad), (0, 0, 0, 160))
            page.alpha_composite(band, (x, y + cell - band.height))
            draw = ImageDraw.Draw(page)
            draw.multiline_text(
                (x + (cell - tw) // 2, y + cell - band.height + pad - box[1]),
                wrapped,
                fill = (255, 255, 255, 255),
                font = font,
                spacing = 4,
                align = "center",
            )
    return _png(page)


@handler("image.grid")
async def grid(ctx: NodeContext) -> dict:
    images = [i for i in ctx.inputs.get("images") or [] if isinstance(i, dict)]
    if not images:
        raise FlowNodeError("No images to lay out")
    if len(images) > 64:
        raise FlowNodeError("A grid takes at most 64 images")
    frames = [(await ctx.media_bytes(i))[0] for i in images]
    captions = [as_text(c) for c in ctx.inputs.get("captions") or []]
    png = await asyncio.to_thread(
        compose_grid,
        frames,
        captions,
        columns = int(ctx.params.get("columns") or 2),
        gap = int(ctx.params.get("gap") or 0),
        background = str(ctx.params.get("background") or "#111111"),
        caption_style = str(ctx.params.get("caption_style") or "comic"),
        cell = max(64, min(int(ctx.params.get("cell_size") or 512), 2048)),
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


def sprite_sheet(frames: list[bytes], size: int, columns: int) -> tuple[bytes, dict]:
    from PIL import Image

    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * size, rows * size), (0, 0, 0, 0))
    for i, data in enumerate(frames):
        sheet.alpha_composite(
            fit(_open(data), size, size, "contain"), ((i % columns) * size, (i // columns) * size)
        )
    meta = {
        "frame_width": size,
        "frame_height": size,
        "columns": columns,
        "rows": rows,
        "count": len(frames),
    }
    return _png(sheet), meta


@handler("image.sprite_sheet")
async def sheet(ctx: NodeContext) -> dict:
    frames_in = [f for f in ctx.inputs.get("frames") or [] if isinstance(f, dict)]
    if not frames_in:
        raise FlowNodeError("No frames to pack")
    if len(frames_in) > 256:
        raise FlowNodeError("A sprite sheet takes at most 256 frames")
    frames = [(await ctx.media_bytes(f))[0] for f in frames_in]
    size = max(8, min(int(ctx.params.get("frame_size") or 64), 1024))
    png, meta = await asyncio.to_thread(
        sprite_sheet, frames, size, int(ctx.params.get("columns") or 4)
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image"), "meta": meta}


def size_of(value: Optional[str]) -> tuple[int, int]:
    return parse_size(value, (832, 480))

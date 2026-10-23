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


OPTION_PARAMS = (
    "loras",
    "sampler",
    "schedule_type",
    "clip_skip",
    "cfg_rescale",
    "shift",
    "mask_blur",
    "only_masked",
    "inpaint_padding",
    "hires_scale",
    "hires_denoise",
    "hires_steps",
    "hires_upscaler",
)


def control_spec(ctx: NodeContext) -> Optional[dict]:
    """The ControlNet settings when a control image is connected, else None."""
    image = ctx.inputs.get("control_image")
    controlnet = str(ctx.params.get("controlnet") or "").strip()
    if image is None:
        if controlnet:
            raise FlowNodeError("Connect a control image for the ControlNet")
        return None
    if not controlnet:
        raise FlowNodeError("Pick a ControlNet for the control image")
    start = float(ctx.params.get("control_start") or 0.0)
    end = ctx.params.get("control_end")
    end = 1.0 if end is None else float(end)
    if start > end:
        raise FlowNodeError("ControlNet start must not be after its end")
    strength = ctx.params.get("control_strength")
    return {
        "id": controlnet,
        "image": image,
        "control_type": str(ctx.params.get("control_type") or "passthrough"),
        "strength": 1.0 if strength is None else float(strength),
        "guidance_start": start,
        "guidance_end": end,
    }


@handler("image.generate")
async def generate(ctx: NodeContext) -> dict:
    width, height = parse_size(ctx.params.get("size"))
    init = ctx.inputs.get("init_image")
    refs = list(ctx.inputs.get("reference") or [])
    workflow = None
    if refs and ctx.inputs.get("mask"):
        raise FlowNodeError(
            "Connect either a mask or reference images: the reference workflow does not take a mask"
        )
    if refs:
        # Studio's reference workflow takes a source image plus extra references.
        workflow = "reference"
        if init is None:
            init, refs = refs[0], refs[1:]
    images = await ctx.generate_image(
        model = ctx.params.get("model"),
        prompt = image_prompt(ctx),
        negative_prompt = str(ctx.params.get("negative_prompt") or ""),
        width = width,
        height = height,
        steps = ctx.params.get("steps") or None,
        guidance = ctx.params.get("guidance"),
        seed = ctx.seed,
        init_image = init,
        strength = ctx.params.get("strength") if workflow is None and init is not None else None,
        references = refs or None,
        workflow = workflow,
        count = int(ctx.params.get("count") or 1),
        mask = ctx.inputs.get("mask"),
        control = control_spec(ctx),
        options = {k: ctx.params.get(k) for k in OPTION_PARAMS},
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
        {"init_image": source, "references": extra or None, "workflow": "reference"},
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


def _max_along(a, radius: int, axis: int):
    """Running max over a 2*radius+1 window along one axis, zero-padded, in O(log radius) passes."""
    import numpy as np

    n, window = a.shape[axis], 2 * radius + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    cur, width = np.pad(a, pad), 1

    def take(x, start, stop):
        return x[start:stop] if axis == 0 else x[:, start:stop]

    while width * 2 <= window:
        size = cur.shape[axis]
        cur = np.maximum(take(cur, 0, size - width), take(cur, width, size))
        width *= 2
    return np.maximum(take(cur, 0, n), take(cur, window - width, window - width + n))


def grow_mask(im, radius: int):
    """Widen the white area by `radius` px on every side: the same result as
    ImageFilter.MaxFilter(2 * radius + 1), but separable, so radius 64 on a 2048 px mask
    takes milliseconds instead of minutes."""
    import numpy as np
    from PIL import Image

    a = np.asarray(im.convert("L"))
    return Image.fromarray(_max_along(_max_along(a, radius, 0), radius, 1), "L")


@handler("image.segment")
async def segment(ctx: NodeContext) -> dict:
    from PIL import Image, ImageOps

    png = await ctx.segment(
        ctx.inputs["image"], str(ctx.params.get("subject") or "the main subject")
    )
    grow = max(0, min(int(ctx.params.get("grow") or 0), 64))
    invert = bool(ctx.params.get("invert"))

    def shape() -> bytes:
        im = Image.open(io.BytesIO(png)).convert("L")
        if grow:
            im = grow_mask(im, grow)
        if invert:
            im = ImageOps.invert(im)
        return _png(im)

    out = await asyncio.to_thread(shape) if grow or invert else png
    return {"mask": await ctx.save_asset(out, "image/png", kind = "image")}


MAX_UPSCALE_SIDE = 8192


@handler("image.upscale")
async def upscale(ctx: NodeContext) -> dict:
    source = ctx.inputs["image"]
    factor = float(ctx.params.get("factor") or 2)
    try:
        images = await ctx.generate_image(
            model = None,
            prompt = source.get("prompt") or "high quality, sharp details",
            seed = ctx.seed,
            init_image = source,
            upscale = factor,
            options = {"upscale_model": ctx.params.get("upscaler")},
        )
    except FlowNodeError as exc:
        # No image model loaded (a cached step before this one), or a family that cannot upscale
        # (FLUX.2-klein among them): a plain resize still gives the size asked for.
        if "not supported" not in str(exc) and "model is loaded" not in str(exc):
            raise
        ctx.log(f"{exc} Resized with Lanczos instead.", level = "warning")
        data, _ = await ctx.media_bytes(source)

        def bigger() -> bytes:
            from PIL import Image

            im = _open(data)
            scale = min(factor, MAX_UPSCALE_SIDE / max(im.size))
            return _png(
                im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            )

        return {
            "image": await ctx.save_asset(
                await asyncio.to_thread(bigger), "image/png", kind = "image"
            )
        }
    return {"image": images[0]}


def pad_canvas(
    data: bytes, left: int, top: int, right: int, bottom: int, feather: int
) -> tuple[bytes, bytes]:
    """The image on a larger canvas plus its outpaint mask (white = the new border to paint)."""
    from PIL import Image, ImageFilter

    im = _open(data)
    width, height = im.width + left + right, im.height + top + bottom
    if width > MAX_GRID_SIDE or height > MAX_GRID_SIDE:
        raise FlowNodeError(
            f"The padded image would be {width}x{height}; keep it under {MAX_GRID_SIDE} px"
        )
    # The border starts as the image's edge colours stretched outwards, which outpaints better than flat black.
    canvas = im.resize((width, height), Image.BILINEAR).filter(ImageFilter.GaussianBlur(24))
    canvas.alpha_composite(im, (left, top))
    mask = Image.new("L", (width, height), 255)
    keep = Image.new("L", (im.width, im.height), 0)
    mask.paste(keep, (left, top))
    if feather:
        # Overlap the old edge a little so the seam is repainted too, but only on the sides that grew.
        x0 = left + (feather if left else 0)
        y0 = top + (feather if top else 0)
        x1 = left + im.width - (feather if right else 0)
        y1 = top + im.height - (feather if bottom else 0)
        inner = Image.new("L", (width, height), 0)
        inner.paste(Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 255), (x0, y0))
        mask = Image.eval(inner, lambda v: 255 - v)
    return _png(canvas.convert("RGB")), _png(mask)


@handler("image.pad")
async def pad(ctx: NodeContext) -> dict:
    data, _ = await ctx.media_bytes(ctx.inputs["image"])
    sides = [
        max(0, min(int(ctx.params.get(k) or 0), 2048)) for k in ("left", "top", "right", "bottom")
    ]
    if not any(sides):
        raise FlowNodeError("Set how far to extend at least one side")
    feather = max(0, min(int(ctx.params.get("feather") or 0), 128))
    image_png, mask_png = await asyncio.to_thread(pad_canvas, data, *sides, feather)
    return {
        "image": await ctx.save_asset(image_png, "image/png", kind = "image"),
        "mask": await ctx.save_asset(mask_png, "image/png", kind = "image"),
    }


def _open(data: bytes) -> Any:
    from PIL import Image

    from core.inference.image_orientation import exif_upright

    im = Image.open(io.BytesIO(data))
    im.load()
    # Upright like Studio's image routes and the browser show it, so a phone photo is not turned sideways.
    return exif_upright(im).convert("RGBA")


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


MAX_GRID_SIDE = 6144


def _wrap_label(text: str, width_px: int, font: Any) -> str:
    """Wrap a row label to `width_px` between words, never inside one."""
    return textwrap.fill(
        text, width = max(6, width_px // max(8, font.size // 2 + 4)), break_long_words = False
    )


# Pillow's built-in caption font has no dashes or ellipsis; they would draw as empty boxes.
_ASCII_PUNCT = str.maketrans({"\u2014": "-", "\u2013": "-", "\u2026": "..."})


def compose_grid(
    frames: list[bytes],
    captions: list[str],
    *,
    columns: int,
    gap: int,
    background: str,
    caption_style: str,
    cell: int,
    row_labels: Optional[list[str]] = None,
    column_labels: Optional[list[str]] = None,
) -> bytes:
    from PIL import Image, ImageColor, ImageDraw

    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    # Big grids shrink their cells so the page stays under MAX_GRID_SIDE (8x8 of 2048 px would be ~1 GiB).
    gap = max(0, min(gap, MAX_GRID_SIDE // (4 * (max(columns, rows) + 1))))
    cell = max(
        16,
        min(
            cell,
            (MAX_GRID_SIDE - (columns + 1) * gap) // columns,
            (MAX_GRID_SIDE - (rows + 1) * gap) // rows,
        ),
    )
    font = _font(max(14, cell // 18))
    row_labels = [str(t).strip().translate(_ASCII_PUNCT) for t in row_labels or []][:rows]
    column_labels = [str(t).strip().translate(_ASCII_PUNCT) for t in column_labels or []][:columns]
    # XY plots: a header band above the columns and a label strip left of the rows; none without labels.
    top = (font.size * 2 + gap) if any(column_labels) else 0
    # Wide enough for the longest word, so a row label wraps between words, never inside one.
    longest = max((font.getlength(w) for t in row_labels for w in t.split()), default = 0)
    left = (
        min(cell, max(font.size * 6, cell // 3, int(longest) + gap)) + gap if any(row_labels) else 0
    )
    width = left + columns * cell + (columns + 1) * gap
    height = top + rows * cell + (rows + 1) * gap
    try:
        bg = ImageColor.getrgb(background)
    except ValueError:
        bg = (17, 17, 17)
    page = Image.new("RGBA", (width, height), (*bg[:3], 255))
    ink = (20, 20, 20, 255) if sum(bg[:3]) > 384 else (235, 235, 235, 255)
    draw = ImageDraw.Draw(page)
    for c, text in enumerate(column_labels):
        x = left + gap + c * (cell + gap)
        box = draw.textbbox((0, 0), text, font = font)
        draw.text(
            (
                x + (cell - (box[2] - box[0])) // 2,
                gap + (top - gap - (box[3] - box[1])) // 2 - box[1],
            ),
            text,
            fill = ink,
            font = font,
        )
    for r, text in enumerate(row_labels):
        y = top + gap + r * (cell + gap)
        wrapped = _wrap_label(text, left - gap, font)
        box = draw.multiline_textbbox((0, 0), wrapped, font = font, spacing = 4)
        draw.multiline_text(
            (gap, y + (cell - (box[3] - box[1])) // 2 - box[1]),
            wrapped,
            fill = ink,
            font = font,
            spacing = 4,
        )
    for i, data in enumerate(frames):
        x = left + gap + (i % columns) * (cell + gap)
        y = top + gap + (i // columns) * (cell + gap)
        panel = fit(_open(data), cell, cell, "cover")
        page.alpha_composite(panel, (x, y))
        text = (
            captions[i].strip().translate(_ASCII_PUNCT) if i < len(captions) and captions[i] else ""
        )
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
        row_labels = _labels(ctx, "row_labels"),
        column_labels = _labels(ctx, "column_labels"),
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


def _labels(ctx: NodeContext, key: str) -> list[str]:
    connected = ctx.inputs.get(key)
    values = connected if connected is not None else ctx.params.get(key)
    return [as_text(v) for v in values or [] if v is not None]


def sprite_sheet(frames: list[bytes], size: int, columns: int) -> tuple[bytes, dict]:
    from PIL import Image

    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    # Like Compose Grid: many big frames shrink so the sheet stays under MAX_GRID_SIDE.
    size = max(8, min(size, MAX_GRID_SIDE // max(columns, rows)))
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

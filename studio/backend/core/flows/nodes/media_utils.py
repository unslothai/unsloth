# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local media steps that need no model: adjust, composite and caption images, and trim or re-time videos."""

from __future__ import annotations

import asyncio
import io
import textwrap
from fractions import Fraction
from typing import Any, Optional

from core.flows.nodes.image import _ASCII_PUNCT, _font, _open, _png
from core.flows.registry import FlowNodeError, NodeContext, handler

MAX_SIDE = 8192
MAX_VIDEO_FRAMES = 20_000
MAX_AUDIO_S = 60 * 60


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def adjust_image(
    data: bytes,
    *,
    crop: Optional[tuple[float, float, float, float]] = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    blur: float = 0.0,
    invert: bool = False,
    flip: str = "none",
    rotate: float = 0.0,
) -> bytes:
    """Crop is (left, top, width, height) as fractions of the image; 1.0 for a factor leaves it unchanged."""
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter

    im = _open(data)
    if crop is not None:
        left, top, w, h = (max(0.0, min(1.0, v)) for v in crop)
        box = (
            round(left * im.width),
            round(top * im.height),
            round(min(1.0, left + w) * im.width),
            round(min(1.0, top + h) * im.height),
        )
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            raise FlowNodeError("The crop leaves no pixels")
        im = im.crop(box)
    alpha = im.getchannel("A")
    rgb = im.convert("RGB")
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
    if blur > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(blur))
    if invert:
        rgb = ImageChops.invert(rgb)
    im = rgb.convert("RGBA")
    im.putalpha(alpha)
    if flip == "horizontal":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == "vertical":
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate % 360:
        im = im.rotate(-rotate, resample = Image.BICUBIC, expand = True)
    return _png(im)


def composite_images(
    base: bytes,
    overlay: bytes,
    *,
    x: float = 0.5,
    y: float = 0.5,
    scale: float = 0.5,
    opacity: float = 1.0,
    mask: Optional[bytes] = None,
) -> bytes:
    """Place `overlay` centred at (x, y), fractions of the base, at `scale` times the base width."""
    from PIL import Image

    page = _open(base)
    top = _open(overlay)
    width = max(1, round(page.width * scale))
    height = max(1, round(top.height * width / max(1, top.width)))
    fit = min(1.0, MAX_SIDE / max(width, height))
    width, height = max(1, round(width * fit)), max(1, round(height * fit))
    top = top.resize((width, height), Image.LANCZOS)
    alpha = top.getchannel("A")
    if mask is not None:
        shape = _open(mask).convert("L").resize((width, height), Image.LANCZOS)
        alpha = Image.composite(alpha, Image.new("L", alpha.size, 0), shape)
    if opacity < 1.0:
        alpha = alpha.point(lambda v: round(v * opacity))
    top.putalpha(alpha)
    left = round(x * page.width - width / 2)
    upper = round(y * page.height - height / 2)
    layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
    layer.paste(top, (left, upper))
    page.alpha_composite(layer)
    return _png(page)


POSITIONS = ("top", "center", "bottom", "top-left", "top-right", "bottom-left", "bottom-right")


def text_overlay(
    data: bytes,
    text: str,
    *,
    size: int = 48,
    color: str = "#ffffff",
    position: str = "bottom",
    outline: int = 2,
) -> bytes:
    from PIL import ImageColor, ImageDraw

    page = _open(data)
    text = text.strip().translate(_ASCII_PUNCT)
    if not text:
        return _png(page)
    font = _font(size)
    try:
        fill = ImageColor.getrgb(color)
    except ValueError:
        fill = (255, 255, 255)
    stroke = (0, 0, 0) if sum(fill[:3]) > 382 else (255, 255, 255)
    draw = ImageDraw.Draw(page)
    chars = max(8, int(page.width / max(1, size * 0.55)))
    wrapped = "\n".join(textwrap.fill(line, width = chars) for line in text.splitlines())
    box = draw.multiline_textbbox((0, 0), wrapped, font = font, spacing = 6, stroke_width = outline)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = max(8, page.width // 40)
    vertical, _, horizontal = position.partition("-")
    if position in ("top", "center", "bottom"):
        horizontal = "center"
    x = {"left": pad, "right": page.width - tw - pad}.get(horizontal, (page.width - tw) // 2)
    y = {"top": pad, "bottom": page.height - th - pad}.get(vertical, (page.height - th) // 2)
    draw.multiline_text(
        (x - box[0], y - box[1]),
        wrapped,
        fill = (*fill[:3], 255),
        font = font,
        spacing = 6,
        align = "center" if horizontal == "center" else horizontal,
        stroke_width = outline,
        stroke_fill = (*stroke, 255),
    )
    return _png(page)


def _upright(frame: Any) -> Any:
    import numpy as np

    array = frame.to_ndarray(format = "rgb24")
    turns = round(float(getattr(frame, "rotation", 0) or 0) / 90) % 4
    return np.ascontiguousarray(np.rot90(array, k = turns)) if turns else array


def _encoder(container: Any, rate: Fraction) -> Any:
    for name in ("libx264", "libopenh264", "mpeg4"):
        try:
            return container.add_stream(name, rate = rate)
        except Exception:  # noqa: BLE001 - not in this FFmpeg build; try the next
            continue
    raise FlowNodeError("This Unsloth Studio has no video encoder")


def edit_video(
    data: bytes,
    *,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    speed: float = 1.0,
    audio: Optional[bytes] = None,
) -> bytes:
    """Trim to [start_s, end_s), play at `speed`, and keep the clip's sound or put `audio` under it."""
    import av

    speed = max(0.25, min(4.0, speed))
    source = av.open(io.BytesIO(data))
    vstream = source.streams.video[0]
    rate = Fraction(vstream.average_rate or 24).limit_denominator(1000)
    # A variable-frame-rate clip averages below its real frame rate: resample on the finer stream rate.
    fine = getattr(vstream, "base_rate", None)
    if fine and rate < fine <= 120:
        rate = Fraction(fine).limit_denominator(1000)
    own_sound = bool(source.streams.audio)
    out_buf = io.BytesIO()
    out = vout = aout = None
    # Frames stream straight from the decoder to the encoder: holding a decoded 4K clip would need tens of GB.
    # Output frames sit on a constant-rate grid over the source timeline (output frame n shows the source at
    # base + n * speed / rate, sample and hold), so variable-frame-rate footage keeps its pacing.
    kept = emitted = decoded = 0
    step = speed / float(rate)
    # Half a step: millisecond-rounded timestamps (WebM) would otherwise land a frame on the wrong grid point.
    slack = min(step, 1.0 / float(rate)) / 2
    base = held = held_t = None
    frame_s = 1.0 / float(rate)

    def emit_until(limit: float) -> None:
        nonlocal emitted
        picture = None
        while base + emitted * step + slack < limit:
            if emitted >= 4 * MAX_VIDEO_FRAMES:
                raise FlowNodeError("The clip is too long to edit here")
            if picture is None:
                picture = av.VideoFrame.from_ndarray(
                    held[: vout.height, : vout.width], format = "rgb24"
                )
            for packet in vout.encode(picture):
                out.mux(packet)
            emitted += 1

    try:
        if start_s > 0 and vstream.time_base:
            source.seek(int(start_s / vstream.time_base), stream = vstream, backward = True)
        for frame in source.decode(vstream):
            decoded += 1
            if decoded > 2 * MAX_VIDEO_FRAMES:
                raise FlowNodeError("The clip is too long to edit here")
            t = float(frame.time or 0.0)
            if t < start_s:
                continue
            if end_s is not None and t >= end_s:
                break
            if kept >= MAX_VIDEO_FRAMES:
                raise FlowNodeError("The clip is too long to edit here")
            array = _upright(frame)
            if out is None:
                height, width = array.shape[:2]
                out = av.open(out_buf, mode = "w", format = "mp4")
                vout = _encoder(out, rate)
                vout.width, vout.height = width - width % 2, height - height % 2
                vout.pix_fmt = "yuv420p"
                if audio is not None or (speed == 1.0 and own_sound):
                    aout = out.add_stream("aac", rate = 44100)
                base = t
            else:
                emit_until(t)
            if getattr(frame, "duration", None) and frame.time_base:
                frame_s = float(frame.duration * frame.time_base) or frame_s
            held, held_t = array, t
            kept += 1
        if out is not None:
            last = held_t + frame_s
            emit_until(min(last, end_s) if end_s is not None else last)
        if out is None:
            raise FlowNodeError("Nothing is left of the clip after trimming")
        for packet in vout.encode():
            out.mux(packet)
        duration = emitted / float(rate)
        if aout is not None:
            track = audio if audio is not None else _slice_audio(data, start_s, start_s + duration)
            if track:
                _mux_audio(out, aout, track, duration)
    finally:
        source.close()
    out.close()
    return out_buf.getvalue()


def video_frame(
    data: bytes,
    which: str = "last",
    at_s: float = 0.0,
) -> bytes:
    """One frame of a clip as PNG: the first, the last, or the one showing at `at_s` seconds."""
    import av
    from PIL import Image

    source = av.open(io.BytesIO(data))
    chosen = None
    try:
        vstream = source.streams.video[0]
        target = (
            at_s
            if which == "time"
            else (source.duration or 0) / 1_000_000 - 2
            if which == "last"
            else 0
        )
        if target > 0 and vstream.time_base:
            source.seek(int(target / vstream.time_base), stream = vstream, backward = True)
        decoded = 0
        for frame in source.decode(vstream):
            decoded += 1
            if decoded > 2 * MAX_VIDEO_FRAMES:
                raise FlowNodeError("The video is too long to read a frame from")
            if which == "time" and chosen is not None and float(frame.time or 0.0) > at_s:
                break
            chosen = frame
            if which == "first":
                break
        image = _upright(chosen) if chosen is not None else None
    finally:
        source.close()
    if image is None:
        raise FlowNodeError("The video has no frames")
    return _png(Image.fromarray(image))


def _slice_audio(data: bytes, start_s: float, end_s: float) -> Optional[bytes]:
    import wave

    import av
    import numpy as np

    source = av.open(io.BytesIO(data))
    if not source.streams.audio:
        source.close()
        return None
    astream = source.streams.audio[0]
    if start_s > 0 and astream.time_base:
        source.seek(int(start_s / astream.time_base), stream = astream, backward = True)
    resampler = av.AudioResampler(format = "s16", layout = "stereo", rate = 44100)
    chunks = []
    first: Optional[float] = None
    budget = int(MAX_AUDIO_S * 44100) * 2
    held = 0
    for frame in source.decode(astream):
        t = float(frame.time or 0.0)
        if t >= end_s:
            break
        if t + frame.samples / max(1, frame.sample_rate) <= start_s:
            continue
        if first is None:
            first = t
        for out in resampler.resample(frame):
            chunk = out.to_ndarray().reshape(-1)
            held += chunk.size
            if held > budget:
                source.close()
                raise FlowNodeError("The clip's sound is too long to edit here")
            chunks.append(chunk)
    source.close()
    if not chunks or first is None:
        return None
    pcm = np.concatenate(chunks).astype(np.int16)
    gap = int((first - start_s) * 44100)
    if gap > 0:
        pcm = np.concatenate([np.zeros(gap * 2, dtype = np.int16), pcm])
        first = start_s
    a = max(0, int((start_s - first) * 44100)) * 2
    b = max(a, int((end_s - first) * 44100)) * 2
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(pcm[a:b].tobytes())
    return buf.getvalue()


def _mux_audio(out: Any, aout: Any, track: bytes, duration: float) -> None:
    import av

    source = av.open(io.BytesIO(track))
    if not source.streams.audio:
        source.close()
        return
    resampler = av.AudioResampler(format = "fltp", layout = "stereo", rate = 44100)
    written = 0
    limit = int(duration * 44100)
    for frame in source.decode(source.streams.audio[0]):
        for chunk in resampler.resample(frame):
            if written >= limit:
                break
            chunk.pts = written
            written += chunk.samples
            for packet in aout.encode(chunk):
                out.mux(packet)
        if written >= limit:
            break
    for packet in aout.encode():
        out.mux(packet)
    source.close()


@handler("image.adjust")
async def adjust(ctx: NodeContext) -> dict:
    data, _ = await ctx.media_bytes(ctx.inputs["image"])
    p = ctx.params
    crop = None
    if p.get("crop"):
        crop = (
            _num(p.get("crop_left"), 0.0, 0.0, 1.0),
            _num(p.get("crop_top"), 0.0, 0.0, 1.0),
            _num(p.get("crop_width"), 1.0, 0.0, 1.0),
            _num(p.get("crop_height"), 1.0, 0.0, 1.0),
        )
    png = await asyncio.to_thread(
        adjust_image,
        data,
        crop = crop,
        brightness = _num(p.get("brightness"), 1.0, 0.0, 4.0),
        contrast = _num(p.get("contrast"), 1.0, 0.0, 4.0),
        saturation = _num(p.get("saturation"), 1.0, 0.0, 4.0),
        blur = _num(p.get("blur"), 0.0, 0.0, 100.0),
        invert = bool(p.get("invert")),
        flip = str(p.get("flip") or "none"),
        rotate = _num(p.get("rotate"), 0.0, -360.0, 360.0),
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


@handler("image.composite")
async def composite(ctx: NodeContext) -> dict:
    base, _ = await ctx.media_bytes(ctx.inputs["image"])
    overlay, _ = await ctx.media_bytes(ctx.inputs["overlay"])
    mask = None
    if ctx.inputs.get("mask"):
        mask, _ = await ctx.media_bytes(ctx.inputs["mask"])
    p = ctx.params
    png = await asyncio.to_thread(
        composite_images,
        base,
        overlay,
        x = _num(p.get("x"), 0.5, -1.0, 2.0),
        y = _num(p.get("y"), 0.5, -1.0, 2.0),
        scale = _num(p.get("scale"), 0.5, 0.01, 4.0),
        opacity = _num(p.get("opacity"), 1.0, 0.0, 1.0),
        mask = mask,
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


@handler("image.text")
async def caption(ctx: NodeContext) -> dict:
    data, _ = await ctx.media_bytes(ctx.inputs["image"])
    text = ctx.inputs.get("text")
    if text is None:
        text = ctx.params.get("text") or ""
    p = ctx.params
    position = str(p.get("position") or "bottom")
    png = await asyncio.to_thread(
        text_overlay,
        data,
        str(text),
        size = int(_num(p.get("size"), 48, 8, 400)),
        color = str(p.get("color") or "#ffffff"),
        position = position if position in POSITIONS else "bottom",
        outline = int(_num(p.get("outline"), 2, 0, 20)),
    )
    return {"image": await ctx.save_asset(png, "image/png", kind = "image")}


@handler("video.edit")
async def video_edit(ctx: NodeContext) -> dict:
    data, _ = await ctx.media_bytes(ctx.inputs["video"])
    track = None
    if ctx.inputs.get("audio"):
        track, _ = await ctx.media_bytes(ctx.inputs["audio"])
    p = ctx.params
    end = p.get("end_s")
    speed = _num(p.get("speed"), 1.0, 0.25, 4.0)
    if speed != 1.0 and track is None:
        ctx.log(
            "Changing the speed drops the clip's own sound; connect an audio track to keep sound",
            "warning",
        )
    mp4 = await asyncio.to_thread(
        edit_video,
        data,
        start_s = _num(p.get("start_s"), 0.0, 0.0, 86400.0),
        end_s = None if end in (None, "", 0) else _num(end, 0.0, 0.0, 86400.0),
        speed = speed,
        audio = track,
    )
    return {"video": await ctx.save_asset(mp4, "video/mp4", kind = "video")}

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import math
import wave

import pytest

from core.flows.nodes import media_utils as mu
from core.flows.registry import FlowNodeError


def _img(
    w,
    h,
    color = (200, 40, 40, 255),
):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGBA", (w, h), color).save(out, "PNG")
    return out.getvalue()


def _open(data):
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGBA")


def _clip(
    frames = 24,
    fps = 12,
    w = 64,
    h = 48,
    tone_s = None,
):
    import av
    import numpy as np

    buf = io.BytesIO()
    out = av.open(buf, mode = "w", format = "mp4")
    v = out.add_stream("mpeg4", rate = fps)
    v.width, v.height, v.pix_fmt = w, h, "yuv420p"
    a = out.add_stream("aac", rate = 44100) if tone_s else None
    for i in range(frames):
        arr = np.full((h, w, 3), min(255, i * 10), dtype = np.uint8)
        for packet in v.encode(av.VideoFrame.from_ndarray(arr, format = "rgb24")):
            out.mux(packet)
    for packet in v.encode():
        out.mux(packet)
    if a is not None:
        n = int(tone_s * 44100)
        pcm = (np.sin(np.arange(n) * 2 * math.pi * 440 / 44100) * 0.3).astype(np.float32)
        step = 1024
        for k in range(0, n, step):
            chunk = np.stack([pcm[k : k + step]] * 2)
            frame = av.AudioFrame.from_ndarray(chunk, format = "fltp", layout = "stereo")
            frame.sample_rate = 44100
            frame.pts = k
            for packet in a.encode(frame):
                out.mux(packet)
        for packet in a.encode():
            out.mux(packet)
    out.close()
    return buf.getvalue()


def _probe(data):
    import av

    src = av.open(io.BytesIO(data))
    frames = sum(1 for _ in src.decode(src.streams.video[0]))
    src.close()
    src = av.open(io.BytesIO(data))
    has_audio = bool(src.streams.audio)
    size = (src.streams.video[0].width, src.streams.video[0].height)
    src.close()
    return frames, has_audio, size


def test_adjust_crop_flip_and_rotate_geometry():
    out = _open(mu.adjust_image(_img(200, 100), crop = (0.25, 0.5, 0.5, 0.5)))
    assert out.size == (100, 50)
    assert _open(mu.adjust_image(_img(200, 100), rotate = 90)).size == (100, 200)
    with pytest.raises(FlowNodeError):
        mu.adjust_image(_img(10, 10), crop = (0.9, 0.9, 0.0, 0.0))


def test_adjust_colour_ops_and_alpha_kept():
    dark = _open(mu.adjust_image(_img(4, 4), brightness = 0.0))
    assert dark.getpixel((0, 0))[:3] == (0, 0, 0)
    inv = _open(mu.adjust_image(_img(4, 4, (200, 40, 40, 128)), invert = True))
    assert inv.getpixel((0, 0)) == (55, 215, 215, 128)
    gray = _open(mu.adjust_image(_img(4, 4), saturation = 0.0)).getpixel((0, 0))
    assert gray[0] == gray[1] == gray[2]
    flipped = mu.adjust_image(_img(8, 8), flip = "horizontal")
    assert _open(flipped).size == (8, 8)


def test_composite_places_and_clips_the_overlay():
    base = _img(100, 100, (0, 0, 0, 255))
    top = _img(10, 10, (255, 255, 255, 255))
    out = _open(mu.composite_images(base, top, x = 0.5, y = 0.5, scale = 0.2))
    assert out.size == (100, 100)
    assert out.getpixel((50, 50))[:3] == (255, 255, 255)
    assert out.getpixel((5, 5))[:3] == (0, 0, 0)
    edge = _open(mu.composite_images(base, top, x = 0.0, y = 0.0, scale = 0.2))
    assert edge.getpixel((2, 2))[:3] == (255, 255, 255) and edge.getpixel((50, 50))[:3] == (0, 0, 0)
    half = _open(mu.composite_images(base, top, scale = 0.2, opacity = 0.5)).getpixel((50, 50))
    assert 110 < half[0] < 145


def test_composite_mask_hides_the_overlay():
    base = _img(40, 40, (0, 0, 0, 255))
    top = _img(40, 40, (255, 255, 255, 255))
    black = _img(40, 40, (0, 0, 0, 255))
    out = _open(mu.composite_images(base, top, scale = 1.0, mask = black))
    assert out.getpixel((20, 20))[:3] == (0, 0, 0)


def test_text_overlay_draws_and_keeps_size():
    base = _img(320, 200, (0, 0, 0, 255))
    out = _open(mu.text_overlay(base, "Hello — world", size = 32, position = "top-left"))
    assert out.size == (320, 200)
    corner = out.crop((0, 0, 160, 60)).convert("L").getextrema()
    assert corner[1] > 200
    assert _open(mu.text_overlay(base, "   ")).getpixel((10, 10))[:3] == (0, 0, 0)


def test_video_trim_and_speed_frame_counts():
    clip = _clip(frames = 24, fps = 12)
    frames, _, size = _probe(mu.edit_video(clip, start_s = 0.5, end_s = 1.5))
    assert frames == 12 and size == (64, 48)
    fast, _, _ = _probe(mu.edit_video(clip, speed = 2.0))
    assert fast == 12
    slow, _, _ = _probe(mu.edit_video(clip, speed = 0.5))
    assert slow == 48
    with pytest.raises(FlowNodeError):
        mu.edit_video(clip, start_s = 10.0)


def test_video_keeps_its_sound_or_takes_a_new_track():
    with_tone = _clip(frames = 24, fps = 12, tone_s = 2.0)
    _, has_audio, _ = _probe(mu.edit_video(with_tone, start_s = 0.5, end_s = 1.5))
    assert has_audio
    _, sped_audio, _ = _probe(mu.edit_video(with_tone, speed = 2.0))
    assert not sped_audio
    tone = io.BytesIO()
    with wave.open(tone, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x10" * 22050 * 3)
    _, dubbed, _ = _probe(mu.edit_video(_clip(), audio = tone.getvalue()))
    assert dubbed


def test_composite_keeps_a_huge_overlay_within_the_size_limit():
    out = _open(mu.composite_images(_img(2048, 64), _img(4, 4000), scale = 4.0))
    assert out.size == (2048, 64)


def test_a_late_sound_keeps_its_offset():
    import numpy as np

    sliced = mu._slice_audio(_clip(frames = 24, fps = 12, tone_s = 1.0), -0.5, 1.0)
    with wave.open(io.BytesIO(sliced)) as w:
        assert w.getnframes() >= int(1.4 * 44100)
        head = np.frombuffer(w.readframes(int(0.4 * 44100)), dtype = np.int16)
    assert not head.any()


def test_video_frame_reads_an_edited_clip():
    clip = mu.edit_video(_clip(frames = 12, fps = 12))
    for which in ("first", "last", "time"):
        assert _open(mu.video_frame(clip, which, 0.5)).size == (64, 48)


def test_a_late_trim_seeks_and_keeps_the_right_frames():
    clip = _clip(frames = 96, fps = 12)
    frames, _, _ = _probe(mu.edit_video(clip, start_s = 6.0, end_s = 7.0))
    assert frames == 12


def test_a_trimmed_sound_holds_only_its_own_span(monkeypatch):
    clip = _clip(frames = 48, fps = 12, tone_s = 4.0)
    sliced = mu._slice_audio(clip, 2.0, 3.0)
    with wave.open(io.BytesIO(sliced)) as w:
        assert abs(w.getnframes() - 44100) < 2048
    monkeypatch.setattr(mu, "MAX_AUDIO_S", 0.5)
    with pytest.raises(FlowNodeError):
        mu._slice_audio(clip, 1.0, 3.0)


def test_video_frame_seeks_near_the_end_instead_of_decoding_everything(monkeypatch):
    clip = _clip(frames = 96, fps = 12)
    first = _open(mu.video_frame(clip, "first", 0.0)).convert("RGB")
    assert first.getpixel((5, 5))[0] < 20
    monkeypatch.setattr(mu, "MAX_VIDEO_FRAMES", 20)
    last = _open(mu.video_frame(clip, "last", 0.0)).convert("RGB")
    assert last.getpixel((5, 5))[0] > 200
    assert _open(mu.video_frame(clip, "time", 1.0)).convert("RGB").getpixel((5, 5))[0] > 100


def test_a_speed_change_without_a_track_warns_that_sound_is_dropped():
    import asyncio

    class _Ctx:
        def __init__(self, params):
            self.inputs = {"video": {"kind": "video", "ref": "x"}}
            self.params, self.logs = params, []

        async def media_bytes(self, media):
            return _clip(frames = 12, fps = 12, tone_s = 1.0), "video/mp4"

        def log(
            self,
            message,
            level = "info",
        ):
            self.logs.append((level, message))

        async def save_asset(
            self,
            data,
            mime,
            kind = None,
        ):
            return {"kind": "video", "ref": "y"}

    fast = _Ctx({"speed": 2.0})
    asyncio.run(mu.video_edit(fast))
    assert fast.logs and fast.logs[0][0] == "warning" and "sound" in fast.logs[0][1]
    same = _Ctx({"speed": 1.0})
    asyncio.run(mu.video_edit(same))
    assert same.logs == []


def test_a_phone_clip_is_turned_upright():
    import numpy as np

    class Frame:
        rotation = 90

        def to_ndarray(self, format):
            arr = np.zeros((48, 64, 3), dtype = np.uint8)
            arr[0, 0] = 255
            return arr

    upright = mu._upright(Frame())
    assert upright.shape == (64, 48, 3) and upright[63, 0].all() and upright.flags["C_CONTIGUOUS"]
    Frame.rotation = 0
    assert mu._upright(Frame()).shape == (48, 64, 3)


def test_edit_video_keeps_variable_frame_rate_pacing():
    from fractions import Fraction

    import av
    import numpy as np

    buf = io.BytesIO()
    out = av.open(buf, mode = "w", format = "mp4")
    v = out.add_stream("mpeg4", rate = 10)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    v.codec_context.time_base = Fraction(1, 10)
    for pts, gray in [(0, 0), (1, 40), (2, 80), (10, 200), (11, 240)]:
        frame = av.VideoFrame.from_ndarray(
            np.full((48, 64, 3), gray, dtype = np.uint8), format = "rgb24"
        )
        frame.pts, frame.time_base = pts, Fraction(1, 10)
        for packet in v.encode(frame):
            out.mux(packet)
    for packet in v.encode():
        out.mux(packet)
    out.close()

    src = av.open(io.BytesIO(mu.edit_video(buf.getvalue())))
    shown = [
        (float(f.time), int(f.to_ndarray(format = "rgb24")[10, 10, 0]))
        for f in src.decode(src.streams.video[0])
    ]
    src.close()
    # The bright frame is shot at 1.0 s; it must not appear before ~1.0 s in the edit.
    first_bright = next(t for t, gray in shown if gray > 150)
    assert first_bright >= 0.9, shown
    assert shown[-1][0] >= 1.0, shown


def test_edit_video_constant_rate_keeps_every_frame_once():
    import av
    clip = _clip(frames = 24, fps = 12)
    for speed, expected in ((1.0, 24), (2.0, 12), (0.5, 48)):
        src = av.open(io.BytesIO(mu.edit_video(clip, speed = speed)))
        count = sum(1 for _ in src.decode(src.streams.video[0]))
        src.close()
        assert count == expected, (speed, count)


def test_edit_video_keeps_every_picture_of_a_millisecond_timed_webm():
    import av
    import numpy as np

    buf = io.BytesIO()
    out = av.open(buf, mode = "w", format = "webm")
    v = out.add_stream("libvpx", rate = 60)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    for i in range(12):
        frame = av.VideoFrame.from_ndarray(
            np.full((48, 64, 3), 10 + i * 20, dtype = np.uint8), format = "rgb24"
        )
        frame.pts = i
        for packet in v.encode(frame):
            out.mux(packet)
    for packet in v.encode():
        out.mux(packet)
    out.close()
    for speed, picks in ((1.0, range(12)), (2.0, range(0, 12, 2))):
        src = av.open(io.BytesIO(mu.edit_video(buf.getvalue(), speed = speed)))
        shown = [
            round((int(f.to_ndarray(format = "rgb24")[10, 10, 0]) - 10) / 20)
            for f in src.decode(src.streams.video[0])
        ]
        src.close()
        assert shown == list(picks), (speed, shown)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Wayland frame reader: SPA pods built by hand, frames turned into RGB, the helper's lifecycle.

The live case needs a PipeWire daemon and WirePlumber: set UNSLOTH_PW_TEST_ROOT to an extracted
root (usr/bin/pipewire, usr/lib/...) or have both on PATH; it skips otherwise.
"""

import os
import shutil
import struct
import subprocess
import sys

import pytest

from core.inference.computer_use.backends import _pipewire as pipewire


def _props(raw: bytes) -> dict:
    return pipewire._parse_format(raw)


def test_the_format_offer_names_raw_video_in_every_reader_format_and_no_modifier():
    raw = pipewire._enum_format()
    size, kind = struct.unpack_from("<II", raw)
    assert kind == pipewire._POD_OBJECT and size + 8 == len(raw) and len(raw) % 8 == 0
    assert struct.unpack_from("<II", raw, 8) == (
        pipewire._OBJECT_FORMAT,
        pipewire._PARAM_ENUM_FORMAT,
    )
    props = _props(raw)
    assert props[pipewire._FORMAT_MEDIA_TYPE] == (pipewire._POD_ID, struct.pack("<I", 2))
    assert props[pipewire._FORMAT_MEDIA_SUBTYPE] == (pipewire._POD_ID, struct.pack("<I", 1))
    # A modifier would ask for DMA-BUF, which this reader can't map.
    assert pipewire._FORMAT_VIDEO_MODIFIER not in props
    assert props[pipewire._FORMAT_VIDEO_FORMAT][0] == pipewire._POD_ID


def test_the_offer_defaults_to_the_portals_stream_size():
    # KWin streams at the offer's default size, so a wrong default pads or crops the screen.
    kind, body = _props(pipewire._enum_format(1280, 800))[pipewire._FORMAT_VIDEO_SIZE]
    assert (kind, struct.unpack("<II", body)) == (pipewire._POD_RECTANGLE, (1280, 800))


def test_a_negotiated_format_gives_its_size_and_format():
    fixed = pipewire._object(
        pipewire._OBJECT_FORMAT,
        pipewire._PARAM_FORMAT,
        [
            (pipewire._FORMAT_VIDEO_FORMAT, pipewire._pod(pipewire._POD_ID, struct.pack("<I", 8))),
            (
                pipewire._FORMAT_VIDEO_SIZE,
                pipewire._pod(pipewire._POD_RECTANGLE, struct.pack("<II", 2560, 1440)),
            ),
        ],
    )
    props = _props(fixed)
    assert struct.unpack("<I", props[pipewire._FORMAT_VIDEO_FORMAT][1]) == (8,)
    assert struct.unpack("<II", props[pipewire._FORMAT_VIDEO_SIZE][1]) == (2560, 1440)


@pytest.mark.parametrize("form, pixel", [
    (7, b"\x01\x02\x03\x09"), (8, b"\x03\x02\x01\x09"), (9, b"\x09\x01\x02\x03"),
    (10, b"\x09\x03\x02\x01"), (11, b"\x01\x02\x03\xff"), (12, b"\x03\x02\x01\xff"),
    (15, b"\x01\x02\x03"), (16, b"\x03\x02\x01"),
])  # fmt: skip
def test_every_offered_format_reads_back_as_rgb_with_a_padded_stride(form, pixel, monkeypatch):
    row = pixel * 3 + b"\0" * 4
    grabber = pipewire.Grabber.__new__(pipewire.Grabber)
    grabber._lock, grabber._closed, grabber._helper = pipewire.threading.Lock(), False, None
    monkeypatch.setattr(
        grabber, "_ask", lambda node, timeout: (3, 2, len(row), form, row * 2), raising = False
    )
    image = grabber.grab(1)
    assert image.mode == "RGB" and image.size == (3, 2)
    assert image.tobytes() == bytes((1, 2, 3)) * 6


def test_a_closed_reader_refuses_and_closing_twice_is_harmless():
    grabber = pipewire.Grabber(lambda: (_ for _ in ()).throw(OSError("no portal")), [5])
    with pytest.raises(pipewire.GrabError):
        grabber.grab(5)
    grabber.close()
    grabber.close()
    with pytest.raises(pipewire.GrabError):
        grabber.grab(5)


def _daemon_root():
    root = os.environ.get("UNSLOTH_PW_TEST_ROOT")
    if root:
        return root
    return None if not (shutil.which("pipewire") and shutil.which("wireplumber")) else "/"


@pytest.mark.skipif(_daemon_root() is None, reason = "no PipeWire daemon to test against")
def test_frames_from_a_live_pipewire_source():
    script = os.environ.get("UNSLOTH_PW_TEST_SCRIPT")
    if not script:
        pytest.skip("UNSLOTH_PW_TEST_SCRIPT names the live harness")
    result = subprocess.run(
        [sys.executable, script, "--root", _daemon_root()],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert "RESULT OK" in result.stdout, result.stdout + result.stderr

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode H.264 frames with the AppImage's GStreamer on the target host."""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

WIDTH, HEIGHT, FRAMES = 320, 240, 12
I420_FRAME_BYTES = WIDTH * HEIGHT * 3 // 2
GST_STATE_NULL, GST_STATE_PLAYING = 1, 4
GST_MESSAGE_EOS, GST_MESSAGE_ERROR = 1 << 0, 1 << 11
EXPORT = re.compile(r'^export ([A-Z0-9_]+)="([^"]*)"$')

# Cover the formats used by the media galleries.
REQUIRED_ELEMENTS = (
    "playbin",
    "decodebin",
    "qtdemux",
    "h264parse",
    "openh264enc",
    "avdec_h264",
    "vp8dec",
    "opusdec",
    "wavparse",
)
# Dictation may use either host audio stack.
CAPTURE_ELEMENTS = ("pulsesrc", "alsasrc")


def _extract(appimage: Path, workdir: Path) -> Path:
    subprocess.run(
        [str(appimage), "--appimage-extract"],
        cwd = workdir,
        check = True,
        stdout = subprocess.DEVNULL,
    )
    return workdir / "squashfs-root"


def _hook_environment(appdir: Path) -> dict[str, str]:
    """The GStreamer and GIO variables AppRun exports, read from the hooks."""

    wanted = {"GIO_MODULE_DIR"}
    environment: dict[str, str] = {}
    for hook in sorted((appdir / "apprun-hooks").glob("*.sh")):
        for line in hook.read_text(encoding = "utf-8", errors = "replace").splitlines():
            match = EXPORT.match(line.strip())
            if not match:
                continue
            name, value = match.groups()
            if not (name.startswith("GST_") or name in wanted):
                continue
            environment[name] = value.replace("${APPDIR}", str(appdir)).replace(
                "$APPDIR", str(appdir)
            )
    missing = {"GST_PLUGIN_SYSTEM_PATH_1_0", "GST_PLUGIN_SCANNER_1_0"} - environment.keys()
    if missing:
        raise SystemExit(f"AppRun hooks export no {', '.join(sorted(missing))}")
    return environment


def main() -> None:
    appimage_value = os.environ.get("APPIMAGE_PATH", "")
    if not appimage_value:
        raise SystemExit("APPIMAGE_PATH must name the AppImage under test")
    appimage = Path(appimage_value).resolve()
    if not appimage.is_file():
        raise SystemExit(f"AppImage does not exist: {appimage}")

    workdir = Path(tempfile.mkdtemp(prefix = "unsloth-appimage-media."))
    try:
        appdir = _extract(appimage, workdir)
        os.environ.update(_hook_environment(appdir))
        os.environ.pop("GIO_EXTRA_MODULES", None)

        gst = ctypes.CDLL(str(appdir / "usr/lib/libgstreamer-1.0.so.0"))
        gst.gst_init(None, None)
        gst.gst_version_string.restype = ctypes.c_char_p
        gst.gst_element_factory_find.restype = ctypes.c_void_p
        gst.gst_element_factory_find.argtypes = [ctypes.c_char_p]
        gst.gst_parse_launch.restype = ctypes.c_void_p
        gst.gst_parse_launch.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        gst.gst_element_set_state.restype = ctypes.c_int
        gst.gst_element_set_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
        gst.gst_element_get_bus.restype = ctypes.c_void_p
        gst.gst_element_get_bus.argtypes = [ctypes.c_void_p]
        gst.gst_bus_timed_pop_filtered.restype = ctypes.c_void_p
        gst.gst_bus_timed_pop_filtered.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
        ]

        print(f"bundled core: {gst.gst_version_string().decode()}")
        absent = [
            name for name in REQUIRED_ELEMENTS if not gst.gst_element_factory_find(name.encode())
        ]
        if not any(gst.gst_element_factory_find(name.encode()) for name in CAPTURE_ELEMENTS):
            absent.append(" or ".join(CAPTURE_ELEMENTS))
        if absent:
            raise SystemExit(
                "The bundled GStreamer registry is missing "
                f"{', '.join(absent)}: a bundled plugin did not load on this host"
            )

        decoded = workdir / "decoded.i420"
        pipeline_description = (
            f"videotestsrc num-buffers={FRAMES} ! "
            f"video/x-raw,width={WIDTH},height={HEIGHT},framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! openh264enc ! h264parse ! "
            "avdec_h264 ! videoconvert ! video/x-raw,format=I420 ! "
            f"filesink location={decoded}"
        )
        error = ctypes.c_void_p()
        pipeline = gst.gst_parse_launch(pipeline_description.encode(), ctypes.byref(error))
        if not pipeline:
            raise SystemExit(f"Could not build the media pipeline: {pipeline_description}")
        if gst.gst_element_set_state(pipeline, GST_STATE_PLAYING) == 0:
            raise SystemExit("The bundled media pipeline refused to start")
        bus = gst.gst_element_get_bus(pipeline)
        finished = gst.gst_bus_timed_pop_filtered(bus, 60 * 1_000_000_000, GST_MESSAGE_EOS)
        failed = gst.gst_bus_timed_pop_filtered(bus, 0, GST_MESSAGE_ERROR)
        gst.gst_element_set_state(pipeline, GST_STATE_NULL)
        if failed or not finished:
            raise SystemExit("The bundled media pipeline errored or never finished")

        size = decoded.stat().st_size if decoded.is_file() else 0
        frames, remainder = divmod(size, I420_FRAME_BYTES)
        if remainder or frames < FRAMES:
            raise SystemExit(
                f"avdec_h264 produced {size} bytes, expected {FRAMES} frames of "
                f"{I420_FRAME_BYTES} bytes"
            )
        print(
            f"PASS bundled GStreamer decoded {frames} H.264 frames "
            f"({size} bytes of I420) on this host"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors = True)


if __name__ == "__main__":
    main()

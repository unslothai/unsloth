# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portal ScreenCast frames read from PipeWire via ctypes in a stdlib-only helper process, so a
fault cannot kill the Studio server; the helper copies a stream's newest buffer only when asked,
so an unchanged screen sends no new buffers."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import select
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

_LIBRARY = "libpipewire-0.3.so.0"

# spa/utils/type.h, spa/param/param.h, spa/param/format.h, spa/param/video/raw.h,
# spa/param/buffers.h, spa/buffer/buffer.h, spa/pod/pod.h
_POD_INT, _POD_ID, _POD_RECTANGLE, _POD_FRACTION = 4, 3, 10, 11
_POD_OBJECT, _POD_CHOICE = 15, 19
_CHOICE_NONE, _CHOICE_RANGE, _CHOICE_ENUM, _CHOICE_FLAGS = 0, 1, 3, 4
_OBJECT_FORMAT, _OBJECT_BUFFERS = 0x40003, 0x40004
_PARAM_ENUM_FORMAT, _PARAM_FORMAT, _PARAM_BUFFERS = 3, 4, 5
_FORMAT_MEDIA_TYPE, _FORMAT_MEDIA_SUBTYPE = 1, 2
_FORMAT_VIDEO_FORMAT, _FORMAT_VIDEO_MODIFIER = 0x20001, 0x20002
_FORMAT_VIDEO_SIZE, _FORMAT_VIDEO_FRAMERATE = 0x20003, 0x20004
_MEDIA_VIDEO, _SUBTYPE_RAW = 2, 1
_BUFFERS_BUFFERS, _BUFFERS_BLOCKS, _BUFFERS_DATA_TYPE = 1, 2, 6
_DATA_MEM_PTR, _DATA_MEM_FD = 1, 2
_STREAM_AUTOCONNECT, _STREAM_MAP_BUFFERS = 1 << 0, 1 << 2
_DIRECTION_INPUT = 0
_STATE_ERROR, _STATE_UNCONNECTED = -1, 0
_EVENTS_VERSION = 2

# SPA video format -> (PIL raw mode into RGB, bytes per pixel). Alpha is dropped: screens are opaque.
_FORMATS = {
    8: ("BGRX", 4),  # BGRx, what mutter and KWin offer first
    7: ("RGBX", 4),  # RGBx
    12: ("BGRX", 4),  # BGRA
    11: ("RGBX", 4),  # RGBA
    9: ("XRGB", 4),  # xRGB
    10: ("XBGR", 4),  # xBGR
    15: ("RGB", 3),  # RGB
    16: ("BGR", 3),  # BGR
}


class GrabError(RuntimeError):
    """The message says why in a few words."""


def available() -> Optional[str]:
    try:
        ctypes.CDLL(ctypes.util.find_library("pipewire-0.3") or _LIBRARY)
    except OSError:
        return (
            "The PipeWire library is missing, which screen capture on Wayland needs. Install "
            "pipewire (libpipewire-0.3-0 on Debian and Ubuntu), then restart Unsloth."
        )
    return None


def _pod(kind: int, body: bytes) -> bytes:
    return struct.pack("<II", len(body), kind) + body + b"\0" * (-len(body) % 8)


def _choice(choice: int, kind: int, size: int, values: bytes) -> bytes:
    return _pod(_POD_CHOICE, struct.pack("<IIII", choice, 0, size, kind) + values)


def _object(kind: int, param: int, props: list[tuple[int, bytes]]) -> bytes:
    body = struct.pack("<II", kind, param)
    for key, value in props:
        body += struct.pack("<II", key, 0) + value
    return _pod(_POD_OBJECT, body)


def _enum_format(width: int = 1920, height: int = 1080) -> bytes:
    formats = list(_FORMATS)
    return _object(
        _OBJECT_FORMAT,
        _PARAM_ENUM_FORMAT,
        [
            (_FORMAT_MEDIA_TYPE, _pod(_POD_ID, struct.pack("<I", _MEDIA_VIDEO))),
            (_FORMAT_MEDIA_SUBTYPE, _pod(_POD_ID, struct.pack("<I", _SUBTYPE_RAW))),
            (
                _FORMAT_VIDEO_FORMAT,
                _choice(
                    _CHOICE_ENUM,
                    _POD_ID,
                    4,
                    struct.pack(f"<{len(formats) + 1}I", formats[0], *formats),
                ),
            ),
            (
                _FORMAT_VIDEO_SIZE,
                _choice(
                    _CHOICE_RANGE,
                    _POD_RECTANGLE,
                    8,
                    struct.pack("<6I", width, height, 1, 1, 16384, 16384),
                ),
            ),
            (
                _FORMAT_VIDEO_FRAMERATE,
                _choice(_CHOICE_RANGE, _POD_FRACTION, 8, struct.pack("<6I", 30, 1, 0, 1, 1000, 1)),
            ),
        ],
    )


def _buffers() -> bytes:
    return _object(
        _OBJECT_BUFFERS,
        _PARAM_BUFFERS,
        [
            (_BUFFERS_BUFFERS, _choice(_CHOICE_RANGE, _POD_INT, 4, struct.pack("<3i", 8, 2, 32))),
            (_BUFFERS_BLOCKS, _pod(_POD_INT, struct.pack("<i", 1))),
            (
                _BUFFERS_DATA_TYPE,
                _choice(
                    _CHOICE_FLAGS,
                    _POD_INT,
                    4,
                    struct.pack("<i", (1 << _DATA_MEM_PTR) | (1 << _DATA_MEM_FD)),
                ),
            ),
        ],
    )


def _parse_format(raw: bytes) -> dict[int, tuple[int, bytes]]:
    """Property key -> (pod type, value body) of a Format object; a Choice gives its first value."""
    size, kind = struct.unpack_from("<II", raw)
    if kind != _POD_OBJECT:
        return {}
    found: dict[int, tuple[int, bytes]] = {}
    at, end = 16, 8 + size
    while at + 16 <= end:
        key, _flags, length, kind = struct.unpack_from("<IIII", raw, at)
        body = raw[at + 16 : at + 16 + length]
        if kind == _POD_CHOICE and len(body) >= 16:
            _choice_type, _choice_flags, child_size, kind = struct.unpack_from("<IIII", body)
            body = body[16 : 16 + child_size]
        found[key] = (kind, body)
        at += 16 + length + (-length % 8)
    return found


class _Chunk(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("stride", ctypes.c_int32),
        ("flags", ctypes.c_int32),
    ]


class _Data(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("fd", ctypes.c_int64),
        ("mapoffset", ctypes.c_uint32),
        ("maxsize", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("chunk", ctypes.POINTER(_Chunk)),
    ]


class _SpaBuffer(ctypes.Structure):
    _fields_ = [
        ("n_metas", ctypes.c_uint32),
        ("n_datas", ctypes.c_uint32),
        ("metas", ctypes.c_void_p),
        ("datas", ctypes.POINTER(_Data)),
    ]


class _PwBuffer(ctypes.Structure):
    _fields_ = [("buffer", ctypes.POINTER(_SpaBuffer)), ("user_data", ctypes.c_void_p)]


_STATE_CHANGED = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p
)
_PARAM_CHANGED = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
_PROCESS = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_BUFFER_EVENT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class _Events(ctypes.Structure):
    # src/pipewire/stream.h, struct pw_stream_events (PW_VERSION_STREAM_EVENTS 2).
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("destroy", ctypes.c_void_p),
        ("state_changed", _STATE_CHANGED),
        ("control_info", ctypes.c_void_p),
        ("io_changed", ctypes.c_void_p),
        ("param_changed", _PARAM_CHANGED),
        ("add_buffer", ctypes.c_void_p),
        ("remove_buffer", _BUFFER_EVENT),
        ("process", _PROCESS),
        ("drained", ctypes.c_void_p),
        ("command", ctypes.c_void_p),
        ("trigger_done", ctypes.c_void_p),
    ]


class _Stream:
    def __init__(self) -> None:
        self.handle: Any = None
        self.held: Any = None
        self.format = 0
        self.width = self.height = 0
        self.error = ""
        self.offer: Any = None
        self.ready = threading.Event()


def _bind(pw: Any) -> None:
    void, vp, cint, u32 = None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
    params = ctypes.POINTER(ctypes.c_void_p)
    for name, restype, argtypes in (
        ("pw_init", void, [vp, vp]),
        ("pw_thread_loop_new", vp, [ctypes.c_char_p, vp]),
        ("pw_thread_loop_get_loop", vp, [vp]),
        ("pw_thread_loop_start", cint, [vp]),
        ("pw_thread_loop_lock", void, [vp]),
        ("pw_thread_loop_unlock", void, [vp]),
        ("pw_context_new", vp, [vp, vp, ctypes.c_size_t]),
        ("pw_context_connect_fd", vp, [vp, cint, vp, ctypes.c_size_t]),
        ("pw_properties_new_string", vp, [ctypes.c_char_p]),
        ("pw_stream_new", vp, [vp, ctypes.c_char_p, vp]),
        ("pw_stream_add_listener", void, [vp, vp, ctypes.POINTER(_Events), vp]),
        ("pw_stream_connect", cint, [vp, cint, u32, cint, params, u32]),
        ("pw_stream_update_params", cint, [vp, params, u32]),
        ("pw_stream_dequeue_buffer", ctypes.POINTER(_PwBuffer), [vp]),
        ("pw_stream_queue_buffer", cint, [vp, ctypes.POINTER(_PwBuffer)]),
    ):
        function = getattr(pw, name)
        function.restype = restype
        function.argtypes = argtypes


def _serve(fd: int, nodes: list[int], sizes: dict[int, tuple[int, int]]) -> None:
    pw = ctypes.CDLL(ctypes.util.find_library("pipewire-0.3") or _LIBRARY)
    _bind(pw)
    pw.pw_init(None, None)
    loop = pw.pw_thread_loop_new(b"unsloth-capture", None)
    context = pw.pw_context_new(pw.pw_thread_loop_get_loop(loop), None, 0)
    if not loop or not context or pw.pw_thread_loop_start(loop) < 0:
        raise SystemExit("PipeWire could not start")
    streams = {node: _Stream() for node in nodes}
    by_index = list(streams.values())

    def stream_of(data: Any) -> _Stream:
        return by_index[(data or 0) - 1]

    def state_changed(data, _old, state, error):
        stream = stream_of(data)
        if state in (_STATE_ERROR, _STATE_UNCONNECTED):
            stream.error = (error or b"").decode("utf-8", "replace") or "the stream ended"
            stream.ready.set()

    def param_changed(data, param_id, param):
        stream = stream_of(data)
        if param_id != _PARAM_FORMAT or not param:
            return
        size = struct.unpack("<I", ctypes.string_at(param, 4))[0]
        props = _parse_format(ctypes.string_at(param, 8 + size))
        kind, body = props.get(_FORMAT_VIDEO_FORMAT, (0, b""))
        stream.format = (
            struct.unpack_from("<I", body)[0] if kind == _POD_ID and len(body) >= 4 else 0
        )
        kind, body = props.get(_FORMAT_VIDEO_SIZE, (0, b""))
        if kind == _POD_RECTANGLE and len(body) >= 8:
            stream.width, stream.height = struct.unpack_from("<II", body)
        if _FORMAT_VIDEO_MODIFIER in props or stream.format not in _FORMATS:
            stream.error = "the compositor offered no frame format this reader takes"
            stream.ready.set()
            return
        pod = ctypes.create_string_buffer(_buffers())
        pw.pw_stream_update_params(stream.handle, (ctypes.c_void_p * 1)(ctypes.addressof(pod)), 1)

    def remove_buffer(data, buffer):
        stream = stream_of(data)
        if stream.held and ctypes.cast(stream.held, ctypes.c_void_p).value == buffer:
            stream.held = None

    def process(data):
        stream = stream_of(data)
        newest = None
        while True:
            buffer = pw.pw_stream_dequeue_buffer(stream.handle)
            if not buffer:
                break
            if newest:
                pw.pw_stream_queue_buffer(stream.handle, newest)
            newest = buffer
        if not newest:
            return
        spa = newest.contents.buffer.contents
        chunk = spa.datas[0].chunk.contents if spa.n_datas and spa.datas[0].chunk else None
        if chunk is None or not spa.datas[0].data or chunk.size == 0:
            pw.pw_stream_queue_buffer(stream.handle, newest)
            return
        if stream.held:
            pw.pw_stream_queue_buffer(stream.handle, stream.held)
        stream.held = newest
        stream.ready.set()

    events = _Events(
        version = _EVENTS_VERSION,
        state_changed = _STATE_CHANGED(state_changed),
        param_changed = _PARAM_CHANGED(param_changed),
        remove_buffer = _BUFFER_EVENT(remove_buffer),
        process = _PROCESS(process),
    )
    # spa_hook is 6 pointers (spa/utils/hook.h); a zeroed, larger block is safe.
    hooks = [(ctypes.c_void_p * 16)() for _ in nodes]
    pw.pw_thread_loop_lock(loop)
    try:
        core = pw.pw_context_connect_fd(context, fd, None, 0)
        if not core:
            raise SystemExit("PipeWire refused the screen sharing connection")
        for index, (node, stream) in enumerate(streams.items(), start = 1):
            properties = pw.pw_properties_new_string(
                b"media.type=Video media.category=Capture media.role=Screen"
            )
            stream.handle = pw.pw_stream_new(core, b"Unsloth computer use", properties)
            if not stream.handle:
                stream.error = "the stream could not be created"
                continue
            pw.pw_stream_add_listener(stream.handle, hooks[index - 1], ctypes.byref(events), index)
            # KWin follows the offer's default size, so default to the portal's stream size.
            stream.offer = ctypes.create_string_buffer(_enum_format(*sizes.get(node, (1920, 1080))))
            result = pw.pw_stream_connect(
                stream.handle,
                _DIRECTION_INPUT,
                node,
                _STREAM_AUTOCONNECT | _STREAM_MAP_BUFFERS,
                (ctypes.c_void_p * 1)(ctypes.addressof(stream.offer)),
                1,
            )
            if result < 0:
                stream.error = f"the stream could not connect ({result})"
    finally:
        pw.pw_thread_loop_unlock(loop)

    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        words = line.split()
        if len(words) != 3 or words[0] != b"grab":
            continue
        stream = streams.get(int(words[1]))
        if stream is None:
            out.write(b"err that screen is not being shared\n")
            out.flush()
            continue
        stream.ready.wait(float(words[2]))
        frame = None
        pw.pw_thread_loop_lock(loop)
        try:
            if stream.held:
                spa = stream.held.contents.buffer.contents
                data, chunk = spa.datas[0], spa.datas[0].chunk.contents
                offset = chunk.offset % max(data.maxsize, 1)
                length = min(chunk.size, data.maxsize - offset)
                bpp = _FORMATS[stream.format][1]
                stride = chunk.stride or stream.width * bpp
                frame = (stride, ctypes.string_at(data.data + offset, length))
        finally:
            pw.pw_thread_loop_unlock(loop)
        if frame is None:
            reason = stream.error or "no frame arrived from the compositor in time"
            out.write(b"err " + reason.encode("utf-8", "replace").replace(b"\n", b" ") + b"\n")
        else:
            stride, pixels = frame
            out.write(
                f"ok {stream.width} {stream.height} {stride} {stream.format} {len(pixels)}\n".encode()
            )
            out.write(pixels)
        out.flush()
    os._exit(0)


class Grabber:
    def __init__(
        self,
        open_fd: Callable[[], int],
        nodes: list[int],
        sizes: Optional[dict[int, tuple[int, int]]] = None,
    ) -> None:
        self._open_fd = open_fd
        self._nodes = [int(node) for node in nodes]
        self._sizes = {int(node): (int(w), int(h)) for node, (w, h) in (sizes or {}).items()}
        self._lock = threading.Lock()
        self._helper: Optional[subprocess.Popen] = None
        self._closed = False
        try:
            self._start()
        except GrabError:
            pass  # grab() reports it, after one more try

    def _start(self) -> None:
        self._stop()
        try:
            fd = self._open_fd()
        except Exception as exc:  # noqa: BLE001 - the portal's own error is shown on grab
            raise GrabError(f"the screen sharing connection could not be opened ({exc})") from exc
        try:
            self._helper = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    os.path.abspath(__file__),
                    "--fd",
                    str(fd),
                    "--nodes",
                    ",".join(map(str, self._nodes)),
                    "--sizes",
                    ",".join(f"{node}:{w}x{h}" for node, (w, h) in self._sizes.items()),
                ],
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                pass_fds = (fd,),
                start_new_session = True,
            )
        except OSError as exc:
            raise GrabError("the screen capture helper could not start") from exc
        finally:
            os.close(fd)

    def _stop(self) -> None:
        helper, self._helper = self._helper, None
        if helper is None:
            return
        for stream in (helper.stdin, helper.stdout):
            try:
                stream.close()
            except OSError:
                pass
        try:
            helper.kill()
            helper.wait(timeout = 2)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _read(self, size: int, deadline: float) -> bytes:
        helper = self._helper
        fd = helper.stdout.fileno()
        chunks, left = [], size
        while left:
            if not select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
                raise GrabError("the screen capture helper stopped answering")
            chunk = os.read(fd, min(left, 1 << 20))
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)

    def _line(self, deadline: float) -> bytes:
        fd = self._helper.stdout.fileno()
        line = b""
        while not line.endswith(b"\n"):
            if not select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
                raise GrabError("the screen capture helper stopped answering")
            byte = os.read(fd, 1)
            if not byte:
                raise EOFError
            line += byte
        return line

    def _ask(self, node: int, timeout: float) -> tuple[int, int, int, int, bytes]:
        if self._helper is None or self._helper.poll() is not None:
            raise EOFError
        try:
            self._helper.stdin.write(f"grab {node} {timeout}\n".encode())
            self._helper.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise EOFError from exc
        deadline = time.monotonic() + timeout + 5.0
        words = self._line(deadline).decode("utf-8", "replace").split(" ", 1)
        if words[0] != "ok":
            raise GrabError(words[1].strip() if len(words) > 1 else "the screen was not captured")
        width, height, stride, form, size = (int(value) for value in words[1].split())
        return width, height, stride, form, self._read(size, deadline)

    def grab(
        self,
        node: int,
        timeout: float = 3.0,
    ) -> Any:
        from PIL import Image

        with self._lock:
            if self._closed:
                raise GrabError("screen capture has been closed")
            try:
                width, height, stride, form, pixels = self._ask(int(node), timeout)
            except EOFError:
                self._start()
                try:
                    width, height, stride, form, pixels = self._ask(int(node), timeout)
                except EOFError:
                    raise GrabError("the screen capture helper stopped") from None
        mode, bpp = _FORMATS.get(form, (None, 0))
        if (
            mode is None
            or width <= 0
            or height <= 0
            or len(pixels) < stride * (height - 1) + width * bpp
        ):
            raise GrabError("the compositor sent a frame this reader can't use")
        # frombytes, not frombuffer: frombuffer maps RGBX in place and hands back an RGBX image.
        return Image.frombytes("RGB", (width, height), pixels, "raw", mode, stride, 1)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop()

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


if __name__ == "__main__":
    arguments = sys.argv[1:]
    _serve(
        int(arguments[arguments.index("--fd") + 1]),
        [int(node) for node in arguments[arguments.index("--nodes") + 1].split(",") if node],
        {
            int(node): tuple(int(part) for part in size.split("x"))
            for node, size in (
                item.split(":")
                for item in arguments[arguments.index("--sizes") + 1].split(",")
                if item
            )
        }
        if "--sizes" in arguments
        else {},
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

bl_info = {
    "name": "Unsloth Studio Live Preview", "author": "Unsloth", "version": (0, 1, 0),
    "blender": (4, 3, 0), "location": "3D View > Sidebar > Unsloth", "category": "3D View",
}

import json
import math
import re
import secrets
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from queue import Empty, Full, Queue

import bpy
import gpu
from bpy.props import StringProperty
from mathutils import Quaternion, Vector

PERIOD = 1 / 15
bridge = None
target = None
handle = None
offscreen = None
drawing = False
last_capture = 0.0


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def png(width, height, rgba):
    rgb = bytearray(width * height * 3)
    for channel in range(3):
        rgb[channel::3] = rgba[channel::4]
    stride = width * 3
    lines = b"".join(b"\0" + rgb[i:i + stride] for i in range(len(rgb) - stride, -1, -stride))

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(lines, 1)) + chunk(b"IEND", b""))


def latest(queue, value):
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(value)
    except Full:
        pass


class Bridge:
    def __init__(self, pairing):
        endpoint, token = pairing.split()
        if not re.fullmatch(r"http://(?:127\.0\.0\.1|\[::1\]):\d+/api/editor-preview/adapter/blender/[\w-]+", endpoint):
            raise ValueError("Paste the Blender pairing code from Unsloth Studio.")
        self.endpoint = endpoint
        self.headers = {"Authorization": "Bearer " + token, "X-Unsloth-Preview-Instance": secrets.token_hex(16)}
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())
        self.frames = Queue(maxsize=1)
        self.gestures = Queue(maxsize=8)
        self.stop = threading.Event()
        self.active = False
        self.status = "Connecting"
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def request(self, path, data=None):
        headers = self.headers if data is None else {**self.headers, "Content-Type": "image/png"}
        req = urllib.request.Request(self.endpoint + path, data=data, headers=headers)
        with self.http.open(req, timeout=.7) as response:
            return response.read()

    def loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                gesture = json.loads(self.request("/input"))
                self.active = gesture.pop("active")
                self.status = "Live" if self.active else "Waiting for Studio"
                if self.active:
                    if any(gesture.values()):
                        try:
                            self.gestures.put_nowait(gesture)
                        except Full:
                            pass
                    try:
                        frame = self.frames.get_nowait()
                    except Empty:
                        pass
                    else:
                        self.request("/frame", png(*frame))
            except urllib.error.HTTPError as error:
                self.active = False
                if error.code in (401, 409):
                    self.status = "Start a new preview in Studio"
                    return
                self.status = "Waiting for Studio"
                self.stop.wait(.5)
            except (OSError, ValueError):
                self.active = False
                self.status = "Waiting for Studio"
                self.stop.wait(.5)
            self.stop.wait(max(0, PERIOD - (time.monotonic() - started)))


def free_buffer():
    global offscreen
    if offscreen is not None:
        try:
            offscreen.free()
        except (ReferenceError, RuntimeError):
            pass
        offscreen = None


def capture():
    global drawing, offscreen, last_capture
    context = bpy.context
    if drawing or not bridge or not bridge.active or context.area is None or context.area.as_pointer() != target:
        return
    now = time.monotonic()
    if now - last_capture < PERIOD:
        return
    region, view = context.region, context.region_data
    if not view or region.width < 1 or region.height < 1:
        return
    scale = min(1, 1280 / region.width, 720 / region.height)
    width, height = max(1, round(region.width * scale)), max(1, round(region.height * scale))
    drawing = True
    last_capture = now
    try:
        if offscreen is None or (width, height) != (offscreen.width, offscreen.height):
            free_buffer()
            offscreen = gpu.types.GPUOffScreen(width, height)
        offscreen.draw_view3d(context.scene, context.view_layer, context.space_data, region,
                              view.view_matrix, view.window_matrix, do_color_management=True)
        with offscreen.bind():
            pixels = gpu.state.active_framebuffer_get().read_color(0, 0, width, height, 4, 0, "UBYTE")
            pixels.dimensions = width * height * 4
            latest(bridge.frames, (width, height, bytes(pixels)))
    except (RuntimeError, ReferenceError):
        free_buffer()
    finally:
        drawing = False


def apply_gesture(view, gesture):
    rotation = view.view_rotation.copy()
    right = rotation @ Vector((1, 0, 0))
    up = rotation @ Vector((0, 1, 0))
    yaw = Quaternion((0, 0, 1), -gesture["orbit_x"] * math.pi)
    pitch = Quaternion(right, -gesture["orbit_y"] * math.pi)
    view.view_rotation = (yaw @ pitch @ rotation).normalized()
    distance = view.view_distance
    view.view_location += (up * gesture["pan_y"] - right * gesture["pan_x"]) * distance * 2
    view.view_distance = max(.01, min(10000, distance * math.exp(gesture["zoom"] * 1.7)))


def tick():
    if bridge is None:
        return None
    areas = (area for window in bpy.context.window_manager.windows for area in window.screen.areas)
    area = next((area for area in areas if area.type == "VIEW_3D" and area.as_pointer() == target), None)
    if area and bridge.active:
        view = area.spaces.active.region_3d
        while True:
            try:
                apply_gesture(view, bridge.gestures.get_nowait())
            except Empty:
                break
        area.tag_redraw()
    return PERIOD


def disconnect():
    global bridge, handle, target
    if bridge:
        bridge.stop.set()
        bridge.thread.join(timeout=1.6)
        bridge = None
    if handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        handle = None
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    free_buffer()
    target = None


def connect(pairing, area):
    global bridge, handle, target
    disconnect()
    target = area.as_pointer()
    bridge = Bridge(pairing)
    handle = bpy.types.SpaceView3D.draw_handler_add(capture, (), "WINDOW", "POST_PIXEL")
    bpy.app.timers.register(tick, persistent=True)


class UNSLOTH_OT_pair(bpy.types.Operator):
    bl_idname = "unsloth.preview_pair"
    bl_label = "Connect preview"

    def execute(self, context):
        try:
            connect(context.window_manager.unsloth_preview_pairing, context.area)
        except ValueError:
            self.report({"ERROR"}, "Paste the Blender pairing code from Unsloth Studio.")
            return {"CANCELLED"}
        context.window_manager.unsloth_preview_pairing = ""
        return {"FINISHED"}


class UNSLOTH_OT_disconnect(bpy.types.Operator):
    bl_idname = "unsloth.preview_disconnect"
    bl_label = "Disconnect"

    def execute(self, _context):
        disconnect()
        return {"FINISHED"}


class UNSLOTH_PT_preview(bpy.types.Panel):
    bl_label = "Unsloth Studio Preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Unsloth"

    def draw(self, context):
        if bridge:
            self.layout.label(text=bridge.status)
            self.layout.operator("unsloth.preview_disconnect")
        else:
            self.layout.prop(context.window_manager, "unsloth_preview_pairing", text="Pairing code")
            self.layout.operator("unsloth.preview_pair")


classes = (UNSLOTH_OT_pair, UNSLOTH_OT_disconnect, UNSLOTH_PT_preview)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.unsloth_preview_pairing = StringProperty(subtype="PASSWORD", options={"SKIP_SAVE"})


def unregister():
    disconnect()
    del bpy.types.WindowManager.unsloth_preview_pairing
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

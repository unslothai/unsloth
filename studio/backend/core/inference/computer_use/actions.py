# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The computer tool's arguments, in whatever dialect a model learned, as validated actions."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from .keys import MAX_REPEAT, chord_text, parse_keys, typed_text_problem

MAX_ACTIONS = 8
MAX_TEXT_CHARS = 4000
MAX_WAIT_S = 10.0
MAX_SCROLL = 30
MAX_APP_NAME_CHARS = 80
# Past any screenshot or 0-1000 grid: a mistake, not a far corner.
_MAX_COORDINATE = 20_000

ACTIONS = (
    "screenshot",
    "zoom",
    "click",
    "double_click",
    "right_click",
    "move",
    "drag",
    "scroll",
    "type",
    "key",
    "wait",
    "open_app",
)

_OP_ALIASES = {
    "left_click": "click",
    "single_click": "click",
    "tap": "click",
    "mouse_click": "click",
    "click_mouse": "click",
    "left_single": "click",
    "left_double": "double_click",
    "doubleclick": "double_click",
    "double_tap": "double_click",
    "right_single": "right_click",
    "rightclick": "right_click",
    "context_click": "right_click",
    "middle_click": "middle_click",
    "triple_click": "triple_click",
    "mouse_move": "move",
    "move_mouse": "move",
    "move_to": "move",
    "hover": "move",
    "left_click_drag": "drag",
    "drag_and_drop": "drag",
    "drag_mouse": "drag",
    "swipe": "drag",
    "keypress": "key",
    "key_press": "key",
    "press_key": "key",
    "press_keys": "key",
    "press": "key",
    "hotkey": "key",
    "shortcut": "key",
    "type_text": "type",
    "input_text": "type",
    "write": "type",
    "paste_text": "type",
    "take_screenshot": "screenshot",
    "capture": "screenshot",
    "see": "screenshot",
    "sleep": "wait",
    "launch_app": "open_app",
    "launch": "open_app",
    "open_application": "open_app",
    "application": "open_app",
    "app": "open_app",
    "cursor_position": "cursor",
}

INPUT_OPS = frozenset({"click", "move", "drag", "scroll", "type", "key"})
POINT_OPS = frozenset({"click", "move", "drag", "scroll", "zoom"})

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BATCH_KEYS = ("then", "actions", "steps")


def _number(raw: Any, what: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{what} must be a number.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a number.")
    return value


def _coordinate(raw: Any, what: str) -> float:
    value = _number(raw, what)
    if not 0 <= value <= _MAX_COORDINATE:
        raise ValueError(f"{what} is off the screen.")
    return value


def _numbers_in(value: Any) -> Optional[list[float]]:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return [_number(value["x"], "x"), _number(value["y"], "y")]
        return None
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple)) for item in value):
            value = [n for item in value for n in item]
        try:
            return [_number(item, "a coordinate") for item in value]
        except ValueError:
            return None
    if isinstance(value, str):
        found = _NUMBER_RE.findall(value)
        return [float(n) for n in found] if found else None
    return None


def _as_point(value: Any) -> Optional[tuple[float, float]]:
    numbers = _numbers_in(value)
    if not numbers:
        return None
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    if len(numbers) == 4:
        return (numbers[0] + numbers[2]) / 2, (numbers[1] + numbers[3]) / 2
    return None


_POINT_KEYS = ("coordinate", "coordinates", "position", "point", "location", "loc", "start_box")
_END_POINT_KEYS = ("end_coordinate", "end", "to", "target", "end_box", "end_point", "destination")
_START_POINT_KEYS = ("start_coordinate", "start", "from", "start_point", "startCoordinate")


def _xy(raw: dict, prefix: str = "") -> Optional[tuple]:
    x, y = raw.get(f"{prefix}x"), raw.get(f"{prefix}y")
    if x is None or y is None or x == "" or y == "":
        return None
    return _coordinate(x, f"{prefix}x"), _coordinate(y, f"{prefix}y")


def _named(raw: dict, keys: tuple) -> Optional[tuple]:
    for key in keys:
        if raw.get(key) is not None:
            point = _as_point(raw[key])
            if point is None:
                raise ValueError(f"{key} must be a point such as [640, 360].")
            return _coordinate(point[0], "x"), _coordinate(point[1], "y")
    return None


def _point(raw: dict) -> Optional[tuple]:
    return _xy(raw) or _named(raw, _POINT_KEYS)


def _need_point(raw: dict, op: str) -> tuple[float, float]:
    point = _point(raw)
    if point is None:
        raise ValueError(f"{op} needs x and y, taken from the latest screenshot.")
    return point


def _modifiers(raw: dict) -> tuple[str, ...]:
    held = raw.get("modifiers") or raw.get("hold_keys") or raw.get("holdKeys")
    if held in (None, "", []):
        return ()
    if isinstance(held, (list, tuple)):
        held = "+".join(str(item) for item in held)
    mods, key = parse_keys(str(held))[0]
    if key is not None:
        raise ValueError("modifiers takes only ctrl, alt, shift or cmd.")
    return mods


def _whole(raw: Any, what: str, low: int, high: int, default: int) -> int:
    if raw is None or raw == "":
        return default
    value = _number(raw, what)
    if not value.is_integer():
        raise ValueError(f"{what} must be a whole number.")
    return max(low, min(int(value), high))


def _scroll(raw: dict) -> dict:
    action: dict = {"op": "scroll"}
    point = _point(raw)
    if point is not None:
        action["x"], action["y"] = point
    direction = str(raw.get("direction") or raw.get("scroll_direction") or "").strip().lower()
    amount = raw.get("amount")
    if amount is None:
        amount = raw.get("scroll_amount", raw.get("scrollCount", raw.get("clicks")))
    dx = dy = 0
    if direction:
        if direction not in ("up", "down", "left", "right"):
            raise ValueError('direction must be "up", "down", "left" or "right".')
        notches = _whole(amount, "amount", 1, MAX_SCROLL, 3)
        dx = {"left": -notches, "right": notches}.get(direction, 0)
        dy = {"up": -notches, "down": notches}.get(direction, 0)
    else:
        # OpenAI's scroll_x / scroll_y and Qwen's pixels are pixel distances; a notch is about 100.
        for key, axis in (("scroll_x", "x"), ("scroll_y", "y"), ("pixels", "qwen")):
            if raw.get(key) in (None, ""):
                continue
            pixels = _number(raw[key], key)
            if pixels == 0:
                continue
            notches = int(math.copysign(max(1, min(MAX_SCROLL, round(abs(pixels) / 100))), pixels))
            if axis == "x":
                dx = notches
            elif axis == "y":
                dy = notches
            else:
                dy = -notches
        if dx == 0 and dy == 0:
            raise ValueError('scroll needs a direction: "up", "down", "left" or "right".')
    action["dx"], action["dy"] = dx, dy
    return action


def _drag(raw: dict) -> dict:
    path = raw.get("path")
    if isinstance(path, (list, tuple)) and len(path) >= 2:
        points = [_as_point(item) for item in path]
        if any(point is None for point in points):
            raise ValueError("path must be a list of points such as [[10, 20], [200, 20]].")
        start, end = points[0], points[-1]
    else:
        start = _xy(raw) or _named(raw, _START_POINT_KEYS)
        end = _xy(raw, "to_") or _named(raw, _END_POINT_KEYS)
        other = _named(raw, _POINT_KEYS)
        if start is None:
            start = other
        elif end is None:
            end = other
    if start is None or end is None:
        raise ValueError("drag needs x and y to start from, and to_x and to_y to end at.")
    return {
        "op": "drag",
        "x": _coordinate(start[0], "x"),
        "y": _coordinate(start[1], "y"),
        "to_x": _coordinate(end[0], "to_x"),
        "to_y": _coordinate(end[1], "to_y"),
    }


def _zoom(raw: dict) -> dict:
    numbers = _numbers_in(raw.get("region") if raw.get("region") is not None else raw.get("box"))
    if not numbers or len(numbers) != 4:
        raise ValueError("zoom needs region: [left, top, right, bottom] in the latest screenshot.")
    x0, y0, x1, y1 = (_coordinate(n, "region") for n in numbers)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("region is empty: right and bottom must be past left and top.")
    return {"op": "zoom", "region": (x0, y0, x1, y1)}


def _text(raw: dict) -> dict:
    text = raw.get("text")
    if text is None:
        text = raw.get("content")
    if not isinstance(text, str) or text == "":
        raise ValueError("type needs text.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters.")
    # One kind of line end, so a CRLF is one Enter, here and in every backend.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    submit = raw.get("submit") is True or raw.get("press_enter") is True
    # UI-TARS ends the text with a newline to submit it.
    if text.endswith("\n") and "\n" not in text[:-1]:
        text, submit = text[:-1], True
    problem = typed_text_problem(text)
    if problem:
        raise ValueError(problem)
    if text == "" and not submit:
        raise ValueError("type needs text.")
    return {"op": "type", "text": text, "submit": submit}


def _key(raw: dict) -> dict:
    value = None
    for name in ("keys", "key", "combo", "combination", "text"):
        if raw.get(name) not in (None, "", []):
            value = raw[name]
            break
    chords = parse_keys(value)
    return {
        "op": "key",
        "chords": chords,
        "repeat": _whole(raw.get("repeat"), "repeat", 1, MAX_REPEAT, 1),
    }


def _one(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("each action is an object with an action name.")
    name = raw.get("action")
    if name is None:
        name = raw.get("type") if isinstance(raw.get("type"), str) else raw.get("op")
    op = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    op = _OP_ALIASES.get(op, op)
    if op == "screenshot":
        action: dict = {"op": "screenshot"}
        display = raw.get("display")
        if display not in (None, ""):
            action["display"] = _whole(display, "display", 1, 16, 1)
        return action
    if op == "cursor":
        return {"op": "cursor"}
    if op == "zoom":
        return _zoom(raw)
    if op in ("click", "double_click", "right_click", "middle_click", "triple_click"):
        x, y = _need_point(raw, op)
        button = str(raw.get("button") or "left").strip().lower()
        count = _whole(raw.get("count", raw.get("clicks", raw.get("clickCount"))), "count", 1, 3, 1)
        if op == "double_click":
            count = 2
        elif op == "triple_click":
            count = 3
        elif op == "right_click":
            button = "right"
        elif op == "middle_click":
            button = "middle"
        if button == "wheel":
            button = "middle"
        if button not in ("left", "right", "middle"):
            raise ValueError('button must be "left", "right" or "middle".')
        return {
            "op": "click",
            "x": x,
            "y": y,
            "button": button,
            "count": count,
            "mods": _modifiers(raw),
        }
    if op == "move":
        x, y = _need_point(raw, op)
        return {"op": "move", "x": x, "y": y}
    if op == "drag":
        return _drag(raw)
    if op == "scroll":
        return _scroll(raw)
    if op == "type":
        return _text(raw)
    if op == "key":
        return _key(raw)
    if op == "wait":
        seconds = raw.get("seconds")
        if seconds is None:
            seconds = raw.get("duration", raw.get("time", 1))
        return {"op": "wait", "seconds": max(0.0, min(_number(seconds, "seconds"), MAX_WAIT_S))}
    if op == "open_app":
        app = raw.get("name") or raw.get("app") or raw.get("application") or raw.get("text")
        app = " ".join(str(app or "").split())
        if not app:
            raise ValueError("open_app needs name: the application to open or bring to the front.")
        if len(app) > MAX_APP_NAME_CHARS:
            raise ValueError("name is too long for an application name.")
        return {"op": "open_app", "name": app}
    if not op:
        raise ValueError(f"action is missing. Use one of: {', '.join(ACTIONS)}.")
    raise ValueError(f"'{name}' is not an action. Use one of: {', '.join(ACTIONS)}.")


def parse_actions(arguments: Any) -> list[dict]:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object with an action.")
    batch = None
    for key in _BATCH_KEYS:
        if isinstance(arguments.get(key), (list, tuple)):
            batch = list(arguments[key])
            break
    has_first = any(arguments.get(key) not in (None, "") for key in ("action", "op")) or (
        isinstance(arguments.get("type"), str) and bool(arguments["type"].strip())
    )
    raws = ([arguments] if has_first or batch is None else []) + (batch or [])
    if not raws:
        raise ValueError(f"action is missing. Use one of: {', '.join(ACTIONS)}.")
    if len(raws) > MAX_ACTIONS:
        raise ValueError(f"at most {MAX_ACTIONS} actions in one call.")
    return [_one(raw) for raw in raws]


def describe(action: dict) -> str:
    """One action in a few words. Typed text is never included in full."""
    op = action["op"]
    if op == "click":
        kind = {"right": "Right-click", "middle": "Middle-click"}.get(action["button"], "Click")
        if action["count"] == 2:
            kind = "Double-click"
        elif action["count"] == 3:
            kind = "Triple-click"
        held = "+".join(action["mods"])
        return f"{kind} at ({action['x']:g}, {action['y']:g})" + (
            f" holding {held}" if held else ""
        )
    if op == "move":
        return f"Move the pointer to ({action['x']:g}, {action['y']:g})"
    if op == "drag":
        return (
            f"Drag from ({action['x']:g}, {action['y']:g}) to "
            f"({action['to_x']:g}, {action['to_y']:g})"
        )
    if op == "scroll":
        parts = []
        if action["dy"]:
            parts.append(f"{'down' if action['dy'] > 0 else 'up'} {abs(action['dy'])}")
        if action["dx"]:
            parts.append(f"{'right' if action['dx'] > 0 else 'left'} {abs(action['dx'])}")
        return "Scroll " + " and ".join(parts)
    if op == "type":
        text = action["text"]
        preview = text if len(text) <= 40 else text[:37] + "..."
        return f'Type "{preview}"' + (" and press Enter" if action["submit"] else "")
    if op == "key":
        keys = " ".join(chord_text(chord) for chord in action["chords"])
        return f"Press {keys}" + (f" x{action['repeat']}" if action["repeat"] > 1 else "")
    if op == "wait":
        return f"Wait {action['seconds']:g}s"
    if op == "open_app":
        return f"Open {action['name']}"
    if op == "zoom":
        return "Zoom in on the screen"
    if op == "cursor":
        return "Read the pointer position"
    return "Take a screenshot"

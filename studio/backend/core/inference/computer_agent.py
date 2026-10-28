# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Visual Computer Use mode for local AI agents (#15532).
Allows vision-capable models (such as Holo4) to interact with desktop applications
through screenshots, mouse movements, clicking, drag-and-drop, scrolling, and keyboard input
within a screenshot -> action -> screenshot visual feedback loop.

Related: #14989 (PC Control / GUI), #10800 (Browser Use), #15447 (Tab Handoff), #15448 (Full-Page Screenshots).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_COMPUTER_NOTE = (
    "Computer actions affect the local desktop: only perform actions requested or confirmed by the user. "
    "A global abort shortcut or stop button is always available to immediately halt control."
)

COMPUTER_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_screenshot",
        "description": (
            "Capture a screenshot of the desktop screen so the model can visually perceive elements, "
            "windows, and coordinates. Part of the visual screenshot -> action -> screenshot feedback loop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "display": {
                    "type": "integer",
                    "description": "Optional display index (0 for primary display).",
                },
                "region": {
                    "type": "object",
                    "description": "Optional sub-region to capture: {x, y, width, height}.",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                    },
                },
            },
        },
    },
}

COMPUTER_CLICK_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_click",
        "description": (
            "Click the mouse at screen coordinates (x, y) from the latest screenshot. "
            "Supports left, right, and middle clicks, as well as double/triple clicks and key modifiers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Horizontal pixel coordinate."},
                "y": {"type": "number", "description": "Vertical pixel coordinate."},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button to click (default is left).",
                },
                "click_count": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "1 for single click, 2 for double click, 3 for triple click (default 1).",
                },
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["ctrl", "alt", "shift", "cmd", "meta"]},
                    "description": "Optional modifier keys to hold during the click.",
                },
            },
            "required": ["x", "y"],
        },
    },
}

COMPUTER_MOVE_MOUSE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_move_mouse",
        "description": (
            "Move the mouse cursor to screen coordinates (x, y) without clicking. "
            "Useful for hover effects, focusing menus, or positioning before an interaction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Horizontal pixel coordinate."},
                "y": {"type": "number", "description": "Vertical pixel coordinate."},
            },
            "required": ["x", "y"],
        },
    },
}

COMPUTER_DRAG_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_drag",
        "description": (
            "Drag from start coordinates (x, y) to destination coordinates (to_x, to_y) and release. "
            "Useful for moving windows, dragging sliders, selecting text, or moving files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Start x coordinate."},
                "y": {"type": "number", "description": "Start y coordinate."},
                "to_x": {"type": "number", "description": "Destination x coordinate."},
                "to_y": {"type": "number", "description": "Destination y coordinate."},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button for dragging (default is left).",
                },
            },
            "required": ["x", "y", "to_x", "to_y"],
        },
    },
}

COMPUTER_MOUSE_DOWN_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_mouse_down",
        "description": "Press and hold down a mouse button at optional coordinates (x, y).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                },
            },
        },
    },
}

COMPUTER_MOUSE_UP_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_mouse_up",
        "description": "Release a previously held mouse button at optional coordinates (x, y).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                },
            },
        },
    },
}

COMPUTER_TYPE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_type",
        "description": (
            "Type a text string into the currently focused desktop application or field. "
            "Optionally press Enter afterwards."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to type."},
                "press_enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter/Return after typing.",
                },
                "clear_first": {
                    "type": "boolean",
                    "description": "Whether to select-all and delete existing field content before typing.",
                },
            },
            "required": ["text"],
        },
    },
}

_KEYS = (
    "Enter",
    "Return",
    "Escape",
    "Tab",
    "Backspace",
    "Delete",
    "Space",
    "ArrowDown",
    "ArrowUp",
    "ArrowLeft",
    "ArrowRight",
    "PageDown",
    "PageUp",
    "Home",
    "End",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
)

COMPUTER_PRESS_KEY_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_press_key",
        "description": (
            "Press a keyboard key or key shortcut combination (e.g., 'Enter', 'Escape', 'Tab', "
            "'Ctrl+c', 'Ctrl+v', 'Cmd+s', 'Alt+Tab')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key or key combination (e.g. 'Enter', 'Escape', 'Ctrl+C', 'Alt+Tab').",
                },
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["ctrl", "alt", "shift", "cmd", "meta"]},
                    "description": "Optional list of modifiers held with key.",
                },
            },
            "required": ["key"],
        },
    },
}

COMPUTER_SCROLL_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_scroll",
        "description": "Scroll the active desktop window or area at (x, y) vertically or horizontally.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction (default is 'down').",
                },
                "amount": {
                    "type": "number",
                    "description": "Number of scroll clicks/steps (default is 3).",
                },
                "x": {"type": "number", "description": "Optional horizontal pixel coordinate to hover before scrolling."},
                "y": {"type": "number", "description": "Optional vertical pixel coordinate to hover before scrolling."},
            },
        },
    },
}

COMPUTER_CURSOR_POSITION_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_cursor_position",
        "description": "Get current mouse cursor position and screen dimensions.",
        "parameters": {"type": "object", "properties": {}},
    },
}

COMPUTER_INTERVAL_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_interval_screenshot",
        "description": (
            "Configure periodic desktop screenshot recording at fixed time intervals (saved to a folder). "
            "Allows the model to follow activities over time like a video trace (#15532)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether periodic screenshot capture is enabled.",
                },
                "interval_seconds": {
                    "type": "number",
                    "description": "Interval between captures in seconds (minimum 1.0, default 5.0).",
                },
                "folder_path": {
                    "type": "string",
                    "description": "Optional custom folder path to save periodic screenshots.",
                },
            },
            "required": ["enabled"],
        },
    },
}

COMPUTER_TOOLS = [
    COMPUTER_SCREENSHOT_TOOL,
    COMPUTER_CLICK_TOOL,
    COMPUTER_MOVE_MOUSE_TOOL,
    COMPUTER_DRAG_TOOL,
    COMPUTER_MOUSE_DOWN_TOOL,
    COMPUTER_MOUSE_UP_TOOL,
    COMPUTER_TYPE_TOOL,
    COMPUTER_PRESS_KEY_TOOL,
    COMPUTER_SCROLL_TOOL,
    COMPUTER_CURSOR_POSITION_TOOL,
    COMPUTER_INTERVAL_SCREENSHOT_TOOL,
]

COMPUTER_TOOL_NAMES = frozenset(tool["function"]["name"] for tool in COMPUTER_TOOLS)
COMPUTER_SCREENSHOT = "computer_screenshot"

_ACTION_TIMEOUT_S = 600.0
_MIN_ACTION_TIMEOUT_S = 30.0
_MAX_POINT = 100_000
_MAX_TEXT_CHARS = 10_000


def is_computer_tool(name: Any) -> bool:
    return isinstance(name, str) and name in COMPUTER_TOOL_NAMES


def _coordinate(arguments: dict, key: str) -> Optional[float]:
    raw = arguments.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} must be a number of pixels.")
    if not (0 <= value <= _MAX_POINT):
        raise ValueError(f"{key} must be between 0 and {_MAX_POINT} pixels.")
    return round(value, 1)


def _validated_computer_request(name: str, arguments: dict) -> dict:
    op = name.removeprefix("computer_")
    if name == COMPUTER_SCREENSHOT:
        display = arguments.get("display")
        region = arguments.get("region")
        req: dict[str, Any] = {"op": "screenshot"}
        if display is not None:
            try:
                req["display"] = int(display)
            except (TypeError, ValueError):
                pass
        if isinstance(region, dict):
            req["region"] = {
                "x": _coordinate(region, "x") or 0.0,
                "y": _coordinate(region, "y") or 0.0,
                "width": max(1.0, float(region.get("width", 100))),
                "height": max(1.0, float(region.get("height", 100))),
            }
        return req

    if name == "computer_click":
        x = _coordinate(arguments, "x")
        y = _coordinate(arguments, "y")
        if x is None or y is None:
            raise ValueError("x and y coordinates are required for clicking.")
        button = str(arguments.get("button") or "left").lower()
        if button not in ("left", "right", "middle"):
            button = "left"
        click_count = arguments.get("click_count") or 1
        try:
            click_count = max(1, min(int(click_count), 3))
        except (TypeError, ValueError):
            click_count = 1
        modifiers = arguments.get("modifiers") or []
        if not isinstance(modifiers, list):
            modifiers = []
        return {
            "op": "click",
            "x": x,
            "y": y,
            "button": button,
            "click_count": click_count,
            "modifiers": [str(m).lower() for m in modifiers if str(m).lower() in ("ctrl", "alt", "shift", "cmd", "meta")],
        }

    if name == "computer_move_mouse":
        x = _coordinate(arguments, "x")
        y = _coordinate(arguments, "y")
        if x is None or y is None:
            raise ValueError("x and y coordinates are required for mouse movement.")
        return {"op": "move", "x": x, "y": y}

    if name == "computer_drag":
        x = _coordinate(arguments, "x")
        y = _coordinate(arguments, "y")
        to_x = _coordinate(arguments, "to_x")
        to_y = _coordinate(arguments, "to_y")
        if any(v is None for v in (x, y, to_x, to_y)):
            raise ValueError("x, y, to_x, and to_y coordinates are required for drag.")
        button = str(arguments.get("button") or "left").lower()
        if button not in ("left", "right", "middle"):
            button = "left"
        return {"op": "drag", "x": x, "y": y, "to_x": to_x, "to_y": to_y, "button": button}

    if name in ("computer_mouse_down", "computer_mouse_up"):
        sub_op = "mouse_down" if name == "computer_mouse_down" else "mouse_up"
        button = str(arguments.get("button") or "left").lower()
        if button not in ("left", "right", "middle"):
            button = "left"
        req = {"op": sub_op, "button": button}
        x = _coordinate(arguments, "x")
        y = _coordinate(arguments, "y")
        if x is not None and y is not None:
            req["x"] = x
            req["y"] = y
        return req

    if name == "computer_type":
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string.")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"text is longer than {_MAX_TEXT_CHARS} characters.")
        return {
            "op": "type",
            "text": text,
            "press_enter": arguments.get("press_enter") is True,
            "clear_first": arguments.get("clear_first") is True,
        }

    if name == "computer_press_key":
        key = str(arguments.get("key") or "").strip()
        if not key:
            raise ValueError("key is required.")
        modifiers = arguments.get("modifiers") or []
        if not isinstance(modifiers, list):
            modifiers = []
        return {
            "op": "press_key",
            "key": key,
            "modifiers": [str(m).lower() for m in modifiers if str(m).lower() in ("ctrl", "alt", "shift", "cmd", "meta")],
        }

    if name == "computer_scroll":
        direction = str(arguments.get("direction") or "down").lower()
        if direction not in ("up", "down", "left", "right"):
            direction = "down"
        try:
            amount = float(arguments.get("amount") or 3.0)
        except (TypeError, ValueError):
            amount = 3.0
        req = {"op": "scroll", "direction": direction, "amount": max(1.0, min(amount, 100.0))}
        x = _coordinate(arguments, "x")
        y = _coordinate(arguments, "y")
        if x is not None and y is not None:
            req["x"] = x
            req["y"] = y
        return req

    if name == "computer_cursor_position":
        return {"op": "cursor_position"}

    if name == "computer_interval_screenshot":
        enabled = arguments.get("enabled") is True
        try:
            interval = float(arguments.get("interval_seconds") or 5.0)
        except (TypeError, ValueError):
            interval = 5.0
        return {
            "op": "interval_screenshot",
            "enabled": enabled,
            "interval_seconds": max(1.0, min(interval, 3600.0)),
            "folder_path": str(arguments.get("folder_path") or "").strip()[:500],
        }

    raise ValueError(f"unknown computer tool {name}.")


def _with_desktop_screenshot(body: str, image: Any, width: Optional[int] = None, height: Optional[int] = None) -> str:
    from .mcp_images import SENTINEL, _png_data_url

    if not isinstance(image, str) or not (image.startswith("data:image/") or len(image) > 50):
        return "Error: could not capture desktop screenshot."
    data = image.split(",", 1)[1] if "," in image else image
    url = _png_data_url(data)
    if url is None:
        # Fallback if PIL is unavailable: verify basic base64 PNG header
        try:
            import base64
            raw = base64.b64decode(data, validate = False)
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                url = f"data:image/png;base64,{data}"
        except Exception:
            pass
    if url is None:
        return "Error: the screenshot was not a readable PNG image."
    size_desc = f" ({width}x{height} pixels)" if width and height else ""
    desc = f"{body}\nDesktop screenshot captured{size_desc}. Inspect the screenshot to locate visual targets and their coordinates."
    return (
        desc
        + "\n"
        + SENTINEL
        + json.dumps([{"data": url.split(",", 1)[1], "mimeType": "image/png"}])
    )


def format_computer_result(name: str, result: dict) -> str:
    if result.get("ok") is not True:
        err = result.get("error") or "the computer action could not be completed."
        return f"Error: {err}"
    if name == COMPUTER_SCREENSHOT:
        w = result.get("width")
        h = result.get("height")
        return _with_desktop_screenshot("Done.", result.get("image"), w, h)
    msg = result.get("message")
    if msg:
        return f"Done: {msg}"
    if name == "computer_cursor_position":
        x = result.get("x", 0)
        y = result.get("y", 0)
        screen_w = result.get("screen_width")
        screen_h = result.get("screen_height")
        dim = f", screen resolution: {screen_w}x{screen_h}" if screen_w and screen_h else ""
        return f"Cursor position: ({x}, {y}){dim}"
    if name == "computer_interval_screenshot":
        status = "enabled" if result.get("enabled") else "disabled"
        folder = result.get("folder") or "default sessions folder"
        return f"Periodic desktop screenshots {status} (saving to: {folder})."
    return "Done."


def execute_computer_tool(
    name: str,
    arguments: dict,
    *,
    cancel_event = None,
    output_callback: Optional[Callable[[Any], None]] = None,
    session_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    from state.computer_actions import (
        begin_computer_action,
        discard_computer_action,
        new_action_id,
        wait_computer_action,
    )
    try:
        from utils.account_context import current_account_id
        acct_id = current_account_id()
    except Exception:
        acct_id = ""

    if output_callback is None:
        return "Error: Computer Use is only available in an interactive Unsloth desktop session."
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        request = _validated_computer_request(name, arguments)
    except ValueError as exc:
        return f"Error: {exc}"

    action_id = new_action_id()
    slot = begin_computer_action(action_id, session_id, acct_id)
    wait = (
        _ACTION_TIMEOUT_S
        if timeout is None
        else max(_MIN_ACTION_TIMEOUT_S, min(float(timeout), _ACTION_TIMEOUT_S))
    )
    try:
        output_callback(
            {
                "computer_action": {
                    "id": action_id,
                    "session_id": session_id or "",
                    **request,
                    "timeout_s": wait,
                }
            }
        )
    except BaseException:
        discard_computer_action(action_id, slot)
        raise

    outcome, result = wait_computer_action(slot, action_id, cancel_event, wait)
    if outcome == "cancelled":
        return "Error: the computer action was aborted by the user."
    if outcome == "expired" or result is None:
        return "Error: the computer action timed out waiting for the desktop application or user approval."
    return format_computer_result(name, result)

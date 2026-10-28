# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for Visual Computer Use mode (#15532).
Tests tool schema, argument validation, desktop action dispatch, emergency abort, and result formatting.
"""

import threading

try:
    import pytest
except ImportError:
    class _PytestMock:
        class raises:
            def __init__(self, expected_exception, match = None):
                self.expected = expected_exception
                self.match = match

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.expected} but nothing was raised")
                if not issubclass(exc_type, self.expected):
                    return False
                if self.match and self.match not in str(exc_val):
                    raise AssertionError(f"Expected match '{self.match}' in '{exc_val}'")
                return True

    pytest = _PytestMock()

from core.inference import computer_agent
from core.inference.mcp_images import SENTINEL
from state import computer_actions


def _request(name: str, arguments: dict) -> dict:
    return computer_agent._validated_computer_request(name, arguments)


def test_tool_catalog_and_membership():
    assert len(computer_agent.COMPUTER_TOOLS) >= 10
    assert "computer_screenshot" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_click" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_move_mouse" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_drag" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_type" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_press_key" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_scroll" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_cursor_position" in computer_agent.COMPUTER_TOOL_NAMES
    assert "computer_interval_screenshot" in computer_agent.COMPUTER_TOOL_NAMES
    assert computer_agent.is_computer_tool("computer_click")
    assert not computer_agent.is_computer_tool("unknown_tool")


def test_screenshot_validation():
    req = _request("computer_screenshot", {})
    assert req == {"op": "screenshot"}

    req_disp = _request("computer_screenshot", {"display": 1})
    assert req_disp == {"op": "screenshot", "display": 1}

    req_reg = _request(
        "computer_screenshot",
        {"region": {"x": 100, "y": 200, "width": 800, "height": 600}},
    )
    assert req_reg["op"] == "screenshot"
    assert req_reg["region"] == {"x": 100.0, "y": 200.0, "width": 800.0, "height": 600.0}


def test_click_validation():
    req = _request("computer_click", {"x": 500, "y": 300})
    assert req == {
        "op": "click",
        "x": 500.0,
        "y": 300.0,
        "button": "left",
        "click_count": 1,
        "modifiers": [],
    }

    req_double = _request(
        "computer_click",
        {"x": 150, "y": 250, "button": "right", "click_count": 2, "modifiers": ["ctrl", "shift"]},
    )
    assert req_double == {
        "op": "click",
        "x": 150.0,
        "y": 250.0,
        "button": "right",
        "click_count": 2,
        "modifiers": ["ctrl", "shift"],
    }

    with pytest.raises(ValueError, match = "x and y coordinates"):
        _request("computer_click", {"x": 100})


def test_move_and_drag_validation():
    req_move = _request("computer_move_mouse", {"x": 1024, "y": 768})
    assert req_move == {"op": "move", "x": 1024.0, "y": 768.0}

    req_drag = _request("computer_drag", {"x": 100, "y": 100, "to_x": 400, "to_y": 500})
    assert req_drag == {
        "op": "drag",
        "x": 100.0,
        "y": 100.0,
        "to_x": 400.0,
        "to_y": 500.0,
        "button": "left",
    }

    with pytest.raises(ValueError, match = "required for drag"):
        _request("computer_drag", {"x": 100, "y": 100})


def test_mouse_down_and_up_validation():
    req_down = _request("computer_mouse_down", {"button": "left", "x": 200, "y": 300})
    assert req_down == {"op": "mouse_down", "button": "left", "x": 200.0, "y": 300.0}

    req_up = _request("computer_mouse_up", {"button": "left"})
    assert req_up == {"op": "mouse_up", "button": "left"}


def test_type_and_press_key_validation():
    req_type = _request("computer_type", {"text": "Hello, Unsloth!", "press_enter": True})
    assert req_type == {
        "op": "type",
        "text": "Hello, Unsloth!",
        "press_enter": True,
        "clear_first": False,
    }

    with pytest.raises(ValueError, match = "text must be a string"):
        _request("computer_type", {"text": 12345})

    req_key = _request("computer_press_key", {"key": "Enter"})
    assert req_key == {"op": "press_key", "key": "Enter", "modifiers": []}

    req_shortcut = _request("computer_press_key", {"key": "c", "modifiers": ["ctrl"]})
    assert req_shortcut == {"op": "press_key", "key": "c", "modifiers": ["ctrl"]}

    with pytest.raises(ValueError, match = "key is required"):
        _request("computer_press_key", {"key": ""})


def test_scroll_validation():
    req = _request("computer_scroll", {"direction": "up", "amount": 5, "x": 300, "y": 400})
    assert req == {"op": "scroll", "direction": "up", "amount": 5.0, "x": 300.0, "y": 400.0}


def test_interval_screenshot_validation():
    req = _request("computer_interval_screenshot", {"enabled": True, "interval_seconds": 3.0})
    assert req == {
        "op": "interval_screenshot",
        "enabled": True,
        "interval_seconds": 3.0,
        "folder_path": "",
    }


def test_format_computer_result_text_and_errors():
    assert computer_agent.format_computer_result("computer_click", {"ok": True}) == "Done."
    assert (
        computer_agent.format_computer_result("computer_click", {"ok": False, "error": "Permission denied"})
        == "Error: Permission denied"
    )

    pos_res = computer_agent.format_computer_result(
        "computer_cursor_position",
        {"ok": True, "x": 500, "y": 300, "screen_width": 1920, "screen_height": 1080},
    )
    assert "Cursor position: (500, 300)" in pos_res
    assert "1920x1080" in pos_res

    interval_res = computer_agent.format_computer_result(
        "computer_interval_screenshot",
        {"ok": True, "enabled": True, "folder": "/tmp/screenshots"},
    )
    assert "Periodic desktop screenshots enabled" in interval_res
    assert "/tmp/screenshots" in interval_res


def test_format_computer_result_screenshot():
    # 1x1 PNG base64
    sample_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    res = computer_agent.format_computer_result(
        "computer_screenshot",
        {"ok": True, "image": sample_png, "width": 1920, "height": 1080},
    )
    assert "Desktop screenshot captured (1920x1080 pixels)" in res
    assert SENTINEL in res
    assert "image/png" in res


def test_state_slot_lifecycle_and_abort():
    aid = computer_actions.new_action_id()
    slot = computer_actions.begin_computer_action(aid, "sess_1", "acct_1")
    assert aid in computer_actions._pending

    # Mismatched account cannot resolve
    assert not computer_actions.resolve_computer_action(aid, {"ok": True}, "sess_1", "other_acct")
    # Matching account resolves
    assert computer_actions.resolve_computer_action(aid, {"ok": True}, "sess_1", "acct_1")

    outcome, result = computer_actions.wait_computer_action(slot, aid, timeout = 1.0)
    assert outcome == "answered"
    assert result == {"ok": True}
    assert aid not in computer_actions._pending


def test_emergency_abort_cancels_all_pending_actions():
    aid1 = computer_actions.new_action_id()
    slot1 = computer_actions.begin_computer_action(aid1, "sess_emergency", "acct_1")

    aid2 = computer_actions.new_action_id()
    slot2 = computer_actions.begin_computer_action(aid2, "sess_emergency", "acct_1")

    aborted = computer_actions.abort_all_computer_actions("sess_emergency")
    assert aborted == 2

    outcome1, res1 = computer_actions.wait_computer_action(slot1, aid1, timeout = 0.5)
    outcome2, res2 = computer_actions.wait_computer_action(slot2, aid2, timeout = 0.5)
    assert outcome1 == "answered"
    assert res1.get("ok") is False
    assert "Aborted by user emergency stop" in res1.get("error", "")


def test_execute_computer_tool_roundtrip(monkeypatch):
    events = []

    def mock_output(event):
        events.append(event)
        act = event.get("computer_action")
        if act:
            # Simulate client answering immediately
            computer_actions.resolve_computer_action(
                act["id"], {"ok": True, "message": "Clicked target"}, act["session_id"], ""
            )

    res = computer_agent.execute_computer_tool(
        "computer_click",
        {"x": 100, "y": 200},
        output_callback = mock_output,
        session_id = "test_sess",
    )
    assert res == "Done: Clicked target"
    assert len(events) == 1
    assert events[0]["computer_action"]["op"] == "click"
    assert events[0]["computer_action"]["x"] == 100.0
    assert events[0]["computer_action"]["y"] == 200.0

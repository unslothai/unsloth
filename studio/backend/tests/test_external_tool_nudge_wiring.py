# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guards for external Studio tool-call nudging (#8907 follow-up)."""

import inspect
from pathlib import Path

from core.inference.studio_tool_loop import ToolLoopPolicy, stream_with_studio_tools


def test_external_tool_loop_accepts_and_gates_nudge_flag():
    assert "nudge_tool_calls" in ToolLoopPolicy.__dataclass_fields__
    src = inspect.getsource(stream_with_studio_tools)
    assert "nudge_enabled(policy.nudge_tool_calls)" in src


def test_external_route_forwards_request_nudge_flag():
    from routes import inference as routes_inference

    external_src = inspect.getsource(routes_inference._proxy_to_external_provider)
    assert external_src.count("nudge_tool_calls = payload.nudge_tool_calls") == 2
    codex_policy = external_src.split("CodexToolPolicy(", 1)[1].split("if studio_tool_payloads", 1)[
        0
    ]
    assert "nudge_tool_calls = payload.nudge_tool_calls" in codex_policy


def test_frontend_forwards_nudge_setting_to_external_tools():
    studio = Path(__file__).resolve().parents[2]
    adapter = (studio / "frontend/src/features/chat/api/chat-adapter.ts").read_text(
        encoding = "utf-8"
    )
    # One local-model request and one external local-tool request.
    assert adapter.count("nudge_tool_calls: runtime.nudgeToolCalls") >= 2

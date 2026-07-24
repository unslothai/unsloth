# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    canonical_tool_call_key,
    coerce_tool_arguments,
    status_for_tool,
    strip_result_for_model,
    tool_event_provenance,
)


def test_append_deferred_nudges_merges_deduped_into_one_message():
    conversation = [{"role": "assistant", "tool_calls": [1]}, {"role": "tool", "content": "r"}]
    nudges = [
        {"role": "user", "content": "duplicate"},
        {"role": "user", "content": "duplicate"},  # dropped: same content
        {"role": "user", "content": "disabled foo"},
    ]
    append_deferred_nudges(conversation, nudges)
    # One user message, after the results, with distinct contents joined.
    assert conversation[2:] == [{"role": "user", "content": "duplicate\n\ndisabled foo"}]
    # Empty is a no-op.
    before = list(conversation)
    append_deferred_nudges(conversation, [])
    assert conversation == before


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _call(
    name: str,
    args,
    call_id: str = "call_0",
) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args) if isinstance(args, dict) else args,
        },
    }


def test_canonical_tool_call_key_sorts_arguments():
    a = canonical_tool_call_key("web_search", {"query": "gpu", "limit": 5})
    b = canonical_tool_call_key("web_search", {"limit": 5, "query": "gpu"})
    c = canonical_tool_call_key("python", {"limit": 5, "query": "gpu"})

    assert a == b
    assert a != c
    assert a == 'web_search:{"limit":5,"query":"gpu"}'


def test_coerce_tool_arguments_parses_json_and_heals_raw_strings():
    parsed = coerce_tool_arguments('{"query":"gpu prices"}', heal = True)
    healed = coerce_tool_arguments("print(1)", heal = True, tool_name = "python")
    raw = coerce_tool_arguments("not-json", heal = False, tool_name = "python")

    assert parsed.arguments == {"query": "gpu prices"}
    assert not parsed.healed
    assert healed.arguments == {"code": "print(1)"}
    assert healed.healed
    assert raw.arguments == {"raw": "not-json"}
    assert not raw.healed


def test_status_and_provenance_match_local_event_conventions():
    assert status_for_tool("web_search", {"query": "gpus"}) == "Searching: gpus"
    assert (
        status_for_tool("web_search", {"url": "https://www.example.com/a"})
        == "Reading: example.com"
    )
    assert status_for_tool("python", {"code": "print(1)\nprint(2)"}) == "Running Python: print(1)"
    assert tool_event_provenance(healed = True, forced = False, provisional = None) == {
        "source": "local",
        "healed": True,
    }


def test_prepare_execute_builds_visible_events_and_model_tool_message():
    controller = ToolLoopController(tools = [_tool("web_search")])
    decision = controller.prepare_call(_call("web_search", {"query": "gpu prices"}))

    assert decision.should_execute
    assert decision.emit_visible_events
    assert decision.status_text == "Searching: gpu prices"
    assert decision.tool_start_payload()["arguments"] == {"query": "gpu prices"}
    assert decision.tool_start_event()["type"] == "tool_start"
    assert decision.as_assistant_tool_call()["function"]["arguments"] == '{"query":"gpu prices"}'

    completion = controller.record_result(decision, "Search result\n__IMAGES__:{...}")

    assert completion.tool_end_payload()["result"] == "Search result\n__IMAGES__:{...}"
    assert completion.tool_end_event()["type"] == "tool_end"
    assert completion.tool_message() == {
        "role": "tool",
        "name": "web_search",
        "content": "Search result",
        "tool_call_id": "call_0",
    }


def test_successful_duplicate_is_internal_noop_and_keeps_remaining_tools():
    controller = ToolLoopController(tools = [_tool("web_search"), _tool("python")])
    first = controller.prepare_call(_call("web_search", {"query": "gpu prices"}, "call_a"))
    controller.record_result(first, "ok")

    duplicate = controller.prepare_call(_call("web_search", {"query": "gpu prices"}, "call_b"))
    completion = controller.record_noop(duplicate)

    assert duplicate.action == "duplicate"
    assert not duplicate.should_execute
    assert not duplicate.emit_visible_events
    duplicate_nudge = completion.model_message()["content"]
    assert duplicate_nudge.startswith(
        "One earlier request to call tool 'web_search' in this batch was not executed"
    )
    assert "previous tool request" not in duplicate_nudge.lower()
    assert "already completed successfully" in duplicate_nudge
    assert "different enabled tool" in duplicate_nudge
    assert completion.model_message()["role"] == "user"
    assert not controller.force_final_answer
    assert [tool["function"]["name"] for tool in controller.active_tools()] == [
        "web_search",
        "python",
    ]


def test_repeated_successful_duplicate_becomes_terminal_after_one_recovery_nudge():
    controller = ToolLoopController(tools = [_tool("web_search"), _tool("python")])
    first = controller.prepare_call(_call("web_search", {"query": "gpu prices"}, "call_a"))
    controller.record_result(first, "ok")

    duplicate_one = controller.prepare_call(_call("web_search", {"query": "gpu prices"}, "call_b"))
    completion_one = controller.record_noop(duplicate_one)

    assert duplicate_one.action == "duplicate"
    assert "already completed successfully" in completion_one.model_message()["content"]
    assert not controller.force_final_answer
    assert [tool["function"]["name"] for tool in controller.active_tools()] == [
        "web_search",
        "python",
    ]

    duplicate_two = controller.prepare_call(_call("web_search", {"query": "gpu prices"}, "call_c"))
    completion_two = controller.record_noop(duplicate_two)

    assert duplicate_two.action == "duplicate"
    assert "already completed successfully" in completion_two.model_message()["content"]
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_failed_call_does_not_block_retry():
    controller = ToolLoopController(tools = [_tool("web_search")])
    first = controller.prepare_call(_call("web_search", {"query": "gpu prices"}))
    controller.record_result(first, "Error: temporary failure")

    retry = controller.prepare_call(_call("web_search", {"query": "gpu prices"}))

    assert retry.should_execute
    assert retry.action == "execute"


def test_empty_enabled_tool_list_blocks_all_tool_calls():
    controller = ToolLoopController(tools = [])
    decision = controller.prepare_call(_call("web_search", {"query": "gpu prices"}))
    completion = controller.record_noop(decision)

    assert decision.action == "disabled"
    assert not decision.emit_visible_events
    assert completion.model_message()["role"] == "user"
    disabled_nudge = completion.model_message()["content"]
    assert disabled_nudge.startswith(
        "One earlier request to call tool 'web_search' in this batch was not executed"
    )
    assert "previous tool request" not in disabled_nudge.lower()
    assert "not enabled" in disabled_nudge
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_disabled_tool_is_internal_noop_not_visible_tool_error():
    controller = ToolLoopController(tools = [_tool("web_search")])
    decision = controller.prepare_call(_call("python", {"code": "print(1)"}))
    completion = controller.record_noop(decision)

    assert decision.action == "disabled"
    assert not decision.emit_visible_events
    assert completion.model_message()["role"] == "user"
    assert "not enabled" in completion.model_message()["content"]
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_render_html_success_filters_active_tools_and_repeat_is_internal():
    controller = ToolLoopController(tools = [_tool("render_html"), _tool("web_search")])
    assert [t["function"]["name"] for t in controller.active_tools()] == [
        "render_html",
        "web_search",
    ]

    first = controller.prepare_call(_call("render_html", {"code": "<html></html>"}, "call_html_1"))
    controller.record_result(first, "Rendered HTML canvas: Demo")

    assert [t["function"]["name"] for t in controller.active_tools()] == ["web_search"]

    repeat = controller.prepare_call(_call("render_html", {"code": "<html></html>"}, "call_html_2"))
    completion = controller.record_noop(repeat)

    assert repeat.action == "render_html_repeat"
    assert not repeat.emit_visible_events
    assert completion.model_message()["role"] == "user"
    assert "Do not call render_html again" in completion.model_message()["content"]
    assert controller.force_final_answer
    assert controller.active_tools() == []


def test_strip_result_for_model_removes_frontend_image_sentinel():
    assert strip_result_for_model('text\n__IMAGES__:{"paths":[]}') == "text"
    assert strip_result_for_model("text __IMAGES__:payload") == "text"
    assert strip_result_for_model("plain text") == "plain text"

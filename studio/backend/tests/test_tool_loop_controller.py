# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    canonical_tool_call_key,
    coerce_arguments_by_schema,
    coerce_tool_arguments,
    status_for_tool,
    strip_result_for_model,
    tool_event_provenance,
)
from core.inference.tool_call_parser import parse_tool_calls_from_text
from core.inference.tools import ALL_TOOLS, _mcp_specs_for_server


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


@pytest.mark.parametrize(
    "url, expected",
    [
        # bare hosts are fetched, so the badge must name them
        ("google.com", "Reading: google.com"),
        ("www.google.com/x", "Reading: google.com"),
        ("//google.com", "Reading: google.com"),
        ("example.com:8443/path", "Reading: example.com"),
        ("github.com/unslothai/unsloth", "Reading: github.com"),
        # still generic for what the fetch layer refuses
        ("/login", "Reading page..."),
        ("javascript:alert(1)", "Reading page..."),
        # urlparse raises on these, outside the fetch's handler: degrade, not raise
        ("https://[::1", "Reading page..."),
        ("https://::1]", "Reading page..."),
        ("//exam／ple.com", "Reading page..."),
        ("//example.com＠", "Reading page..."),
    ],
)
def test_status_names_the_host_for_schemeless_urls(url, expected):
    assert status_for_tool("web_search", {"url": url}) == expected


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


def test_forced_mismatch_keeps_the_required_tool_active():
    controller = ToolLoopController(tools = [_tool("web_search"), _tool("python")])
    decision = controller.prepare_call(
        _call("python", {"code": "print(1)"}),
        allowed_tool_names = {"web_search"},
    )
    completion = controller.record_noop(decision)

    assert decision.action == "forced_mismatch"
    assert "required tool choice" in completion.model_message()["content"]
    assert not controller.force_final_answer
    assert [tool["function"]["name"] for tool in controller.active_tools()] == [
        "web_search",
        "python",
    ]


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


# --- schema-aware argument typing -------------------------------------------------------

_MCP_SERVER = {"id": "notes", "display_name": "Notes"}
_MCP_TOOL = {
    "name": "search",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "fuzzy": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "depth": {"type": ["integer", "null"]},
        },
    },
}
_MCP_PROPS = _MCP_TOOL["inputSchema"]["properties"]
_UNCHANGED = object()


def _mcp_tool_schemas():
    return _mcp_specs_for_server(_MCP_SERVER, [_MCP_TOOL])


def _coerce_one(key, value, props):
    return coerce_arguments_by_schema({key: value}, props)[key]


@pytest.mark.parametrize(
    "key, text, expected",
    [
        ("fuzzy", " False ", False),
        ("limit", "25", 25),
        ("limit", "9007199254740993", 9007199254740993),  # float() would round this
        ("tags", "('a', 'b')", ["a", "b"]),
        ("depth", "null", None),
        ("fuzzy", "null", _UNCHANGED),  # null is not a boolean; None would mean False
    ],
)
def test_a_value_reads_as_the_type_its_schema_declares(key, text, expected):
    got = _coerce_one(key, text, _MCP_PROPS)
    want = text if expected is _UNCHANGED else expected
    # Types too: `False == 0` and `25 == 25.0`, so equality alone would miss a swap.
    assert got == want and type(got) is type(want)
    seam = coerce_tool_arguments(
        json.dumps({key: text}),
        heal = False,
        tool_name = "mcp__notes__search",
        tool_schemas = _mcp_tool_schemas(),
    )
    assert seam.arguments[key] == want and seam.healed is False


@pytest.mark.parametrize(
    "spec",
    [
        # The walk stops on any keyword it does not follow, whichever one it is.
        {"type": "boolean", "$ref": "#/$defs/Flag"},
        # Collapsing a union discards the rest, so it is read only where it is all there is.
        {"anyOf": [{"type": "boolean"}], "properties": {"x": {"type": "boolean"}}},
    ],
)
def test_a_schema_this_walk_cannot_read_is_left_alone(spec):
    assert _coerce_one("f", "false", {"f": spec}) == "false"
    assert _coerce_one("f", "null", {"f": {"anyOf": [spec, {"type": "null"}]}}) == "null"
    # Falsified so the refusal cannot swallow real schemas: `nullable` spells a type union.
    annotated = {"type": "boolean", "title": "F", "enum": [False]}
    assert _coerce_one("f", "false", {"f": annotated}) is False
    assert _coerce_one("f", "null", {"f": {"type": "boolean", "nullable": True}}) is None


@pytest.mark.parametrize(
    "text, repaired",
    [
        ('["a","b"}', ["a", "b"]),  # a closer not matching the innermost opener
        ('["a","b\\"', ["a", 'b"']),  # an open string ending on an ESCAPED quote
    ],
)
def test_a_malformed_container_is_repaired_only_when_healing(text, repaired):
    """Rewriting brackets invents structure, which is what the auto-heal opt-out is for."""

    def at_seam(heal):
        return coerce_tool_arguments(
            json.dumps({"tags": text}),
            heal = heal,
            tool_name = "mcp__notes__search",
            tool_schemas = _mcp_tool_schemas(),
        ).arguments["tags"]

    assert at_seam(True) == repaired
    assert at_seam(False) == text


def test_an_mcp_tool_call_parsed_from_xml_arrives_typed():
    content = (
        "<function=mcp__notes__search>"
        "<parameter=query>ship dates</parameter>"
        "<parameter=limit>25</parameter>"
        "<parameter=fuzzy>false</parameter>"
        '<parameter=tags>["a","b"]</parameter>'
        "<parameter=depth>null</parameter>"
        "</function>"
    )
    calls = parse_tool_calls_from_text(content)
    decision = ToolLoopController(tools = _mcp_tool_schemas()).prepare_call(calls[0])
    assert decision.arguments == {
        "query": "ship dates",
        "limit": 25,
        "fuzzy": False,
        "tags": ["a", "b"],
        "depth": None,
    }
    # The turn replayed to the model carries the typed values too, not the strings.
    assert decision.as_assistant_tool_call()["function"]["arguments"] == (
        '{"depth":null,"fuzzy":false,"limit":25,"query":"ship dates","tags":["a","b"]}'
    )


def test_a_declared_type_nested_in_a_container_is_read_too():
    """`edit_file` declares replace_all inside `edits.items`, so the top level is not enough
    and text that spells a boolean must survive as text."""
    edits = (
        '[{"old_string":"a","new_string":"b","replace_all":"false"},'
        '{"old_string":"false","new_string":"null","replace_all":"true"}]'
    )
    edit_file = next(t for t in ALL_TOOLS if t["function"]["name"] == "edit_file")
    props = edit_file["function"]["parameters"]["properties"]
    typed = [
        {"old_string": "a", "new_string": "b", "replace_all": False},
        {"old_string": "false", "new_string": "null", "replace_all": True},
    ]
    call = {"path": "app.py", "edits": edits}
    assert coerce_arguments_by_schema(call, props) == {"path": "app.py", "edits": typed}
    # An already-typed container is descended into too: its elements can still be text.
    call = {"path": "app.py", "edits": json.loads(edits)}
    assert coerce_arguments_by_schema(call, props) == {"path": "app.py", "edits": typed}

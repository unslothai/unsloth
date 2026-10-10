# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth's UI control frames are opt-in on OpenAI-compatible streams.

Frames like ``tool_status`` / ``reasoning_summary`` carry no ``choices``, so
strict OpenAI clients (openai-python, the Vercel AI SDK, opencode) fail schema
validation mid-stream when they arrive. /v1/chat/completions therefore emits a
clean OpenAI stream by default; the Studio UI opts in with X-Unsloth-Events: 1,
and durable runs (whose event log is replayed to that UI) opt in internally.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.inference.sse_control_frames import (
    ServerToolCallStripper,
    is_ui_control_sse_line,
    strip_server_executed_tool_call,
)
from routes.inference import (
    _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
    UI_STREAM_EVENTS_HEADER,
    _DroppedFrameKeepalive,
    _confirm_gate_has_no_channel,
    _launcher_tool_default_applies,
    _proxy_to_external_provider,
    _ui_stream_events_enabled,
    produce_openai_chat_completions,
)


def _request(headers: list[tuple[bytes, bytes]]):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def test_no_header_means_clean_openai_stream():
    assert _ui_stream_events_enabled(_request([])) is False


def test_header_opts_in():
    req = _request([(UI_STREAM_EVENTS_HEADER.lower().encode(), b"1")])
    assert _ui_stream_events_enabled(req) is True


def test_other_header_values_do_not_opt_in():
    for value in (b"0", b"true", b"yes", b"", b" 1x"):
        req = _request([(UI_STREAM_EVENTS_HEADER.lower().encode(), value)])
        assert _ui_stream_events_enabled(req) is False, value


def test_none_request_is_refused():
    assert _ui_stream_events_enabled(None) is False


def test_background_generation_run_opts_into_control_frames():
    # Durable runs replay SSE lines to the Studio UI, so they must carry the opt-in.
    from core.inference.chat_generation_runs import _background_request
    req = _background_request(app = None, run_id = "run-1", cancel_event = threading.Event())
    assert _ui_stream_events_enabled(req) is True


def test_openai_stream_control_yields_are_gated():
    # Every raw control-frame yield must sit behind the per-request opt-in; keepalive and
    # error chunks are plain SSE and exempt.
    src = inspect.getsource(produce_openai_chat_completions)
    lines = src.splitlines()
    control_yields = (
        'yield f"data: {json.dumps(event)}',
        'yield f"data: {json.dumps(cumulative)}',
        'yield f"data: {status_data}',
    )
    candidate_lines = {
        i + 1
        for i, line in enumerate(lines)
        if any(line.strip().startswith(p) for p in control_yields)
    }
    assert candidate_lines, "control-frame yields disappeared from the producer"

    guarded: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and "_ui_events" in ast.dump(node.test):
            guarded.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    ungated = sorted(candidate_lines - guarded)
    assert not ungated, f"ungated control-frame yields at producer lines {ungated}"


def test_dropped_frames_still_pace_a_keepalive():
    # A gated-off frame still restarts the keepalive wait; unpaced, a Cloudflare tunnel
    # drops the stream at ~100s idle.
    keepalive = _DroppedFrameKeepalive(now = 0.0)
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S - 0.01) is False
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S) is True
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S + 0.01) is False
    assert keepalive.due(now = 2 * _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S) is True


def test_every_gated_frame_falls_back_to_a_keepalive():
    # Dropping a frame must never mean writing nothing, so each opt-in branch carries the
    # paced keepalive on its else side.
    src = inspect.getsource(produce_openai_chat_completions)
    gates = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.If)
        and "_ui_events" in ast.dump(node.test)
        # Frame-emitting gates only; the opt-in also guards bookkeeping that writes nothing.
        and any(isinstance(child, ast.Yield) for child in ast.walk(ast.Module(node.body, [])))
    ]
    assert gates, "the per-request control-frame gate disappeared from the producer"

    missing = [
        node.lineno
        for node in gates
        if not (
            len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.If)
            and "_drop_keepalive" in ast.dump(node.orelse[0].test)
        )
    ]
    assert not missing, f"gated frames dropped with no keepalive at producer lines {missing}"


def test_control_frame_lines_are_recognised_by_type():
    # Shared vocabulary so the passthrough relay and the local gate cannot drift.
    for frame in (
        "tool_start",
        "tool_end",
        "tool_output",
        "tool_args",
        "tool_status",
        "diffusion_frame",
        "reasoning_summary",
    ):
        assert is_ui_control_sse_line('data: {"type": "%s"}' % frame) is True, frame


def test_ordinary_chunks_and_sse_scaffolding_are_not_control_frames():
    for line in (
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        'data: {"choices": [], "usage": {}, "_toolEvent": {"type": "tool_end"}}',
        "data: [DONE]",
        ": keep-alive",
        "event: message",
        "data: not json",
    ):
        assert is_ui_control_sse_line(line) is False, line


def _gate_payload(**kwargs):
    fields = {
        "stream": True,
        "bypass_permissions": False,
        "confirm_tool_calls": None,
        "permission_mode": None,
        "mcp_enabled": False,
        "enabled_tools": None,
        "tool_choice": None,
        "max_tool_calls_per_message": None,
        "enable_tools": True,
        "tools": None,
        "messages": [],
        "response_format": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_confirm_gate_needs_both_a_stream_and_the_frames():
    assert _confirm_gate_has_no_channel(_gate_payload(), False) is True
    assert _confirm_gate_has_no_channel(_gate_payload(), True) is False
    # Non-streaming: an unset mode stays lenient (health checks must not 400).
    assert _confirm_gate_has_no_channel(_gate_payload(stream = False), True) is False
    assert (
        _confirm_gate_has_no_channel(_gate_payload(stream = False, permission_mode = "ask"), True)
        is True
    )
    assert _confirm_gate_has_no_channel(_gate_payload(bypass_permissions = True), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(permission_mode = "off"), False) is False


def test_a_streaming_request_that_can_never_prompt_is_not_refused():
    # An unset mode reads as auto on a stream; deep research sends enabled_tools: [].
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = []), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = ["terminal"]), False) is True
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = None), False) is True
    assert (
        _confirm_gate_has_no_channel(_gate_payload(permission_mode = "ask", enabled_tools = []), False)
        is True
    )


def test_a_request_that_can_run_no_tool_is_not_refused():
    # The catalogue is withdrawn for tool_choice "none" or a spent budget.
    assert _confirm_gate_has_no_channel(_gate_payload(tool_choice = "none"), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(max_tool_calls_per_message = 0), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(max_tool_calls_per_message = 1), False) is True
    assert (
        _confirm_gate_has_no_channel(
            _gate_payload(stream = False, permission_mode = "ask", tool_choice = "none"), True
        )
        is True
    )


_USAGE_EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/features/settings/components/usage-examples.tsx"
)


_LIVE_SMOKE = Path(__file__).resolve().parent / "test_studio_api.py"


def test_the_live_tool_smoke_sends_the_shape_the_examples_hand_out():
    # Example 4 in the live smoke is the same curl the API keys tab shows, so it has to stay
    # runnable in the same way.
    src = _LIVE_SMOKE.read_text(encoding = "utf-8")
    body = src[src.index("def test_curl_with_tools") :]
    body = body[: body.index("\ndef ")]
    assert '"enable_tools": True' in body
    assert '"permission_mode": "off"' in body


def test_the_bundled_api_examples_are_still_runnable():
    # API keys tab snippets stream without control frames; without an explicit mode the
    # gate would refuse them.
    src = _USAGE_EXAMPLES.read_text(encoding = "utf-8")
    tool_branches = src.count("enable_tools")
    assert tool_branches, "the tool variants disappeared from the examples"
    assert (
        src.count('permission_mode": "off"')
        + src.count('permission_mode = "off"')
        + src.count('permission_mode: "off"')
        == tool_branches
    ), "every example that enables tools must pick a permission mode"

    example = _gate_payload(
        enabled_tools = ["web_search", "python", "terminal"],
        permission_mode = "off",
    )
    assert _confirm_gate_has_no_channel(example, False) is False
    assert (
        _confirm_gate_has_no_channel(
            _gate_payload(enabled_tools = ["web_search", "python", "terminal"]), False
        )
        is True
    )


def test_a_structured_type_field_does_not_crash_the_relay():
    # A non-string `type` is unhashable for the frozenset test and would raise.
    for value in ('{"a": 1}', "[1, 2]", "3", "null", "true"):
        line = 'data: {"type": %s, "choices": []}' % value
        assert is_ui_control_sse_line(line) is False, line


def test_the_loops_bare_status_frames_are_held_back_too():
    # RAG autoinjection status frames carry no choices, so strict clients fail on them.
    assert is_ui_control_sse_line('data: {"type": "status", "text": "Searching: x"}') is True
    assert is_ui_control_sse_line('data: {"type": "status", "text": ""}') is True
    # usage and error are the provider's own vocabulary; a client reads them.
    assert is_ui_control_sse_line('data: {"type": "error", "error": {"message": "x"}}') is False
    assert is_ui_control_sse_line('data: {"type": "x", "usage": {"total_tokens": 3}}') is False
    # A context_truncated chunk keeps its choices, so it is a chunk, not a frame.
    assert is_ui_control_sse_line('data: {"choices": [], "context_truncated": {}}') is False


def test_a_call_the_server_runs_itself_is_not_offered_to_the_caller():
    # Server-executed tool_calls relayed to a client would make it run the tool twice.
    assert (
        strip_server_executed_tool_call(
            'data: {"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "c1"}]}}]}'
        )
        is None
    )
    assert (
        strip_server_executed_tool_call(
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
        )
        is None
    )
    kept = strip_server_executed_tool_call(
        'data: {"choices": [{"index": 0, "delta": {"content": "hi", "tool_calls": [{"id": "c"}]}}]}'
    )
    assert kept is not None and "tool_calls" not in kept and '"content":"hi"' in kept
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"total_tokens": 3}}',
        "data: [DONE]",
        ": keep-alive",
    ):
        assert strip_server_executed_tool_call(line) == line, line


def test_the_relay_only_strips_calls_the_loop_owns():
    # On a plain proxy the calls are the caller's own, so the strip is gated on the loop.
    src = inspect.getsource(_proxy_to_external_provider)
    assert "if not _ui_events and policy is not None:" in src
    assert "if not _ui_events and run_studio_tool_loop:" in src
    assert src.count("_tool_call_stripper.strip(line)") == 2


def test_a_legacy_function_call_is_left_for_the_caller():
    # The loop only executes delta.tool_calls, so a legacy function_call is the caller's.
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"function_call": {"name": "f"}}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]}',
    ):
        assert strip_server_executed_tool_call(line) == line, line
    plain = 'data: {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}'
    assert strip_server_executed_tool_call(plain) == plain


def test_a_stop_that_only_looks_final_is_held_back_too():
    # llama.cpp and vLLM finish good tool calls on "stop"; an empty "stop" chunk would end
    # a client before the answer.
    stripper = ServerToolCallStripper()
    call = (
        'data: {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, '
        '"id": "c1", "type": "function", "function": {"name": "python", '
        '"arguments": "{}"}}]}, "finish_reason": null}]}'
    )
    end = 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'

    assert stripper.strip(call) is None
    # Drop the closing "stop" too, or the caller ends the turn on an empty chunk.
    assert stripper.strip(end) is None
    answer = 'data: {"choices": [{"index": 0, "delta": {"content": "56088"}}]}'
    assert stripper.strip(answer) == answer
    assert stripper.strip(end) == end


def test_a_stop_the_loop_will_not_run_past_stays_final():
    # The loop refuses to run calls cut by "length"/"content_filter", so that turn is last.
    for reason in ("length", "content_filter"):
        stripper = ServerToolCallStripper()
        call = (
            'data: {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, '
            '"id": "c1", "type": "function", "function": {"name": "python"}}]}}]}'
        )
        stripper.strip(call)
        end = 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "%s"}]}' % reason
        assert json.loads(stripper.strip(end)[5:])["choices"][0]["finish_reason"] == reason


def _call_chunk(delta, finish = None):
    return "data: " + json.dumps(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


_ONE_CALL = [
    {"index": 0, "id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
]


def test_a_call_co_emitted_with_its_finish_reason_is_still_held_back():
    # Call and finish_reason on ONE line: raise the withheld flag before stripping.
    for reason in ("stop", "tool_calls"):
        stripper = ServerToolCallStripper()
        assert stripper.strip(_call_chunk({"tool_calls": _ONE_CALL}, reason)) is None
        answer = _call_chunk({"content": "56088"})
        assert json.loads(stripper.strip(answer)[5:])["choices"][0]["delta"]["content"] == "56088"
        end = _call_chunk({}, "stop")
        assert json.loads(stripper.strip(end)[5:])["choices"][0]["finish_reason"] == "stop"
        assert stripper.owed_terminal_chunk() is None


def test_a_withheld_call_the_loop_never_runs_still_ends_the_stream():
    # The loop can close without the promised turn; finish_reason is required (openai-node
    # raises "missing finish_reason for choice 0").
    stripper = ServerToolCallStripper()
    assert stripper.strip(_call_chunk({"tool_calls": _ONE_CALL})) is None
    assert stripper.strip(_call_chunk({}, "stop")) is None
    owed = stripper.owed_terminal_chunk()
    assert owed is not None
    payload = json.loads(owed[5:])
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["object"] == "chat.completion.chunk"
    assert (payload["id"], payload["model"]) == ("x", "m")
    assert stripper.owed_terminal_chunk() is None


def test_an_empty_tool_calls_entry_counts_as_a_withheld_call():
    # The strip removes the key whether or not truthy; the flag must read the same condition.
    stripper = ServerToolCallStripper()
    stripper.strip(_call_chunk({"tool_calls": []}))
    assert stripper.strip(_call_chunk({}, "stop")) is None


def test_both_relays_send_a_finish_the_stripper_still_owes():
    # A blanked terminal reason is only safe if something replaces it before [DONE].
    src = inspect.getsource(_proxy_to_external_provider)
    assert src.count("_tool_call_stripper.owed_terminal_chunk()") == 2


def test_a_turn_with_no_withheld_call_keeps_its_own_stop():
    # The state is per turn, not per stream: an ordinary turn must be relayed untouched.
    stripper = ServerToolCallStripper()
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
    ):
        assert stripper.strip(line) == line, line


def test_a_process_wide_tools_on_flag_does_not_refuse_a_plain_request():
    # --enable-tools fills the policy OVERRIDE slot, so the narrower default-only predicate
    # would 400 every ordinary stream on that launcher.
    payload = _gate_payload(enable_tools = None, enabled_tools = None)
    for policy in (None, True):
        with mock.patch("state.tool_policy.get_tool_policy", return_value = policy):
            assert _confirm_gate_has_no_channel(payload, False, ["python"]) is False
            assert _launcher_tool_default_applies(payload, False) is False
    asked = _gate_payload(enable_tools = True, enabled_tools = ["python"])
    for policy in (None, True):
        with mock.patch("state.tool_policy.get_tool_policy", return_value = policy):
            assert _confirm_gate_has_no_channel(asked, False, ["python"]) is True


def test_a_disabled_tool_policy_can_never_prompt():
    # --disable-tools vetoes even explicit enable_tools, so nothing can prompt.
    asked = _gate_payload(enable_tools = True, enabled_tools = ["python"])
    with mock.patch("state.tool_policy.get_tool_policy", return_value = False):
        assert _confirm_gate_has_no_channel(asked, False, ["python"]) is False


def test_the_selected_catalog_beats_a_stale_mcp_flag():
    # The resolved catalogue answers better than the mcp_enabled flag.
    payload = _gate_payload(mcp_enabled = True, enabled_tools = ["search_knowledge_base"])
    assert _confirm_gate_has_no_channel(payload, False) is True
    assert _confirm_gate_has_no_channel(payload, False, ["search_knowledge_base"]) is False
    assert _confirm_gate_has_no_channel(payload, False, ["python"]) is True
    assert _confirm_gate_has_no_channel(payload, False, ["mcp__server__do_thing"]) is True


def test_a_stripped_call_still_paces_a_keepalive():
    # A long argument stream drops every fragment; the relay must still write something.
    src = inspect.getsource(_proxy_to_external_provider)
    assert src.count("_tool_call_stripper.strip(line)") == 2
    assert src.count("if line is None:") == 2
    stripped_blocks = src.count("_drop_keepalive.due()")
    assert stripped_blocks == 4, (
        f"expected a paced keepalive on both control-frame and stripped-call drops, "
        f"found {stripped_blocks}"
    )


def test_the_launcher_default_does_not_claim_a_stream_it_cannot_prompt(monkeypatch):
    # A request that never mentions tools asked for plain chat; gating it would 400 or park.
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "get_tool_policy", lambda: None)
    silent = _gate_payload(enable_tools = None, enabled_tools = None)
    assert _launcher_tool_default_applies(silent, False) is False
    assert _confirm_gate_has_no_channel(silent, False) is False
    assert _launcher_tool_default_applies(silent, True) is True
    asked = _gate_payload(enable_tools = True, enabled_tools = None)
    assert _launcher_tool_default_applies(asked, False) is True
    assert _confirm_gate_has_no_channel(asked, False) is True
    assert (
        _launcher_tool_default_applies(_gate_payload(enable_tools = None, tool_choice = "none"), False)
        is False
    )


def test_both_local_branches_consult_the_launcher_default_rule():
    # The suppression has to happen where tools are switched on, or the loop still opens.
    src = inspect.getsource(produce_openai_chat_completions)
    assert src.count("_launcher_tool_default_applies(payload, _ui_events)") == 2


def test_the_mlx_counter_honours_tool_choice_none():
    # tool_choice "none" withdraws the catalogue, so the count must not price schemas.
    from routes import inference as inf

    src = inspect.getsource(inf._mlx_count_chat_tokens)
    assert 'tool_choice", None) == "none"' in src
    assert 'tool_choice", None) != "none"' in src


def test_tool_choice_none_withdraws_the_catalogue_but_not_the_capability():
    # _sf_template_tools picks the template branch READ; following tool_choice there drops
    # tool_calls history. Withdrawal belongs to _sf_tools_on/_sf_tools_to_use.
    src = inspect.getsource(produce_openai_chat_completions)
    detect = src[src.index("_sf_template_tools = ") :][:400]
    assert 'payload.tool_choice != "none"' not in detect
    assert 'm.role == "tool" or m.tool_calls' in detect
    assert 'if payload.tool_choice == "none":\n        _sf_tools_on = False' in src


def test_a_withheld_call_always_leaves_the_caller_a_finish_reason():
    # [DONE] alone offers no finish_reason to remove; still mint one (openai-node requires it).
    stripper = ServerToolCallStripper()
    call = 'data: {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]}}]}'
    assert stripper.strip(call) is None
    owed = stripper.owed_terminal_chunk()
    assert owed is not None and '"finish_reason":"stop"' in owed
    assert stripper.owed_terminal_chunk() is None


def test_a_removed_reason_owes_a_terminal_even_with_no_call_to_latch_onto():
    # finish_reason "tool_calls" without an emitted call (llama.cpp/vLLM bugs) must still
    # leave a debt.
    stripper = ServerToolCallStripper()
    stripper.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}')
    stripper.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
    )
    owed = stripper.owed_terminal_chunk()
    assert owed is not None and '"finish_reason":"stop"' in owed

    for reason in ("stop", "length", "content_filter"):
        kept = ServerToolCallStripper()
        kept.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}')
        kept.strip(
            'data: {"id": "c", "choices": [{"index": 0, "delta": {},'
            ' "finish_reason": "%s"}]}' % reason
        )
        assert kept.owed_terminal_chunk() is None, reason


def test_a_legacy_finish_after_a_structured_call_is_withheld_too():
    # Gateways may close on legacy "function_call" (LocalAI, litellm).
    stripper = ServerToolCallStripper()
    call = 'data: {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]}}]}'
    assert stripper.strip(call) is None
    out = stripper.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {},'
        ' "finish_reason": "function_call"}]}'
    )
    assert out is None or '"function_call"' not in out

    # A genuine legacy call is the caller's; pending keys on "tool_calls".
    legacy = ServerToolCallStripper()
    passed = legacy.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {"function_call":'
        ' {"name": "f", "arguments": "{}"}}}]}'
    )
    assert passed is not None and "function_call" in passed
    ended = legacy.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {},'
        ' "finish_reason": "function_call"}]}'
    )
    assert ended is not None and '"function_call"' in ended


def test_bypass_still_exempts_a_non_streaming_request():
    # Losing the bypass conjunct would 400 full-access non-streaming runs.
    for extra in ({}, {"permission_mode": "full"}):
        payload = _gate_payload(
            stream = False, bypass_permissions = True, confirm_tool_calls = True, **extra
        )
        assert _confirm_gate_has_no_channel(payload, False) is False, extra

    for extra in ({"confirm_tool_calls": True}, {"permission_mode": "ask"}):
        payload = _gate_payload(stream = False, **extra)
        assert _confirm_gate_has_no_channel(payload, False) is True, extra


def test_a_stream_that_kept_its_own_terminal_is_owed_nothing():
    plain = ServerToolCallStripper()
    plain.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}')
    plain.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'
    )
    assert plain.owed_terminal_chunk() is None

    after_call = ServerToolCallStripper()
    after_call.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]},'
        ' "finish_reason": "tool_calls"}]}'
    )
    after_call.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "a"}}]}')
    after_call.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'
    )
    assert after_call.owed_terminal_chunk() is None


def test_the_mlx_counter_keeps_capability_out_of_the_withdrawal_too():
    from routes import inference as inf

    src = inspect.getsource(inf._mlx_count_chat_tokens)
    detect = src[src.index("_template_tools = ") :][:400]
    assert 'tool_choice", None) != "none"' not in detect
    assert 'm.role == "tool" or m.tool_calls' in detect


def test_external_provider_relay_drops_control_frames_too():
    src = inspect.getsource(_proxy_to_external_provider)
    relays = [line for line in src.splitlines() if line.strip() == 'yield f"{line}\\n\\n"']
    assert relays, "the provider relay yields disappeared"
    assert src.count("is_ui_control_sse_line(line)") == len(
        relays
    ), "every provider relay must hold control frames back from a non-opt-in caller"


def test_a_safetensors_stream_that_disabled_tool_calls_opens_no_loop(monkeypatch):
    # tool_choice "none" is exempted from needing a channel; the safetensors/MLX branch
    # must not open the loop then, or the first high-risk call parks for the hour.
    import asyncio

    from core.inference.api_monitor import ApiMonitor
    from models.inference import ChatCompletionRequest, ChatMessage
    import routes.inference as inf
    from state.tool_policy import reset_tool_policy

    opened_loop = []

    class _Backend:
        active_model_name = "sf-model"
        models = {
            "sf-model": {
                "chat_template_info": {"template": "<tool_call> chatml"},
                "context_length": 2048,
            }
        }

        def generate_chat_response(
            self,
            *,
            messages,
            tools = None,
            stats_holder = None,
            **kw,
        ):
            yield "plain answer"

        def generate_chat_completion_with_tools(self, **kwargs):
            opened_loop.append(kwargs)
            yield {"type": "content", "content": ""}

        def reset_generation_state(self, caller_cancel_event = None):
            pass

        def resize_image(self, image):
            return image

    reset_tool_policy()
    monkeypatch.setattr(inf, "api_monitor", ApiMonitor(max_entries = 8))
    monkeypatch.setattr(
        inf,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = False, supports_tools = False, is_vision = False, context_length = None
        ),
    )
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(
        inf, "_detect_safetensors_features", lambda *a, **k: {"supports_tools": True}
    )

    payload = ChatCompletionRequest(
        model = "default",
        messages = [ChatMessage(role = "user", content = "hi")],
        stream = True,
        enable_tools = True,
        tool_choice = "none",
    )

    async def _run():
        response = await inf.openai_chat_completions(
            payload, request = _request([]), current_subject = "u"
        )
        return [chunk async for chunk in response.body_iterator]

    body = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in asyncio.run(_run()))
    assert "invalid_request_error" not in body
    assert opened_loop == [], "tool_choice: 'none' still opened the safetensors tool loop"
    assert "plain answer" in body

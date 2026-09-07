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
    # Durable runs replay the producer's SSE lines (tool cards included) to the Studio UI,
    # so their synthetic request must carry the opt-in.
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
    # A gated-off frame writes nothing but still restarts the stall-keepalive wait and
    # tool_stream_exec's heartbeat, so an unpaced stream is silent for the whole tool run
    # and a Cloudflare quick tunnel drops it at ~100s idle.
    keepalive = _DroppedFrameKeepalive(now = 0.0)
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S - 0.01) is False
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S) is True
    # Paced, not per-frame: the window restarts from the one just written.
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
    # The vocabulary lives in sse_control_frames so the passthrough relay and the local gate
    # cannot drift apart when a frame type is added.
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
        # An Unsloth extension stamped inside a real chunk still carries choices.
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
        # Explicit, so the launcher-default branch is not what is under test here.
        "enable_tools": True,
        # Read by _request_states_tool_intent when the launcher default is in play.
        "tools": None,
        "messages": [],
        "response_format": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_confirm_gate_needs_both_a_stream_and_the_frames():
    # Either channel missing means the gate has nowhere to ask.
    assert _confirm_gate_has_no_channel(_gate_payload(), False) is True
    assert _confirm_gate_has_no_channel(_gate_payload(), True) is False
    # Non-streaming keeps its old reading: an unset mode stays lenient (a health check must
    # not 400), an explicit one has nowhere to prompt.
    assert _confirm_gate_has_no_channel(_gate_payload(stream = False), True) is False
    assert (
        _confirm_gate_has_no_channel(_gate_payload(stream = False, permission_mode = "ask"), True)
        is True
    )
    # An explicit opt-out of the gate needs no channel at all.
    assert _confirm_gate_has_no_channel(_gate_payload(bypass_permissions = True), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(permission_mode = "off"), False) is False


def test_a_streaming_request_that_can_never_prompt_is_not_refused():
    # An unset permission_mode is read as auto on a stream, the way the loop defaults it, so
    # an always-safe selection is not refused over a prompt that can never fire. Deep research
    # drives its own /v1/chat/completions this way (enabled_tools: []).
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = []), False) is False
    # A selection that can prompt still is.
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = ["terminal"]), False) is True
    # As is an omitted selection, which resolves to every built-in tool.
    assert _confirm_gate_has_no_channel(_gate_payload(enabled_tools = None), False) is True
    # An explicit ask never converges on auto's leniency.
    assert (
        _confirm_gate_has_no_channel(_gate_payload(permission_mode = "ask", enabled_tools = []), False)
        is True
    )


def test_a_request_that_can_run_no_tool_is_not_refused():
    # stream_with_studio_tools withdraws the catalogue unless tool_choice is not "none" and
    # the budget is unspent, so neither shape can reach a prompt. The selector reads neither
    # field, so the catalogue alone cannot answer this.
    assert _confirm_gate_has_no_channel(_gate_payload(tool_choice = "none"), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(max_tool_calls_per_message = 0), False) is False
    # An unspent budget is not a disabled one.
    assert _confirm_gate_has_no_channel(_gate_payload(max_tool_calls_per_message = 1), False) is True
    # Non-streaming keeps its own reading; this only narrows the stream refusal.
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
    # The API keys tab's copy-paste snippets stream with python and terminal enabled and
    # deliberately do not take the control frames, so without an explicit mode the confirm
    # gate would refuse every one of them before generation.
    src = _USAGE_EXAMPLES.read_text(encoding = "utf-8")
    tool_branches = src.count("enable_tools")
    assert tool_branches, "the tool variants disappeared from the examples"
    assert (
        src.count('permission_mode": "off"')
        + src.count('permission_mode = "off"')
        + src.count('permission_mode: "off"')
        == tool_branches
    ), "every example that enables tools must pick a permission mode"

    # And the shape they now send is one the gate admits.
    example = _gate_payload(
        enabled_tools = ["web_search", "python", "terminal"],
        permission_mode = "off",
    )
    assert _confirm_gate_has_no_channel(example, False) is False
    # Without the mode it would not be, which is what the snippets guard against.
    assert (
        _confirm_gate_has_no_channel(
            _gate_payload(enabled_tools = ["web_search", "python", "terminal"]), False
        )
        is True
    )


def test_a_structured_type_field_does_not_crash_the_relay():
    # sanitize_provider_sse_line passes a non-string `type` through, and a frozenset
    # membership test on an unhashable value raises, so a custom provider could end an
    # otherwise relayable stream with a server error.
    for value in ('{"a": 1}', "[1, 2]", "3", "null", "true"):
        line = 'data: {"type": %s, "choices": []}' % value
        assert is_ui_control_sse_line(line) is False, line


def test_the_loops_bare_status_frames_are_held_back_too():
    # build_synthetic_search_exchange brackets a RAG autoinjection with {"type": "status"}
    # frames, written straight onto the relayed stream. "status" is not in the
    # provider-forgery vocabulary, but it carries no choices, so a strict client fails on it
    # exactly like a tool card.
    assert is_ui_control_sse_line('data: {"type": "status", "text": "Searching: x"}') is True
    assert is_ui_control_sse_line('data: {"type": "status", "text": ""}') is True
    # usage and error are the provider's own vocabulary; a client reads them.
    assert is_ui_control_sse_line('data: {"type": "error", "error": {"message": "x"}}') is False
    assert is_ui_control_sse_line('data: {"type": "x", "usage": {"total_tokens": 3}}') is False
    # A context_truncated chunk keeps its choices, so it is a chunk, not a frame.
    assert is_ui_control_sse_line('data: {"choices": [], "context_truncated": {}}') is False


def test_a_call_the_server_runs_itself_is_not_offered_to_the_caller():
    # The loop relays the provider's delta.tool_calls and the finish_reason ending that turn,
    # for a call Unsloth runs and answers later. The catalogue is Unsloth's own, so a client
    # acting on those chunks runs the tool twice, or stops before the real answer arrives.
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
    # A chunk that also carries prose keeps the prose.
    kept = strip_server_executed_tool_call(
        'data: {"choices": [{"index": 0, "delta": {"content": "hi", "tool_calls": [{"id": "c"}]}}]}'
    )
    assert kept is not None and "tool_calls" not in kept and '"content":"hi"' in kept
    # Everything else passes through byte-for-byte.
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"total_tokens": 3}}',
        "data: [DONE]",
        ": keep-alive",
    ):
        assert strip_server_executed_tool_call(line) == line, line


def test_the_relay_only_strips_calls_the_loop_owns():
    # On a plain proxy the calls are the caller's own, so the strip is gated on the loop
    # running: policy on the Codex branch, run_studio_tool_loop on the other.
    src = inspect.getsource(_proxy_to_external_provider)
    assert "if not _ui_events and policy is not None:" in src
    assert "if not _ui_events and run_studio_tool_loop:" in src
    assert src.count("_tool_call_stripper.strip(line)") == 2


def test_a_legacy_function_call_is_left_for_the_caller():
    # The loop reads delta.tool_calls and nothing else, so a legacy function_call is one it
    # never executes: stripping it would drop a call the caller is meant to run, and its
    # matching finish_reason would arrive with no name or arguments behind it.
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"function_call": {"name": "f"}}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]}',
    ):
        assert strip_server_executed_tool_call(line) == line, line
    # And a finish_reason with no call withheld beside it is the caller's own turn ending.
    plain = 'data: {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}'
    assert strip_server_executed_tool_call(plain) == plain


def test_a_stop_that_only_looks_final_is_held_back_too():
    # llama.cpp and vLLM finish a perfectly good structured tool call on "stop", and the loop
    # deliberately runs those (studio_tool_loop keeps "stop" out of `truncated`). Stripping
    # only the call leaves an empty chunk marked finish_reason "stop", so a client ending on
    # the first finish_reason never reads the answer the loop is about to stream.
    stripper = ServerToolCallStripper()
    call = (
        'data: {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, '
        '"id": "c1", "type": "function", "function": {"name": "python", '
        '"arguments": "{}"}}]}, "finish_reason": null}]}'
    )
    end = 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'

    # The call chunk carries nothing else, so it drops entirely, as it already did.
    assert stripper.strip(call) is None
    # So does the "stop" that closes that turn: otherwise the caller gets an empty chunk
    # marked finish_reason "stop" and ends the turn there.
    assert stripper.strip(end) is None
    # The loop's next turn is the caller's to read, finish_reason and all.
    answer = 'data: {"choices": [{"index": 0, "delta": {"content": "56088"}}]}'
    assert stripper.strip(answer) == answer
    assert stripper.strip(end) == end


def test_a_stop_the_loop_will_not_run_past_stays_final():
    # "length" and "content_filter" are the two the loop refuses to run (a call cut off at the
    # token ceiling may be half-written), so that turn really is the last one.
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
    # A provider may put the call and the reason that ends its turn on ONE line; any
    # non-streamed upstream relayed as a single line does. So the withheld-call flag has to be
    # raised before the strip reads it, or this leaks the empty "stop" the gate exists to
    # prevent and then eats the real one that follows.
    for reason in ("stop", "tool_calls"):
        stripper = ServerToolCallStripper()
        assert stripper.strip(_call_chunk({"tool_calls": _ONE_CALL}, reason)) is None
        answer = _call_chunk({"content": "56088"})
        assert json.loads(stripper.strip(answer)[5:])["choices"][0]["delta"]["content"] == "56088"
        # The loop's own turn ends normally, and that reason is the caller's to read.
        end = _call_chunk({}, "stop")
        assert json.loads(stripper.strip(end)[5:])["choices"][0]["finish_reason"] == "stop"
        assert stripper.owed_terminal_chunk() is None


def test_a_withheld_call_the_loop_never_runs_still_ends_the_stream():
    # The loop can close without opening the turn a withheld call promised: a spent tool
    # budget, a discarded call, a provider that failed mid-loop. The reason removed with that
    # call was then the stream's last one, and finish_reason is required -- openai-node raises
    # "missing finish_reason for choice 0".
    stripper = ServerToolCallStripper()
    assert stripper.strip(_call_chunk({"tool_calls": _ONE_CALL})) is None
    assert stripper.strip(_call_chunk({}, "stop")) is None
    owed = stripper.owed_terminal_chunk()
    assert owed is not None
    payload = json.loads(owed[5:])
    assert payload["choices"][0]["finish_reason"] == "stop"
    # Minted in the stream's own envelope, not a bare choices list.
    assert payload["object"] == "chat.completion.chunk"
    assert (payload["id"], payload["model"]) == ("x", "m")
    # Owed once, not on every later poll.
    assert stripper.owed_terminal_chunk() is None


def test_an_empty_tool_calls_entry_counts_as_a_withheld_call():
    # The strip removes the key whether or not it is truthy, so the flag must read the same
    # condition; otherwise the line drops without arming the turn and the next "stop" leaks.
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
    # `unsloth studio run --enable-tools` fills the tool-policy OVERRIDE slot, not the default
    # one, so _tools_on_by_launcher_default_only stops answering for it. Reading that narrower
    # predicate here would 400 every ordinary OpenAI stream on that launcher, `unsloth chat`'s
    # own included, over a slot choice the caller cannot see or act on.
    payload = _gate_payload(enable_tools = None, enabled_tools = None)
    for policy in (None, True):
        with mock.patch("state.tool_policy.get_tool_policy", return_value = policy):
            assert _confirm_gate_has_no_channel(payload, False, ["python"]) is False
            # Tools are withdrawn for that stream instead, so the loop it could not be
            # prompted for never opens.
            assert _launcher_tool_default_applies(payload, False) is False
    # A request that asked for tools itself is still refused: it can be told about the header.
    asked = _gate_payload(enable_tools = True, enabled_tools = ["python"])
    for policy in (None, True):
        with mock.patch("state.tool_policy.get_tool_policy", return_value = policy):
            assert _confirm_gate_has_no_channel(asked, False, ["python"]) is True


def test_a_disabled_tool_policy_can_never_prompt():
    # --disable-tools vetoes even an explicit enable_tools, so no loop opens and nothing
    # can prompt. Refusing there would 400 a request that would have run fine.
    asked = _gate_payload(enable_tools = True, enabled_tools = ["python"])
    with mock.patch("state.tool_policy.get_tool_policy", return_value = False):
        assert _confirm_gate_has_no_channel(asked, False, ["python"]) is False


def test_the_selected_catalog_beats_a_stale_mcp_flag():
    # mcp_enabled arms the classifier on the flag alone. The resolved catalogue answers
    # better: an MCP ask discovery filtered to nothing leaves only always-safe built-ins.
    payload = _gate_payload(mcp_enabled = True, enabled_tools = ["search_knowledge_base"])
    assert _confirm_gate_has_no_channel(payload, False) is True
    assert _confirm_gate_has_no_channel(payload, False, ["search_knowledge_base"]) is False
    # A catalogue that kept something confirmable still needs the channel.
    assert _confirm_gate_has_no_channel(payload, False, ["python"]) is True
    # An MCP tool that survived is not an always-safe built-in, so it still does.
    assert _confirm_gate_has_no_channel(payload, False, ["mcp__server__do_thing"]) is True


def test_a_stripped_call_still_paces_a_keepalive():
    # Same trap as the gated frames: a long argument stream drops every fragment and holds off
    # the loop's stall timer, so the relay must write something.
    src = inspect.getsource(_proxy_to_external_provider)
    assert src.count("_tool_call_stripper.strip(line)") == 2
    # Each strip that drops the line pairs with the paced keepalive before continuing.
    assert src.count("if line is None:") == 2
    stripped_blocks = src.count("_drop_keepalive.due()")
    assert stripped_blocks == 4, (
        f"expected a paced keepalive on both control-frame and stripped-call drops, "
        f"found {stripped_blocks}"
    )


def test_the_launcher_default_does_not_claim_a_stream_it_cannot_prompt(monkeypatch):
    # `unsloth studio run` installs a tools-on default, but a request that never mentions
    # tools cannot be expected to know about the header either. Putting it behind a confirm
    # gate would 400 every ordinary OpenAI call on that launcher, or park it in
    # wait_tool_decision on the first high-risk call. It asked for plain chat.
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "get_tool_policy", lambda: None)
    silent = _gate_payload(enable_tools = None, enabled_tools = None)
    assert _launcher_tool_default_applies(silent, False) is False
    assert _confirm_gate_has_no_channel(silent, False) is False
    # The Studio UI takes the frames, so the default keeps answering for it.
    assert _launcher_tool_default_applies(silent, True) is True
    # A request that asked for tools itself is not the launcher's default and still needs the
    # channel; so does one that stated its intent through the standard fields.
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
    # The safetensors completion withdraws the catalogue outright for tool_choice "none", so a
    # count that still prices the schemas, or trips the MCP discovery 503, no longer describes
    # the prompt generation renders. The MLX branch of the rule the GGUF counter already draws
    # with _client_disabled_tool_calls.
    from routes import inference as inf

    src = inspect.getsource(inf._mlx_count_chat_tokens)
    assert 'tool_choice", None) == "none"' in src
    assert 'tool_choice", None) != "none"' in src


def test_tool_choice_none_withdraws_the_catalogue_but_not_the_capability():
    # _sf_template_tools decides which branch of a named template is READ, not what is
    # rendered. Following tool_choice there reads the plain branch for a tool conversation,
    # turning off _sf_client_tools and dropping the assistant's tool_calls and the result's
    # correlation fields, exactly when "none" asks for the final answer. The withdrawal
    # belongs to _sf_tools_on and _sf_tools_to_use instead.
    src = inspect.getsource(produce_openai_chat_completions)
    detect = src[src.index("_sf_template_tools = ") :][:400]
    assert 'payload.tool_choice != "none"' not in detect
    assert 'm.role == "tool" or m.tool_calls' in detect
    # The catalogue itself is still withdrawn for "none".
    assert 'if payload.tool_choice == "none":\n        _sf_tools_on = False' in src


def test_a_withheld_call_always_leaves_the_caller_a_finish_reason():
    # A provider closing the turn on [DONE] alone offers no finish_reason to remove, so arming
    # the debt only where one was removed left the caller holding a stream whose only chunk
    # was withheld. finish_reason is required in the chunk schema: openai-node raises without.
    stripper = ServerToolCallStripper()
    call = 'data: {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]}}]}'
    assert stripper.strip(call) is None
    owed = stripper.owed_terminal_chunk()
    assert owed is not None and '"finish_reason":"stop"' in owed
    # Minted once, not on every call.
    assert stripper.owed_terminal_chunk() is None


def test_a_removed_reason_owes_a_terminal_even_with_no_call_to_latch_onto():
    # A provider can report finish_reason "tool_calls" for a call its own parser failed to
    # emit (open llama.cpp and vLLM bugs). Nothing is there for the withheld-call flag to
    # latch onto, so the reason was blanked with no debt recorded and the stream ended with no
    # finish_reason at all, the openai-node failure the minting exists to prevent.
    stripper = ServerToolCallStripper()
    stripper.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}')
    stripper.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
    )
    owed = stripper.owed_terminal_chunk()
    assert owed is not None and '"finish_reason":"stop"' in owed

    # The reasons the strip leaves alone are still genuinely final, so nothing is owed.
    for reason in ("stop", "length", "content_filter"):
        kept = ServerToolCallStripper()
        kept.strip('data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}')
        kept.strip(
            'data: {"id": "c", "choices": [{"index": 0, "delta": {},'
            ' "finish_reason": "%s"}]}' % reason
        )
        assert kept.owed_terminal_chunk() is None, reason


def test_a_legacy_finish_after_a_structured_call_is_withheld_too():
    # A gateway may stream modern delta.tool_calls and still close the turn on the legacy
    # "function_call" (LocalAI picks it whenever the client sent no tools, litellm relays it
    # verbatim). The loop runs such a call, so relaying the reason ends the caller's turn
    # early.
    stripper = ServerToolCallStripper()
    call = 'data: {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]}}]}'
    assert stripper.strip(call) is None
    out = stripper.strip(
        'data: {"id": "c", "choices": [{"index": 0, "delta": {},'
        ' "finish_reason": "function_call"}]}'
    )
    assert out is None or '"function_call"' not in out

    # A genuine legacy call is the caller's own to run and keeps its delta and its reason:
    # pending is keyed on "tool_calls", which a function_call delta never sets.
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
    # Losing the bypass conjunct here would 400 full-access non-streaming tool runs that work
    # today. Both shapes the validator can produce: it folds permission_mode "full" and
    # bypass_permissions into each other, and an explicit confirm_tool_calls survives the fold.
    for extra in ({}, {"permission_mode": "full"}):
        payload = _gate_payload(
            stream = False, bypass_permissions = True, confirm_tool_calls = True, **extra
        )
        assert _confirm_gate_has_no_channel(payload, False) is False, extra

    # Without bypass the non-streaming refusal stands: the gate would prompt and cannot.
    for extra in ({"confirm_tool_calls": True}, {"permission_mode": "ask"}):
        payload = _gate_payload(stream = False, **extra)
        assert _confirm_gate_has_no_channel(payload, False) is True, extra


def test_a_stream_that_kept_its_own_terminal_is_owed_nothing():
    # No spurious extra chunk when the caller already has a real finish_reason, whether or not
    # a call was withheld earlier in the stream.
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
    # Same split the completion draws: the catalogue goes for tool_choice "none", the
    # template branch used to read the history does not.
    from routes import inference as inf

    src = inspect.getsource(inf._mlx_count_chat_tokens)
    detect = src[src.index("_template_tools = ") :][:400]
    assert 'tool_choice", None) != "none"' not in detect
    assert 'm.role == "tool" or m.tool_calls' in detect


def test_external_provider_relay_drops_control_frames_too():
    # The provider proxy returns before the local producer's per-yield gates and relays the
    # loop's frames verbatim, so it filters the same vocabulary.
    src = inspect.getsource(_proxy_to_external_provider)
    relays = [line for line in src.splitlines() if line.strip() == 'yield f"{line}\\n\\n"']
    assert relays, "the provider relay yields disappeared"
    assert src.count("is_ui_control_sse_line(line)") == len(
        relays
    ), "every provider relay must hold control frames back from a non-opt-in caller"


def test_a_safetensors_stream_that_disabled_tool_calls_opens_no_loop(monkeypatch):
    # The gate exempts tool_choice: "none" from needing an event channel, on the promise that
    # no call can happen. The safetensors/MLX branch had to be made to keep that promise:
    # nothing on that path read the field (not _sf_use_tools, not _select_request_tools, and
    # generate_chat_completion_with_tools is never passed it), so the loop opened with the
    # full built-in catalogue for a headerless stream just admitted for being unable to call
    # anything. The first high-risk call then wrote a tool_start the relay drops and blocked
    # in wait_tool_decision for the hour.
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
    # No 400: the exemption still admits the request, which is the point of it.
    assert "invalid_request_error" not in body
    # And now it is telling the truth: no loop, so no prompt, so nothing to park on.
    assert opened_loop == [], "tool_choice: 'none' still opened the safetensors tool loop"
    assert "plain answer" in body

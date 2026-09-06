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
import threading

from pathlib import Path
from types import SimpleNamespace

from core.inference.sse_control_frames import (
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
    # Durable runs replay the producer's SSE lines (tool cards included) to the
    # Studio UI, so their synthetic request must carry the opt-in.
    from core.inference.chat_generation_runs import _background_request
    req = _background_request(app = None, run_id = "run-1", cancel_event = threading.Event())
    assert _ui_stream_events_enabled(req) is True


def test_openai_stream_control_yields_are_gated():
    # Every raw control-frame yield in the OpenAI chat producer must sit behind
    # the per-request opt-in; keepalive/error chunks are plain SSE and exempt.
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
    # A gated-off frame writes nothing but still restarts the stall-keepalive wait, and
    # tool_stream_exec's own heartbeat, so an unpaced stream is silent for the whole tool
    # run and a Cloudflare quick tunnel drops it at ~100s idle (_DroppedFrameKeepalive).
    keepalive = _DroppedFrameKeepalive(now = 0.0)
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S - 0.01) is False
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S) is True
    # Paced, not per-frame: the window restarts from the one just written.
    assert keepalive.due(now = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S + 0.01) is False
    assert keepalive.due(now = 2 * _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S) is True


def test_every_gated_frame_falls_back_to_a_keepalive():
    # Companion to the gating check above: dropping a frame must never mean writing
    # nothing, so each opt-in branch carries the paced keepalive on its else side.
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
    # The vocabulary lives in sse_control_frames so the passthrough relay and the local
    # gate cannot drift apart when a frame type is added.
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
    # Non-streaming keeps the reading it has always had: an unset mode stays lenient
    # there (a health check must not 400), an explicit one has nowhere to prompt.
    assert _confirm_gate_has_no_channel(_gate_payload(stream = False), True) is False
    assert (
        _confirm_gate_has_no_channel(_gate_payload(stream = False, permission_mode = "ask"), True)
        is True
    )
    # An explicit opt-out of the gate needs no channel at all.
    assert _confirm_gate_has_no_channel(_gate_payload(bypass_permissions = True), False) is False
    assert _confirm_gate_has_no_channel(_gate_payload(permission_mode = "off"), False) is False


def test_a_streaming_request_that_can_never_prompt_is_not_refused():
    # An unset permission_mode is read as auto on a stream, the way the loop defaults it,
    # so an always-safe selection is not refused over a prompt that can never fire. Deep
    # research drives its own /v1/chat/completions this way (enabled_tools: []).
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
    # stream_with_studio_tools withdraws the catalogue unless tool_choice is not "none"
    # and the budget is unspent, so neither shape can reach a prompt. The selector reads
    # neither field, so the catalogue alone cannot answer this.
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


def test_the_bundled_api_examples_are_still_runnable():
    # Copy-paste snippets from the API keys tab. They stream with python and terminal
    # enabled and deliberately do not take the control frames, so without an explicit
    # mode the confirm gate would refuse every one of them before generation.
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
    # sanitize_provider_sse_line deliberately passes a non-string `type` through, and a
    # frozenset membership test on an unhashable value raises, so a custom provider could
    # end an otherwise relayable stream with a server error.
    for value in ('{"a": 1}', "[1, 2]", "3", "null", "true"):
        line = 'data: {"type": %s, "choices": []}' % value
        assert is_ui_control_sse_line(line) is False, line


def test_the_loops_bare_status_frames_are_held_back_too():
    # build_synthetic_search_exchange brackets a RAG autoinjection with {"type": "status"}
    # frames, which stream_with_studio_tools writes straight onto the relayed stream.
    # "status" is not in the provider-forgery vocabulary, but it carries no choices, so a
    # strict client fails on it exactly like a tool card.
    assert is_ui_control_sse_line('data: {"type": "status", "text": "Searching: x"}') is True
    assert is_ui_control_sse_line('data: {"type": "status", "text": ""}') is True
    # usage and error are the provider's own vocabulary; a client reads them.
    assert is_ui_control_sse_line('data: {"type": "error", "error": {"message": "x"}}') is False
    assert is_ui_control_sse_line('data: {"type": "x", "usage": {"total_tokens": 3}}') is False
    # A context_truncated chunk keeps its choices, so it is a chunk, not a frame.
    assert is_ui_control_sse_line('data: {"choices": [], "context_truncated": {}}') is False


def test_a_call_the_server_runs_itself_is_not_offered_to_the_caller():
    # The loop relays the provider's delta.tool_calls and the finish_reason that ends that
    # turn, for a call Unsloth executes and answers in a later turn. Its catalogue is
    # Unsloth's own, so a client acting on those chunks runs the tool a second time, or
    # stops at the finish_reason before the real answer arrives.
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
    assert src.count("strip_server_executed_tool_call(line)") == 2


def test_a_legacy_function_call_is_left_for_the_caller():
    # The loop reads delta.tool_calls and nothing else, so a legacy function_call is one
    # it never executes: stripping it would drop a call the caller is meant to run, and
    # its matching finish_reason would arrive with no name or arguments behind it.
    for line in (
        'data: {"choices": [{"index": 0, "delta": {"function_call": {"name": "f"}}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]}',
    ):
        assert strip_server_executed_tool_call(line) == line, line
    # And a finish_reason with no call withheld beside it is the caller's own turn ending.
    plain = 'data: {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}'
    assert strip_server_executed_tool_call(plain) == plain


def test_the_selected_catalog_beats_a_stale_mcp_flag():
    # mcp_enabled arms the classifier on the flag alone. Once the catalogue is resolved it
    # answers better: an MCP ask that discovery filtered to nothing leaves only always-safe
    # built-ins, which never prompt.
    payload = _gate_payload(mcp_enabled = True, enabled_tools = ["search_knowledge_base"])
    assert _confirm_gate_has_no_channel(payload, False) is True
    assert _confirm_gate_has_no_channel(payload, False, ["search_knowledge_base"]) is False
    # A catalogue that kept something confirmable still needs the channel.
    assert _confirm_gate_has_no_channel(payload, False, ["python"]) is True
    # An MCP tool that survived is not an always-safe built-in, so it still does.
    assert _confirm_gate_has_no_channel(payload, False, ["mcp__server__do_thing"]) is True


def test_a_stripped_call_still_paces_a_keepalive():
    # Same trap as the gated frames: a long argument stream drops every fragment and keeps
    # the loop's stall timer from firing, so the relay must write something.
    src = inspect.getsource(_proxy_to_external_provider)
    assert src.count("strip_server_executed_tool_call(line)") == 2
    # Each strip that drops the line pairs with the paced keepalive before continuing.
    assert src.count("if line is None:") == 2
    stripped_blocks = src.count("_drop_keepalive.due()")
    assert stripped_blocks == 4, (
        f"expected a paced keepalive on both control-frame and stripped-call drops, "
        f"found {stripped_blocks}"
    )


def test_the_launcher_default_does_not_claim_a_stream_it_cannot_prompt(monkeypatch):
    # `unsloth studio run` installs a tools-on default. A request that never mentions
    # tools cannot be expected to know about the header either, so putting it behind a
    # confirm gate would 400 every ordinary OpenAI call on that launcher, or park it in
    # wait_tool_decision on the first high-risk call. It asked for plain chat.
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "get_tool_policy", lambda: None)
    silent = _gate_payload(enable_tools = None, enabled_tools = None)
    assert _launcher_tool_default_applies(silent, False) is False
    assert _confirm_gate_has_no_channel(silent, False) is False
    # The Studio UI takes the frames, so the default keeps answering for it.
    assert _launcher_tool_default_applies(silent, True) is True
    # A request that asked for tools itself is not the launcher's default, and still needs
    # the channel; so does one that stated its intent through the standard fields.
    asked = _gate_payload(enable_tools = True, enabled_tools = None)
    assert _launcher_tool_default_applies(asked, False) is True
    assert _confirm_gate_has_no_channel(asked, False) is True
    assert _launcher_tool_default_applies(
        _gate_payload(enable_tools = None, tool_choice = "none"), False
    ) is False


def test_both_local_branches_consult_the_launcher_default_rule():
    # The suppression has to happen where tools are switched on, or the loop still opens.
    src = inspect.getsource(produce_openai_chat_completions)
    assert src.count("_launcher_tool_default_applies(payload, _ui_events)") == 2


def test_external_provider_relay_drops_control_frames_too():
    # The provider proxy returns before the local producer's per-yield gates and relays
    # stream_with_studio_tools' frames verbatim, so it filters the same vocabulary.
    src = inspect.getsource(_proxy_to_external_provider)
    relays = [line for line in src.splitlines() if line.strip() == 'yield f"{line}\\n\\n"']
    assert relays, "the provider relay yields disappeared"
    assert src.count("is_ui_control_sse_line(line)") == len(
        relays
    ), "every provider relay must hold control frames back from a non-opt-in caller"

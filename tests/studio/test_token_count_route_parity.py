# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The token count must describe the request the completion actually sends (#7453).

The counter used to apply the process tool policy without first asking which
route the request takes. With ``unsloth run --enable-tools`` set and a completed
tool call still in history, that made it select every built-in tool schema and
append the tool nudge, while the completion forwarded the same request to
llama-server verbatim and sent neither. The bar then reported a prompt that was
larger than the one being generated from.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
BACKEND = WORKDIR / "studio/backend"
REFRESH_TS = WORKDIR / "studio/frontend/src/features/chat/utils/refresh-context-usage.ts"


class _Payload:
    def __init__(
        self,
        *,
        tools = None,
        tool_choice = None,
        messages = None,
        enable_tools = None,
        mcp_enabled = False,
        response_format = None,
    ):
        self.tools = tools
        self.tool_choice = tool_choice
        self.messages = messages or []
        self.enable_tools = enable_tools
        self.mcp_enabled = mcp_enabled
        self.response_format = response_format


class _Backend:
    def __init__(
        self,
        *,
        supports_tools = True,
        supports_tool_passthrough = True,
    ):
        self.supports_tools = supports_tools
        self.supports_tool_passthrough = supports_tool_passthrough


def _route():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from routes import inference
    return inference


TOOL_HISTORY = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


def test_a_cli_tool_policy_does_not_pull_a_passthrough_request_back():
    """`--enable-tools` sets the process policy but does not ask for Unsloth's
    tool loop, so a request carrying tool history still goes to llama-server
    verbatim. The counter must agree, or it prices a tool catalog the completion
    never sends."""
    inference = _route()
    backend = _Backend()

    # The exact shape from the report: every Studio toggle off, one completed
    # tool call still in history, policy on from the CLI.
    payload = _Payload(messages = TOOL_HISTORY)
    assert inference._explicit_studio_tool_loop_requested(payload) is False
    assert inference._takes_tool_passthrough(payload, backend) is True

    # An explicit per-request ask is what takes the loop instead.
    assert (
        inference._takes_tool_passthrough(
            _Payload(messages = TOOL_HISTORY, enable_tools = True), backend
        )
        is False
    )
    assert (
        inference._takes_tool_passthrough(
            _Payload(messages = TOOL_HISTORY, mcp_enabled = True), backend
        )
        is False
    )

    # A client tool catalog is the other contract, and tool_choice "none"
    # withdraws it.
    catalog = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert inference._takes_tool_passthrough(_Payload(tools = catalog), backend) is True
    assert (
        inference._takes_tool_passthrough(_Payload(tools = catalog, tool_choice = "none"), backend)
        is False
    )

    # Plain chat has no contract at all and stays on the normal path.
    assert (
        inference._takes_tool_passthrough(
            _Payload(messages = [{"role": "user", "content": "hi"}]), backend
        )
        is False
    )

    # A backend that cannot forward tools keeps them, so the count is unaffected.
    assert (
        inference._takes_tool_passthrough(
            _Payload(messages = TOOL_HISTORY),
            _Backend(supports_tool_passthrough = False),
        )
        is False
    )


def _tools_priced_by_the_counter(**payload_fields):
    """Run /chat/count_tokens against a stub backend and report the tool catalog
    it handed to the tokenizer."""
    import asyncio

    inference = _route()
    from models.inference import ChatCountTokensRequest

    seen = []

    class _CountingBackend(_Backend):
        is_loaded = True
        model_identifier = "m"

        def count_chat_tokens(
            self,
            messages,
            system = None,
            tools = None,
            strict = False,
        ):
            seen.append(tools)
            return 7

    backend = _CountingBackend()
    original = inference.get_llama_cpp_backend
    inference.get_llama_cpp_backend = lambda: backend
    try:
        asyncio.run(
            inference.chat_count_tokens(
                ChatCountTokensRequest(model = "m", **payload_fields),
                current_subject = "tester",
            )
        )
    finally:
        inference.get_llama_cpp_backend = original
    return seen[0]


def test_a_withdrawn_tool_catalog_is_not_priced():
    """``tool_choice: "none"`` withdraws the caller's catalog from the completion:
    with no tool history the request leaves the passthrough, and the normal GGUF
    path calls ``generate_chat_completion``, which takes no tools at all. Handing
    the catalog to /apply-template then prices schemas nothing renders (#7453).
    """
    catalog = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    plain = [{"role": "user", "content": "hi"}]

    assert _tools_priced_by_the_counter(messages = plain, tools = catalog, tool_choice = "none") is None

    # The passthrough is the one route that does forward the catalog verbatim, so
    # it must keep counting it -- tool history satisfies the contract on its own.
    assert (
        _tools_priced_by_the_counter(messages = TOOL_HISTORY, tools = catalog, tool_choice = "none")
        == catalog
    )
    assert _tools_priced_by_the_counter(messages = plain, tools = catalog) == catalog

    # response_format takes a withdrawn catalog to the passthrough too, and there
    # _build_openai_passthrough_body still drops it, so the count must drop it as well.
    assert (
        _tools_priced_by_the_counter(
            messages = plain,
            tools = catalog,
            tool_choice = "none",
            response_format = {"type": "json_object"},
        )
        is None
    )


def test_the_counter_and_the_completion_ask_the_same_helper():
    """Two copies of this rule drift, and the drift is invisible: the count just
    quietly describes a different request."""
    src = (BACKEND / "routes/inference.py").read_text(encoding = "utf-8")
    calls = re.findall(r"(?<!def )_takes_tool_passthrough\(payload, llama_backend\)", src)
    assert len(calls) == 2, calls
    # Same for which client tools reach the wire: the passthrough body and the counter.
    tool_rule = re.findall(r"(?<!def )_passthrough_client_tools\(payload\)", src)
    assert len(tool_rule) == 2, tool_rule


def test_an_unowned_runtime_getter_is_declined():
    """Compare mode keeps several providers mounted and registration records no
    owner, so the newest getter can belong to a sibling pane. The caller's guard
    checks the thread it asked for, not the getter's owner, so it cannot catch
    that. Pricing the template alone beats reporting another pane's total."""
    src = REFRESH_TS.read_text(encoding = "utf-8")
    body = src.split("function currentRuntimeMessagesGetter", 1)[1].split("\n}", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    # It must no longer hand back whatever happens to be on top of the stack.
    assert ".at(-1)" not in body
    assert "runtimeMessagesGetters.length === 1" in body


def test_the_helper_reads_a_count_payload_which_has_no_tool_choice():
    """The two routes take different request models: ChatCountTokensRequest
    carries no tool_choice and no response_format, so the shared rule has to
    read both defensively or the count endpoint raises instead of counting."""
    inference = _route()
    from models.inference import ChatCountTokensRequest

    payload = ChatCountTokensRequest(messages = TOOL_HISTORY)
    assert not hasattr(payload, "tool_choice")
    assert inference._takes_tool_passthrough(payload, _Backend()) is True

    plain = ChatCountTokensRequest(messages = [{"role": "user", "content": "hi"}])
    assert inference._takes_tool_passthrough(plain, _Backend()) is False


def test_the_passthrough_recount_keeps_adjacent_user_turns():
    """Coalescing is a normal-GGUF-path step, so it must follow the routing.

    ``_openai_messages_for_passthrough`` drops the empty assistant sentinel but
    keeps the two user turns around it, which is the shape a thread has after a
    response is stopped before its first token. Merging them before counting
    renders a prompt that route never sends, and can even count clean where the
    real strict template rejects the unmerged history (#7453).
    """
    src = (BACKEND / "routes/inference.py").read_text(encoding = "utf-8")

    # The passthrough builder is the reference: it does not coalesce.
    builder = src.split("def _openai_messages_for_passthrough", 1)[1].split("\ndef ", 1)[0]
    assert "_coalesce_consecutive_user_turns" not in builder

    # ... so the counter may only coalesce once it knows it is NOT that route.
    counter = src.split("async def chat_count_tokens", 1)[1].split("\n@router", 1)[0]
    assert "if not _takes_passthrough:" in counter
    merge_at = counter.index("_coalesce_consecutive_user_turns(openai_messages)")
    route_at = counter.index("_takes_tool_passthrough(payload, llama_backend)")
    assert route_at < merge_at, "the route must be decided before the merge"

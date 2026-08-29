# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provider-agnostic Unsloth tool loop.

The transport is faked so these exercise the loop itself: turn cycling, the
budget, approvals, and the text-form healing that self-hosted models need. The
scripted streams are shaped like what llama.cpp, vLLM and Ollama actually emit,
including the malformed cases that motivated the healing path.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


def _sse(
    delta = None,
    finish = None,
    **extra,
) -> str:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload: dict = {"choices": [choice]}
    payload.update(extra)
    return "data: " + json.dumps(payload)


_DONE = "data: [DONE]"


def _tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


WEB = _tool("web_search")
PY = _tool("python")


class FakeTransport:
    """Replays scripted turns and records what the loop asked for each time."""

    def __init__(
        self,
        turns,
        *,
        heals = True,
    ):
        self.turns = [list(turn) for turn in turns]
        self.heals_text_tool_calls = heals
        self.requests: list[dict] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    """Record every execute_tool call and return a canned result."""
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments, **kwargs})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    return calls


def _run(
    transport,
    *,
    tools = None,
    tool_choice = None,
    messages = None,
    continue_final_message = False,
    **policy_kwargs,
):
    policy_fields = {
        "tools": tools if tools is not None else [WEB],
        "max_calls": 25,
        "timeout": 300,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    policy_fields.update(policy_kwargs)
    cancel_event = threading.Event()

    async def _collect():
        out = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = messages or [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = tool_choice,
                continue_final_message = continue_final_message,
            ),
            policy = ToolLoopPolicy(**policy_fields),
            cancel_event = cancel_event,
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(_collect())


def _events(lines, kind):
    out = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        payload = json.loads(raw)
        if payload.get("type") == kind:
            out.append(payload)
    return out


def _visible_text(lines) -> str:
    text = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        payload = json.loads(raw)
        if payload.get("type") in ("tool_start", "tool_end"):
            continue
        for choice in payload.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                text.append(content)
    return "".join(text)


# ── Structured tool calls (a well-behaved provider) ───────────────


def test_structured_call_executes_and_continues(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"unsloth"}',
                                },
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "Here is what I found."}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "unsloth"}
    assert len(_events(lines, "tool_start")) == 1
    assert _events(lines, "tool_end")[0]["result"] == "RESULT<web_search>"
    assert "Here is what I found." in _visible_text(lines)

    # The follow-up turn replays assistant tool_calls then the tool result.
    follow_up = transport.requests[1]["messages"]
    assert [message["role"] for message in follow_up[-2:]] == ["assistant", "tool"]
    assert follow_up[-1]["content"] == "RESULT<web_search>"


def test_a_conversation_search_here_gets_the_active_branch(executed):
    """The provider loops share the local paths' tool catalogue.

    So search_conversation is advertised here once a thread has an archive, and needs the
    branch for the same reason: the stored rows are the whole DAG, Retry included.
    """
    branch = [
        {"role": "user", "content": "what was the code"},
        {"role": "assistant", "content": "let me look"},
        {"role": "user", "content": "please"},
    ]
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_c",
                                "function": {
                                    "name": "search_conversation",
                                    "arguments": '{"query":"the code"}',
                                },
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "It was 5150."}), _sse(finish = "stop"), _DONE],
        ]
    )

    _run(transport, tools = [_tool("search_conversation")], messages = branch)

    assert [call["name"] for call in executed] == ["search_conversation"]
    assert executed[0]["conversation_branch"] == branch
    # And a budget, or the tool's clamp is skipped and a model-chosen top_k of 8 appends
    # roughly 4K tokens to a prompt this loop replays. Unsloth cannot measure an external
    # model's window, so the cap is one ordinary recall's worth.
    from core.rag import config as rag_config

    assert (
        executed[0]["conversation_budget_tokens"]
        == rag_config.CHUNK_TOKENS * rag_config.CONVERSATION_ARCHIVE_TOP_K
    )


def test_streamed_tool_name_fragments_are_not_concatenated(executed):
    """llama-server re-sends the whole name as it grows: web -> web_search."""
    transport = FakeTransport(
        [
            [
                _sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web"}}]}),
                _sse({"tool_calls": [{"index": 0, "function": {"name": "web_search"}}]}),
                _sse({"tool_calls": [{"index": 0, "function": {"arguments": '{"query":'}}]}),
                _sse({"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "x"}


# ── Text-form calls (what small self-hosted models actually emit) ──


def test_text_form_tool_call_is_healed_and_executed(executed):
    transport = FakeTransport(
        [
            [
                _sse({"content": "Let me look. "}),
                _sse(
                    {
                        "content": '<tool_call>{"name": "web_search", "arguments": {"query": "unsloth"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            [_sse({"content": "Found it."}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "unsloth"}
    # The markup is consumed, the prose around it survives.
    visible = _visible_text(lines)
    assert "Let me look." in visible
    assert "<tool_call>" not in visible


def test_partial_marker_split_across_deltas_is_not_broken(executed):
    """The signal itself straddles a chunk boundary."""
    transport = FakeTransport(
        [
            [
                _sse({"content": "<tool"}),
                _sse({"content": '_call>{"name": "web_search", "arg'}),
                _sse({"content": 'uments": {"query": "split"}}</tool_call>'}),
                _sse(finish = "stop"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    assert executed[0]["arguments"] == {"query": "split"}
    assert "<tool" not in _visible_text(lines)


def test_unterminated_envelope_is_released_as_prose_and_terminates(executed):
    """The vLLM hang: an envelope the model opens and never closes.

    Nothing may be swallowed. The turn must end with the held text visible
    rather than rendering an empty answer, and no call may be invented.
    """
    transport = FakeTransport(
        [
            [
                _sse({"content": "thinking... "}),
                _sse({"content": '<tool_call>{"name": "web_sea'}),
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    lines = _run(transport)

    assert executed == []
    visible = _visible_text(lines)
    assert "thinking..." in visible
    # The unparseable residue is flushed verbatim, not held forever.
    assert "web_sea" in visible


def test_undeclared_text_call_is_not_promoted(executed):
    """A name outside the selected catalog is data, not a call."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '<tool_call>{"name": "terminal", "arguments": {"command": "id"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    lines = _run(transport, tools = [WEB])

    assert executed == []
    assert "terminal" in _visible_text(lines)


def test_fenced_rehearsal_is_documentation_not_a_call(executed):
    """Markerless syntax quoted in markdown code must never execute (#6967, #8312)."""
    transport = FakeTransport(
        [
            [
                _sse({"content": 'Docs:\n```\npython[ARGS]{"code": "1"}\n```\n'}),
                _sse(finish = "stop"),
                _DONE,
            ]
        ]
    )
    lines = _run(transport, tools = [WEB, PY])

    assert executed == []
    assert "python[ARGS]" in _visible_text(lines)


def test_healing_is_off_for_a_transport_that_does_not_need_it(executed):
    """Codex emits structured calls; its text stream is relayed untouched."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ]
        ],
        heals = False,
    )
    lines = _run(transport)

    assert executed == []
    assert "<tool_call>" in _visible_text(lines)


def test_structured_call_makes_the_healer_dormant(executed):
    """A provider that emits both must not have its text double-counted."""
    transport = FakeTransport(
        [
            [
                _sse({"content": "prefix <tool_c"}),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert [call["name"] for call in executed] == ["web_search"]
    # Held text is flushed when the healer goes dormant, never dropped.
    assert "prefix <tool_c" in _visible_text(lines)


# ── Budget ────────────────────────────────────────────────────────


def test_zero_budget_withdraws_the_catalog(executed):
    transport = FakeTransport([[_sse({"content": "no tools for me"}), _sse(finish = "stop"), _DONE]])
    _run(transport, max_calls = 0)

    assert executed == []
    assert transport.requests[0]["tools"] is None
    assert transport.requests[0]["tool_choice"] == "none"


def test_denied_call_does_not_spend_an_iteration(executed, monkeypatch):
    decisions = ["deny", "allow"]
    monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda session, approval: object())
    monkeypatch.setattr(
        loop_mod,
        "wait_tool_decision",
        lambda slot, approval, cancel_event = None: decisions.pop(0),
    )
    monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda slot, approval: None)
    monkeypatch.setattr(loop_mod, "new_approval_id", lambda: "ap1")

    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "python", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c2",
                                "function": {"name": "python", "arguments": '{"query":"2"}'},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(
        transport,
        tools = [PY],
        max_calls = 1,
        permission_mode = "auto",
        confirm_calls = True,
    )

    # The denial consumed no budget, so the second call still had one left.
    assert [call["name"] for call in executed] == ["python"]
    assert executed[0]["arguments"] == {"query": "2"}


# ── Permissions ───────────────────────────────────────────────────


def test_auto_mode_prompts_only_for_high_risk_calls(executed, monkeypatch):
    slots: list = []
    monkeypatch.setattr(
        loop_mod,
        "begin_tool_decision",
        lambda session, approval: slots.append(approval) or object(),
    )
    monkeypatch.setattr(
        loop_mod, "wait_tool_decision", lambda slot, approval, cancel_event = None: "allow"
    )
    monkeypatch.setattr(loop_mod, "abort_tool_decision", lambda slot, approval: None)
    monkeypatch.setattr(loop_mod, "new_approval_id", lambda: "ap1")

    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(
        transport,
        tools = [WEB, PY],
        permission_mode = "auto",
        confirm_calls = True,
    )

    # web_search is not high-risk, so it runs without an approval card.
    assert slots == []
    assert _events(lines, "tool_start")[0]["awaiting_confirmation"] is False
    assert [call["name"] for call in executed] == ["web_search"]


def test_full_access_disables_the_sandbox_at_execution(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "python", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, tools = [PY], bypass_permissions = True)

    assert executed[0]["disable_sandbox"] is True


def test_sandbox_stays_on_by_default(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "python", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, tools = [PY])

    assert executed[0]["disable_sandbox"] is False


# ── Forced tool choice ────────────────────────────────────────────


def test_forced_choice_is_cleared_after_the_first_execution(executed):
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "answer"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, tool_choice = "required")

    assert transport.requests[0]["tool_choice"] == "required"
    # The result follow-up must be free to answer in prose.
    assert transport.requests[1]["tool_choice"] == "auto"


# ── Behaviours carried over from the local loops (PR #8630) ───────


def test_usage_is_summed_into_one_chunk(executed):
    """A multi-turn answer reports the shape a single-turn one does."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        },
                    }
                ),
                _DONE,
            ],
            [
                _sse({"content": "done"}),
                _sse(finish = "stop"),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 7,
                            "total_tokens": 27,
                            "prompt_tokens_details": {"cached_tokens": 6},
                        },
                    }
                ),
                _DONE,
            ],
        ]
    )
    lines = _run(transport)

    usages = []
    for line in lines:
        raw = line[6:] if line.startswith("data: ") else ""
        if not raw or raw == "[DONE]":
            continue
        payload = json.loads(raw)
        if "usage" in payload:
            usages.append(payload["usage"])

    # One chunk, carrying the sum, including the detail slices pricing reads.
    assert len(usages) == 1
    assert usages[0]["prompt_tokens"] == 30
    assert usages[0]["completion_tokens"] == 12
    assert usages[0]["total_tokens"] == 42
    assert usages[0]["prompt_tokens_details"]["cached_tokens"] == 10


def test_a_repeated_identical_call_does_not_re_execute(executed):
    """The controller's dedup stops a model spending the budget on one call."""
    call = {
        "index": 0,
        "id": "c1",
        "function": {"name": "web_search", "arguments": '{"query":"x"}'},
    }
    transport = FakeTransport(
        [
            [_sse({"tool_calls": [call]}), _sse(finish = "tool_calls"), _DONE],
            [_sse({"tool_calls": [dict(call, id = "c2")]}), _sse(finish = "tool_calls"), _DONE],
            [_sse({"content": "answer"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport)

    assert [c["name"] for c in executed] == ["web_search"]


def test_a_stalled_model_is_nudged_to_act(executed):
    """Small models often say what they will do instead of doing it."""
    transport = FakeTransport(
        [
            [_sse({"content": "I'll search for that now."}), _sse(finish = "stop"), _DONE],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "answer"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, nudge_tool_calls = True)

    assert [c["name"] for c in executed] == ["web_search"]
    # The retry sees the assistant stall before the nudge, not user -> user.
    second = transport.requests[1]["messages"]
    assert [message["role"] for message in second] == ["user", "assistant", "user"]
    assert second[-2]["content"] == "I'll search for that now."


def test_a_reasoning_only_stall_is_nudged_and_replayed(executed):
    """Magistral-style stalls put the whole promise inside the think block.

    Stripping tool markup empties such a turn, so classifying the stripped text
    alone would drop the nudge the local loops still give. The replayed turn must
    be the text that was classified, or the retry is user -> user again.
    """
    stall = "[THINK]I will search now.[/THINK]"
    transport = FakeTransport(
        [
            [_sse({"content": stall}), _sse(finish = "stop"), _DONE],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "answer"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, nudge_tool_calls = True)

    assert [c["name"] for c in executed] == ["web_search"]
    second = transport.requests[1]["messages"]
    assert [message["role"] for message in second] == ["user", "assistant", "user"]
    assert second[-2]["content"] == stall


@pytest.mark.parametrize(
    "stall, merged",
    [
        (
            "[THINK]The user wants a search.[/THINK] I will search now.",
            "The answer is not I will search now.",
        ),
        # Restored as written, not normalised to one space.
        (
            "[THINK]The user wants a search.[/THINK]\n\nI will search now.",
            "The answer is not\n\nI will search now.",
        ),
        # No markup to remove, so the model's own boundary is already at index 0.
        (" I will search now.", "The answer is not I will search now."),
        # The model wrote none, and inventing one would change what it continued.
        ("I will search now.", "The answer is notI will search now."),
    ],
)
def test_a_continued_stall_keeps_the_boundary_the_model_wrote(executed, stall, merged):
    """The merge has no separator, so a trimmed-away space glues two words.

    strip_tool_markup trims, and after a removed [THINK] block that trim sits past
    index 0, so measuring the leading run of the raw turn reports none.
    """
    transport = FakeTransport(
        [
            [_sse({"content": stall}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(
        transport,
        nudge_tool_calls = True,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer is not"},
        ],
        continue_final_message = True,
    )

    second = transport.requests[1]["messages"]
    assert [message["role"] for message in second] == ["user", "assistant", "user"]
    assert second[-2]["content"] == merged


_NOPE_CALL = '<tool_call>{"name": "nope", "arguments": {"q": "I will search now."}}</tool_call>'


@pytest.mark.parametrize(
    "stall, merged",
    [
        # The markup quotes the prose back, so searching the raw turn for the surviving
        # text finds the copy inside the arguments, whose own boundary is the quote.
        (" I will search now." + _NOPE_CALL, "The answer is not I will search now."),
        # Stripped at both ends with nothing between the block and the prose, so there
        # is no boundary to restore and one must not be invented.
        (
            "[THINK]plan[/THINK]I will search now." + _NOPE_CALL,
            "The answer is notI will search now.",
        ),
    ],
)
def test_the_boundary_is_read_off_the_stripped_turn_not_searched_for(executed, stall, merged):
    """The surviving prose can also sit inside the markup that was stripped out."""
    transport = FakeTransport(
        [
            [_sse({"content": stall}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(
        transport,
        auto_heal = False,
        nudge_tool_calls = True,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer is not"},
        ],
        continue_final_message = True,
    )

    second = transport.requests[1]["messages"]
    assert second[-2]["content"] == merged


def test_a_markup_only_stall_is_still_not_nudged(executed):
    """The reasoning fallback runs on the stripped text, so it must not revive Case G."""
    transport = FakeTransport(
        [
            [
                _sse({"content": '<tool_call>{"name": "not_enabled"}</tool_call>'}),
                _sse(finish = "stop"),
                _DONE,
            ],
            [_sse({"content": "SHOULD NOT APPEAR"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport, auto_heal = False, nudge_tool_calls = True)

    assert executed == []
    assert len(transport.requests) == 1


def test_a_stalled_model_is_not_nudged_by_default(executed, monkeypatch):
    """An API caller that omits the opt-in must not get a hidden retry."""
    from core.inference import passthrough_healing

    monkeypatch.setattr(passthrough_healing, "_NUDGE_DEFAULT", False)
    transport = FakeTransport(
        [
            [_sse({"content": "I'll search for that now."}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "SHOULD NOT APPEAR"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport)

    assert executed == []
    assert len(transport.requests) == 1
    assert "SHOULD NOT APPEAR" not in _visible_text(lines)


def test_a_stalled_model_respects_explicit_nudge_off(executed):
    transport = FakeTransport(
        [
            [_sse({"content": "I'll search for that now."}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "SHOULD NOT APPEAR"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, nudge_tool_calls = False)

    assert executed == []
    assert len(transport.requests) == 1


def test_nudging_is_independent_of_text_form_healing(executed):
    """Codex emits structured calls, but still needs plan-without-action recovery."""
    transport = FakeTransport(
        [
            [_sse({"content": "I'll search for that now."}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport, auto_heal = False, nudge_tool_calls = True)

    assert len(transport.requests) == 2


def test_a_finished_answer_is_not_nudged(executed):
    """A real answer must never be re-prompted into calling a tool."""
    answer = (
        "The capital of France is Paris, which has been the seat of government "
        "since the tenth century and remains the largest city in the country."
    )
    transport = FakeTransport([[_sse({"content": answer}), _sse(finish = "stop"), _DONE]])
    _run(transport, nudge_tool_calls = True)

    assert executed == []
    assert len(transport.requests) == 1


def test_the_loop_terminates_against_an_endlessly_calling_model(executed):
    """A provider that always asks for a tool must still stop."""

    class Endless:
        heals_text_tool_calls = True

        def __init__(self):
            self.turns = 0

        def stream(self, *, messages, tools, tool_choice, cancel_event):
            self.turns += 1
            n = self.turns

            async def _gen():
                yield _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"c{n}",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": f"q{n}"}),
                                },
                            }
                        ]
                    }
                )
                yield _sse(finish = "tool_calls")
                yield _DONE

            return _gen()

    transport = Endless()
    _run(transport, max_calls = 3)

    # Budget spent, catalog withdrawn, then one final no-tools pass.
    assert len(executed) == 3
    assert transport.turns <= 5


def test_tool_stdout_streams_while_the_call_runs(executed, monkeypatch):
    """A long call must emit progress rather than going silent."""

    def _execute(
        name,
        arguments,
        output_callback = None,
        **kwargs,
    ):
        if output_callback:
            output_callback("partial line 1\n")
            output_callback("partial line 2\n")
        return "final"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "python", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    lines = _run(transport, tools = [PY])

    progress = [line for line in lines if line.startswith("data: ") and "partial line" in line]
    assert progress, "no live tool output reached the client"


def test_replayed_assistant_content_carries_no_markup(executed):
    """Markup must not go back to the provider: the call replays structurally."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "content": 'Sure. <tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
                    }
                ),
                _sse(finish = "stop"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport)

    replayed = transport.requests[1]["messages"]
    assistant = [m for m in replayed if m.get("role") == "assistant"][-1]
    assert "<tool_call>" not in (assistant.get("content") or "")
    assert assistant.get("tool_calls")


def test_truncated_tool_markup_is_not_replayed_during_a_nudge(executed):
    """A length-truncated call is visible to the user, but not provider context."""
    markup = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    transport = FakeTransport(
        [
            [_sse({"content": f"I'll search now. {markup}"}), _sse(finish = "length"), _DONE],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, nudge_tool_calls = True)

    replayed = transport.requests[1]["messages"]
    assistant = [m for m in replayed if m.get("role") == "assistant"][-1]
    assert assistant["content"] == "I'll search now."


def test_a_markup_only_stall_is_not_nudged(executed):
    """Nothing to continue from, so the retry would have been user -> user.

    An intent phrase buried in an unpromotable call block reads as a stall to the
    classifier but strips to nothing, so there is no assistant turn to append and
    the nudge would merge into the user's own message. Easiest to reach on a
    transport that does not heal text-form calls, which is the Codex shape.
    """
    markup = (
        '<tool_call>{"name": "nope", "arguments": {"q": "I will look this up now"}}</tool_call>'
    )
    transport = FakeTransport(
        [
            [_sse({"content": markup}), _sse(finish = "stop"), _DONE],
            [_sse({"content": "SHOULD NOT APPEAR"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport, auto_heal = False, nudge_tool_calls = True)

    assert len(transport.requests) == 1


def test_conversation_roles_stay_alternating_for_a_strict_server(executed):
    """A no-op only turn must not leave two user turns in a row."""
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "not_a_tool", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport)

    roles = [m["role"] for m in transport.requests[1]["messages"]]
    assert all(not (a == "user" and b == "user") for a, b in zip(roles, roles[1:])), roles


def test_gemini_thought_signature_is_replayed_on_the_assistant_turn(executed):
    """Gemini 3 rejects a replayed functionCall without its thoughtSignature.

    The native translator stows the part-level signature on the tool_call delta
    as extra_content.google.thought_signature, so the accumulator has to carry
    it onto the assistant message or the first post-tool turn is refused.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "web_search", "arguments": ""},
                                "extra_content": {"google": {"thought_signature": "SIG-A"}},
                            }
                        ]
                    }
                ),
                _sse({"tool_calls": [{"index": 0, "function": {"arguments": '{"query":"u"}'}}]}),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "ok"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport)

    assistant = [m for m in transport.requests[1]["messages"] if m.get("role") == "assistant"][-1]
    call = assistant["tool_calls"][0]
    assert call["extra_content"] == {"google": {"thought_signature": "SIG-A"}}
    assert call["function"]["name"] == "web_search"
    assert json.loads(call["function"]["arguments"]) == {"query": "u"}


def _call_delta(index, call_id, name, arguments):
    return {
        "index": index,
        "id": call_id,
        "function": {"name": name, "arguments": arguments},
    }


def test_budget_exhausted_parallel_call_is_replayed_with_its_call(executed):
    """A tool result is only legal next to the call it answers.

    With one slot left and two parallel calls the second is refused, but its
    role="tool" note still goes back to the provider. Without the matching entry
    in the assistant message that note is an orphan, and OpenAI, Anthropic and
    Gemini all reject the follow-up rather than answering.
    """
    transport = FakeTransport(
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            _call_delta(0, "call_a", "web_search", '{"query":"a"}'),
                            _call_delta(1, "call_b", "web_search", '{"query":"b"}'),
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    lines = _run(transport, max_calls = 1)

    assert [call["name"] for call in executed] == ["web_search"]
    replayed = transport.requests[1]["messages"]
    called_ids = {
        call["id"]
        for message in replayed
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    result_ids = {message["tool_call_id"] for message in replayed if message.get("role") == "tool"}
    assert result_ids == {"call_a", "call_b"}
    assert not result_ids - called_ids
    # The refused call is replayed in OpenAI shape only: the parsed arguments
    # dict the loop keeps for itself must not reach the provider.
    exhausted = [
        call
        for message in replayed
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if call["id"] == "call_b"
    ][0]
    assert set(exhausted) == {"id", "type", "function"}
    assert exhausted["function"]["name"] == "web_search"
    assert len(_events(lines, "tool_end")) == 2


def test_unlimited_budget_runs_past_the_old_fixed_turn_cap(executed):
    """ "Max" means max: the sentinel used to fall back to 25 provider turns.

    Both local loops run an unlimited request for as many turns as the model
    asks for, and the fruitless-turn guard already ends a run that executes
    nothing, so a productive run must not stop short of its own answer.
    """
    turns = [
        [
            _sse(
                {"tool_calls": [_call_delta(0, f"call_{n}", "web_search", f'{{"query":"q{n}"}}')]}
            ),
            _sse(finish = "tool_calls"),
            _DONE,
        ]
        for n in range(40)
    ]
    turns.append([_sse({"content": "done"}), _sse(finish = "stop"), _DONE])
    transport = FakeTransport(turns, heals = False)
    lines = _run(transport, max_calls = 9999)

    assert len(executed) == 40
    assert _visible_text(lines) == "done"


def test_a_skipped_duplicate_closes_the_card_the_provider_already_painted(executed):
    """The loop relays the provider's tool_calls delta, so the client paints a
    card for every call. A repeat is answered with a nudge and never executed,
    which used to leave that card running for the rest of the answer.

    The event has to carry the id the provider streamed: a repeated call is
    exactly the one the loop renames to keep the replayed history unambiguous.
    """
    repeat = [
        _sse({"tool_calls": [_call_delta(0, "call_a", "web_search", '{"query":"a"}')]}),
        _sse(finish = "tool_calls"),
        _DONE,
    ]
    transport = FakeTransport(
        [
            list(repeat),
            list(repeat),
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    lines = _run(transport)

    assert len(executed) == 1
    ends = _events(lines, "tool_end")
    assert len(ends) == 2
    assert [end["tool_call_id"] for end in ends] == ["call_a", "call_a"]
    assert ends[1]["result"].startswith("Unsloth did not run this call")
    # Opened as well as closed. The client retires a card id when it closes it,
    # so a second tool_end on the same id resolves to no card and the adapter
    # drops it -- the skip would be invisible again. Announcing it first draws
    # the card the second event closes, and keeps the loop's invariant that
    # every tool_end has a matching tool_start.
    starts = _events(lines, "tool_start")
    assert [start["tool_call_id"] for start in starts] == ["call_a", "call_a"]


def test_a_second_call_at_one_index_keeps_its_own_argument_fragments(executed):
    """Two tool rounds in one response, both streamed at index 0.

    Providers restart ``delta.tool_calls[].index`` at 0 for every round while
    giving each call its own id, and the continuation fragments carrying the
    rest of the arguments are sent bare. Routing those by index alone appended
    round two's tail to round one, producing an unparseable blob and running
    both tools on the wrong arguments.
    """
    transport = FakeTransport(
        [
            [
                _sse({"tool_calls": [_call_delta(0, "call_a", "web_search", '{"query":')]}),
                _sse({"tool_calls": [{"index": 0, "function": {"arguments": '"first"}'}}]}),
                _sse({"tool_calls": [_call_delta(0, "call_b", "web_search", '{"query":')]}),
                _sse({"tool_calls": [{"index": 0, "function": {"arguments": '"second"}'}}]}),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport)

    assert [call["arguments"] for call in executed] == [{"query": "first"}, {"query": "second"}]


def test_a_fragment_naming_its_call_goes_back_to_that_call(executed):
    """Two calls at index 0 with the id repeated on every argument fragment.

    The latest-index mapping only exists to place fragments that carry no id, so
    a fragment that names the call the index opened first has to go back to it
    rather than fork a third slot. Forking left that call with truncated JSON,
    which reaches the tool as ``_raw``, and dropped the fragment for having no
    function name.
    """
    transport = FakeTransport(
        [
            [
                _sse({"tool_calls": [_call_delta(0, "call_a", "web_search", '{"query":')]}),
                _sse({"tool_calls": [_call_delta(0, "call_b", "web_search", '{"query":')]}),
                _sse(
                    {
                        "tool_calls": [
                            {"index": 0, "id": "call_a", "function": {"arguments": '"first"}'}}
                        ]
                    }
                ),
                _sse(
                    {
                        "tool_calls": [
                            {"index": 0, "id": "call_b", "function": {"arguments": '"second"}'}}
                        ]
                    }
                ),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ],
        heals = False,
    )
    _run(transport)

    assert [call["arguments"] for call in executed] == [{"query": "first"}, {"query": "second"}]

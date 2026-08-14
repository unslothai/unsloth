# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool the provider ran must survive into the next turn's prompt.

Hosted and local tools coexist in one turn: Gemini can return a code-execution
result while asking for a local ``web_search``, and OpenAI can generate an image
before requesting one. The hosted output reaches the client as its own
``_toolEvent`` frame, but the loop rebuilds the assistant message from the text
and tool calls it saw, so that output was absent from the conversation replayed
on the follow-up request. The model then answered from the local results alone,
having lost what it had just produced.

Studio's own tool events are written with a top-level ``type`` and never appear
as ``_toolEvent``, so the two sides stay distinguishable and local results are
not replayed twice.
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


_DONE = "data: [DONE]"

WEB = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _hosted_event(payload: dict) -> str:
    """A provider-side tool frame, shaped as external_provider emits it."""
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "_toolEvent": payload,
        }
    )


def _call_line(call_id: str = "c1") -> str:
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )


def _text(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]})


def _finish(reason: str = "tool_calls") -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class FakeTransport:
    heals_text_tool_calls = False
    # The OAI-compat transport sanitizes at ingress, so the loop must not strip
    # the provider's own _toolEvent frames on the way through.
    sanitizes_provider_frames = True

    def __init__(
        self,
        turns,
        *,
        max_turns = 20,
    ):
        self.turns = [list(turn) for turn in turns]
        self.requests: list[dict] = []
        self.max_turns = max_turns

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append({"messages": [dict(m) for m in messages]})
        assert len(self.requests) <= self.max_turns, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    calls: list[str] = []

    def _execute(name, arguments, **kwargs):
        calls.append(name)
        return f"LOCAL<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(transport):
    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [WEB],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(asyncio.wait_for(_collect(), timeout = 30))


def _replayed(transport) -> str:
    """Everything the second request carried, as one searchable blob."""
    assert len(transport.requests) > 1, "the loop never made a follow-up request"
    return json.dumps(transport.requests[1]["messages"])


# ── the gap ──────────────────────────────────────────────────────────


def test_a_hosted_result_reaches_the_follow_up_request(executed):
    """The provider searched, then asked us to run a tool. Both must survive."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_search",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nURL: https://unsloth.ai",
                    }
                ),
                _text("Let me also compute that."),
                _call_line(),
                _finish(),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "https://unsloth.ai" in replayed, "the hosted result was dropped"
    assert "LOCAL<web_search>" in replayed, "the local result was dropped"


def test_the_hosted_result_still_reaches_the_client(executed):
    """Capturing it for the replay must not stop it rendering."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "hosted output",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    out = _run(transport)
    assert any("hosted output" in line and "_toolEvent" in line for line in out)


def test_the_models_own_prose_is_kept_alongside(executed):
    transport = FakeTransport(
        [
            [
                _text("Searching now."),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "hosted output",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "Searching now." in replayed
    assert "hosted output" in replayed


# ── what must not be replayed ────────────────────────────────────────


def test_a_repeated_end_event_is_recorded_once(executed):
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "once only",
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "once only",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert _replayed(transport).count("once only") == 1


def test_a_start_event_alone_adds_nothing(executed):
    """A hosted tool that never reported a result has nothing to replay."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_search",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "hello" in replayed
    assert "web_search result" not in replayed


@pytest.mark.parametrize("result", [None, 42, {"a": 1}])
def test_a_malformed_result_is_ignored(executed, result):
    """A provider is not trusted to send a usable string here.

    A non-string is a malformed frame rather than an outcome, so nothing is
    recorded for it. An empty or blank string is different: the provider did
    report an outcome, and that case is
    ``test_a_silent_hosted_execution_still_reaches_the_next_turn``.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": result,
                    }
                ),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert "[web_search result]" not in _replayed(transport)


def test_an_event_without_a_call_id_is_ignored(executed):
    transport = FakeTransport(
        [
            [
                _hosted_event({"type": "tool_end", "tool_name": "web_search", "result": "orphan"}),
                _text("hello"),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert "orphan" not in _replayed(transport)


def test_a_frontend_image_sentinel_is_not_replayed(executed):
    """__IMAGES__ carries a full data URI for the card, not for the model.

    Replaying it verbatim would put megabytes of base64 into the next request as
    assistant text, costing context and tokens for something the model cannot
    read. Local results already go through the same stripper.
    """
    huge = "data:image/png;base64," + ("A" * 20000)
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "code_execution",
                        "result": '4\n__IMAGES__:["' + huge + '"]',
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "4" in replayed
    assert "__IMAGES__" not in replayed
    assert "AAAA" not in replayed


def test_the_start_events_operation_labels_the_result(executed):
    """The tool_end producers generally omit tool_name.

    For Gemini code execution the code that ran is only in the start event, so a
    result recorded alone replays as an unlabelled value the model cannot
    interpret.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "hosted-1",
                        "arguments": {"language": "python", "code": "print(2 + 2)"},
                    }
                ),
                _hosted_event(
                    {"type": "tool_end", "tool_call_id": "hosted-1", "result": "4"},
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "code_execution" in replayed, "the tool name came only from the start event"
    assert "print(2 + 2)" in replayed, "the operation that produced the result was lost"
    assert "4" in replayed


def test_a_generated_image_is_noted_without_its_bytes(executed):
    """image_generation reports an empty result and carries the picture apart.

    Requiring non-empty text dropped it entirely, but replaying the base64 is
    the same mistake as the sentinel, so record only that it happened.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "image_generation",
                        "tool_call_id": "hosted-1",
                        "arguments": {},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "result": "",
                        "image_b64": "B" * 5000,
                        "image_mime": "image/png",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "image_generation" in replayed
    assert "produced an image" in replayed
    assert "BBBB" not in replayed, "the base64 must not reach the next request"


def test_a_turn_with_no_hosted_tool_replays_exactly_as_before(executed):
    """The common case must be untouched by any of this."""
    transport = FakeTransport(
        [[_text("plain answer"), _call_line(), _finish()], [_finish("stop")], [_DONE]]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "plain answer" in replayed
    assert "result]" not in replayed


# ── the three cases review found after the first cut ─────────────────


def test_a_plot_with_no_stdout_is_still_reported(executed):
    """Gemini code execution can return nothing but the image sentinel.

    Stripping it leaves an empty string and image_b64 is not set on that path,
    so without noticing the sentinel the entry looks empty and the follow-up is
    told nothing was produced.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "hosted-1",
                        "arguments": {"code": "plt.plot(x)"},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "result": '\n__IMAGES__:["data:image/png;base64,' + ("C" * 4000) + '"]',
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "produced an image" in replayed
    assert "CCCC" not in replayed


def test_a_large_hosted_result_is_capped(executed):
    """Local execution caps what the model sees; the hosted copy must too."""
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "code_execution",
                        "result": "D" * 60000,
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "truncated" in replayed
    assert len(replayed) < 40000, f"replayed {len(replayed)} chars"


def test_a_stalled_turn_keeps_its_hosted_result(executed):
    """The model searched, then only said what it was about to do.

    That takes the stall reprompt, which returns to the provider without the
    replay below it, so the reprompted request could no longer see the search
    output it is being told to continue from.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nURL: https://unsloth.ai",
                    }
                ),
                _text("Let me check that."),
                _finish("stop"),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    assert len(transport.requests) > 1, "the stall reprompt never happened"
    assert "https://unsloth.ai" in json.dumps(transport.requests[1]["messages"])


def test_a_stalled_continuation_stays_one_assistant_turn(executed):
    """The stalled turn's replay must merge into a resumed partial.

    On a continuation the conversation ends with the assistant text being
    resumed, and that partial plus what the model just added are one turn.
    Appending instead leaves two assistant messages in a row, which puts a turn
    boundary in the middle of one sentence and is rejected outright by a server
    that enforces role alternation.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nURL: https://unsloth.ai",
                    }
                ),
                _text(" Let me check that."),
                _finish("stop"),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )

    async def _collect():
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "The answer is"},
                ],
                session_id = "s1",
                thread_id = "t1",
                continue_final_message = True,
            ),
            policy = ToolLoopPolicy(
                tools = [WEB],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for _ in agen:
            pass

    asyncio.run(asyncio.wait_for(_collect(), timeout = 30))

    messages = transport.requests[1]["messages"]
    roles = [m["role"] for m in messages]
    assert not any(
        roles[i] == "assistant" and roles[i + 1] == "assistant" for i in range(len(roles) - 1)
    ), f"the resumed partial was split off its own turn: {roles}"
    resumed = [m for m in messages if m["role"] == "assistant"]
    assert len(resumed) == 1
    assert resumed[0]["content"].startswith("The answer is Let me check that.")
    assert "https://unsloth.ai" in resumed[0]["content"]


# ── the label is the operation, not the transport's plumbing ─────────


def test_gemini_code_execution_replays_the_code_not_its_thought_signature(executed):
    """Gemini stows the native part on the same `arguments` the header renders.

    ``arguments.google.native_part`` exists so a follow-up turn can replay
    Gemini's required history shape, and it carries the whole ``executableCode``
    part plus an opaque ``thoughtSignature``. Rendered as prose that is a
    kilobyte of base64 cut off mid-token, and it pushed the header to the 2000
    character cap on every hosted execution.
    """
    signature = "SIG" + ("X" * 3000)
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "code_a",
                        "arguments": {
                            "kind": "code_execution",
                            "language": "python",
                            "code": "print(1 + 1)",
                            "_server_tool": True,
                            "google": {
                                "native_part": {
                                    "parts": [
                                        {
                                            "executableCode": {
                                                "id": "code_a",
                                                "language": "PYTHON",
                                                "code": "print(1 + 1)",
                                            },
                                            "thoughtSignature": signature,
                                        }
                                    ]
                                }
                            },
                        },
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "code_a",
                        "result": "2\n",
                        "google": {"native_part": {"parts": [{"codeExecutionResult": {}}]}},
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "print(1 + 1)" in replayed
    assert "2" in replayed
    assert "XXXX" not in replayed, "the thought signature reached the model"
    assert "native_part" not in replayed
    assert "_server_tool" not in replayed


def test_openai_image_generation_replays_the_prompt_it_actually_used(executed):
    """The prompt is only on the end event; the start opens with an empty one.

    OpenAI emits ``image_generation_call`` on output_item.added before it knows
    the prompt, so reading arguments from the start alone replayed
    ``"prompt": ""`` and the next turn could not tell what had been drawn. The
    same arguments carry the paired reasoning item, whose ``encrypted_content``
    is multi-kilobyte on a zero-data-retention org.
    """
    plumbing = {
        "openai_image_generation_call_id": "ig_abc",
        "openai_response_id": "resp_123",
        "openai_reasoning_item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "E" * 4000,
        },
    }
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "image_generation",
                        "tool_call_id": "ig_abc",
                        "arguments": {"kind": "image", "prompt": "", **plumbing},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "ig_abc",
                        "result": "",
                        "arguments": {
                            "kind": "image",
                            "prompt": "A photorealistic ginger cat",
                            **plumbing,
                        },
                        "image_b64": "B" * 5000,
                        "image_mime": "image/png",
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "A photorealistic ginger cat" in replayed, "the prompt was dropped"
    assert "produced an image" in replayed
    assert "EEEE" not in replayed, "the encrypted reasoning reached the model"
    assert "BBBB" not in replayed


# ── the cap and the envelope are the local ones, not copies ──────────


def test_the_hosted_cap_follows_the_configured_local_one(executed, monkeypatch):
    """An install that lowers the local cap lowers the hosted one too.

    ``UNSLOTH_TOOL_RESULT_MAX_CHARS`` exists so a deployment on a smaller
    context can shrink what a tool result is allowed to occupy. A hosted result
    held to its own hard-coded 16k would still inject far more than the local
    path allows on exactly those installs.
    """
    monkeypatch.setattr(loop_mod.tools_module, "_MAX_OUTPUT_CHARS", 500)
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "code_execution",
                        "result": "D" * 4000,
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "truncated" in replayed
    assert "D" * 600 not in replayed, "the configured cap was ignored"


def test_a_hosted_page_keeps_a_files_line_of_its_own(executed):
    """Only the sandbox tools emit the ``__FILES__`` envelope.

    A fetched page ending in a well formed one is content, and the stripper is
    told the tool's name so it leaves that line alone -- the same call the local
    path makes. Stripping it silently drops the tail of the fetched document
    before the follow-up turn ever sees it.
    """
    page = 'How the envelope looks\n__FILES__:[{"name": "plot.png", "size": 12}]'
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_fetch",
                        "tool_call_id": "hosted-1",
                        "arguments": {"url": "https://unsloth.ai/docs"},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "result": page,
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "How the envelope looks" in replayed
    assert "plot.png" in replayed, "the page's own __FILES__ line was stripped"


# ── a call that ran is a call the next turn hears about ──────────────


def test_a_silent_hosted_execution_still_reaches_the_next_turn(executed):
    """Gemini reports code that printed nothing as an empty result.

    ``codeExecutionResult.output`` is "" on a successful run whose code only
    wrote a file or defined a name, and the executed code is carried on the
    tool_start alone -- never as assistant text. Skipping an entry on an empty
    result therefore drops the whole execution, which is the loss this replay
    exists to prevent.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "code_a",
                        "arguments": {"language": "python", "code": "df.to_parquet('s.pq')"},
                    }
                ),
                _hosted_event({"type": "tool_end", "tool_call_id": "code_a", "result": ""}),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "df.to_parquet" in replayed, "the execution vanished from the replay"
    assert "(no output)" in replayed


def test_a_start_with_no_end_is_still_left_out(executed):
    """The other half of the same rule: a call that never finished says nothing.

    A stream cut between the two halves leaves a start on its own, and reporting
    that as a result would tell the next turn an execution completed when the
    provider never said so.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "code_a",
                        "arguments": {"language": "python", "code": "df.to_parquet('s.pq')"},
                    }
                ),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "df.to_parquet" not in replayed
    assert "no output" not in replayed


def test_a_long_hosted_argument_says_it_was_cut(executed):
    """Anthropic passes the model's whole tool input through as arguments.

    A ``create`` carries the entire file body there and answers with only
    "Created", so the label is the sole record of what was written. Cut without
    a notice it reads as a complete line that simply stops.
    """
    body = "# line\n" * 600
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "code_a",
                        "arguments": {
                            "kind": "text_editor",
                            "command": "create",
                            "path": "/tmp/a.py",
                            "file_text": body,
                        },
                    }
                ),
                _hosted_event({"type": "tool_end", "tool_call_id": "code_a", "result": "Created"}),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = _replayed(transport)
    assert "truncated" in replayed, "the label was cut with no notice"
    assert "Created" in replayed

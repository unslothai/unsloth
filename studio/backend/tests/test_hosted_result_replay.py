# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool the provider ran must survive into the next turn's prompt.

Hosted and local tools coexist in one turn: Gemini can return a code-execution
result while asking for a local ``web_search``. The hosted output reaches the
client as its own ``_toolEvent`` frame, but the loop rebuilds the assistant
message from the text and tool calls it saw, so that output was absent from the
replayed conversation and the model answered from the local results alone.

Unsloth's own tool events carry a top-level ``type`` and never appear as
``_toolEvent``, so local results are not replayed twice.
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
    # the provider's own _toolEvent frames.
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


def _run(transport, *, nudge_tool_calls: bool | None = None):
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
                nudge_tool_calls = nudge_tool_calls,
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
                        "result": "Title: Unsloth\nSnippet: gradient checkpointing lands",
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
    assert "gradient checkpointing lands" in replayed, "the hosted result was dropped"
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
    """A non-string is a malformed frame rather than an outcome, so it records
    nothing. An empty string IS an outcome, covered by
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

    Replaying it verbatim puts megabytes of base64 into the next request. Local
    results already go through the same stripper.
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
    """tool_end generally omits tool_name, and for Gemini code execution the code
    that ran is only in the start event, so a result recorded alone replays as an
    unlabelled value.
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

    Requiring non-empty text dropped it entirely, and replaying the base64 is
    the sentinel mistake again, so record only that it happened.
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


# ── image-only, oversized and stalled turns ──────────────────────────


def test_a_plot_with_no_stdout_is_still_reported(executed):
    """Gemini code execution can return nothing but the image sentinel.

    Stripping it leaves an empty string and image_b64 is unset on that path, so
    unnoticed the entry looks empty and the follow-up is told nothing was made.
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

    That takes the stall reprompt, which returns to the provider from above the
    main replay, so the request could no longer see the search output.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nSnippet: gradient checkpointing lands",
                    }
                ),
                _text("Let me check that."),
                _finish("stop"),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )
    _run(transport, nudge_tool_calls = True)
    assert len(transport.requests) > 1, "the stall reprompt never happened"
    assert "gradient checkpointing lands" in json.dumps(transport.requests[1]["messages"])


def test_a_stalled_continuation_stays_one_assistant_turn(executed):
    """The stalled turn's replay must merge into a resumed partial.

    The partial plus what the model just added are one turn. Appending leaves
    two assistant messages in a row, splitting a sentence across a turn boundary
    and getting rejected by a server that enforces role alternation.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nSnippet: gradient checkpointing lands",
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
                nudge_tool_calls = True,
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
    assert "gradient checkpointing lands" in resumed[0]["content"]


# ── the label is the operation, not the transport's plumbing ─────────


def test_gemini_code_execution_replays_the_code_not_its_thought_signature(executed):
    """Gemini stows the native part on the same `arguments` the header renders.

    ``arguments.google.native_part`` replays Gemini's required history shape and
    carries an opaque ``thoughtSignature``. As prose that is a kilobyte of base64
    cut mid-token, pushing the header to its cap on every hosted execution.
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

    OpenAI emits ``image_generation_call`` before it knows the prompt, so reading
    the start alone replayed ``"prompt": ""``. Those arguments also carry the
    paired reasoning item, multi-kilobyte on a zero-data-retention org.
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

    ``UNSLOTH_TOOL_RESULT_MAX_CHARS`` shrinks what a result may occupy on a
    smaller context; a hosted copy held to its own hard-coded 16k would ignore
    that on exactly those installs.
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

    A fetched page ending in a well formed one is content, so the stripper is
    given the tool's name, as the local path does. Otherwise the tail of the
    document is dropped before the follow-up turn sees it.
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

    ``codeExecutionResult.output`` is "" for a run that only wrote a file, and
    the code is carried on the tool_start alone, never as assistant text, so
    skipping an empty result drops the whole execution.
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

    A stream cut between the halves leaves a start alone, and reporting that
    would tell the next turn an execution completed that never did.
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

    A ``create`` carries the entire file body there and answers only "Created",
    so the label is the sole record of what was written and a silent cut reads
    as the whole of it.
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


def test_a_stalled_turn_keeps_its_thought_signature(executed):
    """Gemini 3 will not take its own turn back without the signature.

    The outbound translator pins the ``thoughtSignature`` back on from
    ``assistant.extra_content`` and nowhere else. The stall reprompt returns to
    the provider from above the main replay, so it has to carry it too.
    """
    signature = "SIGNATURE-abc123"
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "tool_name": "web_search",
                        "result": "Title: Unsloth\nSnippet: gradient checkpointing lands",
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "Let me check that.",
                                    "extra_content": {"google": {"thought_signature": signature}},
                                },
                            }
                        ]
                    }
                ),
                _finish("stop"),
            ],
            [_text("done"), _finish("stop")],
            [_DONE],
        ]
    )
    _run(transport, nudge_tool_calls = True)
    reprompted = transport.requests[1]["messages"]
    stalled = [
        m
        for m in reprompted
        if m["role"] == "assistant" and "gradient checkpointing lands" in str(m.get("content"))
    ]
    assert stalled, "the hosted result never reached the reprompt"
    assert stalled[0].get("extra_content") == {"google": {"thought_signature": signature}}


def test_an_empty_argument_the_model_meant_survives(executed):
    """Anthropic's text editor deletes by replacing with the empty string.

    ``str_replace`` with ``new_str: ""`` is an intentional deletion, and the
    schema allows it, so dropping the key leaves the next turn unable to tell a
    deletion from a replacement whose value was never captured. The provisional
    empty prompt this once guarded against is overwritten by the end event
    anyway, since the two halves merge in order.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "code_execution",
                        "tool_call_id": "edit_1",
                        "arguments": {
                            "kind": "text_editor",
                            "command": "str_replace",
                            "old_str": "debug = True",
                            "new_str": "",
                        },
                    }
                ),
                _hosted_event({"type": "tool_end", "tool_call_id": "edit_1", "result": "Edited"}),
                _call_line(),
                _finish(),
            ],
            [_finish("stop")],
            [_DONE],
        ]
    )
    _run(transport)
    replayed = "".join(
        str(m.get("content")) for m in transport.requests[1]["messages"] if m["role"] == "assistant"
    )
    assert '"new_str":""' in replayed, "the deletion looked like a missing value"


def test_a_page_that_writes_the_image_marker_is_not_an_image(executed):
    """The sentinel is an envelope, not any line mentioning the marker.

    A fetched page documenting the output protocol contains the literal text.
    Reading that as a picture reports an image the turn never produced.
    """
    transport = FakeTransport(
        [
            [
                _hosted_event(
                    {
                        "type": "tool_start",
                        "tool_name": "web_fetch",
                        "tool_call_id": "hosted-1",
                        "arguments": {"url": "https://unsloth.ai/docs/protocol"},
                    }
                ),
                _hosted_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": "hosted-1",
                        "result": "The card reads a line beginning __IMAGES__: and renders it.",
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
    assert "produced an image" not in replayed

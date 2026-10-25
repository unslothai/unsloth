# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Under --no-prefill-assistant llama-server renders a trailing assistant message as a finished
turn and generates a new one after it. A paused attempt of such a request resumes its partial as
that new turn: merged into the earlier answer, the resume continues a prefix that never existed."""

from __future__ import annotations

import threading

import pytest

from core.inference import llama_cpp
from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy,
    delta,
    done,
    finish,
    web_search_tool,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Earlier answer."},
]


def _recorder(monkeypatch, signal, **attributes):
    return PreemptRecorder(
        monkeypatch,
        [
            [delta("Once upon a time"), finish(), done()],
            [delta(" there was a cat."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 1},
        **attributes,
    )


def _plain(monkeypatch, *, continue_final_message = False, **attributes):
    signal = preemption.PreemptSignal()
    rec = _recorder(monkeypatch, signal, **attributes)
    list(
        rec.backend.generate_chat_completion(
            messages = [dict(m) for m in _HISTORY],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = RecordingPolicy(),
            continue_final_message = continue_final_message,
        )
    )
    assert len(rec.payloads) == 2
    return rec.payloads[1]


def _tool_loop(monkeypatch, *, continue_final_message = False, **attributes):
    signal = preemption.PreemptSignal()
    rec = _recorder(monkeypatch, signal, **attributes)
    list(
        rec.backend.generate_chat_completion_with_tools(
            messages = [dict(m) for m in _HISTORY],
            tools = [web_search_tool()],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = RecordingPolicy(),
            continue_final_message = continue_final_message,
            max_tool_iterations = 5,
            max_tokens = 100,
            permission_mode = "off",
        )
    )
    assert len(rec.payloads) == 2
    return rec.payloads[1]


@pytest.mark.parametrize("run", [_plain, _tool_loop], ids = ["plain", "tool-loop"])
def test_a_closed_turn_stays_closed_when_the_server_does_not_prefill(monkeypatch, run):
    resumed = run(monkeypatch, _assistant_prefill_enabled = False)
    assert resumed["messages"][-2] == {"role": "assistant", "content": "Earlier answer."}
    assert resumed["messages"][-1]["role"] == "assistant"
    assert resumed["messages"][-1]["content"] == "Once upon a time"
    assert resumed.get("continue_final_message") is True


@pytest.mark.parametrize("run", [_plain, _tool_loop], ids = ["plain", "tool-loop"])
@pytest.mark.parametrize(
    "attributes, continue_final_message",
    [({}, False), ({"_assistant_prefill_enabled": False}, True)],
    ids = ["server-prefills", "explicit-continue"],
)
def test_a_continued_turn_still_merges(monkeypatch, run, attributes, continue_final_message):
    resumed = run(monkeypatch, continue_final_message = continue_final_message, **attributes)
    assert resumed["messages"][-1]["role"] == "assistant"
    assert resumed["messages"][-1]["content"] == "Earlier answer.Once upon a time"
    assert resumed["messages"][-2]["role"] == "user"


def test_a_thought_resumes_as_its_own_turn_when_the_server_does_not_prefill():
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._assistant_prefill_enabled = False
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "old thought"},
    ]
    checkpoint = preemption.StreamCheckpoint(reasoning_text = "new thought")
    assert backend._assemble_preempt_resume(
        conversation, checkpoint, "", "new thought", continue_final_message = False
    )
    assert conversation[-2]["reasoning_content"] == "old thought"
    assert conversation[-1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "new thought",
    }


@pytest.mark.parametrize(
    "args, env, expected",
    [
        ([], {}, True),
        (["--no-prefill-assistant"], {}, False),
        (["--no_prefill_assistant"], {}, False),
        ([], {"LLAMA_ARG_PREFILL_ASSISTANT": "0"}, False),
        ([], {"LLAMA_ARG_PREFILL_ASSISTANT": "off"}, False),
        (["--prefill-assistant"], {"LLAMA_ARG_PREFILL_ASSISTANT": "0"}, True),
        (["--prefill-assistant", "--no-prefill-assistant"], {}, False),
    ],
)
def test_the_server_prefill_setting_is_read_from_the_launch(args, env, expected):
    assert llama_cpp._assistant_prefill_from_args(args, env) is expected

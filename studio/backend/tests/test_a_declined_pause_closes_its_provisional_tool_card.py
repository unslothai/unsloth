# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause declined while a tool call was still streaming closes the card that call opened, as a
granted pause does: the final pass and an ended turn both leave no card spinning."""

import json

import pytest

from core.inference import llama_preemption as p

from .preempt_fakes import (
    DecliningPolicy,
    PreemptRecorder,
    delta,
    done,
    finish,
    run_tool_loop,
    web_search_tool,
)


def _unfinished_call() -> str:
    call = {
        "index": 0,
        "id": "call_search",
        "type": "function",
        # Long enough for the early card, which small-argument calls never open.
        "function": {"name": "web_search", "arguments": '{"query": "' + "kernel " * 60},
    }
    return (
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}) + "\n"
    )


def _cards(events):
    started = [
        e["tool_call_id"] for e in events if isinstance(e, dict) and e.get("type") == "tool_start"
    ]
    ended = [
        e["tool_call_id"] for e in events if isinstance(e, dict) and e.get("type") == "tool_end"
    ]
    return started, ended


@pytest.mark.parametrize("max_tokens", [100, 1], ids = ["final-pass", "cap-spent"])
def test_a_declined_pause_ends_the_card_its_unfinished_call_opened(monkeypatch, max_tokens):
    monkeypatch.setattr(p, "DEFAULT_MAX_STALLED_RESUMES", 0)
    signal = p.PreemptSignal()
    rec = PreemptRecorder(
        monkeypatch,
        [
            [_unfinished_call(), finish(), done()],
            [delta("The answer."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 1},
    )
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = DecliningPolicy(),
        tools = [web_search_tool()],
        max_tool_iterations = 5,
        max_tokens = max_tokens,
        permission_mode = "off",
    )
    assert not any(
        isinstance(e, dict) and e.get("type") == "preempt" and e.get("state") == "paused"
        for e in events
    ), "the pause was declined"
    assert len(rec.payloads) == (2 if max_tokens > 1 else 1)
    started, ended = _cards(events)
    assert started == ["call_search"], started
    assert set(ended) == set(started), (started, ended)

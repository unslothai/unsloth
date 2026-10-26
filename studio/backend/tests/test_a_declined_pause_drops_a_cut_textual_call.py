# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause declined while a textual tool call was being written hands the final pass only the
prose before it, as a granted resume does: the cut markup is neither replayed nor displayed."""

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


def test_a_declined_pause_replays_and_shows_only_the_prose_before_the_call(monkeypatch):
    monkeypatch.setattr(p, "DEFAULT_MAX_STALLED_RESUMES", 0)
    signal = p.PreemptSignal()
    rec = PreemptRecorder(
        monkeypatch,
        [
            [
                delta("Hi. "),
                delta('<tool_call>{"name": "web_search", "arguments": {"query": "linux'),
                delta(' kernel"}}'),
                finish(),
                done(),
            ],
            [delta("The answer."), finish(), done()],
        ],
        signal = signal,
        pause_after = {0: 2},
    )
    events = run_tool_loop(
        rec.backend,
        signal = signal,
        policy = DecliningPolicy(),
        tools = [web_search_tool()],
        max_tool_iterations = 5,
        max_tokens = 100,
        permission_mode = "off",
    )
    final = rec.payloads[-1]
    assert final.get("continue_final_message") is True
    assert final["messages"][-1] == {"role": "assistant", "content": "Hi. "}
    snapshots = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "content"]
    assert snapshots[-1] == "Hi. The answer."
    assert all("<tool_call>" not in text for text in snapshots)

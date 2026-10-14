# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A conversation with no saved thread keeps its cut between requests, so the prompt prefix is reused."""

from __future__ import annotations

import pytest

from core.inference import checkpoint, llama_cpp, prefix_boundary
from routes import inference as routes


@pytest.fixture(autouse = True)
def _fresh():
    prefix_boundary.clear()
    yield
    prefix_boundary.clear()


def count(messages):
    return sum(max(1, len(str(m.get("content", ""))) // 4) for m in messages)


def _transcript(turns):
    messages = [{"role": "system", "content": "you are a coding agent"}]
    for index in range(turns):
        messages += [
            {"role": "user", "content": f"Step {index}: " + "do the next thing " * 20},
            {"role": "assistant", "content": f"Done with step {index}. " + "details " * 30},
        ]
    return messages + [{"role": "user", "content": "continue"}]


def _fit(messages):
    return llama_cpp._fit_context(messages, context_length = 4096, max_tokens = 512, count_tokens = count)


def _first_kept(fitted):
    return next(m["content"] for m in fitted if m["role"] != "system")


def test_an_api_conversation_keeps_its_cut_while_it_grows(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    starts = []
    for turns in range(30, 46):
        fitted, truncation = _fit(_transcript(turns))
        assert truncation and truncation["fits"]
        starts.append(_first_kept(fitted))

    # Without the memory every request moved the start (16 requests, 16 starts).
    assert len(set(starts)) <= 4
    # And it only moves forward: each cut is a later prefix of the same history.
    assert [starts.index(s) for s in starts] == sorted(starts.index(s) for s in starts)


def test_without_the_memory_the_cut_moves_every_request(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    monkeypatch.setattr(prefix_boundary, "recall", lambda messages, scope = None: 0)
    starts = {_first_kept(_fit(_transcript(turns))[0]) for turns in range(30, 46)}

    assert len(starts) > 8


def test_a_conversation_sharing_only_the_opening_does_not_borrow_the_cut():
    messages = _transcript(40)
    prefix_boundary.remember(messages, 20)
    other = _transcript(40)
    # Same opening, a different turn where the cut fell.
    other[20] = {**other[20], "content": "a different turn"}

    assert prefix_boundary.recall(messages) == 20
    assert prefix_boundary.recall(other) == 0


def test_an_edited_history_forgets_the_cut():
    messages = _transcript(40)
    prefix_boundary.remember(messages, 20)

    assert prefix_boundary.recall(messages[:15] + messages[16:]) == 0


def test_a_thread_with_its_own_boundary_is_not_touched(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    calls = []
    monkeypatch.setattr(prefix_boundary, "recall", lambda *a, **k: calls.append(a) or 0)

    llama_cpp._fit_context(
        _transcript(40),
        context_length = 4096,
        max_tokens = 512,
        count_tokens = count,
        keeps_boundary = True,
    )

    assert calls == []


def test_the_memory_is_bounded():
    for index in range(prefix_boundary._MAX_ENTRIES + 20):
        messages = _transcript(5)
        messages[1] = {**messages[1], "content": f"conversation {index}"}
        prefix_boundary.remember(messages, 4)

    assert len(prefix_boundary._entries) == prefix_boundary._MAX_ENTRIES


def test_the_passthrough_replays_the_cut_before_sending():
    messages = _transcript(40)
    body = {"model": "m", "messages": list(messages)}
    routes._remember_passthrough_cut(messages, body, "truncate_oldest", 30)

    later = {"model": "m", "messages": _transcript(41)}
    dropped = routes._replay_passthrough_cut(later, "truncate_oldest")

    assert dropped >= 30
    assert later["messages"][0]["role"] == "system"
    assert len(later["messages"]) == len(_transcript(41)) - dropped


def test_the_passthrough_leaves_other_policies_alone():
    messages = _transcript(40)
    body = {"model": "m", "messages": list(messages)}
    routes._remember_passthrough_cut(messages, body, "truncate_oldest", 30)

    assert (
        routes._replay_passthrough_cut(
            {"model": "m", "messages": _transcript(41)}, "truncate_middle"
        )
        == 0
    )
    assert routes._replay_passthrough_cut({"model": "m", "messages": _transcript(41)}, None) == 0

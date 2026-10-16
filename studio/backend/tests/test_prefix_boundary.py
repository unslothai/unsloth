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


def _remember(
    messages,
    dropped,
    scope = None,
):
    kept = [messages[0], *messages[1 + dropped :]]
    prefix_boundary.remember(messages, kept, dropped, scope)


SCOPE = ("passthrough", "org/model", 4096)


def _cut_body(messages, dropped):
    return {"messages": [messages[0], *messages[1 + dropped :]]}


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
    assert [starts.index(s) for s in starts] == sorted(starts.index(s) for s in starts)


def test_without_the_memory_the_cut_moves_every_request(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    monkeypatch.setattr(prefix_boundary, "recall", lambda messages, scope = None: 0)
    starts = {_first_kept(_fit(_transcript(turns))[0]) for turns in range(30, 46)}

    assert len(starts) > 8


def test_a_conversation_sharing_only_the_opening_does_not_borrow_the_cut():
    messages = _transcript(40)
    _remember(messages, 20)
    other = _transcript(40)
    other[20] = {**other[20], "content": "a different turn"}

    assert prefix_boundary.recall(messages) == 20
    assert prefix_boundary.recall(other) == 0


def test_an_edited_history_forgets_the_cut():
    messages = _transcript(40)
    _remember(messages, 20)

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
        _remember(messages, 4)

    assert len(prefix_boundary._entries) == prefix_boundary._MAX_ENTRIES


def test_the_passthrough_replays_the_cut_before_sending():
    messages = _transcript(40)
    routes._remember_passthrough_cut(
        messages, _cut_body(messages, 30), "truncate_oldest", 30, SCOPE
    )

    later = {"model": "m", "messages": _transcript(41)}
    dropped = routes._replay_passthrough_cut(later, "truncate_oldest", 4096, SCOPE)

    assert dropped >= 30
    assert later["messages"][0]["role"] == "system"
    assert len(later["messages"]) == len(_transcript(41)) - dropped


def test_the_passthrough_leaves_other_policies_alone():
    messages = _transcript(40)
    routes._remember_passthrough_cut(
        messages, _cut_body(messages, 30), "truncate_oldest", 30, SCOPE
    )

    assert (
        routes._replay_passthrough_cut(
            {"model": "m", "messages": _transcript(41)}, "truncate_middle", 4096, SCOPE
        )
        == 0
    )
    assert (
        routes._replay_passthrough_cut(
            {"model": "m", "messages": _transcript(41)}, None, 4096, SCOPE
        )
        == 0
    )


def test_the_passthrough_sends_a_conversation_that_now_fits_whole():
    messages = _transcript(40)
    routes._remember_passthrough_cut(
        messages, _cut_body(messages, 30), "truncate_oldest", 30, SCOPE
    )
    later = {"model": "m", "messages": _transcript(41)}

    assert routes._replay_passthrough_cut(later, "truncate_oldest", 131072, SCOPE) == 0
    assert len(later["messages"]) == len(_transcript(41))


def test_a_conversation_differing_anywhere_in_the_dropped_part_does_not_borrow_the_cut():
    messages = _transcript(40)
    _remember(messages, 20)
    other = _transcript(40)
    # Same opening and the same turn right before the cut; one turn in between differs.
    other[10] = {**other[10], "content": "another user's turn"}

    assert prefix_boundary.recall(other) == 0


def test_a_kept_protected_turn_inside_the_cut_is_covered_by_the_check():
    messages = _transcript(40)
    # Turn 3 was pinned and kept; the 20 removed turns run past it.
    kept = [messages[0], messages[3], *messages[22:]]
    prefix_boundary.remember(messages, kept, 20)
    other = _transcript(40)
    other[21] = {**other[21], "content": "a turn the cut removed, different here"}

    assert prefix_boundary.recall(messages) == 20
    assert prefix_boundary.recall(other) == 0


def test_tool_schemas_count_toward_the_overflow_check():
    messages = _transcript(12)
    routes._remember_passthrough_cut(messages, _cut_body(messages, 6), "truncate_oldest", 6, SCOPE)
    body = {"messages": _transcript(13)}
    assert routes._replay_passthrough_cut(dict(body), "truncate_oldest", 4096, SCOPE) == 0

    tools = [
        {"type": "function", "function": {"name": f"t{i}", "description": "d" * 400}}
        for i in range(20)
    ]
    assert (
        routes._replay_passthrough_cut({**body, "tools": tools}, "truncate_oldest", 4096, SCOPE)
        >= 6
    )


def test_a_cut_measured_on_another_model_is_not_replayed():
    messages = _transcript(40)
    routes._remember_passthrough_cut(
        messages, _cut_body(messages, 30), "truncate_oldest", 30, SCOPE
    )
    later = {"messages": _transcript(41)}

    assert (
        routes._replay_passthrough_cut(
            later, "truncate_oldest", 4096, ("passthrough", "other/model", 4096)
        )
        == 0
    )


def test_a_changed_tool_catalogue_gets_its_own_cut():
    class Backend:
        model_identifier = "org/model"
        context_length = 4096

    tools = [{"type": "function", "function": {"name": "read_file"}}]
    with_tools = routes._passthrough_cut_scope(Backend, {"tools": tools})

    assert with_tools != routes._passthrough_cut_scope(Backend, {"tools": []})
    assert with_tools == routes._passthrough_cut_scope(Backend, {"tools": list(reversed(tools))})


def test_an_image_does_not_count_as_prompt_text_in_the_overflow_check():
    messages = _transcript(12)
    routes._remember_passthrough_cut(messages, _cut_body(messages, 6), "truncate_oldest", 6, SCOPE)
    later = _transcript(13)
    later[-1] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this picture?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400_000}},
        ],
    }

    assert routes._replay_passthrough_cut({"messages": later}, "truncate_oldest", 4096, SCOPE) == 0


def test_the_cut_scope_reads_a_backend_without_identity_attributes():
    # The passthrough handlers are driven with backend objects that carry neither attribute; the rest of
    # routes/inference.py reads both through getattr, and a bare attribute read here failed every one.
    class Bare:
        pass

    scope = routes._passthrough_cut_scope(Bare(), {"tools": []})
    assert scope == ("passthrough", None, None, ())
    assert routes._replay_passthrough_cut({"messages": []}, "truncate_oldest", None, scope) == 0

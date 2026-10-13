# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/compact: a checkpoint the user asks for, on a chat that still fits."""

from __future__ import annotations

import sys
import types

from core.inference import checkpoint, llama_cpp
from core.inference.checkpoint import fit_checkpoint_context


def count(messages):
    return sum(max(1, len(str(m.get("content", ""))) // 4) for m in messages)


def _thread(pad = 4, chars = 200):
    messages = [{"role": "system", "content": "you are helpful"}]
    for index in range(pad):
        messages += [
            {"role": "user", "content": f"Section {index}. " + "x" * chars},
            {"role": "assistant", "content": f"Section {index} noted."},
        ]
    return messages + [{"role": "user", "content": "continue"}]


def _fit(messages, **kwargs):
    # A window the thread fits in many times over: nothing here overflows.
    kwargs.setdefault("context_length", 100_000)
    kwargs.setdefault("max_tokens", 512)
    kwargs.setdefault("count_tokens", count)
    kwargs.setdefault("can_reset", True)
    return fit_checkpoint_context(messages, **kwargs)


def test_a_chat_that_fits_is_untouched_unless_asked():
    messages = _thread()

    assert _fit(messages) == (messages, None)


def test_compact_now_resets_a_chat_that_fits():
    fitted, truncation = _fit(_thread(), force_reset = True)

    assert [m["role"] for m in fitted] == ["system", "user"]
    assert fitted[-1]["content"] == "continue"
    assert truncation["fits"] is True
    assert truncation["checkpoint_started"] is True
    assert truncation["dropped_messages"] == 8
    assert truncation["manual"] is True


def test_compact_now_never_resets_where_the_turns_could_not_be_archived():
    messages = _thread()

    assert _fit(messages, force_reset = True, can_reset = False) == (messages, None)


def test_compact_now_with_nothing_older_is_a_no_op():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]

    assert _fit(messages, force_reset = True) == (messages, None)


def test_a_manual_epoch_keeps_replaying_while_the_chat_still_fits():
    _, first = _fit(_thread(), force_reset = True)
    later = _thread() + [
        {"role": "assistant", "content": "Carrying on."},
        {"role": "user", "content": "and the next part"},
    ]

    fitted, truncation = _fit(later, sticky_dropped = first["dropped_messages"], sticky_manual = True)

    assert truncation["dropped_messages"] == first["dropped_messages"]
    assert truncation["checkpoint_started"] is False
    assert truncation["manual"] is True
    assert [m["content"] for m in fitted[1:]] == [
        "continue",
        "Carrying on.",
        "and the next part",
    ]
    # An overflow-driven boundary on the same chat still un-compacts once it fits again.
    assert _fit(later, sticky_dropped = first["dropped_messages"]) == (later, None)


def test_an_overflow_reset_after_a_manual_epoch_is_an_ordinary_one():
    _, first = _fit(_thread(), force_reset = True)
    later = _thread(pad = 12, chars = 600)

    _, truncation = _fit(
        later,
        context_length = 1200,
        max_tokens = 200,
        sticky_dropped = first["dropped_messages"],
        sticky_manual = True,
    )

    assert truncation["checkpoint_started"] is True
    # False rather than absent, so it replaces the reply's earlier manual fit when merged.
    assert truncation["manual"] is False


def test_the_rolling_fit_ignores_compact_now(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    messages = _thread()

    fitted, truncation = llama_cpp._fit_context(
        messages,
        context_length = 100_000,
        max_tokens = 512,
        count_tokens = count,
        can_reset = True,
        force_reset = True,
        sticky_manual = True,
    )

    assert (fitted, truncation) == (messages, None)


def test_fit_context_hands_compact_now_to_the_checkpoint_fit(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "checkpoint")
    monkeypatch.setattr(llama_cpp, "_archive_is_degraded", lambda: False)

    fitted, truncation = llama_cpp._fit_context(
        _thread(),
        context_length = 100_000,
        max_tokens = 512,
        count_tokens = count,
        can_reset = True,
        force_reset = True,
    )

    assert [m["role"] for m in fitted] == ["system", "user"]
    assert truncation["manual"] is True


def _stub_studio_db(monkeypatch, rows):
    module = types.SimpleNamespace(list_chat_messages = lambda thread_id: rows)
    package = types.ModuleType("storage")
    package.studio_db = module
    monkeypatch.setitem(sys.modules, "storage", package)
    monkeypatch.setitem(sys.modules, "storage.studio_db", module)


def _rows(**truncation):
    return [
        {"id": "u1", "parentId": None, "role": "user", "content": "Do the work."},
        {
            "id": "a1",
            "parentId": "u1",
            "role": "assistant",
            "content": "Done.",
            "metadata": {
                "custom": {
                    "contextTruncation": {
                        "fits": True,
                        "checkpoint": True,
                        "boundary_messages": 6,
                        **truncation,
                    }
                }
            },
        },
    ]


def test_the_saved_boundary_says_whether_compact_now_started_it(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "checkpoint")
    branch = [
        {"role": "user", "content": "Do the work."},
        {"role": "assistant", "content": "Done."},
    ]

    _stub_studio_db(monkeypatch, _rows(manual = True))
    assert llama_cpp._sticky_compaction_state("t1", branch) == (6, True)
    assert llama_cpp._sticky_compaction_manual("t1", branch) is True

    _stub_studio_db(monkeypatch, _rows())
    assert llama_cpp._sticky_compaction_manual("t1", branch) is False

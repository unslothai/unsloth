# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An attached file larger than the window: keep its start, point at the sandbox copy."""

from __future__ import annotations

from core.inference import checkpoint, llama_cpp
from core.inference.context_window import fit_oversized_attachments

PATH = '".unsloth_attachments/abc123/notes.txt"'
NOTE = (
    "[notes.txt: its text is below, so answer from it. For calculations, the python tool has the "
    f"file at path = {PATH}]\n"
)
FILE = "".join(f"line {i}: the quick brown fox jumps over the lazy dog\n" for i in range(1, 2001))


def count(messages):
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        total += max(1, len(str(content or "")) // 4)
    return total


def _messages(attachment = FILE, note = NOTE):
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "What does line 1000 say?\n" + note + attachment},
    ]


def _fit(messages, context_length = 2048):
    return fit_oversized_attachments(
        messages, context_length = context_length, max_tokens = 512, count_tokens = count
    )


def test_an_attachment_too_big_for_the_window_keeps_its_start_and_names_the_copy():
    fitted = _fit(_messages())
    text = fitted[-1]["content"]

    assert text.startswith("What does line 1000 say?\n" + NOTE + "line 1: ")
    assert "line 2000:" not in text
    assert f"only its first" in text and "of 2000 lines are above" in text
    assert f"path = {PATH}" in text.split(NOTE, 1)[1]
    assert count([fitted[0], fitted[-1]]) <= 2048 - 512
    # Earlier turns are the fit's business, not this one's.
    assert fitted[:-1] == _messages()[:-1]


def test_cutting_twice_changes_nothing():
    once = _fit(_messages())

    assert _fit(once) == once


def test_an_attachment_that_fits_is_untouched():
    messages = _messages(attachment = "line 1: short\n")

    assert _fit(messages) is messages


def test_an_attachment_with_no_sandbox_copy_is_left_to_the_refusal():
    messages = _messages(note = "")

    assert _fit(messages) is messages


def test_a_turn_too_big_without_its_attachment_is_left_to_the_refusal():
    messages = _messages()
    messages[-1]["content"] = "x" * 20_000 + "\n" + NOTE + FILE

    assert _fit(messages) is messages


def test_content_parts_are_cut_in_place():
    messages = _messages()
    messages[-1] = {
        "role": "user",
        "content": [
            {"type": "text", "text": messages[-1]["content"]},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }

    fitted = _fit(messages)

    assert "of 2000 lines are above" in fitted[-1]["content"][0]["text"]
    assert fitted[-1]["content"][1] == messages[-1]["content"][1]


def test_the_compaction_fit_serves_the_turn_instead_of_refusing_it(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")

    fitted, truncation = llama_cpp._fit_context(
        _messages(), context_length = 2048, max_tokens = 512, count_tokens = count
    )

    assert "of 2000 lines are above" in fitted[-1]["content"]
    assert truncation is None or truncation.get("fits") is not False

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An attached file larger than the window: keep its start, point at the sandbox copy."""

from __future__ import annotations

import copy

from core.inference import checkpoint, llama_cpp
from core.inference.context_window import evicted_messages, fit_oversized_attachments

PATH = '".unsloth_attachments/abc123/notes.txt"'


def note(name = "notes.txt", path = PATH):
    return (
        f"[{name}: its text is below, so answer from it. For calculations, the python tool has the "
        f"file at path = {path}]\n"
    )


def attachment(text, name = "notes.txt"):
    return f"<attachment name={name}>\n{text}\n</attachment>"


LINES = "".join(
    f"line {i}: the quick brown fox jumps over the lazy dog\n" for i in range(1, 2001)
).rstrip("\n")
FILE = attachment(LINES)


def count(messages):
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        total += max(1, len(str(content or "")) // 4)
    return total


def _messages(body = FILE, marker = None):
    marker = note() if marker is None else marker
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "What does line 1000 say?\n" + marker + body},
    ]


def _fit(messages, context_length = 2048):
    return fit_oversized_attachments(
        messages, context_length = context_length, max_tokens = 512, count_tokens = count
    )


def test_an_attachment_too_big_for_the_window_keeps_its_start_and_names_the_copy():
    fitted = _fit(_messages())
    text = fitted[-1]["content"]

    assert text.startswith(
        "What does line 1000 say?\n" + note() + "<attachment name=notes.txt>\nline 1: "
    )
    assert "line 2000:" not in text
    # The markup stays balanced, and the wrapper is not counted as lines of the file.
    assert text.count("<attachment name=") == text.count("</attachment>") == 1
    excerpt = text.split("<attachment name=notes.txt>\n", 1)[1].split("\n</attachment>", 1)[0]
    shown = len(excerpt.split("\n"))
    assert f"only its first {shown} of 2000 lines are above" in text
    assert text.index("</attachment>") < text.index("is too long for the context window")
    assert f"path = {PATH}" in text.split(note(), 1)[1]
    assert count([fitted[0], fitted[-1]]) <= 2048 - 512
    assert fitted[:-1] == _messages()[:-1]


def test_the_newest_turn_keeps_its_identity_so_it_is_not_archived_as_evicted():
    messages = _messages()
    before = list(messages)

    fitted = _fit(messages)

    assert evicted_messages(before, fitted) == []


def test_cutting_twice_changes_nothing():
    once = copy.deepcopy(_fit(_messages()))

    assert _fit(copy.deepcopy(once)) == once


def test_an_attachment_that_fits_is_untouched():
    messages = _messages(body = attachment("line 1: short"))

    assert _fit(copy.deepcopy(messages)) == messages


def test_an_attachment_with_no_sandbox_copy_is_left_to_the_refusal():
    messages = _messages(marker = "")

    assert _fit(copy.deepcopy(messages)) == messages


def test_a_turn_too_big_without_its_attachment_is_left_to_the_refusal():
    messages = _messages()
    messages[-1]["content"] = "x" * 20_000 + "\n" + note() + FILE

    assert _fit(copy.deepcopy(messages)) == messages


def test_a_name_with_brackets_is_still_recognised():
    name = "report [final].txt"
    messages = _messages(body = attachment(LINES, name), marker = note(name))

    assert "of 2000 lines are above" in _fit(messages)[-1]["content"]


def test_content_parts_are_cut_in_place():
    messages = _messages()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages[-1] = {
        "role": "user",
        "content": [{"type": "text", "text": messages[-1]["content"]}, image],
    }

    fitted = _fit(messages)

    assert "of 2000 lines are above" in fitted[-1]["content"][0]["text"]
    assert fitted[-1]["content"][1] == image


def test_the_compaction_fit_serves_the_turn_instead_of_refusing_it(monkeypatch):
    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")

    fitted, truncation = llama_cpp._fit_context(
        _messages(), context_length = 2048, max_tokens = 512, count_tokens = count
    )

    assert "of 2000 lines are above" in fitted[-1]["content"]
    assert truncation is None or truncation.get("fits") is not False


def _follow_up(first = None):
    first = first or _messages()
    return first + [
        {
            "role": "assistant",
            "content": "Line 1000 reads: line 1000: the quick brown fox jumps over the lazy dog",
        },
        {"role": "user", "content": "And line 1500?"},
    ]


def test_an_earlier_oversized_attachment_keeps_only_its_path():
    fitted = _fit(_follow_up())
    earlier = fitted[3]["content"]

    assert earlier == (
        "What does line 1000 say?\n[notes.txt (2000 lines) is too long for the context window to show. "
        f"The whole file is at path = {PATH} in the python tool's working directory, so read the part you "
        "need with python.]"
    )
    assert fitted[-1] == _follow_up()[-1]
    assert count(fitted) <= 2048 - 512


def test_a_small_file_beside_an_earlier_oversized_one_stays_whole():
    small_path = '".unsloth_attachments/def456/small.txt"'
    small = note("small.txt", small_path) + attachment("one short line", "small.txt")
    first = _messages()
    first[-1]["content"] += "\n" + small

    earlier = _fit(_follow_up(first))[3]["content"]

    assert "(2000 lines) is too long" in earlier
    assert small in earlier


def test_an_earlier_turn_renders_the_same_on_every_later_request():
    later = _follow_up() + [
        {
            "role": "assistant",
            "content": "Line 1500 reads: line 1500: the quick brown fox jumps over the lazy dog",
        },
        {"role": "user", "content": "And line 2000?"},
    ]

    assert _fit(later)[3] == _fit(_follow_up())[3]


def test_a_conversation_that_fits_skips_the_attachment_checks():
    calls = []

    def counting(messages):
        calls.append(len(messages))
        return count(messages)

    messages = _messages(body = attachment("line 1: short"))
    fit_oversized_attachments(messages, context_length = 2048, max_tokens = 512, count_tokens = counting)

    assert calls == [len(messages)]


def test_a_long_paste_is_cut_inside_its_own_wrapper():
    pasted = f"<pasted_text name=paste.txt bytes=110000>\n{LINES}\n</pasted_text>"
    text = _fit(_messages(body = pasted, marker = note("paste.txt")))[-1]["content"]

    assert text.count("<pasted_text name=") == text.count("</pasted_text>") == 1
    excerpt = text.split("bytes=110000>\n", 1)[1].split("\n</pasted_text>", 1)[0]
    assert f"only its first {len(excerpt.split(chr(10)))} of 2000 lines are above" in text


def test_document_annotations_are_cut_inside_their_wrapper():
    wrapped = f'<document_annotations file="notes.pdf">\n{LINES}\n</document_annotations>'
    text = _fit(_messages(body = wrapped))[-1]["content"]

    assert text.count("<document_annotations") == text.count("</document_annotations>") == 1


def test_a_file_that_fits_the_budget_is_not_trimmed_by_the_margin():
    # Between 87.5% and 100% of the prompt budget (1536): fits as sent.
    lines = "".join(f"row {i:04d} ok\n" for i in range(1, 450)).rstrip("\n")
    messages = _messages(body = attachment(lines))
    assert 0.875 * 1536 < count(messages) <= 1536

    assert _fit(copy.deepcopy(messages)) == messages


def test_a_path_only_turn_that_fits_the_budget_is_served():
    # The system prompt fills 90% of the prompt budget (1536): no room for an excerpt, room for the path.
    messages = _messages()
    messages[0] = {"role": "system", "content": "s" * (4 * 1380)}

    text = _fit(messages)[-1]["content"]

    assert "line 1: " not in text
    assert "(2000 lines) is too long for the context window to show" in text
    assert count([messages[0], messages[-1]]) <= 1536

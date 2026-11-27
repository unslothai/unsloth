# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sandbox note scan finds what the note regex finds, without retrying from every '[' of a long line."""

from __future__ import annotations

import json
import random
import time

import pytest

from core.inference.context_window import (
    _SANDBOX_ATTACHMENT_NOTE_KEY,
    _SANDBOX_ATTACHMENT_NOTE_RE,
    _attachment_spans,
    _sandbox_attachment_notes,
    fit_oversized_attachments,
)


def note(
    name = "a.json",
    path = ".unsloth_attachments/abc/a.json",
    reader = "json.load(open(path))",
):
    return (
        f"[{name}: its text is below, so answer from it. For calculations, the python tool has the "
        f"file at path = {json.dumps(path)}{'; ' + reader if reader else ''}]"
    )


def minified_json(chars):
    row = json.dumps({"id": 0, "tags": [["a", 1], [2]]}, separators = (",", ":"))
    return "[" + ",".join([row] * (chars // (len(row) + 1) + 1)) + "]"


def found(text):
    return [(m.span(), m.groupdict()) for m in _sandbox_attachment_notes(text)]


def expected(text):
    return [(m.span(), m.groupdict()) for m in _SANDBOX_ATTACHMENT_NOTE_RE.finditer(text)]


def test_the_key_is_the_literal_the_note_regex_requires():
    assert _SANDBOX_ATTACHMENT_NOTE_RE.match(note() + "\n")
    assert _SANDBOX_ATTACHMENT_NOTE_KEY in note()
    assert (
        _SANDBOX_ATTACHMENT_NOTE_RE.pattern.replace("\\.", ".").count(_SANDBOX_ATTACHMENT_NOTE_KEY)
        == 1
    )


@pytest.mark.parametrize(
    "text",
    [
        note() + "\n",
        note(),  # no newline at the end of the text: never a note
        note() + "\r\n",  # CRLF: the line does not end in ']'
        note() + "\n" + note("b.csv") + "\n",  # adjacent notes
        note()
        + note("b.csv")
        + "\n",  # two notes on one line: one match, the lazy name stops at the first key
        "[[x" + note("[nested] name") + "\n",  # the match starts at the line's first '['
        "x "
        + _SANDBOX_ATTACHMENT_NOTE_KEY
        + " "
        + note()
        + "\n",  # a key before the line's first '['
        note().replace('path = "', 'path = \\"') + "\n",  # JSON-quoted note: no match
        note()[:-1] + "\n" + note() + "\n",  # unterminated first line
        "[" * 50 + "\n" + note("ü€.txt", "ü/ñ") + "\n" + "]\n",
        note(reader = None) + "\n<attachment>\nbody\n</attachment>",
        note(path = 'we"ird\\p') + "\n",
        "",
        "no notes here [1] [2]\n",
    ],
)
def test_the_scan_finds_exactly_the_note_regex_matches(text):
    assert found(text) == expected(text)


def test_random_texts_find_the_same_notes():
    rng = random.Random(0)
    tokens = [
        _SANDBOX_ATTACHMENT_NOTE_KEY,
        "[",
        "]",
        "]\n",
        '"',
        "\\",
        ";",
        "\n",
        "\r\n",
        "a",
        "ü",
        note(),
    ]
    for _ in range(3000):
        text = "".join(rng.choice(tokens) for _ in range(rng.randint(1, 40)))
        assert found(text) == expected(text), text


def test_a_large_minified_json_attachment_is_scanned_in_linear_time():
    body = minified_json(1_000_000)
    text = "Summarize\n" + note() + '\n<attachment name="a.json">\n' + body + "\n</attachment>"
    started = time.perf_counter()
    spans = list(_attachment_spans(text))
    elapsed = time.perf_counter() - started
    assert len(spans) == 1 and spans[0][2] == body
    # The '[' by '[' retry takes minutes here; a linear scan takes milliseconds.
    assert elapsed < 2.0, elapsed


def test_a_minified_json_line_quoting_the_note_stays_fast():
    quoted = json.dumps(
        [{"content": note() + "\n" + minified_json(200)}] * 2000, separators = (",", ":")
    )
    text = note() + "\n<attachment>\n" + quoted + "\n" + note() + note() + "]\n</attachment>"
    started = time.perf_counter()
    notes = found(text)
    elapsed = time.perf_counter() - started
    # The JSON-escaped quotes never match, so only the real note and the last line's first note are found.
    last = text.rindex(note() + note())
    assert [span for span, _ in notes] == [
        (0, len(note()) + 1),
        (last, len(text) - len("</attachment>")),
    ]
    assert elapsed < 2.0, elapsed


def test_fitting_a_large_minified_json_attachment_stays_fast():
    body = minified_json(1_000_000)
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": note() + '\n<attachment name="a.json">\n' + body + "\n</attachment>",
        },
    ]
    started = time.perf_counter()
    fitted = fit_oversized_attachments(
        messages,
        context_length = 32768,
        max_tokens = 1024,
        count_tokens = lambda ms: sum(len(str(m.get("content"))) // 4 for m in ms),
    )
    assert "is too long for the context window" in fitted[-1]["content"]
    assert time.perf_counter() - started < 10.0

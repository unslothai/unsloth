# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth chat` parses its Markdown when a frame is drawn, not once per streamed chunk."""

from __future__ import annotations

import io

import rich.markdown
from rich.console import Console

import unsloth_cli._inference as inference
from unsloth_cli._inference import stream_markdown

PARA = "Some **bold** text with `code` and a list:\n\n- one\n- two\n\n"


def _chunks(text, step = 3):
    return [text[:n] for n in range(step, len(text) + step, step)]


def _render(chunks, show_thinking, *, terminal):
    buf = io.StringIO()
    console = Console(file = buf, width = 80, force_terminal = terminal, color_system = None)
    raw = stream_markdown(iter([*chunks, {"meta": 1}]), show_thinking, console = console)
    return raw, buf.getvalue()


def test_parses_per_frame_not_per_chunk(monkeypatch):
    built, filtered = [], []
    real_init = rich.markdown.Markdown.__init__
    real_visible = inference.visible_text

    def _counting(self, *args, **kwargs):
        built.append(1)
        real_init(self, *args, **kwargs)

    def _counting_visible(*args, **kwargs):
        filtered.append(1)
        return real_visible(*args, **kwargs)

    monkeypatch.setattr(rich.markdown.Markdown, "__init__", _counting)
    monkeypatch.setattr(inference, "visible_text", _counting_visible)
    text = PARA * 60
    chunks = _chunks(text)
    raw, _ = _render(chunks, False, terminal = False)
    assert raw == text
    assert len(chunks) > 1000
    assert len(built) < 50
    assert len(filtered) < 50


def test_final_output_matches_the_full_reply():
    for text, show in [
        (PARA * 5, False),
        ("<think>hidden reasoning</think>" + PARA, False),
        ("<think>shown reasoning</think>" + PARA, True),
        ("partial <thi", False),
        ("   ", False),
    ]:
        raw, out = _render(_chunks(text), show, terminal = False)
        assert raw == text
        assert out == _render([text], show, terminal = False)[1]


def test_terminal_stream_renders():
    raw, out = _render(_chunks(PARA * 20), False, terminal = True)
    assert raw == PARA * 20
    assert "bold" in out

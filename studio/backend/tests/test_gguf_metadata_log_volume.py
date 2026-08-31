# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF header block says the same things about the same file every time.

Ten call sites reach ``_read_gguf_metadata`` and one ``POST /api/inference/estimate-memory``
walks the header three to five times: 140 of 296 log lines over a 20s four-tab session.
These tests pin both halves of the fix: repeats are demoted, and demoting them does not
change what is detected.
"""

from __future__ import annotations

import logging

import pytest

from core.inference import llama_cpp


@pytest.fixture(autouse = True)
def _clear_seen():
    llama_cpp._GGUF_METADATA_LOGGED.clear()
    yield
    llama_cpp._GGUF_METADATA_LOGGED.clear()


def _gguf(
    tmp_path,
    name = "m.gguf",
    body = b"x" * 64,
):
    p = tmp_path / name
    p.write_bytes(body)
    return str(p)


def test_only_the_first_read_of_a_file_speaks_up(tmp_path):
    path = _gguf(tmp_path)
    assert llama_cpp._note_gguf_metadata_read(path) is True
    for _ in range(10):
        assert llama_cpp._note_gguf_metadata_read(path) is False


def test_a_changed_file_is_described_again(tmp_path):
    """Keyed on identity, not name: a rebuilt GGUF at the same path is worth stating."""
    path = _gguf(tmp_path)
    assert llama_cpp._note_gguf_metadata_read(path) is True
    import os
    import time

    time.sleep(0.01)
    with open(path, "wb") as fh:
        fh.write(b"y" * 128)
    os.utime(path, None)
    assert llama_cpp._note_gguf_metadata_read(path) is True


def test_an_unstattable_path_still_logs(tmp_path):
    """Fails open: an unidentifiable read is better said out loud than swallowed."""
    assert llama_cpp._note_gguf_metadata_read(str(tmp_path / "gone.gguf")) is True


def test_the_cache_is_bounded(tmp_path):
    for i in range(llama_cpp._GGUF_METADATA_LOGGED_MAX + 5):
        llama_cpp._note_gguf_metadata_read(_gguf(tmp_path, f"m{i}.gguf"))
    assert len(llama_cpp._GGUF_METADATA_LOGGED) <= llama_cpp._GGUF_METADATA_LOGGED_MAX


TEMPLATE = (
    "{% if enable_thinking %}<think>{% endif %}"
    "{% if preserve_thinking %}keep{% endif %}"
    # The literal the tool detector actually looks for, not a paraphrase.
    "{% if tools %}{{ tools }}{% endif %}"
)


def test_debug_level_moves_the_capability_lines_off_info(monkeypatch):
    """Asserted against the logger itself: structlog bypasses caplog, so a caplog test
    would pass while the lines still went to info."""
    seen: list[tuple[str, str]] = []
    for level in ("info", "debug"):
        monkeypatch.setattr(
            llama_cpp.logger,
            level,
            lambda msg, *a, _lv = level, **k: seen.append((_lv, str(msg))),
        )

    llama_cpp.detect_reasoning_flags(TEMPLATE, "qwen3.8", log_source = "GGUF metadata")
    first = list(seen)
    seen.clear()
    llama_cpp.detect_reasoning_flags(
        TEMPLATE, "qwen3.8", log_source = "GGUF metadata", log_level = "debug"
    )
    repeat = list(seen)

    assert [lv for lv, _ in first] == ["info"] * len(
        first
    ) and first, "the first read must still describe the model at info"
    assert repeat, "a repeat must still be logged, just not at info"
    assert [lv for lv, _ in repeat] == ["debug"] * len(repeat)
    # Same facts, different level: nothing is lost, only demoted.
    assert [m for _, m in first] == [m for _, m in repeat]


def test_the_level_does_not_change_what_is_detected():
    """A capability that disappears when the line is demoted would be a regression
    hiding behind a quieter log."""
    loud = llama_cpp.detect_reasoning_flags(TEMPLATE, "qwen3.8", log_source = "GGUF metadata")
    quiet = llama_cpp.detect_reasoning_flags(
        TEMPLATE, "qwen3.8", log_source = "GGUF metadata", log_level = "debug"
    )
    assert loud == quiet
    assert loud["supports_reasoning"] is True
    assert loud["supports_preserve_thinking"] is True
    assert loud["supports_tools"] is True

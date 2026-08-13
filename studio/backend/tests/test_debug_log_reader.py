# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reader has to stay cheap on a multi-GB session log (there is no rotation,
only a startup prune of the newest 20 files) and it must not resend what the
caller already has."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.debug_log_reader import (
    MAX_APPEND_BYTES,
    MAX_TAIL_BYTES,
    read_since,
    read_tail,
)


def _write(
    path: Path,
    count: int,
    prefix: str = "line",
) -> None:
    path.write_text("".join(f"{prefix}{i}\n" for i in range(count)), encoding = "utf-8")


def test_an_empty_file_reads_as_no_lines(tmp_path):
    path = tmp_path / "empty.log"
    path.write_text("")
    result = read_tail(path)
    assert result.lines == []
    assert result.size_bytes == 0


def test_the_tail_is_the_newest_lines(tmp_path):
    path = tmp_path / "a.log"
    _write(path, 1001)
    result = read_tail(path)
    assert len(result.lines) == 1000
    assert result.lines[0] == "line1"
    assert result.lines[-1] == "line1000"


def test_a_short_file_returns_everything(tmp_path):
    path = tmp_path / "a.log"
    _write(path, 999)
    assert len(read_tail(path).lines) == 999


def test_a_last_line_without_a_newline_is_still_shown(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("first\nno trailing newline")
    assert read_tail(path).lines == ["first", "no trailing newline"]


def test_a_huge_file_is_read_in_a_bounded_window(tmp_path, monkeypatch):
    """The whole point: cost must not scale with the file."""
    path = tmp_path / "big.log"
    with open(path, "w", encoding = "utf-8") as handle:
        for i in range(200_000):
            handle.write(f"padding line {i} {'a' * 40}\n")
    assert path.stat().st_size > 8 * MAX_TAIL_BYTES

    read_bytes = {"total": 0}
    real_open = open

    class _Counting:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size = -1):
            chunk = self._handle.read(size)
            read_bytes["total"] += len(chunk)
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    def _counting_open(
        file,
        mode = "r",
        *args,
        **kwargs,
    ):
        handle = real_open(file, mode, *args, **kwargs)
        return _Counting(handle) if "b" in mode else handle

    monkeypatch.setattr("builtins.open", _counting_open)
    result = read_tail(path)
    monkeypatch.undo()

    assert len(result.lines) == 1000
    # One block of slack for the block-aligned backward scan.
    assert read_bytes["total"] <= MAX_TAIL_BYTES + 65_536


def test_only_appended_lines_come_back(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("a\nb\n")
    first = read_tail(path)
    with open(path, "a", encoding = "utf-8") as handle:
        handle.write("c\nd\n")
    second = read_since(path, first.cursor)
    assert second.lines == ["c", "d"]
    assert second.reset is False


def test_an_idle_poll_is_empty_and_not_an_error(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("a\n")
    first = read_tail(path)
    second = read_since(path, first.cursor)
    assert second.lines == []
    assert second.reset is False
    assert second.reset_reason is None


def test_repeated_polls_never_resend(tmp_path):
    """A poll loop that resends the tail every tick would look fine in a single
    assertion and be unusable in the UI, so this walks 200 rounds."""
    path = tmp_path / "a.log"
    path.write_text("0\n")
    cursor = read_tail(path).cursor
    delivered = 0
    for i in range(1, 201):
        with open(path, "a", encoding = "utf-8") as handle:
            handle.write(f"{i}\n")
        result = read_since(path, cursor)
        assert result.reset is False, f"unexpected reset on poll {i}"
        cursor = result.cursor
        delivered += len(result.lines)
    assert delivered == 200


def test_a_half_written_line_is_held_back_then_delivered_once(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("a\n")
    cursor = read_tail(path).cursor
    with open(path, "a", encoding = "utf-8") as handle:
        handle.write("partial")
    held = read_since(path, cursor)
    assert held.lines == []
    with open(path, "a", encoding = "utf-8") as handle:
        handle.write(" line\n")
    completed = read_since(path, held.cursor)
    assert completed.lines == ["partial line"]
    assert read_since(path, completed.cursor).lines == []


def test_a_truncated_file_resets_instead_of_returning_garbage(tmp_path):
    path = tmp_path / "a.log"
    _write(path, 50)
    cursor = read_tail(path).cursor
    path.write_text("fresh\n")
    result = read_since(path, cursor)
    assert result.reset is True
    assert result.reset_reason == "truncated"
    assert result.lines == ["fresh"]


def test_a_writer_outrunning_the_reader_is_bounded(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("x\n")
    cursor = read_tail(path).cursor
    with open(path, "a", encoding = "utf-8") as handle:
        for i in range(60_000):
            handle.write(f"flood {i} {'y' * 34}\n")
    result = read_since(path, cursor)
    assert result.dropped_bytes > 0
    assert len(result.lines) <= 2000


def test_an_unusable_cursor_gives_a_fresh_tail_not_an_error(tmp_path):
    path = tmp_path / "a.log"
    _write(path, 5)
    result = read_since(path, "not-a-real-cursor")
    assert result.reset is True
    assert result.reset_reason == "cursor_stale"
    assert len(result.lines) == 5


def test_undecodable_bytes_do_not_raise(tmp_path):
    path = tmp_path / "a.log"
    path.write_bytes(b"good\n\xff\xfe still readable\n")
    assert len(read_tail(path).lines) == 2


def test_crlf_is_normalised(tmp_path):
    path = tmp_path / "a.log"
    path.write_bytes(b"one\r\ntwo\r\n")
    assert read_tail(path).lines == ["one", "two"]


def test_the_reader_redacts_so_a_caller_cannot_forget(tmp_path):
    path = tmp_path / "a.log"
    path.write_text("using hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 now\n")
    assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345" not in read_tail(path).lines[0]

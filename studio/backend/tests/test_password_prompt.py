# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Masked terminal password prompt (auth/terminal_prompt.py): reader echo and
editing, the change loop's validation/re-prompt behavior, and the pure
should-prompt gate. Drives the reader through a scripted fake getch, so no
tty (and no msvcrt on Linux) is needed."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth import terminal_prompt as tp  # noqa: E402


def _fake_getch(keys):
    """Scripted keystroke source: yields one item per _getch() call. Items may
    be multi-char strings to simulate a paste burst arriving in one read."""
    it = iter(keys)

    def getch():
        return next(it)

    return getch


def _read(
    monkeypatch,
    keys,
    prompt = "P: ",
):
    monkeypatch.setattr(tp, "_getch", _fake_getch(keys))
    out = io.StringIO()
    value = tp._read_password(prompt, out = out)
    return value, out.getvalue()


# ── _read_password ───────────────────────────────────────────────────


def test_reader_echoes_one_star_per_char(monkeypatch):
    value, out = _read(monkeypatch, list("secret") + ["\r"])
    assert value == "secret"
    assert out.count("*") == 6
    assert "secret" not in out


def test_reader_backspace_edits_and_erases_star(monkeypatch):
    value, out = _read(monkeypatch, list("abc") + ["\x7f"] + list("d") + ["\n"])
    assert value == "abd"
    assert "\b \b" in out
    # 4 stars were printed (a, b, c, d); one was erased.
    assert out.count("*") == 4


def test_reader_backspace_on_empty_buffer_is_noop(monkeypatch):
    value, out = _read(monkeypatch, ["\x08", "\x7f"] + list("x") + ["\r"])
    assert value == "x"
    assert "\b \b" not in out


def test_reader_paste_burst_delivers_all_chars(monkeypatch):
    # A paste can arrive as one multi-char read; every char must count.
    value, out = _read(monkeypatch, ["pasted-secret", "\r"])
    assert value == "pasted-secret"
    assert out.count("*") == len("pasted-secret")


def test_reader_unicode_password(monkeypatch):
    value, _ = _read(monkeypatch, list("pässwörd✓") + ["\r"])
    assert value == "pässwörd✓"


def test_reader_ignores_other_control_chars(monkeypatch):
    value, _ = _read(monkeypatch, ["\t", "\x1b"] + list("ok") + ["\r"])
    assert value == "ok"


def test_reader_ctrl_c_raises_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(tp, "_getch", _fake_getch(list("ab") + ["\x03"]))
    with pytest.raises(KeyboardInterrupt):
        tp._read_password("P: ", out = io.StringIO())


def test_reader_ctrl_d_on_empty_raises_eof(monkeypatch):
    monkeypatch.setattr(tp, "_getch", _fake_getch(["\x04"]))
    with pytest.raises(EOFError):
        tp._read_password("P: ", out = io.StringIO())


def test_reader_ctrl_d_mid_input_is_ignored(monkeypatch):
    value, _ = _read(monkeypatch, list("ab") + ["\x04"] + list("c") + ["\r"])
    assert value == "abc"


def test_reader_windows_key_prefix_is_ignored(monkeypatch):
    # _getch_windows reports swallowed function-key sequences as "\x00".
    value, _ = _read(monkeypatch, ["\x00"] + list("w") + ["\r"])
    assert value == "w"


# ── prompt_for_password_change ───────────────────────────────────────


def _run_loop(
    monkeypatch,
    keys,
    *,
    min_length = 8,
    current = "bootstrap-pw",
):
    monkeypatch.setattr(tp, "_getch", _fake_getch(keys))
    out = io.StringIO()
    applied = []
    ok = tp.prompt_for_password_change(
        min_length = min_length,
        is_current_password = lambda pw: pw == current,
        apply_change = applied.append,
        out = out,
    )
    return ok, applied, out.getvalue()


def _keys(*lines):
    keys = []
    for line in lines:
        keys.extend(list(line))
        keys.append("\r")
    return keys


def test_loop_success_applies_once(monkeypatch):
    ok, applied, out = _run_loop(monkeypatch, _keys("new-password", "new-password"))
    assert ok is True
    assert applied == ["new-password"]
    assert "Password updated" in out
    assert "new-password" not in out


def test_loop_short_password_reprompts(monkeypatch):
    ok, applied, out = _run_loop(monkeypatch, _keys("short", "long-enough-pw", "long-enough-pw"))
    assert ok is True
    assert applied == ["long-enough-pw"]
    assert "at least 8 characters" in out


def test_loop_rejects_current_password(monkeypatch):
    ok, applied, out = _run_loop(
        monkeypatch, _keys("bootstrap-pw", "fresh-password", "fresh-password")
    )
    assert ok is True
    assert applied == ["fresh-password"]
    assert "must differ" in out


def test_loop_mismatch_reprompts_then_succeeds(monkeypatch):
    ok, applied, out = _run_loop(
        monkeypatch,
        _keys("first-attempt", "typo-attempt", "second-attempt", "second-attempt"),
    )
    assert ok is True
    assert applied == ["second-attempt"]
    assert "do not match" in out


def test_loop_ctrl_c_aborts_without_applying(monkeypatch):
    ok, applied, out = _run_loop(monkeypatch, list("ab") + ["\x03"])
    assert ok is False
    assert applied == []
    assert "aborted" in out


def test_loop_eof_aborts_without_applying(monkeypatch):
    ok, applied, out = _run_loop(monkeypatch, ["\x04"])
    assert ok is False
    assert applied == []
    assert "aborted" in out


def test_loop_ctrl_c_at_confirmation_aborts(monkeypatch):
    ok, applied, _ = _run_loop(monkeypatch, _keys("valid-password") + ["\x03"])
    assert ok is False
    assert applied == []


def test_loop_min_length_counts_code_points(monkeypatch):
    # 8 unicode code points must pass a min_length of 8.
    pw = "pässwörd"
    assert len(pw) == 8
    ok, applied, _ = _run_loop(monkeypatch, _keys(pw, pw))
    assert ok is True
    assert applied == [pw]


# ── should_prompt_password_change ────────────────────────────────────


@pytest.mark.parametrize(
    "tunnel,requires,stdin_tty,stderr_tty,expected",
    [
        (True, True, True, True, True),
        (False, True, True, True, False),  # tunnel not starting (loopback no-op)
        (True, False, True, True, False),  # password already changed
        (True, True, False, True, False),  # piped stdin (headless)
        (True, True, True, False, False),  # redirected stderr
        (False, False, False, False, False),
    ],
)
def test_should_prompt_matrix(tunnel, requires, stdin_tty, stderr_tty, expected):
    assert (
        tp.should_prompt_password_change(
            tunnel_will_start = tunnel,
            requires_change = requires,
            stdin_isatty = stdin_tty,
            stderr_isatty = stderr_tty,
        )
        is expected
    )


def test_stream_eof_aborts_instead_of_submitting(monkeypatch):
    # A dead stream ("" from _getch, e.g. a closed pty) must abort the line,
    # never silently submit the partial password typed so far.
    import io

    err = io.StringIO()
    monkeypatch.setattr(tp, "_getch", _fake_getch(list("abc") + [""]))
    with pytest.raises(EOFError):
        tp._read_password("New password: ", out = err)

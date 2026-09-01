# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A traceback in the log file has to be readable as a traceback.

~/.unsloth/studio/logs is a tee of stdout and stdout is JSON, so every stack trace reached
the reader as one line with its newlines escaped to ``\\n`` -- as the reported Image
Transform failure did. ``with_readable_traceback`` echoes the traceback under the record;
the JSON line itself must survive byte-for-byte for anything parsing the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loggers import config as log_config  # noqa: E402

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/studio/backend/core/inference/diffusion.py", line 5240, in generate\n'
    "    image_latents = self.vae.encode(image)\n"
    "RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same"
)


def _json_renderer():
    import structlog
    return log_config.with_readable_traceback(structlog.processors.JSONRenderer(sort_keys = False))


def _render(event_dict):
    return _json_renderer()(None, "error", event_dict)


def test_record_without_an_exception_is_a_single_json_line():
    out = _render({"event": "loaded", "level": "info"})
    assert "\n" not in out
    assert json.loads(out)["event"] == "loaded"


def test_traceback_is_echoed_as_real_lines_after_the_record():
    out = _render({"event": "request_failed", "exception": _TRACEBACK})
    first, _, rest = out.partition("\n")
    record = json.loads(first)
    assert record["exception"] == _TRACEBACK
    # Each readable line sits behind a prefix so it cannot read as a record of its own.
    prefix = log_config._TRACEBACK_ECHO_PREFIX
    lines = rest.splitlines()
    assert [line.removeprefix(prefix) for line in lines] == _TRACEBACK.splitlines()
    assert lines[0] == f"{prefix}Traceback (most recent call last):"
    assert lines[-1].startswith(f"{prefix}RuntimeError: Input type (float)")


def test_echo_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_PLAIN_TRACEBACKS", "0")
    out = _render({"event": "request_failed", "exception": _TRACEBACK})
    assert "\n" not in out
    assert json.loads(out)["exception"] == _TRACEBACK


def test_blank_and_non_string_exceptions_are_not_echoed():
    for value in ("", "   \n", None, 17):
        out = _render({"event": "e", "exception": value})
        assert "\n" not in out, value


def test_console_renderer_is_left_alone(monkeypatch):
    # Development already prints tracebacks as tracebacks; wrapping would double them.
    import inspect

    source = inspect.getsource(log_config.LogConfig.setup_logging)
    assert "with_readable_traceback(structlog.processors.JSONRenderer" in source
    assert "with_readable_traceback(structlog.dev.ConsoleRenderer" not in source


def test_echoed_copy_is_the_redacted_truncated_one():
    # Reads event_dict["exception"] AFTER filter_sensitive_data and truncation, so no secret re-enters the log.
    huge = "HEAD" + ("x" * 4_000_000) + "\nValueError: nope"
    capped = log_config.truncate_exception({"exception": huge})["exception"]
    out = _render({"event": "request_failed", "exception": capped})
    assert len(out) < 2 * (log_config._MAX_EXC_CHARS + 500)
    assert out.endswith("ValueError: nope")


def test_a_control_heavy_traceback_cannot_outgrow_the_cap_by_escaping():
    # truncate_exception bounds the FIELD; escaping multiplies it six-fold per C0 control, so 16 KiB left as 98 KiB.
    payload = "HEAD\n" + ("\x1b" * 200) + "\n" + "\n".join("\x00" * 400 for _ in range(400))
    capped = log_config.truncate_exception({"exception": payload + "\nValueError: nope"})[
        "exception"
    ]
    out = _render({"event": "request_failed", "exception": capped})
    _, _, echoed = out.partition("\n")
    assert len(echoed) <= log_config._MAX_EXC_CHARS + 200
    for line in echoed.split("\n"):
        assert line.startswith("| "), line
    assert "lines omitted" in echoed
    assert echoed.endswith("ValueError: nope")


def test_an_uncapped_traceback_is_echoed_whole():
    body = "\n".join(f'  File "f{i}.py", line {i}, in fn' for i in range(20))
    out = _render({"event": "request_failed", "exception": f"Traceback:\n{body}\nValueError: nope"})
    _, _, echoed = out.partition("\n")
    assert "lines omitted" not in echoed
    assert echoed.count("\n") == 21


def test_an_exception_message_cannot_forge_a_log_record():
    # CWE-117: messages carry request text and RFC 8259 skips whitespace, so each echoed line must fail json.loads().
    forged = json.dumps({"level": "info", "event": "admin_login", "user": "attacker"})
    out = _render(
        {
            "event": "request_failed",
            "exception": f"Traceback (most recent call last):\nValueError: bad prompt: \n{forged}",
        }
    )
    head, _, echoed = out.partition("\n")
    json.loads(head)
    for line in echoed.split("\n"):
        assert not line[:1].isspace(), line
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        raise AssertionError(f"echoed line parses as a record: {line!r}")


def test_every_echoed_line_carries_the_prefix_including_exotic_separators():
    # splitlines() also breaks on CR, VT, FF, NEL and U+2028/9, so no line rides in on a separator.
    exception = 'Traceback:\r\n  frame\rValueError: x\u2028{"event": "fake"}'
    echoed = _render({"event": "e", "exception": exception}).split("\n")[1:]
    assert echoed
    assert all(line.startswith(log_config._TRACEBACK_ECHO_PREFIX) for line in echoed)
    assert not any("\r" in line for line in echoed)


def test_a_lone_surrogate_cannot_break_the_log_write():
    # A lone surrogate from json.loads raises UnicodeEncodeError on UTF-8 stdout, replacing the real exception.
    import io

    surrogate = json.loads('"\\ud800"')
    out = _render({"event": "request_failed", "exception": f"ValueError: bad prompt: {surrogate}"})
    assert surrogate not in out
    assert "\\ud800" in out
    stream = io.TextIOWrapper(io.BytesIO(), encoding = "utf-8")
    print(out, file = stream)  # must not raise
    out.encode("utf-8")


def test_terminal_controls_are_neutralised():
    # Raw ESC would let request-derived text rewrite what the reader sees, and a backspace run would rub out the prefix.
    exception = "ValueError: \x1b[2Jcleared\x08\x08\x08\x7f and \x9b more"
    out = _render({"event": "request_failed", "exception": exception})
    _, _, echoed = out.partition("\n")
    for raw in ("\x1b", "\x08", "\x7f", "\x9b"):
        assert raw not in echoed
    assert "\\u001b" in echoed and "\\u0008" in echoed
    assert echoed.startswith(log_config._TRACEBACK_ECHO_PREFIX)


def test_bidi_controls_cannot_reorder_the_echoed_line():
    # UAX #9 / Trojan Source (CVE-2021-42574): json.dumps escapes these, so the echo is the only raw path.
    exception = "ValueError: rejected upload ‮gnp.eliforp/sdaolpu/"
    echoed = _render({"event": "request_failed", "exception": exception}).partition("\n")[2]
    assert "‮" not in echoed
    assert "\\u202e" in echoed
    # An unterminated isolate reorders a line too, so the whole set is covered, not just the override.
    exotic = "ValueError: " + "".join(sorted(log_config._BIDI_CONTROLS))
    echoed = _render({"event": "e", "exception": exotic}).partition("\n")[2]
    for ch in log_config._BIDI_CONTROLS:
        assert ch not in echoed
        assert f"\\u{ord(ch):04x}" in echoed


def test_the_escaped_set_is_exactly_unicodes_bidi_controls():
    # Pinned to PropList.txt's Bidi_Control so the set cannot widen into all of category Cf nor narrow to U+202E.
    assert log_config._BIDI_CONTROLS == frozenset(
        chr(c)
        for c in (
            0x061C,
            0x200E,
            0x200F,
            0x202A,
            0x202B,
            0x202C,
            0x202D,
            0x202E,
            0x2066,
            0x2067,
            0x2068,
            0x2069,
        )
    )


def test_zero_width_and_joining_characters_stay_readable():
    # Cf, but they reorder nothing: ZWNJ carries meaning in Persian/Arabic, ZWJ builds emoji sequences.
    exception = "ValueError: ‌بی‌نام and \U0001f469‍\U0001f4bb"
    echoed = _render({"event": "e", "exception": exception}).partition("\n")[2]
    assert "‌" in echoed and "‍" in echoed


def test_ordinary_text_is_left_readable():
    exception = "ValueError: 中文 café — tab:\there"
    echoed = _render({"event": "e", "exception": exception}).partition("\n")[2]
    assert "中文" in echoed and "café" in echoed and "—" in echoed
    assert "\there" in echoed


def test_the_exception_line_survives_a_cap_that_cannot_fit_it():
    # A control-heavy last line overflows the tail budget, and dropping it whole left every frame and no reason.
    frames = "\n".join(f'  File "/app/x{i}.py", line {i}, in fn' for i in range(60))
    payload = (
        "Traceback (most recent call last):\n"
        + frames
        + "\nValueError: rejected upload "
        + ("\x00" * 3000)
    )
    capped = log_config.truncate_exception({"exception": payload})["exception"]
    out = _render({"event": "request_failed", "exception": capped})
    _, _, echoed = out.partition("\n")
    lines = echoed.split("\n")
    assert lines[-1].startswith("| ValueError: rejected upload ")
    assert len(echoed) <= log_config._MAX_EXC_CHARS + 200
    for line in lines:
        assert line.startswith("| "), line

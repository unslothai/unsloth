# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A traceback in the log file has to be readable as a traceback.

~/.unsloth/studio/logs is a tee of stdout and stdout is JSON, so every stack trace
reached the reader as one line with its newlines escaped to ``\\n``. Reported from the
field: the only way to find out why an Image Transform failed was to dig that line out
of the log, and it arrived with, in the reporter's words, all its newlines mangled.

``with_readable_traceback`` echoes the traceback under the record. The JSON line itself
must survive byte-for-byte, because that is what anything parsing the file reads.
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
    # The JSON record is untouched -- still one parseable line, still carrying the
    # escaped exception, so a record-by-record reader sees exactly what it saw before.
    record = json.loads(first)
    assert record["exception"] == _TRACEBACK
    # ...and the human-readable copy follows, as separate physical lines.
    assert rest == _TRACEBACK
    assert rest.splitlines()[0] == "Traceback (most recent call last):"
    assert rest.splitlines()[-1].startswith("RuntimeError: Input type (float)")


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
    # Development already prints tracebacks as tracebacks; wrapping it would double them.
    import inspect

    source = inspect.getsource(log_config.LogConfig.setup_logging)
    assert "with_readable_traceback(structlog.processors.JSONRenderer" in source
    assert "with_readable_traceback(structlog.dev.ConsoleRenderer" not in source


def test_echoed_copy_is_the_redacted_truncated_one():
    # The wrapper reads event_dict["exception"] AFTER filter_sensitive_data and the
    # truncation processor have rewritten it, so a secret cannot re-enter the log
    # through the readable copy and a 2 MB traceback cannot be echoed at full length.
    huge = "HEAD" + ("x" * 4_000_000) + "\nValueError: nope"
    capped = log_config.truncate_exception({"exception": huge})["exception"]
    out = _render({"event": "request_failed", "exception": capped})
    assert len(out) < 2 * (log_config._MAX_EXC_CHARS + 500)
    assert out.endswith("ValueError: nope")

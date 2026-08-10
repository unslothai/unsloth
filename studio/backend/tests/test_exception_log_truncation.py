# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One log record must not be able to grow without bound.

request_failed renders the whole traceback into an "exception" field. That is a
few KB for a normal failure, but an exception whose message embeds a request body
is not: a rejected binary upload produced a single 2.2 MB line. The head (raising
frame) and the tail (exception type and message) are what a reader needs, so the
middle is dropped with a count of what went missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loggers import config as log_config  # noqa: E402


def test_short_tracebacks_are_untouched():
    ev = {"exception": "Traceback...\nValueError: nope"}
    assert log_config.truncate_exception(dict(ev)) == ev


def test_long_traceback_is_capped():
    text = "HEAD" + ("x" * 4_000_000) + "TAILValueError: nope"
    out = log_config.truncate_exception({"exception": text})["exception"]
    assert len(out) < log_config._MAX_EXC_CHARS + 200, len(out)


def test_head_and_tail_survive():
    text = "TRACEBACK_HEAD_MARKER" + ("x" * 4_000_000) + "EXC_TAIL_MARKER"
    out = log_config.truncate_exception({"exception": text})["exception"]
    assert out.startswith("TRACEBACK_HEAD_MARKER")
    assert out.endswith("EXC_TAIL_MARKER")
    assert "chars of traceback omitted" in out


def test_non_string_exception_field_is_ignored():
    ev = {"exception": None}
    assert log_config.truncate_exception(dict(ev)) == ev
    ev2 = {"event": "no exception here"}
    assert log_config.truncate_exception(dict(ev2)) == ev2


def test_cap_can_be_disabled(monkeypatch):
    monkeypatch.setattr(log_config, "_MAX_EXC_CHARS", 0)
    text = "y" * 100_000
    assert log_config.truncate_exception({"exception": text})["exception"] == text


def test_processor_signature_matches_structlog():
    text = "z" * 100_000
    out = log_config._truncate_exception_processor(None, "error", {"exception": text})
    assert len(out["exception"]) < len(text)
